"""
subtype/_data.py — 共享数据加载与预处理。
W1 体检 (01_) 与 W3 内核 (02_) 必须用**完全相同**的预处理，结果才可比。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


def hopkins(X: np.ndarray, m: int = 150, seed: int = 42) -> float:
    """可聚类性。H≈0.5 无结构；H→1 有团块。高维下会被稀释、易低估。"""
    rng = np.random.RandomState(seed)
    n, d = X.shape
    m = min(m, n - 1)
    nbrs = NearestNeighbors(n_neighbors=2).fit(X)
    idx = rng.choice(n, m, replace=False)
    w_dist, _ = nbrs.kneighbors(X[idx], n_neighbors=2)
    w = w_dist[:, 1]
    mins, maxs = X.min(axis=0), X.max(axis=0)
    unif = rng.uniform(mins, maxs, size=(m, d))
    u_dist, _ = nbrs.kneighbors(unif, n_neighbors=1)
    u = u_dist[:, 0]
    denom = u.sum() + w.sum()
    return float(u.sum() / denom) if denom > 0 else 0.5


def load_real(expr_path: str, pheno_path: str):
    """读表达矩阵 + 表型表，自动转置(基因为行时)并按样本取交集对齐。"""
    expr = pd.read_csv(expr_path, sep="\t", index_col=0)
    pheno = pd.read_csv(pheno_path, sep="\t", index_col=0)
    common = expr.index.intersection(pheno.index)
    if len(common) == 0:  # 兼容"基因为行、样本为列"
        expr = expr.T
        common = expr.index.intersection(pheno.index)
    expr, pheno = expr.loc[common], pheno.loc[common]
    return expr, pheno


def filter_labeled(expr: pd.DataFrame, pheno: pd.DataFrame, subtype_col: str):
    """只保留有金标准亚型标签的样本(PAM50 常缺失：正常旁组织/未分型)。"""
    if subtype_col not in pheno.columns:
        return expr, pheno, 0
    valid = pheno[subtype_col].notna() & pheno[subtype_col].astype(str).str.strip().ne("")
    n_drop = int((~valid).sum())
    keep = pheno.index[valid]
    return expr.loc[keep], pheno.loc[keep], n_drop


def top_mad_genes(Xdf: pd.DataFrame, n: int) -> list:
    """按 MAD(中位绝对偏差)选前 n 个高变特征(基因)。"""
    X = Xdf.to_numpy(float)
    mad = np.median(np.abs(X - np.median(X, axis=0)), axis=0)
    return Xdf.columns[np.argsort(mad)[::-1][:n]].tolist()


def exclude_normal_tissue(expr: pd.DataFrame, pheno: pd.DataFrame):
    """剔除癌旁正常组织样本，做纯肿瘤分型。
    TCGA 条码第 4 段前两位=样本类型码：01-09 肿瘤，10-19 正常，20+ 对照。
    """
    def is_tumor(bc: str) -> bool:
        parts = str(bc).split("-")
        if len(parts) < 4 or len(parts[3]) < 2:
            return True  # 无法判定则保留
        try:
            return int(parts[3][:2]) < 10
        except ValueError:
            return True
    mask = pheno.index.to_series().map(is_tumor).to_numpy()
    keep = pheno.index[mask]
    return expr.loc[keep], pheno.loc[keep], int((~mask).sum())


def preprocess(expr: pd.DataFrame, top_genes: int = 1000, n_pcs: int = 50, seed: int = 42):
    """log 稳方差 → 选高变(MAD)基因 → 标准化 → PCA 降噪。返回 (X_genes, Xp_pca, evr)。"""
    X = expr.to_numpy(dtype=float)
    X = np.nan_to_num(X, nan=np.nanmedian(X))
    if X.min() >= 0 and X.max() > 50:  # 像原始计数才 log；Xena 已是 log2 则跳过
        X = np.log1p(X)
    mad = np.median(np.abs(X - np.median(X, axis=0)), axis=0)
    keep = np.argsort(mad)[::-1][:min(top_genes, X.shape[1])]
    X = X[:, keep]
    X = StandardScaler().fit_transform(X)
    n_pcs = min(n_pcs, X.shape[0] - 1, X.shape[1])
    pca = PCA(n_components=n_pcs, random_state=seed).fit(X)
    return X, pca.transform(X), pca.explained_variance_ratio_
