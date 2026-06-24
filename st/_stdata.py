"""
st/_stdata.py — 空间转录组共享数据/预处理(DLPFC benchmarkst 格式)。01/02 复用，保证一致。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


def load_dlpfc(sample_dir):
    """加载一个 DLPFC 样本 → AnnData。
    表达=10x h5；坐标=spatial/tissue_positions_list.csv；金标准=gt/layered/{sample}_{LAYER}_barcodes.txt。
    """
    sample_dir = Path(sample_dir)
    sample = sample_dir.name
    ad = sc.read_10x_h5(next(sample_dir.glob("*_filtered_feature_bc_matrix.h5")))
    ad.var_names_make_unique()
    pos = pd.read_csv(sample_dir / "spatial" / "tissue_positions_list.csv", header=None, index_col=0)
    pos.columns = ["in_tissue", "row", "col", "pxl_row", "pxl_col"]
    gt = pd.Series(index=ad.obs_names, dtype=object)
    for f in sorted((sample_dir / "gt" / "layered").glob(f"{sample}_*_barcodes.txt")):
        layer = f.stem.replace(f"{sample}_", "").replace("_barcodes", "")
        bcs = [ln.strip() for ln in open(f) if ln.strip()]
        gt.loc[gt.index.intersection(bcs)] = layer
    ad.obs["ground_truth"] = gt.values
    common = ad.obs_names.intersection(pos.index)
    ad = ad[common].copy()
    ad.obsm["spatial"] = pos.loc[common, ["pxl_row", "pxl_col"]].to_numpy(float)
    return ad, "ground_truth"


def preprocess(ad, n_top=3000, n_pcs=50):
    """标准 scanpy：normalize → log1p → HVG(n_top) → scale → PCA。
    返回的 adata：.X = 缩放后的 HVG 表达(供 GNN 用)；obsm['X_pca'] = PCA(供基线/聚类用)。"""
    ad = ad.copy()
    sc.pp.filter_genes(ad, min_cells=3)
    sc.pp.normalize_total(ad, target_sum=1e4)
    sc.pp.log1p(ad)
    sc.pp.highly_variable_genes(ad, n_top_genes=n_top)
    ad = ad[:, ad.var.highly_variable].copy()
    sc.pp.scale(ad, max_value=10)
    sc.tl.pca(ad, n_comps=n_pcs)
    return ad


def ari_nmi(labels, gt):
    """ARI/NMI vs 金标准(剔除 NaN spot)。"""
    gt = pd.Series(np.asarray(gt))
    mask = gt.notna().to_numpy()
    gc = pd.Categorical(gt[mask]).codes
    lab = np.asarray(labels)[mask]
    return adjusted_rand_score(gc, lab), normalized_mutual_info_score(gc, lab)


def n_layers(ad, gt_key="ground_truth"):
    return int(pd.Categorical(ad.obs[gt_key][ad.obs[gt_key].notna()]).categories.size)


def refine(labels, coords, k=50):
    """空间近邻多数投票精修(GraphST 标准后处理)。"""
    from sklearn.neighbors import NearestNeighbors
    labels = np.asarray(labels)
    nn = NearestNeighbors(n_neighbors=min(k, len(labels))).fit(coords)
    _, idx = nn.kneighbors(coords)
    out = labels.copy()
    for i in range(len(labels)):
        v, c = np.unique(labels[idx[i]], return_counts=True)
        out[i] = v[c.argmax()]
    return out


def get_emb(adata):
    """从 GraphST 训练后的 adata 取 embedding(高维则 PCA 到 20)。"""
    from sklearn.decomposition import PCA
    for key in ("emb_pca", "emb", "GraphST", "feat"):
        if key in adata.obsm:
            E = np.asarray(adata.obsm[key])
            return PCA(20, random_state=0).fit_transform(E) if E.shape[1] > 30 else E
    raise KeyError(f"no GraphST embedding; obsm={list(adata.obsm.keys())}")
