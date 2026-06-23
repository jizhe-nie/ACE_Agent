"""
subtype/_stability.py — 稳定性/零模型方法学共享helpers（W4/W5 复用）。
（注：03_ 有早期内联副本，逻辑一致；后续小重构会让 03_ 也改为 import 这里。）
"""
from __future__ import annotations

import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score


def make_null_pca(X_genes: np.ndarray, n_pcs: int, seed: int = 42) -> np.ndarray:
    """零模型：每个基因(列)独立打乱 → 毁掉多元/簇结构、保留边际 → PCA。"""
    rng = np.random.RandomState(seed)
    Xn = X_genes.copy()
    for j in range(Xn.shape[1]):
        rng.shuffle(Xn[:, j])
    return PCA(n_components=n_pcs, random_state=seed).fit_transform(Xn)


def ward_partition(Xp: np.ndarray, k: int) -> np.ndarray:
    """Ward 层次聚类切成 k 簇（确定性；不同 k 嵌套，天然成层级）。"""
    return AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(Xp)


def kmeans_partition(Xp: np.ndarray, k: int, seed: int = 42) -> np.ndarray:
    return KMeans(k, n_init=10, random_state=seed).fit_predict(Xp)


def subsample_stability(Xp, k, partition_fn=ward_partition, B=25, frac=0.8, seed=42) -> float:
    """稳定性 = 子采样划分 vs 全量划分的平均一致性(ARI)。返回均值(该数据的统计量)。"""
    ref = partition_fn(Xp, k)
    n = Xp.shape[0]
    rng = np.random.RandomState(seed)
    aris = []
    for b in range(B):
        idx = rng.choice(n, int(frac * n), replace=False)
        aris.append(adjusted_rand_score(ref[idx], partition_fn(Xp[idx], k)))
    return float(np.mean(aris))


def consensus_matrix(Xp, k, partition_fn=ward_partition, B=30, frac=0.8, seed=42) -> np.ndarray:
    """共识矩阵 C[i,j]=两样本在子采样中被分到同簇的频率。"""
    n = Xp.shape[0]
    rng = np.random.RandomState(seed)
    co = np.zeros((n, n)); cnt = np.zeros((n, n))
    for b in range(B):
        idx = rng.choice(n, int(frac * n), replace=False)
        lab = partition_fn(Xp[idx], k)
        for c in np.unique(lab):
            m = idx[lab == c]
            co[np.ix_(m, m)] += 1
        cnt[np.ix_(idx, idx)] += 1
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(cnt > 0, co / cnt, 0.0)


def bh_fdr(pvals: list[float]) -> list[float]:
    """Benjamini-Hochberg FDR 校正，返回 q 值(与输入同序)。"""
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    q = np.empty(n)
    prev = 1.0
    for rank in range(n - 1, -1, -1):
        i = order[rank]
        prev = min(prev, p[i] * n / (rank + 1))
        q[i] = prev
    return q.tolist()
