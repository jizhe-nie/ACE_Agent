"""
ST-W2 收尾 — 官方 SOTA 基线 GraphST。确立"要打败的对象"(在我们这份数据上的真实天花板 ~0.6)。

流程：raw adata(counts + spatial) → 官方 GraphST 训练得 embedding → GMM(k=7) 聚类 + 空间近邻精修(GraphST 标准后处理) → ARI。
对照：非空间 0.20 / 我的 GCN-AE 0.336 / GraphST 论文 ~0.63。
环境：conda run -n Tumor_Subtype_Agent python st/03_graphst_baseline.py --sample-dir data/dlpfc/extracted/DLPFC12/151673
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors

from _stdata import ari_nmi, load_dlpfc, n_layers

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def refine(labels, coords, k=50):
    """GraphST 标准后处理：每个 spot 取其空间近邻的多数标签（空间一致性精修）。"""
    nn = NearestNeighbors(n_neighbors=min(k, len(labels))).fit(coords)
    _, idx = nn.kneighbors(coords)
    out = labels.copy()
    for i in range(len(labels)):
        v, c = np.unique(labels[idx[i]], return_counts=True)
        out[i] = v[c.argmax()]
    return out


def get_emb(adata):
    for key in ("emb_pca", "emb", "GraphST", "feat"):
        if key in adata.obsm:
            E = np.asarray(adata.obsm[key])
            return PCA(20, random_state=0).fit_transform(E) if E.shape[1] > 30 else E
    raise KeyError(f"未找到 GraphST embedding, obsm keys={list(adata.obsm.keys())}")


def main():
    ap = argparse.ArgumentParser(description="GraphST 官方基线")
    ap.add_argument("--sample-dir", required=True)
    ap.add_argument("--seeds", type=int, default=1)
    args = ap.parse_args()

    from GraphST import GraphST
    ad, gt = load_dlpfc(Path(args.sample_dir))
    k = n_layers(ad, gt)
    coords = ad.obsm["spatial"].astype(float)
    print(f"[device] {DEV} | {ad.n_obs} spots × {ad.n_vars} genes | k={k}")

    raw_aris, ref_aris = [], []
    for s in range(args.seeds):
        a = ad.copy()
        torch.manual_seed(s); np.random.seed(s)
        model = GraphST.GraphST(a, device=DEV, random_seed=s)
        a = model.train()
        emb = get_emb(a)
        lab = GaussianMixture(k, covariance_type="full", n_init=5, random_state=s).fit_predict(emb)
        lab_ref = refine(lab, coords)
        raw_aris.append(ari_nmi(lab, ad.obs[gt])[0])
        ref_aris.append(ari_nmi(lab_ref, ad.obs[gt])[0])
        print(f"  seed {s}: ARI raw={raw_aris[-1]:.3f}  refined={ref_aris[-1]:.3f}")

    print(f"\n{'方法':<26}{'ARI(mean±std)':<20}")
    print("-" * 46)
    print(f"{'非空间 KMeans/PCA(参考)':<26}{'~0.201':<20}")
    print(f"{'我的 GCN-AE(参考)':<26}{'~0.336':<20}")
    print(f"{'GraphST + GMM':<26}{f'{np.mean(raw_aris):.3f}±{np.std(raw_aris):.3f}':<20}")
    print(f"{'GraphST + GMM + 精修':<26}{f'{np.mean(ref_aris):.3f}±{np.std(ref_aris):.3f}':<20}")
    print(f"{'GraphST 论文参考':<26}{'~0.63':<20}")

    best = max(np.mean(raw_aris), np.mean(ref_aris))
    print(f"\n[基线确立] GraphST 在 151673 达 ARI={best:.3f}（论文 ~0.63）。")
    print("  → 这就是'要打败的对象'。下一步 ST-W3：找它失效处 → 真创新点。")


if __name__ == "__main__":
    main()
