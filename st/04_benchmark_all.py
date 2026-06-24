"""
ST-W3a — 全 12 样本 DLPFC 基准。SOTA 论文报 12 样本均值；这张表是后续"超越 SOTA"的对照基础，
也用来看 GraphST 在哪些样本上弱(指引创新点)。

每样本：非空间(KMeans/PCA) vs GraphST(emb→GMM+精修)，均用同一聚类器(受控比较)。1 seed(求速，可后续加 seed)。
环境：conda run -n Tumor_Subtype_Agent python st/04_benchmark_all.py
"""
from __future__ import annotations

import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

from _stdata import ari_nmi, get_emb, load_dlpfc, n_layers, preprocess, refine

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROOT = Path("data/dlpfc/extracted/DLPFC12")
SAMPLES = ["151507", "151508", "151509", "151510", "151669", "151670",
           "151671", "151672", "151673", "151674", "151675", "151676"]


def main():
    from GraphST import GraphST
    print(f"[device] {DEV} | 12 样本基准 (1 seed)\n")
    print(f"{'sample':<10}{'k':<4}{'非空间':<10}{'GraphST':<10}")
    print("-" * 34)
    rows = []
    for s in SAMPLES:
        ad, gt = load_dlpfc(ROOT / s)
        k = n_layers(ad, gt)
        coords = ad.obsm["spatial"].astype(float)
        adp = preprocess(ad)
        ns = ari_nmi(KMeans(k, n_init=10, random_state=0).fit_predict(adp.obsm["X_pca"]), ad.obs[gt])[0]
        a = ad.copy()
        torch.manual_seed(0); np.random.seed(0)
        a = GraphST.GraphST(a, device=DEV, random_seed=0).train()
        lab = GaussianMixture(k, covariance_type="full", n_init=5, random_state=0).fit_predict(get_emb(a))
        gs = ari_nmi(refine(lab, coords), ad.obs[gt])[0]
        rows.append((s, k, ns, gs))
        print(f"{s:<10}{k:<4}{ns:<10.3f}{gs:<10.3f}", flush=True)

    arr = np.array([[r[2], r[3]] for r in rows])
    print("-" * 34)
    print(f"{'MEAN':<10}{'':<4}{arr[:,0].mean():<10.3f}{arr[:,1].mean():<10.3f}")
    print(f"\n[基准均值] 非空间={arr[:,0].mean():.3f} | GraphST={arr[:,1].mean():.3f} "
          f"(论文 GraphST ~0.63 用 R mclust)")
    print("  → 这就是 12 样本受控基线。GraphST 最弱的样本即创新可切入处。")
    import pandas as pd
    out = Path(__file__).resolve().parent / "outputs"
    out.mkdir(exist_ok=True)
    pd.DataFrame(rows, columns=["sample", "k", "nonspatial_ari", "graphst_ari"]).to_csv(
        out / "benchmark_all.csv", index=False)


if __name__ == "__main__":
    main()
