"""
ST-W2 — SOTA 级空间图自编码器基线 (GCN-AE, STAGATE 思路)。这是"要打败的基线"/真内核(L2)。

思路（领域 SOTA 的核心机制）：把每个 spot 与空间近邻连成图，用**图卷积**让表示"既保自身表达、又融合邻居"，
自编码器以重构表达为目标学到低维潜表示，再聚类 → 空间域。预期 DLPFC ARI ~0.5-0.6（远超 01 的非空间 0.20 / 简单空间 0.32）。

对照：非空间 KMeans/PCA(=01) vs 本 GCN-AE vs SOTA 参考(STAGATE~0.60/GraphST~0.63)。
环境：conda run -n Tumor_Subtype_Agent python st/02_graph_ae.py --sample-dir data/dlpfc/extracted/DLPFC12/151673
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
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import kneighbors_graph

from _stdata import ari_nmi, load_dlpfc, n_layers, preprocess

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def norm_adj(coords, k=6):
    """空间 kNN 图 → 对称 → Â = D^-1/2 (A+I) D^-1/2（稠密）。"""
    A = kneighbors_graph(coords, k, mode="connectivity", include_self=False)
    A = A.maximum(A.T)                       # 对称
    A = A + sp.eye(A.shape[0])               # 自环
    d = np.asarray(A.sum(1)).ravel()
    dinv = sp.diags(1.0 / np.sqrt(d))
    return (dinv @ A @ dinv).toarray().astype(np.float32)


class GCN_AE(nn.Module):
    def __init__(self, d_in, h=512, z=30):
        super().__init__()
        self.e1 = nn.Linear(d_in, h); self.e2 = nn.Linear(h, z)
        self.d1 = nn.Linear(z, h); self.d2 = nn.Linear(h, d_in)

    def forward(self, X, A):
        h = F.relu(A @ self.e1(X))
        z = A @ self.e2(h)
        h2 = F.relu(A @ self.d1(z))
        xr = A @ self.d2(h2)
        return z, xr


def train_gae(X, A, z=30, epochs=1000, lr=1e-3, wd=1e-4, seed=0):
    torch.manual_seed(seed)
    m = GCN_AE(X.shape[1], z=z).to(DEV)
    opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=wd)
    for _ in range(epochs):
        opt.zero_grad()
        _, xr = m(X, A)
        loss = F.mse_loss(xr, X)
        loss.backward(); opt.step()
    m.eval()
    with torch.no_grad():
        Z, _ = m(X, A)
    return Z.detach().cpu().numpy()


def main():
    ap = argparse.ArgumentParser(description="ST-W2 GCN-AE 基线")
    ap.add_argument("--sample-dir", required=True)
    ap.add_argument("--knn", type=int, default=6)
    ap.add_argument("--epochs", type=int, default=1000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--latent", type=int, default=30)
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()

    print(f"[device] {DEV}")
    ad, gt = load_dlpfc(Path(args.sample_dir))
    k = n_layers(ad, gt)
    adp = preprocess(ad)
    X = torch.tensor(np.asarray(adp.X, dtype=np.float32), device=DEV)
    A = torch.tensor(norm_adj(ad.obsm["spatial"], args.knn), device=DEV)
    gt_arr = ad.obs[gt]
    print(f"[数据] {ad.n_obs} spots × {adp.n_vars} HVG | k={k} | 图 kNN={args.knn} | {DEV}")

    # 非空间基线(=01, 上下文)
    ns_aris = [ari_nmi(KMeans(k, n_init=10, random_state=s).fit_predict(adp.obsm["X_pca"]), gt_arr)[0]
               for s in range(args.seeds)]
    print(f"[非空间 KMeans/PCA] ARI={np.mean(ns_aris):.3f}±{np.std(ns_aris):.3f}")

    # GCN-AE（多 seed）+ GMM(mclust 风格) / KMeans 聚类
    gmm_aris, km_aris = [], []
    for s in range(args.seeds):
        Z = train_gae(X, A, z=args.latent, epochs=args.epochs, lr=args.lr, seed=s)
        lab_g = GaussianMixture(k, covariance_type="full", n_init=5, random_state=s).fit_predict(Z)
        lab_k = KMeans(k, n_init=10, random_state=s).fit_predict(Z)
        gmm_aris.append(ari_nmi(lab_g, gt_arr)[0]); km_aris.append(ari_nmi(lab_k, gt_arr)[0])

    print(f"\n{'方法':<26}{'ARI(mean±std)':<20}")
    print("-" * 46)
    print(f"{'非空间 KMeans/PCA':<26}{f'{np.mean(ns_aris):.3f}±{np.std(ns_aris):.3f}':<20}")
    print(f"{'GCN-AE + KMeans':<26}{f'{np.mean(km_aris):.3f}±{np.std(km_aris):.3f}':<20}")
    print(f"{'GCN-AE + GMM(mclust风)':<26}{f'{np.mean(gmm_aris):.3f}±{np.std(gmm_aris):.3f}':<20}")
    print(f"{'SOTA 参考':<26}{'STAGATE~0.60 / GraphST~0.63':<20}")

    best = max(np.mean(km_aris), np.mean(gmm_aris))
    print("\n[诚实判读]")
    print(f"  GCN-AE 最佳 ARI={best:.3f} vs 非空间 {np.mean(ns_aris):.3f}：{'↑图卷积大幅提升' if best>np.mean(ns_aris)+0.1 else '提升有限'}")
    gap = 0.60 - best
    if gap <= 0.05:
        print(f"  已达 SOTA 量级(差 {gap:.2f}) → 基线复现成功，可作'要打败的对象'，进 W3 找创新。")
    elif gap <= 0.15:
        print(f"  接近 SOTA(差 {gap:.2f}) → 基线基本到位，可调超参再逼近，或直接进 W3。")
    else:
        print(f"  距 SOTA 还差 {gap:.2f} → 基线未达标，需调架构/超参(层数/latent/epochs/图构建)。")


if __name__ == "__main__":
    main()
