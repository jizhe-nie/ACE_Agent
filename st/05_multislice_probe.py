"""
ST-W3b 探针 — 验证"多切片整合"假设(不建大方法，先廉价验证)。

失效分析(#0026)：donor3 的 151669/151670 崩(ARI 0.29/0.26)，但同 donor 相邻片 151671/672 不错。
假设：相邻切片是连续组织，弱片可向相邻片借力 → 联合建模应救起弱片。

设计(隔离"多切片机制"本身)：donor3 的 4 片用**同一联合预处理特征**(共同基因/联合HVG/联合PCA)，
唯一区别在**图**：
  - 单片：每片只用自己的"空间近邻图"，各自跑 GCN-AE → 各片 ARI。
  - 联合：所有 spot 一张图 = 各片空间近邻(块) + **跨片互近邻 MNN**(让片间共享信息)，联合跑 GCN-AE → 各片 ARI。
若弱片(669/670)联合 ARI 明显 > 单片 → 假设成立 → 值得建正式多切片方法；否则转图像多模态。

环境：conda run -n Tumor_Subtype_Agent python st/05_multislice_probe.py
"""
from __future__ import annotations

import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import anndata as adlib
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors

from _stdata import ari_nmi, load_dlpfc, n_layers, preprocess, refine

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROOT = Path("data/dlpfc/extracted/DLPFC12")
DONOR3 = ["151669", "151670", "151671", "151672"]


def spatial_edges(coords, k=6, offset=0):
    nn = NearestNeighbors(n_neighbors=k + 1).fit(coords)
    _, idx = nn.kneighbors(coords)
    e = [(offset + i, offset + j) for i in range(len(coords)) for j in idx[i, 1:]]
    return e


def mnn_edges(Pa, Pb, oa, ob, k=3):
    """两片在 PCA 空间的互近邻 → 跨片边。"""
    na = NearestNeighbors(n_neighbors=k).fit(Pb); _, ia = na.kneighbors(Pa)  # a→b
    nb = NearestNeighbors(n_neighbors=k).fit(Pa); _, ib = nb.kneighbors(Pb)  # b→a
    sa = {(i, j) for i in range(len(Pa)) for j in ia[i]}
    e = []
    for j in range(len(Pb)):
        for i in ib[j]:
            if (i, j) in sa:
                e.append((oa + i, ob + j))
    return e


def norm_adj_sparse(edges, n):
    if not edges:
        edges = []
    r = [e[0] for e in edges] + [e[1] for e in edges] + list(range(n))   # 对称 + 自环
    c = [e[1] for e in edges] + [e[0] for e in edges] + list(range(n))
    A = sp.coo_matrix((np.ones(len(r)), (r, c)), shape=(n, n)).tocsr()
    A.data[:] = 1.0
    d = np.asarray(A.sum(1)).ravel()
    dinv = sp.diags(1.0 / np.sqrt(np.maximum(d, 1e-12)))
    An = (dinv @ A @ dinv).tocoo()
    i = torch.tensor(np.vstack([An.row, An.col]), dtype=torch.long)
    v = torch.tensor(An.data, dtype=torch.float32)
    return torch.sparse_coo_tensor(i, v, (n, n)).coalesce().to(DEV)


class GCN_AE(nn.Module):
    def __init__(self, d, h=512, z=30):
        super().__init__()
        self.e1 = nn.Linear(d, h); self.e2 = nn.Linear(h, z)
        self.d1 = nn.Linear(z, h); self.d2 = nn.Linear(h, d)

    def forward(self, X, A):
        sm = torch.sparse.mm
        h = F.relu(sm(A, self.e1(X))); z = sm(A, self.e2(h))
        h2 = F.relu(sm(A, self.d1(z))); xr = sm(A, self.d2(h2))
        return z, xr


def train(X, A, epochs=800, lr=1e-3, seed=0):
    torch.manual_seed(seed)
    m = GCN_AE(X.shape[1]).to(DEV)
    opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=1e-4)
    for _ in range(epochs):
        opt.zero_grad(); _, xr = m(X, A); F.mse_loss(xr, X).backward(); opt.step()
    with torch.no_grad():
        return m(X, A)[0].cpu().numpy()


def cluster_ari(Z, gt, coords, k, seed=0):
    from sklearn.preprocessing import StandardScaler
    Zs = StandardScaler().fit_transform(Z)
    try:
        lab = GaussianMixture(k, covariance_type="full", reg_covar=1e-3,
                              n_init=5, random_state=seed).fit_predict(Zs)
    except Exception:
        lab = KMeans(k, n_init=10, random_state=seed).fit_predict(Zs)
    return ari_nmi(refine(lab, coords), gt)[0]


def main():
    # 载入 4 片 + 联合预处理(共同基因/联合HVG/联合PCA)
    ads = {}
    for s in DONOR3:
        ad, gt = load_dlpfc(ROOT / s); ad.obs["slice"] = s; ads[s] = ad
    common = None
    for ad in ads.values():
        common = ad.var_names if common is None else common.intersection(ad.var_names)
    joint = adlib.concat([ads[s][:, common] for s in DONOR3], label="slice", keys=DONOR3)
    jp = preprocess(joint)
    X = torch.tensor(np.asarray(jp.X, dtype=np.float32), device=DEV)
    P = jp.obsm["X_pca"]
    sl = joint.obs["slice"].to_numpy()
    coords = joint.obsm["spatial"].astype(float)
    k = 5  # donor3 各片 5 层
    print(f"[device] {DEV} | donor3 联合 {joint.n_obs} spots × {jp.n_vars} HVG | k={k}")

    # 各片在联合特征中的下标
    idxs = {s: np.where(sl == s)[0] for s in DONOR3}

    # 单片图 + 联合图
    single_edges = {}
    all_edges = []
    for s in DONOR3:
        ii = idxs[s]
        e = spatial_edges(coords[ii], k=6, offset=int(ii[0]))  # ii 连续(concat 顺序)
        single_edges[s] = [(a - int(ii[0]), b - int(ii[0])) for (a, b) in e]
        all_edges += e
    # 跨片 MNN
    for a in range(len(DONOR3)):
        for b in range(a + 1, len(DONOR3)):
            sa, sb = DONOR3[a], DONOR3[b]
            all_edges += mnn_edges(P[idxs[sa]], P[idxs[sb]], int(idxs[sa][0]), int(idxs[sb][0]), k=3)
    A_joint = norm_adj_sparse(all_edges, joint.n_obs)

    # 联合训练 → 各片 ARI
    Zj = train(X, A_joint)
    print(f"\n{'sample':<10}{'单片 GCN-AE':<14}{'联合 GCN-AE':<14}{'GraphST(参考)':<14}")
    print("-" * 52)
    gst_ref = {"151669": 0.294, "151670": 0.264, "151671": 0.523, "151672": 0.475}
    single, jointv = [], []
    for s in DONOR3:
        ii = idxs[s]
        Xs = X[ii]; As = norm_adj_sparse(single_edges[s], len(ii))
        Zs = train(Xs, As)
        gt_s = joint.obs["ground_truth"].to_numpy()[ii]
        c_s = coords[ii]
        a_single = cluster_ari(Zs, gt_s, c_s, k)
        a_joint = cluster_ari(Zj[ii], gt_s, c_s, k)
        single.append(a_single); jointv.append(a_joint)
        flag = " <<弱片" if s in ("151669", "151670") else ""
        print(f"{s:<10}{a_single:<14.3f}{a_joint:<14.3f}{gst_ref[s]:<14.3f}{flag}")
    print("-" * 52)
    print(f"{'MEAN':<10}{np.mean(single):<14.3f}{np.mean(jointv):<14.3f}")

    weak_s = np.mean([single[0], single[1]]); weak_j = np.mean([jointv[0], jointv[1]])
    print(f"\n[假设判读] 弱片(669+670) 单片={weak_s:.3f} → 联合={weak_j:.3f} "
          f"({'↑多切片救起弱片, 假设成立' if weak_j > weak_s + 0.03 else '≈无明显增益'})")
    print("  成立 → 值得建正式多切片方法去超 GraphST；不成立 → 转图像多模态。")


if __name__ == "__main__":
    main()
