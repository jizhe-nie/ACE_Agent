"""
W9 — L2 深度内核：多组学自编码器 + DEC 联合深度聚类（真内核），对标弱基线/AE+KMeans/PAM50。

为什么做这个（PM 纠偏后的正路）：W3–W8 一直停在 L0/L1（sklearn on PCA 的弱基线），从未建 L2
深度表示+联合聚类——而 lessons §3 说 L2 才是真内核、且是领域 SOTA 形态。本轮补上：
  - 多组学 AE：表达/CNV 各自编码 → 融合潜空间 z → 各自重构（学到联合表示）。
  - DEC (Xie et al. 2016 深度聚类经典)：AE 预训练 → 用 KMeans 初始化簇心 → 联合优化
    KL(target||soft-assign)+重构（IDEC 风格），让"表示"与"簇结构"一起学（= L2）。

诚实对照（k=5，多 seed mean±std）：
  1. sklearn 弱基线：Ward on PCA(multi)（我们 W8 的老路）
  2. AE + KMeans（深度表示，再聚类）
  3. AE + DEC（联合深度聚类，SOTA 族）—— L2 真内核
  4. PAM50（临床金标准对照）
指标：ARI/NMI vs PAM50（亚型召回）+ 生存 log-rank（全队列，生存未入训，非循环）+ Her2 召回。
**目标：看深度(L2) 到底能不能超过弱基线和 PAM50。结果好坏都直说。**

用法：
  python subtype/08_deep_kernel.py --expr data/brca/HiSeqV2.gz --cnv data/brca/CNV_gistic2.gz \
      --pheno data/brca/BRCA_clinicalMatrix --subtype-col PAM50Call_RNAseq
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
import pandas as pd
import torch
import torch.nn as nn
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import StandardScaler

from _data import exclude_normal_tissue, filter_labeled, load_real, top_mad_genes
from _survival import build_os, logrank

OUT_DIR = Path(__file__).resolve().parent / "outputs"
DEV = "cpu"


def prep(df, top=2000):
    genes = top_mad_genes(df, top)
    X = StandardScaler().fit_transform(df[genes].to_numpy(float))
    return torch.tensor(X, dtype=torch.float32)


class MultiOmicsAE(nn.Module):
    def __init__(self, de, dc, h=512, emb=128, z=64):
        super().__init__()
        self.emb = emb
        self.enc_e = nn.Sequential(nn.Linear(de, h), nn.ReLU(), nn.Linear(h, emb), nn.ReLU())
        self.enc_c = nn.Sequential(nn.Linear(dc, h), nn.ReLU(), nn.Linear(h, emb), nn.ReLU())
        self.fuse = nn.Linear(2 * emb, z)
        self.defuse = nn.Sequential(nn.Linear(z, 2 * emb), nn.ReLU())
        self.dec_e = nn.Sequential(nn.Linear(emb, h), nn.ReLU(), nn.Linear(h, de))
        self.dec_c = nn.Sequential(nn.Linear(emb, h), nn.ReLU(), nn.Linear(h, dc))

    def encode(self, e, c):
        return self.fuse(torch.cat([self.enc_e(e), self.enc_c(c)], 1))

    def forward(self, e, c):
        z = self.encode(e, c)
        d = self.defuse(z)
        return z, self.dec_e(d[:, :self.emb]), self.dec_c(d[:, self.emb:])


def train_ae(ae, E, C, epochs=300, lr=1e-3):
    opt = torch.optim.Adam(ae.parameters(), lr=lr)
    mse = nn.MSELoss()
    for _ in range(epochs):
        opt.zero_grad()
        _, e2, c2 = ae(E, C)
        loss = mse(e2, E) + mse(c2, C)
        loss.backward(); opt.step()
    return ae


def soft_assign(z, mu):
    # student-t kernel (DEC), alpha=1
    d2 = torch.cdist(z, mu) ** 2
    q = 1.0 / (1.0 + d2)
    return q / q.sum(1, keepdim=True)


def dec_refine(ae, E, C, k, seed, iters=200, update_p=20, gamma=0.5, lr=1e-4):
    with torch.no_grad():
        z0 = ae.encode(E, C).cpu().numpy()
    km = KMeans(k, n_init=10, random_state=seed).fit(z0)
    mu = nn.Parameter(torch.tensor(km.cluster_centers_, dtype=torch.float32))
    opt = torch.optim.Adam(list(ae.parameters()) + [mu], lr=lr)
    mse = nn.MSELoss()
    p = None
    for it in range(iters):
        z, e2, c2 = ae(E, C)
        q = soft_assign(z, mu)
        if it % update_p == 0:
            w = (q ** 2) / q.sum(0, keepdim=True)
            p = (w / w.sum(1, keepdim=True)).detach()
        kl = (p * torch.log((p + 1e-8) / (q + 1e-8))).sum(1).mean()
        loss = kl + gamma * (mse(e2, E) + mse(c2, C))
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        return soft_assign(ae.encode(E, C), mu).argmax(1).cpu().numpy()


def metrics(lab, y, pam_names, time, event, erbb2):
    ari = adjusted_rand_score(y, lab)
    nmi = normalized_mutual_info_score(y, lab)
    p = logrank(time, event, lab)[2]
    # Her2 召回（最佳簇）
    best = max(((pam_names[lab == c] == "Her2").sum() for c in np.unique(lab)), default=0)
    recall = best / max(1, int((pam_names == "Her2").sum()))
    return ari, nmi, p, recall


def main():
    ap = argparse.ArgumentParser(description="W9 深度 L2 内核")
    ap.add_argument("--expr", required=True)
    ap.add_argument("--cnv", required=True)
    ap.add_argument("--pheno", required=True)
    ap.add_argument("--subtype-col", default="PAM50Call_RNAseq")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()

    expr, pheno = load_real(args.expr, args.pheno)
    cnv = pd.read_csv(args.cnv, sep="\t", index_col=0).T
    common = expr.index.intersection(cnv.index).intersection(pheno.index)
    expr, cnv, pheno = expr.loc[common], cnv.loc[common], pheno.loc[common]
    expr, pheno, _ = exclude_normal_tissue(expr, pheno)
    expr, pheno, _ = filter_labeled(expr, pheno, args.subtype_col)
    cnv = cnv.loc[pheno.index]
    time, event, valid = build_os(pheno)
    keep = pheno.index[valid]
    expr, cnv, pheno = expr.loc[keep], cnv.loc[keep], pheno.loc[keep]
    time, event = time[valid], event[valid]
    y = pd.Categorical(pheno[args.subtype_col]).codes
    pam = pheno[args.subtype_col].to_numpy()
    erbb2 = cnv["ERBB2"].to_numpy(float) if "ERBB2" in cnv.columns else np.zeros(len(pheno))
    E = prep(expr).to(DEV); C = prep(cnv).to(DEV)
    k = args.k
    print(f"[数据] {E.shape[0]} 例 | 表达{E.shape[1]}+CNV{C.shape[1]} 基因 | 死亡{int(event.sum())} | k={k}\n")

    rows = {}

    # 1. sklearn 弱基线：Ward on PCA(multi)
    Xe = StandardScaler().fit_transform(PCA(30, random_state=0).fit_transform(E.numpy()))
    Xc = StandardScaler().fit_transform(PCA(30, random_state=0).fit_transform(C.numpy()))
    lab = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(np.hstack([Xe, Xc]))
    rows["sklearn 弱基线(Ward/PCA)"] = [metrics(lab, y, pam, time, event, erbb2)]

    # 2&3. 深度：AE+KMeans 与 AE+DEC（多 seed）
    aek, dec = [], []
    for s in range(args.seeds):
        torch.manual_seed(s); np.random.seed(s)
        ae = MultiOmicsAE(E.shape[1], C.shape[1]).to(DEV)
        train_ae(ae, E, C)
        with torch.no_grad():
            z = ae.encode(E, C).cpu().numpy()
        lab_k = KMeans(k, n_init=10, random_state=s).fit_predict(z)
        aek.append(metrics(lab_k, y, pam, time, event, erbb2))
        lab_d = dec_refine(ae, E, C, k, s)
        dec.append(metrics(lab_d, y, pam, time, event, erbb2))
    rows["AE + KMeans(深度表示)"] = aek
    rows["AE + DEC(联合深度,L2)"] = dec

    # 4. PAM50 对照（生存）
    p_pam = logrank(time, event, y)[2]

    print(f"{'方法':<26}{'ARI':<14}{'NMI':<14}{'生存p':<14}{'Her2召回':<10}")
    print("-" * 78)
    for name, runs in rows.items():
        a = np.array(runs)
        m, sd = a.mean(0), a.std(0)
        def cell(i): return f"{m[i]:.3f}±{sd[i]:.3f}" if len(a) > 1 else f"{m[i]:.3f}"
        print(f"{name:<26}{cell(0):<14}{cell(1):<14}{cell(2):<14}{cell(3):<10}")
    print(f"{'PAM50(对照)':<26}{'—':<14}{'—':<14}{f'{p_pam:.3f}':<14}{'1.000':<10}")

    print("\n[诚实判读]")
    base_ari = np.mean(rows['sklearn 弱基线(Ward/PCA)'], 0)[0]
    dec_ari = np.mean(rows['AE + DEC(联合深度,L2)'], 0)[0]
    print(f"  深度 L2(DEC) ARI={dec_ari:.3f} vs sklearn 弱基线 ARI={base_ari:.3f}："
          f"{'深度更好' if dec_ari > base_ari + 0.01 else ('基本持平' if abs(dec_ari-base_ari)<=0.01 else '深度更差')}。")
    print("  ARI 看与 PAM50 一致性；生存 p 看预后区分；Her2 召回看驱动亚型。三者综合判断深度是否真带来提升。")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({n: [tuple(np.mean(r, 0))] for n, r in rows.items()}).to_csv(
        OUT_DIR / "deep_kernel.csv", index=False)
    print(f"\n[输出] 已存 {OUT_DIR}")


if __name__ == "__main__":
    main()
