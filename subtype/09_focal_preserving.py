"""
W10 — 焦点保留的深度多组学融合 (Focal-Preserving DEC, FP-DEC)。本项目第一个**真创新机制**。

问题（W8 朴素融合 + W9 深度融合都暴露）：融合会把**驱动焦点事件**（如 ERBB2 单基因扩增=Her2 型）
淹没在 CNV 的广谱方差里 → Her2 召回从 0.75 掉到 0.43~0.50。

机制（创新点）：在 AE+DEC 主干上加一条**专用焦点通路**——
  - 选 focal CNV 基因（|GISTIC|≥1 的样本占比高 = 在子集里强扩增/缺失的驱动焦点）。
  - 单独 focal_enc/focal_dec 编码+重构这些焦点基因（强制潜表示保住它们）。
  - **聚类空间 = 主潜向量 z ⊕ 焦点嵌入**（焦点直达聚类，不被 2000 基因广谱方差平均掉）。
预期：在保住 W9 生存增益的同时，**拉回 Her2 召回 + 提升 ARI**。

诚实对照（同主干、同超参、3 seeds）：DEC(无焦点, =W9) vs FP-DEC(焦点保留)。指标 ARI/NMI/生存/Her2 召回。
**GPU 自适应**：装了 CUDA 版 torch 即自动用 GPU。结果好坏都直说。

用法：
  python subtype/09_focal_preserving.py --expr data/brca/HiSeqV2.gz --cnv data/brca/CNV_gistic2.gz \
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
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import StandardScaler

from _data import exclude_normal_tissue, filter_labeled, load_real, top_mad_genes
from _survival import build_os, logrank

OUT_DIR = Path(__file__).resolve().parent / "outputs"
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def std_tensor(df, genes):
    return torch.tensor(StandardScaler().fit_transform(df[genes].to_numpy(float)), dtype=torch.float32)


class FPModel(nn.Module):
    """多组学 AE (+ 可选焦点通路)。use_focal=False 时退化为 W9 的 DEC 主干。"""
    def __init__(self, de, dc, df, h=512, emb=128, z=64, fz=16, use_focal=True):
        super().__init__()
        self.emb, self.use_focal = emb, use_focal
        self.enc_e = nn.Sequential(nn.Linear(de, h), nn.ReLU(), nn.Linear(h, emb), nn.ReLU())
        self.enc_c = nn.Sequential(nn.Linear(dc, h), nn.ReLU(), nn.Linear(h, emb), nn.ReLU())
        self.fuse = nn.Linear(2 * emb, z)
        self.defuse = nn.Sequential(nn.Linear(z, 2 * emb), nn.ReLU())
        self.dec_e = nn.Sequential(nn.Linear(emb, h), nn.ReLU(), nn.Linear(h, de))
        self.dec_c = nn.Sequential(nn.Linear(emb, h), nn.ReLU(), nn.Linear(h, dc))
        if use_focal:
            self.focal_enc = nn.Sequential(nn.Linear(df, 64), nn.ReLU(), nn.Linear(64, fz), nn.ReLU())
            self.focal_dec = nn.Sequential(nn.Linear(fz, 64), nn.ReLU(), nn.Linear(64, df))

    def latent(self, e, c, f):
        z = self.fuse(torch.cat([self.enc_e(e), self.enc_c(c)], 1))
        return torch.cat([z, self.focal_enc(f)], 1) if self.use_focal else z

    def forward(self, e, c, f):
        z = self.fuse(torch.cat([self.enc_e(e), self.enc_c(c)], 1))
        d = self.defuse(z)
        e2, c2 = self.dec_e(d[:, :self.emb]), self.dec_c(d[:, self.emb:])
        f2 = self.focal_dec(self.focal_enc(f)) if self.use_focal else None
        return e2, c2, f2


def train_ae(m, E, C, F, epochs=300, lr=1e-3):
    opt = torch.optim.Adam(m.parameters(), lr=lr); mse = nn.MSELoss()
    for _ in range(epochs):
        opt.zero_grad()
        e2, c2, f2 = m(E, C, F)
        loss = mse(e2, E) + mse(c2, C) + (mse(f2, F) if f2 is not None else 0.0)
        loss.backward(); opt.step()
    return m


def soft_assign(z, mu):
    q = 1.0 / (1.0 + torch.cdist(z, mu) ** 2)
    return q / q.sum(1, keepdim=True)


def dec_refine(m, E, C, F, k, seed, iters=200, update_p=20, gamma=0.5, lr=1e-4):
    with torch.no_grad():
        z0 = m.latent(E, C, F).cpu().numpy()
    km = KMeans(k, n_init=10, random_state=seed).fit(z0)
    mu = nn.Parameter(torch.tensor(km.cluster_centers_, dtype=torch.float32, device=DEV))
    opt = torch.optim.Adam(list(m.parameters()) + [mu], lr=lr); mse = nn.MSELoss()
    p = None
    for it in range(iters):
        zc = m.latent(E, C, F)
        q = soft_assign(zc, mu)
        if it % update_p == 0:
            w = (q ** 2) / q.sum(0, keepdim=True)
            p = (w / w.sum(1, keepdim=True)).detach()
        e2, c2, f2 = m(E, C, F)
        recon = mse(e2, E) + mse(c2, C) + (mse(f2, F) if f2 is not None else 0.0)
        loss = (p * torch.log((p + 1e-8) / (q + 1e-8))).sum(1).mean() + gamma * recon
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        return soft_assign(m.latent(E, C, F), mu).argmax(1).cpu().numpy()


def metrics(lab, y, pam, time, event):
    best = max(((pam[lab == c] == "Her2").sum() for c in np.unique(lab)), default=0)
    return (adjusted_rand_score(y, lab), normalized_mutual_info_score(y, lab),
            logrank(time, event, lab)[2], best / max(1, int((pam == "Her2").sum())))


def main():
    ap = argparse.ArgumentParser(description="W10 焦点保留深度融合")
    ap.add_argument("--expr", required=True); ap.add_argument("--cnv", required=True)
    ap.add_argument("--pheno", required=True); ap.add_argument("--subtype-col", default="PAM50Call_RNAseq")
    ap.add_argument("--k", type=int, default=5); ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--n-focal", type=int, default=200)
    args = ap.parse_args()

    print(f"[device] {DEV}" + ("" if DEV == "cuda" else "  (当前 torch 为 CPU 版；装 CUDA torch 后自动用 GPU)"))
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

    # 焦点基因：用**高幅事件**(|GISTIC|>=2, GISTIC 高水平扩增/缺失阈值)的样本占比。
    # 关键修正(W10 bug)：用 |CNV|>=1 会选中"频繁改变的臂级广谱基因"，而非焦点驱动——
    # ERBB2 只在 ~15% Her2 病人里高幅扩增，按"频率"反排到 #1826。高水平阈值才抓得住驱动扩增子。
    cnv_all = cnv.to_numpy(float)
    focality = (np.abs(cnv_all) >= 2).mean(0)
    focal_genes = cnv.columns[np.argsort(focality)[::-1][:args.n_focal]].tolist()
    erbb2_rank = int(np.where(np.argsort(focality)[::-1] == cnv.columns.get_loc("ERBB2"))[0][0]) if "ERBB2" in cnv.columns else -1
    print(f"[焦点基因] 选 {len(focal_genes)} 个 (|CNV|>=2 高幅事件占比最高)；ERBB2 焦点排名 #{erbb2_rank}"
          f" {'(入选)' if 'ERBB2' in focal_genes else '(未入选!)'}")

    E = std_tensor(expr, top_mad_genes(expr, 2000)).to(DEV)
    C = std_tensor(cnv, top_mad_genes(cnv, 2000)).to(DEV)
    F = std_tensor(cnv, focal_genes).to(DEV)
    print(f"[数据] {E.shape[0]} 例 | expr{E.shape[1]}+cnv{C.shape[1]}+focal{F.shape[1]} | 死亡{int(event.sum())} | k={args.k}\n")

    res = {"DEC (无焦点, =W9)": [], "FP-DEC (焦点保留)": []}
    for use_focal, name in [(False, "DEC (无焦点, =W9)"), (True, "FP-DEC (焦点保留)")]:
        for s in range(args.seeds):
            torch.manual_seed(s); np.random.seed(s)
            m = FPModel(E.shape[1], C.shape[1], F.shape[1], use_focal=use_focal).to(DEV)
            train_ae(m, E, C, F)
            lab = dec_refine(m, E, C, F, args.k, s)
            res[name].append(metrics(lab, y, pam, time, event))

    p_pam = logrank(time, event, y)[2]
    print(f"{'方法':<22}{'ARI':<14}{'NMI':<14}{'生存p':<14}{'Her2召回':<10}")
    print("-" * 74)
    for name, runs in res.items():
        a = np.array(runs); m_, sd = a.mean(0), a.std(0)
        print(f"{name:<22}{f'{m_[0]:.3f}±{sd[0]:.3f}':<14}{f'{m_[1]:.3f}±{sd[1]:.3f}':<14}"
              f"{f'{m_[2]:.3f}±{sd[2]:.3f}':<14}{f'{m_[3]:.3f}±{sd[3]:.3f}':<10}")
    print(f"{'PAM50(对照)':<22}{'—':<14}{'—':<14}{f'{p_pam:.3f}':<14}{'1.000':<10}")

    d = np.array(res["DEC (无焦点, =W9)"]).mean(0); f = np.array(res["FP-DEC (焦点保留)"]).mean(0)
    print("\n[诚实判读] FP-DEC vs DEC：")
    print(f"  Her2 召回 {d[3]:.3f} → {f[3]:.3f} ({'↑改善' if f[3]>d[3]+0.02 else ('持平' if abs(f[3]-d[3])<=0.02 else '↓变差')})")
    print(f"  ARI       {d[0]:.3f} → {f[0]:.3f} ({'↑' if f[0]>d[0]+0.01 else ('持平' if abs(f[0]-d[0])<=0.01 else '↓')}) | "
          f"生存p {d[2]:.3f} → {f[2]:.3f} ({'↑更预后' if f[2]<d[2]-0.01 else ('持平' if abs(f[2]-d[2])<=0.01 else '↓')})")
    print(f"  对照 PAM50 生存 p={p_pam:.3f}。结论：焦点机制是否同时救回 Her2 且不伤生存——见上。")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({n: [tuple(np.array(r).mean(0))] for n, r in res.items()}).to_csv(
        OUT_DIR / "focal_preserving.csv", index=False)
    print(f"\n[输出] 已存 {OUT_DIR}")


if __name__ == "__main__":
    main()
