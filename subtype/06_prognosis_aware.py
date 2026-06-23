"""
W7 — 预后感知亚型发现 + 留出验证（B+A 路线）。

W6 负结果：方差驱动的无监督亚型稳定却不预后。W7 让聚类"知道预后"：先按生存挑基因再聚类。
**但这有致命陷阱**：在同一批数据上"按生存挑基因→聚类→再测生存"=循环论证(selection bias)，
等于在测试集调参(lessons §6 大忌)。**正确做法：患者级 train/test 划分**——
训练集挑预后基因+建亚型，**留出测试集**才测生存。本脚本对比三法在**留出测试集**上的生存区分力：
  1. variance（方差选基因，W6 老路，无监督）
  2. prognosis-aware（训练集按生存挑基因）—— 同时打印其"训练集 p"以暴露循环陷阱
  3. PAM50（临床金标准对照）
多次随机划分报 mean±std，看预后感知能否在**诚实的留出集**上真正分开生存。

用法：
  python subtype/06_prognosis_aware.py \
      --expr data/brca/HiSeqV2.gz --pheno data/brca/BRCA_clinicalMatrix --subtype-col PAM50Call_RNAseq
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
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler

from _data import exclude_normal_tissue, filter_labeled, load_real
from _survival import build_os, logrank

OUT_DIR = Path(__file__).resolve().parent / "outputs"


def stratified_split(event, frac_train, seed):
    """按事件分层的患者级划分，保证 train/test 都有死亡事件。"""
    rng = np.random.RandomState(seed)
    tr, te = [], []
    for val in (0, 1):
        idx = np.where(event == val)[0]
        rng.shuffle(idx)
        cut = int(frac_train * len(idx))
        tr += list(idx[:cut]); te += list(idx[cut:])
    return np.array(sorted(tr)), np.array(sorted(te))


def top_mad_genes(Xdf, n):
    X = Xdf.to_numpy(float)
    mad = np.median(np.abs(X - np.median(X, axis=0)), axis=0)
    return Xdf.columns[np.argsort(mad)[::-1][:n]].tolist()


def prognostic_genes(Xdf, time, event, prefilter=2000, topN=100):
    """训练集预后基因：先 MAD 预筛降噪，再对每个基因按中位数二分做 log-rank，取 p 最小的 topN。"""
    cand = top_mad_genes(Xdf, prefilter)
    X = Xdf[cand].to_numpy(float)
    ps = []
    for j in range(X.shape[1]):
        grp = (X[:, j] > np.median(X[:, j])).astype(int)
        ps.append(logrank(time, event, grp)[2] if len(np.unique(grp)) == 2 else 1.0)
    order = np.argsort(ps)[:topN]
    return [cand[j] for j in order]


def fit_repr(Xtr_df, genes, n_pcs=20):
    sc = StandardScaler().fit(Xtr_df[genes].to_numpy(float))
    Xtr = sc.transform(Xtr_df[genes].to_numpy(float))
    n_pcs = min(n_pcs, Xtr.shape[0] - 1, Xtr.shape[1])
    pca = PCA(n_components=n_pcs, random_state=42).fit(Xtr)
    return sc, pca, pca.transform(Xtr)


def assign_nearest(Xp_test, centroids):
    return pairwise_distances(Xp_test, centroids).argmin(axis=1)


def main():
    ap = argparse.ArgumentParser(description="W7 预后感知 + 留出验证")
    ap.add_argument("--expr", required=True)
    ap.add_argument("--pheno", required=True)
    ap.add_argument("--subtype-col", default="PAM50Call_RNAseq")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--R", type=int, default=5, help="随机划分次数")
    ap.add_argument("--frac-train", type=float, default=0.6)
    args = ap.parse_args()

    expr, pheno = load_real(args.expr, args.pheno)
    expr, pheno, _ = exclude_normal_tissue(expr, pheno)
    expr, pheno, _ = filter_labeled(expr, pheno, args.subtype_col)
    time, event, valid = build_os(pheno)
    expr, pheno = expr.loc[pheno.index[valid]], pheno.loc[pheno.index[valid]]
    time, event = time[valid], event[valid]
    pam = pd.Categorical(pheno[args.subtype_col]).codes
    print(f"[队列] {len(time)} 例 | 死亡 {int(event.sum())} | k={args.k} | {args.R} 次留出验证 "
          f"(train {args.frac_train:.0%})\n")

    rec = {"variance(测试集)": [], "prognosis-aware(测试集)": [],
           "prognosis-aware(训练集!)": [], "PAM50(测试集)": []}
    k = args.k
    for s in range(args.R):
        tr, te = stratified_split(event, args.frac_train, seed=s)
        Xtr_df, Xte_df = expr.iloc[tr], expr.iloc[te]
        t_tr, e_tr, t_te, e_te = time[tr], event[tr], time[te], event[te]

        for name, genes in [("variance", top_mad_genes(Xtr_df, 100)),
                            ("prognosis-aware", prognostic_genes(Xtr_df, t_tr, e_tr))]:
            sc, pca, Xp_tr = fit_repr(Xtr_df, genes)
            ref = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(Xp_tr)
            cents = np.vstack([Xp_tr[ref == c].mean(0) for c in range(k)])
            Xp_te = pca.transform(sc.transform(Xte_df[genes].to_numpy(float)))
            lab_te = assign_nearest(Xp_te, cents)
            rec[f"{name}(测试集)"].append(logrank(t_te, e_te, lab_te)[2])
            if name == "prognosis-aware":  # 暴露循环陷阱：训练集 p（同集挑同集测）
                lab_tr = assign_nearest(Xp_tr, cents)
                rec["prognosis-aware(训练集!)"].append(logrank(t_tr, e_tr, lab_tr)[2])
        rec["PAM50(测试集)"].append(logrank(t_te, e_te, pam[te])[2])

    print(f"{'方法':<26}{'log-rank p (mean±std)':<26}{'显著划分比例':<14}")
    print("-" * 66)
    for name, ps in rec.items():
        ps = np.array(ps)
        frac = float((ps < 0.05).mean())
        print(f"{name:<26}{f'{ps.mean():.3f} ± {ps.std():.3f}':<26}{f'{frac:.0%} ({int((ps<0.05).sum())}/{len(ps)})':<14}")

    print("\n[解读]")
    pa_tr = np.mean(rec["prognosis-aware(训练集!)"])
    pa_te = np.mean(rec["prognosis-aware(测试集)"])
    var_te = np.mean(rec["variance(测试集)"])
    pam_te = np.mean(rec["PAM50(测试集)"])
    print(f"  循环陷阱实证：预后感知 训练集 p≈{pa_tr:.1e}（虚低！）vs 测试集 p≈{pa_te:.1e}（诚实）。"
          "→ 必须留出验证，否则自欺。")
    if pa_te < var_te and pa_te < 0.05:
        print(f"  预后感知 在留出集上**真分开了生存**(p≈{pa_te:.1e})，且优于方差基线(p≈{var_te:.1e})。")
    elif pa_te < var_te:
        print(f"  预后感知 在留出集上优于方差基线(p≈{pa_te:.1e} < {var_te:.1e})，但未达显著——需多组学/更大样本。")
    else:
        print(f"  预后感知 在留出集上未超过方差基线(p≈{pa_te:.1e})——表达层预后信号有限，转 W8 多组学。")
    print(f"  对照 PAM50 测试集 p≈{pam_te:.1e}。")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({k2: v for k2, v in rec.items()}).to_csv(OUT_DIR / "prognosis_aware.csv", index=False)
    print(f"\n[输出] 留出验证表已存到 {OUT_DIR}")


if __name__ == "__main__":
    main()
