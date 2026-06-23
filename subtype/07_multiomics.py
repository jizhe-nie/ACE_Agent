"""
W8 — 多组学融合（表达 + CNV）：能否分出 Her2 + 能否改善留出预后。

动机：
- W5 发现表达层分不出 Her2 型；W6/W7 发现表达层预后信号弱、预后感知会过拟合。
- **Her2 亚型的本质是 ERBB2(HER2) 基因的拷贝数扩增(CNV)** → CNV 层应能补上；
  CNV 还带"基因组不稳定"等预后信号。本轮先加 CNV（甲基化 450k 太大，留后续）。

融合：早融合(early integration)——表达、CNV 各自 选高变基因→标准化→PCA(30)→z-score 成块→拼接。
两个测试：
  1. Her2 救回：k=5 聚类，看各簇的 Her2 富集 + 平均 ERBB2 拷贝数（expr-only vs multi）。
  2. 留出预后：W7 患者级 train/test 协议，比 expr-only vs multi 的测试集 log-rank（方差基线，避免预后感知的过拟合）。

用法：
  python subtype/07_multiomics.py --expr data/brca/HiSeqV2.gz --cnv data/brca/CNV_gistic2.gz \
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
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler

from _data import exclude_normal_tissue, filter_labeled, load_real, preprocess, top_mad_genes
from _survival import build_os, logrank

OUT_DIR = Path(__file__).resolve().parent / "outputs"


def load_genes_by_samples(path):
    """读 基因×样本 矩阵 → 转成 样本×基因。"""
    df = pd.read_csv(path, sep="\t", index_col=0)
    return df.T  # 行=样本


def block_repr(df, n_pcs=30, top=2000, seed=42):
    """单组学块表示：选高变→标准化→PCA→z-score 成块（各组学等权拼接）。"""
    _, Xp, _ = preprocess(df, top_genes=top, n_pcs=n_pcs, seed=seed)
    return StandardScaler().fit_transform(Xp)


def fit_omics(df_tr, top=2000, n_pcs=30):
    genes = top_mad_genes(df_tr, top)
    sc = StandardScaler().fit(df_tr[genes].to_numpy(float))
    pca = PCA(n_components=min(n_pcs, len(df_tr) - 1, len(genes)), random_state=42).fit(
        sc.transform(df_tr[genes].to_numpy(float)))
    blk = StandardScaler().fit(pca.transform(sc.transform(df_tr[genes].to_numpy(float))))
    return genes, sc, pca, blk


def transform_omics(df, fitted):
    genes, sc, pca, blk = fitted
    return blk.transform(pca.transform(sc.transform(df[genes].to_numpy(float))))


def stratified_split(event, frac_train, seed):
    rng = np.random.RandomState(seed)
    tr, te = [], []
    for v in (0, 1):
        idx = np.where(event == v)[0]; rng.shuffle(idx)
        cut = int(frac_train * len(idx)); tr += list(idx[:cut]); te += list(idx[cut:])
    return np.array(sorted(tr)), np.array(sorted(te))


def main():
    ap = argparse.ArgumentParser(description="W8 多组学融合")
    ap.add_argument("--expr", required=True)
    ap.add_argument("--cnv", required=True)
    ap.add_argument("--pheno", required=True)
    ap.add_argument("--subtype-col", default="PAM50Call_RNAseq")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--R", type=int, default=5)
    args = ap.parse_args()

    expr, pheno = load_real(args.expr, args.pheno)
    cnv = load_genes_by_samples(args.cnv)
    # 三方样本对齐
    common = expr.index.intersection(cnv.index).intersection(pheno.index)
    expr, cnv, pheno = expr.loc[common], cnv.loc[common], pheno.loc[common]
    expr, pheno, _ = exclude_normal_tissue(expr, pheno)
    expr, pheno, _ = filter_labeled(expr, pheno, args.subtype_col)
    cnv = cnv.loc[pheno.index]
    time, event, valid = build_os(pheno)
    keep = pheno.index[valid]
    expr, cnv, pheno = expr.loc[keep], cnv.loc[keep], pheno.loc[keep]
    time, event = time[valid], event[valid]
    pam = pheno[args.subtype_col].to_numpy()
    erbb2 = cnv["ERBB2"].to_numpy(float) if "ERBB2" in cnv.columns else None
    print(f"[三组学对齐] {len(pheno)} 例(表达∩CNV∩临床, 纯肿瘤, 有生存) | 死亡 {int(event.sum())}")
    if erbb2 is not None:
        print(f"[ERBB2 拷贝数] 按 PAM50: " +
              ", ".join(f"{g}={erbb2[pam==g].mean():+.2f}" for g in pd.unique(pam)))
        print("  (Her2 型应显著高 → 印证 'Her2=ERBB2 扩增', CNV 携带此信号)\n")

    Xe = block_repr(expr); Xc = block_repr(cnv); Xm = np.hstack([Xe, Xc])

    # --- 测试 1: Her2 救回 ---
    print("=== 测试1: Her2 救回 (k=5，看最像 Her2 的簇) ===")
    for name, X in [("expr-only", Xe), ("multi-omics(+CNV)", Xm)]:
        lab = AgglomerativeClustering(n_clusters=args.k, linkage="ward").fit_predict(X)
        best = None
        for c in np.unique(lab):
            m = lab == c
            her2_frac = (pam[m] == "Her2").mean()
            erb = erbb2[m].mean() if erbb2 is not None else float("nan")
            if best is None or her2_frac > best[1]:
                best = (c, her2_frac, int((pam[m] == "Her2").sum()), int(m.sum()), erb)
        c, hf, hn, sz, erb = best
        total_her2 = int((pam == "Her2").sum())
        print(f"  {name:<20} 最佳Her2簇: 召回={hn}/{total_her2} 纯度={hf:.0%} 簇大小={sz} 平均ERBB2_CNV={erb:+.2f}")

    # --- 测试 2: 留出预后 (方差基线，expr vs multi) ---
    print(f"\n=== 测试2: 留出预后 (k={args.k}, {args.R} 次划分, 方差基线) ===")
    rec = {"expr-only(测试)": [], "multi-omics(测试)": []}
    for s in range(args.R):
        tr, te = stratified_split(event, 0.6, seed=s)
        fe = fit_omics(expr.iloc[tr]); fc = fit_omics(cnv.iloc[tr])
        Xtr_e = transform_omics(expr.iloc[tr], fe); Xte_e = transform_omics(expr.iloc[te], fe)
        Xtr_m = np.hstack([Xtr_e, transform_omics(cnv.iloc[tr], fc)])
        Xte_m = np.hstack([Xte_e, transform_omics(cnv.iloc[te], fc)])
        for nm, (Xtr, Xte) in {"expr-only(测试)": (Xtr_e, Xte_e),
                               "multi-omics(测试)": (Xtr_m, Xte_m)}.items():
            ref = AgglomerativeClustering(n_clusters=args.k, linkage="ward").fit_predict(Xtr)
            cents = np.vstack([Xtr[ref == c].mean(0) for c in range(args.k)])
            lab_te = pairwise_distances(Xte, cents).argmin(axis=1)
            rec[nm].append(logrank(time[te], event[te], lab_te)[2])

    print(f"{'方法':<22}{'测试集 log-rank p (mean±std)':<30}{'显著比例':<12}")
    print("-" * 64)
    for nm, ps in rec.items():
        ps = np.array(ps)
        print(f"{nm:<22}{f'{ps.mean():.3f} ± {ps.std():.3f}':<30}{f'{(ps<0.05).mean():.0%}':<12}")

    e_te, m_te = np.mean(rec["expr-only(测试)"]), np.mean(rec["multi-omics(测试)"])
    print("\n[解读]")
    print(f"  加 CNV 后留出预后 p: {e_te:.2e} → {m_te:.2e}（{'改善' if m_te < e_te else '未改善'}）。")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rec).to_csv(OUT_DIR / "multiomics_survival.csv", index=False)
    print(f"\n[输出] 已存 {OUT_DIR}")


if __name__ == "__main__":
    main()
