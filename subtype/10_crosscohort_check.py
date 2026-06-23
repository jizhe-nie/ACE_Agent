"""
W-验证① 跨队列最小验证实验：TCGA-BRCA ↔ METABRIC 的平台漂移有多严重 + 现有简单方法跨队列差多少。

目的（可行性研判的"先验证靶再建模"）：
  - 若"队列内准 vs 跨队列准"的差距**大** → 跨队列稳健是真问题、靶值得全力打。
  - 若差距**小** → 已被解决，换靶。

方法（不建大模型，先用最简单、可解释的基线）：
  - 共同基因 → 各队列内对每个基因 z-score（消一阶平台尺度差）。
  - **最近质心分类器(NearestCentroid)** 预测 PAM50（这正是 PAM50 本身的原理）。
  - 队列内 5 折 CV 准确率/macro-F1 = 上限基线；TCGA→METABRIC 与反向 = 跨队列。
  - 差距 = 队列内 − 跨队列 = 平台漂移的代价。
  - 可视化：合并 PCA 分别按"队列"和"亚型"着色——若按队列分得比按亚型还开，说明平台效应主导。

环境：必须 `conda run -n Tumor_Subtype_Agent python subtype/10_crosscohort_check.py ...`
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
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestCentroid
from sklearn.preprocessing import StandardScaler

from _data import exclude_normal_tissue, filter_labeled, load_real

OUT_DIR = Path(__file__).resolve().parent / "outputs"
CANON = {"luma": "LumA", "lumb": "LumB", "her2": "Her2", "basal": "Basal", "normal": "Normal"}


def norm_labels(s):
    return pd.Series(s, index=getattr(s, "index", None)).astype(str).str.strip().str.lower().map(CANON)


def load_tcga(expr_path, pheno_path, col):
    expr, pheno = load_real(expr_path, pheno_path)
    expr, pheno, _ = exclude_normal_tissue(expr, pheno)
    expr, pheno, _ = filter_labeled(expr, pheno, col)
    lab = norm_labels(pheno[col])
    keep = lab.notna()
    return expr.loc[lab.index[keep]], lab[keep]


def load_metabric(expr_path, clin_path, col="CLAUDIN_SUBTYPE"):
    cl = pd.read_csv(clin_path, sep="\t", skiprows=4).set_index("PATIENT_ID")
    lab = norm_labels(cl[col]); lab = lab[lab.notna()]
    ex = pd.read_csv(expr_path, sep="\t")
    ex = ex.drop(columns=[c for c in ("Entrez_Gene_Id",) if c in ex.columns])
    ex = ex.set_index("Hugo_Symbol")
    ex = ex[~ex.index.duplicated(keep="first")].T  # samples × genes
    common = ex.index.intersection(lab.index)
    return ex.loc[common], lab.loc[common]


def zscore_genes(df):
    return pd.DataFrame(StandardScaler().fit_transform(df.to_numpy(float)),
                        index=df.index, columns=df.columns).fillna(0.0)


def within_cv(X, y, seed=42):
    skf = StratifiedKFold(5, shuffle=True, random_state=seed)
    accs, f1s = [], []
    yv = y.to_numpy()
    for tr, te in skf.split(X, yv):
        nc = NearestCentroid().fit(X[tr], yv[tr])
        pred = nc.predict(X[te])
        accs.append(accuracy_score(yv[te], pred))
        f1s.append(f1_score(yv[te], pred, average="macro"))
    return float(np.mean(accs)), float(np.mean(f1s))


def cross(Xtr, ytr, Xte, yte):
    nc = NearestCentroid().fit(Xtr, ytr.to_numpy())
    pred = nc.predict(Xte)
    return accuracy_score(yte.to_numpy(), pred), f1_score(yte.to_numpy(), pred, average="macro")


def main():
    ap = argparse.ArgumentParser(description="跨队列最小验证")
    ap.add_argument("--tcga-expr", required=True); ap.add_argument("--tcga-pheno", required=True)
    ap.add_argument("--tcga-col", default="PAM50Call_RNAseq")
    ap.add_argument("--mb-expr", required=True); ap.add_argument("--mb-clin", required=True)
    args = ap.parse_args()

    print("[载入] TCGA …"); Xt_df, yt = load_tcga(args.tcga_expr, args.tcga_pheno, args.tcga_col)
    print("[载入] METABRIC …"); Xm_df, ym = load_metabric(args.mb_expr, args.mb_clin)
    genes = Xt_df.columns.intersection(Xm_df.columns)
    print(f"[共同基因] {len(genes)} | TCGA {len(yt)} 例 {dict(yt.value_counts())}")
    print(f"                      METABRIC {len(ym)} 例 {dict(ym.value_counts())}")

    Xt = zscore_genes(Xt_df[genes]); Xm = zscore_genes(Xm_df[genes])
    Xt_np, Xm_np = Xt.to_numpy(), Xm.to_numpy()

    # 队列内基线
    at, ft = within_cv(Xt_np, yt); am, fm = within_cv(Xm_np, ym)
    print(f"\n[队列内 5折CV] TCGA: acc={at:.3f} macroF1={ft:.3f} | METABRIC: acc={am:.3f} macroF1={fm:.3f}")

    # 跨队列
    a_tm, f_tm = cross(Xt_np, yt, Xm_np, ym)   # TCGA→METABRIC
    a_mt, f_mt = cross(Xm_np, ym, Xt_np, yt)   # METABRIC→TCGA
    print(f"[跨队列] TCGA→METABRIC: acc={a_tm:.3f} macroF1={f_tm:.3f}")
    print(f"[跨队列] METABRIC→TCGA: acc={a_mt:.3f} macroF1={f_mt:.3f}")

    within = (at + am) / 2; crossm = (a_tm + a_mt) / 2
    gap = within - crossm
    print(f"\n[差距] 队列内均值 {within:.3f} − 跨队列均值 {crossm:.3f} = **{gap:.3f}**")
    if gap > 0.15:
        print("  → 平台漂移代价大(>0.15)：跨队列稳健是**真问题**，靶值得全力打。")
    elif gap > 0.07:
        print("  → 中等差距：有提升空间，靶可做。")
    else:
        print("  → 差距小：简单方法已跨得不错，需重审靶是否还有创新空间。")

    _plot(Xt_np, Xm_np, yt, ym)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"within_tcga": at, "within_metabric": am,
                   "cross_t2m": a_tm, "cross_m2t": a_mt, "gap": gap}]).to_csv(
        OUT_DIR / "crosscohort_check.csv", index=False)
    print(f"\n[输出] 已存 {OUT_DIR}")


def _plot(Xt, Xm, yt, ym):
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    except Exception:
        return
    X = np.vstack([Xt, Xm])
    cohort = np.array(["TCGA"] * len(Xt) + ["METABRIC"] * len(Xm))
    sub = np.concatenate([yt.to_numpy(), ym.to_numpy()])
    Z = PCA(2, random_state=42).fit_transform(StandardScaler().fit_transform(X))
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    for c in np.unique(cohort):
        m = cohort == c; ax[0].scatter(Z[m, 0], Z[m, 1], s=6, alpha=0.5, label=c)
    ax[0].set_title("PCA colored by COHORT (separation=platform shift)"); ax[0].legend()
    for c in np.unique(sub):
        m = sub == c; ax[1].scatter(Z[m, 0], Z[m, 1], s=6, alpha=0.5, label=c)
    ax[1].set_title("PCA colored by SUBTYPE (PAM50)"); ax[1].legend(fontsize=8)
    for a in ax: a.set_xlabel("PC1"); a.set_ylabel("PC2")
    fig.tight_layout(); fig.savefig(OUT_DIR / "crosscohort_pca.png", dpi=120); plt.close(fig)
    print(f"[图] 平台漂移 PCA 已存 {OUT_DIR/'crosscohort_pca.png'}")


if __name__ == "__main__":
    main()
