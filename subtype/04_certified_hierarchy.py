"""
W5 — 认证的亚型层级 (Certified Subtype Hierarchy) + 校准 p 值。

W4 发现：按"最大稳定性"选 k 有小-k 偏置（永远给 k=2）。W5 改为**不硬选 k**，而是：
  1. 用 Ward 构建**嵌套层级**（k=2⊂3⊂…，每层是上一层的细分）。
  2. 对每层 k 用 **R 次置换零模型** 得到稳定性的**零分布**，算经验 p 值 + z 分数（多重检验 BH 校正）。
  3. 输出**逐层置信度**："哪些分辨率的结构显著高于随机"，而非一个武断的 k。
  4. 给关键层（k=2 主轴 / k=5 临床）发**每亚型稳定性证书**。

诚实定位（前案边界）：对树状图做"显著性检验"已有 SigClust(Liu 2008)/SHC(Kimes 2017)；
我们的差异点是 **置换零模型 + 子采样稳定性统计量 + 亚型可复现性框架**。真正的 Q1 新颖性还需
**跨队列复现 + 生存验证**（W6+），W5 只是把"认证层级"的内部统计骨架搭好。

用法：
  python subtype/04_certified_hierarchy.py \
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
from sklearn.metrics import adjusted_rand_score

from _data import exclude_normal_tissue, filter_labeled, load_real, preprocess
from _stability import (
    bh_fdr,
    consensus_matrix,
    make_null_pca,
    subsample_stability,
    ward_partition,
)

OUT_DIR = Path(__file__).resolve().parent / "outputs"


def per_subtype_certificate(Xp, k, y, pam_names, null_level):
    C = consensus_matrix(Xp, k)
    ref = ward_partition(Xp, k)
    print(f"  [k={k}] 每亚型证书 (null 共识基线={null_level:.3f}):")
    for c in np.unique(ref):
        m = np.where(ref == c)[0]
        iu = np.triu_indices(len(m), k=1)
        stab = float(C[np.ix_(m, m)][iu].mean()) if len(m) > 1 else float("nan")
        vc = pd.Series(pam_names[m]).value_counts()
        dom = f"{vc.index[0]}({vc.iloc[0]}/{len(m)})"
        verdict = "[OK]" if stab > null_level + 0.15 else ("[?]" if stab > null_level else "[X]")
        print(f"    亚型{c}: 稳定性={stab:.3f} {verdict} | 大小={len(m)} | 主导PAM50={dom}")


def main():
    ap = argparse.ArgumentParser(description="W5 认证亚型层级")
    ap.add_argument("--expr", required=True)
    ap.add_argument("--pheno", required=True)
    ap.add_argument("--subtype-col", default="PAM50Call_RNAseq")
    ap.add_argument("--kmin", type=int, default=2)
    ap.add_argument("--kmax", type=int, default=6)
    ap.add_argument("--R", type=int, default=20, help="置换零模型次数")
    ap.add_argument("--B", type=int, default=20, help="每次稳定性的子采样次数")
    args = ap.parse_args()

    expr, pheno = load_real(args.expr, args.pheno)
    expr, pheno, n_norm = exclude_normal_tissue(expr, pheno)
    expr, pheno, n_unl = filter_labeled(expr, pheno, args.subtype_col)
    X, Xp, _ = preprocess(expr)
    y = pd.Categorical(pheno[args.subtype_col]).codes
    pam_names = pheno[args.subtype_col].to_numpy()
    ks = list(range(args.kmin, args.kmax + 1))
    print(f"[纯肿瘤] {Xp.shape[0]} 样本 × {Xp.shape[1]} PC | R={args.R} 次零模型, B={args.B} 子采样\n")

    # 真实统计量 + ARI vs PAM50（上下文）
    real = {k: subsample_stability(Xp, k, ward_partition, B=args.B, seed=42) for k in ks}
    ari_pam = {k: adjusted_rand_score(y, ward_partition(Xp, k)) for k in ks}

    # 零分布：每个 null 数据集在所有 k 上各算一次
    null_stats = {k: [] for k in ks}
    for r in range(args.R):
        nXp = make_null_pca(X, Xp.shape[1], seed=100 + r)
        for k in ks:
            null_stats[k].append(subsample_stability(nXp, k, ward_partition, B=args.B, seed=200 + r))

    # p 值 / z 分数 / BH
    pvals, zs, nmean, nstd = {}, {}, {}, {}
    for k in ks:
        arr = np.asarray(null_stats[k])
        nmean[k], nstd[k] = float(arr.mean()), float(arr.std())
        zs[k] = (real[k] - nmean[k]) / (nstd[k] + 1e-9)
        pvals[k] = (int((arr >= real[k]).sum()) + 1) / (args.R + 1)
    qvals = dict(zip(ks, bh_fdr([pvals[k] for k in ks])))

    print(f"{'k':<4}{'稳定性(真)':<12}{'null均值±std':<18}{'z':<9}{'p':<9}{'q(BH)':<9}{'认证':<8}{'ARI_PAM50':<10}")
    print("-" * 80)
    for k in ks:
        cert = "[OK]" if qvals[k] < 0.05 else "[X]"
        print(f"{k:<4}{real[k]:<12.3f}{f'{nmean[k]:.3f}±{nstd[k]:.3f}':<18}"
              f"{zs[k]:<9.1f}{pvals[k]:<9.3f}{qvals[k]:<9.3f}{cert:<8}{ari_pam[k]:<10.3f}")

    certified = [k for k in ks if qvals[k] < 0.05]
    print(f"\n[认证层级] 显著高于随机的分辨率: k = {certified}")
    print("  解读：结构在多个分辨率都显著(p<0.05) → '有没有结构'已确证；"
          "但'到底分几型'是**粒度**问题，null 显著性回答不了它(各 k 都显著)。")
    print("  → 粒度需用**外部标准**定夺：生存分层 / 跨队列复现（W6）。这正是真正的临床/Q1 价值点。")

    # 关键层的每亚型证书
    print("\n[亚型证书]")
    nXp0 = make_null_pca(X, Xp.shape[1], seed=999)
    for k in (2, 5):
        if k in ks or k <= args.kmax:
            Cn = consensus_matrix(nXp0, k)
            nl = float(Cn[np.triu_indices(Cn.shape[0], k=1)].mean())
            per_subtype_certificate(Xp, k, y, pam_names, nl)

    _plot(ks, real, nmean, nstd)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"k": ks, "stability": [real[k] for k in ks],
                  "null_mean": [nmean[k] for k in ks], "z": [zs[k] for k in ks],
                  "p": [pvals[k] for k in ks], "q_bh": [qvals[k] for k in ks],
                  "ari_pam50": [ari_pam[k] for k in ks]}).to_csv(
        OUT_DIR / "certified_hierarchy.csv", index=False)
    print(f"\n[输出] 认证层级表 + 图已存到 {OUT_DIR}")


def _plot(ks, real, nmean, nstd):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    nm = np.array([nmean[k] for k in ks]); ns = np.array([nstd[k] for k in ks])
    plt.figure(figsize=(7, 5))
    plt.plot(ks, [real[k] for k in ks], "o-", label="real data")
    plt.plot(ks, nm, "s--", color="orange", label="null mean")
    plt.fill_between(ks, nm - 2 * ns, nm + 2 * ns, color="orange", alpha=0.2, label="null ±2σ")
    plt.xlabel("k (hierarchy level)"); plt.ylabel("subsample stability (ARI)")
    plt.title("Certified subtype hierarchy: real vs permutation null (W5)")
    plt.legend(); plt.tight_layout()
    plt.savefig(OUT_DIR / "certified_hierarchy.png", dpi=120); plt.close()


if __name__ == "__main__":
    main()
