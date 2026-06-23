"""
W4 — IP-1 稳定性认证子型（核心创新增量）+ 纯肿瘤基线。

动机（文献痛点）：癌症亚型的主流方法 consensus clustering(ConsensusClusterPlus) 被证明
**过度乐观**——即使数据没有真实结构(null)，它也会"发现"看似稳定的簇并给其背书
（Șenbabaoğlu et al. 2014 的著名批评）。根因：它不和**零模型**比较。

本脚本把"稳定性"从 W3 的事后探针升级为**带 null 校准的选择 + 认证**：
  1. 纯肿瘤：先剔除 -11 癌旁正常组织（见 _data.exclude_normal_tissue）。
  2. 稳定性曲线：对每个候选 k，做 B 次子采样聚类，量"子采样划分与全量划分的一致性(ARI)"= 稳定性。
  3. **null 校准**：把每个基因独立打乱(破坏多元结构、保留边际)→ 同样算稳定性曲线。
     真实稳定性必须**显著高于 null** 才算"有真结构"。gap = 真 - null 最大处即认证的 k*。
  4. **每亚型证书**：在 k* 上用共识矩阵算每个亚型的"成员共聚一致性"，低于 null 水平的亚型标记为"未认证/疑似伪亚型"。

用法：
  python subtype/03_stability_certified.py \
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
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score

from _data import exclude_normal_tissue, filter_labeled, load_real, preprocess

OUT_DIR = Path(__file__).resolve().parent / "outputs"
SEED = 42


def subsample_stability(Xp, k, B=40, frac=0.8, seed=SEED):
    """稳定性 = 子采样划分 vs 全量划分的平均一致性(ARI)。越高越稳。"""
    rng = np.random.RandomState(seed)
    ref = KMeans(k, n_init=10, random_state=seed).fit_predict(Xp)
    n = Xp.shape[0]
    aris = []
    for b in range(B):
        idx = rng.choice(n, int(frac * n), replace=False)
        lab = KMeans(k, n_init=5, random_state=1000 + b).fit_predict(Xp[idx])
        aris.append(adjusted_rand_score(ref[idx], lab))
    return float(np.mean(aris)), float(np.std(aris))


def make_null_pca(X_genes, n_pcs, seed=SEED):
    """零模型：每个基因(列)独立打乱→破坏多元/簇结构、保留各基因边际分布→PCA。"""
    rng = np.random.RandomState(seed)
    Xn = X_genes.copy()
    for j in range(Xn.shape[1]):
        rng.shuffle(Xn[:, j])
    return PCA(n_components=n_pcs, random_state=seed).fit_transform(Xn)


def consensus_matrix(Xp, k, B=40, frac=0.8, seed=SEED):
    """共识矩阵 C[i,j]=两样本在子采样里被分到同簇的频率(仅计两者都被抽中的次数)。"""
    n = Xp.shape[0]
    rng = np.random.RandomState(seed)
    co = np.zeros((n, n))
    cnt = np.zeros((n, n))
    for b in range(B):
        idx = rng.choice(n, int(frac * n), replace=False)
        lab = KMeans(k, n_init=5, random_state=2000 + b).fit_predict(Xp[idx])
        for c in np.unique(lab):
            members = idx[lab == c]
            co[np.ix_(members, members)] += 1
        cnt[np.ix_(idx, idx)] += 1
    with np.errstate(invalid="ignore", divide="ignore"):
        C = np.where(cnt > 0, co / cnt, 0.0)
    return C


def per_cluster_stability(Xp, k, C):
    """每个参考亚型的成员两两共识均值=该亚型稳定性证书。"""
    ref = KMeans(k, n_init=10, random_state=SEED).fit_predict(Xp)
    out = {}
    for c in np.unique(ref):
        m = np.where(ref == c)[0]
        if len(m) < 2:
            out[c] = (len(m), float("nan"))
            continue
        sub = C[np.ix_(m, m)]
        iu = np.triu_indices(len(m), k=1)
        out[c] = (len(m), float(sub[iu].mean()))
    return ref, out


def main():
    ap = argparse.ArgumentParser(description="W4 IP-1 稳定性认证子型")
    ap.add_argument("--expr", required=True)
    ap.add_argument("--pheno", required=True)
    ap.add_argument("--subtype-col", default="PAM50Call_RNAseq")
    ap.add_argument("--kmin", type=int, default=2)
    ap.add_argument("--kmax", type=int, default=7)
    ap.add_argument("--B", type=int, default=40)
    args = ap.parse_args()

    expr, pheno = load_real(args.expr, args.pheno)
    expr, pheno, n_norm = exclude_normal_tissue(expr, pheno)
    expr, pheno, n_unl = filter_labeled(expr, pheno, args.subtype_col)
    print(f"[纯肿瘤] 剔除 {n_norm} 个 -11 正常组织 + {n_unl} 个无标签样本，保留 {len(pheno)} 个肿瘤样本。")
    X, Xp, _ = preprocess(expr)
    y = pd.Categorical(pheno[args.subtype_col]).codes
    Xp_null = make_null_pca(X, Xp.shape[1])
    print(f"[数据] {Xp.shape[0]} 肿瘤样本 × {Xp.shape[1]} PC | PAM50: {dict(pheno[args.subtype_col].value_counts())}\n")

    # --- 纯肿瘤基线：与 W3(含正常组织)对比，看剔除后 ARI 变化 ---
    base = KMeans(5, n_init=10, random_state=SEED).fit_predict(Xp)
    print(f"[纯肿瘤基线] KMeans(k=5) vs PAM50 ARI = {adjusted_rand_score(y, base):.3f}\n")

    # --- null 校准的稳定性曲线 ---
    print(f"{'k':<4}{'真实稳定性':<16}{'null稳定性':<16}{'gap(真-null)':<14}")
    print("-" * 50)
    ks = list(range(args.kmin, args.kmax + 1))
    gaps = {}
    real_curve, null_curve = {}, {}
    for k in ks:
        rm, rs = subsample_stability(Xp, k, B=args.B)
        nm, ns = subsample_stability(Xp_null, k, B=args.B)
        gaps[k] = rm - nm
        real_curve[k], null_curve[k] = rm, nm
        print(f"{k:<4}{f'{rm:.3f}±{rs:.3f}':<16}{f'{nm:.3f}±{ns:.3f}':<16}{gaps[k]:<14.3f}")

    kstar = max(gaps, key=gaps.get)
    print(f"\n[认证 k*] gap 最大 → k* = {kstar} (gap={gaps[kstar]:.3f})")
    print("  解读：真实稳定性显著高于 null 才说明结构真实；gap 小=该 k 的'稳定'多半是 null 也有的假象。")

    # --- 每亚型稳定性证书 (k*) ---
    C = consensus_matrix(Xp, kstar, B=args.B)
    C_null = consensus_matrix(Xp_null, kstar, B=args.B)
    null_level = float(C_null[np.triu_indices(C_null.shape[0], k=1)].mean())
    ref, cl = per_cluster_stability(Xp, kstar, C)
    print(f"\n[每亚型证书] k*={kstar}，null 共识基线={null_level:.3f}（亚型稳定性需明显高于它才算认证）")
    ids = pheno.index.to_numpy()
    pam = pheno[args.subtype_col].to_numpy()
    for c, (sz, stab) in sorted(cl.items()):
        members = np.where(ref == c)[0]
        top_pam = pd.Series(pam[members]).value_counts()
        dom = f"{top_pam.index[0]}({top_pam.iloc[0]}/{sz})"
        verdict = "[OK]认证" if stab > null_level + 0.15 else ("[?]边界" if stab > null_level else "[X]未认证")
        print(f"  亚型{c}: 稳定性={stab:.3f} [{verdict}] | 大小={sz} | 主导PAM50={dom}")

    _plot(ks, real_curve, null_curve, kstar)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"sampleID": ids, "PAM50": pam, "subtype_kstar": ref}).to_csv(
        OUT_DIR / "stability_certified_assignments.csv", index=False)
    print(f"\n[输出] 认证分配 + 稳定性曲线图已存到 {OUT_DIR}")


def _plot(ks, real_curve, null_curve, kstar):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 5))
    plt.plot(ks, [real_curve[k] for k in ks], "o-", label="real data")
    plt.plot(ks, [null_curve[k] for k in ks], "s--", label="null (gene-permuted)")
    plt.axvline(kstar, color="r", ls=":", label=f"certified k*={kstar}")
    plt.xlabel("number of clusters k"); plt.ylabel("subsample stability (ARI)")
    plt.title("Null-calibrated stability for k selection (IP-1)")
    plt.legend(); plt.tight_layout()
    plt.savefig(OUT_DIR / "stability_curve.png", dpi=120); plt.close()


if __name__ == "__main__":
    main()
