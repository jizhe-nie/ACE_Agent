"""
W3 — L1 确定性内核基线（IP-3 多原型/medoid 雏形）+ 严谨多 seed 评估 + 稳定性探针（IP-1 预告）。

底座优先（lessons §4）：这是"要打败的基线"，不是最终方法。最终的 IP-1（稳定性认证）
+ IP-3（端到端 L2 可学习原型）将在此之上构建。

对比 4 法（均设 K=5 对齐 PAM50 五型）：
  1. KMeans            — 质心法（质心可能落在样本不存在的空白处，lessons §5 的"质心陷阱"）
  2. Ward 层次          — 确定性，无随机
  3. K-medoid（单原型） — 每簇 1 个 medoid=真实病人样本，永不落空白，可解释
  4. 多原型-medoid(IP-3)— 过聚类成 M 个微簇→取微簇 medoid→Ward 合并到 K；每个亚型=一组 medoid
                          （密区自然少原型、长条/流形自然多原型；全程用真实样本，最可解释）

评估（lessons §6 严谨性）：
  - 多 seed（随机方法跑 5 个种子）报 ARI/NMI vs PAM50 的 mean±std（单跑数字不可信）。
  - 子采样稳定性探针：重复抽 80% 样本聚类，与全量结果对齐度 → IP-1 的雏形。
  - TCGA-BRCA 每样本=一位病人，故样本级≈患者级，无明显分组泄漏（仍以 sampleID 去重为准）。

用法：
  python subtype/02_kernel_baseline.py \
      --expr data/brca/HiSeqV2.gz --pheno data/brca/BRCA_clinicalMatrix \
      --subtype-col PAM50Call_RNAseq
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
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    pairwise_distances,
)

from _data import filter_labeled, load_real, preprocess

OUT_DIR = Path(__file__).resolve().parent / "outputs"


# --- medoid 工具：簇内"到其它点总距离最小"的真实样本(全局下标) ---
def medoid_of(Xp: np.ndarray, idx: np.ndarray) -> int:
    sub = Xp[idx]
    D = pairwise_distances(sub)
    return int(idx[np.argmin(D.sum(axis=1))])


def lab_kmeans(Xp, k, seed):
    return KMeans(k, n_init=10, random_state=seed).fit_predict(Xp)


def lab_ward(Xp, k, seed=None):
    return AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(Xp)


def lab_kmedoid(Xp, k, seed):
    """KMeans 初分 → 每簇取 medoid → 按最近 medoid 重新分配。单原型、可解释。"""
    lab0 = KMeans(k, n_init=10, random_state=seed).fit_predict(Xp)
    medoids = [medoid_of(Xp, np.where(lab0 == c)[0]) for c in range(k) if (lab0 == c).any()]
    D = pairwise_distances(Xp, Xp[medoids])
    return D.argmin(axis=1)


def lab_multiproto(Xp, k, seed, M=40):
    """IP-3：过聚类 M 微簇 → 微簇 medoid → Ward 合并到 K → 样本归最近微簇 medoid 的亚型。"""
    micro = KMeans(M, n_init=5, random_state=seed).fit_predict(Xp)
    medoid_idx = np.array([medoid_of(Xp, np.where(micro == c)[0])
                           for c in range(M) if (micro == c).any()])
    merge = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(Xp[medoid_idx])
    nearest = pairwise_distances(Xp, Xp[medoid_idx]).argmin(axis=1)
    return merge[nearest], medoid_idx, merge


def score(labels, y):
    return adjusted_rand_score(y, labels), normalized_mutual_info_score(y, labels)


def multiseed(fn, Xp, k, y, seeds):
    aris, nmis = [], []
    for s in seeds:
        lab = fn(Xp, k, s)
        a, n = score(lab, y)
        aris.append(a); nmis.append(n)
    return np.mean(aris), np.std(aris), np.mean(nmis), np.std(nmis)


def stability_probe(Xp, k, y, seed=0, frac=0.8, T=15):
    """IP-1 预告：抽 frac 子集聚类，与全量结果在该子集上的一致性(ARI)。越高越稳。"""
    full, _, _ = lab_multiproto(Xp, k, seed)
    rng = np.random.RandomState(seed)
    n = Xp.shape[0]
    agrees = []
    for t in range(T):
        sub = rng.choice(n, int(frac * n), replace=False)
        sub_lab, _, _ = lab_multiproto(Xp[sub], k, seed + t + 1)
        agrees.append(adjusted_rand_score(full[sub], sub_lab))
    return float(np.mean(agrees)), float(np.std(agrees))


def main():
    ap = argparse.ArgumentParser(description="W3 L1 内核基线")
    ap.add_argument("--expr", required=True)
    ap.add_argument("--pheno", required=True)
    ap.add_argument("--subtype-col", default="PAM50Call_RNAseq")
    ap.add_argument("--k", type=int, default=0, help="0=用 PAM50 类别数")
    ap.add_argument("--seeds", type=int, default=5)
    args = ap.parse_args()

    expr, pheno = load_real(args.expr, args.pheno)
    expr, pheno, n_drop = filter_labeled(expr, pheno, args.subtype_col)
    if n_drop:
        print(f"[过滤] 丢弃 {n_drop} 个无 {args.subtype_col} 标签样本，保留 {len(pheno)}。")
    _, Xp, _ = preprocess(expr)
    y = pd.Categorical(pheno[args.subtype_col]).codes
    k = args.k or pheno[args.subtype_col].nunique()
    seeds = list(range(args.seeds))
    print(f"[数据] {Xp.shape[0]} 样本 × {Xp.shape[1]} PC | K={k} | seeds={seeds}")
    print(f"[标签] {dict(pheno[args.subtype_col].value_counts())}\n")

    print(f"{'方法':<22}{'ARI (mean±std)':<22}{'NMI (mean±std)':<22}")
    print("-" * 66)
    rows = {
        "KMeans (质心)": lab_kmeans,
        "Ward (层次)": lab_ward,
        "K-medoid (单原型)": lab_kmedoid,
    }  # 多原型返回多值，单独在下方算
    for name, fn in rows.items():
        am, as_, nm, ns = multiseed(fn, Xp, k, y, seeds if fn is not lab_ward else [0])
        tag = "" if fn is not lab_ward else " (确定性)"
        print(f"{name:<22}{f'{am:.3f} ± {as_:.3f}':<22}{f'{nm:.3f} ± {ns:.3f}':<22}{tag}")

    # 多原型单独多 seed(它返回多值)
    mp_aris, mp_nmis = [], []
    for s in seeds:
        lab, _, _ = lab_multiproto(Xp, k, s)
        a, n = score(lab, y); mp_aris.append(a); mp_nmis.append(n)
    print(f"{'多原型-medoid (IP-3)':<22}"
          f"{f'{np.mean(mp_aris):.3f} ± {np.std(mp_aris):.3f}':<22}"
          f"{f'{np.mean(mp_nmis):.3f} ± {np.std(mp_nmis):.3f}':<22}")

    # IP-1 稳定性探针
    sm, ss = stability_probe(Xp, k, y)
    print(f"\n[IP-1 稳定性探针] 多原型内核子采样一致性 ARI = {sm:.3f} ± {ss:.3f}  (越高越稳)")

    # IP-3 可解释性：每个亚型的 medoid 范例病人 + 其 PAM50
    print("\n[IP-3 可解释性] 各亚型的 medoid 范例病人(真实样本) 及其 PAM50：")
    lab, medoid_idx, merge = lab_multiproto(Xp, k, 0)
    ids = pheno.index.to_numpy()
    pam = pheno[args.subtype_col].to_numpy()
    for c in range(k):
        mids = medoid_idx[merge == c]
        if len(mids) == 0:
            continue
        exemplars = [(ids[i], pam[i]) for i in mids[:4]]
        share = (lab == c).mean()
        ex_str = ", ".join(f"{sid}({p})" for sid, p in exemplars)
        print(f"  亚型{c} (占比{share:.0%}, {len(mids)}个原型): {ex_str}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "kernel_baseline_assignments.csv"
    pd.DataFrame({"sampleID": ids, "PAM50": pam, "multiproto_subtype": lab}).to_csv(out, index=False)
    print(f"\n[输出] 多原型亚型分配已存: {out}")


if __name__ == "__main__":
    main()
