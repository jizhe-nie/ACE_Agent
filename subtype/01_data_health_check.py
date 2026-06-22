"""
W1 数据体检 (Data Health Check) — 亚型发现的第一步，也是 lessons §4「底座优先」的第 ① 步。

在做任何聚类/亚型方法之前，必须先回答三个问题（否则后面全是空中楼阁）：
  1. 可聚类性 (clusterability):  数据里到底有没有"成团"的结构？还是均匀噪声？
       —— 用 Hopkins 统计量。H≈0.5 = 随机无结构；H→1 = 有明显团块。
  2. 簇 vs 生物学:  数据自带的团块，是否对应已知的生物学亚型 (PAM50)？
       —— 用 ARI/NMI(聚类标签, PAM50)。高 = 结构有生物学意义，值得做。
  3. 簇 vs 批次 (batch):  团块是不是其实在按"实验批次/医院"分，而非生物学？
       —— 用 ARI/NMI(聚类标签, 批次)。高 = 危险！聚的是批次不是亚型，必须先做批次校正。
         （这就是 lessons §7：不处理批次，聚类会忠实地把批次当成亚型。）

用法：
  # 立刻可跑的合成数据演示（无需任何生物数据，用来看懂机制）：
  python subtype/01_data_health_check.py --demo

  # 真实数据（拿到 TCGA-BRCA 后）：
  python subtype/01_data_health_check.py \
      --expr data/brca/expression.tsv \
      --pheno data/brca/phenotype.tsv \
      --subtype-col PAM50 --batch-col TSS

数据格式约定（真实数据）：
  expression.tsv : 行=样本, 列=基因, 数值=表达量 (TSV, 第一列为样本 ID)
  phenotype.tsv  : 行=样本, 含一列亚型标签(如 PAM50) 和一列批次(如 TSS/plate)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Windows 控制台默认 GBK，会因中文/符号报 UnicodeEncodeError；强制 UTF-8 输出。
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

OUT_DIR = Path(__file__).resolve().parent / "outputs"


# ---------------------------------------------------------------------------
# Hopkins 统计量：衡量"可聚类性"。
# 直觉：从真实数据里抽 m 个点，量它们到最近邻的距离 w；再在数据空间里撒 m 个
# 随机均匀点，量它们到最近真实点的距离 u。若数据成团，则随机点离数据更远 → u 远大于 w
# → H = Σu / (Σu+Σw) → 1。若数据本身就是均匀噪声，u≈w → H≈0.5。
# ---------------------------------------------------------------------------
def hopkins(X: np.ndarray, m: int = 150, seed: int = 42) -> float:
    rng = np.random.RandomState(seed)
    n, d = X.shape
    m = min(m, n - 1)
    nbrs = NearestNeighbors(n_neighbors=2).fit(X)

    # w: 真实样本到其最近的"另一个真实样本"的距离
    idx = rng.choice(n, m, replace=False)
    w_dist, _ = nbrs.kneighbors(X[idx], n_neighbors=2)
    w = w_dist[:, 1]  # [:,0] 是自身(距离0)，取第二近

    # u: 随机均匀点到最近真实样本的距离
    mins, maxs = X.min(axis=0), X.max(axis=0)
    unif = rng.uniform(mins, maxs, size=(m, d))
    u_dist, _ = nbrs.kneighbors(unif, n_neighbors=1)
    u = u_dist[:, 0]

    denom = u.sum() + w.sum()
    return float(u.sum() / denom) if denom > 0 else 0.5


# ---------------------------------------------------------------------------
# 合成数据：5 个"生物学亚型"(类比 PAM50 的 5 型) + 3 个"批次"(类比 3 家医院)。
# 生物信号写在前 BIO_GENES 个基因上；批次信号写在另一段基因上(技术指纹)。
# batch_strength 调大 → 批次效应越重 → 体检应当亮红灯。
# ---------------------------------------------------------------------------
def load_demo(n: int = 500, g: int = 2000, batch_strength: float = 1.0, seed: int = 0):
    rng = np.random.RandomState(seed)
    n_sub, n_batch = 5, 3
    bio_genes, batch_genes = 120, 120

    X = rng.normal(0, 1, size=(n, g))
    subtype = rng.randint(0, n_sub, size=n)
    batch = rng.randint(0, n_batch, size=n)

    # 生物学信号：每个亚型在 bio_genes 段有独特的均值偏移
    sub_centers = rng.normal(0, 3.0, size=(n_sub, bio_genes))
    X[:, :bio_genes] += sub_centers[subtype]

    # 批次信号：每个批次在 batch_genes 段有独特的技术偏移(与生物学无关)
    batch_centers = rng.normal(0, 3.0 * batch_strength, size=(n_batch, batch_genes))
    X[:, bio_genes:bio_genes + batch_genes] += batch_centers[batch]

    expr = pd.DataFrame(X, index=[f"S{i:04d}" for i in range(n)],
                        columns=[f"gene{j}" for j in range(g)])
    pheno = pd.DataFrame({
        "PAM50": [f"Subtype{s}" for s in subtype],
        "TSS": [f"Site{b}" for b in batch],
    }, index=expr.index)
    return expr, pheno


def load_real(expr_path: str, pheno_path: str):
    expr = pd.read_csv(expr_path, sep="\t", index_col=0)
    pheno = pd.read_csv(pheno_path, sep="\t", index_col=0)
    # 样本对齐：表达矩阵与表型表取交集(顺序一致)，这是避免"标签错位"的基本功
    common = expr.index.intersection(pheno.index)
    if len(common) == 0:
        # 兼容"基因为行、样本为列"的转置存法
        expr = expr.T
        common = expr.index.intersection(pheno.index)
    expr, pheno = expr.loc[common], pheno.loc[common]
    return expr, pheno


# ---------------------------------------------------------------------------
# 预处理：log 稳定方差 → 选高变基因(信息量最大的) → 标准化 → PCA 降噪。
# 选高变基因 = 用 MAD(中位绝对偏差) 排序取前 top_genes 个。组学有 ~2 万基因，
# 多数是"沉默"或恒定的噪声，只有变化大的基因才携带亚型差异。
# ---------------------------------------------------------------------------
def preprocess(expr: pd.DataFrame, top_genes: int = 1000, n_pcs: int = 50):
    X = expr.to_numpy(dtype=float)
    X = np.nan_to_num(X, nan=np.nanmedian(X))
    # 若像计数/非负大值，做 log1p 稳定方差(RNA-seq 常规操作)
    if X.min() >= 0 and X.max() > 50:
        X = np.log1p(X)
    mad = np.median(np.abs(X - np.median(X, axis=0)), axis=0)
    keep = np.argsort(mad)[::-1][:min(top_genes, X.shape[1])]
    X = X[:, keep]
    X = StandardScaler().fit_transform(X)
    n_pcs = min(n_pcs, X.shape[0] - 1, X.shape[1])
    Xp = PCA(n_components=n_pcs, random_state=42).fit_transform(X)
    return X, Xp, PCA(n_components=n_pcs, random_state=42).fit(X).explained_variance_ratio_


def _assoc(labels, ref):
    """聚类标签与参考标签(亚型/批次)的关联强度：ARI + NMI。"""
    ref_codes = pd.Categorical(ref).codes
    return (adjusted_rand_score(ref_codes, labels),
            normalized_mutual_info_score(ref_codes, labels))


def run_checks(expr: pd.DataFrame, pheno: pd.DataFrame,
               subtype_col: str, batch_col: str, seed: int = 42) -> dict:
    # 只保留有金标准亚型标签的样本：PAM50 常缺失(正常旁组织/未分型)。
    # 三项检查在同一"有标签队列"上比较，ARI 才干净。
    if subtype_col in pheno.columns:
        valid = pheno[subtype_col].notna() & pheno[subtype_col].astype(str).str.strip().ne("")
        n_drop = int((~valid).sum())
        if n_drop:
            keep = pheno.index[valid]
            expr, pheno = expr.loc[keep], pheno.loc[keep]
            print(f"[过滤] 丢弃 {n_drop} 个无 {subtype_col} 标签样本，保留 {len(keep)} 个有金标准样本。")
    print(f"\n[数据] 样本数={expr.shape[0]}, 基因数={expr.shape[1]}")
    has_sub = subtype_col in pheno.columns
    has_batch = batch_col in pheno.columns
    if has_sub:
        print(f"[标签] {subtype_col} 亚型分布: {dict(pheno[subtype_col].value_counts())}")
    if has_batch:
        print(f"[批次] {batch_col} 批次分布: {dict(pheno[batch_col].value_counts())}")

    X_full, Xp, evr = preprocess(expr)

    # --- 检查 1: 可聚类性 ---
    H = hopkins(Xp, seed=seed)
    pc10 = float(evr[:10].sum())
    print("\n=== 检查 1: 可聚类性 (Hopkins) ===")
    print(f"  Hopkins H = {H:.3f}  (≈0.5 无结构 / →1 团块明显)")
    print(f"  前 10 个主成分累计方差 = {pc10:.1%}")
    clusterable = H > 0.65

    # 用与亚型数相同的 k 做一次探针聚类(仅用于体检，不是最终方法)
    k = pheno[subtype_col].nunique() if has_sub else 5
    labels = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(Xp)

    result = {"hopkins": H, "pc10_var": pc10, "k": k, "clusterable": clusterable}

    # --- 检查 2: 簇 vs 生物学亚型 ---
    if has_sub:
        ari_s, nmi_s = _assoc(labels, pheno[subtype_col])
        print("\n=== 检查 2: 簇 vs 生物学亚型 (越高越好) ===")
        print(f"  ARI(聚类, {subtype_col}) = {ari_s:.3f} | NMI = {nmi_s:.3f}")
        result.update(ari_subtype=ari_s, nmi_subtype=nmi_s)

    # --- 检查 3: 簇 vs 批次 ---
    if has_batch:
        ari_b, nmi_b = _assoc(labels, pheno[batch_col])
        print("\n=== 检查 3: 簇 vs 批次 (越低越好！) ===")
        print(f"  ARI(聚类, {batch_col}) = {ari_b:.3f} | NMI = {nmi_b:.3f}")
        result.update(ari_batch=ari_b, nmi_batch=nmi_b)

    # --- 体检结论 ---
    print("\n=== 体检结论 ===")
    # 复核可聚类性：有标签时以"与已知亚型的 ARI"为主判据。
    # 注意 Hopkins 在高维会被噪声基因稀释、系统性低估——不能只看它。
    ari_sub = result.get("ari_subtype", 0.0)
    clusterable = (H > 0.60) or (ari_sub > 0.2)
    result["clusterable"] = clusterable
    if not clusterable:
        print("  [!] 可聚类性弱：Hopkins 低且与已知亚型无对齐，做亚型前需复查特征选择/预处理。")
    elif H <= 0.60 and ari_sub > 0.2:
        print(f"  [OK] 数据可聚类：Hopkins={H:.2f} 偏低是高维稀释的假象，"
              f"但与已知亚型 ARI={ari_sub:.2f} 证明结构真实存在。")
    else:
        print("  [OK] 数据有团块结构，值得做亚型。")
    if has_sub and has_batch:
        bio = result.get("ari_subtype", 0)
        bat = result.get("ari_batch", 0)
        if bat > bio and bat > 0.1:
            print(f"  [STOP] 批次主导：ARI_批次({bat:.3f}) > ARI_亚型({bio:.3f})！"
                  "聚的是批次不是生物学，必须先做批次校正(ComBat/Harmony)再继续。")
        elif bat > 0.1:
            print(f"  [!] 存在批次相关(ARI_批次={bat:.3f})：生物学占优但需在方法里控制批次。")
        else:
            print(f"  [OK] 批次相关弱(ARI_批次={bat:.3f})：结构以生物学为主，可放心推进。")

    _try_plot(Xp, pheno, subtype_col, batch_col)
    return result


def _try_plot(Xp, pheno, subtype_col, batch_col):
    """画 PCA 散点：分别按亚型、按批次着色。一眼看出'点是按生物学还是按批次成团'。"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cols = [c for c in (subtype_col, batch_col) if c in pheno.columns]
    if not cols:
        return
    fig, axes = plt.subplots(1, len(cols), figsize=(6 * len(cols), 5))
    axes = np.atleast_1d(axes)
    for ax, col in zip(axes, cols):
        codes = pd.Categorical(pheno[col])
        for lvl in codes.categories:
            mask = (pheno[col].to_numpy() == lvl)
            ax.scatter(Xp[mask, 0], Xp[mask, 1], s=10, alpha=0.6, label=str(lvl))
        ax.set_title(f"PCA colored by {col}")
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.legend(fontsize=7)
    fig.tight_layout()
    out = OUT_DIR / "health_check_pca.png"
    fig.savefig(out, dpi=120); plt.close(fig)
    print(f"\n[图] 已保存 PCA 散点(按亚型/批次着色): {out}")


def main():
    ap = argparse.ArgumentParser(description="亚型发现 W1 数据体检")
    ap.add_argument("--demo", action="store_true", help="用合成数据演示(无需真实数据)")
    ap.add_argument("--demo-batchy", action="store_true", help="演示'批次主导'的坏情况")
    ap.add_argument("--expr", type=str, help="表达矩阵 TSV (样本×基因)")
    ap.add_argument("--pheno", type=str, help="表型 TSV (含亚型列与批次列)")
    ap.add_argument("--subtype-col", type=str, default="PAM50")
    ap.add_argument("--batch-col", type=str, default="TSS")
    args = ap.parse_args()

    if args.demo or args.demo_batchy:
        strength = 3.0 if args.demo_batchy else 0.25
        print(f"[模式] 合成数据演示 (batch_strength={strength})")
        expr, pheno = load_demo(batch_strength=strength)
    elif args.expr and args.pheno:
        print(f"[模式] 真实数据: {args.expr}")
        expr, pheno = load_real(args.expr, args.pheno)
    else:
        ap.error("请用 --demo 演示，或同时给 --expr 和 --pheno")

    run_checks(expr, pheno, args.subtype_col, args.batch_col)


if __name__ == "__main__":
    main()
