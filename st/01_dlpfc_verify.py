"""
ST 验证① DLPFC 空间域识别：先验证靶——空间信息到底有没有提升空间。

新方向（离开饱和的 BRCA 分型）：空间转录组(ST)。标准基准=人脑 DLPFC(Maynard 2021, 10x Visium)，
每个 spot 有 (x,y) 坐标 + 表达谱 + 人工分层金标准(L1-6 + 白质 WM)。任务=空间域识别(把 spot 聚成分层)。

先验证靶（不建大模型）：
  1. 非空间基线：标准 scanpy 预处理 → PCA → KMeans(k=层数)，ARI vs 金标准。
  2. 简单空间基线：用 spot 的空间近邻平滑 PCA 后再 KMeans（最朴素的"空间感知"）。
  3. 对比已发表 SOTA（DLPFC 151673 上：非空间 Leiden ~0.3-0.4；SpaGCN ~0.45；STAGATE ~0.60；GraphST ~0.63）。
判据：
  - 简单空间基线 > 非空间基线 → 空间信息可利用（机制成立）。
  - 二者都明显 < SOTA(~0.6) → 有清晰提升空间 = 靶值得做（区别于 BRCA 跨队列那种"已无空间"）。

环境：conda run -n Tumor_Subtype_Agent python st/01_dlpfc_verify.py --sample-dir data/dlpfc/151673
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
import scanpy as sc
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.neighbors import NearestNeighbors

def load_dlpfc(sample_dir: Path):
    """加载一个 DLPFC 样本(benchmarkst 格式) → AnnData。
    表达=10x h5；坐标=spatial/tissue_positions_list.csv；金标准=gt/layered/{sample}_{LAYER}_barcodes.txt。
    """
    import pandas as pd
    sample_dir = Path(sample_dir)
    sample = sample_dir.name
    ad = sc.read_10x_h5(next(sample_dir.glob("*_filtered_feature_bc_matrix.h5")))
    ad.var_names_make_unique()
    # 坐标(Visium v1 tissue_positions, 无表头: barcode,in_tissue,row,col,pxl_row,pxl_col)
    pos = pd.read_csv(sample_dir / "spatial" / "tissue_positions_list.csv", header=None, index_col=0)
    pos.columns = ["in_tissue", "row", "col", "pxl_row", "pxl_col"]
    # 金标准: 逐层 barcode 文件 → 标签
    gt = pd.Series(index=ad.obs_names, dtype=object)
    for f in sorted((sample_dir / "gt" / "layered").glob(f"{sample}_*_barcodes.txt")):
        layer = f.stem.replace(f"{sample}_", "").replace("_barcodes", "")
        bcs = [ln.strip() for ln in open(f) if ln.strip()]
        gt.loc[gt.index.intersection(bcs)] = layer
    ad.obs["ground_truth"] = gt.values
    common = ad.obs_names.intersection(pos.index)
    ad = ad[common].copy()
    ad.obsm["spatial"] = pos.loc[common, ["pxl_row", "pxl_col"]].to_numpy(float)
    return ad, "ground_truth"


def preprocess(ad, n_top=3000, n_pcs=50):
    ad = ad.copy()
    sc.pp.filter_genes(ad, min_cells=3)
    sc.pp.normalize_total(ad, target_sum=1e4)
    sc.pp.log1p(ad)
    sc.pp.highly_variable_genes(ad, n_top_genes=n_top)
    ad = ad[:, ad.var.highly_variable]
    sc.pp.scale(ad, max_value=10)
    sc.tl.pca(ad, n_comps=n_pcs)
    return ad


def spatial_smooth(pca, coords, k=6):
    """用空间近邻平滑 PCA：每个 spot = 自身与其 k 个空间最近邻的均值。最朴素的空间感知。"""
    nn = NearestNeighbors(n_neighbors=k + 1).fit(coords)
    _, idx = nn.kneighbors(coords)
    return pca[idx].mean(axis=1)


def ari_nmi(labels, gt):
    mask = ~(gt.isna().to_numpy() if hasattr(gt, "isna") else np.array([g != g for g in gt]))
    g = np.asarray(gt)[mask]
    lab = np.asarray(labels)[mask]
    from pandas import Categorical
    gc = Categorical(g).codes
    return adjusted_rand_score(gc, lab), normalized_mutual_info_score(gc, lab)


def main():
    ap = argparse.ArgumentParser(description="DLPFC 空间域验证")
    ap.add_argument("--sample-dir", required=True)
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()

    ad, gt = load_dlpfc(Path(args.sample_dir))
    if gt is None:
        print("[错误] 未找到金标准分层列，请检查样本目录结构。obs 列：", list(ad.obs.columns)[:20]); return
    import pandas as pd
    valid = ad.obs[gt].notna()
    k = int(pd.Categorical(ad.obs[gt][valid]).categories.size)
    print(f"[数据] {ad.n_obs} spots × {ad.n_vars} genes | 金标准列='{gt}' | 层数 k={k} "
          f"| 分布={dict(ad.obs[gt][valid].value_counts())}")

    adp = preprocess(ad)
    pca = adp.obsm["X_pca"]
    coords = ad.obsm["spatial"].astype(float)

    def multi(emb):
        a, n = [], []
        for s in range(args.seeds):
            lab = KMeans(k, n_init=10, random_state=s).fit_predict(emb)
            ari, nmi = ari_nmi(lab, ad.obs[gt])
            a.append(ari); n.append(nmi)
        return np.mean(a), np.std(a), np.mean(n)

    ns_a, ns_s, ns_n = multi(pca)
    sp_a, sp_s, sp_n = multi(spatial_smooth(pca, coords))
    print(f"\n{'方法':<26}{'ARI(mean±std)':<20}{'NMI':<10}")
    print("-" * 56)
    print(f"{'非空间 KMeans/PCA':<26}{f'{ns_a:.3f}±{ns_s:.3f}':<20}{ns_n:<10.3f}")
    print(f"{'简单空间(近邻平滑)':<26}{f'{sp_a:.3f}±{sp_s:.3f}':<20}{sp_n:<10.3f}")
    print(f"{'SOTA 参考(STAGATE~0.60/GraphST~0.63)':<26}")

    print("\n[诚实判读]")
    print(f"  空间 vs 非空间: {ns_a:.3f} → {sp_a:.3f} ({'↑空间信息有用' if sp_a>ns_a+0.02 else '≈持平/未提升'})")
    gap = 0.60 - sp_a
    if gap > 0.1:
        print(f"  距 SOTA(~0.60) 还差 {gap:.2f} → **有清晰提升空间，靶值得做**。")
    elif gap > 0.03:
        print(f"  距 SOTA 差 {gap:.2f} → 中等空间，需更强方法。")
    else:
        print(f"  已接近 SOTA → 简单方法即够，需重审创新空间。")


if __name__ == "__main__":
    main()
