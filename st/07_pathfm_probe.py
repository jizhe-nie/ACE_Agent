"""
ST-W3b 探针③(最后一发) — 病理基础模型多模态。用 Phikon(Owkin, TCGA 病理图预训练 ViT)替代 ImageNet。

修正 06 的错：ImageNet 不是病理编码器。Phikon 是。且改用**全分辨率 tif** 取 224px 原生 patch(非糊的 hires)。
设计：每 spot 全分辨率 patch → Phikon → 768-d 病理特征 → 与表达 PCA 拼接聚类，比表达-only(同 GMM+精修)。
判据：表达+Phikon > 表达-only 明显 → 病理 FM 多模态值得做(冲一区)；否则两条多模态路都不成，复位目标。
环境：conda run -n Tumor_Subtype_Agent python st/07_pathfm_probe.py
"""
from __future__ import annotations

import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import torch
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from _stdata import ari_nmi, load_dlpfc, n_layers, preprocess, refine

Image.MAX_IMAGE_PIXELS = None  # 允许大 tif
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROOT = Path("data/dlpfc/extracted/DLPFC12")
SAMPLES = ["151507", "151670", "151673"]


def phikon_feats(sample, coords, r=48, bs=128):
    """注：全分辨率 tif 损坏(0页)，回退 hires PNG(与上轮 ImageNet 同输入, 隔离'编码器')。
    r=48 → 96px hires patch(spot~14px+上下文)，processor 内部缩放到 224。"""
    import json as _json
    from transformers import AutoImageProcessor, AutoModel
    mp = "data/phikon"
    proc = AutoImageProcessor.from_pretrained(mp)
    model = AutoModel.from_pretrained(mp).eval().to(DEV)
    sd = ROOT / sample
    sf = _json.load(open(sd / "spatial" / "scalefactors_json.json"))["tissue_hires_scalef"]
    arr = np.asarray(Image.open(sd / "spatial" / "tissue_hires_image.png"))[:, :, :3]
    H, W = arr.shape[:2]
    arr = np.pad(arr, ((r, r), (r, r), (0, 0)), mode="reflect")
    patches = []
    for (pr, pc) in coords:  # pr=row(y), pc=col(x) fullres
        cy = int(round(pr * sf)) + r; cx = int(round(pc * sf)) + r
        cy = min(max(cy, r), H + r - 1); cx = min(max(cx, r), W + r - 1)
        patches.append(Image.fromarray(arr[cy - r:cy + r, cx - r:cx + r]))
    feats = []
    with torch.no_grad():
        for i in range(0, len(patches), bs):
            inp = proc(images=patches[i:i + bs], return_tensors="pt").to(DEV)
            out = model(**inp).last_hidden_state[:, 0, :]  # CLS, 768-d
            feats.append(out.cpu().numpy())
    return np.vstack(feats)


def cluster_ari(emb, gt, coords, k, seed=0):
    Z = StandardScaler().fit_transform(emb)
    try:
        lab = GaussianMixture(k, covariance_type="full", reg_covar=1e-3, n_init=5,
                              random_state=seed).fit_predict(Z)
    except Exception:
        lab = KMeans(k, n_init=10, random_state=seed).fit_predict(Z)
    return ari_nmi(refine(lab, coords), gt)[0]


def main():
    print(f"[device] {DEV} | 编码器=Phikon(owkin/phikon, 病理预训练)\n")
    print(f"{'sample':<10}{'表达-only':<12}{'表达+Phikon':<14}{'GraphST(参考)':<14}")
    print("-" * 50)
    gref = {"151507": 0.598, "151670": 0.264, "151673": 0.504}
    rows = []
    for s in SAMPLES:
        ad, gt = load_dlpfc(ROOT / s)
        k = n_layers(ad, gt)
        coords = ad.obsm["spatial"].astype(float)
        gt_arr = ad.obs[gt]
        expr = preprocess(ad).obsm["X_pca"]
        img = phikon_feats(s, coords)
        img = PCA(50, random_state=0).fit_transform(StandardScaler().fit_transform(img))
        a_e = cluster_ari(expr, gt_arr, coords, k)
        fused = np.hstack([StandardScaler().fit_transform(expr), StandardScaler().fit_transform(img)])
        a_f = cluster_ari(fused, gt_arr, coords, k)
        rows.append((s, a_e, a_f))
        print(f"{s:<10}{a_e:<12.3f}{a_f:<14.3f}{gref[s]:<14.3f}", flush=True)
    print("-" * 50)
    e = np.mean([r[1] for r in rows]); f = np.mean([r[2] for r in rows])
    print(f"{'MEAN':<10}{e:<12.3f}{f:<14.3f}")
    print(f"\n[最后一发判读] 表达-only={e:.3f} → 表达+Phikon={f:.3f} "
          f"({'↑病理 FM 有正交增益, 值得冲一区' if f > e + 0.03 else '≈病理 FM 也无明显增益'})")
    print("  成立 → 病理 FM 多模态=创新方向；不成立 → 多模态两路都不成，诚实复位目标(二区/重订)。")


if __name__ == "__main__":
    main()
