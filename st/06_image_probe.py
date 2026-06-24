"""
ST-W3b 探针② — 图像多模态(H&E 形态学)。验证"组织图像=与表达正交的信息"能否提升域识别(尤其噪片)。

设计(廉价、隔离"图像信息"价值)：
  - 每个 spot 在 hires H&E 图上取一个 patch → 预训练 ResNet18 → 512-d 形态学特征。
  - 聚类对比：表达(PCA) only  vs  表达 ⊕ 图像特征(各自 z-score 拼接)，同一 GMM+精修。
  - 在多个样本(含 GraphST 弱片)看 ARI 是否提升。提升 → 图像多模态值得做；否则两条路都温吞，需重审。
环境：conda run -n Tumor_Subtype_Agent python st/06_image_probe.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as M
import torchvision.transforms as T
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from _stdata import ari_nmi, load_dlpfc, n_layers, preprocess, refine

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROOT = Path("data/dlpfc/extracted/DLPFC12")
SAMPLES = ["151507", "151670", "151673"]  # 强片/弱片/熟悉片各一
NORM = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


def image_feats(sample_dir, coords_fullres, r=16, size=64, bs=256):
    sd = Path(sample_dir)
    sf = json.load(open(sd / "spatial" / "scalefactors_json.json"))["tissue_hires_scalef"]
    img = np.asarray(Image.open(sd / "spatial" / "tissue_hires_image.png"))[:, :, :3]
    H, W = img.shape[:2]
    imgp = np.pad(img, ((r, r), (r, r), (0, 0)), mode="reflect")
    net = M.resnet18(weights=M.ResNet18_Weights.DEFAULT); net.fc = nn.Identity()
    net.eval().to(DEV)
    patches = []
    for (pr, pc) in coords_fullres:
        cy = int(round(pr * sf)) + r; cx = int(round(pc * sf)) + r
        cy = min(max(cy, r), H + r - 1); cx = min(max(cx, r), W + r - 1)
        patches.append(imgp[cy - r:cy + r, cx - r:cx + r])
    feats = []
    tf = T.Compose([T.ToPILImage(), T.Resize((size, size)), T.ToTensor(), NORM])
    with torch.no_grad():
        for i in range(0, len(patches), bs):
            batch = torch.stack([tf(p) for p in patches[i:i + bs]]).to(DEV)
            feats.append(net(batch).cpu().numpy())
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
    print(f"[device] {DEV}\n")
    print(f"{'sample':<10}{'表达-only':<12}{'表达+图像':<12}{'GraphST(参考)':<14}")
    print("-" * 48)
    gref = {"151507": 0.598, "151670": 0.264, "151673": 0.504}
    rows = []
    for s in SAMPLES:
        ad, gt = load_dlpfc(ROOT / s)
        k = n_layers(ad, gt)
        coords = ad.obsm["spatial"].astype(float)  # (pxl_row, pxl_col) fullres
        gt_arr = ad.obs[gt]
        expr = preprocess(ad).obsm["X_pca"]                       # 50-d 表达
        img = image_feats(ROOT / s, coords)                      # 512-d 图像
        img = PCA(50, random_state=0).fit_transform(StandardScaler().fit_transform(img))
        a_e = cluster_ari(expr, gt_arr, coords, k)
        fused = np.hstack([StandardScaler().fit_transform(expr), StandardScaler().fit_transform(img)])
        a_f = cluster_ari(fused, gt_arr, coords, k)
        rows.append((s, a_e, a_f))
        print(f"{s:<10}{a_e:<12.3f}{a_f:<12.3f}{gref[s]:<14.3f}", flush=True)
    print("-" * 48)
    e = np.mean([r[1] for r in rows]); f = np.mean([r[2] for r in rows])
    print(f"{'MEAN':<10}{e:<12.3f}{f:<12.3f}")
    print(f"\n[假设判读] 表达-only={e:.3f} → 表达+图像={f:.3f} "
          f"({'↑图像有正交增益, 值得做' if f > e + 0.03 else '≈图像无明显增益'})")
    print("  成立 → 图像多模态是创新方向；不成立 → 两个探针都温吞，需重审 ST 子任务/目标。")


if __name__ == "__main__":
    main()
