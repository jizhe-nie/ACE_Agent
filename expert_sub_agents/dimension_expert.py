"""
expert_sub_agents/dimension_expert.py
======================================
Dimension Expert (Phase 3 refactor): Hybrid skeleton + LLM decision model.

Instead of asking the LLM to write ~4500 chars of Python from scratch (which
produced a ~60 % success rate due to import omissions, variable-binding bugs,
and anti-patterns like ``artifacts = {}``), we now:

1. Construct a **deterministic code skeleton** that handles imports,
   scaling, error-guards, and the artifacts-write contract.
2. Ask the LLM to produce a **compact JSON decision document** (~700 chars)
   that selects which pipelines to activate and with what hyper-parameters,
   including deep AE architecture topology.
3. Inject that decision into the skeleton and return the merged code.

The sandbox pre-injects ``StandardScaler``, ``PCA``, ``KMeans``,
``GaussianMixture``, ``silhouette_score``, ``calinski_harabasz_score`` and
``davies_bouldin_score`` (see ``CORE_PRE_INJECT`` in coder_sandbox.py), so
the skeleton references them directly without import statements.

An optional 5th pipeline (``ae_kmeans``) leverages a deterministic
AutoEncoder from ``tools/ae_pipeline.py`` for high-dimensional data
(``n_features > 32``).
"""

from __future__ import annotations

import json
import re
from typing import Any

from ACE_Agent.agent_core.schemas import DatasetBundle
from ACE_Agent.expert_sub_agents.base import BaseExpert
from ACE_Agent.tools.llm_client import UniversalLLMClient

# ---------------------------------------------------------------------------
# Deterministic code skeleton template
# ---------------------------------------------------------------------------
_SKELETON = r"""# ===== ACE Dimension Expert Skeleton (Phase 3) =====
# All sklearn types below are pre-injected by the sandbox.
# Data lives in CTX_DATA (read-only); results go into artifacts (write-only).

import numpy as _np

# ---- scale -------------------------------------------------------------
_scaler = {SCALER_CLASS}()
_X = _scaler.fit_transform(CTX_DATA.X)
_n = CTX_DATA.n_samples
_d = CTX_DATA.n_features
_k0 = CTX_DATA.expected_clusters

# ---- pipeline decisions (from LLM) -------------------------------------
_DECISIONS = {DECISIONS_JSON}

# ========================================================================
# PIPELINE 1: PCA + KMeans
# ========================================================================
_d1 = _DECISIONS.get("pipelines", {}).get("pca_kmeans", {})
if _d1.get("active", True):
    _nc1 = int(_d1.get("n_components", min(_d, max(2, _d // 3))))
    _k1 = int(_d1.get("k", _k0))
    try:
        _red1 = PCA(n_components=min(_nc1, _n, _d), random_state=42).fit_transform(_X)
        _lbl1 = KMeans(n_clusters=min(_k1, _n - 1), random_state=42, n_init=10).fit_predict(_red1)
        _sil1 = float(silhouette_score(_red1, _lbl1))
        _chi1 = float(calinski_harabasz_score(_red1, _lbl1))
        artifacts["PCA_KMeans"] = {
            "labels": _lbl1.tolist(),
            "metrics": {
                "score": _sil1, "score_source": "silhouette",
                "silhouette": _sil1, "chi": _chi1,
                "n_components": int(_nc1),
            },
            "plot_path": "",
        }
    except Exception as _e1:
        artifacts["PCA_KMeans_error"] = {
            "labels": [], "metrics": {"score": 0.0, "error": str(_e1)}, "plot_path": ""}

# ========================================================================
# PIPELINE 2: PCA + GMM
# ========================================================================
_d2 = _DECISIONS.get("pipelines", {}).get("pca_gmm", {})
if _d2.get("active", True):
    _nc2 = int(_d2.get("n_components", min(_d, max(2, _d // 3))))
    _k2 = int(_d2.get("k", _k0))
    try:
        _red2 = PCA(n_components=min(_nc2, _n, _d), random_state=42).fit_transform(_X)
        _gmm2 = GaussianMixture(n_components=min(_k2, _n - 1), random_state=42)
        _lbl2 = _gmm2.fit_predict(_red2)
        _sil2 = float(silhouette_score(_red2, _lbl2))
        _chi2 = float(calinski_harabasz_score(_red2, _lbl2))
        artifacts["PCA_GMM"] = {
            "labels": _lbl2.tolist(),
            "metrics": {
                "score": _sil2, "score_source": "silhouette",
                "silhouette": _sil2, "chi": _chi2,
                "n_components": int(_nc2),
            },
            "plot_path": "",
        }
    except Exception as _e2:
        artifacts["PCA_GMM_error"] = {
            "labels": [], "metrics": {"score": 0.0, "error": str(_e2)}, "plot_path": ""}

# ========================================================================
# PIPELINE 3: UMAP + KMeans  (if umap is installed)
# ========================================================================
_UMAP_OK = False
try:
    from umap import UMAP as _UMAP
    _UMAP_OK = True
except Exception:
    pass

_d3 = _DECISIONS.get("pipelines", {}).get("umap_kmeans", {})
if _d3.get("active", True) and _UMAP_OK:
    _nc3 = int(_d3.get("n_components", min(2, _d)))
    _nn3 = int(_d3.get("n_neighbors", min(15, _n - 1)))
    _k3 = int(_d3.get("k", _k0))
    try:
        _red3 = _UMAP(n_components=_nc3, n_neighbors=_nn3, random_state=42).fit_transform(_X)
        _lbl3 = KMeans(n_clusters=min(_k3, _n - 1), random_state=42, n_init=10).fit_predict(_red3)
        _sil3 = float(silhouette_score(_red3, _lbl3))
        _chi3 = float(calinski_harabasz_score(_red3, _lbl3))
        artifacts["UMAP_KMeans"] = {
            "labels": _lbl3.tolist(),
            "metrics": {
                "score": _sil3, "score_source": "silhouette",
                "silhouette": _sil3, "chi": _chi3,
                "n_components": int(_nc3),
            },
            "plot_path": "",
        }
    except Exception as _e3:
        artifacts["UMAP_KMeans_error"] = {
            "labels": [], "metrics": {"score": 0.0, "error": str(_e3)}, "plot_path": ""}

# ========================================================================
# PIPELINE 4: t-SNE + KMeans  (for n <= 5000)
# ========================================================================
_TSNE_OK = False
try:
    from sklearn.manifold import TSNE as _TSNE
    _TSNE_OK = True
except Exception:
    pass

_d4 = _DECISIONS.get("pipelines", {}).get("tsne_kmeans", {})
if _d4.get("active", True) and _TSNE_OK and _n <= 5000:
    _k4 = int(_d4.get("k", _k0))
    try:
        _red4 = _TSNE(n_components=2, random_state=42, max_iter=300).fit_transform(_X)
    except TypeError:
        _red4 = _TSNE(n_components=2, random_state=42, n_iter=300).fit_transform(_X)
    _lbl4 = KMeans(n_clusters=min(_k4, _n - 1), random_state=42, n_init=10).fit_predict(_red4)
    _sil4 = float(silhouette_score(_red4, _lbl4))
    _chi4 = float(calinski_harabasz_score(_red4, _lbl4))
    artifacts["tSNE_KMeans"] = {
        "labels": _lbl4.tolist(),
        "metrics": {
            "score": _sil4, "score_source": "silhouette",
            "silhouette": _sil4, "chi": _chi4,
            "n_components": 2,
        },
        "plot_path": "",
    }

# ========================================================================
# PIPELINE 5: AutoEncoder + KMeans  (high-dim data, if torch is available)
#            Uses Conv-AE for image data, MLP AE otherwise.
# ========================================================================
_AE_OK = False
try:
    {AE_IMPORT}
    _AE_OK = True
except Exception:
    pass

_d5 = _DECISIONS.get("pipelines", {}).get("ae_kmeans", {})
if _d5.get("active", True) and _AE_OK and _d > 32:
    _k5 = int(_d5.get("k", _k0))
    _latent5 = int(_d5.get("latent_dim", min(8, max(2, _d // 4))))
    _epochs5 = int(_d5.get("epochs", 100))
    _hidden5 = _d5.get("hidden_dims", None)
    _lr5 = float(_d5.get("learning_rate", 1e-3))
    _drop5 = float(_d5.get("dropout", 0.2))
    _patience5 = int(_d5.get("early_stopping_patience", 15))
    _noise5 = float(_d5.get("noise_std", 0.15))
    _cluster5 = str(_d5.get("cluster_method", "kmeans"))
    try:
        _result5 = _ae_pipe(
            _X, k=_k5, latent_dim=_latent5, epochs=_epochs5,
            hidden_dims=_hidden5, learning_rate=_lr5,
            dropout=_drop5, early_stopping_patience=_patience5,
            noise_std=_noise5, cluster_method=_cluster5,
            normalize="{NORMALIZE}"{AE_EXTRA_ARGS},
        )
        artifacts["AE_KMeans"] = _result5
    except Exception as _e5:
        artifacts["AE_KMeans_error"] = {
            "labels": [], "metrics": {"score": 0.0, "error": str(_e5)}, "plot_path": ""}

# ========================================================================
# PIPELINE 5B: Res-Attention AE + KMeans  (GAP/embedding features, Phase 6)
#             Uses residual blocks + self-attention bottleneck for semantic
#             features where each dimension carries classification signal.
# ========================================================================
_RES_AE_OK = False
try:
    {RES_AE_IMPORT}
    _RES_AE_OK = True
except Exception:
    pass

_d5b = _DECISIONS.get("pipelines", {}).get("res_ae_kmeans", {})
if _d5b.get("active", True) and _RES_AE_OK and _d > 32:
    _k5b = int(_d5b.get("k", _k0))
    _latent5b = int(_d5b.get("latent_dim", min(8, max(2, _d // 4))))
    _epochs5b = int(_d5b.get("epochs", 100))
    _hidden5b = _d5b.get("hidden_dims", None)
    _lr5b = float(_d5b.get("learning_rate", 1e-3))
    _drop5b = float(_d5b.get("dropout", 0.2))
    _patience5b = int(_d5b.get("early_stopping_patience", 15))
    _noise5b = float(_d5b.get("noise_std", 0.15))
    _heads5b = int(_d5b.get("num_heads", 4))
    _cluster5b = str(_d5b.get("cluster_method", "gmm"))
    try:
        _result5b = _res_ae_pipe(
            _X, k=_k5b, latent_dim=_latent5b, epochs=_epochs5b,
            hidden_dims=_hidden5b, learning_rate=_lr5b,
            dropout=_drop5b, early_stopping_patience=_patience5b,
            noise_std=_noise5b, num_heads=_heads5b,
            cluster_method=_cluster5b, normalize="{NORMALIZE}",
        )
        artifacts["ResAE_KMeans"] = _result5b
    except Exception as _e5b:
        artifacts["ResAE_KMeans_error"] = {
            "labels": [], "metrics": {"score": 0.0, "error": str(_e5b)}, "plot_path": ""}

# ========================================================================
# PIPELINE 6: DEC / IDEC 联合优化 (high-dim data, KL divergence fine-tune)
#            Uses Conv-DEC for image data, MLP DEC otherwise.
# ========================================================================
_DEC_OK = False
try:
    {DEC_IMPORT}
    _DEC_OK = True
except Exception:
    pass

_d6 = _DECISIONS.get("pipelines", {}).get("dec", {})
if _d6.get("active", True) and _DEC_OK and _d > 32:
    _k6 = int(_d6.get("k", _k0))
    _latent6 = int(_d6.get("latent_dim", min(8, max(2, _d // 4))))
    _pretrain6 = int(_d6.get("pretrain_epochs", 100))
    _finetune6 = int(_d6.get("finetune_epochs", 50))
    _hidden6 = _d6.get("hidden_dims", None)
    _lr6 = float(_d6.get("learning_rate", 1e-3))
    _drop6 = float(_d6.get("dropout", 0.2))
    _gamma6 = float(_d6.get("gamma", 0.1))
    _noise6 = float(_d6.get("noise_std", 0.15))
    try:
        _result6 = _dec_pipe(
            _X, k=_k6, latent_dim=_latent6,
            hidden_dims=_hidden6,
            pretrain_epochs=_pretrain6, finetune_epochs=_finetune6,
            pretrain_lr=_lr6, dropout=_drop6,
            gamma=_gamma6, noise_std=_noise6,
            device="auto",
            normalize="{NORMALIZE}"{DEC_EXTRA_ARGS},
        )
        artifacts["DEC"] = _result6
    except Exception as _e6:
        artifacts["DEC_error"] = {
            "labels": [], "metrics": {"score": 0.0, "error": str(_e6)}, "plot_path": ""}

# ========================================================================
# PIPELINE 7: Conv-AE + SelfLabel teacher-student distillation (image data)
#            Phase A: Conv-AE pretrain (ReflectionPad2d + Latent BN)
#            Phase B: GMM pseudo-labels -> CE fine-tune Encoder (frozen Decoder)
#            Production path for MNIST/Fashion-MNIST (ARI >= 0.84 on 70K).
# ========================================================================
_SL_OK = False
try:
    {SELFLABEL_IMPORT}
    _SL_OK = True
except Exception:
    pass

_d7 = _DECISIONS.get("pipelines", {}).get("selflabel", {})
if _d7.get("active", True) and _SL_OK and _d > 32 and CTX_DATA.metadata.get("is_image"):
    _k7 = int(_d7.get("k", _k0))
    _latent7 = int(_d7.get("latent_dim", 32))
    _ae_epochs7 = int(_d7.get("ae_epochs", 150))
    _cluster_epochs7 = int(_d7.get("cluster_epochs", 30))
    _n_iter7 = int(_d7.get("n_iterations", 3))
    _drop7 = float(_d7.get("dropout", 0.2))
    _contrastive7 = float(_d7.get("contrastive_weight", 0.1))
    _bootstrap7 = _d7.get("bootstrap", True)
    _augment7 = _d7.get("augment", True)
    try:
        _result7 = _sl_pipe(
            _X, k=_k7, latent_dim=_latent7,
            ae_epochs=_ae_epochs7, cluster_epochs=_cluster_epochs7,
            n_iterations=_n_iter7, dropout=_drop7,
            contrastive_weight=_contrastive7, bootstrap=_bootstrap7,
            augment=_augment7, normalize="{NORMALIZE}",
        )
        artifacts["SelfLabel"] = _result7
    except Exception as _e7:
        artifacts["SelfLabel_error"] = {
            "labels": [], "metrics": {"score": 0.0, "error": str(_e7)}, "plot_path": ""}
"""

# ---------------------------------------------------------------------------
# LLM decision prompt (constrained)
# ---------------------------------------------------------------------------
_DECISION_SYSTEM_PROMPT = (
    "你是一个降维与聚类参数决策专家。\n\n"
    "你需要为给定的数据集选择合适的降维+聚类管线并设定参数。\n"
    "输出**纯 JSON**，不要 Markdown，不要解释文字。\n\n"
    "## 可用管线\n"
    "1. `pca_kmeans`   — PCA 降维后用 KMeans (几乎总是激活)\n"
    "2. `pca_gmm`      — PCA 降维后用 GaussianMixture (几乎总是激活)\n"
    "3. `umap_kmeans`  — UMAP 流形降维 + KMeans (需要 umap 库，代码自动检测)\n"
    "4. `tsne_kmeans`  — t-SNE 降维 + KMeans (仅 n<=5000 可用，代码自动限流)\n"
    "5. `ae_kmeans`    — AutoEncoder 深度降维 + KMeans (仅 n_features>32 自动激活)\n"
    "   * 对于图像数据 (MNIST/Fashion-MNIST)，系统自动使用 Conv2d/ConvTranspose2d 架构\n"
    "   * Conv-AE 结构: 3层卷积编码器(32→64→128 filters) + 对称解码器\n"
    "6. `dec`          — DEC/IDEC KL散度联合优化 (仅 n_features>32 自动激活，n_samples需>200)\n"
    "   * 对于图像数据，系统自动使用 Conv-DEC (Conv-AE 骨干 + KL 微调)\n"
    "8. `res_ae_kmeans` — 残差+自注意力 AE + KMeans/GMM (Phase 6, GAP/嵌入特征首选)\n"
    "   * 专为语义特征设计（如 CIFAR-10 GAP 64D），非图像非原始像素\n"
    "   * 核心创新: 残差连接 + Multi-Head Self-Attention 隐空间瓶颈\n"
    "   * Self-Attention 在编码器最深隐藏层运行，学习各特征维度之间的交互\n"
    "   * 残差连接防止深度堆叠时的梯度消失\n"
    "   * **当数据为 GAP/嵌入特征时，此管线优于普通 ae_kmeans**\n"
    "   * 仅对非图像的高维数据 (n_features>32) 自动激活\n"
    "7. `selflabel`     — Conv-AE + GMM 师生自蒸馏 (仅图像数据, n_features>32 自动激活)\n"
    "   * Phase A: Conv-AE 预训练 (ReflectionPad2d + Latent BN + Contrastive Loss)\n"
    "   * Phase B: GMM 伪标签 → 冻结Decoder → Cross-Entropy 微调 Encoder\n"
    "   * 这是图像数据的**首选生产路径**，ARI 稳定超过 0.84 (70K MNIST 验证)\n"
    "   * 核心超参: latent_dim=32, ae_epochs=150, contrastive_weight=0.1, bootstrap=True\n\n"
    "## 决策指南\n"
    "- n_features <= 3: 降维意义不大，仅激活 pca_kmeans 和 pca_gmm\n"
    "- 4 <= n_features <= 32: 激活 PCA 管线 + UMAP (若可用)\n"
    "- n_features > 32: 激活全部可用管线（含 ae_kmeans）\n"
    "- n_components 不宜超过 min(n_features, n_samples//2)\n"
    "- k (聚类数) 默认用 expected_clusters 值，不要猜测\n\n"
    "## AE 深度管线调参指南（Phase 4 深度去噪架构）\n"
    "- AE 管线使用**多层堆叠 Denoising AutoEncoder**：\n"
    "  * 每层结构: Linear → BatchNorm1d → LeakyReLU(0.2) → Dropout\n"
    "  * 训练时向输入注入高斯噪声（Denoising AE），验证时不加噪\n"
    "  * Encoder 和 Decoder 对称\n"
    "  * 自带 L2 正则化 (weight_decay=1e-4) + CosineAnnealingLR + Early Stopping\n"
    "- **hidden_dims 选择** (核心调参项，决定网络深度与宽度):\n"
    "  * n_features <= 32: [64, 32] 或 [32, 16]\n"
    "  * 32 < n_features <= 64: [128, 64, 32]\n"
    "  * 64 < n_features <= 128: [256, 128, 64, 32] 或 [128, 64, 32]\n"
    "  * n_features > 128: [256, 128, 64, 32]\n"
    "  * 原则：层数越多，非线性表达能力越强，但小数据集易过拟合\n"
    "  * n_samples < 500 时减少层数（2 层），> 2000 时可加深（3-4 层）\n"
    "- **epochs 选择**: 根据 n_features 动态调整\n"
    "  * 32 < n_features <= 64: epochs=80~150\n"
    "  * 64 < n_features <= 128: epochs=100~200\n"
    "  * n_features > 128: epochs=150~300\n"
    "  * Early Stopping 会在验证损失不再下降时自动终止\n"
    "- **learning_rate 选择**: Adam 初始学习率\n"
    "  * 默认 1e-3，深度网络（≥3层）建议降至 5e-4\n"
    "  * 小数据集（n_samples < 500）建议降至 5e-4 以提高稳定性\n"
    "- **latent_dim 选择**: 瓶颈层维度，通常为 n_features 的 1/4 ~ 1/6\n"
    "  * 若数据类内方差大，适当增大 latent_dim (接近 n_features//4)\n"
    "  * 若需要强压缩，缩小 latent_dim (接近 n_features//6)\n"
    "- **dropout**: 默认 0.2，小数据集可增至 0.3~0.4 防过拟合\n"
    "- **early_stopping_patience**: 默认 15，小数据集可降至 10\n"
    "- **noise_std**: Denoising AE 噪声标准差，默认 0.15\n"
    "  * 小数据集（n_samples < 500）可增至 0.3~0.5 增强正则化（强去噪）\n"
    "  * 大数据集可降至 0.05~0.15\n"
    '- **cluster_method**: 潜在空间聚类方法 "kmeans" 或 "gmm"\n'
    "  * GMM 能捕获非球形簇，多数情况下优于 KMeans\n"
    "  * 小数据集或简单聚类结构用 kmeans\n"
    "  * 默认推荐 gmm\n\n"
    "## DEC / IDEC 联合优化调参指南（Phase 3 核心创新）\n"
    "- DEC 在两阶段训练后通过 KL 散度联合优化编码器与聚类中心：\n"
    "  Phase 1: AE 预训练（与 ae_kmeans 相同）\n"
    "  Phase 2: 移除解码器 → KMeans 初始化聚类中心 → 联合微调\n"
    "  * 软分配 Q: Student's t-distribution kernel (α=1)\n"
    "  * 目标分布 P: Q²/f 归一化（锐化高置信度分配）\n"
    "  * 损失: KL(P||Q)，IDEC 模式额外加 γ·MSE(重建) 保留局部结构\n"
    "- **gamma 选择** (IDEC 重建权重，核心超参):\n"
    "  * gamma=0.0: 纯 DEC（仅优化聚类，可能破坏局部结构）\n"
    "  * gamma=0.05~0.1: 温和 IDEC（推荐默认，保留局部结构的同时优化聚类）\n"
    "  * gamma=0.2~0.5: 强 IDEC（更重重建，适合预训练不充分时）\n"
    "- **finetune_epochs 选择**: KL 微调轮次\n"
    "  * 默认 50，大数据集（n_samples>2000）可增至 80~100\n"
    "  * 收敛判断: label 变化率 < 0.1% 时自动停止\n"
    "- **pretrain_epochs 选择**: 与 ae_kmeans 的 epochs 相同逻辑\n"
    "- DEC 相比 AE_KMeans 的理论优势：\n"
    "  * AE_KMeans: 重建 + 聚类分离 → 重建损失不优化可分性\n"
    "  * DEC: KL 散度直接优化聚类可分性 → ARI 通常显著更高\n"
    "  * IDEC: 保留重建约束防止嵌入空间坍塌\n"
    "- 注意：DEC 需要 n_samples > 200 以确保稳定的软分配估计\n\n"
    "## Conv-AE 图像专用架构 (系统根据 is_image 自动切换)\n"
    "- 当数据为图像时 (metadata.is_image=True)，AE/DEC 自动切换为 Conv2d 架构:\n"
    "  * Encoder: Conv2d(1,32,3,s=2)→BN→ReLU (14×14) → Conv2d(32,64,3,s=2)→BN→ReLU (7×7)\n"
    "    → Conv2d(64,128,3,s=2)→BN→ReLU (4×4) → Flatten→Linear(2048, latent_dim)\n"
    "  * Decoder: Linear→Unflatten → ConvTranspose2d(128→64→32→1) + Sigmoid\n"
    "  * 总参数量 ~150K，比 MLP AE (~2M) 更高效且能捕获空间结构\n"
    "- **Conv-AE 专用超参**:\n"
    "  * epochs: 建议 100~200 (默认 150)\n"
    "  * latent_dim: 建议 32 (图像数据需要更大瓶颈保留结构)\n"
    "  * dropout: 建议 0.1 (Dropout2d 轻正则)\n"
    "  * noise_std: 建议 0.05~0.15 (默认 0.1)\n"
    "  * cluster_method: 推荐 gmm (GMM 能更好捕获潜在空间的非球形簇)\n"
    "- **Conv-DEC 专用超参**:\n"
    "  * pretrain_epochs: Conv-AE 预训练轮次 (默认 150)\n"
    "  * finetune_epochs: KL 微调轮次 (默认 400，大数据集可更短)\n"
    "  * gamma: IDEC 重建权重 (默认 0.1)\n"
    "  * hidden_dims: Conv-AE 不使用此参数 (网络深度由 3 层卷积固定)\n\n"
    "## JSON 格式\n"
    "{\n"
    '  "pipelines": {\n'
    '    "pca_kmeans":  {"active": true, "n_components": <int>, "k": <int>},\n'
    '    "pca_gmm":     {"active": true, "n_components": <int>, "k": <int>},\n'
    '    "umap_kmeans": {"active": true, "n_components": 2, "n_neighbors": <int>, "k": <int>},\n'
    '    "tsne_kmeans": {"active": true, "k": <int>},\n'
    '    "ae_kmeans":   {"active": true, "latent_dim": <int>, "epochs": <int>, "k": <int>,\n'
    '                    "hidden_dims": [<int>, ...], "learning_rate": <float>,\n'
    '                    "dropout": <float>, "early_stopping_patience": <int>,\n'
    '                    "noise_std": <float>, "cluster_method": "gmm"},\n'
    '    "dec":         {"active": true, "latent_dim": <int>, "k": <int>,\n'
    '                    "pretrain_epochs": <int>, "finetune_epochs": <int>,\n'
    '                    "hidden_dims": [<int>, ...], "learning_rate": <float>,\n'
    '                    "dropout": <float>, "gamma": <float>, "noise_std": <float>},\n'
    '    "res_ae_kmeans": {"active": true, "latent_dim": <int>, "epochs": <int>, "k": <int>,\n'
    '                    "hidden_dims": [<int>, ...], "learning_rate": <float>,\n'
    '                    "dropout": <float>, "num_heads": <int>,\n'
    '                    "noise_std": <float>, "cluster_method": "gmm"},\n'
    '    "selflabel":   {"active": true, "latent_dim": 32, "k": <int>,\n'
    '                    "ae_epochs": 150, "cluster_epochs": 30,\n'
    '                    "n_iterations": 3, "dropout": 0.2,\n'
    '                    "contrastive_weight": 0.1, "bootstrap": true,\n'
    '                    "augment": true}\n'
    "  }\n"
    "}\n\n"
    "只输出 JSON。"
)


def _extract_json(text: str) -> dict[str, Any]:
    """Best-effort JSON extraction from LLM output (may have fences)."""
    # Try plain parse first
    stripped = text.strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass
    # Strip ```json fences
    m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", stripped)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    # Fallback: find first { and last }
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(stripped[start : end + 1])
        except json.JSONDecodeError:
            pass
    # Return empty → skeleton uses all defaults
    return {}


def _build_smart_defaults(
    n_features: int, n_samples: int, k: int, is_image: bool = False
) -> dict[str, Any]:
    """Build sensible defaults when LLM decision fails.

    AE hyper-parameters scale with dimensionality and dataset size:
    - Deeper / wider networks for high-dim data
    - Shallower networks + higher dropout for small samples
    - Lower learning rate for deep networks or small datasets
    """
    n_comp = min(n_features, max(2, n_features // 3), n_samples // 2)
    latent_dim = min(12, max(2, n_features // 4))

    # ---- AE architecture defaults ---------------------------------------
    small_dataset = n_samples < 500
    tiny_dataset = n_samples < 200

    # Non-image defaults (set first, overridden below for is_image)
    ae_patience = 10 if small_dataset else 15
    ae_noise = 0.35 if small_dataset else 0.15
    ae_cluster = "gmm"
    dec_gamma = 0.1

    if is_image:
        # Conv-AE defaults for image data
        hidden_dims = None     # not used by Conv-AE
        ae_epochs = 150
        ae_lr = 1e-3
        ae_dropout = 0.1
        latent_dim = 32
        ae_cluster = "gmm"
        ae_noise = 0.1
        ae_patience = 20
        dec_finetune_epochs = 400
        dec_pretrain_epochs = 150
    elif n_features > 128:
        hidden_dims = [256, 128, 64, 32] if not tiny_dataset else [128, 64]
        ae_epochs = 200
        ae_lr = 5e-4
        ae_dropout = 0.3 if small_dataset else 0.2
        latent_dim = min(12, max(2, n_features // 4))
        dec_finetune_epochs = 200 if n_samples > 2000 else (80 if n_samples > 500 else 50)
        dec_pretrain_epochs = ae_epochs
    elif n_features > 64:
        hidden_dims = [128, 64, 32] if not tiny_dataset else [64, 32]
        ae_epochs = 120
        ae_lr = 5e-4 if small_dataset else 1e-3
        ae_dropout = 0.3 if small_dataset else 0.2
        latent_dim = min(12, max(2, n_features // 4))
        dec_finetune_epochs = 200 if n_samples > 2000 else (80 if n_samples > 500 else 50)
        dec_pretrain_epochs = ae_epochs
    elif n_features > 32:
        hidden_dims = [128, 64, 32] if not small_dataset else [64, 32]
        ae_epochs = 80
        ae_lr = 5e-4 if small_dataset else 1e-3
        ae_dropout = 0.3 if small_dataset else 0.2
        latent_dim = min(12, max(2, n_features // 4))
        dec_finetune_epochs = 200 if n_samples > 2000 else (80 if n_samples > 500 else 50)
        dec_pretrain_epochs = ae_epochs
    else:
        hidden_dims = [64, 32]
        ae_epochs = 30
        ae_lr = 1e-3
        ae_dropout = 0.2
        latent_dim = min(12, max(2, n_features // 4))
        dec_finetune_epochs = 50
        dec_pretrain_epochs = ae_epochs

    result = {
        "pipelines": {
            "pca_kmeans": {"active": True, "n_components": n_comp, "k": k},
            "pca_gmm": {"active": True, "n_components": n_comp, "k": k},
            "umap_kmeans": {
                "active": True,
                "n_components": min(2, n_features),
                "n_neighbors": min(15, n_samples - 1),
                "k": k,
            },
            "tsne_kmeans": {"active": n_samples <= 5000, "k": k},
            "ae_kmeans": {
                "active": n_features > 32,
                "latent_dim": latent_dim,
                "epochs": ae_epochs,
                "k": k,
                "hidden_dims": hidden_dims,
                "learning_rate": ae_lr,
                "dropout": ae_dropout,
                "early_stopping_patience": ae_patience,
                "noise_std": ae_noise,
                "cluster_method": ae_cluster,
            },
            "res_ae_kmeans": {
                "active": n_features > 32,
                "latent_dim": latent_dim,
                "epochs": ae_epochs,
                "k": k,
                "hidden_dims": hidden_dims,
                "learning_rate": ae_lr,
                "dropout": ae_dropout,
                "early_stopping_patience": ae_patience,
                "noise_std": ae_noise,
                "num_heads": 4,
                "cluster_method": ae_cluster,
            },
            "dec": {
                "active": n_features > 32 and n_samples > 200,
                "latent_dim": latent_dim,
                "k": k,
                "pretrain_epochs": dec_pretrain_epochs,
                "finetune_epochs": dec_finetune_epochs,
                "hidden_dims": hidden_dims,
                "learning_rate": ae_lr,
                "dropout": ae_dropout,
                "gamma": dec_gamma,
                "noise_std": ae_noise,
            },
        }
    }
    if is_image:
        result["pipelines"]["selflabel"] = {
            "active": n_features > 32,
            "latent_dim": 32,
            "k": k,
            "ae_epochs": 150,
            "cluster_epochs": 30,
            "n_iterations": 3,
            "dropout": 0.2,
            "contrastive_weight": 0.1,
            "bootstrap": True,
            "augment": True,
        }
    return result


# ---------------------------------------------------------------------------
# DimensionExpert
# ---------------------------------------------------------------------------
class DimensionExpert(BaseExpert):
    """Dimension Expert (Phase 3): skeleton + LLM parameter decision.

    Uses ``PRE_INJECT`` to make ``ae_kmeans_pipeline`` available in the
    sandbox without requiring an import statement in the generated code.
    """

    PRE_INJECT: dict[str, Any] = {}  # set in __init__ to avoid import cycles

    def __init__(self) -> None:
        super().__init__("dimension", "维度专家")
        # Deferred import so this module can be loaded without torch.
        pre_inject: dict[str, Any] = {}
        try:
            from ACE_Agent.tools.ae_pipeline import ae_kmeans_pipeline  # noqa: F811

            pre_inject["ae_kmeans_pipeline"] = ae_kmeans_pipeline
        except Exception:
            pass
        try:
            from ACE_Agent.tools.ae_pipeline import conv_ae_kmeans_pipeline  # noqa: F811

            pre_inject["conv_ae_kmeans_pipeline"] = conv_ae_kmeans_pipeline
        except Exception:
            pass
        try:
            from ACE_Agent.tools.ae_pipeline import conv_selflabel_pipeline  # noqa: F811

            pre_inject["conv_selflabel_pipeline"] = conv_selflabel_pipeline
        except Exception:
            pass
        try:
            from ACE_Agent.tools.ae_pipeline import res_ae_kmeans_pipeline  # noqa: F811

            pre_inject["res_ae_kmeans_pipeline"] = res_ae_kmeans_pipeline
        except Exception:
            pass
        try:
            from ACE_Agent.tools.dec_pipeline import dec_pipeline  # noqa: F811

            pre_inject["dec_pipeline"] = dec_pipeline
        except Exception:
            pass
        try:
            from ACE_Agent.tools.dec_pipeline import conv_dec_pipeline  # noqa: F811

            pre_inject["conv_dec_pipeline"] = conv_dec_pipeline
        except Exception:
            pass
        self.PRE_INJECT = pre_inject

    def _generate_code(
        self,
        client: UniversalLLMClient,
        dataset: DatasetBundle,
        prompt: str,
        constraints=None,
    ) -> str:
        n_features = dataset.X.shape[1] if dataset.X.ndim == 2 else 1
        n_samples = dataset.X.shape[0]
        k = dataset.metadata.get("expected_clusters", 3) if dataset.metadata else 3

        # 0. Determine dataset type early
        is_image = bool(dataset.metadata.get("is_image")) if dataset.metadata else False
        original_shape = dataset.metadata.get("original_shape") if dataset.metadata else None

        # 1. Call LLM for pipeline decisions (JSON output)
        small_ds = n_samples < 500
        tiny_ds = n_samples < 200
        if is_image:
            dim_category = "图像数据(Conv-AE)"
            img_h, img_w = original_shape if original_shape else (28, 28)
            ae_hint = (
                f"Conv-AE架构: 3层Conv2d(32→64→128), input={img_h}×{img_w}, "
                f"epochs=100~200, lr=1e-3, latent_dim=32, dropout=0.1, "
                f"noise_std=0.05~0.15, cluster=gmm. hidden_dims 不使用(由卷积结构固定)."
            )
        elif n_features > 128:
            dim_category = "超高维"
            ae_hint = (
                ("hidden_dims=[256,128,64,32]" if not tiny_ds else "hidden_dims=[128,64]")
                + f", epochs=150~300, lr=5e-4, dropout={'0.3' if small_ds else '0.2'}, noise_std={'0.35' if small_ds else '0.15'}, cluster=gmm"
            )
        elif n_features > 64:
            dim_category = "高维"
            ae_hint = (
                ("hidden_dims=[128,64,32]" if not tiny_ds else "hidden_dims=[64,32]")
                + f", epochs=100~200, lr={'5e-4' if small_ds else '1e-3'}, dropout={'0.3' if small_ds else '0.2'}, noise_std={'0.35' if small_ds else '0.15'}, cluster=gmm"
            )
        elif n_features > 32:
            dim_category = "中高维"
            ae_hint = (
                ("hidden_dims=[128,64,32]" if not small_ds else "hidden_dims=[64,32]")
                + f", epochs=80~150, lr={'5e-4' if small_ds else '1e-3'}, dropout={'0.3' if small_ds else '0.2'}, noise_std={'0.35' if small_ds else '0.15'}, cluster=gmm"
            )
        else:
            dim_category = "常规"
            ae_hint = "n_features<=32，AE 管线不会激活"
        user_msg = (
            f"n_samples={n_samples}, n_features={n_features}, expected_clusters={k}。"
            f"数据维度类别: {dim_category}。"
            f"AE调参提示: {ae_hint}。"
            f"请输出管线决策 JSON（含 hidden_dims, learning_rate, dropout, noise_std, cluster_method, early_stopping_patience）。"
        )
        raw = client.chat_completion(
            [{"role": "user", "content": user_msg}],
            self._inject_constraints_prompt(constraints) + _DECISION_SYSTEM_PROMPT,
        ).strip()

        # 2. Parse decision (or use smart defaults on failure)
        decisions = _extract_json(raw)
        if not decisions or "pipelines" not in decisions:
            decisions = _build_smart_defaults(n_features, n_samples, k, is_image)

        # 3. Determine scaler class based on dataset type
        scaler_class = "MinMaxScaler" if is_image else "StandardScaler"
        normalize = "minmax" if is_image else "standard"

        # Conv-AE vs MLP AE backend selection
        selflabel_import = ""

        if is_image and original_shape and len(original_shape) == 2:
            img_h, img_w = original_shape
            input_size = img_h  # square image assumed
            ae_import = "from ACE_Agent.tools.ae_pipeline import conv_ae_kmeans_pipeline as _ae_pipe"
            dec_import = "from ACE_Agent.tools.dec_pipeline import conv_dec_pipeline as _dec_pipe"
            selflabel_import = "from ACE_Agent.tools.ae_pipeline import conv_selflabel_pipeline as _sl_pipe"
            res_ae_import = ""  # Res-AE is for tabular GAP features, not images
            ae_extra = f", input_size={input_size}, base_filters=32"
            dec_extra = f", input_size={input_size}, base_filters=32"
            # Update decisions with conv-specific defaults
            for key in ("ae_kmeans", "dec", "selflabel"):
                pipe = decisions.setdefault("pipelines", {}).setdefault(key, {})
                pipe.setdefault("latent_dim", 32)
                pipe.setdefault("dropout", 0.2)
                pipe.setdefault("noise_std", 0.1)
                if key == "ae_kmeans":
                    pipe.setdefault("epochs", 150)
                    pipe.setdefault("cluster_method", "gmm")
                    pipe.setdefault("batch_size", 128)
                elif key == "dec":
                    pipe.setdefault("pretrain_epochs", 150)
                    pipe.setdefault("finetune_epochs", 400)
                    pipe.setdefault("gamma", 0.1)
                    pipe.setdefault("batch_size", 128)
                else:
                    pipe.setdefault("ae_epochs", 150)
                    pipe.setdefault("cluster_epochs", 30)
                    pipe.setdefault("n_iterations", 3)
                    pipe.setdefault("contrastive_weight", 0.1)
                    pipe.setdefault("bootstrap", True)
                    pipe.setdefault("augment", True)
        else:
            ae_import = "from ACE_Agent.tools.ae_pipeline import ae_kmeans_pipeline as _ae_pipe"
            dec_import = "from ACE_Agent.tools.dec_pipeline import dec_pipeline as _dec_pipe"
            res_ae_import = "from ACE_Agent.tools.ae_pipeline import res_ae_kmeans_pipeline as _res_ae_pipe"
            ae_extra = ""
            dec_extra = ""

        # 4. Inject decision JSON into skeleton (convert JSON bool/null to Python)
        decisions_repr = json.dumps(decisions, ensure_ascii=False)
        decisions_repr = decisions_repr.replace("true", "True").replace("false", "False").replace("null", "None")
        return (
            _SKELETON.replace("{DECISIONS_JSON}", decisions_repr)
            .replace("{SCALER_CLASS}", scaler_class)
            .replace("{NORMALIZE}", normalize)
            .replace("{AE_IMPORT}", ae_import)
            .replace("{RES_AE_IMPORT}", res_ae_import if not is_image else "")
            .replace("{DEC_IMPORT}", dec_import)
            .replace("{SELFLABEL_IMPORT}", selflabel_import)
            .replace("{AE_EXTRA_ARGS}", ae_extra)
            .replace("{DEC_EXTRA_ARGS}", dec_extra)
        )
