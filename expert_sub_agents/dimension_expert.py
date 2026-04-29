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
_scaler = StandardScaler()
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
# ========================================================================
_AE_OK = False
try:
    from ACE_Agent.tools.ae_pipeline import ae_kmeans_pipeline as _ae_pipe
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
        )
        artifacts["AE_KMeans"] = _result5
    except Exception as _e5:
        artifacts["AE_KMeans_error"] = {
            "labels": [], "metrics": {"score": 0.0, "error": str(_e5)}, "plot_path": ""}
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
    "5. `ae_kmeans`    — AutoEncoder 深度降维 + KMeans (仅 n_features>32 自动激活)\n\n"
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
    '                    "noise_std": <float>, "cluster_method": "gmm"}\n'
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


def _build_smart_defaults(n_features: int, n_samples: int, k: int) -> dict[str, Any]:
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

    if n_features > 128:
        hidden_dims = [256, 128, 64, 32] if not tiny_dataset else [128, 64]
        ae_epochs = 200
        ae_lr = 5e-4
        ae_dropout = 0.3 if small_dataset else 0.2
    elif n_features > 64:
        hidden_dims = [128, 64, 32] if not tiny_dataset else [64, 32]
        ae_epochs = 120
        ae_lr = 5e-4 if small_dataset else 1e-3
        ae_dropout = 0.3 if small_dataset else 0.2
    elif n_features > 32:
        hidden_dims = [128, 64, 32] if not small_dataset else [64, 32]
        ae_epochs = 80
        ae_lr = 5e-4 if small_dataset else 1e-3
        ae_dropout = 0.3 if small_dataset else 0.2
    else:
        hidden_dims = [64, 32]
        ae_epochs = 30
        ae_lr = 1e-3
        ae_dropout = 0.2

    ae_patience = 10 if small_dataset else 15
    ae_noise = 0.35 if small_dataset else 0.15
    ae_cluster = "gmm"

    return {
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
        }
    }


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
        try:
            from ACE_Agent.tools.ae_pipeline import ae_kmeans_pipeline  # noqa: F811

            self.PRE_INJECT = {"ae_kmeans_pipeline": ae_kmeans_pipeline}
        except Exception:
            self.PRE_INJECT = {}

    def _generate_code(
        self,
        client: UniversalLLMClient,
        dataset: DatasetBundle,
        prompt: str,
    ) -> str:
        n_features = dataset.X.shape[1] if dataset.X.ndim == 2 else 1
        n_samples = dataset.X.shape[0]
        k = dataset.metadata.get("expected_clusters", 3) if dataset.metadata else 3

        # 1. Call LLM for pipeline decisions (JSON output)
        small_ds = n_samples < 500
        tiny_ds = n_samples < 200
        if n_features > 128:
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
            _DECISION_SYSTEM_PROMPT,
        ).strip()

        # 2. Parse decision (or use smart defaults on failure)
        decisions = _extract_json(raw)
        if not decisions or "pipelines" not in decisions:
            decisions = _build_smart_defaults(n_features, n_samples, k)

        # 3. Inject decision JSON into skeleton (convert JSON bool/null to Python)
        decisions_repr = json.dumps(decisions, ensure_ascii=False)
        decisions_repr = decisions_repr.replace("true", "True").replace("false", "False").replace("null", "None")
        return _SKELETON.replace("{DECISIONS_JSON}", decisions_repr)
