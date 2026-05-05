"""
expert_sub_agents/zoo_expert.py
================================
ZooExpert: 全量算法专家，从 AlgorithmZoo 拉取所有算法并串行运行，
每个算法结果写入 artifacts。

继承 BaseExpert，通过 execute_with_self_correction 走通用的 Think-Act-Fix 循环。
_generate_code 是确定性的（不调用 LLM），直接构造完整可执行代码。

算法覆盖（至少）：
  KMeans, GaussianMixture, DBSCAN, HDBSCAN, AgglomerativeClustering,
  SpectralClustering, OPTICS, Birch, AffinityPropagation, MeanShift

高维数据（>2D）绘图时自动 PCA 降至 2D。

DEPRECATED: 旧 run() 方法保留为向后兼容别名。
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Any

from ACE_Agent.agent_core.schemas import AlgorithmRunResult, DatasetBundle
from ACE_Agent.expert_sub_agents.base import BaseExpert
from ACE_Agent.tools.algorithm_zoo import AlgorithmZoo
from ACE_Agent.tools.llm_client import LLMSettings, UniversalLLMClient

logger = logging.getLogger(__name__)


class ZooExpert(BaseExpert):
    """全量算法专家：确定性地串行运行 AlgorithmZoo 中所有算法。

    _generate_code 不调用 LLM，直接构造完整 Python 代码段，
    通过 BaseExpert 的沙箱执行路径运行，支持自愈重试。
    """

    def __init__(self) -> None:
        super().__init__("zoo", "全量算法专家")
        self.REQUIRES_LLM = False

    # ------------------------------------------------------------------
    # BaseExpert 抽象方法实现
    # ------------------------------------------------------------------

    def _generate_code(
        self,
        client: UniversalLLMClient,  # ZooExpert 不调用 LLM，client 仅满足接口签名
        dataset: DatasetBundle,
        prompt: str,
        constraints=None,
    ) -> str:
        """
        构造完整的串行聚类代码。

        关键设计：
        - 不调用 LLM：zoo 是确定性预置算法库，代码由程序生成。
        - artifacts 约定（全专家通用）：
          artifacts[algo_name] = {
              "labels": <list>,
              "metrics": {"score": float, "silhouette": float,
                          "calinski_harabasz": float, "davies_bouldin": float,
                          [可选] "ari": float, "nmi": float},
              "plot_path": str
          }
          禁止用其他变量名替代 artifacts。
        """
        algos = AlgorithmZoo.get_all_algorithms()
        expected_clusters = dataset.metadata.get("expected_clusters", 3)
        n_features = dataset.X.shape[1] if dataset.X.ndim == 2 else 1

        # 构造每个算法的参数（将占位符替换为实际值）
        algo_configs: list[dict[str, Any]] = []
        for algo in algos:
            params: dict[str, Any] = {}
            for k, v in algo["params"].items():
                params[k] = expected_clusters if v == "expected_clusters" else v
            # DBSCAN: eps=0.5 is tuned for unscaled data; after StandardScaler
            # a tighter eps=0.3 works reliably for low-to-medium noise datasets.
            if algo["name"] == "DBSCAN" and params.get("eps") == 0.5:
                params["eps"] = 0.3
            algo_configs.append({"name": algo["name"], "params": params, "max_samples": algo.get("max_samples")})

        # 序列化算法配置供代码使用
        import json as _json

        algo_configs_repr = _json.dumps(algo_configs, ensure_ascii=False).replace(": null", ": None")

        plot_uses_pca = n_features > 2
        pca_import_line = (
            "from sklearn.decomposition import PCA as _PCA" if plot_uses_pca else "# 数据已是 2D，无需 PCA"
        )
        pca_transform_line = (
            "_X_2d = _PCA(n_components=2, random_state=42).fit_transform(_X_scaled)"
            if plot_uses_pca
            else "_X_2d = _X_scaled"
        )

        # Build code as an unindented multiline string to avoid textwrap.dedent issues
        # caused by injected blocks with different indentation levels.
        lines = [
            "import json as _json",
            "import numpy as _np",
            "import warnings as _warnings",
            "import os as _os",
            '_warnings.filterwarnings("ignore")',
            "",
            "from sklearn.cluster import (",
            "    KMeans, MiniBatchKMeans, DBSCAN, AgglomerativeClustering,",
            "    SpectralClustering, OPTICS, Birch, AffinityPropagation, MeanShift",
            ")",
            "from sklearn.mixture import GaussianMixture",
            "from sklearn.preprocessing import StandardScaler",
            "from sklearn.metrics import (",
            "    silhouette_score, calinski_harabasz_score, davies_bouldin_score,",
            "    adjusted_rand_score, normalized_mutual_info_score",
            ")",
            "",
            "# HDBSCAN: sklearn >= 1.3 内置；如果版本过低则跳过",
            "try:",
            "    from sklearn.cluster import HDBSCAN as _HDBSCAN",
            "    _has_hdbscan = True",
            "except ImportError:",
            "    _has_hdbscan = False",
            "    import logging as _logging_hdb",
            '    _logging_hdb.getLogger("zoo_expert").warning("HDBSCAN 不可用（sklearn < 1.3），已跳过。")',
            "",
            "try:",
            "    import matplotlib",
            '    matplotlib.use("Agg")',
            "    import matplotlib.pyplot as _plt",
            "    _has_matplotlib = True",
            "except ImportError:",
            "    _has_matplotlib = False",
            "",
            "# --- PCA 降维用于绘图（高维数据）---",
            pca_import_line,
            "",
            "def _plot_clusters(X_2d, labels, title, out_path):",
            "    if not _has_matplotlib:",
            "        return out_path",
            "    try:",
            "        _fig, _ax = _plt.subplots(figsize=(6, 4))",
            "        _unique = _np.unique(labels)",
            "        for _lbl in _unique:",
            "            _mask = labels == _lbl",
            '            _ax.scatter(X_2d[_mask, 0], X_2d[_mask, 1], label=f"Cluster {_lbl}", s=12, alpha=0.7)',
            "        _ax.set_title(title)",
            '        _ax.legend(loc="best", fontsize=7)',
            '        _os.makedirs(_os.path.dirname(out_path) if _os.path.dirname(out_path) else ".", exist_ok=True)',
            '        _fig.savefig(out_path, dpi=80, bbox_inches="tight")',
            "        _plt.close(_fig)",
            "        return out_path",
            "    except Exception as _e:",
            '        return f"plot_failed: {_e}"',
            "",
            "def _evaluate(X, y, labels):",
            "    n_labels = len(set(labels) - {-1})",
            "    n_samples = len(labels)",
            "    metrics = {}",
            "    if n_labels < 2 or n_labels >= n_samples:",
            '        metrics["silhouette"] = 0.0',
            '        metrics["calinski_harabasz"] = 0.0',
            '        metrics["davies_bouldin"] = float("inf")',
            '        metrics["score"] = 0.0',
            "        return metrics",
            "    try:",
            '        metrics["silhouette"] = float(silhouette_score(X, labels))',
            "    except Exception:",
            '        metrics["silhouette"] = 0.0',
            "    try:",
            '        metrics["calinski_harabasz"] = float(calinski_harabasz_score(X, labels))',
            "    except Exception:",
            '        metrics["calinski_harabasz"] = 0.0',
            "    try:",
            '        metrics["davies_bouldin"] = float(davies_bouldin_score(X, labels))',
            "    except Exception:",
            '        metrics["davies_bouldin"] = float("inf")',
            "    if y is not None:",
            "        try:",
            '            metrics["ari"] = float(adjusted_rand_score(y, labels))',
            '            metrics["nmi"] = float(normalized_mutual_info_score(y, labels))',
            "        except Exception:",
            "            pass",
            "    # Score priority: ARI (if ground truth) > Silhouette > CH fallback.",
            "    # ARI is unbiased w.r.t. cluster shape; silhouette misranks non-convex clusters.",
            '    if "ari" in metrics:',
            '        metrics["score"] = metrics["ari"]',
            '        metrics["score_source"] = "ari"',
            '    elif metrics["silhouette"] > 0:',
            '        metrics["score"] = metrics["silhouette"]',
            '        metrics["score_source"] = "silhouette"',
            "    else:",
            '        ch = metrics["calinski_harabasz"]',
            '        metrics["score"] = float(ch / (ch + 1000.0)) if ch > 0 else 0.0',
            '        metrics["score_source"] = "calinski_harabasz"',
            "    return metrics",
            "",
            "# --- 准备数据 ---",
            "_scaler = StandardScaler()",
            "_X_scaled = _scaler.fit_transform(X)",
            pca_transform_line,
            "",
            "# --- 算法配置表 ---",
            f"_algo_configs = {algo_configs_repr}",
            "",
            "# --- 大规模数据集熔断: 跳过 O(N²) 算法 ---",
            "_n = X.shape[0]",
            "_skipped = []",
            "for _i in range(len(_algo_configs) - 1, -1, -1):",
            "    _limit = _algo_configs[_i].get('max_samples')",
            "    if _limit is not None and _n > _limit:",
            "        _name = _algo_configs[_i]['name']",
            "        _skipped.append(f'{_name}(N={_n}>{_limit})')",
            "        del _algo_configs[_i]",
            "if _skipped:",
            "    import logging as _zoo_log",
            '    _zoo_log.getLogger("zoo_expert").warning(f"大数据集( N={_n} )跳过 O(N²) 算法: {_skipped}")',
            "",
            "# --- 算法类映射 ---",
            "_algo_map = {",
            '    "KMeans": KMeans,',
            '    "MiniBatchKMeans": MiniBatchKMeans,',
            '    "GaussianMixture": GaussianMixture,',
            '    "DBSCAN": DBSCAN,',
            '    "AgglomerativeClustering": AgglomerativeClustering,',
            '    "SpectralClustering": SpectralClustering,',
            '    "OPTICS": OPTICS,',
            '    "Birch": Birch,',
            '    "AffinityPropagation": AffinityPropagation,',
            '    "MeanShift": MeanShift,',
            "}",
            "if _has_hdbscan:",
            '    _algo_map["HDBSCAN"] = _HDBSCAN',
            "",
            '_output_base = "outputs/zoo"',
            "",
            "for _cfg in _algo_configs:",
            '    _name = _cfg["name"]',
            '    _params = _cfg["params"]',
            "    if _name not in _algo_map:",
            "        continue",
            "    try:",
            "        _Cls = _algo_map[_name]",
            '        if _name == "GaussianMixture":',
            '            _params = {k.replace("n_clusters", "n_components"): v for k, v in _params.items()}',
            "        _model = _Cls(**_params)",
            '        if hasattr(_model, "fit_predict"):',
            "            _labels = _model.fit_predict(_X_scaled)",
            "        else:",
            "            _labels = _model.fit(_X_scaled).predict(_X_scaled)",
            "        _metrics = _evaluate(_X_scaled, y, _labels)",
            '        _plot_path = f"{_output_base}/{_name.lower()}_clusters.png"',
            '        _plot_clusters(_X_2d, _labels, f"{_name} Clustering", _plot_path)',
            '        artifacts[_name] = {"labels": _labels.tolist(), "metrics": _metrics, "plot_path": _plot_path}',
            "    except Exception as _exc:",
            "        import logging as _log",
            '        _log.getLogger("zoo_expert").warning(f"{_name} 运行失败: {_exc}")',
            '        artifacts[_name + "_error"] = {',
            '            "labels": [], "metrics": {"score": 0.0, "error": str(_exc)},',
            '            "plot_path": ""}',
        ]
        return "\n".join(lines) + "\n"

    # ------------------------------------------------------------------
    # 向后兼容别名（deprecated）
    # ------------------------------------------------------------------

    def run(  # noqa: PLR0913
        self,
        dataset: DatasetBundle,
        output_dir: Path,
        algorithm_names: list[str] | None = None,
        *,
        settings: LLMSettings | None = None,
        prompt: str = "运行全量算法",
    ) -> list[AlgorithmRunResult]:
        """
        .. deprecated::
            请改用 execute_with_self_correction()。
            此方法保留仅为向后兼容；内部委托给新接口。
        """
        warnings.warn(
            "ZooExpert.run() is deprecated. Use execute_with_self_correction() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if settings is None:
            from ACE_Agent.tools.llm_client import LLMSettings as _LS

            settings = _LS(enabled=False)
        return self.execute_with_self_correction(dataset, prompt, settings)
