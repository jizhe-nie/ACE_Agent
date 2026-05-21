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
        # Critic 2.0: enforce blocked_algorithms constraint at code-gen level
        if constraints and constraints.get("blocked_algorithms"):
            _blocked = set(constraints["blocked_algorithms"])
            algos = [a for a in algos if a["name"] not in _blocked]
        expected_clusters = dataset.metadata.get("expected_clusters", 3)
        n_features = dataset.X.shape[1] if dataset.X.ndim == 2 else 1

        # ---- Time-series DTW detection -----------------------------------
        _is_ts = (
            isinstance(dataset.metadata, dict)
            and dataset.metadata.get("is_time_series")
        )
        _ts_shape = dataset.metadata.get("ts_shape") if _is_ts else None
        _ts_valid = (
            _ts_shape is not None
            and isinstance(_ts_shape, (list, tuple))
            and len(_ts_shape) == 2
            and all(isinstance(d, int) for d in _ts_shape)
        )
        if _ts_valid:
            _ts_T, _ts_F = int(_ts_shape[0]), int(_ts_shape[1])
        else:
            _ts_T = _ts_F = 0

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
            # SpectralClustering: larger k with mutual k-NN for spread-out manifold classes.
            # Mutual k-NN removes cross-class shortcut edges, so we can afford a larger
            # base k to maintain connectivity of extended manifold classes (e.g., parabola curves).
            # For n=750, k≈22 (vs. old k=7), mutual filtering keeps effective degree ≈8-12.
            if algo["name"] == "SpectralClustering" and params.get("affinity") == "nearest_neighbors":
                import numpy as _np_zoo
                _n_samp = dataset.X.shape[0]
                _n_feat = dataset.X.shape[1] if dataset.X.ndim == 2 else 1
                if _n_feat <= 3:
                    # 2D/3D: 3% of n, bounded [10, 50]
                    params["n_neighbors"] = max(10, min(50, int(0.03 * _n_samp)))
                else:
                    # High-dim: sqrt-based, bounded [10, 50]
                    params["n_neighbors"] = max(10, min(50, int(_np_zoo.sqrt(_n_samp)) // 2))
            algo_configs.append({"name": algo["name"], "params": params, "max_samples": algo.get("max_samples")})

        # 序列化算法配置供代码使用
        import json as _json

        algo_configs_repr = _json.dumps(algo_configs, ensure_ascii=False).replace(": null", ": None").replace(": true", ": True").replace(": false", ": False")

        plot_uses_pca = n_features > 2
        pca_import_line = (
            "from sklearn.decomposition import PCA as _PCA" if plot_uses_pca else "# 数据已是 2D，无需 PCA"
        )
        pca_transform_line = (
            "_X_2d = _PCA(n_components=2, random_state=42).fit_transform(_X_scaled)"
            if plot_uses_pca
            else "_X_2d = _X_scaled"
        )

        # ---- Conditional blocks for time-series DTW support --------------
        if _ts_valid:
            _tslearn_import_lines = [
                "",
                "# --- Time-series DTW imports (tslearn) ---",
                "try:",
                "    from tslearn.clustering import TimeSeriesKMeans as _TSKMeans",
                "    import tslearn.metrics as _ts_metrics",
                "    _has_tslearn = True",
                "except ImportError:",
                "    _has_tslearn = False",
                '    import logging as _log_ts',
                '    _log_ts.getLogger("zoo_expert").warning("tslearn 不可用，跳过 DTW 聚类。")',
            ]
        else:
            _tslearn_import_lines = [""]

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
            *_tslearn_import_lines,
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
            "        _fig, _ax = _plt.subplots(figsize=(8, 6))",
            "        _unique = _np.unique(labels)",
            "        for _lbl in _unique:",
            "            _mask = labels == _lbl",
            '            _ax.scatter(X_2d[_mask, 0], X_2d[_mask, 1], label=f"Cluster {_lbl}", s=12, alpha=0.7)',
            "        _ax.set_title(title)",
            '        _ax.legend(loc="best", fontsize=7)',
            '        _os.makedirs(_os.path.dirname(out_path) if _os.path.dirname(out_path) else ".", exist_ok=True)',
            '        _fig.savefig(out_path, dpi=150, bbox_inches="tight")',
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
            '_output_base = ACE_OUTPUT_DIR + "/zoo" if ACE_OUTPUT_DIR else "outputs/zoo"',
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
            "        # SpectralClustering with nearest_neighbors: build mutual k-NN",
            "        # to eliminate cross-manifold shortcut edges (Half-kernel fix).",
            '        _is_spec_mutual = _name == "SpectralClustering" and _params.get("affinity") == "nearest_neighbors"',
            "        if _is_spec_mutual:",
            "            _k_spec = _params.pop('n_neighbors', 5)",
            '            _adj_fwd_spec = kneighbors_graph(_X_scaled, _k_spec, mode="connectivity", include_self=False)',
            "            _adj_mutual_spec = _adj_fwd_spec.minimum(_adj_fwd_spec.T)",
            '            _params["affinity"] = "precomputed"',
            "            _model = _Cls(**_params)",
            "            _labels = _model.fit_predict(_adj_mutual_spec)",
            "        else:",
            "            _model = _Cls(**_params)",
            '            if hasattr(_model, "fit_predict"):',
            "                _labels = _model.fit_predict(_X_scaled)",
            "            else:",
            "                _labels = _model.fit(_X_scaled).predict(_X_scaled)",
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
        # ---- Append DTW execution block when time-series is detected ----
        if _ts_valid:
            lines.extend([
                "",
                "# --- Time-series DTW pipelines ---",
                "if _has_tslearn:",
                f"    _X_ts = X.reshape(_n, {_ts_T}, {_ts_F})",
                f"    _k = {expected_clusters}",
                "",
                "    # DTW Pipeline 1: TimeSeriesKMeans with Sakoe-Chiba on large N",
                "    try:",
                "        if _n > 500:",
                '            _tskm = _TSKMeans(',
                '                n_clusters=_k, metric="dtw",',
                '                metric_params={"global_constraint": "sakoe_chiba",',
                '                               "sakoe_chiba_radius": 2},',
                "                max_iter=10, random_state=42, n_jobs=1)",
                "        else:",
                '            _tskm = _TSKMeans(',
                '                n_clusters=_k, metric="dtw",',
                "                max_iter=10, random_state=42, n_jobs=1)",
                "        _ts_labels = _tskm.fit_predict(_X_ts)",
                "        _ts_metrics = _evaluate(_X_scaled, y, _ts_labels)",
                '        _plot_clusters(_X_2d, _ts_labels, "TimeSeriesKMeans (DTW)",'
                '                        f"{_output_base}/timeseries_kmeans_clusters.png")',
                '        artifacts["TimeSeriesKMeans"] = {',
                '            "labels": _ts_labels.tolist(), "metrics": _ts_metrics,',
                '            "plot_path": f"{_output_base}/timeseries_kmeans_clusters.png"}',
                "    except Exception as _exc:",
                "        import logging as _log_ts1",
                '        _log_ts1.getLogger("zoo_expert").warning(',
                '            f"TimeSeriesKMeans 运行失败: {_exc}")',
                "",
                "    # DTW Pipeline 2: SpectralClustering with DTW affinity matrix",
                "    try:",
                "        if _n > 500:",
                "            _dtw_dist = _ts_metrics.cdist_dtw(",
                '                _X_ts, global_constraint="sakoe_chiba", sakoe_chiba_radius=2)',
                "        else:",
                "            _dtw_dist = _ts_metrics.cdist_dtw(_X_ts)",
                "        _sigma = float(_np.median(_dtw_dist[_dtw_dist > 0])) \\",
                "                 if _np.any(_dtw_dist > 0) else 1.0",
                "        _aff = _np.exp(-_dtw_dist / max(_sigma, 1e-10))",
                '        _spec = SpectralClustering(',
                '            n_clusters=_k, affinity="precomputed", random_state=42)',
                "        _sp_labels = _spec.fit_predict(_aff)",
                "        _sp_metrics = _evaluate(_X_scaled, y, _sp_labels)",
                '        _plot_clusters(_X_2d, _sp_labels, "SpectralClustering (DTW affinity)",'
                '                        f"{_output_base}/spectral_dtw_clusters.png")',
                '        artifacts["SpectralDTW"] = {',
                '            "labels": _sp_labels.tolist(), "metrics": _sp_metrics,',
                '            "plot_path": f"{_output_base}/spectral_dtw_clusters.png"}',
                "    except Exception as _exc:",
                "        import logging as _log_ts2",
                '        _log_ts2.getLogger("zoo_expert").warning(',
                '            f"SpectralDTW 运行失败: {_exc}")',
            ])
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
