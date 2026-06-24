from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class DatasetBundle:
    name: str
    X: np.ndarray
    y: np.ndarray | None = None
    display_name: str = ""
    description: str = ""
    shape_family: str = "generic"  # e.g., "non_convex", "manifold"
    feature_names: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    feature_mode: str = ""  # e.g., "raw", "resnet", "gap", "cnn_features"


@dataclass
class ModalityProfile:
    """Unified modality descriptor for routing, graph construction, and audit.

    Replaces scattered metadata checks (is_time_series, is_text, is_image,
    is_sparse) with a single structured profile derived once from DatasetBundle
    at pipeline entry.
    """
    modality_type: str = "tabular"
    # "tabular" | "time_series" | "text" | "image"

    distance_metric: str = "euclidean"
    # Metric for k-NN graph construction via sklearn.neighbors.kneighbors_graph.
    # "euclidean" | "cosine".
    # DTW is NOT natively supported by sklearn — time-series graph checks use
    # Euclidean on flattened features as an approximation; actual DTW clustering
    # happens in expert sandboxes via TimeSeriesKMeans.

    l2_normalize: bool = False
    # When True, supervisor L2-normalizes X so Euclidean distance ≡ cosine.

    ts_shape: tuple[int, int] | None = None
    # (time_steps, features_per_step) for time-series reshape in experts.

    ts_original_shape: tuple[int, int] | None = None
    # Original shape before PCA/dim-reduction, if applicable.

    ts_large_n: bool = False
    # N > 500 — experts should use Sakoe-Chiba band or downsample for DTW.

    dim_reduction_hint: str = "pca"
    # Preferred dim-reduction: "pca" | "truncated_svd" | "umap" | "none"

    metadata: dict[str, Any] = field(default_factory=dict)
    # Passthrough of dataset.metadata for sandbox CTX_DATA access.


def detect_modality(dataset: DatasetBundle) -> ModalityProfile:
    """Derive a ModalityProfile from a DatasetBundle.

    Priority (first match wins):
      1. metadata["is_time_series"]  → time_series
      2. metadata["is_text"] or shape_family=="sparse"  → text
      3. metadata["is_image"]  → image
      4. Default → tabular + euclidean
    """
    md = getattr(dataset, "metadata", {}) or {}
    sf = getattr(dataset, "shape_family", "")
    n_samples = dataset.X.shape[0]

    # --- Time-series ---
    if md.get("is_time_series"):
        ts_shape = md.get("ts_shape")
        ts_orig = md.get("ts_shape_original")
        ts_valid = (
            isinstance(ts_shape, (list, tuple))
            and len(ts_shape) == 2
            and all(isinstance(d, int) for d in ts_shape)
        )
        return ModalityProfile(
            modality_type="time_series",
            distance_metric="euclidean",
            l2_normalize=False,
            ts_shape=tuple(ts_shape) if ts_valid else None,
            ts_original_shape=(
                tuple(ts_orig)
                if isinstance(ts_orig, (list, tuple)) and len(ts_orig) == 2
                else None
            ),
            ts_large_n=n_samples > 500,
            dim_reduction_hint="none" if ts_valid else "pca",
            metadata=md,
        )

    # --- Text / sparse ---
    if md.get("is_text") or sf == "sparse":
        return ModalityProfile(
            modality_type="text",
            distance_metric="cosine",
            l2_normalize=True,
            dim_reduction_hint="truncated_svd",
            metadata=md,
        )

    # --- Image ---
    if md.get("is_image"):
        return ModalityProfile(
            modality_type="image",
            distance_metric="euclidean",
            l2_normalize=False,
            dim_reduction_hint="pca",
            metadata=md,
        )

    # --- Default (tabular / generic) ---
    return ModalityProfile(
        modality_type="tabular",
        distance_metric="euclidean",
        metadata=md,
    )


@dataclass
class ExpertRecommendation:
    expert_key: str
    expert_label: str
    priority: int
    role: str
    reason: str


@dataclass
class ChatMessage:
    role: str  # "user" or "assistant"
    content: str


@dataclass
class ProfileReport:
    sample_count: int
    feature_count: int
    negative_ratio: float
    sparsity_ratio: float
    avg_abs_correlation: float
    manifold_hint: bool
    non_convex_hint: bool
    noise_sensitive_hint: bool
    expected_clusters: int | None = None
    structure_class: str = "generic"  # Phase 3: spherical/non_convex/manifold/graph_connected
    geodesic_distortion: float | None = None  # Phase 3: Euclidean-vs-geodesic distortion
    notes: list[str] = field(default_factory=list)
    modality_type: str = "tabular"
    modality_metric: str = "euclidean"


@dataclass
class RoutingDecision:
    profile: ProfileReport
    selected_experts: list[ExpertRecommendation]
    trace: list[str]
    modality: ModalityProfile | None = None


@dataclass
class AlgorithmRunResult:
    algorithm_name: str
    expert_key: str
    expert_label: str
    labels: np.ndarray
    metrics: dict[str, Any]
    plot_path: Path
    code: str = ""  # 存储执行的 Python 代码片段
    params: dict[str, Any] = field(default_factory=dict)
    embedding_path: Path | None = None  # 降维嵌入文件路径，供审计在优胜者特征空间中评估


@dataclass
class SupervisorReport:
    dataset: DatasetBundle
    routing: RoutingDecision
    dataset_plot_path: Path
    output_dir: Path
    results: list[AlgorithmRunResult]
    ranking: list[AlgorithmRunResult]
    executive_summary: str
    decision_trace: list[str]
    latex_path: Path | None = None
    llm_summary: str | None = None
    audit_report: dict[str, Any] | None = None  # Critic post-hoc audit result
    response_type: str = "CLUSTER_TASK"  # "CLUSTER_TASK" 或 "FOLLOW_UP"
