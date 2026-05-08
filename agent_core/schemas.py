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


@dataclass
class RoutingDecision:
    profile: ProfileReport
    selected_experts: list[ExpertRecommendation]
    trace: list[str]


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
