from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class DatasetBundle:
    name: str
    display_name: str
    X: np.ndarray
    y_true: np.ndarray | None
    description: str
    shape_family: str
    feature_names: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)


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
    expected_clusters: int | None
    notes: list[str] = field(default_factory=list)


@dataclass
class ExpertRecommendation:
    expert_key: str
    expert_label: str
    priority: int
    role: str
    reason: str


@dataclass
class RoutingDecision:
    profile: ProfileReport
    selected_experts: list[ExpertRecommendation]
    trace: list[str]


@dataclass
class AlgorithmRunResult:
    expert_key: str
    expert_label: str
    algorithm_name: str
    params: dict[str, Any]
    labels: np.ndarray
    metrics: dict[str, float | int | None]
    narrative: str
    code: str
    plot_path: Path
    artifacts: dict[str, Path] = field(default_factory=dict)
    trace: list[str] = field(default_factory=list)


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
    latex_path: Path
    llm_summary: str | None = None

