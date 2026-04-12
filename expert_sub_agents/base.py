from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False 

from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from ACE_Agent.agent_core.schemas import AlgorithmRunResult, DatasetBundle
from ACE_Agent.tools.coder_sandbox import CoderSandbox


class BaseExpert:
    key = "base"
    label = "Base Expert"

    def __init__(self) -> None:
        self.sandbox = CoderSandbox()

    def run(self, dataset: DatasetBundle, output_dir: Path) -> list[AlgorithmRunResult]:
        raise NotImplementedError

    def _execute_code(
        self,
        dataset: DatasetBundle,
        output_dir: Path,
        algorithm_name: str,
        params: dict[str, Any],
        code: str,
        plot_filename: str,
        trace: list[str] | None = None,
    ) -> AlgorithmRunResult:
        plot_path = output_dir / plot_filename
        execution = self.sandbox.run(
            code,
            {
                "np": np,
                "X": dataset.X.copy(),
                "y_true": None if dataset.y_true is None else dataset.y_true.copy(),
                "params": params,
                "output_path": plot_path,
                "evaluate_labels": evaluate_labels,
                "save_cluster_plot": save_cluster_plot,
                "estimate_dbscan_eps": estimate_dbscan_eps,
                "StandardScaler": StandardScaler,
            },
        )
        labels = np.asarray(execution.result["labels"])
        metrics = dict(execution.result["metrics"])
        narrative = build_narrative(
            expert_label=self.label,
            algorithm_name=algorithm_name,
            dataset_name=dataset.display_name,
            metrics=metrics,
        )
        return AlgorithmRunResult(
            expert_key=self.key,
            expert_label=self.label,
            algorithm_name=algorithm_name,
            params=params,
            labels=labels,
            metrics=metrics,
            narrative=narrative,
            code=execution.code,
            plot_path=Path(execution.result["plot_path"]),
            artifacts={},
            trace=trace or [],
        )


def evaluate_labels(X: np.ndarray, y_true: np.ndarray | None, labels: np.ndarray) -> dict[str, float | int | None]:
    labels = np.asarray(labels)
    unique_labels = np.unique(labels)
    noise_ratio = float(np.mean(labels == -1))
    cluster_labels = [label for label in unique_labels if label != -1]
    cluster_count = len(cluster_labels)
    metrics: dict[str, float | int | None] = {
        "cluster_count": int(cluster_count),
        "noise_ratio": noise_ratio,
        "ami": None,
        "ari": None,
        "silhouette": None,
        "davies_bouldin": None,
        "calinski_harabasz": None,
    }

    if cluster_count > 1 and cluster_count < len(labels):
        try:
            metrics["silhouette"] = float(silhouette_score(X, labels))
        except Exception:
            metrics["silhouette"] = None
        try:
            metrics["davies_bouldin"] = float(davies_bouldin_score(X, labels))
        except Exception:
            metrics["davies_bouldin"] = None
        try:
            metrics["calinski_harabasz"] = float(calinski_harabasz_score(X, labels))
        except Exception:
            metrics["calinski_harabasz"] = None

    if y_true is not None and cluster_count > 1:
        try:
            metrics["ami"] = float(adjusted_mutual_info_score(y_true, labels))
        except Exception:
            metrics["ami"] = None
        try:
            metrics["ari"] = float(adjusted_rand_score(y_true, labels))
        except Exception:
            metrics["ari"] = None

    metrics["score"] = composite_score(metrics)
    return metrics


def composite_score(metrics: dict[str, float | int | None]) -> float:
    score = 0.0
    if metrics.get("ami") is not None:
        score += 0.35 * max(float(metrics["ami"]), 0.0)
    if metrics.get("ari") is not None:
        score += 0.15 * max(float(metrics["ari"]), 0.0)
    if metrics.get("silhouette") is not None:
        score += 0.20 * ((float(metrics["silhouette"]) + 1.0) / 2.0)
    if metrics.get("davies_bouldin") is not None:
        score += 0.15 * (1.0 / (1.0 + float(metrics["davies_bouldin"])))
    if metrics.get("calinski_harabasz") is not None:
        score += 0.15 * min(math.log1p(float(metrics["calinski_harabasz"])) / 6.0, 1.0)
    score -= max(float(metrics.get("noise_ratio", 0.0)) - 0.2, 0.0) * 0.05
    return round(score, 4)


def estimate_dbscan_eps(X: np.ndarray, min_samples: int = 6) -> float:
    scaled = StandardScaler().fit_transform(X)
    neighbors = NearestNeighbors(n_neighbors=min_samples)
    neighbors.fit(scaled)
    distances, _ = neighbors.kneighbors(scaled)
    kth_distance = np.sort(distances[:, -1])
    return float(np.quantile(kth_distance, 0.82))


def save_cluster_plot(
    X: np.ndarray,
    labels: np.ndarray,
    output_path: str | Path,
    title: str,
) -> str:
    output_path = Path(output_path)
    vis = _project_for_visualization(X)
    plt.figure(figsize=(6, 4.5))
    unique = list(np.unique(labels))
    color_map = plt.cm.get_cmap("tab10", max(len(unique), 3))
    for idx, label in enumerate(unique):
        mask = labels == label
        display_name = "噪声点" if label == -1 else f"聚类 {int(label)}"
        color = "#606060" if label == -1 else color_map(idx)
        plt.scatter(vis[mask, 0], vis[mask, 1], s=18, alpha=0.85, label=display_name, color=color)
    plt.title(title)
    plt.xlabel("维度 1")
    plt.ylabel("维度 2")
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()
    return str(output_path)


def save_dataset_preview(dataset: DatasetBundle, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / f"{dataset.name}_dataset.png"
    labels = dataset.y_true if dataset.y_true is not None else np.zeros(dataset.X.shape[0], dtype=int)
    save_cluster_plot(dataset.X, labels, target, f"{dataset.display_name} preview")
    return target


def build_narrative(
    expert_label: str,
    algorithm_name: str,
    dataset_name: str,
    metrics: dict[str, float | int | None],
) -> str:
    ami = _format_metric(metrics.get("ami"))
    silhouette = _format_metric(metrics.get("silhouette"))
    score = _format_metric(metrics.get("score"))
    cluster_count = metrics.get("cluster_count", 0)
    return (
        f"{expert_label} 对 {dataset_name} 使用了 {algorithm_name} 算法。 "
        f"该方法发现了 {cluster_count} 个簇，综合得分为 {score}， "
        f"AMI 为 {ami}，轮廓系数为 {silhouette}。"
    )


def _project_for_visualization(X: np.ndarray) -> np.ndarray:
    if X.shape[1] <= 2:
        return X
    reducer = PCA(n_components=2, random_state=42)
    return reducer.fit_transform(X)


def _format_metric(value: float | int | None) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.3f}"
