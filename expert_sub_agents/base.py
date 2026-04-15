from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
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
        # 准备沙箱环境
        context = {
            "X": dataset.X,
            "y": dataset.y,
            "output_path": output_dir / plot_filename,
            "params": params,
            "evaluate_labels": evaluate_labels,
            "save_cluster_plot": save_cluster_plot,
            "estimate_dbscan_eps": estimate_dbscan_eps,
            "StandardScaler": StandardScaler,
        }

        # 运行沙箱代码
        sandbox_result = self.sandbox.run(code, context)
        
        if sandbox_result.error:
            raise RuntimeError(f"Sandbox execution failed: {sandbox_result.error}")
        
        exec_res = sandbox_result.result
        if not isinstance(exec_res, dict) or "labels" not in exec_res:
            raise ValueError(f"Invalid sandbox result from {algorithm_name}: {exec_res}")

        return AlgorithmRunResult(
            algorithm_name=algorithm_name,
            expert_key=self.key,
            expert_label=self.label,
            labels=np.array(exec_res["labels"]),
            metrics=exec_res.get("metrics", {}),
            plot_path=exec_res.get("plot_path", output_dir / plot_filename),
            code=code,
            params=params,
        )


def evaluate_labels(X: np.ndarray, y_true: np.ndarray | None, labels: np.ndarray) -> dict[str, Any]:
    """
    统一评估聚类结果。支持监督指标（如果 y_true 可用）和无监督指标。
    """
    metrics = {}
    
    # 无监督指标（始终可以计算，只要簇数 > 1）
    n_clusters = len(np.unique(labels[labels != -1]))
    if n_clusters > 1:
        try:
            metrics["silhouette"] = float(silhouette_score(X, labels))
            metrics["calinski_harabasz"] = float(calinski_harabasz_score(X, labels))
            metrics["davies_bouldin"] = float(davies_bouldin_score(X, labels))
        except:
            metrics["silhouette"] = 0.0

    # 监督指标（如果存在真实标签）
    if y_true is not None:
        metrics["ami"] = float(adjusted_mutual_info_score(y_true, labels))
        metrics["ari"] = float(adjusted_rand_score(y_true, labels))
        # 综合得分：平均 AMI 和 Silhouette
        s_score = metrics.get("silhouette", 0.0)
        metrics["score"] = (metrics["ami"] + max(0, s_score)) / 2.0
    else:
        # 无监督情况下的综合得分：直接使用轮廓系数
        metrics["score"] = max(0, metrics.get("silhouette", 0.0))

    return metrics


def save_cluster_plot(X: np.ndarray, labels: np.ndarray, output_path: Path, title: str) -> Path:
    """
    绘制聚类结果图，支持 2D 和 3D。
    """
    plt.figure(figsize=(10, 7))
    
    # 如果维度超过 2，使用 PCA 降维绘图
    if X.shape[1] > 2:
        X_plot = PCA(n_components=2).fit_transform(X)
        title += " (PCA 2D Projection)"
    else:
        X_plot = X

    unique_labels = np.unique(labels)
    colors = plt.cm.get_cmap("tab10")(np.linspace(0, 1, len(unique_labels)))

    for i, label in enumerate(unique_labels):
        mask = (labels == label)
        display_name = "噪声点" if label == -1 else f"簇 {int(label)}"
        plt.scatter(
            X_plot[mask, 0], 
            X_plot[mask, 1], 
            c=[colors[i]], 
            label=display_name, 
            alpha=0.7, 
            edgecolors='w'
        )

    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()
    return output_path


def estimate_dbscan_eps(X: np.ndarray, quantile: float = 0.05) -> float:
    """
    通过最近邻距离分布自动估计 DBSCAN 的 eps 参数。
    """
    neigh = NearestNeighbors(n_neighbors=5)
    neigh.fit(X)
    distances, _ = neigh.kneighbors(X)
    # 取第五个近邻距离的某个分位数
    eps = np.quantile(distances[:, -1], quantile)
    return max(float(eps), 0.01)


def save_dataset_preview(dataset: DatasetBundle, output_dir: Path) -> Path:
    """
    保存原始数据集的预览图。
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / f"{dataset.name}_dataset.png"
    # 如果有真实标签，则按标签着色预览
    labels = dataset.y if dataset.y is not None else np.zeros(dataset.X.shape[0], dtype=int)
    save_cluster_plot(dataset.X, labels, target, f"{dataset.display_name} 原始分布")
    return target


def build_narrative(results: list[AlgorithmRunResult]) -> str:
    """
    根据运行结果构建简单的描述文本（用于生成报告）。
    """
    if not results:
        return "未能获得有效结果。"
    best = max(results, key=lambda r: r.metrics.get("score", 0.0))
    return f"在该专家组内，{best.algorithm_name} 表现最佳，得分为 {best.metrics.get('score', 0.0):.3f}。"
