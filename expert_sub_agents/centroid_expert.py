from __future__ import annotations

import textwrap
from pathlib import Path

from ACE_Agent.agent_core.schemas import AlgorithmRunResult, DatasetBundle
from ACE_Agent.expert_sub_agents.base import BaseExpert


class CentroidExpert(BaseExpert):
    key = "centroid"
    label = "质心专家"

    def run(self, dataset: DatasetBundle, output_dir: Path) -> list[AlgorithmRunResult]:
        expected_clusters = int(dataset.metadata.get("expected_clusters", 3))
        results = []

        kmeans_code = textwrap.dedent(
            f"""
            from sklearn.cluster import KMeans

            scaled = StandardScaler().fit_transform(X)
            model = KMeans(n_clusters={expected_clusters}, n_init=20, random_state=42)
            labels = model.fit_predict(scaled)
            metrics = evaluate_labels(X, y_true, labels)
            plot_path = save_cluster_plot(X, labels, output_path, "质心专家 - KMeans")
            result = {{
                "labels": labels.tolist(),
                "metrics": metrics,
                "plot_path": plot_path,
            }}
            """
        )
        results.append(
            self._execute_code(
                dataset=dataset,
                output_dir=output_dir,
                algorithm_name="KMeans",
                params={"n_clusters": expected_clusters, "n_init": 20},
                code=kmeans_code,
                plot_filename=f"{dataset.name}_centroid_kmeans.png",
                trace=[
                    "在运行 KMeans 前对特征空间进行了标准化。",
                    "使用了数据集元数据中的预期簇数。",
                ],
            )
        )

        gmm_code = textwrap.dedent(
            f"""
            from sklearn.mixture import GaussianMixture

            scaled = StandardScaler().fit_transform(X)
            model = GaussianMixture(n_components={expected_clusters}, covariance_type="full", random_state=42)
            labels = model.fit_predict(scaled)
            metrics = evaluate_labels(X, y_true, labels)
            plot_path = save_cluster_plot(X, labels, output_path, "质心专家 - GMM")
            result = {{
                "labels": labels.tolist(),
                "metrics": metrics,
                "plot_path": plot_path,
            }}
            """
        )
        results.append(
            self._execute_code(
                dataset=dataset,
                output_dir=output_dir,
                algorithm_name="GaussianMixture",
                params={"n_components": expected_clusters, "covariance_type": "full"},
                code=gmm_code,
                plot_filename=f"{dataset.name}_centroid_gmm.png",
                trace=[
                    "拟合了全协方差的高斯混合模型以允许椭圆形的簇结构。",
                    "保持了相同的目标簇数，以便进行公平的质心类方法对比。",
                ],
            )
        )
        return results

