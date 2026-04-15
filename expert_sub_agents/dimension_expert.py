from __future__ import annotations

import textwrap
from pathlib import Path

from ACE_Agent.agent_core.schemas import AlgorithmRunResult, DatasetBundle
from ACE_Agent.expert_sub_agents.base import BaseExpert


class DimensionExpert(BaseExpert):
    key = "dimension"
    label = "维度专家"

    def run(self, dataset: DatasetBundle, output_dir: Path) -> list[AlgorithmRunResult]:
        expected_clusters = int(dataset.metadata.get("expected_clusters", 3))
        results = []

        pca_code = textwrap.dedent(
            f"""
            from sklearn.cluster import KMeans
            from sklearn.decomposition import PCA
            from sklearn.preprocessing import StandardScaler

            scaled = StandardScaler().fit_transform(X)
            embedding = PCA(n_components=2, random_state=42).fit_transform(scaled)
            labels = KMeans(n_clusters={expected_clusters}, n_init=20, random_state=42).fit_predict(embedding)
            metrics = evaluate_labels(embedding, y, labels)
            plot_path = save_cluster_plot(embedding, labels, output_path, "维度专家 - PCA + KMeans")
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
                algorithm_name="PCAPlusKMeans",
                params={"n_clusters": expected_clusters, "embedding": "PCA(2)"},
                code=pca_code,
                plot_filename=f"{dataset.name}_dimension_pca_kmeans.png",
                trace=[
                    "在聚类前将数据压缩到两个主成分方向。",
                    "使用 PCA 作为检查可分性的简单维度基准。",
                ],
            )
        )

        spectral_code = textwrap.dedent(
            f"""
            from sklearn.cluster import KMeans
            from sklearn.manifold import SpectralEmbedding
            from sklearn.preprocessing import StandardScaler

            scaled = StandardScaler().fit_transform(X)
            embedding = SpectralEmbedding(n_components=2, n_neighbors=18, random_state=42).fit_transform(scaled)
            labels = KMeans(n_clusters={expected_clusters}, n_init=20, random_state=42).fit_predict(embedding)
            metrics = evaluate_labels(embedding, y, labels)
            plot_path = save_cluster_plot(embedding, labels, output_path, "维度专家 - 谱嵌入 + KMeans")
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
                algorithm_name="SpectralEmbeddingPlusKMeans",
                params={"embedding": "SpectralEmbedding(2)", "n_clusters": expected_clusters},
                code=spectral_code,
                plot_filename=f"{dataset.name}_dimension_spectral_kmeans.png",
                trace=[
                    "使用基于图的谱嵌入投影数据，以保留流形邻域结构。",
                    "在弯曲结构展开后，对嵌入空间使用 KMeans 进行聚类。",
                ],
            )
        )
        return results
