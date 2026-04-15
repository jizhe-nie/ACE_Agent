from __future__ import annotations

import textwrap
from pathlib import Path

from ACE_Agent.agent_core.schemas import AlgorithmRunResult, DatasetBundle
from ACE_Agent.expert_sub_agents.base import BaseExpert


class MultiViewExpert(BaseExpert):
    key = "multi_view"
    label = "多视图专家"

    def run(self, dataset: DatasetBundle, output_dir: Path) -> list[AlgorithmRunResult]:
        expected_clusters = int(dataset.metadata.get("expected_clusters", 3))

        consensus_code = textwrap.dedent(
            f"""
            import numpy as np
            from sklearn.cluster import AgglomerativeClustering, KMeans
            from sklearn.decomposition import PCA
            from sklearn.manifold import SpectralEmbedding
            from sklearn.preprocessing import StandardScaler

            scaled = StandardScaler().fit_transform(X)
            # 视图 A: 原始特征空间
            view_a = scaled
            # 视图 B: 谱嵌入空间
            view_b = SpectralEmbedding(n_components=2, n_neighbors=12, random_state=42).fit_transform(scaled)
            # 视图 C: PCA 主成分空间
            view_c = PCA(n_components=2, random_state=42).fit_transform(scaled)

            labels_a = KMeans(n_clusters={expected_clusters}, n_init=20, random_state=42).fit_predict(view_a)
            labels_b = KMeans(n_clusters={expected_clusters}, n_init=20, random_state=42).fit_predict(view_b)
            labels_c = KMeans(n_clusters={expected_clusters}, n_init=20, random_state=42).fit_predict(view_c)

            # 构建协同关联矩阵 (Co-association Matrix)
            coassoc = (
                (labels_a[:, None] == labels_a[None, :]).astype(float)
                + (labels_b[:, None] == labels_b[None, :]).astype(float)
                + (labels_c[:, None] == labels_c[None, :]).astype(float)
            ) / 3.0
            distance = 1.0 - coassoc

            # 兼容不同版本的 sklearn
            try:
                consensus = AgglomerativeClustering(
                    n_clusters={expected_clusters},
                    linkage="average",
                    metric="precomputed",
                )
            except TypeError:
                consensus = AgglomerativeClustering(
                    n_clusters={expected_clusters},
                    linkage="average",
                    affinity="precomputed",
                )

            labels = consensus.fit_predict(distance)
            metrics = evaluate_labels(X, y, labels)
            plot_path = save_cluster_plot(X, labels, output_path, "多视图专家 - 共识聚类")
            result = {{
                "labels": labels.tolist(),
                "metrics": metrics,
                "plot_path": plot_path,
            }}
            """
        )
        return [
            self._execute_code(
                dataset=dataset,
                output_dir=output_dir,
                algorithm_name="ConsensusFusion",
                params={"views": ["scaled", "spectral", "pca"], "n_clusters": expected_clusters},
                code=consensus_code,
                plot_filename=f"{dataset.name}_multiview_consensus.png",
                trace=[
                    "构建了同一个数据集的三个互补视图。",
                    "将各视图的分配结果转换为协同关联矩阵，并通过层次共识机制进行融合。",
                ],
            )
        ]
