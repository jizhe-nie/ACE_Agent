from __future__ import annotations

import textwrap
from pathlib import Path

from ACE_Agent.agent_core.schemas import AlgorithmRunResult, DatasetBundle
from ACE_Agent.expert_sub_agents.base import BaseExpert


class TopologyExpert(BaseExpert):
    key = "topology"
    label = "拓扑专家"

    def run(self, dataset: DatasetBundle, output_dir: Path) -> list[AlgorithmRunResult]:
        expected_clusters = int(dataset.metadata.get("expected_clusters", 3))
        results = []

        dbscan_code = textwrap.dedent(
            """
            from sklearn.cluster import DBSCAN
            from sklearn.preprocessing import StandardScaler

            scaled = StandardScaler().fit_transform(X)
            eps = estimate_dbscan_eps(scaled)
            model = DBSCAN(eps=eps, min_samples=6)
            labels = model.fit_predict(scaled)
            metrics = evaluate_labels(X, y, labels)
            plot_path = save_cluster_plot(X, labels, output_path, "拓扑专家 - DBSCAN")
            result = {
                "labels": labels.tolist(),
                "metrics": metrics,
                "plot_path": plot_path,
            }
            """
        )
        results.append(
            self._execute_code(
                dataset=dataset,
                output_dir=output_dir,
                algorithm_name="DBSCAN",
                params={"min_samples": 6, "eps": "auto_knn_quantile"},
                code=dbscan_code,
                plot_filename=f"{dataset.name}_topology_dbscan.png",
                trace=[
                    "根据 K 近邻距离估计了 DBSCAN 的 eps 参数。",
                    "使用密度连通性来保护非凸结构并隔离噪声点。",
                ],
            )
        )

        hac_code = textwrap.dedent(
            f"""
            from sklearn.cluster import AgglomerativeClustering
            from sklearn.preprocessing import StandardScaler

            scaled = StandardScaler().fit_transform(X)
            model = AgglomerativeClustering(n_clusters={expected_clusters}, linkage="single")
            labels = model.fit_predict(scaled)
            metrics = evaluate_labels(X, y, labels)
            plot_path = save_cluster_plot(X, labels, output_path, "拓扑专家 - HAC")
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
                algorithm_name="AgglomerativeSingleLink",
                params={"n_clusters": expected_clusters, "linkage": "single"},
                code=hac_code,
                plot_filename=f"{dataset.name}_topology_hac.png",
                trace=[
                    "应用了单链接层次聚类作为拓扑友好的基准方案。",
                    "强制设定预期簇数，以便与其他专家进行直接对比。",
                ],
            )
        )
        return results
