from __future__ import annotations

import textwrap
from typing import Any


class AlgorithmZoo:
    @staticmethod
    def get_all_algorithms() -> list[dict[str, Any]]:
        return [
            {
                "name": "KMeans",
                "library": "sklearn",
                "category": "centroid",
                "params": {"n_clusters": "expected_clusters", "random_state": 42},
                "doc": "Classic centroid-based clustering.",
                "max_samples": None,  # O(N) — unlimited
            },
            {
                "name": "MiniBatchKMeans",
                "library": "sklearn",
                "category": "centroid",
                "params": {"n_clusters": "expected_clusters", "random_state": 42, "batch_size": 1024},
                "doc": "Mini-batch KMeans for large-scale data.",
                "max_samples": None,  # O(N) — unlimited
            },
            {
                "name": "GaussianMixture",
                "library": "sklearn",
                "category": "centroid",
                "params": {"n_components": "expected_clusters", "random_state": 42},
                "doc": "Probabilistic model assuming Gaussian distributions.",
                "max_samples": None,  # O(N) — unlimited
            },
            {
                "name": "DBSCAN",
                "library": "sklearn",
                "category": "topology",
                "params": {"eps": 0.5, "min_samples": 5},
                "doc": "Density-based spatial clustering.",
                "max_samples": 15000,  # sparse neighbourhood, OK up to ~15K
            },
            {
                "name": "HDBSCAN",
                "library": "sklearn",
                "category": "topology",
                "params": {"min_cluster_size": 5},
                "doc": "Hierarchical DBSCAN.",
                "max_samples": 20000,  # more efficient than DBSCAN
            },
            {
                "name": "AgglomerativeClustering",
                "library": "sklearn",
                "category": "topology",
                "params": {"n_clusters": "expected_clusters"},
                "doc": "Hierarchical clustering with a bottom-up approach.",
                "max_samples": 5000,  # O(N²) distance matrix
            },
            {
                "name": "SpectralClustering",
                "library": "sklearn",
                "category": "topology",
                "params": {"n_clusters": "expected_clusters", "random_state": 42, "affinity": "nearest_neighbors"},
                "doc": "Clustering based on graph Laplacian eigenvalues.",
                "max_samples": 5000,  # O(N²) affinity matrix
            },
            {
                "name": "OPTICS",
                "library": "sklearn",
                "category": "topology",
                "params": {"min_samples": 5},
                "doc": "Ordering Points To Identify the Clustering Structure.",
                "max_samples": 10000,  # reachability graph O(N log N)
            },
            {
                "name": "Birch",
                "library": "sklearn",
                "category": "centroid",
                "params": {"n_clusters": "expected_clusters"},
                "doc": "Balanced Iterative Reducing and Clustering using Hierarchies.",
                "max_samples": None,  # O(N) — unlimited
            },
            {
                "name": "AffinityPropagation",
                "library": "sklearn",
                "category": "centroid",
                "params": {"random_state": 42},
                "doc": "Clustering based on message passing between data points.",
                "max_samples": 3000,  # O(N²) similarity matrix
            },
            {
                "name": "MeanShift",
                "library": "sklearn",
                "category": "topology",
                "params": {},
                "doc": "Centroid-based algorithm that seeks modes of density.",
                "max_samples": 5000,  # O(N²) kernel computation
            },
        ]

    @staticmethod
    def get_algorithm_code(algo_name: str, params: dict[str, Any], display_name: str = "") -> str:
        """
        生成该算法的执行代码片段。
        """
        algo_info = next((a for a in AlgorithmZoo.get_all_algorithms() if a["name"] == algo_name), None)
        if not algo_info:
            raise ValueError(f"Algorithm {algo_name} not found in Zoo.")

        lib = algo_info["library"]
        if lib == "sklearn":
            if algo_name == "GaussianMixture":
                import_stmt = "from sklearn.mixture import GaussianMixture"
                class_name = "GaussianMixture"
            else:
                import_stmt = f"from sklearn.cluster import {algo_name}"
                class_name = algo_name
        else:
            # Placeholder for other libraries
            import_stmt = f"# Library {lib} not fully implemented yet"
            class_name = algo_name

        # Prepare parameters string
        param_items = []
        for k, v in params.items():
            if v == "expected_clusters":
                param_items.append(f"{k}={v}")
            else:
                param_items.append(f"{k}={repr(v)}")
        params_str = ", ".join(param_items)

        code = textwrap.dedent(
            f"""
            {import_stmt}
            from sklearn.preprocessing import StandardScaler

            scaled = StandardScaler().fit_transform(X)
            model = {class_name}({params_str})
            # Handle fit/predict differences
            if hasattr(model, "fit_predict"):
                labels = model.fit_predict(scaled)
            else:
                labels = model.fit(scaled).predict(scaled)

            metrics = evaluate_labels(X, y, labels)
            plot_path = save_cluster_plot(X, labels, output_path, "{display_name or algo_name}")
            result = {{
                "labels": labels.tolist(),
                "metrics": metrics,
                "plot_path": plot_path,
            }}
            """
        )
        return code
