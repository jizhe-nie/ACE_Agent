"""
tools/graph_builder.py
======================
Sparse graph construction for clustering with graph-native algorithms.

All methods are static.  No LLM dependency.  Designed for N=10,000 in <5 s,
N=100,000 in <30 s via kNN sparsity.

Phase 3 (2026-05): Topology-Aware upgrade — graph structure modeling layer.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

import numpy as np
from scipy.sparse import csgraph as _csgraph
from scipy.sparse import csr_matrix, spmatrix
from scipy.sparse import issparse as _issparse
from sklearn.neighbors import NearestNeighbors, kneighbors_graph

_OUTPUTS_DIR = Path(__file__).resolve().parents[1] / "outputs"


class GraphBuilder:
    """Static methods for sparse graph construction and analysis."""

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    @staticmethod
    def build_knn_graph(
        X: np.ndarray,
        k: int | None = None,
        *,
        mutual: bool = True,
        mode: str = "distance",
    ) -> csr_matrix:
        """Build kNN graph adjacency.

        Parameters
        ----------
        X: (n_samples, n_features) feature matrix.
        k: number of neighbours.  Auto-selected as
           ``min(30, max(5, int(sqrt(n))))`` when *None*.
        mutual: when True, only keep edges where both endpoints are mutual
                neighbours (undirected, no one-way edges).
        mode: ``"distance"`` for weighted, ``"connectivity"`` for binary.
        """
        n = X.shape[0]
        if k is None:
            k = min(30, max(5, int(np.sqrt(n))))
        adj: csr_matrix = kneighbors_graph(X, k, mode=mode, include_self=False)
        if mutual:
            adj = adj.minimum(adj.T)  # element-wise min keeps only mutual edges
        return adj

    @staticmethod
    def build_radius_graph(
        X: np.ndarray,
        radius: float | None = None,
        *,
        mode: str = "distance",
    ) -> csr_matrix:
        """Build radius-neighbours graph.

        *radius* auto-computed as the mean kNN distance (k=15) when None.
        """
        n = X.shape[0]
        if radius is None:
            knn = NearestNeighbors(n_neighbors=min(15, n - 1)).fit(X)
            dists, _ = knn.kneighbors(X)
            radius = float(np.mean(dists[:, -1]))
        from sklearn.neighbors import radius_neighbors_graph
        return radius_neighbors_graph(X, radius, mode=mode, include_self=False)

    @staticmethod
    def sparsify(
        adjacency: spmatrix,
        max_degree: int = 50,
    ) -> csr_matrix:
        """Drop edges so each node keeps at most *max_degree* neighbours."""
        adj = adjacency.tocsr(copy=False)
        n = adj.shape[0]
        data: list[float] = []
        indices: list[int] = []
        indptr: list[int] = [0]
        for i in range(n):
            row = adj[i].toarray().ravel()
            nonzero = np.nonzero(row)[0]
            if len(nonzero) > max_degree:
                # Keep the *max_degree* edges with largest weight
                order = np.argsort(row[nonzero])[::-1][:max_degree]
                nonzero = nonzero[order]
            keep = nonzero.tolist()
            indptr.append(indptr[-1] + len(keep))
            for j in keep:
                indices.append(j)
                data.append(float(row[j]))
        return csr_matrix((data, indices, indptr), shape=(n, n))

    # ------------------------------------------------------------------
    # Laplacian
    # ------------------------------------------------------------------

    @staticmethod
    def compute_laplacian(
        adjacency: spmatrix,
        laplacian_type: str = "symmetric_normalized",
    ) -> csr_matrix:
        """Compute graph Laplacian from adjacency.

        *laplacian_type*: ``"unnormalized"`` | ``"symmetric_normalized"`` |
        ``"random_walk"``.
        """
        if laplacian_type == "unnormalized":
            return _csgraph.laplacian(adjacency, normed=False)
        if laplacian_type == "symmetric_normalized":
            return _csgraph.laplacian(adjacency, normed=True)
        if laplacian_type == "random_walk":
            # L_rw = I - D^{-1} A
            adj = adjacency.tocsr(copy=False)
            n = adj.shape[0]
            degrees = np.array(adj.sum(axis=1)).ravel()
            inv_D = csr_matrix(
                ([1.0 / max(d, 1e-12) for d in degrees],
                 (np.arange(n), np.arange(n))),
                shape=(n, n),
            )
            from scipy.sparse import eye
            return eye(n, format="csr") - inv_D @ adj  # type: ignore[no-any-return]
        raise ValueError(f"Unknown laplacian_type: {laplacian_type}")

    # ------------------------------------------------------------------
    # Geodesic distances
    # ------------------------------------------------------------------

    @staticmethod
    def compute_geodesic_distances(
        adjacency: spmatrix,
        indices: np.ndarray | None = None,
    ) -> np.ndarray:
        """All-pairs shortest-path distances via ``scipy.sparse.csgraph``.

        When *indices* is provided returns (len(indices) × N); otherwise
        returns full N×N (only safe for N ≤ 2000).
        """
        adj = adjacency.tocoo(copy=False)
        # Use reciprocal for distance → similarity; clip to avoid inf
        # Build undirected, nonnegative distance graph
        dist_adj = adj.astype(np.float64, copy=True)
        # Safety: replace zeros with large value on non-diagonal
        dist_adj.data[dist_adj.data < 1e-12] = 1e-12
        if indices is not None:
            return _csgraph.shortest_path(
                dist_adj, method="auto", directed=False, indices=indices,
            )
        return _csgraph.shortest_path(dist_adj, method="auto", directed=False)

    # ------------------------------------------------------------------
    # Distortion
    # ------------------------------------------------------------------

    @staticmethod
    def compute_distortion(
        X: np.ndarray,
        graph_distances: np.ndarray,
        sample_size: int = 2000,
    ) -> float:
        """Geodesic-vs-Euclidean distortion.

        ``median(|d_geo - d_euc| / max(d_euc, 1e-8))`` on a random sample.
        > 0.5 → graph-native algorithms are needed.
        """
        n = X.shape[0]
        rng = np.random.RandomState(42)
        if graph_distances.shape[0] == n and graph_distances.shape[1] == n:
            # Full N×N matrix — sample pairs
            if n <= sample_size:
                i_idx = np.arange(n)
            else:
                i_idx = rng.choice(n, sample_size, replace=False)
            ratios = []
            for i in i_idx:
                if graph_distances.shape[1] == n:
                    j_candidates = [j for j in range(n) if j != i]
                    if len(j_candidates) > 100:
                        j_candidates = list(rng.choice(j_candidates, 100, replace=False))
                    for j in j_candidates:
                        d_euc = float(np.linalg.norm(X[i] - X[j]))
                        d_geo = float(graph_distances[i, j])
                        if d_euc > 1e-8 and np.isfinite(d_geo):
                            ratios.append(abs(d_geo - d_euc) / d_euc)
            if not ratios:
                return 0.0
            return float(np.median(ratios))
        else:
            # Anchor-based — limited row count
            return 0.0

    # ------------------------------------------------------------------
    # Wall-crossing detection
    # ------------------------------------------------------------------

    @staticmethod
    def detect_wall_crossings(
        X: np.ndarray,
        adjacency: spmatrix,
        geodesic_dists: np.ndarray,
        euclidean_threshold: float = 0.1,
        geodesic_threshold: int = 5,
        sample_size: int = 1000,
    ) -> list[tuple[int, int]]:
        """Detect point pairs that are Euclidean-near but graph-far.

        These "wall-crossings" indicate where Euclidean clustering would
        incorrectly merge points across a graph boundary.
        """
        _n = X.shape[0]
        rng = np.random.RandomState(42)
        pairs: list[tuple[int, int]] = []
        # Sample candidate pairs from adjacency edges (Euclidean-near)
        adj = adjacency.tocoo(copy=False)
        if adj.nnz == 0:
            return pairs
        candidates = min(adj.nnz, sample_size * 5)
        if candidates < adj.nnz:
            idx = rng.choice(adj.nnz, candidates, replace=False)
        else:
            idx = np.arange(adj.nnz)
        for k in idx:
            i, j = int(adj.row[k]), int(adj.col[k])
            if i >= j:
                continue
            d_euc = float(np.linalg.norm(X[i] - X[j]))
            if d_euc > euclidean_threshold:
                continue
            try:
                d_geo = float(geodesic_dists[i, j])
            except IndexError:
                continue
            if d_geo > geodesic_threshold and np.isfinite(d_geo):
                pairs.append((i, j))
                if len(pairs) >= sample_size:
                    break
        return pairs

    # ------------------------------------------------------------------
    # Visualization helpers
    # ------------------------------------------------------------------

    @staticmethod
    def save_adjacency_image(
        adjacency: spmatrix,
        X: np.ndarray,
        path: str | Path,
        *,
        max_edges: int = 5000,
        labels: np.ndarray | None = None,
        title: str = "kNN Graph",
    ) -> str:
        """Save PNG with kNN graph edges overlaid on scatter of X[:, :2]."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        _OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        path = Path(path)
        if path.suffix != ".png":
            uid = uuid.uuid4().hex[:8]
            path = _OUTPUTS_DIR / f"graph_overlay_{uid}.png"

        import platform
        if platform.system() == "Windows":
            plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
        plt.rcParams["axes.unicode_minus"] = False

        X_np = np.asarray(X, dtype=float)
        if X_np.shape[1] > 2:
            from sklearn.decomposition import PCA
            X_vis = PCA(n_components=2, random_state=42).fit_transform(X_np)
        else:
            X_vis = X_np

        adj = adjacency.tocoo(copy=False)
        fig, ax = plt.subplots(figsize=(8, 7))

        # Down-sample edges for visual clarity
        n_edges = adj.nnz
        if n_edges > max_edges:
            rng = np.random.RandomState(42)
            keep = rng.choice(n_edges, max_edges, replace=False)
            rows, cols, edge_data = adj.row[keep], adj.col[keep], adj.data[keep]
        else:
            rows, cols, edge_data = adj.row, adj.col, adj.data

        if edge_data is not None and len(edge_data) > 0:
            emin, emax = float(np.min(edge_data)), float(np.max(edge_data))
            norm = plt.Normalize(emin, emax) if emax > emin else None
        else:
            norm = None

        for k in range(len(rows)):
            i, j = int(rows[k]), int(cols[k])
            if i >= j:
                continue
            alpha_val = 0.15
            lw_val = 0.4
            color = "#888888"
            if norm is not None:
                w = norm(float(edge_data[k]))
                alpha_val = 0.08 + 0.25 * w
                if w > 0.7:
                    color = "#444444"
                    lw_val = 0.6
            ax.plot(
                [X_vis[i, 0], X_vis[j, 0]],
                [X_vis[i, 1], X_vis[j, 1]],
                color=color, linewidth=lw_val, alpha=alpha_val,
            )

        # Scatter points
        if labels is not None and len(labels) == X_vis.shape[0]:
            ax.scatter(X_vis[:, 0], X_vis[:, 1], c=labels, cmap="tab10",
                       s=2, alpha=0.7, edgecolors="none")
        else:
            ax.scatter(X_vis[:, 0], X_vis[:, 1], s=2, alpha=0.5,
                       color="steelblue", edgecolors="none")
        ax.set_title(title, fontsize=12, weight="bold")
        ax.set_xlabel("x1" if X_np.shape[1] <= 2 else "PC1")
        ax.set_ylabel("x2" if X_np.shape[1] <= 2 else "PC2")
        plt.tight_layout()
        fig.savefig(str(path), dpi=130, bbox_inches="tight")
        plt.close(fig)
        return str(path)

    # ------------------------------------------------------------------
    # Graph metrics for audit / ensemble
    # ------------------------------------------------------------------

    @staticmethod
    def compute_modularity(
        adjacency: spmatrix,
        labels: np.ndarray,
    ) -> float:
        """Graph modularity-like score: (intra-cluster edges) / (total edges)."""
        adj = adjacency.tocoo(copy=False)
        if adj.nnz == 0:
            return 0.0
        intra = 0
        total = 0
        lbl_arr = np.asarray(labels, dtype=int)
        for k in range(adj.nnz):
            i, j = int(adj.row[k]), int(adj.col[k])
            w = float(adj.data[k])
            total += w
            if lbl_arr[i] == lbl_arr[j]:
                intra += w
        if total == 0:
            return 0.0
        return float(intra / total)

    @staticmethod
    def compute_conductance(
        adjacency: spmatrix,
        labels: np.ndarray,
    ) -> float:
        """Average cluster conductance: cross-cluster / total incident edges."""
        adj = adjacency.tocsr(copy=False)
        lbl_arr = np.asarray(labels, dtype=int)
        unique = sorted(set(lbl_arr.tolist()))
        if len(unique) < 2:
            return 1.0
        conductances = []
        for c in unique:
            mask = lbl_arr == c
            # Total edges incident to cluster c (including internal)
            vol = float(adj[mask].sum())
            if vol < 1e-12:
                continue
            # Cross edges: edges from cluster c to outside
            cross = float(adj[mask][:, ~mask].sum())
            conductances.append(cross / vol)
        if not conductances:
            return 0.0
        return float(np.mean(conductances))

    @staticmethod
    def compute_neighborhood_preservation(
        adjacency: spmatrix,
        labels: np.ndarray,
        sample_size: int = 2000,
    ) -> float:
        """Fraction of kNN edges that stay within the same cluster."""
        adj = adjacency.tocoo(copy=False)
        if adj.nnz == 0:
            return 0.0
        lbl_arr = np.asarray(labels, dtype=int)
        n_edges = adj.nnz
        if n_edges <= sample_size:
            idx = np.arange(n_edges)
        else:
            rng = np.random.RandomState(42)
            idx = rng.choice(n_edges, sample_size, replace=False)
        same = 0
        for k in idx:
            if lbl_arr[int(adj.row[k])] == lbl_arr[int(adj.col[k])]:
                same += 1
        return float(same / len(idx))

    # ------------------------------------------------------------------
    # Graph-cut & connectivity-preservation scoring (Phase 3.1)
    # ------------------------------------------------------------------

    @staticmethod
    def normalized_cut_loss(
        adjacency: spmatrix,
        labels: np.ndarray,
    ) -> float:
        """Normalized graph cut: sum of cross-cluster edges / cluster volume.

        Lower is better. Directly penalizes cutting through densely connected
        graph regions — the opposite of Euclidean quadrant cutting.
        """
        adj = adjacency.tocsr(copy=False)
        lbl_arr = np.asarray(labels, dtype=int)
        unique = sorted(set(lbl_arr.tolist()))
        if len(unique) < 2:
            return 1e12  # single cluster = massive penalty
        ncut_total = 0.0
        for c in unique:
            mask = lbl_arr == c
            vol = float(adj[mask].sum())
            if vol < 1e-12:
                ncut_total += 1e6
                continue
            cross = float(adj[mask][:, ~mask].sum())
            ncut_total += cross / vol
        return float(ncut_total / len(unique))

    @staticmethod
    def connectivity_preservation_score(
        X: np.ndarray,
        adjacency: spmatrix,
        labels: np.ndarray,
        geodesic_dists: np.ndarray | None = None,
        sample_size: int = 2000,
    ) -> dict[str, float]:
        """CPS: reward clusters that follow graph connectivity paths.

        For each cluster, samples point pairs:
        - Rewards pairs that are graph-near (short geodesic path) in same cluster
        - Penalizes pairs that are Euclidean-near but graph-far in same cluster
          ("wall-crossing" — points that shouldn't be together)

        Returns dict with:
        - cps: float [0,1] — higher = better connectivity preservation
        - wall_crossing_penalty: float — fraction of pairs penalized
        - geodesic_coherence: float — median geo-distance ratio intra vs inter
        """
        import numpy as np
        lbl_arr = np.asarray(labels, dtype=int)
        X_np = np.asarray(X, dtype=float)
        n = X_np.shape[0]

        # Build geodesic distances if not provided
        if geodesic_dists is None:
            try:
                from scipy.sparse import csgraph as _csgraph
                geodesic_dists = _csgraph.shortest_path(adjacency, method="auto", directed=False)
            except Exception:
                return {"cps": 0.0, "wall_crossing_penalty": 1.0, "geodesic_coherence": 0.0}

        rng = np.random.RandomState(42)
        n_sample = min(sample_size, n)

        # Sample point pairs
        intra_geo_ratios = []
        wall_crossings = 0
        total_intra = 0

        for _ in range(n_sample):
            i = int(rng.randint(0, n))
            j = int(rng.randint(0, n))
            if i == j:
                continue
            d_euc = float(np.linalg.norm(X_np[i] - X_np[j]))
            try:
                d_geo = float(geodesic_dists[i, j])
            except (IndexError, ValueError):
                continue
            if not np.isfinite(d_geo) or d_geo < 1e-12:
                continue

            if lbl_arr[i] == lbl_arr[j]:
                total_intra += 1
                if d_euc > 1e-8:
                    ratio = d_geo / max(d_euc, 1e-8)
                    intra_geo_ratios.append(ratio)
                # Wall-crossing: Euclidean-near but graph-far, yet same cluster
                if d_euc < 0.15 and d_geo > 3.0:
                    wall_crossings += 1

        if not intra_geo_ratios:
            return {"cps": 0.5, "wall_crossing_penalty": 0.0, "geodesic_coherence": 0.0}

        geo_coherence = float(np.median(intra_geo_ratios))
        wc_penalty = wall_crossings / max(total_intra, 1)
        # CPS: high when geo-coherence is low (short graph paths in clusters)
        # and wall-crossing penalty is low
        cps = float(np.exp(-0.3 * geo_coherence) * (1.0 - 0.8 * wc_penalty))
        cps = max(0.0, min(1.0, cps))

        return {
            "cps": cps,
            "wall_crossing_penalty": wc_penalty,
            "geodesic_coherence": geo_coherence,
        }

    @staticmethod
    def detect_axis_aligned_partition(
        X: np.ndarray,
        labels: np.ndarray,
        threshold_ratio: float = 0.7,
    ) -> dict[str, Any]:
        """Detect if clustering is mainly axis-aligned quadrant cuts.

        Checks whether cluster boundaries can be explained by simple x/y
        threshold rules. If ratio > *threshold_ratio*, the partition is
        likely "Euclidean quadrant cutting" rather than graph-respecting.

        Returns dict with:
        - is_axis_aligned: bool
        - x_axis_ratio: float — fraction of cluster variance explained by x-axis split
        - y_axis_ratio: float — same for y-axis
        - axis_score: float [0,1] — 1.0 means fully axis-aligned (bad)
        """
        import numpy as np
        X_np = np.asarray(X, dtype=float)
        lbl_arr = np.asarray(labels, dtype=int)
        n_features = X_np.shape[1]
        if n_features < 2:
            return {"is_axis_aligned": False, "x_axis_ratio": 0.0,
                    "y_axis_ratio": 0.0, "axis_score": 0.0}

        unique = sorted(set(lbl_arr.tolist()))
        if len(unique) < 2:
            return {"is_axis_aligned": False, "x_axis_ratio": 0.0,
                    "y_axis_ratio": 0.0, "axis_score": 0.0}

        # For each cluster, check if its x-range or y-range is
        # primarily bounded by single-axis thresholds
        axis_scores = []
        _x_range_explained = 0.0
        _y_range_explained = 0.0

        for c in unique:
            mask = lbl_arr == c
            Xc = X_np[mask]
            Xc_x = Xc[:, 0]
            Xc_y = Xc[:, 1] if n_features > 1 else np.zeros_like(Xc_x)

            # Check if cluster boundary is a clean axis cut:
            # Points are all on one side of some x/y threshold
            x_min, x_max = float(Xc_x.min()), float(Xc_x.max())
            y_min, y_max = float(Xc_y.min()), float(Xc_y.max())

            # Compute non-cluster points in the bounding box
            non_mask = ~mask
            X_non = X_np[non_mask]
            in_x_range = (X_non[:, 0] >= x_min) & (X_non[:, 0] <= x_max)
            in_y_range = (X_non[:, 1] >= y_min) & (X_non[:, 1] <= y_max)
            in_box = in_x_range & in_y_range

            # If the bounding box is mostly empty of other cluster points,
            # this is likely axis-aligned
            box_total = float(X_non.shape[0])
            if box_total > 0:
                empty_ratio = 1.0 - float(in_box.sum()) / box_total
            else:
                empty_ratio = 1.0
            axis_scores.append(empty_ratio)

        # Global axis check: does a simple x/y split explain clusters?
        # Fit a decision stump on x and y separately
        from sklearn.tree import DecisionTreeClassifier
        # X-axis only
        dt_x = DecisionTreeClassifier(max_depth=1, random_state=42)
        dt_x.fit(X_np[:, :1], lbl_arr)
        x_pred = dt_x.predict(X_np[:, :1])
        x_accuracy = float(np.mean(x_pred == lbl_arr))

        # Y-axis only
        y_accuracy = 0.0
        if n_features > 1:
            dt_y = DecisionTreeClassifier(max_depth=1, random_state=42)
            dt_y.fit(X_np[:, 1:2], lbl_arr)
            y_pred = dt_y.predict(X_np[:, 1:2])
            y_accuracy = float(np.mean(y_pred == lbl_arr))

        axis_score = float(np.mean(axis_scores)) if axis_scores else 0.0
        is_axis_aligned = axis_score > threshold_ratio or max(x_accuracy, y_accuracy) > 0.75

        return {
            "is_axis_aligned": is_axis_aligned,
            "x_axis_accuracy": x_accuracy,
            "y_axis_accuracy": y_accuracy,
            "axis_score": axis_score,
        }

    @staticmethod
    def topology_failure_check(
        X: np.ndarray,
        adjacency: spmatrix,
        labels: np.ndarray,
        geodesic_dists: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Comprehensive topology failure detection.

        Returns dict with:
        - topology_failure: bool — overall failure flag
        - axis_aligned: bool — boundary is axis-aligned
        - high_conductance: bool — graph structure not respected
        - wall_crossings_detected: bool — Euclidean-near pairs in same cluster
        - cps: float — connectivity preservation score
        - axis_score: float — axis-aligned score
        - conductance: float — graph conductance
        - recommendation: str — human-readable diagnosis
        """
        axis_result = GraphBuilder.detect_axis_aligned_partition(X, labels)
        conductance = GraphBuilder.compute_conductance(adjacency, labels)
        cps_result = GraphBuilder.connectivity_preservation_score(
            X, adjacency, labels, geodesic_dists,
        )
        modularity = GraphBuilder.compute_modularity(adjacency, labels)
        n_preservation = GraphBuilder.compute_neighborhood_preservation(adjacency, labels)

        axis_aligned = axis_result.get("is_axis_aligned", False)
        high_conductance = conductance > 0.6
        wall_crossings = cps_result.get("wall_crossing_penalty", 0) > 0.05
        low_cps = cps_result.get("cps", 0.5) < 0.4
        low_modularity = modularity < 0.4

        # Topology failure: any two red flags
        failures = [axis_aligned, high_conductance, wall_crossings, low_cps, low_modularity]
        n_failures = sum(1 for f in failures if f)
        topology_failure = n_failures >= 2

        recommendations = []
        if axis_aligned:
            recommendations.append(
                f"聚类边界近似坐标轴切割(axis_score={axis_result['axis_score']:.2f})。"
                "应使用 graph-based spectral clustering 替代欧氏聚类。"
            )
        if high_conductance:
            recommendations.append(
                f"图传导率过高(conductance={conductance:.2f})，"
                "聚类切断了大量图连通边。应最小化 normalized cut。"
            )
        if wall_crossings:
            recommendations.append(
                f"检测到穿墙近邻(cps={cps_result['cps']:.3f})，"
                "欧氏近但图远的点对被错误分到同簇。"
            )

        return {
            "topology_failure": topology_failure,
            "n_failures": n_failures,
            "axis_aligned": axis_aligned,
            "high_conductance": high_conductance,
            "wall_crossings_detected": wall_crossings,
            "cps": cps_result["cps"],
            "axis_score": axis_result["axis_score"],
            "conductance": conductance,
            "modularity": modularity,
            "neighborhood_preservation": n_preservation,
            "recommendation": "; ".join(recommendations) if recommendations else "拓扑结构良好。",
        }

    @staticmethod
    def spectral_graph_cut_clustering(
        adjacency: spmatrix,
        n_clusters: int,
        *,
        random_state: int = 42,
    ) -> np.ndarray:
        """Proper normalized-cut spectral clustering on the graph.

        Uses the unnormalized Laplacian eigenvectors (Fiedler vector for k=2)
        followed by KMeans for >2 clusters. This is the standard min-Ncut
        relaxation, NOT Euclidean-coordinate clustering.

        Returns labels array.
        """
        import numpy as np
        from scipy.sparse.linalg import eigsh

        n = adjacency.shape[0]
        k = min(n_clusters, n - 1)

        # Unnormalized Laplacian for min-cut relaxation
        L = GraphBuilder.compute_laplacian(adjacency, "unnormalized")

        # Compute k smallest eigenvectors (excluding the trivial zero eigenvector)
        try:
            eigenvalues, eigenvectors = eigsh(L, k=k + 1, which="SM", tol=1e-6,
                                              maxiter=200, return_eigenvectors=True)
            # Sort by eigenvalue ascending
            idx = np.argsort(eigenvalues)
            eigenvectors = eigenvectors[:, idx]
            # Skip the first (zero eigenvalue → constant vector)
            embedding = eigenvectors[:, 1:k + 1].copy()
        except Exception:
            # Fallback: use normalized Laplacian which is better conditioned
            L_norm = GraphBuilder.compute_laplacian(adjacency, "symmetric_normalized")
            eigenvalues, eigenvectors = eigsh(L_norm, k=k + 1, which="SM", tol=1e-6,
                                              maxiter=200, return_eigenvectors=True)
            idx = np.argsort(eigenvalues)
            eigenvectors = eigenvectors[:, idx]
            embedding = eigenvectors[:, 1:k + 1].copy()

        # Normalize rows of embedding to unit length
        norms = np.linalg.norm(embedding, axis=1, keepdims=True)
        norms[norms < 1e-10] = 1e-10
        embedding = embedding / norms

        # KMeans on the spectral embedding
        from sklearn.cluster import KMeans
        labels = KMeans(n_clusters=k, random_state=random_state, n_init=10).fit_predict(embedding)
        return labels.astype(int)

    @staticmethod
    def adaptive_modularity_k(
        adjacency: spmatrix,
        max_k: int = 12,
        min_k: int = 2,
    ) -> int:
        """Find optimal k via modularity maximization on graph structure.

        Runs spectral graph-cut clustering for k=min_k..max_k, selects the
        k that minimizes conductance + maximizes modularity.
        """
        best_k = min_k
        best_score = -1e12
        for k in range(min_k, min(max_k + 1, adjacency.shape[0])):
            try:
                labels = GraphBuilder.spectral_graph_cut_clustering(adjacency, k)
                mod = GraphBuilder.compute_modularity(adjacency, labels)
                cond = GraphBuilder.compute_conductance(adjacency, labels)
                # Score: high modularity + low conductance
                score = mod - 0.5 * cond
                if score > best_score:
                    best_score = score
                    best_k = k
            except Exception:
                continue
        return best_k

    # =====================================================================
    # Phase 3.2: Random Walk / Diffusion Dynamics
    # =====================================================================

    @staticmethod
    def compute_transition_matrix(
        adjacency: spmatrix,
    ) -> csr_matrix:
        """Random walk transition matrix P = D^{-1}A.

        P_ij = probability of transitioning from i to j in one step.
        Row-stochastic (each row sums to 1).
        """
        from scipy.sparse import csr_matrix, diags

        if not _issparse(adjacency):
            adjacency = csr_matrix(adjacency, dtype=float)
        else:
            adjacency = adjacency.tocsr().astype(float)

        degrees = np.asarray(adjacency.sum(axis=1)).ravel()
        degrees[degrees == 0] = 1.0
        D_inv = diags(1.0 / degrees)
        return D_inv @ adjacency

    @staticmethod
    def compute_diffusion_matrix(
        adjacency: spmatrix,
        t: int = 8,
    ) -> np.ndarray | csr_matrix:
        """Compute t-step diffusion P^t from the transition matrix.

        Returns dense for n ≤ 5000; sparse otherwise.
        """
        import numpy as np
        from scipy.sparse import issparse as _issparse

        P = GraphBuilder.compute_transition_matrix(adjacency)
        n = P.shape[0]

        # Binary exponentiation for P^t
        P_t = P.copy()
        remaining = t - 1
        base = P.copy()
        while remaining > 0:
            if remaining & 1:
                P_t = P_t @ base
            base = base @ base
            remaining >>= 1

        if n <= 5000:
            return P_t.toarray() if _issparse(P_t) else np.asarray(P_t)
        return P_t

    @staticmethod
    def compute_diffusion_distances(
        adjacency: spmatrix,
        t: int = 8,
        n_components: int = 16,
        random_state: int = 42,
    ) -> np.ndarray:
        """Diffusion map embedding via spectral decomposition of P^t.

        Returns (n, n_components) embedding where Euclidean distances
        approximate diffusion distances on the graph.
        """
        import numpy as np
        from scipy.sparse.linalg import eigsh

        P = GraphBuilder.compute_transition_matrix(adjacency)
        n = P.shape[0]

        k = min(n_components + 1, n - 1)
        try:
            vals, vecs = eigsh(P.T @ P, k=k, which='LM')
        except Exception:
            try:
                vals, vecs = eigsh(P.T @ P, k=min(6, n - 1), which='LM')
            except Exception:
                from sklearn.decomposition import TruncatedSVD
                svd = TruncatedSVD(n_components=k, random_state=random_state)
                return svd.fit_transform(P)

        # Sort descending by eigenvalue
        idx = np.argsort(vals)[::-1]
        vals = vals[idx]
        vecs = vecs[:, idx]

        # Skip the dominant eigenvector (all ones for stochastic matrix)
        embedding = vecs[:, 1:n_components + 1]
        # Scale by eigenvalues for diffusion distance
        vals = np.maximum(vals[1:n_components + 1], 1e-10)
        embedding = embedding * vals[None, :] ** t

        return np.ascontiguousarray(embedding)

    @staticmethod
    def markov_stability(
        adjacency: spmatrix,
        labels: np.ndarray,
        t: float = 1.0,
    ) -> float:
        """Markov stability — community quality at Markov time t.

        Measures the probability that a random walker stays within the
        same community after t steps, minus the expected probability
        at stationarity.

        R(t) = sum_C [ (P^t_CC) - (pi_C^2) ]

        where P^t_CC is the probability of starting and ending in
        community C, and pi_C is the stationary probability of C.

        Higher values = better community structure at time scale t.
        """
        import numpy as np

        labels = np.asarray(labels, dtype=int).ravel()
        n = adjacency.shape[0]

        # Stationary distribution: degree / total_degree
        degrees = np.asarray(adjacency.sum(axis=1)).ravel().astype(float)
        total_degree = degrees.sum()
        if total_degree == 0:
            return 0.0
        pi = degrees / total_degree

        # Build transition matrix
        P = GraphBuilder.compute_transition_matrix(adjacency)

        # Apply t steps (approximate for non-integer t via eigenvalue scaling)
        if abs(t - round(t)) < 1e-10:
            # Integer steps: use binary exponentiation
            t_int = int(round(t))
            P_t = P.copy()
            remaining = t_int - 1
            base = P.copy()
            while remaining > 0:
                if remaining & 1:
                    P_t = P_t @ base
                base = base @ base
                remaining >>= 1
        else:
            # Non-integer: approximate via spectral decomposition
            from scipy.sparse.linalg import eigsh
            try:
                vals, vecs = eigsh(P.T @ P, k=min(30, n - 1), which='LM')
                idx = np.argsort(vals)[::-1]
                vals = vals[idx]
                vecs = vecs[:, idx]
                vals_t = np.power(np.maximum(vals, 1e-10), t)
                P_t = (vecs * vals_t) @ vecs.T
            except Exception:
                P_t = P  # fallback to one-step

        stability = 0.0
        for c in np.unique(labels):
            mask = labels == c
            pi_c = pi[mask].sum()

            # P^t within community C
            if _issparse(P_t):
                p_cc = sum(P_t[i, j] for i in mask.nonzero()[0] for j in mask.nonzero()[0]
                          if P_t[i, j] > 0)
            else:
                rows = np.where(mask)[0]
                cols = np.where(mask)[0]
                sub = P_t[np.ix_(rows, cols)] if len(rows) * len(cols) <= 1e7 else np.sum(
                    P_t[rows, :][:, cols])
                p_cc = np.sum(sub)

            stability += p_cc - (pi_c * pi_c)

        return float(stability)

    # =====================================================================
    # Phase 3.2: Boundary Quality Audit
    # =====================================================================

    @staticmethod
    def compute_boundary_quality(
        X: np.ndarray,
        adjacency: spmatrix,
        labels: np.ndarray,
        *,
        sample_size: int = 2000,
    ) -> dict[str, Any]:
        """Audit clustering boundary quality from a graph perspective.

        A good boundary:
          - Follows graph bottlenecks (low edge density across)
          - Aligns with narrow graph passages
          - Has low inter-community edge flow

        A bad boundary:
          - Is a straight / axis-aligned cut across the space
          - Has nothing to do with graph edge flow
          - Can be explained by an x/y threshold

        Returns
        -------
        dict with:
          - boundary_quality_score : float  (0–1, higher = better)
          - inter_community_flow : float  (normalised)
          - bottleneck_alignment : float  (0–1)
          - straight_cut_penalty : float  (0–1, higher = worse)
          - threshold_explainable : bool
          - assessment : str  ('good' / 'fair' / 'poor')
        """
        import numpy as np
        from scipy.sparse import csr_matrix
        from scipy.sparse import issparse as _issparse

        X = np.asarray(X, dtype=float)
        labels = np.asarray(labels, dtype=int).ravel()
        n = X.shape[0]

        if not _issparse(adjacency):
            A = csr_matrix(adjacency)
        else:
            A = adjacency.tocsr()

        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels)
        if n_clusters < 2:
            return {
                "boundary_quality_score": 1.0,
                "inter_community_flow": 0.0,
                "bottleneck_alignment": 1.0,
                "straight_cut_penalty": 0.0,
                "threshold_explainable": False,
                "assessment": "good",
                "note": "单簇聚类，无边界可评估。",
            }

        # ---- 1. Inter-community edge flow (normalised) ----
        total_edges = A.sum()
        inter_flow = 0.0
        for _i, ci in enumerate(unique_labels):
            mask_i = labels == ci
            for cj in unique_labels:
                if cj <= ci:
                    continue
                mask_j = labels == cj
                cross = A[mask_i, :][:, mask_j].sum()
                inter_flow += cross
        inter_flow_normalised = inter_flow / max(total_edges, 1.0)

        # ---- 2. Bottleneck alignment ----
        # For each boundary point (close to another cluster), check if
        # its graph degree is low → bottleneck

        rng = np.random.RandomState(42)
        sample_n = min(sample_size, n)
        sample_idx = rng.choice(n, sample_n, replace=False)

        bottleneck_scores = []
        for idx in sample_idx:
            ci = labels[idx]
            other_mask = labels != ci
            if not other_mask.any():
                continue
            # Find the closest point in another cluster
            X_other = X[other_mask]
            dist_to_boundary = np.min(np.sum((X_other - X[idx]) ** 2, axis=1))
            # Graph degree of this point
            degree = A[idx].sum()
            max_degree = max(A.sum(axis=0)).max() or 1
            norm_degree = degree / max_degree
            # Bottleneck: low degree + close to boundary = good boundary alignment
            bottleneck_scores.append(1.0 - norm_degree if dist_to_boundary < 0.1 else 0.0)

        bottleneck_alignment = np.mean(bottleneck_scores) if bottleneck_scores else 0.5

        # ---- 3. Straight-cut penalty (axis-aligned detection) ----
        axis_result = GraphBuilder.detect_axis_aligned_partition(X, labels)
        straight_cut_penalty = axis_result.get("axis_score", 0.0)

        # ---- 4. Threshold explainability ----
        # Can clustering be reproduced by x < threshold or y < threshold?
        threshold_explainable = axis_result.get("is_axis_aligned", False)

        # ---- Composite score ----
        # Base quality: low inter-flow + high bottleneck alignment
        flow_score = 1.0 - inter_flow_normalised
        boundary_quality = (
            0.35 * flow_score
            + 0.35 * bottleneck_alignment
            + 0.30 * (1.0 - straight_cut_penalty)
        )
        boundary_quality = float(np.clip(boundary_quality, 0.0, 1.0))

        if boundary_quality > 0.7:
            assessment = "good"
        elif boundary_quality > 0.4:
            assessment = "fair"
        else:
            assessment = "poor"

        return {
            "boundary_quality_score": boundary_quality,
            "inter_community_flow": float(inter_flow_normalised),
            "bottleneck_alignment": float(bottleneck_alignment),
            "straight_cut_penalty": float(straight_cut_penalty),
            "threshold_explainable": threshold_explainable,
            "assessment": assessment,
        }

    @staticmethod
    def compute_edge_cut_objective(
        adjacency: spmatrix,
        labels: np.ndarray,
    ) -> dict[str, float]:
        """Three edge-cut objectives used as primary optimisation targets
        for graph-connected data (replacing centroid separation).

        - normalized_cut: sum( cut(C, V\\C) / vol(C) ) → lower is better
        - modularity: standard Newman-Girvan modularity → higher is better
        - conductance: mean( cut(C, V\\C) / min(vol(C), vol(V\\C)) ) → lower is better

        These THREE objectives are the primary scoring dimension for
        graph community discovery.  Silhouette / centroid separation is
        NOT used for graph-connected data.
        """
        n_cut = GraphBuilder.normalized_cut_loss(adjacency, labels)
        mod = GraphBuilder.compute_modularity(adjacency, labels)
        cond = GraphBuilder.compute_conductance(adjacency, labels)
        cps = GraphBuilder.connectivity_preservation_score(
            np.zeros((adjacency.shape[0], 1)), adjacency, labels,
        )

        return {
            "normalized_cut": float(n_cut),
            "modularity": float(mod),
            "conductance": float(cond),
            "cps": float(cps.get("cps", 0.0)),
            # Composite: this is the PRIMARY score for graph-connected data
            "edge_cut_composite": float(
                0.30 * (1.0 / max(n_cut, 0.001))  # low n-cut → high score
                + 0.40 * mod  # high modularity → high score
                + 0.20 * (1.0 - cond)  # low conductance → high score
                + 0.10 * cps.get("cps", 0.0)
            ),
        }

    # =====================================================================
    # Phase 4: Wall-Aware Graph Construction
    # =====================================================================
    # The quality of graph community detection is bounded by graph quality.
    # A single wall-crossing edge (Euclidean-near but geodesic-far) can
    # merge two communities that should remain separate, defeating any
    # modularity-based algorithm.
    #
    # The methods below form a pipeline:
    #   1. Build initial mutual kNN
    #   2. Compute shared-neighbor similarity → Jaccard edge weights
    #   3. Apply adaptive local scaling
    #   4. Detect & prune shortcut/wall-crossing edges
    #   5. Sparsify while preserving bottlenecks
    #   6. Compute graph quality metrics
    # =====================================================================

    @staticmethod
    def compute_shared_neighbor_similarity(
        adjacency: spmatrix,
        *,
        k: int | None = None,
    ) -> spmatrix:
        """Compute Jaccard similarity for each edge based on shared neighbors.

        J(i,j) = |N(i) ∩ N(j)| / |N(i) ∪ N(j)|

        High Jaccard → points share many neighbors → genuine connection.
        Low Jaccard → points likely in different graph regions → wall-crossing.

        Returns sparse matrix with Jaccard similarities at edge positions.
        Only edges present in *adjacency* are scored (non-edges are 0).
        """
        import numpy as np
        from scipy.sparse import csr_matrix, lil_matrix

        n = adjacency.shape[0]
        adj = adjacency.tocsr() if _issparse(adjacency) else csr_matrix(adjacency)
        # Binarise
        adj_bin = adj.copy()
        adj_bin.data = np.ones_like(adj_bin.data)
        adj_bin.eliminate_zeros()

        _degrees = np.asarray(adj_bin.sum(axis=1)).ravel().astype(int)

        # For each node, get sorted neighbor list
        neighbor_sets: list[set] = [set() for _ in range(n)]
        for i in range(n):
            row = adj_bin[i]
            neighbor_sets[i] = set(row.indices.tolist())

        result = lil_matrix((n, n), dtype=float)
        for i in range(n):
            row = adj[i]
            ni = neighbor_sets[i]
            if not ni:
                continue
            for j in row.indices:
                if i >= j:
                    continue
                nj = neighbor_sets[j]
                if not nj:
                    continue
                inter = len(ni & nj)
                union = len(ni | nj)
                if union > 0:
                    jaccard = inter / union
                    result[i, j] = jaccard
                    result[j, i] = jaccard

        return result.tocsr()

    @staticmethod
    def build_shared_neighbor_graph(
        X: np.ndarray,
        k: int | None = None,
        *,
        mutual: bool = True,
        jaccard_threshold: float = 0.1,
        random_state: int = 42,
    ) -> spmatrix:
        """Build a kNN graph weighted by shared-neighbor Jaccard similarity.

        Edges with Jaccard < *jaccard_threshold* are pruned entirely.
        Remaining edges receive weight = Jaccard similarity.

        This naturally eliminates wall-crossing edges: two points separated
        by a maze wall share very few kNN neighbors.
        """
        import numpy as np

        n = X.shape[0]
        if k is None:
            k = min(30, max(5, int(np.sqrt(n))))

        # Build initial mutual kNN
        adj = GraphBuilder.build_knn_graph(X, k=k, mutual=mutual, mode="connectivity")

        # Compute Jaccard weights
        jaccard = GraphBuilder.compute_shared_neighbor_similarity(adj, k=k)

        # Combine: adjacency pattern × Jaccard weight
        adj_weighted = adj.multiply(jaccard)

        # Prune low-Jaccard edges
        adj_weighted.data[adj_weighted.data < jaccard_threshold] = 0
        adj_weighted.eliminate_zeros()

        return adj_weighted.tocsr()

    @staticmethod
    def compute_adaptive_bandwidth(
        X: np.ndarray,
        k: int | None = None,
    ) -> np.ndarray:
        """Compute local scaling factor σ_i for each point.

        σ_i = distance to the k-th nearest neighbor.

        This is the self-tuning approach (Zelnik-Manor & Perona, 2004):
        edge weight = exp(-d(i,j)² / (σ_i * σ_j)).
        """
        from sklearn.neighbors import NearestNeighbors

        n = X.shape[0]
        if k is None:
            k = min(15, max(5, n // 20))

        nn = NearestNeighbors(n_neighbors=min(k + 1, n), metric="euclidean")
        nn.fit(X)
        dists, _ = nn.kneighbors(X)
        # dists[:, -1] is distance to k-th neighbor (excluding self for k+1)
        sigma = dists[:, -1].copy()
        sigma[sigma < 1e-8] = 1e-8  # avoid division by zero
        return sigma

    @staticmethod
    def build_adaptive_scaling_graph(
        X: np.ndarray,
        k: int | None = None,
        *,
        mutual: bool = True,
    ) -> spmatrix:
        """Build kNN graph with adaptive local scaling edge weights.

        Weight(i,j) = exp(-d(i,j)² / (σ_i * σ_j))

        where σ_i = distance to k-th neighbor of i.
        Points in dense regions get smaller σ → edges are more local.
        Points in sparse regions get larger σ → edges reach further to
        maintain connectivity.
        """
        import numpy as np
        from sklearn.neighbors import kneighbors_graph

        n = X.shape[0]
        if k is None:
            k = min(30, max(5, int(np.sqrt(n))))

        sigma = GraphBuilder.compute_adaptive_bandwidth(X, k)

        # Get distance-weighted kNN
        adj = kneighbors_graph(X, min(k, n - 1), mode="distance", include_self=False)
        if mutual:
            adj = adj.minimum(adj.T)

        adj = adj.tocsr()
        # Apply local scaling: w_ij = exp(-d_ij² / (σ_i * σ_j))
        for i in range(n):
            row = adj[i]
            for j_idx in range(len(row.indices)):
                j = row.indices[j_idx]
                d = row.data[j_idx]
                w = np.exp(-d * d / (sigma[i] * sigma[j]))
                adj[i, j] = w

        adj.eliminate_zeros()
        return adj.tocsr()

    @staticmethod
    def compute_adaptive_k(
        X: np.ndarray,
        base_k: int = 15,
        *,
        min_k: int = 5,
        max_k: int = 50,
    ) -> np.ndarray:
        """Compute per-point adaptive neighborhood size.

        In dense regions: smaller k (preserve local detail).
        In sparse regions: larger k (maintain connectivity).

        The local density ρ_i is estimated from the mean distance to the
        base_k nearest neighbors. Points with smaller mean distance (dense)
        get reduced k; points with larger mean distance (sparse) get
        increased k.
        """
        import numpy as np
        from sklearn.neighbors import NearestNeighbors

        n = X.shape[0]
        k_use = min(base_k + 1, n)
        nn = NearestNeighbors(n_neighbors=k_use, metric="euclidean")
        nn.fit(X)
        dists, _ = nn.kneighbors(X)
        # Mean distance to base_k neighbors (excluding self)
        mean_dists = dists[:, 1:].mean(axis=1)

        # Normalise to [0, 1]
        d_min, d_max = mean_dists.min(), mean_dists.max()
        if d_max - d_min < 1e-10:
            return np.full(n, base_k, dtype=int)
        norm = (mean_dists - d_min) / (d_max - d_min)

        # Map: low density (norm≈1) → max_k, high density (norm≈0) → min_k
        # Invert: sparse regions need larger k
        adaptive_k = min_k + norm * (max_k - min_k)
        return adaptive_k.astype(int)

    @staticmethod
    def build_adaptive_neighborhood_graph(
        X: np.ndarray,
        base_k: int = 15,
        *,
        mutual: bool = True,
    ) -> spmatrix:
        """Build variable-k kNN graph with per-point adaptive neighborhood size.

        Uses ``compute_adaptive_k`` for per-point k, then builds mutual kNN
        with variable-sized neighborhoods.
        """
        from scipy.sparse import lil_matrix
        from sklearn.neighbors import NearestNeighbors

        n = X.shape[0]
        per_point_k = GraphBuilder.compute_adaptive_k(X, base_k=base_k)
        max_k = per_point_k.max()

        # Get max_k neighbors for all points, then truncate per-point
        nn = NearestNeighbors(n_neighbors=min(max_k + 1, n), metric="euclidean")
        nn.fit(X)
        dists, indices = nn.kneighbors(X)

        # Build adjacency with variable k
        adj = lil_matrix((n, n), dtype=float)
        for i in range(n):
            ki = min(per_point_k[i] + 1, n)
            for j_idx in range(1, ki):  # skip self (index 0)
                j = indices[i, j_idx]
                d = dists[i, j_idx]
                adj[i, j] = 1.0 / max(d, 1e-8)  # inverse distance weight

        adj = adj.tocsr()
        if mutual:
            adj = adj.minimum(adj.T)

        return adj

    @staticmethod
    def detect_shortcut_edges(
        X: np.ndarray,
        adjacency: spmatrix,
        *,
        euclidean_threshold: float = 0.1,
        geodesic_ratio: float = 5.0,
        jaccard_threshold: float = 0.05,
        sample_size: int = 1000,
    ) -> dict[str, Any]:
        """Detect shortcut (wall-crossing) edges in the graph.

        An edge (i,j) is a shortcut if it satisfies ANY of:
          1. Euclidean distance < *euclidean_threshold* BUT geodesic
             distance (graph shortest path around the edge) > *geodesic_ratio*
             times the Euclidean distance.
          2. Shared-neighbor Jaccard similarity < *jaccard_threshold*.

        These edges connect points that are close in Euclidean space but
        belong to disconnected regions in the graph structure — they "cut
        through walls" in maze-like data.

        Returns dict with:
          - shortcut_edge_indices: list of (i, j) tuples
          - shortcut_ratio: float — fraction of edges flagged
          - jaccard_shortcuts: int — detected via Jaccard
          - geodesic_shortcuts: int — detected via geodesic
        """
        import numpy as np
        from scipy.sparse import csr_matrix

        n = X.shape[0]
        adj = adjacency.tocsr() if _issparse(adjacency) else csr_matrix(adjacency)
        adj_bin = adj.copy()
        adj_bin.data = np.ones_like(adj_bin.data, dtype=float)

        total_edges = adj.nnz
        if total_edges == 0:
            return {"shortcut_edge_indices": [], "shortcut_ratio": 0.0,
                    "jaccard_shortcuts": 0, "geodesic_shortcuts": 0}

        # ---- Method 1: Jaccard similarity ----
        jaccard = GraphBuilder.compute_shared_neighbor_similarity(adj_bin)
        jaccard_shortcuts = set()
        jaccard_count = 0
        for i in range(min(n, 5000)):
            row = adj_bin[i]
            for j in row.indices:
                if i >= j:
                    continue
                jac = jaccard[i, j]
                if jac < jaccard_threshold:
                    jaccard_shortcuts.add((i, j))
                    jaccard_count += 1

        # ---- Method 2: Euclidean-vs-geodesic ratio ----
        geodesic_shortcuts = set()
        geo_count = 0

        # Sample edges for geodesic check (expensive)
        rng = np.random.RandomState(42)
        edge_list = []
        for i in range(n):
            row = adj_bin[i]
            for j in row.indices:
                if i < j:
                    edge_list.append((i, j))
        n_edges = len(edge_list)
        sample_n = min(sample_size, n_edges)
        if sample_n > 0:
            sampled = rng.choice(n_edges, sample_n, replace=False) if n_edges > sample_n else range(n_edges)
            sample_set = set()
            for idx in sampled:
                i, j = edge_list[idx]
                d_euc = float(np.linalg.norm(X[i] - X[j]))
                if d_euc > euclidean_threshold:
                    continue
                sample_set.add((i, j))

            if sample_set:
                # Compute geodesic distances on graph WITH edges removed
                # Efficient: for each candidate edge, compute shortest path
                # on the graph MINUS that edge
                from scipy.sparse.csgraph import dijkstra as csg_dijkstra
                for (i, j) in list(sample_set)[:100]:  # limit geodesic checks
                    # Temporarily remove edge and compute shortest path
                    adj_removed = adj_bin.copy().tolil()
                    adj_removed[i, j] = 0
                    adj_removed[j, i] = 0
                    adj_removed = adj_removed.tocsr()
                    try:
                        d_geo = csg_dijkstra(adj_removed, indices=[i], limit=1000)[0, j]
                        if np.isfinite(d_geo) and d_geo > geodesic_ratio * max(d_euc, 1e-8):
                            geodesic_shortcuts.add((i, j))
                            geo_count += 1
                    except Exception:
                        # Disconnected after removal → edge was a bridge (not a shortcut)
                        pass

        # ---- Combine ----
        all_shortcuts = jaccard_shortcuts | geodesic_shortcuts
        return {
            "shortcut_edge_indices": list(all_shortcuts),
            "shortcut_ratio": len(all_shortcuts) / max(total_edges, 1),
            "jaccard_shortcuts": jaccard_count,
            "geodesic_shortcuts": geo_count,
            "total_edges": total_edges,
        }

    @staticmethod
    def prune_shortcut_edges(
        X: np.ndarray,
        adjacency: spmatrix,
        *,
        jaccard_threshold: float = 0.05,
        remove_jaccard_shortcuts: bool = True,
        remove_geodesic_shortcuts: bool = False,
    ) -> tuple[spmatrix, dict[str, Any]]:
        """Remove detected shortcut edges from the graph.

        Returns (pruned_adjacency, pruning_report).
        """
        from scipy.sparse import lil_matrix

        shortcut_info = GraphBuilder.detect_shortcut_edges(
            X, adjacency, jaccard_threshold=jaccard_threshold,
        )

        _n = adjacency.shape[0]
        adj = adjacency.tolil() if _issparse(adjacency) else lil_matrix(adjacency)

        removed = 0
        for i, j in shortcut_info["shortcut_edge_indices"]:
            if adj[i, j] > 0:
                adj[i, j] = 0
                adj[j, i] = 0
                removed += 1

        adj = adj.tocsr()
        adj.eliminate_zeros()

        shortcut_info["edges_removed"] = removed
        shortcut_info["edges_remaining"] = adj.nnz

        return adj, shortcut_info

    @staticmethod
    def sparsify_by_edge_betweenness(
        adjacency: spmatrix,
        *,
        keep_fraction: float = 0.7,
        sample_landmarks: int = 200,
    ) -> spmatrix:
        """Sparsify graph by keeping edges with high betweenness centrality.

        Edges on many shortest paths (bottlenecks) are preserved.  Edges
        on few paths (redundant within dense regions) are pruned.

        Uses landmark-based approximation for efficiency.
        """
        import numpy as np
        from scipy.sparse import csr_matrix, lil_matrix

        n = adjacency.shape[0]
        adj = adjacency.tocsr() if _issparse(adjacency) else csr_matrix(adjacency)

        if adj.nnz < 3:
            return adj

        # Landmark-based edge betweenness approximation
        rng = np.random.RandomState(42)
        n_landmarks = min(sample_landmarks, n)
        landmarks = rng.choice(n, n_landmarks, replace=False)

        from scipy.sparse.csgraph import dijkstra as csg_dijkstra

        # Accumulate edge scores
        edge_scores: dict[tuple, float] = {}

        for lm in landmarks:
            try:
                dists, predecessors = csg_dijkstra(
                    adj, directed=False, indices=[lm],
                    return_predecessors=True, limit=500,
                )
                # For each reachable node, walk back to landmark
                for v in range(n):
                    if not np.isfinite(dists[0, v]) or v == lm:
                        continue
                    _path_nodes = [v]
                    curr = v
                    while curr != lm and predecessors[0, curr] >= 0:
                        p = predecessors[0, curr]
                        if p == curr or p < 0:
                            break
                        e = (min(curr, p), max(curr, p))
                        edge_scores[e] = edge_scores.get(e, 0.0) + 1.0
                        curr = p
            except Exception:
                continue

        if not edge_scores:
            return adj

        # Sort edges by score and keep top keep_fraction
        sorted_edges = sorted(edge_scores.items(), key=lambda x: -x[1])
        n_keep = max(1, int(len(sorted_edges) * keep_fraction))
        _keep_set = {e for e, _ in sorted_edges[:n_keep]}

        # Build pruned adjacency
        pruned = lil_matrix((n, n), dtype=float)
        for (i, j), _s in sorted_edges[:n_keep]:
            w = adj[i, j]
            pruned[i, j] = w
            pruned[j, i] = w

        result = pruned.tocsr()
        result.eliminate_zeros()
        return result

    @staticmethod
    def compute_local_tangent_consistency(
        X: np.ndarray,
        adjacency: spmatrix,
        *,
        sample_size: int = 500,
    ) -> float:
        """Score 0–1 measuring how well edges follow local tangent direction.

        For each point, its local tangent is estimated via PCA on its
        neighborhood.  An edge is "consistent" if its direction aligns
        with the tangent.  Returns the fraction of consistent edges.

        Low score → many edges cut across the manifold.
        """
        import numpy as np
        from scipy.sparse import csr_matrix
        from sklearn.decomposition import PCA

        n = X.shape[0]
        adj = adjacency.tocsr() if _issparse(adjacency) else csr_matrix(adjacency)
        rng = np.random.RandomState(42)
        sample = rng.choice(n, min(sample_size, n), replace=False)

        if X.shape[1] < 2:
            return 0.5

        consistent = 0
        total = 0
        for i in sample:
            row = adj[i]
            indices = row.indices
            if len(indices) < 3:
                continue
            # Local PCA on neighbor coordinates
            nb_pts = X[indices]
            try:
                pca = PCA(n_components=min(2, X.shape[1]))
                pca.fit(nb_pts)
                tangent = pca.components_[0]  # principal direction
                tangent_norm = np.linalg.norm(tangent)
                if tangent_norm < 1e-8:
                    continue
                tangent = tangent / tangent_norm
            except Exception:
                continue

            for j in indices:
                vec = X[j] - X[i]
                v_norm = np.linalg.norm(vec)
                if v_norm < 1e-8:
                    continue
                vec = vec / v_norm
                # Alignment = |dot(tangent, vec)|
                align = abs(np.dot(tangent, vec))
                if align > 0.5:
                    consistent += 1
                total += 1

        return consistent / max(total, 1)

    @staticmethod
    def compute_graph_quality_metrics(
        X: np.ndarray,
        adjacency: spmatrix,
    ) -> dict[str, Any]:
        """Comprehensive graph construction quality audit.

        Returns dict with:
          - shortcut_edge_ratio : float — fraction of edges flagged
          - wall_crossing_ratio : float — point pairs Euclidean-near but geodesic-far
          - manifold_continuity_score : float — tangent consistency (0–1)
          - local_geodesic_consistency : float — local Euclidean/geodesic correlation
          - graph_distortion_score : float — median geodesic/Euclidean distortion
          - graph_quality_pass : bool — is the graph good enough?
          - diagnosis : str — human-readable explanation
          - recommendations : list[str]
        """
        import numpy as np
        from scipy.sparse import csr_matrix

        X = np.asarray(X, dtype=float)
        n = X.shape[0]
        adj = adjacency.tocsr() if _issparse(adjacency) else csr_matrix(adjacency)

        result: dict[str, Any] = {
            "shortcut_edge_ratio": 0.0,
            "wall_crossing_ratio": 0.0,
            "manifold_continuity_score": 0.5,
            "local_geodesic_consistency": 0.5,
            "graph_distortion_score": 0.0,
            "graph_quality_pass": True,
            "diagnosis": "",
            "recommendations": [],
        }

        # ---- 1. Shortcut edge ratio ----
        shortcut_info = GraphBuilder.detect_shortcut_edges(
            X, adj, sample_size=min(300, n),
        )
        result["shortcut_edge_ratio"] = float(shortcut_info["shortcut_ratio"])

        # ---- 2. Graph distortion (geodesic vs Euclidean) ----
        if n <= 2000:
            try:
                geo_dists = GraphBuilder.compute_geodesic_distances(adj)
                result["graph_distortion_score"] = GraphBuilder.compute_distortion(
                    X, geo_dists, sample_size=min(n, 500),
                )
            except Exception:
                pass

        # ---- 3. Wall-crossing detection (Euclidean-near, geodesic-far) ----
        if n <= 2000:
            try:
                pairs = GraphBuilder.detect_wall_crossings(X, adj, geo_dists)
                result["wall_crossing_ratio"] = float(
                    len(pairs) / max(adj.nnz, 1)
                )
            except Exception:
                pass

        # ---- 4. Manifold continuity (tangent consistency) ----
        result["manifold_continuity_score"] = float(
            GraphBuilder.compute_local_tangent_consistency(X, adj)
        )

        # ---- 5. Local geodesic consistency ----
        # Correlation between local Euclidean and geodesic distances
        if n <= 2000:
            try:
                from sklearn.neighbors import NearestNeighbors
                nn = NearestNeighbors(n_neighbors=min(15, n - 1))
                nn.fit(X)
                euc_dists, euc_idx = nn.kneighbors(X)
                local_euc = euc_dists[:, 1:].mean()
                # Local geodesic: mean path length to same neighbors
                local_geo = 0.0
                count = 0
                for i in range(min(n, 500)):
                    for j_idx in range(min(5, euc_idx.shape[1] - 1)):
                        j = euc_idx[i, j_idx + 1]
                        dg = geo_dists[i, j] if n <= 2000 else euc_dists[i, j_idx + 1]
                        if np.isfinite(dg) and dg > 0:
                            local_geo += dg
                            count += 1
                if count > 0:
                    local_geo /= count
                    result["local_geodesic_consistency"] = float(
                        min(local_euc / max(local_geo, 1e-8), 1.0)
                    )
            except Exception:
                pass

        # ---- 6. Diagnosis ----
        red_flags = 0
        recs = result["recommendations"]

        if result["shortcut_edge_ratio"] > 0.1:
            red_flags += 1
            recs.append(
                f"Shortcut edge ratio {result['shortcut_edge_ratio']:.1%} > 10%。"
                "建议使用 build_wall_aware_graph() 或 prune_shortcut_edges()。"
            )
        if result["wall_crossing_ratio"] > 0.05:
            red_flags += 1
            recs.append(
                f"Wall-crossing ratio {result['wall_crossing_ratio']:.1%} > 5%。"
                "欧氏近邻连接了图结构上不连通的区域。"
            )
        if result["manifold_continuity_score"] < 0.4:
            red_flags += 1
            recs.append(
                f"Manifold continuity {result['manifold_continuity_score']:.2f} < 0.4。"
                "许多边不沿局部流形方向。"
            )
        if result["local_geodesic_consistency"] < 0.5:
            recs.append(
                f"Local geodesic consistency {result['local_geodesic_consistency']:.2f} < 0.5。"
                "局部欧氏距离和 geodesic 距离不一致。"
            )
        if result["graph_distortion_score"] > 0.5:
            red_flags += 1
            recs.append(
                f"Graph distortion {result['graph_distortion_score']:.2f} > 0.5。"
                "图结构与欧氏空间严重偏离。"
            )

        result["graph_quality_pass"] = red_flags < 2

        if not result["graph_quality_pass"]:
            result["diagnosis"] = (
                f"图构建质量不合格（{red_flags} 个红旗）。"
                "在此图上的社区发现结果不可信。"
            )
        elif red_flags == 1:
            result["diagnosis"] = "图构建质量有轻微问题，社区发现结果需谨慎解读。"
            result["graph_quality_pass"] = True
        else:
            result["diagnosis"] = "图构建质量良好，适合社区发现。"

        return result

    @staticmethod
    def build_wall_aware_graph(
        X: np.ndarray,
        *,
        base_k: int | None = None,
        mutual: bool = True,
        use_shared_neighbors: bool = True,
        use_adaptive_scaling: bool = True,
        prune_shortcuts: bool = True,
        sparsify: bool = True,
        jaccard_threshold: float = 0.05,
        shortcut_prune_ratio: float = 0.15,
        sparsify_keep: float = 0.70,
        random_state: int = 42,
    ) -> tuple[spmatrix, dict[str, Any]]:
        """Build a wall-aware graph — THE primary graph construction pipeline.

        Pipeline stages:
          1. Mutual kNN with adaptive local scaling (self-tuning weights)
          2. Shared-neighbor Jaccard similarity weighting
          3. Shortcut edge detection and pruning
          4. Bottleneck-preserving sparsification
          5. Graph quality metrics audit

        Returns (adjacency, quality_report).

        This is the recommended entry point for ALL graph-connected data.
        Replaces raw ``kneighbors_graph`` + ``minimum`` calls.
        """
        import numpy as np

        n = X.shape[0]
        X = np.asarray(X, dtype=float)
        if base_k is None:
            base_k = min(30, max(5, int(np.sqrt(n))))

        build_report: dict[str, Any] = {
            "stages": [],
            "base_k": base_k,
            "n": n,
            "final_edge_count": 0,
        }

        # ---- Stage 1: Adaptive scaling mutual kNN ----
        if use_adaptive_scaling:
            adj = GraphBuilder.build_adaptive_scaling_graph(
                X, k=base_k, mutual=mutual,
            )
            build_report["stages"].append("adaptive_scaling")
        else:
            adj = GraphBuilder.build_knn_graph(
                X, k=base_k, mutual=mutual, mode="distance",
            )
            build_report["stages"].append("basic_knn")

        # ---- Stage 2: Shared-neighbor weighting ----
        if use_shared_neighbors:
            jaccard_weights = GraphBuilder.compute_shared_neighbor_similarity(adj)
            adj = adj.multiply(jaccard_weights)
            # Zero out low-Jaccard edges
            adj.data[adj.data < jaccard_threshold] = 0
            adj.eliminate_zeros()
            build_report["stages"].append("shared_neighbor")

        # ---- Stage 3: Shortcut edge pruning ----
        if prune_shortcuts:
            shortcut_info = GraphBuilder.detect_shortcut_edges(
                X, adj, jaccard_threshold=jaccard_threshold,
            )
            build_report["shortcuts_detected"] = len(shortcut_info["shortcut_edge_indices"])
            build_report["shortcut_ratio"] = shortcut_info["shortcut_ratio"]

            adj, prune_report = GraphBuilder.prune_shortcut_edges(
                X, adj, jaccard_threshold=jaccard_threshold,
            )
            build_report["shortcuts_removed"] = prune_report.get("edges_removed", 0)
            build_report["stages"].append("shortcut_pruning")

        # ---- Stage 4: Bottleneck-preserving sparsification ----
        if sparsify and adj.nnz > 100:
            adj = GraphBuilder.sparsify_by_edge_betweenness(
                adj, keep_fraction=sparsify_keep,
            )
            build_report["stages"].append("sparsification")

        # ---- Stage 5: Quality audit ----
        quality_report = GraphBuilder.compute_graph_quality_metrics(X, adj)
        quality_report.update(build_report)
        quality_report["final_edge_count"] = adj.nnz

        return adj, quality_report
