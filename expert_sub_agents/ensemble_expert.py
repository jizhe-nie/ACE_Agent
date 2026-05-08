"""
expert_sub_agents/ensemble_expert.py
=====================================
Ensemble Consensus Expert: Meta-clustering fusion of heterogeneous expert labels.

Instead of selecting the single highest-scoring algorithm, this expert builds a
**Co-association Matrix** from all experts' label vectors and runs
AgglomerativeClustering on the consensus (disagreement) matrix.

Architecture:
  1. Collect labels from all finished experts: ``[labels_0, labels_1, ..., labels_M]``
  2. Build co-association matrix:  C[i,j] = fraction of experts that put i and j
     in the same cluster.
  3. Hierarchical clustering on 1-C with ``n_clusters=k_majority`` (the most
     common k across experts).
  4. Weighted variant: weight each expert by its normalised silhouette score.

Circuit breaker: when N > 20 000, uses Monte Carlo subsampling (10 000 random
pairs) instead of the full N×N matrix to keep memory O(samples) rather than
O(N^2).

Phase 2 (2026-05): Revived from shelved status — Phase 3 deep clustering is
complete, and we now have 10+ algorithms / 7 pipelines worth fusing.
"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import AgglomerativeClustering

from ACE_Agent.agent_core.schemas import AlgorithmRunResult, DatasetBundle
from ACE_Agent.expert_sub_agents.base import BaseExpert
from ACE_Agent.tools.llm_client import UniversalLLMClient

_logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Circuit breaker threshold — above this N, switch to Monte Carlo sampling
# ---------------------------------------------------------------------------
_MC_THRESHOLD = 20_000
_MC_SAMPLE_PAIRS = 10_000

# ---------------------------------------------------------------------------
# Algorithm taxonomy for dynamic vote rebalancing (Phase 3: 3-way)
# ---------------------------------------------------------------------------
_GRAPH_ALGORITHMS = {
    "SpectralClustering", "Spectral_Graph", "Spectral_Laplacian",
    "AgglomerativeClustering", "Agglomerative_Graph", "LabelPropagation",
    "EnsembleConsensus",
}
_DENSITY_ALGORITHMS = {
    "HDBSCAN", "DBSCAN", "OPTICS",
}
_CENTROID_ALGORITHMS = {
    "KMeans", "MiniBatchKMeans", "Birch", "GMM", "GaussianMixture",
    "MeanShift", "AffinityPropagation",
}
# Phase 2.4 topology boost/penalty (kept for backward compat)
_TOPOLOGY_ALGORITHMS = _DENSITY_ALGORITHMS | _GRAPH_ALGORITHMS
_TOPOLOGY_BOOST = 3.0
_CENTROID_PENALTY = 0.25

# Phase 3 diversity constraint targets
_GRAPH_TARGET = 0.50    # min fraction of total weight
_DENSITY_TARGET = 0.30  # min fraction
_CENTROID_CAP = 0.20    # max fraction


class EnsembleConsensusExpert(BaseExpert):
    """Deterministic meta-expert that fuses labels via co-association matrix.

    This expert does **not** use LLM code generation.  It is called directly
    by the Supervisor after all primary experts have produced results.
    """

    def __init__(self) -> None:
        super().__init__("ensemble", "集成共识专家")

    # ------------------------------------------------------------------
    # Public entry point (called by Supervisor)
    # ------------------------------------------------------------------

    def execute_ensemble(
        self,
        results: list[AlgorithmRunResult],
        dataset: DatasetBundle,
        topology_weighting: bool = False,
        diversity_constraints: bool = False,
        geodesic_distortion: float = 0.0,
    ) -> AlgorithmRunResult | None:
        """Fuse labels from *results* and return a consensus result.

        Parameters
        ----------
        topology_weighting:
            When True, boosts topology-friendly algorithms (HDBSCAN, DBSCAN,
            Spectral) and penalises centroid-based algorithms (KMeans, GMM)
            to prevent Euclidean-dominated voting from drowning out
            connectivity-aware results on manifold/non-convex data.
        diversity_constraints:
            When True, enforces graph >= 50%, density >= 30%, centroid <= 20%
            weight proportions. Prunes centroid results if cap is violated,
            boosts graph results if min threshold not met.
        geodesic_distortion:
            When > 0.3, triggers graph connectivity agreement metric.

        Returns ``None`` if there are fewer than 2 valid label sets.
        """
        # Collect valid (labels, score, algorithm_name) triples
        entries: list[tuple[list[int], float, str]] = []
        for r in results:
            lbls = getattr(r, "labels", None)
            if lbls is None or (hasattr(lbls, "__len__") and len(lbls) == 0):
                continue
            if hasattr(lbls, "tolist") or isinstance(lbls, np.ndarray):
                lbls = lbls.tolist()
            algo = getattr(r, "algorithm_name", "unknown")
            entries.append((list(lbls), r.metrics.get("score", 0.0), algo))

        if len(entries) < 2:
            _logger.info("Ensemble: need >= 2 valid label sets, got %d — skipped.", len(entries))
            return None

        n_samples = len(entries[0][0])
        for lbls, _, _ in entries:
            if len(lbls) != n_samples:
                _logger.warning("Ensemble: mismatched label lengths (%d vs %d) — skipped.",
                                n_samples, len(lbls))
                return None

        # ---- Determine k (majority vote among experts) -------------------
        k_counts: dict[int, int] = {}
        for lbls, _, _ in entries:
            k = len(set(lbls))
            k_counts[k] = k_counts.get(k, 0) + 1
        k_consensus = max(k_counts, key=lambda kk: k_counts[kk])

        # ---- Topology-aware score rebalancing ---------------------------
        raw_scores = np.array([s for _, s, _ in entries], dtype=float)
        algo_names = [a for _, _, a in entries]

        # ---- Phase 3: Diversity-constrained expert pruning ---------------
        if diversity_constraints:
            entries, raw_scores, algo_names = self._apply_diversity_constraints(
                entries, raw_scores, algo_names,
            )

        if topology_weighting:
            multipliers = np.ones(len(entries))
            for i, algo in enumerate(algo_names):
                if algo in _TOPOLOGY_ALGORITHMS:
                    multipliers[i] = _TOPOLOGY_BOOST
                elif algo in _CENTROID_ALGORITHMS:
                    multipliers[i] = _CENTROID_PENALTY
            rebalanced = raw_scores * multipliers
            _logger.info(
                "Ensemble: topology_weighting enabled — multipliers: %s",
                {a: f"{m:.2f}" for a, m in zip(algo_names, multipliers, strict=False)},
            )
            scores = rebalanced
        else:
            scores = raw_scores

        # ---- Phase 3.1: CPS-based re-scoring for graph-connected data ---
        _graph_consensus = False
        if geodesic_distortion > 0.3 and diversity_constraints:
            _graph_consensus = True
            try:
                from ACE_Agent.tools.graph_builder import GraphBuilder
                X_np = np.array(dataset.X if hasattr(dataset, "X") else dataset.X)
                adj = GraphBuilder.build_knn_graph(X_np)
                geo_dists = None
                if n_samples <= 3000:
                    geo_dists = GraphBuilder.compute_geodesic_distances(adj)

                # Score each entry by CPS + modularity (not silhouette)
                graph_scores = np.zeros(len(entries), dtype=float)
                for i, (lbls, _, _) in enumerate(entries):
                    lbl_arr = np.array(lbls, dtype=int)
                    cps_r = GraphBuilder.connectivity_preservation_score(
                        X_np, adj, lbl_arr, geo_dists,
                    )
                    mod = GraphBuilder.compute_modularity(adj, lbl_arr)
                    # CPS-weighted score: high CPS + high modularity = good
                    graph_scores[i] = 0.6 * cps_r["cps"] + 0.4 * mod

                # Blend graph scores with original scores
                if graph_scores.max() - graph_scores.min() > 1e-8:
                    graph_weights = (graph_scores - graph_scores.min()) / (graph_scores.max() - graph_scores.min())
                else:
                    graph_weights = np.ones(len(entries))
                # 70% graph-based + 30% original
                scores = 0.7 * graph_weights + 0.3 * scores / max(scores.max(), 1e-8)
                _logger.info(
                    "Ensemble: CPS graph scoring enabled — scores: %s",
                    {a: f"{s:.4f}" for a, s in zip(algo_names, scores, strict=False)},
                )
            except Exception as exc:
                _logger.warning("CPS scoring failed, using original scores: %s", exc)
                _graph_consensus = False

        # ---- Normalise scores to [0,1] for weighted fusion ---------------
        if scores.max() - scores.min() > 1e-8:
            weights = (scores - scores.min()) / (scores.max() - scores.min())
            weights = np.clip(weights + 0.1, 0.1, 1.0)  # floor at 0.1 so no expert is zeroed
        else:
            weights = np.ones(len(entries))

        # ---- Build co-association matrix --------------------------------
        if n_samples > _MC_THRESHOLD:
            consensus_labels, coassoc = self._mc_consensus(
                entries, weights, n_samples, k_consensus
            )
        else:
            consensus_labels, coassoc = self._full_consensus(
                entries, weights, n_samples, k_consensus
            )

        # ---- Inter-expert agreement from co-association matrix -----------
        M = len(entries)
        if n_samples > _MC_THRESHOLD:
            mean_agreement = float(np.mean(
                1.0 - 2.0 * M * coassoc * (1.0 - coassoc) / (M - 1)
            ))
        else:
            n_pairs = n_samples * (n_samples - 1)
            pair_agreement = (
                1.0 - 2.0 * M * coassoc * (1.0 - coassoc) / (M - 1)
            )
            mean_agreement = float((pair_agreement.sum() - n_samples) / n_pairs)
        p = max(0.01, min(0.99, mean_agreement))
        entropy = float(-p * np.log2(p) - (1 - p) * np.log2(1 - p))

        _logger.info(
            "Ensemble: fused %d experts → k=%d, N=%d, agreement=%.3f, entropy=%.3f",
            len(entries), k_consensus, n_samples, mean_agreement, entropy,
        )

        # ---- Downsample co-assoc matrix for frontend heatmap -----------
        max_dim = 500
        if n_samples > max_dim:
            stride = max(n_samples // max_dim, 1)
            coassoc_display = coassoc[::stride, ::stride].astype(np.float16)
        else:
            coassoc_display = coassoc.astype(np.float16)

        expert_names: list[str] = []
        for lbls, _, _ in entries:
            k_e = len(set(lbls))
            expert_names.append(f"k={k_e}")

        # ---- Generate cluster visualization plot -------------------------
        plot_path = _generate_consensus_plot(
            dataset.X, consensus_labels, n_samples, k_consensus
        )

        # ---- Graph connectivity agreement (Phase 3) -----------------------
        graph_agreement = None
        if geodesic_distortion > 0.3:
            graph_agreement = self._compute_graph_connectivity_agreement(
                entries, dataset.X,
            )

        # ---- Stability analysis (Phase 3): high-disagreement regions ------
        disagreement_ratio = None
        if len(entries) >= 3:
            disagreement_ratio = self._compute_disagreement_ratio(
                entries, consensus_labels, n_samples,
            )

        return AlgorithmRunResult(
            algorithm_name="EnsembleConsensus",
            expert_key=self.key,
            expert_label=self.label,
            labels=consensus_labels,
            metrics={
                "score": float(mean_agreement),
                "score_source": "ensemble_agreement",
                "agreement": float(mean_agreement),
                "entropy_of_agreement": float(entropy),
                "n_experts_fused": len(entries),
                "k_consensus": k_consensus,
                "fusion_method": "coassociation_hierarchical",
                **({} if graph_agreement is None else {"graph_connectivity_agreement": graph_agreement}),
                **({} if disagreement_ratio is None else {"high_disagreement_ratio": disagreement_ratio}),
            },
            plot_path=plot_path,
            params={
                "coassoc_matrix": coassoc_display,
                "expert_names": expert_names,
                **({} if graph_agreement is None else {"graph_agreement": graph_agreement}),
                **({} if disagreement_ratio is None else {"disagreement_ratio": disagreement_ratio}),
            },
        )

    # ------------------------------------------------------------------
    # Full N×N co-association matrix
    # ------------------------------------------------------------------

    @staticmethod
    def _classify_entry(algo_name: str) -> str:
        """Classify an algorithm result into graph / density / centroid."""
        if algo_name in _GRAPH_ALGORITHMS:
            return "graph"
        if algo_name in _DENSITY_ALGORITHMS:
            return "density"
        return "centroid"

    @staticmethod
    def _apply_diversity_constraints(
        entries: list[tuple[list[int], float, str]],
        raw_scores: np.ndarray,
        algo_names: list[str],
    ) -> tuple[list[tuple[list[int], float, str]], np.ndarray, list[str]]:
        """Enforce graph >= 50%, density >= 30%, centroid = 0%.

        Phase 3.1 hard exclusion: centroid algorithms (KMeans, GMM, Birch)
        are **entirely removed** from voting for graph-connected data.
        They can only serve as diagnostic baselines, never influence
        the final partition.

        Returns pruned (entries, scores, names) or original if constraints
        cannot be met without leaving < 2 experts.
        """
        if len(entries) < 2:
            return entries, raw_scores, algo_names

        # ---- Hard exclusion: remove ALL centroid results ----
        categories = [EnsembleConsensusExpert._classify_entry(a) for a in algo_names]
        centroid_indices = [i for i, c in enumerate(categories) if c == "centroid"]
        non_centroid_indices = [i for i, c in enumerate(categories) if c != "centroid"]

        if len(non_centroid_indices) >= 2:
            n_removed = len(centroid_indices)
            entries = [entries[i] for i in non_centroid_indices]
            raw_scores = np.array([raw_scores[i] for i in non_centroid_indices])
            algo_names = [algo_names[i] for i in non_centroid_indices]
            categories = [categories[i] for i in non_centroid_indices]
            if n_removed > 0:
                _logger.info(
                    "Diversity: HARD-EXCLUDED %d centroid results from ensemble."
                    " Only graph+density algorithms participate in voting.",
                    n_removed,
                )
        elif len(non_centroid_indices) < 2 and len(entries) >= 2:
            # Emergency: keep centroid entries but with extreme penalty
            _logger.warning(
                "Diversity: would leave only %d experts after excluding centroids,"
                " keeping centroids with severe penalty.",
                len(non_centroid_indices),
            )
            categories = [EnsembleConsensusExpert._classify_entry(a) for a in algo_names]
            for i, c in enumerate(categories):
                if c == "centroid":
                    raw_scores[i] *= 0.01

        return entries, raw_scores, algo_names

    @staticmethod
    def _full_consensus(
        entries: list[tuple[list[int], float, str]],
        weights: np.ndarray,
        n_samples: int,
        k: int,
    ) -> tuple[list[int], np.ndarray]:
        """Build the full co-association matrix and cluster with hierarchical.

        Returns (consensus_labels, coassoc_matrix).
        """
        coassoc = np.zeros((n_samples, n_samples), dtype=np.float64)
        for (lbls, _, _), w in zip(entries, weights, strict=False):
            lbl_arr = np.array(lbls, dtype=np.int32)
            match = lbl_arr[:, None] == lbl_arr[None, :]
            coassoc += w * match
        coassoc /= weights.sum()

        # Hierarchical clustering on disagreement matrix
        dissim = 1.0 - coassoc
        agg = AgglomerativeClustering(
            n_clusters=k, metric="precomputed", linkage="average",
        )
        return agg.fit_predict(dissim).tolist(), coassoc

    # ------------------------------------------------------------------
    # Monte Carlo consensus (N > threshold)
    # ------------------------------------------------------------------

    @staticmethod
    def _mc_consensus(
        entries: list[tuple[list[int], float, str]],
        weights: np.ndarray,
        n_samples: int,
        k: int,
    ) -> tuple[list[int], np.ndarray]:
        """Monte Carlo: sample random pairs, build sparse consensus.

        Returns (consensus_labels, anchor_coassoc_matrix).
        """
        rng = np.random.RandomState(42)
        # Build sparse agreement from sampled pairs
        pair_counts: dict[tuple[int, int], float] = {}
        pair_weight_sum: dict[tuple[int, int], float] = {}

        for (lbls, _, _), w in zip(entries, weights, strict=False):
            lbl_arr = np.array(lbls, dtype=np.int32)
            # Draw _MC_SAMPLE_PAIRS random pairs
            for _ in range(_MC_SAMPLE_PAIRS):
                i, j = int(rng.randint(0, n_samples)), int(rng.randint(0, n_samples))
                if i == j:
                    continue
                key = (min(i, j), max(i, j))
                agree = 1.0 if lbl_arr[i] == lbl_arr[j] else 0.0
                pair_counts[key] = pair_counts.get(key, 0.0) + w * agree
                pair_weight_sum[key] = pair_weight_sum.get(key, 0.0) + w

        # Build sparse distance matrix — use k-NN graph approximation
        # Sample a subset of points as "anchors", cluster anchors, then assign
        n_anchors = min(n_samples, 2000)
        anchor_idx = rng.choice(n_samples, n_anchors, replace=False)
        anchor_set = set(anchor_idx.tolist())

        # Build anchor-to-anchor consensus
        anchor_coassoc = np.zeros((n_anchors, n_anchors), dtype=np.float64)
        for ai, i in enumerate(anchor_idx):
            for aj, j in enumerate(anchor_idx):
                if ai == aj:
                    anchor_coassoc[ai, aj] = 1.0
                    continue
                key = (min(i, j), max(i, j))
                total_w = pair_weight_sum.get(key, 0.0)
                if total_w > 0:
                    anchor_coassoc[ai, aj] = pair_counts.get(key, 0.0) / total_w
                else:
                    anchor_coassoc[ai, aj] = 0.5  # no info → neutral

        anchor_dissim = 1.0 - anchor_coassoc
        clustering = AgglomerativeClustering(
            n_clusters=k, metric="precomputed", linkage="average",
        )
        anchor_labels = clustering.fit_predict(anchor_dissim)

        # Assign remaining points to nearest anchor by consensus agreement
        full_labels = np.full(n_samples, -1, dtype=np.int32)
        for ai, i in enumerate(anchor_idx):
            full_labels[i] = int(anchor_labels[ai])

        # For non-anchor points, find most similar anchor
        for i in range(n_samples):
            if i in anchor_set:
                continue
            best_agree = -1.0
            best_anchor = 0
            for ai, j in enumerate(anchor_idx):
                key = (min(i, j), max(i, j))
                total_w = pair_weight_sum.get(key, 0.0)
                if total_w > 0:
                    agree = pair_counts.get(key, 0.0) / total_w
                else:
                    agree = 0.5
                if agree > best_agree:
                    best_agree = agree
                    best_anchor = ai
            full_labels[i] = int(anchor_labels[best_anchor])

        return full_labels.tolist(), anchor_coassoc

    # ------------------------------------------------------------------
    # Graph connectivity agreement & stability (Phase 3)
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_graph_connectivity_agreement(
        entries: list[tuple[list[int], float, str]],
        X: Any,
    ) -> float:
        """Fraction of intra-cluster edges that exist in the kNN graph.

        Averages across experts. Higher means labels respect graph structure.
        """
        try:
            import numpy as np
            from sklearn.neighbors import NearestNeighbors

            X_np = np.asarray(X, dtype=float)
            n = X_np.shape[0]
            k = min(15, n - 1)
            knn = NearestNeighbors(n_neighbors=k).fit(X_np)
            _, knn_indices = knn.kneighbors(X_np)

            agreements = []
            for lbls, _, _ in entries:
                lbl_arr = np.array(lbls, dtype=int)
                same = 0
                total = 0
                for i in range(min(n, 500)):  # sample points for speed
                    for j in knn_indices[i]:
                        if i >= j:
                            continue
                        total += 1
                        if lbl_arr[i] == lbl_arr[j]:
                            same += 1
                if total > 0:
                    agreements.append(same / total)
            if not agreements:
                return 0.0
            return float(np.mean(agreements))
        except Exception as exc:
            _logger.warning("Graph connectivity agreement failed: %s", exc)
            return 0.0

    @staticmethod
    def _compute_disagreement_ratio(
        entries: list[tuple[list[int], float, str]],
        consensus_labels: list[int],
        n_samples: int,
    ) -> float:
        """Fraction of points where >= 50% of experts disagree with consensus."""
        try:
            import numpy as np
            lbls = [np.array(e[0], dtype=int) for e in entries]
            disagree_count = 0
            for i in range(n_samples):
                votes = sum(1 for l in lbls if l[i] != consensus_labels[i])
                if votes >= len(lbls) / 2:
                    disagree_count += 1
            return disagree_count / n_samples
        except Exception as exc:
            _logger.warning("Disagreement ratio failed: %s", exc)
            return 0.0

    # ------------------------------------------------------------------
    # No LLM code generation needed — this is a deterministic expert
    # ------------------------------------------------------------------

    def _generate_code(
        self,
        client: UniversalLLMClient,
        dataset: DatasetBundle,
        prompt: str,
        constraints=None,
    ) -> str:
        """Ensemble expert does not use LLM — returns a no-op skeleton."""
        return (
            "# EnsembleConsensusExpert is deterministic; "
            "the Supervisor calls execute_ensemble() directly.\n"
            "artifacts['Ensemble_Consensus'] = {'labels': [], 'metrics': {}}"
        )

    def _fix_code(
        self,
        client: UniversalLLMClient,
        old_code: str,
        error: str,
        *,
        attempt: int = 1,
    ) -> str:
        """Not used — self-healing not needed for deterministic expert."""
        return self._generate_code(client, None, "")  # type: ignore[arg-type]


# ==========================================================================
# Helper: generate consensus cluster scatter plot
# ==========================================================================

_OUTPUTS_DIR = Path(__file__).resolve().parents[1] / "outputs"


def _generate_consensus_plot(
    X: Any,
    labels: list[int],
    n_samples: int,
    k: int,
) -> str:
    """Generate a scatter plot of data colored by consensus labels.

    Returns the path to the saved PNG, or empty string on failure.
    """
    try:
        X_np = np.asarray(X, dtype=float)
        if X_np.ndim != 2 or X_np.shape[0] != n_samples:
            return ""

        # Reduce to 2D for visualization if needed
        if X_np.shape[1] > 2:
            from sklearn.decomposition import PCA
            X_vis = PCA(n_components=2, random_state=42).fit_transform(X_np)
            xlabel, ylabel = "PC1", "PC2"
        else:
            X_vis = X_np
            xlabel, ylabel = "x1", "x2"

        # Downsample large datasets for faster rendering
        max_plot_pts = 5000
        if n_samples > max_plot_pts:
            idx = np.random.default_rng(42).choice(n_samples, max_plot_pts, replace=False)
            X_vis = X_vis[idx]
            lbl_vis = np.array(labels, dtype=int)[idx]
        else:
            lbl_vis = np.array(labels, dtype=int)

        _OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        uid = uuid.uuid4().hex[:8]
        path = _OUTPUTS_DIR / f"ensemble_consensus_{uid}.png"

        # Chinese font setup for Windows
        import platform
        if platform.system() == "Windows":
            plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
        plt.rcParams["axes.unicode_minus"] = False

        fig, ax = plt.subplots(figsize=(7, 6))
        scatter = ax.scatter(
            X_vis[:, 0], X_vis[:, 1],
            c=lbl_vis, cmap="tab10", s=3 if n_samples > 2000 else 8,
            alpha=0.6, edgecolors="none",
        )
        ax.set_title(
            f"Ensemble Consensus ({k} clusters, {n_samples} samples)",
            fontsize=12, weight="bold",
        )
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.85)
        cbar.set_label("Cluster", fontsize=9)
        plt.tight_layout()
        fig.savefig(path, dpi=130, bbox_inches="tight")
        plt.close(fig)

        _logger.info("Ensemble consensus plot saved: %s", path)
        return str(path)
    except Exception as exc:
        _logger.warning("Failed to generate ensemble consensus plot: %s", exc)
        return ""
