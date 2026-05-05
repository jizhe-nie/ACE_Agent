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
from typing import Any

import numpy as np
from sklearn.cluster import AgglomerativeClustering

from ACE_Agent.agent_core.schemas import AlgorithmRunResult, DatasetBundle
from ACE_Agent.expert_sub_agents.base import BaseExpert
from ACE_Agent.tools.llm_client import LLMSettings, UniversalLLMClient

_logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Circuit breaker threshold — above this N, switch to Monte Carlo sampling
# ---------------------------------------------------------------------------
_MC_THRESHOLD = 20_000
_MC_SAMPLE_PAIRS = 10_000


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
    ) -> AlgorithmRunResult | None:
        """Fuse labels from *results* and return a consensus result.

        Returns ``None`` if there are fewer than 2 valid label sets.
        """
        # Collect valid (labels, score) tuples
        entries: list[tuple[list[int], float]] = []
        for r in results:
            lbls = getattr(r, "labels", None)
            if lbls is None or (hasattr(lbls, "__len__") and len(lbls) == 0):
                continue
            if hasattr(lbls, "tolist"):
                lbls = lbls.tolist()
            elif isinstance(lbls, np.ndarray):
                lbls = lbls.tolist()
            entries.append((list(lbls), r.metrics.get("score", 0.0)))

        if len(entries) < 2:
            _logger.info("Ensemble: need >= 2 valid label sets, got %d — skipped.", len(entries))
            return None

        n_samples = len(entries[0][0])
        for lbls, _ in entries:
            if len(lbls) != n_samples:
                _logger.warning("Ensemble: mismatched label lengths (%d vs %d) — skipped.",
                                n_samples, len(lbls))
                return None

        # ---- Determine k (majority vote among experts) -------------------
        k_counts: dict[int, int] = {}
        for lbls, _ in entries:
            k = len(set(lbls))
            k_counts[k] = k_counts.get(k, 0) + 1
        k_consensus = max(k_counts, key=lambda kk: k_counts[kk])

        # ---- Normalise scores to [0,1] for weighted fusion ---------------
        scores = np.array([s for _, s in entries])
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
        # For each pair (i,j): s = coassoc*M experts put i,j together.
        # Fraction of expert-pairs that agree:
        #   agree[i,j] = 1 - 2*M*coassoc*(1-coassoc) / (M-1)
        # Average over all (i,j), excluding diagonal.
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
        # Entropy: 0 = perfect agreement, 1 = random
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
        for lbls, _ in entries:
            k_e = len(set(lbls))
            expert_names.append(f"k={k_e}")

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
            },
            plot_path="",
            params={
                "coassoc_matrix": coassoc_display,
                "expert_names": expert_names,
            },
        )

    # ------------------------------------------------------------------
    # Full N×N co-association matrix
    # ------------------------------------------------------------------

    @staticmethod
    def _full_consensus(
        entries: list[tuple[list[int], float]],
        weights: np.ndarray,
        n_samples: int,
        k: int,
    ) -> tuple[list[int], np.ndarray]:
        """Build the full co-association matrix and cluster with hierarchical.

        Returns (consensus_labels, coassoc_matrix).
        """
        coassoc = np.zeros((n_samples, n_samples), dtype=np.float64)
        for (lbls, _), w in zip(entries, weights):
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
        entries: list[tuple[list[int], float]],
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

        for (lbls, _), w in zip(entries, weights):
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
