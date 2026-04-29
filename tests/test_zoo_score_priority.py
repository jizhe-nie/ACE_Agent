"""
tests/test_zoo_score_priority.py
================================
Regression tests for the score-priority fix.

When ground-truth labels are available, ranking must use ARI (unbiased w.r.t.
cluster shape). Silhouette is structurally biased against non-convex clusters
and misranks DBSCAN below KMeans/GMM on moons-like data.

Tests:
1. test_score_equals_ari_when_labels_provided: ARI is used; DBSCAN outranks KMeans.
2. test_score_falls_back_to_silhouette_when_no_labels: silhouette fallback works.
"""

from __future__ import annotations

import sys
from pathlib import Path

_root_parent = str(Path(__file__).resolve().parents[2])
if _root_parent not in sys.path:
    sys.path.insert(0, _root_parent)

import pytest  # noqa: E402
from sklearn.datasets import make_moons  # noqa: E402

from ACE_Agent.agent_core.schemas import DatasetBundle  # noqa: E402
from ACE_Agent.expert_sub_agents.zoo_expert import ZooExpert  # noqa: E402
from ACE_Agent.tools.llm_client import LLMSettings  # noqa: E402


@pytest.fixture()
def moons_bundle_with_labels() -> DatasetBundle:
    X, y = make_moons(n_samples=200, noise=0.05, random_state=42)
    return DatasetBundle(
        name="moons_labeled",
        X=X,
        y=y,
        display_name="Moons (with labels)",
        metadata={"expected_clusters": 2},
    )


@pytest.fixture()
def moons_bundle_no_labels() -> DatasetBundle:
    X, _y = make_moons(n_samples=200, noise=0.05, random_state=42)
    return DatasetBundle(
        name="moons_unlabeled",
        X=X,
        y=None,
        display_name="Moons (no labels)",
        metadata={"expected_clusters": 2},
    )


@pytest.fixture()
def offline_settings() -> LLMSettings:
    return LLMSettings(enabled=False)


def _by_name(results, name):
    matches = [r for r in results if r.algorithm_name == name]
    assert matches, f"{name} not found in results: {[r.algorithm_name for r in results]}"
    return matches[0]


def test_score_equals_ari_when_labels_provided(
    moons_bundle_with_labels: DatasetBundle, offline_settings: LLMSettings
) -> None:
    """With y supplied, metrics['score'] must equal metrics['ari'] and
    score_source must be 'ari'. DBSCAN should outrank KMeans (the core bug)."""
    zoo = ZooExpert()
    results = zoo.execute_with_self_correction(moons_bundle_with_labels, "run all", offline_settings)
    assert results, "No results produced"

    dbscan = _by_name(results, "DBSCAN")
    kmeans = _by_name(results, "KMeans")

    # Score == ARI when labels provided
    assert "ari" in dbscan.metrics, f"ARI missing from DBSCAN metrics: {dbscan.metrics}"
    assert dbscan.metrics.get("score_source") == "ari", (
        f"Expected score_source='ari', got {dbscan.metrics.get('score_source')!r}"
    )
    assert dbscan.metrics["score"] == pytest.approx(dbscan.metrics["ari"]), (
        f"score ({dbscan.metrics['score']}) != ari ({dbscan.metrics['ari']})"
    )

    assert kmeans.metrics.get("score_source") == "ari"
    assert kmeans.metrics["score"] == pytest.approx(kmeans.metrics["ari"])

    # The whole point of the fix: DBSCAN (correct crescents) > KMeans (chopped crescents)
    assert dbscan.metrics["score"] > kmeans.metrics["score"], (
        f"DBSCAN score ({dbscan.metrics['score']:.4f}) should exceed "
        f"KMeans score ({kmeans.metrics['score']:.4f}) on moons with ARI ranking"
    )


def test_score_falls_back_to_silhouette_when_no_labels(
    moons_bundle_no_labels: DatasetBundle, offline_settings: LLMSettings
) -> None:
    """Without y, score must equal silhouette and score_source must be 'silhouette'."""
    zoo = ZooExpert()
    results = zoo.execute_with_self_correction(moons_bundle_no_labels, "run all", offline_settings)
    assert results, "No results produced"

    for r in results:
        # Only check algos that actually clustered (not degenerate n_labels<2 cases)
        sil = r.metrics.get("silhouette", 0.0)
        src = r.metrics.get("score_source")
        # ARI must not be present when y is None
        assert "ari" not in r.metrics, (
            f"{r.algorithm_name}: ari should not be computed when y is None (got metrics={r.metrics})"
        )
        # When silhouette > 0, score_source should be 'silhouette' and score==silhouette
        if sil > 0:
            assert src == "silhouette", (
                f"{r.algorithm_name}: expected score_source='silhouette' (silhouette={sil}), got {src!r}"
            )
            assert r.metrics["score"] == pytest.approx(sil), (
                f"{r.algorithm_name}: score ({r.metrics['score']}) should equal silhouette ({sil}) when no labels"
            )
