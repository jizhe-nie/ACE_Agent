"""
tests/test_zoo_expert.py
========================
P0.5-A: Unit tests for ZooExpert (BaseExpert interface adaptation).

Coverage targets:
- _generate_code produces non-empty, syntactically valid Python
- execute_with_self_correction on make_moons dataset:
    (a) results non-empty
    (b) DBSCAN result exists with metrics["score"] > 0.3  OR  ARI > 0.9
- Deprecated run() alias raises DeprecationWarning
- ZooExpert is a proper subclass of BaseExpert
- Soft-failure path in base.py: success=True but empty artifacts triggers retry
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path
from unittest.mock import MagicMock, patch

_root_parent = str(Path(__file__).resolve().parents[2])
if _root_parent not in sys.path:
    sys.path.insert(0, _root_parent)

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from sklearn.datasets import make_moons  # noqa: E402

from ACE_Agent.agent_core.schemas import DatasetBundle  # noqa: E402
from ACE_Agent.expert_sub_agents.base import BaseExpert  # noqa: E402
from ACE_Agent.expert_sub_agents.zoo_expert import ZooExpert  # noqa: E402
from ACE_Agent.tools.llm_client import LLMSettings  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def moons_bundle() -> DatasetBundle:
    """make_moons(200, noise=0.08) as DatasetBundle."""
    X, y = make_moons(n_samples=200, noise=0.08, random_state=42)
    return DatasetBundle(
        name="moons",
        X=X,
        y=y,
        display_name="Half-Moon Test",
        metadata={"expected_clusters": 2},
    )


@pytest.fixture()
def offline_settings() -> LLMSettings:
    """Unconfigured LLM — forces offline/fallback paths."""
    return LLMSettings(enabled=False)


@pytest.fixture()
def zoo() -> ZooExpert:
    return ZooExpert()


# ---------------------------------------------------------------------------
# Structural tests
# ---------------------------------------------------------------------------


class TestZooExpertStructure:
    def test_is_base_expert_subclass(self, zoo: ZooExpert) -> None:
        assert isinstance(zoo, BaseExpert)

    def test_key_is_zoo(self, zoo: ZooExpert) -> None:
        assert zoo.key == "zoo"

    def test_label_set(self, zoo: ZooExpert) -> None:
        assert isinstance(zoo.label, str)
        assert len(zoo.label) > 0

    def test_has_execute_with_self_correction(self, zoo: ZooExpert) -> None:
        assert callable(getattr(zoo, "execute_with_self_correction", None))


# ---------------------------------------------------------------------------
# _generate_code tests
# ---------------------------------------------------------------------------


class TestZooExpertGenerateCode:
    def test_returns_nonempty_string(self, zoo: ZooExpert, moons_bundle: DatasetBundle) -> None:
        dummy_client = MagicMock()
        code = zoo._generate_code(dummy_client, moons_bundle, "run all")
        assert isinstance(code, str)
        assert len(code) > 100

    def test_code_contains_dbscan(self, zoo: ZooExpert, moons_bundle: DatasetBundle) -> None:
        dummy_client = MagicMock()
        code = zoo._generate_code(dummy_client, moons_bundle, "run all")
        assert "DBSCAN" in code

    def test_code_contains_hdbscan_or_skip(self, zoo: ZooExpert, moons_bundle: DatasetBundle) -> None:
        dummy_client = MagicMock()
        code = zoo._generate_code(dummy_client, moons_bundle, "run all")
        # Either HDBSCAN is present OR there's a graceful skip comment
        assert "HDBSCAN" in code or "hdbscan" in code.lower()

    def test_code_contains_artifacts_assignment(self, zoo: ZooExpert, moons_bundle: DatasetBundle) -> None:
        dummy_client = MagicMock()
        code = zoo._generate_code(dummy_client, moons_bundle, "run all")
        # Must write into artifacts dict
        assert "artifacts[" in code

    def test_code_contains_score_field(self, zoo: ZooExpert, moons_bundle: DatasetBundle) -> None:
        dummy_client = MagicMock()
        code = zoo._generate_code(dummy_client, moons_bundle, "run all")
        assert "score" in code

    def test_code_syntactically_valid(self, zoo: ZooExpert, moons_bundle: DatasetBundle) -> None:
        """Generated code must compile without SyntaxError."""
        dummy_client = MagicMock()
        code = zoo._generate_code(dummy_client, moons_bundle, "run all")
        compile(code, "<zoo_generated>", "exec")  # raises SyntaxError if invalid

    def test_high_dimensional_data_uses_pca(self, zoo: ZooExpert) -> None:
        """5D data should trigger PCA path in generated code."""
        X_5d = np.random.default_rng(0).normal(size=(100, 5))
        bundle_5d = DatasetBundle(name="hd", X=X_5d, y=None, metadata={"expected_clusters": 3})
        dummy_client = MagicMock()
        code = zoo._generate_code(dummy_client, bundle_5d, "run all")
        # PCA降维逻辑应存在
        assert "PCA" in code


# ---------------------------------------------------------------------------
# End-to-end execution tests (no LLM, sandbox runs generated code)
# ---------------------------------------------------------------------------


class TestZooExpertExecution:
    def test_results_nonempty_on_moons(
        self, zoo: ZooExpert, moons_bundle: DatasetBundle, offline_settings: LLMSettings
    ) -> None:
        """(a) 结果非空：至少有一个算法成功。"""
        results = zoo.execute_with_self_correction(moons_bundle, "run all", offline_settings)
        assert len(results) > 0, "Expected at least one AlgorithmRunResult"

    def test_dbscan_result_exists(
        self, zoo: ZooExpert, moons_bundle: DatasetBundle, offline_settings: LLMSettings
    ) -> None:
        """(b) DBSCAN 结果存在。"""
        results = zoo.execute_with_self_correction(moons_bundle, "run all", offline_settings)
        algo_names = [r.algorithm_name for r in results]
        assert "DBSCAN" in algo_names, f"DBSCAN not found in {algo_names}"

    def test_dbscan_score_or_ari_threshold(
        self, zoo: ZooExpert, moons_bundle: DatasetBundle, offline_settings: LLMSettings
    ) -> None:
        """(b) DBSCAN 在 make_moons(noise=0.08) 上 score > 0.3 OR ARI > 0.9。"""
        results = zoo.execute_with_self_correction(moons_bundle, "run all", offline_settings)
        dbscan_results = [r for r in results if r.algorithm_name == "DBSCAN"]
        assert dbscan_results, "DBSCAN result missing"
        metrics = dbscan_results[0].metrics
        score = metrics.get("score", 0.0)
        ari = metrics.get("ari", 0.0)
        assert score > 0.3 or ari > 0.9, (
            f"DBSCAN score={score:.4f}, ARI={ari:.4f}: expected score > 0.3 OR ARI > 0.9 on low-noise moons data"
        )

    def test_results_have_required_metric_fields(
        self, zoo: ZooExpert, moons_bundle: DatasetBundle, offline_settings: LLMSettings
    ) -> None:
        """Every result must have a 'score' field in metrics."""
        results = zoo.execute_with_self_correction(moons_bundle, "run all", offline_settings)
        for r in results:
            assert "score" in r.metrics, f"{r.algorithm_name} missing 'score' in metrics"

    def test_kmeans_result_exists(
        self, zoo: ZooExpert, moons_bundle: DatasetBundle, offline_settings: LLMSettings
    ) -> None:
        """KMeans should always be in results."""
        results = zoo.execute_with_self_correction(moons_bundle, "run all", offline_settings)
        algo_names = [r.algorithm_name for r in results]
        assert "KMeans" in algo_names, f"KMeans not found in {algo_names}"

    def test_last_logs_populated(
        self, zoo: ZooExpert, moons_bundle: DatasetBundle, offline_settings: LLMSettings
    ) -> None:
        """last_logs should be populated after execution."""
        zoo.execute_with_self_correction(moons_bundle, "run all", offline_settings)
        assert hasattr(zoo, "last_logs")
        assert len(zoo.last_logs) > 0


# ---------------------------------------------------------------------------
# Deprecated run() alias
# ---------------------------------------------------------------------------


class TestZooExpertDeprecatedRun:
    def test_run_emits_deprecation_warning(self, zoo: ZooExpert, moons_bundle: DatasetBundle) -> None:
        """run() must emit DeprecationWarning."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            zoo.run(moons_bundle, output_dir=Path("outputs/test"))
        dep_warnings = [w for w in caught if issubclass(w.category, DeprecationWarning)]
        assert len(dep_warnings) >= 1

    def test_run_returns_list(self, zoo: ZooExpert, moons_bundle: DatasetBundle) -> None:
        """Deprecated run() must return a list (possibly empty if no LLM)."""
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = zoo.run(moons_bundle, output_dir=Path("outputs/test"))
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# P0.5-D: Soft-failure path (success=True, empty artifacts → retry)
# ---------------------------------------------------------------------------


class TestSoftFailurePath:
    def test_empty_artifacts_triggers_retry(self, moons_bundle: DatasetBundle, offline_settings: LLMSettings) -> None:
        """
        When sandbox returns success=True but empty artifacts, base.py must
        log a soft-failure message and attempt to retry (up to MAX_RETRIES).
        """
        zoo = ZooExpert()

        # Patch sandbox to always return success=True, artifacts={}
        empty_result = {"success": True, "artifacts": {}, "error": None}
        with patch.object(zoo.sandbox, "execute", return_value=empty_result):
            results = zoo.execute_with_self_correction(moons_bundle, "run all", offline_settings)

        # Results should be empty (all retries produced empty artifacts)
        assert results == []
        # Logs should mention the soft failure
        soft_fail_logs = [log for log in zoo.last_logs if "软失败" in log or "artifacts" in log]
        assert len(soft_fail_logs) > 0, f"Expected soft-failure log entry. Got logs: {zoo.last_logs}"

    def test_error_report_contains_expert_logs(self) -> None:
        """_error_report with expert_logs should embed log snippets in summary."""
        from ACE_Agent.agent_core.supervisor import ACESupervisor

        sv = ACESupervisor()
        report = sv._error_report(
            "所有专家执行均失败。",
            trace=["trace1"],
            expert_logs={
                "centroid": ["log A", "log B", "log C: error details"],
                "zoo": ["zoo log 1"],
            },
        )
        # Should include expert key names and log content
        assert "centroid" in report.executive_summary
        assert "zoo" in report.executive_summary
        # Should include at least part of the last log entry
        assert "error details" in report.executive_summary or "log C" in report.executive_summary
