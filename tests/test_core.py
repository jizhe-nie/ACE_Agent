"""
tests/test_core.py
==================
Core unit tests for ACE Agent infrastructure.

Covers:
- MasterRouter intent analysis (offline / no LLM needed)
- ACESupervisor instantiation and error-path report
- DatasetBundle via generate_dataset
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root's parent is on sys.path for ACE_Agent.* imports
_root_parent = str(Path(__file__).resolve().parents[2])
if _root_parent not in sys.path:
    sys.path.insert(0, _root_parent)

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from ACE_Agent.agent_core.router import MasterRouter  # noqa: E402
from ACE_Agent.agent_core.schemas import DatasetBundle  # noqa: E402
from ACE_Agent.agent_core.supervisor import ACESupervisor  # noqa: E402
from ACE_Agent.tools.data_factory import generate_dataset  # noqa: E402
from ACE_Agent.tools.llm_client import LLMSettings  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def unconfigured_settings() -> LLMSettings:
    """LLMSettings with no API key — forces offline / fallback paths."""
    return LLMSettings(
        provider="DeepSeek",
        base_url="https://api.deepseek.com",
        api_key="",
        model="deepseek-chat",
        enabled=False,
    )


@pytest.fixture()
def blobs_dataset() -> DatasetBundle:
    return generate_dataset("blobs", n_samples=60)


@pytest.fixture()
def moons_dataset() -> DatasetBundle:
    return generate_dataset("moons", n_samples=60)


# ---------------------------------------------------------------------------
# MasterRouter tests
# ---------------------------------------------------------------------------


class TestMasterRouter:
    def test_offline_returns_new_task(self, unconfigured_settings: LLMSettings) -> None:
        """When LLM is not configured, router defaults to NEW_TASK."""
        router = MasterRouter()
        result = router.analyze_intent("分析这个数据集", [], unconfigured_settings)
        assert result.get("intent") == "NEW_TASK"

    def test_offline_reasoning_present(self, unconfigured_settings: LLMSettings) -> None:
        router = MasterRouter()
        result = router.analyze_intent("test", [], unconfigured_settings)
        assert "reasoning" in result
        assert isinstance(result["reasoning"], str)

    def test_intent_keys_present(self, unconfigured_settings: LLMSettings) -> None:
        router = MasterRouter()
        result = router.analyze_intent("hello", [], unconfigured_settings)
        assert "intent" in result

    def test_intent_is_uppercase(self, unconfigured_settings: LLMSettings) -> None:
        router = MasterRouter()
        result = router.analyze_intent("hello", [], unconfigured_settings)
        assert result["intent"] == result["intent"].upper()


# ---------------------------------------------------------------------------
# LLMSettings tests
# ---------------------------------------------------------------------------


class TestLLMSettings:
    def test_is_configured_false_when_empty(self) -> None:
        s = LLMSettings()
        assert not s.is_configured

    def test_is_configured_false_disabled(self) -> None:
        s = LLMSettings(base_url="http://x", api_key="k", model="m", enabled=False)
        assert not s.is_configured

    def test_is_configured_true_when_all_set(self) -> None:
        s = LLMSettings(
            provider="DeepSeek",
            base_url="https://api.deepseek.com",
            api_key="sk-test",
            model="deepseek-chat",
            enabled=True,
        )
        assert s.is_configured


# ---------------------------------------------------------------------------
# generate_dataset tests
# ---------------------------------------------------------------------------


class TestGenerateDataset:
    @pytest.mark.parametrize("name", ["blobs", "moons", "s_curve", "iris"])
    def test_returns_dataset_bundle(self, name: str) -> None:
        ds = generate_dataset(name, n_samples=50)
        assert ds is not None
        assert isinstance(ds.X, np.ndarray)
        assert ds.X.shape[0] > 0

    def test_blobs_shape(self) -> None:
        ds = generate_dataset("blobs", n_samples=120)
        assert ds.X.ndim == 2
        assert ds.X.shape[0] == 120

    def test_display_name_set(self) -> None:
        ds = generate_dataset("blobs")
        assert isinstance(ds.display_name, str)
        assert len(ds.display_name) > 0


# ---------------------------------------------------------------------------
# ACESupervisor instantiation test
# ---------------------------------------------------------------------------


class TestACESupervisor:
    def test_instantiation(self) -> None:
        sv = ACESupervisor()
        assert sv.router is not None
        assert "centroid" in sv.experts
        assert "topology" in sv.experts

    def test_error_report_on_missing_dataset(self, unconfigured_settings: LLMSettings) -> None:
        """Supervisor should return an error report (not crash) when dataset is None."""
        sv = ACESupervisor()
        report = sv.run(
            dataset=None,
            user_prompt="分析数据",
            llm_settings=unconfigured_settings,
            intent_data={"intent": "NEW_TASK", "reasoning": "test"},
        )
        assert report is not None
        assert report.response_type in {"FOLLOW_UP", "CLUSTER_TASK", "ERROR"}

    def test_follow_up_no_dataset(self, unconfigured_settings: LLMSettings) -> None:
        """FOLLOW_UP intent should not require a dataset."""
        sv = ACESupervisor()
        report = sv.run(
            dataset=None,
            user_prompt="请解释轮廓系数的含义",
            llm_settings=unconfigured_settings,
            intent_data={"intent": "FOLLOW_UP", "reasoning": "咨询问题"},
        )
        assert report is not None
        # Should not crash and should return some summary text
        assert isinstance(report.executive_summary, str)
