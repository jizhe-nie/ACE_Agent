"""
tests/test_system_integration.py
=================================
Comprehensive system integration tests covering all Phase 2 features
end-to-end.  Goes beyond unit tests to verify real wiring between modules.

Coverage:
  Phase 1: Critic audit, Dimension expert, Benchmark
  Phase 2.1: Ensemble Consensus Expert, coassoc matrix, conditional trigger
  Phase 2.2: Critic 2.0 closed-loop (action/RETRY/constraints)
  Phase 2.3: HITL label correction, reference_labels constraint
  Phase 0: Sandbox security, LLM client abstraction, RAG engine
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

_root_parent = str(Path(__file__).resolve().parents[2])
if _root_parent not in sys.path:
    sys.path.insert(0, _root_parent)


# ============================================================================
# Module 1: Import chain integrity
# ============================================================================

class TestImportChain:
    """Every module that should be importable, is."""

    def test_import_expert_registry(self) -> None:
        from ACE_Agent.expert_sub_agents import build_expert_registry
        registry = build_expert_registry()
        assert len(registry) == 7, f"Expected 7 experts, got {len(registry)}: {list(registry.keys())}"
        for key in ["centroid", "topology", "zoo", "critic", "dimension", "ensemble", "graph"]:
            assert key in registry, f"Missing expert: {key}"

    def test_import_supervisor(self) -> None:
        from ACE_Agent.agent_core.supervisor import ACESupervisor
        sv = ACESupervisor()
        assert sv.router is not None
        assert sv.knowledge_engine is not None
        assert len(sv.experts) == 7

    def test_import_router(self) -> None:
        from ACE_Agent.agent_core.router import MasterRouter
        router = MasterRouter()
        assert router is not None

    def test_import_schemas(self) -> None:
        from ACE_Agent.agent_core.schemas import (
            AlgorithmRunResult,
            DatasetBundle,
            SupervisorReport,
            RoutingDecision,
        )
        # Verify we can construct key schemas
        ds = DatasetBundle(name="test", X=np.array([[1, 2], [3, 4]]), y=None)
        assert ds.X.shape == (2, 2)
        result = AlgorithmRunResult(
            algorithm_name="KMeans",
            expert_key="centroid",
            expert_label="质心专家",
            labels=[0, 1],
            metrics={"score": 0.85},
            plot_path=Path("test.png"),
        )
        assert result.algorithm_name == "KMeans"

    def test_import_llm_client(self) -> None:
        from ACE_Agent.tools.llm_client import LLMSettings, UniversalLLMClient
        settings = LLMSettings(
            provider="DeepSeek",
            base_url="http://localhost",
            api_key="test",
            model="test",
            enabled=True,
        )
        assert settings.provider == "DeepSeek"

    def test_import_sandbox(self) -> None:
        from ACE_Agent.tools.coder_sandbox import CoderSandbox
        sb = CoderSandbox()
        assert sb is not None

    def test_import_knowledge_engine(self) -> None:
        from ACE_Agent.agent_brain.knowledge_engine import KnowledgeEngine
        # KnowledgeEngine should be importable (may fail if chromadb not available,
        # but the import itself should work)
        assert KnowledgeEngine is not None

    def test_import_latex_generator(self) -> None:
        from ACE_Agent.tools.latex_generator import LatexReportGenerator
        gen = LatexReportGenerator()
        assert gen is not None

    def test_import_all_experts(self) -> None:
        from ACE_Agent.expert_sub_agents.centroid_expert import CentroidExpert
        from ACE_Agent.expert_sub_agents.topology_expert import TopologyExpert
        from ACE_Agent.expert_sub_agents.zoo_expert import ZooExpert
        from ACE_Agent.expert_sub_agents.critic_expert import CriticExpert
        from ACE_Agent.expert_sub_agents.dimension_expert import DimensionExpert
        from ACE_Agent.expert_sub_agents.ensemble_expert import EnsembleConsensusExpert
        # All should instantiate without error
        for cls in [CentroidExpert, TopologyExpert, ZooExpert, CriticExpert, DimensionExpert, EnsembleConsensusExpert]:
            expert = cls()
            assert expert.key is not None
            assert expert.label is not None


# ============================================================================
# Module 2: Constraint injection protocol
# ============================================================================

class TestConstraintInjection:
    """All constraint types produce correct prompt directives."""

    def test_empty_constraints(self) -> None:
        from ACE_Agent.expert_sub_agents.base import BaseExpert
        assert BaseExpert._inject_constraints_prompt(None) == ""
        assert BaseExpert._inject_constraints_prompt({}) == ""

    def test_force_k(self) -> None:
        from ACE_Agent.expert_sub_agents.base import BaseExpert
        result = BaseExpert._inject_constraints_prompt({"force_k": 5})
        assert "k 必须为 5" in result

    def test_blocked_algorithms(self) -> None:
        from ACE_Agent.expert_sub_agents.base import BaseExpert
        result = BaseExpert._inject_constraints_prompt({"blocked_algorithms": ["DBSCAN", "SpectralClustering"]})
        assert "禁止使用以下算法" in result
        assert "DBSCAN" in result
        assert "SpectralClustering" in result

    def test_force_preprocessing(self) -> None:
        from ACE_Agent.expert_sub_agents.base import BaseExpert
        result = BaseExpert._inject_constraints_prompt({"force_preprocessing": "standardize"})
        assert "standardize" in result

    def test_reference_labels_small(self) -> None:
        from ACE_Agent.expert_sub_agents.base import BaseExpert
        result = BaseExpert._inject_constraints_prompt({"reference_labels": [0, 1, 0]})
        assert "HITL" in result
        assert "参考标签" in result
        assert "3 个数据点" in result
        assert "ARI" in result or "NMI" in result

    def test_reference_labels_truncated(self) -> None:
        from ACE_Agent.expert_sub_agents.base import BaseExpert
        labels = list(range(50))
        result = BaseExpert._inject_constraints_prompt({"reference_labels": labels})
        assert "..." in result
        assert "50 个数据点" in result

    def test_all_constraints_combined(self) -> None:
        from ACE_Agent.expert_sub_agents.base import BaseExpert
        result = BaseExpert._inject_constraints_prompt({
            "force_k": 3,
            "blocked_algorithms": ["MeanShift"],
            "force_preprocessing": "pca",
            "reference_labels": [0, 1, 0, 1, 2],
        })
        assert "k 必须为 3" in result
        assert "MeanShift" in result
        assert "pca" in result
        assert "HITL" in result


# ============================================================================
# Module 3: Supervisor routing paths
# ============================================================================

class TestSupervisorRouting:
    """Supervisor.run() handles all intent types correctly."""

    def _settings(self):
        from ACE_Agent.tools.llm_client import LLMSettings
        return LLMSettings(
            provider="DeepSeek",
            base_url="http://unused",
            api_key="test",
            model="unused",
            enabled=True,
        )

    def test_run_follow_up_path(self) -> None:
        """FOLLOW_UP intent does not require a dataset."""
        from ACE_Agent.agent_core.supervisor import ACESupervisor
        from ACE_Agent.tools.llm_client import UniversalLLMClient

        sv = ACESupervisor()

        def fake_chat(self, messages, system_prompt=None, *, caller=None, attempt=1):
            return "这是一个追问回答。"

        with patch.object(UniversalLLMClient, "chat_completion", fake_chat):
            report = sv.run(
                dataset=None,
                user_prompt="为什么选择KMeans?",
                llm_settings=self._settings(),
                intent_data={"intent": "FOLLOW_UP", "reasoning": "追问"},
            )
        assert report.response_type == "FOLLOW_UP"
        assert "追问回答" in report.executive_summary

    def test_run_code_example_path(self) -> None:
        """CODE_EXAMPLE intent generates code without sandbox."""
        from ACE_Agent.agent_core.supervisor import ACESupervisor
        from ACE_Agent.tools.llm_client import UniversalLLMClient

        sv = ACESupervisor()

        def fake_chat(self, messages, system_prompt=None, *, caller=None, attempt=1):
            return "```python\nfrom sklearn.cluster import KMeans\n```"

        with patch.object(UniversalLLMClient, "chat_completion", fake_chat):
            report = sv.run(
                dataset=None,
                user_prompt="生成KMeans代码示例",
                llm_settings=self._settings(),
                intent_data={"intent": "CODE_EXAMPLE", "reasoning": "代码示例"},
            )
        assert report.response_type == "CODE_EXAMPLE"
        assert "```python" in report.executive_summary

    def test_run_new_task_without_dataset_errors(self) -> None:
        """NEW_TASK without dataset returns error report."""
        from ACE_Agent.agent_core.supervisor import ACESupervisor

        sv = ACESupervisor()
        report = sv.run(
            dataset=None,
            user_prompt="分析这个数据集",
            llm_settings=self._settings(),
            intent_data={"intent": "NEW_TASK", "reasoning": "新任务"},
        )
        assert "未识别到数据" in report.executive_summary

    def test_run_new_task_with_constraints(self) -> None:
        """NEW_TASK with HITL constraints passes through to experts."""
        from ACE_Agent.agent_core.supervisor import ACESupervisor
        from ACE_Agent.tools.llm_client import UniversalLLMClient
        from ACE_Agent.agent_core.schemas import DatasetBundle

        sv = ACESupervisor()

        def fake_chat(self, messages, system_prompt=None, *, caller=None, attempt=1):
            return "约束已应用的回答。"

        with patch.object(UniversalLLMClient, "chat_completion", fake_chat):
            ds = DatasetBundle(name="test", X=np.array([[1, 2], [3, 4], [5, 6]]), y=None)
            report = sv.run(
                dataset=ds,
                user_prompt="聚类分析",
                llm_settings=self._settings(),
                intent_data={"intent": "NEW_TASK", "reasoning": "新任务"},
                constraints={"reference_labels": [0, 1, 0]},
            )
        # Should have trace mentioning HITL
        trace_text = "\n".join(report.decision_trace)
        assert "HITL" in trace_text


# ============================================================================
# Module 4: Expert self-correction loop (base.py)
# ============================================================================

class TestSelfCorrectionLoop:
    """Verify the Think-Act-Fix cycle mechanics."""

    def _settings(self):
        from ACE_Agent.tools.llm_client import LLMSettings
        return LLMSettings(
            provider="DeepSeek",
            base_url="http://unused",
            api_key="test",
            model="unused",
            enabled=True,
        )

    @staticmethod
    def _fake_sandbox_execute(code, X, y, **kwargs):
        """Simulate successful sandbox execution with artifacts."""
        return {
            "success": True,
            "artifacts": {
                "KMeans": {
                    "labels": [0, 1, 0, 1],
                    "metrics": {"score": 0.9, "silhouette": 0.85},
                    "plot_path": "",
                }
            },
            "logs": ["运行成功"],
        }

    def test_execute_with_self_correction_success(self) -> None:
        """Expert produces results on first attempt."""
        from ACE_Agent.expert_sub_agents.centroid_expert import CentroidExpert
        from ACE_Agent.agent_core.schemas import DatasetBundle
        from ACE_Agent.tools.llm_client import UniversalLLMClient

        expert = CentroidExpert()

        def fake_generate_code(self, client, dataset, prompt, constraints=None):
            return "print('hello')\nartifacts['KMeans'] = {'labels': [0,1,0,1], 'metrics': {'score': 0.9}, 'plot_path': ''}"

        def fake_chat(self, messages, system_prompt=None, *, caller=None, attempt=1):
            return "fake"

        ds = DatasetBundle(name="test", X=np.array([[1, 2], [3, 4], [5, 6], [7, 8]]), y=None)

        with patch.object(type(expert), "_generate_code", fake_generate_code):
            # sandbox is an instance attribute, not class — use patch.object on instance
            with patch.object(expert, "sandbox") as mock_sb:
                mock_sb.execute = self._fake_sandbox_execute
                with patch.object(UniversalLLMClient, "chat_completion", fake_chat):
                    results = expert.execute_with_self_correction(ds, "test prompt", self._settings())
        assert len(results) == 1
        assert results[0].algorithm_name == "KMeans"
        assert results[0].metrics["score"] == 0.9

    def test_execute_with_constraints_passed_to_generate(self) -> None:
        """Constraints are passed through to _generate_code."""
        from ACE_Agent.expert_sub_agents.centroid_expert import CentroidExpert
        from ACE_Agent.agent_core.schemas import DatasetBundle
        from ACE_Agent.tools.llm_client import UniversalLLMClient

        expert = CentroidExpert()
        captured_constraints = []

        def fake_generate_code(self, client, dataset, prompt, constraints=None):
            captured_constraints.append(constraints)
            return "artifacts['KMeans'] = {'labels': [0,1], 'metrics': {'score': 1.0}, 'plot_path': ''}"

        def fake_chat(self, messages, system_prompt=None, *, caller=None, attempt=1):
            return "fake"

        ds = DatasetBundle(name="test", X=np.array([[1, 2], [3, 4]]), y=None)

        with patch.object(type(expert), "_generate_code", fake_generate_code):
            with patch.object(expert, "sandbox") as mock_sb:
                mock_sb.execute = self._fake_sandbox_execute
                with patch.object(UniversalLLMClient, "chat_completion", fake_chat):
                    expert.execute_with_self_correction(
                        ds, "test", self._settings(),
                        constraints={"reference_labels": [0, 1]},
                    )
        assert len(captured_constraints) == 1
        assert captured_constraints[0]["reference_labels"] == [0, 1]


# ============================================================================
# Module 5: Ensemble Consensus Expert
# ============================================================================

class TestEnsembleIntegration:
    """Ensemble consensus expert works with real AlgorithmRunResult objects."""

    def _settings(self):
        from ACE_Agent.tools.llm_client import LLMSettings
        return LLMSettings(
            provider="DeepSeek",
            base_url="http://unused",
            api_key="test",
            model="unused",
            enabled=True,
        )

    def test_ensemble_with_real_results(self) -> None:
        """Ensemble fuses real result objects properly."""
        from ACE_Agent.expert_sub_agents.ensemble_expert import EnsembleConsensusExpert
        from ACE_Agent.agent_core.schemas import AlgorithmRunResult, DatasetBundle

        expert = EnsembleConsensusExpert()
        ds = DatasetBundle(
            name="test", X=np.random.default_rng(42).standard_normal((100, 2)), y=None,
        )

        r1 = AlgorithmRunResult(
            algorithm_name="KMeans", expert_key="centroid", expert_label="Test",
            labels=[0] * 33 + [1] * 34 + [2] * 33,
            metrics={"score": 0.9},
            plot_path=Path("/tmp/kmeans.png"),
        )
        r1_labels = r1.labels[:100]  # ensure exactly 100

        r2 = AlgorithmRunResult(
            algorithm_name="GMM", expert_key="topology", expert_label="Test",
            labels=[0] * 33 + [1] * 34 + [2] * 33,
            metrics={"score": 0.85},
            plot_path=Path("/tmp/gmm.png"),
        )
        r2.labels = r2.labels[:100]

        # Fix labels length
        rng = np.random.default_rng(42)
        r1_labels = [0] * 33 + [1] * 34 + [2] * 33
        r2_labels = [0] * 33 + [1] * 34 + [2] * 33
        r1.labels = r1_labels
        r2.labels = r2_labels

        result = expert.execute_ensemble([r1, r2], ds)
        assert result is not None
        assert result.algorithm_name == "EnsembleConsensus"
        assert result.labels is not None
        assert len(result.labels) == 100
        assert "coassoc_matrix" in result.params
        assert result.metrics["n_experts_fused"] == 2
        # Two identical label sets → perfect agreement
        assert result.metrics["agreement"] > 0.99

    def test_ensemble_with_divergent_labels(self) -> None:
        """Ensemble handles divergent label sets gracefully."""
        from ACE_Agent.expert_sub_agents.ensemble_expert import EnsembleConsensusExpert
        from ACE_Agent.agent_core.schemas import AlgorithmRunResult, DatasetBundle

        expert = EnsembleConsensusExpert()
        ds = DatasetBundle(
            name="test", X=np.random.default_rng(42).standard_normal((50, 2)), y=None,
        )

        # Two experts with different labeling
        r1 = AlgorithmRunResult(
            algorithm_name="KMeans", expert_key="centroid", expert_label="Test",
            labels=[0] * 25 + [1] * 25,
            metrics={"score": 0.8},
            plot_path=Path("/tmp/kmeans.png"),
        )
        r2 = AlgorithmRunResult(
            algorithm_name="Agglomerative", expert_key="topology", expert_label="Test",
            labels=[0] * 15 + [1] * 20 + [0] * 10 + [1] * 5,
            metrics={"score": 0.6},
            plot_path=Path("/tmp/agg.png"),
        )

        result = expert.execute_ensemble([r1, r2], ds)
        assert result is not None
        assert len(result.labels) == 50
        assert 0.0 < result.metrics["agreement"] < 0.99  # partial agreement
        assert result.metrics["entropy_of_agreement"] > 0


# ============================================================================
# Module 6: Critic 2.0 closed-loop logic
# ============================================================================

class TestCritic20Integration:
    """Critic 2.0 _handle_audit_feedback logic is correct across all actions."""

    def test_clear_action_no_retry(self) -> None:
        from ACE_Agent.agent_core.supervisor import ACESupervisor
        sv = ACESupervisor()
        trace: list[str] = []
        # Dataset is not actually used for CLEAR so we can pass None
        result = sv._handle_audit_feedback(
            {"action": "CLEAR", "endorsement": "endorsed", "confidence_level": 0.9},
            None, "", None, trace, ["centroid"],  # type: ignore[arg-type]
        )
        assert result == []

    def test_warn_action_no_retry_with_trace(self) -> None:
        from ACE_Agent.agent_core.supervisor import ACESupervisor
        sv = ACESupervisor()
        trace: list[str] = []
        result = sv._handle_audit_feedback(
            {"action": "WARN", "endorsement": "qualified"}, None, "", None, trace, ["centroid"],  # type: ignore[arg-type]
        )
        assert result == []
        assert any("WARN" in t for t in trace)

    def test_retry_without_constraints_returns_empty(self) -> None:
        from ACE_Agent.agent_core.supervisor import ACESupervisor
        sv = ACESupervisor()
        trace: list[str] = []
        result = sv._handle_audit_feedback(
            {"action": "RETRY", "retry_constraints": {}}, None, "", None, trace, ["centroid"],  # type: ignore[arg-type]
        )
        assert result == []

    def test_none_audit_returns_empty(self) -> None:
        from ACE_Agent.agent_core.supervisor import ACESupervisor
        sv = ACESupervisor()
        trace: list[str] = []
        result = sv._handle_audit_feedback(None, None, "", None, trace, ["centroid"])  # type: ignore[arg-type]
        assert result == []


# ============================================================================
# Module 7: Sandbox security baseline
# ============================================================================

class TestSandboxSecurity:
    """Verify sandbox blocks dangerous operations."""

    def test_sandbox_blocks_os_system(self) -> None:
        from ACE_Agent.tools.coder_sandbox import CoderSandbox
        sb = CoderSandbox()
        X = np.array([[1, 2], [3, 4]])
        result = sb.execute("import os; os.system('echo pwned')", X, None)
        assert not result["success"]

    def test_sandbox_blocks_subprocess(self) -> None:
        from ACE_Agent.tools.coder_sandbox import CoderSandbox
        sb = CoderSandbox()
        X = np.array([[1, 2], [3, 4]])
        result = sb.execute("import subprocess; subprocess.run(['echo', 'test'])", X, None)
        assert not result["success"]

    def test_sandbox_blocks_file_write(self) -> None:
        from ACE_Agent.tools.coder_sandbox import CoderSandbox
        sb = CoderSandbox()
        X = np.array([[1, 2], [3, 4]])
        result = sb.execute("open('/etc/passwd', 'w').write('x')", X, None)
        assert not result["success"]

    def test_sandbox_allows_valid_clustering(self) -> None:
        from ACE_Agent.tools.coder_sandbox import CoderSandbox
        sb = CoderSandbox()
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]])
        code = """
from sklearn.cluster import KMeans
labels = KMeans(n_clusters=2, n_init=10, random_state=42).fit_predict(X)
artifacts['KMeans'] = {'labels': labels.tolist(), 'metrics': {'score': 0.5}, 'plot_path': ''}
"""
        result = sb.execute(code, X, None)
        assert result["success"]
        assert "KMeans" in result["artifacts"]

    def test_sandbox_data_context_injection(self) -> None:
        from ACE_Agent.tools.coder_sandbox import CoderSandbox
        sb = CoderSandbox()
        X = np.array([[1, 2], [3, 4], [5, 6]])
        code = """
assert CTX_DATA.n_samples == 3
assert CTX_DATA.n_features == 2
labels = [0, 1, 0]
artifacts['Test'] = {'labels': labels, 'metrics': {'score': 1.0}, 'plot_path': ''}
"""
        result = sb.execute(code, X, None)
        assert result["success"]


# ============================================================================
# Module 8: KnowledgeEngine baseline
# ============================================================================

class TestKnowledgeEngineBaseline:
    """KnowledgeEngine can be instantiated and queried (even with empty db)."""

    def test_engine_initializes(self) -> None:
        from ACE_Agent.agent_brain.knowledge_engine import KnowledgeEngine
        engine = KnowledgeEngine()
        assert engine.collection is not None
        assert engine.client is not None

    def test_query_empty_db_returns_empty(self) -> None:
        from ACE_Agent.agent_brain.knowledge_engine import KnowledgeEngine
        engine = KnowledgeEngine()
        result = engine.query("clustering parameter selection")
        # May be empty string or have content depending on prior ingestion
        assert isinstance(result, str)


# ============================================================================
# Module 9: Schema data integrity
# ============================================================================

class TestSchemaIntegrity:
    """Data schemas round-trip correctly."""

    def test_supervisor_report_round_trip(self) -> None:
        from ACE_Agent.agent_core.schemas import (
            AlgorithmRunResult,
            DatasetBundle,
            RoutingDecision,
            SupervisorReport,
        )
        ds = DatasetBundle(name="roundtrip", X=np.array([[1, 2], [3, 4]]), y=None)
        result = AlgorithmRunResult(
            algorithm_name="TestAlgo",
            expert_key="test",
            expert_label="Test Expert",
            labels=[0, 1],
            metrics={"score": 0.95, "silhouette": 0.88},
            plot_path=Path("/tmp/test.png"),
        )
        report = SupervisorReport(
            dataset=ds,
            routing=RoutingDecision(None, [], ["trace1"]),
            dataset_plot_path=Path("/tmp/ds.png"),
            output_dir=Path("/tmp"),
            results=[result],
            ranking=[result],
            executive_summary="一切正常。",
            decision_trace=["trace1", "trace2"],
            audit_report={"endorsement": "endorsed", "confidence_level": 0.9},
            response_type="CLUSTER_TASK",
        )
        assert report.response_type == "CLUSTER_TASK"
        assert len(report.ranking) == 1
        assert report.ranking[0].metrics["score"] == 0.95
        assert report.audit_report["endorsement"] == "endorsed"


# ============================================================================
# Module 10: ZooExpert registry integrity
# ============================================================================

class TestZooExpertRegistry:
    """ZooExpert is properly wired through the registry."""

    def test_zoo_in_registry_and_generates_code(self) -> None:
        from ACE_Agent.expert_sub_agents import build_expert_registry
        registry = build_expert_registry()
        assert "zoo" in registry
        zoo = registry["zoo"]
        assert zoo.key == "zoo"
        # ZooExpert should have the generate_code method
        assert hasattr(zoo, "_generate_code")

    def test_zoo_generates_code_with_sandbox_context(self) -> None:
        from ACE_Agent.expert_sub_agents import build_expert_registry
        registry = build_expert_registry()
        zoo = registry["zoo"]
        from ACE_Agent.agent_core.schemas import DatasetBundle
        from ACE_Agent.tools.llm_client import LLMSettings, UniversalLLMClient

        ds = DatasetBundle(name="test", X=np.array([[1, 2], [3, 4], [5, 6]]), y=None)
        settings = LLMSettings(provider="DeepSeek", base_url="http://u", api_key="k", model="m", enabled=True)

        def fake_chat(self, messages, system_prompt=None, *, caller=None, attempt=1):
            return "from sklearn.cluster import KMeans\nartifacts['KMeans']={'labels':[0,1,0],'metrics':{'score':0.5},'plot_path':''}"

        with patch.object(UniversalLLMClient, "chat_completion", fake_chat):
            code = zoo._generate_code(UniversalLLMClient(settings), ds, "cluster this")
        assert "KMeans" in code or "artifacts" in code or "sklearn" in code.lower()


# ============================================================================
# Module 11: Sandbox resource exceeded handling
# ============================================================================

class TestSandboxResourceHandling:
    """Resource limits are properly surfaced."""

    def test_timeout_detected(self) -> None:
        from ACE_Agent.tools.coder_sandbox import CoderSandbox, SandboxResourceExceeded
        sb = CoderSandbox(timeout_sec=1)
        X = np.array([[1, 2], [3, 4]])
        # Infinite loop should raise SandboxResourceExceeded (timeout)
        with pytest.raises(SandboxResourceExceeded) as exc_info:
            sb.execute("while True: pass", X, None)
        assert exc_info.value.reason == "timeout"

    def test_sandbox_resource_exceeded_importable(self) -> None:
        from ACE_Agent.tools.coder_sandbox import SandboxResourceExceeded
        exc = SandboxResourceExceeded("timeout", "test")
        assert exc.reason == "timeout"


# ============================================================================
# Module 12: code_fence stripping
# ============================================================================

class TestCodeFenceStripping:
    """_strip_code_fences handles all common LLM output patterns."""

    def test_strip_python_fence(self) -> None:
        from ACE_Agent.expert_sub_agents.base import _strip_code_fences
        result = _strip_code_fences("```python\nprint('hello')\n```")
        assert result == "print('hello')"

    def test_strip_bare_fence(self) -> None:
        from ACE_Agent.expert_sub_agents.base import _strip_code_fences
        result = _strip_code_fences("```\ncode here\n```")
        assert result == "code here"

    def test_no_fence_unchanged(self) -> None:
        from ACE_Agent.expert_sub_agents.base import _strip_code_fences
        result = _strip_code_fences("print('hello')")
        assert result == "print('hello')"

    def test_none_input(self) -> None:
        from ACE_Agent.expert_sub_agents.base import _strip_code_fences
        assert _strip_code_fences(None) == ""


# ============================================================================
# Module 13: Error report enrichment
# ============================================================================

class TestErrorReport:
    """Supervisor._error_report enriches with expert logs."""

    def test_error_report_with_expert_logs(self) -> None:
        from ACE_Agent.agent_core.supervisor import ACESupervisor
        sv = ACESupervisor()
        report = sv._error_report(
            "测试错误",
            ["trace1"],
            expert_logs={"centroid": ["log1", "log2", "log3", "log4"]},
        )
        assert "排错信息" in report.executive_summary
        assert "centroid" in report.executive_summary


# ============================================================================
# Module 14: Full pipeline smoke test (no LLM)
# ============================================================================

class TestFullPipelineSmoke:
    """End-to-end pipeline with mocked LLM."""

    def _settings(self):
        from ACE_Agent.tools.llm_client import LLMSettings
        return LLMSettings(
            provider="DeepSeek",
            base_url="http://unused",
            api_key="test",
            model="unused",
            enabled=True,
        )

    def test_full_pipeline_smoke(self) -> None:
        """Run the full supervisor pipeline with all experts mocked."""
        from ACE_Agent.agent_core.supervisor import ACESupervisor
        from ACE_Agent.agent_core.schemas import DatasetBundle
        from ACE_Agent.tools.llm_client import UniversalLLMClient

        sv = ACESupervisor()
        settings = self._settings()
        ds = DatasetBundle(
            name="smoke_test",
            X=np.random.default_rng(123).standard_normal((30, 2)),
            y=None,
        )

        # Simulate LLM always returning a valid clustering script
        def fake_chat(self, messages, system_prompt=None, *, caller=None, attempt=1):
            return (
                "from sklearn.cluster import KMeans\n"
                "labels = KMeans(n_clusters=2, n_init=10, random_state=42).fit_predict(X)\n"
                "artifacts['KMeans'] = {'labels': labels.tolist(), 'metrics': {'score': 0.5, 'silhouette': 0.4}, 'plot_path': ''}"
            )

        with patch.object(UniversalLLMClient, "chat_completion", fake_chat):
            # Patch summarize_report too since it also calls chat_completion
            report = sv.run(
                dataset=ds,
                user_prompt="对这个数据集做聚类分析",
                llm_settings=settings,
                intent_data={"intent": "NEW_TASK", "reasoning": "用户要求聚类"},
            )

        assert report.response_type == "CLUSTER_TASK"
        assert len(report.results) > 0
        # At least one result from each active expert
        assert len(report.results) >= 3  # centroid + topology + zoo
        # Ranking is sorted by score descending
        for i in range(len(report.ranking) - 1):
            assert report.ranking[i].metrics.get("score", 0) >= report.ranking[i + 1].metrics.get("score", 0)
        # Each result has required fields
        for r in report.results:
            assert r.algorithm_name
            assert r.expert_key
            assert r.metrics.get("score") is not None

    def test_full_pipeline_with_hitl_constraints(self) -> None:
        """Full pipeline with HITL reference_labels constraints."""
        from ACE_Agent.agent_core.supervisor import ACESupervisor
        from ACE_Agent.agent_core.schemas import DatasetBundle
        from ACE_Agent.tools.llm_client import UniversalLLMClient

        sv = ACESupervisor()
        settings = self._settings()
        ds = DatasetBundle(
            name="hitl_test",
            X=np.random.default_rng(456).standard_normal((20, 2)),
            y=None,
        )

        def fake_chat(self, messages, system_prompt=None, *, caller=None, attempt=1):
            return (
                "from sklearn.cluster import KMeans\n"
                "labels = KMeans(n_clusters=2, n_init=10, random_state=42).fit_predict(X)\n"
                "artifacts['KMeans'] = {'labels': labels.tolist(), 'metrics': {'score': 0.5}, 'plot_path': ''}"
            )

        with patch.object(UniversalLLMClient, "chat_completion", fake_chat):
            report = sv.run(
                dataset=ds,
                user_prompt="聚类分析",
                llm_settings=settings,
                intent_data={"intent": "NEW_TASK", "reasoning": "测试"},
                constraints={"reference_labels": [0, 1, 0, 1, 0] * 4},
            )
        assert report.response_type == "CLUSTER_TASK"
        trace_text = "\n".join(report.decision_trace)
        assert "HITL" in trace_text
