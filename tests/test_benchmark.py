"""Tests for ACE Agent benchmark suite."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

_root_parent = str(Path(__file__).resolve().parents[2])
if _root_parent not in sys.path:
    sys.path.insert(0, _root_parent)

from sklearn.datasets import make_blobs, make_moons  # noqa: E402

from ACE_Agent.benchmark.config import BenchmarkConfig  # noqa: E402
from ACE_Agent.benchmark.metrics import ClusteringMetricsCalculator  # noqa: E402
from ACE_Agent.benchmark.reporter import BenchmarkReporter  # noqa: E402
from ACE_Agent.benchmark.runner import BenchmarkReport, BenchmarkRunResult, BenchmarkRunner  # noqa: E402
from ACE_Agent.tools.data_factory import generate_dataset  # noqa: E402
from ACE_Agent.tools.llm_client import LLMSettings  # noqa: E402


# ======================================================================
# ClusteringMetricsCalculator
# ======================================================================

class TestClusteringMetricsCalculator:
    """Tests for ClusteringMetricsCalculator."""

    def test_compute_all_with_labels(self) -> None:
        """ARI, Silhouette, CHI, DBI all computed when y is provided."""
        X, y = make_blobs(n_samples=60, centers=3, random_state=42)
        result = ClusteringMetricsCalculator.compute_all(X, y, y)
        assert not np.isnan(result["ari"])
        assert result["ari"] > 0.9  # perfect match
        assert not np.isnan(result["silhouette"])
        assert not np.isnan(result["calinski_harabasz"])
        assert not np.isnan(result["davies_bouldin"])

    def test_compute_all_without_labels(self) -> None:
        """ARI is NaN when y is None; others still computed."""
        X, _ = make_blobs(n_samples=60, centers=3, random_state=42)
        labels = np.random.randint(0, 3, 60)
        result = ClusteringMetricsCalculator.compute_all(X, labels, None)
        assert np.isnan(result["ari"])
        assert not np.isnan(result["silhouette"])

    def test_degenerate_single_cluster(self) -> None:
        """All metrics return NaN for single-label degenerate case."""
        X, _ = make_blobs(n_samples=30, centers=1, random_state=42)
        labels = np.zeros(30, dtype=int)
        result = ClusteringMetricsCalculator.compute_all(X, labels, np.zeros(30, dtype=int))
        assert np.isnan(result["ari"])
        assert np.isnan(result["silhouette"])


class TestSelfHealingStats:
    """Tests for compute_self_healing_stats."""

    def test_success_on_first_try(self) -> None:
        """One attempt, success."""
        logs = [
            "[全量算法专家] 开始分析数据特征并生成代码...",
            "[全量算法专家] 第 1 次尝试运行代码...",
            "[全量算法专家] 运行成功！",
        ]
        stats = ClusteringMetricsCalculator.compute_self_healing_stats(logs)
        assert stats["attempts"] == 1
        assert stats["success"] is True
        assert stats["soft_failures"] == 0
        assert stats["error"] is None

    def test_soft_failure_then_success(self) -> None:
        """Soft failure on first attempt, succeeds on second."""
        logs = [
            "[全量算法专家] 第 1 次尝试运行代码...",
            "[全量算法专家] 软失败：代码运行成功但未产出 artifacts...",
            "[全量算法专家] 正在注入 artifacts 约定并重写代码 (第1次重试)...",
            "[全量算法专家] 第 2 次尝试运行代码...",
            "[全量算法专家] 运行成功！",
        ]
        stats = ClusteringMetricsCalculator.compute_self_healing_stats(logs)
        assert stats["attempts"] == 2
        assert stats["success"] is True
        assert stats["soft_failures"] == 1

    def test_all_retries_exhausted(self) -> None:
        """Three attempts all fail."""
        logs = [
            "[评价专家] 第 1 次尝试运行代码...",
            "[评价专家] 运行失败，错误信息: NameError: name 'foo' is not defined",
            "[评价专家] 第 2 次尝试运行代码...",
            "[评价专家] 运行失败，错误信息: NameError: name 'foo' is not defined",
            "[评价专家] 第 3 次尝试运行代码...",
            "[评价专家] 运行失败，错误信息: NameError: name 'foo' is not defined",
            "[评价专家] 重试次数耗尽，任务失败。",
        ]
        stats = ClusteringMetricsCalculator.compute_self_healing_stats(logs)
        assert stats["attempts"] == 3
        assert stats["success"] is False
        assert stats["soft_failures"] == 0
        assert stats["error"] is not None

    def test_empty_logs(self) -> None:
        """Empty logs produce safe defaults."""
        stats = ClusteringMetricsCalculator.compute_self_healing_stats([])
        assert stats["attempts"] == 0
        assert stats["success"] is False
        assert stats["soft_failures"] == 0


# ======================================================================
# BenchmarkConfig
# ======================================================================

class TestBenchmarkConfig:
    """Tests for BenchmarkConfig."""

    def test_defaults(self) -> None:
        """Default config has sensible values."""
        c = BenchmarkConfig()
        assert len(c.datasets) == 7
        assert c.experts == ["zoo"]
        assert c.n_samples == 480
        assert c.offline_mode is False
        assert c.min_success_rate == 0.80

    def test_custom(self) -> None:
        """Custom config overrides defaults."""
        c = BenchmarkConfig(
            datasets=["blobs", "iris"],
            experts=["zoo", "centroid"],
            n_samples=100,
            offline_mode=True,
        )
        assert c.datasets == ["blobs", "iris"]
        assert c.experts == ["zoo", "centroid"]
        assert c.n_samples == 100
        assert c.offline_mode is True


# ======================================================================
# BenchmarkRunner (offline)
# ======================================================================

class TestBenchmarkRunnerOffline:
    """Offline benchmark tests using ZooExpert only (no LLM required)."""

    @pytest.fixture()
    def config(self, tmp_path: Path) -> BenchmarkConfig:
        return BenchmarkConfig(
            datasets=["blobs", "moons", "iris"],
            experts=["zoo"],
            n_samples=60,
            offline_mode=True,
            output_dir=str(tmp_path),
        )

    @pytest.fixture()
    def config_single(self, tmp_path: Path) -> BenchmarkConfig:
        return BenchmarkConfig(
            datasets=["blobs"],
            experts=["zoo"],
            n_samples=60,
            offline_mode=True,
            output_dir=str(tmp_path),
        )

    def test_resolve_experts_offline(self, config: BenchmarkConfig) -> None:
        """Offline mode keeps zoo (REQUIRES_LLM=False)."""
        runner = BenchmarkRunner(config)
        resolved = runner._resolve_experts()
        assert "zoo" in resolved

    def test_resolve_experts_drops_centroid(self, tmp_path: Path) -> None:
        """Offline mode drops centroid (REQUIRES_LLM=True)."""
        c = BenchmarkConfig(
            datasets=["blobs"], experts=["zoo", "centroid"],
            n_samples=60, offline_mode=True, output_dir=str(tmp_path),
        )
        runner = BenchmarkRunner(c)
        resolved = runner._resolve_experts()
        assert "zoo" in resolved
        assert "centroid" not in resolved

    def test_runner_produces_nonempty_results(self, config: BenchmarkConfig) -> None:
        """Runner produces results for each dataset."""
        runner = BenchmarkRunner(config)
        report = runner.run()
        assert len(report.results) > 0
        # ZooExpert runs ~10 algorithms per dataset × 3 datasets
        assert len(report.results) >= 20

    def test_runner_all_algorithms_present(self, config_single: BenchmarkConfig) -> None:
        """Verify major algorithms appear in results."""
        runner = BenchmarkRunner(config_single)
        report = runner.run()
        algos = {r.algorithm for r in report.results if r.success}
        assert "KMeans" in algos
        assert "DBSCAN" in algos
        assert "GaussianMixture" in algos or "AgglomerativeClustering" in algos

    def test_runner_results_have_metric_fields(self, config_single: BenchmarkConfig) -> None:
        """Each successful result has all metric fields populated."""
        runner = BenchmarkRunner(config_single)
        report = runner.run()
        successes = [r for r in report.results if r.success]
        assert len(successes) > 0
        for r in successes:
            assert r.algorithm != ""
            assert r.dataset != ""
            assert r.expert_key == "zoo"
            assert r.execution_time_ms > 0
            assert not np.isnan(r.silhouette)  # blobs should have valid silhouette

    def test_runner_moons_dbscan_outranks_kmeans(self, config_single: BenchmarkConfig) -> None:
        """On moons, DBSCAN ARI should exceed KMeans ARI (topology beats centroid)."""
        c = BenchmarkConfig(
            datasets=["moons"],
            experts=["zoo"],
            n_samples=200,
            offline_mode=True,
            output_dir=str(config_single.output_dir),
        )
        runner = BenchmarkRunner(c)
        report = runner.run()
        dbscan_ari = None
        kmeans_ari = None
        for r in report.results:
            if r.algorithm == "DBSCAN" and r.success:
                dbscan_ari = r.ari
            if r.algorithm == "KMeans" and r.success:
                kmeans_ari = r.ari
        if dbscan_ari is not None and kmeans_ari is not None:
            assert dbscan_ari > kmeans_ari, (
                f"DBSCAN ARI ({dbscan_ari:.3f}) should beat KMeans ({kmeans_ari:.3f}) on moons"
            )

    def test_runner_summary_structure(self, config: BenchmarkConfig) -> None:
        """Report summary has per_dataset, per_expert, overall keys."""
        runner = BenchmarkRunner(config)
        report = runner.run()
        s = report.summary
        assert "per_dataset" in s
        assert "per_expert" in s
        assert "overall" in s
        for ds_name in config.datasets:
            assert ds_name in s["per_dataset"]
        assert "zoo" in s["per_expert"]
        ov = s["overall"]
        assert ov["total_runs"] > 0
        assert 0.0 <= ov["success_rate"] <= 1.0
        assert ov["total_cost_usd"] >= 0.0

    def test_runner_handles_invalid_dataset(self, tmp_path: Path) -> None:
        """Invalid dataset name produces error result, does not crash."""
        c = BenchmarkConfig(
            datasets=["nonexistent_dataset"],
            experts=["zoo"],
            n_samples=60,
            offline_mode=True,
            output_dir=str(tmp_path),
        )
        runner = BenchmarkRunner(c)
        report = runner.run()
        assert len(report.results) >= 1
        assert any("load_failed" in r.algorithm for r in report.results)


# ======================================================================
# BenchmarkReporter
# ======================================================================

def _make_sample_report() -> BenchmarkReport:
    return BenchmarkReport(
        benchmark_version="1.0",
        timestamp="2026-04-28T00:00:00Z",
        config={"datasets": ["blobs", "moons"], "experts": ["zoo"], "offline_mode": True},
        results=[
            BenchmarkRunResult(
                dataset="blobs", expert_key="zoo", algorithm="KMeans",
                ari=0.95, silhouette=0.72, calinski_harabasz=120.0, davies_bouldin=0.45,
                score=0.95, score_source="ari", success=True, execution_time_ms=1200,
            ),
            BenchmarkRunResult(
                dataset="blobs", expert_key="zoo", algorithm="DBSCAN",
                ari=0.88, silhouette=0.65, calinski_harabasz=90.0, davies_bouldin=0.55,
                score=0.88, score_source="ari", success=True, execution_time_ms=900,
            ),
            BenchmarkRunResult(
                dataset="moons", expert_key="zoo", algorithm="DBSCAN",
                ari=1.0, silhouette=0.30, calinski_harabasz=30.0, davies_bouldin=1.2,
                score=1.0, score_source="ari", success=True, execution_time_ms=1100,
            ),
            BenchmarkRunResult(
                dataset="moons", expert_key="zoo", algorithm="KMeans",
                ari=0.45, silhouette=0.55, calinski_harabasz=80.0, davies_bouldin=0.60,
                score=0.55, score_source="silhouette", success=True, execution_time_ms=800,
            ),
            BenchmarkRunResult(
                dataset="blobs", expert_key="zoo", algorithm="(all_failed)",
                success=False, error_message="Sandbox timeout",
            ),
        ],
        summary={},  # filled below
    )


class TestBenchmarkReporter:
    """Tests for BenchmarkReporter."""

    def test_write_json_creates_file(self, tmp_path: Path) -> None:
        """JSON output file is created with correct content."""
        report = _make_sample_report()
        path = tmp_path / "benchmark_test.json"
        BenchmarkReporter.write_json(report, str(path))
        assert path.exists()
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        assert data["benchmark_version"] == "1.0"
        assert data["timestamp"] == "2026-04-28T00:00:00Z"
        assert len(data["results"]) == 5
        assert "summary" in data

    def test_json_results_have_required_fields(self, tmp_path: Path) -> None:
        """Each result in JSON has all expected keys."""
        report = _make_sample_report()
        path = tmp_path / "benchmark_fields.json"
        BenchmarkReporter.write_json(report, str(path))
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        for r in data["results"]:
            for key in ["dataset", "expert_key", "algorithm", "success", "execution_time_ms"]:
                assert key in r, f"Missing key '{key}' in result"

    def test_write_json_creates_parent_dirs(self, tmp_path: Path) -> None:
        """Nested output path creates intermediate directories."""
        path = tmp_path / "sub" / "deep" / "benchmark.json"
        BenchmarkReporter.write_json(_make_sample_report(), str(path))
        assert path.exists()

    def test_exit_code_zero_when_high_success(self) -> None:
        """Exit 0 when success rate is above threshold."""
        report = _make_sample_report()
        report.summary = {"overall": {"success_rate": 0.80}}
        assert BenchmarkReporter.compute_exit_code(report, 0.75) == 0

    def test_exit_code_one_when_below_threshold(self) -> None:
        """Exit 1 when success rate is below threshold."""
        report = _make_sample_report()
        # 2 of 5 = 40%
        report.results[0].success = True
        report.results[1].success = True
        report.results[2].success = False
        report.results[3].success = False
        report.results[4].success = False
        assert BenchmarkReporter.compute_exit_code(report, 0.80) == 1

    def test_exit_code_edge_threshold(self) -> None:
        """Exactly at threshold passes."""
        report = _make_sample_report()
        report.summary = {"overall": {"success_rate": 2 / 3}}
        assert BenchmarkReporter.compute_exit_code(report, 0.66) == 0

    def test_print_summary_does_not_crash(self, capsys) -> None:
        """print_summary outputs text without exception."""
        report = _make_sample_report()
        BenchmarkReporter.print_summary(report)
        captured = capsys.readouterr()
        assert "ACE Agent Benchmark Report" in captured.out


# ======================================================================
# BenchmarkReport aggregation
# ======================================================================

class TestBenchmarkReportAggregation:
    """Tests for _aggregate() logic."""

    def test_aggregate_counts(self) -> None:
        """Aggregation correctly counts success/failure."""
        runner = BenchmarkRunner(BenchmarkConfig(offline_mode=True))
        runner._results = [
            BenchmarkRunResult(dataset="a", expert_key="z", algorithm="KMeans", success=True),
            BenchmarkRunResult(dataset="a", expert_key="z", algorithm="DBSCAN", success=True),
            BenchmarkRunResult(dataset="a", expert_key="z", algorithm="(all_failed)", success=False),
            BenchmarkRunResult(dataset="b", expert_key="z", algorithm="KMeans", success=True),
        ]
        summary = runner._aggregate()
        assert summary["overall"]["total_runs"] == 4
        assert summary["overall"]["successful"] == 3
        assert summary["overall"]["success_rate"] == 0.75

    def test_aggregate_per_dataset(self) -> None:
        """Per-dataset metrics are correct."""
        runner = BenchmarkRunner(BenchmarkConfig(offline_mode=True))
        runner._results = [
            BenchmarkRunResult(dataset="a", expert_key="z", algorithm="KMeans",
                               score=0.8, ari=0.9, silhouette=0.7, success=True, execution_time_ms=1000),
            BenchmarkRunResult(dataset="a", expert_key="z", algorithm="DBSCAN",
                               score=0.6, ari=0.7, silhouette=0.5, success=True, execution_time_ms=2000),
            BenchmarkRunResult(dataset="b", expert_key="z", algorithm="KMeans",
                               score=0.5, success=True, execution_time_ms=500),
        ]
        summary = runner._aggregate()
        ds_a = summary["per_dataset"]["a"]
        assert ds_a["algorithms_run"] == 2
        assert ds_a["successful"] == 2
        assert ds_a["avg_ari"] == pytest.approx(0.8, 0.01)


# ======================================================================
# BenchmarkRunner with mocked LLM experts
# ======================================================================

class TestBenchmarkWithLLMExperts:
    """Tests with mocked LLM calls for centroid/topology experts."""

    @pytest.fixture()
    def offline_later_config(self, tmp_path: Path) -> BenchmarkConfig:
        return BenchmarkConfig(
            datasets=["moons"],
            experts=["zoo", "centroid"],
            n_samples=60,
            offline_mode=False,
            llm_settings=LLMSettings(enabled=True, base_url="http://x", api_key="x", model="x"),
            output_dir=str(tmp_path),
        )

    def test_centroid_dropped_in_offline(self, tmp_path: Path) -> None:
        """Offline mode drops centroid expert."""
        c = BenchmarkConfig(
            datasets=["moons"], experts=["zoo", "centroid"],
            n_samples=60, offline_mode=True, output_dir=str(tmp_path),
        )
        runner = BenchmarkRunner(c)
        resolved = runner._resolve_experts()
        assert "centroid" not in resolved
        assert "zoo" in resolved


# ======================================================================
# __main__ CLI
# ======================================================================

class TestCLI:
    """Tests for CLI entry point."""

    def test_parse_offline_run(self, tmp_path: Path, monkeypatch) -> None:
        """CLI --offline --output tmp_path exits 0."""
        from ACE_Agent.benchmark.__main__ import main
        output = tmp_path / "cli_test.json"
        # Ensure trace path exists
        (tmp_path / "llm_trace.jsonl").touch()
        rc = main([
            "--offline",
            "--datasets", "blobs",
            "--experts", "zoo",
            "--n-samples", "40",
            "--output", str(output),
        ])
        assert rc == 0
        assert output.exists()

    def test_invalid_dataset_exits(self, tmp_path: Path, monkeypatch) -> None:
        """CLI with invalid dataset exits gracefully."""
        from ACE_Agent.benchmark.__main__ import main
        output = tmp_path / "cli_invalid.json"
        (tmp_path / "llm_trace.jsonl").touch()
        rc = main([
            "--offline",
            "--datasets", "xyz_invalid",
            "--experts", "zoo",
            "--n-samples", "20",
            "--min-success-rate", "0.0",
            "--output", str(output),
        ])
        assert rc == 0
        assert output.exists()


# ======================================================================
# CriticExpert (Phase 1 — independent auditor)
# ======================================================================

_VALID_AUDIT_CODE = r"""
import warnings
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.metrics import (silhouette_score, davies_bouldin_score,
    calinski_harabasz_score, adjusted_rand_score)
from sklearn.model_selection import ShuffleSplit
warnings.filterwarnings('ignore')

X_scaled = StandardScaler().fit_transform(X)
n = len(X_scaled)

# Hopkins
n_sample = min(50, n // 10)
nbrs = NearestNeighbors(n_neighbors=2).fit(X_scaled)
rand_pts = np.random.uniform(X_scaled.min(axis=0), X_scaled.max(axis=0), size=(n_sample, X_scaled.shape[1]))
d_real, _ = nbrs.kneighbors(X_scaled[np.random.choice(n, n_sample, replace=False)])
d_rand, _ = nbrs.kneighbors(rand_pts)
hopkins = np.sum(d_real) / (np.sum(d_real) + np.sum(d_rand))
tendency = "strong" if hopkins > 0.7 else ("moderate" if hopkins > 0.5 else "weak")

max_k = min(15, int(np.sqrt(n)))
k_range = list(range(2, max_k + 1))
sil_scores, dbi_scores, chi_scores = [], [], []
for k in k_range:
    labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(X_scaled)
    sil_scores.append(silhouette_score(X_scaled, labels))
    dbi_scores.append(davies_bouldin_score(X_scaled, labels))
    chi_scores.append(calinski_harabasz_score(X_scaled, labels))
best_k_sil = k_range[int(np.argmax(sil_scores))]
best_k_dbi = k_range[int(np.argmin(dbi_scores))]
best_k_chi = k_range[int(np.argmax(chi_scores))]
k_consensus = int(np.median([best_k_sil, best_k_dbi, best_k_chi]))

rs = ShuffleSplit(n_splits=10, test_size=0.2, random_state=42)
aris = []
for train_idx, _ in rs.split(X_scaled):
    X_train = X_scaled[train_idx]
    prev_labels = None
    for k in range(2, 9):
        labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(X_train)
        if prev_labels is not None:
            aris.append(adjusted_rand_score(prev_labels[:len(labels)], labels))
        prev_labels = labels
stability = float(np.clip(np.mean(aris) if aris else 0.5, 0, 1))

if hopkins > 0.7 and stability > 0.7:
    grade = "excellent"
elif hopkins > 0.5 and stability > 0.5:
    grade = "good"
elif hopkins > 0.3:
    grade = "fair"
else:
    grade = "poor"

score = float(np.clip((hopkins + stability) / 2, 0, 1))

artifacts['Critic_Audit'] = {
    'labels': [],
    'metrics': {
        'score': score,
        'score_source': 'critic_audit',
        'hopkins': round(float(hopkins), 4),
        'cluster_tendency': tendency,
        'stability_score': round(float(stability), 4),
        'best_k_silhouette': best_k_sil,
        'best_k_dbi': best_k_dbi,
        'best_k_chi': best_k_chi,
        'k_consensus': k_consensus,
        'audit_grade': grade,
        'recommendation': 'centroid' if stability > 0.6 else 'topology',
    },
    'plot_path': '',
}
"""


class TestCriticExpert:
    """Tests for CriticExpert (Phase 1)."""

    @pytest.fixture()
    def critic_expert(self):
        from ACE_Agent.expert_sub_agents.critic_expert import CriticExpert
        return CriticExpert()

    def test_is_base_expert_subclass(self, critic_expert) -> None:
        from ACE_Agent.expert_sub_agents.base import BaseExpert
        assert isinstance(critic_expert, BaseExpert)

    def test_key_and_label(self, critic_expert) -> None:
        assert critic_expert.key == "critic"
        assert len(critic_expert.label) > 0

    def test_requires_llm_true(self, critic_expert) -> None:
        assert critic_expert.REQUIRES_LLM is True

    def test_generate_code_returns_string(self, critic_expert) -> None:
        mock_client = MagicMock()
        mock_client.chat_completion.return_value = _VALID_AUDIT_CODE
        dataset = generate_dataset("blobs", n_samples=60)
        code = critic_expert._generate_code(mock_client, dataset, "test")
        assert isinstance(code, str)
        assert len(code) > 0

    def test_generated_code_is_syntactically_valid(self, critic_expert) -> None:
        import ast
        ast.parse(_VALID_AUDIT_CODE)

    def test_audit_code_executes_in_sandbox(self, critic_expert) -> None:
        dataset = generate_dataset("blobs", n_samples=60)
        result = critic_expert.sandbox.execute(_VALID_AUDIT_CODE, dataset.X, dataset.y)
        assert result["success"] is True
        assert "Critic_Audit" in result["artifacts"]
        audit = result["artifacts"]["Critic_Audit"]
        m = audit["metrics"]
        assert m["hopkins"] > 0
        assert m["stability_score"] > 0
        assert m["k_consensus"] >= 2
        assert m["audit_grade"] in ("excellent", "good", "fair", "poor")
        assert m["recommendation"] in ("centroid", "topology", "hybrid")
        assert m["score_source"] == "critic_audit"

    def test_critic_in_registry(self) -> None:
        from ACE_Agent.expert_sub_agents import build_expert_registry
        assert "critic" in build_expert_registry()


# ======================================================================
# DimensionExpert (Phase 3 — skeleton + LLM decision model)
# ======================================================================

_VALID_DIM_DECISION_JSON = """\
{"pipelines": {
    "pca_kmeans":  {"active": true, "n_components": 10, "k": 3},
    "pca_gmm":     {"active": true, "n_components": 10, "k": 3},
    "umap_kmeans": {"active": false, "n_components": 2, "n_neighbors": 15, "k": 3},
    "tsne_kmeans": {"active": false, "k": 3},
    "ae_kmeans":   {"active": false, "latent_dim": 4, "epochs": 10, "k": 3}
}}
"""


class TestDimensionExpert:
    """Tests for DimensionExpert (Phase 3 skeleton + LLM decision)."""

    @pytest.fixture()
    def dim_expert(self):
        from ACE_Agent.expert_sub_agents.dimension_expert import DimensionExpert
        return DimensionExpert()

    def test_is_base_expert_subclass(self, dim_expert) -> None:
        from ACE_Agent.expert_sub_agents.base import BaseExpert
        assert isinstance(dim_expert, BaseExpert)

    def test_key_and_label(self, dim_expert) -> None:
        assert dim_expert.key == "dimension"
        assert len(dim_expert.label) > 0

    def test_requires_llm_true(self, dim_expert) -> None:
        assert dim_expert.REQUIRES_LLM is True

    def test_generate_code_returns_string(self, dim_expert) -> None:
        mock_client = MagicMock()
        mock_client.chat_completion.return_value = _VALID_DIM_DECISION_JSON
        dataset = generate_dataset("high_dim", n_samples=60)
        code = dim_expert._generate_code(mock_client, dataset, "test")
        assert isinstance(code, str)
        assert len(code) > 0
        # Skeleton uses CTX_DATA / pre-injected sklearn, not bare X + import
        assert "CTX_DATA" in code
        # JSON decisions are injected into the skeleton
        assert "DECISIONS" in code

    def test_generated_code_is_syntactically_valid(self, dim_expert) -> None:
        """The skeleton + mock decision must parse as valid Python."""
        import ast
        mock_client = MagicMock()
        mock_client.chat_completion.return_value = _VALID_DIM_DECISION_JSON
        dataset = generate_dataset("high_dim", n_samples=60)
        code = dim_expert._generate_code(mock_client, dataset, "test")
        ast.parse(code)

    def test_dim_code_executes_in_sandbox(self, dim_expert) -> None:
        """Generated skeleton code produces PCA pipelines in sandbox."""
        mock_client = MagicMock()
        mock_client.chat_completion.return_value = _VALID_DIM_DECISION_JSON
        dataset = generate_dataset("high_dim", n_samples=60)
        code = dim_expert._generate_code(mock_client, dataset, "test")
        result = dim_expert.sandbox.execute(
            code, dataset.X, dataset.y,
            pre_inject=dim_expert.PRE_INJECT or None,
            display_name=dataset.display_name,
            expected_clusters=3,
        )
        assert result["success"] is True
        assert "PCA_KMeans" in result["artifacts"]
        assert "PCA_GMM" in result["artifacts"]
        km = result["artifacts"]["PCA_KMeans"]["metrics"]
        assert km["score"] > 0
        gmm = result["artifacts"]["PCA_GMM"]["metrics"]
        assert gmm["score"] > 0

    def test_dim_code_on_blobs_with_high_dim(self, dim_expert) -> None:
        """On 100-dim blobs, PCA pipelines should produce valid results."""
        mock_client = MagicMock()
        mock_client.chat_completion.return_value = _VALID_DIM_DECISION_JSON
        dataset = generate_dataset("high_dim", n_samples=80)
        code = dim_expert._generate_code(mock_client, dataset, "test")
        result = dim_expert.sandbox.execute(
            code, dataset.X, dataset.y,
            pre_inject=dim_expert.PRE_INJECT or None,
            display_name=dataset.display_name,
            expected_clusters=3,
        )
        assert result["success"] is True
        assert "PCA_KMeans" in result["artifacts"]
        sil = result["artifacts"]["PCA_KMeans"]["metrics"]["silhouette"]
        assert sil > 0  # PCA pipelines should produce meaningful clustering

    def test_smart_defaults_kick_in_on_bad_llm_output(self, dim_expert) -> None:
        """When LLM returns garbage, smart defaults produce valid code."""
        mock_client = MagicMock()
        mock_client.chat_completion.return_value = "garbage not json"
        dataset = generate_dataset("high_dim", n_samples=60)
        code = dim_expert._generate_code(mock_client, dataset, "test")
        assert "DECISIONS" in code
        # Code must be valid Python even with garbage LLM input
        import ast
        ast.parse(code)

    def test_ae_pipeline_pre_injected(self, dim_expert) -> None:
        """PRE_INJECT carries ae_kmeans_pipeline when torch is available."""
        if dim_expert.PRE_INJECT:
            assert "ae_kmeans_pipeline" in dim_expert.PRE_INJECT
            assert callable(dim_expert.PRE_INJECT["ae_kmeans_pipeline"])

    def test_dimension_in_registry(self) -> None:
        from ACE_Agent.expert_sub_agents import build_expert_registry
        assert "dimension" in build_expert_registry()
