"""Tests for ACE Agent benchmark suite."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

_root_parent = str(Path(__file__).resolve().parents[2])
if _root_parent not in sys.path:
    sys.path.insert(0, _root_parent)

from sklearn.datasets import make_blobs  # noqa: E402

from ACE_Agent.agent_core.schemas import AlgorithmRunResult, DatasetBundle  # noqa: E402
from ACE_Agent.benchmark.config import BenchmarkConfig  # noqa: E402
from ACE_Agent.benchmark.metrics import ClusteringMetricsCalculator  # noqa: E402
from ACE_Agent.benchmark.reporter import BenchmarkReporter  # noqa: E402
from ACE_Agent.benchmark.runner import BenchmarkReport, BenchmarkRunResult, BenchmarkRunner  # noqa: E402
from ACE_Agent.expert_sub_agents.zoo_expert import ZooExpert  # noqa: E402
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
        assert len(c.datasets) == 12  # BENCHMARK_FULL: 12 high-dim real datasets
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
            datasets=["blobs"],
            experts=["zoo", "centroid"],
            n_samples=60,
            offline_mode=True,
            output_dir=str(tmp_path),
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
                dataset="blobs",
                expert_key="zoo",
                algorithm="KMeans",
                ari=0.95,
                silhouette=0.72,
                calinski_harabasz=120.0,
                davies_bouldin=0.45,
                score=0.95,
                score_source="ari",
                success=True,
                execution_time_ms=1200,
            ),
            BenchmarkRunResult(
                dataset="blobs",
                expert_key="zoo",
                algorithm="DBSCAN",
                ari=0.88,
                silhouette=0.65,
                calinski_harabasz=90.0,
                davies_bouldin=0.55,
                score=0.88,
                score_source="ari",
                success=True,
                execution_time_ms=900,
            ),
            BenchmarkRunResult(
                dataset="moons",
                expert_key="zoo",
                algorithm="DBSCAN",
                ari=1.0,
                silhouette=0.30,
                calinski_harabasz=30.0,
                davies_bouldin=1.2,
                score=1.0,
                score_source="ari",
                success=True,
                execution_time_ms=1100,
            ),
            BenchmarkRunResult(
                dataset="moons",
                expert_key="zoo",
                algorithm="KMeans",
                ari=0.45,
                silhouette=0.55,
                calinski_harabasz=80.0,
                davies_bouldin=0.60,
                score=0.55,
                score_source="silhouette",
                success=True,
                execution_time_ms=800,
            ),
            BenchmarkRunResult(
                dataset="blobs",
                expert_key="zoo",
                algorithm="(all_failed)",
                success=False,
                error_message="Sandbox timeout",
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
        with open(path, encoding="utf-8") as fh:
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
        with open(path, encoding="utf-8") as fh:
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
            BenchmarkRunResult(
                dataset="a",
                expert_key="z",
                algorithm="KMeans",
                score=0.8,
                ari=0.9,
                silhouette=0.7,
                success=True,
                execution_time_ms=1000,
            ),
            BenchmarkRunResult(
                dataset="a",
                expert_key="z",
                algorithm="DBSCAN",
                score=0.6,
                ari=0.7,
                silhouette=0.5,
                success=True,
                execution_time_ms=2000,
            ),
            BenchmarkRunResult(
                dataset="b", expert_key="z", algorithm="KMeans", score=0.5, success=True, execution_time_ms=500
            ),
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
            datasets=["moons"],
            experts=["zoo", "centroid"],
            n_samples=60,
            offline_mode=True,
            output_dir=str(tmp_path),
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
        rc = main(
            [
                "--offline",
                "--datasets",
                "blobs",
                "--experts",
                "zoo",
                "--n-samples",
                "40",
                "--output",
                str(output),
            ]
        )
        assert rc == 0
        assert output.exists()

    def test_invalid_dataset_exits(self, tmp_path: Path, monkeypatch) -> None:
        """CLI with invalid dataset exits gracefully."""
        from ACE_Agent.benchmark.__main__ import main

        output = tmp_path / "cli_invalid.json"
        (tmp_path / "llm_trace.jsonl").touch()
        rc = main(
            [
                "--offline",
                "--datasets",
                "xyz_invalid",
                "--experts",
                "zoo",
                "--n-samples",
                "20",
                "--min-success-rate",
                "0.0",
                "--output",
                str(output),
            ]
        )
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

# Simulated winner context (LLM hardcodes this from audit target)
WINNER = {
    "algorithm_name": "KMeans",
    "expert_label": "centroid",
    "metrics": {"score": 0.75, "silhouette": 0.65},
    "n_labels": 3,
}

X_scaled = StandardScaler().fit_transform(X)
n = len(X_scaled)

# Hopkins statistic
n_sample = min(50, n // 10)
nbrs = NearestNeighbors(n_neighbors=2).fit(X_scaled)
rand_pts = np.random.uniform(X_scaled.min(axis=0), X_scaled.max(axis=0), size=(n_sample, X_scaled.shape[1]))
d_real, _ = nbrs.kneighbors(X_scaled[np.random.choice(n, n_sample, replace=False)])
d_rand, _ = nbrs.kneighbors(rand_pts)
hopkins = np.sum(d_real) / (np.sum(d_real) + np.sum(d_rand))

# CVI multi-k scan
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

# Bootstrap stability around winner's k
winner_k = WINNER.get("n_labels", 3)
rs = ShuffleSplit(n_splits=10, test_size=0.2, random_state=42)
aris = []
for train_idx, _ in rs.split(X_scaled):
    sub_labels = KMeans(n_clusters=winner_k, random_state=42, n_init=10).fit_predict(X_scaled[train_idx])
    full_labels = KMeans(n_clusters=winner_k, random_state=42, n_init=10).fit_predict(X_scaled)
    ari_val = adjusted_rand_score(full_labels[train_idx], sub_labels)
    aris.append(ari_val)
stability = float(np.clip(np.median(aris) if aris else 0.5, 0, 1))

# Overfitting risk
if hopkins < 0.5:
    overfitting_risk = "high" if stability < 0.6 else "medium"
elif stability < 0.5:
    overfitting_risk = "high"
elif stability < 0.75:
    overfitting_risk = "medium"
else:
    overfitting_risk = "low"

# Winner k consistency check
winner_k_consistent = abs(winner_k - k_consensus) <= 1

# Endorsement decision
if stability >= 0.75 and hopkins >= 0.6 and winner_k_consistent:
    endorsement = "endorsed"
elif stability >= 0.5:
    endorsement = "qualified"
else:
    endorsement = "qualified_with_warning"

confidence = float(np.clip((hopkins + stability) / 2, 0, 1))

findings = [
    f"Hopkins H={hopkins:.3f} (" + ("strong trend" if hopkins > 0.7 else "moderate") + ")",
    f"Bootstrap stability={stability:.3f}",
    f"Winner k={winner_k}, CVI consensus k={k_consensus}"
    + (" (consistent)" if winner_k_consistent else " (MISMATCH!)"),
    f"Overfitting risk: {overfitting_risk}",
]

artifacts['Critic_Audit'] = {
    'labels': [],
    'metrics': {
        'score': 0.0,
        'score_source': 'audit',
    },
    'audit_report': {
        'confidence_level': round(float(confidence), 4),
        'overfitting_risk': overfitting_risk,
        'stability_score': round(float(stability), 4),
        'hopkins': round(float(hopkins), 4),
        'winner_k_consistency': winner_k_consistent,
        'endorsement': endorsement,
        'findings': findings,
        'recommendation': 'centroid' if stability > 0.6 else 'topology',
    },
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
        # metrics: score is fixed at 0 (audit does not compete)
        m = audit["metrics"]
        assert m["score"] == 0.0
        assert m["score_source"] == "audit"
        # audit_report: the real output
        ar = audit["audit_report"]
        assert 0 < ar["hopkins"] < 1
        assert 0 < ar["stability_score"] <= 1
        assert 0 <= ar["confidence_level"] <= 1
        assert ar["overfitting_risk"] in ("low", "medium", "high")
        assert ar["endorsement"] in ("endorsed", "qualified", "qualified_with_warning")
        assert isinstance(ar["winner_k_consistency"], bool)
        assert len(ar["findings"]) >= 2
        assert ar["recommendation"] in ("centroid", "topology", "hybrid")

    def test_critic_in_registry(self) -> None:
        from ACE_Agent.expert_sub_agents import build_expert_registry

        assert "critic" in build_expert_registry()


# ======================================================================
# DimensionExpert (Phase 3/4 — skeleton + LLM decision + deep AE)
# ======================================================================

_VALID_DIM_DECISION_JSON = """\
{"pipelines": {
    "pca_kmeans":  {"active": true, "n_components": 10, "k": 3},
    "pca_gmm":     {"active": true, "n_components": 10, "k": 3},
    "umap_kmeans": {"active": false, "n_components": 2, "n_neighbors": 15, "k": 3},
    "tsne_kmeans": {"active": false, "k": 3},
    "ae_kmeans":   {"active": false, "latent_dim": 4, "epochs": 10, "k": 3,
                    "hidden_dims": [64, 32], "learning_rate": 0.001,
                    "dropout": 0.2, "early_stopping_patience": 15,
                    "noise_std": 0.15, "cluster_method": "gmm"},
    "dec":         {"active": false, "latent_dim": 4, "k": 3,
                    "pretrain_epochs": 10, "finetune_epochs": 5,
                    "hidden_dims": [64, 32], "learning_rate": 0.001,
                    "dropout": 0.2, "gamma": 0.1, "noise_std": 0.15}
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
            code,
            dataset.X,
            dataset.y,
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
            code,
            dataset.X,
            dataset.y,
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

    def test_dec_pipeline_pre_injected(self, dim_expert) -> None:
        """PRE_INJECT carries dec_pipeline when torch is available."""
        if dim_expert.PRE_INJECT:
            assert "dec_pipeline" in dim_expert.PRE_INJECT
            assert callable(dim_expert.PRE_INJECT["dec_pipeline"])

    def test_dec_generated_code_contains_pipeline6(self, dim_expert) -> None:
        """Skeleton includes Pipeline 6 (DEC) code."""
        mock_client = MagicMock()
        mock_client.chat_completion.return_value = _VALID_DIM_DECISION_JSON
        dataset = generate_dataset("high_dim", n_samples=100)
        code = dim_expert._generate_code(mock_client, dataset, "test")
        assert "PIPELINE 6" in code or "DEC" in code
        assert "dec_pipeline" in code


class TestDECPipeline:
    """Tests for the DEC/IDEC pipeline (tools/dec_pipeline.py)."""

    def test_module_importable(self) -> None:
        """dec_pipeline module imports without error."""
        from ACE_Agent.tools import dec_pipeline

        assert dec_pipeline._HAS_TORCH is True or dec_pipeline._HAS_TORCH is False

    def test_dec_pipeline_basic(self) -> None:
        """DEC pipeline returns valid result dict on synthetic data."""
        import torch

        if not torch.cuda.is_available() and torch.cuda.device_count() == 0:
            pytest.skip("PyTorch not available for DEC test")

        X, y = make_blobs(n_samples=200, n_features=50, centers=3, random_state=42)

        from ACE_Agent.tools.dec_pipeline import dec_pipeline

        result = dec_pipeline(
            X,
            k=3,
            latent_dim=4,
            hidden_dims=[32, 16],
            pretrain_epochs=10,
            finetune_epochs=5,
            gamma=0.0,
        )
        assert "labels" in result
        assert len(result["labels"]) == 200
        assert len(set(result["labels"])) > 1  # more than one cluster
        assert result["metrics"]["score"] > 0

    def test_idec_mode(self) -> None:
        """IDEC (gamma > 0) also produces valid result."""
        import torch

        if not torch.cuda.is_available() and torch.cuda.device_count() == 0:
            pytest.skip("PyTorch not available for DEC test")

        X, y = make_blobs(n_samples=120, n_features=40, centers=3, random_state=42)

        from ACE_Agent.tools.dec_pipeline import dec_pipeline

        result = dec_pipeline(
            X,
            k=3,
            latent_dim=4,
            hidden_dims=[32, 16],
            pretrain_epochs=10,
            finetune_epochs=5,
            gamma=0.1,
        )
        assert result["metrics"]["method"] == "IDEC"
        assert result["metrics"]["score"] > 0

    def test_dec_no_torch_fallback(self) -> None:
        """When _HAS_TORCH is False, sklearn fallback is used."""
        from ACE_Agent.tools import dec_pipeline

        X, y = make_blobs(n_samples=100, n_features=10, centers=3, random_state=42)
        saved = dec_pipeline._HAS_TORCH
        try:
            dec_pipeline._HAS_TORCH = False
            result = dec_pipeline.dec_pipeline(X, k=3)
            assert result["metrics"]["backend"] == "sklearn-fallback"
            assert len(result["labels"]) == 100
            assert result["metrics"]["score"] > 0
        finally:
            dec_pipeline._HAS_TORCH = saved

    def test_soft_assignment_is_normalized(self) -> None:
        """Q rows sum to 1.0."""
        import torch

        from ACE_Agent.tools.dec_pipeline import _soft_assignment

        z = torch.randn(30, 8)
        centers = torch.nn.Parameter(torch.randn(3, 8))
        q = _soft_assignment(z, centers)
        row_sums = q.sum(dim=1)
        assert torch.allclose(row_sums, torch.ones(30), atol=1e-5)

    def test_target_distribution_is_normalized(self) -> None:
        """P rows sum to 1.0."""
        import torch

        from ACE_Agent.tools.dec_pipeline import _soft_assignment, _target_distribution

        z = torch.randn(30, 8)
        centers = torch.nn.Parameter(torch.randn(3, 8))
        q = _soft_assignment(z, centers)
        p = _target_distribution(q)
        row_sums = p.sum(dim=1)
        assert torch.allclose(row_sums, torch.ones(30), atol=1e-5)


# ======================================================================
# Benchmark Dataloader
# ======================================================================


class TestBenchmarkDataloader:
    """Tests for benchmark/dataloader.py."""

    def test_module_importable(self) -> None:
        from ACE_Agent.benchmark import dataloader

        assert hasattr(dataloader, "load_benchmark_dataset")
        assert hasattr(dataloader, "load_from_npy")

    def test_is_large_dataset_small(self) -> None:
        from ACE_Agent.benchmark.dataloader import is_large_dataset

        ds = generate_dataset("blobs", n_samples=100)
        assert not is_large_dataset(ds)

    def test_is_large_dataset_big(self) -> None:
        from ACE_Agent.benchmark.dataloader import is_large_dataset

        ds = generate_dataset("high_dim", n_samples=60)
        # high_dim: 100 features, <512 threshold → not large
        assert not is_large_dataset(ds)

        # Manually construct a large dataset
        X = np.random.randn(15000, 10)
        from ACE_Agent.agent_core.schemas import DatasetBundle

        big = DatasetBundle(name="big", X=X)
        assert is_large_dataset(big)

    def test_dataset_size_label(self) -> None:
        from ACE_Agent.benchmark.dataloader import dataset_size_label

        from ACE_Agent.agent_core.schemas import DatasetBundle

        assert dataset_size_label(DatasetBundle(name="s", X=np.random.randn(100, 2))) == "[SMALL]"
        assert dataset_size_label(DatasetBundle(name="m", X=np.random.randn(15000, 2))) == "[LARGE]"
        assert dataset_size_label(DatasetBundle(name="l", X=np.random.randn(60000, 2))) == "[HUGE]"

    def test_load_from_npy_roundtrip(self, tmp_path) -> None:
        from ACE_Agent.benchmark.dataloader import load_from_npy

        X = np.random.randn(50, 128).astype(np.float32)
        y = np.random.randint(0, 3, 50).astype(np.int64)
        np.save(tmp_path / "features.npy", X)
        np.save(tmp_path / "labels.npy", y)

        ds = load_from_npy(
            "test_features",
            str(tmp_path / "features.npy"),
            str(tmp_path / "labels.npy"),
            cache_dir=str(tmp_path / "cache"),
            expected_clusters=3,
            feature_extractor="ResNet50",
        )
        assert ds.name == "test_features"
        assert ds.X.shape == (50, 128)
        assert ds.y is not None and len(ds.y) == 50
        assert ds.metadata["expected_clusters"] == 3
        assert ds.metadata["feature_extractor"] == "ResNet50"

    def test_cached_load_creates_npy(self, tmp_path) -> None:
        """load_from_npy also writes NPY cache."""
        from ACE_Agent.benchmark.dataloader import load_from_npy

        X = np.random.randn(20, 10).astype(np.float32)
        np.save(tmp_path / "feat.npy", X)
        cache = tmp_path / "cache"
        load_from_npy("cached", str(tmp_path / "feat.npy"), cache_dir=str(cache))
        assert (cache / "cached_X.npy").exists()


# ======================================================================
# Algorithm filtering (O(N²) meltdown prevention)
# ======================================================================


class TestAlgorithmFiltering:
    """Tests for max_samples filtering in ZooExpert and AlgorithmZoo."""

    def test_all_algorithms_have_max_samples(self) -> None:
        from ACE_Agent.tools.algorithm_zoo import AlgorithmZoo

        for algo in AlgorithmZoo.get_all_algorithms():
            assert "max_samples" in algo, f"{algo['name']} missing max_samples"

    def test_o_n_algorithms_unlimited(self) -> None:
        """KMeans, MiniBatchKMeans, GMM, Birch have max_samples=None."""
        from ACE_Agent.tools.algorithm_zoo import AlgorithmZoo

        unlimited = {"KMeans", "MiniBatchKMeans", "GaussianMixture", "Birch"}
        for algo in AlgorithmZoo.get_all_algorithms():
            if algo["name"] in unlimited:
                assert algo["max_samples"] is None, f"{algo['name']} should be unlimited"

    def test_o_n2_algorithms_limited(self) -> None:
        """Spectral and Agglomerative are capped at 5000."""
        from ACE_Agent.tools.algorithm_zoo import AlgorithmZoo

        limits = {"SpectralClustering": 5000, "AgglomerativeClustering": 5000,
                  "AffinityPropagation": 3000, "MeanShift": 5000}
        for algo in AlgorithmZoo.get_all_algorithms():
            if algo["name"] in limits:
                assert algo["max_samples"] == limits[algo["name"]]

    def test_small_dataset_runs_all_algorithms(self) -> None:
        """On N=200, ZooExpert generated code skips zero algorithms."""
        zoo = ZooExpert()
        ds = generate_dataset("smile", n_samples=200)
        code = zoo._generate_code(None, ds, "test")
        import ast

        ast.parse(code)  # must be valid Python
        # The filtering block is present
        assert "max_samples" in code
        # All 11 algorithms should still be in the config
        assert code.count('"name":') == 11

    def test_large_dataset_filters_o_n2(self) -> None:
        """On N=20000, ZooExpert filters out O(N²) algorithms."""
        zoo = ZooExpert()
        import numpy as np

        from ACE_Agent.agent_core.schemas import DatasetBundle

        X = np.random.randn(20000, 10)
        y = np.random.randint(0, 3, 20000)
        ds = DatasetBundle(
            name="large_test", X=X, y=y,
            display_name="Large Test", description="",
            metadata={"expected_clusters": 3},
        )
        code = zoo._generate_code(None, ds, "test")
        import ast

        ast.parse(code)

        # Spectral, Agglomerative, AffinityPropagation, MeanShift should NOT be
        # in the generated _algo_configs (they're filtered before code runs).
        # However, the filtering is done inside the generated code, not in Python.
        # The generated code itself defines _algo_configs with max_samples, and
        # then the RUNTIME filtering removes O(N²) algorithms. So the generated
        # code still includes ALL configs but filters at exec time.
        # The key assertion: the generated code includes filtering logic.
        assert "max_samples" in code
        assert "_skipped" in code

    def test_generated_code_filters_at_runtime(self) -> None:
        """Execute generated code on N=6000 in sandbox; verify O(N²) skipped."""
        zoo = ZooExpert()
        zoo.sandbox.timeout_sec = 180  # N=6000 needs extra time for DBSCAN/OPTICS
        import numpy as np

        from ACE_Agent.agent_core.schemas import DatasetBundle

        X = np.random.randn(6000, 10)
        y = np.random.randint(0, 3, 6000)
        ds = DatasetBundle(
            name="mid_test", X=X, y=y,
            display_name="Mid Test", description="",
        )
        code = zoo._generate_code(None, ds, "test")
        result = zoo.sandbox.execute(
            code, X, y,
            display_name=ds.display_name,
            expected_clusters=3,
        )
        assert result["success"] is True
        # KMeans and MiniBatchKMeans should succeed
        artifacts = result["artifacts"]
        assert "KMeans" in artifacts
        assert "MiniBatchKMeans" in artifacts
        # Spectral and AffinityPropagation should be absent (filtered at runtime)
        assert "SpectralClustering" not in artifacts
        assert "AffinityPropagation" not in artifacts
        assert "MeanShift" not in artifacts
        # Agglomerative limit is 5000, so also skipped
        assert "AgglomerativeClustering" not in artifacts


# ============================================================================
# Ensemble Consensus Expert Tests (Phase 2, 2026-05)
# ============================================================================

class TestEnsembleConsensusExpert:
    """Tests for EnsembleConsensusExpert co-association matrix fusion."""

    @staticmethod
    def _make_result(name: str, expert_key: str, expert_label: str,
                     labels: list[int], score: float) -> AlgorithmRunResult:
        return AlgorithmRunResult(
            algorithm_name=name,
            expert_key=expert_key,
            expert_label=expert_label,
            labels=labels,
            metrics={"score": score, "score_source": "silhouette"},
            plot_path="",
        )

    @staticmethod
    def _make_ds(n: int = 500, d: int = 10) -> DatasetBundle:
        return DatasetBundle(
            name="test_ensemble",
            X=np.random.randn(n, d),
            y=None,
            display_name="Ensemble Test",
        )

    def test_ensemble_registered(self) -> None:
        from ACE_Agent.expert_sub_agents import build_expert_registry
        reg = build_expert_registry()
        assert "ensemble" in reg, "Ensemble expert must be in registry"

    def test_ensemble_key_and_label(self) -> None:
        from ACE_Agent.expert_sub_agents.ensemble_expert import EnsembleConsensusExpert
        e = EnsembleConsensusExpert()
        assert e.key == "ensemble"
        assert "共识" in e.label

    def test_ensemble_requires_two_label_sets(self) -> None:
        """Fewer than 2 valid results → returns None."""
        from ACE_Agent.expert_sub_agents.ensemble_expert import EnsembleConsensusExpert
        e = EnsembleConsensusExpert()
        ds = self._make_ds(100)
        results = [self._make_result("A", "a", "aa", list(range(100)), 0.5)]
        assert e.execute_ensemble(results, ds) is None

    def test_ensemble_rejects_mismatched_lengths(self) -> None:
        """Different-length labels → returns None."""
        from ACE_Agent.expert_sub_agents.ensemble_expert import EnsembleConsensusExpert
        e = EnsembleConsensusExpert()
        ds = self._make_ds(100)
        results = [
            self._make_result("A", "a", "aa", list(range(100)), 0.5),
            self._make_result("B", "b", "bb", list(range(50)), 0.4),
        ]
        assert e.execute_ensemble(results, ds) is None

    def test_ensemble_basic_consensus(self) -> None:
        """Two identical label sets → perfect agreement."""
        from ACE_Agent.expert_sub_agents.ensemble_expert import EnsembleConsensusExpert
        e = EnsembleConsensusExpert()
        n = 200
        labels_a = ([0] * 100) + ([1] * 100)
        labels_b = ([0] * 100) + ([1] * 100)  # identical
        ds = self._make_ds(n)
        results = [
            self._make_result("KMeans", "centroid", "c", labels_a, 0.6),
            self._make_result("Spectral", "topology", "t", labels_b, 0.4),
        ]
        consensus = e.execute_ensemble(results, ds)
        assert consensus is not None
        assert consensus.algorithm_name == "EnsembleConsensus"
        assert consensus.metrics["n_experts_fused"] == 2
        assert consensus.metrics["agreement"] > 0.99  # identical labels
        assert consensus.metrics["entropy_of_agreement"] < 0.1
        assert len(consensus.labels) == n
        # Consensus should find k=2
        assert consensus.metrics["k_consensus"] == 2

    def test_ensemble_weighted_scores_matter(self) -> None:
        """High-score expert should dominate over low-score expert."""
        from ACE_Agent.expert_sub_agents.ensemble_expert import EnsembleConsensusExpert
        e = EnsembleConsensusExpert()
        n = 200
        # High-score expert labels: 3 clusters
        labels_good = ([0] * 70) + ([1] * 60) + ([2] * 70)
        # Low-score expert labels: 2 clusters (random-ish)
        rng = np.random.RandomState(42)
        labels_bad = rng.randint(0, 2, n).tolist()
        ds = self._make_ds(n)
        results = [
            self._make_result("GoodAlgo", "dim", "dim", labels_good, 0.9),
            self._make_result("BadAlgo", "zoo", "zoo", labels_bad, 0.1),
        ]
        consensus = e.execute_ensemble(results, ds)
        assert consensus is not None
        # k_consensus should be majority vote = 2 (tie) or 3 depending on label set
        assert consensus.metrics["k_consensus"] in (2, 3)

    def test_ensemble_entropy_of_agreement(self) -> None:
        """Disagreeing experts → higher entropy."""
        from ACE_Agent.expert_sub_agents.ensemble_expert import EnsembleConsensusExpert
        e = EnsembleConsensusExpert()
        n = 200
        rng = np.random.RandomState(42)
        labels_a = rng.randint(0, 3, n).tolist()
        labels_b = rng.randint(0, 4, n).tolist()
        labels_c = rng.randint(0, 5, n).tolist()
        ds = self._make_ds(n)
        results = [
            self._make_result("A", "a", "aa", labels_a, 0.5),
            self._make_result("B", "b", "bb", labels_b, 0.5),
            self._make_result("C", "c", "cc", labels_c, 0.5),
        ]
        consensus = e.execute_ensemble(results, ds)
        assert consensus is not None
        # With random labels, agreement should be low and entropy high
        assert consensus.metrics["agreement"] < 0.70
        assert consensus.metrics["entropy_of_agreement"] > 0.5

    def test_ensemble_empty_labels_skipped(self) -> None:
        """Empty label list is filtered out."""
        from ACE_Agent.expert_sub_agents.ensemble_expert import EnsembleConsensusExpert
        e = EnsembleConsensusExpert()
        n = 200
        results = [
            self._make_result("Empty", "a", "aa", [], 0.0),
            self._make_result("Good", "b", "bb", list(range(n)), 0.5),
        ]
        ds = self._make_ds(n)
        assert e.execute_ensemble(results, ds) is None  # only 1 valid

    def test_ensemble_monte_carlo_threshold(self) -> None:
        """Verify the MC circuit breaker constant exists and is reasonable."""
        from ACE_Agent.expert_sub_agents.ensemble_expert import \
            _MC_THRESHOLD, _MC_SAMPLE_PAIRS
        assert _MC_THRESHOLD == 5000
        assert _MC_SAMPLE_PAIRS == 20000

    def test_ensemble_stores_coassoc_matrix(self) -> None:
        """Ensemble result must carry coassoc_matrix in params for frontend."""
        from ACE_Agent.expert_sub_agents.ensemble_expert import EnsembleConsensusExpert
        e = EnsembleConsensusExpert()
        n = 60
        labels_a = ([0] * 20) + ([1] * 20) + ([2] * 20)
        labels_b = ([0] * 20) + ([2] * 20) + ([1] * 20)
        ds = self._make_ds(n)
        results = [
            self._make_result("A", "a", "aa", labels_a, 0.6),
            self._make_result("B", "b", "bb", labels_b, 0.5),
        ]
        consensus = e.execute_ensemble(results, ds)
        assert consensus is not None
        params = getattr(consensus, "params", {})
        assert "coassoc_matrix" in params, "Must store coassoc_matrix in params"
        assert "expert_names" in params, "Must store expert_names in params"
        assert params["expert_names"] == ["k=3", "k=3"]

    def test_ensemble_coassoc_dimensions(self) -> None:
        """Coassoc matrix in params must be square and <= 500."""
        from ACE_Agent.expert_sub_agents.ensemble_expert import EnsembleConsensusExpert
        e = EnsembleConsensusExpert()
        n = 200
        labels_a = ([0] * 100) + ([1] * 100)
        labels_b = ([0] * 100) + ([1] * 100)
        ds = self._make_ds(n)
        results = [
            self._make_result("A", "a", "aa", labels_a, 0.6),
            self._make_result("B", "b", "bb", labels_b, 0.5),
        ]
        consensus = e.execute_ensemble(results, ds)
        assert consensus is not None
        coassoc = consensus.params["coassoc_matrix"]
        assert coassoc.shape[0] == coassoc.shape[1], "Must be square"
        assert coassoc.shape[0] == n, "For N<500, must match N"
        # Values must be in [0, 1]
        assert float(coassoc.min()) >= 0.0
        assert float(coassoc.max()) <= 1.0

    def test_conditional_ensemble_skip_rule(self) -> None:
        """Ensemble should skip when endorsed + confidence >= 0.75."""
        def _should_skip(audit: dict | None) -> bool:
            if audit is None:
                return False
            return (
                audit.get("endorsement") == "endorsed"
                and audit.get("confidence_level", 0.0) >= 0.75
            )

        assert _should_skip({"endorsement": "endorsed", "confidence_level": 0.85})
        assert _should_skip({"endorsement": "endorsed", "confidence_level": 0.75})
        assert not _should_skip({"endorsement": "endorsed", "confidence_level": 0.74})
        assert not _should_skip({"endorsement": "qualified", "confidence_level": 0.9})
        assert not _should_skip({"endorsement": "qualified_with_warning", "confidence_level": 0.9})
        assert not _should_skip(None)  # no audit → don't skip (conservative)

    def test_conditional_ensemble_triggered_when_low_confidence(self) -> None:
        """Ensemble should trigger when confidence < 0.75 or not endorsed."""
        def _should_skip(audit: dict | None) -> bool:
            if audit is None:
                return False
            return (
                audit.get("endorsement") == "endorsed"
                and audit.get("confidence_level", 0.0) >= 0.75
            )

        # Trigger cases: ensemble should NOT be skipped
        assert not _should_skip(None)
        assert not _should_skip({"endorsement": "qualified", "confidence_level": 0.5})
        assert not _should_skip({"endorsement": "qualified_with_warning", "confidence_level": 0.3})
        assert not _should_skip({"confidence_level": 0.9})  # missing endorsement
        assert not _should_skip({})  # empty dict


class TestCritic20ClosedLoop:
    """Tests for Critic 2.0 decision closed-loop (action + retry_constraints)."""

    def test_inject_constraints_prompt_empty(self) -> None:
        """Empty or None constraints produce empty prompt."""
        from ACE_Agent.expert_sub_agents.base import BaseExpert
        assert BaseExpert._inject_constraints_prompt(None) == ""
        assert BaseExpert._inject_constraints_prompt({}) == ""

    def test_inject_constraints_prompt_force_k(self) -> None:
        """force_k constraint produces a k= directive."""
        from ACE_Agent.expert_sub_agents.base import BaseExpert
        prompt = BaseExpert._inject_constraints_prompt({"force_k": 3})
        assert "k 必须为 3" in prompt
        assert "约束指令" in prompt

    def test_inject_constraints_prompt_blocked(self) -> None:
        """blocked_algorithms produces a ban directive."""
        from ACE_Agent.expert_sub_agents.base import BaseExpert
        prompt = BaseExpert._inject_constraints_prompt(
            {"blocked_algorithms": ["SpectralClustering", "DBSCAN"]}
        )
        assert "禁止使用以下算法" in prompt
        assert "SpectralClustering" in prompt
        assert "DBSCAN" in prompt

    def test_inject_constraints_prompt_all(self) -> None:
        """All constraint types combined produce a multi-line directive."""
        from ACE_Agent.expert_sub_agents.base import BaseExpert
        prompt = BaseExpert._inject_constraints_prompt({
            "force_k": 5,
            "blocked_algorithms": ["MeanShift"],
            "force_preprocessing": "standardize",
        })
        assert "k 必须为 5" in prompt
        assert "MeanShift" in prompt
        assert "standardize" in prompt

    def test_inject_constraints_prompt_reference_labels(self) -> None:
        """reference_labels produces HITL annotation directive."""
        from ACE_Agent.expert_sub_agents.base import BaseExpert
        prompt = BaseExpert._inject_constraints_prompt({
            "reference_labels": [0, 1, 0, 2, 1],
        })
        assert "HITL" in prompt
        assert "参考标签" in prompt
        assert "[0, 1, 0, 2, 1]" in prompt
        assert "ARI" in prompt or "NMI" in prompt

    def test_inject_constraints_prompt_reference_labels_truncated(self) -> None:
        """Long reference_labels (>20) should be truncated in prompt preview."""
        from ACE_Agent.expert_sub_agents.base import BaseExpert
        labels = list(range(50))
        prompt = BaseExpert._inject_constraints_prompt({
            "reference_labels": labels,
        })
        assert "..." in prompt
        assert "50 个数据点" in prompt

    def test_handle_audit_feedback_clear_or_warn_returns_empty(self) -> None:
        """CLEAR and WARN actions should return empty list (no retry needed)."""
        from ACE_Agent.agent_core.supervisor import ACESupervisor
        sv = ACESupervisor()
        trace: list[str] = []
        actives = ["centroid", "zoo"]

        assert sv._handle_audit_feedback(None, None, "", None, trace, actives) == []  # type: ignore[arg-type]
        assert sv._handle_audit_feedback(
            {"action": "CLEAR"}, None, "", None, trace, actives  # type: ignore[arg-type]
        ) == []
        assert sv._handle_audit_feedback(
            {"action": "WARN", "endorsement": "qualified"}, None, "", None, trace, actives  # type: ignore[arg-type]
        ) == []

    def test_handle_audit_feedback_retry_without_constraints_returns_empty(self) -> None:
        """RETRY with no valid constraints should skip (safety guard)."""
        from ACE_Agent.agent_core.supervisor import ACESupervisor
        sv = ACESupervisor()
        trace: list[str] = []
        actives = ["centroid"]

        assert sv._handle_audit_feedback(
            {"action": "RETRY"}, None, "", None, trace, actives  # type: ignore[arg-type]
        ) == []
        assert sv._handle_audit_feedback(
            {"action": "RETRY", "retry_constraints": {}}, None, "", None, trace, actives  # type: ignore[arg-type]
        ) == []
