"""Benchmark runner: orchestrates experts across datasets and collects metrics."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from ACE_Agent.agent_core.schemas import AlgorithmRunResult
from ACE_Agent.benchmark.config import BenchmarkConfig
from ACE_Agent.benchmark.metrics import ClusteringMetricsCalculator
from ACE_Agent.expert_sub_agents import build_expert_registry
from ACE_Agent.tools.data_factory import generate_dataset
from ACE_Agent.tools.llm_client import LLMSettings


@dataclass
class BenchmarkRunResult:
    """Per-algorithm result within a benchmark run."""

    dataset: str
    expert_key: str
    algorithm: str
    ari: float = float("nan")
    silhouette: float = float("nan")
    calinski_harabasz: float = float("nan")
    davies_bouldin: float = float("nan")
    score: float = 0.0
    score_source: str = ""
    retries_used: int = 0
    execution_time_ms: int = 0
    tokens_prompt: int = 0
    tokens_completion: int = 0
    estimated_cost_usd: float = 0.0
    success: bool = False
    error_message: str | None = None


@dataclass
class BenchmarkReport:
    """Complete benchmark report."""

    benchmark_version: str = "1.0"
    timestamp: str = ""
    config: dict[str, Any] = field(default_factory=dict)
    results: list[BenchmarkRunResult] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)


class BenchmarkRunner:
    """Runs experts against datasets and collects performance metrics."""

    _TRACE_REL_PATH = Path("outputs") / "llm_trace.jsonl"

    def __init__(self, config: BenchmarkConfig) -> None:
        self.config = config
        self._registry = build_expert_registry()
        self._results: list[BenchmarkRunResult] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> BenchmarkReport:
        """Execute full benchmark suite. Returns structured report."""
        self._results = []
        experts = self._resolve_experts()

        for ds_name in self.config.datasets:
            try:
                dataset = generate_dataset(
                    ds_name,
                    n_samples=self.config.n_samples,
                    noise=self.config.noise,
                    random_state=self.config.random_state,
                )
            except Exception as exc:
                self._results.append(
                    BenchmarkRunResult(
                        dataset=ds_name,
                        expert_key="(none)",
                        algorithm="(load_failed)",
                        success=False,
                        error_message=f"Dataset load failed: {exc}",
                    )
                )
                continue

            # N×D cost-aware timeout (matching supervisor pre-flight gate)
            n = dataset.X.shape[0]
            d = dataset.X.shape[1]
            size_tag = "[HUGE]" if n > 50000 else ("[LARGE]" if n > 10000 else "[SMALL]")
            cost = n * d

            if cost >= 50_000_000:
                adaptive_timeout = 300
                print(f"  {size_tag} {dataset.display_name}: {n}×{d} (N×D={cost/1e6:.1f}M — Tier 3, 300s)")
            elif cost >= 10_000_000:
                adaptive_timeout = 300
                print(f"  {size_tag} {dataset.display_name}: {n}×{d} (N×D={cost/1e6:.1f}M — Tier 2, 300s)")
            elif cost >= 2_000_000:
                adaptive_timeout = 120
                print(f"  {size_tag} {dataset.display_name}: {n}×{d} (N×D={cost/1e6:.1f}M — Tier 1, 120s)")
            elif n > 10000:
                adaptive_timeout = min(n // 100, 180)
                print(f"  {size_tag} {dataset.display_name}: {n} samples × {d} features")
            else:
                adaptive_timeout = 0
                print(f"  {size_tag} {dataset.display_name}: {n} samples × {d} features")

            prompt = f"请分析 {dataset.display_name} 数据集，运行所有可用算法。"
            for expert_key in experts:
                expert = self._registry.get(expert_key)
                if expert is None:
                    continue
                if adaptive_timeout > 0:
                    sandbox = getattr(expert, "sandbox", None)
                    if sandbox is not None:
                        sandbox.timeout_sec = adaptive_timeout
                try:
                    results = self._run_expert_on_dataset(
                        expert_key,
                        expert,
                        dataset,
                        prompt,
                    )
                    self._results.extend(results)
                except Exception as exc:
                    self._results.append(
                        BenchmarkRunResult(
                            dataset=ds_name,
                            expert_key=expert_key,
                            algorithm="(expert_crashed)",
                            success=False,
                            error_message=f"Expert crashed: {exc}",
                        )
                    )

        return self._build_report()

    # ------------------------------------------------------------------
    # Expert resolution
    # ------------------------------------------------------------------

    def _resolve_experts(self) -> list[str]:
        """Filter expert list based on offline_mode and REQUIRES_LLM."""
        resolved: list[str] = []
        for key in self.config.experts:
            expert = self._registry.get(key)
            if expert is None:
                continue
            if self.config.offline_mode and getattr(expert, "REQUIRES_LLM", True):
                continue
            resolved.append(key)
        return resolved

    # ------------------------------------------------------------------
    # Per-expert execution
    # ------------------------------------------------------------------

    def _run_expert_on_dataset(
        self,
        expert_key: str,
        expert: Any,
        dataset: Any,
        prompt: str,
    ) -> list[BenchmarkRunResult]:
        settings = self._make_settings()
        trace_path = Path(self.config.output_dir) / "llm_trace.jsonl"
        trace_start = self._trace_size(trace_path)

        t0 = time.monotonic()
        algo_results: list[AlgorithmRunResult] = expert.execute_with_self_correction(
            dataset,
            prompt,
            settings,
        )
        elapsed_ms = int((time.monotonic() - t0) * 1000)

        logs = getattr(expert, "last_logs", [])
        heal_stats = ClusteringMetricsCalculator.compute_self_healing_stats(logs)
        cost_delta = self._extract_cost_delta(trace_path, trace_start)

        results: list[BenchmarkRunResult] = []
        for ar in algo_results:
            ext_metrics = ClusteringMetricsCalculator.compute_all(
                dataset.X,
                ar.labels,
                dataset.y,
            )
            results.append(
                BenchmarkRunResult(
                    dataset=dataset.name,
                    expert_key=expert_key,
                    algorithm=ar.algorithm_name,
                    ari=ext_metrics["ari"],
                    silhouette=ext_metrics["silhouette"],
                    calinski_harabasz=ext_metrics["calinski_harabasz"],
                    davies_bouldin=ext_metrics["davies_bouldin"],
                    score=float(ar.metrics.get("score") or 0.0),
                    score_source=str(ar.metrics.get("score_source", "")),
                    retries_used=heal_stats["attempts"],
                    execution_time_ms=elapsed_ms,
                    tokens_prompt=cost_delta.get("prompt_tokens", 0),
                    tokens_completion=cost_delta.get("completion_tokens", 0),
                    estimated_cost_usd=cost_delta.get("cost_usd", 0.0),
                    success=True,
                )
            )

        if not algo_results and heal_stats.get("error"):
            results.append(
                BenchmarkRunResult(
                    dataset=dataset.name,
                    expert_key=expert_key,
                    algorithm="(all_failed)",
                    success=False,
                    error_message=heal_stats["error"],
                    execution_time_ms=elapsed_ms,
                    retries_used=heal_stats["attempts"],
                )
            )

        return results

    # ------------------------------------------------------------------
    # Cost isolation
    # ------------------------------------------------------------------

    @staticmethod
    def _trace_size(path: Path) -> int:
        try:
            return path.stat().st_size
        except OSError:
            return 0

    @staticmethod
    def _extract_cost_delta(trace_path: Path, start_byte: int) -> dict[str, Any]:
        """Read trace lines appended after start_byte; sum tokens and cost."""
        prompt_tokens = 0
        completion_tokens = 0
        cost_usd = 0.0
        try:
            with open(trace_path, encoding="utf-8") as fh:
                if start_byte > 0:
                    fh.seek(start_byte)
                for line in fh:
                    try:
                        record = json.loads(line.strip())
                        prompt_tokens += int(record.get("prompt_tokens", 0))
                        completion_tokens += int(record.get("completion_tokens", 0))
                        cost_usd += float(record.get("cost_usd", 0.0))
                    except (json.JSONDecodeError, ValueError, KeyError):
                        pass
        except OSError:
            pass
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "cost_usd": round(cost_usd, 6),
        }

    # ------------------------------------------------------------------
    # Settings
    # ------------------------------------------------------------------

    def _make_settings(self) -> LLMSettings:
        if self.config.llm_settings is not None:
            return self.config.llm_settings
        return LLMSettings(enabled=False)

    # ------------------------------------------------------------------
    # Aggregation and report building
    # ------------------------------------------------------------------

    def _build_report(self) -> BenchmarkReport:
        return BenchmarkReport(
            benchmark_version="1.0",
            timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            config={
                "datasets": self.config.datasets,
                "experts": self.config.experts,
                "offline_mode": self.config.offline_mode,
                "n_samples": self.config.n_samples,
                "random_state": self.config.random_state,
            },
            results=self._results,
            summary=self._aggregate(),
        )

    def _aggregate(self) -> dict[str, Any]:
        per_dataset: dict[str, dict[str, Any]] = {}
        per_expert: dict[str, dict[str, Any]] = {}

        for r in self._results:
            ds = per_dataset.setdefault(
                r.dataset,
                {
                    "algorithms_run": 0,
                    "successful": 0,
                    "total_score": 0.0,
                    "total_ari": 0.0,
                    "ari_count": 0,
                    "total_silhouette": 0.0,
                    "silhouette_count": 0,
                    "total_time_ms": 0,
                    "total_cost_usd": 0.0,
                },
            )
            ds["algorithms_run"] += 1
            if r.success:
                ds["successful"] += 1
                ds["total_score"] += r.score
                if not np.isnan(r.ari):
                    ds["total_ari"] += r.ari
                    ds["ari_count"] += 1
                if not np.isnan(r.silhouette):
                    ds["total_silhouette"] += r.silhouette
                    ds["silhouette_count"] += 1
            ds["total_time_ms"] += r.execution_time_ms
            ds["total_cost_usd"] += r.estimated_cost_usd

            ex = per_expert.setdefault(
                r.expert_key,
                {
                    "algorithms_run": 0,
                    "successful": 0,
                    "total_score": 0.0,
                    "total_retries": 0,
                    "total_time_ms": 0,
                    "total_cost_usd": 0.0,
                },
            )
            ex["algorithms_run"] += 1
            if r.success:
                ex["successful"] += 1
                ex["total_score"] += r.score
                ex["total_retries"] += r.retries_used
            ex["total_time_ms"] += r.execution_time_ms
            ex["total_cost_usd"] += r.estimated_cost_usd

        # Normalise
        for ds in per_dataset.values():
            n = max(ds["successful"], 1)
            ds["avg_score"] = round(ds["total_score"] / n, 4)
            ds["avg_ari"] = round(ds["total_ari"] / max(ds["ari_count"], 1), 4) if ds["ari_count"] > 0 else None
            ds["avg_silhouette"] = (
                round(ds["total_silhouette"] / max(ds["silhouette_count"], 1), 4)
                if ds["silhouette_count"] > 0
                else None
            )
            ds["success_rate"] = round(ds["successful"] / ds["algorithms_run"], 4) if ds["algorithms_run"] > 0 else 0.0

        for ex in per_expert.values():
            n = max(ex["successful"], 1)
            ex["avg_score"] = round(ex["total_score"] / n, 4)
            ex["avg_retries"] = round(ex["total_retries"] / n, 2)
            ex["avg_latency_ms"] = round(ex["total_time_ms"] / max(ex["algorithms_run"], 1), 0)
            ex["success_rate"] = round(ex["successful"] / ex["algorithms_run"], 4) if ex["algorithms_run"] > 0 else 0.0

        total_runs = len(self._results)
        total_success = sum(1 for r in self._results if r.success)
        total_cost = round(sum(r.estimated_cost_usd for r in self._results), 6)
        total_time = sum(r.execution_time_ms for r in self._results)

        return {
            "per_dataset": per_dataset,
            "per_expert": per_expert,
            "overall": {
                "total_runs": total_runs,
                "successful": total_success,
                "success_rate": round(total_success / total_runs, 4) if total_runs > 0 else 0.0,
                "total_cost_usd": total_cost,
                "total_time_ms": total_time,
            },
        }
