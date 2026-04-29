"""Benchmark reporter: JSON output, console summary, CI exit code."""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

from ACE_Agent.benchmark.runner import BenchmarkReport


class BenchmarkReporter:
    """Formats and outputs benchmark results."""

    @staticmethod
    def write_json(report: BenchmarkReport, path: str | Path) -> Path:
        """Write report as pretty-printed JSON. Creates parent dirs if needed."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        serializable = {
            "benchmark_version": report.benchmark_version,
            "timestamp": report.timestamp,
            "config": report.config,
            "results": [
                {
                    "dataset": r.dataset,
                    "expert_key": r.expert_key,
                    "algorithm": r.algorithm,
                    "ari": r.ari if r.ari == r.ari else None,
                    "silhouette": r.silhouette if r.silhouette == r.silhouette else None,
                    "calinski_harabasz": r.calinski_harabasz if r.calinski_harabasz == r.calinski_harabasz else None,
                    "davies_bouldin": r.davies_bouldin if r.davies_bouldin == r.davies_bouldin else None,
                    "score": r.score,
                    "score_source": r.score_source,
                    "retries_used": r.retries_used,
                    "execution_time_ms": r.execution_time_ms,
                    "tokens_prompt": r.tokens_prompt,
                    "tokens_completion": r.tokens_completion,
                    "estimated_cost_usd": r.estimated_cost_usd,
                    "success": r.success,
                    "error_message": r.error_message,
                }
                for r in report.results
            ],
            "summary": report.summary,
        }
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(serializable, fh, ensure_ascii=False, indent=2)
        return out

    @staticmethod
    def default_output_path(output_dir: str = "outputs") -> str:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        return os.path.join(output_dir, f"benchmark_{ts}.json")

    @staticmethod
    def print_summary(report: BenchmarkReport) -> None:
        """Print human-readable summary to stdout."""
        print("\n" + "=" * 72)
        print("  ACE Agent Benchmark Report")
        print("=" * 72)
        print(f"  Timestamp : {report.timestamp}")
        print(f"  Datasets  : {', '.join(report.config.get('datasets', []))}")
        print(f"  Experts   : {', '.join(report.config.get('experts', []))}")
        print(f"  Mode      : {'offline' if report.config.get('offline_mode') else 'online'}")
        ov = report.summary.get("overall", {})
        print(
            f"  Total runs: {ov.get('total_runs', 0)}  "
            f"Success rate: {ov.get('success_rate', 0):.1%}  "
            f"Cost: ${ov.get('total_cost_usd', 0):.4f}  "
            f"Time: {ov.get('total_time_ms', 0) / 1000:.1f}s"
        )
        print("-" * 72)

        pd = report.summary.get("per_dataset", {})
        if pd:
            print(f"  {'Dataset':<12} {'Algos':>5} {'Success':>8} {'AvgARI':>8} {'AvgSil':>8} {'AvgScore':>9}")
            print("  " + "-" * 56)
            for ds_name, ds in sorted(pd.items()):
                ari_str = f"{ds.get('avg_ari', 0):.4f}" if ds.get("avg_ari") is not None else "    N/A"
                print(
                    f"  {ds_name:<12} {ds.get('algorithms_run', 0):>5} "
                    f"{ds.get('success_rate', 0.0):>7.1%} {ari_str:>8} "
                    f"{ds.get('avg_silhouette', 0) or 0.0:>8.4f} "
                    f"{ds.get('avg_score', 0) or 0.0:>9.4f}"
                )

        pe = report.summary.get("per_expert", {})
        if pe:
            print("-" * 72)
            print(f"  {'Expert':<18} {'Runs':>5} {'Success':>8} {'AvgRetry':>9} {'AvgLat':>8} {'Cost':>8}")
            print("  " + "-" * 63)
            for ex_key, ex in sorted(pe.items()):
                print(
                    f"  {ex_key:<18} {ex.get('algorithms_run', 0):>5} "
                    f"{ex.get('success_rate', 0.0):>7.1%} "
                    f"{ex.get('avg_retries', 0) or 0.0:>9.2f} "
                    f"{ex.get('avg_latency_ms', 0) or 0:>7.0f}ms"
                    f" ${ex.get('total_cost_usd', 0) or 0.0:>7.4f}"
                )
        print("=" * 72 + "\n")

    @staticmethod
    def compute_exit_code(report: BenchmarkReport, min_success_rate: float = 0.80) -> int:
        """Return 0 if overall success_rate >= min_success_rate, else 1."""
        ov = report.summary.get("overall", {})
        rate = ov.get("success_rate", 0.0)
        return 0 if rate >= min_success_rate else 1
