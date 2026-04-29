"""CLI entry point: python -m ACE_Agent.benchmark [options]."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Ensure the project-root parent is on sys.path so that "ACE_Agent" resolves to
# the project root itself, not the (possibly empty) ACE_Agent/ subdirectory.
_project_parent = str(Path(__file__).resolve().parents[2])
if _project_parent not in sys.path:
    sys.path.insert(0, _project_parent)

from ACE_Agent.benchmark.config import BenchmarkConfig  # noqa: E402
from ACE_Agent.benchmark.reporter import BenchmarkReporter  # noqa: E402
from ACE_Agent.benchmark.runner import BenchmarkRunner  # noqa: E402
from ACE_Agent.tools.llm_client import LLMSettings  # noqa: E402


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="ACE Agent Benchmark Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  python -m ACE_Agent.benchmark --offline\n"
            "  python -m ACE_Agent.benchmark --datasets blobs,moons,iris --experts zoo --n-samples 200\n"
        ),
    )
    p.add_argument(
        "--datasets",
        default="blobs,moons,iris,wine,s_curve,digits,smile",
    )
    p.add_argument("--experts", default="zoo")
    p.add_argument("--offline", action="store_true", default=False)
    p.add_argument("--n-samples", type=int, default=480)
    p.add_argument("--noise", type=float, default=0.06)
    p.add_argument("--output", default=None)
    p.add_argument("--min-success-rate", type=float, default=0.80)
    return p.parse_args(argv)


def _build_settings() -> LLMSettings | None:
    # 1) Load from .ace_demo_config.json (populated by Streamlit web UI)
    try:
        from ACE_Agent.tools.settings_store import load_settings  # noqa: E402

        config = load_settings()
        if config.get("llm_enabled") and config.get("llm_api_key"):
            return LLMSettings(
                provider=config.get("llm_provider", "DashScope"),
                base_url=config.get("llm_base_url", ""),
                api_key=config["llm_api_key"],
                model=config.get("llm_model", "qwen-plus"),
                enabled=True,
            )
    except Exception:
        pass

    # 2) Fallback: environment variables
    enabled = os.environ.get("ACE_LLM_ENABLED", "").lower() == "true"
    base_url = os.environ.get("ACE_LLM_BASE_URL", "")
    api_key = os.environ.get("ACE_LLM_API_KEY", "")
    model = os.environ.get("ACE_LLM_MODEL", "")
    if enabled and base_url and api_key and model:
        return LLMSettings(
            provider="DashScope",
            base_url=base_url,
            api_key=api_key,
            model=model,
            enabled=True,
        )
    return None


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    config = BenchmarkConfig(
        datasets=[d.strip() for d in args.datasets.split(",") if d.strip()],
        experts=[e.strip() for e in args.experts.split(",") if e.strip()],
        n_samples=args.n_samples,
        noise=args.noise,
        offline_mode=args.offline,
        llm_settings=_build_settings(),
        min_success_rate=args.min_success_rate,
    )

    if config.offline_mode:
        print("[benchmark] Offline mode — only non-LLM experts will run.")
    elif config.llm_settings is None:
        print("[benchmark] LLM not configured; switching to offline mode.")
        config.offline_mode = True

    runner = BenchmarkRunner(config)
    report = runner.run()

    output_path = args.output or BenchmarkReporter.default_output_path()
    BenchmarkReporter.write_json(report, output_path)
    BenchmarkReporter.print_summary(report)

    exit_code = BenchmarkReporter.compute_exit_code(report, config.min_success_rate)
    print(f"[benchmark] Report saved to: {output_path}")
    print(f"[benchmark] Exit code: {exit_code} "
          f"(success_rate={report.summary['overall']['success_rate']:.1%}, "
          f"threshold={config.min_success_rate:.0%})")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
