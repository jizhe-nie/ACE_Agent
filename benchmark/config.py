"""Benchmark configuration."""

from __future__ import annotations

from dataclasses import dataclass, field

from ACE_Agent.tools.llm_client import LLMSettings

# ---------------------------------------------------------------------------
# Stratified benchmark preset lists
# ---------------------------------------------------------------------------
BENCHMARK_SMOKE = ["blobs", "moons", "iris"]  # ~5 min, CI every commit
BENCHMARK_FULL = [
    "iris", "wine", "digits", "usps", "pendigits", "letter",
    "har", "reuters", "mnist", "fashion_mnist", "cifar10_raw",
]  # ~30 min, daily/PR merge
BENCHMARK_EXHAUSTIVE = [
    "blobs", "moons", "smile", "s_curve",
    "iris", "wine", "digits", "usps", "pendigits", "letter",
    "har", "reuters", "mnist", "fashion_mnist",
    "cifar10_raw", "cifar10_gap", "cifar10_resnet",
    "coil20",
]  # ~2 h, release/paper


@dataclass
class BenchmarkConfig:
    """Tunable parameters for a benchmark run."""

    datasets: list[str] = field(default_factory=lambda: list(BENCHMARK_FULL))
    experts: list[str] = field(default_factory=lambda: ["zoo"])
    n_samples: int = 480
    noise: float = 0.06
    random_state: int = 42
    timeout_per_dataset_sec: int = 300
    output_dir: str = "outputs"
    offline_mode: bool = False
    llm_settings: LLMSettings | None = None
    min_success_rate: float = 0.80
