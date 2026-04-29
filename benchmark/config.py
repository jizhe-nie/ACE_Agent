"""Benchmark configuration."""
from __future__ import annotations

from dataclasses import dataclass, field

from ACE_Agent.tools.llm_client import LLMSettings


@dataclass
class BenchmarkConfig:
    """Tunable parameters for a benchmark run."""

    datasets: list[str] = field(default_factory=lambda: [
        "blobs", "moons", "iris", "wine", "s_curve", "digits", "smile",
    ])
    experts: list[str] = field(default_factory=lambda: ["zoo"])
    n_samples: int = 480
    noise: float = 0.06
    random_state: int = 42
    timeout_per_dataset_sec: int = 300
    output_dir: str = "outputs"
    offline_mode: bool = False
    llm_settings: LLMSettings | None = None
    min_success_rate: float = 0.80
