from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from ACE_Agent.agent_core.supervisor import ACESupervisor
from ACE_Agent.tools.data_factory import generate_dataset, list_demo_datasets


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the ACE Agent clustering demo in CLI mode.")
    parser.add_argument("--dataset", choices=list_demo_datasets(), default="smile")
    parser.add_argument("--samples", type=int, default=420)
    parser.add_argument("--noise", type=float, default=0.06)
    args = parser.parse_args()

    dataset = generate_dataset(args.dataset, n_samples=args.samples, noise=args.noise, random_state=42)
    report = ACESupervisor().run(dataset=dataset, user_prompt=f"Analyze the {args.dataset} dataset.")
    print(report.executive_summary)
    print(f"LaTeX report: {report.latex_path}")
    for item in report.ranking[:5]:
        print(
            f"- {item.expert_label} / {item.algorithm_name}: "
            f"score={float(item.metrics.get('score', 0.0)):.3f}, "
            f"AMI={item.metrics.get('ami')}, silhouette={item.metrics.get('silhouette')}"
        )


if __name__ == "__main__":
    main()
