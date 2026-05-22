from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from ACE_Agent.agent_core.schemas import DatasetBundle
from ACE_Agent.agent_core.supervisor import ACESupervisor
from ACE_Agent.tools.data_factory import generate_dataset, list_demo_datasets
from ACE_Agent.tools.llm_client import LLMSettings, MultiLLMConfig
from ACE_Agent.tools.settings_store import load_settings


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the ACE Agent clustering demo in CLI mode.")
    parser.add_argument("--dataset", choices=list_demo_datasets(), default="smile")
    parser.add_argument("--samples", type=int, default=420)
    parser.add_argument("--noise", type=float, default=0.06)
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--interactive", action="store_true", help="Enable interactive follow-up mode.")
    args = parser.parse_args()

    # 加载 LLM 配置
    config = load_settings()
    llm_settings = LLMSettings(
        base_url=config.get("llm_base_url", ""),
        api_key=config.get("llm_api_key", ""),
        model=config.get("llm_model", "gpt-3.5-turbo"),
        enabled=config.get("llm_enabled", False),
    )

    supervisor = ACESupervisor()
    dataset = generate_dataset(args.dataset, n_samples=args.samples, noise=args.noise, random_state=42)

    # 第一次运行
    first_prompt = args.prompt or f"请分析 {args.dataset} 数据集。"
    llm_config = MultiLLMConfig(worker=llm_settings)
    _process_and_print(supervisor, dataset, first_prompt, llm_config)

    # 如果是交互模式或没有指定 prompt，询问用户是否要追问
    if args.interactive:
        print("\n" + "=" * 50)
        print("进入交互模式（输入 'exit' 或 'quit' 退出）。")
        while True:
            try:
                user_input = input("\n[追问] > ").strip()
                if not user_input:
                    continue
                if user_input.lower() in ["exit", "quit", "退出", "q"]:
                    break

                # 追问逻辑：不重新传 dataset
                _process_and_print(supervisor, None, user_input, llm_config)

            except KeyboardInterrupt:
                break
    else:
        print("\n提示: 使用 --interactive 选项可以进入连续追问模式。")


def _process_and_print(
    supervisor: ACESupervisor, dataset: DatasetBundle | None, prompt: str, llm_config: MultiLLMConfig
):
    print(f"\n[ACE] 处理请求: {prompt}")
    report = supervisor.run(dataset=dataset, user_prompt=prompt, llm_config=llm_config)

    if report.response_type == "FOLLOW_UP":
        print("\n[ACE 分析回复]:")
        print(report.llm_summary)
    else:
        print("\n[ACE 任务报告摘要]:")
        print(report.executive_summary)
        print(f"LaTeX 报告路径: {report.latex_path}")
        print("\n算法排行榜:")
        for item in report.ranking[:5]:
            print(
                f"- {item.expert_label} / {item.algorithm_name}: "
                f"score={float(item.metrics.get('score') or 0.0):.3f}, "
                f"AMI={_fmt(item.metrics.get('ami'))}, silhouette={_fmt(item.metrics.get('silhouette'))}"
            )
        if report.llm_summary:
            print("\n[LLM 综合点评]:")
            print(report.llm_summary)


def _fmt(value):
    try:
        return f"{float(value):.3f}"
    except Exception:
        return "n/a"


if __name__ == "__main__":
    main()
