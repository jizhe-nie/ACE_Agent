from __future__ import annotations

from pathlib import Path
import sys

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd
import streamlit as st

from ACE_Agent.agent_core.supervisor import ACESupervisor
from ACE_Agent.agent_core.schemas import SupervisorReport
from ACE_Agent.tools.data_factory import (
    DATASET_LABELS,
    generate_dataset,
    infer_dataset_from_prompt,
    list_demo_datasets,
)
from ACE_Agent.tools.llm_client import LLMSettings
from ACE_Agent.tools.settings_store import load_settings, save_settings


st.set_page_config(page_title="ACE Agent Demo", layout="wide")


def main() -> None:
    settings = _sidebar_settings()
    _init_state()

    st.title("ACE Agent")
    st.caption("多 Agent 聚类 demo：主控路由、专家执行、指标评估、图文报告、LaTeX 输出。")

    control_col, info_col = st.columns([1.15, 1.0])
    with control_col:
        dataset_name = st.selectbox(
            "Demo 数据集",
            list_demo_datasets(),
            format_func=lambda value: DATASET_LABELS[value],
            key="selected_dataset",
        )
        sample_count = st.slider("样本数", min_value=180, max_value=900, value=480, step=30, key="sample_count")
        noise = st.slider("噪声强度", min_value=0.01, max_value=0.18, value=0.06, step=0.01, key="noise")
        seed = st.number_input("随机种子", min_value=0, max_value=9999, value=42, step=1, key="seed")
    with info_col:
        st.markdown(
            """
            **页面说明**

            - 侧边栏保存 OpenAI 兼容接口配置，用户可以自行选择模型并保存。
            - 对话区会展示主控 Agent 的总结。
            - “决策轨迹 / 推理摘要”展示的是可解释的路由与实验过程，不是底层模型的原始隐式思维链。
            """
        )

    st.divider()
    _render_messages()

    prompt = st.chat_input("例如：请分析笑脸数据，并比较拓扑方法和质心方法。")
    if prompt:
        _handle_prompt(
            prompt=prompt,
            default_dataset=dataset_name,
            sample_count=int(sample_count),
            noise=float(noise),
            seed=int(seed),
            settings=settings,
        )
        st.rerun()


def _sidebar_settings() -> LLMSettings:
    saved = load_settings()
    st.sidebar.header("模型配置")
    enabled = st.sidebar.toggle(
        "启用大模型润色",
        value=bool(saved.get("enabled", False)),
        help="关闭后依然可以完整运行聚类与报告流程，只是不调用外部模型生成润色摘要。",
    )
    base_url = st.sidebar.text_input("Base URL", value=saved.get("base_url", ""))
    api_key = st.sidebar.text_input("API Key", value=saved.get("api_key", ""), type="password")
    model = st.sidebar.text_input("Model", value=saved.get("model", ""))
    temperature = st.sidebar.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=float(saved.get("temperature", 0.2)),
        step=0.05,
    )
    if st.sidebar.button("保存模型配置", use_container_width=True):
        save_settings(
            {
                "enabled": enabled,
                "base_url": base_url,
                "api_key": api_key,
                "model": model,
                "temperature": temperature,
            }
        )
        st.sidebar.success("配置已保存到 ACE_Agent/.ace_demo_config.json")

    return LLMSettings(
        base_url=base_url,
        api_key=api_key,
        model=model,
        temperature=temperature,
        enabled=enabled,
    )


def _init_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": (
                    "欢迎来到 ACE Agent demo。你可以直接在聊天框里说“请分析笑脸数据”或“帮我看看 S 形数据”，"
                    "我会让主控 Agent 自动路由到合适的聚类专家。"
                ),
                "report": None,
            }
        ]


def _render_messages() -> None:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("report") is not None:
                _render_report(message["report"])


def _handle_prompt(
    prompt: str,
    default_dataset: str,
    sample_count: int,
    noise: float,
    seed: int,
    settings: LLMSettings,
) -> None:
    st.session_state.messages.append({"role": "user", "content": prompt, "report": None})

    # 获取上一条报告（如果存在）
    last_report = next((m["report"] for m in reversed(st.session_state.messages) if m["report"]), None)
    
    # 意图判断：是否是针对已有报告的提问
    is_question = any(q in prompt for q in ["为什么", "怎么", "如何", "解释", "哪个", "解释下"])
    
    if last_report and is_question:
        with st.spinner("正在基于当前结果回答您的疑问..."):
            if settings.is_configured:
                # 调用 LLM 进行对话式回答
                from ACE_Agent.tools.llm_client import OpenAICompatibleClient
                client = OpenAICompatibleClient(settings)
                # 构造一个简单的对话上下文
                context_payload = {
                    "question": prompt,
                    "existing_report_summary": last_report.executive_summary,
                    "metrics": [
                        {"algo": r.algorithm_name, "score": r.metrics.get("score")} 
                        for r in last_report.ranking[:3]
                    ]
                }
                answer = client.summarize_report({"type": "follow_up", "data": context_payload})
                if answer:
                    st.session_state.messages.append({"role": "assistant", "content": answer, "report": None})
                    return

    # 如果不是提问，或者是提问但没有配置 LLM/没有上文，则执行新聚类
    inferred_dataset = infer_dataset_from_prompt(prompt) or default_dataset
    dataset = generate_dataset(
        dataset_name=inferred_dataset,
        n_samples=sample_count,
        noise=noise,
        random_state=seed,
    )

    with st.spinner("主控 Agent 正在调度专家并运行实验..."):
        report = ACESupervisor().run(
            dataset=dataset,
            user_prompt=prompt,
            llm_settings=settings,
        )

    content = report.llm_summary or report.executive_summary
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": content,
            "report": report,
        }
    )


def _render_report(report: SupervisorReport) -> None:
    top = report.ranking[0]
    metric_cols = st.columns(4)
    metric_cols[0].metric("最佳算法", top.algorithm_name)
    metric_cols[1].metric("所属专家", top.expert_label)
    metric_cols[2].metric("综合得分", f"{float(top.metrics.get('score', 0.0)):.3f}")
    metric_cols[3].metric("AMI", _fmt_metric(top.metrics.get("ami")))

    with st.expander("决策轨迹 / 推理摘要", expanded=True):
        for line in report.decision_trace:
            st.markdown(f"- {line}")

    result_table = pd.DataFrame(
        [
            {
                "expert": item.expert_label,
                "algorithm": item.algorithm_name,
                "score": round(float(item.metrics.get("score", 0.0)), 4),
                "AMI": _float_or_none(item.metrics.get("ami")),
                "Silhouette": _float_or_none(item.metrics.get("silhouette")),
                "clusters": item.metrics.get("cluster_count"),
                "noise_ratio": _float_or_none(item.metrics.get("noise_ratio")),
            }
            for item in report.ranking
        ]
    )
    st.dataframe(result_table, use_container_width=True, hide_index=True)

    image_cols = st.columns(2)
    with image_cols[0]:
        st.image(str(report.dataset_plot_path), caption=f"数据集预览：{report.dataset.display_name}", use_container_width=True)
    with image_cols[1]:
        st.image(str(top.plot_path), caption=f"优胜方案结果：{top.expert_label} / {top.algorithm_name}", use_container_width=True)

    with st.expander("查看 Top-3 子 Agent 代码", expanded=False):
        for item in report.ranking[:3]:
            st.markdown(f"**{item.expert_label} / {item.algorithm_name}**")
            st.code(item.code, language="python")

    latex_bytes = Path(report.latex_path).read_bytes()
    st.download_button(
        "下载 LaTeX 报告",
        data=latex_bytes,
        file_name=report.latex_path.name,
        mime="application/x-tex",
        use_container_width=True,
        key=f"dl_btn_{report.output_dir.name}"
    )
    st.caption(f"输出目录：`{report.output_dir}`")


def _fmt_metric(value: float | int | None) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.3f}"


def _float_or_none(value: float | int | None) -> float | None:
    if value is None:
        return None
    return round(float(value), 4)


if __name__ == "__main__":
    main()
