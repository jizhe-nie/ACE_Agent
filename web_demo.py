from __future__ import annotations

from pathlib import Path
import sys
import os

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
    load_custom_dataset,
)
from ACE_Agent.tools.llm_client import LLMSettings
from ACE_Agent.tools.settings_store import load_settings, save_settings


st.set_page_config(page_title="ACE Agent Demo", layout="wide")


def main() -> None:
    settings = _sidebar_settings()
    _init_state()

    st.title("ACE Agent")
    st.caption("具备主控路由、多专家执行、指标评估及图文报告生成的自动化聚类分析系统。")

    # 数据配置区
    with st.expander("数据配置", expanded=True):
        source_tab1, source_tab2 = st.tabs(["内置 Demo 数据", "上传自定义数据"])
        
        with source_tab1:
            col1, col2 = st.columns([1, 1])
            with col1:
                dataset_name = st.selectbox(
                    "选择数据集模板",
                    [d for d in list_demo_datasets() if d != "custom"],
                    format_func=lambda value: DATASET_LABELS[value],
                    key="selected_dataset",
                )
            with col2:
                sample_count = st.slider("样本数", min_value=180, max_value=2000, value=480, step=30, key="sample_count")
            
            col3, col4 = st.columns([1, 1])
            with col3:
                noise = st.slider("噪声强度", min_value=0.01, max_value=0.18, value=0.06, step=0.01, key="noise")
            with col4:
                seed = st.number_input("随机种子", min_value=0, max_value=9999, value=42, step=1, key="seed")
            
            uploaded_file = None

        with source_tab2:
            st.markdown("上传 CSV 或 Excel 文件。系统将尝试自动识别特征列。")
            uploaded_file = st.file_uploader("选择文件", type=["csv", "xlsx", "xls"])
            if uploaded_file:
                st.info(f"已就绪: {uploaded_file.name}")

    if st.button("重置对话历史"):
        st.session_state.messages = []
        st.session_state.supervisor = ACESupervisor()
        st.rerun()

    st.divider()
    _render_messages()

    prompt = st.chat_input("输入指令或进行追问（如：为什么 Spectral 算法表现最好？）")
    if prompt:
        _handle_prompt(
            prompt=prompt,
            default_dataset=dataset_name,
            sample_count=int(sample_count),
            noise=float(noise),
            seed=int(seed),
            settings=settings,
            uploaded_file=uploaded_file,
        )
        st.rerun()


def _sidebar_settings() -> LLMSettings:
    saved = load_settings()
    st.sidebar.header("模型配置")
    enabled = st.sidebar.toggle(
        "启用大模型润色/追问",
        value=bool(saved.get("enabled", False)),
    )
    base_url = st.sidebar.text_input("Base URL", value=saved.get("base_url", ""))
    api_key = st.sidebar.text_input("API Key", value=saved.get("api_key", ""), type="password")
    model = st.sidebar.text_input("Model", value=saved.get("model", ""))
    temperature = st.sidebar.slider("Temperature", 0.0, 1.0, float(saved.get("temperature", 0.2)))
    
    if st.sidebar.button("保存配置"):
        save_settings({"enabled": enabled, "base_url": base_url, "api_key": api_key, "model": model, "temperature": temperature})
        st.sidebar.success("配置已保存")

    return LLMSettings(base_url=base_url, api_key=api_key, model=model, temperature=temperature, enabled=enabled)


def _init_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "supervisor" not in st.session_state:
        st.session_state.supervisor = ACESupervisor()


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
    uploaded_file: any = None,
) -> None:
    st.session_state.messages.append({"role": "user", "content": prompt, "report": None})
    supervisor = st.session_state.supervisor

    # 1. 自动识别数据集逻辑
    dataset = None
    # 只有当 supervisor 认为这是 NEW_TASK 时，才需要加载数据集
    # 我们先让 supervisor 决定意图
    intent = supervisor.router.analyze_intent(prompt, supervisor.memory, settings)
    
    if intent == "NEW_TASK":
        if uploaded_file is not None:
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name
            try:
                dataset = load_custom_dataset(tmp_path)
                dataset.display_name = f"上传数据: {uploaded_file.name}"
            finally:
                if os.path.exists(tmp_path): os.remove(tmp_path)
        else:
            inferred = infer_dataset_from_prompt(prompt)
            active_dataset_name = inferred or default_dataset
            dataset = generate_dataset(active_dataset_name, n_samples=sample_count, noise=noise, random_state=seed)

    # 2. 调用 Supervisor
    with st.spinner("ACE Agent 正在分析..." if intent == "FOLLOW_UP" else "正在调度专家运行聚类任务..."):
        report = supervisor.run(dataset=dataset, user_prompt=prompt, llm_settings=settings)

    # 3. 处理展示逻辑
    content = report.llm_summary or report.executive_summary
    st.session_state.messages.append({
        "role": "assistant",
        "content": content,
        "report": report if report.response_type == "CLUSTER_TASK" else None
    })


def _render_report(report: SupervisorReport) -> None:
    top = report.ranking[0]
    st.subheader(f"分析报告: {report.dataset.display_name}")
    
    metric_cols = st.columns(4)
    metric_cols[0].metric("最佳算法", top.algorithm_name)
    metric_cols[1].metric("所属专家", top.expert_label)
    metric_cols[2].metric("综合得分", f"{float(top.metrics.get('score', 0.0)):.3f}")
    metric_cols[3].metric("AMI", _fmt_metric(top.metrics.get("ami")))

    with st.expander("决策轨迹", expanded=False):
        for line in report.decision_trace:
            st.markdown(f"- {line}")

    result_table = pd.DataFrame([
        {
            "专家": item.expert_label,
            "算法": item.algorithm_name,
            "综合得分": round(float(item.metrics.get("score", 0.0)), 4),
            "AMI": _float_or_none(item.metrics.get("ami")),
            "轮廓系数": _float_or_none(item.metrics.get("silhouette")),
            "簇数量": item.metrics.get("cluster_count"),
        }
        for item in report.ranking
    ])
    st.dataframe(result_table, use_container_width=True, hide_index=True)

    image_cols = st.columns(2)
    with image_cols[0]:
        st.image(str(report.dataset_plot_path), caption="原始数据分布", use_container_width=True)
    with image_cols[1]:
        st.image(str(top.plot_path), caption=f"优胜方案结果 ({top.algorithm_name})", use_container_width=True)

    latex_bytes = Path(report.latex_path).read_bytes()
    st.download_button(
        "下载 PDF/LaTeX 报告",
        data=latex_bytes,
        file_name=report.latex_path.name,
        mime="application/x-tex",
        use_container_width=True,
        key=f"dl_btn_{report.output_dir.name}_{hash(report.executive_summary)}"
    )


def _fmt_metric(value: any) -> str:
    if value is None: return "n/a"
    try: return f"{float(value):.3f}"
    except: return str(value)


def _float_or_none(value: any) -> float | None:
    if value is None: return None
    try: return round(float(value), 4)
    except: return None


if __name__ == "__main__":
    main()
