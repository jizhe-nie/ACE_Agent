from __future__ import annotations

import json
import os
import sys
import uuid
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

os.environ["OMP_NUM_THREADS"] = "1"  # suppress KMeans thread warning

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402,F401  (kept for possible downstream use)
import streamlit as st  # noqa: E402

from ACE_Agent.agent_core.supervisor import ACESupervisor
from ACE_Agent.tools.data_factory import (
    DATASET_LABELS,
    generate_dataset,
    infer_dataset_from_prompt,
    list_demo_datasets,
    load_custom_dataset,
)
from ACE_Agent.tools.llm_client import LLMSettings, UniversalLLMClient
from ACE_Agent.tools.settings_store import DEFAULT_PROVIDERS, SessionManager, SettingsStore

# ---------------------------------------------------------------------------
# Trace path for LLM monitoring panel
# ---------------------------------------------------------------------------
_TRACE_PATH = Path(__file__).resolve().parent / "outputs" / "llm_trace.jsonl"

st.set_page_config(page_title="ACE Agent", layout="wide", initial_sidebar_state="expanded")


# ---------------------------------------------------------------------------
# Cost constants (USD per 1K tokens) — used in sidebar cost estimate
# These mirror values in llm_client.py; keep in sync if pricing changes.
# ---------------------------------------------------------------------------
_COST_TABLE: dict[str, dict[str, float]] = {
    "DeepSeek":  {"input": 0.00014, "output": 0.00028},
    "DashScope": {"input": 0.0004,  "output": 0.0012},
    "OpenAI":    {"input": 0.005,   "output": 0.015},
    "Moonshot":  {"input": 0.001,   "output": 0.003},
    "Gemini":    {"input": 0.00035, "output": 0.00105},
}


# ---------------------------------------------------------------------------
# Session-level LLM cost state (accumulated across all LLM calls in session)
# ---------------------------------------------------------------------------
def _init_cost_state() -> None:
    defaults = {
        "llm_call_count": 0,
        "llm_retry_count": 0,
        "llm_prompt_tokens": 0,
        "llm_completion_tokens": 0,
        "llm_cost_usd": 0.0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _accumulate_cost(client: UniversalLLMClient) -> None:
    """Merge a client's per-session cost summary into global session state."""
    s = client.get_cost_summary()
    st.session_state.llm_call_count += s["call_count"]
    st.session_state.llm_retry_count += s["retry_count"]
    st.session_state.llm_prompt_tokens += s["total_prompt_tokens"]
    st.session_state.llm_completion_tokens += s["total_completion_tokens"]
    st.session_state.llm_cost_usd += s["estimated_cost_usd"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@st.cache_resource
def _get_supervisor() -> ACESupervisor:
    return ACESupervisor()


def _init_state() -> None:
    _init_cost_state()
    for k, v in {
        "current_session_id": str(uuid.uuid4()),
        "messages": [],
        "supervisor": _get_supervisor(),
        "settings_store": SettingsStore(),
        "session_manager": SessionManager(),
    }.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _read_trace_stats() -> dict[str, int]:
    """Read llm_trace.jsonl and compute cumulative stats from disk."""
    stats = {"calls": 0, "retries": 0, "prompt_tokens": 0, "completion_tokens": 0}
    if not _TRACE_PATH.exists():
        return stats
    try:
        for line in _TRACE_PATH.read_text(encoding="utf-8").splitlines():
            try:
                rec = json.loads(line)
                if rec.get("event") == "provider_fallback":
                    stats["retries"] += 1
                    continue
                stats["calls"] += 1
                stats["prompt_tokens"] += rec.get("prompt_tokens", 0)
                stats["completion_tokens"] += rec.get("completion_tokens", 0)
                if rec.get("is_retry"):
                    stats["retries"] += 1
            except json.JSONDecodeError:
                continue
    except Exception:
        pass
    return stats


# ---------------------------------------------------------------------------
# Sidebar UI
# ---------------------------------------------------------------------------


def _sidebar_ui() -> tuple[LLMSettings, LLMSettings | None]:
    """Render sidebar; return (primary_settings, fallback_settings)."""
    sm = st.session_state.session_manager
    ss = st.session_state.settings_store

    with st.sidebar:
        st.title("ACE Agent")

        if st.button("+ 新建对话", use_container_width=True):
            st.session_state.current_session_id = str(uuid.uuid4())
            st.session_state.messages = []
            st.rerun()

        st.divider()
        st.subheader("历史会话")
        for s in sm.sessions[:10]:
            col1, col2 = st.columns([0.8, 0.2])
            title = s.get("metadata", {}).get("title", s["id"][:8])
            if col1.button(title, key=f"s_{s['id']}", use_container_width=True):
                st.session_state.current_session_id = s["id"]
                st.session_state.messages = s["messages"]
                st.rerun()
            if col2.button("X", key=f"d_{s['id']}"):
                sm.delete_session(s["id"])
                st.rerun()

        # Add some visual spacing
        for _ in range(3):
            st.sidebar.write("")

        # ------------------------------------------------------------------
        # Model configuration
        # ------------------------------------------------------------------
        with st.popover("Model Config", use_container_width=True):
            provider_names = list(DEFAULT_PROVIDERS.keys())

            # Primary provider
            st.markdown("**Primary Provider**")
            active_p = st.selectbox(
                "供应商",
                provider_names,
                index=provider_names.index(ss.get("active_provider", "DeepSeek")),
                key="primary_provider_sel",
            )
            p_cfg = DEFAULT_PROVIDERS[active_p]
            api_key = st.text_input(
                "API Key",
                value=ss.get("api_keys", {}).get(active_p, ""),
                type="password",
                key="primary_api_key",
            )
            model = st.selectbox(
                "模型",
                p_cfg["models"],
                index=(
                    0
                    if ss.get("model") not in p_cfg["models"]
                    else p_cfg["models"].index(ss.get("model"))
                ),
                key="primary_model_sel",
            )

            st.divider()

            # Fallback provider
            st.markdown("**Fallback Provider** (optional)")
            fallback_options = ["(disabled)"] + provider_names
            saved_fallback = ss.get("fallback_provider", "(disabled)")
            fallback_p = st.selectbox(
                "Fallback 供应商",
                fallback_options,
                index=(
                    fallback_options.index(saved_fallback)
                    if saved_fallback in fallback_options
                    else 0
                ),
                key="fallback_provider_sel",
            )
            fallback_api_key = ""
            fallback_model = ""
            if fallback_p != "(disabled)":
                fb_cfg = DEFAULT_PROVIDERS[fallback_p]
                fallback_api_key = st.text_input(
                    "Fallback API Key",
                    value=ss.get("api_keys", {}).get(fallback_p, ""),
                    type="password",
                    key="fallback_api_key",
                )
                fallback_model = st.selectbox(
                    "Fallback 模型",
                    fb_cfg["models"],
                    index=0,
                    key="fallback_model_sel",
                )

            if st.button("保存配置", use_container_width=True):
                keys = ss.get("api_keys", {})
                keys[active_p] = api_key
                if fallback_p != "(disabled)":
                    keys[fallback_p] = fallback_api_key
                ss.save({
                    "active_provider": active_p,
                    "api_keys": keys,
                    "model": model,
                    "fallback_provider": fallback_p,
                    "fallback_model": fallback_model,
                })
                st.rerun()

        st.divider()

        # ------------------------------------------------------------------
        # LLM call monitoring panel (P0-3)
        # ------------------------------------------------------------------
        with st.expander("LLM Call Monitor", expanded=False):
            trace_stats = _read_trace_stats()
            provider_cost = _COST_TABLE.get(active_p, {"input": 0.0, "output": 0.0})
            est_cost = (
                trace_stats["prompt_tokens"] / 1000.0 * provider_cost["input"]
                + trace_stats["completion_tokens"] / 1000.0 * provider_cost["output"]
            )

            col_a, col_b = st.columns(2)
            col_a.metric("Total Calls", trace_stats["calls"])
            col_b.metric("Retries", trace_stats["retries"])

            col_c, col_d = st.columns(2)
            col_c.metric("Prompt Tokens", f"{trace_stats['prompt_tokens']:,}")
            col_d.metric("Completion Tokens", f"{trace_stats['completion_tokens']:,}")

            st.metric("Est. Cost (USD)", f"${est_cost:.4f}")
            st.caption(
                f"Prices: input ${provider_cost['input']}/1K, "
                f"output ${provider_cost['output']}/1K ({active_p})"
            )

            if st.button("Clear Trace Log", use_container_width=True):
                if _TRACE_PATH.exists():
                    _TRACE_PATH.write_text("", encoding="utf-8")
                # Reset session cost counters
                for k in ["llm_call_count", "llm_retry_count", "llm_prompt_tokens",
                          "llm_completion_tokens", "llm_cost_usd"]:
                    st.session_state[k] = 0
                st.rerun()

    # Build settings objects
    primary_settings = LLMSettings(
        provider=active_p,
        base_url=p_cfg["base_url"],
        api_key=api_key,
        model=model,
        temperature=ss.get("temperature", 0.2),
    )

    fallback_settings: LLMSettings | None = None
    if fallback_p != "(disabled)" and fallback_api_key and fallback_model:
        fb_cfg = DEFAULT_PROVIDERS[fallback_p]
        fallback_settings = LLMSettings(
            provider=fallback_p,
            base_url=fb_cfg["base_url"],
            api_key=fallback_api_key,
            model=fallback_model,
            temperature=ss.get("temperature", 0.2),
        )

    return primary_settings, fallback_settings


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    _init_state()
    settings, fallback_settings = _sidebar_ui()

    st.title("ACE Agent")
    st.caption("基于 Orchestrator 架构的自愈式多代理聚类系统")

    with st.expander("Data Config", expanded=not st.session_state.messages):
        t1, t2 = st.tabs(["内置数据", "上传数据"])
        with t1:
            c1, c2 = st.columns(2)
            ds_name = c1.selectbox(
                "模板",
                [d for d in list_demo_datasets() if d != "custom"],
                format_func=lambda v: DATASET_LABELS[v],
            )
            sc = c2.slider("样本量", 180, 2000, 480, 30)
            c3, c4 = st.columns(2)
            noise = c3.slider("噪声", 0.01, 0.18, 0.06, 0.01)
            seed = c4.number_input("随机种子", 0, 9999, 42)
        with t2:
            uploaded_file = st.file_uploader("上传 CSV/Excel", type=["csv", "xlsx", "xls"])

        if st.button("Preview Data Distribution", use_container_width=True) or uploaded_file:
            st.divider()
            with st.spinner("正在绘制原始分布..."):
                preview_ds = None
                if uploaded_file:
                    import tempfile

                    with tempfile.NamedTemporaryFile(
                        delete=False, suffix=Path(uploaded_file.name).suffix
                    ) as tmp:
                        tmp.write(uploaded_file.getvalue())
                        tmp_path = tmp.name
                    preview_ds = load_custom_dataset(tmp_path)
                    os.remove(tmp_path)
                else:
                    preview_ds = generate_dataset(
                        ds_name, n_samples=sc, noise=noise, random_state=seed
                    )

                if preview_ds:
                    st.subheader(f"数据预览: {preview_ds.display_name}")
                    pc1, pc2 = st.columns([0.7, 0.3])
                    with pc1:
                        import platform

                        if platform.system() == "Windows":
                            plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
                        plt.rcParams["axes.unicode_minus"] = False
                        fig, ax = plt.subplots(figsize=(8, 4.5))
                        ax.scatter(
                            preview_ds.X[:, 0],
                            preview_ds.X[:, 1],
                            c="black",
                            s=10,
                            alpha=0.5,
                            edgecolors="none",
                        )
                        ax.set_title("原始特征空间分布 (特征1 vs 特征2)", fontsize=10)
                        ax.set_xlabel("Feature 1")
                        ax.set_ylabel("Feature 2")
                        ax.grid(True, linestyle=":", alpha=0.6)
                        st.pyplot(fig)
                        plt.close(fig)
                    with pc2:
                        st.write("**数据统计**")
                        st.metric("样本总数", preview_ds.X.shape[0])
                        st.metric("特征维度", preview_ds.X.shape[1])
                        st.info("提示：请在下方输入指令来启动智能体聚类任务。")

    _render_messages()
    if prompt := st.chat_input("输入指令，例如：使用谱聚类分析这个数据集..."):
        _handle_prompt(prompt, ds_name, sc, noise, seed, settings, fallback_settings, uploaded_file)
        st.rerun()


# ---------------------------------------------------------------------------
# Message rendering
# ---------------------------------------------------------------------------


def _render_messages() -> None:
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            if m.get("thought"):
                with st.expander("Agent Thinking Trace", expanded=False):
                    for line in m["thought"].split("\n"):
                        if "[RAG]" in line or "【RAG】" in line:
                            st.info(line)
                        elif "失败" in line or "错误" in line:
                            st.error(line)
                        elif "成功" in line:
                            st.success(line)
                        else:
                            st.write(line)
            st.markdown(m["content"])
            if m.get("report"):
                _render_report(m["report"])


@st.cache_data
def _cached_load_custom(file_bytes: bytes, file_name: str):  # type: ignore[return]
    import tempfile

    suffix = Path(file_name).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    try:
        ds = load_custom_dataset(tmp_path)
        ds.display_name = f"上传: {file_name}"
        return ds
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def _handle_prompt(
    prompt: str,
    ds_name: str,
    sc: int,
    noise: float,
    seed: int,
    settings: LLMSettings,
    fallback_settings: LLMSettings | None,
    uploaded_file,  # type: ignore[type-arg]
) -> None:
    st.session_state.messages.append({"role": "user", "content": prompt})
    supervisor = st.session_state.supervisor

    with st.chat_message("assistant"):
        with st.status("ACE Orchestrator 正在运行...", expanded=True) as status:
            st.write("正在通过 MasterRouter 进行语义识别...")

            # Build router client with fallback support
            router_client = UniversalLLMClient(settings, fallback_settings, caller="router")
            intent = supervisor.router.analyze_intent(prompt, supervisor.memory, settings)
            st.write(f"意图判定: **{intent.get('intent')}** ({intent.get('reasoning')})")
            _accumulate_cost(router_client)

            dataset = None
            if intent.get("intent") == "NEW_TASK":
                st.write("正在画像并准备数据集...")
                if uploaded_file:
                    dataset = _cached_load_custom(uploaded_file.getvalue(), uploaded_file.name)
                else:
                    inferred = infer_dataset_from_prompt(prompt)
                    dataset = generate_dataset(
                        inferred or ds_name, n_samples=sc, noise=noise, random_state=seed
                    )

            st.write("正在激活专家 Agent 并监控自愈执行...")
            report = supervisor.run(
                dataset=dataset,
                user_prompt=prompt,
                llm_settings=settings,
                intent_data=intent,
            )
            status.update(label="分析任务完成", state="complete", expanded=False)

        thought = "\n".join(report.decision_trace)
        st.markdown(report.llm_summary or report.executive_summary)
        if report.response_type == "CLUSTER_TASK":
            _render_report(report)

    st.session_state.messages.append({
        "role": "assistant",
        "content": report.llm_summary or report.executive_summary,
        "thought": thought,
        "report": report if report.response_type == "CLUSTER_TASK" else None,
    })
    st.session_state.session_manager.save_session(
        st.session_state.current_session_id,
        st.session_state.messages,
        {
            "title": prompt[:30],
            "dataset": dataset.display_name if dataset else "追问",
        },
    )


def _render_report(r) -> None:  # type: ignore[type-arg]
    ranking = r.ranking if hasattr(r, "ranking") else r["ranking"]
    dataset = r.dataset if hasattr(r, "dataset") else r["dataset"]
    top = ranking[0]
    ds_name_display = (
        dataset.display_name if hasattr(dataset, "display_name") else dataset["display_name"]
    )
    st.subheader(f"分析报告: {ds_name_display}")
    c = st.columns(4)
    algo_name = top.algorithm_name if hasattr(top, "algorithm_name") else top["algorithm_name"]
    score = float(
        top.metrics["score"] if hasattr(top, "metrics") else top["metrics"]["score"]
    )
    c[0].metric("优胜算法", algo_name)
    c[2].metric("评分", f"{score:.3f}")
    cols = st.columns(2)
    raw_plot = r.dataset_plot_path if hasattr(r, "dataset_plot_path") else r["dataset_plot_path"]
    top_plot = top.plot_path if hasattr(top, "plot_path") else top["plot_path"]
    cols[0].image(str(raw_plot), caption="原始分布")
    cols[1].image(str(top_plot), caption="最优聚类结果")


if __name__ == "__main__":
    main()
