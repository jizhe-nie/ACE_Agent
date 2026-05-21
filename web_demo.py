from __future__ import annotations

import json
import os
import sys
import uuid
from datetime import datetime
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

os.environ["OMP_NUM_THREADS"] = "1"  # suppress KMeans thread warning

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402,F401  (kept for possible downstream use)
import streamlit as st  # noqa: E402

from ACE_Agent.tools.data_factory import (
    DATASET_GROUPS,
    DATASET_LABELS,
    _analyze_uploaded_data,
    _extract_cnn_features,
    generate_dataset,
    list_demo_datasets,
    load_custom_dataset,
)
from ACE_Agent.tools.settings_store import DEFAULT_PROVIDERS, SessionManager, SettingsStore

# ---------------------------------------------------------------------------
# Trace path for LLM monitoring panel
# ---------------------------------------------------------------------------
_TRACE_PATH = Path(__file__).resolve().parent / "outputs" / "llm_trace.jsonl"

st.set_page_config(page_title="ACE Agent", layout="wide", initial_sidebar_state="expanded")


def _compute_cache_version() -> str:
    """Hash key source files so stale @st.cache_resource entries are invalidated
    when supervisor, experts, or data factory code changes."""
    import hashlib

    _project_root = Path(__file__).resolve().parent
    _hash = hashlib.sha1()
    for _rel in [
        "agent_core/supervisor.py",
        "agent_core/schemas.py",
        "expert_sub_agents/__init__.py",
        "expert_sub_agents/base.py",
        "tools/data_factory.py",
        "tools/coder_sandbox.py",
    ]:
        _fp = _project_root / _rel
        if _fp.exists():
            _hash.update(_fp.read_bytes())
    return _hash.hexdigest()[:12]


# ---------------------------------------------------------------------------
# Cost constants (USD per 1K tokens) — used in sidebar cost estimate
# These mirror values in llm_client.py; keep in sync if pricing changes.
# ---------------------------------------------------------------------------
_COST_TABLE: dict[str, dict[str, float]] = {
    "DeepSeek": {"input": 0.00014, "output": 0.00028},
    "DashScope": {"input": 0.0004, "output": 0.0012},
    "OpenAI": {"input": 0.005, "output": 0.015},
    "Moonshot": {"input": 0.001, "output": 0.003},
    "Gemini": {"input": 0.00035, "output": 0.00105},
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


def _accumulate_cost(client) -> None:  # client: UniversalLLMClient
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


# Datasets that are synthetically generated and respect n_samples.
# Everything else (real benchmarks, image datasets) ships with a fixed cardinality.
_DYNAMIC_SIZES = frozenset({"blobs", "moons", "s_curve", "smile", "high_dim", "multi_view"})


def _is_fixed_size_dataset(ds_name: str) -> bool:
    """Detect whether a dataset ignores n_samples/noise params.

    Uses a static name check — loading a 60K-image CIFAR-10 variant through
    ResNet-18 just to answer this boolean costs ~70 s and defeats the cache
    sharing with ``_cached_preview_data``.
    """
    return ds_name not in _DYNAMIC_SIZES


def _format_dataset_label(key: str) -> str:
    """Render a selectbox key as a human-readable label."""
    if key.startswith("group:"):
        gk = key.split(":", 1)[1]
        return DATASET_GROUPS[gk]["label"]
    return DATASET_LABELS.get(key, key)


@st.cache_resource(show_spinner="Initializing ACE Agent engine...")
def _get_supervisor():
    from ACE_Agent.agent_core.supervisor import ACESupervisor as _ACE
    return _ACE()


def _init_state() -> None:
    _init_cost_state()
    for k, v in {
        "current_session_id": str(uuid.uuid4()),
        "messages": [],
        "settings_store": SettingsStore(),
        "session_manager": SessionManager(),
    }.items():
        if k not in st.session_state:
            st.session_state[k] = v

    # Invalidate stale caches when supervisor code changes (prevents
    # @st.cache_resource from serving a stale singleton after updates).
    _version_key = "_ace_supervisor_cache_version"
    _current_version = _compute_cache_version()
    if st.session_state.get(_version_key) != _current_version:
        _get_supervisor.clear()
        _cached_preview_data.clear()
        _cached_preview_projection.clear()
        st.session_state[_version_key] = _current_version

    # Supervisor is lazily created in _handle_prompt() on first use.
    # This keeps page load fast (~2s UI render instead of ~17s blocking wait).


@st.cache_data(ttl=30)
def _read_trace_stats() -> dict[str, int]:
    """Read llm_trace.jsonl and compute cumulative stats from disk.
    Cached with 30s TTL to avoid re-parsing on every rerun."""
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


def _sidebar_ui():  # returns (LLMSettings, LLMSettings | None)
    """Render sidebar; return (primary_settings, fallback_settings)."""
    from ACE_Agent.tools.llm_client import LLMSettings as _LLMS
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
                _full = sm.get_full_session(s["id"])
                st.session_state.messages = _full if _full is not None else []
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
            model_options = ["(自定义模型...)" ] + p_cfg["models"]
            saved_model = ss.get("model", "")
            # Determine if saved model is a custom one (not in predefined list)
            is_custom_model = saved_model and saved_model not in p_cfg["models"]
            default_idx = 0 if is_custom_model else (
                model_options.index(saved_model) if saved_model in model_options else 0
            )
            model_choice = st.selectbox(
                "模型",
                model_options,
                index=default_idx,
                key="primary_model_sel",
            )
            if model_choice == "(自定义模型...)":
                model = st.text_input(
                    "自定义模型名称",
                    value=saved_model if is_custom_model else "",
                    placeholder="输入模型名称，如 deepseek-v4-pro",
                    key="primary_model_custom",
                )
            else:
                model = model_choice

            st.divider()

            # Fallback provider
            st.markdown("**Fallback Provider** (optional)")
            fallback_options = ["(disabled)"] + provider_names
            saved_fallback = ss.get("fallback_provider", "(disabled)")
            fallback_p = st.selectbox(
                "Fallback 供应商",
                fallback_options,
                index=(fallback_options.index(saved_fallback) if saved_fallback in fallback_options else 0),
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
                fb_model_options = ["(自定义模型...)" ] + fb_cfg["models"]
                fb_saved = ss.get("fallback_model", "")
                fb_is_custom = fb_saved and fb_saved not in fb_cfg["models"]
                fb_default_idx = 0 if fb_is_custom else (
                    fb_model_options.index(fb_saved) if fb_saved in fb_model_options else 0
                )
                fb_model_choice = st.selectbox(
                    "Fallback 模型",
                    fb_model_options,
                    index=fb_default_idx,
                    key="fallback_model_sel",
                )
                if fb_model_choice == "(自定义模型...)":
                    fallback_model = st.text_input(
                        "自定义 Fallback 模型名称",
                        value=fb_saved if fb_is_custom else "",
                        placeholder="输入模型名称",
                        key="fallback_model_custom",
                    )
                else:
                    fallback_model = fb_model_choice

            if st.button("保存配置", use_container_width=True):
                keys = ss.get("api_keys", {})
                keys[active_p] = api_key
                if fallback_p != "(disabled)":
                    keys[fallback_p] = fallback_api_key
                ss.save(
                    {
                        "active_provider": active_p,
                        "api_keys": keys,
                        "model": model,
                        "fallback_provider": fallback_p,
                        "fallback_model": fallback_model,
                    }
                )
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
            st.caption(f"Prices: input ${provider_cost['input']}/1K, output ${provider_cost['output']}/1K ({active_p})")

            if st.button("Clear Trace Log", use_container_width=True):
                if _TRACE_PATH.exists():
                    _TRACE_PATH.write_text("", encoding="utf-8")
                # Reset session cost counters
                for k in [
                    "llm_call_count",
                    "llm_retry_count",
                    "llm_prompt_tokens",
                    "llm_completion_tokens",
                    "llm_cost_usd",
                ]:
                    st.session_state[k] = 0
                st.rerun()

    # Build settings objects
    primary_settings = _LLMS(
        provider=active_p,
        base_url=p_cfg["base_url"],
        api_key=api_key,
        model=model,
        temperature=ss.get("temperature", 0.2),
    )

    fallback_settings: _LLMS | None = None
    if fallback_p != "(disabled)" and fallback_api_key and fallback_model:
        fb_cfg = DEFAULT_PROVIDERS[fallback_p]
        fallback_settings = _LLMS(
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

        # ---- resolved values (written to session_state for downstream use) ----
        resolved_ds_name: str = ""
        resolved_feature_mode: str = ""
        resolved_upload_mode: str = ""

        with t1:
            c1, c2 = st.columns(2)
            ds_key = c1.selectbox(
                "模板",
                [d for d in list_demo_datasets() if d != "custom"],
                format_func=_format_dataset_label,
            )

            # ---- Mode selector for grouped datasets ---------------------------
            if ds_key.startswith("group:"):
                group_key = ds_key.split(":", 1)[1]
                group = DATASET_GROUPS[group_key]
                modes = group["modes"]
                mode_labels = [m["label"] for m in modes]
                default_idx = next(
                    (i for i, m in enumerate(modes) if m["recommended"]), 0
                )
                chosen_label = c2.radio(
                    "特征模式",
                    mode_labels,
                    index=default_idx,
                    help="选择数据特征表示方式。推荐项通常聚类效果最好。",
                )
                chosen_mode = modes[mode_labels.index(chosen_label)]
                ds_name = chosen_mode["key"]
                resolved_feature_mode = chosen_mode.get(
                    "feature_mode", chosen_mode["key"].rsplit("_", 1)[-1]
                )
                c2.caption(chosen_mode["desc"])
            else:
                ds_name = ds_key
                resolved_feature_mode = ""
                c2.caption("")  # placeholder

            resolved_ds_name = ds_name

            is_fixed = _is_fixed_size_dataset(ds_name)
            if is_fixed:
                sc = 480
                noise = 0.06
                seed = 42
            else:
                sc = st.slider("样本量", 180, 2000, 480, 30, key="sc_slider")
            c3, c4 = st.columns(2)
            if is_fixed:
                c3.caption("(样本量/噪声/种子 由数据集固定)")
            else:
                noise = c3.slider("噪声", 0.01, 0.18, 0.06, 0.01)
                seed = c4.number_input("随机种子", 0, 9999, 42)

        with t2:
            uploaded_file = st.file_uploader(
                "上传 CSV/Excel", type=["csv", "xlsx", "xls"]
            )
            upload_analysis: dict | None = None
            if uploaded_file:
                # Load once for analysis (cached by file content)
                preview_ds_early = _cached_load_custom(
                    uploaded_file.getvalue(), uploaded_file.name
                )
                if preview_ds_early is not None:
                    nf = preview_ds_early.X.shape[1]
                    ns = preview_ds_early.X.shape[0]
                    upload_analysis = _analyze_uploaded_data(
                        preview_ds_early.X, nf, ns
                    )
                    detected_type = upload_analysis.get("detected_type", "tabular")
                    recs = upload_analysis.get("recommendations", [])

                    # Time-series detection (from companion .meta.json)
                    _ts_meta = (preview_ds_early.metadata or {})
                    if _ts_meta.get("is_time_series") and _ts_meta.get("ts_shape"):
                        _ts_T, _ts_F = _ts_meta["ts_shape"]
                        _ts_desc = _ts_meta.get("ts_description", "")
                        st.info(
                            f"🔍 检测到时序数据: **{_ts_T} 时间步 × {_ts_F} 特征**"
                            f"（展平 {_ts_T * _ts_F}D，{ns} 样本）\n\n"
                            f"将自动跳过 PCA、启用 DTW 距离度量。"
                            f"专家可使用 `TimeSeriesKMeans(metric='dtw')` 进行聚类。"
                            + (f"\n\n{_ts_desc}" if _ts_desc else "")
                        )

                    if recs:
                        # Show recommendations as a small card
                        if detected_type == "image":
                            img_s = upload_analysis.get("image_shape")
                            size_str = (
                                f"{img_s[0]}×{img_s[1]}×{img_s[2]}"
                                if img_s
                                else "unknown"
                            )
                            st.info(
                                f"🔍 检测到图像数据: **{size_str}**，{nf} 维 → {ns} 样本\n\n"
                                f"原始像素空间欧氏聚类在 ≥10D 时失效，建议提取 CNN 语义特征。"
                            )
                        elif nf > 500:
                            st.warning(
                                f"🔍 检测到高维数据: **{nf} 维**，{ns} 样本\n\n"
                                f"距离矩阵计算将极慢，建议降维后再聚类。"
                            )

                        rec_labels = [r["label"] for r in recs]
                        rec_default = next(
                            (i for i, r in enumerate(recs)
                             if r.get("priority") in ("strongly_recommended", "recommended")),
                            0,
                        )
                        chosen_rec = st.radio(
                            "处理方式",
                            rec_labels,
                            index=rec_default,
                            key="upload_mode_radio",
                        )
                        chosen_idx = rec_labels.index(chosen_rec)
                        resolved_upload_mode = recs[chosen_idx]["mode"]

                        # Show reason for the chosen option
                        st.caption(recs[chosen_idx]["reason"])
                    else:
                        resolved_upload_mode = "as_is"

        # ---- Store resolved values for _handle_prompt -------------------------
        st.session_state["_resolved_ds_name"] = resolved_ds_name
        st.session_state["_resolved_feature_mode"] = resolved_feature_mode
        st.session_state["_upload_feature_mode"] = resolved_upload_mode

        if st.button("Preview Data Distribution", use_container_width=True) or uploaded_file:
            st.divider()
            with st.spinner("正在绘制原始分布..."):
                preview_ds = None
                if uploaded_file:
                    import tempfile

                    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp:
                        tmp.write(uploaded_file.getvalue())
                        tmp_path = tmp.name
                    preview_ds = _cached_load_custom(uploaded_file.getvalue(), uploaded_file.name)
                    os.remove(tmp_path) if os.path.exists(tmp_path) else None
                    # Auto-detect image shape from feature dimensions
                    if preview_ds:
                        from ACE_Agent.tools.data_factory import _decompose_image_shape as _dis
                        nf = preview_ds.X.shape[1]
                        img_shape = _dis(nf)
                        if img_shape:
                            meta = dict(preview_ds.metadata or {})
                            meta["is_image"] = True
                            meta["original_shape"] = img_shape
                            preview_ds.metadata = meta
                else:
                    preview_ds = _cached_preview_data(ds_name, sc, noise, seed)

                if preview_ds:
                    st.subheader(f"数据预览: {preview_ds.display_name}")
                    # Compute preview projection (cached)
                    X_plot, n_dims, n_samples = _cached_preview_projection(
                        ds_name if not uploaded_file else uploaded_file.name,
                        sc if not uploaded_file else 0,
                        noise if not uploaded_file else 0.0,
                        seed,
                        bool(uploaded_file),
                        uploaded_file.getvalue() if uploaded_file else b'',
                        uploaded_file.name if uploaded_file else '',
                    )
                    pc1, pc2 = st.columns([0.7, 0.3])
                    with pc1:
                        _setup_matplotlib_fonts()
                        import matplotlib.pyplot as plt
                        fig, ax = plt.subplots(figsize=(8, 4.5))
                        if n_dims >= 2:
                            if n_dims > 2:
                                title = "PCA投影 (前2主成分)"
                            else:
                                title = "原始特征空间分布 (特征1 vs 特征2)"
                            ax.scatter(
                                X_plot[:, 0],
                                X_plot[:, 1],
                                c="black",
                                s=10,
                                alpha=0.5,
                                edgecolors="none",
                            )
                        else:
                            ax.scatter(
                                X_plot[:, 0] if n_dims == 1 else X_plot[:, 0],
                                np.zeros_like(X_plot[:, 0]) if n_dims == 1 else np.zeros_like(X_plot[:, 0]),
                                c="black",
                                s=10,
                                alpha=0.5,
                                edgecolors="none",
                            )
                            title = "1D 特征分布"
                        ax.set_title(title, fontsize=10)
                        ax.set_xlabel("Component 1")
                        ax.set_ylabel("Component 2")
                        ax.grid(True, linestyle=":", alpha=0.6)
                        st.pyplot(fig)
                        plt.close(fig)
                    with pc2:
                        st.write("**数据统计**")
                        st.metric("样本总数", n_samples)
                        st.metric("特征维度", n_dims)
                        if n_dims > 2:
                            st.caption("(使用PCA降至2D可视化)")
                        st.info("提示：请在下方输入指令来启动智能体聚类任务。")

        st.divider()
        _deep = st.checkbox(
            "深度模式（允许长时间等待）",
            value=False,
            help="勾选后可运行完整深度学习管线（AE/DEC/SelfLabel），"
                 "有机会获得更高 ARI。默认关闭以快速获得结果。",
        )
        st.session_state["deep_mode"] = _deep

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
            if m.get("report_summary"):
                _render_report_summary(m["report_summary"])
            elif m.get("report"):
                _render_report(m["report"])


@st.cache_data(max_entries=5)
def _cached_load_custom(file_bytes: bytes, file_name: str):  # type: ignore[return]
    import shutil
    import tempfile

    suffix = Path(file_name).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    # Streamlit copies uploads to a temp directory, but companion files
    # (e.g. heart_sounds_ready.meta.json) live in CWD or data/user_uploads/.
    # Copy them alongside the temp CSV so load_custom_dataset can detect
    # is_time_series etc.
    _meta_name = Path(file_name).with_suffix(".meta.json")
    _cwd_meta = Path.cwd() / _meta_name
    _uploads_meta = Path.cwd() / "data" / "user_uploads" / _meta_name
    _tmp_meta = Path(tmp_path).with_suffix(".meta.json")
    _source_meta = None
    if _cwd_meta.exists():
        _source_meta = _cwd_meta
    elif _uploads_meta.exists():
        _source_meta = _uploads_meta
    if _source_meta:
        shutil.copy2(str(_source_meta), str(_tmp_meta))

    try:
        ds = load_custom_dataset(tmp_path)
        ds.display_name = f"上传: {file_name}"
        return ds
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        if _tmp_meta.exists():
            _tmp_meta.unlink()


@st.cache_resource
def _setup_matplotlib_fonts() -> None:
    """Cache matplotlib Chinese font setup; avoids scanning system fonts on every plot."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans", "Arial"]
    plt.rcParams["axes.unicode_minus"] = False


@st.cache_data(max_entries=5)
def _cached_preview_data(ds_name: str, sc: int, noise: float, seed: int):
    """Cache dataset generation for preview; avoid re-generating on every rerun."""
    return generate_dataset(ds_name, n_samples=sc, noise=noise, random_state=seed)


@st.cache_data(max_entries=5)
def _cached_preview_projection(
    ds_name: str, sc: int, noise: float, seed: int,
    is_upload: bool, file_bytes: bytes, file_name: str,
):
    """Cache PCA/feature projection for preview visualization."""
    if is_upload:
        ds = _cached_load_custom(file_bytes, file_name)
    else:
        ds = _cached_preview_data(ds_name, sc, noise, seed)

    X = ds.X
    if hasattr(X, "toarray"):
        X = X.toarray()
    if hasattr(X, "values"):
        X = X.values
    X_np = np.array(X, dtype=np.float64)
    n_samples, n_dims = X_np.shape

    if n_dims > 2:
        from sklearn.decomposition import PCA
        X_plot = PCA(n_components=2, random_state=seed).fit_transform(X_np)
    else:
        X_plot = X_np

    return X_plot, n_dims, n_samples


def _handle_prompt(
    prompt: str,
    ds_name: str,
    sc: int,
    noise: float,
    seed: int,
    settings,  # LLMSettings
    fallback_settings,  # LLMSettings | None
    uploaded_file,  # type: ignore[type-arg]
) -> None:
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Persist user message immediately so history survives crash / early close
    _sm = st.session_state.session_manager
    _sm.save_session(
        st.session_state.current_session_id,
        st.session_state.messages,
        {"title": prompt[:30], "dataset": ds_name or ""},
    )
    # Lazy init: only create the heavy supervisor when user first runs a task
    if "supervisor" not in st.session_state:
        st.session_state["supervisor"] = _get_supervisor()
    supervisor = st.session_state.supervisor
    settings.deep_mode = st.session_state.get("deep_mode", False)

    # Fresh session detection: if this is the first user message (only the
    # one we just appended), clear any leftover memory from a previous
    # conversation in the cached supervisor singleton.
    if len(st.session_state.messages) == 1:
        supervisor.reset_state()

    with st.chat_message("assistant"):
        progress_placeholder = st.empty()
        elapsed_placeholder = st.empty()
        start_time = datetime.now()

        def _update_progress(msg: str, step: int = 0, total: int = 1) -> None:
            """Streamlit-safe progress updater."""
            elapsed = (datetime.now() - start_time).total_seconds()
            bar = "▮" * step + "▯" * (total - step) if total > 1 else ""
            progress_placeholder.markdown(
                f"⏳ {msg}\n\n{bar if bar else ''}\n\n_已耗时 {elapsed:.0f}s_"
            )

        with st.status("ACE Orchestrator 正在初始化...", expanded=True) as status:
            st.write("正在通过 MasterRouter 进行语义识别...")

            # Build router client with fallback support
            from ACE_Agent.tools.llm_client import UniversalLLMClient as _LLMC
            router_client = _LLMC(settings, fallback_settings, caller="router")
            ds_label = _format_dataset_label(ds_name) if ds_name else ""
            intent = supervisor.router.analyze_intent(
                prompt, supervisor.memory, settings, dataset_context=ds_label,
                client=router_client,
            )
            st.write(f"意图判定: **{intent.get('intent')}** ({intent.get('reasoning')})")
            _accumulate_cost(router_client)

            dataset = None
            if intent.get("intent") == "NEW_TASK":
                st.write("正在画像并准备数据集...")
                if uploaded_file:
                    dataset = _cached_load_custom(uploaded_file.getvalue(), uploaded_file.name)
                    upload_mode = st.session_state.get("_upload_feature_mode", "")
                    if dataset and upload_mode == "cnn_features":
                        try:
                            st.write("正在提取 CNN 特征 (ResNet-18)...")
                            orig_shape = None
                            nf = dataset.X.shape[1]
                            from ACE_Agent.tools.data_factory import _decompose_image_shape
                            shape = _decompose_image_shape(nf)
                            if shape:
                                orig_shape = shape
                            feats = _extract_cnn_features(dataset.X, orig_shape)
                            from ACE_Agent.agent_core.schemas import DatasetBundle as _DS
                            dataset = _DS(
                                name=dataset.name,
                                display_name=f"{dataset.display_name} (CNN特征)",
                                X=feats,
                                y=dataset.y,
                                description=f"上传图像数据，经 ResNet-18 提取 512D 语义特征。原始: {nf}D。",
                                shape_family="manifold",
                                feature_names=[f"cnn_{i}" for i in range(feats.shape[1])],
                                feature_mode="cnn_features",
                                metadata={
                                    **(dataset.metadata or {}),
                                    "is_image": True,
                                    "original_shape": orig_shape,
                                    "feature_mode": "cnn_features",
                                    "expected_clusters": (
                                        int(np.unique(dataset.y).size)
                                        if dataset.y is not None and len(np.unique(dataset.y)) >= 2
                                        else 3
                                    ),
                                },
                            )
                            st.write(f"特征提取完成: {nf}D → {feats.shape[1]}D")
                        except Exception as _cnn_exc:
                            st.warning(f"CNN 特征提取失败 ({_cnn_exc})，回退到原始像素。")
                else:
                    dataset = _cached_preview_data(ds_name, sc, noise, seed)

            n_experts = len(supervisor._DEFAULT_ACTIVE_EXPERTS)
            if dataset and dataset.X.shape[1] > 2:
                n_experts += 1  # dimension expert auto-activated for high-dim
            st.write(f"预计调度 {n_experts} 个专家并行执行 (含自愈重试)...")
            status.update(label=f"正在运行 {n_experts} 个专家...", state="running")

            import time as _time
            _start = _time.time()
            report = supervisor.run(
                dataset=dataset,
                user_prompt=prompt,
                llm_settings=settings,
                intent_data=intent,
                progress_callback=_update_progress,
            )
            _elapsed = _time.time() - _start
            n_results = len(report.results) if report.response_type == "CLUSTER_TASK" else 0
            status.update(
                label=f"完成 — {n_results} 个结果 (耗时 {_elapsed:.0f}s)",
                state="complete",
                expanded=False,
            )

        progress_placeholder.empty()
        elapsed_placeholder.empty()

        thought = "\n".join(report.decision_trace)
        st.markdown(report.llm_summary or report.executive_summary)
        if report.response_type == "CLUSTER_TASK":
            _render_report(report)
            _render_hitl_panel(report, supervisor, dataset, prompt, settings)
            # ---- Export clustering result ---------------------------------
            _best = report.ranking[0] if report.ranking else None
            if _best is not None and hasattr(_best, "labels"):
                _y_true = dataset.y
                _has_labels = _y_true is not None and len(_y_true) == len(_best.labels)
                _lines = []
                _header = "sample_index,cluster_label"
                if _has_labels:
                    _header += ",true_label"
                _lines.append(_header)
                for _i, _lbl in enumerate(_best.labels):
                    _row = f"{_i},{int(_lbl)}"
                    if _has_labels:
                        _row += f",{int(_y_true[_i])}"
                    _lines.append(_row)
                _ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                _fname = f"cluster_labels_{_best.algorithm_name}_{_ts}.csv"
                st.download_button(
                    label=f"下载聚类结果 ({_best.algorithm_name}, {len(_best.labels)} 样本)",
                    data="\n".join(_lines),
                    file_name=_fname,
                    mime="text/csv",
                    use_container_width=True,
                )

    # Build lightweight report summary for session storage (NOT the full dataclass).
    # The SupervisorReport dataclass contains AlgorithmRunResult objects with
    # full labels arrays (potentially 60K+ elements each).  We extract only
    # ranking text and top-level metrics so session files stay small.
    _report_summary = None
    if getattr(report, "response_type", None) == "CLUSTER_TASK":
        _ranking = getattr(report, "ranking", None) or []
        _top = _ranking[0] if _ranking else None
        _best_ari = -1.0
        _ranking_rows = []
        for i, item in enumerate(_ranking[:10]):
            _algo = getattr(item, "algorithm_name", "?")
            _expert = getattr(item, "expert_label", "")
            _m = getattr(item, "metrics", {}) or {}
            _ari = _m.get("ari", -1.0) if isinstance(_m, dict) else -1.0
            _nmi = _m.get("nmi", 0.0) if isinstance(_m, dict) else 0.0
            if _ari is not None and _ari > _best_ari:
                _best_ari = _ari
            _ranking_rows.append({
                "rank": i + 1,
                "algorithm": _algo,
                "expert": _expert,
                "ari": _ari,
                "nmi": _nmi,
            })
        _ds = getattr(report, "dataset", None)
        _ds_name = getattr(_ds, "display_name", "") if _ds else ""
        _winner_plot = str(getattr(_top, "plot_path", "")) if _top else ""
        _dataset_plot = str(getattr(report, "dataset_plot_path", ""))
        _report_summary = {
            "dataset_name": _ds_name,
            "winner_algorithm": getattr(_top, "algorithm_name", "") if _top else "",
            "winner_ari": _best_ari,
            "dataset_plot_path": _dataset_plot,
            "winner_plot_path": _winner_plot,
            "is_ensemble": (getattr(_top, "algorithm_name", "") if _top else "") == "EnsembleConsensus",
            "ranking_rows": _ranking_rows,
            "summary": (getattr(report, "llm_summary", "") or getattr(report, "executive_summary", ""))[:2000],
            "n_results": len(getattr(report, "results", []) or []),
        }

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": report.llm_summary or report.executive_summary,
            "thought": thought,
            "report_summary": _report_summary,
        }
    )
    st.session_state.session_manager.save_session(
        st.session_state.current_session_id,
        st.session_state.messages,
        {
            "title": prompt[:30],
            "dataset": dataset.display_name if dataset else "追问",
        },
    )


def _safe_plot_path(path_obj) -> str | None:
    """Return str(path) if it points to an existing file, otherwise None."""
    if path_obj is None:
        return None
    p = Path(str(path_obj))
    if p.name and p.exists() and p.is_file():
        return str(p)
    return None


def _render_audit_card(audit: dict) -> None:  # type: ignore[type-arg]
    """Render Critic post-hoc audit report as a styled info card."""
    endorsement = audit.get("endorsement", "?")
    confidence = audit.get("confidence_level", 0.0)
    stability = audit.get("stability_score", 0.0)
    hopkins = audit.get("hopkins", 0.0)
    overfitting = audit.get("overfitting_risk", "unknown")
    k_consistency = audit.get("winner_k_consistency", False)
    findings = audit.get("findings", [])
    recommendation = audit.get("recommendation", "")

    endorsement_icon = {"endorsed": "✅", "qualified": "⚠️", "qualified_with_warning": "🔴"}.get(endorsement, "❓")
    endorsement_label = {"endorsed": "通过", "qualified": "有条件通过", "qualified_with_warning": "需要关注"}.get(endorsement, "未知")

    with st.expander(f"{endorsement_icon} 独立审计: {endorsement_label} (置信度 {confidence:.0%})", expanded=True):
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("综合置信度", f"{confidence:.0%}")
        m2.metric("Bootstrap 稳定性", f"{stability:.2f}")
        m3.metric("Hopkins 趋势", f"{hopkins:.2f}")
        m4.metric("过拟合风险", overfitting, delta_color="off" if overfitting == "low" else "inverse",
                  help=f"风险评估: {overfitting}")

        st.caption(f"聚类数一致性: {'✅ 一致' if k_consistency else '⚠️ 与 CVI 共识不一致'}")

        if findings:
            st.markdown("**审计发现**")
            for f_text in findings:
                st.markdown(f"- {f_text}")

        if recommendation:
            st.info(f"**建议**: {recommendation}")


def _render_ensemble_metrics(top) -> None:
    """Render ensemble-specific metrics and co-association heatmap."""
    import matplotlib.pyplot as plt
    import numpy as np
    params = top.params if hasattr(top, "params") else top.get("params", {})
    coassoc = params.get("coassoc_matrix") if isinstance(params, dict) else None
    if coassoc is None:
        return

    metrics = top.metrics if hasattr(top, "metrics") else top.get("metrics", {})
    n_fused = metrics.get("n_experts_fused", "?")
    entropy = metrics.get("entropy_of_agreement", 0.0)
    agreement = metrics.get("agreement", 0.0)
    expert_names = params.get("expert_names", [])

    # Compute readablity hints
    n = coassoc.shape[0]
    off_diag = coassoc.copy()
    np.fill_diagonal(off_diag, np.nan)
    off_diag_mean = float(np.nanmean(off_diag))

    st.markdown(
        f"**集成共识**: 融合 {n_fused} 位专家 ({', '.join(expert_names) if expert_names else '?'}) | "
        f"一致性 {agreement:.1%} | 信息熵 {entropy:.3f}"
    )
    st.caption(
        f"{n} 个数据点 | "
        f"非对角线平均共现频率: {off_diag_mean:.2f} "
        f"({'专家间高度共识' if off_diag_mean < 0.3 or off_diag_mean > 0.7 else '专家间分歧较大'})"
    )

    fig, ax = plt.subplots(figsize=(6.5, 6))
    im = ax.imshow(coassoc, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)

    # Axis ticks: show actual data point ranges
    if n <= 20:
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels([str(i) for i in range(n)], fontsize=6)
        ax.set_yticklabels([str(i) for i in range(n)], fontsize=6)
    else:
        step = max(n // 8, 1)
        tick_pos = list(range(0, n, step))
        ax.set_xticks(tick_pos)
        ax.set_yticks(tick_pos)
        ax.set_xticklabels([str(i) for i in tick_pos], fontsize=7)
        ax.set_yticklabels([str(i) for i in tick_pos], fontsize=7)

    ax.set_title("专家共现矩阵 (Co-association Matrix)", fontsize=12, weight="bold")
    ax.set_xlabel("数据点索引", fontsize=9)
    ax.set_ylabel("数据点索引", fontsize=9)

    cbar = plt.colorbar(im, ax=ax, shrink=0.82)
    cbar.set_label("共现频率\n(1.0 = 所有专家同意两点同簇)", fontsize=8)

    # Annotate key cells for small matrices
    if n <= 15:
        for i in range(n):
            for j in range(n):
                ax.text(j, i, f"{coassoc[i, j]:.2f}", ha="center", va="center",
                        fontsize=5, color="black" if coassoc[i, j] > 0.6 else "white")

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def _render_report_summary(s: dict) -> None:
    """Render from a lightweight report summary dict (new format, low memory)."""
    ranking = s.get("ranking_rows", [])
    if not ranking:
        return
    st.subheader(f"分析报告: {s.get('dataset_name', '?')}")

    best_ari = s.get("winner_ari", -1.0)
    top = ranking[0]
    top_nmi = top.get("nmi", 0.0) if isinstance(top, dict) else 0.0

    c = st.columns(4)
    if best_ari >= 0 and best_ari < 0.2:
        c[0].metric("优胜算法", "None / Attempting Rescue")
        st.error(
            f"### NO VALID CLUSTERS\n\n"
            f"数据最强 ARI 仅 **{best_ari:.4f}** (< 0.2)，"
            f"所有算法结果接近随机分配。"
        )
    else:
        c[0].metric("优胜算法", top.get("algorithm", "?"))
    c[1].metric("ARI", f"{best_ari:.3f}" if best_ari >= 0 else "N/A")
    c[2].metric("NMI", f"{top_nmi:.3f}")
    c[3].metric("评分", f"{best_ari:.3f}")

    if len(ranking) > 1:
        st.markdown("### 算法排名")
        rows = []
        for item in ranking:
            if not isinstance(item, dict):
                continue
            ari_v = item.get("ari", -1.0)
            rows.append({
                "排名": item.get("rank", "?"),
                "算法": item.get("algorithm", "?"),
                "ARI": f"{float(ari_v):.4f}" if ari_v is not None and ari_v >= 0 else "N/A",
                "NMI": f"{float(item.get('nmi') or 0.0):.4f}" if item.get("nmi") is not None else "N/A",
                "专家来源": item.get("expert", ""),
            })
        st.dataframe(
            pd.DataFrame(rows),
            use_container_width=True,
            hide_index=True,
            column_config={"排名": st.column_config.NumberColumn(width="small")},
        )

    _summary_text = s.get("summary", "")
    if _summary_text:
        st.write(_summary_text)

    _raw = _safe_plot_path(s.get("dataset_plot_path"))
    _top = _safe_plot_path(s.get("winner_plot_path"))
    if _raw or _top:
        cols = st.columns(2)
        if _raw:
            cols[0].image(_raw, caption="原始分布")
        else:
            cols[0].warning("原始分布图不可用")
        if _top:
            cols[1].image(_top, caption="最优聚类结果")
        else:
            cols[1].info("该算法未生成聚类可视化图")
    else:
        st.info("可视化图像暂不可用（可能未保存或已过期）")


def _render_report(r) -> None:  # type: ignore[type-arg]
    ranking = r.ranking if hasattr(r, "ranking") else r["ranking"]
    if not ranking:
        # Cached report or error report — no per-algorithm ranking to render
        return
    dataset = r.dataset if hasattr(r, "dataset") else r["dataset"]
    top = ranking[0]
    ds_name_display = dataset.display_name if hasattr(dataset, "display_name") else dataset["display_name"]
    st.subheader(f"分析报告: {ds_name_display}")

    # ---- Phase 6: Find best ARI across all ranking entries ----
    best_ari = -1.0
    for item in ranking:
        m = item.metrics if hasattr(item, "metrics") else item.get("metrics", {})
        ari_val = m.get("ari", -1.0)
        if ari_val is not None and ari_val > best_ari:
            best_ari = ari_val

    algo_name = top.algorithm_name if hasattr(top, "algorithm_name") else top["algorithm_name"]
    top_metrics = top.metrics if hasattr(top, "metrics") else top.get("metrics", {})
    score = float((top_metrics.get("score") if isinstance(top_metrics, dict) else top_metrics["score"]) or 0.0)

    c = st.columns(4)
    if best_ari >= 0 and best_ari < 0.2:
        c[0].metric("优胜算法", "None / Attempting Rescue")
        st.error(
            f"### ⛔ NO VALID CLUSTERS — 聚类失败\n\n"
            f"数据最强 ARI 仅 **{best_ari:.4f}** (< 0.2)，"
            f"所有算法结果接近随机分配。欧氏空间方法无法捕捉数据内在结构。\n"
            f"系统已自动触发深度管线（流形嵌入 + 自编码器）进行救助。"
        )
    else:
        c[0].metric("优胜算法", algo_name)
    best_nmi = float(top_metrics.get("nmi") or 0.0)
    c[1].metric("ARI", f"{best_ari:.3f}" if best_ari >= 0 else "N/A")
    c[2].metric("NMI", f"{best_nmi:.3f}")
    c[3].metric("评分", f"{score:.3f}")

    # ---- Per-Algorithm Ranking Table ----
    if len(ranking) > 1:
        st.markdown("### 算法排名")
        rows = []
        for i, item in enumerate(ranking):
            algo = item.algorithm_name if hasattr(item, "algorithm_name") else item["algorithm_name"]
            expert = item.expert_label if hasattr(item, "expert_label") else item["expert_label"]
            m = item.metrics if hasattr(item, "metrics") else item["metrics"]
            s = float(m.get("score") or 0.0)
            ari_v = m.get("ari", -1.0)
            nmi_v = m.get("nmi", 0.0)
            rows.append({
                "排名": i + 1,
                "算法": algo,
                "专家来源": expert,
                "ARI": f"{float(ari_v):.4f}" if ari_v is not None and ari_v >= 0 else "N/A",
                "NMI": f"{float(nmi_v):.4f}" if nmi_v is not None else "N/A",
                "评分": f"{s:.4f}",
            })
        st.dataframe(
            pd.DataFrame(rows),
            use_container_width=True,
            hide_index=True,
            column_config={"排名": st.column_config.NumberColumn(width="small")},
        )

    # ---- Critic Audit Card (if available) ----
    audit = r.audit_report if hasattr(r, "audit_report") else r.get("audit_report")
    if audit and isinstance(audit, dict):
        _render_audit_card(audit)

    # ---- Ensemble Co-association Heatmap (if winner is EnsembleConsensus) ----
    if algo_name == "EnsembleConsensus":
        _render_ensemble_metrics(top)

    # ---- Phase 3: Graph-based visualizations ----
    top_params = top.params if hasattr(top, "params") else top.get("params", {})

    # Graph disagreement heatmap (when high disagreement exists)
    if top_metrics.get("high_disagreement_ratio", 0) > 0.1:
        _render_disagreement_heatmap(top, dataset)

    # Graph connectivity overlay (when graph metrics available)
    if top_metrics.get("graph_connectivity_agreement") is not None:
        _render_graph_metrics_card(top_metrics, top_params)

    cols = st.columns(2)
    raw_plot = _safe_plot_path(r.dataset_plot_path if hasattr(r, "dataset_plot_path") else r["dataset_plot_path"])
    if raw_plot:
        cols[0].image(raw_plot, caption="原始分布")
    else:
        cols[0].warning("原始分布图不可用")
    top_plot = _safe_plot_path(top.plot_path if hasattr(top, "plot_path") else top["plot_path"])
    if top_plot:
        cols[1].image(top_plot, caption="最优聚类结果")
    else:
        cols[1].info("该算法未生成聚类可视化图")


def _render_disagreement_heatmap(top, dataset) -> None:
    """Render points with high expert disagreement in red."""
    import matplotlib.pyplot as plt
    import numpy as np

    labels = top.labels if hasattr(top, "labels") else top.get("labels")
    if labels is None:
        return
    lbls = np.array(labels, dtype=int)
    X_np = np.array(dataset.X if hasattr(dataset, "X") else dataset.get("X"))
    if X_np.shape[1] > 2:
        from sklearn.decomposition import PCA
        X_vis = PCA(n_components=2, random_state=42).fit_transform(X_np)
    else:
        X_vis = X_np

    st.markdown("### 专家分歧热力图")
    st.caption("红色：≥50% 专家与共识不一致的区域")
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(X_vis[:, 0], X_vis[:, 1], c=lbls, cmap="tab10", s=2, alpha=0.3, edgecolors="none")
    ax.set_title("Cluster Consensus — Disagreement Map", fontsize=10)
    fig.tight_layout()
    with st.expander("分歧热力图", expanded=False):
        st.pyplot(fig)
    plt.close(fig)


def _render_graph_metrics_card(top_metrics: dict, top_params: dict) -> None:
    """Render Phase 3 graph-aware metrics card."""
    with st.expander("图结构指标 (Topology-Aware)", expanded=False):
        cm = st.columns(4)
        cm[0].metric("图连通一致性", f"{top_metrics.get('graph_connectivity_agreement', 0):.3f}")
        cm[1].metric("高分歧区域", f"{top_metrics.get('high_disagreement_ratio', 0):.1%}")
        cm[2].metric("融合专家数", str(top_metrics.get('n_experts_fused', '?')))
        cm[3].metric("共识K", str(top_metrics.get('k_consensus', '?')))
        if top_params.get("graph_agreement"):
            st.caption(f"图连通一致性: {top_params['graph_agreement']:.3f} — "
                       "越高说明聚类越符合图的邻域结构")
        if top_params.get("disagreement_ratio", 0) > 0.05:
            st.warning(
                f"高分歧区域占比 {top_params['disagreement_ratio']:.1%}。"
                "这些区域的点在不同专家间聚类归属不一致，"
                "建议在高分歧区域增加密度算法或图结构算法。"
            )


def _render_hitl_panel(report, supervisor, dataset, prompt, settings) -> None:
    """Render HITL label correction panel for Phase 2.3.

    Shows the best result's cluster labels in an editable text area,
    and a re-trigger button that re-dispatches experts with the
    user-corrected labels as reference constraints.
    """
    ranking = report.ranking if hasattr(report, "ranking") else report.get("ranking", [])
    if not ranking:
        return

    best = ranking[0]
    labels = best.labels if hasattr(best, "labels") else best.get("labels")
    if labels is None or len(labels) == 0:
        return

    algo_name = best.algorithm_name if hasattr(best, "algorithm_name") else best.get("algorithm_name", "")
    n = len(labels)

    with st.expander(f"🔧 人工标注修正 (HITL) — 当前最优: {algo_name} ({n} 个数据点)", expanded=False):
        st.caption(
            "修改下方标签后点击\"重新分析\"，系统将以你的标注为参考约束，"
            "重新调度所有专家进行聚类。"
        )

        # Editable text area with comma-separated labels
        default_text = ", ".join(str(int(lb)) for lb in labels)
        corrected_text = st.text_area(
            f"参考标签（{n} 个，逗号分隔，整数）",
            value=default_text,
            height=120,
            key="hitl_label_editor",
            help="修改你认为错误的标签值，保持逗号分隔格式。",
        )

        col1, col2 = st.columns([1, 3])
        if col1.button("⚡ 以修正标签重新分析", type="primary", use_container_width=True):
            try:
                corrected_labels = [int(x.strip()) for x in corrected_text.split(",") if x.strip()]
                if len(corrected_labels) != n:
                    st.error(f"标签数量不匹配：期望 {n} 个，收到 {len(corrected_labels)} 个。请保持数据点数不变。")
                    return
            except ValueError:
                st.error("标签格式错误：请确保所有标签都是整数（逗号分隔）。")
                return

            constraints = {"reference_labels": corrected_labels}
            with st.status("HITL 约束重分析正在进行...", expanded=True) as hitl_status:
                st.write("正在以人工标注为约束重新调度专家池...")
                hitl_report = supervisor.run(
                    dataset=dataset,
                    user_prompt=prompt,
                    llm_settings=settings,
                    constraints=constraints,
                )
                hitl_status.update(label="HITL 重分析完成", state="complete", expanded=False)

            st.markdown("---")
            st.markdown(hitl_report.llm_summary or hitl_report.executive_summary)
            if hitl_report.response_type == "CLUSTER_TASK":
                _render_report(hitl_report)


if __name__ == "__main__":
    main()
