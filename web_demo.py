from __future__ import annotations
from pathlib import Path
import sys, os, uuid, json
if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

import os
os.environ["OMP_NUM_THREADS"] = "1" # 抑制 KMeans 警告

import pandas as pd

import streamlit as st
import matplotlib.pyplot as plt
from ACE_Agent.agent_core.supervisor import ACESupervisor

from ACE_Agent.tools.data_factory import (
    DATASET_LABELS, generate_dataset, infer_dataset_from_prompt,
    list_demo_datasets, load_custom_dataset
)
from ACE_Agent.tools.llm_client import LLMSettings
from ACE_Agent.tools.settings_store import SettingsStore, SessionManager, DEFAULT_PROVIDERS

st.set_page_config(page_title="ACE Agent", layout="wide", initial_sidebar_state="expanded")

@st.cache_resource
def _get_supervisor():
    sv = ACESupervisor()
    return sv

def _init_state():
    for k, v in {"current_session_id": str(uuid.uuid4()), "messages": [], 
                 "supervisor": _get_supervisor(), "settings_store": SettingsStore(), 
                 "session_manager": SessionManager()}.items():
        if k not in st.session_state: st.session_state[k] = v

def _sidebar_ui():
    sm, ss = st.session_state.session_manager, st.session_state.settings_store
    with st.sidebar:
        st.title("🛡️ ACE Agent")
        if st.button("➕ 新建对话", use_container_width=True):
            st.session_state.current_session_id, st.session_state.messages = str(uuid.uuid4()), []
            st.rerun()
        st.divider()
        st.subheader("历史会话")
        for s in sm.sessions[:10]:
            col1, col2 = st.columns([0.8, 0.2])
            if col1.button(s.get("metadata", {}).get("title", s["id"][:8]), key=f"s_{s['id']}", use_container_width=True):
                st.session_state.current_session_id, st.session_state.messages = s['id'], s['messages']
                st.rerun()
            if col2.button("🗑️", key=f"d_{s['id']}"): sm.delete_session(s['id']); st.rerun()
        for _ in range(5): st.sidebar.write("")
        with st.popover("⚙️ 模型配置", use_container_width=True):
            active_p = st.selectbox("供应商", list(DEFAULT_PROVIDERS.keys()), 
                                    index=list(DEFAULT_PROVIDERS.keys()).index(ss.get("active_provider", "DeepSeek")))
            p_cfg = DEFAULT_PROVIDERS[active_p]
            api_key = st.text_input("API Key", value=ss.get("api_keys", {}).get(active_p, ""), type="password")
            model = st.selectbox("模型", p_cfg["models"], index=0 if ss.get("model") not in p_cfg["models"] else p_cfg["models"].index(ss.get("model")))
            if st.button("保存", use_container_width=True):
                keys = ss.get("api_keys", {}); keys[active_p] = api_key
                ss.save({"active_provider": active_p, "api_keys": keys, "model": model})
                st.rerun()
    return LLMSettings(provider=active_p, base_url=p_cfg["base_url"], api_key=api_key, model=model, temperature=ss.get("temperature", 0.2))

def main():
    _init_state()
    settings = _sidebar_ui()
    st.title("ACE Agent")
    st.caption("基于 Orchestrator 架构的自愈式多代理聚类系统")
    with st.expander("📊 数据配置", expanded=not st.session_state.messages):
        t1, t2 = st.tabs(["内置数据", "上传数据"])
        with t1:
            c1, c2 = st.columns(2)
            ds_name = c1.selectbox("模板", [d for d in list_demo_datasets() if d != "custom"], format_func=lambda v: DATASET_LABELS[v])
            sc = c2.slider("样本量", 180, 2000, 480, 30)
            c3, c4 = st.columns(2)
            noise = c3.slider("噪声", 0.01, 0.18, 0.06, 0.01)
            seed = c4.number_input("随机种子", 0, 9999, 42)
        with t2: uploaded_file = st.file_uploader("上传 CSV/Excel", type=["csv", "xlsx", "xls"])

        # 新增：数据即时预览功能
        if st.button("🔍 预览数据分布", use_container_width=True) or uploaded_file:
            st.divider()
            with st.spinner("正在绘制原始分布..."):
                preview_ds = None
                if uploaded_file:
                    import tempfile
                    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp:
                        tmp.write(uploaded_file.getvalue()); tmp_path = tmp.name
                    preview_ds = load_custom_dataset(tmp_path)
                    os.remove(tmp_path)
                else:
                    preview_ds = generate_dataset(ds_name, n_samples=sc, noise=noise, random_state=seed)
                
                if preview_ds:
                    st.subheader(f"数据预览: {preview_ds.display_name}")
                    pc1, pc2 = st.columns([0.7, 0.3])
                    with pc1:
                        # 解决中文乱码
                        import platform
                        if platform.system() == "Windows":
                            plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
                        plt.rcParams['axes.unicode_minus'] = False
                        
                        fig, ax = plt.subplots(figsize=(8, 4.5))
                        # 如果是高维数据，提醒用户只展示了前两个特征
                        ax.scatter(preview_ds.X[:, 0], preview_ds.X[:, 1], c='black', s=10, alpha=0.5, edgecolors='none')
                        ax.set_title(f"原始特征空间分布 (特征1 vs 特征2)", fontsize=10)
                        ax.set_xlabel("Feature 1")
                        ax.set_ylabel("Feature 2")
                        ax.grid(True, linestyle=':', alpha=0.6)
                        st.pyplot(fig)
                        plt.close(fig) # 释放内存
                    with pc2:
                        st.write("📈 **数据统计**")
                        st.metric("样本总数", preview_ds.X.shape[0])
                        st.metric("特征维度", preview_ds.X.shape[1])
                        st.info("提示：请在下方输入指令（如：'分析这个数据集'）来启动智能体聚类任务。")

    _render_messages()
    if prompt := st.chat_input("输入指令，例如：使用谱聚类分析这个数据集..."):
        _handle_prompt(prompt, ds_name, sc, noise, seed, settings, uploaded_file)
        st.rerun()

def _render_messages():
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            if m.get("thought"):
                with st.expander("💭 代理思考流 (Thinking Trace)", expanded=False):
                    for line in m["thought"].split("\n"):
                        if "【RAG】" in line: st.info(line) # 突出显示 RAG 信息
                        elif "失败" in line or "错误" in line: st.error(line)
                        elif "成功" in line: st.success(line)
                        else: st.write(line)
            st.markdown(m["content"])
            if m.get("report"): _render_report(m["report"])

@st.cache_data
def _cached_load_custom(file_bytes, file_name):
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
        if os.path.exists(tmp_path): os.remove(tmp_path)

def _handle_prompt(prompt, ds_name, sc, noise, seed, settings, uploaded_file):
    st.session_state.messages.append({"role": "user", "content": prompt})
    supervisor = st.session_state.supervisor
    
    with st.chat_message("assistant"):
        with st.status("🛠️ ACE Orchestrator 正在运行...", expanded=True) as status:
            st.write("🔍 正在通过 MasterRouter 进行语义识别...")
            intent = supervisor.router.analyze_intent(prompt, supervisor.memory, settings)
            st.write(f"意图判定: **{intent.get('intent')}** ({intent.get('reasoning')})")
            
            dataset = None
            if intent.get("intent") == "NEW_TASK":
                st.write("📊 正在画像并准备数据集...")
                if uploaded_file:
                    dataset = _cached_load_custom(uploaded_file.getvalue(), uploaded_file.name)
                else:
                    inferred = infer_dataset_from_prompt(prompt)
                    dataset = generate_dataset(inferred or ds_name, n_samples=sc, noise=noise, random_state=seed)
            
            st.write("🤖 正在激活专家 Agent 并监控自愈执行...")
            report = supervisor.run(dataset=dataset, user_prompt=prompt, llm_settings=settings, intent_data=intent)
            status.update(label="✅ 分析任务完成", state="complete", expanded=False)
            
        thought = "\n".join(report.decision_trace)
        st.markdown(report.llm_summary or report.executive_summary)
        if report.response_type == "CLUSTER_TASK": _render_report(report)
        
    # 3. 结果保存
    st.session_state.messages.append({
        "role": "assistant", 
        "content": report.llm_summary or report.executive_summary, 
        "thought": thought, 
        "report": report if report.response_type == "CLUSTER_TASK" else None
    })
    st.session_state.session_manager.save_session(st.session_state.current_session_id, st.session_state.messages, {"title": prompt[:30], "dataset": dataset.display_name if dataset else "追问"})

def _render_report(r):
    ranking = r.ranking if hasattr(r, 'ranking') else r['ranking']
    dataset = r.dataset if hasattr(r, 'dataset') else r['dataset']
    top = ranking[0]
    st.subheader(f"分析报告: {dataset.display_name if hasattr(dataset, 'display_name') else dataset['display_name']}")
    c = st.columns(4)
    c[0].metric("优胜算法", top.algorithm_name if hasattr(top, 'algorithm_name') else top['algorithm_name'])
    c[2].metric("评分", f"{float(top.metrics['score'] if hasattr(top, 'metrics') else top['metrics']['score']):.3f}")
    cols = st.columns(2)
    cols[0].image(str(r.dataset_plot_path if hasattr(r, 'dataset_plot_path') else r['dataset_plot_path']), caption="原始分布")
    cols[1].image(str(top.plot_path if hasattr(top, 'plot_path') else top['plot_path']), caption="最优聚类结果")

if __name__ == "__main__": main()
