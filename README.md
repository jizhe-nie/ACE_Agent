# ACE Agent (Automated Clustering Expert) 🛡️

ACE Agent 是一个生产级的多智能体自主聚类分析系统。它不仅是一个自动化脚本，而是一个模拟资深数据科学家思维链条的智能体系统：**语义识别 → 任务编排 → 专家自愈执行 → 多维评估 → 知识发现。**

## 🚀 核心特性

### 1. 基于编排器 (Orchestrator) 的多代理架构
系统由主控 (Master) 统一调度，专家 (Specialists) 各司其职：
- **MasterRouter**: 纯 LLM 驱动的语义中枢，精准识别“新任务”与“深度追问”，彻底废除硬编码逻辑。
- **Self-Correction Experts**: 专家智能体具备 **Think-Act-Fix** 循环能力。代码执行失败时，会自动结合 Traceback 错误信息进行自我 Debug，直至产出正确结果。
- **Orchestrator (Supervisor)**：统筹全局生命周期，管理代理间的消息流与状态同步。

### 2. 沉浸式交互体验
- **透明化思考流 (Thinking Trace)**：实时展示主控决策与专家纠错过程，让 AI 的每一步行动都清晰可见。
- **持久化会话管理**：支持多轮对话历史保存与恢复（基于 JSON 存储，规划迁移至 SQLite）。
- **多供应商配置中心**：内置支持 DeepSeek, 通义千问 (DashScope), Kimi (Moonshot), OpenAI, Gemini 等国内外主流大模型。

### 3. 专业级分析与报告
- **自愈式代码沙箱**: 在受限环境下安全执行 Python 代码，支持自动安装缺失逻辑与 Matplotlib 中文乱码修复。
- **学术级报告生成**: 自动产出包含决策轨迹、指标排行、可视化分布的 LaTeX 与 PDF 报告。
- **全量数据集支持**: 覆盖从合成数据（Moons, Smile）到真实世界大规模数据（MNIST, 20Newsgroups, UCI Mfeat）。

---

## 🛠️ 快速开始

### 1. 环境准备
```bash
conda create -n ACE_Agent python=3.10
conda activate ACE_Agent
pip install -r requirements.txt
cp .env.example .env  # 在此处配置你的 API Key
```

### 2. 启动 Web 交互界面
```bash
streamlit run web_demo.py
```
*在侧边栏底部的 ⚙️ Settings 中一键配置并切换不同的模型供应商。*

---

## 🏗️ 系统架构图

1.  **Agent Core (`agent_core/`)**:
    - `router.py`: LLM 语义路由与意图决策。
    - `supervisor.py`: 核心编排器，负责任务分发与结果汇编。
2.  **Expert Sub-Agents (`expert_sub_agents/`)**:
    - `base.py`: 定义自愈循环逻辑。
    - `centroid_expert.py`: 质心专家。
    - `topology_expert.py`: 拓扑专家。
3.  **Tools (`tools/`)**:
    - `llm_client.py`: 统一供应商适配器。
    - `coder_sandbox.py`: 具备中文支持的受限执行环境。
    - `settings_store.py`: 持久化存储引擎。

---

## 📝 最近更新
- ✅ **架构跨越**: 完成从“自动化脚本”到“Orchestrator-Agent”模式的深度重构。
- ✅ **专家自愈**: 实现了子代理的代码报错自动重写功能 (Max 3 retries)。
- ✅ **全界面汉化**: 针对国内科研使用场景，优化了全中文交互界面。
- ✅ **可视化增强**: 彻底解决了聚类图表在 Win32 环境下的中文乱码问题。
- ✅ **历史存档**: 侧边栏支持会话历史列表显示与持久化保存。
