# ACE Agent (Automated Clustering Expert) 🛡️

ACE Agent 是一个生产级的、具备**自愈能力**的多智能体自主聚类分析系统。它将大语言模型（LLM）作为大脑，实现了从数据画像、意图路由到代码生成、自动调试及学术报告生成的全流程闭环。

## 🚀 核心架构与特性

### 1. Orchestrator-Worker 编排架构
- **MasterRouter (决策中枢)**: 摒弃传统关键词匹配，采用纯语义识别。精准区分“新任务分析”与“历史追问解析”。
- **ACESupervisor (全局编排)**: 维护会话状态与记忆，协调不同领域的专家 Agent。
- **Self-Healing Experts (自愈专家)**: 核心专家 Agent (质心、拓扑等) 遵循 **Think-Act-Fix** 循环。当代码在沙箱中运行报错（如维度不匹配、API 变更）时，Agent 会分析 Traceback 并自动修复代码，上限 3 次重试。

### 2. 深度 EDA 与可视化
- **Black-Dot 预览**: 支持上传数据后的即时特征分布预览，采用黑色点状图进行无偏展示。
- **中文原生支持**: 针对 Windows 环境优化的字体注入技术，彻底解决 Matplotlib 图表乱码。
- **性能优化**: 引入 `st.cache_data` 缓存机制，避免大数据集重复加载导致的性能损耗。

### 3. 安全沙箱执行
- **隔离环境**: 在受限的 Python 命名空间内执行生成代码，拦截 `__import__` 等高危操作。
- **数据注入**: 无需文件读写，直接在内存中对接 NumPy 对象，规避文件路径报错。

---

## 🏗️ 核心代码逻辑分布

- `agent_core/router.py`: LLM 意图判定逻辑。
- `agent_core/supervisor.py`: 专家调度与结果聚合。
- `expert_sub_agents/base.py`: **关键** - 定义了专家的自愈循环框架。
- `tools/coder_sandbox.py`: 包含安全策略与中文字体配置。
- `web_demo.py`: 基于 Streamlit 的交互式工作台。

---

## 🛠️ 部署指南

### 环境依赖
```bash
# 推荐 Python 3.10+
pip install -r requirements.txt
pip install charset-normalizer # 解决 Requests 依赖警告
```

### 运行
```bash
streamlit run web_demo.py
```

---

## 📑 工程说明书
项目根目录下包含 `ACE_Agent_Engineering_Handbook.tex`，该文档以学术级规格详细记录了系统的通信协议、自愈数学模型及复现路径。

---

## 🔮 路线图 (Roadmap)
- [ ] **RAG 理论增强 (Current Focus)**: 引入聚类专业知识库，使 Agent 具备引用学术文献的能力。
- [ ] **Critic Agent (评审机制)**: 引入独立审计 Agent，对聚类质量进行数学层面的二次校验。
- [ ] **多视图联动**: 支持 T-SNE / UMAP 等高级降维算法的自动选择与对比。
