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

## 🔮 发展路线图 (Roadmap 2024-2025)

ACE Agent 的愿景是成为**数据科学领域的“自动驾驶仪”**。我们将分阶段实现从“自动化分析”到“智能化共创”的跨越：

### 第一阶段：深度专家矩阵 (Short-term)
- **[TODO] 维度专家 (Dimension Expert)**: 集成 PCA、t-SNE、UMAP 及 AutoEncoder。针对高维、稀疏数据自动决策降维路径，解决“维度灾难”。
- **[TODO] 评价专家 (Critic Agent)**: 引入独立审计机制。不仅看轮廓系数，还将引入稳定性检验、重采样交叉验证等统计学指标对结果进行“二次质检”。
- **[TODO] 深度表示专家**: 引入基于 PyTorch 的深度聚类 (Deep Clustering) 能力，支持图像与复杂非线性特征的自动提取。

### 第二阶段：交互式创新 (Mid-term)
- **[TODO] Human-in-the-Loop**: 允许用户对聚类结果进行在线标记，Agent 通过约束聚类 (Constrained Clustering) 实时微调算法权重。
- **[TODO] 集成聚类 (Ensemble Consensus)**: 实现多算法并行后的“一致性融合”，通过 Meta-Clustering 显著提升结果的稳健性。
- **[TODO] 参数演化预览**: 交互式展示参数空间对结果的影响（如 DBSCAN 的 Epsilon 扫描），由 Agent 给出最佳建议。

### 第三阶段：学科特化与生态 (Long-term)
- **[TODO] 行业 RAG 插件系统**: 提供“生信插件”、“金融审计插件”、“遥感插件”。用户可切换领域知识库，使 Agent 具备行业特有的“聚类直觉”。
- **[TODO] ACE-as-a-Service**: 提供标准 RESTful API 接口，支持将 ACE 引擎集成到私有云或企业级 MLOps 流水线中。
- **[TODO] 多模态支持**: 扩展至文本主题聚类与跨模态特征对齐聚类任务。

---
