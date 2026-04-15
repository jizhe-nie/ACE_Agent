# ACE Agent Demo

ACE Agent is a lightweight multi-agent clustering demo built around a master router,
specialized expert agents, synthetic sklearn-style datasets, and a Streamlit web UI.

## What this demo includes

- A master agent that profiles data and routes work to expert agents
- Five expert agents for centroid, topology, dimensionality, deep representation,
  and multi-view consensus clustering
- Deterministic code generation plus sandboxed execution for each expert
- Synthetic demo datasets: blobs, moons, s-curve, and smile
- Automatic metrics, figures, and LaTeX report generation
- A Streamlit page for model configuration, chat-driven analysis, and visible
  decision trace

## Run

```bash
streamlit run ACE_Agent/web_demo.py
```

## Smoke test

```bash
python ACE_Agent/demo_runner.py --dataset smile
```

## Notes

- The web UI stores model settings in `ACE_Agent/.ace_demo_config.json`.
- LLM access is optional. If no API is configured, the system still runs the full
  clustering pipeline and produces deterministic reports.
- The UI shows a "decision trace" rather than raw hidden chain-of-thought.

该项目的实现逻辑遵循典型的代理协作模式，分为以下五个关键层次：

## 1. 代理大脑 (Agent Brain) —— 知识与决策逻辑

- **位置**: `agent_brain/`
- **作用**: 存储算法选择的“边界”和“规则”。
  - `taxonomy_rules.md`: 定义了聚类算法的分类学规则（什么时候用基于密度的，什么时候用基于质心的）。
  - `algorithm_limits.md`: 规定了不同算法在数据规模、维度上的限制。
  - `metric_logic.json`: 包含了评估指标（如轮廓系数、DBI等）的评价逻辑。

## 2. 专家子代理 (Expert Sub-Agents) —— 垂直领域的执行者

- **位置**: `expert_sub_agents/`
- **实现**: 每个文件代表一个特定领域的专家。
  - `centroid_expert.py`: 负责基于质心的算法（如 K-Means）。
  - `topology_expert.py`: 负责基于拓扑或密度的算法（如 DBSCAN, HAC）。
  - `dimension_expert.py`: 负责降维后的聚类（如 PCA + K-Means）。
  - `deep_representation.py`: 负责利用深度学习（如 AutoEncoder）进行表征学习后再聚类。
  - `multi_view_expert.py`: 负责多视图共识分析。

## 3. 代理核心控制 (Agent Core) —— 调度与路由

- **位置**: `agent_core/`
- **实现**:
  - `router.py`: 根据数据集的特征（在 `data_factory` 生成时获取）将任务分配给合适的专家。
  - `supervisor.py`: 充当“主考官”，汇总所有专家的分析结果，解决冲突，并决定最终的最优方案。

## 4. 工具链 (Tools) —— 底层支撑

- **位置**: `tools/`
- **关键功能**:
  - `data_factory.py`: 负责生成测试数据集（如 S-Curve, Moons, Smile 等）或读取用户数据。
  - `coder_sandbox.py`: 提供一个安全的环境，让 LLM 生成的 Python 代码能够实际运行算法并绘图。
  - `latex_generator.py`: 将专家的文字分析、评估指标和绘图路径转换成专业的 LaTeX 源代码。
  - `llm_client.py`: 统一的 LLM 接口，支持调用不同的模型进行推理。

## 5. 输出与演示 (Outputs & Demos)

- **位置**: `outputs/`, `web_demo.py`
- **结果**: 系统运行后，会在 `outputs/` 下创建带有时间戳的文件夹，包含：
  - 各种算法的聚类效果对比图 (`.png`)。
  - 自动生成的分析报告 (`ace_report.tex` 及其编译后的 `.pdf`)。

---

## 整个工作流 (Workflow) 实现过程

1. **初始化**: 用户通过 `web_demo.py` (Streamlit) 或 `demo_runner.py` 启动任务。
2. **数据画像**: `data_factory` 生成数据后，`supervisor` 调用 LLM 分析数据特征（维度、样本数、可能的分布形态）。
3. **专家分工**: `router` 根据数据画像，激活对应的 `expert_sub_agents`。例如，如果数据是流形结构的（如 S-Curve），则会重点激活 `topology_expert`。
4. **自主实验**:
   - 每个专家调用 `coder_sandbox` 编写并运行 Python 代码。
   - 产生实验结果（指标评分和可视化图像）。
5. **共识评审**: `supervisor` 收集所有专家的报告，对比不同维度的优劣，撰写最终的“执行摘要”。
6. **报告汇编**: `latex_generator` 将所有图表和专家的文字描述整合，渲染成一份完整的学术级聚类分析报告。

## 总结

这个项目是一个“数据科学代理”的典型实践。它不仅仅是跑几个聚类算法，而是模拟了人类数据科学家“观察数据 -> 猜测分布 -> 尝试不同算法 -> 对比指标 -> 撰写总结”的完整思维链条。

▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄

✦ 这是一个非常有深度的反馈，老师的意图是希望将你的项目从一个“功能演示脚本”提升为一个“真正的生产级智能体系统（Production-Ready Agentic System）”。

以下是对老师意图的详细分析、具体实现逻辑、可行性评估及行动建议：

### 1. 改进对话方式：支持追问（Contextual Memory）

*   **老师意图**：解决“健忘”问题。目前 Agent 处于“一问一答”的无状态模式。老师希望 Agent 具备上下文感知（Context-Awareness）能力，能分清什么是“新任务”，什么是“对旧结果的深入探讨”。
*   **具体实现**：
    *   **会话管理（Session Management）**：在 `supervisor.py` 或 `router.py` 中引入 `history` 列表，存储之前的用户输入和 Agent 的输出（包括中间结果的元数据）。
    *   **意图识别（Intention Recognition）**：每次用户提问时，先由 LLM 判断意图：是 `NEW_TASK`（重新跑算法）还是 `FOLLOW_UP`（解释/分析已有结果）。
    *   **结果持久化**：聚类结果（图像路径、指标）需要缓存在内存或临时数据库中，以便追问时直接读取，而不需要重新计算。
*   **可行性**：极高。这是目前 Agent 开发的标准配置。

### 2. 增加算法库：大规模聚类算法与智能选优

*   **老师意图**：增加系统的“厚度”和“专业性”。目前的几个 Expert 只是冰山一角。老师希望系统能像一个“资深数据科学家”，不仅懂几种算法，还能根据数据特征（如：数据量大小、维度、是否有噪声、形状）自动匹配最优解。
*   **具体实现**：
    *   **算法插件化**：不要为每种算法写一个子 Agent 文件。建立一个 `AlgorithmZoo`，统一封装 `scikit-learn`、`PyClustering`、`HDBSCAN` 等库。
    *   **元学习路由（Meta-Learning Router）**：在 `router.py` 中增加逻辑：提取数据的统计特征（偏度、峰度、稀疏度、样本量），由 LLM 或规则引擎输出：“该数据适合密度聚类（DBSCAN）而非中心聚类（KMeans）”。
    *   **全量遍历模式**：提供一个 `ExhaustiveMode`，利用多线程跑完所有算法，最后根据 Silhouette Score（轮廓系数）等指标排序。
*   **可行性**：中等偏上。算法实现不难（调库），难点在于如何高效地调度和对比上百种算法的结果而不崩溃。

### 3. 增加检索增强检索（RAG）

*   **老师意图**：增加“理论深度”。当用户问“为什么这个结果不好？”时，Agent 不应该只是猜，而是去查询聚类相关的教材、论文或文档。这能显著降低 LLM 的幻觉，让回答更具有学术权威性。
*   **具体实现**：
    *   **知识库构建**：收集聚类分析的经典教材（如《模式识别》）、算法官方文档（Scikit-learn docs）的 PDF/Markdown。
    *   **向量化存储**：使用 FAISS 或 ChromaDB 将这些文档存入向量数据库（项目已有 `memory_vdb` 文件夹，正好可用）。
    *   **RAG 工作流**：用户提问 -> 检索向量库相关段落 -> 将段落连同问题发给 LLM -> LLM 生成带有理论依据的回答。
*   **可行性**：高。这是目前最成熟的 Agent 增强技术。

---

### 可行性总结与优先级建议

| 需求                | 难度  | 价值  | 建议                                          |
|-------------------|------|------|---------------------------------------------|
| 1. 支持追问         | ★★☆☆☆ | ★★★★★ | 优先第一步。这是用户体验的基石。                |
| 2. 算法库扩展       | ★★★★☆ | ★★★★☆ | 核心竞争力。可以先扩充到 20-30 种主流算法。     |
| 3. RAG 集成         | ★★★☆☆ | ★★★★☆ | 加分项。让 Agent 看起来更像“专家”而非“黑盒”。   |

---

### 你该如何开始？

我建议分阶段推进，你可以先从改进对话逻辑开始，因为这能立刻让你在下一次向老师演示时，表现出巨大的进步。

#### 第一步：修改 `router.py` 和 `supervisor.py` 实现“追问”

你需要引入一个简单的状态机。

**操作思路：**

1.  在 `ACE_Agent/agent_core/schemas.py` 中定义一个 `Message` 结构。
2.  在 `supervisor.py` 中维护一个 `memory` 列表。
3.  在调用 `router` 之前，把 `memory` 传给 LLM，问它：“用户是在问新任务，还是在问关于上一个结果的问题？”

#### 第二步：扩充算法池（动态生成 Expert）

不要手动写 100 个文件，而是写一个 `GenericExpert`，它能根据参数动态调用不同的算法。

**操作思路：**

1.  创建一个 `tools/algorithm_registry.py`，注册 50 种算法。
2.  修改路由逻辑，让 LLM 输出一个“候选算法列表”。

#### 第三步：简单的 RAG 实现

你可以先放几个 Markdown 文件到 `agent_brain`，尝试用向量检索。

---

你想先从哪一部分开始改进？
如果你准备好了，我可以先帮你写出“支持追问”的逻辑框架代码。

