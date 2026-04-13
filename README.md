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

