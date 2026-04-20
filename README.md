# ACE Agent (Automated Clustering Expert) 🛡️

ACE Agent 是一个 **MVP / 早期可用阶段**的、具备**自愈能力**的多智能体自主聚类分析系统。它将大语言模型（LLM）作为大脑，实现了从数据画像、意图路由到代码生成、自动调试及学术报告生成的全流程闭环。

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

ACE Agent 的愿景是成为**数据科学领域的“自动驾驶仪”**。

### 第一阶段：评估闭环与维度感知 (Phase 1 - DONE)
- **[DONE] 维度专家 (Dimension Expert)**: 实现 PCA / SpectralEmbedding 管线，已完全适配 Think-Act-Fix 自愈框架。
- **[DONE] 评价专家 (Critic Agent)**: 引入独立审计机制。实现 Hopkins Statistic (聚类趋势) 审计与 K-稳定性校验，为聚类结果提供数学层面的“二次质检”。
- **[DONE] 深度表示专家**: 基于 PyTorch 的 AE + KMeans 骨架已就位，支持非线性特征提取与设备（CUDA/CPU）自适应。
- **[DONE] 智能调度升级**: `ACESupervisor` 实现维度感知逻辑，处理 >2D 高维数据时自动激活维度专家进行预处理。

### 第二阶段：交互式创新 (Mid-term - WIP)
- **[TODO] Human-in-the-Loop**: 允许用户对聚类结果进行在线标记，Agent 通过约束聚类 (Constrained Clustering) 实时微调算法权重。
- **[TODO] 集成聚类 (Ensemble Consensus)**: 实现多算法并行后的“一致性融合”，通过 Meta-Clustering 显著提升结果的稳健性。
- **[TODO] 参数演化预览**: 交互式展示参数空间对结果的影响（如 DBSCAN 的 Epsilon 扫描）。

### 第三阶段：深度聚类强化 (Mid-term - Upcoming)
- **[TODO] 完整深度聚类**: 实现 DEC / IDEC 等端到端聚类优化。
- **[TODO] 显存自适应策略**: 针对 8G/16G 显存自动优化 Batch Size 与混合精度。

---

## 🧭 工程审查与优先级规划 (PM Review, 2026-04-20 Update)

### 1. 现状基线 (Baseline Audit)
- **测试状态**: 96 项测试全部通过（Coverage ~65%）。
- **环境要求**: 必须在 Conda `Tumor_Subtype_Agent` 环境下运行。
- **可观测性**: 已实现 `outputs/llm_trace.jsonl` 日志，支持 Token 计量与 Fallback 追踪。

### 2. 路线图风险分级 (Risk Triage)

**🟢 低风险（路径清晰）**
- 维度专家、深度表示、集成聚类、参数扫描：库成熟，骨架已有，可继续推进。

**🟡 中风险（被低估）**
- **Critic Agent**：定位模糊，与 Supervisor 选型逻辑存在重叠；须先定义"独立审计方 vs 投票人"的职责边界。
- **Human-in-the-Loop 约束聚类**：实为准独立子系统（标注 UI + 增量训练 + 约束求解），需单独立项。
- **RAG 插件系统**：需向量库、语料治理、版本管理；当前 `knowledge_engine.py` 仅 102 行，不具备支撑三领域并行的能力。
- **ACE-as-a-Service**：缺 FastAPI 层、鉴权、任务队列、限流/计费；Dockerfile 存在但未构成服务化闭环。

**🔴 路线图未覆盖的关键盲区（Must-Fix Before Scale-out）**

| 维度 | 缺失项 | 业务风险 |
| --- | --- | --- |
| 工程质量 | 无 CI、无覆盖率、无 lint/type-check | 自愈改代码时无回归保障 |
| 可观测性 | 无 LLM 调用追踪、Token 成本监控、自愈失败率看板 | 生产化定位困难 |
| 数据安全 | 沙箱无资源限额；无 PII 扫描 | 企业场景卡点 |
| 成本治理 | 自愈 3 次 × N 专家 → Token 指数膨胀 | 脏数据即烧钱 |
| 评估基准 | 无标准 benchmark 套件 | 无法量化"自愈"效果 |
| LLM 抽象 | `llm_client.py` 疑似单供应商硬编码 | 切换/降级/本地模型困难 |
| 用户协作 | 无多用户、无会话持久化 | 仅限单机 Demo |
| 合规 | 无脱敏、无审计日志、无模型卡 | 金融/生信无法落地 |
| 文档一致性 | README 与代码实况脱节 | 交接/招募成本高 |

### 3. 修订后的优先级路线图 (Revised Roadmap)

**核心原则：先夯地基 → 再扩专家 → 最后谈生态。**

#### Phase 0 — 工程基线 (目标 2 周内)
- [x] 对齐 README 与代码实况（移除不实"生产级"表述、更新 [TODO] 状态）
- [x] 引入 CI：`pytest` + `ruff` + `mypy`，最小覆盖率门槛 60%（当前实测 ~62%，`--cov-fail-under=30` 作为 CI 门槛；TODO Phase 1 提升至 60）
- [x] LLM 调用日志 + Token 计量 + 自愈重试指标（落盘 `outputs/llm_trace.jsonl`；`caller` 字段区分 generate / fix:N 调用；侧边栏实时累计展示）
- [x] 沙箱加硬限额：wall-clock timeout（默认 60s，`ACE_SANDBOX_TIMEOUT_SEC` 可配）+ 内存上限（默认 2 GiB，基于 delta-RSS 监控，Windows psutil 实现）；超限抛出 `SandboxResourceExceeded(reason)`
- [x] `llm_client.py` 抽象化：`LLMProvider` ABC + `OpenAICompatibleProvider` / `DeepSeekProvider` / `DashScopeProvider` / `OpenAIProvider`；主供应商失败自动 fallback（最多 1 次）；侧边栏新增 Fallback Provider 配置

#### Phase 1 — 评估闭环优先于新专家 (目标 1 个月)
- [ ] **Critic Agent**：明确为"独立审计方"，输出稳定性/重采样/CVI 指标
- [ ] **Benchmark 套件**：UCI 标准数据集 + 合成数据，量化自愈成功率与结果质量
- [ ] Dimension Expert：API 收敛 + 单测补齐
- [ ] 文档：Engineering Handbook 与实际代码对齐

#### Phase 2 — 交互与稳健性 (目标 2 个月)
- [ ] 集成聚类（Ensemble Consensus）：技术难度低、用户感知强，优先做 Demo 亮点
- [ ] 参数演化预览（Epsilon 扫描等）
- [ ] Human-in-the-Loop **降级版**：结果打标 + 重新触发，不做在线约束求解

#### Phase 3 — 深度聚类启用 (目标 3 个月)
> **硬件已就位**：本地 RTX 4060 Ti 8G（开发/小模型）+ 远端 NVIDIA A4000 16G（训练/中等模型）。故 PyTorch 深度聚类从"砍/延后项"升级为 Phase 3 正式阶段。
- [ ] 深度聚类基线：DEC / IDEC / AutoEncoder + KMeans 联合优化
- [ ] 表格数据深度表示：Contrastive / SCARF 类嵌入
- [ ] 显存自适应策略：batch size、AMP 混合精度、按显卡容量自动选档（8G vs 16G）
- [ ] GPU/CPU 回退路径：无 CUDA 环境自动降级到 sklearn 基线
- [ ] 与 Benchmark 套件对接：深度方法必须在 Phase 1 的 benchmark 上跑出可复现指标

#### Phase 4 — 学科特化与扩展 (目标 4 个月+)
- [ ] RAG 插件：**单一领域（建议生信，语料开放）先跑通**，禁止三领域并行
- [ ] 多模态：图像/文本主题聚类（复用 Phase 3 的深度表示能力）
- [ ] 稳定性强化、用户反馈沉淀

#### Phase 5 — ACE-as-a-Service (长期设想，暂不启动)
> **定位调整**：仅作为功能完整后的最终设想，优先级最低；Phase 0–4 全部完成并验证后再立项。
- [ ] RESTful API、鉴权、任务队列、限流/计费等服务化能力

#### 🗑️ 砍/延后项
- **ACE-as-a-Service 提前启动**：禁止在 Phase 0–4 完成前分配资源
- **三领域 RAG 并行**：禁止同时启动多个领域插件

### 4. 规划决策约束 (Governance)

后续任何 PR、Issue、功能提案必须满足：
1. **地基优先**：Phase 0 未完成前，不新增 Roadmap 一期以外的专家。
2. **可度量**：新增自愈/聚类能力必须附带 benchmark 结果。
3. **成本可控**：涉及 LLM 调用的改动必须评估 Token 成本上限。
4. **文档同步**：代码变更须同步更新本 README 与 Engineering Handbook，禁止"TODO 与现实脱节"再次发生。
5. **冲突仲裁**：本章节（PM Review）优先级高于上文 Roadmap；若需调整本章，须显式在 PR 说明中标注 "PM-Review-Override"。

### P0.5 紧急修复（2026-04-20）

> Phase 0 之后、Phase 1 之前的补丁，修复三个生产验收中发现的 Bug。

#### Bug 1 — 月牙数据集误选 SpectralClustering 为最优
- **根因**：`supervisor.py` 硬编码 `selected_keys = ["centroid", "topology"]`，
  密度聚类专家（zoo 含 DBSCAN/HDBSCAN）从未被激活，只能在两个非密度专家里选最优。
- **修复**：`ZooExpert` 适配为 `BaseExpert` 接口（继承，实现 `_generate_code`），
  `supervisor.py` 改用 `build_expert_registry()` 完整注册表，
  默认激活策略升级为 `["centroid", "topology", "zoo"]`。
  改动文件：`expert_sub_agents/zoo_expert.py`、`agent_core/supervisor.py`。

#### Bug 2 — "生成代码示例"被误判为 NEW_TASK 重新跑实验
- **根因**：`router.py` 的 prompt 规则"提到具体算法名 + 要求执行 → NEW_TASK"
  将"生成代码示例"与"执行新实验"混淆。
- **修复**：Router system prompt 新增第三种意图 `CODE_EXAMPLE`，
  明确区分"要代码本身"与"要执行实验"；`supervisor.py` 新增
  `_handle_code_example()` 分流，只用 LLM 生成 Markdown 代码块，
  不走沙箱、不生成图、不更新 `last_report`；
  异常兜底从 NEW_TASK 改为 FOLLOW_UP，降低误触发风险。
  改动文件：`agent_core/router.py`、`agent_core/supervisor.py`。

#### Bug 3 — 专家日志显示"运行成功"但最终报"所有专家均失败"
- **根因**：`base.py` 只检查 `run_result["success"]`，
  不检查 `artifacts` 是否非空；LLM 生成代码未写入 artifacts 时
  success=True 但 artifacts 为空，外层 `if not all_results` 触发通用错误且原因被吞掉。
- **修复**：`base.py` 判据改为 `success AND artifacts 非空` 才视为成功；
  新增软失败路径（success=True 但 artifacts 空 → 注入 artifacts 约定提示后重试）；
  `_error_report` 增强：汇总各专家最后 3 行日志作为排错依据；
  `_generate_code` 文档增加 artifacts 约定说明；
  `LatexReportGenerator` 对 CODE_EXAMPLE 类型主动跳过（不崩溃）。
  改动文件：`expert_sub_agents/base.py`、`agent_core/supervisor.py`、`tools/latex_generator.py`。

---
