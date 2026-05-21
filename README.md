# ACE Agent (Automated Clustering Expert) 🛡️

ACE Agent 是一个 **MVP / 早期可用阶段**的、具备**自愈能力**的多智能体自主聚类分析系统。它将大语言模型（LLM）作为大脑，实现了从数据画像、意图路由到代码生成、自动调试及学术报告生成的全流程闭环。

## 🚀 核心架构与特性

### 1. Orchestrator-Worker 编排架构
- **MasterRouter (决策中枢)**: 摒弃传统关键词匹配，采用纯语义识别。精准区分“新任务分析”与“历史追问解析”。
- **ACESupervisor (全局编排)**: 维护会话状态与记忆，协调不同领域的专家 Agent。
- **Self-Healing Experts (自愈专家)**: 核心专家 Agent (质心、拓扑、降维等) 遵循 **Think-Act-Fix** 循环。当代码在沙箱中运行报错（如维度不匹配、API 变更）时，Agent 会分析 Traceback 并自动修复代码，上限 3 次重试。
- **Dimension Expert (Phase 3 混合模式)**: 采用"强类型骨架 + LLM JSON 决策"架构。LLM 不生成完整代码，而是输出约 140 token 的 JSON 管线决策；确定性 Python 骨架负责 import、异常守卫和 artifacts 写入。支持 5 条降维+聚类管线（PCA+KMeans / PCA+GMM / UMAP+KMeans / tSNE+KMeans / AE+KMeans），在高维数据（>32 features）上自动激活 AutoEncoder 深度管线。
- **ModalityProfile 模态感知**: 集中式模态描述对象 (`schemas.py`)，自动检测数据类型 (tabular/time_series/text/image) 并推断距离度量 (euclidean/cosine)。贯穿 graph 构图 → supervisor 专家路由 → 无标签排名 (metric-aware Silhouette) → Critic 审计全链路，消除硬编码欧氏距离盲区。
- **DataContext 不可变数据上下文**: 数据集以只读 `CTX_DATA` 对象注入沙箱，包含 X, y, n_samples, n_features 等元数据，消除 LLM 生成代码中常见的变量绑定错误。

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
- `expert_sub_agents/base.py`: **关键** - 定义了专家的自愈循环框架 + PRE_INJECT 预注入机制。
- `expert_sub_agents/dimension_expert.py`: **Phase 3 重构** - 强类型骨架 + LLM JSON 决策，7 管线降维专家（含 SelfLabel 师生蒸馏，70K MNIST ARI 0.8454）。
- `tools/coder_sandbox.py`: 包含安全策略、DataContext 不可变数据上下文、CORE_PRE_INJECT 预注入。
- `web_demo.py`: 基于 Streamlit 的交互式工作台。

---

## 🛠️ 部署指南

### 环境依赖
```bash
# 推荐 Python 3.10+
pip install -r requirements.txt
# RAG 知识引擎依赖 (语义检索)
pip install sentence-transformers pypdf
# 拓扑专家 HDBSCAN/OPTICS 依赖
pip install hdbscan
# 若需 UMAP 降维管线 (推荐):
pip install umap-learn
```

### 运行
```bash
streamlit run web_demo.py
```

---

## 📑 工程说明书
项目根目录下包含 `ACE_Agent_Learning_Guide.tex`（学习指南）和 `ACE_Agent_Engineering_Handbook.tex`（工程手册），以学术级规格详细记录了系统的架构设计、通信协议、自愈数学模型及复现路径。编译方式见手册内说明。

---

## 🔮 发展路线图 (PM Review, 2026-05-08 Revision)

> **注意**：本章节为 ACE Agent 唯一权威路线图，替代此前所有版本。所有 Phase 状态均基于代码实况审计，而非历史声明。

### 1. 现状基线 (Baseline Audit, 2026-05-08 Update)
- **测试状态**: **134 项测试全部通过**（87 benchmark + 47 系统集成），Coverage ~65%（CI 门槛 `--cov-fail-under=30`）。
- **环境要求**: 必须在 Conda `Tumor_Subtype_Agent` 环境下运行。
- **可观测性**: 已实现 `outputs/llm_trace.jsonl` 日志 + `outputs/benchmark_*.json` 基准报告，支持 Token 计量与 Fallback 追踪。
- **专家状态**: 7 个活跃专家（centroid / topology / zoo / critic / dimension / ensemble / graph）；**critic** 完成独立后验审计重构 + Critic 2.0 决策闭环；**dimension** 完成 Conv-AE + SelfLabel 师生蒸馏管线，70K MNIST ARI **0.8454**；**ensemble** 完成 Co-association Matrix 共识融合 + 多样性约束 + 质心算法硬过滤；**graph** 完成原生 MCL/Louvain/Leiden 社区发现，条件激活（仅当 geodesic_distortion > 0.35 时）；deep_representation 和 multi_view 已从默认池移除。
- **数据集**: 21 个数据集（11 个 2D/3D 合成 + 10 个真实高维），Phase 4 计划新增 7 个高维真实数据集（USPS/Reuters/HAR/CIFAR-10/Pendigits/Letter/COIL-20）。
- **Benchmark 套件**: 已就位（`benchmark/` 包），支持 CLI 一键运行（`python -m ACE_Agent.benchmark --datasets ... --experts ...`）、离线/在线双模；digits(64维) 全专家 100% 成功率，UMAP_KMeans ARI 0.8764 当前最高分；**70K MNIST SelfLabel ARI 0.8454 为深度管线最高分**。
- **沙箱安全**: Phase 0 资源限额（timeout + memory）+ **2026-04-29 高危方法拦截**（os.remove / os.system / sys.exit / subprocess 等封禁），见 `tools/coder_sandbox.py`。
- **代码规模**: ~7,300 行核心代码（含 benchmark），~3,300 行测试代码（134 项全部通过，含 47 项系统集成测试）。

### 2. 路线图风险分级 (Risk Triage)

**🟢 低风险（路径清晰）**
- 维度专家、深度表示、集成聚类、参数扫描：库成熟，骨架已有，可继续推进。

**🟡 中风险（被低估）**
- **Critic Agent**：✅ 已解决 — 职责明确为独立后验审计方，输出结构化 audit_report，Web UI / LaTeX 均已对接。
- **Human-in-the-Loop 约束聚类**：实为准独立子系统（标注 UI + 增量训练 + 约束求解），需单独立项。
- **RAG 插件系统**：需语料治理、版本管理、多 Collection 并行检索；当前 `knowledge_engine.py` (~140 行，ChromaDB + SentenceTransformer) 已可稳定查询，但距三领域并行仍有差距。
- **ACE-as-a-Service**：缺 FastAPI 层、鉴权、任务队列、限流/计费；Dockerfile 存在但未构成服务化闭环。

**🔴 关键盲区治理状态（Phase 0 审计更新）**

| 维度 | 缺失项 | 状态 | 备注 |
| --- | --- | --- | --- |
| 工程质量 | 无 CI、无覆盖率、无 lint/type-check | ✅ 已解决 | Phase 0：GitHub Actions CI + ruff + mypy + pytest-cov |
| 可观测性 | 无 LLM 调用追踪、Token 成本监控 | ✅ 已解决 | Phase 0：`llm_trace.jsonl` + caller 字段 + 侧边栏展示 |
| 数据安全 | 沙箱无资源限额 | ✅ 已解决 | Phase 0：wall-clock timeout + delta-RSS 2 GiB 硬限额 |
| 数据安全 | 沙箱未拦截 os.remove / sys.exit 等高危方法 | ✅ 已解决 | 2026-04-29：`_safe_import` 拦截 18 个高危方法 + 封禁 subprocess 等 15 个危险模块 |
| LLM 抽象 | `llm_client.py` 单供应商硬编码 | ✅ 已解决 | Phase 0：ABC + 4 种 Provider + 自动 fallback |
| 成本治理 | 自愈 3 次 × N 专家 → Token 指数膨胀 | 🟡 部分缓解 | 自愈重试已追踪，但策略层面无预算上限熔断 |
| 评估基准 | 无标准 benchmark 套件 | ✅ 已解决 | Phase 1：`benchmark/` 包，29 项测试，离线/在线模式，CI 集成 |
| 用户协作 | 无多用户、无会话持久化 | 🔴 未解决 | 非 MVP 阶段优先事项 |
| 合规 | 无脱敏、无审计日志、无模型卡 | 🔴 未解决 | 非 MVP 阶段优先事项 |
| 文档一致性 | README 与代码实况脱节 | ✅ 已解决 | 路线图统一 + Engineering Handbook v2.0 重写 |

### 3. 修订后的优先级路线图 (Revised Roadmap)

**核心原则：先夯地基 → 再扩专家 → 最后谈生态。**

#### Phase 3.2 — 原生图社区发现 (2026-05 DONE)
- [x] **Native Graph Expert**: 摆脱欧氏坐标，仅基于邻接矩阵进行分区。
- [x] **Wall-Aware 构图**: 引入 Jaccard 剪枝与局部缩放权重，识别流形间的"防火墙"。
- [x] **Louvain/MCL 集成**: 支持大规模图社区发现，采用模块度 (Modularity) 替代轮廓系数评分。

#### Phase 6 — 智能诊断与评估增强 (2026-05 DONE)
- [x] **NMI 指标**: 新增归一化互信息作为一级排序指标，与 ARI 并列展示。ARI 看配对一致性，NMI 看信息增益，两者互补交叉验证。
- [x] **Hopkins 预检门禁**: 专家派发前对 500 样本快速估算 Hopkins，< 0.3 自动跳过密度/拓扑/图专家，省一半无效计算。
- [x] **审计严格化**: 当 |winner_k - CVI_k| > 3 且 Hopkins < 0.3 时，endorsement 强制降级至 `qualified_with_warning`。
- [x] **结果缓存**: 数据集哈希 → 跳过重复计算；自动绑定 git HEAD，代码改动即自动失效；支持"重新"/"重跑"关键词强制绕过。
- [x] **大样本智能降采样**: N > 10K 时分层抽样至子集派发专家，避免 O(N²) 超时，共 11 处排名调用统一使用 working_dataset 消除 ARI 静默归零。
- [x] **Router 数据集感知**: `analyze_intent()` 接收当前已选数据集上下文，"请分析该数据"在新会话中正确路由为 NEW_TASK。
- [x] **多项关键 Bug 修复**: 高维数据误触发 UMAP 流形展开（512D→3D 失真）；`weighted_median` polyfill（scipy < 1.13 兼容）；`ACEJsonEncoder` 支持 numpy 标量类型。

#### Phase 7 — 极限性能优化与 70K 样本支持 (2026-05 DONE)
- [x] **Session 数据瘦身**: 递归剥离大数组，彻底解决 `.ace_sessions.json` 文件膨胀导致的 UI 瘫痪。
- [x] **Ensemble 采样熔断**: $N > 5000$ 时自动切换为 Monte Carlo 随机对采样。
- [x] **Hopkins 门禁**: 预检数据聚类倾向，Hopkins < 0.3 时自动跳过昂贵的拓扑专家。
- [x] **数据集缓存限制**: `st.cache_data(max_entries=5)` 防止内存长期驻留。
- [x] **沙箱自适应超时**: 根据样本量自动调整超时时间 (60s -> 240s)。

#### Phase 8 — 多 LLM 混合架构 (2026-05 进行中)
- [x] **ModalityProfile 模态感知系统**: 集中式 dataclass + `detect_modality()` 检测函数，自动推断距离度量 (euclidean/cosine)，贯穿 graph 构图 / 专家路由 / 无标签排名 / Critic 审计全链路。支持 time_series (DTW hint)、text/sparse (cosine + L2 normalize)、image、tabular 四种模态。 (**2026-05-18 DONE**)
- [x] **无标签排名 metric 感知**: 当无 ground-truth 标签时，Silhouette 自动使用 ModalityProfile 推断的距离度量 (text→cosine, 其他→euclidean)，消除文本数据欧氏距离评分偏差。 (**2026-05-18 DONE**)
- [x] **降维模态感知**: `dim_reduction_hint` 接入 dimension expert 全链路 — LLM 决策提示 + `_build_smart_defaults()` 模态感知，text→TruncatedSVD，time_series→PCA。 (**2026-05-19 DONE**)
- [x] **KnowledgeEngine 路径健壮性修复**: `db_path` / `docs_dir` 改为 `__file__` 基准绝对路径，消除 CWD 依赖。 (**2026-05-18 DONE**)
- [x] **Session 持久化修复**: 用户消息立即写盘 + 异常捕获扩展，解决重启后历史会话丢失。 (**2026-05-19 DONE**)
- [x] **项目文件清理**: 根目录散落 .png、outputs/ 300+ 历史图片、benchmark_cache 等残留全部清理，.gitignore 加固。 (**2026-05-19 DONE**)
- [ ] **异构模型路由**: 调度 GPT-4o 负责审计 (Critic)，DeepSeek-Coder 负责代码生成。
- [ ] **多方审查机制**: 当 Critic 裁决为 RETRY 时，引入第二个 LLM 进行交叉验证。
- [ ] **性能分级**: 轻量级任务 (FOLLOW_UP) 路由到廉价模型，复杂任务 (NEW_TASK) 路由到旗舰模型。

#### Phase 1 — 评估闭环优先于新专家 (目标 1 个月)
- [x] **Critic Agent**：独立审计方，输出稳定性/重采样/CVI 指标 + Web UI 审计卡片 + LaTeX 审计章节 (**2026-04-30 DONE**)
- [x] **Benchmark 套件**：UCI 标准数据集 + 合成数据，量化自愈成功率与结果质量 (**2026-04-28 DONE**)
- [x] **Dimension Expert**：Phase 3 重构完成 — 强类型骨架 + LLM JSON 决策 5 管线混合模式，digits 数据集 100% 成功率零重试，UMAP_KMeans ARI 0.8764 (**2026-04-29 DONE**)
- [x] 文档：Engineering Handbook 与实际代码对齐 (**2026-04-28 DONE，v2.0 重写**)

#### Phase 2 — 集成聚类与可观测性深度强化 (2 个月, 2026-05 启动)

> **背景**：Phase 3 深度聚类已超额完成任务（SelfLabel ARI 0.8454），工程基线（Phase 0/1）稳固。
> 现进入"从单兵作战转向兵团作战"阶段——用集成共识替代单最优选择，用反馈闭环替代事后审计。

##### 2.1 Ensemble Consensus Expert — P0 (Week 1-2) ✅ **DONE (2026-05-04)**
- [x] **Co-association Matrix 融合**：新建 `expert_sub_agents/ensemble_expert.py`
  - 接收各专家产出的 `labels` 列表，构建共识矩阵 $C_{ij} = \frac{1}{M} \sum \mathbb{1}[l_m(i) = l_m(j)]$
  - 对差异矩阵 $1-C$ 运行 AgglomerativeClustering 得 `consensus_labels`
  - 加权模式：按各专家 `score` 归一化后加权投票
  - 指标：`n_experts_fused`, `entropy_of_agreement` (Shannon 熵衡量一致性)，`agreement`（专家对一致性比率）
- [x] Supervisor 集成：在 rank + audit 之后新增 `_execute_ensemble()` 步骤，共识结果参与最终排名
- [x] O(N²) 熔断：N > 20K 时对共识矩阵做 Monte Carlo 稀疏采样（10K 随机对 + 2K anchor）
- [x] Benchmark 测试：13 项 ensemble 专项测试全部通过
- [x] **共现矩阵可视化**：Web UI 渲染 YlOrRd 热力图，展示专家共识矩阵（降采样 max 500×500）
- [x] **条件触发门控**：Critic endorsed + confidence ≥ 0.75 时自动跳过 Ensemble，仅在置信度不足时触发融合拯救

##### 2.2 Critic 2.0 决策闭环 — P1 (Week 3-4) ✅ **DONE (2026-05-05)**
- [x] **条件触发门控（简化版）**：Critic endorsed + confidence ≥ 0.75 时跳过 Ensemble；qualified/warning 时触发
- [x] **完整 Critic 2.0 闭环**：
  - `audit_report` 新增 `action: "CLEAR" | "WARN" | "RETRY"` 字段
  - `retry_constraints` 含 `force_k`, `blocked_algorithms`, `force_preprocessing`
- [x] Supervisor 新增 `_handle_audit_feedback()`：RETRY 时构造约束指令，重新调度专家池
- [x] `max_retries=2` 硬限制 + 每次重试扩大 timeout_margin
- [x] 约束传递协议：`BaseExpert._generate_code()` 链增加 `constraints: dict | None` 参数（全部 7 个专家适配）
- [x] Benchmark 测试：6 项 Critic 2.0 专项测试全部通过

##### 2.3 Human-in-the-Loop 降级版 ✅ **DONE (2026-05-05)**
- [x] 结果打标：Web UI 新增 HITL 人工标注修正面板（`_render_hitl_panel`），支持标签编辑 + 格式校验
- [x] 约束传递：`_inject_constraints_prompt` 扩展 `reference_labels` 约束类型，含截断预览 + ARI/NMI 对齐提示
- [x] 重新触发：`supervisor.run()` 新增 `constraints` 参数，贯穿 `_execute_full_analysis` → `execute_with_self_correction`
- [x] 不涉及在线约束求解（COP-KMeans 等），保持轻量

#### Phase 3 — 深度聚类启用 (目标 3 个月)
> **硬件已就位**：本地 RTX 4060 Ti 8G（开发/小模型）+ 远端 NVIDIA A4000 16G（训练/中等模型）。故 PyTorch 深度聚类从"砍/延后项"升级为 Phase 3 正式阶段。
- [x] **AE_KMeans 深度架构 (Phase 4 升级)**：深层堆叠 Denoising AE（Linear→BatchNorm1d→LeakyReLU→Dropout）+ CosineAnnealingLR + Early Stopping + GMM 潜在聚类；LLM 可控 8 个超参（hidden_dims, learning_rate, dropout, noise_std, cluster_method 等）；digits ARI 0.5094（+57% vs 浅层 AE 0.3238），手调参数达 0.6102 (**2026-04-29 DONE**)
- [x] **DEC / IDEC 联合优化**：已完成 DEC/IDEC KL 散度联合优化（`tools/dec_pipeline.py`）但发现 KL 类联合训练在 Conv-AE 骨干上会导致表征退化（ARI 0.41-0.44 vs 基线 0.56）。转为自标注师生蒸馏路线 (**2026-05-03 DONE**)
- [x] **Conv-AE + SelfLabel 师生蒸馏 (Phase 5)**：Phase A Conv-AE 预训练（ReflectionPad2d + Latent BN + SimCLR-lite 对比损失）→ Phase B GMM 伪标签 → 冻结 Decoder → Cross-Entropy 微调 Encoder + Bootstrap + ReduceLROnPlateau。70K MNIST ARI **0.8454**（+0.15 vs 纯 Conv-AE+GMM，+0.28 vs MLP AE），已集成到 `dimension_expert.py` 作为图像数据首选生产路径 (**2026-05-04 DONE**)
- [ ] 表格数据深度表示：Contrastive / SCARF 类嵌入
- [ ] 显存自适应策略：batch size、AMP 混合精度、按显卡容量自动选档（8G vs 16G）
- [x] **GPU/CPU 回退路径**：PyTorch 模块级安全导入 + `_sklearn_fallback_pipeline` (PCA+KMeans/GMM) 自动降级 (**2026-04-30 DONE**)
- [x] **与 Benchmark 套件对接**：Conv-AE + SelfLabel 在 70K MNIST (完整数据集) 上跑出可复现 ARI 0.8454，dim=16/24/32 全维度验证通过。Scaler / OOM 回退 / O(N²) 熔断均已整合 (**2026-05-04 DONE**)

#### Phase 4 — 高维真实数据转向 (2026-05-08 重新规划)

> **背景**：导师反馈指出，迷宫/抛物线等复杂 2D 数据无测试优化必要，项目后续面向高维真实数据。
> Phase 4 原计划（RAG / KIM / 多模态）后移到 Phase 5，Phase 4 聚焦**数据集升级 + 高维适配**。

##### 4.1 数据集基础设施升级 — P0 (Week 1)

> **当前问题**：21 个数据集中 11 个是 2D/3D 合成 toy 数据（blobs/moons/spiral/pathbased/square/
> half_kernel/t4_8k/t7_10k 等），高维真实数据仅 8 个，且缺少聚类论文常用的对标基准。

**新增 Tier 1 数据集（必加，与主流论文对标）**

| 数据集 | 维度 | 样本数 | 类别数 | 领域 | 来源 |
|--------|------|--------|--------|------|------|
| **USPS** | 256 | 9,298 | 10 | 手写数字 | sklearn `fetch_openml("usps")` |
| **Reuters-21578** | ~2,000 (TF-IDF) | ~10,000 | 4 | 文本聚类 | nltk / sklearn `fetch_openml` |
| **HAR** (Human Activity Recognition) | 561 | 10,299 | 6 | 传感器时序 | UCI / sklearn `fetch_openml` |

**新增 Tier 2 数据集（应加，覆盖多领域）**

| 数据集 | 维度 | 样本数 | 类别数 | 领域 | 来源 |
|--------|------|--------|--------|------|------|
| **CIFAR-10** | 3072 (raw) / 512 (预训练特征) | 60,000 | 10 | 自然图像 | torchvision |
| **Pendigits** | 16 | 10,992 | 10 | 手写笔迹 | UCI / sklearn |
| **Letter Recognition** | 16 | 20,000 | 26 | 字符识别 | UCI / sklearn |
| **COIL-20** | ~1024 (HOG/pixel) | 1,440 | 20 | 物体识别 | sklearn / openml |

**CIFAR-10 三种特征模式切换说明**：

| 模式 | 维度 | 说明 | 切换方式 |
|------|------|------|----------|
| `cifar10_raw` | 3072 (32×32×3) | 原始像素展平 | `feature_mode="raw"` (默认) |
| `cifar10_gap` | 64 (8×8 GAP) | 全局平均池化降维 | `feature_mode="gap"` |
| `cifar10_resnet` | 512 | 预训练 ResNet-18 倒数第二层特征 | `feature_mode="resnet18"` |

三种模式通过 `data_factory.py` 的 `generate_dataset("cifar10", feature_mode="...")` 参数切换：
- **手动切换**：benchmark CLI 或 web_demo 中指定模式字符串，代码根据 `feature_mode` 走不同加载分支
- **不会自动切换**：因为不同模式维度差异巨大(raw 3072D vs gap 64D)，自动切换会破坏可复现性。建议 benchmark 中分别注册为独立数据集项，一次性跑通三个配置的对比

**基准测试重定位**：
- `benchmark/config.py` 默认 datasets 从 7 个 2D toy → 10 个高维真实数据
  - 新默认列表：`iris, wine, digits, usps, pendigits, letter, har, reuters, mnist, fashion_mnist, cifar10_raw`
- 保留 `blobs/moons` 作为 CI smoke test 快速验证（~5s）
- 新增分层 benchmark 策略：
  - `smoke` (3 数据集, ~5min) → CI 每次提交
  - `full` (11 数据集, ~30min) → 每日/PR 合并前
  - `exhaustive` (全量, ~2h) → 发版/论文前

##### 4.2 2D 数据瘦身 — P1 (Week 1)

**现有 2D SIPU/CHAMELEON 数据集处理方案（待你确认）**：

当前 6 个复杂 2D 数据的详细状态：

| 数据集 | 键名 | 样本 | 标签 | shape_family | 当前用途 |
|--------|------|------|------|-------------|----------|
| **pathbased** | `pathbased` | 300 | 3 类 | `non_convex` | SIPU 环形基准 |
| **Square** | `square` | ~800 | 5 类 | `non_convex` | 嵌套方形 |
| **Spiral** | `spiral_sipu` | 312 | 3 类 | `manifold` | 三螺旋线 |
| **Half-kernel** | `half_kernel` | ~750 | 4 类 | `non_convex` | 高斯+抛物线 |
| **t4.8k** | `t4_8k` | 8,000 | **无标签** | `non_convex` | 面具多密度 |
| **t7.10k** | `t7_10k` | 10,000 | **无标签** | `manifold` | 迷宫连通域 |

处理选项：

- **方案 A（推荐）**：保留代码不删除，但从默认 benchmark/FIXED_SIZE_DATASETS 中移除。代码保留在 `data_factory.py` 中，web_demo 下拉框隐藏（或标记 `[DEPRECATED]`），benchmark 默认列表不包含。若将来审稿需要 2D 可视化证据可随时恢复。
- **方案 B**：全部删除相关代码（`data_factory.py` 中的生成函数 + `FIXED_SIZE_DATASETS` 注册 + web_demo 列表项），彻底清理。
- **方案 C**：仅保留有标签的 4 个(pathbased/square/spiral/half_kernel)，删除 t4_8k/t7_10k（无标签无法 ARI 评估，价值最低）。

**推荐方案 A**：代码即文档，删除无益；但从默认路径移除确保不影响日常开发效率。

##### 4.3 深度管线高维适配 — P2 (Week 1-2)
- [ ] DimensionExpert 激活阈值调整：`n_features > 32` → `n_features > 16`（覆盖 Pendigits/Letter 等 16D 数据）
- [ ] PCA+KMeans baseline 管线确保所有高维数据集可达
- [ ] AE_KMeans / SelfLabel 在 CIFAR-10 (raw+gap+resnet18) 上首轮验证
- [ ] 高维数据上 Critic audit 默认开启 `fast_audit`（跳过 graph/boundary 任务避免 O(N²) 爆炸）
- [ ] 运行全量 online benchmark：11 数据集 × 7 专家

##### 4.4 Benchmark 报告体系升级 — P3 (Week 2)
- [ ] 生成 11 数据集 × 7 专家的完整 benchmark 报告矩阵
- [ ] 按领域分组对比：图像(MNIST/Fashion-MNIST/CIFAR-10/USPS) vs 文本(Reuters/News) vs 传感器(HAR) vs 笔迹(Pendigits/Letter)
- [ ] 输出 per-algorithm ranking table（跨数据集汇总）
- [ ] 更新 LaTeX 报告模板：支持多数据集汇总 + 跨领域对比表格
- [ ] 文档同步：README + Engineering Handbook

#### Phase 5 — 学科特化与扩展 (原 Phase 4，后移，目标 4 个月+)

##### 5.1 RAG 知识引擎升级 — P1 (Week 5-6)
> **2026-05-04 决策**：经评估 Milvus/Qdrant/Dify 等方案后，决定**在现有 ChromaDB 基础上深度升级而非替换**。
> 理由：Chromadb 已稳定集成（`knowledge_engine.py`），零运维嵌入式部署；核心痛点不在向量库本身，而在分块/元数据/混合检索缺失。
- [ ] **A. 学术论文逻辑分块 (Logical Chunking)**：PDF 解析时用正则提取 Section Heading
- [ ] **B. 双模式检索 (Lightweight / High Quality)**：侧边栏开关控制
- [ ] **C. 多 Collection 领域感知并行检索**：`ace_bioinfo` / `ace_general` / `ace_cv` 三 collection 并行查询
- [ ] **元数据模型** + **`build_index.py`** 离线预构建脚本

##### 5.2 其他
- [ ] **KIM (Knowledge Integration Mechanism)**：跨语言脚本集成机制
- [ ] 多模态：图像/文本主题聚类
- [ ] 稳定性强化、用户反馈沉淀

#### Phase 6 — ACE-as-a-Service (长期设想，暂不启动)
> **定位调整**：仅作为功能完整后的最终设想，优先级最低；Phase 0–5 全部完成并验证后再立项。
- [ ] RESTful API、鉴权、任务队列、限流/计费等服务化能力

#### 🗑️ 砍/延后项
- **ACE-as-a-Service 提前启动**：禁止在 Phase 0–4 完成前分配资源
- **三领域 RAG 并行**：禁止同时启动多个领域插件
- **LLM Token 预算熔断**：低优先级，当前自愈追踪已到位，暂不实现硬性上限

### 4. 2026-05-19 路线图评估与后续建议

#### 当前完成度总览

| Phase | 状态 | 关键成果 |
|-------|------|---------|
| Phase 0 (工程基线) | ✅ DONE | CI/CD + ruff + mypy + pytest-cov + 沙箱安全 |
| Phase 1 (评估闭环) | ✅ DONE | Critic + Benchmark + Dimension Expert 重构 |
| Phase 2 (集成聚类) | ✅ DONE | Ensemble Consensus + Critic 2.0 闭环 + HITL Lite |
| Phase 3 (深度聚类) | ✅ DONE | Conv-AE + SelfLabel 师生蒸馏 (ARI 0.8454) |
| Phase 6 (智能诊断) | ✅ DONE | NMI + Hopkins 门禁 + 结果缓存 + 审计严格化 |
| Phase 7 (性能优化) | ✅ DONE | Session 瘦身 + Ensemble 熔断 + 自适应超时 |
| Phase 8 (模态感知) | 🟡 80% | ModalityProfile + 降维感知 + Session 持久化 ✅ / 异构模型路由 ❌ |
| Phase 4 (高维真实数据) | 🔴 5% | 数据集加载代码完备，但从未 benchmark 验证 |
| Phase 5 (RAG/KIM) | ⬜ 0% | 受 Governance 约束，Phase 4 完成前不启动 |

#### 建议后续 4 步

**Step 1 — Phase 4.1 数据集首轮 Benchmark (P0, 1-2 天)**
- 运行 `BENCHMARK_FULL`（12 数据集 × zoo 专家），建立高维真实数据基线
- 重点验证：reuters (cosine)、news (cosine)、har (time_series)、cifar10_raw (image)
- 输出 per-algorithm ranking table，识别各数据集的 SOTA 算法
- 预期发现：text 数据集上 cosine 路由 vs 欧氏路由的 ARI 差异；cifar10_raw 上 image pipeline 是否真正生效

**Step 2 — Phase 4.2 2D 数据瘦身 + UI 清理 (P1, 半天)**
- 方案 A（推荐）：保留代码，从 UI 下拉框和 benchmark 默认列表中移除 6 个 SIPU 2D 数据
- 清理 web_demo.py 中废弃的 gallery_legacy / dimension / zoo 静态目录引用

**Step 3 — Dimension Expert TruncatedSVD 管线 (P1, 1 天)**
- `dim_reduction_hint` 已接入，但骨架中尚无专用 `truncated_svd_kmeans` 管线
- 新增管线 #8：TruncatedSVD → KMeans (sparse text 首选路径)
- 更新 `_DECISION_SYSTEM_PROMPT` 和 `_build_smart_defaults()` 对应激活逻辑

**Step 4 — Phase 4.3 深度管线高维适配验证 (P2, 1-2 天)**
- DimensionExpert 激活阈值 `n_features > 32` → `n_features > 16`
- Conv-AE SelfLabel 在 cifar10_raw/cifar10_gap/cifar10_resnet 上首轮验证
- 高维数据 Critic audit 默认开启 `fast_audit`

### 5. 规划决策约束 (Governance)

后续任何 PR、Issue、功能提案必须满足：
1. **地基优先**：Phase 4 未完成前，不启动 Phase 5 (RAG/KIM)。
2. **可度量**：新增自愈/聚类能力必须附带 benchmark 结果。
3. **成本可控**：涉及 LLM 调用的改动必须评估 Token 成本上限。
4. **文档同步**：代码变更须同步更新本 README 与 Engineering Handbook，禁止"TODO 与现实脱节"再次发生。
5. **冲突仲裁**：本章节为唯一权威路线图；若需调整，须显式在 PR 说明中标注 "PM-Review-Override"。

### 2026-05-19 更新 — 降维模态感知 + 项目清理 + Session 持久化修复

**降维与模态感知深化**
- `dim_reduction_hint` 从死代码接入全链路：supervisor 注入 metadata → dimension expert LLM 决策提示 → `_build_smart_defaults()` 模态感知
- 文本数据 (reuters, news)：自动禁用 UMAP/t-SNE，引导 TruncatedSVD + cosine 聚类
- 时序数据 (har)：自动禁用 UMAP/t-SNE，保留 PCA 保持时序连续性
- **har 数据集修复**: metadata 新增 `is_time_series: True`（之前漏标，时序模态未激活）
- **news 加入 BENCHMARK_FULL**: 新增第二个 text/sparse 基准测试点

**项目文件清理**
- 根目录清理：删除 18 个散落 .png 文件（dbscan/kmeans/gmm/spectral 等调试输出）
- `outputs/` 清理：删除 300+ 张 `ensemble_consensus_*.png`、`dimension/`、`zoo/`、`gallery_legacy/`（47 文件）、100+ 历史运行目录
- 其他清理：`benchmark_cache/`、`clustering_output/`、`_minted/`、`artifacts/`、`plots/`、`output/`、LaTeX build artifacts
- `.gitignore` 加固：新增 `/*.png` `/*.pdf` `/*.npy` `output/*` 防御规则

**关键 Bug 修复**
- **Ensemble 图表路径**: `execute_ensemble()` 新增 `output_dir` 参数，共识热力图跟随运行结果目录而非散落 `outputs/` 根
- **LaTeX 生成器空路径**: 空 `plot_path` / `dataset_plot_path` 不再生成 `\includegraphics{}` 导致编译报错
- **Session 持久化**: 用户消息在发送时立即写盘（之前仅在助手响应后延迟保存，崩溃即丢失）；`save_session()` 异常捕获从 `OSError` 扩展为 `Exception`（防御 JSON 序列化 TypeError）

**当前状态**
- **测试**: 132 通过 / 0 失败 / 2 跳过
- **数据集**: 28 个（12 个高维真实 + 6 个合成 + 4 个 SIPU + 6 个 Phase 4 新数据）
- **模态覆盖**: tabular / text (cosine) / time_series / image，BENCHMARK_FULL 含 12 数据集
- **核心代码**: ~23,000 行 Python，60 个 .py 文件

**提交记录** (9 commits):
```
cf32b45 docs: update README with ModalityProfile + KnowledgeEngine fixes, add deps
500fbb4 fix: resolve KnowledgeEngine relative-path fragility
f9d1f3f chore: cleanup — remove generated files, update .gitignore, prune dead code
d9b0dd0 feat: harden image detection heuristics in data_factory
ff1394e feat: ModalityProfile — centralized modality-aware pipeline
1fe98d8 feat: web UI — time-series detection, companion file upload, cluster CSV export
b32d0bf fix: dimension pipeline embedding export + AE/DEC adaptive params
6a925f7 fix: session persistence OSError hardening + expert embedding_path propagation
```

### 2026-05-21 更新 — DTW 时序聚类 + LaTeX 编译修复 + 历史会话恢复 + 渲染健壮性

**心音数据 DTW 时序聚类**
- 新建 `data/user_uploads/heart_normal.meta.json` / `heart_abnormal.meta.json` 伴生元数据文件
- 心音 Mel 频谱图数据 (63 时间步 × 64 mel bands) 上传后自动识别为时序模态
- `web_demo.py` 新增 `data/user_uploads/` 备选 meta.json 搜索路径
- PCA Gate 时序豁免：T ≤ 128 且 N ≤ 5000 时跳过 PCA，保留原始时序形状供 DTW 使用
- Cost Budget 用时序公式 N×T² 替代 N×D，避免 DTW 计算被误判超时
- `zoo_expert.py` 代码生成新增 DTW 双管线：TimeSeriesKMeans(metric="dtw") + SpectralDTW (cdist_dtw → precomputed)
- `topology_expert.py` LLM 系统提示词注入 DTW 拓扑分析说明（cdist_dtw, Sakoe-Chiba 加速）
- N > 500 时自动启用 Sakoe-Chiba 带约束加速

**历史会话渲染修复**
- **根因**: `settings_store.py` 的 key-name 黑名单将 `_report_summary["ranking"]` 误清空为 `[]`
- **修复**: 重命名为 `"ranking_rows"` 避开黑名单，历史会话恢复后排名表、图片、摘要完整显示

**TypeError 崩溃修复**
- **根因**: Python `dict.get("score", 0)` 在 key 存在但 value 为 `None` 时返回 `None`，`float(None)` 崩溃
- **修复**: 全部 8 处脆弱点改为 `.get("score") or 0.0`，覆盖 6 个文件 (`web_demo.py`, `supervisor.py`, `latex_generator.py`, `demo_runner.py`, `benchmark/runner.py`)
- 上游防御：topology expert 提示词新增无标签时 `score = silhouette_score()` 指令

**LaTeX 编译修复（4 项根因）**
- **双重转义**: `_latex_escape` 用占位符先存 `\`，其他字符转义完再恢复，消除 `\textbackslash\{\}` 污染
- **图片路径丢失**: `_p.name` → `relative_to(output_dir)`，保留子目录结构（`zoo/kmeans.png`）
- **Markdown/Emoji 污染**: 新增 `_md_summary_to_latex()` 将 `##`, `**`, `---`, 行内代码等转为 LaTeX 命令
- **`_format_optional` 安全化**: 增加 try/except 防止非 float 值崩溃
- `latex_generator.py` import 路径修正: `ACE_Agent.agent_core.schemas` → `agent_core.schemas`

**输出文件组织结构化**
- 新增 `ACE_OUTPUT_DIR` 沙箱环境变量，指向时间戳运行目录
- 全部 5 个专家 (dimension/graph/zoo/centroid/topology) 的输出图片写入 `{output_dir}/{expert}/` 下
- 不再散落在平铺共享目录，不同 run 之间互不覆盖

**图表分辨率提升**
- 全部 6 处 `dpi` 从 80/100 → 150，`figsize` 从 (6,4) → (8,6)，Web Demo 中图表可读性大幅提升
- 覆盖：`centroid_expert.py`, `topology_expert.py`, `graph_expert.py`, `ensemble_expert.py`, `zoo_expert.py`, `dimension_expert.py`, `supervisor.py`

**改动文件**: `web_demo.py`, `agent_core/supervisor.py`, `tools/coder_sandbox.py`, `tools/latex_generator.py`, `expert_sub_agents/topology_expert.py`, `expert_sub_agents/zoo_expert.py`, `expert_sub_agents/centroid_expert.py`, `expert_sub_agents/dimension_expert.py`, `expert_sub_agents/graph_expert.py`, `expert_sub_agents/ensemble_expert.py`, `scripts/demo_runner.py`, `benchmark/runner.py`

---
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
