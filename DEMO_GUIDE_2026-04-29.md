# ACE Agent 演示手册 — 2026-04-29 周进展汇报

> 面向导师/评审的现场演示指南。涵盖本周四项核心交付、分环节演示脚本、关键指标展示、以及下一步推进方向。

---

## 一、本周工作概览（4 项核心交付）

| # | 交付项 | 类别 | 关键结果 |
|---|---|---|---|
| 1 | **AE_KMeans 深度架构升级 (Phase 4)** | 算法能力 | digits ARI 0.3238 → 0.5094 (+57%)，手调 0.6102 突破 0.6 目标线 |
| 2 | **沙箱安全加固** | 基础设施 | 拦截 18 个高危 os/sys 方法 + 封禁 15 个危险模块 |
| 3 | **Critic 架构重构** | 系统架构 | 从"并行投票人"重构为"后验独立审计方"，裁判与运动员分离 |
| 4 | **工程质量基线** | 工程规范 | CI 全绿（ruff+mypy+pytest-cov），142 测试通过，README 路线图与代码对齐 |

---

## 二、演示数据集推荐

现有 12 个内置数据集，按演示目的分为三档：

### 第一档：必演（展示核心能力）

| 数据集 | 特征 | 演示目的 | 推荐参数 |
|---|---|---|---|
| **smile** (笑脸) | 2维，非凸，3类，可视化直观 | 开场暖场，展示基本聚类流程 + Router 意图识别 | `n=480, noise=0.06` |
| **moons** (月牙) | 2维，高度非凸，2类 | 展示密度聚类（DBSCAN/HDBSCAN）vs 质心方法的差异，Zoo 专家价值 | `n=420, noise=0.06` |
| **digits** (手写数字) | 64维，10类 | **核心展示**：Dimension Expert 自动激活 + AE_KMeans 深度管线产出 | `n=480` |

### 第二档：选演（展示特定特性）

| 数据集 | 特征 | 演示目的 |
|---|---|---|
| **high_dim** (高维) | 100维，5类 | 展示高维自动感知 + 降维管线自动激活 |
| **iris** (鸢尾花) | 4维，3类 | 经典基线，展示有标签时的 ARI 评估 |
| **blobs** (团状) | 2维，3类 | 最简单场景，展示 KMeans/GMM 基线性能 |

### 第三档：扩展（展示深度能力）

| 数据集 | 特征 | 演示目的 |
|---|---|---|
| **mnist** (手写体) | 784维，10类 | 真实高维图像数据，展示 AE 深度管线在超高维上的表现 |
| **s_curve** (流形) | 3维流形，3类 | 展示流形学习方法（UMAP）的价值 |

> **建议**：时间充裕时加演 **mnist**（2000 样本，784 维），强调 Phase 3 深度聚类管线的硬件优势（本地 RTX 4060 Ti 8G + 远端 A4000 16G）。

---

## 三、演示脚本（按环节分步）

### 环节 1：开场 — 基本聚类流程（约 5 分钟）

**目的**：展示系统从意图识别到结果产出的完整闭环。

**操作步骤**：
1. 侧边栏 → Model Config → 确认 API Key 已配置（DeepSeek 推荐）
2. Data Config → 内置数据 → 选择 **smile (笑脸数据集)**
3. 样本量保持 `480`，点击 "Preview Data Distribution" 展示原始分布
4. 在聊天框输入：
   > 帮我分析这个笑脸数据集，运行所有可用的聚类算法，选出最优结果。

**展示要点**：
- **语义路由**：观察状态栏显示 "意图判定: NEW_TASK"，而非关键词匹配
- **专家激活**：trace 显示 `[centroid] / [topology] / [zoo]` 三家并行执行，dimension 因维度=2 不激活
- **自愈日志**：若某专家首次失败，观察 `[fix:1]` 自动修复重试
- **排行榜**：多个算法按 score 排序，优胜者高亮
- **可视化**：原始分布图 vs 最优聚类结果图对比

**讲解话术**：
> "ACE Agent 不是传统的调参工具。你只需要用自然语言描述需求，系统通过 LLM 语义路由自动判定意图，然后并行调度多个专家 Agent。每个专家在受限沙箱中生成并执行代码，如果出错会自动分析 Traceback 并修复，最多重试 3 次。"

---

### 环节 2：密度聚类 — Zoo 专家价值（约 5 分钟）

**目的**：展示非凸数据上密度专家不可替代的价值，以及 P0.5 Bug 修复效果。

**操作步骤**：
1. Data Config → 选择 **moons (月牙数据集)**，样本量 `420`
2. 输入指令：
   > 分析这个月牙数据集的聚类结构。

**展示要点**：
- **Zoo 专家激活**：trace 中显示 `[zoo]` 与 centroid/topology 并行运行
- **DBSCAN/HDBSCAN 胜出**：对于非凸月牙数据，密度方法理应优于 KMeans/Spectral
- **Critic 后验审计**：trace 最后显示 `【审计】对最优结果 'DBSCAN' 启动独立后验审计...` → `裁决: endorsed/qualified`
- 强调：这是 Bug 1 修复后的效果（此前月牙数据会错误选 Spectral 为最优）

**讲解话术**：
> "这张月牙数据集有两个交错的新月形，传统 KMeans 会强行切出错误的球状簇。我们在此前发现了一个关键 Bug —— 密度聚类专家（Zoo）从未被激活，导致系统只能在非密度方法里瞎选。修复后，DBSCAN/HDBSCAN 能正确识别这种非凸结构。此外，注意看 trace 最后一行 —— Critic 审计专家现在不在并行池中，而是在最优结果选出后独立进行后验审查。"

---

### 环节 3：核心展示 — AE_KMeans 深度管线（约 10 分钟）

**目的**：展示本周最大亮点——AE_KMeans 深度架构升级效果。

**操作步骤**：
1. Data Config → 选择 **digits (手写数字特征)**，样本量 `480`
2. 输入指令：
   > 对这个高维手写数字数据集进行降维和聚类分析，使用所有可用管线。

**展示要点**：

**a) 自动感知**
- trace 显示 `检测到数据维度为 64，已自动激活维度专家`
- Dimension Expert 的 `PRE_INJECT` 注入了 AE/Torch 模块

**b) 多管线并行**
- 查看 trace 中 Dimension Expert 的 artifacts：
  - `PCA_KMeans` — 经典线性降维
  - `PCA_GMM` — 概率模型聚类
  - `AE_KMeans` — **深度去噪自编码器管线**
  - `UMAP_KMeans`（若 UMAP 已安装）
- 强调：6 条管线确定性骨架执行，LLM 仅做 ~180 token 的 JSON 参数决策

**c) 深度 AE 架构**
- 如果不便查看代码，口头介绍：
  > "AE_KMeans 管线本周完成了从浅层 MLP 到深层堆叠去噪自编码器的升级。新架构每层包含 Linear → BatchNorm1d → LeakyReLU → Dropout，训练使用 CosineAnnealing 学习率调度 + Early Stopping 防止过拟合，潜在空间使用 GMM 而非 KMeans 聚类——因为 GMM 能捕获椭圆状分布。大模型可以通过 JSON 决策控制 8 个超参数：隐藏层维度、学习率、dropout 率、噪声标准差、聚类方法等。"

**d) 量化结果**
- 选中排行榜，展示：
  - `AE_KMeans ARI ≈ 0.51`（benchmark 480 样本子集）
  - `UMAP_KMeans ARI ≈ 0.88`（当前最佳）
  - 提及全量 1797 样本手调 AE_KMeans 达 0.6102
- 打开侧边栏 "LLM Call Monitor" 展示 Token 消耗和成本

**讲解话术**：
> "这是我们本周最大的进展。上周 AE_KMeans 在 digits 上的 ARI 只有 0.3238，是所有管线中垫底的。经过三步重构——深层非线性架构、防过拟合机制、LLM 全参数控制——ARI 提升了 57% 达到 0.5094。但我们清醒地认识到离 UMAP_KMeans 的 0.88 还有本质差距，原因是两阶段训练（先重建后聚类）的优化目标不一致——这正是我们下一步 DEC 联合优化的方向。"

---

### 环节 4（可选）：Critic 审计详情（约 3 分钟）

**目的**：展示 Critic 重构后的审计报告结构。

**操作步骤**：
1. 在上一轮聚类完成后，不重新跑数据
2. 查看 trace 中审计相关内容

**或在对话中追问**：
> 刚才的最优结果有多可靠？帮我审查一下。

**展示要点**：
- Critic 输出的 `audit_report` 结构：
  - `endorsement`: endorsed / qualified / qualified_with_warning
  - `confidence_level`: 0-1 综合置信度
  - `overfitting_risk`: low / medium / high
  - `stability_score`: Bootstrap 稳定性
  - `hopkins`: 聚类趋势
  - `winner_k_consistency`: 优胜者 k 与 CVI 共识是否一致

**讲解话术**：
> "Critic 之前作为'运动员'跟聚类专家一起跑，产出竞争性 score 混在同一个排行榜里——这就像让裁判也上场踢球。这周我们把它重构为'独立审计方'，在最优结果选出后才运行，只输出审计报告不参与排名。审计包括 Hopkins 统计检验、Bootstrap 稳定性分析和过拟合风险评估。"

---

### 环节 5：追问与知识检索（约 3 分钟）

**目的**：展示 FOLLOW_UP 意图和 RAG 知识增强。

**操作步骤**：
1. 在上一轮聚类完成后
2. 输入追问：
   > 为什么这个数据集上 DBSCAN 比 KMeans 表现好？
3. 再输入：
   > 给我一段用 sklearn 做 DBSCAN 聚类的代码示例。

**展示要点**：
- **FOLLOW_UP**：Router 识别为追问而非新任务，不重新跑实验，基于上轮报告上下文回答
- **CODE_EXAMPLE**：Router 识别为代码示例请求，不走沙箱，直接用 LLM 生成 Markdown 代码块
- **RAG 增强**：trace 显示 `【RAG】成功检索到相关学术理论片段`（如果知识库已配置）

---

## 四、技术亮点速查表

| 维度 | 指标 | 当前值 | 说明 |
|---|---|---|---|
| **测试** | 测试数量 | 142 项全通过 | 96 核心 + 46 benchmark |
| **测试** | 覆盖率 | ~65% | CI 门槛 30%（已远超） |
| **代码** | 核心规模 | ~6,500 行 | 含 benchmark 套件 |
| **CI** | Lint/Type | ruff + mypy 全绿 | GitHub Actions 自动运行 |
| **可观测** | LLM 追踪 | `llm_trace.jsonl` | 每次调用记录 prompt/completion tokens + cost |
| **安全** | 沙箱拦截 | 18 方法 + 15 模块 | os.remove/system/exit, subprocess, socket 等 |
| **安全** | 资源限额 | 60s timeout + 2 GiB 内存 | 硬限制，超限抛出异常 |
| **LLM** | 供应商 | 4 种 Provider | DeepSeek / DashScope / OpenAI / Moonshot + 自动 fallback |
| **专家** | 已注册 | 7 个 | centroid / topology / zoo / critic / dimension / deep_representation / multi_view |
| **算法** | 聚类算法 | 10 种 | KMeans, GMM, DBSCAN, HDBSCAN, Agglomerative, Spectral, OPTICS, Birch, AffinityPropagation, MeanShift |
| **降维管线** | Dimension Expert | 5 条 | PCA+KMeans / PCA+GMM / UMAP+KMeans / tSNE+KMeans / AE+KMeans |
| **AE 深度** | AE_KMeans ARI | 0.5094 (+57%) | 深层去噪 AE + GMM 潜在聚类 |
| **最佳** | UMAP_KMeans ARI | 0.8764 | digits 64 维当前最高分 |

---

## 五、未来推进方向

### 短期（Phase 3 续）

| 优先级 | 任务 | 说明 |
|---|---|---|
| **P0** | **DEC / IDEC 联合优化** | 当前 AE+KMeans 两阶段训练（重建+聚类分离）是 ARI 瓶颈。DEC 通过 KL 散度联合优化嵌入和聚类分配，目标突破 ARI 0.7 |
| **P1** | **显存自适应** | 根据 8G (RTX 4060 Ti) vs 16G (A4000) 自动选 batch/精度档位 |
| **P1** | **GPU/CPU 回退** | 无 CUDA 环境自动降级到 sklearn 基线 |
| **P2** | **Contrastive 表格嵌入** | SCARF 类对比学习用于表格数据深度表示 |

### 中期（Phase 4）

| 优先级 | 任务 | 说明 |
|---|---|---|
| **P1** | **RAG 插件（生信单领域）** | 向量库 + 语料治理，先跑通生物信息学一个领域 |
| **P2** | **Human-in-the-Loop 降级版** | 结果打标 + 重新触发，不做在线约束求解 |

### 长期（Phase 5，暂缓）

- ACE-as-a-Service：仅作为功能完整后的最终设想

### 已搁置

- 集成聚类（Ensemble Consensus）：资源集中于深度管线
- LLM Token 预算熔断：当前追踪已到位，暂不设硬上限

---

## 六、演示环境检查清单

**演示前确认**：

- [ ] Conda 环境 `Tumor_Subtype_Agent` 已激活
- [ ] API Key 已配置（侧边栏 Model Config → API Key 填入）
- [ ] UMAP 已安装：`pip install umap-learn`（可选，增强降维管线）
- [ ] PyTorch 可用：`python -c "import torch; print(torch.cuda.is_available())"`（AE 管线依赖）
- [ ] 142 测试全绿：`python -m pytest ACE_Agent/tests/ -x --tb=short`
- [ ] 启动 Web Demo：`streamlit run ACE_Agent/web_demo.py`
- [ ] LLM Call Monitor 已清零（可选，便于观察实时 Token 消耗）

**常见问题预案**：

| 问题 | 应对 |
|---|---|
| LLM API 超时/失败 | 展示 Fallback Provider 自动切换机制（trace 中可见 `provider_fallback` 事件） |
| AE 管线报错 | 展示自愈机制：trace 中可见 `fix:1` 重试日志，自动修复后成功 |
| 聚类效果不佳 | 强调这是 MVP 阶段，非生产系统；且不同算法有不同适用场景，正体现多专家并行的价值 |
| Token 消耗高 | 展示侧边栏成本面板，说明自愈重试已被追踪，未来可设预算熔断 |
