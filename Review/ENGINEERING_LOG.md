# ACE Agent 工程日志 (Engineering Log)

> **本文件性质**：项目经理（PM）维护的**单一、追加式（append-only）工程日志**。
> 自 2026-06-22 起由 Claude（接手 PM 角色）创建并维护。
> 此前的架构审计由 Gemini CLI 负责，记录在 `Review/2026-*/assessment.md`；
> 单点 Bug 修复记录在 `Review/fix/NNN_*.md`。本日志是**总账**：
> 记录全局审阅结论、后续每一次问题发现、决策、修改与验证。
>
> **追加规范**：
> - 每条新记录追加到文末「## 后续追加区」，编号递增（#0002、#0003…）。
> - 每条记录必须含：日期 / 类型（发现·决策·修改·验证）/ 严重度 / 关联文件:行 / 结论。
> - 涉及具体 Bug 修复时，在 `Review/fix/` 建立对应编号文件，并在本日志引用它。
> - 不要在此文件堆叠代码长片段；只放结论与指针。

---

## 日志 #0001 — 2026-06-22 — 项目接手 · 全量审阅总报告

**类型**：发现（基线审计） · **PM**：Claude · **范围**：全仓 62 个 Python 文件 / 24,374 行 + 文档 + 配置

### 一、综述：项目目的

ACE Agent（Automated Clustering Expert）是一个**以 LLM 为大脑的多专家自主聚类分析系统**。
用户提供「数据集 + 自然语言指令」，系统自动完成：意图路由 → 数据画像 → 多专家并发出码 →
受限执行器中运行并自愈纠错 → 结果排名 → Critic 后验审计 → 集成共识 → 中文报告 + LaTeX 输出。

定位（README 自述）：**MVP / 早期可用阶段**的研究型 Demo，而非生产服务。核心卖点是
「Think-Act-Fix 自愈」+「ARI 一票否决排名」+「后验审计闭环」三件套。

### 二、实现方法与框架

**端到端主链路**（入口 `web_demo.py` → `ACESupervisor.run()`）：

```
用户 prompt + dataset
  → MasterRouter.analyze_intent()         意图: NEW_TASK / FOLLOW_UP / CODE_EXAMPLE
  → KnowledgeEngine.query()               RAG 注入（ChromaDB + SentenceTransformer）
  → detect_modality()                     模态: tabular/time_series/text/image + 距离度量
  → preflight.classify_data_structure()   结构分类 + geodesic_distortion
  → preflight.fast_hopkins()              聚类倾向门禁（<0.3 跳过密度/谱/图专家）
  → preflight.run_preflight_gates()       规模预算/降维/降采样/流形预处理
  → 专家派发循环（见“谬误#3：实为串行”）
       centroid / topology / zoo / dimension / graph  （critic、ensemble 另行调用）
  → ranking.compute_informed_ranking()    ARI 一票否决 + 共识陷阱检测 + 质心禁令
  → （条件触发）Geodesic Deep Pipeline    UMAP 救助 + 图社区发现 + 降维回落
  → reflection.execute_audit()            Critic 后验审计（3 级超时降级）
  → reflection.handle_audit_feedback()    Critic 2.0 约束重试闭环（≤2 轮）
  → _execute_ensemble()                   Co-association 共识融合（含诚实退避门）
  → reflection.generate_llm_summary()     LLM-3 中文摘要
  → reflection.assemble_final_report()    报告组装 + LaTeX
```

**模块清单（核实后的真实职责与规模）**：

| 模块 | 行数 | 职责 | 备注 |
|------|------|------|------|
| `agent_core/supervisor.py` | 1504 | 主控编排、缓存、集成调用 | 含约 17 个死委托方法（见冗余#1） |
| `agent_core/preflight.py` | 1124 | 派发前：模态/结构/门禁/降维 | 14 个纯函数，无状态 |
| `agent_core/ranking.py` | 357 | 派发后：ARI 排名/陷阱/交叉验证 | 设计清晰 |
| `agent_core/reflection.py` | 476 | 审计/Critic2.0/摘要/组装 | 3 级审计降级逻辑较绕 |
| `agent_core/router.py` | 110 | LLM 意图三分类 | 含死分支（谬误#1） |
| `agent_core/schemas.py` | 192 | 数据类 + detect_modality | 设计良好 |
| `expert_sub_agents/base.py` | 357 | BaseExpert 自愈循环 + 负向优化回退 | 核心 |
| `expert_sub_agents/*` | ~3300 | 7 专家（centroid/topology/zoo/critic/dimension/ensemble/graph） | dimension/graph 最复杂 |
| `tools/coder_sandbox.py` | 820 | 受限执行器（非安全沙箱） | 见谬误#4（最大风险） |
| `tools/llm_client.py` | 458 | 多 provider + fallback + 成本追踪 | 含错误哨兵不一致（谬误#2） |
| `tools/graph_builder.py` | 1891 | 图构建/geodesic/社区发现工具库 | 最大单文件 |
| `tools/data_factory.py` | 1193 | 21 数据集加载 | if/elif 链（约 27 分支，技术债） |
| `web_demo.py` | 1420 | Streamlit 单文件工作台 | `_sidebar_ui` 约 286 行偏臃肿 |
| `benchmark/` | ~870 | 离线/在线基准套件 + CLI | |

**关键设计决策（与 CLAUDE.md 一致，已在代码中核实）**：
- **ARI 唯一排名**（有标签时）：内部指标排除出评分公式，防 Silhouette 偏袒质心算法。
- **共识陷阱检测**：Ensemble 自报一致性≥0.7 但 ARI 低于最佳个体专家 → 判过拟合降级。
- **连通性预检质心禁令**：geodesic_distortion>0.35 时 KMeans/GMM/Birch 禁止夺冠。
- **多级审计超时降级**：normal → fast_audit → auto-relaxed（500 样本 + 2× 超时）。
- **诚实失败/退避**：全专家 ARI<0.2 跳过救助与集成，向用户如实报告失败。

### 三、现状真值（健康度盘点）

- **测试**：单元层抽样 **77/77 通过**（test_core 16 + test_fence_stripping 6 + test_p0 32 +
  test_zoo_expert 21 + test_zoo_score_priority 2；合计 ~61s），核心基础设施健康。
  README 声称 201 项；**实测全量 = 230 项**（见 #0003 全量跑结果：**222 passed / 6 failed /
  2 skipped，耗时 22m23s**）。耗时极长源于真实聚类 + 多次 `ACESupervisor()` 冷启动
  （ChromaDB + SentenceTransformer + torch），并非挂死；快速验证仍建议用免冷启动文件组。
  6 个失败为**测试隔离/资源脆弱性**（隔离复跑全过），非代码 Bug，详见 **#0003** 与待办 T-1。
- **CI**：`.github/workflows/ci.yml` 存在，含 ruff + mypy（continue-on-error 宽松）+ pytest
  覆盖率门槛 `--cov-fail-under=30`（偏低，README 计划提到 60）。
- **工程卫生**：pyproject 配置 ruff/pytest/coverage 规范；pre-commit 存在。MVP 级别尚可。
- **既往修复**：`Review/fix/` 已积累 14 条 Bug 记录，多为元数据传播、维度诅咒、沙箱超时坍缩类。
- **历史审计**：Gemini CLI 出具 5 份 assessment（最新 Audit-04 给 92/100）。注意其结论偏乐观，
  且部分文档已与代码脱节（见冗余#3）。

### 四、冗余清单（Redundancy）

| # | 冗余项 | 证据 | 处置建议 |
|---|--------|------|----------|
| R1 | **supervisor.py 约 17 个死委托方法** | `_classify_data_structure`/`_connectivity_pre_check`/`_compute_best_ari`/`_check_topology_failure`/`_detect_image_data`/`_detect_manifold_topology`/`_apply_highdim_reduction`/`_apply_hard_dim_reduction`/`_subsample_large_dataset`/`_compute_data_cost_budget`/`_apply_manifold_preprocessing`/`_execute_audit`/`_fast_hopkins`/`_prepare_output_dir`/`_save_raw_plot`/`_compute_informed_ranking`/`_cross_validate_graph_winner` 全部无调用方（生产流程直接调模块函数）；仅 `_handle_audit_feedback` 被测试引用，`_execute_ensemble` 内部使用 | 删除无调用者的 shim；测试用到的改为直接调模块函数后再删 |
| R2 | **`expert_sub_agents/deep_representation.py` 死代码** | 已从 `build_expert_registry()` 移除，仅存于 `supervisor.py:40` 与 `__init__.py:24` 的过时注释 | 删除文件 + 清理注释 |
| R3 | **7× `GEMINI.md` 散落 + 与代码脱节** | agent_core/agent_brain/benchmark/docs/expert_sub_agents/scripts/tools 各一份；`agent_core/GEMINI.md` 称 router 判定「QUESTION」意图且「决定激活哪些专家」——二者均错（实为 CODE_EXAMPLE，且选专家在 supervisor 不在 router） | 与 CLAUDE.md/README 合并去重；过时内容更正或删除 |
| R4 | **心音/DTW 模板子项目混入主仓** | `tools/heart_sound_demo.py`、`template_classifier.py`（自述 "standalone, no ACE dependency"）、`template_selector.py`、`ace_template_selector.py`、`evaluate_templates.py` + `templates/*.npy`（normal/abnormal 心音模板）；引用另一 conda 环境 `Tumor_Subtype_Agent` | 与 PM 确认归属：剥离到独立仓，或明确为 ACE 的「应用案例」并隔离到 `examples/` |
| R5 | **大量未跟踪产物/新代码** | git status：Review/、data/、cluster_data/、benchmark_cache/、docs/handbook/、scripts/experiments/、templates/ 等未入库 | 补 `.gitignore` 规则（数据/缓存）+ 提交应入库的源码 |

### 五、谬误与缺陷清单（Defects）

| # | 严重度 | 缺陷 | 证据 | 影响 |
|---|--------|------|------|------|
| D1 | 低 | **router 死分支三元式** | `router.py:107` `"FOLLOW_UP" if ("?" in prompt or "？" in prompt) else "FOLLOW_UP"` 两支恒等 | 逻辑无效；疑似本意是问号→FOLLOW_UP、否则→NEW_TASK，被写坏。需澄清意图后修正 |
| D2 | 中 | **LLM 错误哨兵不一致** | `llm_client.py:365` 返回 `"Error during LLM call: ..."`，但 :368 及下游用 `startswith("Error:")` 判错——该串不以 `Error:` 开头 | 非 fallback 路径的 LLM 错误会被当作**有效内容**：base.py `reply or old_code` → 当作代码下发沙箱必然失败；router/summary 把错误文当答案 |
| D3 | 中 | **「并行派发」声明不实** | `CLAUDE.md:39` 写 "Expert dispatch (parallel)"，但 `supervisor.py:492` 是 `for idx,key in enumerate(active_experts)` 串行循环，全文件无 Thread/async/joblib | 能力虚标；多专家串行使端到端延迟随专家数线性增长 |
| D4 | **高** | **沙箱非安全边界** | `coder_sandbox.py` 同进程 `exec()`；`__import__` 白名单 + os/sys 属性黑名单可被常规手段绕过（如 `().__class__.__bases__[0].__subclasses__()` 触达任意类）；线程超时无法杀线程（失控计算持续耗资源）；内存为 delta-RSS 轮询尽力而为 | 仅在「信任 LLM 输出、本地单机」前提下安全。若开放为服务=任意代码执行风险。文档已坦承，但威胁模型须显式化 |
| D5 | 低 | **`.env` 明文密钥落盘** | `.env` 含真实 `ACE_LLM_API_KEY=sk-...`（已 gitignore，未入库 ✅） | 共享机器/快照泄露面；建议密钥轮换 + 文档化 secret 管理 |
| D6 | 低 | **环境耦合到无关项目** | README 要求在 conda `Tumor_Subtype_Agent` 下运行；无独立 env 规格（仅 requirements.txt） | 复现性差；新人易踩坑 |
| D7 | 低 | **结果缓存近乎永久失效** | `supervisor._code_version()` 用 git HEAD + `git status --porcelain` 脏标记；当前海量未跟踪文件使其恒为 `-dirty` → 缓存基本不命中 | 开发期缓存收益丧失（功能正确，仅低效）；与 R5 联动 |

### 六、结论与建议

**总体判断**：架构思路扎实、设计决策有学术依据，自愈/审计/排名三件套是真实亮点；
代码工程化达到 MVP 中上水平。但存在**三类系统性问题**：
(a) 重构遗留的死代码/兼容层未清理（R1、R2）；
(b) 文档与代码脱节、能力虚标（R3、D3）；
(c) 范围蔓延——心音子项目混入主仓（R4）。
安全方面 D4 是唯一的「高」，但在当前本地 Demo 定位下属可接受的已知限制。

**整改优先级（建议路线）**：
1. ✅ **P0（正确性）**：D2 错误哨兵统一为 `Error:` 前缀；D1 澄清并修正 router 分支。→ #0002
2. ✅ **P0（诚实性）**：D3 把 CLAUDE.md 的「parallel」改为「sequential」。→ #0002
3. ✅ **P1（瘦身）**：R1 删 17 个死 shim、R2 删 deep_representation.py。→ #0004
4. ✅ **P1（文档）**：R3 删除 7 份 GEMINI.md，确立 CLAUDE.md + 本日志为唯一权威。→ #0004
5. ✅ **P1（边界）**：R4 心音子项目已剥离，本仓清理完毕。→ #0004
6. ⬜ **P2（卫生）**：R5 整理 .gitignore 与入库；D5 密钥管理；D6 独立 env；D4 威胁模型显式化。
7. ⬜ **P2（验证）**：T-1 冷启动极慢；T-2 全量套件隔离脆弱性（见 #0003）。

> 以上每项落地时，在「后续追加区」追加 #00xx 记录（发现/决策/修改/验证），
> 涉及 Bug 的同步建 `Review/fix/NNN_*.md`。

---

## 后续追加区

<!--
追加模板：

## 日志 #00NN — YYYY-MM-DD — 一句话标题
**类型**：发现 / 决策 / 修改 / 验证   **严重度**：高/中/低   **关联**：file.py:line, Review/fix/NNN_*.md
**背景**：……
**结论 / 动作**：……
**验证**：（命令 + 结果）
-->

## 日志 #0002 — 2026-06-22 — P0 正确性修复（D1/D2/D3）

**类型**：修改 + 验证 · **严重度**：D2 中 / D1 低 / D3 低（文档）
**关联**：`tools/llm_client.py`、`agent_core/router.py`、`CLAUDE.md:39`、`Review/fix/015`、`Review/fix/016`
**决策来源**：用户选定「P0 正确性修复」线。

**动作**：
1. **D2（中）LLM 错误哨兵统一**：`llm_client.py` 非 fallback 错误返回由
   `"Error during LLM call: ..."` 改为 `"Error: LLM call failed: ..."`，对齐其自身契约
   （:313「error string starting with `Error:`」）与 :368 成本统计判据。→ `Review/fix/016`。
2. **D1（低）router 死分支清除**：`router.py` 异常兜底的恒等三元式
   `"FOLLOW_UP" if (...) else "FOLLOW_UP"` 简化为无条件 `"FOLLOW_UP"`，与类文档「异常兜底
   FOLLOW_UP」一致；本意无歧义，无需追加 PM 决策。→ `Review/fix/015`。
3. **D3（低/文档）并行声明纠正**：`CLAUDE.md:39` 的 "Expert dispatch (parallel)" 改为
   "Expert dispatch (sequential loop over active_experts)"，与 `supervisor.py:492` 串行循环
   实况一致（全文件无并发原语）。未来若真要并发，再单独立项。

**验证**（均已确认）：
- 回归：`pytest tests/test_p0.py tests/test_core.py tests/test_fence_stripping.py`
  = **54 passed, exit 0（28.97s）**，无冷启动、无挂起，改动无回归。
- D2 桩测：用 chat() 抛错的桩 provider 走非 fallback 分支，`chat_completion()` 返回
  `'Error: LLM call failed: boom-net'`，`.startswith('Error:')` == **True**（修复前为 False）→ 契约修复确认。
- D1：`X if cond else X` → `X` 语义等价于原结果值，test_p0（含 router 单测）全过，零回归。
- 注：`test_follow_up.py` 在本会话**冷启动挂起**（150s 超时），系 `ACESupervisor()` 加载
  ChromaDB+SentenceTransformer 所致，**非本次改动引入**——归入 T-1 处理。

**待办联动**：D2 的下游消费侧（`base.py._fix_code/_generate_code`、router、summarize_report
未统一校验 `Error:` 哨兵）属 P1 健壮性改进，已在 fix/016 标注，后续追加。

## 日志 #0003 — 2026-06-22 — 全量测试套件实测：222 passed / 6 failed（隔离脆弱性）

**类型**：验证 + 发现 · **严重度**：中（CI 可靠性，非功能正确性）· **状态**：open（待根因）
**关联**：`tests/test_zoo_expert.py`、`tests/test_zoo_score_priority.py`、`Review/fix/010`

**背景**：会话初启动的后台全量 `pytest -q` 历时 **22m23s** 后完成（exit 0 但有失败），
结果 **222 passed / 6 failed / 2 skipped / 72 warnings**。该任务在 P0 改动之前启动，
跑的是改动前代码，故失败**与 #0002 无关**。

**6 个失败**（全部集中在 zoo）：
- `test_zoo_expert.py`：test_results_nonempty_on_moons / test_dbscan_result_exists /
  test_dbscan_score_or_ari_threshold / test_kmeans_result_exists
- `test_zoo_score_priority.py`：test_score_equals_ari_when_labels_provided /
  test_score_falls_back_to_silhouette_when_no_labels

**关键判定**：单独复跑这两文件 → `23 passed, exit 0 (36.7s)`。**隔离下全过**，
故非代码 Bug，而是**全量套件的测试隔离/资源脆弱性**。

**根因假设**（待证）：
1. Windows 上 MKL KMeans/MiniBatchKMeans 已知内存泄漏（套件 warnings 多次报）——
   222 项真实聚类累积后，zoo 沙箱在尾部触达内存看门狗/超时而失败。
2. 全局态污染：某前置测试改了 `os.environ["ACE_SANDBOX_TIMEOUT_SEC"]` 或
   `.ace_result_cache.json` 等共享状态，影响后续 zoo 沙箱执行。
3. 非杀死线程的沙箱超时（见 D4）累积的僵尸线程导致尾部资源退化。

**动作**：暂不改（未根因，避免盲改）。归入 T-1 同时新开 **T-2**：
设 `OMP_NUM_THREADS=2`（消 MKL 泄漏）+ 测试夹具复位环境变量/缓存 + 评估 zoo 测试加
`pytest` 资源标记或 `--forked` 隔离。根因落定并修复后建 `Review/fix/017` 并在此引用。

**验证命令留痕**：
`pytest tests/test_zoo_expert.py tests/test_zoo_score_priority.py` → 23 passed（隔离）。

## 日志 #0004 — 2026-06-22 — 瘦身 + 去重 + 心音子项目清理（R1/R2/R3/R4）

**类型**：修改 + 验证 · **严重度**：低（清理，无功能面变更）· **授权**：用户明示
**关联**：`agent_core/supervisor.py`、`CLAUDE.md`、删除清单见下

**动作**：
- **R4 心音子项目清理**（用户确认已剥离）：删除 5 个工具脚本
  （`heart_sound_demo.py` / `template_classifier.py` / `template_selector.py` /
  `ace_template_selector.py` / `evaluate_templates.py`）+ `templates/` 目录（DTW 模板 .npy/.csv）
  + `data/user_uploads/heart_{abnormal,normal,sounds_ready}.csv(.meta.json)`。
  **共释放约 262MB**。`data/user_uploads/clustering_data*` 非心音，已保留。
  删除前确认：5 个脚本零 import 引用；`templates/*.npy` 仅被这些脚本读取；
  data_factory 无心音数据集加载器（mel 提及纯属图像形状启发式注释，保留）。
- **R2 死代码**：删除 `expert_sub_agents/deep_representation.py`（已下线、`__init__` 不导入、仅注释引用）。
- **R1 死委托层**：从 `supervisor.py` 切除 **17 个零调用 shim**，**1504 → 1332 行（−172）**。
  保留 `_execute_ensemble`（内部调用）、`_handle_audit_feedback`（测试调用）及全部真实方法。
- **R3 文档去重**：删除全部 **7 份 GEMINI.md**（与 CLAUDE.md 重复 + 指向已删文件 + 既有漂移）。
  确立 **CLAUDE.md + 本工程日志为唯一权威文档**。
- **顺带**：修正 supervisor.py 顶部 docstring 残留的 `deep_representation`/`_execute_audit()`
  引用与「并行」措辞；CLAUDE.md 模块表 supervisor 行数 ~1500 → ~1332。

**验证**（全部通过）：
- 导入：`ACESupervisor` 正常导入，`build_expert_registry()` 产出 7 专家
  （centroid/critic/dimension/ensemble/graph/topology/zoo）；17 shim 已消失、5 保留方法在位。
- 测试：`pytest test_p0 + test_core + test_fence_stripping + test_zoo_expert +
  test_zoo_score_priority` → **77 passed（51s）**，无回归（与瘦身前一致）。
- Lint：`ruff check supervisor.py router.py llm_client.py` → All checks passed。
- 变更集：tracked 删除 2（deep_representation.py、ace_template_selector.py）+ tracked 修改 4
  （CLAUDE.md、supervisor.py、router.py、llm_client.py）；其余为 untracked 直接移除。**未提交**。

**过程留痕（值得记）**：R1 用「断言锚点 + 按行区间删除」的一次性脚本执行。
脚本的前置断言**两次拦截了行号错误**——其中一次若不拦截会误删要保留的 `_error_report`。
印证「删代码前先 assert 边界」的价值，已据权威读取校正后通过。脚本运行后自删。

**剩余队列**：R5（.gitignore/未跟踪文件治理，本仓仍有 data/、cluster_data/、benchmark_cache/
等大体量未跟踪物）、D4-D7、T-1/T-2。等指示继续。
