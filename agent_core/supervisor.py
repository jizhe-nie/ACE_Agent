from __future__ import annotations

import contextlib
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from ACE_Agent.agent_brain.knowledge_engine import KnowledgeEngine
from ACE_Agent.agent_core.router import MasterRouter
from ACE_Agent.agent_core.schemas import (
    AlgorithmRunResult,
    DatasetBundle,
    ProfileReport,
    RoutingDecision,
    SupervisorReport,
)
from ACE_Agent.expert_sub_agents import build_expert_registry
from ACE_Agent.tools.graph_builder import GraphBuilder
from ACE_Agent.tools.latex_generator import LatexReportGenerator
from ACE_Agent.tools.llm_client import LLMSettings, UniversalLLMClient


class ACESupervisor:
    """主控编排器 (Orchestrator)：协调多代理完成复杂任务。

    P0.5 变更（2026-04-20）：
    - 专家注册表改用 build_expert_registry()，包含 zoo 专家（含 DBSCAN/HDBSCAN）。
    - 默认激活策略：centroid + topology + zoo（三家并行）。
    - dimension / deep_representation / multi_view 已注册但默认不激活；
    - Critic 重构为后验审计方（2026-04-29）：从并行池移除，在选出最优结果后
      独立运行 _execute_audit()，输出结构化 audit_report，不参与排名。
    - 新增 CODE_EXAMPLE 分流路径（只返回代码 Markdown，不走沙箱）。
    - 每个专家执行用 try/except 包裹，异常转为日志不中断编排。
    - _error_report 增强：汇总每个专家最后 3 行日志作为排错依据。
    """

    # 默认激活的专家 key（ensemble 在第 2 步由 _execute_ensemble 独立调用）
    _DEFAULT_ACTIVE_EXPERTS: list[str] = ["centroid", "topology", "zoo"]

    def __init__(self) -> None:
        self.router = MasterRouter()
        self.knowledge_engine = KnowledgeEngine()
        try:
            self.knowledge_engine.ingest_docs()
        except Exception as e:
            print(f"知识库初始化警告: {e}")

        self.memory: list[dict[str, Any]] = []
        self.last_report: SupervisorReport | None = None

        # 完整专家注册表（含 zoo, critic, dimension, deep）
        self.experts: dict[str, Any] = build_expert_registry()

    def run(
        self,
        dataset: DatasetBundle | None,
        user_prompt: str,
        llm_settings: LLMSettings,
        intent_data: dict[str, Any] | None = None,
        constraints: dict[str, Any] | None = None,
        progress_callback: Any = None,
    ) -> SupervisorReport:
        """核心编排流程。

        progress_callback: optional callable(message: str, step: int, total: int)
            for real-time UI progress updates.
        """
        def _progress(msg: str, step: int = 0, total: int = 1) -> None:
            if progress_callback:
                try:
                    progress_callback(msg, step, total)
                except Exception:
                    pass

        # 1. 语义路由
        if not intent_data:
            intent_data = self.router.analyze_intent(user_prompt, self.memory, llm_settings)

        intent = str(intent_data.get("intent", "NEW_TASK")).upper()
        reasoning = intent_data.get("reasoning", "接收到指令")

        # RAG 增强：检索背景知识
        knowledge_context = self.knowledge_engine.query(user_prompt)
        if knowledge_context:
            trace_msg = "【RAG】成功检索到相关学术理论片段，已注入上下文。"
            user_prompt = f"{knowledge_context}\n用户指令: {user_prompt}"
        else:
            trace_msg = "【RAG】未在知识库中匹配到强相关的理论支持。"

        trace = [f"【主控】确认意图: {intent}", f"【逻辑】{reasoning}", trace_msg]

        # HITL约束检测
        if constraints and constraints.get("reference_labels"):
            trace.append(
                f"【HITL】检测到人工标注参考标签（{len(constraints['reference_labels'])} 个数据点），"
                "将作为约束传递给各专家。"
            )

        # 2. 意图分流
        if intent == "FOLLOW_UP":
            return self._handle_follow_up(user_prompt, llm_settings, trace)

        if intent == "CODE_EXAMPLE":
            return self._handle_code_example(user_prompt, llm_settings, trace)

        # 3. NEW_TASK：需要数据集
        if not dataset:
            return self._error_report(
                "意图识别为新任务，但未识别到数据。请先配置/预览数据，或直接进行学术提问。",
                trace,
                expert_logs={},
            )

        # Phase 1 增强：高维数据自动感知
        n_features = dataset.X.shape[1]
        active_experts = list(self._DEFAULT_ACTIVE_EXPERTS)
        if n_features > 2 and "dimension" not in active_experts:
            trace.append(f"【主控】检测到数据维度为 {n_features}，已自动激活维度专家。")
            active_experts.append("dimension")

        # Phase 3 Topology-Aware: classify data structure and route experts
        structure = self._classify_data_structure(dataset)
        trace.append(
            f"【数据结构分类】{structure['structure_class']}"
            f"（geodesic_distortion={structure.get('geodesic_distortion', 0):.3f}）"
        )
        strategy = structure.get("recommended_strategy", {})
        if strategy.get("activate_graph_expert") and "graph" not in active_experts:
            trace.append("【主控】检测到图连通结构，已激活图结构专家（GraphExpert）。")
            active_experts.append("graph")
        if strategy.get("centroid_suppress") and "centroid" in active_experts:
            active_experts.remove("centroid")
            trace.append("【主控】已抑制质心专家（欧氏距离算法不适合图连通结构数据）。")

        # Phase 3.1: maze connectivity mode detection
        maze_connectivity_mode = structure.get("structure_class") == "graph_connected"
        if maze_connectivity_mode:
            trace.append(
                "【主控】已启用迷宫连通域模式（maze_connectivity_mode）。"
                "优化优先级：图连通性 > 最短路径一致性 > 瓶颈保留。"
                "质心算法已硬性排除，CPS 为主要评分指标。"
            )

        # Phase 3.2: community native mode — graph community discovery is PRIMARY
        community_native_mode = structure.get("structure_class") == "graph_connected"
        if community_native_mode:
            trace.append(
                "【主控】已启用社区发现原生模式（community_native_mode）。"
                "图社区发现为 PRIMARY pipeline。聚类数由 modularity 最优决定。"
                "集成共识仅用于稳定性分析，不覆盖图分区结果。"
            )

        return self._execute_full_analysis(
            dataset, user_prompt, llm_settings, trace, active_experts,
            constraints=constraints, progress_callback=progress_callback,
            structure=structure,
            maze_connectivity_mode=maze_connectivity_mode,
            community_native_mode=community_native_mode,
        )

    # ------------------------------------------------------------------
    # 完整分析流
    # ------------------------------------------------------------------

    def _execute_full_analysis(
        self,
        dataset: DatasetBundle,
        prompt: str,
        settings: LLMSettings,
        trace: list[str],
        active_experts: list[str],
        constraints: dict[str, Any] | None = None,
        progress_callback: Any = None,
        structure: dict[str, Any] | None = None,
        maze_connectivity_mode: bool = False,
        community_native_mode: bool = False,
    ) -> SupervisorReport:
        """执行完整的自动化聚类实验流。"""
        if structure is None:
            structure = {}

        def _progress(msg: str, step: int = 0, total: int = 1) -> None:
            if progress_callback:
                try:
                    progress_callback(msg, step, total)
                except Exception:
                    pass

        output_dir = self._prepare_output_dir(dataset.name)
        all_results: list[AlgorithmRunResult] = []
        expert_logs: dict[str, list[str]] = {}

        # ---- Phase 2.4: Manifold pre-processing for complex topology ----
        # Detect non-convex / manifold data and apply UMAP / SpectralEmbedding
        # BEFORE experts are dispatched, so they cluster the "unfolded" data.
        # Phase 3: uses _classify_data_structure() result for detection.
        working_dataset = dataset
        _manifold_detected = structure.get("structure_class") in ("manifold", "non_convex", "graph_connected") or self._detect_manifold_topology(dataset)
        if _manifold_detected and dataset.X.shape[1] > 2:
            embedded = self._apply_manifold_preprocessing(dataset, trace)
            if embedded is not None:
                working_dataset = embedded
                trace.append("【流形预处理】已将嵌入数据集作为后续专家的输入。")

        n_experts = len(active_experts)

        for idx, key in enumerate(active_experts):
            expert = self.experts.get(key)
            _progress(f"正在运行 {key} 专家 ({idx + 1}/{n_experts})...", idx + 1, n_experts)
            if expert is None:
                trace.append(f"【主控】警告：专家 '{key}' 未在注册表中找到，跳过。")
                continue
            try:
                expert_results = expert.execute_with_self_correction(working_dataset, prompt, settings, constraints=constraints)
                all_results.extend(expert_results)
                trace.extend(expert.last_logs)
                expert_logs[key] = list(expert.last_logs)
                n_results = len(expert_results)
                _progress(f"[{key}] 完成 — 产出 {n_results} 个结果", idx + 1, n_experts)
            except Exception as exc:
                err_msg = f"【主控】专家 '{key}' 执行时发生未捕获异常: {exc}"
                trace.append(err_msg)
                expert_logs[key] = [err_msg]
                _progress(f"[{key}] 异常: {exc}", idx + 1, n_experts)

        if not all_results:
            return self._error_report(
                "所有专家执行均失败。",
                trace,
                expert_logs=expert_logs,
            )

        # Phase 4: Graph quality gate — check if graph construction was sound
        _graph_quality_ok = True
        _graph_shortcut_ratio = 0.0
        for r in all_results:
            if r.algorithm_name == "_graph_meta":
                _gq = r.metrics.get("graph_quality_pass", True)
                _sr = r.metrics.get("shortcut_ratio", 0.0)
                if not _gq:
                    _graph_quality_ok = False
                    _graph_shortcut_ratio = _sr
                    trace.append(
                        f"【图质量门禁】图构建质量不合格！"
                        f"shortcut_ratio={_sr:.2%}。"
                        f"在此图上的社区发现结果不可信。"
                    )
                break
        # Also check for graph quality warning artifact
        for r in all_results:
            if r.algorithm_name == "_graph_quality_warning":
                _warn_msg = r.metrics.get("warning", "图质量警告")
                _graph_quality_ok = False
                _graph_shortcut_ratio = r.metrics.get("shortcut_ratio", 0.0)
                trace.append(f"【图质量门禁】{_warn_msg}")
                break

        if not _graph_quality_ok and community_native_mode:
            trace.append(
                "【图质量门禁】shortcut edge ratio 过高，图拓扑不可信。"
                "建议：增大 kNN_k 或使用 build_wall_aware_graph() 重新构造图。"
                "当前社区发现结果仅供参考，不作为最终输出。"
            )

        # Phase 5.1: 几何连通性预检 — 检测长程曲线结构
        _conn_check = self._connectivity_pre_check(dataset, trace)
        if _conn_check["long_range_curve"]:
            trace.append(
                "【连通性预检】检测到长程曲线/流形结构！"
                "KMeans/GMM 等质心算法禁止判为优胜，即使 ARI 暂时领先。"
                "强制拓扑/密度专家进行网格参数搜索。"
            )
            # Inject grid-search constraint for topology & zoo experts
            _grid_constraint = {
                "grid_search": True,
                "reason": "connectivity_pre_check_detected_long_range_curve",
            }
            if constraints is None:
                from copy import deepcopy
                constraints = deepcopy(_grid_constraint)
            else:
                constraints = {**constraints, **_grid_constraint}

        # 排序与摘要 — Phase 5: ARI 一票否决制 + 连通性禁令
        ranking = self._compute_informed_ranking(
            all_results, dataset, trace,
            centroid_ban=_conn_check["centroid_ban"],
        )
        best = ranking[0]

        # Phase 5.1: 低 ARI 看板预警
        _best_ari = self._compute_best_ari(ranking, dataset)
        if _best_ari is not None and _best_ari < 0.7:
            trace.append(
                f"⚠️ 预警：当前最高 ARI 仅 {_best_ari:.3f}（< 0.7），"
                f"所有算法均未能有效捕捉数据结构，聚类结果仅供参考，"
                f"建议人工干预参数或更换算法策略。"
            )

        # Phase 5.1: 全局低 ARI 诚实失败检测
        _global_low_ari = False
        if dataset.y is not None:
            from sklearn.metrics import adjusted_rand_score as _global_ari_fn
            _y_true = np.asarray(dataset.y, dtype=int).ravel()
            _all_aris = []
            for _r in all_results:
                _rl = getattr(_r, "labels", None)
                if _rl is not None and hasattr(_rl, "__len__") and len(_rl) > 0:
                    try:
                        _all_aris.append(float(_global_ari_fn(_y_true, np.asarray(_rl, dtype=int).ravel())))
                    except Exception:
                        pass
            if _all_aris and max(_all_aris) < 0.5:
                _global_low_ari = True
                trace.append(
                    f"【低召回警告】所有算法 ARI 均 < 0.5（max={max(_all_aris):.3f}），"
                    f"数据结构疑似高度复杂（极窄薄流形/并行曲线），当前模型召回率不足。"
                    f"触发 Critic 增强级 Bootstrapping 稳定性校验。"
                )

        # Phase 5.3: FAILED verdict — when all independent ARI < 0.7 and labels exist
        _deep_pipeline_triggered = False
        _all_independent_aris: list[float] = []
        if dataset.y is not None:
            from sklearn.metrics import adjusted_rand_score as _fail_ari_fn
            _y_fail = np.asarray(dataset.y, dtype=int).ravel()
            for _r in all_results:
                if _r.algorithm_name in ("EnsembleConsensus",) or _r.algorithm_name.startswith("_"):
                    continue
                _rl = getattr(_r, "labels", None)
                if _rl is not None and hasattr(_rl, "__len__") and len(_rl) > 0:
                    try:
                        _all_independent_aris.append(
                            float(_fail_ari_fn(_y_fail, np.asarray(_rl, dtype=int).ravel()))
                        )
                    except Exception:
                        pass
            if _all_independent_aris and max(_all_independent_aris) < 0.7:
                _deep_pipeline_triggered = True
                _best_independent_ari = max(_all_independent_aris)
                trace.append(
                    f"【FAILED / 结构识别失败】所有独立算法 ARI 均低于 0.7"
                    f"（最高 ARI={_best_independent_ari:.3f}），"
                    f"系统判定当前欧氏空间方法无法有效捕捉数据结构。"
                    f"自动触发 Deep Pipeline（DimensionExpert + UMAP 流形嵌入）。"
                )
                _progress("FAILED 裁决：触发 Geodesic Deep Pipeline...")
                _deep_results: list[AlgorithmRunResult] = []

                # Path A: UMAP embedding + topology expert (OPTICS/HDBSCAN on manifold)
                try:
                    import numpy as _gd_np
                    _X_orig = _gd_np.asarray(working_dataset.X, dtype=float)
                    _n_feat = _X_orig.shape[1] if _X_orig.ndim == 2 else 1
                    if _n_feat > 2:
                        import umap
                        _reducer = umap.UMAP(
                            n_components=min(3, _n_feat),
                            n_neighbors=min(30, max(5, _X_orig.shape[0] // 50)),
                            min_dist=0.1, metric="euclidean", random_state=42,
                        )
                        _X_umap = _reducer.fit_transform(_X_orig)
                    else:
                        _X_umap = _X_orig
                    _umap_ds = DatasetBundle(
                        name=f"{working_dataset.name}_umap_rescue",
                        display_name=f"{working_dataset.display_name} (UMAP Rescue)",
                        X=_X_umap.astype(float),
                        y=working_dataset.y,
                        description=f"UMAP embedding of {working_dataset.display_name} for geodesic rescue.",
                        shape_family="manifold",
                        metadata={**working_dataset.metadata, "preprocessing": "umap_rescue"},
                    )
                    trace.append(
                        f"【Geodesic Pipeline】UMAP 嵌入完成 → {_X_umap.shape[1]}D,"
                        f" 派发 TopologyExpert (HDBSCAN + mutual k-NN Spectral)。"
                    )
                    topo = self.experts.get("topology")
                    if topo is not None:
                        _topo_results = topo.execute_with_self_correction(
                            _umap_ds, prompt, settings, constraints=constraints
                        )
                        if _topo_results:
                            _deep_results.extend(_topo_results)
                            trace.extend(topo.last_logs)
                            expert_logs["topology_rescue"] = list(topo.last_logs)
                            trace.append(
                                f"【Geodesic Pipeline】TopologyExpert 产出 {len(_topo_results)} 个结果。"
                            )
                except Exception as _umap_exc:
                    trace.append(f"【Geodesic Pipeline】UMAP 路径失败: {_umap_exc}")

                # Path B: Graph community discovery on wall-aware graph
                try:
                    _graph_expert = self.experts.get("graph")
                    if _graph_expert is not None:
                        _graph_results = _graph_expert.execute_with_self_correction(
                            working_dataset, prompt, settings, constraints=constraints
                        )
                        if _graph_results:
                            _deep_results.extend(_graph_results)
                            trace.extend(_graph_expert.last_logs)
                            expert_logs["graph_rescue"] = list(_graph_expert.last_logs)
                            trace.append(
                                f"【Geodesic Pipeline】GraphExpert 产出 {len(_graph_results)} 个结果。"
                            )
                except Exception as _graph_exc:
                    trace.append(f"【Geodesic Pipeline】Graph 路径失败: {_graph_exc}")

                # Path C: fallback to DimensionExpert if A and B both failed
                if not _deep_results:
                    trace.append("【Geodesic Pipeline】路径 A+B 均失败，回落 DimensionExpert。")
                    dim_expert = self.experts.get("dimension")
                    if dim_expert is not None:
                        try:
                            _dim_results = dim_expert.execute_with_self_correction(
                                working_dataset, prompt, settings, constraints=constraints
                            )
                            if _dim_results:
                                _deep_results.extend(_dim_results)
                                trace.extend(dim_expert.last_logs)
                                expert_logs["dimension_fallback"] = list(dim_expert.last_logs)
                        except Exception as _dim_exc:
                            trace.append(f"【Geodesic Pipeline】DimensionExpert 回落也失败: {_dim_exc}")

                if _deep_results:
                    all_results.extend(_deep_results)
                    ranking = self._compute_informed_ranking(
                        all_results, dataset, trace,
                        centroid_ban=_conn_check["centroid_ban"],
                    )
                    best = ranking[0]
                    _new_best_ari = self._compute_best_ari(ranking, dataset)
                    trace.append(
                        f"【Geodesic Pipeline 完成】产出 {len(_deep_results)} 个结果，"
                        f"重新排名后最佳 ARI={_new_best_ari or 'N/A'}"
                    )
                else:
                    trace.append("【Geodesic Pipeline】所有路径均未产出有效结果。")

        # 后验审计：Critic 独立审查最优结果
        _progress("正在进行 Critic 后验审计...")
        audit_report = self._execute_audit(best, dataset, settings, trace)

        # Phase 5.1: 当全局 ARI < 0.5 时，追加诚实失败评估
        if _global_low_ari and audit_report:
            audit_report["global_low_ari"] = True
            if audit_report.get("stability_score", 1.0) < 0.5:
                audit_report["honest_failure"] = True
                audit_report["endorsement"] = "qualified_with_warning"
                audit_report.setdefault("findings", []).append(
                    "Bootstrapping 稳定性 < 0.5 且全局 ARI < 0.5：算法正在'强行聚类'，"
                    "结果在重采样下极不稳定。"
                )
                if not audit_report.get("recommendation"):
                    audit_report["recommendation"] = (
                        "该数据集结构复杂（极窄薄流形/并行曲线），现有模型召回率低。"
                        "建议：1) 使用 OPTICS 可达性图分析；2) 引入 k-NN 局部连接谱聚类；"
                        "3) 考虑将数据投影到流形坐标后再聚类。"
                    )
                trace.append(
                    "【诚实失败】Bootstrapping 校验确认：聚类结果在重采样下极其不稳定，"
                    "算法正在强行聚类。已向用户如实报告数据结构复杂、模型召回率低。"
                )

        # Phase 5.3: inject FAILED verdict into audit_report when Deep Pipeline was triggered
        if _deep_pipeline_triggered:
            if audit_report is None:
                audit_report = {}
            audit_report["verdict"] = "FAILED"
            audit_report["structure_recognition_failed"] = True
            audit_report.setdefault("findings", []).insert(
                0,
                f"结构识别失败：所有独立算法 ARI < 0.7（最高={max(_all_independent_aris):.3f}），"
                f"欧氏空间方法无法捕捉数据内在结构。已自动触发 Deep Pipeline 流形嵌入。"
            )
            if not audit_report.get("recommendation"):
                audit_report["recommendation"] = (
                    "数据内在结构在欧氏空间中不可分。建议：1) 使用 UMAP/t-SNE 流形嵌入后聚类；"
                    "2) 采用基于图结构的社区发现方法；3) 考虑使用深度聚类（DEC/IDEC）"
                    "或自标签蒸馏（SelfLabel）。"
                )
            trace.append(
                "【FAILED 裁决】已注入审计报告：结构识别失败，"
                "建议采用流形嵌入或图结构方法。"
            )

        # Critic 2.0 反馈闭环：审计→约束重试→复验
        retry_results = self._handle_audit_feedback(
            audit_report, working_dataset, prompt, settings, trace, active_experts
        )
        if retry_results:
            all_results.extend(retry_results)
            ranking = self._compute_informed_ranking(all_results, dataset, trace,
                                                     centroid_ban=_conn_check["centroid_ban"])
            best = ranking[0]
            trace.append("【Critic 2.0】约束重试完成，已重新排名。")
            _progress("正在重新审计约束重试结果...")
            audit_report = self._execute_audit(best, dataset, settings, trace)

        # 集成共识：仅在 Critic 对单一最优结果有保留时触发
        # Phase 3.1: maze mode 始终触发 ensemble 以便 graph consensus
        #
        # Phase 5.3 Honest Retreat: 当所有独立专家 ARI < 0.4 时，禁止生成
        # 集成融合结果，直接向用户诚实报告失败。
        _honest_retreat = False
        if dataset.y is not None:
            import numpy as _hr_np
            from sklearn.metrics import adjusted_rand_score as _hr_ari
            _hr_y = _hr_np.asarray(dataset.y, dtype=int).ravel()
            _hr_max_ari = 0.0
            for _hr_r in all_results:
                if _hr_r.algorithm_name in ("EnsembleConsensus",) or _hr_r.algorithm_name.startswith("_"):
                    continue
                _hr_l = getattr(_hr_r, "labels", None)
                if _hr_l is not None and hasattr(_hr_l, "__len__") and len(_hr_l) > 0:
                    try:
                        _hr_a = float(_hr_ari(_hr_y, _hr_np.asarray(_hr_l, dtype=int).ravel()))
                        if _hr_a > _hr_max_ari:
                            _hr_max_ari = _hr_a
                    except Exception:
                        pass
            if _hr_max_ari < 0.4:
                _honest_retreat = True
                trace.append(
                    f"【集成诚实退避】所有独立专家 ARI < 0.4（max={_hr_max_ari:.3f}），"
                    f"系统拒绝生成集成融合结果。"
                    f"原因：数据内在结构极其复杂，现有模型全线失效，"
                    f"强行融合只会产生具有误导性的一致性报告。"
                    f"建议：1) 手动调整 k-NN 图参数；2) 引入领域知识约束；"
                    f"3) 考虑非欧氏距离度量。"
                )

        _should_ensemble = True
        if _honest_retreat:
            _should_ensemble = False
            trace.append("【集成诚实退避】已跳过 Ensemble 融合步骤。")
        elif maze_connectivity_mode:
            trace.append("【集成】迷宫连通域模式：跳过单一评审，强制触发集成共识。")
        elif audit_report is None:
            trace.append("【集成】无审计报告，保守触发集成共识。")
        elif (
            audit_report.get("endorsement") == "endorsed"
            and audit_report.get("confidence_level", 0.0) >= 0.75
        ):
            _should_ensemble = False
            trace.append(
                "【集成】Critic 已 endorsement 单一最优结果"
                f"（置信度 {audit_report.get('confidence_level', 0):.0%}），跳过集成共识。"
            )
        else:
            trace.append(
                f"【集成】Critic 裁决为 '{audit_report.get('endorsement')}'"
                f"（置信度 {audit_report.get('confidence_level', 0):.0%}），触发集成融合拯救。"
            )

        if _should_ensemble:
            # ---- Topology-aware weighting gate ---------------------------
            # Phase 3: uses structure for graph-aware diversity constraints
            _topology_mode = self._detect_manifold_topology(dataset, audit_report)
            _graph_mode = structure.get("recommended_strategy", {}).get("activate_graph_expert", False)
            _geodesic = structure.get("geodesic_distortion", 0.0)
            _diversity = _graph_mode or structure.get("structure_class") in ("graph_connected",)

            if _topology_mode:
                trace.append(
                    "【集成】检测到复杂流形/非凸拓扑特征，启用拓扑加权模式"
                    "（密度算法权重×3.0，距离算法权重×0.25）。"
                )
            if _diversity:
                trace.append(
                    "【集成】检测到图连通结构，启用图感知多样性约束"
                    "（图算法≥50%, 密度算法≥30%, 质心算法≤20%）。"
                )
            consensus_result = self._execute_ensemble(
                all_results, working_dataset, trace,
                topology_weighting=_topology_mode,
                diversity_constraints=_diversity,
                geodesic_distortion=_geodesic,
            )
            if consensus_result is not None:
                all_results.append(consensus_result)
                # Re-rank with consensus result included
                ranking = self._compute_informed_ranking(all_results, dataset, trace,
                                                         centroid_ban=_conn_check["centroid_ban"])
                best = ranking[0]
                # Phase 3.2: community native mode — ensemble is stability-only
                if community_native_mode:
                    consensus_result.metrics["role"] = "stability_analysis"
                    trace.append(
                        "【社区发现原生模式】集成共识仅用于稳定性分析，"
                        "不覆盖图社区发现分区结果。"
                    )

        # Phase 3.2: community native mode — prioritize graph community discovery
        if community_native_mode:
            # Find the graph community discovery result (primary output)
            _graph_community = None
            for r in all_results:
                if r.algorithm_name == "GraphCommunity_Result":
                    _graph_community = r
                    break
            if _graph_community is not None:
                _gc_score = _graph_community.metrics.get("modularity", 0.0)
                trace.append(
                    f"【社区发现原生模式】图社区发现为主输出"
                    f"（方法={_graph_community.metrics.get('best_method', '?')},"
                    f" 社区数={_graph_community.metrics.get('n_communities', '?')},"
                    f" modularity={_gc_score:.4f}）。"
                    f"聚类数由 modularity 最优自然决定，非预定义 k。"
                )
                # Re-rank via informed scoring (respects ARI priority when labels exist)
                ranking = self._compute_informed_ranking(all_results, dataset, trace,
                                                         centroid_ban=_conn_check["centroid_ban"])
                best = ranking[0]
                if best.algorithm_name == "GraphCommunity_Result":
                    trace.append(
                        "【社区发现原生模式】图社区发现结果已通过 Informed Ranking 确认为最优。"
                    )
                else:
                    trace.append(
                        f"【社区发现原生模式】Informed Ranking 选择了 '{best.algorithm_name}'"
                        f"（ARI/NMI 优先于 edge_cut score），覆盖图社区发现结果。"
                    )

        # Phase 5: Cross-validation check for graph algorithm winners
        self._cross_validate_graph_winner(best, dataset, all_results, trace)

        # Phase 3.1: topology failure detection (maze connectivity mode)
        topology_failure_report: dict[str, Any] | None = None
        if maze_connectivity_mode and best is not None:
            topology_failure_report = self._check_topology_failure(
                working_dataset, best, trace,
            )
            if topology_failure_report and topology_failure_report.get("topology_failure"):
                trace.append(
                    "【拓扑失败检测】聚类结果疑似拓扑失败！"
                    f"({topology_failure_report.get('n_failures', 0)}/5 个红旗)"
                    f" — {topology_failure_report.get('recommendation', '')}"
                )
                # Merge topology failure into audit report if audit exists
                if audit_report:
                    audit_report["topology_failure"] = True
                    audit_report["topology_failure_details"] = topology_failure_report
                    if audit_report.get("endorsement") == "endorsed":
                        audit_report["endorsement"] = "qualified_with_warning"
                        trace.append(
                            "【拓扑失败检测】审计裁决从 'endorsed' 降级为"
                            " 'qualified_with_warning'。"
                        )
                else:
                    audit_report = {
                        "topology_failure": True,
                        "topology_failure_details": topology_failure_report,
                        "endorsement": "qualified_with_warning",
                        "confidence_level": 0.3,
                        "recommendation": topology_failure_report.get("recommendation", ""),
                    }
                    trace.append(
                        "【拓扑失败检测】无现有审计报告，生成拓扑失败审计。"
                    )
            elif topology_failure_report:
                trace.append(
                    "【拓扑失败检测】通过 —"
                    f" (CPS={topology_failure_report.get('cps', 0):.3f},"
                    f" conductance={topology_failure_report.get('conductance', 0):.3f},"
                    f" modularity={topology_failure_report.get('modularity', 0):.3f})"
                )

        client = UniversalLLMClient(settings)
        summary = client.summarize_report(
            {
                "user_intent": prompt,
                "dataset": dataset.display_name,
                "best_algo": best.algorithm_name,
                "metrics": best.metrics,
                "score_source": best.metrics.get("score_source", "silhouette"),
                "all_results": [
                    {
                        "algo": r.algorithm_name,
                        "score": r.metrics.get("score", 0.0),
                        "score_source": r.metrics.get("score_source", "silhouette"),
                    }
                    for r in all_results
                ],
            }
        )

        report = SupervisorReport(
            dataset=dataset,
            routing=RoutingDecision(
                profile=ProfileReport(dataset.X.shape[0], dataset.X.shape[1], 0, 0, 0, False, False, False),
                selected_experts=[],
                trace=trace,
            ),
            dataset_plot_path=self._save_raw_plot(dataset, output_dir),
            output_dir=output_dir,
            results=all_results,
            ranking=ranking,
            executive_summary=summary or "聚类分析完成。",
            decision_trace=trace,
            audit_report=audit_report,
            response_type="CLUSTER_TASK",
        )

        # 自动生成 LaTeX（静默失败，不阻塞 UI；CODE_EXAMPLE 类型会在生成器内跳过）
        with contextlib.suppress(Exception):
            report.latex_path = LatexReportGenerator().generate(report)

        self.last_report = report
        self.memory.append({"role": "user", "content": prompt})
        self.memory.append({"role": "assistant", "content": report.executive_summary})
        return report

    # ------------------------------------------------------------------
    # 后验审计
    # ------------------------------------------------------------------

    def _execute_audit(
        self,
        winner: AlgorithmRunResult,
        dataset: DatasetBundle,
        settings: LLMSettings,
        trace: list[str],
    ) -> dict[str, Any] | None:
        """Run CriticExpert as a post-hoc auditor on the winning result.

        Phase 5.3: sets a shorter sandbox timeout (50% of main clustering
        time) for the audit.  On timeout, degrades to a basic audit
        recommendation rather than failing entirely.
        """
        critic = self.experts.get("critic")
        if critic is None:
            trace.append("【审计】Critic 专家未注册，跳过审计。")
            return None

        fast_audit = bool(getattr(settings, "fast_audit", False))
        mode_label = "快速审计" if fast_audit else "完整审计"
        trace.append(
            f"【审计】对最优结果 '{winner.algorithm_name}' 启动 {mode_label}..."
        )

        # ---- Phase 5.3: shorter timeout for audit sandbox -----------------
        import os
        _prev_timeout = os.environ.get("ACE_SANDBOX_TIMEOUT_SEC")
        _audit_timeout = "45"  # default: 45 seconds for audit
        if _prev_timeout is not None:
            _audit_timeout = str(max(20, int(_prev_timeout) // 2))
        os.environ["ACE_SANDBOX_TIMEOUT_SEC"] = _audit_timeout

        try:
            audit = critic.execute_audit(winner, dataset, settings)
            if audit:
                endorsement = audit.get("endorsement", "?")
                confidence = audit.get("confidence_level", "?")
                trace.append(
                    f"【审计】完成 — 裁决: {endorsement}, 置信度: {confidence}"
                )
            else:
                # ---- Degraded audit fallback --------------------------------
                critic_diag = "; ".join(critic.last_logs[-2:]) if critic.last_logs else "（无诊断信息）"
                trace.append(
                    f"【审计】审计未产出有效报告（可能超时），降级为初级审计建议。"
                    f"{critic_diag}"
                )
                # Synthesize a minimal degraded audit
                audit = {
                    "endorsement": "qualified",
                    "action": "WARN",
                    "confidence_level": 0.3,
                    "findings": ["审计超时，无法完成全量分析。建议人工检查聚类质量。"],
                    "recommendation": "审计沙箱超时（timeout=" + _audit_timeout + "s），请考虑启用 fast_audit 模式或减小数据集。",
                    "degraded": True,
                }
            return audit
        except Exception as exc:
            trace.append(f"【审计】Critic 审计过程异常: {exc}")
            return {
                "endorsement": "qualified",
                "action": "WARN",
                "confidence_level": 0.2,
                "findings": [f"审计异常: {exc}"],
                "recommendation": "审计无法完成，建议人工判断。",
                "degraded": True,
            }
        finally:
            # Restore previous timeout
            if _prev_timeout is not None:
                os.environ["ACE_SANDBOX_TIMEOUT_SEC"] = _prev_timeout
            elif "ACE_SANDBOX_TIMEOUT_SEC" in os.environ:
                del os.environ["ACE_SANDBOX_TIMEOUT_SEC"]

    # ------------------------------------------------------------------
    # Critic 2.0 反馈闭环
    # ------------------------------------------------------------------

    def _handle_audit_feedback(
        self,
        audit_report: dict | None,
        dataset: DatasetBundle,
        prompt: str,
        settings: LLMSettings,
        trace: list[str],
        active_experts: list[str],
    ) -> list[AlgorithmRunResult]:
        """Critic 2.0 closed-loop: RETRY with constraints, max 2 attempts.

        When the audit finds the best result untrustworthy (action=RETRY),
        re-dispatch the expert pool with constraint instructions derived
        from the audit findings.
        """
        if audit_report is None:
            return []

        action = audit_report.get("action", "CLEAR")
        if action == "CLEAR":
            return []
        if action == "WARN":
            trace.append("【Critic 2.0】审计裁决为 WARN，接受当前结果（不重试）。")
            return []

        # action == "RETRY"
        constraints = audit_report.get("retry_constraints", {})
        if not constraints:
            trace.append("【Critic 2.0】RETRY 但无有效约束，跳过重试。")
            return []

        trace.append(
            f"【Critic 2.0】审计裁决为 RETRY，启动约束重试..."
            f" force_k={constraints.get('force_k')},"
            f" blocked={constraints.get('blocked_algorithms')}"
        )

        for attempt in range(1, 3):  # max_retries=2
            trace.append(f"【Critic 2.0】第 {attempt}/2 次约束重试...")
            retry_results: list[AlgorithmRunResult] = []

            for key in active_experts:
                expert = self.experts.get(key)
                if expert is None:
                    continue
                try:
                    if attempt > 1:
                        import os
                        os.environ.setdefault("ACE_SANDBOX_TIMEOUT_SEC", "120")
                    results = expert.execute_with_self_correction(
                        dataset, prompt, settings, constraints=constraints,
                    )
                    retry_results.extend(results)
                    trace.extend(expert.last_logs)
                except Exception as exc:
                    trace.append(
                        f"【Critic 2.0】专家 '{key}' 约束重试异常: {exc}"
                    )

            if retry_results:
                trace.append(
                    f"【Critic 2.0】第 {attempt} 次重试产出 {len(retry_results)} 个结果。"
                )
                return retry_results

            trace.append(f"【Critic 2.0】第 {attempt} 次重试未产出有效结果。")

        return []

    # ------------------------------------------------------------------
    # 集成共识
    # ------------------------------------------------------------------

    def _execute_ensemble(
        self,
        all_results: list[AlgorithmRunResult],
        dataset: DatasetBundle,
        trace: list[str],
        topology_weighting: bool = False,
        diversity_constraints: bool = False,
        geodesic_distortion: float = 0.0,
    ) -> AlgorithmRunResult | None:
        """Run EnsembleConsensusExpert to fuse all expert labels.

        Builds a co-association matrix from all valid result labels and
        produces consensus labels via hierarchical clustering.

        When *topology_weighting* is True, topology-friendly algorithms
        (HDBSCAN, DBSCAN, Spectral) receive a 3× vote boost and centroid
        algorithms (KMeans, GMM) are penalised to 0.25×.

        When *diversity_constraints* is True, enforces graph ≥50%,
        density ≥30%, centroid ≤20% weight proportions.

        Returns ``None`` if the ensemble expert is unavailable or fewer
        than 2 valid label sets exist.
        """
        ensemble = self.experts.get("ensemble")
        if ensemble is None:
            trace.append("【集成】Ensemble 专家未注册，跳过共识融合。")
            return None

        valid_count = sum(
            1 for r in all_results
            if getattr(r, "labels", None) is not None
            and len(getattr(r, "labels", [])) > 0
        )
        if valid_count < 2:
            trace.append(f"【集成】有效标签集不足 ({valid_count} < 2)，跳过共识融合。")
            return None

        mode_parts = []
        if topology_weighting:
            mode_parts.append("拓扑加权")
        if diversity_constraints:
            mode_parts.append("多样性约束")
        mode_label = f" {'+'.join(mode_parts)}" if mode_parts else ""
        trace.append(f"【集成】对 {valid_count} 套专家标签启动 Co-association 共识融合{mode_label}...")
        try:
            result = ensemble.execute_ensemble(
                all_results, dataset,
                topology_weighting=topology_weighting,
                diversity_constraints=diversity_constraints,
                geodesic_distortion=geodesic_distortion,
            )
            if result is not None:
                trace.append(
                    f"【集成】完成 — 融合 {result.metrics.get('n_experts_fused', '?')} 位专家, "
                    f"一致性={result.metrics.get('agreement', 0):.3f}"
                )
            else:
                trace.append("【集成】共识融合未产出有效结果。")
            return result
        except Exception as exc:
            trace.append(f"【集成】共识融合异常: {exc}")
            return None

    # ------------------------------------------------------------------
    # Data structure classification (Phase 3 Topology-Aware)
    # ------------------------------------------------------------------

    @staticmethod
    def _classify_data_structure(
        dataset: DatasetBundle,
    ) -> dict[str, Any]:
        """Classify data into: spherical / non_convex / manifold /
        graph_connected / density / hierarchical.

        Returns a dict with:
        - structure_class: str
        - geodesic_distortion: float
        - recommended_strategy: dict
          - activate_graph_expert: bool
          - topology_boost: bool
          - centroid_suppress: bool
        """
        sf = getattr(dataset, "shape_family", "generic")
        n_features = dataset.X.shape[1] if hasattr(dataset.X, "shape") else 0
        n_samples = dataset.X.shape[0]

        # Default: use shape_family hint
        structure_class = sf

        # Compute geodesic distortion for 2D/3D data (fast)
        geodesic_distortion = 0.0
        wall_crossings = 0
        if 1 <= n_features <= 5 and n_samples >= 50:
            try:
                import numpy as np
                X_np = np.asarray(dataset.X, dtype=float)
                adj = GraphBuilder.build_knn_graph(X_np)
                if n_samples <= 2000:
                    geo_dists = GraphBuilder.compute_geodesic_distances(adj)
                    geodesic_distortion = GraphBuilder.compute_distortion(X_np, geo_dists, sample_size=min(n_samples, 500))
                else:
                    # Sample anchors for large N
                    rng = np.random.RandomState(42)
                    n_anchors = min(500, n_samples)
                    anchors = rng.choice(n_samples, n_anchors, replace=False)
                    geo_dists = GraphBuilder.compute_geodesic_distances(adj, indices=anchors)
                    geodesic_distortion = GraphBuilder.compute_distortion(X_np, geo_dists, sample_size=min(n_anchors, 200))
            except Exception:
                pass

        # Detect wall-crossings when distortion is high
        if geodesic_distortion > 0.3 and n_samples <= 5000:
            try:
                import numpy as np
                X_np = np.asarray(dataset.X, dtype=float)
                adj = GraphBuilder.build_knn_graph(X_np)
                if n_samples <= 2000:
                    geo_dists = GraphBuilder.compute_geodesic_distances(adj)
                else:
                    rng = np.random.RandomState(42)
                    n_anchors = min(500, n_samples)
                    anchors = rng.choice(n_samples, n_anchors, replace=False)
                    geo_dists = GraphBuilder.compute_geodesic_distances(adj, indices=anchors)
                pairs = GraphBuilder.detect_wall_crossings(X_np, adj, geo_dists)
                wall_crossings = len(pairs)
            except Exception:
                pass

        # ---- Classification rules ----
        if geodesic_distortion > 0.5:
            structure_class = "graph_connected"
        elif sf in ("non_convex", "manifold") and geodesic_distortion > 0.3:
            structure_class = "graph_connected"
        elif sf == "manifold" and geodesic_distortion > 0.15:
            structure_class = "graph_connected"
        elif sf in ("non_convex",) and geodesic_distortion <= 0.1:
            structure_class = "non_convex"

        # ---- Strategy ----
        activate_graph_expert = structure_class in ("graph_connected",) or geodesic_distortion > 0.5
        topology_boost = structure_class in ("graph_connected", "non_convex", "manifold")
        centroid_suppress = structure_class in ("graph_connected",) and geodesic_distortion > 0.3

        return {
            "structure_class": structure_class,
            "geodesic_distortion": geodesic_distortion,
            "wall_crossings": wall_crossings,
            "recommended_strategy": {
                "activate_graph_expert": activate_graph_expert,
                "topology_boost": topology_boost,
                "centroid_suppress": centroid_suppress,
            },
        }

    # ------------------------------------------------------------------
    # Phase 5.2: Geometric connectivity pre-check
    # ------------------------------------------------------------------

    @staticmethod
    def _connectivity_pre_check(
        dataset: DatasetBundle,
        trace: list[str],
    ) -> dict[str, Any]:
        """Run a lightweight k-NN connectivity check BEFORE expert dispatch.

        Detects long-range curve / manifold structure where centroid
        algorithms (KMeans, GMM) produce physically meaningless spherical
        partitions.  When such structure is found, returns a centroid ban
        set that prevents those algorithms from taking the top ranking spot.

        Returns a dict with:
        - long_range_curve: bool
        - centroid_ban: set[str] (empty if no ban)
        - geodesic_distortion: float
        """
        _CENTROID_ALGOS = {"KMeans", "MiniBatchKMeans", "GaussianMixture", "GMM", "Birch"}
        n_samples = dataset.X.shape[0]
        n_features = dataset.X.shape[1] if hasattr(dataset.X, "shape") else 0

        if n_samples < 50 or n_features > 10:
            return {"long_range_curve": False, "centroid_ban": set(), "geodesic_distortion": 0.0}

        try:
            import numpy as np
            X_np = np.asarray(dataset.X, dtype=float)
            adj = GraphBuilder.build_knn_graph(X_np, k=min(10, max(3, int(np.sqrt(n_samples)) // 4)))

            if n_samples <= 2000:
                geo_dists = GraphBuilder.compute_geodesic_distances(adj)
                distortion = GraphBuilder.compute_distortion(X_np, geo_dists, sample_size=min(n_samples, 500))
            else:
                rng = np.random.RandomState(42)
                n_anchors = min(500, n_samples)
                anchors = rng.choice(n_samples, n_anchors, replace=False)
                geo_dists = GraphBuilder.compute_geodesic_distances(adj, indices=anchors)
                distortion = GraphBuilder.compute_distortion(X_np, geo_dists, sample_size=min(n_anchors, 200))
        except Exception:
            return {"long_range_curve": False, "centroid_ban": set(), "geodesic_distortion": 0.0}

        long_range_curve = distortion > 0.35

        if long_range_curve:
            trace.append(
                f"【连通性预检】geodesic_distortion={distortion:.3f} > 0.35，"
                f"判定为长程曲线/流形结构，质心算法（KMeans/GMM/Birch）禁止优胜。"
            )
        else:
            trace.append(
                f"【连通性预检】geodesic_distortion={distortion:.3f}，未触发质心禁令。"
            )

        return {
            "long_range_curve": long_range_curve,
            "centroid_ban": _CENTROID_ALGOS if long_range_curve else set(),
            "geodesic_distortion": distortion,
        }

    # ------------------------------------------------------------------
    # Compute best ARI across a ranking (for dashboard warning)
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_best_ari(
        ranking: list[AlgorithmRunResult],
        dataset: DatasetBundle,
    ) -> float | None:
        """Return the maximum ARI among the given results, or None if labels
        are unavailable."""
        y_true = dataset.y
        if y_true is None:
            return None
        try:
            from sklearn.metrics import adjusted_rand_score
            y_true_arr = np.asarray(y_true, dtype=int).ravel()
            best = 0.0
            for r in ranking:
                labels = getattr(r, "labels", None)
                if labels is None or not hasattr(labels, "__len__") or len(labels) == 0:
                    continue
                labels_arr = np.asarray(labels, dtype=int).ravel()
                ari = float(adjusted_rand_score(y_true_arr, labels_arr))
                if ari > best:
                    best = ari
            return best
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Phase 3.1: Topology failure detection for maze connectivity mode
    # ------------------------------------------------------------------

    @staticmethod
    def _check_topology_failure(
        dataset: DatasetBundle,
        result: AlgorithmRunResult,
        trace: list[str],
    ) -> dict[str, Any] | None:
        """Run comprehensive topology failure check on the winning result.

        Only invoked when maze_connectivity_mode is active (graph_connected
        data).  Uses GraphBuilder.topology_failure_check to detect:
        axis-aligned partitions, high conductance, wall-crossings,
        low CPS, and low modularity.

        Returns the failure report dict, or None if the check cannot run.
        """
        labels = getattr(result, "labels", None)
        if labels is None or len(labels) == 0:
            trace.append("【拓扑检测】跳过：最优结果无有效标签。")
            return None

        try:
            import numpy as np
            X_np = np.asarray(dataset.X, dtype=float)
            n_samples = X_np.shape[0]

            adj = GraphBuilder.build_knn_graph(X_np)
            geo_dists = None
            if n_samples <= 2000:
                geo_dists = GraphBuilder.compute_geodesic_distances(adj)

            labels_arr = np.asarray(labels, dtype=int).ravel()
            report = GraphBuilder.topology_failure_check(
                X_np, adj, labels_arr, geo_dists,
            )
            return report
        except Exception as exc:
            trace.append(f"【拓扑检测】失败: {exc}")
            return None

    # ------------------------------------------------------------------
    # Topology / Manifold detection & preprocessing (Phase 2.4)
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_manifold_topology(
        dataset: DatasetBundle,
        audit_report: dict[str, Any] | None = None,
    ) -> bool:
        """Return True if the data likely has complex manifold / non-convex topology.

        Detection heuristics (any one triggers the flag):
        1. Dataset metadata explicitly marks shape_family as manifold or non_convex.
        2. Critic audit reports Hopkins > 0.6 and overfitting_risk != 'low'.
        3. Low-dimensional data (≤ 5 features) — 2D/3D datasets are often
           the exact topology benchmarks this mechanism targets.
        """
        # Heuristic 1: explicit metadata
        sf = getattr(dataset, "shape_family", None)
        if sf in ("manifold", "non_convex"):
            return True

        # Heuristic 2: Critic audit signals
        if audit_report and isinstance(audit_report, dict):
            hopkins = audit_report.get("hopkins", 0.0)
            overfit = audit_report.get("overfitting_risk", "low")
            dbcv = audit_report.get("dbcv_score", None)
            if isinstance(hopkins, (int, float)) and hopkins > 0.6 and overfit != "low":
                return True
            # DBCV < 0 signals that density separation is poor —
            # strongly indicative of manifold topology that centroid
            # algorithms cannot handle.
            if isinstance(dbcv, (int, float)) and dbcv < 0.0:
                return True

        # Heuristic 3: low-dimensional data
        n_features = getattr(dataset, "X", None)
        if n_features is not None:
            d = n_features.shape[1] if hasattr(n_features, "shape") else 0
            if 1 <= d <= 5:
                return True

        return False

    def _apply_manifold_preprocessing(
        self,
        dataset: DatasetBundle,
        trace: list[str],
    ) -> DatasetBundle | None:
        """Reduce high-D manifold data to a 2D/3D embedding via UMAP or
        SpectralEmbedding before clustering experts are dispatched.

        Returns a new DatasetBundle with the transformed feature matrix,
        or ``None`` if preprocessing could not be applied.
        """
        try:
            import numpy as np

            X = np.asarray(dataset.X, dtype=float)
            n_samples, n_features = X.shape

            # For already-2D data, skip preprocessing — it's already visualisable
            if n_features <= 2:
                return dataset

            target_dim = min(3, n_features - 1, 16)
            trace.append(
                f"【流形预处理】{n_features}D → {target_dim}D manifold embedding..."
            )

            # Prefer UMAP for topology preservation
            try:
                import umap  # type: ignore[import-untyped]

                reducer = umap.UMAP(
                    n_components=target_dim,
                    n_neighbors=min(30, max(5, n_samples // 50)),
                    min_dist=0.1,
                    metric="euclidean",
                    random_state=42,
                )
                X_embedded = reducer.fit_transform(X)
                method = "UMAP"
            except ImportError:
                # Fallback: SpectralEmbedding
                from sklearn.manifold import SpectralEmbedding

                trace.append("【流形预处理】UMAP 不可用，回落至 SpectralEmbedding。")
                emb = SpectralEmbedding(
                    n_components=target_dim,
                    affinity="nearest_neighbors",
                    random_state=42,
                )
                X_embedded = emb.fit_transform(X)
                method = "SpectralEmbedding"

            trace.append(
                f"【流形预处理】{method} 完成 → {X_embedded.shape[1]}D 嵌入。"
            )

            return DatasetBundle(
                name=f"{dataset.name}_manifold",
                display_name=f"{dataset.display_name} (流形嵌入)",
                X=X_embedded.astype(float),
                y=dataset.y,
                description=f"{dataset.description} 经 {method} {target_dim}D 流形嵌入预处理。",
                shape_family="manifold",
                feature_names=[f"emb_{i}" for i in range(X_embedded.shape[1])],
                metadata={
                    **dataset.metadata,
                    "preprocessing": method,
                    "original_dim": n_features,
                },
            )
        except Exception as exc:
            trace.append(f"【流形预处理】失败: {exc}")
            return None

    # ------------------------------------------------------------------
    # FOLLOW_UP 路径
    # ------------------------------------------------------------------

    def _handle_follow_up(self, prompt: str, settings: LLMSettings, trace: list[str]) -> SupervisorReport:
        """纯 LLM 驱动的追问或学术咨询处理。"""
        client = UniversalLLMClient(settings)

        if self.last_report:
            context = {
                "last_summary": self.last_report.executive_summary,
                "ranking": [
                    {"algo": r.algorithm_name, "score": r.metrics.get("score")} for r in self.last_report.ranking
                ],
            }
            system_msg = f"你是一个数据科学专家。请基于以下聚类背景及检索到的知识回答用户问题：\n{context}"
        else:
            system_msg = "你是一个数据科学专家。请基于检索到的学术背景知识回答用户的理论咨询。"

        res = client.chat_completion([{"role": "system", "content": system_msg}, {"role": "user", "content": prompt}])
        trace.append("【主控】正在基于知识库与会话上下文进行深度解析...")

        report = SupervisorReport(
            dataset=(
                self.last_report.dataset
                if self.last_report
                else DatasetBundle("Consultation", np.array([[0, 0]]), None)
            ),
            routing=(self.last_report.routing if self.last_report else RoutingDecision(None, [], trace)),
            dataset_plot_path=(self.last_report.dataset_plot_path if self.last_report else Path("")),
            output_dir=self.last_report.output_dir if self.last_report else Path(""),
            results=[],
            ranking=self.last_report.ranking if self.last_report else [],
            executive_summary=res or "无法生成回答。",
            decision_trace=trace,
            response_type="FOLLOW_UP",
        )
        return report

    # ------------------------------------------------------------------
    # CODE_EXAMPLE 路径（P0.5-C 新增）
    # ------------------------------------------------------------------

    def _handle_code_example(self, prompt: str, settings: LLMSettings, trace: list[str]) -> SupervisorReport:
        """处理 CODE_EXAMPLE 意图：用 LLM 生成自包含代码，不走沙箱，不生成图。

        返回 SupervisorReport(response_type="CODE_EXAMPLE")，
        executive_summary 为 Markdown 代码块字符串。
        """
        trace.append("【主控】识别为 CODE_EXAMPLE 意图，生成代码示例（不执行实验）。")
        client = UniversalLLMClient(settings)

        code_system = (
            "你是一个 Python 数据科学专家。用户要求你提供一段**完整可运行的** Python 代码示例。\n"
            "要求：\n"
            "1. 代码必须自包含（包含所有 import）。\n"
            "2. 数据使用 sklearn.datasets 内置数据集或 make_* 函数构造，不读取外部文件。\n"
            "3. 必须包含：算法调用、结果可视化（matplotlib）、指标计算（轮廓系数等）。\n"
            "4. 用中文注释解释关键步骤。\n"
            "5. 只返回 Markdown 代码块，格式为 ```python ... ```，不要额外解释。"
        )
        raw = client.chat_completion([{"role": "user", "content": prompt}], code_system)
        # 确保结果包裹在 markdown 代码块里
        if raw and "```" not in raw:
            code_md = f"```python\n{raw.strip()}\n```"
        else:
            code_md = raw or "```python\n# 无法生成代码示例，请检查 LLM 配置。\n```"

        trace.append("【主控】代码示例生成完毕。")

        # 复用上次报告的 dataset/routing/output_dir 字段，保持 UI 不崩溃
        placeholder_ds = (
            self.last_report.dataset if self.last_report else DatasetBundle("code_example", np.array([[0, 0]]), None)
        )
        report = SupervisorReport(
            dataset=placeholder_ds,
            routing=(self.last_report.routing if self.last_report else RoutingDecision(None, [], trace)),
            dataset_plot_path=(self.last_report.dataset_plot_path if self.last_report else Path("")),
            output_dir=self.last_report.output_dir if self.last_report else Path(""),
            results=[],
            ranking=self.last_report.ranking if self.last_report else [],
            executive_summary=code_md,
            decision_trace=trace,
            response_type="CODE_EXAMPLE",
        )
        # 不更新 self.last_report（不覆盖上次真实分析结果）
        return report

    # ------------------------------------------------------------------
    # 辅助方法
    # ------------------------------------------------------------------

    def _prepare_output_dir(self, name: str) -> Path:
        path = Path(f"outputs/{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _save_raw_plot(self, dataset: DatasetBundle, out_dir: Path) -> Path:
        import matplotlib.pyplot as plt

        path = out_dir / "raw_data.png"
        plt.figure(figsize=(6, 4))
        plt.scatter(dataset.X[:, 0], dataset.X[:, 1], c="gray", alpha=0.5, s=10)
        plt.title(f"Dataset: {dataset.display_name}")
        plt.savefig(path)
        plt.close()
        return path

    # ------------------------------------------------------------------
    # Phase 5.1: Informed ranking — ARI one-vote veto when labels exist
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_informed_ranking(
        all_results: list[AlgorithmRunResult],
        dataset: DatasetBundle,
        trace: list[str],
        centroid_ban: set[str] | None = None,
    ) -> list[AlgorithmRunResult]:
        """Rank results.  When ground-truth labels exist, **ARI is the sole
        ranking criterion** — internal metrics (Silhouette, Edge Cut,
        modularity) are excluded from the scoring formula.

        Additionally detects the **consensus trap**: when EnsembleConsensus
        has high self-reported agreement (≥ 0.7) but its ARI is materially
        lower than the best individual expert, the ensemble result is
        flagged as overfitting / consensus-bias and the highest-ARI
        individual algorithm takes the top spot.

        When *centroid_ban* is non-empty, KMeans/GMM/MiniBatchKMeans/Birch
        results have their ARI zeroed so they cannot take the top spot.
        This is the connectivity pre-check veto: on long-range curve data,
        centroid algorithms produce physically meaningless partitions.
        """
        y_true = dataset.y
        has_labels = y_true is not None

        if not has_labels:
            return sorted(
                all_results,
                key=lambda r: r.metrics.get("score", 0.0),
                reverse=True,
            )

        from sklearn.metrics import adjusted_rand_score

        y_true_arr = np.asarray(y_true, dtype=int).ravel()

        # ---- compute ARI for every result -----------------------------------
        IS_ENSEMBLE_NAME = "EnsembleConsensus"
        best_individual_ari = -1.0
        best_individual_name = ""
        ensemble_entry: tuple[float, float, AlgorithmRunResult] | None = None

        entries: list[tuple[float, float, float, AlgorithmRunResult]] = []
        # Each entry: (ari, nmi, internal_raw, result)

        for r in all_results:
            labels = getattr(r, "labels", None)
            if labels is None or not hasattr(labels, "__len__") or len(labels) == 0:
                entries.append((-1.0, 0.0, 0.0, r))
                continue

            try:
                labels_arr = np.asarray(labels, dtype=int).ravel()
                ari = float(adjusted_rand_score(y_true_arr, labels_arr))
            except Exception:
                ari = 0.0

            # ---- connectivity pre-check centroid veto --------------------
            if centroid_ban and r.algorithm_name in centroid_ban:
                ari = 0.0

            nmi = float(r.metrics.get("nmi", 0.0))
            internal_raw = float(r.metrics.get("score", 0.0))
            entries.append((ari, nmi, internal_raw, r))

            # Track best individual (non-ensemble, non-internal-meta)
            is_individual = (
                r.algorithm_name != IS_ENSEMBLE_NAME
                and not r.algorithm_name.startswith("_")
            )
            if is_individual and ari > best_individual_ari:
                best_individual_ari = ari
                best_individual_name = r.algorithm_name

            if r.algorithm_name == IS_ENSEMBLE_NAME:
                ensemble_entry = (ari, internal_raw, r)

        # ---- consensus trap detection ---------------------------------------
        if ensemble_entry is not None and best_individual_ari > 0:
            ens_ari, ens_agreement, ens_r = ensemble_entry
            ari_gap = best_individual_ari - ens_ari
            if ari_gap > 0.03 and ens_agreement >= 0.7:
                trace.append(
                    f"【一致性陷阱】检测到 Consensus Trap！"
                    f"EnsembleConsensus 一致性={ens_agreement:.3f} 但 ARI={ens_ari:.3f}，"
                    f"独立专家 '{best_individual_name}' 的 ARI={best_individual_ari:.3f}"
                    f"（差距={ari_gap:.3f}）。"
                    f"判定集成结果为过拟合/共识偏差，降级处理。"
                )
                # Demote ensemble: set its ARI to a value just below best_individual
                # so it falls in ranking but stays available for reference
                ens_r.metrics["consensus_trap"] = True
                ens_r.metrics["consensus_trap_gap"] = round(ari_gap, 4)
                ens_r.metrics["outperformed_by"] = best_individual_name

        # ---- ARI-only sort (internal metrics excluded from scoring) ---------
        # Tiebreak: higher internal score first
        entries.sort(key=lambda x: (x[0], x[2]), reverse=True)

        best = entries[0]
        best_ari = best[0]
        best_name = best[3].algorithm_name
        is_ensemble_winner = best_name == IS_ENSEMBLE_NAME

        if is_ensemble_winner and best_ari >= best_individual_ari - 0.01:
            trace.append(
                f"【优选排名】ARI 一票否决制: EnsembleConsensus ARI={best_ari:.3f}"
                f" ≥ 最佳独立专家 ARI={best_individual_ari:.3f}，集成结果可信。"
            )
        elif is_ensemble_winner:
            trace.append(
                f"【优选排名】ARI 一票否决制: EnsembleConsensus ARI={best_ari:.3f}"
                f" 微弱领先，但独立专家 '{best_individual_name}' ARI={best_individual_ari:.3f}"
                f" 更优。请关注一致性陷阱风险。"
            )
        else:
            trace.append(
                f"【优选排名】ARI 一票否决制: best='{best_name}'"
                f" ARI={best_ari:.3f}"
                f" (internal={best[3].metrics.get('score', 0):.3f})"
            )
            if ensemble_entry is not None:
                ens_ari_v = ensemble_entry[0]
                if best_ari > ens_ari_v + 0.03:
                    trace.append(
                        f"【优选排名】EnsembleConsensus ARI={ens_ari_v:.3f} 被"
                        f" '{best_name}' (ARI={best_ari:.3f}) 一票否决。"
                        f"内部指标（一致性={ensemble_entry[1]:.3f}）不参与 ARI 判定。"
                    )

        return [e[3] for e in entries]

    # ------------------------------------------------------------------
    # Phase 5: Cross-validation check for graph algorithm winners
    # ------------------------------------------------------------------

    @staticmethod
    def _cross_validate_graph_winner(
        best: AlgorithmRunResult,
        dataset: DatasetBundle,
        all_results: list[AlgorithmRunResult],
        trace: list[str],
    ) -> None:
        """When a graph algorithm wins, auto-compare against DBSCAN/KMeans.

        If agreement is very low and ARI differs significantly, emit a
        "Metric Artifact Warning" suggesting the internal metric may be
        misleading.
        """
        _GRAPH_KEYS = {"graph", "GraphCommunity_Result"}
        is_graph_winner = (
            best.algorithm_name == "GraphCommunity_Result"
            or best.expert_key in _GRAPH_KEYS
        )
        if not is_graph_winner:
            return

        y_true = dataset.y
        if y_true is None:
            return

        from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

        best_labels = getattr(best, "labels", None)
        if best_labels is None or not hasattr(best_labels, "__len__") or len(best_labels) == 0:
            return

        y_true_arr = np.asarray(y_true, dtype=int).ravel()
        best_labels_arr = np.asarray(best_labels, dtype=int).ravel()

        try:
            best_ari = float(adjusted_rand_score(y_true_arr, best_labels_arr))
            best_nmi = float(normalized_mutual_info_score(y_true_arr, best_labels_arr))
        except Exception:
            return

        # Find DBSCAN/KMeans results for comparison
        ref_algos = {"KMeans", "DBSCAN", "HDBSCAN", "MiniBatchKMeans", "GMM", "GaussianMixture"}
        best_ref_ari = -1.0
        best_ref_name = ""
        for r in all_results:
            if r.algorithm_name not in ref_algos:
                continue
            r_labels = getattr(r, "labels", None)
            if r_labels is None or not hasattr(r_labels, "__len__") or len(r_labels) == 0:
                continue
            try:
                r_labels_arr = np.asarray(r_labels, dtype=int).ravel()
                r_ari = float(adjusted_rand_score(y_true_arr, r_labels_arr))
                if r_ari > best_ref_ari:
                    best_ref_ari = r_ari
                    best_ref_name = r.algorithm_name
            except Exception:
                continue

        if best_ref_ari < 0:
            return

        ari_gap = best_ref_ari - best_ari
        if ari_gap > 0.15:
            trace.append(
                f"【指标伪影警告】图算法 '{best.algorithm_name}' 的 ARI={best_ari:.3f}"
                f" 显著低于 '{best_ref_name}' 的 ARI={best_ref_ari:.3f}（差距={ari_gap:.3f}）。"
                f"内部指标（modularity/edge_cut）可能高估了图算法的真实聚类质量。"
                f"建议：检查 geodesic_distortion，考虑使用非图算法或混合集成。"
            )

    # ------------------------------------------------------------------
    # Error / fallback report
    # ------------------------------------------------------------------

    def _error_report(
        self,
        msg: str,
        trace: list[str],
        expert_logs: dict[str, list[str]] | None = None,
    ) -> SupervisorReport:
        """构造错误报告。

        P0.5-D：当 expert_logs 非空时，汇总每个专家最后 3 行日志
        作为排错依据，而不是只输出一句通用错误。
        """
        debug_lines: list[str] = [msg]
        if expert_logs:
            debug_lines.append("\n排错信息（各专家最后日志）：")
            for key, logs in expert_logs.items():
                last_3 = logs[-3:] if logs else ["（无日志）"]
                debug_lines.append(f"  [{key}] " + " | ".join(last_3))
        full_msg = "\n".join(debug_lines)

        return SupervisorReport(
            dataset=DatasetBundle("error", np.array([[0, 0]]), None),
            routing=RoutingDecision(None, [], trace),
            dataset_plot_path=Path(""),
            output_dir=Path(""),
            results=[],
            ranking=[],
            executive_summary=full_msg,
            decision_trace=trace,
            response_type="FOLLOW_UP",
        )
