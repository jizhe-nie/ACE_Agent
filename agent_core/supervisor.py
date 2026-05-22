from __future__ import annotations

import contextlib
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from ACE_Agent.agent_brain.knowledge_engine import KnowledgeEngine
from ACE_Agent.agent_core import preflight, reflection
from ACE_Agent.agent_core import ranking as _ranking
from ACE_Agent.agent_core.router import MasterRouter
from ACE_Agent.agent_core.schemas import (
    AlgorithmRunResult,
    DatasetBundle,
    ModalityProfile,
    RoutingDecision,
    SupervisorReport,
    detect_modality,
)
from ACE_Agent.expert_sub_agents import build_expert_registry
from ACE_Agent.tools.latex_generator import LatexReportGenerator
from ACE_Agent.tools.llm_client import LLMSettings, MultiLLMConfig, UniversalLLMClient


def _image_shape_hint(n_features: int) -> str | None:
    """Return a shape hint string (e.g. ``"32×32×3"``) or ``None``."""
    return preflight.image_shape_hint(n_features)


class ACESupervisor:
    """主控编排器 (Orchestrator)：协调多代理完成复杂任务。

    P0.5 变更（2026-04-20）：
    - 专家注册表改用 build_expert_registry()，包含 zoo 专家（含 DBSCAN/HDBSCAN）。
    - 默认激活策略：centroid + topology + zoo（三家并行）。
    - dimension / deep_representation 已注册但默认不激活；
    - Critic 重构为后验审计方（2026-04-29）：从并行池移除，在选出最优结果后
      独立运行 _execute_audit()，输出结构化 audit_report，不参与排名。
    - 新增 CODE_EXAMPLE 分流路径（只返回代码 Markdown，不走沙箱）。
    - 每个专家执行用 try/except 包裹，异常转为日志不中断编排。
    - _error_report 增强：汇总每个专家最后 3 行日志作为排错依据。
    """

    # 默认激活的专家 key（ensemble 在第 2 步由 _execute_ensemble 独立调用）
    _DEFAULT_ACTIVE_EXPERTS: list[str] = ["centroid", "topology", "zoo"]

    _RESULT_CACHE_PATH: Path = Path(__file__).resolve().parents[1] / ".ace_result_cache.json"
    _CACHE_MAX_ENTRIES: int = 50

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

        # Result cache: (dataset_hash → best_ari, best_nmi, winner, ts)
        self._result_cache: dict[str, dict[str, Any]] = self._load_cache()

    # ------------------------------------------------------------------
    # Result cache (avoid recomputing known dataset results)
    # ------------------------------------------------------------------

    @staticmethod
    def _code_version() -> str:
        """Return a short version string that changes when the code changes.
        Uses the git HEAD hash + dirty-flag if available; otherwise falls back
        to the mtime of key source files.
        """
        try:
            import subprocess
            root = Path(__file__).resolve().parents[1]
            r = subprocess.run(
                ["git", "rev-parse", "--short=8", "HEAD"],
                cwd=str(root), capture_output=True, text=True, timeout=5,
            )
            if r.returncode == 0 and r.stdout.strip():
                ver = r.stdout.strip()
                # Detect dirty working tree — untracked files or modified tracked
                # files in directories that affect clustering results.
                dirty = subprocess.run(
                    ["git", "status", "--porcelain",
                     "agent_core/", "expert_sub_agents/", "tools/", "benchmark/"],
                    cwd=str(root), capture_output=True, text=True, timeout=5,
                )
                if dirty.returncode == 0 and dirty.stdout.strip():
                    ver += "-dirty"
                return ver
        except Exception:
            pass
        # Fallback: hash mtimes of key modules that affect clustering results
        try:
            h = hashlib.sha256()
            for rel in ["agent_core/supervisor.py", "agent_core/schemas.py",
                        "expert_sub_agents/", "tools/coder_sandbox.py"]:
                p = Path(__file__).resolve().parents[1] / rel
                if p.is_file():
                    h.update(str(p.stat().st_mtime).encode())
                elif p.is_dir():
                    for f in sorted(p.glob("*.py")):
                        h.update(str(f.stat().st_mtime).encode())
            return h.hexdigest()[:8]
        except Exception:
            return "unknown"

    @staticmethod
    def _dataset_hash(dataset: DatasetBundle) -> str:
        """Fast hash of dataset X + y + feature_mode.  Samples 1000 rows
        max to keep hashing fast even for large datasets."""
        try:
            h = hashlib.sha256()
            X = np.asarray(dataset.X, dtype=float)
            n = X.shape[0]
            # Hash shape and a sample of rows
            h.update(f"{X.shape}|{X.dtype}".encode())
            step = max(1, n // 1000)
            for i in range(0, min(n, 1000)):
                row = X[i * step] if i * step < n else X[-1]
                h.update(row.tobytes())
            if dataset.y is not None:
                y = np.asarray(dataset.y).ravel()
                h.update(y[:min(len(y), 1000)].tobytes())
            fm = getattr(dataset, "feature_mode", "") or ""
            h.update(fm.encode())
            return h.hexdigest()[:32]
        except Exception:
            return ""

    def _load_cache(self) -> dict[str, dict[str, Any]]:
        current_ver = self._code_version()
        try:
            if self._RESULT_CACHE_PATH.exists():
                raw = json.loads(self._RESULT_CACHE_PATH.read_text(encoding="utf-8"))
                # Drop entries from a different code version — stale results
                # are misleading when algorithms have changed.
                return {
                    k: v for k, v in raw.items()
                    if v.get("code_version") == current_ver
                }
        except Exception:
            pass
        return {}

    def _check_cache(self, dataset: DatasetBundle) -> dict[str, Any] | None:
        dh = self._dataset_hash(dataset)
        if not dh:
            return None
        return self._result_cache.get(dh)

    def _update_cache(
        self,
        dataset: DatasetBundle,
        best_ari: float,
        best_nmi: float,
        winner: str,
        ranking_rows: list[dict[str, object]] | None = None,
        executive_summary: str = "",
        output_dir: str = "",
        dataset_plot_path: str = "",
        winner_plot_path: str = "",
    ) -> None:
        dh = self._dataset_hash(dataset)
        if not dh:
            return
        self._result_cache[dh] = {
            "best_ari": round(best_ari, 4),
            "best_nmi": round(best_nmi, 4),
            "winner": winner,
            "ts": datetime.now().isoformat(),
            "code_version": self._code_version(),
            "ranking_rows": ranking_rows or [],
            "executive_summary": executive_summary[:3000],
            "output_dir": output_dir,
            "dataset_plot_path": dataset_plot_path,
            "winner_plot_path": winner_plot_path,
        }
        # Cap cache size
        if len(self._result_cache) > self._CACHE_MAX_ENTRIES:
            keys = sorted(self._result_cache.keys())[:self._CACHE_MAX_ENTRIES]
            self._result_cache = {k: self._result_cache[k] for k in keys}
        try:
            self._RESULT_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
            self._RESULT_CACHE_PATH.write_text(
                json.dumps(self._result_cache, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception:
            pass

    def reset_state(self) -> None:
        """Clear accumulated conversation memory for a fresh session.

        Called when the user starts a new conversation in the web UI, so the
        cached supervisor singleton doesn't leak memory/context from the
        previous session into the new one.
        """
        self.memory.clear()
        self.last_report = None

    def run(
        self,
        dataset: DatasetBundle | None,
        user_prompt: str,
        llm_config: MultiLLMConfig | None = None,
        intent_data: dict[str, Any] | None = None,
        constraints: dict[str, Any] | None = None,
        progress_callback: Any = None,
    ) -> SupervisorReport:
        """核心编排流程。

        progress_callback: optional callable(message: str, step: int, total: int)
            for real-time UI progress updates.
        """
        if llm_config is None:
            llm_config = MultiLLMConfig()
        # Cache for this run so downstream methods can use role-specific LLMs
        self._current_llm_config = llm_config
        worker_settings = llm_config.get_worker()

        def _progress(msg: str, step: int = 0, total: int = 1) -> None:
            if progress_callback:
                with contextlib.suppress(Exception):
                    progress_callback(msg, step, total)

        # 1. 语义路由
        if not intent_data:
            _ds_ctx = dataset.display_name if dataset else ""
            intent_data = self.router.analyze_intent(
                user_prompt, self.memory, llm_config.get_router(), dataset_context=_ds_ctx,
            )

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

        # HITL: 提示词级约束注入（非 COP-KMeans 等约束优化器）
        if constraints and constraints.get("reference_labels"):
            trace.append(
                f"【HITL】检测到人工标注参考标签（{len(constraints['reference_labels'])} 个数据点），"
                "将作为提示词约束注入各专家（非严格 must-link/cannot-link）。"
            )

        # 2. 意图分流
        if intent == "FOLLOW_UP":
            return self._handle_follow_up(user_prompt, llm_config.get_router(), trace)

        if intent == "CODE_EXAMPLE":
            return self._handle_code_example(user_prompt, worker_settings, trace)

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

        # ---- Modality detection (unified profile for routing/graph/audit) --
        modality = detect_modality(dataset)
        trace.append(
            f"【模态检测】{modality.modality_type}"
            f"（metric={modality.distance_metric}）"
        )

        # Phase 3 Topology-Aware: classify data structure and route experts
        structure = preflight.classify_data_structure(dataset, modality=modality)
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

        # Phase 3.3: hierarchical structure routing
        if structure.get("is_hierarchical"):
            trace.append(
                f"【分层结构检测】Ward linkage 分析发现主导性 k=2 分裂"
                f"（分裂比={structure.get('hierarchical_k2_ratio', 0):.2f}），"
                f"数据具有清晰的分层结构。"
                f"建议：将 k=2 结果作为主要分层的有效替代方案；"
                f"若需更细粒度聚类，考虑对子簇递归聚类。"
            )

        # Phase 6: Result cache check — avoid recomputing known datasets
        # Skip cache when user explicitly asks to re-run (算法改进后需要重跑验证)
        _force_rerun = any(kw in user_prompt for kw in ("重新", "重跑", "再跑", "再分析", "再次"))
        _cached = None if _force_rerun else self._check_cache(dataset)
        if _cached:
            trace.append(
                f"【结果缓存】该数据集已在此前运行过，缓存结果: "
                f"winner='{_cached.get('winner', '?')}', "
                f"ARI={_cached.get('best_ari', 0):.3f}, "
                f"NMI={_cached.get('best_nmi', 0):.3f} "
                f"(缓存时间: {_cached.get('ts', '?')})。"
                f"将跳过本次计算，直接返回缓存结论。"
            )
            # Reconstruct ranking list from cached lightweight rows
            _cached_ranking: list[AlgorithmRunResult] = []
            for _row in (_cached.get("ranking_rows") or []):
                _cached_ranking.append(AlgorithmRunResult(
                    algorithm_name=str(_row.get("algorithm", "?")),
                    expert_key="",
                    expert_label=str(_row.get("expert", "")),
                    labels=np.array([]),
                    metrics={
                        "ari": _row.get("ari", -1.0),
                        "nmi": _row.get("nmi", 0.0),
                    },
                    plot_path=Path(""),
                ))
            # Set winner plot path from cache
            _winner_plot = _cached.get("winner_plot_path", "")
            if _cached_ranking and _winner_plot:
                _cached_ranking[0].plot_path = Path(_winner_plot)
            _cached_summary = _cached.get("executive_summary", "") or (
                f"## 缓存命中\n\n"
                f"该数据集 `{dataset.display_name}` 此前已经运行过完整分析。\n\n"
                f"- 优胜算法: **{_cached.get('winner', '?')}**\n"
                f"- ARI: **{_cached.get('best_ari', 0):.3f}**\n"
                f"- NMI: **{_cached.get('best_nmi', 0):.3f}**\n"
                f"- 缓存时间: {_cached.get('ts', '?')}\n\n"
                f"如需重新计算，请清除缓存文件 `.ace_result_cache.json` 或更换数据集参数。"
            )
            return SupervisorReport(
                dataset=dataset,
                routing=RoutingDecision(None, [], trace, modality=modality),
                dataset_plot_path=Path(_cached.get("dataset_plot_path", "") or ""),
                output_dir=Path(_cached.get("output_dir", "") or ""),
                results=_cached_ranking,
                ranking=_cached_ranking,
                executive_summary=_cached_summary,
                decision_trace=trace,
                response_type="CLUSTER_TASK",
            )

        # Phase 6: Hopkins pre-check gate — skip doomed experts early
        _hopkins = preflight.fast_hopkins(dataset.X)
        trace.append(
            f"【Hopkins预检】快速评估值 = {_hopkins:.3f}"
            f"{'（< 0.3，数据聚类倾向极弱）' if _hopkins < 0.3 else '（聚类倾向正常）'}"
        )
        if _hopkins < 0.3:
            _removed = []
            for _k in ("topology", "zoo", "graph"):
                if _k in active_experts:
                    active_experts.remove(_k)
                    _removed.append(_k)
            if _removed:
                trace.append(
                    f"【Hopkins门禁】Hopkins={_hopkins:.3f} < 0.3，"
                    f"密度/谱/图专家在低聚类倾向数据上几乎必然失败，"
                    f"已自动跳过: {', '.join(_removed)}。"
                    f"保留质心+降维管线以节省 {_hopkins:.0%} 无效计算时间。"
                )

        result = self._execute_full_analysis(
            dataset, user_prompt, worker_settings, trace, active_experts,
            constraints=constraints, progress_callback=progress_callback,
            modality=modality,
            structure=structure,
            maze_connectivity_mode=maze_connectivity_mode,
            community_native_mode=community_native_mode,
        )
        # Update result cache after successful run
        if result.ranking:
            best = result.ranking[0]
            best_m = best.metrics if isinstance(best.metrics, dict) else {}
            best_ari = float(best_m.get("ari", -1.0))
            best_nmi = float(best_m.get("nmi", 0.0))
            if best_ari > -0.5:
                _ranking_rows = []
                for i, item in enumerate(result.ranking[:10]):
                    _m = getattr(item, "metrics", {}) or {}
                    _ranking_rows.append({
                        "rank": i + 1,
                        "algorithm": getattr(item, "algorithm_name", "?"),
                        "expert": getattr(item, "expert_label", ""),
                        "ari": _m.get("ari", -1.0) if isinstance(_m, dict) else -1.0,
                        "nmi": _m.get("nmi", 0.0) if isinstance(_m, dict) else 0.0,
                    })
                _ds_plot = str(getattr(result, "dataset_plot_path", ""))
                _winner_plot = str(getattr(best, "plot_path", ""))
                _out_dir = str(getattr(result, "output_dir", ""))
                self._update_cache(
                    dataset, best_ari, best_nmi, best.algorithm_name,
                    ranking_rows=_ranking_rows,
                    executive_summary=getattr(result, "executive_summary", ""),
                    output_dir=_out_dir,
                    dataset_plot_path=_ds_plot,
                    winner_plot_path=_winner_plot,
                )
        return result

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
        modality: ModalityProfile | None = None,
    ) -> SupervisorReport:
        """执行完整的自动化聚类实验流。"""
        if structure is None:
            structure = {}
        if modality is None:
            modality = detect_modality(dataset)

        def _progress(msg: str, step: int = 0, total: int = 1) -> None:
            if progress_callback:
                with contextlib.suppress(Exception):
                    progress_callback(msg, step, total)

        output_dir = preflight.prepare_output_dir(dataset.name)
        all_results: list[AlgorithmRunResult] = []
        expert_logs: dict[str, list[str]] = {}

        # ---- Pre-flight gates (delegated to preflight module) ----------
        preflight_result = preflight.run_preflight_gates(
            dataset, prompt, trace, active_experts,
            settings=settings, constraints=constraints,
            structure=structure, modality=modality,
        )
        working_dataset = preflight_result["working_dataset"]
        prompt = preflight_result["prompt"]
        constraints = preflight_result["constraints"]
        active_experts = preflight_result["active_experts"]
        _base_timeout = preflight_result["base_timeout"]
        modality = preflight_result["modality"]

        n_experts = len(active_experts)

        for idx, key in enumerate(active_experts):
            expert = self.experts.get(key)
            # Apply adaptive timeout + output_dir to this expert's sandbox
            if expert is not None and hasattr(expert, "sandbox"):
                with contextlib.suppress(Exception):
                    expert.sandbox.timeout_sec = _base_timeout
                    expert.sandbox.output_dir = str(output_dir)
            _progress(f"正在运行 {key} 专家 ({idx + 1}/{n_experts})...", idx + 1, n_experts)
            if expert is None:
                trace.append(f"【主控】警告：专家 '{key}' 未在注册表中找到，跳过。")
                continue
            try:
                _expert_c = dict(constraints) if constraints else {}
                if key == "dimension" and getattr(settings, "deep_mode", False):
                    _expert_c["deep_mode"] = True
                expert_results = expert.execute_with_self_correction(
                    working_dataset, prompt, settings, constraints=_expert_c or None
                )
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
        _conn_check = preflight.connectivity_pre_check(dataset, trace, modality=modality)
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
        # Use working_dataset (may be subsampled/preprocessed) so labels
        # and y_true have matching lengths.  Using the original dataset
        # when experts ran on a subset silently zeroes all ARI values.
        ranking = _ranking.compute_informed_ranking(
            all_results, working_dataset, trace,
            centroid_ban=_conn_check["centroid_ban"],
            modality=modality,
        )
        best = ranking[0]

        # Phase 5.1: 低 ARI 看板预警
        _best_ari = _ranking.compute_best_ari(ranking, working_dataset)
        if _best_ari is not None and _best_ari < 0.7:
            trace.append(
                f"⚠️ 预警：当前最高 ARI 仅 {_best_ari:.3f}（< 0.7），"
                f"所有算法均未能有效捕捉数据结构，聚类结果仅供参考，"
                f"建议人工干预参数或更换算法策略。"
            )

        # Phase 5.1: 全局低 ARI 诚实失败检测
        _global_low_ari = False
        if working_dataset.y is not None:
            from sklearn.metrics import adjusted_rand_score as _global_ari_fn
            _y_true = np.asarray(working_dataset.y, dtype=int).ravel()
            _all_aris = []
            for _r in all_results:
                _rl = getattr(_r, "labels", None)
                if _rl is not None and hasattr(_rl, "__len__") and len(_rl) > 0:
                    with contextlib.suppress(Exception):
                        _all_aris.append(float(_global_ari_fn(_y_true, np.asarray(_rl, dtype=int).ravel())))
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
        if working_dataset.y is not None:
            from sklearn.metrics import adjusted_rand_score as _fail_ari_fn
            _y_fail = np.asarray(working_dataset.y, dtype=int).ravel()
            for _r in all_results:
                if _r.algorithm_name in ("EnsembleConsensus",) or _r.algorithm_name.startswith("_"):
                    continue
                _rl = getattr(_r, "labels", None)
                if _rl is not None and hasattr(_rl, "__len__") and len(_rl) > 0:
                    with contextlib.suppress(Exception):
                        _all_independent_aris.append(
                            float(_fail_ari_fn(_y_fail, np.asarray(_rl, dtype=int).ravel()))
                        )
            if _all_independent_aris:
                _max_independent_ari = max(_all_independent_aris)
                if _max_independent_ari < 0.2:
                    # ARI < 0.2: data structure fundamentally inaccessible via
                    # Euclidean methods — UMAP embedding cannot rescue this.
                    trace.append(
                        f"【诚实失败快速出口】ARI < 0.2（max={_max_independent_ari:.3f}），"
                        f"数据结构无法通过欧氏空间方法捕捉，跳过救助管线。"
                    )
                    _deep_pipeline_triggered = False
                elif _max_independent_ari < 0.7:
                    _deep_pipeline_triggered = True
                    _best_independent_ari = _max_independent_ari
                    trace.append(
                        f"【FAILED / 结构识别失败】所有独立算法 ARI 均低于 0.7"
                        f"（最高 ARI={_best_independent_ari:.3f}），"
                        f"系统判定当前欧氏空间方法无法有效捕捉数据结构。"
                        f"自动触发 Deep Pipeline（DimensionExpert + UMAP 流形嵌入）。"
                    )
                    _progress("FAILED 裁决：触发 Geodesic Deep Pipeline...")
                    _deep_results: list[AlgorithmRunResult] = []

                if _deep_pipeline_triggered:
                    # Phase 5.4: image semantic awareness — detect image-shaped data
                    # (e.g. 3072=32×32×3 for CIFAR-10, 784=28×28 for MNIST).
                    # Raw pixel clustering fails beyond ~10D; flag for conv pipeline.
                    _is_image_data = preflight.detect_image_data(dataset)
                    if _is_image_data:
                        _n_features_raw = dataset.X.shape[1] if dataset.X.ndim == 2 else 1
                        trace.append(
                            f"【图像语义感知】检测到图像形数据"
                            f"（{_n_features_raw}D = {_is_image_data}），"
                            f"原始像素聚类在 ≥10D 时失效。"
                            f"如需语义分组请使用 CNN 特征提取或 Conv-AE。"
                        )

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
                        ranking = _ranking.compute_informed_ranking(
                            all_results, working_dataset, trace,
                            centroid_ban=_conn_check["centroid_ban"],
                            modality=modality,
                        )
                        best = ranking[0]
                        _new_best_ari = _ranking.compute_best_ari(ranking, working_dataset)
                        _new_best_internal = best.metrics.get("score") or 0.0 if hasattr(best, "metrics") else 0.0
                        if _new_best_ari is not None and _new_best_ari > 0:
                            _ari_display = f"{_new_best_ari:.3f}"
                        else:
                            _ari_display = f"internal={_new_best_internal:.3f}"
                        trace.append(
                            f"【Geodesic Pipeline 完成】产出 {len(_deep_results)} 个结果，"
                            f"重新排名后最佳 {_ari_display}"
                        )
                    else:
                        trace.append("【Geodesic Pipeline】所有路径均未产出有效结果。")

        # 后验审计：Critic 独立审查最优结果
        # Phase 5.4: audit in the reduced space (working_dataset), not the
        # original high-dim dataset — avoids dimension-cursed distance
        # matrices that cause 120s+ timeout even with subsampling.
        _progress("正在进行 Critic 后验审计...")
        _audit_ds = working_dataset if working_dataset is not dataset else dataset

        # 审计特征空间对齐：若优胜算法使用了降维嵌入，在嵌入空间中审计
        _embed_path = getattr(best, "embedding_path", None)
        if _embed_path and _embed_path.exists():
            _embed_X = np.load(str(_embed_path))
            from dataclasses import replace as _dc_replace
            _audit_ds = _dc_replace(_audit_ds, X=_embed_X.astype(float))
            trace.append(f"【审计】切换至优胜算法嵌入空间 ({_embed_X.shape[1]}D) 进行审计。")

        audit_report = reflection.execute_audit(
            self.experts.get("critic"), best, _audit_ds, settings, trace, modality=modality,
        )

        # Phase 5.4: Audit collapse detection — when metrics are sentinel
        # values (negative = not computed due to sandbox timeout), flag
        # the audit as collapsed.  Distinguish from genuinely near-zero
        # metrics that may be valid for difficult datasets.
        if audit_report:
            _sv = audit_report.get("stability_score")
            _hv = audit_report.get("hopkins")
            _cv = audit_report.get("confidence_level")
            _stab = float(_sv if _sv is not None else -1)
            _hop = float(_hv if _hv is not None else -1)
            _conf = float(_cv if _cv is not None else 1)
            _not_computed = _stab < 0 or _hop < 0
            _near_zero = 0 <= _stab <= 0.01 and 0 <= _hop <= 0.01
            if (_not_computed or _near_zero) and _conf < 0.5:
                audit_report["audit_collapse"] = True
                if _not_computed:
                    _collapse_msg = (
                        f"审计引擎坍缩：bootstrap_stability={_stab:.2f}, hopkins={_hop:.2f}。"
                        f"审计执行超时导致指标未完成计算（哨兵值）。"
                        f"聚类结果的实际可信度无法验证，请人工审查。"
                    )
                else:
                    _collapse_msg = (
                        f"审计引擎坍缩：bootstrap_stability={_stab:.2f}, hopkins={_hop:.2f}。"
                        f"所有审计指标接近零，聚类质量无法通过内部验证确认。"
                        f"请优先关注聚类结果的 ARI/NMI 而非审计指标。"
                    )
                audit_report.setdefault("findings", []).insert(0, _collapse_msg)
                if not audit_report.get("recommendation"):
                    audit_report["recommendation"] = (
                        "审计引擎在高维数据上超时坍缩，所有校验指标不可用。"
                        "建议：1) 启用 fast_audit 模式重试；2) 手动检查聚类结果有效性；"
                        "3) 对 >100D 数据先做 PCA 降维再审计。"
                    )
                trace.append(f"【审计坍缩警告】{_collapse_msg}")

        # Phase 5.1: 当全局 ARI < 0.5 时，追加诚实失败评估
        if _global_low_ari and audit_report:
            audit_report["global_low_ari"] = True
            if (audit_report.get("stability_score") or 1.0) < 0.5:
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

        # Phase 5.3: FAILED/HEALED verdict — check rescue outcome
        if _deep_pipeline_triggered:
            if audit_report is None:
                audit_report = {}
            # Check whether rescue produced a significant improvement
            _rescue_best_ari = _ranking.compute_best_ari(ranking, dataset)
            _rescue_succeeded = (
                _rescue_best_ari is not None
                and _rescue_best_ari > max(_all_independent_aris)
                and _rescue_best_ari >= 0.7
            )
            if _rescue_succeeded:
                audit_report["verdict"] = "HEALED"
                audit_report["structure_recognition_healed"] = True
                audit_report.setdefault("findings", []).insert(
                    0,
                    f"结构识别已修复：Geodesic Pipeline 成功产出 ARI={_rescue_best_ari:.3f}"
                    f"（初始独立专家最高 ARI={max(_all_independent_aris):.3f}）。"
                    f"流形嵌入方法有效捕捉了数据内在结构。"
                )
                if not audit_report.get("recommendation"):
                    audit_report["recommendation"] = (
                        "当前数据内在结构需通过流形嵌入（UMAP）揭示。"
                        "建议后续对此类高维数据默认启用流形预处理。"
                    )
                trace.append(
                    f"【HEALED / 救助成功】Geodesic Pipeline 将 ARI 从"
                    f" {max(_all_independent_aris):.3f} 提升至 {_rescue_best_ari:.3f}，"
                    f"结构识别已修复。"
                )
            else:
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
        retry_results = reflection.handle_audit_feedback(
            audit_report, working_dataset, prompt, settings, trace, active_experts,
            self.experts, all_results=all_results,
        )
        if retry_results:
            all_results.extend(retry_results)
            # ---- Critic 2.0: enforce blocked_algorithms at collection level ----
            _blocked = list(audit_report.get("retry_constraints", {}).get("blocked_algorithms", []) or [])
            if _blocked:
                _before_count = len(all_results)
                all_results = [r for r in all_results if r.algorithm_name not in set(_blocked)]
                _removed = _before_count - len(all_results)
                if _removed > 0:
                    trace.append(
                        f"【Critic 2.0】已从结果池移除 {_removed} 个被封锁算法: {_blocked}"
                    )
            ranking = _ranking.compute_informed_ranking(
                all_results, working_dataset, trace,
                centroid_ban=_conn_check["centroid_ban"],
                blocked_algorithms=_blocked if _blocked else None,
                modality=modality,
            )
            best = ranking[0]
            trace.append("【Critic 2.0】约束重试完成，已重新排名。")
            _progress("正在重新审计约束重试结果...")
            audit_report = reflection.execute_audit(
                self.experts.get("critic"), best, _audit_ds, settings, trace, modality=modality,
            )

        # 集成共识：仅在 Critic 对单一最优结果有保留时触发
        # Phase 3.1: maze mode 始终触发 ensemble 以便 graph consensus
        #
        # Phase 5.3 Honest Retreat: 当所有独立专家 ARI < 0.4 且
        # Geodesic Pipeline 未产出高内部质量结果时，禁止生成集成融合
        # 结果，直接向用户诚实报告失败。
        _honest_retreat = False
        if working_dataset.y is not None:
            import numpy as _hr_np
            from sklearn.metrics import adjusted_rand_score as _hr_ari
            _hr_y = _hr_np.asarray(working_dataset.y, dtype=int).ravel()
            _hr_max_ari = 0.0
            _hr_best_internal = 0.0
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
                _hr_s = (_hr_r.metrics.get("score") or 0.0) if hasattr(_hr_r, "metrics") else 0.0
                if _hr_s > _hr_best_internal:
                    _hr_best_internal = _hr_s
            # Honest retreat only when ARI < 0.4 AND no high-quality rescue exists
            _has_rescue_quality = _deep_pipeline_triggered and _hr_best_internal > 0.5
            # Tightened gate: absolute floor at ARI < 0.2 always skips;
            # ARI < 0.3 only skips when rescue didn't produce high internal quality.
            if _hr_max_ari < 0.2 or (_hr_max_ari < 0.3 and not _has_rescue_quality):
                _honest_retreat = True
                trace.append(
                    f"【集成诚实退避】所有独立专家 ARI < 0.4（max={_hr_max_ari:.3f}），"
                    f"系统拒绝生成集成融合结果。"
                    f"原因：数据内在结构极其复杂，现有模型全线失效，"
                    f"强行融合只会产生具有误导性的一致性报告。"
                    f"建议：1) 手动调整 k-NN 图参数；2) 引入领域知识约束；"
                    f"3) 考虑非欧氏距离度量。"
                )
            elif _hr_max_ari < 0.4 and _has_rescue_quality:
                trace.append(
                    f"【集成诚实退避-覆盖】ARI 低（max={_hr_max_ari:.3f}）但"
                    f"Geodesic Pipeline 产出高内部质量结果"
                    f"（best internal={_hr_best_internal:.3f}），允许集成融合。"
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
            _topology_mode = preflight.detect_manifold_topology(dataset, audit_report)
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
                output_dir=output_dir,
            )
            if consensus_result is not None:
                all_results.append(consensus_result)
                # Re-rank with consensus result included
                ranking = _ranking.compute_informed_ranking(all_results, working_dataset, trace,
                                                         centroid_ban=_conn_check["centroid_ban"],
                                                         modality=modality)
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
                ranking = _ranking.compute_informed_ranking(all_results, working_dataset, trace,
                                                         centroid_ban=_conn_check["centroid_ban"],
                                                         modality=modality)
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
        _ranking.cross_validate_graph_winner(best, dataset, all_results, trace)

        # Phase 3.1: topology failure detection (maze connectivity mode)
        topology_failure_report: dict[str, Any] | None = None
        if maze_connectivity_mode and best is not None:
            topology_failure_report = _ranking.check_topology_failure(
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

        # ---- LLM-3 Reflection summary + Report assembly -------------------
        _summary_best_ari = _best_ari
        _llm_cfg = self._current_llm_config if hasattr(self, "_current_llm_config") else None
        summary = reflection.generate_llm_summary(
            best, dataset, prompt, all_results, _summary_best_ari,
            llm_config=_llm_cfg, fallback_settings=settings,
        )

        dataset_plot_path = preflight.save_raw_plot(dataset, output_dir)
        report = reflection.assemble_final_report(
            dataset, output_dir, all_results, ranking, summary,
            audit_report, trace, modality, dataset_plot_path,
        )

        with contextlib.suppress(Exception):
            report.latex_path = LatexReportGenerator().generate(report)

        self.last_report = report
        self._last_report_light = {
            "executive_summary": report.executive_summary,
            "ranking": [
                {"algorithm_name": r.algorithm_name, "metrics": dict(r.metrics)}
                for r in (report.ranking or [])
            ],
            "dataset": report.dataset,
            "routing": report.routing,
            "dataset_plot_path": report.dataset_plot_path,
            "output_dir": report.output_dir,
            "response_type": report.response_type,
        }
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
        modality: ModalityProfile | None = None,
    ) -> dict[str, Any] | None:
        return reflection.execute_audit(
            self.experts.get("critic"), winner, dataset, settings, trace,
            modality=modality,
        )

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
        all_results: list[AlgorithmRunResult] | None = None,
    ) -> list[AlgorithmRunResult]:
        return reflection.handle_audit_feedback(
            audit_report, dataset, prompt, settings, trace,
            active_experts, self.experts, all_results,
        )

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
        output_dir: Path = Path(""),
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
                output_dir=output_dir,
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
        *,
        modality: ModalityProfile | None = None,
    ) -> dict[str, Any]:
        return preflight.classify_data_structure(dataset, modality=modality)

    # ------------------------------------------------------------------
    # Phase 5.2: Geometric connectivity pre-check
    # ------------------------------------------------------------------

    @staticmethod
    def _connectivity_pre_check(
        dataset: DatasetBundle,
        trace: list[str],
        *,
        modality: ModalityProfile | None = None,
    ) -> dict[str, Any]:
        return preflight.connectivity_pre_check(dataset, trace, modality=modality)

    # ------------------------------------------------------------------
    # Compute best ARI across a ranking (for dashboard warning)
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_best_ari(
        ranking: list[AlgorithmRunResult],
        dataset: DatasetBundle,
    ) -> float | None:
        return _ranking.compute_best_ari(ranking, dataset)

    # ------------------------------------------------------------------
    # Phase 3.1: Topology failure detection for maze connectivity mode
    # ------------------------------------------------------------------

    @staticmethod
    def _check_topology_failure(
        dataset: DatasetBundle,
        result: AlgorithmRunResult,
        trace: list[str],
    ) -> dict[str, Any] | None:
        return _ranking.check_topology_failure(dataset, result, trace)

    # ------------------------------------------------------------------
    # Image semantic detection (Phase 5.4)
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_image_data(dataset: DatasetBundle) -> str | None:
        return preflight.detect_image_data(dataset)

    # ------------------------------------------------------------------
    # Topology / Manifold detection & preprocessing (Phase 2.4)
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_manifold_topology(
        dataset: DatasetBundle,
        audit_report: dict[str, Any] | None = None,
    ) -> bool:
        return preflight.detect_manifold_topology(dataset, audit_report)

    @staticmethod
    def _apply_highdim_reduction(
        dataset: DatasetBundle,
        trace: list[str],
    ) -> DatasetBundle | None:
        return preflight.apply_highdim_reduction(dataset, trace)

    @staticmethod
    def _apply_hard_dim_reduction(
        dataset: DatasetBundle,
        target_dim: int,
        trace: list[str],
        *,
        modality: ModalityProfile | None = None,
    ) -> DatasetBundle | None:
        return preflight.apply_hard_dim_reduction(
            dataset, target_dim, trace, modality=modality,
        )

    @staticmethod
    def _subsample_large_dataset(
        dataset: DatasetBundle,
        max_samples: int = 10_000,
        trace: list[str] | None = None,
    ) -> DatasetBundle | None:
        return preflight.subsample_large_dataset(dataset, max_samples, trace)

    @staticmethod
    def _compute_data_cost_budget(dataset: DatasetBundle) -> dict[str, Any]:
        return preflight.compute_data_cost_budget(dataset)

    def _apply_manifold_preprocessing(
        self,
        dataset: DatasetBundle,
        trace: list[str],
    ) -> DatasetBundle | None:
        return preflight.apply_manifold_preprocessing(dataset, trace)

    # ------------------------------------------------------------------
    # FOLLOW_UP 路径
    # ------------------------------------------------------------------

    def _handle_follow_up(self, prompt: str, settings: LLMSettings, trace: list[str]) -> SupervisorReport:
        """纯 LLM 驱动的追问或学术咨询处理。"""
        client = UniversalLLMClient(settings)
        _prev = self._last_report_light if hasattr(self, "_last_report_light") else None

        if _prev:
            context = {
                "last_summary": _prev["executive_summary"],
                "ranking": [
                    {"algo": r["algorithm_name"], "score": r["metrics"].get("score")}
                    for r in (_prev.get("ranking") or [])
                ],
            }
            system_msg = f"你是一个数据科学专家。请基于以下聚类背景及检索到的知识回答用户问题：\n{context}"
        else:
            system_msg = "你是一个数据科学专家。请基于检索到的学术背景知识回答用户的理论咨询。"

        res = client.chat_completion([{"role": "system", "content": system_msg}, {"role": "user", "content": prompt}])
        trace.append("【主控】正在基于知识库与会话上下文进行深度解析...")

        report = SupervisorReport(
            dataset=(
                _prev["dataset"]
                if _prev
                else DatasetBundle("Consultation", np.array([[0, 0]]), None)
            ),
            routing=(_prev["routing"] if _prev else RoutingDecision(None, [], trace)),
            dataset_plot_path=(_prev["dataset_plot_path"] if _prev else Path("")),
            output_dir=_prev["output_dir"] if _prev else Path(""),
            results=[],
            ranking=[],
            executive_summary=res or "无法生成回答。",
            decision_trace=trace,
            response_type="FOLLOW_UP",
        )
        return report

    # ------------------------------------------------------------------
    # CODE_EXAMPLE 路径（P0.5-C 新增）
    # ------------------------------------------------------------------

    def _handle_code_example(self, prompt: str, settings: LLMSettings, trace: list[str]) -> SupervisorReport:
        """处理 CODE_EXAMPLE 意图：用 LLM 生成自包含代码，不进入执行器，不生成图。

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
        _prev = self._last_report_light if hasattr(self, "_last_report_light") else None
        placeholder_ds = (
            _prev["dataset"] if _prev else DatasetBundle("code_example", np.array([[0, 0]]), None)
        )
        report = SupervisorReport(
            dataset=placeholder_ds,
            routing=(_prev["routing"] if _prev else RoutingDecision(None, [], trace)),
            dataset_plot_path=(_prev["dataset_plot_path"] if _prev else Path("")),
            output_dir=_prev["output_dir"] if _prev else Path(""),
            results=[],
            ranking=[],
            executive_summary=code_md,
            decision_trace=trace,
            response_type="CODE_EXAMPLE",
        )
        # 不更新 self.last_report（不覆盖上次真实分析结果）
        return report

    # ------------------------------------------------------------------
    # Phase 6: Fast Hopkins pre-check (gatekeeper before expert dispatch)
    # ------------------------------------------------------------------

    @staticmethod
    def _fast_hopkins(X: np.ndarray, n_samples: int = 200, seed: int = 42) -> float:
        return preflight.fast_hopkins(X, n_samples, seed)

    # ------------------------------------------------------------------
    # 辅助方法
    # ------------------------------------------------------------------

    def _prepare_output_dir(self, name: str) -> Path:
        return preflight.prepare_output_dir(name)

    def _save_raw_plot(self, dataset: DatasetBundle, out_dir: Path) -> Path:
        return preflight.save_raw_plot(dataset, out_dir)

    # ------------------------------------------------------------------
    # Phase 5.1: Informed ranking — ARI one-vote veto when labels exist
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_informed_ranking(
        all_results: list[AlgorithmRunResult],
        dataset: DatasetBundle,
        trace: list[str],
        centroid_ban: set[str] | None = None,
        blocked_algorithms: list[str] | None = None,
        *,
        modality: ModalityProfile | None = None,
    ) -> list[AlgorithmRunResult]:
        return _ranking.compute_informed_ranking(
            all_results, dataset, trace, centroid_ban, blocked_algorithms,
            modality=modality,
        )

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
        _ranking.cross_validate_graph_winner(best, dataset, all_results, trace)

    # ------------------------------------------------------------------
    # Error / fallback report
    # ------------------------------------------------------------------

    def _error_report(
        self,
        msg: str,
        trace: list[str],
        expert_logs: dict[str, list[str]] | None = None,
    ) -> SupervisorReport:
        return reflection.build_error_report(msg, trace, expert_logs)
