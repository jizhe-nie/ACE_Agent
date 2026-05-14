from __future__ import annotations

import contextlib
import hashlib
import json
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
from ACE_Agent.tools.latex_generator import LatexReportGenerator
from ACE_Agent.tools.llm_client import LLMSettings, UniversalLLMClient


def _decompose_dim_to_image_hint(n_features: int) -> str | None:
    """Factorize *n_features* into H×W or H×W×C image dimensions.

    Returns a shape hint string (e.g. ``"32×32×3"``) or ``None``.
    Only triggers for plausible image dimensions: n_features ≥ 256 (≈16×16)
    and each spatial dimension ≥ 8.
    """
    if n_features < 256:
        return None
    for channels in (3, 1):
        if n_features % channels != 0:
            continue
        n_flat = n_features // channels
        root = int(round(n_flat ** 0.5))
        for h in range(max(8, root - 8), root + 9):
            if n_flat % h == 0:
                w = n_flat // h
                if 8 <= w <= 2048 and 0.2 <= h / w <= 5.0:
                    if channels == 3:
                        return f"{h}×{w}×3"
                    return f"{h}×{w}"
    return None


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
        Uses the git HEAD hash if available; otherwise falls back to the
        mtime of key source files.
        """
        try:
            import subprocess
            root = Path(__file__).resolve().parents[1]
            r = subprocess.run(
                ["git", "rev-parse", "--short=8", "HEAD"],
                cwd=str(root), capture_output=True, text=True, timeout=5,
            )
            if r.returncode == 0 and r.stdout.strip():
                return r.stdout.strip()
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

    def _update_cache(self, dataset: DatasetBundle, best_ari: float, best_nmi: float, winner: str) -> None:
        dh = self._dataset_hash(dataset)
        if not dh:
            return
        self._result_cache[dh] = {
            "best_ari": round(best_ari, 4),
            "best_nmi": round(best_nmi, 4),
            "winner": winner,
            "ts": datetime.now().isoformat(),
            "code_version": self._code_version(),
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
            _ds_ctx = dataset.display_name if dataset else ""
            intent_data = self.router.analyze_intent(
                user_prompt, self.memory, llm_settings, dataset_context=_ds_ctx,
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

        # Phase 6: Result cache check — avoid recomputing known datasets
        # Skip cache when user explicitly asks to re-run (算法改进后需要重跑验证)
        _force_rerun = any(kw in user_prompt for kw in ("重新", "重跑", "再跑", "再分析"))
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
            # Build a minimal cached report
            return SupervisorReport(
                dataset=dataset,
                routing=RoutingDecision(None, [], trace),
                dataset_plot_path=Path(""),
                output_dir=Path(""),
                results=[],
                ranking=[],
                executive_summary=(
                    f"## 缓存命中\n\n"
                    f"该数据集 `{dataset.display_name}` 此前已经运行过完整分析。\n\n"
                    f"- 优胜算法: **{_cached.get('winner', '?')}**\n"
                    f"- ARI: **{_cached.get('best_ari', 0):.3f}**\n"
                    f"- NMI: **{_cached.get('best_nmi', 0):.3f}**\n"
                    f"- 缓存时间: {_cached.get('ts', '?')}\n\n"
                    f"如需重新计算，请清除缓存文件 `.ace_result_cache.json` 或更换数据集参数。"
                ),
                decision_trace=trace,
                response_type="CLUSTER_TASK",
            )

        # Phase 6: Hopkins pre-check gate — skip doomed experts early
        _hopkins = self._fast_hopkins(dataset.X)
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
            dataset, user_prompt, llm_settings, trace, active_experts,
            constraints=constraints, progress_callback=progress_callback,
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
                self._update_cache(dataset, best_ari, best_nmi, best.algorithm_name)
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

        # ---- Phase 5.4: High-dim dimension gatekeeper --------------------
        # When n_features > 100, force PCA (95% variance) before dispatching
        # experts. High-dim raw data causes audit collapse and degrades
        # clustering quality due to the curse of dimensionality.
        working_dataset = dataset
        _n_features_raw = dataset.X.shape[1] if dataset.X.ndim == 2 else 1
        if _n_features_raw > 100:
            _highdim_result = self._apply_highdim_reduction(dataset, trace)
            if _highdim_result is not None:
                working_dataset = _highdim_result

        # ---- Phase 6: Image-aware routing ---------------------------------
        # When is_image=True and n_features > 500 (raw pixels), Euclidean
        # distance is semantically meaningless.  Activate DimensionExpert
        # so its Conv-AE / deep embedding pipelines are available.
        # Centroid algorithms (KMeans/GMM) on raw pixels will naturally rank
        # low via ARI — no need to hard-suppress them.
        _img_meta = dataset.metadata or {}
        if _img_meta.get("is_image") and _n_features_raw > 500:
            if "dimension" not in active_experts:
                active_experts.append("dimension")
                trace.append(
                    f"【图像路由】检测到图像数据 ({_n_features_raw}D)，"
                    f"已激活维度专家（深度嵌入管线）。"
                    f"提示：原始像素欧氏距离无语义判别力，建议使用 ResNet 特征模式 (cifar10_resnet)。"
                )

        # ---- Phase 2.4: Manifold pre-processing for complex topology ----
        # Detect non-convex / manifold data and apply UMAP / SpectralEmbedding
        # BEFORE experts are dispatched, so they cluster the "unfolded" data.
        # Phase 3: uses _classify_data_structure() result for detection.
        #
        # IMPORTANT: Skip when n_features > 100.  In high-dim spaces geodesic
        # distortion is unreliable (curse of dimensionality makes all point-pair
        # distances nearly equal), so manifold detection produces false positives.
        # UMAP on 512D → 3D destroys the semantic structure that CNN features encode.
        _manifold_detected = (
            working_dataset.X.shape[1] <= 100
            and (
                structure.get("structure_class") in ("manifold", "non_convex", "graph_connected")
                or self._detect_manifold_topology(working_dataset)
            )
        )
        if _manifold_detected and working_dataset.X.shape[1] > 2:
            embedded = self._apply_manifold_preprocessing(working_dataset, trace)
            if embedded is not None:
                working_dataset = embedded
                trace.append("【流形预处理】已将嵌入数据集作为后续专家的输入。")

        # ---- Adaptive sandbox timeout for large datasets -----------------
        _n_samples = working_dataset.X.shape[0]
        _base_timeout = 60
        if _n_samples > 20000:
            _base_timeout = 240
        elif _n_samples > 10000:
            _base_timeout = 180
        elif _n_samples > 5000:
            _base_timeout = 120
        elif _n_samples > 2000:
            _base_timeout = 90
        if _base_timeout > 60:
            trace.append(
                f"【沙箱超时】数据集 {_n_samples} 样本，"
                f"沙箱超时调整为 {_base_timeout}s。"
            )

        # ---- Large-sample downsampling (N > 30K) ---------------------------
        # O(N^2) algorithms (DBSCAN/OPTICS/HDBSCAN in Topology/Zoo experts)
        # time out on large datasets.  Subsampling before dispatch keeps
        # experts fast while preserving label distribution.
        _subsample_result = self._subsample_large_dataset(working_dataset, trace=trace)
        if _subsample_result is not None:
            working_dataset = _subsample_result

        n_experts = len(active_experts)

        for idx, key in enumerate(active_experts):
            expert = self.experts.get(key)
            # Apply adaptive timeout to this expert's sandbox
            if expert is not None and hasattr(expert, "sandbox"):
                try:
                    expert.sandbox.timeout_sec = _base_timeout
                except Exception:
                    pass
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
        # Use working_dataset (may be subsampled/preprocessed) so labels
        # and y_true have matching lengths.  Using the original dataset
        # when experts ran on a subset silently zeroes all ARI values.
        ranking = self._compute_informed_ranking(
            all_results, working_dataset, trace,
            centroid_ban=_conn_check["centroid_ban"],
        )
        best = ranking[0]

        # Phase 5.1: 低 ARI 看板预警
        _best_ari = self._compute_best_ari(ranking, working_dataset)
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
        if working_dataset.y is not None:
            from sklearn.metrics import adjusted_rand_score as _fail_ari_fn
            _y_fail = np.asarray(working_dataset.y, dtype=int).ravel()
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

                # Phase 5.4: image semantic awareness — detect image-shaped data
                # (e.g. 3072=32×32×3 for CIFAR-10, 784=28×28 for MNIST).
                # Raw pixel clustering fails beyond ~10D; flag for conv pipeline.
                _is_image_data = self._detect_image_data(dataset)
                if _is_image_data:
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
                    ranking = self._compute_informed_ranking(
                        all_results, working_dataset, trace,
                        centroid_ban=_conn_check["centroid_ban"],
                    )
                    best = ranking[0]
                    _new_best_ari = self._compute_best_ari(ranking, working_dataset)
                    _new_best_internal = best.metrics.get("score", 0) if hasattr(best, "metrics") else 0
                    _ari_display = f"{_new_best_ari:.3f}" if (_new_best_ari is not None and _new_best_ari > 0) else f"internal={_new_best_internal:.3f}"
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
        audit_report = self._execute_audit(best, _audit_ds, settings, trace)

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
                        f"审计沙箱超时导致指标未完成计算（哨兵值）。"
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
            _rescue_best_ari = self._compute_best_ari(ranking, dataset)
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
        retry_results = self._handle_audit_feedback(
            audit_report, working_dataset, prompt, settings, trace, active_experts
        )
        if retry_results:
            all_results.extend(retry_results)
            ranking = self._compute_informed_ranking(all_results, working_dataset, trace,
                                                     centroid_ban=_conn_check["centroid_ban"])
            best = ranking[0]
            trace.append("【Critic 2.0】约束重试完成，已重新排名。")
            _progress("正在重新审计约束重试结果...")
            audit_report = self._execute_audit(best, _audit_ds, settings, trace)

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
                _hr_s = _hr_r.metrics.get("score", 0.0) if hasattr(_hr_r, "metrics") else 0.0
                if _hr_s > _hr_best_internal:
                    _hr_best_internal = _hr_s
            # Honest retreat only when ARI < 0.4 AND no high-quality rescue exists
            _has_rescue_quality = _deep_pipeline_triggered and _hr_best_internal > 0.5
            if _hr_max_ari < 0.4 and not _has_rescue_quality:
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
                ranking = self._compute_informed_ranking(all_results, working_dataset, trace,
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
                ranking = self._compute_informed_ranking(all_results, working_dataset, trace,
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

        # Phase 6: detect global failure for honest reporting in summary
        _summary_best_ari = _best_ari  # from Phase 5.1 warning block above

        client = UniversalLLMClient(settings)
        summary = client.summarize_report(
            {
                "user_intent": prompt,
                "dataset": dataset.display_name,
                "best_algo": best.algorithm_name,
                "metrics": best.metrics,
                "score_source": best.metrics.get("score_source", "silhouette"),
                "best_ari": _summary_best_ari,
                "all_algorithms_failed": _summary_best_ari is not None and _summary_best_ari < 0.2,
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

        # ---- Phase 5.4: adaptive timeout + memory for audit sandbox -----------
        _n_features = dataset.X.shape[1] if dataset.X.ndim == 2 else 1
        _n_samples = dataset.X.shape[0]
        # Scale timeout with data dimensionality: base 45s, +5s per 100D
        _audit_timeout_sec = 45 + max(0, (_n_features - 100) // 100 * 5)
        # For large samples with high dim, add extra budget
        if _n_samples > 5000 and _n_features > 200:
            _audit_timeout_sec += 15
        _audit_timeout_sec = min(_audit_timeout_sec, 120)  # cap at 120s
        # Directly set sandbox attributes — env vars are read once at construction
        _prev_sandbox_timeout = critic.sandbox.timeout_sec
        _prev_sandbox_memory = critic.sandbox.memory_mb
        critic.sandbox.timeout_sec = _audit_timeout_sec
        # Elastic memory: bump to 4 GiB for high-dim distance matrices
        if _n_features > 500:
            critic.sandbox.memory_mb = 4096
            trace.append(
                f"【审计内存】{_n_features}D 审计上调内存配额至 4 GiB"
                f"（防止高维距离矩阵 OOM）。"
            )

        try:
            audit = critic.execute_audit(winner, dataset, settings)
            if audit and not audit.get("degraded"):
                endorsement = audit.get("endorsement", "?")
                confidence = audit.get("confidence_level", "?")
                trace.append(
                    f"【审计】完成 — 裁决: {endorsement}, 置信度: {confidence}"
                )
            elif audit and audit.get("degraded"):
                # ---- Degraded: retry with fast_audit mode -------------------
                trace.append(
                    "【审计】首次审计降级（超时/部分失败），自动切换 fast_audit 模式重试..."
                )
                try:
                    _fast_settings = settings
                    if hasattr(settings, "model_copy"):
                        _fast_settings = settings.model_copy(update={"fast_audit": True})
                    else:
                        from dataclasses import replace
                        _fast_settings = replace(settings, fast_audit=True)
                    _fast_audit = critic.execute_audit(winner, dataset, _fast_settings)
                    if _fast_audit and not _fast_audit.get("degraded"):
                        _fast_audit["fast_audit_retry"] = True
                        audit = _fast_audit
                        trace.append(
                            f"【审计】fast_audit 重试成功 — "
                            f"裁决: {audit.get('endorsement', '?')},"
                            f" 置信度: {audit.get('confidence_level', '?')}"
                        )
                    else:
                        # ---- Phase 6: auto-relax timeout + shrink sample to 500 ----
                        _relaxed_timeout = _audit_timeout_sec * 2
                        trace.append(
                            "【审计】fast_audit 仍失败，触发自适应宽松模式："
                            f"超时 {_audit_timeout_sec}s → {_relaxed_timeout}s，"
                            "采样率降至 500 样本。"
                        )
                        critic.sandbox.timeout_sec = _relaxed_timeout
                        try:
                            _relaxed_settings = _fast_settings
                            if hasattr(_fast_settings, "model_copy"):
                                _relaxed_settings = _fast_settings.model_copy(
                                    update={"fast_audit": True, "audit_relaxed": True}
                                )
                            else:
                                from dataclasses import replace
                                _relaxed_settings = replace(
                                    _fast_settings, fast_audit=True, audit_relaxed=True
                                )
                            _relaxed_audit = critic.execute_audit(
                                winner, dataset, _relaxed_settings,
                            )
                            if _relaxed_audit and not _relaxed_audit.get("degraded"):
                                _relaxed_audit["fast_audit_retry"] = True
                                _relaxed_audit["audit_relaxed"] = True
                                audit = _relaxed_audit
                                trace.append(
                                    f"【审计】自适应宽松重试成功 — "
                                    f"裁决: {audit.get('endorsement', '?')},"
                                    f" 置信度: {audit.get('confidence_level', '?')}"
                                )
                            else:
                                trace.append(
                                    "【审计】自适应宽松重试仍失败，使用降级审计报告。"
                                )
                        finally:
                            critic.sandbox.timeout_sec = _prev_sandbox_timeout
                except Exception as _fast_exc:
                    trace.append(f"【审计】fast_audit 重试异常 ({_fast_exc})，使用降级审计报告。")
            else:
                # ---- Audit returned None (sandbox timeout) — retry fast_audit ----
                trace.append(
                    "【审计】审计未产出有效报告（sandbox 超时），自动切换 fast_audit 重试..."
                )
                _retry_succeeded = False
                try:
                    # Bump timeout and enable fast_audit
                    critic.sandbox.timeout_sec = max(_audit_timeout_sec * 2, 90)
                    _fast_settings = settings
                    if hasattr(settings, "model_copy"):
                        _fast_settings = settings.model_copy(update={"fast_audit": True})
                    else:
                        from dataclasses import replace
                        _fast_settings = replace(settings, fast_audit=True)
                    _fast_audit = critic.execute_audit(winner, dataset, _fast_settings)
                    if _fast_audit and not _fast_audit.get("degraded"):
                        _fast_audit["fast_audit_retry"] = True
                        audit = _fast_audit
                        _retry_succeeded = True
                        trace.append(
                            f"【审计】fast_audit 重试成功 — "
                            f"裁决: {audit.get('endorsement', '?')},"
                            f" 置信度: {audit.get('confidence_level', '?')}"
                        )
                except Exception:
                    pass

                if not _retry_succeeded:
                    critic_diag = "; ".join(critic.last_logs[-2:]) if critic.last_logs else "（无诊断信息）"
                    trace.append(
                        f"【审计】fast_audit 重试仍失败，降级为初级审计建议。"
                        f"{critic_diag}"
                    )
                    audit = {
                        "endorsement": "qualified",
                        "action": "WARN",
                        "confidence_level": 0.3,
                        "findings": ["审计超时，无法完成全量分析。建议人工检查聚类质量。"],
                        "recommendation": "审计沙箱超时（timeout=" + str(_audit_timeout_sec) + "s），请考虑启用 fast_audit 模式或减小数据集。",
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
            critic.sandbox.timeout_sec = _prev_sandbox_timeout
            critic.sandbox.memory_mb = _prev_sandbox_memory

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

        # Compute geodesic distortion for ≤50D data.
        # Uses anchor sampling for N > 2000 to keep cost bounded.
        geodesic_distortion = 0.0
        wall_crossings = 0
        if 1 <= n_features <= 50 and n_samples >= 50:
            from ACE_Agent.tools.graph_builder import GraphBuilder as _GB
            try:
                import numpy as np
                X_np = np.asarray(dataset.X, dtype=float)
                adj = _GB.build_knn_graph(X_np)
                if n_samples <= 2000:
                    geo_dists = _GB.compute_geodesic_distances(adj)
                    geodesic_distortion = _GB.compute_distortion(X_np, geo_dists, sample_size=min(n_samples, 500))
                else:
                    # Sample anchors for large N
                    rng = np.random.RandomState(42)
                    n_anchors = min(500, n_samples)
                    anchors = rng.choice(n_samples, n_anchors, replace=False)
                    geo_dists = _GB.compute_geodesic_distances(adj, indices=anchors)
                    geodesic_distortion = _GB.compute_distortion(X_np, geo_dists, sample_size=min(n_anchors, 200))
            except Exception:
                pass

        # Detect wall-crossings when distortion is high
        if geodesic_distortion > 0.3 and n_samples <= 5000:
            from ACE_Agent.tools.graph_builder import GraphBuilder as _GB_wall
            try:
                import numpy as np
                X_np = np.asarray(dataset.X, dtype=float)
                adj = _GB_wall.build_knn_graph(X_np)
                if n_samples <= 2000:
                    geo_dists = _GB_wall.compute_geodesic_distances(adj)
                else:
                    rng = np.random.RandomState(42)
                    n_anchors = min(500, n_samples)
                    anchors = rng.choice(n_samples, n_anchors, replace=False)
                    geo_dists = _GB_wall.compute_geodesic_distances(adj, indices=anchors)
                pairs = _GB_wall.detect_wall_crossings(X_np, adj, geo_dists)
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

        if n_samples < 50:
            return {"long_range_curve": False, "centroid_ban": set(), "geodesic_distortion": 0.0}

        # For high-dim data (> 50D), geodesic distortion on raw features is
        # unreliable due to the curse of dimensionality — skip pre-check.
        if n_features > 50:
            return {"long_range_curve": False, "centroid_ban": set(), "geodesic_distortion": 0.0}

        from ACE_Agent.tools.graph_builder import GraphBuilder as _GB2
        try:
            import numpy as np
            X_np = np.asarray(dataset.X, dtype=float)
            adj = _GB2.build_knn_graph(X_np, k=min(10, max(3, int(np.sqrt(n_samples)) // 4)))

            if n_samples <= 2000:
                geo_dists = _GB2.compute_geodesic_distances(adj)
                distortion = _GB2.compute_distortion(X_np, geo_dists, sample_size=min(n_samples, 500))
            else:
                rng = np.random.RandomState(42)
                n_anchors = min(500, n_samples)
                anchors = rng.choice(n_samples, n_anchors, replace=False)
                geo_dists = _GB2.compute_geodesic_distances(adj, indices=anchors)
                distortion = _GB2.compute_distortion(X_np, geo_dists, sample_size=min(n_anchors, 200))
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
        data).  Uses _GB.topology_failure_check to detect:
        axis-aligned partitions, high conductance, wall-crossings,
        low CPS, and low modularity.

        Returns the failure report dict, or None if the check cannot run.
        """
        labels = getattr(result, "labels", None)
        if labels is None or len(labels) == 0:
            trace.append("【拓扑检测】跳过：最优结果无有效标签。")
            return None

        from ACE_Agent.tools.graph_builder import GraphBuilder as _GB3
        try:
            import numpy as np
            X_np = np.asarray(dataset.X, dtype=float)
            n_samples = X_np.shape[0]

            adj = _GB3.build_knn_graph(X_np)
            geo_dists = None
            if n_samples <= 2000:
                geo_dists = _GB3.compute_geodesic_distances(adj)

            labels_arr = np.asarray(labels, dtype=int).ravel()
            report = _GB3.topology_failure_check(
                X_np, adj, labels_arr, geo_dists,
            )
            return report
        except Exception as exc:
            trace.append(f"【拓扑检测】失败: {exc}")
            return None

    # ------------------------------------------------------------------
    # Image semantic detection (Phase 5.4)
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_image_data(dataset: DatasetBundle) -> str | None:
        """Detect image-shaped data by factorizing the feature count.

        Returns a human-readable shape hint (e.g. ``"32×32×3"``) if the
        feature count can be decomposed into H×W or H×W×C, or ``None`` if not.

        Does NOT flag data that is already a CNN / feature embedding
        (e.g. ResNet-18 512D, GAP 64D) — those are semantic vectors, not
        raw pixel grids.
        """
        # If the dataset already carries a feature_mode that is not "raw",
        # the features are semantic embeddings, not pixel values.
        fm = getattr(dataset, "feature_mode", "") or ""
        if fm in ("resnet", "resnet18", "gap", "cnn_features", "simclr"):
            return None
        meta_fm = (dataset.metadata or {}).get("feature_mode", "")
        if meta_fm in ("resnet", "resnet18", "gap", "cnn_features", "simclr"):
            return None
        # Explicit is_image=False means this is known non-image data.
        if (dataset.metadata or {}).get("is_image") is False:
            return None
        n = dataset.X.shape[1] if dataset.X.ndim == 2 else 1
        return _decompose_dim_to_image_hint(n)

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

    @staticmethod
    def _apply_highdim_reduction(
        dataset: DatasetBundle,
        trace: list[str],
    ) -> DatasetBundle | None:
        """Reduce high-dim data (>100D) via PCA retaining 95% variance
        before clustering experts are dispatched.

        High-dimensional raw data causes the curse of dimensionality,
        audit collapse (Hopkins/Bootstrap all zeros), and degraded
        clustering quality.  PCA acts as a mandatory dimension gatekeeper.

        Performance: uses randomized SVD with a 100-component cap.
        For n_samples > 5000, fits PCA on a 5000-sample subset then
        transforms the full dataset — avoids minutes-long SVD on
        60k×3072 matrices.
        """
        try:
            import numpy as np
            from sklearn.decomposition import PCA

            X = np.asarray(dataset.X, dtype=float)
            n_samples, n_features = X.shape

            if n_features <= 100:
                return None

            # Phase 6: image data carries semantic info in raw pixel dimensions.
            # PCA on raw pixels destroys class-discriminative structure (it
            # preserves brightness variance, not category separability).
            # Route to DimensionExpert's Conv-AE pipeline instead.
            _meta = dataset.metadata or {}
            if _meta.get("is_image"):
                trace.append(
                    f"【高维门禁】跳过 PCA：数据为图像 ({n_features}D ="
                    f" {_meta.get('original_shape', '?')})，"
                    f"语义信息需通过 Conv-AE 而非 PCA 提取。"
                )
                return None

            # Single PCA fit: randomized solver, capped at 100 components
            n_components_max = min(100, n_features, n_samples - 1)

            if n_samples > 5000:
                # Fit on a random subset for speed, transform all data
                rng = np.random.default_rng(42)
                fit_idx = rng.choice(n_samples, size=min(5000, n_samples), replace=False)
                pca = PCA(n_components=n_components_max, svd_solver="randomized",
                          random_state=42)
                pca.fit(X[fit_idx])
                cumsum = np.cumsum(pca.explained_variance_ratio_)
                n_keep = min(int(np.searchsorted(cumsum, 0.95) + 1), len(cumsum))
                n_keep = max(n_keep, 8)
                X_reduced = pca.transform(X)[:, :n_keep]
            else:
                pca = PCA(n_components=n_components_max, svd_solver="randomized",
                          random_state=42)
                X_reduced = pca.fit_transform(X)
                cumsum = np.cumsum(pca.explained_variance_ratio_)
                n_keep = min(int(np.searchsorted(cumsum, 0.95) + 1), len(cumsum))
                n_keep = max(n_keep, 8)
                if n_keep < X_reduced.shape[1]:
                    X_reduced = X_reduced[:, :n_keep]

            trace.append(
                f"【高维门禁】PCA {n_features}D → {n_keep}D"
                f"（保留 {cumsum[n_keep - 1]:.1%} 方差）。"
            )

            return DatasetBundle(
                name=f"{dataset.name}_pca{n_keep}",
                display_name=f"{dataset.display_name} (PCA{n_keep})",
                X=X_reduced.astype(float),
                y=dataset.y,
                description=f"PCA-reduced from {n_features}D to {n_keep}D (95% variance).",
                shape_family=dataset.shape_family,
                feature_names=[f"PC{i + 1}" for i in range(n_keep)],
                metadata={**(dataset.metadata or {}), "preprocessing": "pca_highdim"},
            )
        except Exception as exc:
            trace.append(f"【高维门禁】PCA 降维失败 ({exc})，继续使用原始数据。")
            return None

    @staticmethod
    def _subsample_large_dataset(
        dataset: DatasetBundle,
        max_samples: int = 10_000,
        trace: list[str] | None = None,
    ) -> DatasetBundle | None:
        """Stratified downsample when N > max_samples so O(N^2) experts
        (Topology / Zoo) don't timeout.  Keeps the full dataset for final
        evaluation; only the working copy passed to experts is subsampled.
        """
        trace = trace or []
        X = np.asarray(dataset.X, dtype=float)
        n_samples = X.shape[0]
        if n_samples <= max_samples:
            return None

        rng = np.random.default_rng(42)
        y = np.asarray(dataset.y).ravel() if dataset.y is not None else None

        if y is not None and len(np.unique(y)) > 1:
            from sklearn.model_selection import train_test_split
            try:
                _, _, _, _, idx_sub, _ = train_test_split(
                    X, y, np.arange(n_samples),
                    train_size=max_samples,
                    stratify=y,
                    random_state=42,
                )
            except ValueError:
                # Stratify fails when a class has < 2 samples — fall back to random
                idx_sub = rng.choice(n_samples, size=max_samples, replace=False)
        else:
            idx_sub = rng.choice(n_samples, size=max_samples, replace=False)

        idx_sub = np.sort(idx_sub)
        X_sub = X[idx_sub].copy()
        y_sub = y[idx_sub].copy() if y is not None else None

        trace.append(
            f"【大样本降采样】{n_samples} → {max_samples} 样本"
            f"{'（分层抽样，保留类别比例）' if y is not None and len(np.unique(y)) > 1 else '（随机抽样）'}。"
            f"专家将在子集上运行，O(N^2) 算法（DBSCAN/OPTICS/HDBSCAN）避免超时。"
        )

        return DatasetBundle(
            name=f"{dataset.name}_sub{max_samples}",
            display_name=f"{dataset.display_name} (sub{max_samples})",
            X=X_sub,
            y=y_sub,
            description=f"Subsampled from {n_samples} to {max_samples} samples.",
            shape_family=dataset.shape_family,
            feature_names=dataset.feature_names,
            metadata={**(dataset.metadata or {}), "preprocessing": "subsample", "original_n_samples": n_samples},
        )

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
    # Phase 6: Fast Hopkins pre-check (gatekeeper before expert dispatch)
    # ------------------------------------------------------------------

    @staticmethod
    def _fast_hopkins(X: np.ndarray, n_samples: int = 200, seed: int = 42) -> float:
        """Estimate Hopkins statistic on a small subset for fast gating.

        Returns a value in [0, 1].  Values near 0.5 suggest uniformly
        distributed data (no clustering tendency); values near 1.0 suggest
        strong clustering tendency.

        Uses *n_samples* reference points and a subsample of at most 2000
        data points to keep the O(N^2) distance computation fast.
        """
        try:
            from sklearn.neighbors import NearestNeighbors

            X_arr = np.asarray(X, dtype=float)
            n_total = X_arr.shape[0]
            rng = np.random.default_rng(seed)

            # Work on a subsample for speed
            n_work = min(n_total, 2000)
            if n_work < n_total:
                idx_work = rng.choice(n_total, size=n_work, replace=False)
                X_work = X_arr[idx_work]
            else:
                X_work = X_arr

            n_ref = min(n_samples, n_work // 2)
            if n_ref < 10:
                return 0.5

            # Sample reference points from data
            idx_ref = rng.choice(n_work, size=n_ref, replace=False)
            X_ref_data = X_work[idx_ref]

            # Generate uniform reference points within data bounding box
            mins = X_work.min(axis=0)
            maxs = X_work.max(axis=0)
            ranges = maxs - mins
            ranges[ranges == 0] = 1.0
            X_unif = rng.uniform(low=mins, high=maxs, size=(n_ref, X_work.shape[1]))

            # Nearest-neighbour distances (fit once on X_work)
            nn = NearestNeighbors(n_neighbors=2, algorithm="auto").fit(X_work)
            dist_data, _ = nn.kneighbors(X_ref_data, return_distance=True)
            dist_unif, _ = nn.kneighbors(X_unif, return_distance=True)
            # Use distance to 1st neighbour (index 0 is self for data points)
            d_data = dist_data[:, 1] if dist_data.shape[1] > 1 else dist_data[:, 0]
            d_unif = dist_unif[:, 0]

            sum_d = np.sum(d_data)
            sum_u = np.sum(d_unif)
            if sum_d + sum_u < 1e-15:
                return 0.5
            hopkins = sum_u / (sum_d + sum_u)
            return float(np.clip(hopkins, 0.0, 1.0))
        except Exception:
            return 0.5

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
        X = np.asarray(dataset.X, dtype=float)
        if X.shape[1] < 2:
            return path
        if X.shape[1] > 2:
            from sklearn.decomposition import PCA
            X = PCA(n_components=2, random_state=42).fit_transform(X)
        plt.figure(figsize=(6, 4))
        plt.scatter(X[:, 0], X[:, 1], c="gray", alpha=0.5, s=10)
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

        from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

        y_true_arr = np.asarray(y_true, dtype=int).ravel()

        # ---- compute ARI for every result -----------------------------------
        IS_ENSEMBLE_NAME = "EnsembleConsensus"
        best_individual_ari = -1.0
        best_individual_name = ""
        ensemble_entry: tuple[float, float, AlgorithmRunResult] | None = None

        entries: list[tuple[float, float, float, AlgorithmRunResult]] = []
        # Each entry: (ari, nmi, internal_raw, result)

        # Phase 5.3: collect labeled ARIs first to determine rescue threshold
        _pre_aris: list[float] = []
        for r in all_results:
            _lbl = getattr(r, "labels", None)
            if _lbl is not None and hasattr(_lbl, "__len__") and len(_lbl) > 0:
                try:
                    _pre_aris.append(
                        float(adjusted_rand_score(y_true_arr, np.asarray(_lbl, dtype=int).ravel()))
                    )
                except Exception:
                    pass
        _max_labeled_ari = max(_pre_aris) if _pre_aris else 0.0

        for r in all_results:
            labels = getattr(r, "labels", None)
            internal_raw = float(r.metrics.get("score") or 0.0)
            nmi = float(r.metrics.get("nmi") or 0.0)

            if labels is None or not hasattr(labels, "__len__") or len(labels) == 0:
                # When all labeled results have low ARI (< 0.5), unlabeled
                # rescue results (e.g. from Geodesic Pipeline) compete via
                # internal score capped at 0.65 so they can surface above
                # poor labeled results but can't fake a high ARI.
                if _max_labeled_ari < 0.5:
                    pseudo_ari = min(internal_raw, 0.65) if internal_raw > 0 else 0.0
                else:
                    pseudo_ari = -1.0
                entries.append((pseudo_ari, nmi, internal_raw, r))
                continue

            try:
                labels_arr = np.asarray(labels, dtype=int).ravel()
                ari = float(adjusted_rand_score(y_true_arr, labels_arr))
                # Compute NMI here so every result has it (not just zoo expert)
                try:
                    nmi = float(normalized_mutual_info_score(y_true_arr, labels_arr))
                except Exception:
                    nmi = 0.0
                r.metrics["nmi"] = nmi
                r.metrics["ari"] = ari
            except Exception:
                ari = 0.0

            # ---- connectivity pre-check centroid veto --------------------
            if centroid_ban and r.algorithm_name in centroid_ban:
                ari = 0.0

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
        # Tiebreak: NMI first, then internal score
        entries.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)

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
            _best_nmi = best[1]
            trace.append(
                f"【优选排名】ARI 一票否决制: best='{best_name}'"
                f" ARI={best_ari:.3f}, NMI={_best_nmi:.3f}"
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
