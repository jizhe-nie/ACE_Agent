"""
Post-ranking audit, Critic 2.0 closed-loop retry, and LLM-3 reflection.

Extracted from ACESupervisor.  Functions that previously accessed
``self.experts`` now take the needed expert(s) as explicit parameters.
"""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Any

import numpy as np

from ACE_Agent.agent_core.schemas import (
    AlgorithmRunResult,
    DatasetBundle,
    ModalityProfile,
    ProfileReport,
    RoutingDecision,
    SupervisorReport,
)
from ACE_Agent.tools.llm_client import LLMSettings, MultiLLMConfig, UniversalLLMClient

# ---------------------------------------------------------------------------
# Critic audit execution
# ---------------------------------------------------------------------------

def execute_audit(
    critic_expert: Any,  # CriticExpert (lazy import to avoid circular deps)
    winner: AlgorithmRunResult,
    dataset: DatasetBundle,
    settings: LLMSettings,
    trace: list[str],
    *,
    modality: ModalityProfile | None = None,
) -> dict[str, Any] | None:
    """Run CriticExpert as a post-hoc auditor on the winning result.

    Phase 5.3: sets a shorter sandbox timeout (50% of main clustering
    time) for the audit.  On timeout, degrades to a basic audit
    recommendation rather than failing entirely.
    """
    if critic_expert is None:
        trace.append("【审计】Critic 专家未注册，跳过审计。")
        return None

    fast_audit = bool(getattr(settings, "fast_audit", False))

    # ---- Phase 5.4: adaptive timeout + memory for audit sandbox -----------
    _n_features = dataset.X.shape[1] if dataset.X.ndim == 2 else 1
    _n_samples = dataset.X.shape[0]

    # Auto fast-track for high-dim data to avoid guaranteed sandbox timeout
    if _n_features > 500 and not fast_audit:
        fast_audit = True
        trace.append(
            f"【审计】{_n_features}D 高维数据自动切换快速审计模式"
            f"（跳过全量审计以避免执行超时坍缩）。"
        )

    mode_label = "快速审计" if fast_audit else "完整审计"
    trace.append(
        f"【审计】对最优结果 '{winner.algorithm_name}' 启动 {mode_label}..."
    )
    # Scale timeout with data dimensionality: base 45s, +5s per 100D
    _audit_timeout_sec = 45 + max(0, (_n_features - 100) // 100 * 5)
    # For large samples with high dim, add extra budget
    if _n_samples > 5000 and _n_features > 200:
        _audit_timeout_sec += 15
    _audit_timeout_sec = min(_audit_timeout_sec, 240 if _n_features > 500 else 120)
    # Directly set sandbox attributes — env vars are read once at construction
    _prev_sandbox_timeout = critic_expert.sandbox.timeout_sec
    _prev_sandbox_memory = critic_expert.sandbox.memory_mb
    critic_expert.sandbox.timeout_sec = _audit_timeout_sec
    # Elastic memory: bump to 4 GiB for high-dim distance matrices
    if _n_features > 500:
        critic_expert.sandbox.memory_mb = 4096
        trace.append(
            f"【审计内存】{_n_features}D 审计上调内存配额至 4 GiB"
            f"（防止高维距离矩阵 OOM）。"
        )

    try:
        audit = critic_expert.execute_audit(winner, dataset, settings, modality=modality)
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
                _fast_audit = critic_expert.execute_audit(
                    winner, dataset, _fast_settings, modality=modality,
                )
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
                    critic_expert.sandbox.timeout_sec = _relaxed_timeout
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
                        _relaxed_audit = critic_expert.execute_audit(
                            winner, dataset, _relaxed_settings, modality=modality,
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
                        critic_expert.sandbox.timeout_sec = _prev_sandbox_timeout
            except Exception as _fast_exc:
                trace.append(
                    f"【审计】fast_audit 重试异常 ({_fast_exc})，使用降级审计报告。"
                )
        else:
            # ---- Audit returned None (sandbox timeout) — retry fast_audit ----
            trace.append(
                "【审计】审计未产出有效报告（执行超时），自动切换 fast_audit 重试..."
            )
            _retry_succeeded = False
            try:
                # Bump timeout and enable fast_audit
                critic_expert.sandbox.timeout_sec = max(_audit_timeout_sec * 2, 90)
                _fast_settings = settings
                if hasattr(settings, "model_copy"):
                    _fast_settings = settings.model_copy(update={"fast_audit": True})
                else:
                    from dataclasses import replace
                    _fast_settings = replace(settings, fast_audit=True)
                _fast_audit = critic_expert.execute_audit(
                    winner, dataset, _fast_settings, modality=modality,
                )
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
                # ---- Phase 6: auto-relaxed tier — last resort ----
                _relaxed_timeout = max(_audit_timeout_sec * 2, 90)
                trace.append(
                    "【审计】fast_audit 重试失败，触发自适应宽松模式："
                    f"超时 {_audit_timeout_sec}s → {_relaxed_timeout}s，"
                    "最大采样 500 样本。"
                )
                try:
                    critic_expert.sandbox.timeout_sec = _relaxed_timeout
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
                    _relaxed_audit = critic_expert.execute_audit(
                        winner, dataset, _relaxed_settings, modality=modality,
                    )
                    if _relaxed_audit and not _relaxed_audit.get("degraded"):
                        _relaxed_audit["fast_audit_retry"] = True
                        _relaxed_audit["audit_relaxed"] = True
                        audit = _relaxed_audit
                        _retry_succeeded = True
                        trace.append(
                            f"【审计】自适应宽松重试成功 — "
                            f"裁决: {audit.get('endorsement', '?')},"
                            f" 置信度: {audit.get('confidence_level', '?')}"
                        )
                except Exception as _relaxed_exc:
                    trace.append(f"【审计】自适应宽松重试异常 ({_relaxed_exc})。")

            if not _retry_succeeded:
                critic_diag = (
                    "; ".join(critic_expert.last_logs[-2:])
                    if critic_expert.last_logs
                    else "（无诊断信息）"
                )
                trace.append(
                    f"【审计】自适应宽松重试仍失败，降级为初级审计建议。"
                    f"{critic_diag}"
                )
                audit = {
                    "endorsement": "qualified",
                    "action": "WARN",
                    "confidence_level": 0.3,
                    "findings": ["审计超时，无法完成全量分析。建议人工检查聚类质量。"],
                    "recommendation": (
                        "审计执行超时（timeout=" + str(_audit_timeout_sec)
                        + "s），请考虑启用 fast_audit 模式或减小数据集。"
                    ),
                    "degraded": True,
                    "stability_score": -1,
                    "hopkins": -1,
                }
    finally:
        # Restore sandbox settings
        try:
            critic_expert.sandbox.timeout_sec = _prev_sandbox_timeout
            critic_expert.sandbox.memory_mb = _prev_sandbox_memory
        except Exception:
            pass

    return audit


# ---------------------------------------------------------------------------
# Critic 2.0 feedback loop: RETRY with constraints
# ---------------------------------------------------------------------------

def handle_audit_feedback(
    audit_report: dict | None,
    dataset: DatasetBundle,
    prompt: str,
    settings: LLMSettings,
    trace: list[str],
    active_experts: list[str],
    experts: dict[str, Any],
    all_results: list[AlgorithmRunResult] | None = None,
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
    # ---- Structural failure pre-check: skip RETRY if max ARI < 0.2 ----
    if all_results and dataset.y is not None:
        import numpy as _np_retry
        from sklearn.metrics import adjusted_rand_score as _ari_retry
        _y_true_retry = _np_retry.asarray(dataset.y, dtype=int).ravel()
        _max_ari_retry = 0.0
        for _r in all_results:
            _rl = getattr(_r, "labels", None)
            if _rl is not None and hasattr(_rl, "__len__") and len(_rl) > 0:
                with contextlib.suppress(Exception):
                    _a = float(_ari_retry(
                        _y_true_retry, _np_retry.asarray(_rl, dtype=int).ravel(),
                    ))
                    if _a > _max_ari_retry:
                        _max_ari_retry = _a
        if _max_ari_retry < 0.2:
            trace.append(
                f"【诚实失败快速出口】ARI < 0.2（max={_max_ari_retry:.3f}），"
                f"数据结构无法通过欧氏空间方法捕捉。"
                f"结构性问题无法通过约束重试修复，跳过 RETRY。"
            )
            return []
        # Upper-bound gate: when the best result already achieves ARI ≥ 0.7,
        # RETRY can only make things worse (blocking the winner + re-ranking
        # among weaker algorithms degrades the final outcome).
        if _max_ari_retry >= 0.7:
            trace.append(
                f"【RETRY 门禁】最佳 ARI={_max_ari_retry:.3f} ≥ 0.7，"
                f"当前结果已充分捕捉数据结构，跳过 RETRY（避免封锁优胜算法导致退化）。"
            )
            return []

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
            expert = experts.get(key)
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


# ---------------------------------------------------------------------------
# LLM-3 Reflection summary generation
# ---------------------------------------------------------------------------

def generate_llm_summary(
    best: AlgorithmRunResult,
    dataset: DatasetBundle,
    prompt: str,
    all_results: list[AlgorithmRunResult],
    best_ari: float | None,
    llm_config: MultiLLMConfig | None,
    fallback_settings: LLMSettings,
) -> str | None:
    """Generate a Chinese-language clustering summary via LLM-3 (Reflection)."""
    if llm_config is not None:
        reflection_settings = llm_config.get_reflection()
    else:
        reflection_settings = fallback_settings
    client = UniversalLLMClient(reflection_settings)
    summary = client.summarize_report(
        {
            "user_intent": prompt,
            "dataset": dataset.display_name,
            "best_algo": best.algorithm_name,
            "metrics": best.metrics,
            "score_source": best.metrics.get("score_source", "silhouette"),
            "best_ari": best_ari,
            "all_algorithms_failed": best_ari is not None and best_ari < 0.2,
            "all_results": [
                {
                    "algo": r.algorithm_name,
                    "score": r.metrics.get("score") or 0.0,
                    "score_source": r.metrics.get("score_source", "silhouette"),
                }
                for r in all_results
            ],
        }
    )
    return summary


# ---------------------------------------------------------------------------
# Final report assembly
# ---------------------------------------------------------------------------

def assemble_final_report(
    dataset: DatasetBundle,
    output_dir: Path,
    all_results: list[AlgorithmRunResult],
    ranking: list[AlgorithmRunResult],
    summary: str | None,
    audit_report: dict[str, Any] | None,
    trace: list[str],
    modality: ModalityProfile,
    dataset_plot_path: Path,
    response_type: str = "CLUSTER_TASK",
) -> SupervisorReport:
    """Build the SupervisorReport from post-ranking outputs."""
    report = SupervisorReport(
        dataset=dataset,
        routing=RoutingDecision(
            profile=ProfileReport(
                dataset.X.shape[0], dataset.X.shape[1],
                0, 0, 0, False, False, False,
                modality_type=modality.modality_type,
                modality_metric=modality.distance_metric,
            ),
            selected_experts=[],
            trace=trace,
            modality=modality,
        ),
        dataset_plot_path=dataset_plot_path,
        output_dir=output_dir,
        results=all_results,
        ranking=ranking,
        executive_summary=summary or "聚类分析完成。",
        decision_trace=trace,
        audit_report=audit_report,
        response_type=response_type,
    )
    return report


# ---------------------------------------------------------------------------
# Error / fallback report
# ---------------------------------------------------------------------------

def build_error_report(
    msg: str,
    trace: list[str],
    expert_logs: dict[str, list[str]] | None = None,
) -> SupervisorReport:
    """Construct an error report with per-expert log diagnostics."""
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
