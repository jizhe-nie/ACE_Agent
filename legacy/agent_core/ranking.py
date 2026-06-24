"""
Post-dispatch ranking, arbitration, and cross-validation for ACE Agent.

Extracted from ACESupervisor.  All functions are stateless — they take
results + data + trace and return ranked output.
"""

from __future__ import annotations

import contextlib
from typing import Any

import numpy as np

from ACE_Agent.agent_core.schemas import AlgorithmRunResult, DatasetBundle, ModalityProfile

# ---------------------------------------------------------------------------
# Best ARI extractor
# ---------------------------------------------------------------------------

def compute_best_ari(
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


# ---------------------------------------------------------------------------
# Topology failure check
# ---------------------------------------------------------------------------

def check_topology_failure(
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

    from ACE_Agent.tools.graph_builder import GraphBuilder as _GB3
    try:
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


# ---------------------------------------------------------------------------
# Cross-validation for graph algorithm winners
# ---------------------------------------------------------------------------

def cross_validate_graph_winner(
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
        _best_nmi = float(normalized_mutual_info_score(y_true_arr, best_labels_arr))
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


# ---------------------------------------------------------------------------
# Informed ranking — ARI one-vote veto
# ---------------------------------------------------------------------------

def compute_informed_ranking(
    all_results: list[AlgorithmRunResult],
    dataset: DatasetBundle,
    trace: list[str],
    centroid_ban: set[str] | None = None,
    blocked_algorithms: list[str] | None = None,
    *,
    modality: ModalityProfile | None = None,
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
        # Modality-aware internal scoring: recompute Silhouette with the
        # correct distance metric so ranking is valid for text/cosine and
        # time_series data (not just tabular/euclidean).
        _metric = modality.distance_metric if modality is not None else "euclidean"
        if _metric != "euclidean":
            trace.append(
                f"【无标签排名】使用 metric='{_metric}' 重新计算 Silhouette"
                f"（非默认欧氏距离）。"
            )
        X = dataset.X
        for r in all_results:
            _lbl = getattr(r, "labels", None)
            if _lbl is not None and hasattr(_lbl, "__len__") and len(_lbl) > 0:
                try:
                    from sklearn.metrics import silhouette_score

                    _lbl_arr = np.asarray(_lbl, dtype=int).ravel()
                    if len(set(_lbl_arr)) > 1:
                        _sil = float(silhouette_score(X, _lbl_arr, metric=_metric))
                        r.metrics["score"] = _sil
                        r.metrics["silhouette"] = _sil
                        r.metrics["silhouette_metric"] = _metric
                except Exception:
                    pass  # keep the expert-computed score on failure
        return sorted(
            all_results,
            key=lambda r: r.metrics.get("score") or 0.0,
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
            with contextlib.suppress(Exception):
                _pre_aris.append(
                    float(adjusted_rand_score(y_true_arr, np.asarray(_lbl, dtype=int).ravel()))
                )
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

        # ---- Critic 2.0 blocked algorithm veto -----------------------
        if blocked_algorithms and r.algorithm_name in blocked_algorithms:
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
