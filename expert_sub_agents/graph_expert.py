"""
expert_sub_agents/graph_expert.py
==================================
Graph Community Discovery Expert — the PRIMARY pipeline for graph-connected data.

Phase 3.2 (2026-05): Complete rewrite.
- Replaced Euclidean clustering skeleton with native graph community discovery.
- All algorithms operate on adjacency matrices, NOT coordinates.
- Best partition selected by modularity, NOT silhouette.
- Cluster count determined by graph structure, NOT pre-defined k.

Community detection algorithms (primary):
  Louvain, MCL (Markov Cluster Algorithm), Label Propagation,
  Spectral Graph Partition, Random Walk / Diffusion, Leiden/Infomap (if available)

Scoring: edge-cut objectives (normalized cut, modularity, conductance, CPS)
"""

from __future__ import annotations

import json
import re
from typing import Any

from ACE_Agent.agent_core.schemas import DatasetBundle
from ACE_Agent.expert_sub_agents.base import BaseExpert
from ACE_Agent.tools.llm_client import UniversalLLMClient

# ===========================================================================
# Phase 3.2 Skeleton — Native Graph Community Discovery
# ===========================================================================
_SKELETON = r"""# ===== ACE Graph Community Discovery Skeleton (Phase 3.2) =====
# ALL clustering is native graph community detection on the kNN adjacency.
# Coordinates are used ONLY to build the graph, NOT to cluster.
# Modules (pre-injected by sandbox): StandardScaler, kneighbors_graph,
#   csgraph, sparse, numpy (as _np)

import numpy as _np

# ---- scale -------------------------------------------------------------
_scaler = StandardScaler()
_X = _scaler.fit_transform(CTX_DATA.X)
_n = CTX_DATA.n_samples
_d = CTX_DATA.n_features

# ---- decisions from LLM -------------------------------------------------
_DECISIONS = {DECISIONS_JSON}

# ========================================================================
# WALL-AWARE GRAPH CONSTRUCTION (Phase 4)
# ========================================================================
# The graph construction quality determines the upper bound of community
# detection accuracy.  A single wall-crossing edge can merge two communities.
#
# Pipeline:
#   1. Mutual kNN with adaptive local scaling
#   2. Shared-neighbor Jaccard pruning (removes wall-crossing edges)
#   3. Shortcut edge detection
#   4. Graph quality audit

_knn = int(_DECISIONS.get("knn_k", min(30, max(5, int(_np.sqrt(_n))))))
_mutual = bool(_DECISIONS.get("mutual", True))
_wall_aware = bool(_DECISIONS.get("wall_aware", True))
_mode = str(_DECISIONS.get("graph_mode", "distance"))

# ---- Stage 1: Initial mutual kNN ----
_adj = kneighbors_graph(_X, min(_knn, _n - 1), mode=_mode, include_self=False)
if _mutual:
    _adj = _adj.minimum(_adj.T)
_adj = _adj.tocsr()

# ---- Stage 2: Adaptive local scaling weights ----
if _wall_aware:
    try:
        # Compute per-point bandwidth: distance to k-th neighbor
        from sklearn.neighbors import NearestNeighbors as _NN
        _nn_model = _NN(n_neighbors=min(_knn + 1, _n), metric='euclidean')
        _nn_model.fit(_X)
        _dists, _ = _nn_model.kneighbors(_X)
        _sigma = _dists[:, -1].copy()
        _sigma[_sigma < 1e-8] = 1e-8
        # Reweight edges: w_ij = exp(-d_ij^2 / (sigma_i * sigma_j))
        for _i in range(min(_n, 5000)):
            _row = _adj[_i]
            for _jj in range(len(_row.indices)):
                _j = _row.indices[_jj]
                _d = _row.data[_jj]
                _w = _np.exp(-_d * _d / (_sigma[_i] * _sigma[_j]))
                _adj[_i, _j] = _w
        _adj.eliminate_zeros()
    except Exception:
        pass

# ---- Stage 3: Shared-neighbor Jaccard pruning ----
_jaccard_threshold = float(_DECISIONS.get("jaccard_threshold", 0.05))
_shortcuts_pruned = 0
if _wall_aware and _n <= 10000:
    try:
        # Build neighbor sets
        _adj_bin = _adj.copy()
        _adj_bin.data = _np.ones_like(_adj_bin.data)
        _nb_sets = [set(_adj_bin[_i].indices.tolist()) for _i in range(_n)]
        # Prune low-Jaccard edges
        from scipy.sparse import lil_matrix as _lil
        _adj_new = _lil((_n, _n), dtype=float)
        for _i in range(_n):
            _row = _adj[_i]
            _ni = _nb_sets[_i]
            if not _ni:
                continue
            for _j in _row.indices:
                if _i >= _j:
                    continue
                _nj = _nb_sets[_j]
                if not _nj:
                    continue
                _inter = len(_ni & _nj)
                _union = len(_ni | _nj)
                _jac = _inter / max(_union, 1)
                if _jac >= _jaccard_threshold:
                    _w = _adj[_i, _j] * _jac  # weight × Jaccard
                    _adj_new[_i, _j] = _w
                    _adj_new[_j, _i] = _w
                else:
                    _shortcuts_pruned += 1
        _adj = _adj_new.tocsr()
    except Exception:
        pass

# ---- Stage 4: Graph quality audit ----
_graph_quality_pass = True
_shortcut_ratio = 0.0
if _wall_aware:
    _total_edges = _adj.nnz
    if _total_edges > 0:
        _shortcut_ratio = _shortcuts_pruned / max(_total_edges + _shortcuts_pruned, 1)
    _graph_quality_pass = _shortcut_ratio < 0.20  # tolerate up to 20% suspect edges
    if not _graph_quality_pass:
        artifacts["_graph_quality_warning"] = {
            "labels": [],
            "metrics": {
                "score": 0.0,
                "warning": f"Graph quality FAILED: shortcut_ratio={_shortcut_ratio:.2%}",
                "shortcut_ratio": float(_shortcut_ratio),
                "edges_pruned": int(_shortcuts_pruned),
            },
            "plot_path": "",
        }

_adj.data[_adj.data < 0] = 0
_adj.eliminate_zeros()

# ---- geodesic distortion ------------------------------------------------
_geodesic_distortion = 0.0
if _n <= 2000:
    try:
        _geo_dists = csgraph.shortest_path(_adj, method="auto", directed=False)
        _sample = min(_n, 500)
        _s_idx = _np.random.RandomState(42).choice(_n, _sample, replace=False)
        _ratios = []
        for _i in _s_idx:
            for _j in _s_idx:
                if _i >= _j:
                    continue
                _de = float(_np.linalg.norm(_X[_i] - _X[_j]))
                _dg = float(_geo_dists[_i, _j])
                if _de > 1e-8 and _np.isfinite(_dg):
                    _ratios.append(abs(_dg - _de) / _de)
        if _ratios:
            _geodesic_distortion = float(_np.median(_ratios))
    except Exception:
        pass

# ========================================================================
# COMMUNITY DISCOVERY PIPELINES
# ========================================================================
# Each pipeline runs a native graph community detection algorithm.
# Output = community partition labels (NOT Euclidean clusters).
# Score = edge-cut objectives (normalized cut, modularity, conductance).
# ========================================================================

_community_results = []
_modularity_results = {}

def _modularity(_adj_matrix, _labels):
    # Newman-Girvan modularity
    _lbl = _np.asarray(_labels, dtype=int).ravel()
    _m2 = _adj_matrix.sum()
    if _m2 == 0:
        return 0.0
    _deg = _np.asarray(_adj_matrix.sum(axis=1)).ravel()
    _q = 0.0
    for _c in set(_lbl.tolist()):
        _mask = _lbl == _c
        _w_in = _adj_matrix[_mask, :][:, _mask].sum()
        _d_c = _deg[_mask].sum()
        _q += (_w_in - _d_c * _d_c / _m2) / _m2
    return float(_q)

def _conductance(_adj_matrix, _labels):
    # Mean conductance across clusters
    _lbl = _np.asarray(_labels, dtype=int).ravel()
    _uniq = list(set(_lbl.tolist()))
    if len(_uniq) < 2:
        return 0.0
    _conds = []
    for _c in _uniq:
        _mask_in = _lbl == _c
        _mask_out = ~_mask_in
        _cut = _adj_matrix[_mask_in, :][:, _mask_out].sum()
        _vol_in = _adj_matrix[_mask_in, :].sum()
        _vol_out = _adj_matrix[_mask_out, :].sum()
        _denom = min(_vol_in, _vol_out)
        if _denom > 0:
            _conds.append(_cut / _denom)
    return float(_np.mean(_conds)) if _conds else 1.0

def _normalized_cut(_adj_matrix, _labels):
    # Normalized cut: sum( cut(C,V\C) / vol(C) )
    _lbl = _np.asarray(_labels, dtype=int).ravel()
    _uniq = list(set(_lbl.tolist()))
    if len(_uniq) < 2:
        return 0.0
    _total = 0.0
    for _c in _uniq:
        _mask_in = _lbl == _c
        _mask_out = ~_mask_in
        _cut = _adj_matrix[_mask_in, :][:, _mask_out].sum()
        _vol = _adj_matrix[_mask_in, :].sum()
        if _vol > 0:
            _total += _cut / _vol
    return float(_total / len(_uniq))

def _edge_cut_score(_adj_matrix, _labels):
    # Primary score: composite of edge-cut objectives
    _nc = _normalized_cut(_adj_matrix, _labels)
    _mod = _modularity(_adj_matrix, _labels)
    _cond = _conductance(_adj_matrix, _labels)
    # Invert n-cut and conductance (lower is better) into a positive score
    _score = (1.0 / max(_nc, 0.001)) * 0.3 + _mod * 0.4 + (1.0 - _cond) * 0.3
    return {
        "score": float(_score),
        "score_source": "edge_cut",
        "normalized_cut": float(_nc),
        "modularity": float(_mod),
        "conductance": float(_cond),
    }

def _compute_n_preservation(_adj_matrix, _labels):
    # Fraction of kNN edges that stay within the same cluster
    _lbl = _np.asarray(_labels, dtype=int).ravel()
    _adj_csr = _adj_matrix.tocsr()
    _same = 0
    _total = 0
    for _i in range(min(_adj_csr.shape[0], 5000)):
        _row = _adj_csr[_i]
        for _j in _row.indices:
            if _lbl[_i] == _lbl[_j]:
                _same += 1
            _total += 1
    return float(_same / max(_total, 1))

# =====================================================================
# PIPELINE 1: Louvain (greedy modularity optimisation)
# =====================================================================
_d_louvain = _DECISIONS.get("algorithms", {}).get("louvain", {})
if _d_louvain.get("active", True):
    try:
        # ---- deterministic Louvain implementation ----
        _n_lv = _n
        _A_lv = _adj.copy().tocsr()
        _A_lv = _A_lv + _A_lv.T
        _A_lv.data *= 0.5
        _m2_lv = _A_lv.sum() * 2
        _labels_lv = _np.arange(_n_lv, dtype=int)
        if _m2_lv > 0:
            _rng = _np.random.RandomState(42)
            for _pass_lv in range(20):
                _improved = False
                _order = _rng.permutation(_n_lv)
                for _v in _order:
                    _v_comm = _labels_lv[_v]
                    _row = _A_lv[_v]
                    _k_v = _row.sum()
                    _mask_cur = _labels_lv == _v_comm
                    _d_cur_no_v = _A_lv[_mask_cur, :].sum() - _k_v
                    _k_v_in_cur = _A_lv[_v, _mask_cur].sum()
                    _gain_remove = -(_k_v_in_cur - _k_v * _d_cur_no_v / _m2_lv) * 2 / _m2_lv
                    _best_gain = 0.0
                    _best_comm = _v_comm
                    _seen = {_v_comm}
                    for _nb in _row.indices:
                        _c = int(_labels_lv[_nb])
                        if _c in _seen:
                            continue
                        _seen.add(_c)
                        _mask_c = _labels_lv == _c
                        _d_c = _A_lv[_mask_c, :].sum()
                        _k_v_in_c = _A_lv[_v, _mask_c].sum()
                        _delta = _k_v_in_c - _k_v * _d_c / _m2_lv
                        _gain = _gain_remove + _delta * 2 / _m2_lv
                        if _gain > _best_gain:
                            _best_gain = _gain
                            _best_comm = _c
                    if _best_comm != _v_comm:
                        _labels_lv[_v] = _best_comm
                        _improved = True
                if not _improved:
                    break
        _uniq_lv = list(set(_labels_lv.tolist()))
        _map_lv = {_o: _i for _i, _o in enumerate(_uniq_lv)}
        _lbl_lv = _np.array([_map_lv[l] for l in _labels_lv], dtype=int)
        _n_comm_lv = len(_uniq_lv)
        if _n_comm_lv >= 2:
            _metrics_lv = _edge_cut_score(_adj, _lbl_lv)
            _metrics_lv["k"] = _n_comm_lv
            _metrics_lv["method"] = "louvain"
            _metrics_lv["n_preservation"] = _compute_n_preservation(_adj, _lbl_lv)
            artifacts["Louvain"] = {
                "labels": _lbl_lv.tolist(),
                "metrics": _metrics_lv,
                "plot_path": "",
            }
            _community_results.append(("louvain", _lbl_lv, _metrics_lv["score"]))
            _modularity_results["louvain"] = _metrics_lv["modularity"]
    except Exception as _e_lv:
        artifacts["Louvain_error"] = {
            "labels": [], "metrics": {"score": 0.0, "error": str(_e_lv)}, "plot_path": ""}

# =====================================================================
# PIPELINE 2: MCL (Markov Cluster Algorithm)
# =====================================================================
_d_mcl = _DECISIONS.get("algorithms", {}).get("mcl", {})
if _d_mcl.get("active", True):
    try:
        _inflate = float(_d_mcl.get("inflate_factor", 2.0))
        _expand = int(_d_mcl.get("expand_factor", 2))
        # Build transition matrix
        _A_mcl = _adj.copy().tocsr().astype(float)
        _A_mcl = _A_mcl + sparse.eye(_n, dtype=float)
        _col_sum = _np.asarray(_A_mcl.sum(axis=0)).ravel()
        _col_sum[_col_sum == 0] = 1.0
        from scipy.sparse import diags as _diags
        _M = _A_mcl @ _diags(1.0 / _col_sum)
        _M = _M.tocsc()
        for _iter_mcl in range(100):
            _prev_nnz = _M.nnz
            _M = (_M ** _expand).tocsc()
            _M.data = _M.data ** _inflate
            _cs = _np.asarray(_M.sum(axis=0)).ravel()
            _cs[_cs == 0] = 1.0
            _M = _M @ _diags(1.0 / _cs)
            _M.data[_np.abs(_M.data) < 1e-6] = 0
            _M.eliminate_zeros()
            if _M.nnz == _prev_nnz:
                break
        # Connected components of M = communities
        _lbl_mcl = _np.full(_n, -1, dtype=int)
        _curr_label = 0
        for _start in range(_n):
            if _lbl_mcl[_start] != -1:
                continue
            _stack = [_start]
            _lbl_mcl[_start] = _curr_label
            while _stack:
                _v = _stack.pop()
                _row_mcl = _M.getrow(_v)
                if hasattr(_row_mcl, 'indices'):
                    for _nb in _row_mcl.indices:
                        if _lbl_mcl[_nb] == -1:
                            _lbl_mcl[_nb] = _curr_label
                            _stack.append(_nb)
            _curr_label += 1
        _n_mcl = len(set(_lbl_mcl.tolist()))
        if _n_mcl >= 2:
            _metrics_mcl = _edge_cut_score(_adj, _lbl_mcl)
            _metrics_mcl["k"] = _n_mcl
            _metrics_mcl["method"] = "mcl"
            _metrics_mcl["n_preservation"] = _compute_n_preservation(_adj, _lbl_mcl)
            artifacts["MCL"] = {
                "labels": _lbl_mcl.tolist(),
                "metrics": _metrics_mcl,
                "plot_path": "",
            }
            _community_results.append(("mcl", _lbl_mcl, _metrics_mcl["score"]))
            _modularity_results["mcl"] = _metrics_mcl["modularity"]
    except Exception as _e_mcl:
        artifacts["MCL_error"] = {
            "labels": [], "metrics": {"score": 0.0, "error": str(_e_mcl)}, "plot_path": ""}

# =====================================================================
# PIPELINE 3: Label Propagation
# =====================================================================
_d_lp = _DECISIONS.get("algorithms", {}).get("label_propagation", {})
if _d_lp.get("active", True):
    try:
        _lbl_lp = _np.arange(_n, dtype=int)
        _rng_lp = _np.random.RandomState(42)
        _A_lp = _adj.tocsr()
        for _iter_lp in range(100):
            _changed = 0
            _order_lp = _rng_lp.permutation(_n)
            for _v in _order_lp:
                _row = _A_lp[_v]
                if len(_row.indices) == 0:
                    continue
                _nb_labels = _lbl_lp[_row.indices]
                _counts = _np.bincount(_nb_labels, minlength=_n)
                _max_c = _counts.max()
                _best = _np.where(_counts == _max_c)[0]
                _new_label = _best[_rng_lp.randint(len(_best))]
                if _new_label != _lbl_lp[_v]:
                    _lbl_lp[_v] = _new_label
                    _changed += 1
            if _changed == 0:
                break
        _uniq_lp, _lbl_lp2 = _np.unique(_lbl_lp, return_inverse=True)
        _n_lp = len(_uniq_lp)
        if _n_lp >= 2:
            _metrics_lp = _edge_cut_score(_adj, _lbl_lp2)
            _metrics_lp["k"] = _n_lp
            _metrics_lp["method"] = "label_propagation"
            _metrics_lp["n_preservation"] = _compute_n_preservation(_adj, _lbl_lp2)
            artifacts["LabelPropagation"] = {
                "labels": _lbl_lp2.tolist(),
                "metrics": _metrics_lp,
                "plot_path": "",
            }
            _community_results.append(("label_propagation", _lbl_lp2, _metrics_lp["score"]))
            _modularity_results["label_propagation"] = _metrics_lp["modularity"]
    except Exception as _e_lp:
        artifacts["LabelPropagation_error"] = {
            "labels": [], "metrics": {"score": 0.0, "error": str(_e_lp)}, "plot_path": ""}

# =====================================================================
# PIPELINE 4: Spectral Graph Partition (normalized cut)
# =====================================================================
_d_sp = _DECISIONS.get("algorithms", {}).get("spectral", {})
if _d_sp.get("active", True):
    try:
        from scipy.sparse.linalg import eigsh as _eigsh
        _k_sp = max(2, min(12, int(_np.sqrt(_n) // 2)))
        _L_sp = csgraph.laplacian(_adj, normed=False)
        try:
            _vals_sp, _vecs_sp = _eigsh(_L_sp, k=_k_sp + 1, which="SM", tol=1e-6, maxiter=200)
        except Exception:
            _L_norm = csgraph.laplacian(_adj, normed=True)
            _vals_sp, _vecs_sp = _eigsh(_L_norm, k=_k_sp + 1, which="SM", tol=1e-6, maxiter=200)
        _idx_sp = _np.argsort(_vals_sp)
        _emb_sp = _vecs_sp[:, _idx_sp[1:_k_sp + 1]]
        _norms_sp = _np.linalg.norm(_emb_sp, axis=1, keepdims=True)
        _norms_sp[_norms_sp < 1e-10] = 1e-10
        _emb_sp = _emb_sp / _norms_sp
        from sklearn.cluster import KMeans as _KMeans
        # Modularity-optimal k sweep for spectral
        _best_k_sp = _k_sp
        _best_mod_sp = -999.0
        _best_lbl_sp = None
        for _k_try in range(2, _k_sp + 1):
            try:
                _lbl_try = _KMeans(n_clusters=_k_try, n_init=10, random_state=42).fit_predict(_emb_sp)
                _mod_try = _modularity(_adj, _lbl_try)
                if _mod_try > _best_mod_sp:
                    _best_mod_sp = _mod_try
                    _best_k_sp = _k_try
                    _best_lbl_sp = _lbl_try
            except Exception:
                continue
        if _best_lbl_sp is not None and _best_k_sp >= 2:
            _metrics_sp = _edge_cut_score(_adj, _best_lbl_sp)
            _metrics_sp["k"] = _best_k_sp
            _metrics_sp["method"] = "spectral_graph_partition"
            _metrics_sp["n_preservation"] = _compute_n_preservation(_adj, _best_lbl_sp)
            artifacts["SpectralGraphPartition"] = {
                "labels": _best_lbl_sp.tolist(),
                "metrics": _metrics_sp,
                "plot_path": "",
            }
            _community_results.append(("spectral", _best_lbl_sp, _metrics_sp["score"]))
            _modularity_results["spectral"] = _metrics_sp["modularity"]
    except Exception as _e_sp:
        artifacts["SpectralGraphPartition_error"] = {
            "labels": [], "metrics": {"score": 0.0, "error": str(_e_sp)}, "plot_path": ""}

# =====================================================================
# PIPELINE 5: Random Walk / Diffusion Community Detection
# =====================================================================
_d_rw = _DECISIONS.get("algorithms", {}).get("random_walk", {})
if _d_rw.get("active", _n <= 5000):  # auto-active for manageable graphs
    try:
        _deg_rw = _np.asarray(_adj.sum(axis=1)).ravel()
        _deg_rw[_deg_rw == 0] = 1.0
        from scipy.sparse import diags as _diags_rw
        _P_rw = _diags_rw(1.0 / _deg_rw) @ _adj
        # Symmetric normalized Laplacian for diffusion embedding
        _D_sqrt_inv = _diags_rw(1.0 / _np.sqrt(_deg_rw))
        _L_sym = sparse.eye(_n) - _D_sqrt_inv @ _adj @ _D_sqrt_inv
        _k_eig_rw = min(_n - 1, 30)
        try:
            _vals_rw, _vecs_rw = _eigsh(_L_sym, k=_k_eig_rw, which='SM', tol=1e-4)
        except Exception:
            _vals_rw, _vecs_rw = _eigsh(_L_sym, k=min(_k_eig_rw, _n - 2), which='SM')
        _emb_rw = _vecs_rw[:, :min(16, _vecs_rw.shape[1])]
        # Sweep k via modularity
        _best_k_rw = 3
        _best_mod_rw = -999.0
        _best_lbl_rw = None
        for _k_try_rw in range(2, min(12, _n // 5 + 2)):
            try:
                _lbl_try = _KMeans(n_clusters=_k_try_rw, n_init=5, random_state=42).fit_predict(_emb_rw)
                _mod_try = _modularity(_adj, _lbl_try)
                if _mod_try > _best_mod_rw:
                    _best_mod_rw = _mod_try
                    _best_k_rw = _k_try_rw
                    _best_lbl_rw = _lbl_try
            except Exception:
                continue
        if _best_lbl_rw is not None and _best_k_rw >= 2:
            _metrics_rw = _edge_cut_score(_adj, _best_lbl_rw)
            _metrics_rw["k"] = _best_k_rw
            _metrics_rw["method"] = "random_walk_diffusion"
            _metrics_rw["n_preservation"] = _compute_n_preservation(_adj, _best_lbl_rw)
            artifacts["RandomWalkDiffusion"] = {
                "labels": _best_lbl_rw.tolist(),
                "metrics": _metrics_rw,
                "plot_path": "",
            }
            _community_results.append(("random_walk", _best_lbl_rw, _metrics_rw["score"]))
            _modularity_results["random_walk"] = _metrics_rw["modularity"]
    except Exception as _e_rw:
        artifacts["RandomWalkDiffusion_error"] = {
            "labels": [], "metrics": {"score": 0.0, "error": str(_e_rw)}, "plot_path": ""}

# =====================================================================
# SELECT BEST COMMUNITY PARTITION
# =====================================================================
# The BEST partition is selected by modularity, NOT silhouette.
# This is the key difference from Euclidean clustering.
_best_method = "fallback"
_best_labels = _np.zeros(_n, dtype=int)
_best_modularity = -999.0

if _community_results:
    for _bm, _bl, _bs in _community_results:
        _bm_q = _modularity_results.get(_bm, 0.0)
        if _bm_q > _best_modularity:
            _best_modularity = _bm_q
            _best_method = _bm
            _best_labels = _np.asarray(_bl, dtype=int).ravel()

# =====================================================================
# OUTPUT: Best community partition
# =====================================================================
_n_communities = len(set(_best_labels.tolist()))
_edge_cut = _edge_cut_score(_adj, _best_labels)

# ---- visualization: scatter plot of best community partition -------------
_plot_path = ""
try:
    _fig, _ax = plt.subplots(figsize=(8, 6))
    # Reduce to 2D if high-dimensional
    if _X.shape[1] > 2:
        try:
            from sklearn.decomposition import PCA as _PCA_vis
            _X_vis = _PCA_vis(n_components=2, random_state=42).fit_transform(_X)
        except Exception:
            _X_vis = _X[:, :2]
    else:
        _X_vis = _X
    # Sample for speed when many points
    if _n > 1000:
        _rng_v = _np.random.RandomState(42)
        _samp_idx = _rng_v.choice(_n, 1000, replace=False)
        _X_plt = _X_vis[_samp_idx]
        _L_plt = _best_labels[_samp_idx]
    else:
        _X_plt = _X_vis
        _L_plt = _best_labels
    _uniq = sorted(set(int(_l) for _l in _L_plt))
    _cmap = plt.cm.get_cmap("tab20", len(_uniq)) if len(_uniq) <= 20 else plt.cm.get_cmap("viridis", len(_uniq))
    for _cidx, _c in enumerate(_uniq):
        _mask = _L_plt == _c
        _ax.scatter(_X_plt[_mask, 0], _X_plt[_mask, 1],
                    s=16, alpha=0.75, label=f"C{_c}",
                    color=_cmap(_cidx), edgecolors="none")
    _ax.set_title(f"Graph Community: {_best_method} ({_n_communities} communities)")
    _ax.legend(loc="upper right", fontsize=6, ncol=2, markerscale=0.6)
    _plot_dir = ACE_OUTPUT_DIR + "/graph" if ACE_OUTPUT_DIR else "outputs/graph"
    _os.makedirs(_plot_dir, exist_ok=True)
    _plot_path = f"{_plot_dir}/graph_community_result.png"
    _fig.savefig(_plot_path, dpi=150, bbox_inches="tight")
    plt.close(_fig)
except Exception:
    pass

artifacts["GraphCommunity_Result"] = {
    "labels": _best_labels.tolist(),
    "metrics": {
        "score": float(_edge_cut["score"]),
        "score_source": "edge_cut",
        "best_method": _best_method,
        "n_communities": _n_communities,
        "modularity": float(_best_modularity),
        "normalized_cut": float(_edge_cut["normalized_cut"]),
        "conductance": float(_edge_cut["conductance"]),
        "n_preservation": _compute_n_preservation(_adj, _best_labels),
        "all_modularities": {str(k): float(v) for k, v in _modularity_results.items()},
    },
    "plot_path": _plot_path,
}

# ---- store meta for downstream ensemble / audit ---------------------------
artifacts["_graph_meta"] = {
    "labels": [],
    "metrics": {
        "score": 0.0,
        "geodesic_distortion": float(_geodesic_distortion),
        "knn_k": int(_knn),
        "mutual": bool(_mutual),
        "wall_aware": bool(_wall_aware),
        "community_mode": True,
        "best_community_method": _best_method,
        "n_communities": _n_communities,
        "graph_quality_pass": bool(_graph_quality_pass),
        "shortcut_ratio": float(_shortcut_ratio),
        "shortcuts_pruned": int(_shortcuts_pruned),
        "all_modularities": {str(k): float(v) for k, v in _modularity_results.items()},
    },
    "plot_path": "",
}
"""

# ===========================================================================
# LLM decision prompt — graph community discovery params
# ===========================================================================
_DECISION_SYSTEM_PROMPT = (
    "你是一个图结构社区发现参数决策专家。\n\n"
    "你需要为给定的数据集选择图构建参数和社区发现管线。\n"
    "输出**纯 JSON**，不要 Markdown，不要解释文字。\n\n"
    "## 可用管线（图社区发现 — 不依赖欧氏坐标）\n"
    "1. `louvain`            — Louvain 贪婪模块度最大化\n"
    "2. `mcl`                — Markov Cluster Algorithm (流扩散+膨胀)\n"
    "3. `label_propagation`  — 标签传播社区检测\n"
    "4. `spectral`           — 谱图分割 (normalized cut relaxation)\n"
    "5. `random_walk`        — 随机游走扩散距离聚类\n\n"
    "## 决策指南\n"
    "- knn_k 自动计算规则:\n"
    "  * n <= 100: k = max(5, n//10)\n"
    "  * 100 < n <= 2000: k = int(sqrt(n))\n"
    "  * 2000 < n <= 20000: k = min(30, max(10, int(sqrt(n)/2)))\n"
    "  * n > 20000: k = 15\n"
    "- mutual: 强制 mutual kNN (无单向边)，非凸流形/迷宫数据必须 True\n"
    "- wall_aware: 是否启用墙感知图构建（共享邻居+局部缩放+Jaccard剪枝），迷宫数据必须 True\n"
    "- jaccard_threshold: Jaccard 相似度阈值，低于此值的边被剪枝（默认 0.05）\n"
    "- graph_mode: \"distance\" (加权边，推荐) 或 \"connectivity\" (二值边)\n"
    "- inflate_factor (仅 MCL): 1.5-3.0，越大簇越多\n\n"
    "## 特别规则\n"
    "- 对于迷宫/连通域数据 (n >= 5000, d == 2):\n"
    "  * mutual = True (关键！)\n"
    "  * knn_k 稍大，min(25, max(10, int(sqrt(n)/3))) — 保持连通性\n"
    "  * 所有管线激活\n"
    "- 对于流形/非凸数据 (d <= 3):\n"
    "  * mutual = True\n"
    "  * louvain + mcl + spectral 激活\n\n"
    "## JSON 格式\n"
    "{\n"
    '  "knn_k": <int>,\n'
    '  "mutual": <bool>,\n'
    '  "wall_aware": <bool>,\n'
    '  "jaccard_threshold": <float>,\n'
    '  "graph_mode": "distance",\n'
    '  "algorithms": {\n'
    '    "louvain":            {"active": true},\n'
    '    "mcl":                {"active": true, "inflate_factor": 2.0, "expand_factor": 2},\n'
    '    "label_propagation":  {"active": true},\n'
    '    "spectral":           {"active": true},\n'
    '    "random_walk":        {"active": true}\n'
    "  }\n"
    "}\n\n"
    "只输出 JSON。"
)


def _extract_json(text: str) -> dict[str, Any]:
    stripped = text.strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass
    m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", stripped)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(stripped[start : end + 1])
        except json.JSONDecodeError:
            pass
    return {}


def _build_graph_defaults(n_samples: int, n_features: int) -> dict[str, Any]:
    _k = max(2, min(15, int(n_samples ** 0.5) // 2))
    return {
        "knn_k": min(30, max(5, int(n_samples ** 0.5))),
        "mutual": True,
        "wall_aware": True,
        "jaccard_threshold": 0.05,
        "graph_mode": "distance",
        "algorithms": {
            "louvain": {"active": True},
            "mcl": {"active": True, "inflate_factor": 2.0, "expand_factor": 2},
            "label_propagation": {"active": n_samples <= 5000},
            "spectral": {"active": True},
            "random_walk": {"active": n_samples <= 5000},
        },
    }


class GraphExpert(BaseExpert):
    """Graph community discovery expert (Phase 3.2).

    This is the PRIMARY pipeline for graph-connected data.
    All clustering is native graph community detection — NOT Euclidean
    clustering with graph reweighting.

    Activated conditionally when ``_classify_data_structure()`` detects
    graph-connected or high geodesic-distortion datasets.
    """

    def __init__(self) -> None:
        super().__init__("graph", "图结构专家")

    def _generate_code(
        self,
        client: UniversalLLMClient,
        dataset: DatasetBundle,
        prompt: str,
        constraints=None,
    ) -> str:
        n_features = dataset.X.shape[1] if dataset.X.ndim == 2 else 1
        n_samples = dataset.X.shape[0]
        shape_family = getattr(dataset, "shape_family", "generic")

        # 1. LLM decision for graph community discovery parameters
        is_2d = n_features <= 3
        is_maze = n_samples >= 5000 and is_2d
        context = ""
        if is_maze:
            context = (
                f"迷宫/连通域数据 ({n_samples}点, {n_features}D). "
                "knn_k取 min(25, max(10, int(sqrt(n)/3))) 保持连通性，mutual=True."
                "所有管线激活."
            )
        elif shape_family in ("non_convex", "manifold"):
            context = (
                f"非凸/流形数据 ({n_samples}点, {n_features}D). "
                "mutual=True, louvain+mcl+spectral 激活."
            )
        else:
            context = (
                f"通用数据 ({n_samples}点, {n_features}D). 推荐默认参数."
            )
        user_msg = (
            f"n_samples={n_samples}, n_features={n_features}, "
            f"shape_family={shape_family}. "
            f"{context}"
            f"请输出图社区发现决策 JSON。"
        )
        raw = client.chat_completion(
            [{"role": "user", "content": user_msg}],
            self._inject_constraints_prompt(constraints) + _DECISION_SYSTEM_PROMPT,
        ).strip()

        decisions = _extract_json(raw)
        if not decisions:
            decisions = _build_graph_defaults(n_samples, n_features)

        decisions_repr = json.dumps(decisions, ensure_ascii=False)
        decisions_repr = decisions_repr.replace("true", "True").replace("false", "False").replace("null", "None")
        return _SKELETON.replace("{DECISIONS_JSON}", decisions_repr)
