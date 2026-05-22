"""
Pre-dispatch analysis and data preparation for ACE Agent.

Extracted from ACESupervisor to keep supervisor.py focused on orchestration.
All functions here are stateless — they take data + trace and return decisions.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from ACE_Agent.agent_core.schemas import DatasetBundle, ModalityProfile

# ---------------------------------------------------------------------------
# Image shape hint
# ---------------------------------------------------------------------------

def image_shape_hint(n_features: int) -> str | None:
    """Return a shape hint string (e.g. ``"32×32×3"``) or ``None``."""
    from ACE_Agent.tools.data_factory import _decompose_image_shape

    shape = _decompose_image_shape(n_features)
    if shape is None:
        return None
    h, w, c = shape
    if c == 3:
        return f"{h}×{w}×3"
    return f"{h}×{w}"


# ---------------------------------------------------------------------------
# Data structure classification
# ---------------------------------------------------------------------------

def classify_data_structure(
    dataset: DatasetBundle,
    *,
    modality: ModalityProfile | None = None,
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

    # ---- Geodesic distortion computation -----------------------------
    # k-NN in raw >50D space is unreliable (curse of dimensionality).
    # For high-D data, PCA-reduce to ≤32D preserving 95% variance first.
    geodesic_distortion = 0.0
    wall_crossings = 0
    _pca_variance = 1.0
    _geo_pca_applied = False
    _X_for_geo = np.asarray(dataset.X, dtype=float)
    _metric = modality.distance_metric if modality else "euclidean"
    if n_features > 50 and n_samples >= 50:
        try:
            from sklearn.decomposition import PCA
            _max_components = min(32, n_features, n_samples - 1)
            _pca_tmp = PCA(n_components=_max_components, random_state=42)
            _pca_tmp.fit(_X_for_geo)
            _cumvar = np.cumsum(_pca_tmp.explained_variance_ratio_)
            _n95 = int(np.searchsorted(_cumvar, 0.95) + 1)
            _n_keep = max(2, min(_max_components, _n95))
            _pca_for_geo = PCA(n_components=_n_keep, random_state=42)
            _X_for_geo = _pca_for_geo.fit_transform(_X_for_geo)
            _pca_variance = float(np.sum(_pca_for_geo.explained_variance_ratio_))
            _geo_pca_applied = True
        except Exception:
            _X_for_geo = np.empty((0, 0))
    _n_geo_features = _X_for_geo.shape[1] if _X_for_geo.ndim == 2 else 0
    if _X_for_geo.shape[0] >= 50 and _n_geo_features >= 1:
        from ACE_Agent.tools.graph_builder import GraphBuilder as _GB
        try:
            _k = min(30, max(5, int(np.sqrt(n_samples))))
            adj = _GB.build_knn_graph(_X_for_geo, k=_k, metric=_metric)
            if n_samples <= 2000:
                geo_dists = _GB.compute_geodesic_distances(adj)
                geodesic_distortion = _GB.compute_distortion(
                    _X_for_geo, geo_dists, sample_size=min(n_samples, 500),
                )
            else:
                rng = np.random.RandomState(42)
                n_anchors = min(500, n_samples)
                anchors = rng.choice(n_samples, n_anchors, replace=False)
                geo_dists = _GB.compute_geodesic_distances(adj, indices=anchors)
                geodesic_distortion = _GB.compute_distortion(
                    _X_for_geo, geo_dists, sample_size=min(n_anchors, 200),
                )
        except Exception:
            pass

    # Detect wall-crossings when distortion is high
    if geodesic_distortion > 0.3 and n_samples <= 5000 and _n_geo_features >= 1:
        from ACE_Agent.tools.graph_builder import GraphBuilder as _GB_wall
        try:
            adj = _GB_wall.build_knn_graph(_X_for_geo, metric=_metric)
            if n_samples <= 2000:
                geo_dists = _GB_wall.compute_geodesic_distances(adj)
            else:
                rng = np.random.RandomState(42)
                n_anchors = min(500, n_samples)
                anchors = rng.choice(n_samples, n_anchors, replace=False)
                geo_dists = _GB_wall.compute_geodesic_distances(adj, indices=anchors)
            pairs = _GB_wall.detect_wall_crossings(_X_for_geo, adj, geo_dists)
            wall_crossings = len(pairs)
        except Exception:
            pass

    # ---- Hierarchical structure detection (Ward linkage) ------------
    _is_hierarchical = False
    _hierarchical_k2_ratio = 0.0
    if n_samples >= 30 and _n_geo_features >= 1:
        try:
            from scipy.cluster.hierarchy import ward
            _n_ward = min(n_samples, 200)
            if _n_ward < n_samples:
                _idx_ward = np.random.RandomState(42).choice(
                    n_samples, _n_ward, replace=False,
                )
                _X_ward = _X_for_geo[_idx_ward]
            else:
                _X_ward = _X_for_geo
            _Z = ward(_X_ward)
            _merge_dists = _Z[:, 2]
            if len(_merge_dists) >= 3:
                _k2_drop = _merge_dists[-1] / (_merge_dists.sum() + 1e-12)
                _hierarchical_k2_ratio = float(_k2_drop)
                if _hierarchical_k2_ratio > 0.6:
                    _is_hierarchical = True
        except Exception:
            pass

    # ---- Classification rules ----
    if _is_hierarchical and geodesic_distortion < 0.3:
        structure_class = "hierarchical"
    elif (geodesic_distortion > 0.5
            or (sf in ("non_convex", "manifold") and geodesic_distortion > 0.3)
            or (sf == "manifold" and geodesic_distortion > 0.15)):
        structure_class = "graph_connected"
    elif sf in ("non_convex",) and geodesic_distortion <= 0.1:
        structure_class = "non_convex"

    # ---- Strategy ----
    activate_graph_expert = (
        structure_class in ("graph_connected",) or geodesic_distortion > 0.5
    )
    topology_boost = structure_class in (
        "graph_connected", "non_convex", "manifold", "hierarchical",
    )
    centroid_suppress = (
        structure_class in ("graph_connected",) and geodesic_distortion > 0.3
    )

    return {
        "structure_class": structure_class,
        "geodesic_distortion": geodesic_distortion,
        "wall_crossings": wall_crossings,
        "is_hierarchical": _is_hierarchical,
        "hierarchical_k2_ratio": _hierarchical_k2_ratio,
        "pca_variance_retained": _pca_variance if _geo_pca_applied else 1.0,
        "recommended_strategy": {
            "activate_graph_expert": activate_graph_expert,
            "topology_boost": topology_boost,
            "centroid_suppress": centroid_suppress,
        },
    }


# ---------------------------------------------------------------------------
# Connectivity pre-check
# ---------------------------------------------------------------------------

def connectivity_pre_check(
    dataset: DatasetBundle,
    trace: list[str],
    *,
    modality: ModalityProfile | None = None,
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
        return {
            "long_range_curve": False,
            "centroid_ban": set(),
            "geodesic_distortion": 0.0,
        }

    # For high-dim data (> 50D), geodesic distortion on raw features is
    # unreliable due to the curse of dimensionality — PCA-reduce first.
    if n_features > 50:
        try:
            from sklearn.decomposition import PCA as _pca_conn
            X_raw = np.asarray(dataset.X, dtype=float)
            _max_c = min(32, n_features, n_samples - 1)
            _pca_tmp = _pca_conn(n_components=_max_c, random_state=42).fit(X_raw)
            _cumvar = np.cumsum(_pca_tmp.explained_variance_ratio_)
            _n95 = int(np.searchsorted(_cumvar, 0.95) + 1)
            _n_keep = max(2, min(_max_c, _n95))
            X_for_conn = _pca_conn(
                n_components=_n_keep, random_state=42,
            ).fit_transform(X_raw)
        except Exception:
            return {
                "long_range_curve": False,
                "centroid_ban": set(),
                "geodesic_distortion": 0.0,
            }
    else:
        X_for_conn = np.asarray(dataset.X, dtype=float)

    from ACE_Agent.tools.graph_builder import GraphBuilder as _GB2
    _conn_metric = modality.distance_metric if modality else "euclidean"
    try:
        adj = _GB2.build_knn_graph(
            X_for_conn,
            k=min(10, max(3, int(np.sqrt(n_samples)) // 4)),
            metric=_conn_metric,
        )

        if n_samples <= 2000:
            geo_dists = _GB2.compute_geodesic_distances(adj)
            distortion = _GB2.compute_distortion(
                X_for_conn, geo_dists, sample_size=min(n_samples, 500),
            )
        else:
            rng = np.random.RandomState(42)
            n_anchors = min(500, n_samples)
            anchors = rng.choice(n_samples, n_anchors, replace=False)
            geo_dists = _GB2.compute_geodesic_distances(adj, indices=anchors)
            distortion = _GB2.compute_distortion(
                X_for_conn, geo_dists, sample_size=min(n_anchors, 200),
            )
    except Exception:
        return {
            "long_range_curve": False,
            "centroid_ban": set(),
            "geodesic_distortion": 0.0,
        }

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


# ---------------------------------------------------------------------------
# Hopkins pre-check
# ---------------------------------------------------------------------------

def fast_hopkins(X: np.ndarray, n_samples: int = 200, seed: int = 42) -> float:
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
        X_unif = rng.uniform(
            low=mins, high=maxs, size=(n_ref, X_work.shape[1]),
        )

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


# ---------------------------------------------------------------------------
# Cost budget gate (N×D based)
# ---------------------------------------------------------------------------

def compute_data_cost_budget(dataset: DatasetBundle) -> dict[str, Any]:
    """Pre-flight scale gate based on N×D cost metric.

    Returns a dict with gating decisions that prevent sandbox timeout
    on large / high-dimensional datasets.  Unlike the existing N-based
    timeout (which ignores D), this gate uses the product N×D as a
    proxy for both distance-matrix cost and memory pressure.

    Tier 0 (N×D <  2M): no restrictions
    Tier 1 (N×D ≥  2M): PCA ≤50D, block Agglomerative/Spectral/Affinity/MeanShift, 120s
    Tier 2 (N×D ≥ 10M): as Tier 1 + block DBSCAN/OPTICS, downsample N>20k→2k, 300s
    Tier 3 (N×D ≥ 50M): PCA ≤32D, aggressive downsample, block HDBSCAN/Birch too, 300s
    """
    X = np.asarray(dataset.X, dtype=float)
    n, d = X.shape

    # For time-series with small T, DTW cost is O(N²·T²), not O(N²·D).
    # Use T² as the effective dimension so the gate doesn't over-react
    # to high feature counts that encode short temporal sequences.
    _md = dataset.metadata or {}
    _is_small_ts = False
    if _md.get("is_time_series") and _md.get("ts_shape"):
        _ts_T = int(_md["ts_shape"][0])
        if _ts_T <= 128:
            _is_small_ts = True
            cost = n * (_ts_T ** 2)
        else:
            cost = n * d
    else:
        cost = n * d

    result: dict[str, Any] = {
        "n_samples": n,
        "n_features": d,
        "cost_score": cost,
        "force_dim_reduce": False,
        "cap_dims": 0,
        "block_o_n2_algorithms": [],
        "force_downsample_to": None,
        "timeout_tier": 0,
        "log_message": "",
    }

    if cost < 2_000_000:
        return result

    if cost >= 2_000_000:
        result["timeout_tier"] = 1  # 120s
        result["block_o_n2_algorithms"] = [
            "AgglomerativeClustering", "SpectralClustering",
            "AffinityPropagation", "MeanShift",
        ]
        if not _is_small_ts:
            result["force_dim_reduce"] = True
            result["cap_dims"] = min(50, d)
        result["log_message"] = (
            f"【规模门禁 Tier 1】{'时序成本' if _is_small_ts else 'N×D'}={cost / 1e6:.1f}M ≥ 2M。"
            + (f"强制降维至≤{result['cap_dims']}D，" if result["force_dim_reduce"] else "保留原始时序形状，")
            + f"屏蔽O(N²)算法: {', '.join(result['block_o_n2_algorithms'])}。"
            f"执行超时升至 120s。"
        )

    if cost >= 10_000_000:
        result["timeout_tier"] = 2  # 300s
        result["block_o_n2_algorithms"].extend(["DBSCAN", "OPTICS"])
        if not _is_small_ts:
            result["force_dim_reduce"] = True
            result["cap_dims"] = min(50, d)
        if n > 20000:
            result["force_downsample_to"] = 2000
            result["log_message"] += (
                f" N>20000，追加降采样至 {result['force_downsample_to']}。"
            )
        else:
            result["log_message"] = (
                f"【规模门禁 Tier 2】{'时序成本' if _is_small_ts else 'N×D'}={cost / 1e6:.1f}M ≥ 10M。"
                + (f"强制降维至≤{result['cap_dims']}D，" if result["force_dim_reduce"] else "保留原始时序形状，")
                + f"屏蔽O(N²)/密度算法: {', '.join(result['block_o_n2_algorithms'])}。"
                f"执行超时升至 300s。"
            )

    if cost >= 50_000_000:
        result["cap_dims"] = min(32, d)
        result["force_downsample_to"] = min(2000, max(500, n // 10))
        result["block_o_n2_algorithms"].extend(["HDBSCAN", "Birch"])
        result["force_dim_reduce"] = True
        result["log_message"] = (
            f"【规模门禁 Tier 3】{'时序成本' if _is_small_ts else 'N×D'}={cost / 1e6:.1f}M ≥ 50M。"
            f"强制降维至≤{result['cap_dims']}D + 降采样至 {result['force_downsample_to']}。"
            f"仅保留轻量算法。"
        )

    return result


# ---------------------------------------------------------------------------
# Image data detection
# ---------------------------------------------------------------------------

def detect_image_data(dataset: DatasetBundle) -> str | None:
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
    return image_shape_hint(n)


# ---------------------------------------------------------------------------
# Manifold topology detection
# ---------------------------------------------------------------------------

def detect_manifold_topology(
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


# ---------------------------------------------------------------------------
# High-dimensional reduction (>200D → PCA 95% variance)
# ---------------------------------------------------------------------------

def apply_highdim_reduction(
    dataset: DatasetBundle,
    trace: list[str],
) -> DatasetBundle | None:
    """Reduce high-dim data (>200D) via PCA retaining 95% variance
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
        from sklearn.decomposition import PCA

        X = np.asarray(dataset.X, dtype=float)
        n_samples, n_features = X.shape

        if n_features <= 200:
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

        # Time-series data: when the number of time steps (T) is small and
        # N is moderate, DTW on the raw shape is viable without PCA.
        # DTW complexity is O(N²·T²), not O(N²·D), so T=63 with N=3000
        # is far cheaper than Euclidean O(N²·D) on 4032D.
        # For small-T time-series, we SKIP PCA to preserve the temporal
        # ordering — experts reshape in-sandbox and use DTW directly.
        if _meta.get("is_time_series") and _meta.get("ts_shape"):
            _ts_T, _ts_F = _meta["ts_shape"]
            if _ts_T <= 128 and n_samples <= 5000:
                trace.append(
                    f"【时序路由】T={_ts_T}≤128, N={n_samples}≤5000。"
                    f"DTW 复杂度 O(N²·{_ts_T}²) 可控，"
                    f"跳过 PCA 保留原始时序形状 ({_ts_T}×{_ts_F}) 供 DTW 聚类使用。"
                )
                return None
            # Higher cap for time-series: 200D lets Spectral survive while
            # keeping most of the temporal signal.
            n_components_max = min(200, n_features, n_samples - 1)
            pca = PCA(n_components=n_components_max, svd_solver="randomized",
                      random_state=42)
            X_reduced = pca.fit_transform(X)
            cumsum = np.cumsum(pca.explained_variance_ratio_)
            # 99% variance for time-series (vs 95% default) — more signal preserved
            n_keep = min(int(np.searchsorted(cumsum, 0.99) + 1), len(cumsum))
            n_keep = max(n_keep, min(16, len(cumsum)))
            if n_keep < X_reduced.shape[1]:
                X_reduced = X_reduced[:, :n_keep]

            trace.append(
                f"【时序路由】PCA {n_features}D → {n_keep}D"
                f"（保留 {cumsum[n_keep - 1]:.1%} 方差，"
                f"原时序形状 {_ts_T}×{_ts_F}）。"
            )

            # Build updated metadata: keep is_time_series but clear
            # ts_shape so experts don't attempt the invalid reshape.
            _ts_meta = {**(_meta or {}),
                        "preprocessing": "pca_highdim_ts",
                        "is_time_series": True,
                        "ts_shape": None,
                        "ts_shape_original": [_ts_T, _ts_F]}
            return DatasetBundle(
                name=f"{dataset.name}_pca{n_keep}_ts",
                display_name=f"{dataset.display_name} (PCA{n_keep})",
                X=X_reduced.astype(float),
                y=dataset.y,
                description=f"PCA-reduced from {n_features}D to {n_keep}D (99% variance, time-series).",
                shape_family=dataset.shape_family,
                feature_names=[f"PC{i + 1}" for i in range(n_keep)],
                metadata=_ts_meta,
            )

        # Single PCA fit: randomized solver, capped at 100 components
        n_components_max = min(200, n_features, n_samples - 1)

        if n_samples > 5000:
            # Fit on a random subset for speed, transform all data
            rng = np.random.default_rng(42)
            fit_idx = rng.choice(
                n_samples, size=min(5000, n_samples), replace=False,
            )
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


# ---------------------------------------------------------------------------
# Hard dimension cap (scale-gate driven)
# ---------------------------------------------------------------------------

def apply_hard_dim_reduction(
    dataset: DatasetBundle,
    target_dim: int,
    trace: list[str],
    *,
    modality: ModalityProfile | None = None,
) -> DatasetBundle | None:
    """Hard-cap dimension reduction to target_dim using TruncatedSVD or PCA.

    Unlike apply_highdim_reduction (variance-based, triggers at D>200),
    this is invoked by the data-size gate and guarantees output dimension
    ≤ target_dim regardless of variance retained.
    """
    try:
        X = np.asarray(dataset.X, dtype=float)
        n_samples, n_features = X.shape
        target_dim = min(target_dim, n_features, n_samples - 1)

        if n_features <= target_dim:
            return None

        # Modality-aware reducer selection
        _md = dataset.metadata or {}
        dim_hint = _md.get("dim_reduction_hint", "pca")
        if modality is not None:
            dim_hint = modality.dim_reduction_hint

        if dim_hint == "truncated_svd":
            from sklearn.decomposition import TruncatedSVD
            reducer = TruncatedSVD(n_components=target_dim, random_state=42)
            method = "TruncatedSVD"
        else:
            from sklearn.decomposition import PCA
            reducer = PCA(
                n_components=target_dim, svd_solver="randomized",
                random_state=42,
            )
            method = "PCA"

        # Fit on subset for speed when N is large
        if n_samples > 5000:
            rng = np.random.default_rng(42)
            fit_idx = rng.choice(
                n_samples, size=min(5000, n_samples), replace=False,
            )
            reducer.fit(X[fit_idx])
            X_reduced = reducer.transform(X)
        else:
            X_reduced = reducer.fit_transform(X)

        trace.append(
            f"【硬降维门禁】{method} {n_features}D → {target_dim}D"
            f"（规模门禁强制，非方差保留模式）。"
        )

        return DatasetBundle(
            name=f"{dataset.name}_{method.lower()}{target_dim}",
            display_name=f"{dataset.display_name} ({method}{target_dim})",
            X=X_reduced.astype(float),
            y=dataset.y,
            description=f"{method}-reduced from {n_features}D to {target_dim}D for scale safety.",
            shape_family=dataset.shape_family,
            feature_names=[f"{method[:3]}{i + 1}" for i in range(target_dim)],
            metadata={**(_md or {}), "preprocessing": f"hard_{method.lower()}"},
        )
    except Exception as exc:
        trace.append(f"【硬降维门禁】降维失败 ({exc})，继续使用原始数据。")
        return None


# ---------------------------------------------------------------------------
# Large-dataset subsampling
# ---------------------------------------------------------------------------

def subsample_large_dataset(
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
        metadata={**(dataset.metadata or {}), "preprocessing": "subsample",
                  "original_n_samples": n_samples},
    )


# ---------------------------------------------------------------------------
# Manifold preprocessing (UMAP / SpectralEmbedding)
# ---------------------------------------------------------------------------

def apply_manifold_preprocessing(
    dataset: DatasetBundle,
    trace: list[str],
) -> DatasetBundle | None:
    """Reduce high-D manifold data to a 2D/3D embedding via UMAP or
    SpectralEmbedding before clustering experts are dispatched.

    Returns a new DatasetBundle with the transformed feature matrix,
    or ``None`` if preprocessing could not be applied.
    """
    try:
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


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def prepare_output_dir(name: str) -> Path:
    path = Path(f"outputs/{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_raw_plot(dataset: DatasetBundle, out_dir: Path) -> Path:
    import matplotlib.pyplot as plt

    path = out_dir / "raw_data.png"
    X = np.asarray(dataset.X, dtype=float)
    if X.shape[1] < 2:
        return path
    if X.shape[1] > 2:
        from sklearn.decomposition import PCA
        X = PCA(n_components=2, random_state=42).fit_transform(X)
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c="gray", alpha=0.5, s=10)
    plt.title(f"Dataset: {dataset.display_name}")
    plt.savefig(path, dpi=150)
    plt.close()
    return path


# ---------------------------------------------------------------------------
# Orchestrator: run all pre-flight gates
# ---------------------------------------------------------------------------

def run_preflight_gates(
    dataset: DatasetBundle,
    prompt: str,
    trace: list[str],
    active_experts: list[str],
    *,
    settings: Any,
    constraints: dict[str, Any] | None = None,
    structure: dict[str, Any] | None = None,
    modality: ModalityProfile | None = None,
) -> dict[str, Any]:
    """Execute all pre-dispatch gates: dimension reduction, routing,
    cost budget, subsampling.  Returns updated working state.

    Returns a dict with:
    - working_dataset: DatasetBundle (may differ from input after reduction/subsampling)
    - prompt: str (may have hints injected)
    - constraints: dict (may have blocked_algorithms merged)
    - active_experts: list[str]
    - trace: list[str] (updated in-place)
    - base_timeout: int
    """
    if structure is None:
        structure = {}
    if modality is None:
        from ACE_Agent.agent_core.schemas import detect_modality
        modality = detect_modality(dataset)

    working_dataset = dataset

    # ---- Phase 5.4: High-dim dimension gatekeeper --------------------
    _n_features_raw = dataset.X.shape[1] if dataset.X.ndim == 2 else 1
    if _n_features_raw > 200:
        _highdim_result = apply_highdim_reduction(dataset, trace)
        if _highdim_result is not None:
            working_dataset = _highdim_result

    # Inject dim_reduction_hint into working_dataset metadata
    _aug_meta = dict(working_dataset.metadata) if working_dataset.metadata else {}
    _aug_meta["dim_reduction_hint"] = modality.dim_reduction_hint
    _aug_meta["modality_type"] = modality.modality_type
    _aug_meta["distance_metric"] = modality.distance_metric
    working_dataset = replace(working_dataset, metadata=_aug_meta)

    # ---- Phase 6: Image-aware routing ---------------------------------
    _img_meta = dataset.metadata or {}
    if _img_meta.get("is_image") and _n_features_raw > 500 and "dimension" not in active_experts:
        active_experts.append("dimension")
        trace.append(
            f"【图像路由】检测到图像数据 ({_n_features_raw}D)，"
            f"已激活维度专家（深度嵌入管线）。"
            f"提示：原始像素欧氏距离无语义判别力，建议使用 ResNet 特征模式 (cifar10_resnet)。"
        )

    # ---- Phase 2.4: Manifold pre-processing for complex topology ----
    _manifold_detected = (
        working_dataset.X.shape[1] <= 100
        and (
            structure.get("structure_class") in ("manifold", "non_convex", "graph_connected")
            or detect_manifold_topology(working_dataset)
        )
    )
    if _manifold_detected and working_dataset.X.shape[1] > 2:
        embedded = apply_manifold_preprocessing(working_dataset, trace)
        if embedded is not None:
            working_dataset = embedded
            trace.append("【流形预处理】已将嵌入数据集作为后续专家的输入。")

    # ---- Sparse / text data cosine-space routing --------------------
    if modality.l2_normalize:
        from sklearn.preprocessing import normalize as _l2_norm
        _X_normed = _l2_norm(working_dataset.X, norm="l2")
        working_dataset = replace(working_dataset, X=_X_normed)
        trace.append(
            f"【余弦路由】检测到{modality.modality_type}数据，已对特征做 L2 归一化。"
            f"后续所有欧氏距离等效为余弦距离。"
        )

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
            f"【执行超时】数据集 {_n_samples} 样本，"
            f"超时调整为 {_base_timeout}s。"
        )
    if getattr(settings, "deep_mode", False) and _n_samples > 5000:
        _base_timeout += 60
        trace.append(f"【执行超时】深度模式 +60s → {_base_timeout}s。")

    # DTW distance matrix is O(N²T), roughly 30× slower than Euclidean.
    if modality.modality_type == "time_series" and modality.ts_large_n:
        _base_timeout = max(_base_timeout, 180)
        trace.append(
            f"【执行超时】时序数据 DTW 上调至 {_base_timeout}s"
            f"（{_n_samples} 样本，O(N²T) 计算密集）。"
        )

    # ---- Pre-flight data-size gate (N×D based) -------------------------
    _budget = compute_data_cost_budget(working_dataset)
    if _budget["log_message"]:
        trace.append(_budget["log_message"])

    # Override timeout from cost-budget tier
    _tier_timeouts = {0: 60, 1: 120, 2: 300}
    _base_timeout = _tier_timeouts.get(_budget["timeout_tier"], _base_timeout)

    # Hard dimension cap (separate from the variance-based PCA above)
    if _budget["force_dim_reduce"] and working_dataset.X.shape[1] > _budget["cap_dims"]:
        _capped = apply_hard_dim_reduction(
            working_dataset, _budget["cap_dims"], trace, modality=modality,
        )
        if _capped is not None:
            working_dataset = _capped

    # Cost-driven downsampling
    if _budget.get("force_downsample_to"):
        _sub = subsample_large_dataset(
            working_dataset,
            max_samples=_budget["force_downsample_to"],
            trace=trace,
        )
        if _sub is not None:
            working_dataset = _sub

    # Build expert constraints with blocked algorithms
    _blocked = _budget.get("block_o_n2_algorithms", [])
    if _blocked:
        if constraints is None:
            constraints = {}
        _existing_blocked = constraints.get("blocked_algorithms", [])
        constraints = {**constraints,
                       "blocked_algorithms": list(set(_existing_blocked + _blocked))}

    # ---- Large-sample downsampling (fallback, N > 10K) ------------------
    if not _budget.get("force_downsample_to"):
        _subsample_result = subsample_large_dataset(
            working_dataset, trace=trace,
        )
        if _subsample_result is not None:
            working_dataset = _subsample_result

    # ---- Sparse data: inject cosine hint into expert prompt ----------
    if modality.modality_type == "text":
        _cosine_hint = (
            "\n\n【重要提示】数据已做 L2 归一化，当前欧氏距离等效于余弦距离。"
            "请优先使用 cosine_similarity 或直接在 L2 归一化数据上使用 KMeans（等价于 SphericalKMeans），"
            "k-NN 图构建也应优先使用 cosine 相似度。"
        )
        prompt = prompt + _cosine_hint

    # ---- Time-series data: inject DTW hint into expert prompt ---------
    if modality.modality_type == "time_series":
        _ts_shape = modality.ts_shape
        _ts_orig = modality.ts_original_shape
        _n_samples = working_dataset.X.shape[0]
        _n_features = working_dataset.X.shape[1]

        if _ts_shape and len(_ts_shape) == 2:
            _ts_T, _ts_F = _ts_shape
            _dtw_hint = (
                f"\n\n【时间序列路由】此数据为时间序列，原始形状为 ({_ts_T} 时间步 × {_ts_F} 特征)，"
                f"已展平为 {_ts_T * _ts_F}D，共 {_n_samples} 个样本。"
                f"执行环境已预注入 TimeSeriesKMeans (from tslearn.clustering)，支持 DTW 距离。"
            )
            if _n_samples > 500:
                _dtw_hint += (
                    f"\n⚠️ {_n_samples} 样本下完整 DTW 极易超时。加速策略：\n"
                    f"方案A（推荐）: 先 stratify 降采样至 500~800 样本，"
                    f"再使用 TimeSeriesKMeans(metric='dtw')。\n"
                    f"方案B: TimeSeriesKMeans(n_clusters=k, metric='dtw',"
                    f" metric_params={{'sakoe_chiba_radius': 3}}, max_iter=10, random_state=42)。\n"
                    f"方案C: 用 tslearn.metrics.cdist_dtw 预计算 DTW 距离矩阵，"
                    f"传入 SpectralClustering(affinity='precomputed')。\n"
                    f"步骤: 1) X_ts = X.reshape(n_samples, {_ts_T}, {_ts_F})\n"
                    f"      2) 选方案A/B/C执行聚类\n"
                    f"      3) labels 写回 artifacts['labels']"
                )
            else:
                _dtw_hint += (
                    f"\n1. X_ts = X.reshape(n_samples, {_ts_T}, {_ts_F})\n"
                    f"2. TimeSeriesKMeans(n_clusters=k, metric='dtw', max_iter=10, random_state=42)\n"
                    f"3. DTW 自动处理时间轴伸缩，适合心音/语音等时序频谱数据。"
                )
        else:
            _ts_desc = ""
            if _ts_orig and len(_ts_orig) == 2:
                _ts_desc = f"（原始时序形状 {_ts_orig[0]}×{_ts_orig[1]}）"
            _dtw_hint = (
                f"\n\n【时间序列路由】此数据源自时间序列{_ts_desc}，"
                f"已 PCA 降维至 {_n_features}D（保留 99% 方差）。"
                f"原始时序形状已不可用（PCA 混合了时间轴），"
                f"请使用降维空间中的有效方法："
                f"\n1. SpectralClustering + k-NN 图（对非球形结构最有效）"
                f"\n2. HDBSCAN / OPTICS（密度聚类，可发现任意形状簇）"
                f"\n3. 可用 tslearn.metrics.cdist_dtw 在降维空间中计算 DTW 距离，"
                f"传入 SpectralClustering(affinity='precomputed')"
            )
        _dtw_hint += "\n请优先使用 DTW 距离或谱方法替代欧氏距离进行聚类。"
        prompt = prompt + _dtw_hint

    return {
        "working_dataset": working_dataset,
        "prompt": prompt,
        "constraints": constraints,
        "active_experts": list(active_experts),
        "base_timeout": _base_timeout,
        "modality": modality,
    }
