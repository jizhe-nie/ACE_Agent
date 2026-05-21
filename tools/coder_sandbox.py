"""
tools/coder_sandbox.py
======================
Safe Python execution sandbox for ACE Agent (Phase 0).

Security features:
- Restricted builtins (allowlist only)
- Dangerous os/sys methods intercepted (os.remove, os.system, sys.exit, etc.)
- High-risk modules blocked entirely (subprocess, socket, pickle, shutil, etc.)
- Wall-clock timeout via threading.Timer
- Memory upper bound monitored via psutil (Windows-compatible; no resource module needed)
- SandboxResourceExceeded raised with typed reason string: "timeout" | "memory" | "cpu"

Configuration (environment variables or constructor kwargs):
  ACE_SANDBOX_TIMEOUT_SEC   : wall-clock timeout in seconds (default 60)
  ACE_SANDBOX_MEMORY_MB     : RSS memory ceiling in MiB (default 2048 = 2 GiB)

Pre-injection (Phase 3):
  Core sklearn types are pre-injected into every sandbox namespace so that
  LLM-generated code need not import them.  A read-only DataContext object
  (CTX_DATA) carries X, y and metadata to eliminate ``NameError: data is
  not defined`` and similar variable-binding mistakes.
"""

from __future__ import annotations

import os
import threading
import time
import traceback
from dataclasses import dataclass
from typing import Any

import numpy as np
import psutil
from loguru import logger

# ---------------------------------------------------------------------------
# Defaults / configuration
# ---------------------------------------------------------------------------
_DEFAULT_TIMEOUT_SEC: int = int(os.environ.get("ACE_SANDBOX_TIMEOUT_SEC", "60"))
_DEFAULT_MEMORY_MB: int = int(os.environ.get("ACE_SANDBOX_MEMORY_MB", "2048"))


# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------
class SandboxResourceExceeded(RuntimeError):
    """
    Raised when sandbox execution breaches a resource limit.

    Attributes:
        reason: "timeout" | "memory" | "cpu"
        detail: Human-readable description including the actual measured value.
    """

    def __init__(self, reason: str, detail: str = ""):
        self.reason = reason
        self.detail = detail
        super().__init__(f"SandboxResourceExceeded[{reason}]: {detail}")


# ---------------------------------------------------------------------------
# Data context (immutable) — carries X, y and metadata into the sandbox
# ---------------------------------------------------------------------------
class DataContext:
    """Lightweight read-only bag injected as ``CTX_DATA`` into every sandbox.

    LLM-generated code accesses the dataset through this context instead of
    bare ``X`` / ``y`` global variables.  Repeated reassignment of an
    attribute raises ``TypeError`` so that accidental shadowing (e.g.
    ``CTX_DATA = ...`` by-name re-binding still works, but ``CTX_DATA.X = 42``
    fails fast with a clear message.
    """

    __slots__ = ("X", "y", "n_samples", "n_features", "expected_clusters", "has_labels", "display_name", "metadata")

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray | None,
        display_name: str = "",
        expected_clusters: int = 3,
        metadata: dict[str, Any] | None = None,
    ):
        object.__setattr__(self, "X", X)
        object.__setattr__(self, "y", y)
        object.__setattr__(self, "n_samples", X.shape[0])
        object.__setattr__(self, "n_features", X.shape[1] if X.ndim == 2 else 1)
        object.__setattr__(self, "expected_clusters", expected_clusters)
        object.__setattr__(self, "has_labels", y is not None)
        object.__setattr__(self, "display_name", display_name)
        object.__setattr__(self, "metadata", metadata or {})

    def __setattr__(self, name: str, value: Any) -> None:
        raise TypeError(
            f"CTX_DATA is read-only.  Cannot set '{name}'.  Use the skeleton variables or artifacts to store results."
        )

    def __repr__(self) -> str:
        return (
            f"DataContext(n_samples={self.n_samples}, n_features={self.n_features},"
            f" expected_clusters={self.expected_clusters}, has_labels={self.has_labels})"
        )


# ---------------------------------------------------------------------------
# DBCV (Density-Based Cluster Validation) — pre-injected into sandbox
# ---------------------------------------------------------------------------
def _dbcv_score(X: np.ndarray, labels: np.ndarray) -> float:
    """Compute DBCV index for arbitrary-shaped clustering quality.

    DBCV measures the ratio of inter-cluster density separation to
    intra-cluster density sparseness.  Range approx [-1, 1]; higher is better.
    DBCV < 0 means clusters are not well-separated in density space.
    """
    import numpy as _np

    _X = _np.asarray(X, dtype=float)
    _labels = _np.asarray(labels, dtype=int).ravel()
    unique_lbls = _np.unique(_labels)
    unique_lbls = unique_lbls[unique_lbls >= 0]  # exclude noise
    n_clusters = len(unique_lbls)
    n_samples = _X.shape[0]

    if n_clusters < 2 or n_clusters >= n_samples:
        return 0.0

    from sklearn.neighbors import NearestNeighbors

    n_neighbors = min(10, n_samples - 1)
    nn = NearestNeighbors(n_neighbors=n_neighbors).fit(_X)
    dists, _indices = nn.kneighbors(_X)
    core_dists = dists[:, -1]  # distance to k-th neighbor

    # Per-cluster density sparseness: median core distance
    sparseness = {}
    for lbl in unique_lbls:
        mask = _labels == lbl
        if mask.sum() > 1:
            sparseness[lbl] = float(_np.median(core_dists[mask]))
        else:
            sparseness[lbl] = 0.0

    # Inter-cluster density separation: min pairwise distance between clusters
    separations = []
    unique_arr = list(unique_lbls)
    for i in range(len(unique_arr)):
        for j in range(i + 1, len(unique_arr)):
            ci, cj = unique_arr[i], unique_arr[j]
            mi = _labels == ci
            mj = _labels == cj
            # Min point-to-point distance between two clusters
            # Use a sampled approach for large clusters
            ni, nj = mi.sum(), mj.sum()
            if ni * nj <= 10000:
                # Full cross-distance
                Xi, Xj = _X[mi], _X[mj]
                from scipy.spatial.distance import cdist
                dmat = cdist(Xi, Xj, metric="euclidean")
                min_d = float(dmat.min())
            else:
                # Sampled approach
                rng = _np.random.RandomState(42)
                si = rng.choice(ni, min(100, ni), replace=False)
                sj = rng.choice(nj, min(100, nj), replace=False)
                Xi, Xj = _X[mi][si], _X[mj][sj]
                from scipy.spatial.distance import cdist
                dmat = cdist(Xi, Xj, metric="euclidean")
                min_d = float(dmat.min())
            separations.append(min_d)

    if not separations:
        return 0.0

    mean_sep = float(_np.mean(separations))
    mean_sparse = float(_np.mean(list(sparseness.values())))
    denom = max(mean_sep, mean_sparse)
    if denom < 1e-12:
        return 0.0
    return float((mean_sep - mean_sparse) / denom)


# ---------------------------------------------------------------------------
# Core modules pre-injected into every sandbox so LLM code can use them
# without import statements.  Additional expert-specific modules are merged
# via the ``pre_inject`` kwarg of ``execute()``.
# ---------------------------------------------------------------------------
def _build_core_pre_inject() -> dict[str, Any]:
    """Construct the pre-injection dict for core sklearn types.

    These are available **without import** inside every sandbox execution.
    """
    # ---- polyfill scipy.stats.weighted_median (added in scipy 1.13.0) ----
    try:
        import scipy.stats as _sp_stats
        if not hasattr(_sp_stats, "weighted_median"):
            import numpy as _np

            def _weighted_median(data, weights):
                """Polyfill for scipy>=1.13.0 weighted_median."""
                data = _np.asarray(data)
                weights = _np.asarray(weights, dtype=float)
                if weights.sum() == 0:
                    raise ValueError("Total weight must be positive.")
                idx = _np.argsort(data)
                cdf = _np.cumsum(weights[idx])
                cdf /= cdf[-1]
                cut = _np.searchsorted(cdf, 0.5)
                if cut == 0:
                    return float(data[idx[0]])
                w0 = cdf[cut - 1]
                w1 = cdf[cut]
                if w1 - w0 < 1e-15:
                    return float(data[idx[cut]])
                t = (0.5 - w0) / (w1 - w0)
                return float((1 - t) * data[idx[cut - 1]] + t * data[idx[cut]])

            _sp_stats.weighted_median = _weighted_median
    except Exception:
        pass

    modules: dict[str, Any] = {}
    # --- pre-processing ---
    try:
        from sklearn.preprocessing import StandardScaler, MinMaxScaler, normalize  # noqa: F811

        modules["StandardScaler"] = StandardScaler
        modules["MinMaxScaler"] = MinMaxScaler
        modules["normalize"] = normalize
    except Exception:
        pass
    # --- cosine / pairwise distances (essential for text/sparse data) ---
    try:
        from sklearn.metrics.pairwise import cosine_similarity  # noqa: F811

        modules["cosine_similarity"] = cosine_similarity
    except Exception:
        pass
    # --- decomposition ---
    try:
        from sklearn.decomposition import PCA  # noqa: F811

        modules["PCA"] = PCA
    except Exception:
        pass
    # --- manifold / dim-reduction ---
    try:
        from umap import UMAP  # noqa: F811
        modules["UMAP"] = UMAP
    except Exception:
        pass
    try:
        from sklearn.manifold import (  # noqa: F811
            TSNE,
            Isomap,
            MDS,
            LocallyLinearEmbedding,
            SpectralEmbedding,
        )
        modules["TSNE"] = TSNE
        modules["Isomap"] = Isomap
        modules["MDS"] = MDS
        modules["LocallyLinearEmbedding"] = LocallyLinearEmbedding
        modules["SpectralEmbedding"] = SpectralEmbedding
    except Exception:
        pass
    try:
        from sklearn.decomposition import TruncatedSVD  # noqa: F811
        modules["TruncatedSVD"] = TruncatedSVD
    except Exception:
        pass
    # --- clustering ---
    try:
        from sklearn.cluster import KMeans  # noqa: F811

        modules["KMeans"] = KMeans
    except Exception:
        pass
    try:
        from sklearn.mixture import GaussianMixture  # noqa: F811

        modules["GaussianMixture"] = GaussianMixture
    except Exception:
        pass
    # --- metrics ---
    try:
        from sklearn.metrics import (
            silhouette_score,
            calinski_harabasz_score,
            davies_bouldin_score,
            adjusted_rand_score,
            normalized_mutual_info_score,
        )

        modules["silhouette_score"] = silhouette_score
        modules["calinski_harabasz_score"] = calinski_harabasz_score
        modules["davies_bouldin_score"] = davies_bouldin_score
        modules["adjusted_rand_score"] = adjusted_rand_score
        modules["normalized_mutual_info_score"] = normalized_mutual_info_score
    except Exception:
        pass
    # --- graph / nearest-neighbours (Phase 3 Topology-Aware) ---
    try:
        from sklearn.neighbors import (  # noqa: F811
            NearestNeighbors,
            kneighbors_graph,
            radius_neighbors_graph,
        )
        modules["NearestNeighbors"] = NearestNeighbors
        modules["kneighbors_graph"] = kneighbors_graph
        modules["radius_neighbors_graph"] = radius_neighbors_graph
    except Exception:
        pass
    try:
        from scipy.sparse import csgraph  # noqa: F811
        modules["csgraph"] = csgraph
    except Exception:
        pass
    try:
        import scipy.sparse as sparse  # noqa: F811
        modules["sparse"] = sparse
    except Exception:
        pass
    # --- density / topology algorithms (Critic bootstrap stability needs these) ---
    try:
        from sklearn.cluster import DBSCAN, OPTICS, AgglomerativeClustering  # noqa: F811
        from sklearn.cluster import SpectralClustering as _SpectralClustering

        class _SafeSpectralClustering(_SpectralClustering):
            """Wrapper that prevents scipy 'k >= N' error by capping n_clusters.

            When the affinity matrix is sparse, scipy eigsh requires k < N.
            LLM-generated code often sets n_clusters too high for sub-sampled
            data, causing::

                TypeError: Cannot use scipy.linalg.eigh for sparse A with k >= N

            This wrapper silently caps n_clusters to max(2, n_samples - 1) at
            fit time, which is always safe.
            """
            def fit(self, X, y=None):
                n_samples = X.shape[0] if hasattr(X, 'shape') else len(X)
                safe_k = max(2, n_samples - 1)
                if self.n_clusters > safe_k:
                    self.n_clusters = safe_k
                return super().fit(X, y)

        modules["DBSCAN"] = DBSCAN
        modules["OPTICS"] = OPTICS
        modules["SpectralClustering"] = _SafeSpectralClustering
        modules["AgglomerativeClustering"] = AgglomerativeClustering
    except Exception:
        pass
    try:
        from sklearn.cluster import HDBSCAN  # noqa: F811
        modules["HDBSCAN"] = HDBSCAN
    except Exception:
        pass
    # --- time-series clustering (DTW via tslearn) ---
    try:
        from tslearn.clustering import TimeSeriesKMeans  # noqa: F811

        modules["TimeSeriesKMeans"] = TimeSeriesKMeans
        # Also expose tslearn.metrics.dtw for pairwise distance computation
        import tslearn.metrics as _ts_metrics  # noqa: F811
        modules["tslearn_metrics"] = _ts_metrics
    except Exception:
        pass
    # --- model selection (Critic stratified sampling) ---
    try:
        from sklearn.model_selection import StratifiedShuffleSplit  # noqa: F811
        modules["StratifiedShuffleSplit"] = StratifiedShuffleSplit
    except Exception:
        pass
    # --- tree (Critic boundary quality audit) ---
    try:
        from sklearn.tree import DecisionTreeClassifier  # noqa: F811
        modules["DecisionTreeClassifier"] = DecisionTreeClassifier
    except Exception:
        pass
    # --- dbcv_score (density-based cluster validation) ---
    modules["dbcv_score"] = _dbcv_score
    return modules


CORE_PRE_INJECT: dict[str, Any] = _build_core_pre_inject()


# ---------------------------------------------------------------------------
# Allowed builtins
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Safe import and module wrappers (sandbox hardening)
# ---------------------------------------------------------------------------
_DENIED_OS_ATTRS: set[str] = {
    "chmod", "chown", "execv", "execve", "fork", "kill", "popen",
    "remove", "removedirs", "rmdir", "spawnl", "spawnle", "system",
    "unlink",
}
_DENIED_SYS_ATTRS: set[str] = {"exit", "_exit", "call_tracing", "setprofile", "settrace"}
_BLOCKED_MODULES: set[str] = {
    "ctypes", "ftplib", "http", "http.client", "http.server",
    "imaplib", "multiprocessing", "pickle", "requests", "shutil",
    "smtplib", "socket", "socketserver", "subprocess", "telnetlib",
    "urllib", "urllib.request", "urllib.parse",
}


class _SafeModule:
    """Proxy that wraps a module and denies access to dangerous attributes."""

    def __init__(self, real_module: Any, denied: set[str]) -> None:
        object.__setattr__(self, "_real", real_module)
        object.__setattr__(self, "_denied", denied)

    def __getattr__(self, name: str) -> Any:
        if name in self._denied:
            modname = getattr(self._real, "__name__", "?")
            raise PermissionError(
                f"Sandbox: {modname}.{name} is blocked for security reasons"
            )
        return getattr(self._real, name)

    def __setattr__(self, name: str, value: Any) -> None:
        object.__setattr__(self, name, value)

    def __dir__(self) -> list[str]:
        return [x for x in dir(self._real) if x not in self._denied]


def _safe_import(
    name: str,
    globals: Any = None,
    locals: Any = None,
    fromlist: tuple[str, ...] = (),
    level: int = 0,
) -> Any:
    """Sandbox-aware ``__import__``.

    - Entirely blocked modules raise ``ImportError``.
    - ``os`` and ``sys`` are returned pre-wrapped with dangerous attributes
      stripped.
    - Everything else passes through to the real ``__import__``.
    """
    if name in _BLOCKED_MODULES:
        raise ImportError(f"Module '{name}' is not allowed inside the sandbox")
    module = __import__(name, globals, locals, fromlist, level)
    if name == "os":
        return _SafeModule(module, _DENIED_OS_ATTRS)
    if name == "sys":
        return _SafeModule(module, _DENIED_SYS_ATTRS)
    return module


SAFE_BUILTINS: dict[str, Any] = {
    "__build_class__": __build_class__,
    "__import__": _safe_import,
    # Standard exception hierarchy
    "AttributeError": AttributeError,
    "Exception": Exception,
    "FileNotFoundError": FileNotFoundError,
    "ImportError": ImportError,
    "IndexError": IndexError,
    "KeyError": KeyError,
    "ModuleNotFoundError": ModuleNotFoundError,
    "NameError": NameError,
    "NotImplementedError": NotImplementedError,
    "OSError": OSError,
    "PermissionError": PermissionError,
    "RuntimeError": RuntimeError,
    "TypeError": TypeError,
    "ValueError": ValueError,
    # Builtin functions (safe — no I/O, no code execution)
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "dict": dict,
    "dir": dir,
    "enumerate": enumerate,
    "filter": filter,
    "float": float,
    "getattr": getattr,
    "globals": globals,
    "hasattr": hasattr,
    "int": int,
    "isinstance": isinstance,
    "iter": iter,
    "len": len,
    "list": list,
    "locals": locals,
    "map": map,
    "max": max,
    "min": min,
    "next": next,
    "object": object,
    "pow": pow,
    "print": print,
    "range": range,
    "reversed": reversed,
    "round": round,
    "set": set,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "super": super,
    "tuple": tuple,
    "type": type,
    "vars": vars,
    "zip": zip,
}


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------
@dataclass
class SandboxExecution:
    code: str
    result: dict[str, Any]
    error: str | None = None


# ---------------------------------------------------------------------------
# Artifacts-shape detector (used by the broadened rescue in execute())
# ---------------------------------------------------------------------------
def _looks_like_artifacts(v: Any) -> bool:
    """Return True if ``v`` looks like the expert artifacts contract.

    The expert contract is::

        artifacts[algo_name] = {
            "labels": <list | np.ndarray>,
            "metrics": {...},
            "plot_path": str,
        }

    We identify this shape by requiring ``v`` to be a non-empty dict whose
    values include at least one dict containing a ``"labels"`` key. This is
    strong enough to reject typical config dicts (``{"seed": 42}``) while
    tolerant of missing optional fields like ``metrics`` / ``plot_path``.
    """
    if not isinstance(v, dict) or not v:
        return False
    return any(isinstance(sub, dict) and "labels" in sub for sub in v.values())


# ---------------------------------------------------------------------------
# Memory-monitoring context manager
# ---------------------------------------------------------------------------
class _MemoryWatchdog(threading.Thread):
    """
    Background thread that polls the current process RSS every 0.5 s.
    Sets self.exceeded = True if the RSS *growth* since watchdog start exceeds limit_mb.

    We track the delta (RSS_current - RSS_baseline) rather than absolute RSS so that
    baseline process memory (numpy, torch, etc.) does not count against the limit.
    This makes the memory limit meaningful: "the sandbox code may not allocate more
    than limit_mb bytes on top of the existing process footprint."
    """

    def __init__(self, limit_mb: int, poll_interval: float = 0.5):
        super().__init__(daemon=True)
        self.limit_mb = limit_mb
        self.poll_interval = poll_interval
        self.exceeded = False
        self.peak_delta_mb: float = 0.0
        self._stop_event = threading.Event()
        # Record baseline before sandbox code runs
        try:
            self._baseline_mb: float = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
        except Exception:
            self._baseline_mb = 0.0

    @property
    def peak_mb(self) -> float:
        """Total RSS at peak (baseline + peak delta) — for human-readable messages."""
        return self._baseline_mb + self.peak_delta_mb

    def run(self) -> None:
        proc = psutil.Process(os.getpid())
        while not self._stop_event.is_set():
            try:
                current_mb = proc.memory_info().rss / (1024 * 1024)
                delta_mb = max(0.0, current_mb - self._baseline_mb)
                if delta_mb > self.peak_delta_mb:
                    self.peak_delta_mb = delta_mb
                if delta_mb > self.limit_mb:
                    self.exceeded = True
                    return
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                return
            time.sleep(self.poll_interval)

    def stop(self) -> None:
        self._stop_event.set()


# ---------------------------------------------------------------------------
# Main sandbox class
# ---------------------------------------------------------------------------
class CoderSandbox:
    """
    Execute generated Python code in a restricted namespace with resource guards.

    Args:
        timeout_sec: Wall-clock timeout in seconds (default: ACE_SANDBOX_TIMEOUT_SEC env or 60).
        memory_mb:   RSS memory ceiling in MiB (default: ACE_SANDBOX_MEMORY_MB env or 2048).
    """

    def __init__(
        self,
        timeout_sec: int = _DEFAULT_TIMEOUT_SEC,
        memory_mb: int = _DEFAULT_MEMORY_MB,
    ):
        self.timeout_sec = timeout_sec
        self.memory_mb = memory_mb
        self.output_dir = ""  # set by supervisor before expert dispatch

    # ------------------------------------------------------------------
    # Public execute interface (used by expert agents)
    # ------------------------------------------------------------------

    def execute(
        self,
        code: str,
        X: np.ndarray,
        y: np.ndarray | None = None,
        *,
        pre_inject: dict[str, Any] | None = None,
        display_name: str = "",
        expected_clusters: int = 3,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Execute a code string inside the sandbox namespace.

        Parameters
        ----------
        code : str
            Python source to execute.
        X : np.ndarray
            Feature matrix.
        y : np.ndarray or None
            Optional label / ground-truth vector (may be None).
        pre_inject : dict or None
            Expert-specific name→object mappings injected alongside the
            core pre-injection set.  Takes precedence over CORE_PRE_INJECT
            on key conflict.
        display_name : str
            Dataset display name exposed via ``CTX_DATA.display_name``.
        expected_clusters : int
            Expected number of clusters exposed via ``CTX_DATA.expected_clusters``.

        Returns
        -------
        dict with keys: success (bool), artifacts (dict), error (str|None).
        Raises SandboxResourceExceeded if timeout or memory limit is breached.
        """
        # Build pre-injection: expert overrides core
        merged_pre: dict[str, Any] = dict(CORE_PRE_INJECT)
        if pre_inject:
            merged_pre.update(pre_inject)

        # Last-resort sanitization: convert JSON literals that LLMs leak into
        # Python code.  Expert-layer _sanitize_python_literals() runs earlier,
        # but code that reaches the sandbox through bypass paths (critic audit,
        # ensemble internal runs, etc.) must also be protected.
        import re as _re_sbx
        code = _re_sbx.sub(r'\btrue\b', 'True', code)
        code = _re_sbx.sub(r'\bfalse\b', 'False', code)
        code = _re_sbx.sub(r'\bnull\b', 'None', code)

        artifacts: dict[str, Any] = {}
        exec_env: dict[str, Any] = {
            "__builtins__": SAFE_BUILTINS,
            "__name__": "__ace_sandbox__",
            "X": X,
            "y": y,
            "artifacts": artifacts,
            "np": np,
            "CTX_DATA": DataContext(
                X,
                y,
                display_name=display_name,
                expected_clusters=expected_clusters,
                metadata=metadata,
            ),
            "ACE_OUTPUT_DIR": self.output_dir or "",
            **merged_pre,
        }

        try:
            import matplotlib
            import platform

            matplotlib.use("Agg")  # non-interactive backend, safe in threads
            import matplotlib.pyplot as plt

            if platform.system() == "Windows":
                plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
            plt.rcParams["axes.unicode_minus"] = False
            exec_env["plt"] = plt

        except Exception:
            pass  # matplotlib unavailable — non-fatal

        error_msg: str | None = None
        try:
            self._run_with_limits(code, exec_env)

            # Sync artifacts from exec_env in case the code re-assigned the
            # name (e.g. ``artifacts = {}``). The exec_env binding is the
            # authoritative one after execution.
            artifacts = exec_env.get("artifacts", artifacts)

            if not artifacts:
                # Keep legacy rescue of "result" (most common fallback)
                if "result" in exec_env and _looks_like_artifacts(exec_env["result"]):
                    artifacts = exec_env["result"]
                else:
                    # Broader rescue: scan user-defined vars, skip injected ones
                    # and callables. This catches cases where the LLM wrote
                    # results into an arbitrarily-named dict variable.
                    INJECTED = {"__builtins__", "__name__", "X", "y", "artifacts", "np", "plt"}
                    for name, value in exec_env.items():
                        if name in INJECTED or name.startswith("_") or callable(value):
                            continue
                        if _looks_like_artifacts(value):
                            artifacts = value
                            break

            return {"success": True, "artifacts": artifacts, "error": None}

        except SandboxResourceExceeded:
            raise  # let caller (self-healing loop) handle it

        except Exception as exc:
            error_msg = f"沙箱执行失败: {exc}\n{traceback.format_exc()}"
            logger.error(error_msg)
            return {"success": False, "artifacts": {}, "error": error_msg}

    # ------------------------------------------------------------------
    # Legacy run() interface (backward-compat)
    # ------------------------------------------------------------------

    def run(self, code: str, context: dict[str, Any]) -> SandboxExecution:
        """Legacy interface kept for backward compatibility."""
        exec_env: dict[str, Any] = {
            "__builtins__": SAFE_BUILTINS,
            "__name__": "__ace_sandbox__",
            **context,
        }
        try:
            self._run_with_limits(code, exec_env)
            result = exec_env.get("result", exec_env.get("artifacts", {}))
            return SandboxExecution(code=code, result=result)
        except SandboxResourceExceeded:
            raise
        except Exception as exc:
            err_msg = f"Sandbox execution failed: {exc}\n{traceback.format_exc()}"
            return SandboxExecution(code=code, result={}, error=err_msg)

    # ------------------------------------------------------------------
    # Core execution with limits
    # ------------------------------------------------------------------

    def _run_with_limits(self, code: str, exec_env: dict[str, Any]) -> None:
        """
        Execute ``code`` inside ``exec_env`` enforcing wall-clock timeout
        and memory limits.

        Raises:
            SandboxResourceExceeded: on timeout or memory breach.
            Any exception raised inside the executed code.
        """
        exc_holder: list[BaseException] = []
        done_event = threading.Event()

        def _target() -> None:
            try:
                exec(code, exec_env, exec_env)  # noqa: S102
            except Exception as exc:
                exc_holder.append(exc)
            finally:
                done_event.set()

        watchdog = _MemoryWatchdog(limit_mb=self.memory_mb)
        watchdog.start()

        worker = threading.Thread(target=_target, daemon=True)
        worker.start()

        # Wait up to timeout_sec for the worker to finish
        finished = done_event.wait(timeout=self.timeout_sec)

        watchdog.stop()
        watchdog.join(timeout=1.0)

        if not finished:
            raise SandboxResourceExceeded(
                "timeout",
                f"Execution exceeded {self.timeout_sec}s wall-clock limit.",
            )

        if watchdog.exceeded:
            raise SandboxResourceExceeded(
                "memory",
                f"Sandbox allocated {watchdog.peak_delta_mb:.1f} MiB above baseline, "
                f"exceeding {self.memory_mb} MiB limit (total RSS: {watchdog.peak_mb:.1f} MiB).",
            )

        # Re-raise any exception from the worker thread
        if exc_holder:
            raise exc_holder[0]
