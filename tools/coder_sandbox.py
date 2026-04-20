"""
tools/coder_sandbox.py
======================
Safe Python execution sandbox for ACE Agent (Phase 0).

Security features:
- Restricted builtins (allowlist only)
- Wall-clock timeout via threading.Timer
- Memory upper bound monitored via psutil (Windows-compatible; no resource module needed)
- SandboxResourceExceeded raised with typed reason string: "timeout" | "memory" | "cpu"

Configuration (environment variables or constructor kwargs):
  ACE_SANDBOX_TIMEOUT_SEC   : wall-clock timeout in seconds (default 60)
  ACE_SANDBOX_MEMORY_MB     : RSS memory ceiling in MiB (default 2048 = 2 GiB)

The existing __import__ interception and Chinese font setup are preserved.
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
# Allowed builtins
# ---------------------------------------------------------------------------
SAFE_BUILTINS: dict[str, Any] = {
    "__build_class__": __build_class__,
    "__import__": __import__,
    "abs": abs,
    "all": all,
    "any": any,
    "Exception": Exception,
    "dict": dict,
    "dir": dir,
    "enumerate": enumerate,
    "float": float,
    "getattr": getattr,
    "globals": globals,
    "hasattr": hasattr,
    "int": int,
    "isinstance": isinstance,
    "len": len,
    "list": list,
    "locals": locals,
    "max": max,
    "min": min,
    "object": object,
    "print": print,
    "range": range,
    "round": round,
    "set": set,
    "str": str,
    "sum": sum,
    "super": super,
    "tuple": tuple,
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
    for sub in v.values():
        if isinstance(sub, dict) and "labels" in sub:
            return True
    return False


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

    # ------------------------------------------------------------------
    # Public execute interface (used by expert agents)
    # ------------------------------------------------------------------

    def execute(self, code: str, X: np.ndarray, y: np.ndarray | None = None) -> dict[str, Any]:
        """
        Execute a code string with X/y injected into the namespace.

        Returns a dict with keys: success (bool), artifacts (dict), error (str|None).
        Raises SandboxResourceExceeded if timeout or memory limit is breached.
        """
        artifacts: dict[str, Any] = {}
        exec_env: dict[str, Any] = {
            "__builtins__": SAFE_BUILTINS,
            "__name__": "__ace_sandbox__",
            "X": X,
            "y": y,
            "artifacts": artifacts,
            "np": np,
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

            if not artifacts:
                # First: keep legacy rescue of "result" (most common)
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
