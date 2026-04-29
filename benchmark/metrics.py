"""Clustering quality metrics and self-healing statistics."""
from __future__ import annotations

import math
import re
from typing import Any, Callable

import numpy as np
from sklearn.metrics import (
    adjusted_rand_score,
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)


class ClusteringMetricsCalculator:
    """Compute all standard clustering quality metrics from (X, labels_pred, y_true)."""

    @staticmethod
    def _safe_metric(fn: Callable[..., float], *args: Any, **kwargs: Any) -> float:
        """Return NaN on any exception (degenerate clusters, single label, etc.)."""
        try:
            result = float(fn(*args, **kwargs))
            if math.isfinite(result):
                return result
            return float("nan")
        except Exception:
            return float("nan")

    @staticmethod
    def compute_all(
        X: np.ndarray,
        labels_pred: np.ndarray,
        y_true: np.ndarray | None = None,
    ) -> dict[str, float]:
        """Compute ARI, Silhouette, Calinski-Harabasz, Davies-Bouldin.

        ARI is NaN when y_true is None or labels are degenerate.
        """
        labels = np.asarray(labels_pred).ravel()
        n_unique = len(np.unique(labels))
        if n_unique < 2 or n_unique >= len(labels):
            return {
                "ari": float("nan"),
                "silhouette": float("nan"),
                "calinski_harabasz": float("nan"),
                "davies_bouldin": float("nan"),
            }
        return {
            "ari": ClusteringMetricsCalculator._safe_metric(
                adjusted_rand_score, y_true, labels,
            ) if y_true is not None else float("nan"),
            "silhouette": ClusteringMetricsCalculator._safe_metric(
                silhouette_score, X, labels,
            ),
            "calinski_harabasz": ClusteringMetricsCalculator._safe_metric(
                calinski_harabasz_score, X, labels,
            ),
            "davies_bouldin": ClusteringMetricsCalculator._safe_metric(
                davies_bouldin_score, X, labels,
            ),
        }

    @staticmethod
    def compute_self_healing_stats(logs: list[str]) -> dict[str, Any]:
        """Parse expert.last_logs to extract retry count and success/failure.

        Returns dict with:
            attempts: int       -- max attempt number seen (0 if no log)
            success: bool       -- True if any attempt succeeded
            soft_failures: int  -- count of soft-failure entries
            error: str | None   -- last error message if failed
        """
        attempts = 0
        success = False
        soft_failures = 0
        error: str | None = None

        for line in logs:
            m = re.search(r"第\s*(\d+)\s*次尝试运行代码", line)
            if m:
                attempts = max(attempts, int(m.group(1)))
            if "运行成功" in line:
                success = True
            if "软失败" in line:
                soft_failures += 1
            if "运行失败" in line or "沙箱资源超限" in line:
                error = line[:300]

        if not success and attempts == 0:
            for line in logs:
                if "失败" in line or "异常" in line or "超限" in line:
                    error = line[:300]
                    break

        return {
            "attempts": attempts,
            "success": success,
            "soft_failures": soft_failures,
            "error": error,
        }
