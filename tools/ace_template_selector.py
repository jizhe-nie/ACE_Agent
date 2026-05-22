"""
ACE-driven per-class template extraction for heart sound data.

Unlike template_selector.py (which hardcodes TimeSeriesKMeans+DTW), this script
lets ACE Agent's full expert dispatch decide the best clustering algorithm for
each class independently.  The workflow:

  1. Load heart_sounds_ready.csv, split by label → normal / abnormal
  2. For each class, reshape to (N, 63, 64), wrap as DatasetBundle
  3. Call ACESupervisor.run() — full expert pool, routing, audit, ensemble
  4. Extract best result's cluster labels
  5. Compute cluster centers (templates) in the original 3D space
  6. Evaluate template classification accuracy via nearest-template voting

Usage:
  python tools/ace_template_selector.py

Requires:
  - .ace_demo_config.json with valid LLM credentials (populated by Streamlit UI)
  - data/user_uploads/heart_sounds_ready.csv
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT.parent))  # D:\PycharmProject → ACE_Agent package

from ACE_Agent.agent_core.schemas import DatasetBundle  # noqa: E402
from ACE_Agent.agent_core.supervisor import ACESupervisor  # noqa: E402
from ACE_Agent.tools.llm_client import LLMSettings, MultiLLMConfig  # noqa: E402
from ACE_Agent.tools.settings_store import load_settings  # noqa: E402

_OUTPUT_DIR = _PROJECT_ROOT / "templates" / "ace_templates"


def load_and_split_data(csv_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load CSV, split into normal/abnormal, reshape to (N, 63, 64).

    Returns:
        X_normal_3d, X_abnormal_3d, X_normal_flat, X_abnormal_flat
    """
    print(f"Loading {csv_path}...")
    data = np.loadtxt(csv_path, delimiter=",", skiprows=1, dtype=np.float32)
    X_flat = data[:, :-1]   # (N, 4032)
    y = data[:, -1].astype(int)  # (N,)  0=normal, 1=abnormal

    print(f"  Total: {len(y)} samples (normal={int((y == 0).sum())}, abnormal={int((y == 1).sum())})")

    mask_normal = y == 0
    mask_abnormal = y == 1

    X_n_flat = X_flat[mask_normal]   # (N_n, 4032)
    X_a_flat = X_flat[mask_abnormal]  # (N_a, 4032)
    X_n_3d = X_n_flat.reshape(-1, 63, 64)
    X_a_3d = X_a_flat.reshape(-1, 63, 64)

    print(f"  Reshaped: normal {X_n_3d.shape}, abnormal {X_a_3d.shape}")
    return X_n_3d, X_a_3d, X_n_flat, X_a_flat


def build_dataset(X_flat: np.ndarray, class_name: str, ts_shape: tuple) -> DatasetBundle:
    """Build a DatasetBundle for one class with time-series metadata."""
    T, F = ts_shape
    return DatasetBundle(
        name=f"heart_sound_{class_name}",
        X=X_flat,
        y=None,  # No ground-truth sub-labels — pure unsupervised within class
        display_name=f"心音-{class_name} ({T}×{F})",
        description=f"心音{ {'normal': '正常', 'abnormal': '异常'}[class_name] }样本 —— "
                    f"ACE 专家调度探索{ {'normal': '正常', 'abnormal': '异常'}[class_name] }心音的子类型模式",
        metadata={
            "is_time_series": True,
            "is_image": False,  # prevent 4032 = 32×42×3 false positive
            "ts_shape": [T, F],
            "ts_description": f"Mel频谱图 ({F} mel bands × {T} time frames)",
        },
    )


def load_llm_config() -> LLMSettings:
    cfg = load_settings()
    return LLMSettings(
        provider=cfg.get("llm_provider", "DeepSeek"),
        base_url=cfg.get("llm_base_url", ""),
        api_key=cfg.get("llm_api_key", ""),
        model=cfg.get("llm_model", ""),
        temperature=0.2,
        enabled=cfg.get("llm_enabled", True),
    )


def extract_templates(X_3d: np.ndarray, labels: np.ndarray, class_name: str) -> dict:
    """From ACE's clustering result, extract per-cluster templates.

    Args:
        X_3d: (N, 63, 64) time-series data for one class
        labels: (N,) cluster assignment from ACE's best result
        class_name: "normal" or "abnormal"

    Returns:
        dict with templates, cluster_sizes, inertia (DTW), etc.
    """
    from tslearn.metrics import cdist_dtw

    unique_k = np.unique(labels)
    n_clusters = len(unique_k)
    T, F = X_3d.shape[1], X_3d.shape[2]
    templates = np.zeros((n_clusters, T, F), dtype=np.float32)

    cluster_sizes = {}
    total_inertia = 0.0

    for ci, c in enumerate(sorted(unique_k)):
        mask = labels == c
        c_samples = X_3d[mask]
        # Cluster center = barycenter (mean in Euclidean space)
        # For DTW-aware center, we'd need DTW barycenter averaging (DBA).
        # Using the sample closest to the mean as prototype:
        c_mean = c_samples.mean(axis=0, keepdims=True)  # (1, T, F)
        dists = cdist_dtw(c_samples, c_mean).ravel()
        best_i = int(np.argmin(dists))
        templates[ci] = c_samples[best_i]
        total_inertia += float(dists.sum())
        cluster_sizes[int(c)] = {
            "size": int(mask.sum()),
            "pct": float(mask.sum() / len(labels) * 100),
        }

    print(f"  {class_name}: {n_clusters} templates extracted")
    print(f"    Cluster sizes: { {k: v['size'] for k, v in cluster_sizes.items()} }")
    print(f"    Total DTW inertia (to mean): {total_inertia:.0f}")

    return {
        "templates": templates,
        "cluster_sizes": cluster_sizes,
        "inertia": total_inertia,
        "n_clusters": n_clusters,
        "n_samples": len(labels),
    }


def evaluate_templates(
    X_n: np.ndarray, X_a: np.ndarray,
    tmpl_n: np.ndarray, tmpl_a: np.ndarray,
    sample_size: int = 1000,
) -> float:
    """Evaluate template classification accuracy via nearest-template voting."""
    from tslearn.metrics import cdist_dtw

    k_n, k_a = tmpl_n.shape[0], tmpl_a.shape[0]
    all_tmpl = np.vstack([tmpl_n, tmpl_a])  # (k_n+k_a, T, F)
    tmpl_classes = np.array([0] * k_n + [1] * k_a)  # 0=normal, 1=abnormal

    rng = np.random.default_rng(42)
    n_use = min(sample_size, X_n.shape[0])
    idx_n = rng.choice(X_n.shape[0], size=n_use, replace=False)
    n_use_a = min(sample_size, X_a.shape[0])
    idx_a = rng.choice(X_a.shape[0], size=n_use_a, replace=False)

    X_test = np.vstack([X_n[idx_n], X_a[idx_a]])
    y_true = np.array([0] * n_use + [1] * n_use_a)

    D = cdist_dtw(X_test, all_tmpl)
    y_pred = tmpl_classes[D.argmin(axis=1)]
    acc = float((y_pred == y_true).mean())

    print(f"\n  Template classification on {len(y_true)} samples: {acc:.2%}")
    return acc


def main():
    csv_path = _PROJECT_ROOT / "data" / "user_uploads" / "heart_sounds_ready.csv"
    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found")
        sys.exit(1)

    # ---- Load & split -------------------------------------------------------
    X_n_3d, X_a_3d, X_n_flat, X_a_flat = load_and_split_data(str(csv_path))
    ts_shape = (63, 64)

    # ---- LLM config ---------------------------------------------------------
    print("\nLoading LLM config from .ace_demo_config.json...")
    llm = load_llm_config()
    if not llm.is_configured:
        print("ERROR: LLM not configured.  Run the Streamlit UI first to set API keys.")
        sys.exit(1)
    print(f"  Provider: {llm.provider}, Model: {llm.model}")

    # ---- Run ACE on each class ----------------------------------------------
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results = {}
    for class_name, X_3d, X_flat in [
        ("normal", X_n_3d, X_n_flat),
        ("abnormal", X_a_3d, X_a_flat),
    ]:
        print(f"\n{'='*70}")
        print(f"  ACE Agent → {class_name} 心音聚类 ({X_flat.shape[0]} samples, {X_flat.shape[1]}D)")
        print(f"{'='*70}")

        ds = build_dataset(X_flat, class_name, ts_shape)
        supervisor = ACESupervisor()
        prompt = (
            f"对{ {'normal': '正常', 'abnormal': '异常'}[class_name] }心音的 Mel 频谱图进行聚类分析。"
            f"每条数据是 63 时间帧 × 64 Mel 频带的二维时序，已展平为 4032 维向量。"
            f"请自动发现{ {'normal': '正常', 'abnormal': '异常'}[class_name] }心音内部的子类型模式，"
            f"并给出最佳聚类结果。"
        )

        t0 = time.time()
        report = supervisor.run(
            dataset=ds,
            user_prompt=prompt,
            llm_config=MultiLLMConfig(worker=llm),
        )
        elapsed = time.time() - t0
        print(f"\n  ACE 完成 ({elapsed:.0f}s), {len(report.results)} 个结果")

        # ---- Extract best clustering ----------------------------------------
        if not report.ranking:
            print(f"  WARNING: {class_name} 无有效聚类结果，跳过")
            results[class_name] = None
            continue

        best = report.ranking[0]
        print(f"  最佳算法: {best.algorithm_name} (expert={best.expert_key})")
        if best.metrics:
            _sc = best.metrics.get("silhouette_score", "?")
            print(f"  Silhouette: {_sc}")

        # ---- Extract templates from ACE clusters ----------------------------
        tmpl_result = extract_templates(X_3d, best.labels, class_name)
        results[class_name] = tmpl_result

        # Save
        tmpl_path = _OUTPUT_DIR / f"{class_name}_templates.npy"
        np.save(tmpl_path, tmpl_result["templates"])
        print(f"  Templates saved → {tmpl_path}")

        # Save cluster labels
        label_lines = ["sample_index,cluster_id"]
        for i, lbl in enumerate(best.labels):
            label_lines.append(f"{i},{int(lbl)}")
        lbl_path = _OUTPUT_DIR / f"{class_name}_labels.csv"
        lbl_path.write_text("\n".join(label_lines), encoding="utf-8")
        print(f"  Labels saved → {lbl_path}")

    # ---- Evaluate -----------------------------------------------------------
    print(f"\n{'='*70}")
    print("  Template Classification Evaluation")
    print(f"{'='*70}")

    if results.get("normal") and results.get("abnormal"):
        acc = evaluate_templates(
            X_n_3d, X_a_3d,
            results["normal"]["templates"],
            results["abnormal"]["templates"],
        )

        # ---- Summary ---------------------------------------------------------
        summary = {
            "method": "ACE Agent per-class clustering → template extraction",
            "normal": {
                "n_samples": results["normal"]["n_samples"],
                "n_templates": results["normal"]["n_clusters"],
                "cluster_sizes": results["normal"]["cluster_sizes"],
                "dtw_inertia": results["normal"]["inertia"],
            },
            "abnormal": {
                "n_samples": results["abnormal"]["n_samples"],
                "n_templates": results["abnormal"]["n_clusters"],
                "cluster_sizes": results["abnormal"]["cluster_sizes"],
                "dtw_inertia": results["abnormal"]["inertia"],
            },
            "template_classification_accuracy": round(float(acc), 4),
        }
        summary_path = _OUTPUT_DIR / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\n  Summary saved → {summary_path}")
        print(f"\n  Done! ACE 模板提取完成，分类准确率: {acc:.2%}")
    else:
        print("  Cannot evaluate — one or both classes had no results")


if __name__ == "__main__":
    main()
