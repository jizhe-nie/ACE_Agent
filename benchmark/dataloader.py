"""
benchmark/dataloader.py
=======================
Dynamic data loading for large-scale benchmark datasets.

Supports:
- **torchvision** standard datasets (MNIST, Fashion-MNIST) with automatic
  download, [0,1] normalisation, flatten, and NPY disk caching.
- **Pre-extracted features** from arbitrary pipelines (e.g. ResNet50 2048-dim)
  via ``load_from_npy()``.

All functions return a ``DatasetBundle`` so the BenchmarkRunner works
without modification.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from ACE_Agent.agent_core.schemas import DatasetBundle

_logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Torchvision dataset registry
# ---------------------------------------------------------------------------
_TORCHVISION_CONFIG: dict[str, dict[str, Any]] = {
    "mnist_full": {
        "torch_name": "MNIST",
        "n_classes": 10,
        "image_shape": (28, 28),
        "n_features": 784,
        "display_name": "MNIST Full (70K × 784)",
        "description": "MNIST handwritten digits. 70,000 samples, 28×28 grayscale images flattened to 784-D. 10 classes (0-9).",
        "expected_clusters": 10,
    },
    "fashion_mnist": {
        "torch_name": "FashionMNIST",
        "n_classes": 10,
        "image_shape": (28, 28),
        "n_features": 784,
        "display_name": "Fashion-MNIST (70K × 784)",
        "description": "Zalando Fashion-MNIST. 70,000 samples, 28×28 grayscale images flattened to 784-D. 10 classes (T-shirt, Trouser, etc.).",
        "expected_clusters": 10,
    },
}

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_benchmark_dataset(
    name: str,
    cache_dir: str = "benchmark_cache",
) -> DatasetBundle:
    """Load a benchmark dataset by name; cache as NPY on disk.

    Supported names: ``"mnist_full"``, ``"fashion_mnist"``.

    Parameters
    ----------
    name : str
        Dataset key (must be in ``_TORCHVISION_CONFIG``).
    cache_dir : str
        Directory for NPY cache files.  Created if missing.

    Returns
    -------
    DatasetBundle
    """
    cfg = _TORCHVISION_CONFIG.get(name)
    if cfg is None:
        raise ValueError(
            f"Unknown benchmark dataset '{name}'. Supported: "
            f"{list(_TORCHVISION_CONFIG)}."
        )

    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    x_path = cache_path / f"{name}_X.npy"
    y_path = cache_path / f"{name}_y.npy"

    if x_path.exists() and y_path.exists():
        _logger.info("Loading %s from NPY cache (%s).", name, cache_path)
        X = np.load(x_path)
        y = np.load(y_path)
    else:
        _logger.info("Downloading %s via torchvision...", name)
        X, y = _download_torchvision(cfg)
        np.save(x_path, X)
        np.save(y_path, y)
        _logger.info("Cached %s to %s (%d samples, %d features).",
                      name, cache_path, X.shape[0], X.shape[1])

    return DatasetBundle(
        name=name,
        X=X.astype(np.float32),
        y=y.astype(np.int64),
        display_name=cfg["display_name"],
        description=cfg["description"],
        shape_family="manifold",
        metadata={
            "expected_clusters": cfg["expected_clusters"],
            "n_classes": cfg["n_classes"],
            "is_image": True,
            "source": f"torchvision.{cfg['torch_name']}",
            "original_shape": cfg["image_shape"],
        },
    )


def load_from_npy(
    name: str,
    features_path: str,
    labels_path: str | None = None,
    *,
    cache_dir: str = "benchmark_cache",
    display_name: str = "",
    description: str = "",
    expected_clusters: int | None = None,
    feature_extractor: str = "",
) -> DatasetBundle:
    """Load a pre-extracted feature matrix (.npy or .npz).

    Use this to inject high-dimensional visual features from external
    pipelines (e.g. ResNet50 2048-D embeddings).

    Parameters
    ----------
    name : str
        Internal dataset key.
    features_path : str
        Path to a ``.npy`` file (shape ``(N, D)``) or ``.npz`` archive
        (must contain key ``"features"`` and optionally ``"labels"``).
    labels_path : str or None
        Optional path to a ``.npy`` file of integer labels ``(N,)``.
        Overrides labels contained in ``features_path`` if that is a
        ``.npz``.
    display_name : str
        Human-readable name.
    expected_clusters : int or None
        Number of ground-truth clusters (used by clustering algorithms
        that require *k* upfront).

    Returns
    -------
    DatasetBundle
    """
    fp = Path(features_path)
    if not fp.exists():
        raise FileNotFoundError(f"Feature file not found: {fp}")

    y = None
    if fp.suffix == ".npz":
        archive = np.load(fp)
        X = archive["features"]
        y = archive.get("labels", None)
        if y is not None:
            y = y.astype(np.int64).ravel()
    else:
        X = np.load(fp)

    if labels_path:
        y = np.load(labels_path).astype(np.int64).ravel()

    n_features = X.shape[1]
    if not display_name:
        display_name = f"{name} ({X.shape[0]} × {n_features})"
    if expected_clusters is None and y is not None:
        expected_clusters = int(len(np.unique(y)))

    metadata: dict[str, Any] = {
        "expected_clusters": expected_clusters or 3,
        "source": str(fp),
        "feature_extractor": feature_extractor,
    }

    cache_dir_path = Path(cache_dir)
    cache_dir_path.mkdir(parents=True, exist_ok=True)
    np.save(cache_dir_path / f"{name}_X.npy", X)
    if y is not None:
        np.save(cache_dir_path / f"{name}_y.npy", y)

    return DatasetBundle(
        name=name,
        X=X.astype(np.float32),
        y=y.astype(np.int64) if y is not None else None,
        display_name=display_name,
        description=description or f"Pre-extracted features: {fp.name}",
        shape_family="image_features",
        metadata=metadata,
    )


def is_large_dataset(dataset: DatasetBundle) -> bool:
    """Return True if the dataset warrants O(N²) algorithm skipping."""
    n = dataset.X.shape[0]
    d = dataset.X.shape[1]
    return n > 10000 or d > 512


def dataset_size_label(dataset: DatasetBundle) -> str:
    """Human-readable size label: ``"[SMALL]"`` / ``"[LARGE]"`` / ``"[HUGE]"``."""
    n = dataset.X.shape[0]
    if n > 50000:
        return "[HUGE]"
    if n > 10000:
        return "[LARGE]"
    return "[SMALL]"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _download_torchvision(cfg: dict) -> tuple[np.ndarray, np.ndarray]:  # type: ignore[type-arg]
    """Download a torchvision dataset and return (X_flat, y)."""
    from torchvision import datasets, transforms  # noqa: E402

    transform = transforms.Compose(
        [
            transforms.ToTensor(),  # PIL [0,255] → float [0,1]; NO Normalize — image AE needs sparse background
        ]
    )
    torch_name: str = cfg["torch_name"]
    ds_cls = getattr(datasets, torch_name)

    # Load both train and test sets, concatenate
    train = ds_cls(root="./data", train=True, download=True, transform=transform)
    test = ds_cls(root="./data", train=False, download=True, transform=transform)

    images: list[np.ndarray] = []
    labels: list[int] = []
    for ds in (train, test):
        for img, lbl in ds:  # type: ignore[var-annotated]
            images.append(img.numpy().ravel())  # 28×28 → 784
            labels.append(int(lbl))

    return np.vstack(images), np.array(labels)
