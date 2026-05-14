from __future__ import annotations

import contextlib
import os
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.datasets import (
    fetch_20newsgroups,
    fetch_openml,
    load_digits,
    load_iris,
    load_wine,
    make_blobs,
    make_moons,
    make_s_curve,
)

from ACE_Agent.agent_core.schemas import DatasetBundle

# ---------------------------------------------------------------------------
# Local cache for remote datasets — avoids repeated HTTP fetches from OpenML
# ---------------------------------------------------------------------------
_CACHE_DIR = os.path.join("data", "cache")


def _openml_cache(name: str, version: int = 1, *, cache_key: str | None = None):
    """Fetch from OpenML with local ``.npz`` cache.

    Returns ``(X, y)`` as numpy arrays.  On first call, downloads from
    OpenML and persists to *data/cache/<cache_key>.npz*; subsequent calls
    load directly from disk.
    """
    cache_key = cache_key or name
    cache_path = os.path.join(_CACHE_DIR, f"{cache_key}.npz")
    if os.path.exists(cache_path):
        logger.info("从本地缓存加载 %s (%s)", name, cache_path)
        cached = np.load(cache_path, allow_pickle=True)
        return cached["X"], cached["y"]
    logger.info("从 OpenML 下载 %s (首次，将缓存到 %s)", name, cache_path)
    os.makedirs(_CACHE_DIR, exist_ok=True)
    data = fetch_openml(name, version=version, parser="auto")
    X = np.asarray(data.data, dtype=float)
    y = np.asarray(data.target)
    np.savez_compressed(cache_path, X=X, y=y)
    logger.info("已缓存 %s → %s  (%d x %d)", name, cache_path, X.shape[0], X.shape[1])
    return X, y


DATASET_LABELS = {
    "blobs": "团状数据集",
    "moons": "月牙数据集",
    "s_curve": "S型数据集",
    "smile": "笑脸数据集",
    "high_dim": "高维稀疏数据",
    "multi_view": "多视图模拟数据",
    "iris": "Iris 鸢尾花",
    "wine": "Wine 葡萄酒",
    "digits": "Optdigits 手写数字特征",
    "mnist": "MNIST 原始手写体",
    "mnist_full": "MNIST Full (70K)",
    "fashion_mnist": "Fashion-MNIST (70K)",
    "news": "20 Newsgroups 文本",
    "mfeat": "Multiple Features 多视图手写体",
    # -- 4 classic 2D clustering benchmarks (SIPU + programmatic, 调试保留) --
    "pathbased": "Pathbased 环形缺口",
    "square": "Square 嵌套方形",
    "spiral_sipu": "Spiral 三螺旋线",
    "half_kernel": "Half-kernel 抛物线边界",
    # -- Phase 4: 高维真实聚类基准 --
    "usps": "USPS 手写数字 (256D)",
    "reuters": "Reuters-21578 文本 (2K-D)",
    "har": "HAR 人体活动识别 (561D)",
    "cifar10_raw": "CIFAR-10 原始像素 (3072D)",
    "cifar10_gap": "CIFAR-10 GAP降维 (64D)",
    "cifar10_resnet": "CIFAR-10 ResNet18特征 (512D)",
    "pendigits": "Pendigits 笔迹数字 (16D)",
    "letter": "Letter 字母识别 (16D)",
    "coil20": "COIL-20 物体识别 (1024D)",
    "custom": "自定义上传数据",
}

# ---------------------------------------------------------------------------
# Dataset groups: datasets with multiple feature representations.
# Each group has a label and a list of modes.  A mode's "key" is the flat
# dataset_name understood by generate_dataset().
# ---------------------------------------------------------------------------
DATASET_GROUPS: dict[str, dict] = {
    "cifar10": {
        "label": "CIFAR-10",
        "modes": [
            {
                "key": "cifar10_resnet",
                "label": "ResNet-18 特征 (512D)",
                "recommended": True,
                "desc": "ImageNet预训练CNN，语义特征，聚类效果最好",
            },
            {
                "key": "cifar10_gap",
                "label": "GAP 降维 (64D)",
                "recommended": False,
                "desc": "全局平均池化，速度快但语义弱于ResNet",
            },
            {
                "key": "cifar10_raw",
                "label": "原始像素 (3072D) ⚠️",
                "recommended": False,
                "desc": "像素空间欧氏距离不承载语义，聚类基本无效",
            },
        ],
    },
}

# Reverse lookup: group label → group key
_GROUP_LABEL_TO_KEY: dict[str, str] = {g["label"]: k for k, g in DATASET_GROUPS.items()}

# Set of flat keys that belong to a group (so list_demo_datasets can exclude them)
_GROUPED_FLAT_KEYS: set[str] = {
    m["key"] for g in DATASET_GROUPS.values() for m in g["modes"]
}


def _decompose_image_shape(n_features: int) -> tuple[int, int, int] | None:
    """Try to decompose *n_features* into H×W or H×W×C image dimensions.

    Returns (H, W, C) on success, or None if no reasonable decomposition exists.
    C=1 for grayscale, C=3 for RGB.
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
                    return (h, w, channels)
    return None


def _infer_shape_family(X: np.ndarray, n_features: int) -> str:
    """Quick heuristic to infer shape_family for datasets without metadata.

    Uses PCA variance concentration and dimension count as signals.
    """
    if n_features <= 3:
        # Low-dim data — could be manifold or spherical
        try:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=min(n_features, 2), random_state=42)
            pca.fit(X)
            ratio = pca.explained_variance_ratio_
            # If first component dominates, likely manifold
            if len(ratio) >= 2 and ratio[0] > 0.75:
                return "manifold"
            return "non_convex"
        except Exception:
            return "manifold"

    # For higher dimensions, check PCA variance concentration
    try:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=min(10, n_features), random_state=42)
        pca.fit(X[: min(2000, X.shape[0])])
        cumsum = np.cumsum(pca.explained_variance_ratio_)
        # If 2 components capture < 30% variance → sparse/scattered
        if cumsum[1] < 0.3:
            return "sparse"
        # If 5 components capture > 90% → likely spherical low-rank
        if len(cumsum) >= 5 and cumsum[4] > 0.9:
            return "spherical"
        if len(cumsum) >= 3 and cumsum[2] > 0.8:
            return "spherical"
        return "manifold"
    except Exception:
        return "unknown"


def _analyze_uploaded_data(
    X: np.ndarray, n_features: int, n_samples: int
) -> dict:
    """Analyze uploaded data and return detection results + recommendations.

    Returns a dict with keys:
    - detected_type: "image" | "tabular"
    - image_shape: (H, W, C) tuple if image, else None
    - recommendations: list of dicts with mode/label/reason/priority
    """
    result: dict = {
        "detected_type": "tabular",
        "image_shape": None,
        "recommendations": [],
    }

    # 1. Image detection from feature dimensions
    img_shape = _decompose_image_shape(n_features)
    if img_shape:
        h, w, c = img_shape
        result["detected_type"] = "image"
        result["image_shape"] = img_shape
        if c == 1:
            size_str = f"{h}×{w} (灰度)"
        else:
            size_str = f"{h}×{w}×{c} (RGB)"
        result["recommendations"].append({
            "mode": "cnn_features",
            "label": f"提取 CNN 特征后聚类 ({size_str})",
            "reason": (
                f"检测到图像数据 {size_str}，"
                f"原始像素空间的欧氏距离不承载语义信息。"
                f"建议使用预训练 CNN 提取语义特征后再聚类。"
            ),
            "priority": "strongly_recommended",
        })
        result["recommendations"].append({
            "mode": "raw_pixels",
            "label": f"直接使用原始像素聚类 ({n_features}D) ⚠️",
            "reason": (
                "像素空间聚类在 ≥10D 时失效——类内散度远大于类间差异。"
                "保留此选项仅用于对比验证。"
            ),
            "priority": "not_recommended",
        })

    # 2. Extreme high-dim warning
    if n_features > 500 and not img_shape:
        target_dim = min(64, n_features // 10)
        result["recommendations"].append({
            "mode": "pca_first",
            "label": f"PCA 降至 {target_dim}D 后聚类",
            "reason": (
                f"当前 {n_features} 维，距离矩阵 O(N²D) 计算将极慢"
                f"且受维度灾难影响。建议先 PCA 降维。"
            ),
            "priority": "recommended",
        })

    # 3. Large sample warning
    if n_samples > 10000:
        result["recommendations"].append({
            "mode": "sample_first",
            "label": f"降采样至 {min(5000, n_samples // 4)} 样本",
            "reason": (
                f"当前 {n_samples} 样本，k-NN 图构建 O(N²) 将严重超时。"
                f"建议降采样后再分析。"
            ),
            "priority": "recommended",
        })

    return result


def _extract_cnn_features(
    images: np.ndarray, original_shape: tuple | None = None
) -> np.ndarray:
    """Extract ResNet-18 features from uploaded image data.

    Parameters
    ----------
    images : np.ndarray, shape (N, D)
        Flattened image array.  If *original_shape* is not given the
        function tries to auto-decompose the dimension count.
    original_shape : (H, W) or (H, W, C), optional

    Returns
    -------
    features : np.ndarray, shape (N, 512)
    """
    try:
        import torch
        import torch.nn as nn
        from torchvision import transforms as T
        from torchvision.models import ResNet18_Weights, resnet18
    except ImportError:
        raise ImportError("需要 PyTorch + torchvision 来提取 CNN 特征")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights = ResNet18_Weights.IMAGENET1K_V1 if ResNet18_Weights else None
    model = resnet18(weights=weights)
    model = nn.Sequential(*list(model.children())[:-1])
    model = model.to(device).eval()

    # Reshape: (N, D) → (N, C, H, W)
    n_samples, n_feat = images.shape
    if original_shape and len(original_shape) == 2:
        h, w = original_shape
        c = 1
    elif original_shape and len(original_shape) == 3:
        h, w, c = original_shape
    else:
        shape = _decompose_image_shape(n_feat)
        if shape is None:
            raise ValueError(f"Cannot decompose {n_feat}D into image dimensions")
        h, w, c = shape

    # Reshape to (N, H, W, C) then to (N, C, H, W)
    if c == 1:
        imgs_reshaped = images.reshape(n_samples, h, w)
        imgs_reshaped = np.stack([imgs_reshaped] * 3, axis=1)
    else:
        imgs_reshaped = images.reshape(n_samples, h, w, c).transpose(0, 3, 1, 2)

    transform = T.Compose([
        T.Resize(224),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    batch_size = 256
    features_list = []
    for i in range(0, n_samples, batch_size):
        batch = torch.from_numpy(imgs_reshaped[i: i + batch_size]).float().to(device)
        batch = transform(batch) if batch.shape[-1] >= 32 else transform(
            torch.nn.functional.interpolate(batch, size=(224, 224), mode="bilinear")
        )
        with torch.no_grad():
            feats = model(batch).squeeze(-1).squeeze(-1)
        features_list.append(feats.cpu().numpy())

    return np.concatenate(features_list, axis=0)


def load_custom_dataset(file_path: str | Path) -> DatasetBundle:
    path = Path(file_path)
    logger.info(f"正在从文件加载自定义数据: {path}")
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    elif path.suffix.lower() in [".xls", ".xlsx"]:
        df = pd.read_excel(path)
    else:
        raise ValueError(f"不支持的文件格式: {path.suffix}")

    for col in df.columns:
        if df[col].dtype == object:
            with contextlib.suppress(ValueError, TypeError):
                df[col] = pd.to_numeric(df[col])

    target_col = None
    y = None
    last_col = df.columns[-1]
    if df[last_col].nunique() < len(df) / 5:
        target_col = last_col
        y_raw = df[target_col].values
        from sklearn.preprocessing import LabelEncoder

        try:
            y = LabelEncoder().fit_transform(y_raw.astype(str))
            logger.info(f"自动识别标签列: {target_col}")
        except Exception as e:
            logger.warning(f"标签编码失败: {e}")
            y = None

    feature_df = df.drop(columns=[target_col]) if target_col else df
    X_df = feature_df.select_dtypes(include=[np.number])

    if X_df.isnull().any().any():
        logger.info("检测到缺失值，正在使用均值填充...")
        from sklearn.impute import SimpleImputer

        imputer = SimpleImputer(strategy="mean")
        X_filled = imputer.fit_transform(X_df)
        X_df = pd.DataFrame(X_filled, columns=X_df.columns, index=X_df.index)

    X = X_df.values
    feature_names = X_df.columns.tolist()

    if X.shape[1] == 0:
        raise ValueError("数据集中未找到有效的数值型特征列。")

    n_features = X.shape[1]
    n_samples = X.shape[0]

    # Auto-detect image shape from feature dimensions
    img_shape = _decompose_image_shape(n_features)
    is_image = img_shape is not None
    original_shape = (img_shape[0], img_shape[1]) if img_shape else None
    if is_image:
        logger.info(
            f"自动检测到图像数据: {n_features}D = {img_shape[0]}×{img_shape[1]}"
            f"{'×' + str(img_shape[2]) if img_shape[2] > 1 else ''}"
        )

    # Auto-infer shape_family for custom datasets
    shape_family = _infer_shape_family(X, n_features)
    logger.info(f"推断数据结构类型: {shape_family} (n_features={n_features})")

    # Estimate expected cluster count from label column
    expected_clusters = None
    if y is not None:
        expected_clusters = int(np.unique(y).size)
        if expected_clusters < 2:
            expected_clusters = None

    meta = {
        "file_path": str(path),
        "original_columns": df.columns.tolist(),
        "is_image": is_image,
        "source": "custom_upload",
        "n_samples": n_samples,
    }
    if original_shape:
        meta["original_shape"] = original_shape
    if expected_clusters is not None:
        meta["expected_clusters"] = expected_clusters

    return DatasetBundle(
        name="custom",
        display_name=f"自定义数据 ({path.name})",
        X=X.astype(float),
        y=y if y is None else y.astype(int),
        description=(
            f"从文件 {path.name} 加载的自定义数据集，"
            f"包含 {n_samples} 个样本和 {n_features} 个数值特征。"
            + (f" 自动检测为图像数据 ({img_shape[0]}×{img_shape[1]})。" if is_image else "")
        ),
        shape_family=shape_family,
        feature_names=feature_names,
        metadata=meta,
    )


def infer_dataset_from_prompt(prompt: str) -> str | None:
    normalized = prompt.lower()
    mapping = {
        "blobs": ["blob", "blobs", "球形", "高斯团", "簇团"],
        "moons": ["moon", "moons", "双月", "月牙"],
        "s_curve": ["s-curve", "s curve", "s形", "scurve", "流形"],
        "smile": ["smile", "smiley", "笑脸", "笑脸数据"],
        "high_dim": ["high_dim", "high dim", "高维", "多维", "高维度"],
        "multi_view": ["multi_view", "multi view", "多视图模拟"],
        "iris": ["iris", "鸢尾花", "经典数据"],
        "wine": ["wine", "葡萄酒", "酒"],
        "pendigits": ["pendigits", "pen digit", "笔迹", "手写笔迹"],
        "digits": ["digits", "手写数字特征", "optdigits"],
        "letter": ["letter", "字母识别", "字符识别"],
        "mnist": ["mnist", "原始手写体", "图像聚类"],
        "mnist_full": ["mnist full", "mnist_full", "完整mnist", "手写数字全集"],
        "fashion_mnist": ["fashion", "fashion_mnist", "fashion-mnist", "时尚"],
        "usps": ["usps", "手写数字", "邮政"],
        "reuters": ["reuters", "路透社", "文本", "reuter"],
        "har": ["har", "人体活动", "活动识别", "传感器", "加速度"],
        "news": ["news", "newsgroups", "文本聚类", "20news"],
        "mfeat": ["mfeat", "multi-feature", "真实多视图", "多特征"],
        "pathbased": ["pathbased", "path-based", "path based", "环形", "缺口"],
        "square": ["square", "squares", "nested square", "嵌套方形", "方块"],
        "spiral_sipu": ["spiral", "螺旋", "螺旋线"],
        "half_kernel": ["half-kernel", "half_kernel", "parable", "抛物线", "边界"],
        "cifar10_gap": ["cifar10 gap", "cifar10_gap", "cifar gap"],
        "cifar10_resnet": ["cifar10 resnet", "cifar10_resnet", "cifar resnet"],
        "cifar10_raw": ["cifar10 raw", "cifar10_raw", "cifar raw", "cifar10", "cifar"],
        "coil20": ["coil20", "coil-20", "coil 20", "物体识别", "obj"],
    }
    for dataset_name, keywords in mapping.items():
        if any(keyword in normalized for keyword in keywords):
            return dataset_name
    return None


def list_demo_datasets() -> list[str]:
    """Return keys for the UI selector.

    Flat datasets appear as their raw key.  Multi-mode datasets appear as
    ``group:<group_key>`` so the UI can show a mode picker.
    """
    keys: list[str] = []
    for k in DATASET_LABELS:
        if k == "custom":
            continue
        if k not in _GROUPED_FLAT_KEYS:
            keys.append(k)
    for gk in DATASET_GROUPS:
        keys.append(f"group:{gk}")
    return keys


def generate_dataset(
    dataset_name: str,
    n_samples: int = 480,
    noise: float = 0.06,
    random_state: int = 42,
) -> DatasetBundle:
    dataset_name = dataset_name.lower()

    # 1. 合成数据集逻辑
    if dataset_name == "blobs":
        X, y = make_blobs(
            n_samples=n_samples, centers=3, n_features=2, cluster_std=[0.9, 1.1, 0.75], random_state=random_state
        )
        return DatasetBundle(
            name="blobs",
            display_name=DATASET_LABELS["blobs"],
            X=X.astype(float),
            y=y.astype(int),
            description="Three compact Gaussian-like groups. Ideal for centroid methods.",
            shape_family="spherical",
            feature_names=["x1", "x2"],
            metadata={"expected_clusters": 3},
        )

    if dataset_name == "moons":
        X, y = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
        return DatasetBundle(
            name="moons",
            display_name=DATASET_LABELS["moons"],
            X=X.astype(float),
            y=y.astype(int),
            description="两个交错的月牙，具有非凸拓扑结构。",
            shape_family="non_convex",
            feature_names=["x1", "x2"],
            metadata={"expected_clusters": 2},
        )

    if dataset_name == "s_curve":
        X, t = make_s_curve(n_samples=n_samples, noise=noise, random_state=random_state)
        y = np.digitize(t, bins=np.quantile(t, [1 / 3, 2 / 3]), right=False)
        return DatasetBundle(
            name="s_curve",
            display_name=DATASET_LABELS["s_curve"],
            X=X.astype(float),
            y=y.astype(int),
            description="A curved 3D manifold. Good for embedding plus clustering demos.",
            shape_family="manifold",
            feature_names=["x", "y", "z"],
            metadata={"expected_clusters": 3},
        )

    if dataset_name == "smile":
        return _make_smile_dataset(n_samples=n_samples, noise=noise, random_state=random_state)

    if dataset_name == "high_dim":
        X, y = make_blobs(
            n_samples=n_samples, centers=5, n_features=100, cluster_std=noise * 10 + 1.0, random_state=random_state
        )
        return DatasetBundle(
            name="high_dim",
            display_name=DATASET_LABELS["high_dim"],
            X=X.astype(float),
            y=y.astype(int),
            description="100维高斯簇，适合测试维度灾难处理。",
            shape_family="spherical",
            feature_names=[f"f{i}" for i in range(100)],
            metadata={"expected_clusters": 5},
        )

    # 2. Scikit-learn 经典数据集
    if dataset_name in ["iris", "wine", "digits"]:
        loader = {"iris": load_iris, "wine": load_wine, "digits": load_digits}[dataset_name]
        data = loader()
        return DatasetBundle(
            name=dataset_name,
            display_name=DATASET_LABELS[dataset_name],
            X=data.data.astype(float),
            y=data.target.astype(int),
            description=data.DESCR.split("\n\n")[0],
            shape_family="spherical",
            feature_names=getattr(data, "feature_names", [f"f{i}" for i in range(data.data.shape[1])]),
            metadata={"expected_clusters": len(np.unique(data.target))},
        )

    # 3. 深度聚类/高维数据集
    if dataset_name == "mnist":
        X, y_raw = _openml_cache("mnist_784", version=1, cache_key="mnist")
        X, y = X[:2000], y_raw[:2000].astype(int)
        return DatasetBundle(
            name="mnist",
            display_name=DATASET_LABELS["mnist"],
            X=X.astype(float),
            y=y,
            description="原始手写数字像素数据 (784维)，此处截取前2000个样本。",
            shape_family="manifold",
            feature_names=[f"px{i}" for i in range(784)],
            metadata={"expected_clusters": 10, "is_image": True,
                      "original_shape": (28, 28)},
        )

    if dataset_name in ("mnist_full", "fashion_mnist"):
        from ACE_Agent.benchmark.dataloader import load_benchmark_dataset

        logger.info("正在加载 %s 数据集 (torchvision, 首次需下载)...", dataset_name)
        return load_benchmark_dataset(dataset_name)

    if dataset_name == "news":
        logger.info("正在加载 20 Newsgroups 文本数据并进行 TF-IDF 向量化...")
        from sklearn.feature_extraction.text import TfidfVectorizer

        newsgroups = fetch_20newsgroups(subset="all", remove=("headers", "footers", "quotes"))
        vectorizer = TfidfVectorizer(max_features=1000, stop_words="english")
        X = vectorizer.fit_transform(newsgroups.data[:2000]).toarray()
        y = newsgroups.target[:2000]
        return DatasetBundle(
            name="news",
            display_name=DATASET_LABELS["news"],
            X=X.astype(float),
            y=y.astype(int),
            description="20个新闻组的文本聚类数据，已降采样至2000样本，1000维 TF-IDF 特征。",
            shape_family="sparse",
            feature_names=vectorizer.get_feature_names_out().tolist(),
            metadata={"expected_clusters": 20},
        )

    # 4. 真实多视图数据集
    if dataset_name == "mfeat":
        return _load_mfeat_dataset()

    if dataset_name == "multi_view":  # 模拟多视图
        return _make_simulated_multi_view(n_samples, noise, random_state)

    # 5. SIPU classic 2D clustering benchmarks
    if dataset_name == "pathbased":
        return _load_pathbased()

    if dataset_name == "square":
        return _make_square_dataset(n_samples=800, noise=0.01, random_state=random_state)

    if dataset_name == "spiral_sipu":
        return _load_spiral_sipu()

    if dataset_name == "half_kernel":
        return _make_half_kernel_dataset(n_samples=750, noise=0.02, random_state=random_state)

    # 6. Phase 4: high-dimensional real clustering benchmarks
    if dataset_name == "usps":
        return _load_usps()

    if dataset_name == "reuters":
        return _load_reuters()

    if dataset_name == "har":
        return _load_har()

    if dataset_name in ("cifar10_raw", "cifar10_gap", "cifar10_resnet"):
        mode = dataset_name.replace("cifar10_", "")
        if mode == "resnet":
            mode = "resnet18"
        return _load_cifar10(feature_mode=mode)

    if dataset_name == "pendigits":
        return _load_pendigits()

    if dataset_name == "letter":
        return _load_letter()

    if dataset_name == "coil20":
        return _load_coil20()

    raise ValueError(f"Unsupported dataset: {dataset_name}")


def _load_mfeat_dataset() -> DatasetBundle:
    logger.info("正在从 UCI 下载 Multiple Features (mfeat) 数据集...")
    base_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/mfeat/mfeat-"
    views = ["fou", "fac", "kar"]  # 傅里叶系数, 轮廓相关, Karhunen-Love
    dfs = []
    for v in views:
        url = base_url + v
        dfs.append(pd.read_csv(url, sep=r"\s+", header=None))

    # 标签：10个类别 (0-9)，每个类别 200 个样本
    y = np.repeat(np.arange(10), 200)
    X = np.hstack([df.values for df in dfs])

    feature_names = []
    for i, v in enumerate(views):
        feature_names.extend([f"view{i + 1}_{v}_{j}" for j in range(dfs[i].shape[1])])

    return DatasetBundle(
        name="mfeat",
        display_name=DATASET_LABELS["mfeat"],
        X=X.astype(float),
        y=y.astype(int),
        description="UCI 真实多视图手写体数据集。View1: 傅里叶, View2: 轮廓, View3: K-L 系数。",
        shape_family="multi_view",
        feature_names=feature_names,
        metadata={"expected_clusters": 10, "view_dims": [df.shape[1] for df in dfs]},
    )


def _make_simulated_multi_view(n_samples: int, noise: float, random_state: int) -> DatasetBundle:
    rng = np.random.default_rng(random_state)
    latent_y = rng.integers(0, 3, size=n_samples)
    latent_X = np.zeros((n_samples, 2))
    for i in range(3):
        latent_X[latent_y == i] = rng.normal(loc=[i * 3, i * 3], scale=1.0, size=(np.sum(latent_y == i), 2))
    V1 = latent_X @ rng.normal(size=(2, 10)) + rng.normal(scale=noise * 5, size=(n_samples, 10))
    V2 = np.sin(latent_X @ rng.normal(size=(2, 10))) + rng.normal(scale=noise * 5, size=(n_samples, 10))
    X = np.hstack([V1, V2])
    return DatasetBundle(
        name="multi_view",
        display_name=DATASET_LABELS["multi_view"],
        X=X.astype(float),
        y=None,
        description="模拟多视角拼接数据（线性视角 + 非线性视角）。",
        shape_family="manifold",
        feature_names=[f"v1_{i}" for i in range(10)] + [f"v2_{i}" for i in range(10)],
        metadata={"expected_clusters": 3},
    )


def _make_smile_dataset(n_samples: int, noise: float, random_state: int) -> DatasetBundle:
    rng = np.random.default_rng(random_state)
    eye_size = max(n_samples // 6, 40)
    mouth_size = n_samples - (2 * eye_size)
    left_eye = rng.normal(loc=(-1.15, 1.05), scale=(0.11, 0.11), size=(eye_size, 2))
    right_eye = rng.normal(loc=(1.15, 1.05), scale=(0.11, 0.11), size=(eye_size, 2))
    angles = rng.uniform(np.deg2rad(205), np.deg2rad(335), size=mouth_size)
    radii = rng.normal(loc=1.85, scale=0.06, size=mouth_size)
    mouth = np.column_stack([radii * np.cos(angles), radii * np.sin(angles) - 0.15])
    X = np.vstack([left_eye, right_eye, mouth]) + rng.normal(scale=noise, size=(n_samples, 2))
    y = np.concatenate([np.zeros(eye_size, dtype=int), np.ones(eye_size, dtype=int), np.full(mouth_size, 2, dtype=int)])
    return DatasetBundle(
        name="smile",
        display_name=DATASET_LABELS["smile"],
        X=X.astype(float),
        y=y,
        description="A smiley-face style dataset.",
        shape_family="non_convex",
        feature_names=["x1", "x2"],
        metadata={"expected_clusters": 3},
    )


# ==========================================================================
# SIPU classic 2D clustering benchmark helpers
# ==========================================================================

_BENCHMARK_CACHE = Path(__file__).resolve().parents[1] / "benchmark_cache"
_SIPU_BASE_URL = "https://cs.uef.fi/sipu/datasets"


def _sipu_cache_path(name: str) -> Path:
    return _BENCHMARK_CACHE / f"{name}.txt"


def _download_sipu(name: str) -> Path:
    """Download a SIPU .txt dataset into benchmark_cache/ if not present."""
    cache_path = _sipu_cache_path(name)
    if cache_path.exists():
        logger.info(f"SIPU '{name}' 已有缓存: {cache_path}")
        return cache_path
    _BENCHMARK_CACHE.mkdir(parents=True, exist_ok=True)
    url = f"{_SIPU_BASE_URL}/{name}.txt"
    logger.info(f"正在从 SIPU 下载 '{name}' → {url}")
    urllib.request.urlretrieve(url, cache_path)
    logger.info(f"下载完成: {cache_path} ({cache_path.stat().st_size} bytes)")
    return cache_path


# ------------------------------------------------------------------
# Dataset: Pathbased (SIPU)
# 300 samples, 2D + labels, 3 classes
# Shape: a circular ring with a gap + 3 inner high-density clusters
# ------------------------------------------------------------------
def _load_pathbased() -> DatasetBundle:
    path = _download_sipu("pathbased")
    data = pd.read_csv(path, sep=r"\s+", header=None).values
    X = data[:, :2].astype(float)
    y = data[:, 2].astype(int) - 1  # SIPU labels are 1-indexed
    return DatasetBundle(
        name="pathbased",
        display_name=DATASET_LABELS["pathbased"],
        X=X,
        y=y,
        description="Pathbased: 一个带缺口的环形外框 + 3 个高密度内部簇。"
                    "测试算法对非凸形状和多密度簇的识别能力。",
        shape_family="non_convex",
        feature_names=["x1", "x2"],
        metadata={"expected_clusters": 3, "source": "SIPU", "n_samples": 300},
    )


# ------------------------------------------------------------------
# Dataset: Spiral (SIPU)
# 312 samples, 2D + labels, 3 classes
# Shape: 3 interleaving Archimedean spirals
# ------------------------------------------------------------------
def _load_spiral_sipu() -> DatasetBundle:
    path = _download_sipu("spiral")
    data = pd.read_csv(path, sep=r"\s+", header=None).values
    X = data[:, :2].astype(float)
    y = data[:, 2].astype(int) - 1  # SIPU labels are 1-indexed
    return DatasetBundle(
        name="spiral_sipu",
        display_name=DATASET_LABELS["spiral_sipu"],
        X=X,
        y=y,
        description="Spiral: 三条交错的阿基米德螺旋线。"
                    "测试基于图/连通性的算法处理复杂流形的能力。",
        shape_family="manifold",
        feature_names=["x1", "x2"],
        metadata={"expected_clusters": 3, "source": "SIPU", "n_samples": 312},
    )


# ------------------------------------------------------------------
# Dataset: Square / Nested Squares (programmatic)
# ~800 samples, 2D + labels, 5 classes
# Shape: outer square frame + 4 inner square blocks in 2×2 grid
# ------------------------------------------------------------------
def _make_square_dataset(
    n_samples: int = 800, noise: float = 0.01, random_state: int = 42
) -> DatasetBundle:
    rng = np.random.default_rng(random_state)
    # Outer square frame (4 segments with a gap in top side)
    frame_n = n_samples // 3
    inner_n = (n_samples - frame_n) // 4
    pieces: list[np.ndarray] = []
    labels_list: list[np.ndarray] = []

    def _line_segment(start, end, n, noise_std):
        t = rng.uniform(0, 1, n).reshape(-1, 1)
        pts = start + t * (end - start)
        return pts + rng.normal(0, noise_std, pts.shape)

    # Outer frame: bottom, right, top (with gap), left
    frame_pts = [
        _line_segment(np.array([-3, -3]), np.array([3, -3]), frame_n // 4, noise * 4),
        _line_segment(np.array([3, -3]), np.array([3, 3]), frame_n // 4, noise * 4),
        _line_segment(np.array([-1.5, 3]), np.array([1.5, 3]), frame_n // 4, noise * 4),  # top with gap
        _line_segment(np.array([-3, -3]), np.array([-3, 3]), frame_n // 4, noise * 4),
    ]
    pieces.extend(frame_pts)
    labels_list.extend([np.full(len(p), 0) for p in frame_pts])

    # 4 inner square blocks centred at (±1.5, ±1.5)
    inner_centers = [(-1.5, -1.5), (1.5, -1.5), (-1.5, 1.5), (1.5, 1.5)]
    for i, (cx, cy) in enumerate(inner_centers):
        block = rng.uniform(low=-0.6, high=0.6, size=(inner_n, 2))
        block[:, 0] += cx
        block[:, 1] += cy
        block += rng.normal(0, noise * 2, block.shape)
        pieces.append(block)
        labels_list.append(np.full(inner_n, i + 1))

    X = np.vstack(pieces)
    y = np.concatenate(labels_list)
    return DatasetBundle(
        name="square",
        display_name=DATASET_LABELS["square"],
        X=X.astype(float),
        y=y.astype(int),
        description="Square: 外层方形边框(顶部有缺口) + 4 个内嵌方块（2×2 排列）。"
                    "测试算法在不规则形状、间隙和非高斯分布上的鲁棒性。",
        shape_family="non_convex",
        feature_names=["x1", "x2"],
        metadata={"expected_clusters": 5},
    )


# ------------------------------------------------------------------
# Dataset: Half-kernel / Parabola (programmatic)
# ~750 samples, 2D + labels, 4 classes
# Shape: parabolic boundary + 3 inner Gaussian clusters
# ------------------------------------------------------------------
def _make_half_kernel_dataset(
    n_samples: int = 750, noise: float = 0.02, random_state: int = 42
) -> DatasetBundle:
    rng = np.random.default_rng(random_state)
    # 3 inner Gaussian clusters
    n_inner = 3 * (n_samples // 5)
    inner_labels = rng.integers(0, 3, n_inner)
    centers = np.array([[-1.0, -0.5], [1.0, -0.5], [0.0, 1.0]])
    X_inner = rng.normal(0, 0.25, (n_inner, 2)) + centers[inner_labels]

    # Parabolic boundary: y = x² - 2, sampling along the curve
    n_boundary = n_samples - n_inner
    x_boundary = rng.uniform(-2.5, 2.5, n_boundary)
    y_boundary = x_boundary ** 2 - 1.8
    X_boundary = np.column_stack([x_boundary, y_boundary])
    X_boundary += rng.normal(0, noise * 3, X_boundary.shape)

    X = np.vstack([X_inner, X_boundary])
    y = np.concatenate([inner_labels, np.full(n_boundary, 3)])
    return DatasetBundle(
        name="half_kernel",
        display_name=DATASET_LABELS["half_kernel"],
        X=X.astype(float),
        y=y.astype(int),
        description="Half-kernel: 抛物线 y=x² 噪声边界 + 3 个高斯内簇。"
                    "测试算法对曲线边界和多密度的区分能力。",
        shape_family="non_convex",
        feature_names=["x1", "x2"],
        metadata={"expected_clusters": 4},
    )


# ==========================================================================
# Phase 4: High-dimensional real clustering benchmark loaders
# ==========================================================================


def _load_usps() -> DatasetBundle:
    """USPS handwritten digits. 256 features, 9298 samples, 10 classes."""
    X, y_raw = _openml_cache("usps", version=1, cache_key="usps")
    y = np.asarray(y_raw, dtype=int)
    return DatasetBundle(
        name="usps",
        display_name=DATASET_LABELS["usps"],
        X=X,
        y=y,
        description="USPS 手写数字数据集：256 维灰度像素特征，9298 样本，10 类。",
        shape_family="spherical",
        feature_names=[f"px{i}" for i in range(256)],
        metadata={"expected_clusters": 10, "source": "OpenML/USPS", "n_samples": len(X)},
    )


def _load_reuters() -> DatasetBundle:
    """Reuters-21578 TF-IDF text clustering. ~2000 features, 4 classes."""
    logger.info("正在加载 Reuters-21578 数据集...")
    from sklearn.feature_extraction.text import TfidfVectorizer

    try:
        import nltk
        nltk_data_dir = os.path.join(os.path.expanduser("~"), "nltk_data")
        nltk.data.path.append(nltk_data_dir)
        nltk.download("reuters", quiet=True)
        nltk.download("punkt", quiet=True)
        from nltk.corpus import reuters as _reuters
        fileids = _reuters.fileids()
        categories = _reuters.categories()
        target_cats = sorted(categories)[:4]
        docs, labels = [], []
        for fid in fileids:
            cat_set = set(_reuters.categories(fid))
            matched = [c for c in target_cats if c in cat_set]
            if matched:
                docs.append(" ".join(_reuters.words(fid)))
                labels.append(target_cats.index(matched[0]))
    except Exception:
        logger.warning("NLTK Reuters 加载失败，尝试 sklearn fetch_openml 回退...")
        openml_attempts = [
            {"name": "Reuters-21578", "version": 1},
            {"name": "reuters-21578", "version": 1},
            {"data_id": 433},
        ]
        # Try local cache first (from prior successful OpenML download)
        _reuters_cache = os.path.join(_CACHE_DIR, "reuters.npz")
        if os.path.exists(_reuters_cache):
            logger.info("从本地缓存加载 reuters (%s)", _reuters_cache)
            _c = np.load(_reuters_cache, allow_pickle=True)
            X = _c["X"]
            y = _c["y"]
        else:
            data = None
            for attempt in openml_attempts:
                try:
                    data = fetch_openml(parser="auto", **attempt)
                    break
                except Exception:
                    continue
            if data is not None:
                X_raw = data.data
                y_raw = data.target
                if hasattr(X_raw, "toarray"):
                    X_raw = X_raw.toarray()
                X = np.asarray(X_raw, dtype=float)
                y = np.asarray(pd.factorize(y_raw)[0], dtype=int)
                os.makedirs(_CACHE_DIR, exist_ok=True)
                np.savez_compressed(_reuters_cache, X=X, y=y)
                logger.info("已缓存 reuters → %s", _reuters_cache)
            else:
                X = None
                y = None
        if X is not None and y is not None:
            return DatasetBundle(
                name="reuters",
                display_name=DATASET_LABELS["reuters"],
                X=X,
                y=y,
                description="Reuters-21578 文本聚类：~2000 维 TF-IDF 特征，~10000 样本，4 类。",
                shape_family="sparse",
                feature_names=[f"word_{i}" for i in range(X.shape[1])],
                metadata={"expected_clusters": 4, "source": "OpenML/Reuters",
                          "n_samples": len(X)},
            )
        # Ultimate fallback: synthetic sparse text-like data
        logger.warning("OpenML Reuters 回退也失败，生成合成稀疏文本数据作为兜底...")
        rng = np.random.RandomState(42)
        n_samples = 1000
        n_features = 500
        X = rng.uniform(0, 1, (n_samples, n_features)) * (
            rng.uniform(0, 1, n_features) > 0.85
        )
        y = rng.randint(0, 4, n_samples)
        return DatasetBundle(
            name="reuters",
            display_name=DATASET_LABELS["reuters"],
            X=X,
            y=y,
            description="Reuters-21578 替代数据（原数据集暂不可用）。",
            shape_family="sparse",
            metadata={"expected_clusters": 4, "source": "synthetic_fallback",
                      "n_samples": n_samples},
        )

    vectorizer = TfidfVectorizer(max_features=2000, stop_words="english")
    X = vectorizer.fit_transform(docs).toarray()
    y = np.array(labels, dtype=int)
    return DatasetBundle(
        name="reuters",
        display_name=DATASET_LABELS["reuters"],
        X=X.astype(float),
        y=y,
        description="Reuters-21578 文本聚类：~2000 维 TF-IDF 特征，~10000 样本，4 类。",
        shape_family="sparse",
        feature_names=vectorizer.get_feature_names_out().tolist(),
        metadata={"expected_clusters": 4, "source": "NLTK/Reuters", "n_samples": len(X)},
    )


def _load_har() -> DatasetBundle:
    """Human Activity Recognition. 561 features, 10299 samples, 6 classes."""
    X, y_raw = _openml_cache("har", version=1, cache_key="har")
    y = pd.factorize(y_raw)[0]
    return DatasetBundle(
        name="har",
        display_name=DATASET_LABELS["har"],
        X=X,
        y=np.asarray(y, dtype=int),
        description="HAR 人体活动识别：561 维传感器时序特征，10299 样本，6 类。",
        shape_family="spherical",
        feature_names=[f"feat_{i}" for i in range(561)],
        metadata={"expected_clusters": 6, "source": "UCI/HAR", "n_samples": len(X)},
    )


def _load_cifar10(feature_mode: str = "raw") -> DatasetBundle:
    """CIFAR-10 with three feature modes: raw(3072D) / gap(64D) / resnet18(512D)."""
    logger.info("正在加载 CIFAR-10 数据集 (mode=%s)...", feature_mode)
    try:
        import torch
        import torchvision
        import torchvision.transforms as T
    except ImportError:
        raise ImportError(
            "CIFAR-10 需要 PyTorch + torchvision。请安装: pip install torch torchvision"
        )

    if feature_mode in ("gap", "resnet18"):
        import torch.nn as nn

    from sklearn.preprocessing import LabelEncoder

    transform = T.Compose([T.ToTensor()])
    trainset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
    images = np.concatenate(
        [trainset.data, testset.data], axis=0
    ).astype(np.float32)
    labels_raw = np.concatenate([trainset.targets, testset.targets], axis=0)
    y = LabelEncoder().fit_transform(labels_raw.astype(str))

    _is_image = False
    _original_shape = None
    if feature_mode == "raw":
        X = images.reshape(images.shape[0], -1)
        feat_names = [f"px{i}" for i in range(3072)]
        n_feat = 3072
        _is_image = True
        _original_shape = (32, 32)
        _image_channels = 3
    elif feature_mode == "gap":
        # 8x8 global average pooling on each channel
        X_gap = images.reshape(images.shape[0], 3, 32, 32)
        pooled = nn.AvgPool2d(kernel_size=4, stride=4)(torch.from_numpy(X_gap))
        X = pooled.reshape(images.shape[0], -1).numpy()
        n_feat = X.shape[1]
        feat_names = [f"gap_{i}" for i in range(n_feat)]
    elif feature_mode == "resnet18":
        # Pre-trained ResNet-18 features (penultimate layer)
        try:
            from torchvision.models import ResNet18_Weights, resnet18
        except ImportError:
            from torchvision.models import resnet18
            ResNet18_Weights = None

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        weights = ResNet18_Weights.IMAGENET1K_V1 if ResNet18_Weights else None
        model = resnet18(weights=weights)
        model = nn.Sequential(*list(model.children())[:-1])
        model = model.to(device).eval()
        n_feat = 512
        feat_names = [f"resnet_{i}" for i in range(n_feat)]

        # Batch inference
        transform_resnet = T.Compose([T.Resize(224), T.ToTensor(),
                                       T.Normalize(mean=[0.485, 0.456, 0.406],
                                                   std=[0.229, 0.224, 0.225])])
        batch_size = 512
        features_list = []
        for i in range(0, len(images), batch_size):
            batch_imgs = images[i: i + batch_size]
            batch_tensors = torch.stack(
                [transform_resnet(T.ToPILImage()(img.astype(np.uint8)))
                 for img in batch_imgs]
            ).to(device)
            with torch.no_grad():
                feats = model(batch_tensors).squeeze(-1).squeeze(-1)
            features_list.append(feats.cpu().numpy())
        X = np.concatenate(features_list, axis=0)
    else:
        raise ValueError(f"Unknown CIFAR-10 feature_mode: {feature_mode}")

    return DatasetBundle(
        name=f"cifar10_{feature_mode}",
        display_name=DATASET_LABELS.get(
            f"cifar10_{feature_mode}",
            DATASET_LABELS.get("cifar10_resnet", f"CIFAR-10 {feature_mode}")
        ),
        X=X.astype(float),
        y=y.astype(int),
        description=f"CIFAR-10 ({feature_mode} 特征): {n_feat} 维，60000 样本，10 类。",
        shape_family="manifold",
        feature_names=feat_names,
        feature_mode=feature_mode,
        metadata={"expected_clusters": 10, "source": "torchvision/CIFAR-10",
                  "n_samples": len(X), "feature_mode": feature_mode,
                  "is_image": _is_image,
                  "original_shape": _original_shape},
    )


def _load_pendigits() -> DatasetBundle:
    """Pendigits pen-based handwritten digits. 16 features, 10992 samples, 10 classes."""
    X, y_raw = _openml_cache("pendigits", version=1, cache_key="pendigits")
    y = pd.factorize(y_raw)[0]
    return DatasetBundle(
        name="pendigits",
        display_name=DATASET_LABELS["pendigits"],
        X=X,
        y=np.asarray(y, dtype=int),
        description="Pendigits 笔迹手写数字：16 维笔划特征，10992 样本，10 类。",
        shape_family="spherical",
        feature_names=[f"feat_{i}" for i in range(16)],
        metadata={"expected_clusters": 10, "source": "UCI/Pendigits", "n_samples": len(X)},
    )


def _load_letter() -> DatasetBundle:
    """Letter Recognition. 16 features, 20000 samples, 26 classes."""
    X, y_raw = _openml_cache("letter", version=1, cache_key="letter")
    y = pd.factorize(y_raw)[0]
    return DatasetBundle(
        name="letter",
        display_name=DATASET_LABELS["letter"],
        X=X,
        y=np.asarray(y, dtype=int),
        description="Letter 字母识别：16 维像素统计特征，20000 样本，26 类。",
        shape_family="spherical",
        feature_names=[f"feat_{i}" for i in range(16)],
        metadata={"expected_clusters": 26, "source": "UCI/Letter", "n_samples": len(X)},
    )


def _load_coil20() -> DatasetBundle:
    """COIL-20 object recognition. ~1024 features (HOG/pixel), 1440 samples, 20 classes."""
    try:
        X, y_raw = _openml_cache("coil20", version=1, cache_key="coil20")
    except Exception:
        X, y_raw = _openml_cache("COIL-20", version=1, cache_key="coil20")
    y = pd.factorize(y_raw)[0]
    return DatasetBundle(
        name="coil20",
        display_name=DATASET_LABELS["coil20"],
        X=X,
        y=y.astype(int),
        description="COIL-20 物体识别：~1024 维像素/HOG 特征，1440 样本，20 个物体类别。",
        shape_family="manifold",
        feature_names=[f"feat_{i}" for i in range(X.shape[1])],
        metadata={"expected_clusters": 20, "source": "Columbia/COIL-20",
                  "n_samples": len(X), "is_image": True,
                  "original_shape": (32, 32)},
    )
