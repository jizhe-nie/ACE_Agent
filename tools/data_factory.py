from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.datasets import make_blobs, make_moons, make_s_curve

from ACE_Agent.agent_core.schemas import DatasetBundle


import pandas as pd
from loguru import logger

DATASET_LABELS = {
    "blobs": "团状数据集",
    "moons": "月牙数据集",
    "s_curve": "S型数据集",
    "smile": "笑脸数据集",
    "high_dim": "高维稀疏数据",
    "multi_view": "多视图无标签数据",
    "custom": "自定义上传数据",
}


def load_custom_dataset(file_path: str | Path) -> DatasetBundle:
    path = Path(file_path)
    logger.info(f"正在从文件加载自定义数据: {path}")
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    elif path.suffix.lower() in [".xls", ".xlsx"]:
        df = pd.read_excel(path)
    else:
        raise ValueError(f"不支持的文件格式: {path.suffix}")

    # 尝试自动识别特征和标签
    # 假设最后一列是标签（如果是数值型且唯一值较少），或者全部作为特征
    if df.iloc[:, -1].dtype in [object, int] and df.iloc[:, -1].nunique() < len(df) / 5:
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
        # 尝试编码标签
        from sklearn.preprocessing import LabelEncoder
        y = LabelEncoder().fit_transform(y)
        feature_names = df.columns[:-1].tolist()
    else:
        X = df.values
        y = None
        feature_names = df.columns.tolist()

    return DatasetBundle(
        name="custom",
        display_name=f"自定义数据 ({path.name})",
        X=X.astype(float),
        y=y if y is None else y.astype(int),
        description=f"从文件 {path.name} 加载的自定义数据集，包含 {X.shape[0]} 个样本和 {X.shape[1]} 个特征。",
        shape_family="unknown",
        feature_names=feature_names,
        metadata={"file_path": str(path)},
    )


def infer_dataset_from_prompt(prompt: str) -> str | None:
    normalized = prompt.lower()
    mapping = {
        "blobs": ["blob", "blobs", "球形", "高斯团", "簇团"],
        "moons": ["moon", "moons", "双月", "月牙"],
        "s_curve": ["s-curve", "s curve", "s形", "scurve", "流形"],
        "smile": ["smile", "smiley", "笑脸", "笑脸数据"],
        "high_dim": ["high_dim", "high dim", "高维", "多维", "高维度"],
        "multi_view": ["multi_view", "multi view", "多视图", "多视角", "无标签"],
    }
    for dataset_name, keywords in mapping.items():
        if any(keyword in normalized for keyword in keywords):
            return dataset_name
    return None


def list_demo_datasets() -> list[str]:
    return list(DATASET_LABELS.keys())


def generate_dataset(
    dataset_name: str,
    n_samples: int = 480,
    noise: float = 0.06,
    random_state: int = 42,
) -> DatasetBundle:
    dataset_name = dataset_name.lower()
    if dataset_name == "blobs":
        X, y = make_blobs(
            n_samples=n_samples,
            centers=3,
            n_features=2,
            cluster_std=[0.9, 1.1, 0.75],
            random_state=random_state,
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
        bins = np.quantile(t, [1 / 3, 2 / 3])
        y = np.digitize(t, bins=bins, right=False)
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
            n_samples=n_samples,
            centers=5,
            n_features=100,
            cluster_std=noise * 10 + 1.0,
            random_state=random_state,
        )
        return DatasetBundle(
            name="high_dim",
            display_name=DATASET_LABELS["high_dim"],
            X=X.astype(float),
            y=y.astype(int),
            description="100维空间中的5个簇。直接测距可能会遭遇维度灾难，非常适合考察降维与深度表征专家。",
            shape_family="spherical",
            feature_names=[f"f{i}" for i in range(100)],
            metadata={"expected_clusters": 5},
        )

    if dataset_name == "multi_view":
        rng = np.random.default_rng(random_state)
        # 隐藏的真实标签，但不对聚类算法公开
        latent_y = rng.integers(0, 3, size=n_samples)
        latent_X = np.zeros((n_samples, 2))
        for i in range(3):
            latent_X[latent_y == i] = rng.normal(loc=[i*3, i*3], scale=1.0, size=(np.sum(latent_y == i), 2))
        
        # 视角1：线性特征映射
        W1 = rng.normal(size=(2, 10))
        V1 = latent_X @ W1 + rng.normal(scale=noise * 5, size=(n_samples, 10))
        
        # 视角2：非线性特征映射（如正弦波形变形）
        W2 = rng.normal(size=(2, 10))
        V2 = np.sin(latent_X @ W2) + rng.normal(scale=noise * 5, size=(n_samples, 10))
        
        X = np.hstack([V1, V2])
        
        return DatasetBundle(
            name="multi_view",
            display_name=DATASET_LABELS["multi_view"],
            X=X.astype(float),
            y=None,  # 完全无标签，测试系统自动降级到内部评估的能力
            description="无标签的多视角拼接数据（线性视角 + 非线性视角）。系统将仅依赖轮廓系数等无监督内部指标进行打分评估。",
            shape_family="manifold",
            feature_names=[f"v1_{i}" for i in range(10)] + [f"v2_{i}" for i in range(10)],
            metadata={"expected_clusters": 3},
        )

    raise ValueError(f"Unsupported dataset: {dataset_name}")


def _make_smile_dataset(n_samples: int, noise: float, random_state: int) -> DatasetBundle:
    rng = np.random.default_rng(random_state)
    eye_size = max(n_samples // 6, 40)
    mouth_size = n_samples - (2 * eye_size)

    left_eye = rng.normal(loc=(-1.15, 1.05), scale=(0.11, 0.11), size=(eye_size, 2))
    right_eye = rng.normal(loc=(1.15, 1.05), scale=(0.11, 0.11), size=(eye_size, 2))

    angles = rng.uniform(np.deg2rad(205), np.deg2rad(335), size=mouth_size)
    radii = rng.normal(loc=1.85, scale=0.06, size=mouth_size)
    mouth = np.column_stack(
        [
            radii * np.cos(angles),
            radii * np.sin(angles) - 0.15,
        ]
    )

    X = np.vstack([left_eye, right_eye, mouth])
    X += rng.normal(scale=noise, size=X.shape)
    y = np.concatenate(
        [
            np.zeros(eye_size, dtype=int),
            np.ones(eye_size, dtype=int),
            np.full(mouth_size, 2, dtype=int),
        ]
    )

    return DatasetBundle(
        name="smile",
        display_name=DATASET_LABELS["smile"],
        X=X.astype(float),
        y=y,
        description="A smiley-face style dataset with two eye clusters and one curved mouth cluster.",
        shape_family="non_convex",
        feature_names=["x1", "x2"],
        metadata={"expected_clusters": 3},
    )

