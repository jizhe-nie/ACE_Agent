from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.datasets import (
    make_blobs, make_moons, make_s_curve, 
    load_iris, load_wine, load_digits, 
    fetch_openml, fetch_20newsgroups
)

from ACE_Agent.agent_core.schemas import DatasetBundle

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
    "news": "20 Newsgroups 文本",
    "mfeat": "Multiple Features 多视图手写体",
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

    for col in df.columns:
        if df[col].dtype == object:
            try:
                df[col] = pd.to_numeric(df[col])
            except (ValueError, TypeError):
                pass

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

    return DatasetBundle(
        name="custom",
        display_name=f"自定义数据 ({path.name})",
        X=X.astype(float),
        y=y if y is None else y.astype(int),
        description=f"从文件 {path.name} 加载的自定义数据集，包含 {X.shape[0]} 个样本和 {X.shape[1]} 个数值特征。",
        shape_family="unknown",
        feature_names=feature_names,
        metadata={"file_path": str(path), "original_columns": df.columns.tolist()},
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
        "digits": ["digits", "手写数字特征", "optdigits"],
        "mnist": ["mnist", "原始手写体", "图像聚类"],
        "news": ["news", "newsgroups", "文本聚类", "20news"],
        "mfeat": ["mfeat", "multi-feature", "真实多视图", "多特征"],
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
    
    # 1. 合成数据集逻辑
    if dataset_name == "blobs":
        X, y = make_blobs(n_samples=n_samples, centers=3, n_features=2, cluster_std=[0.9, 1.1, 0.75], random_state=random_state)
        return DatasetBundle(name="blobs", display_name=DATASET_LABELS["blobs"], X=X.astype(float), y=y.astype(int), 
                             description="Three compact Gaussian-like groups. Ideal for centroid methods.", shape_family="spherical", 
                             feature_names=["x1", "x2"], metadata={"expected_clusters": 3})

    if dataset_name == "moons":
        X, y = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
        return DatasetBundle(name="moons", display_name=DATASET_LABELS["moons"], X=X.astype(float), y=y.astype(int),
                             description="两个交错的月牙，具有非凸拓扑结构。", shape_family="non_convex",
                             feature_names=["x1", "x2"], metadata={"expected_clusters": 2})

    if dataset_name == "s_curve":
        X, t = make_s_curve(n_samples=n_samples, noise=noise, random_state=random_state)
        y = np.digitize(t, bins=np.quantile(t, [1/3, 2/3]), right=False)
        return DatasetBundle(name="s_curve", display_name=DATASET_LABELS["s_curve"], X=X.astype(float), y=y.astype(int),
                             description="A curved 3D manifold. Good for embedding plus clustering demos.", shape_family="manifold",
                             feature_names=["x", "y", "z"], metadata={"expected_clusters": 3})

    if dataset_name == "smile":
        return _make_smile_dataset(n_samples=n_samples, noise=noise, random_state=random_state)

    if dataset_name == "high_dim":
        X, y = make_blobs(n_samples=n_samples, centers=5, n_features=100, cluster_std=noise * 10 + 1.0, random_state=random_state)
        return DatasetBundle(name="high_dim", display_name=DATASET_LABELS["high_dim"], X=X.astype(float), y=y.astype(int),
                             description="100维高斯簇，适合测试维度灾难处理。", shape_family="spherical",
                             feature_names=[f"f{i}" for i in range(100)], metadata={"expected_clusters": 5})

    # 2. Scikit-learn 经典数据集
    if dataset_name in ["iris", "wine", "digits"]:
        loader = {"iris": load_iris, "wine": load_wine, "digits": load_digits}[dataset_name]
        data = loader()
        return DatasetBundle(name=dataset_name, display_name=DATASET_LABELS[dataset_name], X=data.data.astype(float), y=data.target.astype(int),
                             description=data.DESCR.split("\n\n")[0], shape_family="spherical",
                             feature_names=getattr(data, "feature_names", [f"f{i}" for i in range(data.data.shape[1])]),
                             metadata={"expected_clusters": len(np.unique(data.target))})

    # 3. 深度聚类/高维数据集
    if dataset_name == "mnist":
        logger.info("正在从 OpenML 下载 MNIST 数据集 (可能耗时)...")
        mnist = fetch_openml("mnist_784", version=1, parser="auto")
        X, y = mnist.data.values[:2000], mnist.target.values[:2000].astype(int)
        return DatasetBundle(name="mnist", display_name=DATASET_LABELS["mnist"], X=X.astype(float), y=y,
                             description="原始手写数字像素数据 (784维)，此处截取前2000个样本。", shape_family="manifold",
                             feature_names=[f"px{i}" for i in range(784)], metadata={"expected_clusters": 10})

    if dataset_name == "news":
        logger.info("正在加载 20 Newsgroups 文本数据并进行 TF-IDF 向量化...")
        from sklearn.feature_extraction.text import TfidfVectorizer
        newsgroups = fetch_20newsgroups(subset="all", remove=("headers", "footers", "quotes"))
        vectorizer = TfidfVectorizer(max_features=1000, stop_words="english")
        X = vectorizer.fit_transform(newsgroups.data[:2000]).toarray()
        y = newsgroups.target[:2000]
        return DatasetBundle(name="news", display_name=DATASET_LABELS["news"], X=X.astype(float), y=y.astype(int),
                             description="20个新闻组的文本聚类数据，已降采样至2000样本，1000维 TF-IDF 特征。", shape_family="sparse",
                             feature_names=vectorizer.get_feature_names_out().tolist(), metadata={"expected_clusters": 20})

    # 4. 真实多视图数据集
    if dataset_name == "mfeat":
        return _load_mfeat_dataset()

    if dataset_name == "multi_view": # 模拟多视图
        return _make_simulated_multi_view(n_samples, noise, random_state)

    raise ValueError(f"Unsupported dataset: {dataset_name}")


def _load_mfeat_dataset() -> DatasetBundle:
    logger.info("正在从 UCI 下载 Multiple Features (mfeat) 数据集...")
    base_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/mfeat/mfeat-"
    views = ["fou", "fac", "kar"] # 傅里叶系数, 轮廓相关, Karhunen-Love
    dfs = []
    for v in views:
        url = base_url + v
        dfs.append(pd.read_csv(url, sep="\s+", header=None))
    
    # 标签：10个类别 (0-9)，每个类别 200 个样本
    y = np.repeat(np.arange(10), 200)
    X = np.hstack([df.values for df in dfs])
    
    feature_names = []
    for i, v in enumerate(views):
        feature_names.extend([f"view{i+1}_{v}_{j}" for j in range(dfs[i].shape[1])])
        
    return DatasetBundle(
        name="mfeat",
        display_name=DATASET_LABELS["mfeat"],
        X=X.astype(float),
        y=y.astype(int),
        description="UCI 真实多视图手写体数据集。View1: 傅里叶, View2: 轮廓, View3: K-L 系数。",
        shape_family="multi_view",
        feature_names=feature_names,
        metadata={"expected_clusters": 10, "view_dims": [df.shape[1] for df in dfs]}
    )


def _make_simulated_multi_view(n_samples: int, noise: float, random_state: int) -> DatasetBundle:
    rng = np.random.default_rng(random_state)
    latent_y = rng.integers(0, 3, size=n_samples)
    latent_X = np.zeros((n_samples, 2))
    for i in range(3):
        latent_X[latent_y == i] = rng.normal(loc=[i*3, i*3], scale=1.0, size=(np.sum(latent_y == i), 2))
    V1 = latent_X @ rng.normal(size=(2, 10)) + rng.normal(scale=noise * 5, size=(n_samples, 10))
    V2 = np.sin(latent_X @ rng.normal(size=(2, 10))) + rng.normal(scale=noise * 5, size=(n_samples, 10))
    X = np.hstack([V1, V2])
    return DatasetBundle(name="multi_view", display_name=DATASET_LABELS["multi_view"], X=X.astype(float), y=None,
                         description="模拟多视角拼接数据（线性视角 + 非线性视角）。", shape_family="manifold",
                         feature_names=[f"v1_{i}" for i in range(10)] + [f"v2_{i}" for i in range(10)],
                         metadata={"expected_clusters": 3})


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
    return DatasetBundle(name="smile", display_name=DATASET_LABELS["smile"], X=X.astype(float), y=y,
                         description="A smiley-face style dataset.", shape_family="non_convex",
                         feature_names=["x1", "x2"], metadata={"expected_clusters": 3})
