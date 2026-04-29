import numpy as np
import pandas as pd

# 设置随机种子保证结果可复现
np.random.seed(42)


def generate_rice_data(n_samples=300):
    # 品种 A: 长粒型 (e.g., 籼稻)
    cluster_a = np.random.multivariate_normal(
        mean=[8.5, 2.2, 3.8, 0.85, 25.0], cov=[[0.1, 0.02, 0.05, 0.01, 0.5]] * 5, size=n_samples // 3
    )

    # 品种 B: 圆粒型 (e.g., 粳稻)
    cluster_b = np.random.multivariate_normal(
        mean=[5.5, 3.0, 1.8, 0.70, 28.0], cov=[[0.08, 0.03, 0.04, 0.02, 0.6]] * 5, size=n_samples // 3
    )

    # 品种 C: 中间型或变异品种
    cluster_c = np.random.multivariate_normal(
        mean=[7.0, 2.6, 2.7, 0.78, 26.5], cov=[[0.15, 0.05, 0.1, 0.03, 0.8]] * 5, size=n_samples // 3
    )

    data = np.vstack([cluster_a, cluster_b, cluster_c])
    columns = ["Length", "Width", "Aspect_Ratio", "Transparency", "Weight_per_1000"]
    df = pd.DataFrame(data, columns=columns)

    # 添加真实标签用于后续验证 (测试时可删除此列)
    df["True_Label"] = [0] * 100 + [1] * 100 + [2] * 100

    # 打乱顺序
    df = df.sample(frac=1).reset_index(drop=True)
    return df


df = generate_rice_data()
df.to_csv("test_rice_seeds.csv", index=False)
print("测试文件 'test_rice_seeds.csv' 已生成。")
