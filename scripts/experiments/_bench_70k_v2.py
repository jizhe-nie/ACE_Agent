"""70K MNIST — high-capacity Conv-AE benchmark."""
import logging
import time

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")

from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from ACE_Agent.benchmark.dataloader import load_benchmark_dataset

ds = load_benchmark_dataset("mnist_full")
X, y = ds.X, ds.y
print(f"X: {X.shape}")

results = {}

# === 1. Conv-AE + GMM (base_filters=64, dim=32, epochs=200) ===
print("\n" + "=" * 70)
print("[1/3] Conv-AE + GMM (filters=64, dim=32, epochs=200)")
from ACE_Agent.tools.ae_pipeline import conv_ae_kmeans_pipeline

t0 = time.time()
r = conv_ae_kmeans_pipeline(
    X, k=10, latent_dim=32, epochs=200, base_filters=64,
    cluster_method="gmm", normalize="minmax",
)
t1 = time.time()
ari = adjusted_rand_score(y, np.array(r["labels"]))
nmi = normalized_mutual_info_score(y, np.array(r["labels"]))
print(f"  ARI={ari:.4f} NMI={nmi:.4f} Sil={r['metrics']['silhouette']:.4f} time={t1 - t0:.0f}s")
results["Conv_f64_d32"] = (ari, nmi)

# === 2. Conv-AE + KMeans (filters=64, dim=32, epochs=200) ===
print("\n" + "=" * 70)
print("[2/3] Conv-AE + KMeans (filters=64, dim=32, epochs=200)")
t0 = time.time()
r = conv_ae_kmeans_pipeline(
    X, k=10, latent_dim=32, epochs=200, base_filters=64,
    cluster_method="kmeans", normalize="minmax",
)
t1 = time.time()
ari = adjusted_rand_score(y, np.array(r["labels"]))
nmi = normalized_mutual_info_score(y, np.array(r["labels"]))
print(f"  ARI={ari:.4f} NMI={nmi:.4f} Sil={r['metrics']['silhouette']:.4f} time={t1 - t0:.0f}s")
results["Conv_f64_d32_KM"] = (ari, nmi)

# === 3. Conv-AE + GMM (filters=64, dim=32, epochs=300) ===
print("\n" + "=" * 70)
print("[3/3] Conv-AE + GMM (filters=64, dim=32, epochs=300)")
t0 = time.time()
r = conv_ae_kmeans_pipeline(
    X, k=10, latent_dim=32, epochs=300, base_filters=64,
    cluster_method="gmm", normalize="minmax",
)
t1 = time.time()
ari = adjusted_rand_score(y, np.array(r["labels"]))
nmi = normalized_mutual_info_score(y, np.array(r["labels"]))
print(f"  ARI={ari:.4f} NMI={nmi:.4f} Sil={r['metrics']['silhouette']:.4f} time={t1 - t0:.0f}s")
results["Conv_f64_d32_e300"] = (ari, nmi)

# === Summary ===
print("\n" + "=" * 70)
print("SUMMARY — High-capacity 70K MNIST")
print("=" * 70)
for name, (ari_val, nmi_val) in results.items():
    print(f"  {name:<25} ARI={ari_val:.4f} NMI={nmi_val:.4f}")
print("=" * 70)
best = max(results.items(), key=lambda x: x[1][0])
print(f"Best: {best[0]} ARI={best[1][0]:.4f}")
