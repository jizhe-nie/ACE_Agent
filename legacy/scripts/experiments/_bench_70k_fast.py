"""Quick 70K MNIST benchmark — 3 core experiments."""
import logging
import time

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")

from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from ACE_Agent.benchmark.dataloader import load_benchmark_dataset

ds = load_benchmark_dataset("mnist_full")
X, y = ds.X, ds.y
print(f"X: {X.shape}, y: {y.shape}")

results = {}

# === 1. Conv-AE + GMM (dim=32, epochs=150) ===
print("\n" + "=" * 70)
print("[1/3] Conv-AE + GMM (latent_dim=32, epochs=150)")
from ACE_Agent.tools.ae_pipeline import conv_ae_kmeans_pipeline

t0 = time.time()
r = conv_ae_kmeans_pipeline(
    X, k=10, latent_dim=32, epochs=150,
    cluster_method="gmm", normalize="minmax",
)
t1 = time.time()
ari = adjusted_rand_score(y, np.array(r["labels"]))
nmi = normalized_mutual_info_score(y, np.array(r["labels"]))
print(f"  ARI={ari:.4f} NMI={nmi:.4f} Sil={r['metrics']['silhouette']:.4f} time={t1 - t0:.0f}s")
results["Conv_GMM_dim32"] = (ari, nmi, r["metrics"])

# === 2. Conv-AE + GMM (dim=64, epochs=150) ===
print("\n" + "=" * 70)
print("[2/3] Conv-AE + GMM (latent_dim=64, epochs=150)")
t0 = time.time()
r = conv_ae_kmeans_pipeline(
    X, k=10, latent_dim=64, epochs=150,
    cluster_method="gmm", normalize="minmax",
)
t1 = time.time()
ari = adjusted_rand_score(y, np.array(r["labels"]))
nmi = normalized_mutual_info_score(y, np.array(r["labels"]))
print(f"  ARI={ari:.4f} NMI={nmi:.4f} Sil={r['metrics']['silhouette']:.4f} time={t1 - t0:.0f}s")
results["Conv_GMM_dim64"] = (ari, nmi, r["metrics"])

# === 3. MLP AE + GMM baseline ===
print("\n" + "=" * 70)
print("[3/3] MLP AE + GMM baseline (latent_dim=32, epochs=100)")
from ACE_Agent.tools.ae_pipeline import ae_kmeans_pipeline

t0 = time.time()
r = ae_kmeans_pipeline(
    X, k=10, latent_dim=32, epochs=100,
    cluster_method="gmm", normalize="minmax",
)
t1 = time.time()
ari = adjusted_rand_score(y, np.array(r["labels"]))
nmi = normalized_mutual_info_score(y, np.array(r["labels"]))
print(f"  ARI={ari:.4f} NMI={nmi:.4f} Sil={r['metrics']['silhouette']:.4f} time={t1 - t0:.0f}s")
results["MLP_GMM"] = (ari, nmi, r["metrics"])

# === Summary ===
print("\n" + "=" * 70)
print("SUMMARY — 70K MNIST")
print("=" * 70)
print(f"{'Method':<35} {'ARI':>8} {'NMI':>8} {'Silhouette':>12}")
print("-" * 70)
for name, (ari_val, nmi_val, metrics) in results.items():
    print(f"{name:<35} {ari_val:>8.4f} {nmi_val:>8.4f} {metrics['silhouette']:>12.4f}")
print("=" * 70)
best = max(results.items(), key=lambda x: x[1][0])
print(f"Conv-AE over MLP: ARI delta = {results['Conv_GMM_dim32'][0] - results['MLP_GMM'][0]:+.4f}")
