"""70K MNIST benchmark v4 — augment=False for AE pretraining (no translation).

Key insight from augmentation A/B test: translation hurts MNIST ARI
(0.5649 → 0.5094). Self-labeling student phase still uses augmentation
for classifier robustness.

Compares:
  1. SelfLabel d=16, augment=False (most compact + clean AE)
  2. SelfLabel d=24, augment=False
  3. SelfLabel d=32, augment=False
  4. Conv-AE + GMM baseline d=16, augment=False
"""
import sys
import time

import numpy as np

sys.path.insert(0, "D:/PycharmProject")
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")

from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from ACE_Agent.benchmark.dataloader import load_benchmark_dataset

ds = load_benchmark_dataset("mnist_full")
X, y = ds.X, ds.y
print(f"X: {X.shape}, y: {y.shape}")

results = {}

# Shared params: NO translation during AE pretraining
SHARED = dict(
    k=10, ae_epochs=150, cluster_epochs=30, n_iterations=3,
    dropout=0.2, contrastive_weight=0.1, bootstrap=True,
    augment=False,  # <-- no translation during AE pretraining
    base_filters=32, batch_size=128, normalize="minmax",
)

# ============================================================================
# [1/4] SelfLabel d=16, augment=False
# ============================================================================
print("\n" + "=" * 70)
print("[1/4] SelfLabel d=16 (no translation AE, contrastive=0.1)")
print("=" * 70)
from ACE_Agent.tools.ae_pipeline import conv_selflabel_pipeline

t0 = time.time()
r = conv_selflabel_pipeline(X, latent_dim=16, **SHARED)
t1 = time.time()
ari = adjusted_rand_score(y, np.array(r["labels"]))
nmi = normalized_mutual_info_score(y, np.array(r["labels"]))
print(f"  ARI={ari:.4f} NMI={nmi:.4f} Sil={r['metrics']['silhouette']:.4f} time={t1-t0:.0f}s")
results["SelfLabel_d16"] = (ari, nmi, r["metrics"])

# ============================================================================
# [2/4] SelfLabel d=24, augment=False
# ============================================================================
print("\n" + "=" * 70)
print("[2/4] SelfLabel d=24 (no translation AE, contrastive=0.1)")
print("=" * 70)
t0 = time.time()
r = conv_selflabel_pipeline(X, latent_dim=24, **SHARED)
t1 = time.time()
ari = adjusted_rand_score(y, np.array(r["labels"]))
nmi = normalized_mutual_info_score(y, np.array(r["labels"]))
print(f"  ARI={ari:.4f} NMI={nmi:.4f} Sil={r['metrics']['silhouette']:.4f} time={t1-t0:.0f}s")
results["SelfLabel_d24"] = (ari, nmi, r["metrics"])

# ============================================================================
# [3/4] SelfLabel d=32, augment=False
# ============================================================================
print("\n" + "=" * 70)
print("[3/4] SelfLabel d=32 (no translation AE, contrastive=0.1)")
print("=" * 70)
t0 = time.time()
r = conv_selflabel_pipeline(X, latent_dim=32, **SHARED)
t1 = time.time()
ari = adjusted_rand_score(y, np.array(r["labels"]))
nmi = normalized_mutual_info_score(y, np.array(r["labels"]))
print(f"  ARI={ari:.4f} NMI={nmi:.4f} Sil={r['metrics']['silhouette']:.4f} time={t1-t0:.0f}s")
results["SelfLabel_d32"] = (ari, nmi, r["metrics"])

# ============================================================================
# [4/4] Conv-AE + GMM baseline d=16, augment=False
# ============================================================================
print("\n" + "=" * 70)
print("[4/4] Conv-AE + GMM d=16 (no translation, contrastive=0.1)")
print("=" * 70)
from ACE_Agent.tools.ae_pipeline import conv_ae_kmeans_pipeline

t0 = time.time()
r = conv_ae_kmeans_pipeline(
    X, k=10, latent_dim=16, epochs=150, base_filters=32,
    cluster_method="gmm", normalize="minmax",
    augment=False, contrastive_weight=0.1,
)
t1 = time.time()
ari = adjusted_rand_score(y, np.array(r["labels"]))
nmi = normalized_mutual_info_score(y, np.array(r["labels"]))
print(f"  ARI={ari:.4f} NMI={nmi:.4f} Sil={r['metrics']['silhouette']:.4f} time={t1-t0:.0f}s")
results["ConvGMM_d16"] = (ari, nmi, r["metrics"])

# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 70)
print("SUMMARY — SelfLabel (no translation AE) + Dim Sweep on 70K MNIST")
print("=" * 70)
print(f"{'Method':<25} {'ARI':>8} {'NMI':>8} {'Silhouette':>12} {'Iters':>6}")
print("-" * 70)
for name, (ari_val, nmi_val, metrics) in results.items():
    iters = metrics.get("iterations", "-")
    print(f"{name:<25} {ari_val:>8.4f} {nmi_val:>8.4f} {metrics['silhouette']:>12.4f} {str(iters):>6}")
print("=" * 70)
best = max(results.items(), key=lambda x: x[1][0])
print(f"Best: {best[0]} ARI={best[1][0]:.4f}")
