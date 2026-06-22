"""Full ConvTranspose2d-AE benchmark on 70K MNIST."""
import logging
import time

import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(name)s] %(levelname)s: %(message)s')

from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from ACE_Agent.benchmark.dataloader import load_benchmark_dataset

ds = load_benchmark_dataset('mnist_full')
X, y = ds.X, ds.y
n_total = X.shape[0]
print(f'X: {X.shape}, y: {y.shape}')

print('=' * 70)
print('ConvTranspose2d-AE Benchmark — 70K MNIST')
print('=' * 70)

# ---- 1. ConvTranspose2d-AE + GMM (full 70K, 150 epochs) --------------------
print('\n[1/3] ConvTranspose2d-AE + GMM (70K, latent_dim=32, epochs=150)')
from ACE_Agent.tools.ae_pipeline import conv_ae_kmeans_pipeline

t0 = time.time()
r1 = conv_ae_kmeans_pipeline(
    X, k=10, latent_dim=32, epochs=150,
    cluster_method='gmm', normalize='minmax',
)
t1 = time.time()
ari1 = adjusted_rand_score(y, np.array(r1['labels']))
nmi1 = normalized_mutual_info_score(y, np.array(r1['labels']))
print(f'  ARI={ari1:.4f} NMI={nmi1:.4f} Sil={r1["metrics"]["silhouette"]:.4f} '
      f'time={t1-t0:.0f}s backend={r1["metrics"]["backend"]}')

# ---- 2. ConvTranspose2d-AE + KMeans (comparison) ---------------------------
print('\n[2/3] ConvTranspose2d-AE + KMeans (70K, latent_dim=32, epochs=150)')
t0 = time.time()
r2 = conv_ae_kmeans_pipeline(
    X, k=10, latent_dim=32, epochs=150,
    cluster_method='kmeans', normalize='minmax',
)
t1 = time.time()
ari2 = adjusted_rand_score(y, np.array(r2['labels']))
nmi2 = normalized_mutual_info_score(y, np.array(r2['labels']))
print(f'  ARI={ari2:.4f} NMI={nmi2:.4f} Sil={r2["metrics"]["silhouette"]:.4f} '
      f'time={t1-t0:.0f}s')

# ---- 3. MLP AE baseline (for reference) ------------------------------------
print('\n[3/3] MLP AE + GMM baseline (70K, latent_dim=32, epochs=100)')
from ACE_Agent.tools.ae_pipeline import ae_kmeans_pipeline

t0 = time.time()
r3 = ae_kmeans_pipeline(
    X, k=10, latent_dim=32, epochs=100,
    cluster_method='gmm', normalize='minmax',
)
t1 = time.time()
ari3 = adjusted_rand_score(y, np.array(r3['labels']))
nmi3 = normalized_mutual_info_score(y, np.array(r3['labels']))
print(f'  ARI={ari3:.4f} NMI={nmi3:.4f} Sil={r3["metrics"]["silhouette"]:.4f} '
      f'time={t1-t0:.0f}s backend={r3["metrics"]["backend"]}')

# ---- Summary ---------------------------------------------------------------
print('\n' + '=' * 70)
print('SUMMARY — 70K MNIST')
print('=' * 70)
print(f'{"Method":<30} {"ARI":>8} {"NMI":>8} {"Silhouette":>12}')
print('-' * 70)
print(f'{"ConvTranspose2d-AE + GMM":<30} {ari1:>8.4f} {nmi1:>8.4f} {r1["metrics"]["silhouette"]:>12.4f}')
print(f'{"ConvTranspose2d-AE + KMeans":<30} {ari2:>8.4f} {nmi2:>8.4f} {r2["metrics"]["silhouette"]:>12.4f}')
print(f'{"MLP AE + GMM (baseline)":<30} {ari3:>8.4f} {nmi3:>8.4f} {r3["metrics"]["silhouette"]:>12.4f}')
print('=' * 70)
print(f'\nConv-AE improvement over MLP: ARI delta = {ari1-ari3:+.4f}, NMI delta = {nmi1-nmi3:+.4f}')
