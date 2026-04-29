"""
tools/ae_pipeline.py
====================
Deterministic AutoEncoder + KMeans pipeline for high-dimensional data.

Used by DimensionExpert when ``CTX_DATA.n_features > 32``.
Runs a shallow MLP auto-encoder on CPU (or GPU if available), then
clusters the latent representations with KMeans.

This module is pre-injected into the sandbox so the code skeleton can
call ``ae_kmeans_pipeline()`` without importing it.
"""
from __future__ import annotations

import warnings
import numpy as np

warnings.filterwarnings("ignore")


def _get_device() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def build_autoencoder(n_features: int, latent_dim: int, hidden_dims: tuple = ()) -> "torch.nn.Module":
    """Construct a shallow MLP auto-encoder.

    Encoder: n_features → hidden → latent
    Decoder: latent → hidden → n_features
    """
    import torch.nn as nn

    if not hidden_dims:
        hidden = max(16, n_features // 3)
    else:
        hidden = hidden_dims[0] if len(hidden_dims) == 1 else hidden_dims[0]

    class AE(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(n_features, hidden),
                nn.ReLU(True),
                nn.Linear(hidden, latent_dim),
            )
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, hidden),
                nn.ReLU(True),
                nn.Linear(hidden, n_features),
            )

        def forward(self, x):
            z = self.encoder(x)
            return self.decoder(z), z

    return AE()


def train_ae(model, X: np.ndarray, epochs: int = 30, batch_size: int = 64,
             device: str = "cpu") -> np.ndarray:
    """Train the auto-encoder and return latent-space representations."""
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    X_t = torch.tensor(X, dtype=torch.float32)
    loader = DataLoader(TensorDataset(X_t), batch_size=min(batch_size, len(X_t)), shuffle=True)

    model.to(device)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()

    for _ in range(epochs):
        for (batch,) in loader:
            batch = batch.to(device)
            recon, _ = model(batch)
            loss = loss_fn(recon, batch)
            opt.zero_grad()
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        _, latents = model(X_t.to(device))
    return latents.cpu().numpy()


def ae_kmeans_pipeline(
    X: np.ndarray, k: int, *, latent_dim: int = 0, epochs: int = 30
) -> dict:
    """End-to-end AE+KMeans pipeline.

    Returns a dict with keys ``labels``, ``metrics``, ``plot_path``
    ready for insertion into the sandbox artifacts.
    """
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import silhouette_score

    X_scaled = StandardScaler().fit_transform(X)
    n_features = X_scaled.shape[1]

    if latent_dim <= 0:
        latent_dim = min(8, max(2, n_features // 4))
    device = _get_device()

    ae = build_autoencoder(n_features, latent_dim)
    latent_repr = train_ae(ae, X_scaled, epochs=epochs, device=device)

    labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(latent_repr)
    sil = float(silhouette_score(latent_repr, labels))

    return {
        "labels": labels.tolist(),
        "metrics": {
            "score": sil,
            "score_source": "silhouette",
            "silhouette": sil,
            "latent_dim": latent_dim,
            "epochs": epochs,
        },
        "plot_path": "",
    }
