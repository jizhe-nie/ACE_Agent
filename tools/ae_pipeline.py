"""
tools/ae_pipeline.py
====================
Deterministic Deep AutoEncoder + KMeans pipeline for high-dimensional data.

Architecture (v2):
  - Deep stacked encoder: n_features → hidden_dims[0] → ... → hidden_dims[-1] → latent_dim
  - Symmetric decoder with BatchNorm1d + LeakyReLU + Dropout per layer
  - L2 regularization (weight_decay) in Adam optimizer
  - Validation-based Early Stopping with best-model restoration

Used by DimensionExpert when ``CTX_DATA.n_features > 32``.
Pre-injected into the sandbox so the code skeleton can call
``ae_kmeans_pipeline()`` without importing it.
"""

from __future__ import annotations

import copy
import warnings

import numpy as np
import torch

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# GPU / CPU detection
# ---------------------------------------------------------------------------
def _get_device() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


# ---------------------------------------------------------------------------
# Deep AutoEncoder builder
# ---------------------------------------------------------------------------
def build_autoencoder(
    n_features: int,
    hidden_dims: list[int] | None = None,
    latent_dim: int = 8,
    dropout: float = 0.2,
) -> torch.nn.Module:
    """Build a deep stacked auto-encoder.

    Encoder::
        n_features → hidden_dims[0] → ... → hidden_dims[-1] → latent_dim

    Decoder (symmetric)::
        latent_dim → hidden_dims[-1] → ... → hidden_dims[0] → n_features

    Each hidden layer is wrapped as::
        Linear → BatchNorm1d → LeakyReLU(0.2) → Dropout

    The final encoder layer (→ latent_dim) and final decoder layer
    (→ n_features) are plain ``Linear`` without activation or dropout,
    so the latent space stays unrestricted and the output is unbounded.

    Parameters
    ----------
    n_features : int
        Input dimensionality.
    hidden_dims : list[int] or None
        List of hidden layer widths for the encoder (excluding latent_dim).
        If ``None`` or empty, a sensible default is generated from n_features.
    latent_dim : int
        Bottleneck dimension.
    dropout : float
        Dropout probability applied after each LeakyReLU (default 0.2).
    """
    import torch.nn as nn

    if not hidden_dims:
        # Sensible default: 3-layer encoder scaling with input dimension
        if n_features > 128:
            hidden_dims = [256, 128, 64, 32]
        elif n_features > 64:
            hidden_dims = [128, 64, 32]
        elif n_features > 32:
            hidden_dims = [64, 32]
        else:
            hidden_dims = [max(32, n_features * 2), max(16, n_features)]

    # Clip extremely wide layers to avoid VRAM blow-up
    hidden_dims = [min(h, 512) for h in hidden_dims]

    encoder_layers: list[nn.Module] = []
    in_dim = n_features
    for h_dim in hidden_dims:
        encoder_layers.extend(
            [
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(dropout),
            ]
        )
        in_dim = h_dim
    encoder_layers.append(nn.Linear(in_dim, latent_dim))

    decoder_layers: list[nn.Module] = []
    in_dim = latent_dim
    for h_dim in reversed(hidden_dims):
        decoder_layers.extend(
            [
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(dropout),
            ]
        )
        in_dim = h_dim
    decoder_layers.append(nn.Linear(in_dim, n_features))

    class DeepAE(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = nn.Sequential(*encoder_layers)
            self.decoder = nn.Sequential(*decoder_layers)

        def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            z = self.encoder(x)
            return self.decoder(z), z

    return DeepAE()


# ---------------------------------------------------------------------------
# Training loop with L2 reg + Early Stopping
# ---------------------------------------------------------------------------
def train_ae(
    model: torch.nn.Module,
    X: np.ndarray,
    *,
    epochs: int = 100,
    batch_size: int = 64,
    device: str = "cpu",
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    early_stopping_patience: int = 15,
    noise_std: float = 0.15,
) -> np.ndarray:
    """Train the auto-encoder and return latent-space representations.

    Uses Denoising AE (add Gaussian noise to input, reconstruct clean)
    with an 80/20 train/validation split for Early Stopping.  A
    ``CosineAnnealingLR`` scheduler ramps the learning rate down over
    the course of training.

    Training stops when validation loss fails to improve for
    ``early_stopping_patience`` consecutive epochs, and the best
    model weights are restored.

    Parameters
    ----------
    model : nn.Module
        Auto-encoder model (from ``build_autoencoder``).
    X : np.ndarray
        Input data of shape ``(n_samples, n_features)``.
    epochs : int
        Maximum training epochs.
    batch_size : int
        Mini-batch size (capped at n_samples).
    device : str
        Torch device string.
    learning_rate : float
        Adam initial learning rate.
    weight_decay : float
        L2 regularization strength on all parameters.
    early_stopping_patience : int
        Stop after this many epochs without validation improvement.
    noise_std : float
        Standard deviation of Gaussian noise added to input during
        training (Denoising AE).  0.0 disables.
    """
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    X_t = torch.tensor(X, dtype=torch.float32)
    n_total = len(X_t)

    # ---- train / validation split -----------------------------------------
    n_val = max(1, int(n_total * 0.2))
    n_train = n_total - n_val
    indices = torch.randperm(n_total)
    train_idx, val_idx = indices[:n_train], indices[n_train:]

    train_loader = DataLoader(
        TensorDataset(X_t[train_idx]),
        batch_size=min(batch_size, n_train),
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(X_t[val_idx]),
        batch_size=min(batch_size, n_val),
        shuffle=False,
    )

    model.to(device)
    loss_fn = torch.nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=learning_rate * 0.01)

    best_val_loss = float("inf")
    best_model_state = None
    patience_counter = 0

    for _epoch in range(epochs):
        # ---- train step (Denoising AE) ------------------------------------
        model.train()
        train_loss = 0.0
        for (batch,) in train_loader:
            clean = batch.to(device)
            # Add Gaussian noise → model must reconstruct clean input
            if noise_std > 0:
                noisy = clean + torch.randn_like(clean) * noise_std
            else:
                noisy = clean
            recon, _ = model(noisy)
            loss = loss_fn(recon, clean)
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_loss += loss.item() * len(batch)
        train_loss /= n_train

        # ---- validation step (no noise) -----------------------------------
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for (batch,) in val_loader:
                batch = batch.to(device)
                recon, _ = model(batch)
                val_loss += loss_fn(recon, batch).item() * len(batch)
        val_loss /= n_val

        scheduler.step()

        # ---- early stopping check -----------------------------------------
        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                break

    # Restore best weights
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # Encode all data to latent space
    model.eval()
    with torch.no_grad():
        _, latents = model(X_t.to(device))
    return latents.cpu().numpy()


# ---------------------------------------------------------------------------
# End-to-end pipeline
# ---------------------------------------------------------------------------
def ae_kmeans_pipeline(
    X: np.ndarray,
    k: int,
    *,
    latent_dim: int = 0,
    epochs: int = 100,
    hidden_dims: list[int] | None = None,
    learning_rate: float = 1e-3,
    dropout: float = 0.2,
    early_stopping_patience: int = 15,
    noise_std: float = 0.15,
    cluster_method: str = "kmeans",
) -> dict:
    """End-to-end Deep Denoising AE + clustering pipeline.

    After training a deep denoising auto-encoder, clusters the latent
    representations with either KMeans or GaussianMixture.

    Returns a dict with keys ``labels``, ``metrics``, ``plot_path``
    ready for insertion into the sandbox artifacts.

    Parameters
    ----------
    X : np.ndarray
        Input data of shape ``(n_samples, n_features)``.
    k : int
        Number of clusters.
    latent_dim : int
        Bottleneck dimension (0 → auto-compute as ``min(8, max(2, n_features//4))``).
    epochs : int
        Maximum training epochs (Early Stopping may stop earlier).
    hidden_dims : list[int] or None
        Encoder hidden layer widths (None → auto-compute from n_features).
    learning_rate : float
        Adam initial learning rate.
    dropout : float
        Dropout probability in each hidden layer.
    early_stopping_patience : int
        Early Stopping patience (epochs without validation improvement).
    noise_std : float
        Gaussian noise std for Denoising AE (0.0 disables).
    cluster_method : str
        ``"kmeans"`` or ``"gmm"`` — clustering algorithm in latent space.
    """
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import StandardScaler

    X_scaled = StandardScaler().fit_transform(X)
    n_features = X_scaled.shape[1]

    if latent_dim <= 0:
        latent_dim = min(8, max(2, n_features // 4))

    device = _get_device()

    ae = build_autoencoder(n_features, hidden_dims, latent_dim, dropout)
    latent_repr = train_ae(
        ae,
        X_scaled,
        epochs=epochs,
        device=device,
        learning_rate=learning_rate,
        weight_decay=1e-4,
        early_stopping_patience=early_stopping_patience,
        noise_std=noise_std,
    )

    if cluster_method == "gmm":
        from sklearn.mixture import GaussianMixture

        labels = GaussianMixture(n_components=k, random_state=42, n_init=3).fit_predict(latent_repr)
        method_name = "GMM"
    else:
        from sklearn.cluster import KMeans

        labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(latent_repr)
        method_name = "KMeans"

    sil = float(silhouette_score(latent_repr, labels))

    return {
        "labels": labels.tolist(),
        "metrics": {
            "score": sil,
            "score_source": "silhouette",
            "silhouette": sil,
            "latent_dim": latent_dim,
            "epochs": epochs,
            "hidden_dims": hidden_dims or [],
            "cluster_method": method_name,
        },
        "plot_path": "",
    }
