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
import logging
import warnings

import numpy as np

warnings.filterwarnings("ignore")

_logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Safe torch import
# ---------------------------------------------------------------------------
_HAS_TORCH = False
_DEVICE = "cpu"
try:
    import torch  # noqa: I200

    _HAS_TORCH = True
    if torch.cuda.is_available():
        _DEVICE = "cuda"
        _logger.info("CUDA GPU detected, using GPU for AE training.")
    else:
        _logger.info("CUDA not available, AE training will run on CPU.")
except ImportError:
    _logger.info("PyTorch not installed, AE pipeline will use sklearn fallback.")


# ---------------------------------------------------------------------------
# GPU / CPU detection (backward-compatible alias)
# ---------------------------------------------------------------------------
def _get_device() -> str:
    return _DEVICE


# ---------------------------------------------------------------------------
# Data scaling helper
# ---------------------------------------------------------------------------
def _scale_data(X: np.ndarray, normalize: str = "standard") -> np.ndarray:
    """Scale input data according to *normalize* policy.

    ``"standard"`` — Z-score (backward-compatible default).
    ``"minmax"``  — MinMax to [0, 1] (for image data like MNIST).
    ``"none"``    — Pass-through (data is already pre-scaled).
    """
    if normalize == "none":
        return X
    if normalize == "minmax":
        from sklearn.preprocessing import MinMaxScaler
        return MinMaxScaler().fit_transform(X)
    from sklearn.preprocessing import StandardScaler
    return StandardScaler().fit_transform(X)


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
    hidden_dims = [min(h, 2048) for h in hidden_dims]

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
    normalize: str = "standard",
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
    normalize : str
        ``"standard"`` (default) — Z-score scaling.
        ``"minmax"`` — MinMax to [0,1]; use for image data (MNIST etc.).
        ``"none"`` — no scaling, data is already pre-normalised.
    """
    # ---- no-torch fast-path: pure sklearn fallback --------------------------
    if not _HAS_TORCH:
        return _sklearn_fallback_pipeline(
            X, k, latent_dim=latent_dim or 0, cluster_method=cluster_method,
            normalize=normalize)

    from sklearn.metrics import silhouette_score

    X_scaled = _scale_data(X, normalize)
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
            "backend": f"torch-{device}",
        },
        "plot_path": "",
    }


# ---------------------------------------------------------------------------
# Residual + Self-Attention AutoEncoder for GAP / embedding features (Phase 6)
# ---------------------------------------------------------------------------
def build_res_attention_autoencoder(
    n_features: int,
    hidden_dims: list[int] | None = None,
    latent_dim: int = 8,
    dropout: float = 0.2,
    num_heads: int = 4,
) -> torch.nn.Module:
    """Build residual AE with self-attention bottleneck for semantic features.

    Designed for GAP / embedding data (e.g. CIFAR-10 64D GAP) where:
    - Each dimension is a semantic feature (not raw pixel)
    - Features are correlated and benefit from attention-based reweighting
    - Residual connections prevent vanishing gradients in deeper stacks

    Architecture::

        Input → ResBlock(s) → SelfAttention → Latent → ResBlock(s) → Output

    The self-attention sits at the pre-bottleneck layer (before compression
    to ``latent_dim``), letting the model learn which feature combinations
    carry the most classification signal.
    """

    import torch.nn as nn

    if not hidden_dims:
        if n_features > 128:
            hidden_dims = [256, 128, 64, 32]
        elif n_features > 64:
            hidden_dims = [128, 64, 32]
        elif n_features > 32:
            hidden_dims = [64, 32]
        else:
            hidden_dims = [max(32, n_features * 2), max(16, n_features)]
    hidden_dims = [min(h, 2048) for h in hidden_dims]

    # ---- Residual block ----------------------------------------------------
    class ResidualBlock(nn.Module):
        def __init__(
            self, in_dim: int, out_dim: int, dp: float
        ) -> None:
            super().__init__()
            self.main = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.BatchNorm1d(out_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(dp),
            )
            self.skip = (
                nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.main(x) + self.skip(x)

    # ---- Self-attention bottleneck ----------------------------------------
    class SelfAttentionBottleneck(nn.Module):
        """Multi-head self-attention operating across feature sub-spaces.

        Reshapes (B, D) → (B, num_heads, D//num_heads) so each attention
        head sees a slice of the feature vector, then aggregates.
        """

        def __init__(self, embed_dim: int, heads: int = 4) -> None:
            super().__init__()
            assert embed_dim % heads == 0, (
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({heads})"
            )
            self.heads = heads
            self.head_dim = embed_dim // heads
            self.qkv = nn.Linear(embed_dim, embed_dim * 3)
            self.out_proj = nn.Linear(embed_dim, embed_dim)
            self.norm = nn.LayerNorm(embed_dim)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            B, D = x.shape
            qkv = self.qkv(x)  # (B, 3*D)
            q, k, v = qkv.chunk(3, dim=-1)
            # Reshape: (B, heads, head_dim)
            q = q.view(B, self.heads, self.head_dim)
            k = k.view(B, self.heads, self.head_dim)
            v = v.view(B, self.heads, self.head_dim)
            # Scaled dot-product attention across feature heads
            scale = self.head_dim ** 0.5
            attn_weights = (q * k).sum(dim=-1) / scale  # (B, heads)
            attn_weights = attn_weights.softmax(dim=-1).unsqueeze(-1)
            attended = (v * attn_weights).view(B, D)
            return self.norm(x + self.out_proj(attended))

    # ---- Build encoder ----------------------------------------------------
    encoder_blocks: list[nn.Module] = []
    in_dim = n_features
    for h_dim in hidden_dims:
        encoder_blocks.append(ResidualBlock(in_dim, h_dim, dropout))
        in_dim = h_dim

    # Self-attention on the deepest hidden representation
    attn_bottleneck = SelfAttentionBottleneck(in_dim, heads=num_heads)

    # Final compression to latent
    encoder_blocks.append(nn.Linear(in_dim, latent_dim))

    # ---- Build decoder (symmetric, no residual) ---------------------------
    decoder_blocks: list[nn.Module] = []
    in_dim = latent_dim
    for h_dim in reversed(hidden_dims):
        decoder_blocks.extend([
            nn.Linear(in_dim, h_dim),
            nn.BatchNorm1d(h_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
        ])
        in_dim = h_dim
    decoder_blocks.append(nn.Linear(in_dim, n_features))

    class ResAttentionAE(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = nn.Sequential(*encoder_blocks)
            self.attention = attn_bottleneck
            self.decoder = nn.Sequential(*decoder_blocks)

        def forward(
            self, x: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            h = self.encoder[:-1](x)  # all residual blocks
            h = self.attention(h)      # self-attention reweighting
            z = self.encoder[-1](h)    # final linear to latent
            return self.decoder(z), z

    return ResAttentionAE()


def res_ae_kmeans_pipeline(
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
    num_heads: int = 4,
    cluster_method: str = "kmeans",
    normalize: str = "standard",
) -> dict:
    """Residual + Self-Attention AE + clustering pipeline (Phase 6).

    Uses ``build_res_attention_autoencoder()`` which adds residual
    skip-connections and a multi-head self-attention bottleneck to the
    standard deep AE architecture.

    Intended for GAP / embedding features (e.g. 64D CIFAR-10 GAP) where
    each dimension carries semantic meaning and inter-feature attention
    helps isolate classification-relevant signal.
    """
    if not _HAS_TORCH:
        return _sklearn_fallback_pipeline(
            X, k, latent_dim=latent_dim or 0, cluster_method=cluster_method,
            normalize=normalize)

    from sklearn.metrics import silhouette_score

    X_scaled = _scale_data(X, normalize)
    n_features = X_scaled.shape[1]

    if latent_dim <= 0:
        latent_dim = min(8, max(2, n_features // 4))

    # Auto-adjust heads based on the attention dimension
    _hidden = hidden_dims or []
    attn_dim = _hidden[-1] if _hidden else (
        32 if n_features <= 32 else (64 if n_features <= 64 else 128)
    )
    _num_heads = min(num_heads, attn_dim // 4)
    _num_heads = max(2, _num_heads)

    device = _get_device()
    ae = build_res_attention_autoencoder(
        n_features, hidden_dims, latent_dim, dropout, _num_heads,
    )
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
        labels = GaussianMixture(
            n_components=k, random_state=42, n_init=3,
        ).fit_predict(latent_repr)
        method_name = "GMM"
    else:
        from sklearn.cluster import KMeans
        labels = KMeans(
            n_clusters=k, random_state=42, n_init=10,
        ).fit_predict(latent_repr)
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
            "backend": f"torch-resattn-{device}",
            "num_heads": _num_heads,
        },
        "plot_path": "",
    }


# ---------------------------------------------------------------------------
# Conv-AE pipeline — for image data (MNIST, Fashion-MNIST, etc.)
# ---------------------------------------------------------------------------
def build_conv_autoencoder(
    latent_dim: int = 32,
    input_channels: int = 1,
    input_size: int = 28,
    base_filters: int = 32,
    dropout: float = 0.1,
) -> torch.nn.Module:
    """Build a Conv2d/ConvTranspose2d autoencoder for image data.

    Pads input to the next multiple of 8 (e.g. 28→32) so the
    ConvTranspose2d decoder hits exact spatial sizes without
    negative output_padding.

    Encoder (3× Conv2d stride-2 on padded input)::
        Pad(H→H8) → Conv(f, 3, s=2) → BN → ReLU → Dropout2d
        → Conv(2f, 3, s=2) → BN → ReLU → Dropout2d
        → Conv(4f, 3, s=2) → BN → ReLU
        → Flatten → Linear(flat_dim, latent_dim)

    Decoder (3× ConvTranspose2d, learnable upsampling)::
        Linear(latent_dim, flat_dim) → Unflatten
        → ConvTranspose2d(4f→2f, 3, s=2, p=1, op=1) → BN → ReLU
        → ConvTranspose2d(2f→f, 3, s=2, p=1, op=1) → BN → ReLU
        → ConvTranspose2d(f→C, 3, s=2, p=1, op=1) → Sigmoid
        → Crop to original size

    Parameters
    ----------
    latent_dim : int
        Bottleneck dimension (default 32 for image data).
    input_channels : int
        Number of input channels (1 for grayscale).
    input_size : int
        Square image dimension (e.g. 28). Padded to next multiple of 8.
    base_filters : int
        Filters in first conv layer (doubles per stage).
    dropout : float
        Dropout2d probability after each encoder ReLU.
    """
    import torch.nn as nn
    import torch.nn.functional as F  # noqa: N812

    f = base_filters

    # Pad to next multiple of 8 for clean ConvTranspose2d sizing
    pad_to = ((input_size + 7) // 8) * 8
    pad_amt = (pad_to - input_size) // 2
    bottleneck_spatial = pad_to // 8  # after 3× stride-2
    conv_flat_dim = (4 * f) * bottleneck_spatial * bottleneck_spatial

    encoder = nn.Sequential(
        nn.Conv2d(input_channels, f, 3, stride=2, padding=1),
        nn.BatchNorm2d(f),
        nn.ReLU(inplace=True),
        nn.Dropout2d(dropout),
        nn.Conv2d(f, 2 * f, 3, stride=2, padding=1),
        nn.BatchNorm2d(2 * f),
        nn.ReLU(inplace=True),
        nn.Dropout2d(dropout),
        nn.Conv2d(2 * f, 4 * f, 3, stride=2, padding=1),
        nn.BatchNorm2d(4 * f),
        nn.ReLU(inplace=True),
        nn.Flatten(),
        nn.Dropout(dropout),                       # regularisation before bottleneck
        nn.Linear(conv_flat_dim, latent_dim),
        nn.BatchNorm1d(latent_dim),                # latent-space regularisation → N(0,1)
    )

    # ConvTranspose2d decoder — learnable upsampling (no bilinear Upsample)
    decoder = nn.Sequential(
        nn.Linear(latent_dim, conv_flat_dim),
        nn.Unflatten(1, (4 * f, bottleneck_spatial, bottleneck_spatial)),
        nn.ConvTranspose2d(4 * f, 2 * f, 3, stride=2, padding=1, output_padding=1),
        nn.BatchNorm2d(2 * f),
        nn.ReLU(inplace=True),
        nn.ConvTranspose2d(2 * f, f, 3, stride=2, padding=1, output_padding=1),
        nn.BatchNorm2d(f),
        nn.ReLU(inplace=True),
        nn.ConvTranspose2d(f, input_channels, 3, stride=2, padding=1, output_padding=1),
        nn.Sigmoid(),
    )

    class ConvAE(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = encoder
            self.decoder = decoder
            self._input_size = input_size
            self._pad_to = pad_to
            self._pad_amt = pad_amt
            self._input_channels = input_channels
            self._flat_dim = input_channels * input_size * input_size

        def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            # x: (N, C*H*W) flat → reshape to (N, C, H, W)
            x_2d = x.view(-1, self._input_channels, self._input_size, self._input_size)
            # Reflection-pad to divisible-by-8 size (avoids zero-pad edge artifacts)
            x_pad = F.pad(x_2d, [self._pad_amt] * 4, mode="reflect")
            z = self.encoder(x_pad)
            recon_pad = self.decoder(z)
            # Crop back to original size
            a = self._pad_amt
            recon_2d = recon_pad[:, :, a:a + self._input_size, a:a + self._input_size]
            recon_flat = recon_2d.reshape(-1, self._flat_dim)
            return recon_flat, z

    return ConvAE()


def _augment_batch(
    x: torch.Tensor, input_size: int, noise_std: float, device: str,
    translate: bool = True,
) -> torch.Tensor:
    """Apply random ±2px translation + Gaussian noise to a flat batch.

    When *translate* is False, only Gaussian noise is added (no translation).
    """
    import torch.nn.functional as F

    if noise_std > 0:
        x = x + torch.randn_like(x) * noise_std
    if not translate:
        return x
    bsz = x.shape[0]
    imgs_2d = x.view(bsz, 1, input_size, input_size)
    padded = F.pad(imgs_2d, [2, 2, 2, 2], mode="constant", value=0)
    cropped = torch.zeros(bsz, 1, input_size, input_size, device=device)
    for i in range(bsz):
        dx = torch.randint(0, 5, (1,)).item()
        dy = torch.randint(0, 5, (1,)).item()
        cropped[i] = padded[i, :, dy:dy + input_size, dx:dx + input_size]
    return cropped.view(bsz, -1)


def _nt_xent_loss(
    z_a: torch.Tensor, z_b: torch.Tensor, temperature: float = 0.5
) -> torch.Tensor:
    """NT-Xent (SimCLR) contrastive loss between two augmented views.

    Normalises both views, constructs a 2N×2N similarity matrix, and
    applies cross-entropy with positives at offset N.
    """
    import torch.nn.functional as F

    z_a = F.normalize(z_a, dim=1)
    z_b = F.normalize(z_b, dim=1)
    z = torch.cat([z_a, z_b], dim=0)          # (2N, D)
    sim = torch.mm(z, z.t()) / temperature     # (2N, 2N)
    n = z_a.shape[0]
    mask = torch.eye(2 * n, device=z.device).bool()
    sim = sim.masked_fill(mask, -float("inf"))
    labels = torch.cat([torch.arange(n) + n, torch.arange(n)]).to(z.device)
    return F.cross_entropy(sim, labels)


def train_conv_ae(
    model: torch.nn.Module,
    X: np.ndarray,
    *,
    epochs: int = 150,
    batch_size: int = 128,
    device: str = "cpu",
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    early_stopping_patience: int = 20,
    noise_std: float = 0.1,
    augment: bool = False,
    contrastive_weight: float = 0.0,
    contrastive_temperature: float = 0.5,
    img_size: int = 28,
) -> np.ndarray:
    """Train a Conv-AE and return latent-space representations.

    Uses Denoising AE with 80/20 train/validation split, CosineAnnealingLR,
    and Early Stopping with best-model restoration.

    When *augment* is True, applies random ±2px translation during training.
    When *contrastive_weight* > 0, adds SimCLR NT-Xent loss between two
    randomly augmented views to encourage augmentation-invariant encodings.

    Returns
    -------
    np.ndarray of shape ``(n_samples, latent_dim)`` — latent codes.
    """
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    X_t = torch.tensor(X, dtype=torch.float32)
    n_total = len(X_t)

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
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=epochs, eta_min=learning_rate * 0.01
    )

    best_val_loss = float("inf")
    best_model_state = None
    patience_counter = 0

    for _ in range(epochs):
        model.train()
        train_loss = 0.0
        train_mse = 0.0
        train_contrast = 0.0
        for (batch,) in train_loader:
            clean = batch.to(device)
            if noise_std > 0:
                noisy = clean + torch.randn_like(clean) * noise_std
            else:
                noisy = clean

            if augment:
                noisy = _augment_batch(noisy, img_size, 0, device, translate=True)

            recon, z = model(noisy)
            loss = loss_fn(recon, clean)
            train_mse += loss.item() * len(batch)

            # ---- SimCLR-lite contrastive loss -------------------------------
            if contrastive_weight > 0:
                # Second view: noise always, translation only if augment=True
                view_b = _augment_batch(clean.clone(), img_size, noise_std, device,
                                        translate=augment)
                _, z_b = model(view_b)
                contrast_loss = _nt_xent_loss(z, z_b, contrastive_temperature)
                loss = loss + contrastive_weight * contrast_loss
                train_contrast += contrast_loss.item() * len(batch)

            opt.zero_grad()
            loss.backward()
            opt.step()
            train_loss += loss.item() * len(batch)
        train_loss /= n_train

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for (batch,) in val_loader:
                batch = batch.to(device)
                recon, _ = model(batch)
                val_loss += loss_fn(recon, batch).item() * len(batch)
        val_loss /= n_val

        scheduler.step()

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    model.eval()
    with torch.no_grad():
        _, latents = model(X_t.to(device))
    return latents.cpu().numpy()


def conv_ae_kmeans_pipeline(
    X: np.ndarray,
    k: int,
    *,
    latent_dim: int = 32,
    input_size: int = 28,
    base_filters: int = 32,
    epochs: int = 150,
    batch_size: int = 128,
    learning_rate: float = 1e-3,
    dropout: float = 0.1,
    noise_std: float = 0.1,
    early_stopping_patience: int = 20,
    augment: bool = True,
    contrastive_weight: float = 0.1,
    contrastive_temperature: float = 0.5,
    cluster_method: str = "kmeans",
    normalize: str = "minmax",
) -> dict:
    """End-to-end Conv-AE + clustering pipeline for image data.

    Trains a Conv2d/ConvTranspose2d denoising autoencoder, then clusters
    the latent representations with KMeans or GaussianMixture.

    Falls back to random labels on OOM (GPU VRAM exhaustion).

    Parameters
    ----------
    X : np.ndarray
        Input data of shape ``(n_samples, n_features)`` (flat pixels).
    k : int
        Number of clusters.
    latent_dim : int
        Bottleneck dimension (default 32 for image data).
    input_size : int
        Square image dimension (28 for MNIST/Fashion-MNIST).
    base_filters : int
        Filters in first Conv layer. Reduce from 32→16 if OOM.
    epochs : int
        Maximum training epochs (Early Stopping may stop earlier).
    cluster_method : str
        ``"kmeans"`` or ``"gmm"``.
    normalize : str
        Default ``"minmax"`` — to [0,1] for image pixels.
    """
    if not _HAS_TORCH:
        return _sklearn_fallback_pipeline(
            X, k, latent_dim=latent_dim, cluster_method=cluster_method, normalize=normalize
        )

    from sklearn.metrics import silhouette_score

    X_scaled = _scale_data(X, normalize)

    if latent_dim <= 0:
        latent_dim = max(16, min(256, (input_size ** 2) // 24))

    device = _get_device()

    try:
        ae = build_conv_autoencoder(
            latent_dim=latent_dim,
            input_channels=1,
            input_size=input_size,
            base_filters=base_filters,
            dropout=dropout,
        )
        latent_repr = train_conv_ae(
            ae,
            X_scaled,
            epochs=epochs,
            batch_size=batch_size,
            device=device,
            learning_rate=learning_rate,
            noise_std=noise_std,
            early_stopping_patience=early_stopping_patience,
            augment=augment,
            contrastive_weight=contrastive_weight,
            contrastive_temperature=contrastive_temperature,
            img_size=input_size,
        )
    except RuntimeError as e:
        if "out of memory" in str(e).lower() and base_filters > 8:
            _logger.warning("Conv-AE OOM with base_filters=%d, retrying with %d",
                            base_filters, base_filters // 2)
            return conv_ae_kmeans_pipeline(
                X, k,
                latent_dim=latent_dim, input_size=input_size,
                base_filters=base_filters // 2,
                epochs=epochs, batch_size=batch_size // 2,
                learning_rate=learning_rate, dropout=dropout,
                noise_std=noise_std, early_stopping_patience=early_stopping_patience,
                cluster_method=cluster_method, normalize=normalize,
            )
        _logger.warning("Conv-AE failed: %s, falling back to MLP AE", e)
        return ae_kmeans_pipeline(
            X, k,
            latent_dim=latent_dim, epochs=epochs,
            cluster_method=cluster_method, normalize=normalize,
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
            "base_filters": base_filters,
            "cluster_method": method_name,
            "backend": f"torch-conv-{device}",
        },
        "plot_path": "",
    }


# ---------------------------------------------------------------------------
# Self-labeling iterative refinement (DeepCluster-like)
# ---------------------------------------------------------------------------
def conv_selflabel_pipeline(
    X: np.ndarray,
    k: int,
    *,
    latent_dim: int = 32,
    input_size: int = 28,
    base_filters: int = 32,
    ae_epochs: int = 150,
    cluster_epochs: int = 30,
    n_iterations: int = 3,
    learning_rate: float = 1e-3,
    cluster_lr: float = 1e-4,
    batch_size: int = 128,
    dropout: float = 0.1,
    noise_std: float = 0.1,
    augment: bool = True,
    contrastive_weight: float = 0.1,
    conf_threshold: float = 0.5,
    bootstrap: bool = True,
    device: str = "auto",
    normalize: str = "minmax",
) -> dict:
    """Self-labeling iterative refinement (Teacher-Student Distillation).

    Unlike DEC/IDEC (KL divergence), this uses hard pseudo-labels from GMM
    (the "teacher") to fine-tune an encoder+classifier (the "student") with
    cross-entropy loss. Cross-entropy provides strong discriminative gradients
    that widen inter-class margins in latent space.

    Pipeline:
    1. Conv-AE pretraining (+ optional SimCLR contrastive loss)
    2. GMM clustering on frozen AE features → pseudo-labels (teacher)
    3. For each iteration:
       a. Encoder + Linear(k) classifier (student) trained with CE loss
          on bootstrap-resampled high-confidence pseudo-labels
       b. ReduceLROnPlateau triggers when validation acc plateaus
       c. Re-extract features → GMM re-clustering → new pseudo-labels
       d. Stop if label change rate < 0.1% or sil decreases 2×
    4. Return best labels across all iterations
    """

    import torch
    import torch.nn.functional as F
    from sklearn.metrics import silhouette_score
    from sklearn.mixture import GaussianMixture
    from torch.utils.data import DataLoader, TensorDataset

    if device == "auto":
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"

    X_s = _scale_data(X, normalize)
    X_t = torch.tensor(X_s, dtype=torch.float32)
    n_samples = len(X_t)

    # ---- Phase 1: Conv-AE pretraining (+ contrastive) -----------------------
    ae = build_conv_autoencoder(
        latent_dim=latent_dim,
        input_channels=1,
        input_size=input_size,
        base_filters=base_filters,
        dropout=dropout,
    )
    train_conv_ae(
        ae, X_s, epochs=ae_epochs, batch_size=batch_size, device=device,
        learning_rate=learning_rate, noise_std=noise_std, augment=augment,
        contrastive_weight=contrastive_weight, img_size=input_size,
    )

    # ---- Phase 2: Initial GMM clustering (teacher) --------------------------
    ae.eval()
    with torch.no_grad():
        _, z = ae(X_t.to(device))
    z_np = z.cpu().numpy()

    gmm = GaussianMixture(n_components=k, random_state=42, n_init=20)
    labels = gmm.fit_predict(z_np)

    best_labels = labels.copy()
    best_sil = float(silhouette_score(z_np, labels))
    _logger.info("Initial GMM: sil=%.4f", best_sil)

    # ---- Phase 3: Iterative refinement (teacher→student distillation) ------
    prev_labels = labels.copy()
    sil_decrease_count = 0

    for iteration in range(n_iterations):
        # 3a. Student: encoder + Linear(k) classifier
        class ClassifierHead(torch.nn.Module):
            def __init__(self, enc, latent_d, n_cls, img_sz, img_ch=1):
                super().__init__()
                self.encoder = enc
                self.head = torch.nn.Linear(latent_d, n_cls)
                self._img_size = img_sz
                self._img_ch = img_ch
                pad_to = ((img_sz + 7) // 8) * 8
                self._pad_amt = (pad_to - img_sz) // 2

            def forward(self, x):
                x_2d = x.view(-1, self._img_ch, self._img_size, self._img_size)
                x_pad = F.pad(x_2d, [self._pad_amt] * 4, mode="reflect")
                return self.head(self.encoder(x_pad))

        clf = ClassifierHead(ae.encoder, latent_dim, k, input_size).to(device)

        # High-confidence sample selection
        probs = gmm.predict_proba(z_np)
        max_prob = probs.max(axis=1)
        conf_mask = max_prob > conf_threshold
        conf_idx = np.where(conf_mask)[0]

        if len(conf_idx) < k * 10:
            _logger.warning("Too few high-confidence samples (%d), using all", len(conf_idx))
            conf_idx = np.arange(n_samples)

        # Bootstrap resampling: draw with replacement from confident set
        if bootstrap:
            rng = np.random.RandomState(42 + iteration)
            train_idx = conf_idx[rng.choice(len(conf_idx), size=len(conf_idx), replace=True)]
        else:
            train_idx = conf_idx

        # 80/20 train/val split for ReduceLROnPlateau monitoring
        n_tr = int(len(train_idx) * 0.8)
        tr_idx = train_idx[:n_tr]
        va_idx = train_idx[n_tr:]

        X_tr, y_tr = X_t[tr_idx], torch.tensor(labels[tr_idx], dtype=torch.long)
        X_va, y_va = X_t[va_idx], torch.tensor(labels[va_idx], dtype=torch.long)

        train_loader_clf = DataLoader(
            TensorDataset(X_tr, y_tr),
            batch_size=min(batch_size, len(tr_idx)), shuffle=True,
        )
        val_loader_clf = DataLoader(
            TensorDataset(X_va, y_va),
            batch_size=min(batch_size, len(va_idx)), shuffle=False,
        )

        opt = torch.optim.Adam(clf.parameters(), lr=cluster_lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="max", factor=0.5, patience=5, min_lr=cluster_lr * 0.01, verbose=False,
        )
        ce_fn = torch.nn.CrossEntropyLoss()

        best_val_acc = 0.0
        for _ in range(cluster_epochs):
            clf.train()
            for batch_x, batch_y in train_loader_clf:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)

                if augment:
                    batch_x = _augment_batch(batch_x, input_size, 0, device)

                logits = clf(batch_x)
                loss = ce_fn(logits, batch_y)

                opt.zero_grad()
                loss.backward()
                opt.step()

            # Validation
            clf.eval()
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for batch_x, batch_y in val_loader_clf:
                    batch_x = batch_x.to(device)
                    batch_y = batch_y.to(device)
                    logits = clf(batch_x)
                    val_correct += (logits.argmax(dim=1) == batch_y).sum().item()
                    val_total += len(batch_y)
            val_acc = val_correct / max(val_total, 1)
            scheduler.step(val_acc)

            if val_acc > best_val_acc:
                best_val_acc = val_acc

        # 3b. Re-extract features from AE (use full model for proper padding)
        ae.eval()
        with torch.no_grad():
            _, z_new = ae(X_t.to(device))
        z_new_np = z_new.cpu().numpy()

        # Re-cluster with GMM
        new_gmm = GaussianMixture(n_components=k, random_state=42, n_init=20)
        new_labels = new_gmm.fit_predict(z_new_np)
        new_sil = float(silhouette_score(z_new_np, new_labels))

        # Track best
        if new_sil > best_sil:
            best_sil = new_sil
            best_labels = new_labels.copy()
            sil_decrease_count = 0

        # 3c. Convergence check
        delta = (new_labels != prev_labels).mean()
        _logger.info("Iter %d: sil=%.4f (best=%.4f) delta=%.4f val_acc=%.4f",
                      iteration + 1, new_sil, best_sil, delta, best_val_acc)

        if new_sil < best_sil:
            sil_decrease_count += 1

        if delta < 0.001 or sil_decrease_count >= 2:
            _logger.info("Self-label converged at iteration %d", iteration + 1)
            break

        prev_labels = new_labels.copy()
        labels = new_labels
        z_np = z_new_np
        gmm = new_gmm

    return {
        "labels": best_labels.tolist(),
        "metrics": {
            "score": best_sil,
            "score_source": "silhouette",
            "silhouette": best_sil,
            "latent_dim": latent_dim,
            "ae_epochs": ae_epochs,
            "iterations": iteration + 1,
            "cluster_method": "SelfLabel-GMM",
            "backend": f"torch-conv-{device}",
        },
        "plot_path": "",
    }


# ---------------------------------------------------------------------------
# Sklearn fallback (PCA + clustering) — used when torch is unavailable
# ---------------------------------------------------------------------------
def _sklearn_fallback_pipeline(
    X: np.ndarray,
    k: int,
    *,
    latent_dim: int = 0,
    cluster_method: str = "kmeans",
    normalize: str = "standard",
) -> dict:
    """Pure sklearn fallback: PCA reduction → KMeans / GMM clustering.

    Used automatically by ``ae_kmeans_pipeline`` when PyTorch is not
    installed.  Returns the same dict shape so callers don't need to
    special-case.
    """
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_score

    X_scaled = _scale_data(X, normalize)
    n_features = X_scaled.shape[1]

    if latent_dim <= 0:
        latent_dim = min(8, max(2, n_features // 4))

    reducer = PCA(n_components=min(latent_dim, n_features), random_state=42)
    latent_repr = reducer.fit_transform(X_scaled)

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
            "cluster_method": method_name,
            "backend": "sklearn-fallback",
        },
        "plot_path": "",
    }
