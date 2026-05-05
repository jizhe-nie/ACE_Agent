"""
tools/dec_pipeline.py
=====================
DEC / IDEC deep embedded clustering for high-dimensional data.

Reference
---------
- Xie, Girshick, Farhadi. "Unsupervised Deep Embedding for Clustering
  Analysis" (DEC), ICML 2016.
- Guo, Gao, et al. "Improved Deep Embedded Clustering ..." (IDEC), IJCAI 2017.

Architecture (v2 — academic alignment)
--------------------------------------
Phase 0 — Greedy layer-wise pretraining of stacked denoising autoencoders
          (SDAE), each layer trained as a single-hidden-layer DAE with
          Gaussian noise injection.  After stacking, a short global
          fine-tuning pass stabilises the full autoencoder.
Phase 1 — Remove decoder, initialise cluster centres via KMeans on latent
          codes, then jointly fine-tune encoder + centres with KL-divergence
          loss (DEC) or KL + reconstruction loss (IDEC).
Phase 2 — Optimiser: SGD + Momentum (0.9), following the original paper.
          Target distribution P is updated every *update_interval* batches
          (default 140) to prevent high-frequency oscillation and cluster
          collapse.

Default encoder for MNIST / Fashion-MNIST (n_features > 256):
    Input → 500 → 500 → 2000 → 10 (latent)
"""

from __future__ import annotations

import copy
import logging
from typing import Any

import numpy as np

_logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Safe torch import (consistent with ae_pipeline.py)
# ---------------------------------------------------------------------------
_HAS_TORCH = False
try:
    import torch
    import torch.nn.functional as F  # noqa: N812

    _HAS_TORCH = True
except ImportError:
    _logger.info("PyTorch not installed; DEC pipeline unavailable.")


# ========================================================================
# Data scaling helper
# ========================================================================

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
    # default: "standard"
    from sklearn.preprocessing import StandardScaler
    return StandardScaler().fit_transform(X)


# ========================================================================
# Phase 0 — Greedy layer-wise pretraining (SDAE)
# ========================================================================

def _greedy_layerwise_pretrain(
    X: np.ndarray,
    layer_dims: list[int],
    *,
    epochs_per_layer: int = 50,
    batch_size: int = 256,
    learning_rate: float = 0.01,
    noise_std: float = 0.3,
    dropout: float = 0.2,
    global_finetune_epochs: int = 100,
    device: str = "cpu",
) -> torch.nn.Module:
    """Greedy layer-wise pretraining of a stacked denoising autoencoder.

    Each layer is trained as a single-hidden-layer DAE:
        input_dim → hidden_dim → input_dim (reconstruction)

    After training layer *i*, its encoder half transforms the data for
    layer *i+1*.  Finally, all encoder and decoder layers are stacked
    and a short global fine-tuning pass is run.

    Parameters
    ----------
    X : (N, D) float array
        Standardised input data.
    layer_dims : list[int]
        Encoder hidden widths from input to latent, e.g. [500, 500, 2000, 10].
    epochs_per_layer : int
        Training epochs for each individual DAE layer.
    global_finetune_epochs : int
        Epochs of end-to-end AE fine-tuning after stacking all layers.
    device : str

    Returns
    -------
    torch.nn.Module — the full stacked autoencoder with pretrained weights.
    """
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    X_t = torch.tensor(X, dtype=torch.float32)
    n_total = len(X_t)
    current_data = X_t.clone()

    encoder_layers: list[nn.Module] = []
    decoder_layers: list[nn.Module] = []

    # Build dims list: [n_features] + layer_dims
    full_dims = [X.shape[1]] + list(layer_dims)

    for level in range(len(full_dims) - 1):
        in_dim = full_dims[level]
        hid_dim = full_dims[level + 1]

        # ---- build single-layer DAE ----
        enc = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            nn.BatchNorm1d(hid_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
        )
        dec = nn.Sequential(
            nn.Linear(hid_dim, in_dim),
        )

        dae = nn.Module()
        dae.encoder = enc
        dae.decoder = dec
        dae.to(device)

        # Build DataLoader from current (transformed) data
        layer_loader = DataLoader(
            TensorDataset(current_data),
            batch_size=min(batch_size, n_total),
            shuffle=True,
        )

        opt = torch.optim.SGD(dae.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)
        loss_fn = nn.MSELoss()

        dae.train()
        for _ in range(epochs_per_layer):
            for (batch,) in layer_loader:
                clean = batch.to(device)
                noisy = clean + torch.randn_like(clean) * noise_std
                recon = dae.decoder(dae.encoder(noisy))
                loss = loss_fn(recon, clean)
                opt.zero_grad()
                loss.backward()
                opt.step()

        # ---- save trained encoder / decoder (detached copy, keep on device) ----
        encoder_layers.append(copy.deepcopy(dae.encoder).to(device))
        decoder_layers.insert(0, copy.deepcopy(dae.decoder).to(device))

        # ---- transform data for next layer ----
        dae.eval()
        with torch.no_grad():
            current_data = dae.encoder(current_data.to(device)).cpu()

        _logger.info(
            "  Layer %d/%d: %d→%d  pretrained.",
            level + 1, len(full_dims) - 1, in_dim, hid_dim,
        )

    # ---- stack into full AE ----
    stacked_encoder = nn.Sequential(*encoder_layers)
    stacked_decoder = nn.Sequential(*decoder_layers)

    full_ae = _StackedAutoEncoder(stacked_encoder, stacked_decoder)
    full_ae.to(device)

    # ---- global fine-tuning pass ----
    if global_finetune_epochs > 0:
        _logger.info("  Global AE fine-tuning: %d epochs ...", global_finetune_epochs)
        full_ae.train()
        global_loader = DataLoader(
            TensorDataset(X_t),
            batch_size=min(batch_size, n_total),
            shuffle=True,
        )
        opt = torch.optim.SGD(full_ae.parameters(), lr=learning_rate * 0.1, momentum=0.9, weight_decay=1e-4)
        loss_fn = nn.MSELoss()
        for _ in range(global_finetune_epochs):
            for (batch,) in global_loader:
                clean = batch.to(device)
                noisy = clean + torch.randn_like(clean) * noise_std * 0.5
                recon, _ = full_ae(noisy)
                loss = loss_fn(recon, clean)
                opt.zero_grad()
                loss.backward()
                opt.step()

    full_ae.eval()
    return full_ae


class _StackedAutoEncoder(torch.nn.Module):
    """Wrapper that exposes the same ``(recon, latent)`` interface as
    ``build_autoencoder`` so the rest of the pipeline works unchanged."""

    def __init__(self, encoder: torch.nn.Module, decoder: torch.nn.Module):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x: torch.Tensor):
        z = self.encoder(x)
        r = self.decoder(z)
        return r, z


# ========================================================================
# Phase 1 — pretraining (thin wrapper, used without layer-wise path)
# ========================================================================

def _pretrain_ae(
    X: np.ndarray,
    hidden_dims: list[int] | None = None,
    latent_dim: int = 8,
    epochs: int = 100,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    dropout: float = 0.2,
    noise_std: float = 0.15,
    early_stopping_patience: int = 15,
    device: str = "cpu",
) -> tuple[torch.nn.Module, np.ndarray]:
    """Pretrain a deep denoising auto-encoder; return (model, latent_codes)."""
    from ACE_Agent.tools.ae_pipeline import build_autoencoder, train_ae

    n_features = X.shape[1]
    model = build_autoencoder(n_features, hidden_dims, latent_dim, dropout)
    latents = train_ae(
        model,
        X,
        epochs=epochs,
        batch_size=batch_size,
        device=device,
        learning_rate=learning_rate,
        weight_decay=1e-4,
        early_stopping_patience=early_stopping_patience,
        noise_std=noise_std,
    )
    return model, latents


# ========================================================================
# Phase 2 — DEC / IDEC clustering
# ========================================================================

def _soft_assignment(
    z: torch.Tensor,
    centers: torch.nn.Parameter,
) -> torch.Tensor:
    """Student's t-distribution kernel Q (α = 1).

    q_ij = (1 + ||z_i - μ_j||²)⁻¹ / Σ_j'(1 + ||z_i - μ_j'||²)⁻¹
    """
    diff = z.unsqueeze(1) - centers.unsqueeze(0)  # (N, K, D)
    sq_dist = (diff ** 2).sum(dim=2)              # (N, K)
    q_raw = 1.0 / (1.0 + sq_dist)                  # kernel
    q = q_raw / q_raw.sum(dim=1, keepdim=True)     # normalise
    return q


def _target_distribution(
    q: "torch.Tensor",
    conf_threshold: float = 0.0,
) -> "torch.Tensor":
    """Auxiliary target distribution P derived from Q.

    p_ij = (q_ij² / f_j) / Σ_j'(q_ij'² / f_j')
    where  f_j = Σ_i q_ij  (soft cluster frequency).

    When *conf_threshold* > 0, low-confidence samples (max(Q) ≤ threshold)
    use Q itself as target instead of the sharpened P, preventing noisy
    assignments from corrupting cluster centres.
    """
    f = q.sum(dim=0)                           # (K,) soft cluster sizes
    q_sq_f = (q ** 2) / (f + 1e-10)             # (N, K), epsilon guards div-by-zero
    p = q_sq_f / (q_sq_f.sum(dim=1, keepdim=True) + 1e-10)

    if conf_threshold > 0:
        max_q, _ = q.max(dim=1)                  # (N,)
        mask = (max_q > conf_threshold).float().unsqueeze(1)  # (N, 1)
        # High-confidence → P (sharpened); low-confidence → Q (soft self-target)
        p = mask * p + (1.0 - mask) * q

    return p


def dec_train(
    model: torch.nn.Module,
    X: np.ndarray,
    k: int,
    *,
    # Pretraining
    pretrain_epochs: int = 300,
    pretrain_lr: float = 0.01,
    # DEC finetuning
    finetune_epochs: int = 200,
    finetune_lr: float = 0.001,
    centres_lr: float = 0.01,
    batch_size: int = 64,
    # IDEC
    gamma: float = 0.0,               # reconstruction weight (>0 → IDEC)
    # KL annealing (smooth takeover from reconstruction to clustering)
    gamma_init: float = 10.0,          # initial gamma (reconstruction-dominant phase)
    gamma_warmup_epochs: int = 50,     # epochs at gamma_init before annealing
    gamma_anneal_epochs: int = 100,    # epochs to linearly decay gamma_init → gamma
    # Confidence threshold for target distribution P
    conf_threshold: float = 0.0,       # if >0, only sharpen high-confidence assignments
    # Convergence
    tol: float = 1e-3,                 # fraction of changed labels
    update_interval: int = 140,        # update P every N batches
    # Misc
    device: str = "cpu",
    noise_std: float = 0.3,
    use_layerwise: bool = True,
    layer_dims: list[int] | None = None,
    epochs_per_layer: int = 50,
    use_sgd_finetune: bool = False,
    use_sgd_pretrain: bool = False,
    early_stopping_patience: int = 50,
    normalize: str = "standard",
) -> dict[str, Any]:
    """DEC / IDEC clustering with KL annealing.

    Phase 0 (optional) — greedy layer-wise pretrain → full AE.
    Phase 1 — pretrain / global fine-tune autoencoder.
    Phase 2 — KMeans init → KL-divergence joint fine-tuning with
              annealed reconstruction weight.

    KL Annealing
    ------------
    gamma starts at *gamma_init* (e.g. 10.0 — pure reconstruction focus),
    stays there for *gamma_warmup_epochs*, then linearly decays to *gamma*
    over *gamma_anneal_epochs*.  This prevents KL divergence from destroying
    the AE's learned representations before clusters stabilise.

    Early epochs:  loss = KL(P||Q) + gamma_init × MSE(recon, input)
    Late epochs:   loss = KL(P||Q) + gamma × MSE(recon, input)

    Parameters
    ----------
    gamma : float
        Final reconstruction weight after annealing (0.0 = vanilla DEC).
    gamma_init : float
        Initial reconstruction weight (default 10.0 — strong reconstruction).
    gamma_warmup_epochs : int
        Epochs to hold at gamma_init before annealing begins.
    gamma_anneal_epochs : int
        Epochs over which gamma linearly decays from gamma_init to gamma.
    conf_threshold : float
        If > 0, only compute sharpened P for samples where max(Q) exceeds
        this value.  Low-confidence samples use Q as self-target instead.
    tol : float
        Convergence threshold — fraction of samples that change cluster
        assignment between two consecutive target-distribution updates.
    update_interval : int
        Number of batches between target-distribution P recomputations.
        DEC paper uses ~140; lower values risk cluster centre collapse.
    use_layerwise : bool
        If True, use greedy layer-wise SDAE pretraining before global
        fine-tuning (recommended for deep networks).
    normalize : str
        ``"standard"`` (default) — Z-score scaling.
        ``"minmax"`` — MinMax to [0,1]; use for image data (MNIST etc.).
        ``"none"`` — no scaling, data is already pre-normalised.
    device : str
        Torch device string ("cuda" or "cpu").
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    X_scaled = _scale_data(X, normalize)
    X_t = torch.tensor(X_scaled, dtype=torch.float32)
    n_samples, n_features = X_t.shape

    # ---- Phase 0 / 1: AE pretraining -----------------------------------------
    model.to(device)

    if use_layerwise and layer_dims:
        _logger.info("DEC: using greedy layer-wise pretrain (%s)", layer_dims)
        # Model already pretrained via _greedy_layerwise_pretrain.
        # Short Adam polish pass for stability.
        _pretrain_ae_inplace(
            model, X_scaled,
            epochs=min(pretrain_epochs // 6, 50),
            batch_size=batch_size,
            learning_rate=pretrain_lr * 0.1,
            noise_std=noise_std * 0.5,
            device=device,
            use_sgd=use_sgd_pretrain,
            early_stopping_patience=early_stopping_patience,
        )
    else:
        _pretrain_ae_inplace(
            model, X_scaled,
            epochs=pretrain_epochs,
            batch_size=batch_size,
            learning_rate=pretrain_lr,
            noise_std=noise_std,
            device=device,
            use_sgd=use_sgd_pretrain,
            early_stopping_patience=early_stopping_patience,
        )

    # encode all data to latent space
    model.eval()
    with torch.no_grad():
        _, latents = model(X_t.to(device))
    latents_np = latents.cpu().numpy()

    # ---- Phase 2: initialise cluster centres via KMeans ---------------------
    km = KMeans(n_clusters=k, random_state=42, n_init=20)
    km_labels = km.fit_predict(latents_np)
    centres_init = torch.tensor(km.cluster_centers_, dtype=torch.float32, device=device)

    # trainable cluster centres
    centres = torch.nn.Parameter(centres_init.clone())

    # Optimiser: Adam (default — better convergence in our setting)
    # SGD + Momentum available via `use_sgd_finetune=True`
    if use_sgd_finetune:
        opt = torch.optim.SGD(
            [
                {"params": model.encoder.parameters(), "lr": finetune_lr},
                {"params": [centres], "lr": centres_lr},
            ],
            momentum=0.9,
            weight_decay=1e-4,
        )
    else:
        opt = torch.optim.Adam(
            [
                {"params": model.encoder.parameters(), "lr": finetune_lr},
                {"params": [centres], "lr": centres_lr},
            ]
        )
    loss_fn = torch.nn.MSELoss()

    # ---- DEC / IDEC training loop -------------------------------------------
    xb = X_t.to(device)  # persistent device tensor

    # Initial full-dataset Q and P
    _target = lambda q: _target_distribution(q, conf_threshold)  # noqa: E731
    model.eval()
    with torch.no_grad():
        _, z_full = model(xb)
        q_full = _soft_assignment(z_full, centres)
    p_full = _target(q_full).detach()
    prev_labels = q_full.argmax(dim=1).cpu().numpy()

    best_labels = prev_labels.copy()
    best_loss = float("inf")
    best_model_state = copy.deepcopy(model.state_dict())
    best_centres = centres.data.clone()

    batch_count = 0

    for epoch in range(finetune_epochs):
        # ---- KL annealing: compute current gamma ----------------------------
        if gamma_init > gamma and gamma > 0:
            if epoch < gamma_warmup_epochs:
                current_gamma = gamma_init
            elif epoch < gamma_warmup_epochs + gamma_anneal_epochs:
                progress = (epoch - gamma_warmup_epochs) / gamma_anneal_epochs
                current_gamma = gamma_init + (gamma - gamma_init) * progress
            else:
                current_gamma = gamma
        else:
            current_gamma = gamma

        model.train()
        epoch_kl = 0.0
        epoch_recon = 0.0

        indices = torch.randperm(n_samples)
        effective_bs = min(batch_size, n_samples)

        for start in range(0, n_samples, effective_bs):
            batch_idx = indices[start : start + effective_bs]
            batch = xb[batch_idx]

            recon, z = model(batch)
            q_batch = _soft_assignment(z, centres)       # (B, K)
            p_batch = p_full[batch_idx]                   # (B, K) from full P

            kl_loss = F.kl_div(q_batch.log(), p_batch, reduction="batchmean")

            loss: torch.Tensor = kl_loss
            if current_gamma > 0:
                recon_loss = loss_fn(recon, batch)
                loss = kl_loss + current_gamma * recon_loss
                epoch_recon += recon_loss.item()

            opt.zero_grad()
            loss.backward()
            opt.step()

            epoch_kl += kl_loss.item()
            batch_count += 1

            # Intra-epoch P update (DEC paper: every update_interval batches)
            if batch_count % update_interval == 0:
                model.eval()
                with torch.no_grad():
                    _, z_full = model(xb)
                    q_full = _soft_assignment(z_full, centres)
                p_full = _target(q_full).detach()
                model.train()

        # ---- convergence check (every epoch) --------------------------------
        model.eval()
        with torch.no_grad():
            _, z_full = model(xb)
            q_full = _soft_assignment(z_full, centres)
        new_labels = q_full.argmax(dim=1).cpu().numpy()
        delta = (new_labels != prev_labels).mean()

        # Update P from full-dataset Q for next epoch (with confidence threshold)
        p_full = _target(q_full).detach()

        if delta < tol and epoch >= 10:
            best_labels = new_labels
            break

        prev_labels = new_labels

        total = epoch_kl + epoch_recon * gamma
        if total < best_loss:
            best_loss = total
            best_labels = new_labels
            best_model_state = copy.deepcopy(model.state_dict())
            best_centres = centres.data.clone()

    # ---- restore best -----------------------------------------------
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        centres.data.copy_(best_centres)

    # ---- final clustering result -------------------------------------------
    model.eval()
    with torch.no_grad():
        _, z_all = model(X_t.to(device))
        q_final = _soft_assignment(z_all, centres)
    final_labels = q_final.argmax(dim=1).cpu().numpy()

    sil = float(silhouette_score(latents_np, final_labels))

    return {
        "labels": final_labels.tolist(),
        "metrics": {
            "score": sil,
            "score_source": "silhouette",
            "silhouette": sil,
            "latent_dim": z_all.shape[1],
            "pretrain_epochs": pretrain_epochs,
            "finetune_epochs": epoch + 1,
            "gamma": gamma,
            "gamma_init": gamma_init,
            "conf_threshold": conf_threshold,
            "backend": f"torch-{device}",
            "method": "IDEC" if gamma > 0 else "DEC",
        },
        "plot_path": "",
    }


# ---------------------------------------------------------------------------
# In-place pretraining (avoids double AE construction)
# ---------------------------------------------------------------------------
def _pretrain_ae_inplace(
    model: torch.nn.Module,
    X: np.ndarray,
    *,
    epochs: int = 100,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    early_stopping_patience: int = 15,
    noise_std: float = 0.15,
    device: str = "cpu",
    use_sgd: bool = False,
) -> None:
    """Pretrain autoencoder in-place on *X* (standardised)."""
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

    if use_sgd:
        opt = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
    else:
        opt = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=learning_rate * 0.01)

    best_val = float("inf")
    best_state = None
    patience = 0

    for _ in range(epochs):
        model.train()
        for (batch,) in train_loader:
            clean = batch.to(device)
            noisy = clean + torch.randn_like(clean) * noise_std if noise_std > 0 else clean
            recon, _ = model(noisy)
            loss = loss_fn(recon, clean)
            opt.zero_grad()
            loss.backward()
            opt.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for (batch,) in val_loader:
                batch = batch.to(device)
                recon, _ = model(batch)
                val_loss += loss_fn(recon, batch).item() * len(batch)
        val_loss /= n_val
        scheduler.step()

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            patience = 0
        else:
            patience += 1
            if patience >= early_stopping_patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)


# ========================================================================
# End-to-end pipeline (analogous to ae_kmeans_pipeline)
# ========================================================================

def dec_pipeline(
    X: np.ndarray,
    k: int,
    *,
    latent_dim: int = 0,
    hidden_dims: list[int] | None = None,
    pretrain_epochs: int = 300,
    finetune_epochs: int = 200,
    pretrain_lr: float = 0.01,
    finetune_lr: float = 0.001,
    dropout: float = 0.2,
    gamma: float = 0.1,
    gamma_init: float = 10.0,
    gamma_warmup_epochs: int = 50,
    gamma_anneal_epochs: int = 100,
    conf_threshold: float = 0.0,
    noise_std: float = 0.3,
    update_interval: int = 140,
    device: str = "auto",
    use_layerwise: bool = True,
    epochs_per_layer: int = 50,
    use_sgd_finetune: bool = False,
    use_sgd_pretrain: bool = False,
    centres_lr: float = 0.01,
    batch_size: int = 256,
    early_stopping_patience: int = 50,
    normalize: str = "standard",
) -> dict:
    """End-to-end DEC/IDEC pipeline.

    1. Build deep AE (with optional greedy layer-wise SDAE pretraining).
    2. KMeans init cluster centres on latent codes.
    3. Jointly fine-tune encoder + centres via KL divergence (DEC) or
       KL + reconstruction loss (IDEC when *gamma* > 0).

    Parameters
    ----------
    X : np.ndarray, shape (N, D)
        Input data.
    k : int
        Number of clusters.
    latent_dim : int
        Bottleneck dimension (0 → auto: 10 for high-dim data).
    hidden_dims : list[int] | None
        Encoder hidden widths (None → auto: [500,500,2000] for n_features>256).
    gamma : float
        Reconstruction weight in IDEC mode; 0.0 = vanilla DEC.
    update_interval : int
        Batches between target-distribution P recomputations.
    use_layerwise : bool
        Enable greedy layer-wise SDAE pretraining.
    device : str
        ``"auto"`` uses GPU if available.
    """
    # ---- no-torch fast-path -----------------------------------------------
    if not _HAS_TORCH:
        return _sklearn_fallback(X, k, latent_dim, normalize)

    X_s = _scale_data(X, normalize)
    n_features = X_s.shape[1]

    if device == "auto":
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"

    # ---- auto network architecture -----------------------------------------
    if latent_dim <= 0:
        latent_dim = 32 if n_features > 256 else min(8, max(2, n_features // 4))

    if hidden_dims is None:
        if n_features > 256:
            hidden_dims = [500, 500, 2000]          # DEC paper for MNIST
        elif n_features > 128:
            hidden_dims = [256, 128, 64, 32]
        elif n_features > 64:
            hidden_dims = [128, 64, 32]
        elif n_features > 32:
            hidden_dims = [64, 32]
        else:
            hidden_dims = [max(32, n_features * 2), max(16, n_features)]

    # ---- build model -------------------------------------------------------
    layer_dims = list(hidden_dims) + [latent_dim]  # full encoder path

    if use_layerwise:
        model = _greedy_layerwise_pretrain(
            X_s,
            layer_dims,
            epochs_per_layer=epochs_per_layer,
            batch_size=256,
            learning_rate=pretrain_lr,
            noise_std=noise_std,
            dropout=dropout,
            global_finetune_epochs=min(pretrain_epochs // 3, 100),
            device=device,
        )
    else:
        from ACE_Agent.tools.ae_pipeline import build_autoencoder
        model = build_autoencoder(n_features, hidden_dims, latent_dim, dropout)

    # ---- DEC / IDEC --------------------------------------------------------
    result = dec_train(
        model,
        X_s,
        k,
        pretrain_epochs=pretrain_epochs,
        pretrain_lr=pretrain_lr,
        finetune_epochs=finetune_epochs,
        finetune_lr=finetune_lr,
        gamma=gamma,
        gamma_init=gamma_init,
        gamma_warmup_epochs=gamma_warmup_epochs,
        gamma_anneal_epochs=gamma_anneal_epochs,
        conf_threshold=conf_threshold,
        update_interval=update_interval,
        device=device,
        noise_std=noise_std,
        use_layerwise=use_layerwise,
        layer_dims=layer_dims,
        epochs_per_layer=epochs_per_layer,
        use_sgd_finetune=use_sgd_finetune,
        use_sgd_pretrain=use_sgd_pretrain,
        centres_lr=centres_lr,
        batch_size=batch_size,
        early_stopping_patience=early_stopping_patience,
        normalize=normalize,
    )

    result["metrics"]["latent_dim"] = latent_dim
    result["metrics"]["hidden_dims"] = hidden_dims
    result["metrics"]["layerwise"] = use_layerwise
    return result


# ========================================================================
# Conv-DEC pipeline — DEC/IDEC with Conv-AE backbone for image data
# ========================================================================

def conv_dec_pipeline(
    X: np.ndarray,
    k: int,
    *,
    latent_dim: int = 32,
    input_size: int = 28,
    base_filters: int = 32,
    pretrain_epochs: int = 150,
    finetune_epochs: int = 400,
    pretrain_lr: float = 1e-3,
    finetune_lr: float = 0.001,
    centres_lr: float = 0.01,
    batch_size: int = 128,
    dropout: float = 0.1,
    gamma: float = 0.1,
    gamma_init: float = 10.0,
    gamma_warmup_epochs: int = 50,
    gamma_anneal_epochs: int = 100,
    conf_threshold: float = 0.9,
    noise_std: float = 0.1,
    update_interval: int = 140,
    tol: float = 1e-3,
    device: str = "auto",
    normalize: str = "minmax",
) -> dict:
    """DEC/IDEC with Conv2d/ConvTranspose2d autoencoder backbone.

    Phase 1 — Conv-AE pretraining (Denoising AE).
    Phase 2 — KMeans centre init + KL-divergence joint fine-tuning with
              KL annealing (gamma starts high → decays) and confidence
              threshold (only sharpens P for high-confidence assignments).

    Falls back through lighter Conv-AE → MLP AE → sklearn on OOM.

    Parameters
    ----------
    X : np.ndarray, shape (N, D)
        Flat image data (e.g. 784 for 28×28).
    k : int
        Number of clusters.
    latent_dim : int
        Bottleneck dimension.
    input_size : int
        Square image dimension.
    base_filters : int
        Filters in first Conv layer (doubles per stage).
    gamma : float
        IDEC reconstruction weight; 0.0 = vanilla DEC.
    normalize : str
        ``"minmax"`` (default) for image pixels.
    """
    if not _HAS_TORCH:
        return _sklearn_fallback(X, k, latent_dim, normalize)

    X_s = _scale_data(X, normalize)
    n_features = X_s.shape[1]

    if device == "auto":
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"

    if latent_dim <= 0:
        latent_dim = 32

    # ---- Phase 0: Conv-AE pretraining ------------------------------------
    _logger.info("Conv-DEC: pretraining Conv-AE (%d filters, latent=%d)", base_filters, latent_dim)

    try:
        from ACE_Agent.tools.ae_pipeline import build_conv_autoencoder, train_conv_ae

        conv_ae = build_conv_autoencoder(
            latent_dim=latent_dim,
            input_channels=1,
            input_size=input_size,
            base_filters=base_filters,
            dropout=dropout,
        )

        _ = train_conv_ae(
            conv_ae,
            X_s,
            epochs=pretrain_epochs,
            batch_size=batch_size,
            device=device,
            learning_rate=pretrain_lr,
            noise_std=noise_std,
        )
    except RuntimeError as e:
        if "out of memory" in str(e).lower() and base_filters > 8:
            _logger.warning("Conv-DEC OOM, retrying with base_filters=%d", base_filters // 2)
            return conv_dec_pipeline(
                X, k,
                latent_dim=latent_dim, input_size=input_size,
                base_filters=base_filters // 2,
                pretrain_epochs=pretrain_epochs,
                finetune_epochs=finetune_epochs,
                pretrain_lr=pretrain_lr, finetune_lr=finetune_lr,
                centres_lr=centres_lr, batch_size=batch_size // 2,
                dropout=dropout, gamma=gamma,
                gamma_init=gamma_init, gamma_warmup_epochs=gamma_warmup_epochs,
                gamma_anneal_epochs=gamma_anneal_epochs, conf_threshold=conf_threshold,
                noise_std=noise_std,
                update_interval=update_interval, tol=tol,
                device=device, normalize=normalize,
            )
        _logger.warning("Conv-DEC failed: %s, falling back to MLP DEC", e)
        return dec_pipeline(
            X, k, latent_dim=latent_dim,
            pretrain_epochs=pretrain_epochs, finetune_epochs=finetune_epochs,
            pretrain_lr=pretrain_lr, finetune_lr=finetune_lr,
            dropout=0.2, gamma=gamma,
            gamma_init=gamma_init, conf_threshold=conf_threshold,
            noise_std=0.15, device=device, normalize=normalize,
        )

    # ---- Phase 1: DEC/IDEC fine-tuning -----------------------------------
    # Pass pretrain_epochs=0 to skip the in-place MLP pretraining in dec_train
    result = dec_train(
        conv_ae,
        X_s,
        k,
        pretrain_epochs=0,              # already pretrained above
        pretrain_lr=pretrain_lr,
        finetune_epochs=finetune_epochs,
        finetune_lr=finetune_lr,
        centres_lr=centres_lr,
        batch_size=batch_size,
        gamma=gamma,
        gamma_init=gamma_init,
        gamma_warmup_epochs=gamma_warmup_epochs,
        gamma_anneal_epochs=gamma_anneal_epochs,
        conf_threshold=conf_threshold,
        tol=tol,
        update_interval=update_interval,
        device=device,
        noise_std=noise_std,
        use_layerwise=False,
        normalize=normalize,
    )

    result["metrics"]["latent_dim"] = latent_dim
    result["metrics"]["base_filters"] = base_filters
    result["metrics"]["arch"] = "conv"
    result["metrics"]["backend"] = f"torch-conv-{device}"
    return result


# ========================================================================
# Deep GMM Clustering — DEC-style KL with GMM responsibilities
# ========================================================================

def _gmm_responsibilities(
    z: "torch.Tensor",
    means: "torch.Tensor",
    cov_logits: "torch.Tensor",
    weight_logits: "torch.Tensor",
    temperature: float = 2.0,
    min_std: float = 0.3,
) -> "torch.Tensor":
    """GMM responsibilities (soft assignments) Q, temperature-scaled.

    q_ik ∝ exp( log(π_k · N(z_i|μ_k, Σ_k)) / τ )

    Temperature τ > 1 softens assignments, preventing collapse to one-hot
    that would produce overly-aggressive KL gradients. The minimum std
    bound prevents covariances from shrinking to zero.

    Returns
    -------
    q : (N, K) normalised over K (Σ_k q_ik = 1).
    """
    import math
    import torch.nn.functional as F

    D = z.shape[1]
    K = means.shape[0]

    # Softplus-style clamping: σ ∈ [min_std, ∞)
    cov_diag = F.softplus(cov_logits) + min_std ** 2     # (K, D) ≥ min_std²
    log_det = torch.log(cov_diag).sum(dim=1)              # (K,)
    weights = F.softmax(weight_logits, dim=0)              # (K,) ∈ [0,1], Σ=1

    diff = z.unsqueeze(1) - means.unsqueeze(0)             # (N, K, D)
    mahalanobis = ((diff ** 2) / cov_diag.unsqueeze(0)).sum(dim=2)  # (N, K)

    log_prob = -0.5 * (D * math.log(2 * math.pi) + log_det.unsqueeze(0) + mahalanobis)
    log_joint = log_prob + torch.log(weights + 1e-10).unsqueeze(0)

    # Temperature scaling: τ > 1 makes Q softer (like Student's t heavy tail)
    log_q = (log_joint / temperature) - torch.logsumexp(log_joint / temperature, dim=1, keepdim=True)
    return torch.exp(log_q)


def gmm_cluster_train(
    model: "torch.nn.Module",
    X: np.ndarray,
    k: int,
    *,
    # Pretraining
    pretrain_epochs: int = 100,
    pretrain_lr: float = 1e-3,
    # GMM clustering
    cluster_epochs: int = 200,
    cluster_lr: float = 1e-4,
    batch_size: int = 128,
    gamma: float = 0.5,                # final clustering weight
    gamma_warmup: int = 30,            # epochs at gamma=0 (pure AE)
    gamma_anneal: int = 80,           # epochs to linearly increase 0→gamma
    # GMM kernel
    gmm_temperature: float = 2.0,     # >1 softens assignments (like Student's t heavy tail)
    gmm_min_std: float = 0.3,         # minimum component std (prevents collapse)
    # Misc
    device: str = "cpu",
    noise_std: float = 0.1,
    early_stopping_patience: int = 30,
    normalize: str = "minmax",
) -> dict[str, Any]:
    """Deep GMM clustering — KL(P||Q) with GMM responsibilities.

    Unlike pure GMM log-likelihood (which can collapse to one component),
    this uses DEC-style self-training: Q = GMM responsibilities, P = target
    distribution derived from Q, and the loss is KL(P||Q).

    The GMM component models cluster *shape* via learned diagonal covariances,
    while the KL self-training mechanism prevents collapse.

    Gamma scheduling (INVERSE annealing):
      - Epochs 0..warmup:     gamma=0     (pure AE reconstruction)
      - Epochs warmup..+anneal: gamma increases 0→gamma (clustering gently activates)
      - Epochs after:          gamma=gamma (balanced)

    Parameters
    ----------
    gamma : float
        Final KL clustering weight (default 0.5).
    gamma_warmup : int
        Epochs to train with reconstruction only before clustering activates.
    gamma_anneal : int
        Epochs over which gamma linearly increases from 0 to its final value.
    """
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.metrics import silhouette_score

    X_s = _scale_data(X, normalize)
    X_t = torch.tensor(X_s, dtype=torch.float32)
    n_samples = len(X_t)

    model.to(device)

    # ---- Phase 0: AE pretraining (reconstruction only) --------------------
    from ACE_Agent.tools.ae_pipeline import train_conv_ae

    _ = train_conv_ae(
        model, X_s,
        epochs=pretrain_epochs, batch_size=batch_size,
        device=device, learning_rate=pretrain_lr,
        noise_std=noise_std,
        early_stopping_patience=early_stopping_patience,
    )

    # ---- Extract latent codes and initialise GMM ---------------------------
    model.eval()
    with torch.no_grad():
        _, z0 = model(X_t.to(device))
    z0_np = z0.cpu().numpy()

    # KMeans init for means
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=k, random_state=42, n_init=20)
    km_labels = km.fit_predict(z0_np)
    means_init = torch.tensor(km.cluster_centers_, dtype=torch.float32, device=device)

    # GMM parameters
    means = torch.nn.Parameter(means_init)
    cov_logits = torch.nn.Parameter(torch.zeros(k, z0.shape[1], device=device))   # σ²=1 initially
    weight_logits = torch.nn.Parameter(torch.zeros(k, device=device))              # uniform

    # ---- Phase 1: Joint GMM + reconstruction training ----------------------
    n_val = max(1, int(n_samples * 0.2))
    n_train = n_samples - n_val
    indices = torch.randperm(n_samples)
    train_idx, val_idx = indices[:n_train], indices[n_train:]

    train_loader = DataLoader(
        TensorDataset(X_t[train_idx]),
        batch_size=min(batch_size, n_train), shuffle=False,
    )
    val_loader = DataLoader(
        TensorDataset(X_t[val_idx]),
        batch_size=min(batch_size, n_val), shuffle=False,
    )

    opt = torch.optim.Adam(
        [
            {"params": model.parameters(), "lr": cluster_lr},
            {"params": [means, cov_logits, weight_logits], "lr": cluster_lr * 10},
        ],
        weight_decay=1e-4,
    )
    mse_fn = torch.nn.MSELoss()

    best_val = float("inf")
    best_state = None
    patience = 0
    prev_labels = None
    tol = 0.001  # label change tolerance for early stopping

    for epoch in range(cluster_epochs):
        # ---- gamma schedule (INVERSE annealing: 0 → gamma) ------------------
        if epoch < gamma_warmup:
            cur_gamma = 0.0
        elif epoch < gamma_warmup + gamma_anneal:
            cur_gamma = gamma * (epoch - gamma_warmup) / gamma_anneal
        else:
            cur_gamma = gamma

        # ---- Step 1: Compute target P from full dataset (stable, no grad) -
        if cur_gamma > 0:
            model.eval()
            with torch.no_grad():
                _, z_all = model(X_t.to(device))
                q_all = _gmm_responsibilities(z_all, means, cov_logits, weight_logits, temperature=gmm_temperature, min_std=gmm_min_std)
            p_all = _target_distribution(q_all).detach()
        else:
            p_all = None

        # ---- Step 2: Training -----------------------------------------------
        model.train()
        train_mse = 0.0
        train_kl = 0.0

        if cur_gamma > 0 and p_all is not None:
            # ---- Clustering phase: KL only (DEC-style, no reconstruction) --
            for batch_idx, (batch,) in enumerate(train_loader):
                clean = batch.to(device)
                _, z = model(clean)

                start = batch_idx * batch_size
                end = start + len(batch)
                p_batch = p_all[train_idx[start:end]]
                q = _gmm_responsibilities(z, means, cov_logits, weight_logits, temperature=gmm_temperature, min_std=gmm_min_std)
                kl = (p_batch * (torch.log(p_batch + 1e-10) - torch.log(q + 1e-10))).sum(dim=1).mean()

                opt.zero_grad()
                (cur_gamma * kl).backward()
                opt.step()
                train_kl += kl.item() * len(batch)
            train_kl /= n_train
        else:
            # ---- Warmup phase: reconstruction only (pure AE) ----------------
            for (batch,) in train_loader:
                clean = batch.to(device)
                noisy = clean + torch.randn_like(clean) * noise_std if noise_std > 0 else clean
                recon, _ = model(noisy)
                recon_loss = mse_fn(recon, clean)

                opt.zero_grad()
                recon_loss.backward()
                opt.step()
                train_mse += recon_loss.item() * len(batch)
            train_mse /= n_train

        # Validation — track label change for early stopping
        model.eval()
        val_mse = 0.0
        with torch.no_grad():
            for (batch,) in val_loader:
                clean = batch.to(device)
                recon, _ = model(clean)
                val_mse += mse_fn(recon, clean).item() * len(batch)
        val_mse /= n_val

        # Label-based early stopping (DEC-style: stop when < tol% labels change)
        if cur_gamma > 0:
            with torch.no_grad():
                _, z_full = model(X_t.to(device))
                q_full = _gmm_responsibilities(z_full, means, cov_logits, weight_logits, temperature=gmm_temperature, min_std=gmm_min_std)
                cur_labels = q_full.argmax(dim=1).cpu().numpy()
            if prev_labels is not None:
                delta = (cur_labels != prev_labels).mean()
                if delta < tol and epoch > gamma_warmup + gamma_anneal:
                    break
            prev_labels = cur_labels

        # Save best model by validation MSE
        if val_mse < best_val - 1e-6:
            best_val = val_mse
            best_state = copy.deepcopy(model.state_dict())
            best_means = means.detach().clone()
            best_covs = cov_logits.detach().clone()
            best_weights = weight_logits.detach().clone()
            patience = 0
        else:
            patience += 1
            if patience >= early_stopping_patience:
                break

    # Restore best
    if best_state is not None:
        model.load_state_dict(best_state)
        means.data.copy_(best_means)
        cov_logits.data.copy_(best_covs)
        weight_logits.data.copy_(best_weights)

    # ---- Final clustering --------------------------------------------------
    model.eval()
    with torch.no_grad():
        _, z_all = model(X_t.to(device))

    # Assign labels via GMM responsibilities
    z_all_np = z_all.cpu().numpy()
    means_np = means.detach().cpu().numpy()
    cov_np = np.log(1 + np.exp(cov_logits.detach().cpu().numpy())) + gmm_min_std ** 2
    weights_np = F.softmax(weight_logits.detach(), dim=0).cpu().numpy()

    # Compute responsibilities
    final_labels = _gmm_assign_labels(z_all_np, means_np, cov_np, weights_np)

    sil = float(silhouette_score(z_all_np, final_labels))

    return {
        "labels": final_labels.tolist(),
        "metrics": {
            "score": sil,
            "score_source": "silhouette",
            "silhouette": sil,
            "latent_dim": z_all.shape[1],
            "pretrain_epochs": pretrain_epochs,
            "cluster_epochs": epoch + 1,
            "gamma": gamma,
            "cluster_method": "GMM-loglik",
            "backend": f"torch-{device}",
        },
        "plot_path": "",
    }


def _gmm_assign_labels(
    z: np.ndarray,
    means: np.ndarray,
    cov_diag: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """Hard cluster assignment via GMM responsibilities (MAP).

    Assigns each sample to the cluster k that maximises
    log π_k + log N(z_i | μ_k, diag(σ²_k)).
    """
    from scipy.special import logsumexp

    D = z.shape[1]
    K = means.shape[0]

    diff = z[:, None, :] - means[None, :, :]          # (N, K, D)
    sq_maha = ((diff ** 2) / cov_diag[None, :, :]).sum(axis=2)  # (N, K)
    log_det = np.log(cov_diag).sum(axis=1)              # (K,)
    log_prob = -0.5 * (D * np.log(2 * np.pi) + log_det[None, :] + sq_maha)  # (N, K)
    log_joint = log_prob + np.log(weights + 1e-10)[None, :]

    return np.argmax(log_joint, axis=1)


def conv_gmm_pipeline(
    X: np.ndarray,
    k: int,
    *,
    latent_dim: int = 32,
    input_size: int = 28,
    base_filters: int = 32,
    pretrain_epochs: int = 150,
    cluster_epochs: int = 200,
    pretrain_lr: float = 1e-3,
    cluster_lr: float = 1e-4,
    batch_size: int = 128,
    dropout: float = 0.1,
    gamma: float = 0.5,
    gamma_warmup: int = 30,
    gamma_anneal: int = 80,
    noise_std: float = 0.1,
    gmm_temperature: float = 2.0,
    gmm_min_std: float = 0.3,
    device: str = "auto",
    normalize: str = "minmax",
) -> dict:
    """End-to-end Deep GMM clustering with Conv-AE backbone.

    Phase 1 — Conv-AE pretraining.
    Phase 2 — DEC-style KL(P||Q) with GMM responsibilities (not Student's t).

    GMM covariances are softplus-regularised (min_std prevents collapse)
    and temperature scaling (τ > 1) softens assignments for stable gradients.

    Gamma is ANNEALED IN (0 → gamma) rather than decayed, so the
    AE representations stabilise before clustering activates.

    Falls back through lighter Conv-AE → sklearn on OOM.
    """
    if not _HAS_TORCH:
        return _sklearn_fallback(X, k, latent_dim, normalize)

    X_s = _scale_data(X, normalize)

    if device == "auto":
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"

    if latent_dim <= 0:
        latent_dim = 32

    try:
        from ACE_Agent.tools.ae_pipeline import build_conv_autoencoder

        conv_ae = build_conv_autoencoder(
            latent_dim=latent_dim,
            input_channels=1,
            input_size=input_size,
            base_filters=base_filters,
            dropout=dropout,
        )

        result = gmm_cluster_train(
            conv_ae, X_s, k,
            pretrain_epochs=pretrain_epochs,
            pretrain_lr=pretrain_lr,
            cluster_epochs=cluster_epochs,
            cluster_lr=cluster_lr,
            batch_size=batch_size,
            gamma=gamma,
            gamma_warmup=gamma_warmup,
            gamma_anneal=gamma_anneal,
            noise_std=noise_std,
            gmm_temperature=gmm_temperature,
            gmm_min_std=gmm_min_std,
            device=device,
            normalize=normalize,
        )
    except RuntimeError as e:
        if "out of memory" in str(e).lower() and base_filters > 8:
            _logger.warning("Conv-GMM OOM, retrying with base_filters=%d", base_filters // 2)
            return conv_gmm_pipeline(
                X, k,
                latent_dim=latent_dim, input_size=input_size,
                base_filters=base_filters // 2,
                pretrain_epochs=pretrain_epochs,
                cluster_epochs=cluster_epochs,
                pretrain_lr=pretrain_lr, cluster_lr=cluster_lr,
                batch_size=batch_size // 2,
                dropout=dropout, gamma=gamma,
                gamma_warmup=gamma_warmup, gamma_anneal=gamma_anneal,
                noise_std=noise_std,
                gmm_temperature=gmm_temperature, gmm_min_std=gmm_min_std,
                device=device, normalize=normalize,
            )
        _logger.warning("Conv-GMM failed: %s, falling back to sklearn", e)
        return _sklearn_fallback(X, k, latent_dim, normalize)

    result["metrics"]["latent_dim"] = latent_dim
    result["metrics"]["base_filters"] = base_filters
    result["metrics"]["arch"] = "conv-gmm"
    result["metrics"]["backend"] = f"torch-conv-{device}"
    return result


# ---------------------------------------------------------------------------
# Sklearn fallback (when torch unavailable)
# ---------------------------------------------------------------------------
def _sklearn_fallback(
    X: np.ndarray, k: int, latent_dim: int = 0, normalize: str = "standard"
) -> dict:
    """PCA + KMeans/GMM fallback — same shape as dec_pipeline output."""
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_score

    X_s = _scale_data(X, normalize)
    n_features = X_s.shape[1]
    if latent_dim <= 0:
        latent_dim = min(8, max(2, n_features // 4))

    reduced = PCA(n_components=min(latent_dim, n_features), random_state=42).fit_transform(X_s)
    from sklearn.cluster import KMeans

    labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(reduced)
    sil = float(silhouette_score(reduced, labels))

    return {
        "labels": labels.tolist(),
        "metrics": {
            "score": sil,
            "score_source": "silhouette",
            "silhouette": sil,
            "latent_dim": latent_dim,
            "method": "DEC-fallback",
            "backend": "sklearn-fallback",
        },
        "plot_path": "",
    }
