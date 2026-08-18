"""
TopK Sparse Autoencoder — vendored from repos/bio-sae/src/sae_model.py (pinned commit
9277f5d) to keep the C2S-Scale atlas comparable to the published Geneformer/scGPT atlas.

One deliberate change from the reference: the decoder unit-normalization axis is exposed
as `decoder_norm_dim` and defaults to 0. Feature directions are the COLUMNS of the
(d_model, n_features) decoder weight, i.e. dim=0. The reference used dim=1 (rows); we must
reconcile against repos/bio-sae/src/02_train_sae.py and the reference's decoder-cosine
computation before trusting cross-model comparisons (see projects/c2s-scale/README.md).
"""

from __future__ import annotations

import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class TopKSAE(nn.Module):
    """Encoder d_model -> n_features with hard TopK; decoder n_features -> d_model."""

    def __init__(self, d_model, n_features, k, decoder_norm_dim: int = 0, dtype=torch.float32):
        super().__init__()
        self.d_model = d_model
        self.n_features = n_features
        self.k = k
        self.decoder_norm_dim = decoder_norm_dim  # 0 == unit-norm feature-direction columns
        self.W_enc = nn.Linear(d_model, n_features, bias=True, dtype=dtype)
        self.W_dec = nn.Linear(n_features, d_model, bias=True, dtype=dtype)
        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_uniform_(self.W_enc.weight, nonlinearity="relu")
        nn.init.zeros_(self.W_enc.bias)
        nn.init.kaiming_uniform_(self.W_dec.weight, nonlinearity="linear")
        self.normalize_decoder()
        nn.init.zeros_(self.W_dec.bias)

    def encode(self, x):
        h = self.W_enc(x)
        topk_values, topk_indices = torch.topk(h, self.k, dim=-1)
        topk_values = F.relu(topk_values)
        h_sparse = torch.zeros_like(h)
        h_sparse.scatter_(-1, topk_indices, topk_values)
        return h_sparse, topk_indices

    def decode(self, h_sparse):
        return self.W_dec(h_sparse)

    def forward(self, x):
        h_sparse, topk_indices = self.encode(x)
        return self.decode(h_sparse), h_sparse, topk_indices

    def loss(self, x, x_hat):
        return F.mse_loss(x, x_hat)  # MSE only — TopK enforces sparsity, no L1

    @torch.no_grad()
    def normalize_decoder(self):
        self.W_dec.weight.data = F.normalize(self.W_dec.weight.data, dim=self.decoder_norm_dim)

    @torch.no_grad()
    def decoder_directions(self) -> torch.Tensor:
        """(n_features, d_model) unit feature directions, for cosine/tracking analyses."""
        return F.normalize(self.W_dec.weight.data.T, dim=1)

    def save(self, path):
        torch.save(
            {
                "model_state_dict": self.state_dict(),
                "d_model": self.d_model,
                "n_features": self.n_features,
                "k": self.k,
                "decoder_norm_dim": self.decoder_norm_dim,
            },
            path,
        )

    @classmethod
    def load(cls, path, device="cpu"):
        s = torch.load(path, map_location=device, weights_only=False)
        m = cls(s["d_model"], s["n_features"], s["k"], s.get("decoder_norm_dim", 0))
        m.load_state_dict(s["model_state_dict"])
        return m.to(device)


class SAETrainer:
    """Adam training loop with per-step decoder renorm, logging, checkpointing."""

    def __init__(self, sae, lr=3e-4, device="cpu"):
        self.sae = sae.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(sae.parameters(), lr=lr)
        self.step = 0
        self.log = []

    def train_step(self, batch):
        self.sae.train()
        batch = batch.to(self.device)
        x_hat, _, _ = self.sae(batch)
        loss = self.sae.loss(batch, x_hat)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.sae.normalize_decoder()  # non-negotiable: project decoder back to unit norm
        self.step += 1
        return loss.item()

    def train_epoch(self, activations, batch_size=4096, log_every=1000,
                    checkpoint_dir=None, checkpoint_every=10000):
        n = len(activations)
        indices = np.random.permutation(n)
        epoch_losses = []
        for start in range(0, n, batch_size):
            batch_idx = indices[start:min(start + batch_size, n)]
            batch = torch.tensor(activations[batch_idx], dtype=torch.float32)
            epoch_losses.append(self.train_step(batch))
            if self.step % log_every == 0:
                with torch.no_grad():
                    self.sae.eval()
                    bg = batch.to(self.device)
                    x_hat, h_sparse, _ = self.sae(bg)
                    mse = F.mse_loss(bg, x_hat).item()
                    total_var = bg.var(dim=0).sum().item()
                    resid_var = (bg - x_hat).var(dim=0).sum().item()
                    var_expl = 1.0 - resid_var / max(total_var, 1e-10)
                    freq = (h_sparse > 0).float().mean(dim=0)
                    dead = int((freq == 0).sum())
                self.log.append({"step": self.step, "loss": mse, "var_explained": var_expl,
                                 "dead_features": dead})
                print(f"  step {self.step:>6d} | mse {mse:.6f} | var_expl {var_expl:.4f} | "
                      f"dead {dead}/{self.sae.n_features}")
            if checkpoint_dir and self.step % checkpoint_every == 0:
                self.sae.save(os.path.join(checkpoint_dir, f"sae_step_{self.step}.pt"))
        return float(np.mean(epoch_losses))

    def save_log(self, path):
        with open(path, "w") as f:
            json.dump(self.log, f, indent=2)
