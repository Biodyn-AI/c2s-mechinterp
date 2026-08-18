"""
Phase 1 (atlas): train a TopK SAE on one extracted layer, save weights + mean + metrics.

Reads layer_{L:02d}_activations.npy (float16 or float32) from the run dir, casts to float32,
mean-centres, trains a TopK SAE (decoder_norm_dim=0 — the correct axis, see README), and writes:
  sae/sae_L{L:02d}.pt     TopKSAE checkpoint (loaded by feature_catalog/coactivation/cross_layer)
  sae/mu_L{L:02d}.npy     (1, d_model) float32 pre-encoder mean
  sae/train_L{L:02d}.json Phase-1 signatures (var explained, dead %, decoder cos)

Usage:
  python train_sae.py --run runs/2b_atlas --layer 13 --expansion 4 --k 32 --epochs 8
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

try:  # works as `python -m c2s_mechinterp.<mod>` and as a plain script
    from .sae_topk import SAETrainer, TopKSAE
except ImportError:
    from sae_topk import SAETrainer, TopKSAE


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--expansion", type=int, default=4)
    p.add_argument("--k", type=int, default=32)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--val-frac", type=float, default=0.05)
    p.add_argument("--decoder-norm-dim", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sae-subdir", default="sae",
                   help="output subdir under --run for SAE files "
                        "(e.g. sae_matched_x2 to avoid clobbering the existing sae/)")
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    acts = np.load(os.path.join(args.run, f"layer_{args.layer:02d}_activations.npy")).astype(np.float32)
    if not np.isfinite(acts).all():  # fp16 overflow (massive activations) -> inf; clip so training can't NaN
        n_bad = int((~np.isfinite(acts)).sum())
        print(f"  WARNING: {n_bad} non-finite activations -> clipping (re-extract in float32 for correctness)")
        acts = np.nan_to_num(acts, nan=0.0, posinf=60000.0, neginf=-60000.0)
    n, d = acts.shape
    print(f"L{args.layer:02d}: {n:,} rows x {d} | d_SAE {args.expansion * d} | k {args.k} | {device}")

    mu = acts.mean(0, keepdims=True).astype(np.float32)
    perm = np.random.permutation(n)
    n_val = max(1, int(args.val_frac * n))
    Xtr = acts[perm[n_val:]] - mu
    Xval = torch.tensor(acts[perm[:n_val]] - mu, device=device)

    sae = TopKSAE(d, args.expansion * d, args.k, decoder_norm_dim=args.decoder_norm_dim)
    trainer = SAETrainer(sae, lr=3e-4, device=device)
    for ep in range(args.epochs):
        loss = trainer.train_epoch(Xtr, batch_size=args.batch_size, log_every=500)
        print(f"  epoch {ep}: train mse {loss:.5f}")

    sae.eval()
    with torch.no_grad():
        xhat, hval, _ = sae(Xval)
        tot = Xval.var(0).sum().item()
        var_expl = 1 - (Xval - xhat).var(0).sum().item() / max(tot, 1e-10)
        dead = int((hval > 0).float().mean(0).eq(0).sum())
        dirs = sae.decoder_directions().cpu().numpy()
        s = dirs[np.random.choice(len(dirs), min(512, len(dirs)), replace=False)]
        cos = s @ s.T
        mean_cos = float(np.abs(cos[~np.eye(len(s), dtype=bool)]).mean())

    os.makedirs(os.path.join(args.run, args.sae_subdir), exist_ok=True)
    sae.save(os.path.join(args.run, args.sae_subdir, f"sae_L{args.layer:02d}.pt"))
    np.save(os.path.join(args.run, args.sae_subdir, f"mu_L{args.layer:02d}.npy"), mu)
    metrics = {"layer": args.layer, "n_rows": n, "d_model": d, "n_features": args.expansion * d,
               "k": args.k, "epochs": args.epochs, "variance_explained": var_expl,
               "dead_features": dead, "dead_frac": dead / (args.expansion * d),
               "mean_decoder_cos": mean_cos}
    with open(os.path.join(args.run, args.sae_subdir, f"train_L{args.layer:02d}.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  var_expl {var_expl:.4f} | dead {dead}/{args.expansion*d} "
          f"({100*dead/(args.expansion*d):.2f}%) | dec_cos {mean_cos:.4f}")


if __name__ == "__main__":
    main()
