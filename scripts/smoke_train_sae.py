"""
Smoke-test SAE trainer + Phase-2 peek for the C2S-Scale adapter.

Trains a TopK SAE on one extracted layer and reports the Phase-1 validation signatures
(variance explained, dead features, mean pairwise decoder cosine), then does a light
Phase-2 "does it look biological?" peek: for the highest-frequency features, the top genes
by mean activation (using row_gene_names from extraction).

Usage:
  python smoke_train_sae.py --run runs/2b_smoke --layer 13 --expansion 4 --k 32 --epochs 4
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from sae_topk import SAETrainer, TopKSAE


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True, help="extraction output dir")
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--expansion", type=int, default=4)
    p.add_argument("--k", type=int, default=32)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--peek-features", type=int, default=15)
    p.add_argument("--decoder-norm-dim", type=int, default=0,
                   help="0 = unit-norm feature-direction columns (correct); 1 = reference's row norm")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    acts = np.load(os.path.join(args.run, f"layer_{args.layer:02d}_activations.npy"))
    with open(os.path.join(args.run, "row_gene_names.json")) as f:
        gene_names = np.array(json.load(f))
    n, d = acts.shape
    print(f"layer {args.layer}: {n} rows x {d} dims | expansion {args.expansion} -> "
          f"d_SAE {args.expansion * d} | k {args.k}")

    # mean-centre (persist mu like the reference)
    mu = acts.mean(0, keepdims=True)
    perm = np.random.permutation(n)
    n_val = max(1, int(args.val_frac * n))
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    Xtr = (acts[tr_idx] - mu).astype(np.float32)
    Xval = torch.tensor((acts[val_idx] - mu).astype(np.float32), device=device)

    sae = TopKSAE(d, args.expansion * d, args.k, decoder_norm_dim=args.decoder_norm_dim)
    trainer = SAETrainer(sae, lr=3e-4, device=device)
    for ep in range(args.epochs):
        loss = trainer.train_epoch(Xtr, batch_size=args.batch_size, log_every=200)
        print(f"epoch {ep}: mean train mse {loss:.6f}")

    # ---- Phase-1 signatures on held-out val ----
    sae.eval()
    with torch.no_grad():
        xhat, hval, _ = sae(Xval)
        tot = Xval.var(0).sum().item()
        res = (Xval - xhat).var(0).sum().item()
        var_expl = 1 - res / max(tot, 1e-10)
        freq = (hval > 0).float().mean(0).cpu().numpy()
        dead = int((freq == 0).sum())
        dirs = sae.decoder_directions().cpu().numpy()  # (n_features, d)
        s = dirs[np.random.choice(len(dirs), min(512, len(dirs)), replace=False)]
        cos = s @ s.T
        off = cos[~np.eye(len(s), dtype=bool)]
        mean_abs_cos = float(np.abs(off).mean())
    print("\n=== Phase-1 signatures (target: var>0.75, dead<2%, cos~0.03-0.05) ===")
    print(f"  variance explained : {var_expl:.4f}")
    print(f"  dead features      : {dead}/{sae.n_features} ({100*dead/sae.n_features:.2f}%)")
    print(f"  mean |decoder cos| : {mean_abs_cos:.4f}")

    # ---- Phase-2 peek: top genes for the most-active features ----
    uniq, inv = np.unique(gene_names, return_inverse=True)
    M = np.zeros((sae.n_features, len(uniq)), dtype=np.float64)
    fcount = np.zeros(sae.n_features)
    Xall = torch.tensor((acts - mu).astype(np.float32), device=device)
    with torch.no_grad():
        for start in range(0, n, 8192):
            h = sae.encode(Xall[start:start + 8192])[0].cpu().numpy()
            r, fidx = np.nonzero(h > 0)
            np.add.at(M, (fidx, inv[start + r]), h[r, fidx])
            np.add.at(fcount, fidx, 1.0)
    order = np.argsort(-fcount)
    order = order[fcount[order] > 0][: args.peek_features]
    print(f"\n=== Phase-2 peek: top genes for {len(order)} highest-frequency features "
          f"(<CELL> excluded if present) ===")
    for f in order:
        top = uniq[np.argsort(-M[f])[:10]]
        top = [g for g in top if g != "<CELL>"]
        print(f"  F{int(f):<5d} freq {fcount[f]/n:.3f} | {' '.join(top)}")

    os.makedirs(os.path.join(args.run, "sae"), exist_ok=True)
    sae.save(os.path.join(args.run, "sae", f"sae_L{args.layer:02d}.pt"))
    np.save(os.path.join(args.run, "sae", f"mu_L{args.layer:02d}.npy"), mu)
    print(f"\nsaved SAE -> {args.run}/sae/sae_L{args.layer:02d}.pt")


if __name__ == "__main__":
    main()
