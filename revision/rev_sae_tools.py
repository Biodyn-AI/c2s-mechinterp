"""Sparse-autoencoder utilities for the revision runs.

Subcommands
  train      same recipe and random-number use as code/train_sae.py (TopK SAE, Adam 3e-4, batch 4096,
             5% held-out split, decoder columns unit-norm), with free k / dictionary size / seed / epochs,
             plus wall time, peak memory and firing statistics over ALL rows.
  eval       evaluate a stored SAE on (re-extracted) activations, on the same held-out split.
  untrained  build a control SAE with random encoder directions whose row norms match a trained SAE
             (bias kept or zeroed), zero training steps.
  match      share of features of SAE A whose decoder direction has |cosine| >= thr with some feature of B.

All paths follow the atlas run-dir layout: <run>/layer_XX_activations.npy, <run>/<sae-subdir>/sae_LXX.pt,
mu_LXX.npy, train_LXX.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np


def projection_spec(path, width, layer):
    """Validate an optional E4 preset without loading the activation matrix into RAM."""
    projection = {'mode': 'identity', 'original_width': width, 'removed_dimensions': [],
                  'variance_reference': 'original input'}
    if not path:
        return projection
    with open(path, 'rb') as handle:
        raw = handle.read()
    preset = json.loads(raw)
    if isinstance(preset, dict):
        if 'layer' in preset and preset['layer']!=layer:
            raise ValueError('dimension preset belongs to a different layer')
        if 'indices' not in preset:
            raise ValueError('dimension preset is missing its indices list')
        indices = preset['indices']
    else:
        indices = preset
    if not isinstance(indices, list) or any(type(index) is not int for index in indices):
        raise ValueError('--drop-dims requires a JSON integer list or an object containing indices')
    if len(set(indices))!=len(indices) or any(index<0 or index>=width for index in indices):
        raise ValueError('removed dimensions must be unique and within the activation width')
    if len(indices)>=width:
        raise ValueError('cannot remove every input dimension')
    projection.update(mode='zero_selected_columns', removed_dimensions=sorted(indices),
                      retained_input_dimensions=width-len(indices), preset_file=str(path),
                      preset_sha256=hashlib.sha256(raw).hexdigest(),
                      variance_reference='projected input after zeroing selected columns')
    return projection


def zero_columns_in_place(activations, indices):
    """Mutate only the existing RAM copy; writing through a memory map is forbidden."""
    if isinstance(activations, np.memmap) or not activations.flags.writeable:
        raise ValueError('dimension removal requires a writable RAM array, never a memory-mapped file')
    if activations.ndim!=2 or any(type(index) is not int for index in indices):
        raise ValueError('dimension removal requires a matrix and integer column indices')
    if len(set(indices))!=len(indices) or any(index<0 or index>=activations.shape[1] for index in indices):
        raise ValueError('invalid removed input dimensions')
    activations[:, indices] = 0.


def _load_acts(run, layer):
    acts = np.load(os.path.join(run, f"layer_{layer:02d}_activations.npy")).astype(np.float32)
    n_bad = int((~np.isfinite(acts)).sum())
    if n_bad:
        raise ValueError(f'{n_bad} nonfinite activations at layer {layer}; re-extract in float32')
    if acts.ndim != 2 or acts.shape[0] < 2 or acts.shape[1] < 1:
        raise ValueError('activations must be a nonempty matrix with at least two rows')
    return acts, n_bad


def _firing_stats(sae, acts, mu, device, chunk=20000):
    import torch
    cnt = torch.zeros(sae.n_features, dtype=torch.float64, device=device)
    l0 = 0.0
    se = 0.0
    tot = 0.0
    mu_t = torch.tensor(mu, device=device)
    col_mean = torch.tensor(acts.mean(0), device=device) - mu_t[0]
    with torch.no_grad():
        for s in range(0, len(acts), chunk):
            x = torch.tensor(acts[s:s + chunk], device=device) - mu_t
            xh, h, _ = sae(x)
            if not bool(torch.isfinite(xh).all() and torch.isfinite(h).all()):
                raise ValueError('nonfinite SAE outputs during firing evaluation')
            cnt += (h > 0).double().sum(0)
            l0 += float((h > 0).float().sum())
            se += float(((x - xh) ** 2).sum())
            tot += float(((x - col_mean) ** 2).sum())
    freq = (cnt / len(acts)).cpu().numpy()
    alive = freq > 0
    q = np.quantile(freq[alive], [0.01, 0.1, 0.5, 0.9, 0.99]).tolist() if alive.any() else [0] * 5
    return {"dead_all_rows": int((~alive).sum()), "dead_frac_all_rows": float((~alive).mean()),
            "mean_active_per_position": l0 / len(acts),
            "firing_freq_quantiles_1_10_50_90_99": q,
            "share_features_firing_gt_10pct": float((freq > 0.10).mean()),
            "share_features_firing_lt_1e-4": float(((freq < 1e-4) & alive).mean()),
            "fraction_variance_unexplained_all_rows": se / max(tot, 1e-12)}, freq


def cmd_train(a):
    import torch
    from sae_topk import SAETrainer, TopKSAE
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t_all = time.time()
    header = np.load(os.path.join(a.run, f'layer_{a.layer:02d}_activations.npy'), mmap_mode='r')
    if header.ndim!=2:
        raise ValueError('activations must be a matrix')
    projection = projection_spec(a.drop_dims, header.shape[1], a.layer)
    if a.drop_dims and os.path.realpath(os.path.join(a.run, a.sae_subdir))==os.path.realpath(os.path.join(a.run, 'sae')):
        raise ValueError('--drop-dims requires a separate --sae-subdir, such as sae_dropmassive')
    if a.drop_dims and not projection['removed_dimensions']:
        out = os.path.join(a.run, a.sae_subdir)
        os.makedirs(out, exist_ok=True)
        with open(os.path.join(out, f'train_L{a.layer:02d}.json'), 'w') as handle:
            json.dump({'layer': a.layer, 'status': 'skipped_no_dimensions', 'input_projection': projection,
                       'reason': 'No selected dimensions; no training or causal-mechanism comparison performed'}, handle, indent=2)
        print(f'L{a.layer:02d}: no selected dimensions, E4 skipped', flush=True)
        return
    del header
    acts, n_bad = _load_acts(a.run, a.layer)
    if projection['removed_dimensions']:
        zero_columns_in_place(acts, projection['removed_dimensions'])
    n, d = acts.shape
    n_features = a.n_features or a.expansion * d
    print(f"L{a.layer:02d}: {n:,} rows x {d} | features {n_features} | k {a.k} | epochs {a.epochs} | "
          f"seed {a.seed} | {device}", flush=True)
    mu = acts.mean(0, keepdims=True).astype(np.float32)
    perm = np.random.permutation(n)
    n_val = max(1, int(a.val_frac * n))
    Xtr = acts[perm[n_val:]] - mu
    Xval = torch.tensor(acts[perm[:n_val]] - mu, device=device)
    sae = TopKSAE(d, n_features, a.k, decoder_norm_dim=0)
    trainer = SAETrainer(sae, lr=a.lr, device=device)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    losses = []
    for ep in range(a.epochs):
        loss = trainer.train_epoch(Xtr, batch_size=a.batch_size, log_every=500)
        if not np.isfinite(loss):
            raise ValueError(f'nonfinite SAE training loss at epoch {ep}')
        losses.append(loss)
        print(f"  epoch {ep}: train mse {loss:.5f}", flush=True)
    train_s = time.time() - t0
    sae.eval()
    with torch.no_grad():
        xhat, hval, _ = sae(Xval)
        if not bool(torch.isfinite(xhat).all() and torch.isfinite(hval).all()):
            raise ValueError('nonfinite SAE validation outputs')
        tot = Xval.var(0).sum().item()
        var_expl = 1 - (Xval - xhat).var(0).sum().item() / max(tot, 1e-10)
        dead = int((hval > 0).float().mean(0).eq(0).sum())
        dirs = sae.decoder_directions().cpu().numpy()
        s = dirs[np.random.choice(len(dirs), min(512, len(dirs)), replace=False)]
        cos = s @ s.T
        mean_cos = float(np.abs(cos[~np.eye(len(s), dtype=bool)]).mean())
    del Xtr
    extra, freq = _firing_stats(sae, acts, mu, device)
    out = os.path.join(a.run, a.sae_subdir)
    os.makedirs(out, exist_ok=True)
    sae.save(os.path.join(out, f"sae_L{a.layer:02d}.pt"))
    np.save(os.path.join(out, f"mu_L{a.layer:02d}.npy"), mu)
    np.save(os.path.join(out, f"freq_L{a.layer:02d}.npy"), freq.astype(np.float32))
    metrics = {"layer": a.layer, "status": "trained", "input_projection": projection,
               "n_rows": n, "d_model": d, "n_features": n_features, "k": a.k,
               "epochs": a.epochs, "seed": a.seed, "lr": a.lr, "batch_size": a.batch_size,
               "variance_explained": var_expl, "dead_features": dead, "dead_frac": dead / n_features,
               "mean_decoder_cos": mean_cos, "epoch_train_mse": losses, "n_nonfinite_clipped": n_bad,
               "train_seconds": round(train_s, 1), "total_seconds": round(time.time() - t_all, 1),
               "peak_gpu_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2) if device == "cuda" else None,
               "positions_per_feature": n / n_features, **extra}
    if a.drop_dims:
        metrics['variance_explained_projected_input'] = var_expl
    with open(os.path.join(out, f"train_L{a.layer:02d}.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  var_expl {var_expl:.4f} | dead(val) {dead}/{n_features} ({100 * dead / n_features:.2f}%) | "
          f"dead(all rows) {extra['dead_all_rows']} | dec_cos {mean_cos:.4f} | train {train_s:.0f}s", flush=True)


def cmd_eval(a):
    import torch
    from sae_topk import TopKSAE
    device = "cuda" if torch.cuda.is_available() else "cpu"
    np.random.seed(a.seed)
    acts, n_bad = _load_acts(a.run, a.layer)
    n, d = acts.shape
    sdir = os.path.join(a.sae_run or a.run, a.sae_subdir)
    stored = {}
    tj = os.path.join(sdir, f"train_L{a.layer:02d}.json")
    if os.path.exists(tj):
        stored = json.load(open(tj))
    projection = stored.get('input_projection', {'mode': 'identity', 'removed_dimensions': [],
                                                 'variance_reference': 'original input'})
    if projection['mode']=='zero_selected_columns':
        zero_columns_in_place(acts, projection['removed_dimensions'])
    sae = TopKSAE.load(os.path.join(sdir, f"sae_L{a.layer:02d}.pt"), device=device).eval()
    mu = np.load(os.path.join(sdir, f"mu_L{a.layer:02d}.npy")).astype(np.float32).reshape(1, d)
    perm = np.random.permutation(n)
    n_val = max(1, int(a.val_frac * n))
    Xval = torch.tensor(acts[perm[:n_val]] - mu, device=device)
    with torch.no_grad():
        xhat, hval, _ = sae(Xval)
        if not bool(torch.isfinite(xhat).all() and torch.isfinite(hval).all()):
            raise ValueError('nonfinite SAE evaluation outputs')
        tot = Xval.var(0).sum().item()
        var_expl = 1 - (Xval - xhat).var(0).sum().item() / max(tot, 1e-10)
        dead = int((hval > 0).float().mean(0).eq(0).sum())
    extra, _ = _firing_stats(sae, acts, mu, device)
    res = {"layer": a.layer, "variance_explained_reextracted": var_expl, "dead_val_reextracted": dead,
           "variance_explained_stored": stored.get("variance_explained"),
           "dead_val_stored": stored.get("dead_features"), "n_features": sae.n_features,
           "mu_shift_rel": float(np.linalg.norm(acts.mean(0) - mu[0]) / max(np.linalg.norm(mu[0]), 1e-9)),
           "n_nonfinite_clipped": n_bad, "input_projection": projection, **extra}
    with open(a.out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"L{a.layer:02d}: var_expl re-extracted {var_expl:.4f} vs stored "
          f"{stored.get('variance_explained', float('nan')):.4f} | dead(val) {dead} vs "
          f"{stored.get('dead_features')} | dead(all) {extra['dead_all_rows']} | "
          f"mu shift {res['mu_shift_rel']:.2e}", flush=True)


def cmd_untrained(a):
    import torch
    from sae_topk import TopKSAE
    src = os.path.join(a.sae_run, a.sae_subdir)
    trained = TopKSAE.load(os.path.join(src, f"sae_L{a.layer:02d}.pt"), device="cpu")
    torch.manual_seed(a.seed)
    new = TopKSAE(trained.d_model, trained.n_features, trained.k, decoder_norm_dim=0)
    with torch.no_grad():
        w = new.W_enc.weight
        target = trained.W_enc.weight.norm(dim=1, keepdim=True)
        w.mul_(target / w.norm(dim=1, keepdim=True).clamp(min=1e-12))
        if a.zero_bias:
            new.W_enc.bias.zero_()
        else:
            new.W_enc.bias.copy_(trained.W_enc.bias)
        new.W_dec.bias.copy_(trained.W_dec.bias)
        rel = float(((new.W_enc.weight.norm(dim=1) - target[:, 0]).abs() / target[:, 0].clamp(min=1e-12)).max())
    out = os.path.join(a.out_run, "sae")
    os.makedirs(out, exist_ok=True)
    new.save(os.path.join(out, f"sae_L{a.layer:02d}.pt"))
    np.save(os.path.join(out, f"mu_L{a.layer:02d}.npy"),
            np.load(os.path.join(src, f"mu_L{a.layer:02d}.npy")))
    with open(os.path.join(out, f"untrained_L{a.layer:02d}.json"), "w") as f:
        json.dump({"layer": a.layer, "seed": a.seed, "zero_bias": bool(a.zero_bias),
                   "max_rel_row_norm_error": rel, "source": src}, f, indent=2)
    print(f"untrained control L{a.layer:02d}: zero_bias={a.zero_bias} | max rel norm err {rel:.2e}")


def cmd_match(a):
    import torch
    from sae_topk import TopKSAE
    device = "cuda" if torch.cuda.is_available() else "cpu"
    A = TopKSAE.load(a.sae_a, device=device).decoder_directions()
    B = TopKSAE.load(a.sae_b, device=device).decoder_directions()
    best = torch.zeros(len(A), device=device)
    for s in range(0, len(A), 2048):
        best[s:s + 2048] = (A[s:s + 2048] @ B.T).abs().max(dim=1).values
    best = best.cpu().numpy()
    res = {"sae_a": a.sae_a, "sae_b": a.sae_b, "n_a": int(len(A)), "n_b": int(len(B)),
           "share_matched_0.7": float((best >= 0.7).mean()), "share_matched_0.9": float((best >= 0.9).mean()),
           "mean_best_cos": float(best.mean()), "median_best_cos": float(np.median(best))}
    with open(a.out, "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--code-dir", required=True)
    sub = p.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--run", required=True)
    t.add_argument("--layer", type=int, required=True)
    t.add_argument("--expansion", type=int, default=4)
    t.add_argument("--n-features", type=int, default=0)
    t.add_argument("--k", type=int, default=32)
    t.add_argument("--epochs", type=int, default=6)
    t.add_argument("--batch-size", type=int, default=4096)
    t.add_argument("--val-frac", type=float, default=0.05)
    t.add_argument("--lr", type=float, default=3e-4)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--sae-subdir", default="sae")
    t.add_argument('--drop-dims', default='', help='E4 JSON dimension indices; zero only the RAM input, save in a separate --sae-subdir')
    e = sub.add_parser("eval")
    e.add_argument("--run", required=True)
    e.add_argument("--sae-run", default="")
    e.add_argument("--sae-subdir", default="sae")
    e.add_argument("--layer", type=int, required=True)
    e.add_argument("--val-frac", type=float, default=0.05)
    e.add_argument("--seed", type=int, default=0)
    e.add_argument("--out", required=True)
    u = sub.add_parser("untrained")
    u.add_argument("--sae-run", required=True)
    u.add_argument("--sae-subdir", default="sae")
    u.add_argument("--layer", type=int, required=True)
    u.add_argument("--out-run", required=True)
    u.add_argument("--zero-bias", action="store_true")
    u.add_argument("--seed", type=int, default=123)
    m = sub.add_parser("match")
    m.add_argument("--sae-a", required=True)
    m.add_argument("--sae-b", required=True)
    m.add_argument("--out", required=True)
    a = p.parse_args()
    sys.path.insert(0, a.code_dir)
    {"train": cmd_train, "eval": cmd_eval, "untrained": cmd_untrained, "match": cmd_match}[a.cmd](a)


if __name__ == "__main__":
    main()
