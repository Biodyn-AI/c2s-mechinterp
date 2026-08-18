"""
Phase-2/3 adapter for C2S-Scale: build the per-feature catalog for one atlas layer.

Adapted from repos/bio-sae/src/03_analyze_features.py (pinned commit 9277f5d). Same logic —
streaming per-feature activation stats, top max-activating genes, decoder-cosine, and Phase-3
SVD/superposition comparison — only the I/O is remapped to the C2S-Scale run-dir contract:
  * activations come from layer_{L:02d}_activations.npy (n_rows, d_model), streamed in chunks
  * genes come from row_gene_names.json (SYMBOLS, one per row; "<CELL>" rows skipped when
    grouping by gene). This replaces the reference's gene_ids + token_id_to_gene_name.json.
  * SAE + pre-encoder mean come from sae/sae_L{L:02d}.pt and sae/mu_L{L:02d}.npy.

Phase 3 (superposition): top-50 SVD of a 50k sample of (acts - mu); each feature's max |cos|
with the SVD axes; is_svd_aligned = max|cos| > 0.7 (the pipeline's alignment threshold).

Writes feature_catalog_L{L:02d}.json into the run dir. The top_genes schema matches what
annotate_features.py reads downstream (it keys on g['gene_name']).

Usage:
  python feature_catalog.py --run runs/2b_atlas --layer 13
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:  # works as `python -m c2s_mechinterp.<mod>` and as a plain script
    from .sae_topk import TopKSAE
except ImportError:
    from sae_topk import TopKSAE

# ---- faithful reference parameters ----
CHUNK_SIZE = 10000          # stream activations in chunks; never load all into GPU at once
TOP_GENES_PER_FEATURE = 20  # top max-activating genes kept per feature (by mean activation)
N_SVD_AXES = 50             # Phase-3: compare feature directions to top-50 SVD axes
SVD_SAMPLE = 50000          # rows sampled for the SVD fit
SVD_ALIGN_THRESH = 0.7      # |cos| above this == aligned with an SVD axis (not novel)
CELL_TOKEN = "<CELL>"       # cell-summary rows: skipped when grouping by gene


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Not JSON serializable: {type(obj)}")


def parse_args():
    p = argparse.ArgumentParser(description="C2S-Scale Phase-2/3 SAE feature catalog")
    p.add_argument("--run", required=True, help="atlas run dir (holds layer_*_activations.npy + sae/)")
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--seed", type=int, default=42, help="seed for the SVD + decoder-cosine samples")
    return p.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 70)
    print("C2S-SCALE: FEATURE CATALOG")
    print(f"  Run: {args.run} | Layer: {args.layer:02d} | device: {device}")
    print("=" * 70)
    total_t0 = time.time()

    # ============================================================
    # STEP 1: Load SAE, activations, gene names, pre-encoder mean
    # ============================================================
    print("\nSTEP 1: Loading SAE and data...")

    sae = TopKSAE.load(os.path.join(args.run, "sae", f"sae_L{args.layer:02d}.pt"), device=device)
    sae.eval()
    n_features = sae.n_features
    print(f"  SAE: {sae.d_model}d -> {n_features} features, k={sae.k}")

    act_path = os.path.join(args.run, f"layer_{args.layer:02d}_activations.npy")
    activations = np.load(act_path, mmap_mode="r")
    n_samples, d_model = activations.shape
    print(f"  Activations: {activations.shape}")

    mu = np.load(os.path.join(args.run, "sae", f"mu_L{args.layer:02d}.npy"))  # (1, d_model)
    mu = mu.astype(np.float32).reshape(1, d_model)

    with open(os.path.join(args.run, "row_gene_names.json")) as f:
        gene_names = json.load(f)  # list[str], length n_rows, "<CELL>" for cell-summary rows
    if len(gene_names) != n_samples:
        raise ValueError(f"row_gene_names.json ({len(gene_names)}) != activations rows ({n_samples})")
    print(f"  Gene names: {len(gene_names)} rows")

    # ============================================================
    # STEP 2: Streaming per-feature activation stats + per-gene accumulation
    # ============================================================
    print("\nSTEP 2: Computing feature activations (streaming)...")
    t0 = time.time()

    feature_sum = np.zeros(n_features, dtype=np.float64)
    feature_count = np.zeros(n_features, dtype=np.int64)  # times feature fired
    feature_n = 0

    # per-feature -> per-gene running sum / count (sparse, gene SYMBOL keyed)
    gene_feature_sum = defaultdict(lambda: defaultdict(float))
    gene_feature_count = defaultdict(lambda: defaultdict(int))

    for start in range(0, n_samples, CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, n_samples)
        batch_act = np.asarray(activations[start:end], dtype=np.float32) - mu
        batch_tensor = torch.tensor(batch_act, dtype=torch.float32, device=device)
        with torch.no_grad():
            h_sparse, _ = sae.encode(batch_tensor)
        h_np = h_sparse.detach().cpu().numpy()

        # global feature stats (over ALL rows, incl. "<CELL>")
        active_mask = h_np > 0
        feature_sum += h_np.sum(axis=0)
        feature_count += active_mask.sum(axis=0)
        feature_n += len(batch_act)

        # per-gene-per-feature stats (skip "<CELL>" rows; only active features stay sparse)
        for i in range(len(batch_act)):
            gname = gene_names[start + i]
            if gname == CELL_TOKEN:
                continue
            for fi in np.where(h_np[i] > 0)[0]:
                fi = int(fi)
                gene_feature_sum[fi][gname] += float(h_np[i, fi])
                gene_feature_count[fi][gname] += 1

        if end % 100000 < CHUNK_SIZE:
            print(f"    Processed {end:,}/{n_samples:,}")

    print(f"  Feature activation computation: {time.time()-t0:.1f}s")

    feature_mean_act = feature_sum / max(feature_n, 1)
    feature_freq = feature_count / max(feature_n, 1)
    dead_mask = feature_count == 0
    n_alive = int((~dead_mask).sum())
    n_dead = int(dead_mask.sum())
    alive_indices = np.where(~dead_mask)[0]
    print(f"  Alive: {n_alive}/{n_features} | Dead: {n_dead}/{n_features}")
    if n_alive:
        print(f"  Mean activation freq (alive): {feature_freq[~dead_mask].mean():.4f}")

    # ============================================================
    # STEP 3: Build feature catalog (top genes by mean activation)
    # ============================================================
    print("\nSTEP 3: Building feature catalog...")

    catalog = []
    for fi in range(n_features):
        entry = {
            "feature_idx": fi,
            "is_dead": bool(dead_mask[fi]),
            "activation_freq": float(feature_freq[fi]),
            "mean_activation": float(feature_mean_act[fi]),
            "fire_count": int(feature_count[fi]),
            "top_genes": [],
            "n_unique_genes": 0,
            "max_svd_alignment": 0.0,
            "is_svd_aligned": False,
        }
        if not dead_mask[fi]:
            gene_acts = gene_feature_sum.get(fi, {})
            gene_counts = gene_feature_count.get(fi, {})
            gene_mean_acts = {g: gene_acts[g] / gene_counts[g]
                              for g in gene_acts if gene_counts[g] > 0}
            sorted_genes = sorted(gene_mean_acts.items(), key=lambda x: -x[1])
            entry["top_genes"] = [
                {"gene_name": g, "mean_activation": float(act), "fire_count": int(gene_counts[g])}
                for g, act in sorted_genes[:TOP_GENES_PER_FEATURE]
            ]
            entry["n_unique_genes"] = len(gene_acts)
        catalog.append(entry)

    # ============================================================
    # STEP 4: Decoder-direction analysis (mean pairwise |cosine|)
    # ============================================================
    print("\nSTEP 4: Analyzing decoder directions...")

    dirs = sae.decoder_directions().detach().cpu().numpy()  # (n_features, d_model), unit-normed
    mean_cos = 0.0
    alive_dirs = None
    if n_alive > 0:
        alive_dirs = dirs[alive_indices]
        if n_alive > 100:
            sample_size = min(1000, n_alive * (n_alive - 1) // 2)
            idx_a = np.random.randint(0, n_alive, sample_size)
            idx_b = np.random.randint(0, n_alive, sample_size)
            mask = idx_a != idx_b
            cosines = np.sum(alive_dirs[idx_a[mask]] * alive_dirs[idx_b[mask]], axis=1)
            mean_cos = float(np.mean(np.abs(cosines)))
        else:
            cos_matrix = alive_dirs @ alive_dirs.T
            np.fill_diagonal(cos_matrix, 0)
            mean_cos = float(np.mean(np.abs(cos_matrix)))
        print(f"  Mean |cosine| between features: {mean_cos:.4f}")

    # ============================================================
    # STEP 5: Phase-3 SVD / superposition comparison
    # ============================================================
    print("\nSTEP 5: Comparing to SVD axes...")

    svd_sample_size = min(SVD_SAMPLE, n_samples)
    svd_idx = np.sort(np.random.choice(n_samples, svd_sample_size, replace=False))
    svd_data = np.asarray(activations[svd_idx], dtype=np.float32) - mu
    print(f"  Computing SVD on {svd_sample_size:,} sampled rows...")
    _, S, Vt = np.linalg.svd(svd_data, full_matrices=False)
    svd_axes = Vt[:N_SVD_AXES]  # (N_SVD_AXES, d_model)
    svd_variance = S ** 2 / (S ** 2).sum()
    print(f"  Top-{N_SVD_AXES} SVD axes explain {svd_variance[:N_SVD_AXES].sum()*100:.1f}% variance")

    n_svd_aligned = 0
    n_novel = 0
    if n_alive > 0:
        alignments = alive_dirs @ svd_axes.T          # (n_alive, N_SVD_AXES)
        max_alignment = np.max(np.abs(alignments), axis=1)
        svd_aligned = max_alignment > SVD_ALIGN_THRESH
        n_svd_aligned = int(svd_aligned.sum())
        n_novel = int(n_alive - n_svd_aligned)
        for i, fi in enumerate(alive_indices):
            catalog[int(fi)]["max_svd_alignment"] = float(max_alignment[i])
            catalog[int(fi)]["is_svd_aligned"] = bool(svd_aligned[i])
        print(f"  SVD-aligned (|cos|>{SVD_ALIGN_THRESH}): {n_svd_aligned}/{n_alive} "
              f"({100*n_svd_aligned/n_alive:.1f}%)")
        print(f"  Novel (non-SVD-aligned): {n_novel}/{n_alive} "
              f"({100*n_novel/n_alive:.1f}%)")

    svd_info = {
        "n_svd_axes": N_SVD_AXES,
        "svd_sample_size": svd_sample_size,
        "align_threshold": SVD_ALIGN_THRESH,
        "variance_explained_top50": float(svd_variance[:N_SVD_AXES].sum()),
        "singular_values": S[:N_SVD_AXES].tolist(),
        "variance_ratios": svd_variance[:N_SVD_AXES].tolist(),
    }
    np.save(os.path.join(args.run, f"svd_axes_L{args.layer:02d}.npy"), svd_axes)

    # ============================================================
    # STEP 6: Save catalog
    # ============================================================
    print("\nSTEP 6: Saving feature catalog...")

    summary = {
        "run": args.run,
        "layer": args.layer,
        "n_features": n_features,
        "k": sae.k,
        "n_alive": n_alive,
        "n_dead": n_dead,
        "mean_feature_cosine": mean_cos,
        "n_svd_aligned": n_svd_aligned,
        "n_novel": n_novel,
        "svd_info": svd_info,
    }
    output = {"summary": summary, "features": catalog}

    out_path = os.path.join(args.run, f"feature_catalog_L{args.layer:02d}.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=_json_default)

    print(f"\n{'=' * 70}")
    print(f"DONE. Total time: {(time.time()-total_t0)/60:.1f} min")
    print(f"  Alive/dead: {n_alive}/{n_dead} of {n_features}")
    print(f"  Mean |decoder cos|: {mean_cos:.4f}")
    print(f"  SVD-aligned: {n_svd_aligned} | Novel: {n_novel}")
    print(f"  Catalog: {out_path}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
