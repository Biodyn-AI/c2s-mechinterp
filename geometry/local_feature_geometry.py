"""
local_feature_geometry.py — SAE-decoder-direction geometry for the C2S-Scale-2B atlas.

RUNS ON LOCAL DATA ONLY. No GPU, no model, no raw activations. Consumes only the 26
trained TopK SAE checkpoints at runs/2b_atlas/sae/sae_L{00..25}.pt. Each SAE exposes
`decoder_directions()` -> (n_features=9216, d_model=2304) unit-norm feature directions
(decoder columns, decoder_norm_dim=0). These directions ARE the learned feature geometry:
the residual-stream axes onto which the model writes gene-program features.

Computes, per layer:
  (A) FEATURE INTRINSIC DIMENSION of the 9216-direction cloud in R^2304:
        - participation ratio PR = (Σλ)^2 / Σλ^2 of the second-moment eigenvalues,
          where λ are eigenvalues of C = Dᵀ D (2304x2304). PR is the "effective number
          of ambient axes" the feature directions spread across. (Identity used:
          Σλ = trace(C) = Σ‖d_i‖^2 = n_features since rows are unit-norm; Σλ^2 =
          ‖C‖_F^2 = Σ_ij cos_ij^2, so PR = n_features^2 / Σ_ij cos_ij^2 — ties directly
          to the cosine geometry in (B).)
        - PCA effective dim: #eigenvalues to reach 90% / 95% of Σλ.
        - normalized ID = PR / d_model  (1.0 == isotropic, uses the whole space).
  (B) QUASI-ORTHOGONALITY / SUPERPOSITION GEOMETRY:
        - mean, median, p99, max of |cos| over all C(9216,2) off-diagonal feature pairs.
        - fraction of pairs with |cos| > {0.2,0.3,0.5}.
        - RANDOM-DIRECTION NULL: 9216 iid Gaussian unit vectors in R^2304 give the
          isotropic baseline E[|cos|] ≈ sqrt(2/(π d)). Report observed/null ratio.
          If observed ≈ null, the dictionary is maximally quasi-orthogonal (clean
          superposition); excess mean|cos| is genuine geometric clustering of features.
  (C) CROSS-LAYER GEOMETRY (all layers share the one residual-stream basis R^2304):
        - feature-level matched cosine: for each L0 feature, best |cos| to any feature
          at layer L (argmax). Report mean/median and persistence@0.7 / @0.9. (Recovers
          the cross_layer_tracking.json headline from the SAE weights directly.)
        - SUBSPACE alignment via principal angles between the top-R principal axes of the
          feature cloud (top-R eigenvectors of C) at layer A vs layer B. mean cos(θ_i)
          = how much the occupied subspace rotates with depth. Computed consecutive and
          L0->all.

Confound notes (honest):
  * The random-direction null is the correct baseline for quasi-orthogonality; a
    "feature-shuffle" null is meaningless for pure-direction statistics (shuffling
    feature identity does not change the set of directions).
  * These directions were learned on K562 activations; co-expression structure in K562
    could shape the geometry. That confound bites the FUNCTIONAL clustering
    (local_functional_geometry.py), not the pure superposition/ID counts here.
  * Sign of a decoder direction is arbitrary -> all cross-feature/cross-layer comparisons
    use |cos|.

Usage:
    python local_feature_geometry.py \
        --run ../runs/2b_atlas \
        --out results/feature_geometry.json \
        [--subspace-r 50] [--seed 0]

Outputs: results/feature_geometry.json (all numbers) and prints a compact per-layer table.
"""


# --- make sibling modules importable when run from anywhere ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "c2s_mechinterp"))

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "code"))
from sae_topk import TopKSAE  # noqa: E402


def load_dirs(run: str, layer: int) -> np.ndarray | None:
    """Unit-norm decoder directions (n_features, d_model) for a layer, or None."""
    path = os.path.join(run, "sae", f"sae_L{layer:02d}.pt")
    if not os.path.exists(path):
        return None
    sae = TopKSAE.load(path, device="cpu")
    return sae.decoder_directions().cpu().numpy().astype(np.float64)


def second_moment_eigs(D: np.ndarray) -> np.ndarray:
    """Eigenvalues (descending) of C = Dᵀ D, the (d_model, d_model) second-moment matrix."""
    C = D.T @ D  # (d_model, d_model)
    eig = np.linalg.eigvalsh(C)  # ascending
    return eig[::-1].copy()


def participation_ratio(eig: np.ndarray) -> float:
    s1 = float(eig.sum())
    s2 = float((eig ** 2).sum())
    return (s1 * s1) / s2 if s2 > 0 else 0.0


def pca_eff_dim(eig: np.ndarray, frac: float) -> int:
    tot = eig.sum()
    if tot <= 0:
        return 0
    c = np.cumsum(eig) / tot
    return int(np.searchsorted(c, frac) + 1)


def cosine_stats(D: np.ndarray, block: int = 1024) -> dict:
    """Off-diagonal |cos| distribution over all feature pairs, computed in row-blocks
    to bound memory. D rows are unit-norm, so cos = D @ D.T."""
    n = D.shape[0]
    thresholds = [0.2, 0.3, 0.5]
    cnt = {t: 0 for t in thresholds}
    total_pairs = 0
    s_abs = 0.0
    s_sq = 0.0          # Σ_i<j cos^2  (for PR cross-check)
    mx = 0.0
    # reservoir of |cos| for percentiles (sample to keep memory bounded)
    rng = np.random.default_rng(12345)
    sample = []
    sample_cap = 2_000_000
    for i0 in range(0, n, block):
        i1 = min(i0 + block, n)
        C = D[i0:i1] @ D.T  # (b, n)
        A = np.abs(C)
        # zero the diagonal entries that fall in this block
        for r in range(i0, i1):
            A[r - i0, r] = 0.0
        # keep only j > i to count each pair once
        for r in range(i0, i1):
            row = A[r - i0, r + 1:]
            if row.size == 0:
                continue
            total_pairs += row.size
            s_abs += float(row.sum())
            s_sq += float((row * row).sum())
            m = float(row.max())
            if m > mx:
                mx = m
            for t in thresholds:
                cnt[t] += int((row > t).sum())
            if len(sample) * 0 == 0 and rng.random() < 0.5 and len(sample) < sample_cap:
                # subsample this row for percentile estimation
                k = min(row.size, 256)
                idx = rng.integers(0, row.size, size=k)
                sample.append(row[idx])
    samp = np.concatenate(sample) if sample else np.array([0.0])
    return {
        "n_pairs": int(total_pairs),
        "mean_abs_cos": s_abs / max(total_pairs, 1),
        "median_abs_cos": float(np.median(samp)),
        "p99_abs_cos": float(np.percentile(samp, 99)),
        "max_abs_cos": mx,
        "frac_gt_0.2": cnt[0.2] / max(total_pairs, 1),
        "frac_gt_0.3": cnt[0.3] / max(total_pairs, 1),
        "frac_gt_0.5": cnt[0.5] / max(total_pairs, 1),
        "_sum_offdiag_cos_sq": s_sq,  # internal, for PR cross-check
    }


def random_null_mean_abs_cos(n: int, d: int, seed: int = 0, trials: int = 1) -> dict:
    """Empirical mean |cos| for n iid Gaussian unit vectors in R^d, plus the closed form."""
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(trials):
        G = rng.standard_normal((n, d))
        G /= np.linalg.norm(G, axis=1, keepdims=True)
        # sample pairs to estimate mean|cos| cheaply
        idx_a = rng.integers(0, n, size=200000)
        idx_b = rng.integers(0, n, size=200000)
        keep = idx_a != idx_b
        c = np.abs(np.sum(G[idx_a[keep]] * G[idx_b[keep]], axis=1))
        means.append(float(c.mean()))
    return {
        "empirical_mean_abs_cos": float(np.mean(means)),
        "closed_form_mean_abs_cos": float(np.sqrt(2.0 / (np.pi * d))),
    }


def principal_angle_alignment(evecs_a: np.ndarray, evecs_b: np.ndarray) -> dict:
    """Mean cos of principal angles between two R-dim subspaces (columns orthonormal)."""
    M = evecs_a.T @ evecs_b  # (R, R)
    sv = np.linalg.svd(M, compute_uv=False)
    sv = np.clip(sv, 0.0, 1.0)
    return {"mean_principal_cos": float(sv.mean()), "min_principal_cos": float(sv.min())}


def top_eigvecs(D: np.ndarray, r: int) -> np.ndarray:
    """Top-r right singular vectors (principal axes of the direction cloud), (d_model, r)."""
    # eigh on C = Dᵀ D
    C = D.T @ D
    w, V = np.linalg.eigh(C)      # ascending
    return V[:, ::-1][:, :r].copy()


def main():
    t0 = time.time()
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="../runs/2b_atlas")
    ap.add_argument("--out", default="results/feature_geometry.json")
    ap.add_argument("--layers", type=int, nargs="+", default=list(range(26)))
    ap.add_argument("--subspace-r", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    run = args.run
    layers = sorted(set(args.layers))
    print("=" * 72)
    print("C2S-Scale-2B FEATURE GEOMETRY (local, SAE decoder directions only)")
    print("=" * 72)

    per_layer = {}
    evec_cache = {}   # only the tiny (d_model, r) principal axes are cached
    d_model = None

    for L in layers:
        D = load_dirs(run, L)
        if D is None:
            print(f"  L{L:02d}: MISSING")
            continue
        n, d = D.shape
        d_model = d
        eig = second_moment_eigs(D)
        pr = participation_ratio(eig)
        cs = cosine_stats(D)
        # PR cross-check via cosines: PR = n^2 / (n + 2*Σ_i<j cos^2)
        pr_from_cos = (n * n) / (n + 2.0 * cs.pop("_sum_offdiag_cos_sq"))
        rec = {
            "layer": L,
            "n_features": int(n),
            "d_model": int(d),
            "participation_ratio": pr,
            "pr_normalized": pr / d,
            "pr_crosscheck_from_cosines": pr_from_cos,
            "pca_dim_90pct": pca_eff_dim(eig, 0.90),
            "pca_dim_95pct": pca_eff_dim(eig, 0.95),
            "top1_eig_frac": float(eig[0] / eig.sum()),
            **cs,
        }
        per_layer[L] = rec
        evec_cache[L] = top_eigvecs(D, args.subspace_r)
        del D
        print(f"  L{L:02d}: PR={pr:7.1f} ({pr/d:5.3f}·d) pca90={rec['pca_dim_90pct']:4d} "
              f"pca95={rec['pca_dim_95pct']:4d} mean|cos|={rec['mean_abs_cos']:.4f} "
              f"max|cos|={rec['max_abs_cos']:.3f} f>0.3={rec['frac_gt_0.3']:.2e}")

    # random-direction null (depends only on n,d)
    n0 = per_layer[layers[0]]["n_features"] if per_layer else 9216
    null = random_null_mean_abs_cos(n0, d_model or 2304, seed=args.seed)
    print(f"\n  RANDOM NULL mean|cos|: empirical={null['empirical_mean_abs_cos']:.4f} "
          f"closed-form={null['closed_form_mean_abs_cos']:.4f}")

    # -------- cross-layer geometry --------
    avail = sorted(per_layer.keys())
    ref = avail[0]
    Dref = load_dirs(run, ref)
    matched = {}   # feature-level matched cosine ref->L
    print(f"\n  CROSS-LAYER (ref L{ref:02d}) feature matched-cosine + subspace alignment:")
    for L in avail:
        DL = load_dirs(run, L)
        # matched cosine: max over columns of |Dref @ DL.T|, computed in blocks
        best = np.zeros(Dref.shape[0])
        for i0 in range(0, Dref.shape[0], 1024):
            i1 = min(i0 + 1024, Dref.shape[0])
            C = np.abs(Dref[i0:i1] @ DL.T)
            best[i0:i1] = C.max(axis=1)
        sub = principal_angle_alignment(evec_cache[ref], evec_cache[L])
        matched[L] = {
            "target_layer": L,
            "mean_matched_cos": float(best.mean()),
            "median_matched_cos": float(np.median(best)),
            "persist_0.7": float((best >= 0.7).mean()),
            "persist_0.9": float((best >= 0.9).mean()),
            "subspace_mean_principal_cos": sub["mean_principal_cos"],
            "subspace_min_principal_cos": sub["min_principal_cos"],
        }
        print(f"    L{ref:02d}->L{L:02d}: matched mean|cos|={matched[L]['mean_matched_cos']:.3f} "
              f"persist@0.7={matched[L]['persist_0.7']:.1%} "
              f"subspace_cos(R={args.subspace_r})={sub['mean_principal_cos']:.3f}")
        del DL

    # consecutive subspace alignment
    consec = []
    for a, b in zip(avail[:-1], avail[1:]):
        sub = principal_angle_alignment(evec_cache[a], evec_cache[b])
        consec.append({"from": a, "to": b, **sub})

    out = {
        "meta": {
            "run": os.path.abspath(run),
            "n_layers": len(avail),
            "subspace_r": args.subspace_r,
            "note": "decoder_norm_dim=0; unit-norm feature directions; |cos| throughout",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "random_null": null,
        "per_layer": [per_layer[L] for L in avail],
        "cross_layer_ref": ref,
        "cross_layer_matched": [matched[L] for L in avail],
        "consecutive_subspace_alignment": consec,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Saved: {args.out}  ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
