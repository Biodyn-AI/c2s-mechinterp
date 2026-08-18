"""
Phase-4 adapter for C2S-Scale: cross-layer SAE feature tracking by decoder cosine.

Ports repos/bio-sae/src/06_cross_layer.py to the atlas run-dir contract. A feature at
layer L "matches" a feature at layer L' if their (unit-normed) decoder directions have
absolute cosine >= 0.7 — same direction in residual-stream space, presumably the same
concept. For each feature in A we keep its single best match in B (argmax |cos|), exactly
as the reference.

This script needs ONLY the SAE weights — no activations. It loads each layer's
sae/sae_L{L:02d}.pt, takes sae.decoder_directions() -> (n_features, d_model) unit dirs,
and reproduces the reference's structural analyses:
  1. consecutive-layer persistence (adjacent pairs in --layers)
  2. per-feature lifecycle tracked forward from the first layer (first/last layer, span)
  4. feature emergence: features at deeper layers matching nothing at the first layer
  5. long-range persistence (first layer -> every layer)

Dropped from the reference: Analyses 3 & 6 (persistence-vs-annotation, top-gene Jaccard)
and the annotation columns of Analysis 4 — they need feature_catalog.json /
feature_annotations.json, which are not part of the C2S-Scale atlas contract.

Usage:
  python cross_layer.py --run runs/2b_atlas --layers 0 6 13 20 25
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:  # works as `python -m c2s_mechinterp.<mod>` and as a plain script
    from .sae_topk import TopKSAE  # noqa: E402
except ImportError:
    from sae_topk import TopKSAE  # noqa: E402

MATCH_THRESHOLD = 0.7       # |cosine| threshold for "same feature" (reference default)
HIGH_MATCH_THRESHOLD = 0.9  # strong-match threshold (reference default)


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Not JSON serializable: {type(obj)}")


def parse_args():
    p = argparse.ArgumentParser(description="C2S-Scale Phase-4 cross-layer feature tracking")
    p.add_argument("--run", required=True, help="atlas run dir (holds sae/sae_L{L:02d}.pt)")
    p.add_argument("--layers", type=int, nargs="+", required=True,
                   help="transformer block indices to track (tracked in ascending order)")
    p.add_argument("--threshold", type=float, default=MATCH_THRESHOLD,
                   help="|cosine| match threshold (default 0.7)")
    p.add_argument("--high-threshold", type=float, default=HIGH_MATCH_THRESHOLD,
                   help="strong-match |cosine| threshold (default 0.9)")
    return p.parse_args()


def load_layer_dirs(run, layer):
    """Load a layer's SAE and return unit-normed decoder directions (n_features, d_model).

    Returns None if the checkpoint is missing. Loaded on CPU — we only need the weights,
    never the activations.
    """
    path = os.path.join(run, "sae", f"sae_L{layer:02d}.pt")
    if not os.path.exists(path):
        return None
    sae = TopKSAE.load(path, device="cpu")
    return sae.decoder_directions().cpu().numpy().astype(np.float32)  # (n_features, d_model)


def compute_matches(dirs_a, dirs_b, threshold=MATCH_THRESHOLD):
    """Find matching features between two layers.

    dirs_a, dirs_b are already unit-normed, so cos = dirs_a @ dirs_b.T. Sign of a decoder
    direction is arbitrary, so we match on |cos|. For each feature in A we keep its single
    best match in B (argmax), if it clears the threshold.

    Returns: list of (feat_a, feat_b, abs_cosine) tuples.
    """
    if len(dirs_a) == 0 or len(dirs_b) == 0:
        return []
    cos_matrix = np.abs(dirs_a @ dirs_b.T)  # (n_a, n_b)
    best_b_for_a = np.argmax(cos_matrix, axis=1)
    best_cos_for_a = cos_matrix[np.arange(len(dirs_a)), best_b_for_a]
    matches = []
    for i, (bi, cos) in enumerate(zip(best_b_for_a, best_cos_for_a)):
        if cos >= threshold:
            matches.append((int(i), int(bi), float(cos)))
    return matches


def main():
    total_t0 = time.time()
    args = parse_args()
    match_th = args.threshold
    high_th = args.high_threshold

    print("=" * 70)
    print("PHASE 4: CROSS-LAYER FEATURE TRACKING")
    print(f"  Match threshold:      {match_th}")
    print(f"  High-match threshold: {high_th}")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Load decoder directions for every requested layer (ascending order)
    # ------------------------------------------------------------------
    print("\nLoading SAE decoder directions...")
    layers_data = {}
    for layer in sorted(set(args.layers)):
        dirs = load_layer_dirs(args.run, layer)
        if dirs is None:
            print(f"  Layer {layer:02d}: MISSING (no sae/sae_L{layer:02d}.pt)")
            continue
        layers_data[layer] = dirs
        print(f"  Layer {layer:02d}: {dirs.shape[0]} features x {dirs.shape[1]} dims")

    available_layers = sorted(layers_data.keys())
    if len(available_layers) < 2:
        raise SystemExit("Need at least two loadable layers to track features across.")
    print(f"\n  {len(available_layers)} layers loaded")

    ref_layer = available_layers[0]
    ref_dirs = layers_data[ref_layer]
    n_ref = ref_dirs.shape[0]

    # ------------------------------------------------------------------
    # Analysis 1: consecutive-layer persistence
    # ------------------------------------------------------------------
    print(f"\n{'=' * 70}")
    print("ANALYSIS 1: CONSECUTIVE-LAYER FEATURE PERSISTENCE")
    print(f"{'=' * 70}")

    adjacent_results = []
    for i in range(len(available_layers) - 1):
        la = available_layers[i]
        lb = available_layers[i + 1]
        n_a = layers_data[la].shape[0]
        matches = compute_matches(layers_data[la], layers_data[lb], threshold=match_th)
        high_matches = [m for m in matches if m[2] >= high_th]
        persistence_rate = len(matches) / max(n_a, 1)
        high_rate = len(high_matches) / max(n_a, 1)
        mean_cos = float(np.mean([m[2] for m in matches])) if matches else 0.0
        res = {
            "from_layer": la,
            "to_layer": lb,
            "n_matches": len(matches),
            "n_high_matches": len(high_matches),
            "persistence_rate": float(persistence_rate),
            "high_persistence_rate": float(high_rate),
            "mean_match_cosine": mean_cos,
            "n_features_from": n_a,
        }
        adjacent_results.append(res)
        print(f"  L{la:>2d}->L{lb:>2d}: {len(matches):>5d}/{n_a} persist "
              f"({persistence_rate:.1%}), {len(high_matches)} strong (>{high_th}), "
              f"mean cos={mean_cos:.3f}")

    # ------------------------------------------------------------------
    # Analysis 2: feature lifecycle (first/last appearance from ref layer)
    # ------------------------------------------------------------------
    print(f"\n{'=' * 70}")
    print("ANALYSIS 2: FEATURE LIFECYCLE TRACKING")
    print(f"{'=' * 70}")
    print(f"\n  Tracking features forward from L{ref_layer:02d}...")

    feature_chains = {fi: {ref_layer: (fi, 1.0)} for fi in range(n_ref)}
    for layer in available_layers[1:]:
        matches = compute_matches(ref_dirs, layers_data[layer], threshold=match_th)
        for feat_a, feat_b, cos in matches:
            feature_chains[feat_a][layer] = (feat_b, cos)

    lifespans = []
    for fi, chain in feature_chains.items():
        last_layer = max(chain.keys())
        lifespans.append({
            "feature_idx": fi,
            "first_layer": ref_layer,
            "last_layer": last_layer,
            "span": last_layer - ref_layer + 1,
            "n_layers_present": len(chain),
            "layers": sorted(chain.keys()),
        })

    transient = [f for f in lifespans if f["n_layers_present"] <= 3]
    moderate = [f for f in lifespans if 4 <= f["n_layers_present"] <= 10]
    persistent = [f for f in lifespans if f["n_layers_present"] > 10]

    print(f"  L{ref_layer:02d} features: {n_ref}")
    print(f"  Transient (1-3 layers):   {len(transient)} ({len(transient)/n_ref*100:.1f}%)")
    print(f"  Moderate (4-10 layers):   {len(moderate)} ({len(moderate)/n_ref*100:.1f}%)")
    print(f"  Persistent (11+ layers):  {len(persistent)} ({len(persistent)/n_ref*100:.1f}%)")

    span_hist = defaultdict(int)
    for f in lifespans:
        span_hist[f["n_layers_present"]] += 1
    print("\n  Span distribution (# layers present):")
    for n_layers in sorted(span_hist.keys()):
        bar = "#" * (span_hist[n_layers] // 20 + 1)
        print(f"    {n_layers:>2d} layers: {span_hist[n_layers]:>5d} {bar}")

    # ------------------------------------------------------------------
    # Analysis 4: feature emergence (novel features vs the first layer)
    # ------------------------------------------------------------------
    print(f"\n{'=' * 70}")
    print("ANALYSIS 4: FEATURE EMERGENCE BY LAYER")
    print(f"{'=' * 70}")
    print(f"\n  Features NOT matching any L{ref_layer:02d} feature (novel at deeper layers):")

    emergence = []
    for layer in available_layers[1:]:
        ld = layers_data[layer]
        matches = compute_matches(ld, ref_dirs, threshold=match_th)  # reverse: L -> ref
        matched_l_feats = set(m[0] for m in matches)
        n_l = ld.shape[0]
        n_unmatched = n_l - len(matched_l_feats)
        unmatch_rate = n_unmatched / max(n_l, 1)
        emergence.append({
            "layer": layer,
            "n_novel": int(n_unmatched),
            "n_features": int(n_l),
            "novel_rate": float(unmatch_rate),
        })
        print(f"    L{layer:>2d}: {n_unmatched:>5d}/{n_l} novel ({unmatch_rate:.1%})")

    # ------------------------------------------------------------------
    # Analysis 5: long-range tracking (first layer -> every layer)
    # ------------------------------------------------------------------
    print(f"\n{'=' * 70}")
    print(f"ANALYSIS 5: LONG-RANGE FEATURE TRACKING (L{ref_layer:02d} -> every layer)")
    print(f"{'=' * 70}")

    long_range = []
    for layer in available_layers:
        if layer == ref_layer:
            long_range.append({
                "target_layer": ref_layer,
                "n_matches": n_ref,
                "persistence_rate": 1.0,
                "mean_cosine": 1.0,
            })
            print(f"  L{ref_layer:>2d}->L{ref_layer:>2d}: {n_ref:>5d}/{n_ref} persist "
                  f"(100.0%), mean cos=1.000")
            continue
        matches = compute_matches(ref_dirs, layers_data[layer], threshold=match_th)
        rate = len(matches) / max(n_ref, 1)
        mean_cos = float(np.mean([m[2] for m in matches])) if matches else 0.0
        long_range.append({
            "target_layer": layer,
            "n_matches": len(matches),
            "persistence_rate": float(rate),
            "mean_cosine": mean_cos,
        })
        print(f"  L{ref_layer:>2d}->L{layer:>2d}: {len(matches):>5d}/{n_ref} persist "
              f"({rate:.1%}), mean cos={mean_cos:.3f}")

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    out_path = os.path.join(args.run, "cross_layer_tracking.json")
    output = {
        "config": {
            "match_threshold": match_th,
            "high_match_threshold": high_th,
            "layers": available_layers,
            "n_layers": len(available_layers),
            "reference_layer": ref_layer,
        },
        "adjacent_persistence": adjacent_results,
        "long_range_persistence": long_range,
        "feature_emergence": emergence,
        "lifecycle_summary": {
            "n_ref_features": n_ref,
            "n_transient": len(transient),
            "n_moderate": len(moderate),
            "n_persistent": len(persistent),
            "span_histogram": {str(k): v for k, v in sorted(span_hist.items())},
        },
        "feature_lifecycles": lifespans,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=_json_default)

    print(f"\n  Saved: {out_path}")
    print(f"\n{'=' * 70}")
    print(f"DONE. {len(available_layers)} layers, ref L{ref_layer:02d}, "
          f"{len(persistent)} persistent features. Total time: {time.time()-total_t0:.1f}s")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
