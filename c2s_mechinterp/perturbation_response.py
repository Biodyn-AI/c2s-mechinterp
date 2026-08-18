"""
Phase 8 v2 (robust): perturbation-response / regulatory-logic test with proper statistics.

Upgrades over phase8_perturbation.py (which used pooled effect-size and blew up where control
std -> 0): per-CELL feature vectors + per-feature two-sided Mann-Whitney U (perturbed vs control)
with BH-FDR, |Cohen's d| > 0.5 for "responding"; and a Fisher's-exact enrichment for
TF-specificity (are responding features enriched for features whose top-20 genes overlap the TF's
TRRUST targets, vs the alive-feature background) with BH across TFs. This matches the pipeline's
stated Phase-8 methodology, not just the reference's simpler code.

Per-cell feature vector = mean SAE feature activation across that cell's gene rows at the probe layer.

Usage:
  python phase8_perturbation_v2.py --h5ad data/replogle_essential.h5ad --run runs/2b_atlas \
      --layer 16 --model vandijklab/C2S-Scale-Gemma-2-2B --ontology-dir data/ontologies \
      --perturbation-col perturbation --control-label control \
      --n-tf 48 --cells-per-target 30 --n-control-cells 300 --out runs/2b_atlas/phase8v2_L16.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from scipy.stats import fisher_exact, mannwhitneyu
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:  # works as `python -m c2s_mechinterp.<mod>` and as a plain script
    from .cell_sentences import anndata_to_ranked_genes, attribute_tokens_to_genes, build_cell_sentence
except ImportError:
    from cell_sentences import anndata_to_ranked_genes, attribute_tokens_to_genes, build_cell_sentence
try:  # works as `python -m c2s_mechinterp.<mod>` and as a plain script
    from .sae_topk import TopKSAE
except ImportError:
    from sae_topk import TopKSAE


def parse_args():
    p = argparse.ArgumentParser(description="C2S-Scale Phase 8 v2 (robust)")
    p.add_argument("--h5ad", required=True)
    p.add_argument("--run", required=True)
    p.add_argument("--layer", type=int, default=16)
    p.add_argument("--model", required=True)
    p.add_argument("--ontology-dir", required=True)
    p.add_argument("--perturbation-col", default="perturbation")
    p.add_argument("--control-label", default="control")
    p.add_argument("--max-genes", type=int, default=512)
    p.add_argument("--n-tf", type=int, default=48, help="# TRRUST TF targets to test")
    p.add_argument("--n-nontf", type=int, default=20, help="# non-TF targets as contrast")
    p.add_argument("--cells-per-target", type=int, default=30)
    p.add_argument("--n-control-cells", type=int, default=300)
    p.add_argument("--min-cells", type=int, default=15)
    p.add_argument("--effect-thresh", type=float, default=0.5)
    p.add_argument("--fdr", type=float, default=0.05)
    p.add_argument("--out", required=True)
    return p.parse_args()


def load_trrust(ontology_dir):
    tfs, targets = set(), {}
    with open(os.path.join(ontology_dir, "trrust_human.tsv")) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            tf, tgt = parts[0].upper(), parts[1].upper()
            tfs.add(tf)
            targets.setdefault(tf, set()).add(tgt)
    return tfs, targets


def cell_feature_matrix(cell_indices, adata, tok, model, sae, mu, layer, max_genes, device, max_len):
    """(n_used, n_features) — per-cell mean SAE feature activation across the cell's gene rows."""
    import torch
    vecs = []
    for ci in cell_indices:
        genes = anndata_to_ranked_genes(adata, int(ci), max_genes=max_genes)
        if not genes:
            continue
        cs = build_cell_sentence(genes, prompt=True)
        enc = tok(cs.text, return_offsets_mapping=True, return_tensors="pt",
                  truncation=True, max_length=max_len)
        t2g = attribute_tokens_to_genes(enc.pop("offset_mapping")[0].tolist(), cs.gene_spans)
        with torch.no_grad():
            out = model(**{k: v.to(device) for k, v in enc.items()}, output_hidden_states=True)
        hid = out.hidden_states[layer + 1][0].float().cpu().numpy()
        rows = [hid[t2g == g].mean(0) for g in range(len(genes)) if (t2g == g).any()]
        if not rows:
            continue
        X = np.asarray(rows, np.float32) - mu
        with torch.no_grad():
            h, _ = sae.encode(torch.tensor(X, dtype=torch.float32, device=device))
        vecs.append(h.detach().cpu().numpy().mean(0))
    return np.asarray(vecs, np.float64) if vecs else np.zeros((0, sae.n_features))


def main():
    import anndata
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    L = args.layer
    rng = np.random.RandomState(42)
    trrust_tfs, trrust_targets = load_trrust(args.ontology_dir)

    adata = anndata.read_h5ad(args.h5ad)
    pert = adata.obs[args.perturbation_col].astype(str).values
    ctrl_idx = np.where(pert == args.control_label)[0]
    if len(ctrl_idx) == 0:
        raise SystemExit(f"no control cells with {args.perturbation_col}=={args.control_label!r}")
    if len(ctrl_idx) > args.n_control_cells:
        ctrl_idx = rng.choice(ctrl_idx, args.n_control_cells, replace=False)

    uniq, counts = np.unique(pert, return_counts=True)
    present = {g: int(c) for g, c in zip(uniq, counts)
              if g != args.control_label and c >= args.min_cells}
    tf_t = sorted([g for g in present if g.upper() in trrust_tfs], key=lambda g: -present[g])[: args.n_tf]
    nontf = list(rng.permutation(sorted([g for g in present if g.upper() not in trrust_tfs])))[: args.n_nontf]
    targets = tf_t + list(nontf)
    print(f"targets: {len(tf_t)} TFs + {len(nontf)} non-TF | controls: {len(ctrl_idx)} | probe L{L}")

    tok = AutoTokenizer.from_pretrained(args.model)
    max_len = min(getattr(tok, "model_max_length", 8192) or 8192, 8192)
    try:
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="auto")
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map="auto")
    model = model.eval()
    sae = TopKSAE.load(os.path.join(args.run, "sae", f"sae_L{L:02d}.pt"), device=device)
    sae.eval()
    nf = sae.n_features
    mu = np.load(os.path.join(args.run, "sae", f"mu_L{L:02d}.npy")).astype(np.float32).reshape(1, -1)
    with open(os.path.join(args.run, f"feature_catalog_L{L:02d}.json")) as f:
        catalog = json.load(f)["features"]
    feat_top = {feat["feature_idx"]: {g["gene_name"].upper() for g in feat.get("top_genes", [])}
                for feat in catalog}

    print("control baseline (per-cell)...")
    C = cell_feature_matrix(ctrl_idx, adata, tok, model, sae, mu, L, args.max_genes, device, max_len)
    alive = np.where((C > 0).any(0))[0]
    print(f"  control cells {C.shape[0]} | alive features {len(alive)}")

    results = []
    for ti, tgt in enumerate(targets):
        tidx = np.where(pert == tgt)[0]
        if len(tidx) > args.cells_per_target:
            tidx = rng.choice(tidx, args.cells_per_target, replace=False)
        P = cell_feature_matrix(tidx, adata, tok, model, sae, mu, L, args.max_genes, device, max_len)
        if P.shape[0] < 3:
            continue
        pvals = np.ones(nf)
        d = np.zeros(nf)
        for f in alive:
            a, b = P[:, f], C[:, f]
            if a.std() == 0 and b.std() == 0:
                continue
            try:
                pvals[f] = mannwhitneyu(a, b, alternative="two-sided").pvalue
            except ValueError:
                pvals[f] = 1.0
            pooled = np.sqrt((a.var() + b.var()) / 2) + 1e-9
            d[f] = (a.mean() - b.mean()) / pooled
        q = np.ones(nf)
        q[alive] = multipletests(pvals[alive], alpha=args.fdr, method="fdr_bh")[1]
        responding = np.where((q < args.fdr) & (np.abs(d) > args.effect_thresh))[0]
        R = set(int(x) for x in responding)

        known = trrust_targets.get(tgt.upper(), set())
        Tset = set(int(f) for f in alive if feat_top.get(int(f), set()) & known)
        A = set(int(f) for f in alive)
        RT, Ronly, Tonly = len(R & Tset), len(R - Tset), len(Tset - R)
        neither = len(A - R - Tset)
        if (RT + Ronly) > 0 and (RT + Tonly) > 0 and len(Tset) > 0:
            spec_p = fisher_exact([[RT, Ronly], [Tonly, neither]], alternative="greater")[1]
        else:
            spec_p = 1.0
        results.append({"target": tgt, "is_trrust_tf": tgt.upper() in trrust_tfs,
                        "n_known_targets": len(known), "n_cells": int(P.shape[0]),
                        "n_responding": len(R), "n_resp_overlap_targets": RT,
                        "spec_p": float(spec_p)})
        print(f"  [{ti+1}/{len(targets)}] {tgt}{' [TF]' if tgt.upper() in trrust_tfs else ''}: "
              f"cells {P.shape[0]}, responding {len(R)}, overlap {RT}, spec_p {spec_p:.3g}")

    tfs = [r for r in results if r["is_trrust_tf"] and r["n_known_targets"] > 0]
    if tfs:
        qs = multipletests([r["spec_p"] for r in tfs], alpha=0.05, method="fdr_bh")[1]
        for r, qq in zip(tfs, qs):
            r["spec_q"] = float(qq)
    n_spec = sum(1 for r in tfs if r.get("spec_q", 1) < 0.05)
    n_detect = sum(1 for r in results if r["n_responding"] > 0)
    summary = {"layer": L, "n_targets": len(results), "n_tfs_tested": len(tfs),
               "detection_rate": n_detect / max(len(results), 1),
               "n_tfs_specific": n_spec, "tf_specificity_rate": n_spec / max(len(tfs), 1),
               "n_control_cells": int(C.shape[0]), "n_alive": int(len(alive)),
               "mean_responding": float(np.mean([r["n_responding"] for r in results])) if results else 0}
    with open(args.out, "w") as f:
        json.dump({"summary": summary, "targets": results}, f, indent=2)

    print("\n" + "=" * 64)
    print(f"PHASE 8 v2 (layer {L}) — Wilcoxon+BH+Fisher, full Replogle")
    print(f"  detection rate:      {100*summary['detection_rate']:.1f}% ({n_detect}/{len(results)})")
    print(f"  TF-specificity rate: {100*summary['tf_specificity_rate']:.1f}% "
          f"({n_spec}/{len(tfs)} TFs, BH q<0.05) — reference Geneformer 6.2%")
    print(f"  saved: {args.out}")
    print("=" * 64)


if __name__ == "__main__":
    main()
