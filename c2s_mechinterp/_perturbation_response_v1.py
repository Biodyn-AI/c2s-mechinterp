"""
Phase 8 (reframed for C2S-Scale): perturbation-response / regulatory-logic test.

The central scientific test: when a TF is knocked down (CRISPRi), do the SAE features that
respond correspond to that TF's KNOWN regulatory targets — i.e. does the model encode directed
regulation, or only co-expression? Reference: repos/bio-sae/src/09_perturbation_response.py
(Geneformer: 92% detection, 6.2% TF-specificity → co-expression, not regulation).

Reframe for the C2S-Scale decoder LLM (vs the reference's gene-token encoder):
  * cells -> cell sentences -> gene-token residual-stream activations at the probe layer
    (same representation the atlas SAE was trained on), encoded through that layer's SAE.
  * control baseline = non-targeting cells' per-feature mean/std/freq over their gene rows.
  * per target: per-feature effect = (pert_mean - ctrl_mean) / ctrl_std; a feature "responds"
    if |effect| > 0.5; it is "specific" if its top-20 genes (atlas catalog) contain the target
    or overlap the target's TRRUST targets.
  * summary: detection rate (any responding feature) and TF-specificity rate (TFs with >=1
    specific responding feature).

Usage:
  python phase8_perturbation.py --h5ad data/replogle_k562_subset.h5ad --run runs/2b_atlas \
      --layer 16 --model vandijklab/C2S-Scale-Gemma-2-2B --ontology-dir data/ontologies \
      --n-nontf 36 --n-control-cells 500 --out runs/2b_atlas/phase8_L16.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

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
    p = argparse.ArgumentParser(description="C2S-Scale Phase 8: perturbation-response test")
    p.add_argument("--h5ad", required=True)
    p.add_argument("--run", required=True, help="atlas run dir (holds sae/ + feature_catalog_L*.json)")
    p.add_argument("--layer", type=int, default=16, help="probe layer (~2/3 through the stack)")
    p.add_argument("--model", required=True)
    p.add_argument("--ontology-dir", required=True, help="holds trrust_human.tsv")
    p.add_argument("--max-genes", type=int, default=512)
    p.add_argument("--min-cells", type=int, default=8, help="min perturbed cells to test a target")
    p.add_argument("--n-nontf", type=int, default=36, help="# non-TF targets to include as contrast")
    p.add_argument("--n-control-cells", type=int, default=500)
    p.add_argument("--effect-thresh", type=float, default=0.5)
    p.add_argument("--gene-col", default="gene")
    p.add_argument("--control-label", default="non-targeting")
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


def extract_feature_stats(cell_indices, adata, tok, model, sae, mu, layer, max_genes, device, max_len):
    """Accumulate per-feature sum/sqsum/active-count over all gene rows of the given cells."""
    import torch
    nf = sae.n_features
    fsum = np.zeros(nf, np.float64)
    fsq = np.zeros(nf, np.float64)
    fact = np.zeros(nf, np.float64)
    npos = 0
    for ci in cell_indices:
        genes = anndata_to_ranked_genes(adata, int(ci), max_genes=max_genes)
        if not genes:
            continue
        cs = build_cell_sentence(genes, prompt=True)
        enc = tok(cs.text, return_offsets_mapping=True, return_tensors="pt",
                  truncation=True, max_length=max_len)
        tok2gene = attribute_tokens_to_genes(enc.pop("offset_mapping")[0].tolist(), cs.gene_spans)
        with torch.no_grad():
            out = model(**{k: v.to(device) for k, v in enc.items()}, output_hidden_states=True)
        hidden = out.hidden_states[layer + 1][0].float().cpu().numpy()
        rows = [hidden[tok2gene == g].mean(0) for g in range(len(genes)) if (tok2gene == g).any()]
        if not rows:
            continue
        X = np.asarray(rows, dtype=np.float32) - mu
        with torch.no_grad():
            h, _ = sae.encode(torch.tensor(X, dtype=torch.float32, device=device))
        h = h.detach().cpu().numpy()
        fsum += h.sum(0)
        fsq += (h ** 2).sum(0)
        fact += (h > 0).sum(0)
        npos += len(rows)
    return fsum, fsq, fact, npos


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
    gene_col = adata.obs[args.gene_col].astype(str).values
    control_idx = np.where(gene_col == args.control_label)[0]
    if len(control_idx) > args.n_control_cells:
        control_idx = rng.choice(control_idx, args.n_control_cells, replace=False)

    # target selection: all present TRRUST TFs + a sample of non-TF targets
    uniq, counts = np.unique(gene_col, return_counts=True)
    present = {g: int(c) for g, c in zip(uniq, counts)
              if g != args.control_label and c >= args.min_cells}
    tf_targets = sorted([g for g in present if g.upper() in trrust_tfs], key=lambda g: -present[g])
    nontf = sorted([g for g in present if g.upper() not in trrust_tfs], key=lambda g: -present[g])
    nontf = list(rng.permutation(nontf))[: args.n_nontf]
    target_list = tf_targets + nontf
    print(f"targets: {len(tf_targets)} TRRUST TFs + {len(nontf)} non-TF | control cells: {len(control_idx)}")
    print(f"TFs: {tf_targets}")

    # model + SAE + catalog
    tok = AutoTokenizer.from_pretrained(args.model)
    max_len = min(getattr(tok, "model_max_length", 8192) or 8192, 8192)
    try:
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="auto")
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map="auto")
    model = model.eval()
    sae = TopKSAE.load(os.path.join(args.run, "sae", f"sae_L{L:02d}.pt"), device=device)
    sae.eval()
    mu = np.load(os.path.join(args.run, "sae", f"mu_L{L:02d}.npy")).astype(np.float32).reshape(1, -1)

    with open(os.path.join(args.run, f"feature_catalog_L{L:02d}.json")) as f:
        catalog = json.load(f)["features"]
    feat_top_genes = {feat["feature_idx"]: {g["gene_name"].upper() for g in feat.get("top_genes", [])}
                      for feat in catalog}

    # control baseline
    print("control baseline...")
    cs, csq, cact, cnpos = extract_feature_stats(control_idx, adata, tok, model, sae, mu, L,
                                                 args.max_genes, device, max_len)
    cmean = cs / max(cnpos, 1)
    cstd = np.sqrt(np.maximum(csq / max(cnpos, 1) - cmean ** 2, 1e-10))
    cfreq = cact / max(cnpos, 1)
    print(f"  control positions: {cnpos:,} | active features: {int((cfreq>0).sum())}")

    # per target
    results = []
    for ti, tgt in enumerate(target_list):
        tidx = np.where(gene_col == tgt)[0]
        ps, _, pact, pnpos = extract_feature_stats(tidx, adata, tok, model, sae, mu, L,
                                                   args.max_genes, device, max_len)
        if pnpos == 0:
            continue
        pmean = ps / pnpos
        effect = (pmean - cmean) / cstd
        ae = np.abs(effect)
        known = trrust_targets.get(tgt.upper(), set())
        is_tf = tgt.upper() in trrust_tfs
        order = np.argsort(-ae)[:20]
        n_resp = n_spec = 0
        top = []
        for fi in order:
            fi = int(fi)
            if ae[fi] < args.effect_thresh:
                continue
            n_resp += 1
            fg = feat_top_genes.get(fi, set())
            spec = (tgt.upper() in fg) or (len(fg & known) > 0)
            n_spec += int(spec)
            top.append({"feature_idx": fi, "effect": float(effect[fi]), "specific": bool(spec)})
        results.append({"target": tgt, "is_trrust_tf": is_tf, "n_known_targets": len(known),
                        "n_cells": int(len(tidx)), "n_positions": pnpos,
                        "n_responding": n_resp, "n_specific": n_spec,
                        "max_abs_effect": float(ae.max()), "top_changed": top})
        print(f"  [{ti+1}/{len(target_list)}] {tgt}{' [TF]' if is_tf else ''}: "
              f"{len(tidx)} cells, {n_resp} resp, {n_spec} specific, max|eff| {ae.max():.2f}")

    tfs = [r for r in results if r["is_trrust_tf"]]
    n_tf_spec = sum(1 for r in tfs if r["n_specific"] > 0)
    n_detect = sum(1 for r in results if r["n_responding"] > 0)
    summary = {
        "layer": L, "n_targets": len(results), "n_trrust_tfs": len(tfs),
        "detection_rate": n_detect / max(len(results), 1),
        "n_tfs_with_specific_response": n_tf_spec,
        "tf_specificity_rate": n_tf_spec / max(len(tfs), 1),
        "mean_responding": float(np.mean([r["n_responding"] for r in results])) if results else 0,
        "control_positions": cnpos, "n_control_cells": int(len(control_idx)),
    }
    with open(args.out, "w") as f:
        json.dump({"summary": summary, "targets": results}, f, indent=2)

    print("\n" + "=" * 64)
    print(f"PHASE 8 (layer {L}) — preliminary on the Replogle subset")
    print(f"  detection rate:      {100*summary['detection_rate']:.1f}% ({n_detect}/{len(results)} targets respond)")
    print(f"  TF-specificity rate: {100*summary['tf_specificity_rate']:.1f}% "
          f"({n_tf_spec}/{len(tfs)} TFs) — reference Geneformer: 6.2%")
    print(f"  -> {'co-expression, not regulation (null replicates)' if summary['tf_specificity_rate']<0.15 else 'ELEVATED specificity — investigate'}")
    print(f"  saved: {args.out}")
    print("=" * 64)


if __name__ == "__main__":
    main()
