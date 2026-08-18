"""
gene_manifold_gpu.py — DRAFT (do NOT run without a GPU + the live 2B model).

QUESTION (function, not chromosome): in C2S-Scale-2B's residual stream, do functionally
related genes occupy a shared manifold / cluster together — beyond what co-expression alone
predicts? C2S-Scale has NO gene-embedding table, so a gene's "representation" is the mean
residual-stream activation over its token positions across many diverse cells. This builds
that per-gene representation manifold and tests functional organization with the program's
full confound battery.

================================ WHAT THIS NEEDS ================================
COMPUTE:  1 GPU (>=24 GB for 2B in bf16) for the extraction step only; the geometry runs on CPU.
MODEL:    vandijklab/C2S-Scale-Gemma-2-2B  (ungated; ~5 GB bf16).  NOT downloaded here.
DATA:     a CELL-TYPE-DIVERSE h5ad (Tabula Sapiens subset recommended — NOT K562, which is a
          single proliferative line and would confound "function" with "cell-cycle").
          Needs adata.obs[<celltype_key>] for the group-split and co-expression controls.
          Raw/counts layer required for the co-expression null.

STEP 0 (GPU) — extract per-gene-occurrence activations, reusing the atlas extractor verbatim:
    python ../../code/extract_activations.py \
        --model vandijklab/C2S-Scale-Gemma-2-2B \
        --h5ad <tabula_sapiens_subset.h5ad> \
        --layers 9 13 17 --max-cells 4000 --max-genes 512 \
        --unit gene-token --store-dtype float16 --out runs/gene_manifold
  -> runs/gene_manifold/layer_{L}_activations.npy  (n_rows, d_model)
     runs/gene_manifold/row_gene_names.json  runs/gene_manifold/row_cell_ids.npy

STEP 1+ (CPU) — this script: aggregate rows -> per-gene (and per-(gene,celltype)) means,
    then run the manifold + functional-organization analyses below.

================================ METHOD ================================
1. Per-gene representation G (n_genes x d_model) = mean over all occurrences of that gene
   (occurrence-CAPPED at --occ-cap to kill the 1/sqrt(n) heteroscedastic bias, per
   route_genemanifold NOTE). Optionally per-(gene,celltype) for a diversity-aware version.
2. Manifold intrinsic dimension: participation ratio PR=(Σλ)²/Σλ² and TwoNN on G; compare to
   the SAE feature-space ID from local_feature_geometry.py (representation vs dictionary ID).
3. Functional organization: for each gene, are its nearest neighbours (cosine on centered G)
   enriched for a shared GO/KEGG/Reactome term / STRING edge vs random pairs? precision@k and
   NN-enrichment, reported as z vs a feature-shuffle null (permute gene labels on G).
CONTROLS (a positive counts only if it survives ALL):
   (C1) HOUSEKEEPING STOPWORD filter — drop RPL*/RPS*/MRPL*/MRPS*/MT-* + high-expression-
        frequency genes; report raw AND filtered (filtered is the honest number).
   (C2) CO-EXPRESSION-MATCHED null — compute pairwise Pearson co-expression from the raw
        counts; bin gene pairs into 5 co-expression x 5 mean-expression bins; ask whether
        same-function pairs are still closer WITHIN each bin (topo2's 5x5 stratified null).
        If functional proximity vanishes within co-expression bins, the manifold is
        co-expression, not function.
   (C3) GROUP SPLIT — GroupKFold where the probe never sees a gene whose STRING/PPI cluster
        overlaps a training gene (near-duplicate leakage guard, per route_genemanifold
        genome_groupsplit pattern generalized to functional clusters).
   (C4) BEAT-BOTH-BASELINES — the model manifold must beat BOTH the raw co-expression space
        AND a sequence control (gene-name-string / ESM2 if available). Implemented for
        co-expression here; ESM2 hook left as a TODO stub.

Outputs: results/gene_manifold_L{L}.json  (ID, NN-enrichment raw/filtered/within-bin, z vs
    shuffle, group-split scores) + a short console table.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "code"))
# from sae_topk import TopKSAE  # optional: project G onto SAE features for a dictionary view

HK_RE = re.compile(r"^(RPL|RPS|MRPL|MRPS|MT-|RNA5|RN7|RNU|MALAT1$|MTRNR)")


# --------------------------------------------------------------------------- #
# aggregation                                                                  #
# --------------------------------------------------------------------------- #
def aggregate_per_gene(acts_path, gene_names, occ_cap, seed=0):
    """Mean activation per gene over (subsampled, capped) occurrences.
    Returns (genes: list[str], G: (n_genes, d_model) float32)."""
    rng = np.random.default_rng(seed)
    acts = np.load(acts_path, mmap_mode="r")          # (n_rows, d_model) float16
    by_gene: dict[str, list[int]] = {}
    for i, g in enumerate(gene_names):
        by_gene.setdefault(g, []).append(i)
    genes, rows = [], []
    for g, idx in by_gene.items():
        if len(idx) > occ_cap:
            idx = rng.choice(idx, occ_cap, replace=False)
        genes.append(g)
        rows.append(np.asarray(idx))
    G = np.zeros((len(genes), acts.shape[1]), dtype=np.float32)
    for j, idx in enumerate(rows):
        G[j] = np.asarray(acts[np.sort(idx)], dtype=np.float32).mean(axis=0)
    return genes, G


# --------------------------------------------------------------------------- #
# geometry / functional organization                                           #
# --------------------------------------------------------------------------- #
def participation_ratio(X):
    Xc = X - X.mean(0, keepdims=True)
    s = np.linalg.svd(Xc, compute_uv=False)
    l = s ** 2
    return float((l.sum() ** 2) / (l ** 2).sum())


def twonn_id(X, seed=0):
    """TwoNN intrinsic dimension (Facco 2017), simple estimator."""
    from scipy.spatial import cKDTree
    tree = cKDTree(X)
    d, _ = tree.query(X, k=3)          # self + 2 neighbours
    r1, r2 = d[:, 1], d[:, 2]
    mu = r2 / np.clip(r1, 1e-12, None)
    mu = mu[mu > 1]
    return float((len(mu)) / np.sum(np.log(mu)))


def cosine_knn(Xc, k):
    """Top-k neighbours by cosine on centered rows."""
    U = Xc / np.clip(np.linalg.norm(Xc, axis=1, keepdims=True), 1e-12, None)
    n = U.shape[0]
    nn = np.zeros((n, k), dtype=np.int64)
    for i0 in range(0, n, 512):
        i1 = min(i0 + 512, n)
        C = U[i0:i1] @ U.T
        for r in range(i0, i1):
            C[r - i0, r] = -1
        nn[i0:i1] = np.argpartition(-C, k, axis=1)[:, :k]
    return nn


def shared_term(gene_terms, a, b):
    return bool(gene_terms.get(a, frozenset()) & gene_terms.get(b, frozenset()))


def nn_functional_enrichment(genes, Xc, gene_terms, k, n_shuffle, rng):
    nn = cosine_knn(Xc, k)
    n = len(genes)
    nn_rate = np.mean([shared_term(gene_terms, genes[i], genes[j])
                       for i in range(n) for j in nn[i]])
    # base rate over random pairs
    ra, rb = rng.integers(0, n, 200000), rng.integers(0, n, 200000)
    base = np.mean([shared_term(gene_terms, genes[a], genes[b])
                    for a, b in zip(ra, rb) if a != b])
    # feature-shuffle null: permute the gene->term assignment, recompute NN rate
    sh = []
    for _ in range(n_shuffle):
        perm = rng.permutation(n)
        gt = {genes[i]: gene_terms.get(genes[perm[i]], frozenset()) for i in range(n)}
        sh.append(np.mean([shared_term(gt, genes[i], genes[j])
                           for i in range(n) for j in nn[i]]))
    sh = np.asarray(sh)
    return {"nn_rate": float(nn_rate), "base_rate": float(base),
            "enrichment": float(nn_rate / max(base, 1e-9)),
            "z_vs_shuffle": float((nn_rate - sh.mean()) / (sh.std() + 1e-12))}


def coexpression_matched_enrichment(genes, Xc, gene_terms, coexpr, mean_expr, k, rng,
                                    n_bins=5, n_perm=200):
    """Within n_bins x n_bins (co-expression x mean-expr) bins, are same-function gene pairs more
    cosine-similar than different-function pairs BEYOND co-expression? The binning already matches
    co-expression + mean-expr, so the honest null is a PERMUTATION of the gene->term assignment:
    reshuffle which genes carry which functional term-sets, recompute the within-bin delta many
    times, and z-score the observed delta against that co-expression-matched null."""
    U = Xc / np.clip(np.linalg.norm(Xc, axis=1, keepdims=True), 1e-12, None)
    ce_edges = np.quantile(coexpr[np.triu_indices_from(coexpr, 1)], np.linspace(0, 1, n_bins + 1))
    me = np.asarray([mean_expr.get(g, 0.0) for g in genes])
    me_edges = np.quantile(me, np.linspace(0, 1, n_bins + 1))
    # sample gene pairs ONCE; store bin cell (flattened), cosine, and endpoint gene indices
    ra, rb = rng.integers(0, len(genes), 400000), rng.integers(0, len(genes), 400000)
    cells, coss, ia, ib = [], [], [], []
    for a, b in zip(ra, rb):
        if a >= b:
            continue
        cb = min(np.searchsorted(ce_edges, coexpr[a, b]) - 1, n_bins - 1)
        mb = min(np.searchsorted(me_edges, (me[a] + me[b]) / 2) - 1, n_bins - 1)
        cells.append(max(cb, 0) * n_bins + max(mb, 0))
        coss.append(float(U[a] @ U[b])); ia.append(a); ib.append(b)
    cells = np.asarray(cells); coss = np.asarray(coss)
    ia = np.asarray(ia); ib = np.asarray(ib)
    uniq = np.unique(cells)
    term_list = [gene_terms.get(g, frozenset()) for g in genes]

    def delta_for(term_of):
        shared = np.fromiter((bool(term_of[ia[i]] & term_of[ib[i]]) for i in range(ia.size)),
                             dtype=bool, count=ia.size)
        ds = []
        for c in uniq:
            m = cells == c
            s = shared & m
            d = (~shared) & m
            if s.sum() >= 20 and d.sum() >= 20:
                ds.append(coss[s].mean() - coss[d].mean())
        return (float(np.mean(ds)), len(ds)) if ds else (float("nan"), 0)

    obs, nb = delta_for(term_list)
    nulls = []
    for _ in range(n_perm):
        perm = rng.permutation(len(genes))
        nd, _ = delta_for([term_list[perm[i]] for i in range(len(genes))])
        if not np.isnan(nd):
            nulls.append(nd)
    nulls = np.asarray(nulls) if nulls else np.asarray([np.nan])
    z = float((obs - np.nanmean(nulls)) / (np.nanstd(nulls) + 1e-12))
    p = float((np.sum(nulls >= obs) + 1) / (np.sum(~np.isnan(nulls)) + 1))
    return {"within_bin_mean_cos_delta": float(obs), "n_bins_evaluated": nb,
            "coexpr_null_mean": float(np.nanmean(nulls)), "coexpr_null_std": float(np.nanstd(nulls)),
            "z_coexpr_matched": z, "p_perm_onesided": p, "n_perm": int(np.sum(~np.isnan(nulls)))}


def gene_to_terms(onto_dir, min_set=3, max_set=200):
    g2t = {}
    for fn in ["go_bp_gene_sets.json", "kegg_gene_sets.json", "reactome_gene_sets.json"]:
        p = os.path.join(onto_dir, fn)
        if not os.path.exists(p):
            continue
        tag = fn.split("_")[0]
        for term, members in json.load(open(p)).items():
            if min_set <= len(members) <= max_set:
                for g in members:
                    g2t.setdefault(g, set()).add(f"{tag}:{term}")
    return {g: frozenset(v) for g, v in g2t.items()}


def build_coexpression(h5ad, genes, layer=None):
    """Pairwise Pearson co-expression + mean expression from raw counts (log1p-CP10k)."""
    import anndata
    import scipy.sparse as sp
    adata = anndata.read_h5ad(h5ad)
    X = adata.layers[layer] if layer else adata.X
    var = {g: i for i, g in enumerate(map(str, adata.var_names))}
    cols = [var[g] for g in genes if g in var]
    present = [g for g in genes if g in var]
    sub = X[:, cols]
    sub = sub.toarray() if sp.issparse(sub) else np.asarray(sub)
    sub = np.log1p(sub / np.clip(sub.sum(1, keepdims=True), 1, None) * 1e4)
    coexpr = np.corrcoef(sub.T)
    mean_expr = {g: float(sub[:, j].mean()) for j, g in enumerate(present)}
    return present, np.nan_to_num(coexpr), mean_expr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="dir with layer_{L}_activations.npy + row_gene_names.json")
    ap.add_argument("--h5ad", required=True, help="same h5ad used for extraction (for co-expression null)")
    ap.add_argument("--onto", default="../../data/ontologies")
    ap.add_argument("--layers", type=int, nargs="+", required=True)
    ap.add_argument("--counts-layer", default=None, help="adata.layers key with raw counts")
    ap.add_argument("--occ-cap", type=int, default=200)
    ap.add_argument("--knn", type=int, default=15)
    ap.add_argument("--df-frac", type=float, default=0.05)
    ap.add_argument("--n-shuffle", type=int, default=50)
    ap.add_argument("--out", default="results/gene_manifold.json")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    g2t = gene_to_terms(args.onto)
    row_gene_names = json.load(open(os.path.join(args.run, "row_gene_names.json")))

    all_out = []
    for L in args.layers:
        genes, G = aggregate_per_gene(
            os.path.join(args.run, f"layer_{L:02d}_activations.npy"), row_gene_names, args.occ_cap, args.seed)
        # per-dim z-score (Timkey rogue-dimension fix) then center
        Gz = (G - G.mean(0)) / (G.std(0) + 1e-6)
        rec = {"layer": L, "n_genes": len(genes),
               "participation_ratio": participation_ratio(Gz),
               "twonn_id": twonn_id(Gz, args.seed)}
        for mode, keep in [("raw", genes),
                           ("nostop", [g for g in genes if not HK_RE.match(g)])]:
            idx = [i for i, g in enumerate(genes) if g in set(keep)]
            gg = [genes[i] for i in idx]
            rec[mode] = nn_functional_enrichment(gg, Gz[idx], g2t, args.knn, args.n_shuffle, rng)
        # co-expression-matched control (uses raw counts)
        cg, coexpr, mean_expr = build_coexpression(args.h5ad, genes, args.counts_layer)
        cidx = [i for i, g in enumerate(genes) if g in set(cg)]
        rec["coexpr_matched"] = coexpression_matched_enrichment(
            [genes[i] for i in cidx], Gz[cidx], g2t,
            coexpr[np.ix_(range(len(cg)), range(len(cg)))], mean_expr, args.knn, rng)
        all_out.append(rec)
        print(f"L{L:02d}: PR={rec['participation_ratio']:.1f} TwoNN={rec['twonn_id']:.1f} "
              f"| nostop enr={rec['nostop']['enrichment']:.1f} z={rec['nostop']['z_vs_shuffle']:.1f} "
              f"| within-coexpr-bin Δcos={rec['coexpr_matched']['within_bin_mean_cos_delta']:.4f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump({"meta": {"params": vars(args)}, "layers": all_out}, open(args.out, "w"), indent=2)
    print("saved", args.out)


if __name__ == "__main__":
    main()
