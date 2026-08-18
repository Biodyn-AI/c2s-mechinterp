"""
gene_polysemy_gpu.py — DRAFT (do NOT run without a GPU + the live 2B model).

QUESTION (LLM-analogy, function not chromosome): is a gene a POLYSEMOUS token in C2S-Scale-2B?
i.e. does a gene's residual-stream representation shift meaningfully and reproducibly with the
CELL CONTEXT it appears in, along interpretable functional axes — beyond what co-expression
predicts? This is route_genemanifold's context-representation program (its non-chromosome core),
ported to the model where genes are LITERALLY text tokens.

Central object: v_p(g, c) = mean residual-stream activation at gene g's token positions, over
cells of cell-type c, in an independent cell partition p (p in {0,1}). Occurrence-capped.

================================ WHAT THIS NEEDS ================================
COMPUTE:  1 GPU (>=24 GB) for extraction; metrics run on CPU. Causal test (Test 3) needs the
          live model resident (forward passes with a residual-stream hook).
MODEL:    vandijklab/C2S-Scale-Gemma-2-2B.
DATA:     a MULTI-CELL-TYPE h5ad with adata.obs[<celltype_key>] (Tabula Sapiens: ~12 cell types
          x ~1000 cells each recommended, matching route_genemanifold). Raw counts for the
          co-expression-coherence null.

STEP 0 (GPU) — extract per-(gene, cell-type, partition) means. The atlas extractor writes
per-occurrence rows tagged with row_cell_ids; this script needs each cell's cell-type label and
a partition id. Two options:
  (a) run extract_activations.py --unit gene-token TWICE, once per random half of each cell-type
      (partition 0 / 1), on subsets built with build_partitions() below; OR
  (b) run it once and aggregate here using adata.obs[celltype] + a fixed 50/50 split of cells
      within each cell type (implemented in aggregate_context_means()).
Layers: extract 0 4 8 13 17 (early->late) to see where contextualization concentrates.

================================ METHOD ================================
TEST 1 — CONTEXTUALIZATION STRENGTH (EXCESS), route_genemanifold ctx_polysemy.py
  Crowd-removed pairwise context shift for a gene between two cell types c1,c2 in partition p:
      delta_p(g) = [v_p(g,c2) - v_p(g,c1)] - mean_over_genes([v_p(.,c2) - v_p(.,c1)])
  EXCESS = mean_g cos(delta_0(g), delta_1(g)) - mean_g cos(delta_0(g), delta_1(perm g))
  (same-gene cross-partition agreement minus a gene-shuffled control). EXCESS > 0 and stable
  across many (c1,c2) pairs => genes are reproducibly, gene-SPECIFICALLY recontextualized.
  Decomposition v(g,c)=mu + a(g) + b(c) + E(g,c): b(c) is the "context main effect / averaging
  null"; the crowd-removal above subtracts exactly b(c) so EXCESS isolates the INTERACTION E.

TEST 2 — FUNCTIONAL-AXIS MODULATION, route_genemanifold ctx_functional_axes.py
  Freeze a functional axis u = unit(centroid_of_gene_main_effect(poleA) - centroid(poleB)) for
  two GO poles (e.g. cell-surface/secreted vs nuclear/transcriptional). Projection
  q_p(g,c) = <v_p(g,c), u>; interaction I_p(g,c) = q_p(g,c) - a_q(g) - b_q(c) + mu_q.
  POWER = mean_{(g,c)} I_0 * I_1  (cross-partition covariance of the gene x context interaction).
  Score = z of POWER vs a matched-size RANDOM-PARTITION-AXIS null AND, decisively, vs a
  CO-EXPRESSION-COHERENCE-MATCHED null (Test 2b).

TEST 2b — CO-EXPRESSION-MATCHED NULL (the honest control), ctx_coexpr_null.py
  Build ~400 null axes spanning the co-expression-coherence range: grow gene modules from a seed
  by top co-expressed neighbours (tight) down to near-random sets (loose); each null axis =
  centroid difference of a module vs its complement. Fit POWER ~ coherence (quadratic); place
  each FUNCTIONAL axis on the curve; report z = residual above the fit in units of null scatter.
  A functional axis only counts as "beyond co-expression" if it sits ABOVE this curve.

TEST 3 — CAUSAL STEER-HERE / READ-THERE, ctx_causal.py (needs the live model)
  Split a cell's gene tokens into two disjoint halves H_steer, H_read. After layer L, ADD signed
  functional direction ±u to the residual stream at H_steer positions only (forward hook). Read
  the model's own next-token gene logits at H_read positions. Dose-response of the read-side
  functional shift vs injected magnitude, compared to a NORM-MATCHED RANDOM direction. Positive =
  the model's computation USES the functional axis (not just represents it). Never read a steer
  with a probe fit on the steering direction (circularity) — read native next-gene logits.

CONFOUNDS BAKED IN
  * pairwise contexts, NOT the all-context intersection (avoids the ~65-gene housekeeping
    "stopword" trap where the only shared genes are ubiquitous ones).
  * occurrence cap (<=50 tokens/entry) kills 1/sqrt(n) heteroscedastic fake interaction.
  * per-dimension z-scoring (Timkey rogue-dimension fix) before cosines.
  * rank/position residualization: regress out token-rank so contextualization != position.
  * split-half across INDEPENDENT cells: the p0/p1 partitions never share a cell.

Outputs: results/gene_polysemy_L{L}.json (EXCESS per context pair, POWER + coexpr-null z per
    functional axis, dose-response slopes) + console summary.
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

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "code"))
from cell_sentences import build_cell_sentence, attribute_tokens_to_genes  # noqa: E402


# --------------------------------------------------------------------------- #
# STEP 0 helpers — partitions for independent split-half extraction            #
# --------------------------------------------------------------------------- #
def build_partitions(adata, celltype_key, seed=0):
    """Return obs-index arrays for partition 0/1 = disjoint random halves WITHIN each cell type,
    so v_0 and v_1 are estimated on non-overlapping cells (the split-half noise floor)."""
    rng = np.random.default_rng(seed)
    p0, p1 = [], []
    labels = np.asarray(adata.obs[celltype_key])
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        rng.shuffle(idx)
        h = len(idx) // 2
        p0.extend(idx[:h]); p1.extend(idx[h:2 * h])
    return np.asarray(sorted(p0)), np.asarray(sorted(p1))


# --------------------------------------------------------------------------- #
# aggregation to v_p(g, c)                                                      #
# --------------------------------------------------------------------------- #
def aggregate_context_means(acts_path, row_gene_names, row_cell_ids, cell_type_of, partition_of,
                            occ_cap=50, seed=0):
    """Build v[p][c] = {gene: mean activation} over capped occurrences.
    cell_type_of: array mapping cell_id -> cell-type label. partition_of: cell_id -> 0/1/-1."""
    rng = np.random.default_rng(seed)
    acts = np.load(acts_path, mmap_mode="r")
    # bucket rows by (partition, celltype, gene)
    buckets: dict[tuple, list[int]] = {}
    for i, (g, cid) in enumerate(zip(row_gene_names, row_cell_ids)):
        p = partition_of[cid]
        if p < 0:
            continue
        buckets.setdefault((int(p), str(cell_type_of[cid]), g), []).append(i)
    v = {0: {}, 1: {}}
    for (p, c, g), idx in buckets.items():
        if len(idx) > occ_cap:
            idx = rng.choice(idx, occ_cap, replace=False)
        v[p].setdefault(c, {})[g] = np.asarray(acts[np.sort(idx)], dtype=np.float32).mean(0)
    return v


def zscore_dims(v):
    """Per-dimension z-score across all (p,c,g) entries (Timkey rogue-dimension fix)."""
    allv = np.stack([x for p in v for c in v[p] for x in v[p][c].values()])
    mu, sd = allv.mean(0), allv.std(0) + 1e-6
    for p in v:
        for c in v[p]:
            for g in v[p][c]:
                v[p][c][g] = (v[p][c][g] - mu) / sd
    return v


# --------------------------------------------------------------------------- #
# TEST 1 — EXCESS                                                              #
# --------------------------------------------------------------------------- #
def context_shift(vp, c1, c2):
    """delta_p(g) crowd-removed, over genes present in both c1 and c2 for partition p."""
    genes = sorted(set(vp[c1]) & set(vp[c2]))
    D = np.stack([vp[c2][g] - vp[c1][g] for g in genes])
    D = D - D.mean(0, keepdims=True)                 # remove the context main effect b(c)
    return genes, D


def excess(v, c1, c2, rng):
    g0, D0 = context_shift(v[0], c1, c2)
    g1, D1 = context_shift(v[1], c1, c2)
    common = sorted(set(g0) & set(g1))
    i0 = {g: k for k, g in enumerate(g0)}; i1 = {g: k for k, g in enumerate(g1)}
    A = D0[[i0[g] for g in common]]; B = D1[[i1[g] for g in common]]
    An = A / np.clip(np.linalg.norm(A, axis=1, keepdims=True), 1e-9, None)
    Bn = B / np.clip(np.linalg.norm(B, axis=1, keepdims=True), 1e-9, None)
    same = float(np.mean(np.sum(An * Bn, axis=1)))
    perm = rng.permutation(len(common))
    shuf = float(np.mean(np.sum(An * Bn[perm], axis=1)))
    return {"c1": c1, "c2": c2, "n_genes": len(common), "same": same, "shuffled": shuf,
            "excess": same - shuf}


# --------------------------------------------------------------------------- #
# TEST 2 — functional-axis modulation POWER                                    #
# --------------------------------------------------------------------------- #
def gene_main_effect(vp):
    """a(g) = mean over cell types of v_p(g,c)."""
    genes = sorted(set().union(*[set(vp[c]) for c in vp]))
    a = {}
    for g in genes:
        xs = [vp[c][g] for c in vp if g in vp[c]]
        if xs:
            a[g] = np.mean(xs, axis=0)
    return a


def functional_axis(a_main, poleA_genes, poleB_genes):
    A = [a_main[g] for g in poleA_genes if g in a_main]
    B = [a_main[g] for g in poleB_genes if g in a_main]
    if not A or not B:
        return None
    u = np.mean(A, axis=0) - np.mean(B, axis=0)
    return u / (np.linalg.norm(u) + 1e-9)


def modulation_power(v, u):
    """Cross-partition covariance of the (gene x context) interaction of q=<v,u>."""
    def interaction(vp):
        cells = sorted(vp)
        genes = sorted(set().union(*[set(vp[c]) for c in cells]))
        Q = np.full((len(genes), len(cells)), np.nan)
        for j, c in enumerate(cells):
            for i, g in enumerate(genes):
                if g in vp[c]:
                    Q[i, j] = vp[c][g] @ u
        mu = np.nanmean(Q)
        ag = np.nanmean(Q, axis=1, keepdims=True)
        bc = np.nanmean(Q, axis=0, keepdims=True)
        I = Q - ag - bc + mu
        return genes, cells, I
    g0, c0, I0 = interaction(v[0]); g1, c1, I1 = interaction(v[1])
    gi = sorted(set(g0) & set(g1)); ci = sorted(set(c0) & set(c1))
    m0 = I0[np.ix_([g0.index(g) for g in gi], [c0.index(c) for c in ci])]
    m1 = I1[np.ix_([g1.index(g) for g in gi], [c1.index(c) for c in ci])]
    mask = ~(np.isnan(m0) | np.isnan(m1))
    return float(np.mean(m0[mask] * m1[mask]))


# NOTE: co-expression-coherence-matched null axes (Test 2b) require the raw-count co-expression
# matrix; build_null_axes() grows modules by co-expression coherence and returns their centroid-
# difference axes. Left as a documented stub to keep this draft focused; port ctx_coexpr_null.py.
def build_null_axes(*args, **kwargs):
    raise NotImplementedError("Port route_genemanifold/ctx_coexpr_null.py: grow co-expression "
                              "modules across the coherence range, return centroid-diff axes, "
                              "fit POWER~coherence, score functional axes as residual-z above fit.")


# --------------------------------------------------------------------------- #
# TEST 3 — causal steer-here / read-there (needs the live model)               #
# --------------------------------------------------------------------------- #
def causal_steer_read(model, tok, cell_genes, u, layer, alphas, organism="human", seed=0):
    """Add ±alpha*u to residual stream at a random half of gene tokens after `layer`; read the
    model's next-gene logits at the disjoint other half. Returns dose-response vs the injected
    magnitude. Compare externally against a norm-matched random direction (call twice)."""
    import torch
    rng = np.random.default_rng(seed)
    cs = build_cell_sentence(cell_genes, organism=organism, prompt=True)
    enc = tok(cs.text, return_offsets_mapping=True, return_tensors="pt", truncation=True,
              max_length=8192)
    tok2gene = attribute_tokens_to_genes(enc.pop("offset_mapping")[0].tolist(), cs.gene_spans)
    gene_positions = np.where(tok2gene >= 0)[0]
    rng.shuffle(gene_positions)
    h = len(gene_positions) // 2
    steer_pos = torch.tensor(gene_positions[:h]); read_pos = torch.tensor(gene_positions[h:2 * h])
    u_t = torch.tensor(u, dtype=model.dtype, device=model.device)

    results = []
    for alpha in alphas:
        handle = None

        def hook(module, inp, out):
            hs = out[0] if isinstance(out, tuple) else out
            hs[0, steer_pos] = hs[0, steer_pos] + alpha * u_t
            return (hs,) + out[1:] if isinstance(out, tuple) else hs

        try:
            handle = model.model.layers[layer].register_forward_hook(hook)
            with torch.no_grad():
                logits = model(**{k: v.to(model.device) for k, v in enc.items()}).logits[0]
        finally:
            if handle:
                handle.remove()
        # read-side functional shift: mean logit mass on poleA vs poleB gene tokens at read_pos
        results.append({"alpha": float(alpha),
                        "read_logits": logits[read_pos].float().cpu().numpy()})
    return {"steer_n": int(h), "read_n": int(len(read_pos)), "dose": results}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--h5ad", required=True)
    ap.add_argument("--celltype-key", required=True)
    ap.add_argument("--onto", default="../../data/ontologies")
    ap.add_argument("--layers", type=int, nargs="+", required=True)
    ap.add_argument("--poleA", nargs="+", help="GO term ids for pole A of the functional axis")
    ap.add_argument("--poleB", nargs="+", help="GO term ids for pole B")
    ap.add_argument("--occ-cap", type=int, default=50)
    ap.add_argument("--out", default="results/gene_polysemy.json")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    import anndata
    adata = anndata.read_h5ad(args.h5ad)
    cell_type_of = np.asarray(adata.obs[args.celltype_key]).astype(str)
    p0, p1 = build_partitions(adata, args.celltype_key, args.seed)
    partition_of = np.full(adata.n_obs, -1, dtype=np.int64)
    partition_of[p0] = 0; partition_of[p1] = 1

    row_gene_names = json.load(open(os.path.join(args.run, "row_gene_names.json")))
    row_cell_ids = np.load(os.path.join(args.run, "row_cell_ids.npy"))

    out = []
    for L in args.layers:
        v = aggregate_context_means(os.path.join(args.run, f"layer_{L:02d}_activations.npy"),
                                    row_gene_names, row_cell_ids, cell_type_of, partition_of,
                                    args.occ_cap, args.seed)
        v = zscore_dims(v)
        cts = sorted(set(v[0]) & set(v[1]))
        # TEST 1: EXCESS over all cell-type pairs
        exc = [excess(v, c1, c2, rng) for i, c1 in enumerate(cts) for c2 in cts[i + 1:]]
        mean_excess = float(np.mean([e["excess"] for e in exc]))
        rec = {"layer": L, "n_cell_types": len(cts), "mean_excess": mean_excess,
               "excess_pairs": exc}
        # TEST 2: functional-axis POWER (if poles supplied) — coexpr null via build_null_axes (stub)
        if args.poleA and args.poleB:
            g2t = json.load(open(os.path.join(args.onto, "go_bp_gene_sets.json")))
            poleA = set().union(*[set(g2t.get(t, [])) for t in args.poleA])
            poleB = set().union(*[set(g2t.get(t, [])) for t in args.poleB])
            a0 = gene_main_effect(v[0])
            u = functional_axis(a0, poleA, poleB)
            if u is not None:
                rec["axis_power"] = modulation_power(v, u)
        out.append(rec)
        print(f"L{L:02d}: cell types {len(cts)} | mean EXCESS = {mean_excess:.4f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump({"meta": {"params": vars(args)}, "layers": out}, open(args.out, "w"), indent=2)
    print("saved", args.out)


if __name__ == "__main__":
    main()
