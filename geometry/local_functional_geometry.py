"""
local_functional_geometry.py — is the SAE feature geometry FUNCTIONALLY organized?

RUNS ON LOCAL DATA ONLY. No GPU, no model, no raw activations. Consumes:
  - runs/2b_atlas/sae/sae_L{L}.pt            (decoder directions = feature geometry)
  - runs/2b_atlas/feature_catalog_L{L}.json  (top-20 genes per feature)   [17 layers]
  - data/ontologies/{go_bp,kegg,reactome}_gene_sets.json, string_ppi_edges.json

Core question (task deliverable 2, function-not-chromosome):
  Do features that are NEAR each other in decoder-direction space (|cos| large) also
  share biological FUNCTION (overlapping top genes / a common GO-KEGG-Reactome term /
  STRING PPI edges)? If yes, the learned geometry is functionally organized — functionally
  related gene programs occupy nearby residual-stream directions.

Method
  For each catalog layer:
   1. Feature geometry: each alive feature -> unit decoder direction; nearest neighbours
      by |cos| (top-k, computed in row-blocks).
   2. Feature function: each feature -> set of top genes (from catalog); mapped to the set
      of ontology terms it "hits" (a term counts only if >= MIN_HITS of the feature's top
      genes belong to it, to suppress single-gene noise). Also a STRING-PPI adjacency:
      features A,B are PPI-linked if any top-gene of A shares a >=700 STRING edge with a
      top-gene of B.
   3. Coherence of geometry: among nearest-neighbour feature pairs, the rate of
        - gene-Jaccard > 0                (share a listed gene)
        - shared ontology term            (co-annotated)
        - STRING PPI link
      compared to (a) the base rate over random feature pairs and (b) a FEATURE-SHUFFLE
      NULL that permutes the gene/term assignment across decoder directions and recomputes.
      Enrichment = NN-rate / base-rate; z = (NN-rate − shuffle-mean) / shuffle-sd.
   4. Continuous check: Spearman(|cos|, gene-Jaccard) over a large random pair sample.

Confounds (honest)
  * HOUSEKEEPING "STOPWORD" TRAP. Ribosomal/mito gene families (RPL*/RPS*/MRPL*/MRPS*/MT-*)
    and any gene appearing in the top-genes of > --df-frac of features are treated as
    stopwords. Everything is computed BOTH raw and stopword-filtered; the filtered numbers
    are the honest ones (shared housekeeping genes create spurious functional overlap).
  * FEATURE-SHUFFLE NULL is the correct null here: it holds the geometry and the gene/term
    multiset fixed and only breaks the geometry<->function pairing, so any enrichment above
    it is real structure, not a base-rate artifact.
  * CO-EXPRESSION CONFOUND (unresolved locally). The SAE was trained on K562 activations, so
    features firing on co-expressed genes can be BOTH geometrically close AND share top genes
    for reasons of transcriptional co-occurrence, not shared molecular function. Ontology/PPI
    relatedness (external DBs) is stronger evidence than raw gene-Jaccard, but co-expressed
    genes are also co-annotated, so this test shows the geometry is *co-expression/function*
    organized and cannot fully separate the two without a co-expression-matched null (needs
    raw counts / the model — drafted for the GPU stage, see drafts/).

Usage
    python local_functional_geometry.py \
        --run ../runs/2b_atlas --onto ../data/ontologies \
        --out results/functional_geometry.json \
        [--knn 10] [--min-hits 2] [--df-frac 0.05] [--n-shuffle 50]
"""


# --- make sibling modules importable when run from anywhere ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "c2s_mechinterp"))

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "code"))
from sae_topk import TopKSAE  # noqa: E402

HK_RE = re.compile(r"^(RPL|RPS|MRPL|MRPS|MT-|RNA5|RN7|RNU|MALAT1$|MTRNR)")


def load_catalog_topgenes(path: str, top_n: int) -> tuple[list[int], list[list[str]]]:
    """Return (feature_indices, top_gene_lists) for alive features with genes."""
    d = json.load(open(path))
    feats = d["features"]
    idxs, genes = [], []
    for f in feats:
        if f.get("is_dead"):
            continue
        tg = [g["gene_name"] for g in f.get("top_genes", [])[:top_n]]
        if len(tg) >= 3:
            idxs.append(int(f["feature_idx"]))
            genes.append(tg)
    return idxs, genes


def load_dirs(run: str, layer: int) -> np.ndarray:
    sae = TopKSAE.load(os.path.join(run, "sae", f"sae_L{layer:02d}.pt"), device="cpu")
    return sae.decoder_directions().cpu().numpy().astype(np.float32)


def build_stopwords(gene_lists: list[list[str]], df_frac: float) -> set[str]:
    """Housekeeping families + high-document-frequency genes."""
    n = len(gene_lists)
    df = {}
    for gl in gene_lists:
        for g in set(gl):
            df[g] = df.get(g, 0) + 1
    stop = {g for g in df if HK_RE.match(g)}
    stop |= {g for g, c in df.items() if c > df_frac * n}
    return stop


def gene_to_terms(onto_dir: str, min_set=3, max_set=200) -> dict[str, set[str]]:
    """Map gene -> set of ontology term ids (GO BP + KEGG + Reactome), skipping huge/tiny sets."""
    g2t: dict[str, set[str]] = {}
    for fn in ["go_bp_gene_sets.json", "kegg_gene_sets.json", "reactome_gene_sets.json"]:
        p = os.path.join(onto_dir, fn)
        if not os.path.exists(p):
            continue
        tag = fn.split("_")[0]
        for term, members in json.load(open(p)).items():
            if not (min_set <= len(members) <= max_set):
                continue
            tid = f"{tag}:{term}"
            for g in members:
                g2t.setdefault(g, set()).add(tid)
    return g2t


def load_string_edges(onto_dir: str, level="pairs_700") -> dict[str, set[str]]:
    p = os.path.join(onto_dir, "string_ppi_edges.json")
    adj: dict[str, set[str]] = {}
    if not os.path.exists(p):
        return adj
    d = json.load(open(p))
    for a, b in d.get(level, []):
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)
    return adj


def feature_terms(gene_lists, g2t, stop, min_hits) -> list[set[str]]:
    """Per feature: ontology terms hit by >= min_hits of its (non-stopword) top genes."""
    out = []
    for gl in gene_lists:
        cnt: dict[str, int] = {}
        for g in gl:
            if g in stop:
                continue
            for t in g2t.get(g, ()):  # noqa
                cnt[t] = cnt.get(t, 0) + 1
        out.append({t for t, c in cnt.items() if c >= min_hits})
    return out


def feature_genesets(gene_lists, stop) -> list[frozenset]:
    return [frozenset(g for g in gl if g not in stop) for gl in gene_lists]


def knn_by_cosine(D: np.ndarray, k: int, block=1024) -> np.ndarray:
    """(n, k) indices of each row's top-k neighbours by |cos| (excludes self)."""
    n = D.shape[0]
    nn = np.zeros((n, k), dtype=np.int64)
    for i0 in range(0, n, block):
        i1 = min(i0 + block, n)
        C = np.abs(D[i0:i1] @ D.T)  # (b, n)
        for r in range(i0, i1):
            C[r - i0, r] = -1.0     # mask self
        part = np.argpartition(-C, k, axis=1)[:, :k]
        # order them (nice-to-have)
        rows = np.arange(i1 - i0)[:, None]
        ord_ = np.argsort(-C[rows, part], axis=1)
        nn[i0:i1] = part[rows, ord_]
    return nn


def pair_rates(idx_pairs, genesets, termsets, ppi_expanded):
    """Fraction of pairs sharing a gene / a term / a PPI edge.
    ppi_expanded[a] = genes-of-A UNION all STRING neighbours of genes-of-A, so a PPI link is
    a single set intersection ppi_expanded[a] & genesets[b] (fast, no nested loop)."""
    ng = nt = npp = 0
    m = len(idx_pairs)
    for a, b in idx_pairs:
        if genesets[a] & genesets[b]:
            ng += 1
        if termsets[a] & termsets[b]:
            nt += 1
        if ppi_expanded[a] & genesets[b]:
            npp += 1
    return ng / max(m, 1), nt / max(m, 1), npp / max(m, 1)


def analyze_layer(run, layer, onto_dir, g2t, ppi_adj, args, rng):
    cat = os.path.join(run, f"feature_catalog_L{layer:02d}.json")
    idxs, gene_lists = load_catalog_topgenes(cat, args.top_n)
    D = load_dirs(run, layer)
    D = D[idxs]  # align geometry to catalogued alive features
    n = len(idxs)

    out = {"layer": layer, "n_features_used": n}
    nn = knn_by_cosine(D, args.knn)
    nn_pairs = [(i, int(j)) for i in range(n) for j in nn[i]]
    for mode, stop in [("raw", set()), ("nostop", build_stopwords(gene_lists, args.df_frac))]:
        genesets = feature_genesets(gene_lists, stop)
        termsets = feature_terms(gene_lists, g2t, stop, args.min_hits)
        # PPI: expand each feature's gene set with its STRING neighbours (one-time)
        ppi_expanded = [gs | frozenset().union(*(ppi_adj.get(g, frozenset()) for g in gs)) if gs
                        else frozenset() for gs in genesets]

        # random base-rate pairs
        ra = rng.integers(0, n, size=args.n_random)
        rb = rng.integers(0, n, size=args.n_random)
        keep = ra != rb
        rand_pairs = list(zip(ra[keep].tolist(), rb[keep].tolist()))

        nn_g, nn_t, nn_p = pair_rates(nn_pairs, genesets, termsets, ppi_expanded)
        bg, bt, bp = pair_rates(rand_pairs, genesets, termsets, ppi_expanded)

        # FEATURE-SHUFFLE NULL: permute function-assignment across directions, recompute NN rates
        sh_g, sh_t, sh_p = [], [], []
        for _ in range(args.n_shuffle):
            perm = rng.permutation(n)
            gs = [genesets[p] for p in perm]
            ts = [termsets[p] for p in perm]
            ps = [ppi_expanded[p] for p in perm]
            g_, t_, p_ = pair_rates(nn_pairs, gs, ts, ps)
            sh_g.append(g_); sh_t.append(t_); sh_p.append(p_)

        def z(obs, arr):
            arr = np.asarray(arr)
            return float((obs - arr.mean()) / (arr.std() + 1e-12))

        # continuous Spearman(|cos|, gene-Jaccard) over random pairs
        from scipy.stats import spearmanr
        cos_s, jac_s = [], []
        for a, b in rand_pairs[:20000]:
            c = abs(float(D[a] @ D[b]))
            ua, ub = genesets[a], genesets[b]
            j = len(ua & ub) / max(len(ua | ub), 1)
            cos_s.append(c); jac_s.append(j)
        rho, pval = spearmanr(cos_s, jac_s)

        out[mode] = {
            "nn_share_gene": nn_g, "base_share_gene": bg,
            "enrich_gene": nn_g / max(bg, 1e-9), "z_gene_vs_shuffle": z(nn_g, sh_g),
            "nn_share_term": nn_t, "base_share_term": bt,
            "enrich_term": nn_t / max(bt, 1e-9), "z_term_vs_shuffle": z(nn_t, sh_t),
            "nn_ppi_link": nn_p, "base_ppi_link": bp,
            "enrich_ppi": nn_p / max(bp, 1e-9), "z_ppi_vs_shuffle": z(nn_p, sh_p),
            "spearman_cos_jaccard": float(rho), "spearman_p": float(pval),
            "n_stopwords": len(stop),
        }
    return out


def main():
    t0 = time.time()
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="../runs/2b_atlas")
    ap.add_argument("--onto", default="../data/ontologies")
    ap.add_argument("--out", default="results/functional_geometry.json")
    ap.add_argument("--layers", type=int, nargs="+", default=None,
                    help="default = all layers that have a feature_catalog_L*.json")
    ap.add_argument("--top-n", type=int, default=20)
    ap.add_argument("--knn", type=int, default=10)
    ap.add_argument("--min-hits", type=int, default=2)
    ap.add_argument("--df-frac", type=float, default=0.05)
    ap.add_argument("--n-random", type=int, default=200000)
    ap.add_argument("--n-shuffle", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    if args.layers is None:
        cats = sorted(glob.glob(os.path.join(args.run, "feature_catalog_L*.json")))
        args.layers = [int(re.search(r"L(\d+)", os.path.basename(c)).group(1)) for c in cats]

    print("=" * 72)
    print("C2S-Scale-2B FUNCTIONAL GEOMETRY (local: decoder dirs + catalogs + ontologies)")
    print(f"  layers={args.layers}  knn={args.knn}  min_hits={args.min_hits}  df_frac={args.df_frac}")
    print("=" * 72)
    g2t = gene_to_terms(args.onto)
    ppi_adj = load_string_edges(args.onto)
    print(f"  ontology genes mapped: {len(g2t)}   STRING nodes: {len(ppi_adj)}\n")

    results = []
    for L in args.layers:
        try:
            r = analyze_layer(args.run, L, args.onto, g2t, ppi_adj, args, rng)
        except (json.JSONDecodeError, ValueError) as e:
            print(f"  L{L:02d}: SKIPPED (unreadable catalog: {type(e).__name__})")
            continue
        results.append(r)
        ns = r["nostop"]
        print(f"  L{L:02d} n={r['n_features_used']:4d} | NOSTOP  "
              f"gene enr={ns['enrich_gene']:5.1f} (z{ns['z_gene_vs_shuffle']:6.1f}) | "
              f"term enr={ns['enrich_term']:5.1f} (z{ns['z_term_vs_shuffle']:6.1f}) | "
              f"ppi enr={ns['enrich_ppi']:5.1f} (z{ns['z_ppi_vs_shuffle']:6.1f}) | "
              f"rho(cos,jac)={ns['spearman_cos_jaccard']:.3f}")

    out = {
        "meta": {
            "run": os.path.abspath(args.run),
            "params": vars(args),
            "note": "enrich = nearest-neighbour rate / random-pair base rate; "
                    "z = vs feature-shuffle null; nostop = housekeeping-filtered (honest).",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "layers": results,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"\n  Saved: {args.out}  ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
