"""Feature annotation with a full multiple-testing correction and measured chance floors.

A feature is represented by its top genes (as stored in feature_catalog_L*.json). For one feature:

  * GO biological process, KEGG and Reactome terms, and TRRUST regulons (the target set of each
    transcription factor), restricted to the gene universe U and kept when
    MIN_TERM <= |term in U| <= MAX_TERM: a one-sided hypergeometric test for EVERY kept term.
    A term with fewer than MIN_OVERLAP shared genes gets p = 1 (it still counts as a test).
  * STRING: the number of interacting pairs among the feature's genes, against the number expected
    from the universe pair density, one-sided Poisson test.
  * TRRUST pair enrichment and TF membership enrichment are two further tests.
  * Benjamini-Hochberg within the feature, across all of its tests (all kept terms + 3).

A feature is "annotated" if any test has q < ALPHA.

Chance floors, scored by the same code on random gene sets with the same sizes:
  uniform : genes drawn uniformly from U
  freq    : genes drawn with probability proportional to how often they occur in real top-gene lists

Also reported: an empirical calibration (share of features whose smallest p-value beats the 5% / 1%
quantile of the smallest p-values of frequency-matched random sets) and a global BH over all
feature x term tests.

Usage:
  python annotate_v2.py --catalog feature_catalog_L13.json --universe row_gene_names.json \
      --ont-dir ontologies_v2 --out annotations_v2_L13.json
  (--legacy reads the older file layout: go_bp_gene_sets.json, kegg_gene_sets.json,
   reactome_gene_sets.json, string_ppi_edges.json[pairs_700], trrust_human.tsv)
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import os
import time

import numpy as np
from scipy import sparse
from scipy.stats import hypergeom, poisson

SOURCES = ["GO_BP", "KEGG", "Reactome", "TRRUST", "STRING", "TRRUST_pairs", "TF_membership"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--catalog", help="feature_catalog_L*.json")
    p.add_argument("--gene-sets", help="alternative input: JSON {name: [genes]} or [[genes], ...]")
    p.add_argument("--universe", required=True, help="row_gene_names.json or a JSON list of genes")
    p.add_argument("--ont-dir", required=True)
    p.add_argument("--legacy", action="store_true")
    p.add_argument("--out", required=True)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--min-term", type=int, default=5)
    p.add_argument("--max-term", type=int, default=500)
    p.add_argument("--min-overlap", type=int, default=2)
    p.add_argument("--n-floor-seeds", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-terms-stored", type=int, default=10000)
    p.add_argument("--no-floors", action="store_true")
    return p.parse_args()


# --------------------------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------------------------
def load_universe(path):
    with open(path) as f:
        g = json.load(f)
    return sorted({str(x).upper() for x in g if x and x != "<CELL>"})


def load_libraries(ont_dir, legacy):
    libs = {}
    names = ({"GO_BP": "go_bp_gene_sets.json", "KEGG": "kegg_gene_sets.json",
              "Reactome": "reactome_gene_sets.json"} if legacy else
             {"GO_BP": "go_bp.json", "KEGG": "kegg.json", "Reactome": "reactome.json"})
    for src, fn in names.items():
        with open(os.path.join(ont_dir, fn)) as f:
            libs[src] = {t: {g.upper() for g in genes} for t, genes in json.load(f).items()}
    # TRRUST regulons
    reg = {}
    with open(os.path.join(ont_dir, "trrust_human.tsv")) as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) >= 2:
                reg.setdefault(p[0].upper(), set()).add(p[1].upper())
    libs["TRRUST"] = {f"{tf} targets": tg for tf, tg in reg.items()}
    # STRING pairs
    pairs = set()
    if legacy:
        with open(os.path.join(ont_dir, "string_ppi_edges.json")) as f:
            for a, b in json.load(f)["pairs_700"]:
                a, b = a.upper(), b.upper()
                if a != b:
                    pairs.add((a, b) if a < b else (b, a))
    else:
        fn = sorted(glob.glob(os.path.join(ont_dir, "string_v*_pairs.tsv.gz")))[-1]
        with gzip.open(fn, "rt") as f:
            for line in f:
                a, b = line.rstrip("\n").split("\t")
                pairs.add((a, b) if a < b else (b, a))
    return libs, pairs


def load_feature_sets(args):
    """Returns (ids, gene lists, n_features_total, n_alive)."""
    if args.catalog:
        with open(args.catalog) as f:
            cat = json.load(f)
        feats = cat["features"] if isinstance(cat, dict) else cat
        alive = [ft for ft in feats if not ft.get("is_dead", False)]
        ids = [int(ft.get("feature_idx", i)) for i, ft in enumerate(alive)]
        sets = [[(g["gene_name"] if isinstance(g, dict) else g).upper() for g in ft.get("top_genes", [])]
                for ft in alive]
        return ids, sets, len(feats), len(alive)
    with open(args.gene_sets) as f:
        gs = json.load(f)
    if isinstance(gs, dict):
        ids, sets = list(gs.keys()), [[g.upper() for g in v] for v in gs.values()]
    else:
        ids, sets = list(range(len(gs))), [[g.upper() for g in v] for v in gs]
    return ids, sets, len(sets), len(sets)


# --------------------------------------------------------------------------------------------
# Scorer
# --------------------------------------------------------------------------------------------
class Scorer:
    def __init__(self, universe, libs, string_pairs, min_term, max_term, min_overlap, alpha,
                 trrust_pairs=(), trrust_tfs=()):
        self.U = universe
        self.gi = {g: i for i, g in enumerate(universe)}
        self.N = len(universe)
        self.min_overlap, self.alpha = min_overlap, alpha
        rows, cols, self.term_src, self.term_name = [], [], [], []
        for src in ["GO_BP", "KEGG", "Reactome", "TRRUST"]:
            for name, genes in libs[src].items():
                idx = [self.gi[g] for g in genes if g in self.gi]
                if min_term <= len(idx) <= max_term:
                    j = len(self.term_name)
                    rows.extend(idx)
                    cols.extend([j] * len(idx))
                    self.term_src.append(src)
                    self.term_name.append(name)
        self.n_terms = len(self.term_name)
        self.T = sparse.csc_matrix((np.ones(len(rows), np.int32), (rows, cols)),
                                   shape=(self.N, self.n_terms))
        self.K = np.asarray(self.T.sum(0)).ravel().astype(np.int64)
        self.term_src_arr = np.array(self.term_src)
        # STRING network restricted to U
        ei, ej = [], []
        for a, b in string_pairs:
            if a in self.gi and b in self.gi:
                ei.append(self.gi[a])
                ej.append(self.gi[b])
        self.M = len(ei)
        A = sparse.coo_matrix((np.ones(self.M, np.int32), (ei, ej)), shape=(self.N, self.N))
        self.A = (A + A.T).tocsr()
        self.deg = np.asarray(self.A.sum(1)).ravel().astype(np.float64)
        # A pair test is a density-based Poisson approximation, not an ontology term.
        # Preserve these identities separately so circuits never call two pair tests a shared term.
        self.R = self._adjacency(trrust_pairs)
        self.M_trrust = self.R.nnz // 2
        self.tf_mask = np.array([g in trrust_tfs for g in universe], dtype=np.int32)
        self.m_tests = self.n_terms + 3
        self.counts = {s: int((self.term_src_arr == s).sum()) for s in ["GO_BP", "KEGG", "Reactome", "TRRUST"]}

    def _adjacency(self, pairs):
        unique = {tuple(sorted((a, b))) for a, b in pairs
                  if a != b and a in self.gi and b in self.gi}
        ii = [self.gi[a] for a, b in unique]
        jj = [self.gi[b] for a, b in unique]
        matrix = sparse.coo_matrix((np.ones(len(ii), np.int32), (ii, jj)),
                                   shape=(self.N, self.N))
        return (matrix + matrix.T).tocsr()

    def to_matrix(self, sets):
        r, c = [], []
        for i, s in enumerate(sets):
            idx = sorted({self.gi[g] for g in s if g in self.gi})
            r.extend([i] * len(idx))
            c.extend(idx)
        return sparse.csr_matrix((np.ones(len(r), np.int32), (r, c)), shape=(len(sets), self.N))

    def score(self, sets, keep_terms=0):
        """Score gene sets. Returns dict of arrays (+ per-set term lists if keep_terms > 0)."""
        F = self.to_matrix(sets)
        n = np.asarray(F.sum(1)).ravel().astype(np.int64)
        O = (F @ self.T).tocsr()                                  # overlaps (sets x terms)
        # STRING: edges inside each set, and the degree-based expectation
        FA = F @ self.A
        k_str = np.asarray(FA.multiply(F).sum(1)).ravel() / 2.0
        possible_pairs = max(self.N * (self.N - 1) / 2, 1)
        set_pairs = n * (n - 1) / 2
        lam = set_pairs * self.M / possible_pairs
        # Pair counts can be one; min_overlap applies only to gene-set overlap tests.
        p_str = np.where(k_str > 0, poisson.sf(k_str - 1, lam), 1.0)
        k_tr = np.asarray((F @ self.R).multiply(F).sum(1)).ravel() / 2
        lam_tr = set_pairs * self.M_trrust / possible_pairs
        p_tr = np.where(k_tr > 0, poisson.sf(k_tr - 1, lam_tr), 1.0)
        k_tf = np.asarray(F @ self.tf_mask).ravel()
        p_tf = hypergeom.sf(k_tf - 1, self.N, int(self.tf_mask.sum()), n)
        extras = [("STRING", "interaction enrichment", k_str, lam, p_str),
                  ("TRRUST_pairs", "regulatory pair enrichment", k_tr, lam_tr, p_tr),
                  ("TF_membership", "transcription factor enrichment", k_tf,
                   n * self.tf_mask.sum() / max(self.N, 1), p_tf)]

        n_sets = len(sets)
        min_p = np.ones(n_sets)
        annotated = np.zeros(n_sets, bool)
        by_src = {s: np.zeros(n_sets, bool) for s in SOURCES}
        n_sig = np.zeros(n_sets, np.int32)
        cand_p = []                                               # all candidate p (for global BH)
        details = [None] * n_sets
        significant_term_ids = [[] for _ in sets]
        m = self.m_tests
        for i in range(n_sets):
            if n[i] < 3:
                continue
            lo, hi = O.indptr[i], O.indptr[i + 1]
            cols, k = O.indices[lo:hi], O.data[lo:hi]
            sel = k >= self.min_overlap
            cols, k = cols[sel], k[sel]
            p = hypergeom.sf(k - 1, self.N, self.K[cols], n[i]) if len(k) else np.zeros(0)
            allp = np.concatenate([p, [entry[4][i] for entry in extras]])
            order = np.argsort(allp, kind="stable")
            ps = allp[order]
            q = np.minimum.accumulate((ps * m / np.arange(1, len(ps) + 1))[::-1])[::-1]
            q = np.minimum(q, 1.0)
            min_p[i] = ps[0] if len(ps) else 1.0
            sig = q < self.alpha
            cand_p.append(allp[allp < 1.0])
            if sig.any():
                annotated[i] = True
                n_sig[i] = int(sig.sum())
                for jj in np.where(sig)[0]:
                    o = order[jj]
                    src = extras[o - len(p)][0] if o >= len(p) else self.term_src[cols[o]]
                    if o < len(p):
                        significant_term_ids[i].append(f"{src}::{self.term_name[cols[o]]}")
                    by_src[src][i] = True
            if keep_terms:
                top = []
                for jj in range(min(len(ps), keep_terms)):
                    if not (q[jj] < self.alpha) and jj > 0:
                        break
                    o = order[jj]
                    if o >= len(p):
                        src, label, observed, expected, _ = extras[o - len(p)]
                        top.append({"source": src, "term": label,
                                    "k": float(observed[i]), "expected": float(expected[i]),
                                    "p": float(ps[jj]), "q": float(q[jj])})
                    else:
                        c = cols[o]
                        Kc = int(self.K[c])
                        genes = [self.U[g] for g in F.indices[F.indptr[i]:F.indptr[i + 1]]
                                 if self.T[g, c]]
                        top.append({"source": self.term_src[c], "term": self.term_name[c],
                                    "k": int(k[o]), "K": Kc, "n": int(n[i]),
                                    "fold": float((k[o] / n[i]) / (Kc / self.N)),
                                    "p": float(ps[jj]), "q": float(q[jj]), "genes": genes})
                details[i] = top
        cand = np.concatenate(cand_p) if cand_p else np.zeros(0)
        return {"n": n, "min_p": min_p, "annotated": annotated, "by_src": by_src, "n_sig": n_sig,
                "cand_p": cand, "details": details, "k_string": k_str,
                "k_trrust": k_tr, "expected_string": lam, "p_string": p_str,
                "p_trrust_pairs": p_tr, "significant_term_ids": significant_term_ids}


def global_bh_rate(res, m_tests, alpha):
    """Share of sets with >=1 discovery under BH across ALL set x test pairs."""
    n_sets = len(res["n"])
    m_total = n_sets * m_tests
    p = np.sort(res["cand_p"])
    if len(p) == 0 or m_total == 0:
        return 0.0, 0.0
    ok = np.where(p <= alpha * np.arange(1, len(p) + 1) / m_total)[0]
    if len(ok) == 0:
        return 0.0, 0.0
    thr = p[ok[-1]]
    return float((res["min_p"] <= thr).sum() / max(len(res["n"]), 1)), float(thr)


def main():
    args = parse_args()
    t0 = time.time()
    rng = np.random.default_rng(args.seed)
    U = load_universe(args.universe)
    libs, pairs = load_libraries(args.ont_dir, args.legacy)
    regulatory_pairs, factor_names = set(), set()
    with open(os.path.join(args.ont_dir, "trrust_human.tsv")) as handle:
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            if len(fields) >= 2:
                factor, target = fields[0].upper(), fields[1].upper()
                factor_names.add(factor)
                regulatory_pairs.add((factor, target))
    sc = Scorer(U, libs, pairs, args.min_term, args.max_term, args.min_overlap, args.alpha,
                trrust_pairs=regulatory_pairs, trrust_tfs=factor_names)
    ids, sets, n_total, n_alive = load_feature_sets(args)
    sets_u = [sorted({g for g in s if g in sc.gi}) for s in sets]
    print(f"universe {sc.N} genes | tests per feature {sc.m_tests} ({sc.counts}, 3 network/membership tests; "
          f"{sc.M} STRING pairs in universe) | {len(sets)} gene sets", flush=True)

    real = sc.score(sets_u, keep_terms=args.max_terms_stored)
    n_sets = len(sets_u)
    rate = float(real["annotated"].sum() / max(n_sets, 1))
    summary = {
        "n_features_total": n_total, "n_alive": n_alive, "n_sets": n_sets,
        "n_testable": int((real["n"] >= 3).sum()),
        "n_annotated": int(real["annotated"].sum()), "annotation_rate": rate,
        "rate_by_source": {s: float(real["by_src"][s].sum() / max(n_sets, 1)) for s in SOURCES},
        "universe_size": sc.N, "tests_per_feature": sc.m_tests, "terms_by_source": sc.counts,
        "string_pairs_in_universe": sc.M, "trrust_pairs_in_universe": sc.M_trrust,
        "pair_null": "Poisson approximation using universe pair density",
        "scorer_version": "3-plan-aligned", "alpha": args.alpha, "min_term": args.min_term,
        "max_term": args.max_term, "min_overlap": args.min_overlap, "legacy_libraries": bool(args.legacy),
        "median_set_size": float(np.median(real["n"])) if n_sets else 0.0,
    }
    n_ontology = sum(bool(ids_) for ids_ in real["significant_term_ids"])
    summary.update({"n_ontology_annotated": n_ontology,
                    "n_network_or_membership_only": int(real["annotated"].sum()) - n_ontology,
                    "ontology_annotation_rate_alive": n_ontology / max(n_sets, 1),
                    "ontology_annotation_rate_all_features": n_ontology / max(n_total, 1),
                    "annotation_rate_all_features": int(real["annotated"].sum()) / max(n_total, 1)})
    g_rate, g_thr = global_bh_rate(real, sc.m_tests, args.alpha)
    summary["global_bh"] = {"annotation_rate": g_rate, "p_threshold": g_thr}

    if not args.no_floors:
        sizes = real["n"]
        counts = {}
        for s in sets_u:
            for g in s:
                counts[g] = counts.get(g, 0) + 1
        pool = np.array(sorted(counts))
        w = np.array([counts[g] for g in pool], float)
        w /= w.sum()
        Uarr = np.array(U)
        floors = {"uniform": [], "freq": []}
        null_minp = {"uniform": [], "freq": []}
        ontology_floor = {"uniform": [], "freq": []}
        by_src_floor = {"uniform": {s: [] for s in SOURCES}, "freq": {s: [] for s in SOURCES}}
        gb = {"uniform": [], "freq": []}
        for _ in range(args.n_floor_seeds):
            for kind in ("uniform", "freq"):
                rs = []
                for nn in sizes:
                    if nn == 0:
                        rs.append([])
                    elif kind == "uniform":
                        rs.append(list(rng.choice(Uarr, size=int(nn), replace=False)))
                    else:
                        rs.append(list(rng.choice(pool, size=int(min(nn, len(pool))), replace=False, p=w)))
                r = sc.score(rs)
                floors[kind].append(float(r["annotated"].sum() / max(n_sets, 1)))
                ontology_floor[kind].append(float(np.mean([bool(ids_) for ids_ in r["significant_term_ids"]])))
                null_minp[kind].append(r["min_p"][r["n"] >= 3])
                for s in SOURCES:
                    by_src_floor[kind][s].append(float(r["by_src"][s].sum() / max(n_sets, 1)))
                gb[kind].append(global_bh_rate(r, sc.m_tests, args.alpha)[0])
        summary["floors"] = {
            kind: {"mean": float(np.mean(v)), "sd": float(np.std(v)), "values": v,
                   "ontology_mean": float(np.mean(ontology_floor[kind])), "ontology_values": ontology_floor[kind],
                   "by_source_mean": {s: float(np.mean(by_src_floor[kind][s])) for s in SOURCES},
                   "global_bh_mean": float(np.mean(gb[kind]))}
            for kind, v in floors.items()}
        summary["excess_over_freq_floor"] = rate - summary["floors"]["freq"]["mean"]
        nm = np.sort(np.concatenate(null_minp["freq"]))
        # Inclusive ranks and the plus-one correction handle discrete/tied p-values.
        # A raw lower quantile with <= can otherwise declare every all-p=1 set significant.
        empirical_p = (1 + np.searchsorted(nm, real["min_p"], side="right")) / (len(nm) + 1)
        null_empirical_p = (1 + np.searchsorted(nm, nm, side="right")) / (len(nm) + 1)
        emp = {}
        for a in (0.05, 0.01):
            selected = empirical_p <= a
            emp[f"alpha_{a}"] = {"rate": float(selected.sum() / max(n_sets, 1)),
                                 "null_rate_check": float((null_empirical_p <= a).mean()),
                                 "calibration": "inclusive plus-one empirical p against pooled frequency null"}

        summary["empirical_min_p"] = emp

    feats = []
    for i, fid in enumerate(ids):
        rec = {"feature_idx": fid, "n_genes": int(real["n"][i]), "annotated": bool(real["annotated"][i]),
               "min_p": float(real["min_p"][i]), "n_sig": int(real["n_sig"][i]),
               "significant_term_ids": real["significant_term_ids"][i]}
        if real["details"][i]:
            rec["terms"] = real["details"][i]
        feats.append(rec)
    summary["seconds"] = round(time.time() - t0, 1)
    with open(args.out, "w") as f:
        json.dump({"summary": summary, "features": feats}, f)
    fl = summary.get("floors", {})
    print(f"annotated {summary['n_annotated']}/{n_sets} = {100 * rate:.1f}% | floors: uniform "
          f"{100 * fl.get('uniform', {}).get('mean', float('nan')):.1f}%, freq "
          f"{100 * fl.get('freq', {}).get('mean', float('nan')):.1f}% | global-BH {100 * g_rate:.1f}% | "
          f"by source { {s: round(100 * v, 1) for s, v in summary['rate_by_source'].items()} } | "
          f"{summary['seconds']}s", flush=True)


if __name__ == "__main__":
    main()
