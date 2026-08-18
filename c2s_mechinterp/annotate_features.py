"""
Phase-4 adapter for C2S-Scale: annotate SAE features against biological ontologies.

Faithful port of repos/bio-sae/src/04_annotate_features.py (pinned). Only the I/O is
changed to the C2S-Scale atlas run-dir contract:
  * reads   feature_catalog_L{L:02d}.json   (from feature_catalog.py) in --run
  * ontologies + TRRUST come from --ontology-dir (trrust_human.tsv lives inside it),
    replacing the reference's hardcoded ONTOLOGY_DIR / TRRUST_PATH
  * writes  feature_annotations_L{L:02d}.json  into --run

For each alive feature's top-gene set the same tests run, unchanged:
  * GO Biological Process / KEGG / Reactome : Fisher's exact (alternative='greater')
  * STRING PPI (pairs_700 + pairs_900)      : within-set edge-density enrichment
  * TRRUST TF-target                        : within-set edge-density enrichment
  * TRRUST TFs                              : Fisher's exact TF over-representation
BH-FDR (alpha 0.05) is applied across all p-value tests together, exactly as the reference.

Usage:
  python annotate_features.py --run runs/2b_atlas --layer 13 \
      --ontology-dir /Volumes/Crucial\\ X6/MacBook/Code/biomi_automation/projects/c2s-scale/data/ontologies
"""

from __future__ import annotations

import argparse
import json
import os
import time
import warnings
from collections import defaultdict

import numpy as np
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")

# ============================================================
# Configuration (identical thresholds to the reference)
# ============================================================
DEFAULT_ONTOLOGY_DIR = (
    "/Volumes/Crucial X6/MacBook/Code/biomi_automation/projects/c2s-scale/data/ontologies"
)

TOP_N_GENES = 20        # Top genes per feature to test
MIN_OVERLAP = 2         # Minimum overlap for testing
MIN_TERM_SIZE = 5       # Minimum genes in a term
MAX_TERM_SIZE = 500     # Maximum genes in a term
BH_ALPHA = 0.05         # BH correction threshold


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Not JSON serializable: {type(obj)}")


def _gene_symbol(g):
    """Extract a gene symbol from a top_genes entry (dict {'gene_name':...} or bare str)."""
    if isinstance(g, str):
        return g
    if isinstance(g, dict):
        return g.get("gene_name") or g.get("gene") or g.get("name") or g.get("symbol")
    return None


def _feature_genes(feat):
    """Uppercased set of a feature's top-gene symbols (contract uses gene symbols directly)."""
    genes = set()
    for g in feat.get("top_genes", []):
        sym = _gene_symbol(g)
        if sym:
            genes.add(sym.upper())
    return genes


def _is_dead(feat):
    return bool(feat.get("is_dead", False))


def _feature_idx(feat, fallback):
    idx = feat.get("feature_idx")
    if idx is None:
        idx = feat.get("feature_id")
    return fallback if idx is None else idx


def load_ontologies(ontology_dir):
    """Load all biological ontologies from ontology_dir (contract layout)."""
    ontologies = {}

    # GO Biological Process
    go_bp_path = os.path.join(ontology_dir, "go_bp_gene_sets.json")
    if os.path.exists(go_bp_path):
        with open(go_bp_path) as f:
            raw = json.load(f)
        ontologies["GO_BP"] = {k: set(v) for k, v in raw.items()
                               if MIN_TERM_SIZE <= len(v) <= MAX_TERM_SIZE}
        print(f"  GO BP: {len(ontologies['GO_BP'])} terms")

    # KEGG
    kegg_path = os.path.join(ontology_dir, "kegg_gene_sets.json")
    if os.path.exists(kegg_path):
        with open(kegg_path) as f:
            raw = json.load(f)
        ontologies["KEGG"] = {k: set(v) for k, v in raw.items()
                              if MIN_TERM_SIZE <= len(v) <= MAX_TERM_SIZE}
        print(f"  KEGG: {len(ontologies['KEGG'])} terms")

    # Reactome
    reactome_path = os.path.join(ontology_dir, "reactome_gene_sets.json")
    if os.path.exists(reactome_path):
        with open(reactome_path) as f:
            raw = json.load(f)
        ontologies["Reactome"] = {k: set(v) for k, v in raw.items()
                                  if MIN_TERM_SIZE <= len(v) <= MAX_TERM_SIZE}
        print(f"  Reactome: {len(ontologies['Reactome'])} terms")

    # STRING PPI (edges, not gene sets — different analysis)
    string_path = os.path.join(ontology_dir, "string_ppi_edges.json")
    if os.path.exists(string_path):
        with open(string_path) as f:
            raw = json.load(f)
        string_edges = set()
        for tier in ["pairs_700", "pairs_900"]:
            if tier in raw:
                for pair in raw[tier]:
                    if len(pair) == 2:
                        a, b = pair[0].upper(), pair[1].upper()
                        string_edges.add((min(a, b), max(a, b)))
        ontologies["STRING_edges"] = string_edges
        print(f"  STRING PPI: {len(string_edges)} edges")

    # TRRUST (tf<TAB>target<TAB>mode<TAB>pmid, no header)
    trrust_path = os.path.join(ontology_dir, "trrust_human.tsv")
    if os.path.exists(trrust_path):
        trrust_edges = set()
        trrust_tfs = set()
        with open(trrust_path) as f:
            for line in f:
                line = line.rstrip("\n")
                if not line:
                    continue
                parts = line.split("\t")
                if len(parts) < 2:
                    continue
                tf, tgt = parts[0].upper(), parts[1].upper()
                trrust_edges.add((tf, tgt))
                trrust_tfs.add(tf)
        ontologies["TRRUST_edges"] = trrust_edges
        ontologies["TRRUST_tfs"] = trrust_tfs
        print(f"  TRRUST: {len(trrust_edges)} edges, {len(trrust_tfs)} TFs")

    return ontologies


def test_gene_set_enrichment(gene_set, term_genes, background_size):
    """Fisher's exact test (alternative='greater') for gene-set enrichment.

    Returns (odds_ratio, p_value, overlap_genes); None odds_ratio if under MIN_OVERLAP.
    """
    overlap = gene_set & term_genes
    if len(overlap) < MIN_OVERLAP:
        return None, 1.0, set()

    a = len(overlap)
    b = len(gene_set) - a
    c = len(term_genes) - a
    d = background_size - a - b - c

    table = [[a, b], [c, d]]
    odds_ratio, p_value = fisher_exact(table, alternative="greater")

    return odds_ratio, p_value, overlap


def test_edge_enrichment(gene_set, edge_set):
    """Count within-set edges among a feature's top genes.

    Returns (n_edges_within, n_possible, density).
    """
    gene_list = list(gene_set)
    n = len(gene_list)
    if n < 2:
        return 0, 0, 0.0

    n_possible = n * (n - 1) // 2
    n_within = 0
    for i in range(n):
        for j in range(i + 1, n):
            a, b = gene_list[i].upper(), gene_list[j].upper()
            edge = (min(a, b), max(a, b))
            if edge in edge_set:
                n_within += 1

    density = n_within / max(n_possible, 1)
    return n_within, n_possible, density


def main():
    parser = argparse.ArgumentParser(description="C2S-Scale Phase-4 SAE feature annotation")
    parser.add_argument("--run", required=True, help="atlas run dir (holds feature_catalog_L*.json)")
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--ontology-dir", default=DEFAULT_ONTOLOGY_DIR)
    args = parser.parse_args()

    run_dir = args.run
    layer = args.layer

    print("=" * 70)
    print("C2S-SCALE PHASE 4: ANNOTATE SAE FEATURES")
    print(f"  Run: {run_dir}  Layer: {layer:02d}")
    print("=" * 70)

    total_t0 = time.time()

    # ============================================================
    # STEP 1: Load feature catalog
    # ============================================================
    print("\nSTEP 1: Loading feature catalog...")

    catalog_path = os.path.join(run_dir, f"feature_catalog_L{layer:02d}.json")
    with open(catalog_path) as f:
        catalog_data = json.load(f)

    if isinstance(catalog_data, list):
        features = catalog_data
        summary = {}
    else:
        features = catalog_data["features"]
        summary = catalog_data.get("summary", {})

    n_features = summary.get("n_features", len(features))
    n_alive = summary.get("n_alive", sum(1 for f in features if not _is_dead(f)))
    print(f"  Features: {n_features} total, {n_alive} alive")

    # ============================================================
    # STEP 2: Load ontologies
    # ============================================================
    print("\nSTEP 2: Loading ontologies...")
    ontologies = load_ontologies(args.ontology_dir)

    # Build background gene set (all genes that appear in any feature)
    all_genes = set()
    for feat in features:
        all_genes |= _feature_genes(feat)
    background_size = len(all_genes)
    print(f"  Background gene set: {background_size} genes")

    # ============================================================
    # STEP 3: Run enrichment tests
    # ============================================================
    print("\nSTEP 3: Running enrichment tests...")
    t0 = time.time()

    all_pvals = []  # gene-set tests carry 'p_value'; edge tests do not

    alive_features = [f for f in features if not _is_dead(f)]
    print(f"  Testing {len(alive_features)} alive features...")

    for fi, feat in enumerate(alive_features):
        if fi % 500 == 0 and fi > 0:
            print(f"    Feature {fi}/{len(alive_features)}...")

        top_genes = _feature_genes(feat)
        if len(top_genes) < 3:
            continue

        feature_idx = _feature_idx(feat, fi)

        # Test gene-set ontologies (GO, KEGG, Reactome)
        for ont_name in ["GO_BP", "KEGG", "Reactome"]:
            if ont_name not in ontologies:
                continue
            for term, term_genes in ontologies[ont_name].items():
                term_genes_upper = set(g.upper() for g in term_genes)
                odds_ratio, pval, overlap = test_gene_set_enrichment(
                    top_genes, term_genes_upper, background_size)
                if odds_ratio is not None:
                    all_pvals.append({
                        "feature_idx": feature_idx,
                        "ontology": ont_name,
                        "term": term,
                        "p_value": pval,
                        "odds_ratio": odds_ratio,
                        "overlap": list(overlap),
                        "n_overlap": len(overlap),
                        "term_size": len(term_genes),
                    })

        # Test edge-based enrichment (STRING, TRRUST)
        for edge_ont in ["STRING_edges", "TRRUST_edges"]:
            if edge_ont not in ontologies:
                continue
            n_within, n_possible, density = test_edge_enrichment(
                top_genes, ontologies[edge_ont])
            if n_within > 0:
                all_pvals.append({
                    "feature_idx": feature_idx,
                    "ontology": edge_ont,
                    "term": "pairwise_enrichment",
                    "n_edges_within": n_within,
                    "n_possible": n_possible,
                    "density": density,
                })

        # Test TF enrichment (are top genes enriched for TFs?)
        if "TRRUST_tfs" in ontologies:
            tf_overlap = top_genes & ontologies["TRRUST_tfs"]
            if len(tf_overlap) >= MIN_OVERLAP:
                n_tf_bg = len(ontologies["TRRUST_tfs"] & all_genes)
                odds_ratio, pval, _ = test_gene_set_enrichment(
                    top_genes, ontologies["TRRUST_tfs"] & all_genes, background_size)
                if odds_ratio is not None:
                    all_pvals.append({
                        "feature_idx": feature_idx,
                        "ontology": "TRRUST_TF_enrichment",
                        "term": "transcription_factors",
                        "p_value": pval,
                        "odds_ratio": odds_ratio,
                        "overlap": list(tf_overlap),
                        "n_overlap": len(tf_overlap),
                        "term_size": n_tf_bg,
                    })

    print(f"  Total tests: {len(all_pvals)}")
    print(f"  Enrichment testing time: {time.time()-t0:.1f}s")

    # ============================================================
    # STEP 4: BH correction (across all p-value tests together)
    # ============================================================
    print("\nSTEP 4: BH correction...")

    pval_tests = [t for t in all_pvals if "p_value" in t]
    edge_tests = [t for t in all_pvals if "p_value" not in t]

    if pval_tests:
        raw_pvals = np.array([t["p_value"] for t in pval_tests])
        reject, corrected, _, _ = multipletests(raw_pvals, alpha=BH_ALPHA, method="fdr_bh")

        for i, test in enumerate(pval_tests):
            test["p_adjusted"] = float(corrected[i])
            test["significant"] = bool(reject[i])

        n_significant = int(reject.sum())
        print(f"  Significant tests (BH<{BH_ALPHA}): {n_significant}/{len(pval_tests)}")
    else:
        n_significant = 0

    # ============================================================
    # STEP 5: Build per-feature annotation summary
    # ============================================================
    print("\nSTEP 5: Building annotation summary...")

    feature_annotations = defaultdict(list)
    for test in pval_tests:
        if test.get("significant", False):
            feature_annotations[test["feature_idx"]].append({
                "ontology": test["ontology"],
                "term": test["term"],
                "p_adjusted": test["p_adjusted"],
                "odds_ratio": test["odds_ratio"],
                "n_overlap": test["n_overlap"],
                "overlap_genes": test.get("overlap", []),
            })

    for test in edge_tests:
        if test.get("n_edges_within", 0) > 0:
            feature_annotations[test["feature_idx"]].append({
                "ontology": test["ontology"],
                "term": test["term"],
                "n_edges": test["n_edges_within"],
                "density": test["density"],
            })

    n_annotated = len(feature_annotations)
    print(f"  Features with at least one annotation: {n_annotated}/{n_alive}")
    print(f"  Annotation rate: {100*n_annotated/max(n_alive,1):.1f}%")

    # Count by ontology
    ont_counts = defaultdict(int)
    for fidx, anns in feature_annotations.items():
        for ann in anns:
            ont_counts[ann["ontology"]] += 1
    for ont, count in sorted(ont_counts.items(), key=lambda x: -x[1]):
        print(f"    {ont}: {count} feature-term associations")

    # ============================================================
    # STEP 6: Save annotations
    # ============================================================
    print("\nSTEP 6: Saving annotations...")

    output = {
        "summary": {
            "run_dir": run_dir,
            "layer": layer,
            "n_features": n_features,
            "n_alive": n_alive,
            "n_annotated": n_annotated,
            "annotation_rate": n_annotated / max(n_alive, 1),
            "n_significant_tests": int(n_significant),
            "total_tests": len(pval_tests),
            "bh_alpha": BH_ALPHA,
            "ontology_counts": dict(ont_counts),
        },
        "feature_annotations": {str(k): v for k, v in feature_annotations.items()},
        "all_significant_tests": [t for t in pval_tests if t.get("significant", False)],
        "edge_enrichments": edge_tests,
    }

    out_path = os.path.join(run_dir, f"feature_annotations_L{layer:02d}.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=_json_default)

    total_time = time.time() - total_t0
    print(f"\n{'=' * 70}")
    print(f"DONE. Total time: {total_time/60:.1f} min")
    print(f"  Annotated features: {n_annotated}/{n_alive} "
          f"({100*n_annotated/max(n_alive,1):.1f}%)")
    print(f"  Significant enrichments: {n_significant}")
    print(f"  Annotations: {out_path}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
