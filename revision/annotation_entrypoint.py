"""Bridge the public --run/--layer annotation CLI to the full-family v3 scorer."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import sys


ONTOLOGY_SOURCES = {"GO_BP", "KEGG", "Reactome", "TRRUST"}
NETWORK_SOURCES = {"STRING", "TRRUST_pairs", "TF_membership"}


def library_layout(directory, requested):
    """Layout describes file names, never the multiple-testing procedure."""
    directory = Path(directory)
    common = (directory / "trrust_human.tsv").is_file()
    versioned = common and all((directory / name).is_file() for name in ("go_bp.json", "kegg.json", "reactome.json")) and bool(list(directory.glob("string_v*_pairs.tsv.gz")))
    snapshot = common and all((directory / name).is_file() for name in ("go_bp_gene_sets.json", "kegg_gene_sets.json", "reactome_gene_sets.json", "string_ppi_edges.json"))
    available = {"versioned": versioned, "snapshot": snapshot}
    if requested != "auto":
        if not available[requested]:
            raise ValueError(f"Incomplete {requested} ontology library at {directory}")
        return requested
    matches = [name for name, present in available.items() if present]
    if len(matches) != 1:
        raise ValueError("Ontology layout is missing or ambiguous; choose --library-layout versioned or snapshot and supply all files")
    return matches[0]


def add_compatibility_fields(payload, layer, universe, layout, alpha):
    """Keep v3 records while exposing true ontology terms to existing atlas consumers.

    A shared pair-count test is not a shared functional term. Network evidence therefore
    has its own mapping. The old odds_ratio field is not populated with a fold enrichment.
    """
    ontology, network = {}, {}
    source_counts, network_counts = Counter(), Counter()
    for feature in payload["features"]:
        identity = str(feature["feature_idx"])
        terms = [term for term in feature.get("terms", []) if term.get("q", 1) < alpha]
        annotated_terms = [term for term in terms if term["source"] in ONTOLOGY_SOURCES]
        networks = [term for term in terms if term["source"] in NETWORK_SOURCES]
        complete = set(feature["significant_term_ids"])
        if complete != {f"{term['source']}::{term['term']}" for term in annotated_terms}:
            raise ValueError(f"Significant term details were capped or are inconsistent for feature {identity}")
        if annotated_terms:
            ontology[identity] = [{"ontology": term["source"], "term": term["term"],
                                   "p_adjusted": term["q"], "raw_p": term["p"],
                                   "fold_enrichment": term["fold"], "n_overlap": term["k"],
                                   "overlap_genes": term.get("genes", [])} for term in annotated_terms]
        if networks:
            network[identity] = networks
        source_counts.update({term["source"] for term in annotated_terms})
        network_counts.update({term["source"] for term in networks})
    summary = payload["summary"]
    summary.update({"layer": layer, "n_features": summary["n_features_total"],
                    "n_annotated_any": summary["n_annotated"], "n_ontology_annotated": len(ontology),
                    "n_network_evidence": len(network),
                    "ontology_annotation_rate": len(ontology) / max(summary["n_alive"], 1),
                    "ontology_counts": dict(source_counts), "network_counts": dict(network_counts)})
    payload["feature_annotations"], payload["network_evidence"] = ontology, network
    payload["metadata"] = {
        "library_layout": layout, "schema": "full-family-v3-with-ontology-compatibility",
        "universe": {"file": universe.name, "sha256": hashlib.sha256(universe.read_bytes()).hexdigest(),
                     "definition": "All observed input gene symbols, or an explicit full-universe JSON; not the union of selected feature genes"},
        "correction": {"method": "Benjamini-Hochberg", "scope": "within each feature",
                       "family": "all eligible ontology/regulon terms plus STRING pair, TRRUST pair and TF-membership tests",
                       "zero_or_one_overlap": "p=1 and retained in the test count", "alpha": alpha},
        "ontology_sources": sorted(ONTOLOGY_SOURCES), "network_sources": sorted(NETWORK_SOURCES),
        "pair_null": "one-sided Poisson approximation using pair density in the full observed universe",
        "membership_null": "one-sided hypergeometric test for TF membership",
        "floors": "Size-matched uniform and feature-gene-frequency-weighted random sets scored by the identical full family",
        "summary_annotation_rate": "any significant ontology, network-pair or TF-membership test among atlas-active features",
        "compatibility_annotation_records": "only significant ontology/regulon terms; fold_enrichment is not an odds ratio",
        "source_counts": "features with at least one significant term/test from each source; source groups overlap",
    }
    return payload


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--ontology-dir", type=Path, required=True)
    parser.add_argument("--universe", type=Path, help="Full observed gene list; defaults to RUN/row_gene_names.json")
    parser.add_argument("--library-layout", choices=("auto", "versioned", "snapshot"), default="auto")
    parser.add_argument("--n-floor-seeds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--min-term", type=int, default=5)
    parser.add_argument("--max-term", type=int, default=500)
    args = parser.parse_args(argv)
    universe = args.universe or args.run / "row_gene_names.json"
    catalogue = args.run / f"feature_catalog_L{args.layer:02d}.json"
    if not universe.is_file():
        parser.error("The full observed gene universe is required: supply row_gene_names.json in --run or provide --universe. Top-gene-only backgrounds are not supported.")
    if args.n_floor_seeds < 1:
        parser.error("--n-floor-seeds must be positive so the annotation chance floor is measured")
    try:
        layout = library_layout(args.ontology_dir, args.library_layout)
    except ValueError as error:
        parser.error(str(error))
    symbols = {str(gene).upper() for gene in json.loads(universe.read_text()) if gene and gene != "<CELL>"}
    content = json.loads(catalogue.read_text())
    features = content["features"] if isinstance(content, dict) else content
    selected = {str(gene["gene_name"] if isinstance(gene, dict) else gene).upper()
                for feature in features if not feature.get("is_dead", False) for gene in feature.get("top_genes", [])}
    if not selected <= symbols:
        parser.error(f"Catalogue genes are absent from the full universe: {sorted(selected - symbols)[:10]}")
    output = args.run / f"feature_annotations_L{args.layer:02d}.json"
    # Failed scoring or compatibility validation cannot leave a valid-looking final file.
    temporary = output.with_suffix(".pending.json")
    scorer = Path(__file__).with_name("annotate_v2.py")
    command = [sys.executable, str(scorer), "--catalog", str(catalogue), "--universe", str(universe),
               "--ont-dir", str(args.ontology_dir), "--out", str(temporary), "--n-floor-seeds", str(args.n_floor_seeds),
               "--seed", str(args.seed), "--alpha", str(args.alpha), "--min-term", str(args.min_term),
               "--max-term", str(args.max_term), "--max-terms-stored", "10000"]
    if layout == "snapshot":
        command.append("--legacy")  # File layout only; v3 statistics remain the default.
    subprocess.run(command, check=True)
    payload = add_compatibility_fields(json.loads(temporary.read_text()), args.layer, universe, layout, args.alpha)
    temporary.write_text(json.dumps(payload, allow_nan=False))
    temporary.replace(output)
    print(f"Full-family annotations written to {output}")


if __name__ == "__main__":
    main()
