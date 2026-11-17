"""Export scorer-v3 atlas data into a new staging directory, without touching the live atlas.

Example on the pod (UMAP, when requested, belongs on the pod):
  python export_revision_atlas.py --run runs/2b_atlas --annotations results/annotations/2b \
    --evaluation-dir results \
    --positions-dir atlas/public/data --compute-missing-positions --out results/atlas_staging \
    --phase8-statistics results/phase8_k562/statistics

All significant ontology/regulon terms require numeric saved q/fold details. Network pair and
TF-membership evidence are exported separately. Existing annotation summary fields refer to
ontology/regulon annotation; explicit *_any fields retain the scorer's combined evidence counts.
The old browser needs the accompanying UI text/type updates before this staging data is published.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import gzip
import json
import math
from pathlib import Path
import shutil


ONTOLOGY_SOURCES = {"GO_BP", "KEGG", "Reactome", "TRRUST"}
NETWORK_SOURCES = {"STRING", "TRRUST_pairs", "TF_membership"}


def load_json(path):
    with open(path) as handle:
        return json.load(handle)


def save_json(path, payload):
    path.write_bytes(json_bytes(payload))


def json_bytes(payload):
    return json.dumps(payload, separators=(",", ":"), allow_nan=False).encode("utf-8")


def write_resource(out, logical_path, decoded, compress):
    """Hash the exact UTF-8 JSON bytes, not a later parser's reserialisation."""
    encoded = gzip.compress(decoded, compresslevel=9, mtime=0) if compress else decoded
    encoded_sha = hashlib.sha256(encoded).hexdigest()
    logical = Path(logical_path)
    path = logical.with_name(f"{logical.stem}.{encoded_sha[:16]}.json.gz") if compress else logical
    destination = out/path
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(encoded)
    return {"path": path.as_posix(), "encoding": "gzip" if compress else "identity",
            "encoded_bytes": len(encoded), "decoded_bytes": len(decoded),
            "encoded_sha256": encoded_sha, "decoded_sha256": hashlib.sha256(decoded).hexdigest()}


def write_annotation_chunks(out, layer, detailed, chunk_size=256):
    if not 1 <= chunk_size <= 512:
        raise ValueError("annotation chunk size must be between 1 and 512")
    groups = defaultdict(dict)
    for feature, record in sorted(detailed.items(), key=lambda item: int(item[0])):
        feature_id = int(feature)
        if feature_id < 0 or str(feature_id) != feature:
            raise ValueError(f"Feature ID is not a canonical nonnegative integer: {feature}")
        groups[feature_id // chunk_size][feature] = record
    chunks, files = {}, {}
    for index, records in groups.items():
        logical = f"annotations/L{layer:02d}/chunk_{index:04d}.json"
        files[logical] = write_resource(out, logical, json_bytes(records), compress=True)
        chunks[str(index)] = {"resource": logical, "first_feature_id": index*chunk_size,
                              "last_feature_id": (index+1)*chunk_size-1,
                              "n_features": len(records)}
    return {"chunk_size": chunk_size, "n_features": len(detailed), "chunks": chunks}, files


def finalize_transport(out, annotation_layers, chunk_files, gzip_threshold=1024*1024):
    """Compress large whole resources once; annotation monoliths are never written."""
    files = dict(chunk_files)
    for path in sorted(out.glob("*.json")):
        if path.name in ("data_transport.json", "export_manifest.json"):
            raise ValueError(f"Transport metadata already exists in staging: {path}")
        decoded = path.read_bytes()
        compress = len(decoded) >= gzip_threshold
        files[path.name] = write_resource(out, path.name, decoded, compress)
        if compress:
            path.unlink()
    transport = {"schema": "c2s-atlas-transport-v1", "atlas_schema": "c2s-atlas-v4",
                 "files": files, "annotations": annotation_layers,
                 "range_definition": "first_feature_id and last_feature_id are inclusive fixed feature-ID bin bounds",
                 "integrity": "SHA256 of encoded file bytes and decoded UTF-8 JSON bytes before parsing",
                 "annotation_completeness": "All exported feature objects, terms, significant IDs, network evidence and saved gene records; no display truncation of detail files"}
    save_json(out/"data_transport.json", transport)
    return transport


def check_delivery_limits(out, max_site_bytes=950_000_000, max_file_bytes=100*1024*1024, additional_files=()):
    sizes = [(path.relative_to(out).as_posix(), path.stat().st_size) for path in out.rglob("*") if path.is_file()]
    sizes.extend(additional_files)
    for name, size in sizes:
        if size >= max_file_bytes:
            raise ValueError(f"Publication file-size limit exceeded: {name} has {size} bytes (limit is strictly below {max_file_bytes})")
    total = sum(size for _, size in sizes)
    if total >= max_site_bytes:
        raise ValueError(f"Publication site-size limit exceeded: {total} bytes (limit is strictly below {max_site_bytes})")
    return {"published_bytes": total, "n_files": len(sizes), "largest_file_bytes": max((size for _, size in sizes), default=0)}


def validate_annotation(row):
    if "significant_term_ids" not in row:
        raise ValueError("Complete significant_term_ids are required; old/capped annotation output is not publishable")
    complete = set(row["significant_term_ids"])
    significant = [term for term in row.get("terms", []) if term.get("q", 1) < 0.05]
    detailed = {f"{term['source']}::{term['term']}": term for term in significant if term.get("source") in ONTOLOGY_SOURCES}
    if complete != set(detailed):
        raise ValueError(f"Missing numeric details or inconsistent complete IDs for feature {row['feature_idx']}: {sorted(complete ^ set(detailed))[:5]}")
    for term in detailed.values():
        if not all(isinstance(term.get(key), (int, float)) and math.isfinite(term[key]) for key in ("q", "fold", "k")):
            raise ValueError(f"Significant ontology term lacks numeric q/fold/overlap: {term['term']}")
    network = [term for term in significant if term.get("source") in NETWORK_SOURCES]
    if bool(complete or network) != bool(row["annotated"]):
        raise ValueError(f"Combined annotation flag disagrees with saved evidence for feature {row['feature_idx']}")
    if len(complete) + len(network) != row["n_sig"]:
        raise ValueError(f"Significant evidence count is incomplete for feature {row['feature_idx']}")
    return detailed, network


def export_feature(catalogue, annotation, module):
    terms, network = validate_annotation(annotation)
    ordered = sorted(terms.items(), key=lambda item: (item[1]["q"], item[0]))
    best = ordered[0][1] if ordered else None
    genes = catalogue.get("top_genes", [])
    compact = {"i": int(catalogue["feature_idx"]), "d": bool(catalogue["is_dead"]),
               "f": catalogue["activation_freq"], "ma": catalogue["mean_activation"], "fc": catalogue["fire_count"],
               "na": len(terms), "ne": len(network), "annotated_any": bool(annotation["annotated"]),
               "m": int(module), "sv": bool(catalogue["is_svd_aligned"]),
               "lb": best["term"] if best else "", "to": best["source"] if best else "none",
               "tg": [{"n": gene["gene_name"], "a": gene["mean_activation"]} for gene in genes[:5]]}
    detailed = {"genes": [{"n": gene["gene_name"], "a": gene["mean_activation"], "fc": gene["fire_count"]} for gene in genes],
                "significant_term_ids": sorted(terms),
                "anns": [{"o": term["source"], "t": term["term"], "id": identity,
                          "p": term["q"], "raw_p": term.get("p"), "q": term["q"],
                          "or": term["fold"], "fold": term["fold"], "effect_kind": "fold_enrichment",
                          "n": term["k"], "g": term.get("genes", [])} for identity, term in ordered],
                "network_evidence": network}
    return compact, detailed


def export_positions(args, layer, destination, n_features):
    cached = Path(args.positions_dir) / f"layer_{layer:02d}_positions.json" if args.positions_dir else None
    if cached and cached.exists():
        positions = load_json(cached)
        if len(positions) != n_features or any(len(point) != 2 for point in positions):
            raise ValueError(f"Position shape disagrees with catalogue: {cached}")
        save_json(destination, positions)
        return {"kind": "provided_dictionary_positions", "source": cached.name,
                "sha256": hashlib.sha256(cached.read_bytes()).hexdigest(),
                "requirement": "caller supplies positions from these same saved dictionaries"}
    if not args.compute_missing_positions:
        raise ValueError(f"Layer {layer} has no positions. Provide --positions-dir or run --compute-missing-positions on the pod")
    import numpy as np
    import torch
    import umap
    checkpoint = torch.load(Path(args.run) / f"sae/sae_L{layer:02d}.pt", map_location="cpu", weights_only=False)
    directions = checkpoint["model_state_dict"]["W_dec.weight"].numpy().T.copy()
    del checkpoint
    positions = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine", random_state=42, n_components=2).fit_transform(directions)
    if positions.shape != (n_features, 2) or not np.isfinite(positions).all():
        raise ValueError(f"Invalid generated positions for layer {layer}")
    save_json(destination, positions.round(5).tolist())
    return {"kind": "UMAP_decoder_directions", "seed": 42, "metric": "cosine", "n_neighbors": 15, "min_dist": 0.1}


def export_perturbations(directory, out, labels, layer=16):
    if not directory:
        save_json(out / "perturbation_response.json", {"summary": {"available": False}, "targets": []})
        return
    import numpy as np
    directory = Path(directory)
    statistics = load_json(directory / "regulatory_statistics.json")
    records = [row for row in statistics["groups"] if row["representation"] == "model" and row["layer"] == layer and row["kind"] in ("tf", "nontf")]
    with np.load(directory / f"feature_statistics_L{layer:02d}.npz") as arrays:
        lookup = {str(group): i for i, group in enumerate(arrays["groups"])}
        targets = []
        for row in records:
            ri = lookup[row["group"]]
            responding = np.flatnonzero(arrays["responding"][ri])
            strongest = responding[np.argsort(-np.abs(arrays["effect"][ri, responding]))[:20]]
            targets.append({"gene": row["group"], "tf": row["kind"] == "tf", "nk": row["n_eligible_known_target_genes"],
                            "nr": row["n_responding"], "ns": row["specificity"]["n_target_responses"],
                            "specificity_q": row["specificity"].get("q"), "n_cells": row["n_cells"],
                            "top": [{"i": int(fi), "es": float(arrays["effect"][ri, fi]), "lb": labels.get((layer, int(fi)), "")} for fi in strongest]})
    factors = [row for row in records if row["kind"] == "tf"]
    n_specific = sum(row["specificity"].get("q", 1) < 0.05 for row in factors)
    save_json(out / "perturbation_response.json", {"summary": {"available": True, "layer": layer,
        "n_targets_processed": len(records), "n_trrust_tfs": len(factors), "n_tfs_with_specific_response": n_specific,
        "frac_tfs_specific": n_specific / max(len(factors), 1), "n_targets_responding": sum(row["n_responding"] > 0 for row in records),
        "definition": "two-sided Mann-Whitney/BH plus |d|>0.5; unsigned target-set specificity; nontargeting pseudo groups excluded from this display"}, "targets": targets})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--evaluation-dir", required=True, help="Fresh 2b_eval_LXX.json files; original training metrics are never substituted")
    parser.add_argument("--out", required=True, help="New empty staging directory; never existing atlas/public/data")
    parser.add_argument("--layers", type=int, nargs="+", default=list(range(26)))
    parser.add_argument("--positions-dir")
    parser.add_argument("--compute-missing-positions", action="store_true")
    parser.add_argument("--phase8-statistics")
    parser.add_argument("--gene-index-cap", type=int, default=30)
    parser.add_argument("--ontology-index-cap", type=int, default=100)
    parser.add_argument("--annotation-chunk-size", type=int, default=256)
    parser.add_argument("--gzip-threshold-bytes", type=int, default=1024*1024)
    parser.add_argument("--max-site-bytes", type=int, default=950_000_000, help="Strict published-data budget, including the final manifest; maximum 1 GB")
    parser.add_argument("--max-file-bytes", type=int, default=100*1024*1024, help="Strict per-file budget; maximum 100 MiB")
    args = parser.parse_args()
    if not 1 <= args.annotation_chunk_size <= 512 or args.gzip_threshold_bytes < 1:
        parser.error("annotation chunk size must be 1--512 and gzip threshold must be positive")
    if not 0 < args.max_site_bytes <= 1_000_000_000 or not 0 < args.max_file_bytes <= 100*1024*1024:
        parser.error("Publication budgets must be positive and cannot exceed 1 GB total or 100 MiB per file")
    out, run = Path(args.out).resolve(), Path(args.run)
    if out.parts[-2:] == ("public", "data") or out.exists() and any(out.iterdir()):
        parser.error("--out must be an empty staging directory outside atlas/public/data")
    # Check the entire requested evaluation set before creating any staged output.
    evaluation_paths = {layer: Path(args.evaluation_dir) / f"2b_eval_L{layer:02d}.json" for layer in args.layers}
    for evaluation_path in evaluation_paths.values():
        if not evaluation_path.is_file():
            parser.error(f"Fresh evaluation required: {evaluation_path}")
    out.mkdir(parents=True, exist_ok=True)
    gene_index, ontology_index = defaultdict(list), defaultdict(list)
    ontology_counts_by_layer, gene_counts_by_layer = defaultdict(Counter), defaultdict(Counter)
    layer_summaries, modules, labels, input_hashes, coordinate_provenance = [], [], {}, {}, {}
    annotation_layers, chunk_files = {}, {}
    svd = {"per_layer": {}, "aggregate": {}}
    for layer in args.layers:
        catalogue_path, annotation_path = run / f"feature_catalog_L{layer:02d}.json", Path(args.annotations) / f"ann_v2_L{layer:02d}.json"
        catalogue, annotations = load_json(catalogue_path), load_json(annotation_path)
        if annotations["summary"].get("scorer_version") != "3-plan-aligned":
            raise ValueError(f"Unrecognised scorer version in {annotation_path}")
        input_hashes[catalogue_path.name] = hashlib.sha256(catalogue_path.read_bytes()).hexdigest()
        input_hashes[annotation_path.name] = hashlib.sha256(annotation_path.read_bytes()).hexdigest()
        by_id = {int(row["feature_idx"]): row for row in annotations["features"]}
        catalogue_ids = [int(row["feature_idx"]) for row in catalogue["features"]]
        if len(by_id) != len(annotations["features"]) or len(set(catalogue_ids)) != len(catalogue_ids):
            raise ValueError(f"Duplicate annotation or catalogue feature IDs at layer {layer}")
        module_path = run / f"coactivation_modules_L{layer:02d}.json"
        module_data = load_json(module_path) if module_path.exists() else {"module_assignment": {}, "modules": []}
        if module_path.exists():
            input_hashes[module_path.name] = hashlib.sha256(module_path.read_bytes()).hexdigest()
        compact, detailed = [], {}
        ontology_source_counts, network_source_counts = Counter(), Counter()
        n_ontology, n_network, n_any = 0, 0, 0
        for feature in catalogue["features"]:
            fi = int(feature["feature_idx"])
            if not feature["is_dead"] and fi not in by_id:
                raise ValueError(f"Alive feature {fi} is missing from layer {layer} annotation output")
            # Dead features were never tested by the scorer and carry no annotation evidence.
            row = by_id.get(fi, {"feature_idx": fi, "annotated": False, "n_sig": 0, "significant_term_ids": []})
            small, full = export_feature(feature, row, module_data["module_assignment"].get(str(fi), -1))
            compact.append(small)
            detailed[str(fi)] = full
            labels[(layer, fi)] = small["lb"]
            n_ontology += bool(small["na"])
            n_network += bool(small["ne"])
            n_any += bool(small["annotated_any"])
            ontology_source_counts.update({term["o"] for term in full["anns"]})
            network_source_counts.update({term["source"] for term in full["network_evidence"]})
            for term in full["anns"]:
                ontology_index[term["id"]].append({"l": layer, "i": fi, "p": term["q"], "o": term["o"]})
                ontology_counts_by_layer[term["id"]][str(layer)] += 1
            for rank, gene in enumerate(feature.get("top_genes", [])[:5]):
                gene_index[gene["gene_name"].upper()].append({"l": layer, "i": fi, "r": rank, "lb": small["lb"], "m": small["m"]})
                gene_counts_by_layer[gene["gene_name"].upper()][str(layer)] += 1
        if n_any != annotations["summary"]["n_annotated"]:
            raise ValueError(f"Combined evidence count disagrees with scorer summary at layer {layer}")
        save_json(out / f"layer_{layer:02d}_features.json", compact)
        annotation_layers[str(layer)], resources = write_annotation_chunks(out, layer, detailed, args.annotation_chunk_size)
        chunk_files.update(resources)
        coordinate_provenance[str(layer)] = export_positions(args, layer, out / f"layer_{layer:02d}_positions.json", len(compact))
        for module in module_data["modules"]:
            counts = Counter(identity for fi in module["features"] for identity in detailed.get(str(fi), {}).get("significant_term_ids", []))
            modules.append({"layer": layer, "id": module["module_id"], "n": len(module["features"]), "features": module["features"],
                            "top_anns": [{"t": identity, "c": count} for identity, count in counts.most_common(5)]})
        cs = catalogue["summary"]
        evaluation_path = evaluation_paths[layer]
        evaluation = load_json(evaluation_path)
        if evaluation.get("layer") != layer or evaluation.get("n_features") != len(compact):
            raise ValueError(f"Fresh evaluation layer/dictionary size disagrees with catalogue: {evaluation_path}")
        variance_explained = evaluation["variance_explained_reextracted"]
        heldout_dead = evaluation["dead_val_reextracted"]
        if not isinstance(variance_explained, (int, float)) or not math.isfinite(variance_explained):
            raise ValueError(f"Invalid fresh variance explained: {evaluation_path}")
        if not isinstance(heldout_dead, int) or not 0 <= heldout_dead <= len(compact):
            raise ValueError(f"Invalid fresh held-out inactive count: {evaluation_path}")
        input_hashes[evaluation_path.name] = hashlib.sha256(evaluation_path.read_bytes()).hexdigest()
        n_alive = sum(not feature["d"] for feature in compact)
        # Nulls were scored on the alive feature sets. Display all observed/null rates
        # per dictionary element, retaining explicit alive rates for re-use elsewhere.
        floor_scale = n_alive / len(compact)
        floors_any, floors_ontology = {}, {}
        for kind, floor in annotations["summary"].get("floors", {}).items():
            floors_any[kind] = {"mean": floor["mean"] * floor_scale,
                "sd": floor["sd"] * floor_scale,
                "values": [value * floor_scale for value in floor["values"]]}
            if "ontology_mean" in floor and "ontology_values" in floor:
                values = [value * floor_scale for value in floor["ontology_values"]]
                mean = floor["ontology_mean"] * floor_scale
                floors_ontology[kind] = {"mean": mean, "values": values,
                    "sd": math.sqrt(sum((value - mean) ** 2 for value in values) / max(len(values), 1))}
        layer_summaries.append({"layer": layer, "alive": n_alive, "dead": len(compact) - n_alive,
            "annotated": n_ontology, "annotation_rate": n_ontology / len(compact),
            "annotation_rate_alive": n_ontology / max(n_alive, 1),
            "annotated_any": n_any, "annotation_rate_any": n_any / len(compact),
            "annotation_rate_any_alive": n_any / max(n_alive, 1), "network_evidence": n_network,
            "ontology_counts": dict(ontology_source_counts), "network_counts": dict(network_source_counts),
            "n_modules": len(module_data["modules"]), "n_svd_aligned": cs["n_svd_aligned"], "n_novel": cs["n_novel"],
            "mean_feature_cosine": cs["mean_feature_cosine"], "variance_explained": variance_explained,
            "annotation_floors_any": floors_any, "annotation_floors_ontology": floors_ontology,
            "heldout_dead_features": heldout_dead, "annotation_background": "all dictionary features",
            "reconstruction_metrics_source": evaluation_path.name})
        sv = cs["svd_info"]["variance_explained_top50"]
        svd["per_layer"][str(layer)] = {"svd_variance": sv, "sae_variance": variance_explained,
            "gain": variance_explained / max(sv, 1e-12), "n_aligned": cs["n_svd_aligned"], "n_novel": cs["n_novel"]}
    summary = {"total_features": sum(row["alive"] + row["dead"] for row in layer_summaries),
        "total_alive": sum(row["alive"] for row in layer_summaries), "total_annotated": sum(row["annotated"] for row in layer_summaries),
        "total_annotated_any": sum(row["annotated_any"] for row in layer_summaries),
        "total_network_evidence": sum(row["network_evidence"] for row in layer_summaries),
        "total_modules": len(modules), "total_novel": sum(row["n_novel"] for row in layer_summaries),
        "n_layers": len(args.layers), "n_features_per_layer": len(compact), "layers": layer_summaries,
        "annotation_rate_denominator": "all_dictionary_features",
        "gene_index_top_genes_per_feature": 5, "gene_index_per_gene_cap": args.gene_index_cap}
    save_json(out / "global_summary.json", summary)
    save_json(out / "modules.json", modules)
    save_json(out / "gene_index.json", {gene: sorted(entries, key=lambda row: (row["r"], row["l"], row["i"]))[:args.gene_index_cap] for gene, entries in gene_index.items()})
    save_json(out / "gene_index_counts.json", {gene: {"total": len(gene_index[gene]), "by_layer": dict(counts)} for gene, counts in gene_counts_by_layer.items()})
    save_json(out / "ontology_index.json", {term: sorted(entries, key=lambda row: (row["p"], row["l"], row["i"]))[:args.ontology_index_cap] for term, entries in ontology_index.items()})
    save_json(out / "ontology_index_counts.json", {term: {"total": len(ontology_index[term]), "by_layer": dict(counts)} for term, counts in ontology_counts_by_layer.items()})
    svd["aggregate"] = {"total_aligned": sum(row["n_svd_aligned"] for row in layer_summaries), "total_novel": summary["total_novel"],
                        "pct_novel": summary["total_novel"] / max(summary["total_features"], 1)}
    save_json(out / "svd_comparison.json", svd)
    shutil.copyfile(run / "cross_layer_tracking.json", out / "cross_layer_tracking.json")
    input_hashes["cross_layer_tracking.json"] = hashlib.sha256((run / "cross_layer_tracking.json").read_bytes()).hexdigest()
    export_perturbations(args.phase8_statistics, out, labels)
    save_json(out / "causal_patching.json", {"summary": {"available": False}, "features": []})
    # A distinct field names the effect correctly; pmi is retained solely for current browser compatibility.
    graph = load_json(run / "circuit_graph.json")
    input_hashes["circuit_graph.json"] = hashlib.sha256((run / "circuit_graph.json").read_bytes()).hexdigest()
    pairs = defaultdict(lambda: defaultdict(list))
    for edge in graph["edges"]:
        sl, tl, sf, tf = (int(edge[key]) for key in ("source_layer", "target_layer", "source_feature", "target_feature"))
        if sl in args.layers and tl in args.layers:
            pairs[(sl, tl)][sf].append({"b": tf, "pmi": abs(edge["cohens_d"]), "effect_size": edge["cohens_d"], "lb": labels.get((tl, tf), "")})
    save_json(out / "cross_layer_graph.json", {f"L{sl:02d}_L{tl:02d}": {"summary": {
        "n_sources": len(sources), "n_edges": sum(map(len, sources.values())),
        "n_edges_stored": sum(map(len, sources.values())),
        "n_display_edges": sum(min(len(dependencies), 5) for dependencies in sources.values()),
        "stored_cap_per_source_target_layer": 50, "display_cap_per_source_target_layer": 5,
        "graph_is_capped": True, "metric": "Cohen d"},
        "deps": [{"a": sf, "deps": sorted(dependencies, key=lambda row: -row["pmi"])[:5]} for sf, dependencies in sources.items()]} for (sl, tl), sources in pairs.items()})
    transport = finalize_transport(out, annotation_layers, chunk_files, args.gzip_threshold_bytes)
    resources_by_path = {record["path"]: record for record in transport["files"].values()}
    outputs = {}
    for path in sorted(out.rglob("*")):
        if path.is_file():
            name = path.relative_to(out).as_posix()
            outputs[name] = {"bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            if name in resources_by_path:
                outputs[name].update({key: resources_by_path[name][key] for key in ("encoding", "decoded_bytes", "decoded_sha256")})
    manifest = {"schema_version": "c2s-atlas-v4", "scorer_version": "3-plan-aligned", "status": "complete",
        "model": "vandijklab/C2S-Scale-Gemma-2-2B", "layers": args.layers, "inputs_sha256": input_hashes,
        "output_files": outputs, "positions": coordinate_provenance,
        "transport": {"index": "data_transport.json", "schema": transport["schema"], "annotation_chunk_size": args.annotation_chunk_size,
                      "gzip_threshold_bytes": args.gzip_threshold_bytes, "max_site_bytes": args.max_site_bytes,
                      "max_file_bytes": args.max_file_bytes, "payload_bytes_excluding_this_manifest": sum(record["bytes"] for record in outputs.values()),
                      "annotation_chunk_count": len(chunk_files),
                      "largest_annotation_encoded_bytes": max((record["encoded_bytes"] for record in chunk_files.values()), default=0),
                      "largest_annotation_decoded_bytes": max((record["decoded_bytes"] for record in chunk_files.values()), default=0),
                      "budget_definition": "Published data includes all nested chunks, whole resources, transport index and this final manifest"},
        "definitions": {"annotated": "at least one BH q<0.05 ontology/regulon term", "annotated_any": "ontology/regulon OR network-pair OR TF-membership evidence",
                        "dead": "never fires on all atlas positions; heldout_dead_features comes from fresh held-out evaluation",
                        "reconstruction_metrics": "variance_explained_reextracted and dead_val_reextracted from required fresh 2b_eval_LXX.json",
                        "annotation_rates_and_floors": "per all dictionary features; null alive-set rates rescaled by alive/total; explicit *_alive observed rates are also retained",
                        "circuit_graph": "n_edges/n_edges_stored count saved top-50-per-source/target-layer edges; n_display_edges counts the top-5 export; omitted full measured edges cannot be recovered from this file",
                        "p": "BH-adjusted q (legacy browser field alias)", "or": "fold enrichment, not odds ratio (legacy browser field alias)",
                        "ontology_index": "display-capped by term; complete counts are in ontology_index_counts; all terms remain in per-feature files"},
        "gene_search_scope": "Only top five driving genes per feature are indexed; per-gene entries are display-capped. gene_index_counts gives complete counts within this top-five scope, not all top-twenty memberships; per-feature files retain the full saved gene lists.",
        "display_caps": {"gene_index_per_gene": args.gene_index_cap, "ontology_index_per_term": args.ontology_index_cap,
                         "graph_dependencies_per_source_target_layer": 5},
        "requires_ui_updates": ["v4 gzip transport and on-demand feature-ID chunks", "dynamic 26-layer selectors and descriptions", "separate ontology and network evidence panels and counts", "label q and fold rather than p and OR", "draw annotation chance baselines", "use complete ontology counts when index is capped", "remove unvalidated U-shape and regulatory-direction conclusions"]}
    encoded_manifest = json_bytes(manifest)
    delivery = check_delivery_limits(out, args.max_site_bytes, args.max_file_bytes, [("export_manifest.json", len(encoded_manifest))])
    (out/"export_manifest.json").write_bytes(encoded_manifest)
    print(f"Exported {summary['n_layers']} layers and {summary['total_features']} features to {out}; {delivery['published_bytes']} published bytes in {delivery['n_files']} files")


if __name__ == "__main__":
    main()
