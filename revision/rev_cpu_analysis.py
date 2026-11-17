"""CPU analyses for CBAC revision: circuits, biological categories and dimensionality.

Run substantive analyses on the pod. Every command saves its inputs and definitions with the
statistics. Circuit edges use effect/consistency cutoffs, not a multiple-testing-corrected p-value.
Bootstrap intervals resample source features because edges sharing a source are dependent.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np


REAL_SOURCES = ("GO_BP", "KEGG", "Reactome", "TRRUST")
CATEGORY_RULES = [
    ("haem and erythroid", ("erythro", "hemoglobin", "haemoglobin", "heme ", "haem ", "porphyrin")),
    ("translation and ribosome", ("ribosom", "translation", "translational", "trna aminoacyl")),
    ("RNA processing", ("splic", "rna processing", "rna transport", "mrna", "rrna", "ribonucleoprotein", "nonsense-mediated")),
    ("protein folding and degradation", ("proteasom", "protein folding", "chaperon", "ubiquitin", "protein catabolic", "unfolded protein")),
    ("cell cycle and mitosis", ("cell cycle", "mitotic", "mitosis", "spindle", "chromosome segreg", "cytokinesis", "meiotic", "anaphase", "centrosom")),
    ("DNA replication and repair", ("dna replication", "dna repair", "dna damage", "recombination", "recombinational", "mismatch", "nucleotide excision")),
    ("chromatin and transcription", ("chromatin", "histone", "transcription", "nucleosome", "epigen")),
    ("energy and other metabolism", ("metaboli", "glycol", "oxidative phosphorylation", "respiratory electron", "citric acid", "atp synthesis", "biosynth", "fatty acid")),
    ("immune and signalling", ("immune", "interleukin", "interferon", "cytokine", "antigen", "signaling", "signalling", "lymphocyte", "receptor")),
    ("stress and apoptosis", ("apopt", "stress", "cell death", "autophagy", "hypoxia")),
    ("trafficking and membranes", ("vesicle", "traffick", "membrane", "endocyt", "exocyt", "golgi", "lysosom", "transport")),
]
CATEGORIES = [name for name, _ in CATEGORY_RULES] + ["other"]


def read_json(path):
    with open(path) as handle:
        return json.load(handle)


def write_json(path, payload):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)


def annotation_terms(feature, sources=REAL_SOURCES, alpha=0.05):
    """Keep significant named ontology terms; never treat STRING pair evidence as a term."""
    if "significant_term_ids" in feature:
        complete = {tuple(term.split("::", 1)) for term in feature["significant_term_ids"] if "::" in term}
        return {term for term in complete if term[0] in sources}
    return {(term["source"], term["term"]) for term in feature.get("terms", [])
            if term.get("source") in sources and term.get("q", 1) < alpha}


def load_annotations(directory, sources=REAL_SOURCES):
    tables = {}
    for path in sorted(Path(directory).glob("ann_v2_L*.json")):
        layer = int(path.stem.split("_L")[-1])
        rows = read_json(path)["features"]
        if any("significant_term_ids" not in row for row in rows):
            raise ValueError(f"{path} lacks complete significant_term_ids; rerun annotate_v2.py scorer v3 before using circuit/category statistics")
        tables[layer] = {int(row["feature_idx"]): annotation_terms(row, sources) for row in rows}
    if not tables:
        raise ValueError(f"No ann_v2_L*.json files found in {directory}")
    return tables


def cluster_bootstrap(clusters, n_boot=2000, seed=0):
    """Rows are source-level [shared count, eligible edge count, expected shared count]."""
    clusters = np.asarray(clusters, dtype=float)
    if clusters.ndim != 2 or clusters.shape[1] != 3 or not len(clusters):
        raise ValueError("Need at least one source cluster with three count columns")
    rng = np.random.default_rng(seed)
    aggregate = clusters.sum(0)
    if aggregate[1] <= 0:
        raise ValueError("Source clusters contain no eligible edges")
    observed, baseline = aggregate[[0, 2]] / aggregate[1]
    values = []
    for _ in range(n_boot):
        draw = clusters[rng.integers(len(clusters), size=len(clusters))].sum(0)
        if draw[1] <= 0:
            continue  # The shared-term fraction is undefined in all-empty resamples.
        values.append([draw[0] / draw[1], draw[2] / draw[1], (draw[0] - draw[2]) / draw[1]])
    values = np.asarray(values)
    return {key: {"estimate": float(point), "ci95": np.quantile(values[:, i], [0.025, 0.975]).tolist()}
            for i, (key, point) in enumerate(zip(("observed", "baseline", "excess"),
                                                (observed, baseline, observed - baseline)))}


def circuit_baseline(edges, terms, n_boot=2000, seed=0):
    """Compute exact random-target expectations, then cluster-bootstrap their excess.

    For each eligible edge, its source stays fixed and its target is drawn uniformly from
    ontology-annotated features in that target layer. The second baseline draws both endpoints
    independently from annotated features in their respective layers. Both retain layer pairs.
    """
    pools = {layer: {idx: value for idx, value in table.items() if value} for layer, table in terms.items()}
    inverted = {}
    for layer, table in pools.items():
        inverted[layer] = defaultdict(set)
        for idx, value in table.items():
            for term in value:
                inverted[layer][term].add(idx)
    cache = {}

    def probability(source_terms, target_layer):
        key = (frozenset(source_terms), target_layer)
        if key not in cache:
            matching = set()
            for term in source_terms:
                matching.update(inverted[target_layer].get(term, ()))
            cache[key] = len(matching) / len(pools[target_layer])
        return cache[key]

    clusters = {(int(edge["source_layer"]), int(edge["source_feature"])): np.zeros(3) for edge in edges}
    fully_cache, layer_counts = {}, Counter()
    eligible, examples = 0, []
    for edge in edges:
        sl, sf, tl, tf = (int(edge[k]) for k in ("source_layer", "source_feature", "target_layer", "target_feature"))
        source, target = pools.get(sl, {}).get(sf), pools.get(tl, {}).get(tf)
        if not source or not target:
            continue
        eligible += 1
        shared = bool(source & target)
        expectation = probability(source, tl)
        cluster = clusters.setdefault((sl, sf), np.zeros(3))
        cluster += [shared, 1, expectation]
        layer_counts[(sl, tl)] += 1
        if shared:
            examples.append({**edge, "shared_terms": sorted([list(x) for x in source & target])})
    if not eligible:
        raise ValueError("No edges have significant ontology annotations at both endpoints")
    for sl, tl in layer_counts:
        fully_cache[(sl, tl)] = float(np.mean([probability(value, tl) for value in pools[sl].values()]))
    counts = np.asarray(list(clusters.values()))
    bootstrap = cluster_bootstrap(counts, n_boot, seed)
    n_shared, _, n_expected = counts.sum(0)
    examples.sort(key=lambda row: -abs(row.get("cohens_d", 0)))
    return {"n_edges": len(edges), "n_annotated_edges": eligible, "n_shared": int(n_shared),
            "n_source_clusters": len(clusters), "observed": float(n_shared / eligible),
            "same_source_random_target": float(n_expected / eligible),
            "fully_random_annotated_pair": sum(layer_counts[k] * value for k, value in fully_cache.items()) / eligible,
            "excess": float((n_shared - n_expected) / eligible), "source_bootstrap": bootstrap,
            "source_counts": [{"source_layer": sl, "source_feature": sf, "shared": int(v[0]),
                               "eligible": int(v[1]), "expected_shared": float(v[2])} for (sl, sf), v in clusters.items()],
            "strong_shared_examples": examples[:30], "n_boot": n_boot, "seed": seed,
            "definition": "Complete GO/KEGG/Reactome/TRRUST-regulon IDs with q<0.05; excludes pair/membership evidence; exact uniform annotated-target expectation; source-cluster percentile bootstrap"}


def edge_mask(effects, consistency, n_active, d_threshold=0.5, consistency_threshold=0.7):
    return ((np.abs(effects) > d_threshold) & (consistency > consistency_threshold) & (np.asarray(n_active) >= 2))


def compare_edges(first, second):
    """Compare retained edge sets; do not interpret missing capped edges as measured nulls."""
    def index(rows):
        return {tuple(int(row[k]) for k in ("source_layer", "source_feature", "target_layer", "target_feature")): row["cohens_d"] for row in rows}
    a, b = index(first), index(second)
    shared, union = sorted(a.keys() & b.keys()), a.keys() | b.keys()
    correlation = None
    if len(shared) > 1:
        x, y = [a[key] for key in shared], [b[key] for key in shared]
        if np.std(x) > 0 and np.std(y) > 0:
            correlation = float(np.corrcoef(x, y)[0, 1])
    return {"n_first": len(a), "n_second": len(b), "n_shared": len(shared),
            "jaccard": len(shared) / len(union) if union else None,
            "fraction_first_recovered": len(shared) / len(a) if a else None,
            "effect_pearson_shared_edges": correlation,
            "note": "Correlation conditions on retained edges in both runs; capped input graphs omit measured effects"}


def best_decoder_matches(first, second, block_size=512):
    first = np.asarray(first, dtype=np.float32)
    second = np.asarray(second, dtype=np.float32)
    first = first / np.maximum(np.linalg.norm(first, axis=1, keepdims=True), 1e-12)
    second = second / np.maximum(np.linalg.norm(second, axis=1, keepdims=True), 1e-12)
    targets, cosines = np.zeros(len(first), dtype=np.int64), np.zeros(len(first))
    for start in range(0, len(first), block_size):
        similarity = np.abs(first[start:start + block_size] @ second.T)
        best = similarity.argmax(1)
        targets[start:start + len(best)] = best
        cosines[start:start + len(best)] = similarity[np.arange(len(best)), best]
    return targets, cosines


def load_edges(path):
    path = Path(path)
    if path.suffix == ".json":
        payload = read_json(path)
        return payload["edges"], {"source": str(path), "capped_graph": True}
    edges = []
    for filename in sorted(path.glob("circuit_L*.npz")):
        sl = int(filename.stem.split("_L")[-1])
        with np.load(filename) as arrays:
            # Preserve the decision made in float64 before statistics were stored in float32.
            mask = arrays["edge_mask"] if "edge_mask" in arrays else edge_mask(
                arrays["d_all"], arrays["cons_all"], arrays["n_cells_active"][:, None, None])
            for fi, li, target in zip(*np.where(mask)):
                edges.append({"source_layer": sl, "source_feature": int(arrays["features"][fi]),
                              "target_layer": int(arrays["downstream_layers"][li]), "target_feature": int(target),
                              "cohens_d": float(arrays["d_all"][fi, li, target])})
    if not edges:
        raise ValueError(f"No qualifying circuit edges found at {path}")
    return edges, {"source": str(path), "capped_graph": False}


def edge_rule_power(n_cells, n_sim=20000, seed=0):
    """Normal-delta model for the *joint* effect and sign-consistency edge rule.

    This is a sensitivity calculation conditional on a source being active in every cell.
    It is not a corrected significance test or a power claim for non-Gaussian biological deltas.
    """
    rng = np.random.default_rng(seed)
    noise = rng.normal(size=(n_sim, n_cells))
    means, deviations = noise.mean(1), noise.std(1, ddof=1)
    curve = []
    for effect in np.arange(0, 1.525, 0.025):
        signs = (noise + effect > 0).mean(1)
        called = edge_mask((means + effect) / deviations, np.maximum(signs, 1 - signs), n_cells)
        curve.append({"population_standardized_effect": float(round(effect, 3)), "call_probability": float(called.mean())})
    detectable = next((row["population_standardized_effect"] for row in curve if row["call_probability"] >= 0.8), None)
    return {"n_active_cells": n_cells, "n_sim": n_sim, "seed": seed, "curve": curve,
            "null_call_probability": curve[0]["call_probability"], "minimum_grid_effect_at_80pct": detectable,
            "assumption": "Independent normal per-cell deltas; |sample mean/sample SD|>0.5 and majority sign fraction>0.7"}


def category_for_term(term):
    label = term.lower()
    return next((name for name, patterns in CATEGORY_RULES if any(p in label for p in patterns)), "other")


def persistence_categories(first, last, source_ids, target_ids, cosines, threshold, n_boot, seed):
    chosen = [(first.get(int(s), set()), last.get(int(t), set()))
              for s, t, cosine in zip(source_ids, target_ids, cosines)
              if cosine >= threshold and first.get(int(s)) and last.get(int(t))]
    if not chosen:
        return {"n_matched_annotated_pairs": 0}
    target_pool = [value for value in last.values() if value]
    pairs = np.asarray([[bool(a & b), np.mean([bool(a & c) for c in target_pool])] for a, b in chosen])
    rng = np.random.default_rng(seed)
    boots = [pairs[rng.integers(len(pairs), size=len(pairs))].mean(0) for _ in range(n_boot)]
    boots = np.asarray(boots)
    return {"n_matched_annotated_pairs": len(pairs), "observed_same_category": float(pairs[:, 0].mean()),
            "random_target_baseline": float(pairs[:, 1].mean()),
            "excess": float(np.diff(pairs.mean(0))[0] * -1),
            "excess_ci95": np.quantile(boots[:, 0] - boots[:, 1], [0.025, 0.975]).tolist(),
            "cosine_threshold": threshold, "note": "Best decoder matches can reuse targets; descriptive source bootstrap"}


def dimension_metrics(vectors):
    """PR of centered covariance; TwoNN MLE in the original Euclidean coordinates."""
    from sklearn import config_context
    from sklearn.neighbors import NearestNeighbors
    vectors = np.asarray(vectors, dtype=np.float64)
    if vectors.ndim != 2 or len(vectors) < 4 or vectors.shape[1] < 2 or not np.isfinite(vectors).all():
        raise ValueError("Need a finite matrix with at least four rows and two columns")
    centered = vectors - vectors.mean(0)
    covariance = centered.T @ centered if vectors.shape[1] <= len(vectors) else centered @ centered.T
    variance = float(np.trace(covariance))
    pr = variance ** 2 / float(np.square(covariance).sum()) if variance else 0.0
    with config_context(working_memory=128):
        distances = NearestNeighbors(n_neighbors=3, algorithm="brute", n_jobs=1).fit(vectors).kneighbors(return_distance=True)[0]
    # kneighbors(X=None) already excludes each query's self, so use the first two columns.
    first, second = distances[:, 0], distances[:, 1]
    valid = (first > 1e-12) & (second > first)
    logs = np.log(second[valid] / first[valid])
    return {"n_genes": len(vectors), "n_coordinates": vectors.shape[1], "participation_ratio": pr,
            "twonn_id": float(len(logs) / logs.sum()) if len(logs) and logs.sum() > 0 else None,
            "twonn_valid_rows": int(valid.sum()), "twonn_excluded_rows": int((~valid).sum())}


def cmd_circuits(args):
    edges, provenance = load_edges(args.circuits)
    summary = circuit_baseline(edges, load_annotations(args.annotations), args.n_boot, args.seed)
    summary["inputs"] = {**provenance, "annotations": args.annotations}
    summary["edge_rule_sensitivity"] = [edge_rule_power(n, args.n_sim, args.seed) for n in (50, 100)]
    totals = Counter((row["source_layer"], row["source_feature"]) for row in edges)
    rng = np.random.default_rng(args.seed)
    counts = np.asarray(list(totals.values()))
    draws = [int(counts[rng.integers(len(counts), size=len(counts))].sum()) for _ in range(args.n_boot)]
    summary["total_edge_count_source_bootstrap_ci95"] = np.quantile(draws, [0.025, 0.975]).tolist()
    if args.compare:
        comparison, other_provenance = load_edges(args.compare)
        summary["comparison"] = {**compare_edges(edges, comparison), "other_inputs": other_provenance}
    write_json(args.out, summary)


def cmd_biology(args):
    layers, categories_by_layer = [], {}
    for path in sorted(Path(args.annotations).glob("ann_v2_L*.json")):
        layer = int(path.stem.split("_L")[-1])
        annotation = read_json(path)
        catalogue = {row["feature_idx"]: row for row in read_json(Path(args.run) / f"feature_catalog_L{layer:02d}.json")["features"]}
        counts, examples, by_feature = Counter(), {}, {}
        for row in annotation["features"]:
            if "significant_term_ids" not in row:
                raise ValueError(f"{path} lacks complete significant_term_ids; rerun annotation with scorer v3")
            sig = [term for term in row.get("terms", []) if term.get("source") in REAL_SOURCES and term.get("q", 1) < 0.05]
            categories = {category_for_term(term) for _, term in annotation_terms(row)}
            by_feature[int(row["feature_idx"])] = categories
            counts.update(categories)
            for category in categories:
                displayed = [term for term in sig if category_for_term(term["term"]) == category]
                if not displayed:
                    continue  # Complete IDs determine counts; examples need saved q/fold details.
                best = min(displayed, key=lambda term: term["q"])
                example = {"feature_idx": row["feature_idx"], "term": best,
                           "top_genes": catalogue.get(row["feature_idx"], {}).get("top_genes", [])[:10]}
                if category not in examples or best["q"] < examples[category]["term"]["q"]:
                    examples[category] = example
        n_annotated = sum(bool(value) for value in by_feature.values())
        modules_path = Path(args.run) / f"coactivation_modules_L{layer:02d}.json"
        modules = []
        if modules_path.exists():
            for module in read_json(modules_path)["modules"]:
                mc = Counter(category for idx in module["features"] for category in by_feature.get(int(idx), ()))
                modules.append({"module_id": module["module_id"], "n_features": len(module["features"]), "category_counts": dict(mc)})
        layers.append({"layer": layer, "n_features": len(catalogue), "n_ontology_annotated": n_annotated,
                       "category_counts": dict(counts),
                       "category_share_all_features": {c: counts[c] / len(catalogue) for c in CATEGORIES},
                       "category_share_annotated_features": {c: counts[c] / max(n_annotated, 1) for c in CATEGORIES},
                       "examples": examples, "modules": modules})
        categories_by_layer[layer] = by_feature
    summary = {"layers": layers, "categories": CATEGORIES, "category_rules": CATEGORY_RULES,
               "definition": "Complete significant ontology/regulon IDs assign categories; network pair and TF membership tests excluded; overlapping feature shares need not sum to one; examples use displayed terms with saved q/fold details",
               "inputs": {"annotations": args.annotations, "catalogues": args.run}}
    if args.sae_from and args.sae_to:
        import torch
        def directions(path):
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            return checkpoint["model_state_dict"]["W_dec.weight"].numpy().T.copy()
        targets, cosines = best_decoder_matches(directions(args.sae_from), directions(args.sae_to))
        matches_path = args.out + ".matches.npz"
        np.savez_compressed(matches_path, source_ids=np.arange(len(targets)), target_ids=targets, cosines=cosines)
        args.matches = matches_path
        summary["inputs"].update({"sae_from": args.sae_from, "sae_to": args.sae_to, "matches": matches_path})
    if args.matches:
        with np.load(args.matches) as matches:
            summary["persistent_categories"] = persistence_categories(categories_by_layer[args.from_layer], categories_by_layer[args.to_layer],
                matches["source_ids"], matches["target_ids"], matches["cosines"], 0.7, args.n_boot, args.seed)
    write_json(args.out, summary)


def choose_cell_indices(n_observations, n_cells, rng, explicit_path=None):
    """An explicit design overrides sampling, retaining its exact cell order and population."""
    if explicit_path:
        values = read_json(explicit_path)
        if not isinstance(values, list) or not values or any(type(value) is not int for value in values):
            raise ValueError('--cell-indices must contain a nonempty JSON list of integer row indices')
        indices = np.asarray(values, dtype=np.int64)
        if len(set(values))!=len(values) or (indices<0).any() or (indices>=n_observations).any():
            raise ValueError('--cell-indices contains duplicate or out-of-bounds rows')
        return indices
    return np.sort(rng.choice(n_observations, min(n_cells, n_observations), replace=False))


def cmd_dimensions(args):
    import anndata
    import scipy.sparse as sp
    adata = anndata.read_h5ad(args.h5ad, backed="r")
    rng = np.random.default_rng(args.seed)
    indices = choose_cell_indices(adata.n_obs, args.n_cells, rng, args.cell_indices)
    # Backed h5ad reads require sorted rows; restore an explicitly supplied order afterwards.
    order = np.argsort(indices)
    subset = adata[indices[order]].to_memory()
    adata.file.close()
    counts = subset.layers[args.counts_layer] if args.counts_layer else subset.X
    expression = counts.toarray().astype(np.float32) if sp.issparse(counts) else np.asarray(counts, dtype=np.float32)
    expression = expression[np.argsort(order)]
    expression = np.log1p(expression / np.maximum(expression.sum(1, keepdims=True), 1) * 1e4)
    names = np.asarray(subset.var_names).astype(str)
    model = np.load(args.model_vectors) if args.model_vectors else None
    if model is not None:
        # Preserve the model gene order and use exactly the same genes in all expression controls.
        lookup = {gene: i for i, gene in enumerate(names)}
        model_rows = np.asarray([i for i, gene in enumerate(model["genes"]) if str(gene) in lookup])
        gene_rows = np.asarray([lookup[str(model["genes"][i])] for i in model_rows])
    else:
        gene_rows = np.where(expression.sum(0) > 0)[0]
        model_rows = None
    if args.gene_cap and len(gene_rows) > args.gene_cap:
        selected = np.sort(rng.choice(len(gene_rows), args.gene_cap, replace=False))
        gene_rows = gene_rows[selected]
        if model_rows is not None:
            model_rows = model_rows[selected]
    vectors = expression[:, gene_rows].T
    del expression
    rows = [{"representation": "expression_log1p_CP10k", **dimension_metrics(vectors)}]
    shuffled = vectors.copy()
    for row in shuffled:
        rng.shuffle(row)
    rows.append({"representation": "expression_cells_shuffled_within_gene", **dimension_metrics(shuffled)})
    del shuffled
    rows.append({"representation": "expression_shape_iid_normal", **dimension_metrics(rng.normal(size=vectors.shape))})
    if model is not None:
        for layer in args.layers:
            values = model[f"L{layer:02d}"][model_rows]
            rows.append({"representation": f"model_L{layer:02d}", **dimension_metrics(values)})
            rows.append({"representation": f"model_L{layer:02d}_shape_iid_normal", **dimension_metrics(rng.normal(size=values.shape))})
    write_json(args.out, {"rows": rows, "cell_indices": indices.tolist(), "genes": names[gene_rows].tolist(),
                         "cell_selection": "explicit matching extraction design" if args.cell_indices else "seeded sample",
                         "inputs": vars(args), "definition": "Rows are genes; PR centers coordinate columns; TwoNN Euclidean MLE excludes zero/tied nearest distances; controls keep representation shape"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("circuits", "biology", "dimensions"):
        command = commands.add_parser(name)
        command.add_argument("--out", required=True)
        command.add_argument("--seed", type=int, default=0)
        if name in ("circuits", "biology"):
            command.add_argument("--annotations", required=True)
            command.add_argument("--n-boot", type=int, default=2000)
    circuits = commands.choices["circuits"]
    circuits.add_argument("--circuits", required=True, help="Original capped circuit_graph.json or directory of complete circuit_L*.npz")
    circuits.add_argument("--n-sim", type=int, default=20000)
    circuits.add_argument("--compare", help="Optional second graph JSON or circuit NPZ directory")
    biology = commands.choices["biology"]
    biology.add_argument("--run", required=True)
    biology.add_argument("--matches", help="Optional NPZ with source_ids,target_ids,cosines from actual decoder matching")
    biology.add_argument("--sae-from", help="Optional source checkpoint; supply with --sae-to to compute actual matches")
    biology.add_argument("--sae-to", help="Optional target checkpoint")
    biology.add_argument("--from-layer", type=int, default=0)
    biology.add_argument("--to-layer", type=int, default=25)
    dimensions = commands.choices["dimensions"]
    dimensions.add_argument("--h5ad", required=True)
    dimensions.add_argument("--counts-layer", default="")
    dimensions.add_argument("--model-vectors", help="NPZ from rev_misc.py gene-means, keys genes and Lxx")
    dimensions.add_argument("--layers", type=int, nargs="+", default=[9, 13, 17])
    dimensions.add_argument("--n-cells", type=int, default=4000)
    dimensions.add_argument("--cell-indices", help="JSON list of exact h5ad cell indices used for model extraction; overrides --n-cells")
    dimensions.add_argument("--gene-cap", type=int, default=0)
    args = parser.parse_args()
    if args.command == "biology" and bool(args.sae_from) != bool(args.sae_to):
        parser.error("--sae-from and --sae-to must be supplied together")
    {"circuits": cmd_circuits, "biology": cmd_biology, "dimensions": cmd_dimensions}[args.command](args)


if __name__ == "__main__":
    main()
