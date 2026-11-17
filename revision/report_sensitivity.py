"""Assemble a source-hashed sensitivity report from saved JSON, without rescoring.

Run on the pod with --root /workspace/rev1. This reads one catalogue and one
annotation file at a time and writes only its own reporting artifact.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import resource
import time

ONTOLOGY = {"GO_BP", "KEGG", "Reactome", "TRRUST"}
NETWORK = {"STRING", "TRRUST_pairs", "TF_membership"}
FIRING_KEYS = ("dead_all_rows", "dead_frac_all_rows", "mean_active_per_position",
    "firing_freq_quantiles_1_10_50_90_99", "share_features_firing_gt_10pct",
    "share_features_firing_lt_1e-4", "fraction_variance_unexplained_all_rows")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(actual, expected, label):
    require(math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-12),
            f"{label}: saved {actual} differs from recount {expected}")


def indexed(rows, label):
    result = {int(row["feature_idx"]): row for row in rows}
    require(len(result) == len(rows), f"duplicate feature IDs in {label}")
    return result


def annotation_report(catalogue, annotation):
    """Recount saved decisions and rescale active-set floors to all features."""
    cat = indexed(catalogue["features"], "catalogue")
    active = {key for key, row in cat.items() if not row["is_dead"]}
    rows = indexed(annotation["features"], "annotation")
    summary = annotation["summary"]
    total, alive = len(cat), len(active)
    require(total > 0 and alive > 0, "empty dictionary or active feature set")
    require(set(rows) == active, "annotation IDs must equal catalogue-active IDs")
    for key, expected in (("n_features_total", total), ("n_alive", alive), ("n_sets", alive)):
        require(summary[key] == expected, f"{key} disagrees with catalogue recount")
    for key, expected in (("n_features", total), ("n_alive", alive), ("n_dead", total - alive)):
        require(catalogue["summary"][key] == expected, f"catalogue {key} disagrees with rows")
    require(summary["scorer_version"] == "3-plan-aligned", "full-family v3 annotations required")
    require(summary["tests_per_feature"] == sum(summary["terms_by_source"].values()) + 3,
            "test-family count must include three separate network/membership tests")
    n_any = n_ontology = n_network = n_network_only = 0
    source_counts = {source: 0 for source in ONTOLOGY | NETWORK}
    for feature in rows.values():
        ids = feature["significant_term_ids"]
        require(len(ids) == len(set(ids)), "duplicate complete significant IDs")
        significant = [term for term in feature.get("terms", []) if term["q"] < summary["alpha"]]
        sources = {term["source"] for term in significant}
        require(sources <= ONTOLOGY | NETWORK, "unknown significant source")
        numeric_ids = {f'{term["source"]}::{term["term"]}' for term in significant
                       if term["source"] in ONTOLOGY}
        require(set(ids) == numeric_ids, "complete significant IDs lack exact numeric details")
        for term in significant:
            require(math.isfinite(term["q"]), "nonfinite significant q")
            if term["source"] in ONTOLOGY:
                require(all(math.isfinite(term[key]) for key in ("p", "fold")),
                        "missing/nonfinite significant ontology numeric details")
        require(feature["n_sig"] == len(significant), "n_sig disagrees with complete numeric details")
        require(bool(feature["annotated"]) == bool(significant), "annotated flag disagrees with terms")
        any_hit, ontology_hit = bool(significant), bool(ids)
        network_hit = bool(sources & NETWORK)
        n_any += any_hit
        n_ontology += ontology_hit
        n_network += network_hit
        n_network_only += any_hit and not ontology_hit
        for source in sources:
            source_counts[source] += 1
    for key, expected in (("n_annotated", n_any), ("n_ontology_annotated", n_ontology),
                          ("n_network_or_membership_only", n_network_only)):
        require(summary[key] == expected, f"{key} disagrees with annotation rows")
    for key, expected in (("annotation_rate", n_any / alive),
        ("annotation_rate_all_features", n_any / total),
        ("ontology_annotation_rate_alive", n_ontology / alive),
        ("ontology_annotation_rate_all_features", n_ontology / total)):
        close(summary[key], expected, key)
    for source, rate in summary.get("rate_by_source", {}).items():
        close(rate, source_counts[source] / alive, "rate_by_source:" + source)
    floors = {}
    for kind in ("uniform", "freq"):
        saved = summary["floors"][kind]
        require(len(saved["values"]) == len(saved["ontology_values"]) > 0, "missing floor replicates")
        close(saved["mean"], sum(saved["values"]) / len(saved["values"]), "floor mean")
        close(saved["ontology_mean"], sum(saved["ontology_values"]) / len(saved["ontology_values"]), "ontology floor mean")
        active_floor = {"any_mean": saved["mean"], "any_values": saved["values"],
                        "ontology_mean": saved["ontology_mean"], "ontology_values": saved["ontology_values"]}
        all_floor = {key: ([v * alive / total for v in value] if isinstance(value, list)
                         else value * alive / total) for key, value in active_floor.items()}
        floors[kind] = {"n_seeds": len(saved["values"]), "catalogue_active": active_floor,
                        "all_features": all_floor}
    return {"n_features": total, "n_catalogue_active": alive, "n_catalogue_inactive": total - alive,
        "n_testable": summary["n_testable"], "n_any": n_any, "n_ontology": n_ontology,
        "n_network_evidence": n_network, "n_network_only": n_network_only,
        "source_counts": source_counts,
        "rates_all_features": {"any": n_any / total, "ontology": n_ontology / total},
        "rates_catalogue_active": {"any": n_any / alive, "ontology": n_ontology / alive},
        "floors": floors, "configuration": {key: summary.get(key) for key in (
            "scorer_version", "legacy_libraries", "universe_size", "tests_per_feature", "terms_by_source",
            "string_pairs_in_universe", "trrust_pairs_in_universe", "pair_null", "alpha",
            "min_term", "max_term", "min_overlap", "median_set_size")},
        "verification": "exact saved-ID/decision/numeric-detail/count/rate recount; no rescoring"}


def metric_report(training, evaluation, baseline=False):
    if baseline:
        require(evaluation is not None, "baseline requires fresh 2b_eval_LXX.json")
        variance = evaluation["variance_explained_reextracted"]
        dead = evaluation["dead_val_reextracted"]
        source, firing = "fresh_2b_evaluation", evaluation
    else:
        variance, dead = training["variance_explained"], training["dead_features"]
        source, firing = "new_variant_training_metrics", training
    return {"variance_explained": variance, "heldout_inactive": dead,
        "evaluation_source": source, "mean_decoder_cos": training["mean_decoder_cos"],
        "decoder_cos_source": "checkpoint-training JSON; fixed weights; mean absolute off-diagonal cosine",
        "firing": {key: firing[key] for key in FIRING_KEYS if key in firing},
        "training_configuration": {key: training.get(key) for key in (
            "d_model", "n_features", "k", "epochs", "seed", "lr", "batch_size", "n_rows")}}


class Sources:
    def __init__(self, root):
        self.root, self.files = root, {}

    def read(self, path):
        path = Path(path)
        payload = path.read_bytes()
        self.files[str(path.relative_to(self.root))] = {
            "sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)}
        return json.loads(payload)


def saved_entry(sources, name, layer, run, annotation_path, training_path=None,
                evaluation_path=None, baseline=False):
    paths = [run / f"feature_catalog_L{layer:02d}.json", annotation_path]
    if training_path:
        paths.append(training_path)
    if evaluation_path:
        paths.append(evaluation_path)
    missing = [str(path.relative_to(sources.root)) for path in paths if not path.is_file()]
    if missing:
        return {"name": name, "layer": layer, "status": "pending", "missing": missing}
    catalogue, annotation = sources.read(paths[0]), sources.read(paths[1])
    result = {"name": name, "layer": layer, "status": "complete",
        "sources": [str(path.relative_to(sources.root)) for path in paths],
        "annotation": annotation_report(catalogue, annotation)}
    if training_path:
        result["metrics"] = metric_report(sources.read(training_path),
            sources.read(evaluation_path) if evaluation_path else None, baseline=baseline)
        result["metrics"]["heldout_inactive_fraction"] = (
            result["metrics"]["heldout_inactive"] / result["annotation"]["n_features"])
        firing_dead = result["metrics"]["firing"].get("dead_all_rows")
        if firing_dead is not None:
            require(firing_dead == result["annotation"]["n_catalogue_inactive"],
                    f"{name}: metrics/catalogue all-row inactivity mismatch")
    return result


def build_report(root):
    sources = Sources(root)
    runs, results = root / "runs", root / "results"
    atlas = runs / "2b_atlas"
    sweeps = []
    for layer in (0, 13):
        for k in (8, 16, 32, 64, 128):
            baseline = k == 32
            name = f"sweep_L{layer:02d}_k{k}_x4"
            run = atlas if baseline else runs / name
            ann = results / "annotations" / ("2b" if baseline else name) / f"ann_v2_L{layer:02d}.json"
            sweeps.append(saved_entry(sources, name, layer, run, ann, run / f"sae/train_L{layer:02d}.json",
                results / f"2b_eval_L{layer:02d}.json" if baseline else None, baseline))
    for expansion in (2, 8):
        name = f"sweep_L13_k32_x{expansion}"
        run = runs / name
        sweeps.append(saved_entry(sources, name, 13, run,
            results / "annotations" / name / "ann_v2_L13.json", run / "sae/train_L13.json"))
    libraries = []
    for layer in (0, 13, 22):
        pair = {kind: saved_entry(sources, kind, layer, atlas,
            results / "annotations" / directory / f"ann_v2_L{layer:02d}.json")
            for kind, directory in (("full", "2b"), ("legacy", "2b_legacy"))}
        pair["layer"] = layer
        pair["status"] = "complete" if all(pair[k]["status"] == "complete" for k in ("full", "legacy")) else "pending"
        if pair["status"] == "complete":
            require(not pair["full"]["annotation"]["configuration"]["legacy_libraries"], "full library flag incorrect")
            require(pair["legacy"]["annotation"]["configuration"]["legacy_libraries"], "legacy library flag incorrect")
        libraries.append(pair)
    pools = [saved_entry(sources, f"pool_{kind}", 13, runs / f"pool_{kind}",
        results / f"annotations/pool_{kind}/ann_v2_L13.json") for kind in ("first", "last")]
    pool_metrics = results / "pool_comparison.json"
    pooling = {"status": "complete" if pool_metrics.is_file() and all(x["status"] == "complete" for x in pools)
               else "pending", "entries": pools,
               "comparison": sources.read(pool_metrics) if pool_metrics.is_file() else None}
    untrained = []
    for layer in (0, 13, 22):
        name = f"untrained_L{layer:02d}"
        entry = saved_entry(sources, name, layer, runs / name,
            results / "annotations" / name / f"ann_v2_L{layer:02d}.json")
        metadata = runs / name / f"sae/untrained_L{layer:02d}.json"
        if metadata.is_file():
            entry["initialization"] = sources.read(metadata)
        elif entry["status"] == "complete":
            entry.update(status="pending", missing=[str(metadata.relative_to(root))])
        untrained.append(entry)
    return {"schema": "c2s-sensitivity-report-v1", "created_utc": datetime.now(timezone.utc).isoformat(),
        "units": {"primary_annotation_denominator": "all dictionary features",
            "catalogue_active_denominator": "features firing at least once across all 614400 atlas gene-token positions",
            "heldout_inactive": "features not firing in saved held-out evaluation positions; distinct from catalogue inactivity",
            "floors": "saved rates over active gene sets, rescaled by n_catalogue_active/n_features for all-feature comparisons",
            "firing_quantiles": "1st/10th/50th/90th/99th percentiles among features active across all rows",
            "rare_firing_share": "active features with frequency <1e-4 divided by all dictionary features",
            "baseline_metrics": "fresh re-extracted 2b_eval VE/inactivity/firing; decoder cosine from fixed-checkpoint training JSON",
            "new_variant_metrics": "saved six-epoch new-variant training held-out VE/inactivity and all-row firing",
            "library_comparison": "full-family BH in both conditions; archival reduced library snapshot changes the tested term family, not the scorer"},
        "sweeps": sweeps, "n_complete_sweeps_including_baselines": sum(x["status"] == "complete" for x in sweeps),
        "library_sensitivity": libraries, "pooling": pooling, "untrained": untrained,
        "sources": sources.files,
        "limitations": ["Random-floor replicates describe the saved three seeded null draws; they are not independent trained models.",
            "Only saved decisions and metrics are reported; no annotation rescoring, checkpoint loading, or training occurs.",
            "Pooling/untrained entries remain explicitly pending until their required outputs exist; untrained reconstruction is not inferred."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    report = build_report(args.root.resolve())
    report["reporter_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    report["runtime_seconds"] = time.monotonic() - started
    report["peak_process_rss_kib_linux"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".partial")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(args.out)
    print(json.dumps({"out": str(args.out), "complete_sweeps_including_baselines": report["n_complete_sweeps_including_baselines"],
        "pooling": report["pooling"]["status"], "untrained": [x["status"] for x in report["untrained"]],
        "runtime_seconds": report["runtime_seconds"], "peak_rss_kib": report["peak_process_rss_kib_linux"]}))


if __name__ == "__main__":
    main()
