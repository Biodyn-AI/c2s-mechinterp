"""Aggregate audited saved text-name deletion retrieval, without model execution.

Macro means and TF bootstrap intervals call the existing summarize_retrieval
implementation. No gene scores, target labels, or paired statistics are refitted.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path

STATS_PATH = Path(__file__).parent / "pod/rev_phase8_stats.py"
if not STATS_PATH.is_file():
    STATS_PATH = Path(__file__).with_name("rev_phase8_stats.py")
SPEC = importlib.util.spec_from_file_location("text_deletion_saved_statistics", STATS_PATH)
STATS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(STATS)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def summarize_line(payload, bootstrap, seed):
    meta, records = payload["meta"], payload["retrieval"]
    factors, layers = meta["tfs"], meta["layers"]
    require(len(set(factors)) == len(factors) and len(set(layers)) == len(layers), "duplicate metadata TF/layer")
    keys = [(row["tf"], row["layer"]) for row in records]
    require(len(keys) == len(set(keys)), "duplicate TF-layer retrieval rows")
    require(set(keys) == {(tf, layer) for tf in factors for layer in layers}, "incomplete or foreign TF-layer grid")
    require(meta["n_pairs"] > 0, "missing TF-cell pairs")
    for key in ("max_pre_deletion_cosdist_mean", "max_pre_deletion_cosdist_max"):
        require(math.isfinite(meta[key]) and meta[key] >= 0, "missing/nonfinite pre-deletion noise")
    output = {key: meta[key] for key in ("n_pairs", "max_pre_deletion_cosdist_mean", "max_pre_deletion_cosdist_max")}
    output.update(n_screened_named_tfs=len(factors), screened_named_tfs=factors,
        n_named_gene_universe=len(meta["genes"]), noise_check_layer=layers[0], layers=[])
    for layer in layers:
        family = [row for row in records if row["layer"] == layer]
        evaluable = [row for row in family if row["average_precision"] is not None]
        for row in evaluable:
            require(0 < row["n_positive"] < row["n"], "evaluable factor requires both classes")
            for key in ("average_precision", "auroc", "prevalence", "random_label_expected_ap"):
                require(math.isfinite(row[key]), f"missing/nonfinite retrieval metric: {key}")
        summary = STATS.summarize_retrieval([{"saved": row} for row in family], "saved", bootstrap, seed)
        keep = {key: value for key, value in summary.items() if key in ("n_tfs_evaluable", "bootstrap_unit")
                or any(key in (f"mean_{metric}", f"mean_{metric}_ci95") for metric in
                       ("average_precision", "auroc", "prevalence", "random_label_expected_ap"))}
        keep.update(layer=layer, n_tfs_screened=len(family), evaluable_tfs=[row["tf"] for row in evaluable],
            evaluable_downstream_genes_min=min((row["n"] for row in evaluable), default=None),
            evaluable_downstream_genes_max=max((row["n"] for row in evaluable), default=None))
        output["layers"].append(keep)
    return output


def build_report(results, bootstrap=5000, seed=20261117):
    require(bootstrap > 0, "TF bootstrap count must be positive")
    results = Path(results)
    lines, sources = {}, {}
    for line in ("k562", "rpe1"):
        relative = f"phase8_{line}/statistics/insilico_retrieval.json"
        path = results / relative
        raw = path.read_bytes()
        sources[relative] = hashlib.sha256(raw).hexdigest()
        lines[line] = summarize_line(json.loads(raw), bootstrap, seed)
    return {"schema": "text-name-deletion-report-v1", "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(), "bootstrap": bootstrap, "seed": seed,
        "bootstrap_unit": "evaluable transcription factors; saved per-factor retrieval held fixed",
        "summary_implementation_sha256": hashlib.sha256(STATS_PATH.read_bytes()).hexdigest(),
        "reporter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_sha256": sources, "lines": lines,
        "definitions": {"score": "paired mean cosine-distance contrast: TF-name deletion minus nearby-name deletion",
            "screening": "TF names occurring in control sentences; this screened family differs from measured knockdown targets",
            "universe": "per-factor downstream genes with at least five paired observations, excluding the deleted factor itself",
            "noise": "first sampled layer: mean and maximum of per-pair maximum pre-deletion cosine distance; only deletion ranks greater than five are checked",
            "macro": "unweighted means over factors with both target and nontarget retrieval items",
            "null": "mean exact finite random-label AP expectation, preserving each saved score-tie structure"},
        "limitations": ["The downstream retrieval universes differ between cell lines and factors.",
            "The nearby-name control does not match subword token length.",
            "Deleting model text is not a biological knockout or evidence of biological regulatory recovery.",
            "TF bootstrap intervals condition on the saved per-factor cell responses; they do not quantify new-cell or training variability."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20261117)
    args = parser.parse_args()
    report = build_report(args.results, args.bootstrap, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".partial")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(args.out)
    print(json.dumps({"out": str(args.out), "status": report["status"],
        "screened": {line: value["n_screened_named_tfs"] for line, value in report["lines"].items()}}))


if __name__ == "__main__":
    main()
