"""
2B vs 27B SAE-atlas scaling comparison for C2S-Scale.

Extracts matched per-layer metrics from the local atlas outputs (no GPU / model needed):
  - SAE quality:   variance_explained, dead_frac      (sae/train_L{L}.json)
  - annotation:    annotation_rate                    (feature_annotations_L{L}.json summary)
  - superposition: n_svd_aligned / n_alive, var_top50 (feature_catalog_L{L}.json summary)
  - modules:       n_modules, module_coverage         (coactivation_modules_L{L}.json summary)

Compares only layers present for BOTH models per metric. Rates (not absolute counts) are used
because d_SAE differs (2B 9216 vs 27B 18432). Writes runs/scaling_comparison.{json,md}.
"""

from __future__ import annotations

import json
import os
from statistics import mean

RUNS = "/Volumes/Crucial X6/MacBook/Code/biomi_automation/projects/c2s-scale/runs"
MODELS = {"2B": ("2b_atlas", 2304, 9216, 26), "27B": ("27b_atlas", 4608, 18432, 46)}
LAYERS = range(26)  # matched range (2B has 26 layers total; 27B L0-25)


def _load(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def collect(model):
    d, dmodel, dsae, nlayers = MODELS[model]
    run = os.path.join(RUNS, d)
    rows = {}
    for L in LAYERS:
        LL = f"{L:02d}"
        r = {}
        t = _load(os.path.join(run, "sae", f"train_L{LL}.json"))
        if t:
            r["var_explained"] = t.get("variance_explained")
            r["dead_frac"] = t.get("dead_frac")
            r["dec_cos"] = t.get("mean_decoder_cos")
        a = _load(os.path.join(run, f"feature_annotations_L{LL}.json"))
        if a and "summary" in a:
            r["annotation_rate"] = a["summary"].get("annotation_rate")
        c = _load(os.path.join(run, f"feature_catalog_L{LL}.json"))
        if c and "summary" in c:
            s = c["summary"]
            alive = s.get("n_alive") or 1
            r["svd_aligned_frac"] = (s.get("n_svd_aligned") or 0) / alive
            r["var_top50_svd"] = (s.get("svd_info") or {}).get("variance_explained_top50")
        m = _load(os.path.join(run, f"coactivation_modules_L{LL}.json"))
        if m and "summary" in m:
            r["n_modules"] = m["summary"].get("n_modules")
            r["module_coverage"] = m["summary"].get("module_coverage")
        if r:
            rows[L] = r
    return {"d_model": dmodel, "d_sae": dsae, "layers": rows}


def _series(data, model, key):
    return {L: r[key] for L, r in data[model]["layers"].items() if r.get(key) is not None}


def _fmt(x, p=3):
    return f"{x:.{p}f}" if isinstance(x, (int, float)) else "—"


def main():
    data = {m: collect(m) for m in MODELS}

    metrics = [
        ("var_explained", "SAE variance explained", 3),
        ("dead_frac", "dead-feature fraction", 3),
        ("dec_cos", "mean |decoder cos|", 4),
        ("annotation_rate", "ontology annotation rate", 3),
        ("svd_aligned_frac", "SVD-aligned fraction (superposition)", 4),
        ("var_top50_svd", "variance in top-50 SVD axes", 3),
        ("n_modules", "co-activation modules/layer", 1),
        ("module_coverage", "module coverage", 3),
    ]

    lines = []
    lines.append("# C2S-Scale SAE atlas — 2B vs 27B scaling comparison\n")
    lines.append(f"- 2B: d_model 2304, d_SAE 9216 (4x), k=32 | 27B: d_model 4608, d_SAE 18432 (4x), k=32")
    lines.append(f"- Matched layers L0-25. Rates compared (d_SAE differs). 27B late layers (26-45) not run.\n")

    summary = {}
    lines.append("## Per-metric summary (mean over layers present in BOTH models)\n")
    lines.append("| metric | 2B (n) | 27B (n) | Δ (27B−2B) |")
    lines.append("|---|---|---|---|")
    for key, label, p in metrics:
        s2, s27 = _series(data, "2B", key), _series(data, "27B", key)
        common = sorted(set(s2) & set(s27))
        if not common:
            lines.append(f"| {label} | — | — | (no overlap) |")
            continue
        m2 = mean(s2[L] for L in common)
        m27 = mean(s27[L] for L in common)
        summary[key] = {"n_common": len(common), "mean_2B": m2, "mean_27B": m27, "delta": m27 - m2}
        lines.append(f"| {label} | {_fmt(m2, p)} ({len(common)}) | {_fmt(m27, p)} ({len(common)}) | {_fmt(m27 - m2, p)} |")

    # depth trend of dead features (both, all 26 layers)
    lines.append("\n## Depth trend — dead-feature fraction by layer (all 26 layers)\n")
    lines.append("| layer | 2B var | 2B dead | 27B var | 27B dead |")
    lines.append("|---|---|---|---|---|")
    for L in LAYERS:
        r2 = data["2B"]["layers"].get(L, {})
        r27 = data["27B"]["layers"].get(L, {})
        lines.append(f"| L{L:02d} | {_fmt(r2.get('var_explained'))} | {_fmt(r2.get('dead_frac'))} "
                     f"| {_fmt(r27.get('var_explained'))} | {_fmt(r27.get('dead_frac'))} |")

    report = "\n".join(lines)
    with open(os.path.join(RUNS, "scaling_comparison.md"), "w") as f:
        f.write(report + "\n")
    with open(os.path.join(RUNS, "scaling_comparison.json"), "w") as f:
        json.dump({"models": {m: {"d_model": data[m]["d_model"], "d_sae": data[m]["d_sae"],
                                  "n_layers_with_data": len(data[m]["layers"])} for m in MODELS},
                   "metric_summary": summary}, f, indent=2)
    print(report)
    print(f"\nsaved -> {RUNS}/scaling_comparison.md + .json")


if __name__ == "__main__":
    main()
