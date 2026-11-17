"""Frozen K562-to-RPE1 phase readouts with full-panel sentence selection matched.

Rank positive genes across each cell's full panel, break expression ties by panel order,
take the top k, then project onto shared gene columns. This follows cell_sentences.py.
Shared-first selection is saved only as a diagnostic. Shared-column baselines still cannot
use identities absent from the source panel; selection matching is not identity equivalence.

Run substantial work on the pod. This helper never downloads data or changes cloud resources.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import time

import numpy as np


def ranked_positive_indices(row, k):
    """Exactly the positive-gene and stable-tie rule used for a cell sentence."""
    row = np.asarray(row)
    nonzero = np.flatnonzero(row > 0)
    return nonzero[np.argsort(-row[nonzero], kind="stable")[:k]]


def encode_selected_shared(expression, shared_columns, k):
    """Select full-panel genes first; retain their full-panel ranks after projection."""
    from scipy import sparse
    shared_columns = np.asarray(shared_columns, dtype=int)
    lookup = {int(column): index for index, column in enumerate(shared_columns)}
    ranks = np.zeros((expression.shape[0], len(shared_columns)), dtype=np.float64)
    magnitudes = np.zeros_like(ranks)
    n_selected, n_nonshared = [], []
    for ci in range(expression.shape[0]):
        row = expression[ci]
        row = row.toarray().ravel() if sparse.issparse(row) else np.asarray(row).ravel()
        if not np.isfinite(row).all() or np.any(row < 0):
            raise ValueError("Expression inputs must be finite and nonnegative")
        selected = ranked_positive_indices(row, k)
        n_selected.append(len(selected))
        nonshared = 0
        for rank, column in enumerate(selected):
            target = lookup.get(int(column))
            if target is None:
                nonshared += 1
            else:
                ranks[ci, target] = k - rank
                magnitudes[ci, target] = row[column]
        n_nonshared.append(nonshared)
    return ranks, magnitudes, {"n_selected": n_selected, "n_nonshared_selected": n_nonshared}


def resultant_agreement(predicted, truth):
    return float(abs(np.mean(np.exp(1j * (np.asarray(predicted) - np.asarray(truth))))))


def encode_shared_first(expression, shared_columns, k):
    """Diagnostic selection among shared genes, still breaking ties by original panel order."""
    shared_columns = np.asarray(shared_columns, dtype=int)
    panel_order = np.argsort(shared_columns, kind="stable")
    restricted = expression[:, shared_columns[panel_order]]
    # Return the final columns to the same shared-gene order used by the primary arms.
    return encode_selected_shared(restricted, np.argsort(panel_order), k)


def median_rotated_error(predicted, truth):
    residual = np.asarray(predicted) - np.asarray(truth)
    offset = np.angle(np.mean(np.exp(-1j * residual)))
    errors = np.abs(np.angle(np.exp(1j * (residual + offset))))
    return {"median_error_deg": float(np.degrees(np.median(errors))),
            "fitted_rotation_deg": float(np.degrees(offset))}


def paired_contrasts(predictions, truth, pairs, n_boot=5000, seed=0):
    """One target-cell resample per draw is shared by every arm; source fits stay fixed."""
    rng = np.random.default_rng(seed)
    residuals = {name: np.exp(1j * (values - truth)) for name, values in predictions.items()}
    point = {name: float(abs(values.mean())) for name, values in residuals.items()}
    draws = {name: np.empty(n_boot) for name in predictions}
    for draw in range(n_boot):
        indices = rng.integers(len(truth), size=len(truth))
        for name, values in residuals.items():
            draws[name][draw] = abs(values[indices].mean())
    contrasts = {}
    for first, second in pairs:
        difference = draws[first] - draws[second]
        low, high = np.quantile(difference, [0.025, 0.975])
        interpretation = "unresolved_difference"
        if low > 0:
            interpretation = "first_arm_higher"
        elif high < 0:
            interpretation = "first_arm_lower"
        contrasts[f"{first}_minus_{second}"] = {
            "point_delta": point[first] - point[second], "bootstrap_mean_delta": float(difference.mean()),
            "ci95": [float(low), float(high)], "bootstrap_fraction_positive": float((difference > 0).mean()),
            "interpretation": interpretation,
            "interpretation_limit": "An interval spanning zero is an unresolved difference, not equivalence or parity",
            "bootstrap_delta_draws": difference.tolist(),
        }
    return contrasts


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_states(directory, n_observations, layer, k):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest.get("unit") != "cell-summary" or manifest.get("max_genes") != k:
        raise ValueError(f"State manifest must describe cell-summary representations with max_genes={k}: {directory}")
    cell_ids = np.load(directory / "row_cell_ids.npy")
    states = np.load(directory / f"layer_{layer:02d}_activations.npy", mmap_mode="r")
    if states.ndim != 2 or cell_ids.ndim != 1 or len(cell_ids) != len(states):
        raise ValueError("State rows and cell identities must align exactly; truncation is not permitted")
    if not np.issubdtype(cell_ids.dtype, np.integer) or len(np.unique(cell_ids)) != len(cell_ids):
        raise ValueError("Cell-summary row identities must be unique integers")
    if len(cell_ids) < 3 or cell_ids.min() < 0 or cell_ids.max() >= n_observations:
        raise ValueError("State cell identities are empty or outside the h5ad rows")
    if not np.isfinite(states).all():
        raise ValueError("Nonfinite model states are not permitted")
    return np.asarray(states, dtype=np.float64), cell_ids, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-h5ad", type=Path, required=True)
    parser.add_argument("--target-h5ad", type=Path, required=True)
    parser.add_argument("--source-states", type=Path, required=True)
    parser.add_argument("--target-states", type=Path, required=True)
    parser.add_argument("--route-dir", type=Path, required=True, help="Directory containing the archived cc_phase.py")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=21)
    parser.add_argument("--k", type=int, default=512)
    parser.add_argument("--alpha", type=float, default=1000)
    parser.add_argument("--n-boot", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.k < 1 or args.n_boot < 1 or args.alpha <= 0:
        parser.error("--k, --n-boot and --alpha must be positive")
    started = time.monotonic()
    import anndata
    from scipy import sparse
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    phase_path = args.route_dir / "cc_phase.py"
    spec = importlib.util.spec_from_file_location("selection_transfer_phase", phase_path)
    phase_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(phase_module)
    source = anndata.read_h5ad(args.source_h5ad)
    target = anndata.read_h5ad(args.target_h5ad)
    source_states, source_ids, source_manifest = load_states(args.source_states, source.n_obs, args.layer, args.k)
    target_states, target_ids, target_manifest = load_states(args.target_states, target.n_obs, args.layer, args.k)
    if source_states.shape[1] != target_states.shape[1] or source_manifest.get("model") != target_manifest.get("model"):
        raise ValueError("Source and target states must use the same model width and checkpoint")
    source_phase, _, _, source_orientation = phase_module.phase_angle_oriented(source, seed=args.seed)
    target_phase, _, _, target_orientation = phase_module.phase_angle_oriented(target, seed=args.seed)
    source_phase, target_phase = source_phase[source_ids], target_phase[target_ids]
    source_symbols = np.char.upper(np.asarray(source.var_names).astype(str))
    target_symbols = np.char.upper(np.asarray(target.var_names).astype(str))
    if len(set(source_symbols)) != len(source_symbols) or len(set(target_symbols)) != len(target_symbols):
        raise ValueError("Gene symbols must be unique after uppercasing; resolve aliases explicitly")
    shared = sorted(set(source_symbols) & set(target_symbols))
    if len(shared) < 2:
        raise ValueError("At least two shared gene identities are required")
    source_lookup, target_lookup = ({gene: i for i, gene in enumerate(symbols)} for symbols in (source_symbols, target_symbols))
    source_columns = np.array([source_lookup[gene] for gene in shared])
    target_columns = np.array([target_lookup[gene] for gene in shared])
    source_expression, target_expression = source.X[source_ids], target.X[target_ids]

    def dense_shared(expression, columns):
        values = expression[:, columns]
        return values.toarray().astype(np.float64) if sparse.issparse(values) else np.asarray(values, dtype=np.float64)

    source_full, target_full = dense_shared(source_expression, source_columns), dense_shared(target_expression, target_columns)
    source_rank, source_mag, source_diagnostic = encode_selected_shared(source_expression, source_columns, args.k)
    target_rank, target_mag, target_diagnostic = encode_selected_shared(target_expression, target_columns, args.k)
    source_old_rank, _, _ = encode_shared_first(source_expression, source_columns, args.k)
    target_old_rank, _, _ = encode_shared_first(target_expression, target_columns, args.k)
    source_targets = np.column_stack([np.cos(source_phase), np.sin(source_phase)])

    def frozen_fit_apply(source_values, target_values):
        scaler = StandardScaler().fit(source_values)
        fitted = Ridge(alpha=args.alpha).fit(scaler.transform(source_values), source_targets)
        predicted = fitted.predict(scaler.transform(target_values))
        return np.arctan2(predicted[:, 1], predicted[:, 0])

    predictions = {
        "model": frozen_fit_apply(source_states, target_states),
        "expression_full_shared": frozen_fit_apply(source_full, target_full),
        "expression_topk_magnitude_fullpanel": frozen_fit_apply(source_mag, target_mag),
        "expression_topk_rank_fullpanel": frozen_fit_apply(source_rank, target_rank),
        "diagnostic_shared_first_rank": frozen_fit_apply(source_old_rank, target_old_rank),
        "constant": np.full_like(target_phase, np.angle(np.mean(np.exp(1j * target_phase)))),
        "random": np.random.default_rng(args.seed).uniform(0, 2 * np.pi, len(target_phase)),
    }
    points = {name: {"R_diff": resultant_agreement(values, target_phase), **median_rotated_error(values, target_phase)}
              for name, values in predictions.items()}
    rank_name, mag_name, full_name = "expression_topk_rank_fullpanel", "expression_topk_magnitude_fullpanel", "expression_full_shared"
    contrasts = paired_contrasts(predictions, target_phase, [("model", rank_name), ("model", full_name),
        (full_name, rank_name), (full_name, mag_name), (mag_name, rank_name), ("model", "diagnostic_shared_first_rank")], args.n_boot, args.seed)

    def selection_diagnostic(primary, diagnostic, counts):
        changed = np.any(primary != diagnostic, axis=1)
        return {**counts, "n_cells_differing_from_shared_first_selection": int(changed.sum()),
                "fraction_cells_differing_from_shared_first_selection": float(changed.mean()),
                "mean_selected_nonshared_genes": float(np.mean(counts["n_nonshared_selected"]))}

    inputs = {"source_h5ad": args.source_h5ad, "target_h5ad": args.target_h5ad, "phase_code": phase_path, "analysis_code": Path(__file__)}
    for label, directory in (("source", args.source_states), ("target", args.target_states)):
        for name in ("manifest.json", "row_cell_ids.npy", f"layer_{args.layer:02d}_activations.npy"):
            inputs[f"{label}_{name}"] = directory / name
    record = {
        "schema": "selection-matched-frozen-transfer-v1", "n_source": len(source_phase), "n_target": len(target_phase),
        "n_source_panel_genes": source.n_vars, "n_target_panel_genes": target.n_vars, "n_shared_genes": len(shared),
        "layer": args.layer, "k": args.k, "ridge_alpha": args.alpha, "point": points, "contrasts": contrasts,
        "primary_contrast": f"model_minus_{rank_name}", "primary_interpretation": contrasts[f"model_minus_{rank_name}"]["interpretation"],
        "selection_diagnostics": {"source": selection_diagnostic(source_rank, source_old_rank, source_diagnostic),
                                  "target": selection_diagnostic(target_rank, target_old_rank, target_diagnostic)},
        "phase_orientation": {"source": source_orientation, "target": target_orientation},
        "source_phase_concentration": float(abs(np.exp(1j * source_phase).mean())),
        "target_phase_concentration": float(abs(np.exp(1j * target_phase).mean())),
        "predicted_target_phases": {name: values.tolist() for name, values in predictions.items()},
        "source_phase": source_phase.tolist(), "target_phase": target_phase.tolist(),
        "source_cell_indices": source_ids.tolist(), "target_cell_indices": target_ids.tolist(), "shared_genes": shared,
        "definitions": {
            "selection": "positive genes from full panel, descending expression with stable panel-order ties, top k, then shared-column projection",
            "rank_weight": "k minus zero-based rank within the full-panel positive-gene list; absent shared genes get zero",
            "expression_values": "h5ad.X as provided; no second normalisation",
            "readout": "source-only StandardScaler and Ridge of cosine/sine phase, frozen on target; all fits use float64",
            "diagnostic_shared_first_rank": "shared genes restricted before top-k selection, retaining original panel-order tie breaks; excluded from primary comparison",
            "bootstrap": "paired target-cell percentile intervals with fixed source fits; not source-sampling or cross-experiment uncertainty",
            "bootstrap_fraction_positive": "descriptive bootstrap proportion, not a calibrated hypothesis-test p-value",
            "R_diff": "absolute mean complex angular residual, invariant to a global rotation; not circular R-squared",
            "constant_floor": "a constant phase predictor; R_diff equals measured target concentration",
            "identity_coverage_limit": "expression arms use shared gene columns only; the model may also use nonshared gene identities",
            "null_interval": "CI spanning zero means unresolved difference; no equivalence margin or equivalence test is specified",
        },
        "n_boot": args.n_boot, "seed": args.seed, "seconds": time.monotonic() - started,
        "inputs": {name: {"file": path.name, "sha256": file_sha256(path)} for name, path in inputs.items()},
        "software": {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "scikit-learn", "anndata")},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".pending")
    temporary.write_text(json.dumps(record, indent=2, allow_nan=False))
    temporary.replace(args.out)
    print(json.dumps({"primary_contrast": record["primary_contrast"], "point": points,
                      "primary_ci95": contrasts[record["primary_contrast"]]["ci95"],
                      "interpretation": record["primary_interpretation"], "out": str(args.out)}, indent=2))


if __name__ == "__main__":
    main()
