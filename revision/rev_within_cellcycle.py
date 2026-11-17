"""Recover the within-K562 input ladder using the archived cell-cycle ruler, on the pod.

  python rev_within_cellcycle.py --h5ad data/canonical_k562.h5ad --states data/act_k562 \
      --route-dir route_genemanifold_c2s --out results/within_k562_cellcycle.json

Positive genes are ranked over the FULL gene panel with stable panel-order ties before top-512
selection. Full expression, selected magnitudes, binary membership, selected ranks and L21 states
use the same cells. As in archived cc_benchmark_c2s.pca_k, PCA50 and StandardScaler are fitted on
the whole cohort before five-fold cosine/sine Ridge(alpha=10). This preserves the archived ruler;
it is not a protocol with preprocessing fitted separately in each training fold. Phase labels
come from marker expression, which advantages the expression baseline by construction.

Paired cell-bootstrap intervals hold all OOF fits fixed. Intervals spanning zero are unresolved
differences; no equivalence margin is specified. Archived state manifests lack expression-file
hashes: present-input alignment checks and hashes cannot retroactively prove their cohort linkage.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import time

import numpy as np
from rev_selection_matched_transfer import file_sha256, load_states, ranked_positive_indices


def encode_topk_inputs(expression, k=512):
    matrix = np.asarray(expression)
    if matrix.ndim!=2 or not np.isfinite(matrix).all() or (matrix<0).any() or k<1:
        raise ValueError('Input ladder requires a finite nonnegative expression matrix and positive k')
    magnitude, binary, rank = (np.zeros_like(matrix) for _ in range(3))
    for row_index, row in enumerate(matrix):
        selected = ranked_positive_indices(row, k)
        magnitude[row_index, selected] = row[selected]
        binary[row_index, selected] = 1
        rank[row_index, selected] = k-np.arange(len(selected))
    return {'expression_topk_magnitude': magnitude, 'expression_topk_binary': binary,
            'expression_topk_rank': rank}


def circular_r2(predicted, truth):
    predicted, truth = np.asarray(predicted, float), np.asarray(truth, float)
    if predicted.shape!=truth.shape or truth.ndim!=1 or not np.isfinite(predicted).all() or not np.isfinite(truth).all():
        raise ValueError('Predicted and true phase vectors must align and be finite')
    mean = np.angle(np.exp(1j*truth).mean())
    baseline = np.mean(1-np.cos(truth-mean))
    if baseline<=1e-12:
        raise ValueError('Circular R-squared is undefined for a degenerate phase cohort')
    return float(1-np.mean(1-np.cos(predicted-truth))/(baseline+1e-12))


def pca_k(expression, k=50):
    """The archived cc_benchmark_c2s.pca_k implementation, retaining input precision."""
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    return StandardScaler().fit_transform(PCA(min(k, expression.shape[1], expression.shape[0]-1),
                     random_state=0).fit_transform(expression-expression.mean(0)))


def oof_phase(predictors, phase):
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import KFold
    targets, predictions = np.column_stack([np.cos(phase), np.sin(phase)]), np.zeros((len(phase), 2))
    folds = np.zeros(len(phase), int)
    for fold, (train, test) in enumerate(KFold(5, shuffle=True, random_state=0).split(predictors)):
        predictions[test] = Ridge(alpha=10.).fit(predictors[train], targets[train]).predict(predictors[test])
        folds[test] = fold
    return np.arctan2(predictions[:, 1], predictions[:, 0]), folds


def paired_cell_bootstrap(predictions, truth, pairs, n_boot=5000, seed=0):
    if n_boot<1:
        raise ValueError('At least one bootstrap draw is required')
    truth = np.asarray(truth)
    point = {name: circular_r2(values, truth) for name, values in predictions.items()}
    errors = {name: 1-np.cos(values-truth) for name, values in predictions.items()}
    complex_phase = np.exp(1j*truth)
    rng = np.random.default_rng(seed)
    draws = {name: np.empty(n_boot) for name in predictions}
    for draw in range(n_boot):
        indices = rng.integers(len(truth), size=len(truth))
        denominator = 1-abs(complex_phase[indices].mean())
        if denominator<=1e-12:
            raise ValueError('Degenerate phase bootstrap sample')
        for name, error in errors.items():
            draws[name][draw] = 1-error[indices].mean()/(denominator+1e-12)
    contrasts = {}
    for first, second in pairs:
        delta = draws[first]-draws[second]
        lower, upper = np.quantile(delta, [.025, .975])
        interpretation = 'first_arm_higher' if lower>0 else 'first_arm_lower' if upper<0 else 'unresolved_difference'
        contrasts[f'{first}_minus_{second}'] = {'point_delta': point[first]-point[second],
            'ci95': [float(lower), float(upper)], 'interpretation': interpretation,
            'interpretation_limit': 'CI spanning zero is an unresolved difference, not equivalence or parity'}
    return {'point': {name: {'circular_r2': value, 'ci95': np.quantile(draws[name], [.025, .975]).tolist()}
                      for name, value in point.items()}, 'contrasts': contrasts}


def check_cache_validation(path, files, model, layer):
    """An optional GPU validation record must refer to exactly these current input files."""
    record = json.loads(Path(path).read_text())
    if record['schema']!='sampled-cell-summary-cache-validation-v1' or record['model']!=model or record['layer']!=layer:
        raise ValueError('Cache validation is for a different model, layer or protocol')
    if record['all_sampled_rows_pass'] is not True or record['n_sampled']<1:
        raise ValueError('Sampled cache validation did not pass')
    for label, source in files.items():
        if record['inputs'][label]['sha256']!=file_sha256(source):
            raise ValueError(f'Cache validation input differs from current {label}')
    return {'status': 'sampled_validation_passed_for_current_inputs', 'n_sampled': record['n_sampled'],
            'validation_sha256': file_sha256(path), 'coverage_limit': record['coverage_limit']}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--h5ad', type=Path, required=True)
    parser.add_argument('--states', type=Path, required=True)
    parser.add_argument('--route-dir', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--model', default='vandijklab/C2S-Scale-Gemma-2-2B')
    parser.add_argument('--layer', type=int, default=21)
    parser.add_argument('--k', type=int, default=512)
    parser.add_argument('--expected-cells', type=int, default=3000)
    parser.add_argument('--n-boot', type=int, default=5000)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--cache-validation', type=Path, help='Optional successful sampled GPU validation JSON; all cohort/state hashes must match')
    args = parser.parse_args()
    import anndata
    from scipy import sparse
    started = time.monotonic()
    cohort = anndata.read_h5ad(args.h5ad)
    if cohort.n_obs!=args.expected_cells or cohort.n_obs<5 or not cohort.obs_names.is_unique:
        raise ValueError('The supplied canonical cohort has the wrong size or duplicate cell names')
    states, ids, manifest = load_states(args.states, cohort.n_obs, args.layer, args.k)
    if manifest['model']!=args.model or args.layer not in manifest['layers'] or manifest['prompt'] is not True:
        raise ValueError('States must match the requested model, layer and prompted sentence construction')
    if manifest['n_cells']!=cohort.n_obs or manifest['n_rows']!=len(states) or manifest['d_model']!=states.shape[1]:
        raise ValueError('State manifest sizes do not match the cohort and activation array')
    if not np.array_equal(np.sort(ids), np.arange(cohort.n_obs)):
        raise ValueError('States must cover exactly the same cohort once, without row truncation or substitution')
    validation = {'status': 'not supplied'}
    if args.cache_validation:
        validation = check_cache_validation(args.cache_validation,
            {'h5ad': args.h5ad, 'states': args.states/f'layer_{args.layer:02d}_activations.npy',
             'row_ids': args.states/'row_cell_ids.npy', 'manifest': args.states/'manifest.json'}, args.model, args.layer)
    phase_path = args.route_dir/'cc_phase.py'
    specification = importlib.util.spec_from_file_location('archived_cc_phase', phase_path)
    phase_module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(phase_module)
    full_phase, _, _, orientation = phase_module.phase_angle_oriented(cohort, seed=0)
    phase = np.asarray(full_phase)[ids]
    full = cohort.X.toarray() if sparse.issparse(cohort.X) else np.asarray(cohort.X)
    expression = np.asarray(full[ids])
    representations = {'expression_full': expression, **encode_topk_inputs(expression, args.k), 'model': states}
    predictions, dimensions = {}, {}
    for name, matrix in representations.items():
        coordinates = pca_k(matrix)
        predictions[name], folds = oof_phase(coordinates, phase)
        dimensions[name] = coordinates.shape[1]
    predictions['constant'] = np.full_like(phase, np.angle(np.exp(1j*phase).mean()))
    predictions['random'] = np.random.default_rng(args.seed).uniform(0, 2*np.pi, len(phase))
    estimates = paired_cell_bootstrap(predictions, phase,
        [('model', 'expression_topk_rank'), ('model', 'expression_topk_binary'), ('model', 'expression_full')], args.n_boot, args.seed)
    files = {'h5ad': args.h5ad, 'phase_code': phase_path, 'analysis_code': Path(__file__),
             'selection_code': Path(__file__).with_name('rev_selection_matched_transfer.py')}
    for name in ('manifest.json', 'row_cell_ids.npy', f'layer_{args.layer:02d}_activations.npy'):
        files[name] = args.states/name
    output = {'schema': 'within-k562-input-ladder-v1', 'model': args.model, 'layer': args.layer, 'k': args.k,
              'n_cells': len(phase), 'n_genes': cohort.n_vars, **estimates,
              'cell_indices': ids.tolist(), 'cell_ids': np.asarray(cohort.obs_names)[ids].astype(str).tolist(),
              'phase': phase.tolist(), 'predicted_phases': {name: values.tolist() for name, values in predictions.items()},
              'fold_ids': folds.tolist(), 'pca_dimensions': dimensions, 'phase_orientation': orientation,
              'cohort_checks': {'expected_cells': args.expected_cells, 'complete_unique_index_coverage': True,
                  'manifest_model_unit_layer_prompt_sizes': True,
                  'sampled_cache_validation': validation,
                  'archived_linkage_limit': 'Archived state manifest has no expression-file hash or obs_names; current hashes and positional checks preserve supplied provenance but cannot independently prove the historical cohort linkage'},
              'definitions': {'selection': 'Full-panel positive-gene ranking, descending expression, stable panel-order ties; top k before any projection',
                  'expression': 'h5ad.X as provided, no second normalization',
                  'preprocessing': 'Whole-cohort PCA50(random_state=0) then StandardScaler, before five-fold Ridge; archived ruler, not fold-fitted preprocessing',
                  'probe': '5-fold KFold(shuffle=True,random_state=0), Ridge(alpha=10) on cos/sin of oriented phase',
                  'circular_r2': '1 - mean(1-cos(prediction-phase)) / (mean(1-cos(phase-circular_mean(phase))) + 1e-12)',
                  'constant': 'Whole-cohort mean-phase reference; zero circular R-squared by score definition, not a fitted predictor',
                  'random': 'Independent uniform phases with recorded seed',
                  'bootstrap': 'Paired cohort-cell percentile intervals with fixed OOF fits and resampled score denominator; not fold-refitting or cross-experiment uncertainty',
                  'phase_labels': 'Derived from marker expression; expression baselines have a definitional advantage',
                  'comparison': 'No equivalence margin or equivalence test; intervals spanning zero are unresolved differences'},
              'n_boot': args.n_boot, 'seed': args.seed, 'seconds': time.monotonic()-started,
              'inputs': {name: {'file': path.name, 'sha256': file_sha256(path)} for name, path in files.items()},
              'software': {name: importlib.metadata.version(name) for name in ('numpy', 'scipy', 'scikit-learn', 'anndata')}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix+'.pending')
    temporary.write_text(json.dumps(output, indent=2, allow_nan=False))
    temporary.replace(args.out)
    print(json.dumps({'point': output['point'], 'contrasts': output['contrasts'], 'out': str(args.out)}, indent=2), flush=True)


if __name__=='__main__':
    main()
