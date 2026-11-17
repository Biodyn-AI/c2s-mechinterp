"""Summarize E3 residual spectra and join corresponding SAE dead-feature measurements.

Reads rev_extract.py NPZ keys eig_Lxx, rms_Lxx, mean_Lxx and scalars_Lxx. The first three
scalars are row count, mean row L2 norm and maximum absolute activation; optional fourth is
the median of per-cell median absolute activations. This fourth value is NOT the median of
all pooled scalar activations. Covariance spectra are centered; RMS includes the mean.

Layer correlations are descriptive Spearman ranks. Layers share cells and model weights,
so no independent-layer hypothesis tests or p-values are calculated.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

import numpy as np
from scipy.stats import rankdata


def layer_summary(eigenvalues, rms, mean, scalars):
    eigenvalues, rms, mean, scalars = [np.asarray(x, np.float64) for x in (eigenvalues, rms, mean, scalars)]
    if not all(np.isfinite(x).all() for x in (eigenvalues, rms, mean, scalars)):
        raise ValueError('nonfinite residual statistics')
    width = len(eigenvalues)
    if not width or len(rms) != width or len(mean) != width or len(scalars) < 3:
        raise ValueError('residual statistic widths or scalar layout do not match')
    if (eigenvalues < 0).any() or (rms < 0).any():
        raise ValueError('covariance eigenvalues and RMS values must be nonnegative')
    eigenvalues = np.sort(eigenvalues)[::-1]
    trace = float(eigenvalues.sum())
    pr = trace**2 / float((eigenvalues**2).sum()) if trace > 0 else 0.
    median_rms, max_rms = float(np.median(rms)), float(rms.max())
    cell_median = float(scalars[3]) if len(scalars) >= 4 else None
    mean_norm = float(np.linalg.norm(mean))
    return {'width': width, 'n_gene_positions': int(scalars[0]),
            'centered_covariance_trace': trace, 'centered_covariance_pr': pr,
            'centered_covariance_pr_fraction_width': pr / width,
            'top1_variance_share': float(eigenvalues[0]/trace) if trace else None,
            'top10_variance_share': float(eigenvalues[:10].sum()/trace) if trace else None,
            'rms_max': max_rms, 'rms_median': median_rms,
            'rms_max_over_median': max_rms/median_rms if median_rms else None,
            'n_dimensions_rms_gt_10x_median': int((rms > 10*median_rms).sum()),
            'max_abs_activation': float(scalars[2]),
            'mean_row_l2_norm': float(scalars[1]), 'mean_vector_l2_norm': mean_norm,
            'mean_vector_norm_over_mean_row_norm': mean_norm/float(scalars[1]) if scalars[1] else None,
            'median_of_cell_median_abs_activation': cell_median,
            'max_abs_over_median_of_cell_medians': float(scalars[2])/cell_median if cell_median else None}


def rank_correlation(x, y):
    pairs = [(a, b) for a, b in zip(x, y)
             if a is not None and b is not None and np.isfinite(a) and np.isfinite(b)]
    record = {'n_layers': len(pairs), 'spearman_rho': None, 'interpretation': 'descriptive, dependent layers'}
    if len(pairs) < 3:
        return record
    values = np.asarray(pairs)
    rx, ry = rankdata(values[:, 0]), rankdata(values[:, 1])
    rx, ry = rx-rx.mean(), ry-ry.mean()
    denominator = float(np.linalg.norm(rx) * np.linalg.norm(ry))
    if denominator:
        record['spearman_rho'] = float(np.clip(np.dot(rx, ry)/denominator, -1, 1))
    return record


def optional_json(path):
    return json.loads(path.read_text()) if path.exists() else {}


def summarize_model(path, train_directory, evaluation_directory=None, expected_layers=None):
    spectrum = np.load(path)
    layers = sorted(int(match.group(1)) for key in spectrum.files
                    for match in [re.fullmatch(r'eig_L(\d+)', key)] if match)
    if not layers:
        raise ValueError(f'no covariance spectra in {path}')
    records, sources = [], {}
    for layer in layers:
        record = layer_summary(*(spectrum[f'{key}_L{layer:02d}'] for key in ('eig', 'rms', 'mean', 'scalars')))
        train_path = train_directory / f'train_L{layer:02d}.json'
        evaluation_path = (evaluation_directory / f'2b_eval_L{layer:02d}.json'
                           if evaluation_directory is not None else None)
        trained = optional_json(train_path)
        evaluated = optional_json(evaluation_path) if evaluation_path else {}
        features = evaluated.get('n_features', trained.get('n_features'))
        original_dead = trained.get('dead_frac')
        if original_dead is None and features and 'dead_features' in trained:
            original_dead = trained['dead_features']/features
        reextracted_dead = (evaluated['dead_val_reextracted']/features
                            if features and 'dead_val_reextracted' in evaluated else None)
        all_rows_dead = evaluated.get('dead_frac_all_rows', trained.get('dead_frac_all_rows'))
        record.update(layer=layer, relative_depth=layer/max(max(layers), 1),
                      n_sae_features=features,
                      dead_fraction_heldout_original=original_dead,
                      dead_fraction_heldout_reextracted=reextracted_dead,
                      dead_fraction_heldout=(reextracted_dead if reextracted_dead is not None else original_dead),
                      dead_fraction_all_rows=all_rows_dead,
                      variance_explained_original=trained.get('variance_explained'),
                      variance_explained_reextracted=evaluated.get('variance_explained_reextracted'))
        records.append(record)
        for source in (train_path, evaluation_path):
            if source is not None and source.exists():
                sources[str(source)] = hashlib.sha256(source.read_bytes()).hexdigest()
    if expected_layers is not None:
        for layer in expected_layers:
            found = next((record for record in records if record['layer'] == layer), None)
            if found is None or found['dead_fraction_heldout'] is None:
                raise ValueError(f'missing expected SAE dead-feature metrics at layer {layer}')
    measurements = ['centered_covariance_pr', 'centered_covariance_pr_fraction_width',
                    'top1_variance_share', 'top10_variance_share', 'rms_max_over_median',
                    'n_dimensions_rms_gt_10x_median', 'max_abs_activation',
                    'max_abs_over_median_of_cell_medians', 'mean_vector_norm_over_mean_row_norm']
    correlations = {dead: {measure: rank_correlation([r[measure] for r in records], [r[dead] for r in records])
                           for measure in measurements}
                    for dead in ('dead_fraction_heldout', 'dead_fraction_all_rows')}
    return {'layers': records, 'descriptive_rank_correlations': correlations, 'metric_source_sha256': sources}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stats-2b', type=Path, required=True)
    parser.add_argument('--stats-27b', type=Path, required=True)
    parser.add_argument('--train-2b', type=Path, required=True)
    parser.add_argument('--eval-2b', type=Path, required=True)
    parser.add_argument('--train-27b', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = {'experiment': 'E3',
              '2b': summarize_model(args.stats_2b, args.train_2b, args.eval_2b, list(range(26))),
              '27b_matched': summarize_model(args.stats_27b, args.train_27b,
                                             expected_layers=[0, 7, 14, 24, 31, 38, 45]),
              'definitions': {'covariance': 'centered across gene positions; PR=(sum eigenvalues)^2/sum squared eigenvalues',
                              'rms': 'per-dimension root mean square; includes the mean',
                              'median_activation': 'median of per-cell medians of absolute pooled activations',
                              'dead_heldout': 're-extracted held-out fraction where available, otherwise training held-out fraction',
                              'dead_all_rows': 'never fires across all atlas gene positions',
                              'correlations': 'descriptive within model; layers are dependent, no p-values'},
              'source_sha256': {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                                for path in (args.stats_2b, args.stats_27b, Path(__file__))}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(f'{args.out}: {len(result["2b"]["layers"])} 2B layers, '
          f'{len(result["27b_matched"]["layers"])} 27B layers; descriptive correlations only')


if __name__ == '__main__':
    main()
