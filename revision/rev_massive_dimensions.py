"""E4 dimension presets and descriptive comparisons, without copying activation arrays.

  python rev_massive_dimensions.py preset --stats results/scale-extract_residual_stats.npz \
      --layers 24 38 --out-dir results/dropmassive_dimensions
  python rev_massive_dimensions.py compare --run runs/27b_matched --layers 24 38 \
      --projected-subdir sae_dropmassive --out results/dropmassive_comparison.json

Presets select dimensions with RMS strictly greater than ten times the layer median. Empty or
degenerate presets explicitly skip training; they are not counted as causal mechanism tests.
Dead-feature fractions are compared under the same training recipe. Reconstruction variance is
reported for each input separately; no original-versus-projected variance-explained delta is made.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def high_rms_preset(rms, layer, multiplier=10.):
    rms = np.asarray(rms, float)
    if rms.ndim!=1 or not len(rms) or not np.isfinite(rms).all() or (rms<0).any():
        raise ValueError('RMS values must be a finite nonnegative vector')
    median = float(np.median(rms))
    indices = np.flatnonzero(rms>multiplier*median).tolist() if median>0 else []
    return {'layer': layer, 'status': 'selected' if indices else 'skip_no_high_rms_dimensions' if median>0 else 'skip_zero_median_rms',
            'width': len(rms), 'indices': indices, 'median_rms': median, 'threshold_multiplier': multiplier,
            'rms_threshold': multiplier*median, 'selected_rms': rms[indices].tolist(),
            'definition': 'Per-dimension RMS > 10 x median dimension RMS; selection independent of SAE dead-feature outcomes'}


def compare_records(baseline, projected, layer):
    if projected.get('status')=='skipped_no_dimensions':
        return {'layer': layer, 'status': 'skipped_no_dimensions', 'input_projection': projected['input_projection'],
                'interpretation': 'No selected dimensions; no mechanism test performed'}
    for key in ('n_features', 'k', 'epochs', 'seed'):
        if baseline[key]!=projected[key]:
            raise ValueError(f'Unmatched E4 training recipe: {key}')
    if projected['input_projection']['mode']!='zero_selected_columns':
        raise ValueError('Projected comparison lacks explicit input projection metadata')
    return {'layer': layer, 'status': 'compared', 'input_projection': projected['input_projection'],
            'training_recipe': {key: baseline[key] for key in ('n_features', 'k', 'epochs', 'seed')},
            'dead_fraction_heldout_original': baseline['dead_frac'],
            'dead_fraction_heldout_projected': projected['dead_frac'],
            'dead_fraction_heldout_delta': projected['dead_frac']-baseline['dead_frac'],
            'dead_fraction_all_rows_original': baseline['dead_frac_all_rows'],
            'dead_fraction_all_rows_projected': projected['dead_frac_all_rows'],
            'dead_fraction_all_rows_delta': projected['dead_frac_all_rows']-baseline['dead_frac_all_rows'],
            'variance_explained_original_input': baseline['variance_explained'],
            'variance_explained_projected_input': projected['variance_explained'],
            'interpretation': 'Different input variance references; variance explained is not a like-for-like contrast'}


def write(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, allow_nan=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest='command', required=True)
    preset = subparsers.add_parser('preset')
    preset.add_argument('--stats', type=Path, required=True)
    preset.add_argument('--layers', type=int, nargs='+', default=[24, 38])
    preset.add_argument('--out-dir', type=Path, required=True)
    compare = subparsers.add_parser('compare')
    compare.add_argument('--run', type=Path, required=True)
    compare.add_argument('--layers', type=int, nargs='+', default=[24, 38])
    compare.add_argument('--projected-subdir', default='sae_dropmassive')
    compare.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.command=='preset':
        checksum = hashlib.sha256(args.stats.read_bytes()).hexdigest()
        records = []
        with np.load(args.stats) as arrays:
            for layer in args.layers:
                record = high_rms_preset(arrays[f'rms_L{layer:02d}'], layer)
                record.update(source_stats=str(args.stats), source_stats_sha256=checksum)
                write(args.out_dir/f'dimensions_L{layer:02d}.json', record)
                records.append(record)
        write(args.out_dir/'presets.json', {'experiment': 'E4', 'layers': records})
    else:
        records, sources = [], {}
        for layer in args.layers:
            first = args.run/'sae'/f'train_L{layer:02d}.json'
            second = args.run/args.projected_subdir/f'train_L{layer:02d}.json'
            records.append(compare_records(json.loads(first.read_text()), json.loads(second.read_text()), layer))
            for path in (first, second): sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        write(args.out, {'experiment': 'E4', 'layers': records, 'source_sha256': sources,
                         'definition': 'Dimension ablation using original-width, capacity-matched, identically seeded dictionaries; dead-fraction contrasts only'})


if __name__=='__main__':
    main()
