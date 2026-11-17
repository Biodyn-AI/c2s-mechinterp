#!/usr/bin/env python3
"""Descriptive circuit-effect correlations over all common eligible tested columns.

Uses complete saved d_all vectors, including zero/subthreshold/nonpassing effects.
Source eligibility requires at least two active cells in both runs. It preserves a
separate explicitly selection-conditioned shared-passing-edge correlation.
"""
import argparse
import json
from pathlib import Path
import resource
import sys

import numpy as np
from scipy.stats import rankdata

from audit_experimental_saved import Audit


def aligned_vector(values, stored_layers, common_layers):
    position = {int(layer): index for index, layer in enumerate(stored_layers)}
    return np.asarray(values)[[position[layer] for layer in common_layers]].reshape(-1)


def correlations(first, second):
    first, second = np.asarray(first, float), np.asarray(second, float)
    if first.shape != second.shape or not np.isfinite(first).all() or not np.isfinite(second).all():
        raise ValueError('Effect vectors must be aligned and finite')
    output = {'n_effect_pairs': len(first), 'pearson': None, 'spearman': None}
    if len(first) > 1 and np.std(first) > 0 and np.std(second) > 0:
        output['pearson'] = float(np.corrcoef(first, second)[0, 1])
        output['spearman'] = float(np.corrcoef(rankdata(first), rankdata(second))[0, 1])
    return output


def source_summary(first, second, first_mask, second_mask, first_active, second_active):
    eligible = first_active >= 2 and second_active >= 2
    a = np.asarray(first_mask, bool) & (first_active >= 2)
    b = np.asarray(second_mask, bool) & (second_active >= 2)
    shared, union = a & b, a | b
    return {'common_eligible': bool(eligible), 'n_active_first': int(first_active), 'n_active_second': int(second_active),
            'n_first_passing': int(a.sum()), 'n_second_passing': int(b.sum()), 'n_shared_passing': int(shared.sum()),
            'passing_jaccard': float(shared.sum()/union.sum()) if union.any() else None,
            'all_tested': correlations(first, second) if eligible else None,
            'shared_passing_only': correlations(np.asarray(first)[shared], np.asarray(second)[shared])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--first', type=Path, help='Default: ROOT/results/circuit_50')
    parser.add_argument('--second', type=Path, help='Default: ROOT/results/circuit_100')
    parser.add_argument('--source-layers', type=int, nargs='+', default=[7])
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    first = args.first or args.root/'results/circuit_50'
    second = args.second or args.root/'results/circuit_100'
    audit, records = Audit(), []
    pooled_first, pooled_second, shared_first, shared_second = [], [], [], []
    n_first = n_second = n_shared = 0
    required = [directory/f'circuit_L{layer:02d}{suffix}' for directory in [first, second]
                for layer in args.source_layers for suffix in ['.npz', '_summary.json']]
    if not all([audit.require(path) for path in required]):
        report = {'passed': False, 'status': 'pending', 'pending_files': audit.pending}
        args.out.parent.mkdir(parents=True, exist_ok=True); args.out.write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps(report)); raise SystemExit(1)
    for layer in args.source_layers:
        audit.context = f'L{layer}: '
        summaries = [audit.read(directory/f'circuit_L{layer:02d}_summary.json') for directory in [first, second]]
        for index, expected_cells in enumerate([50, 100]):
            audit.equal('run cell count', summaries[index]['n_cells'], expected_cells)
            audit.equal('same effect threshold', summaries[index]['d_threshold'], .5)
            audit.equal('same consistency threshold', summaries[index]['consistency_threshold'], .7)
        archives = [directory/f'circuit_L{layer:02d}.npz' for directory in [first, second]]
        for path in archives: audit.hash(path)
        with np.load(archives[0], allow_pickle=False) as a, np.load(archives[1], allow_pickle=False) as b:
            features = [archive['features'].astype(int) for archive in [a, b]]
            layers = [archive['downstream_layers'].astype(int).tolist() for archive in [a, b]]
            common_layers = list(range(layer+1, 26))
            for values in layers: audit.equal('full common downstream layer family', sorted(values), common_layers)
            audit.equal('same prespecified source feature IDs', set(features[0]), set(features[1]))
            effects = [archive['d_all'] for archive in [a, b]]
            masks = [archive['edge_mask'].astype(bool) for archive in [a, b]]
            active = [archive['n_cells_active'].astype(int) for archive in [a, b]]
            for index, archive in enumerate([a, b]):
                audit.equal('source activity half-count join', active[index], archive['n_half'].sum(axis=0))
                audit.equal('dictionary target width', effects[index].shape[2], 9216)
                audit.check('complete effect vectors finite', np.isfinite(effects[index]).all())
                audit.equal('array/mask alignment', list(masks[index].shape), list(effects[index].shape))
                audit.equal('source count alignment', len(features[index]), effects[index].shape[0])
                audit.equal('ablated pass count from active sources', summaries[index]['n_ablated_passes'], int(active[index].sum()))
            second_index = {feature: index for index, feature in enumerate(features[1])}
            for index, feature in enumerate(features[0]):
                other = second_index[feature]
                x, y = [aligned_vector(effects[side][position], layers[side], common_layers)
                        for side, position in [(0, index), (1, other)]]
                xmask, ymask = [aligned_vector(masks[side][position], layers[side], common_layers)
                                for side, position in [(0, index), (1, other)]]
                result = source_summary(x, y, xmask, ymask, active[0][index], active[1][other])
                result.update(source_layer=layer, source_feature=int(feature), n_target_layers=len(common_layers), n_target_features_per_layer=9216)
                records.append(result)
                n_first += result['n_first_passing']; n_second += result['n_second_passing']; n_shared += result['n_shared_passing']
                if result['common_eligible']:
                    pooled_first.append(x); pooled_second.append(y)
                    common_mask = xmask & ymask
                    shared_first.append(x[common_mask]); shared_second.append(y[common_mask])
    pooled = correlations(np.concatenate(pooled_first) if pooled_first else [], np.concatenate(pooled_second) if pooled_second else [])
    shared = correlations(np.concatenate(shared_first) if shared_first else [], np.concatenate(shared_second) if shared_second else [])
    comparison = {'n_first': n_first, 'n_second': n_second, 'n_shared': n_shared,
                  'jaccard': n_shared/(n_first+n_second-n_shared) if n_first+n_second-n_shared else None,
                  'fraction_first_recovered': n_shared/n_first if n_first else None,
                  'effect_pearson_shared_edges': shared['pearson']}
    old_path = args.root/'results/circuit_stability.json'
    if old_path.exists():
        old = audit.read(old_path)['comparison']
        for key, value in comparison.items(): audit.equal('existing shared-only comparison '+key, old[key], value, atol=1e-7)
    audit.hash(Path(__file__))
    report = {'audit': 'Independent circuit stability over full common eligible tested families',
              'passed': audit.passed, 'status': 'complete' if audit.passed else 'failed',
              'n_checks': audit.n_checks, 'findings': audit.failed, 'source_layers': args.source_layers,
              'n_source_features': len(records), 'n_common_eligible_source_features': sum(row['common_eligible'] for row in records),
              'per_source': records, 'pooled_all_tested': pooled,
              'selection_conditioned_shared_passing': {'correlations': shared, 'edge_comparison': comparison,
                    'label': 'Conditions on passing the edge rule in both runs; not an all-tested correlation.'},
              'source_sha256': audit.sources,
              'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform == 'darwin' else 1024),
              'definitions': {'eligible_source': 'At least two source-active cells in each run; same prespecified source set.',
                              'full_tested_family': 'Every common downstream layer/feature column, including zeros and nonpassing effects.',
                              'passing_mask': 'Saved full-precision edge_mask plus source activity >=2; no thresholding of rounded d_all.',
                              'pooled_weight': 'Each tested source/layer/target-feature effect pair contributes once.',
                              'input_precision': 'Stored float32 d_all; no fresh model effects are recomputed.'},
              'limitations': ['Prespecified source features are a subset of the dictionary; all-tested refers to their tested downstream families.',
                              'The 50-cell panel is nested within the 100-cell panel; this is not independent-sample replication.',
                              'Target columns and source effects are dependent. Correlations are descriptive; no p-values or independent-pair intervals.']}
    args.out.parent.mkdir(parents=True, exist_ok=True); args.out.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({key: report[key] for key in ['passed', 'n_source_features', 'n_common_eligible_source_features', 'pooled_all_tested', 'selection_conditioned_shared_passing']}))
    raise SystemExit(0 if audit.passed else 1)


if __name__ == '__main__': main()
