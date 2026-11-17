"""Independent arithmetic and provenance audit of saved transfer/within/cache results.

No analysis-helper functions are imported, no prediction models are refitted, and no GPU data
is extracted. Bootstrap draws are independently regenerated from saved phases/predictions.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest
from sklearn.model_selection import KFold


def sha(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1<<20), b''): digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    cache, source_hashes = {}, {}
    def read(path):
        path = Path(path)
        if path not in cache:
            raw = path.read_bytes()
            cache[path] = json.loads(raw)
            source_hashes[str(path)] = hashlib.sha256(raw).hexdigest()
        return cache[path]
    transfer = read(args.results/'selection_matched_transfer.json')
    within = read(args.results/'within_k562_cellcycle.json')
    caches = {line: read(args.results/f'cache_validation_{line}.json') for line in ('k562', 'rpe1')}
    ledger = read(args.results/'retained_numeric_ledger.json')
    checks, findings = [], []
    def check(label, saved, independent):
        if isinstance(independent, (float, np.floating)):
            passed = bool(np.isclose(saved, independent, rtol=1e-10, atol=1e-12))
        elif isinstance(independent, np.ndarray):
            passed = (bool(np.allclose(saved, independent, rtol=1e-10, atol=1e-12))
                      if np.issubdtype(independent.dtype, np.number) else saved==independent.tolist())
            independent = independent.tolist()
        else: passed = saved==independent
        checks.append({'check': label, 'passed': bool(passed), 'saved': saved, 'independent': independent})
    # Resolve every ledger JSON pointer, even entries marked superseded or provisional.
    resolved, unresolvable, unaudited_derivations = 0, [], []
    def pointer(document, path):
        value = document
        for token in path.strip('/').split('/') if path else []:
            token = token.replace('~1', '/').replace('~0', '~')
            value = value[int(token)] if isinstance(value, list) else value[token]
        return value
    def steering_paths(document):
        return {arm: np.asarray([np.abs(np.angle(np.exp(1j*np.diff([point['phase'] for point in row['traj'][arm]])))).sum()
                                  for row in document['rows']]) for arm in ('manifold', 'linear', 'random')}
    for claim in ledger['claims']:
        if 'json_pointer' not in claim or 'absolute_artifact' not in claim: continue
        path = Path(claim['absolute_artifact'])
        if not path.is_file():
            unresolvable.append(claim['id']); continue
        document = read(path)
        try:
            value = pointer(document, claim['json_pointer'])
            if claim['status']=='derived_recomputed':
                identity = claim['id']
                if identity=='gene_geometry_permutations':
                    value = [row['coexpr_matched']['n_perm'] for row in document['layers']]
                elif identity=='benchmark_model_wins':
                    value = sum(record[metric]>document['expression'][k][metric]
                                for layer in document['model'].values() for k, record in layer.items()
                                for metric in ('order', 'classify', 'auroc'))
                elif identity.endswith('_mean_path_length'):
                    arm = identity.split('.json_', 1)[1].split('_mean_path_length', 1)[0]
                    value = float(steering_paths(document)[arm].mean())
                elif identity.endswith('_manifold_longer_count') or identity.endswith('_path_length_sign_p'):
                    paths = steering_paths(document)
                    count = int((paths['manifold']>paths['linear']).sum())
                    value = float(binomtest(count, len(paths['manifold']), .5, alternative='greater').pvalue) if identity.endswith('_sign_p') else count
                elif identity.endswith('_instrument_R_diff'):
                    initial = np.array([row['traj']['manifold'][0]['phase'] for row in document['rows']])
                    source = np.array([row['src'] for row in document['rows']])*2*np.pi/document['n_bins']
                    value = float(abs(np.exp(1j*(initial-source)).mean()))
                elif identity.startswith('L21_') and identity.endswith('_target_progress'):
                    arm = identity.split('_')[1]
                    knots = path.parent/'manifold_knots_cc.npz'
                    with np.load(knots) as arrays: centers = arrays['centers']
                    source_hashes[str(knots)] = sha(knots)
                    changes = []
                    for row in document['rows']:
                        target = centers[row['tgt']]
                        phases = [row['traj'][arm][0]['phase'], row['traj'][arm][-1]['phase']]
                        distance = np.abs(np.angle(np.exp(1j*(np.asarray(phases)-target))))
                        changes.append(distance[0]-distance[1])
                    value = float(np.mean(changes))
                elif identity.startswith('name_') and identity.endswith('_relative_class_separation'):
                    value /= pointer(document, claim['denominator_pointer'])
                elif identity=='submitted_references_all_cited': value = len(document)
                else:
                    unaudited_derivations.append(identity)
                    continue
            check(f'ledger pointer {claim["id"]}', claim['value'], np.asarray(value) if isinstance(value, list) else value)
            resolved += 1
        except (KeyError, IndexError, ValueError): unresolvable.append(claim['id'])
    if unresolvable: findings.append({'severity': 'error', 'finding': 'Ledger pointers cannot be resolved', 'claims': unresolvable})
    # Recompute within-cell scores by the circular loss and re-estimated bootstrap denominator.
    truth = np.asarray(within['phase'])
    names = list(within['predicted_phases'])
    predicted = np.asarray([within['predicted_phases'][name] for name in names])
    mean_phase = np.arctan2(np.sin(truth).mean(), np.cos(truth).mean())
    denominator = np.mean(1-np.cos(truth-mean_phase))
    loss = 1-np.cos(predicted-truth)
    points = 1-loss.mean(1)/(denominator+1e-12)
    for name, point in zip(names, points): check(f'within {name} circular R-squared', within['point'][name]['circular_r2'], float(point))
    rng = np.random.default_rng(within['seed'])
    draws = np.empty((within['n_boot'], len(names)))
    for draw in range(within['n_boot']):
        indices = rng.integers(len(truth), size=len(truth))
        sampled = truth[indices]
        center = np.arctan2(np.sin(sampled).mean(), np.cos(sampled).mean())
        baseline = np.mean(1-np.cos(sampled-center))
        draws[draw] = 1-loss[:, indices].mean(1)/(baseline+1e-12)
    for index, name in enumerate(names):
        check(f'within {name} bootstrap CI', within['point'][name]['ci95'], np.quantile(draws[:, index], [.025, .975]))
    for name, contrast in within['contrasts'].items():
        first, second = name.split('_minus_', 1)
        i, j = names.index(first), names.index(second)
        check(f'within {name} point delta', contrast['point_delta'], float(points[i]-points[j]))
        interval = np.quantile(draws[:, i]-draws[:, j], [.025, .975])
        check(f'within {name} paired CI', contrast['ci95'], interval)
        interpretation = 'first_arm_higher' if interval[0]>0 else 'first_arm_lower' if interval[1]<0 else 'unresolved_difference'
        check(f'within {name} interval interpretation', contrast['interpretation'], interpretation)
    expected_folds = np.zeros(len(truth), int)
    for fold, (_, test) in enumerate(KFold(5, shuffle=True, random_state=0).split(expected_folds)):
        expected_folds[test] = fold
    check('within fold IDs', within['fold_ids'], expected_folds)
    check('within complete positional cohort coverage', sorted(within['cell_indices']), list(range(within['n_cells'])))
    check('within unique cell names', len(set(within['cell_ids'])), within['n_cells'])
    # Independently regenerate transfer residual means and paired bootstrap intervals.
    truth_transfer = np.asarray(transfer['target_phase'])
    transfer_names = list(transfer['predicted_target_phases'])
    predictions_transfer = np.asarray([transfer['predicted_target_phases'][name] for name in transfer_names])
    residuals = np.exp(1j*(predictions_transfer-truth_transfer))
    point_transfer = np.abs(residuals.mean(1))
    for name, point in zip(transfer_names, point_transfer): check(f'transfer {name} R_diff', transfer['point'][name]['R_diff'], float(point))
    rng = np.random.default_rng(transfer['seed'])
    transfer_draws = np.empty((transfer['n_boot'], len(transfer_names)))
    for draw in range(transfer['n_boot']):
        indices = rng.integers(len(truth_transfer), size=len(truth_transfer))
        transfer_draws[draw] = np.abs(residuals[:, indices].mean(1))
    for name, contrast in transfer['contrasts'].items():
        first, second = name.split('_minus_', 1)
        i, j = transfer_names.index(first), transfer_names.index(second)
        delta = transfer_draws[:, i]-transfer_draws[:, j]
        check(f'transfer {name} point delta', contrast['point_delta'], float(point_transfer[i]-point_transfer[j]))
        check(f'transfer {name} paired CI', contrast['ci95'], np.quantile(delta, [.025, .975]))
        check(f'transfer {name} saved bootstrap draws', contrast['bootstrap_delta_draws'], delta)
        interval = np.quantile(delta, [.025, .975])
        interpretation = 'first_arm_higher' if interval[0]>0 else 'first_arm_lower' if interval[1]<0 else 'unresolved_difference'
        check(f'transfer {name} interval interpretation', contrast['interpretation'], interpretation)
    # Audit recorded sample pass decisions and join their file hashes to the retained statistics.
    for line, validation in caches.items():
        passed = []
        for row in validation['rows']:
            passed.append(row['cosine']>=validation['criteria']['minimum_cosine'] and
                          row['relative_l2']<=validation['criteria']['maximum_relative_l2'] and row['expected_is_tied_for_nearest'])
        check(f'{line} sampled cache row count', validation['n_sampled'], len(passed))
        check(f'{line} sampled cache pass count', validation['n_passed'], sum(passed))
        check(f'{line} all sampled cache rows pass', validation['all_sampled_rows_pass'], all(passed))
        prefix = 'source' if line=='k562' else 'target'
        for cached_key, transfer_key in [('h5ad', f'{prefix}_h5ad'), ('states', f'{prefix}_layer_21_activations.npy'),
                                          ('row_ids', f'{prefix}_row_cell_ids.npy'), ('manifest', f'{prefix}_manifest.json')]:
            check(f'{line} cache / transfer {cached_key} SHA256', validation['inputs'][cached_key]['sha256'], transfer['inputs'][transfer_key]['sha256'])
    check('within attached validation SHA256', within['cohort_checks']['sampled_cache_validation']['validation_sha256'], sha(args.results/'cache_validation_k562.json'))
    for label in ('h5ad', 'manifest.json', 'row_cell_ids.npy', 'layer_21_activations.npy'):
        key = {'h5ad':'h5ad', 'manifest.json':'manifest', 'row_cell_ids.npy':'row_ids', 'layer_21_activations.npy':'states'}[label]
        check(f'within / sampled K562 cache {label} SHA256', within['inputs'][label]['sha256'], caches['k562']['inputs'][key]['sha256'])
    statuses = Counter(claim['status'] for claim in ledger['claims'])
    pending = [claim['id'] for claim in ledger['claims'] if claim['id'].startswith('selection_matched') and 'pending_cache' in claim['status']]
    if pending: findings.append({'severity': 'bookkeeping', 'finding': 'Selection-transfer ledger status still says cache validation pending despite saved passing sampled checks', 'n_claims': len(pending)})
    within_entries = [claim for claim in ledger['claims'] if claim['id'].startswith('within_')]
    if not within_entries: findings.append({'severity': 'bookkeeping', 'finding': 'Ledger has no new within-K562 ladder entries although saved points/predictions/paired intervals are available'})
    if not any(claim['id'].startswith('cache_') for claim in ledger['claims']):
        findings.append({'severity': 'bookkeeping', 'finding': 'Ledger has no saved sampled-cache-validation count/criterion entries'})
    if any(issue['id']=='within_K562_intermediate_ladder_missing' and not issue.get('status', '').startswith('resolved') for issue in ledger['issues']):
        findings.append({'severity': 'bookkeeping', 'finding': 'Ledger missing-within-ladder issue is stale'})
    failures = [record for record in checks if not record['passed']]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({'audit': 'V1 independent retained saved-result arithmetic/provenance',
        'all_arithmetic_checks_passed': not failures and not unresolvable, 'n_checks': len(checks),
        'n_ledger_pointers_resolved': resolved, 'ledger_status_counts': dict(statuses), 'findings': findings,
        'unaudited_derived_ledger_entries': unaudited_derivations,
        'checks': checks, 'limitations': ['No model readouts or expression PCA/Ridge fits were rerun; saved predictions were audited.',
            'Cached-state pass criteria and hash linkage were audited; fresh GPU vectors were not independently extracted.',
            'Sampled cache validation covers 20 rows per cohort, not every cached row.'],
        'source_sha256': {**source_hashes, str(Path(__file__)): sha(Path(__file__))}}, indent=2, allow_nan=False))
    print(json.dumps({'audit': str(args.out), 'n_checks': len(checks), 'arithmetic_passed': not failures,
                      'failed_check_names': [record['check'] for record in failures], 'findings': findings}, indent=2))
    if failures or unresolvable: raise SystemExit(2)


if __name__=='__main__': main()
