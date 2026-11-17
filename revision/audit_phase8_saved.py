#!/usr/bin/env python3
"""Independent saved-array V1 audit of response BH, specificity and retrieval.

Does not import experiment implementations, refit a model, or read the large h5ad.
Per-item bootstrap fits are not rerun. Across-TF means and their saved percentile
intervals are reproduced from the saved per-TF records, using the documented RNG.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import resource
import sys

import numpy as np
from scipy.stats import fisher_exact, mannwhitneyu, t as student_t
from sklearn.metrics import average_precision_score, roc_auc_score
from statsmodels.stats.multitest import multipletests

from audit_experimental_saved import Audit


METRICS = ['average_precision', 'auroc', 'ap_lift', 'prevalence', 'prevalence_lift',
           'random_label_expected_ap', 'lift_over_random_expected_ap']


def random_ap(scores, n_positive):
    """Average conditional prefix precision over tied score groups under label permutation."""
    n = len(scores)
    if n_positive == 0:
        return 0.
    if n == 1:
        return 1.
    group_sizes = np.unique(scores, return_counts=True)[1][::-1]
    prefix = np.cumsum(group_sizes)
    expected_precision = (1+(prefix-1)*(n_positive-1)/(n-1))/prefix
    return float(np.dot(group_sizes/n, expected_precision))


def check_response(arrays, audit, alpha, effect_threshold):
    p, q, effect, responding = [np.asarray(arrays[key]) for key in ['p', 'q', 'effect', 'responding']]
    audit.check('response vector alignment', p.shape == q.shape == effect.shape == responding.shape)
    audit.check('response p/effect finite and p within unit interval', np.isfinite(effect).all() and np.all(np.isfinite(p) & (p >= 0) & (p <= 1)))
    expected_q = multipletests(p, method='fdr_bh')[1]
    audit.equal('full-column BH including p=1 columns', q, expected_q, atol=1e-300)
    audit.check('responding flag equals strict q/effect rule', np.array_equal(responding, (expected_q < alpha) & (np.abs(effect) > effect_threshold)))


def check_raw_feature_sample(matrix, rows, groups, responses, audit):
    """Recompute prespecified boundary/strong/random feature tests on exact saved cells."""
    control = np.array([index for index, row in enumerate(rows) if row['kind'] == 'control'])
    audit.equal('feature matrix rows match extracted cells', len(matrix), len(rows))
    rng = np.random.default_rng(20261117)
    for position, group in enumerate(groups):
        group_rows = np.array([index for index, row in enumerate(rows) if row['group'] == group])
        # Four strongest effects, four q=.05 boundaries, four outcome-independent
        # random columns: selection changes coverage, never the test being checked.
        columns = np.unique(np.r_[np.argsort(-np.abs(responses['effect'][position]), kind='stable')[:4],
            np.argsort(np.abs(responses['q'][position]-.05), kind='stable')[:4],
            rng.choice(matrix.shape[1], min(4, matrix.shape[1]), replace=False)])
        perturbed = matrix[np.ix_(group_rows, columns)].astype(float)
        controls = matrix[np.ix_(control, columns)].astype(float)
        # Identical constant columns contain no evidence of a group difference.
        # Explicitly use p=1 rather than a version-dependent 0/0 tie correction.
        varying = (perturbed.var(0)+controls.var(0) > 0) | (perturbed.mean(0) != controls.mean(0))
        pvalues = np.ones(len(columns))
        if varying.any():
            pvalues[varying] = mannwhitneyu(perturbed[:, varying], controls[:, varying], axis=0,
                                           method='asymptotic', alternative='two-sided').pvalue
        effects = ((perturbed.mean(0)-controls.mean(0))/
                   np.maximum(np.sqrt((perturbed.var(0)+controls.var(0))/2), 1e-9))
        audit.equal(group+' sampled raw-cell Mann–Whitney p', responses['p'][position, columns], pvalues, atol=1e-300)
        audit.equal(group+' sampled raw-cell effect', responses['effect'][position, columns], effects)
    return {'n_conditions': len(groups), 'maximum_features_per_condition': 12,
            'control_alive_features': int(np.any(matrix[control] > 0, axis=0).sum()),
            'all_group_eligible_features': int(np.any(matrix != 0, axis=0).sum())}


def metric_summary(records, bootstrap, seed):
    selected = [row for row in records if row.get('average_precision') is not None]
    rng = np.random.default_rng(seed)
    output = {'n_tfs_evaluable': len(selected)}
    for metric in METRICS:
        values = np.array([row[metric] for row in selected])
        output['mean_'+metric] = float(np.mean(values)) if len(values) else None
        output['mean_'+metric+'_ci95'] = None
        if len(values) and bootstrap:
            replicates = rng.choice(values, size=(bootstrap, len(values)), replace=True).mean(axis=1)
            output['mean_'+metric+'_ci95'] = np.quantile(replicates, [.025, .975]).tolist()
    return output


def check_retrieval(saved, labels, scores, audit):
    labels, scores = np.asarray(labels, bool), np.asarray(scores, float)
    positives, n = int(labels.sum()), len(labels)
    prevalence = positives/n if n else 0.
    expected_ap = random_ap(scores, positives) if n else 0.
    for key, expected in [('n', n), ('n_positive', positives), ('prevalence', prevalence), ('random_label_expected_ap', expected_ap)]:
        audit.equal('retrieval '+key, saved[key], expected)
    if not positives or positives == n:
        for key in ['average_precision', 'auroc', 'ap_lift', 'prevalence_lift', 'lift_over_random_expected_ap']:
            audit.equal('undefined retrieval '+key, saved[key], None)
        return
    ap = float(average_precision_score(labels, scores))
    expected = {'average_precision': ap, 'auroc': float(roc_auc_score(labels, scores)),
                'ap_lift': ap/prevalence, 'prevalence_lift': ap/prevalence,
                'lift_over_random_expected_ap': ap/expected_ap}
    for key, value in expected.items():
        audit.equal('retrieval '+key, saved[key], value)
    for key in ['average_precision_ci95', 'auroc_ci95', 'ap_lift_ci95', 'lift_over_random_expected_ap_ci95']:
        bounds = saved[key]
        audit.check('saved item-bootstrap '+key+' ordered finite bounds', bounds is None or (len(bounds) == 2 and np.isfinite(bounds).all() and bounds[0] <= bounds[1]))


def check_specificity(saved, responding, labels, eligible, audit):
    r, target = responding & eligible, labels & eligible
    table = [[int(np.sum(r & target)), int(np.sum(r & ~target))],
             [int(np.sum(~r & target & eligible)), int(np.sum(~r & ~target & eligible))]]
    p = float(fisher_exact(table, alternative='greater')[1]) if np.any(target) and np.any(r) else 1.
    for key, value in [('table', table), ('p', p), ('n_eligible', int(eligible.sum())),
                       ('n_target_columns', int(target.sum())), ('n_non_target_columns', int(np.sum(eligible & ~target))),
                       ('n_target_responses', table[0][0]), ('n_responding_eligible', int(r.sum())),
                       ('evaluable', bool(target.any() and np.any(eligible & ~target)))]:
        audit.equal('specificity '+key, saved[key], value, atol=1e-300 if key == 'p' else 1e-10)


def audit_insilico(stats_dir, feature_dir, targets, audit):
    raw_path = feature_dir/'insilico.npz'
    if not raw_path.exists():
        return {'available': False}
    paths = [raw_path, feature_dir/'insilico_meta.json', stats_dir/'insilico_statistics.npz',
             stats_dir/'insilico_retrieval.json']
    if not all([audit.require(path) for path in paths]):
        return {'available': False}
    for path in [paths[0], paths[2]]:
        audit.hash(path)
    meta, retrieval = audit.read(paths[1]), audit.read(paths[3])
    with np.load(paths[0], allow_pickle=False) as raw, np.load(paths[2], allow_pickle=False) as saved:
        count = np.asarray(raw['count'], float)[..., None]
        total_difference = raw['sum_tf']-raw['sum_ctrl']
        means = total_difference/np.maximum(count, 1)
        squared_deviation = np.maximum(raw['sum_d2']-total_difference**2/np.maximum(count, 1), 0)
        sem = np.sqrt(squared_deviation/np.maximum(count-1, 1)/np.maximum(count, 1))
        statistic = means/np.maximum(sem, 1e-12)
        eligible = np.broadcast_to(count >= 5, means.shape)
        pvalues = np.where(eligible, 2*student_t.sf(np.abs(statistic), np.maximum(count-1, 1)), 1.)
        for key, value in [('mean_difference', means), ('standard_error', sem), ('t', statistic),
                           ('eligible', eligible), ('p', pvalues)]:
            audit.equal('in-silico sufficient-statistic '+key, saved[key], value)
        saved_q = saved['q']  # Inflate the compressed matrix once, not once per TF/layer.
        genes = np.asarray(meta['genes'])
        for tf_index, factor in enumerate(meta['tfs']):
            known = targets.get(factor.upper(), set())-{factor.upper()}
            labels = np.array([gene.upper() in known for gene in genes])
            candidates = np.array([gene.upper() != factor.upper() for gene in genes])
            for layer_index, layer in enumerate(meta['layers']):
                audit.context = f'in-silico/{factor}/L{layer}: '
                qvalues = multipletests(pvalues[tf_index, :, layer_index], method='fdr_bh')[1]
                audit.equal('BH includes every gene', saved_q[tf_index, :, layer_index], qvalues, atol=1e-300)
                keep = eligible[tf_index, :, layer_index] & candidates
                record = next(row for row in retrieval['retrieval'] if row['tf'] == factor and row['layer'] == layer)
                check_retrieval(record, labels[keep], means[tf_index, keep, layer_index], audit)
    return {'available': True, 'n_tfs': len(meta['tfs']), 'n_layers': len(meta['layers'])}


def audit_phase8(stats_dir, catalog_dir, trrust_path, audit):
    required = [stats_dir/name for name in ['manifest.json', 'regulatory_statistics.json', 'gene_statistics.npz']]
    if not all([audit.require(path) for path in required]):
        return {}
    manifest, regulatory = audit.read(required[0]), audit.read(required[1])
    arguments = manifest['arguments']
    for source, expected_hash in manifest.get('source_sha256', {}).items():
        source_path = Path(source)
        if audit.require(source_path):
            audit.hash(source_path)
            audit.equal('manifest source hash '+source, audit.sources[source], expected_hash)
    alpha, threshold = arguments['alpha'], arguments['effect_threshold']
    bootstrap, seed = arguments['bootstrap'], arguments['seed']
    audit.hash(trrust_path)
    targets = {}
    for line in trrust_path.read_text().splitlines():
        fields = line.split('\t')
        if len(fields) >= 2:
            targets.setdefault(fields[0].upper(), set()).add(fields[1].upper())
    audit.hash(required[2])
    genes_arrays = np.load(required[2], allow_pickle=False)
    genes, groups = genes_arrays['genes'].astype(str), genes_arrays['groups'].astype(str).tolist()
    audit.equal('gene universe unique', len(set(genes)), len(genes))
    audit.equal('manifest gene width', manifest['n_genes'], len(genes))
    records = regulatory['groups']
    keys = [(r['representation'], r['layer'], r['group']) for r in records]
    audit.equal('group/representation/layer records unique', len(set(keys)), len(keys))
    provenance_checked = False
    feature_dir = Path(arguments['features_dir']) if arguments.get('features_dir') else None
    row_counts, row_kinds = {}, {}
    if feature_dir is not None:
        row_path, design_path = feature_dir/'rows.json', Path(arguments['design'])
        if all([audit.require(row_path), audit.require(design_path)]):
            rows, design = audit.read(row_path), audit.read(design_path)
            planned = {(group['group'], int(cell)) for group in design['groups'] for cell in group['cells']}
            observed = [(row['group'], int(row['cell_index'])) for row in rows]
            audit.equal('extracted group/cell rows unique', len(set(observed)), len(observed))
            audit.check('extracted rows all belong to exact selected design', set(observed) <= planned)
            audit.equal('statistics cell IDs equal extracted row order', genes_arrays['cell_indices'], [row['cell_index'] for row in rows])
            audit.equal('manifest extracted-cell count', manifest['n_extracted_cells'], len(rows))
            for row in rows:
                row_counts[row['group']] = row_counts.get(row['group'], 0)+1
                row_kinds[row['group']] = row['kind']
            controls = [group for group in row_counts if row_kinds[group] == 'control']
            audit.equal('single exact control cohort', len(controls), 1)
            audit.equal('control-cell count', manifest['n_control_cells'], row_counts[controls[0]])
            audit.check('saved conditions equal all extracted noncontrol groups with >=3 cells',
                        set(groups) == {group for group, count in row_counts.items() if row_kinds[group] != 'control' and count >= 3})
            provenance_checked = True
    families = [(r['representation'], r['layer']) for r in regulatory['summary']]
    raw_sample_results = {}
    for representation, layer in families:
        audit.context = f'phase8/{stats_dir.parent.name}/{representation}/L{layer}: '
        family = [r for r in records if r['representation'] == representation and r['layer'] == layer]
        audit.equal('every extracted condition retained', set(r['group'] for r in family), set(groups))
        by_group = {r['group']: r for r in family}
        if representation == 'model':
            path = stats_dir/f'feature_statistics_L{layer:02d}.npz'
            catalog_path = catalog_dir/f'feature_catalog_L{layer:02d}.json'
            if not all([audit.require(path), audit.require(catalog_path)]):
                continue
            audit.hash(path)
            arrays = np.load(path, allow_pickle=False)
            audit.equal('feature-array group order', arrays['groups'].astype(str).tolist(), groups)
            catalogue = audit.read(catalog_path)['features']
            memberships = {int(row['feature_idx']): {str(gene['gene_name'] if isinstance(gene, dict) else gene).upper()
                                                    for gene in row['top_genes']} for row in catalogue}
            eligible = arrays['eligible'].astype(bool)
            active_genes = set().union(*(names for feature, names in memberships.items() if eligible[feature]))
            denominator = np.zeros(len(genes))
            gene_to_position = {gene: index for index, gene in enumerate(genes)}
            feature_gene_indices = {}
            for feature, names in memberships.items():
                indices = [gene_to_position[name] for name in names if name in gene_to_position]
                feature_gene_indices[feature] = indices
                denominator[indices] += 1
        else:
            arrays = genes_arrays
            eligible = arrays[representation+'_eligible'].astype(bool)
        # NPZ indexing decompresses the whole named array. Cache one family's small
        # response matrices so the per-condition loop does not inflate them repeatedly.
        responses = {key: arrays[key if representation == 'model' else representation+'_'+key]
                     for key in ['p', 'q', 'effect', 'responding']}
        saved_gene_scores = arrays['gene_scores'] if representation == 'model' else None
        if representation == 'model' and provenance_checked:
            raw_path = feature_dir/f'feat_L{layer:02d}.npz'
            if audit.require(raw_path):
                audit.hash(raw_path)
                with np.load(raw_path, allow_pickle=False) as raw:
                    matrix = raw['X']
                audit.check('raw extracted features finite', np.isfinite(matrix).all())
                audit.check('Fisher feature background is nonzero in any group', np.array_equal(eligible, np.any(matrix != 0, axis=0)))
                raw_sample_results[str(layer)] = check_raw_feature_sample(matrix, rows, groups, responses, audit)
                del matrix
        for index, group in enumerate(groups):
            record = by_group[group]
            audit.context = f'phase8/{stats_dir.parent.name}/{representation}/L{layer}/{group}: '
            response = {key: values[index] for key, values in responses.items()}
            if provenance_checked:
                audit.equal('exact group-cell count', record['n_cells'], row_counts[group])
                audit.equal('exact control-cell count', record['n_control_cells'], manifest['n_control_cells'])
                audit.equal('group kind matches extraction', record['kind'], row_kinds[group])
            check_response(response, audit, alpha, threshold)
            audit.equal('response count', record['n_responding'], int(np.sum(response['responding'])))
            audit.equal('self-target exclusion flag', record['self_target_excluded'], True)
            audit.equal('perturbed-gene retrieval exclusion flag', record['perturbed_gene_excluded_from_gene_retrieval'], True)
            known = targets.get(group.upper(), set())- {group.upper()} if record['kind'] == 'tf' else set()
            gene_labels = np.array([gene in known for gene in genes])
            candidates = genes != group.upper()
            if representation == 'model':
                labels = np.array([bool(memberships[feature] & known) for feature in range(len(eligible))])
                check_specificity(record['specificity'], response['responding'].astype(bool), labels, eligible, audit)
                check_retrieval(record['feature_retrieval'], labels[eligible], np.abs(response['effect'][eligible]), audit)
                scores = np.zeros(len(genes))
                for feature, indices in feature_gene_indices.items():
                    scores[indices] += abs(response['effect'][feature])
                scores /= np.maximum(denominator, 1)
                audit.equal('model gene scores from mean absolute catalogue effects', saved_gene_scores[index], scores)
                audit.equal('eligible distinct known-target genes', record['n_eligible_known_target_genes'], len(known & active_genes))
            else:
                scores = np.abs(response['effect'])
                check_specificity(record['specificity'], response['responding'].astype(bool), gene_labels, eligible & candidates, audit)
                audit.equal('eligible distinct known-target genes', record['n_eligible_known_target_genes'], int(np.sum(gene_labels & eligible)))
            check_retrieval(record['gene_retrieval'], gene_labels[candidates], scores[candidates], audit)
        if representation == 'model':
            arrays.close()
        factors = [r for r in family if r['kind'] == 'tf']
        audit.context = f'phase8/{stats_dir.parent.name}/{representation}/L{layer}/family: '
        adjusted = multipletests([r['specificity']['p'] for r in factors], method='fdr_bh')[1]
        audit.equal('specificity BH retains every screened TF', [r['specificity']['q'] for r in factors], adjusted, atol=1e-300)
        saved = next(row for row in regulatory['summary'] if row['representation'] == representation and row['layer'] == layer)
        evaluable = [r for r in factors if r['specificity']['evaluable'] and r['n_eligible_known_target_genes'] >= 1]
        for key, value in [('n_tfs', len(factors)), ('n_tfs_screened', len(factors)), ('n_tfs_evaluable', len(evaluable)),
                           ('n_tfs_unevaluable', len(factors)-len(evaluable)),
                           ('n_specific', int(np.sum(adjusted < alpha))), ('n_groups', len(family)),
                           ('n_responding_groups', sum(r['n_responding'] > 0 for r in family))]:
            audit.equal(key, saved[key], value)
        for minimum in [1, 3, 5]:
            selected = [r for r in factors if r['specificity']['evaluable'] and r['n_eligible_known_target_genes'] >= minimum]
            expected = {'n_tfs_evaluable': len(selected), 'n_specific_full_family_bh': sum(r['specificity']['q'] < alpha for r in selected)}
            audit.equal(f'target coverage sensitivity >= {minimum}', saved['target_coverage_sensitivity'][f'min_{minimum}_eligible_target_genes'], expected)
        for kind in ['gene_retrieval', 'feature_retrieval']:
            aggregate = metric_summary([r[kind] for r in factors if kind in r], bootstrap, seed)
            for key, value in aggregate.items():
                audit.equal(kind+' TF-bootstrap '+key, saved[kind][key], value)
        pseudo = [{'group': r['group'], 'n_responding': r['n_responding']} for r in family if r['kind'] == 'pseudo']
        audit.equal('pseudo-control groups retained', saved['pseudo_controls'], pseudo)
    genes_arrays.close()
    insilico = audit_insilico(stats_dir, feature_dir, targets, audit) if feature_dir else {'available': False}
    return {'n_groups': len(groups), 'n_genes': len(genes), 'n_representation_layer_families': len(families),
            'n_records': len(records), 'exact_row_provenance_checked': provenance_checked,
            'raw_feature_test_sample': raw_sample_results, 'in_silico': insilico, 'summary': regulatory['summary']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stats', type=Path, required=True)
    parser.add_argument('--catalog-dir', type=Path, required=True)
    parser.add_argument('--trrust', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    audit = Audit()
    results = audit_phase8(args.stats, args.catalog_dir, args.trrust, audit)
    audit.hash(Path(__file__))
    report = {'audit': 'V1 independent saved phase8 arithmetic/provenance',
              'passed': audit.passed and not audit.pending, 'n_checks': audit.n_checks,
              'findings': audit.failed, 'pending_files': audit.pending, 'results': results,
              'source_sha256': audit.sources,
              'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform == 'darwin' else 1024),
              'limitations': ['Full response BH uses retained p-values. For model features, up to12 strong/boundary/random columns per condition are independently recalculated from exact saved cell matrices; other raw response tests are not rerun.',
                              'No h5ad expression normalisation or raw cell/feature response tests are rerun.',
                              'Across-TF bootstrap means/intervals are reproduced. Per-item intervals are checked for finite ordered bounds only.']}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps({key: report[key] for key in ['passed', 'n_checks', 'findings', 'pending_files']}))
    raise SystemExit(0 if report['passed'] else 1)


if __name__ == '__main__':
    main()
