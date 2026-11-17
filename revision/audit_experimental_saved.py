#!/usr/bin/env python3
"""Independent V1 arithmetic/provenance audit, streaming one layer at a time.

No experiment implementation is imported and no models, SAE fits or raw activation
matrices are opened. Use --mode final to require every core output, including all
26 original-atlas layers. Partial mode reports absent outputs as pending, never verified.
"""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import json
import math
from pathlib import Path
import resource
import sys

import numpy as np
from scipy.stats import rankdata


NAMED = {'GO_BP', 'KEGG', 'Reactome', 'TRRUST'}


class Audit:
    def __init__(self):
        self.n_checks = 0
        self.failed = []
        self.pending = []
        self.sources = {}
        self.context = ''

    @property
    def passed(self):
        return not self.failed

    def check(self, label, condition, saved=None, expected=None):
        self.n_checks += 1
        if not bool(condition):
            self.failed.append({'check': self.context + label, 'saved': saved, 'independent': expected})

    def equal(self, label, saved, expected, *, atol=1e-10):
        numeric = isinstance(expected, (float, int, np.number, list, np.ndarray)) and not isinstance(expected, bool)
        try:
            # Millions of scalar count checks need no NumPy temporary arrays.
            if isinstance(expected, (int, np.integer)) and isinstance(saved, (int, np.integer)):
                same = saved == expected
            elif numeric and np.isscalar(expected) and np.isscalar(saved):
                same = math.isclose(saved, expected, rel_tol=1e-7, abs_tol=atol)
            else:
                same = np.allclose(saved, expected, rtol=1e-7, atol=atol, equal_nan=False) if numeric else saved == expected
        except (TypeError, ValueError):
            same = (np.array_equal(saved, expected) if isinstance(saved, np.ndarray) or isinstance(expected, np.ndarray)
                    else saved == expected)
        def small(value):
            if isinstance(value, np.ndarray):
                return small(value.tolist()) if value.size <= 12 else {'shape': list(value.shape)}
            if isinstance(value, np.generic):
                return small(value.item())
            if isinstance(value, float) and not np.isfinite(value):
                return str(value)
            if isinstance(value, (list, tuple)):
                return [small(item) for item in value]
            return value
        self.check(label, same, small(saved), small(expected))

    def require(self, path):
        path = Path(path)
        if not path.is_file():
            if str(path) not in self.pending:
                self.pending.append(str(path))
            return False
        return True

    def hash(self, path):
        path = Path(path)
        digest = hashlib.sha256()
        with path.open('rb') as handle:
            for block in iter(lambda: handle.read(1 << 20), b''):
                digest.update(block)
        self.sources[str(path)] = digest.hexdigest()

    def read(self, path):
        raw = Path(path).read_bytes()
        self.sources[str(path)] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw)


def annotation_counts(annotation, audit):
    summary, rows = annotation['summary'], annotation['features']
    total, alive, alpha = summary['n_features_total'], len(rows), summary['alpha']
    by_source = Counter()
    ids = [row['feature_idx'] for row in rows]
    audit.equal('annotation IDs unique', len(set(ids)), len(ids))
    ontology, annotated, testable, extra_unknown = 0, 0, 0, 0
    for row in rows:
        index = row['feature_idx']
        audit.check(f'F{index} complete IDs present', 'significant_term_ids' in row)
        complete = row.get('significant_term_ids', [])
        audit.equal(f'F{index} complete IDs unique', len(set(complete)), len(complete))
        sources = {term.split('::', 1)[0] for term in complete if '::' in term}
        audit.check(f'F{index} named ID sources valid', sources <= NAMED)
        visible = [term for term in row.get('terms', []) if term.get('q', 1) < alpha]
        named_visible = {term['source']+'::'+term['term'] for term in visible if term['source'] in NAMED}
        audit.check(f'F{index} displayed significant terms in complete IDs', named_visible <= set(complete))
        extras = {term['source'] for term in visible if term['source'] not in NAMED}
        n_extra = row['n_sig'] - len(complete)
        audit.check(f'F{index} number of significant network tests', len(extras) <= n_extra <= 3,
                    n_extra, 'between visible network evidence count and 3')
        extra_unknown += int(n_extra > len(extras))
        by_source.update(sources | extras)
        ontology += bool(complete)
        annotated += bool(row['annotated'])
        testable += row['n_genes'] >= 3
        audit.equal(f'F{index} any-evidence flag equals significant-count flag', row['annotated'], row['n_sig'] > 0)
        audit.check(f'F{index} ontology implies any-evidence', not complete or row['annotated'])
    expected_counts = {'n_alive': alive, 'n_sets': alive, 'n_testable': testable, 'n_annotated': annotated,
                       'n_ontology_annotated': ontology, 'n_network_or_membership_only': annotated-ontology,
                       'median_set_size': float(np.median([r['n_genes'] for r in rows]))}
    for key, value in expected_counts.items():
        audit.equal(key, summary.get(key), value)
    rates = {'annotation_rate': annotated/max(alive, 1), 'annotation_rate_all_features': annotated/max(total, 1),
             'ontology_annotation_rate_alive': ontology/max(alive, 1),
             'ontology_annotation_rate_all_features': ontology/max(total, 1)}
    for key, value in rates.items():
        audit.equal(key, summary.get(key), value)
    for source, rate in summary['rate_by_source'].items():
        if source in NAMED or not extra_unknown:
            audit.equal(f'{source} rate from complete IDs/visible network evidence', rate, by_source[source]/max(alive, 1))
        else:
            audit.check(f'{source} rate within bounds from capped network details',
                        by_source[source]-1e-7 <= rate*alive <= by_source[source]+extra_unknown+1e-7)
    threshold = summary['global_bh']['p_threshold']
    global_rate = sum(row['min_p'] <= threshold for row in rows)/max(alive, 1)
    audit.equal('global BH rate from stored minimum p/threshold', summary['global_bh']['annotation_rate'], global_rate)
    floors = {}
    for kind, floor in summary.get('floors', {}).items():
        values, named = np.asarray(floor['values']), np.asarray(floor['ontology_values'])
        audit.equal(f'{kind} floor seeds aligned', len(values), len(named))
        audit.equal(f'{kind} floor mean', floor['mean'], values.mean())
        audit.equal(f'{kind} floor SD', floor['sd'], values.std(ddof=0))
        audit.equal(f'{kind} ontology floor mean', floor['ontology_mean'], named.mean())
        audit.check(f'{kind} ontology floor bounded by any-evidence floor', np.all((named >= 0) & (named <= values) & (values <= 1)))
        audit.check(f'{kind} floor rates have integer alive-feature numerator',
                    np.allclose(np.r_[values, named]*alive, np.rint(np.r_[values, named]*alive), atol=1e-6, rtol=0))
        floors[kind] = {'any_alive': float(values.mean()), 'ontology_alive': float(named.mean()),
                        'any_all_features': float(values.mean()*alive/total),
                        'ontology_all_features': float(named.mean()*alive/total)}
    if 'excess_over_freq_floor' in summary:
        audit.equal('any-evidence excess over frequency floor', summary['excess_over_freq_floor'],
                    annotated/max(alive, 1)-floors['freq']['any_alive'])
    return {'n_features': total, 'n_alive': alive, 'n_annotated': annotated, 'n_ontology': ontology,
            'any_rate_alive': rates['annotation_rate'], 'ontology_rate_alive': rates['ontology_annotation_rate_alive'],
            'ontology_rate_all_features': rates['ontology_annotation_rate_all_features'],
            'frequency_ontology_floor_all_features': floors.get('freq', {}).get('ontology_all_features'),
            'floors': floors, 'features_with_capped_network_details': extra_unknown}


def catalogue_checks(catalogue, annotation, audit, n_positions=None, evaluation=None, modules=None):
    rows, summary = catalogue['features'], catalogue['summary']
    ids = [row['feature_idx'] for row in rows]
    alive_ids = {row['feature_idx'] for row in rows if not row['is_dead']}
    audit.equal('catalogue IDs unique', len(set(ids)), len(ids))
    audit.check('catalogue IDs contiguous', set(ids) == set(range(len(rows))))
    for key, expected in [('n_features', len(rows)), ('n_alive', len(alive_ids)), ('n_dead', len(rows)-len(alive_ids))]:
        audit.equal('catalogue '+key, summary.get(key), expected)
    annotation_ids = {row['feature_idx'] for row in annotation['features']}
    audit.check('annotation IDs equal live catalogue IDs', annotation_ids == alive_ids)
    audit.equal('annotation total denominator equals catalogue', annotation['summary']['n_features_total'], len(rows))
    frequencies = np.asarray([row['activation_freq'] for row in rows])
    counts = np.asarray([row['fire_count'] for row in rows])
    audit.check('catalogue frequencies finite within unit interval', np.all(np.isfinite(frequencies) & (frequencies >= 0) & (frequencies <= 1)))
    audit.check('catalogue dead flags match zero firing counts', all(row['is_dead'] == (row['fire_count'] == 0) for row in rows))
    if n_positions is not None:
        audit.equal('frequency equals fire count / all positions', frequencies, counts/n_positions)
    aligned = sum(bool(row.get('is_svd_aligned', False)) for row in rows if not row['is_dead'])
    audit.equal('catalogue SVD-aligned count', summary.get('n_svd_aligned'), aligned)
    audit.equal('catalogue SVD-novel alive denominator', summary.get('n_novel'), len(alive_ids)-aligned)
    if evaluation:
        audit.equal('evaluation feature width', evaluation['n_features'], len(rows))
        audit.equal('evaluation nonfinite activations rejected', evaluation['n_nonfinite_clipped'], 0)
        audit.equal('evaluation all-row dead count', evaluation['dead_all_rows'], len(rows)-len(alive_ids))
        audit.equal('evaluation all-row dead fraction', evaluation['dead_frac_all_rows'], 1-len(alive_ids)/len(rows))
        audit.equal('evaluation mean active per position', evaluation['mean_active_per_position'], frequencies.sum())
        audit.equal('evaluation firing quantiles', evaluation['firing_freq_quantiles_1_10_50_90_99'],
                    np.quantile(frequencies[frequencies > 0], [.01, .1, .5, .9, .99]) if np.any(frequencies > 0) else np.zeros(5))
        audit.equal('evaluation share firing >10%', evaluation['share_features_firing_gt_10pct'], float((frequencies > .1).mean()))
        audit.equal('evaluation share alive firing <1e-4', evaluation['share_features_firing_lt_1e-4'], float(((frequencies < 1e-4) & (frequencies > 0)).mean()))
        audit.check('held-out inactive count is bounded and at least all-row count',
                    len(rows)-len(alive_ids) <= evaluation['dead_val_reextracted'] <= len(rows))
    if modules:
        member_ids = [int(index) for module in modules['modules'] for index in module['features']]
        audit.equal('module feature IDs unique across communities', len(set(member_ids)), len(member_ids))
        audit.check('module members alive', set(member_ids) <= alive_ids)
        for module in modules['modules']:
            audit.equal(f'module {module["module_id"]} size', module['n_features'], len(module['features']))
            audit.check(f'module {module["module_id"]} minimum size', len(module['features']) >= 3)
        module_summary = modules['summary']
        for key, value in [('n_modules', len(modules['modules'])), ('n_features', len(rows)),
                           ('n_alive_features', len(alive_ids)), ('n_features_in_modules', len(member_ids)),
                           ('module_coverage', len(member_ids)/max(len(alive_ids), 1))]:
            audit.equal('module '+key, module_summary[key], value)
        assignment = {str(index): module['module_id'] for module in modules['modules'] for index in module['features']}
        audit.equal('module assignment equals explicit member lists', modules['module_assignment'], assignment)
    return {'n_svd_aligned': aligned, 'n_modules': len(modules['modules']) if modules else None,
            'variance_explained_reextracted': evaluation.get('variance_explained_reextracted') if evaluation else None,
            'dead_fraction_heldout': evaluation['dead_val_reextracted']/len(rows) if evaluation else None,
            'dead_fraction_all_rows': 1-len(alive_ids)/len(rows)}


def circuit_from_source_counts(rows, n_boot, seed):
    counts = np.asarray([[row['shared'], row['eligible'], row['expected_shared']] for row in rows], float)
    if not len(counts) or counts.shape[1] != 3 or not np.isfinite(counts).all():
        raise ValueError('Invalid source-level circuit counts')
    if np.any(counts < 0) or np.any(counts[:, 0] > counts[:, 1]) or np.any(counts[:, 2] > counts[:, 1]):
        raise ValueError('Source counts violate count/probability bounds')
    shared, eligible, expected = counts.sum(0)
    if eligible <= 0:
        raise ValueError('No annotated edges')
    point = np.array([shared, expected, shared-expected])/eligible
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(n_boot):
        a, denominator, b = counts[rng.integers(0, len(counts), len(counts))].sum(0)
        if denominator:
            samples.append([a/denominator, b/denominator, (a-b)/denominator])
    bounds = np.quantile(samples, [.025, .975], axis=0)
    return {'n_annotated_edges': int(eligible), 'n_shared': int(shared), 'n_source_clusters': len(rows),
            'observed': float(point[0]), 'same_source_random_target': float(point[1]), 'excess': float(point[2]),
            'source_bootstrap': {key: {'estimate': float(point[index]), 'ci95': bounds[:, index].tolist()}
                                 for index, key in enumerate(['observed', 'baseline', 'excess'])}}


def residual_metrics(eigenvalues, rms, mean, scalars):
    eig, rms, mean, scalars = [np.asarray(value, float) for value in [eigenvalues, rms, mean, scalars]]
    if not all(np.isfinite(value).all() for value in [eig, rms, mean, scalars]):
        raise ValueError('Nonfinite residual moments')
    if len(eig) != len(rms) or len(eig) != len(mean) or len(scalars) < 3 or np.any(eig < 0):
        raise ValueError('Invalid residual dimensions')
    eig = np.sort(eig)[::-1]
    width, trace = len(eig), eig.sum()
    participation = trace**2/np.dot(eig, eig) if trace else 0.
    median_rms = float(np.median(rms))
    norm_mean = float(np.linalg.norm(mean))
    cell_median = float(scalars[3]) if len(scalars) > 3 else None
    return {'width': width, 'n_gene_positions': int(scalars[0]),
            'centered_covariance_trace': float(trace), 'centered_covariance_pr': float(participation),
            'centered_covariance_pr_fraction_width': float(participation/width),
            'top1_variance_share': float(eig[0]/trace) if trace else None,
            'top10_variance_share': float(eig[:10].sum()/trace) if trace else None,
            'rms_max': float(rms.max()), 'rms_median': median_rms,
            'rms_max_over_median': float(rms.max()/median_rms) if median_rms else None,
            'n_dimensions_rms_gt_10x_median': int((rms > 10*median_rms).sum()),
            'max_abs_activation': float(scalars[2]), 'mean_row_l2_norm': float(scalars[1]),
            'mean_vector_l2_norm': norm_mean,
            'mean_vector_norm_over_mean_row_norm': norm_mean/float(scalars[1]) if scalars[1] else None,
            'median_of_cell_median_abs_activation': cell_median,
            'max_abs_over_median_of_cell_medians': float(scalars[2]/cell_median) if cell_median else None,
            'moment_centered_trace': float(np.dot(rms, rms)-np.dot(mean, mean))}


def reference_family(directory, universe, minimum, maximum, audit):
    """Rebuild test-family membership after universe intersection, never score features."""
    identifiers, counts = set(), Counter()
    for source, name in [('GO_BP', 'go_bp.json'), ('KEGG', 'kegg.json'), ('Reactome', 'reactome.json')]:
        for term, genes in audit.read(directory/name).items():
            if minimum <= len({str(g).upper() for g in genes} & universe) <= maximum:
                identifiers.add(source+'::'+term)
                counts[source] += 1
    trrust_path = directory/'trrust_human.tsv'
    audit.hash(trrust_path)
    regulons, regulatory = {}, set()
    for line in trrust_path.read_text().splitlines():
        fields = line.split('\t')
        if len(fields) >= 2:
            factor, target = fields[0].upper(), fields[1].upper()
            regulons.setdefault(factor, set()).add(target)
            if factor != target and factor in universe and target in universe:
                regulatory.add(tuple(sorted([factor, target])))
    for factor, genes in regulons.items():
        if minimum <= len(genes & universe) <= maximum:
            identifiers.add('TRRUST::'+factor+' targets')
            counts['TRRUST'] += 1
    pairs_paths = list(directory.glob('string_v*_pairs.tsv.gz'))
    if len(pairs_paths) != 1:
        raise ValueError('Need one unambiguous saved STRING library')
    audit.hash(pairs_paths[0])
    pairs = set()
    with gzip.open(pairs_paths[0], 'rt') as handle:
        for line in handle:
            a, b = line.rstrip('\n').split('\t')
            a, b = a.upper(), b.upper()
            if a != b and a in universe and b in universe:
                pairs.add(tuple(sorted([a, b])))
    return identifiers, dict(counts), len(pairs), len(regulatory)


def audit_layers(root, audit, models, layers):
    outputs = {}
    base_rows = root/'runs/2b_atlas/row_gene_names.json'
    if not audit.require(base_rows):
        return outputs
    universe = {str(gene).upper() for gene in audit.read(base_rows) if gene and gene != '<CELL>'}
    families = {}
    for model in models:
        run = root/'runs'/('2b_atlas' if model == '2b' else '27b_atlas')
        out = []
        row_path = run/'row_gene_names.json'
        n_positions = len(audit.read(row_path)) if row_path.exists() else None
        for layer in layers:
            audit.context = f'{model}/L{layer:02d}: '
            annotation = root/'results/annotations'/model/f'ann_v2_L{layer:02d}.json'
            catalogue = run/f'feature_catalog_L{layer:02d}.json'
            required = [annotation, catalogue]
            evaluation = root/'results'/f'2b_eval_L{layer:02d}.json'
            module_path = run/f'coactivation_modules_L{layer:02d}.json'
            if model == '2b':
                required += [evaluation, module_path]
            available = [audit.require(path) for path in required]
            if not all(available):
                continue
            try:
                ann, catalog = audit.read(annotation), audit.read(catalogue)
                audit.equal('full scorer version', ann['summary']['scorer_version'], '3-plan-aligned')
                audit.equal('full libraries rather than pilot', ann['summary']['legacy_libraries'], False)
                family_key = (ann['summary']['min_term'], ann['summary']['max_term'])
                if family_key not in families:
                    families[family_key] = reference_family(root/'ontologies_v2', universe, *family_key, audit)
                known, counts, string_pairs, regulatory_pairs = families[family_key]
                audit.equal('universe restriction', ann['summary']['universe_size'], len(universe))
                audit.equal('full family includes all named terms plus three tests', ann['summary']['tests_per_feature'], sum(counts.values())+3)
                audit.equal('retained named families', ann['summary']['terms_by_source'], counts)
                audit.equal('STRING pair background restriction', ann['summary']['string_pairs_in_universe'], string_pairs)
                audit.equal('TRRUST pair background restriction', ann['summary']['trrust_pairs_in_universe'], regulatory_pairs)
                audit.check('all significant IDs belong to retained full family',
                            all(set(row.get('significant_term_ids', [])) <= known for row in ann['features']))
                record = annotation_counts(ann, audit)
                record.update(catalogue_checks(catalog, ann, audit, n_positions,
                    audit.read(evaluation) if model == '2b' else None,
                    audit.read(module_path) if module_path.exists() else None))
                record['layer'] = layer
                out.append(record)
                del ann, catalog
            except (KeyError, TypeError, ValueError) as error:
                audit.check('saved schema is complete', False, str(error), None)
        outputs[model] = {'n_layers_completed': len(out), 'layers': out}
    return outputs


def audit_circuits(root, audit):
    path = root/'results/circuit_baseline.json'
    audit.context = 'circuits: '
    if not audit.require(path):
        return {}
    saved = audit.read(path)
    expected = circuit_from_source_counts(saved['source_counts'], saved['n_boot'], saved['seed'])
    identities = [(row['source_layer'], row['source_feature']) for row in saved['source_counts']]
    audit.equal('source units unique', len(identities), len(set(identities)))
    for key, value in expected.items():
        if key != 'source_bootstrap':
            audit.equal(key, saved[key], value)
    for key, value in expected['source_bootstrap'].items():
        for metric in ['estimate', 'ci95']:
            audit.equal(f'source bootstrap {key} {metric}', saved['source_bootstrap'][key][metric], value[metric])
    return expected


def audit_residual(root, audit):
    audit.context = 'residual: '
    path = root/'results/residual_summary.json'
    if not audit.require(path):
        return {}
    summary, outputs = audit.read(path), {}
    for model, stats_name, run_name, expected_layers, trained_layers in [
        ('2b', 'atlas-extract', '2b_atlas', list(range(26)), set(range(26))),
        ('27b_matched', 'scale-extract', '27b_matched', list(range(46)), {0, 7, 14, 24, 31, 38, 45})]:
        statistics = root/'results'/f'{stats_name}_residual_stats.npz'
        if not audit.require(statistics):
            continue
        audit.hash(statistics)
        by_layer = {row['layer']: row for row in summary[model]['layers']}
        audit.context = f'residual/{model}: '
        audit.equal('unique residual layer rows', len(by_layer), len(summary[model]['layers']))
        audit.equal(model+' residual layer coverage', sorted(by_layer), expected_layers)
        with np.load(statistics, allow_pickle=False) as arrays:
            for layer, saved in by_layer.items():
                audit.context = f'residual/{model}/L{layer:02d}: '
                values = residual_metrics(*(arrays[f'{kind}_L{layer:02d}'] for kind in ['eig', 'rms', 'mean', 'scalars']))
                for key, value in values.items():
                    if key != 'moment_centered_trace':
                        audit.equal(key, saved.get(key), value)
                audit.equal('covariance trace equals E||x||² - ||E x||²', values['centered_covariance_trace'], values['moment_centered_trace'])
                trained_path = root/'runs'/run_name/'sae'/f'train_L{layer:02d}.json'
                trained = None
                # Spectra cover every tap; matched 27B dictionaries cover only
                # the seven prespecified relative-depth taps. Missing metrics at
                # other taps are intentional and must remain explicitly absent.
                if layer in trained_layers and audit.require(trained_path):
                    trained = audit.read(trained_path)
                    features = trained['n_features']
                    original_dead = trained.get('dead_frac', trained.get('dead_features', 0)/features)
                    audit.equal('original held-out dead fraction join', saved['dead_fraction_heldout_original'], original_dead)
                    audit.equal('original variance-explained join', saved['variance_explained_original'], trained.get('variance_explained'))
                elif layer not in trained_layers:
                    for metric in ['n_sae_features', 'dead_fraction_heldout_original', 'dead_fraction_heldout',
                                   'dead_fraction_all_rows', 'variance_explained_original', 'variance_explained_reextracted']:
                        audit.equal('untrained tap has no SAE join: '+metric, saved.get(metric), None)
                if model == '2b':
                    evaluated_path = root/'results'/f'2b_eval_L{layer:02d}.json'
                    if audit.require(evaluated_path):
                        evaluated = audit.read(evaluated_path)
                        audit.equal('re-extracted held-out dead fraction join', saved['dead_fraction_heldout'], evaluated['dead_val_reextracted']/evaluated['n_features'])
                        audit.equal('all-row dead fraction join', saved['dead_fraction_all_rows'], evaluated['dead_frac_all_rows'])
                        audit.equal('re-extracted variance-explained join', saved['variance_explained_reextracted'], evaluated['variance_explained_reextracted'])
                elif trained is not None:
                    audit.equal('all-row dead fraction join', saved['dead_fraction_all_rows'], trained['dead_frac_all_rows'])
        for dead, correlations in summary[model]['descriptive_rank_correlations'].items():
            for measure, record in correlations.items():
                pairs = [(row[dead], row[measure]) for row in by_layer.values() if row[dead] is not None and row[measure] is not None]
                audit.equal(f'{model} {dead}/{measure} layer count', record['n_layers'], len(pairs))
                expected = None
                if len(pairs) >= 3:
                    x, y = np.asarray(pairs).T
                    if np.std(rankdata(x)) and np.std(rankdata(y)):
                        expected = float(np.corrcoef(rankdata(x), rankdata(y))[0, 1])
                audit.equal(f'{model} {dead}/{measure} descriptive rho', record['spearman_rho'], expected)
                audit.check('dependent-layer correlations contain no p-value claim', not any(k in record for k in ['p', 'pvalue', 'p_value']))
        outputs[model] = {'n_layers': len(by_layer)}
    return outputs


def audit_phase8_outputs(root, audit):
    from audit_phase8_saved import audit_phase8
    outputs = {}
    for cohort in ['k562', 'rpe1']:
        audit.context = f'phase8/{cohort}: '
        outputs[cohort] = audit_phase8(root/'results'/f'phase8_{cohort}'/'statistics',
            root/'runs/2b_atlas', root/'ontologies_v2/trrust_human.tsv', audit)
    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True, help='Experiment root containing runs/ and results/.')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--mode', choices=['partial', 'final'], default='final')
    parser.add_argument('--models', nargs='+', choices=['2b', '27b_x4'], default=['2b', '27b_x4'])
    parser.add_argument('--layers', nargs='+', type=int, default=list(range(26)), help='Partial mode only; final always requires all 26.')
    parser.add_argument('--components', nargs='+', choices=['layers', 'circuits', 'residual', 'phase8'],
                        default=['layers', 'circuits', 'residual', 'phase8'])
    args = parser.parse_args()
    if args.mode == 'final' and args.layers != list(range(26)):
        parser.error('Final mode requires all 26 layers; use partial mode for interim checks.')
    audit, results = Audit(), {}
    for name, operation in [('layers', lambda: audit_layers(args.root, audit, args.models, args.layers)),
                             ('circuits', lambda: audit_circuits(args.root, audit)),
                             ('residual', lambda: audit_residual(args.root, audit)),
                             ('phase8', lambda: audit_phase8_outputs(args.root, audit))]:
        if name in args.components:
            try:
                results[name] = operation()
            except (KeyError, TypeError, ValueError, OSError) as error:
                audit.check(name+' complete readable schema', False, str(error), None)
    audit.hash(Path(__file__))
    if 'phase8' in args.components:
        phase8_helper = Path(__file__).with_name('audit_phase8_saved.py')
        if audit.require(phase8_helper):
            audit.hash(phase8_helper)
    max_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024)
    report = {'audit': 'V1 independent saved experimental arithmetic/provenance',
              'mode': args.mode, 'arithmetic_passed': audit.passed,
              'requested_components': args.components, 'requested_models': args.models, 'requested_layers': args.layers,
              'complete': not audit.pending, 'passed': audit.passed and not audit.pending,
              'n_checks': audit.n_checks, 'findings': audit.failed, 'pending_files': audit.pending,
              'results': results, 'source_sha256': audit.sources, 'peak_rss_bytes': max_rss,
              'limitations': ['No model/SAE/refit calculations or raw activation matrices are opened.',
                              'Complete named IDs determine all-feature ontology rates; capped term details only bound undisplayed network-source rates.',
                              'Stored chance seed-rate arithmetic is checked; null gene sets and their p-values are not independently regenerated.',
                              'Global BH rate is checked at the saved threshold; full threshold regeneration requires unsaved complete p-value vectors.',
                              'Circuit bootstrap is independently reproduced from saved source-level counts, not fresh model interventions.',
                              'Phase8 response p-values are retained inputs. Full-column BH, specificity Fisher/full-TF BH, self-excluded retrieval, and TF-bootstrap means/intervals are recomputed; per-item intervals receive only finite/order checks.']}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k: report[k] for k in ['mode', 'arithmetic_passed', 'complete', 'n_checks', 'findings', 'peak_rss_bytes']}))
    raise SystemExit(0 if audit.passed and (args.mode == 'partial' or not audit.pending) else 1)


if __name__ == '__main__':
    main()
