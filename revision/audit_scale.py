#!/usr/bin/env python3
"""Audit and report seven fixed-capacity 2B/27B depth pairs, using saved outputs only."""
import argparse
import gc
import json
from pathlib import Path
import resource
import sys

import numpy as np

from audit_experimental_saved import Audit, annotation_counts, catalogue_checks
from audit_annotation_dense import references
from audit_sensitivity import dense_sample


LAYERS_2B = [0, 4, 8, 13, 17, 21, 25]
LAYERS_27B = [0, 7, 14, 24, 31, 38, 45]


def paired_contrast(first, second):
    if first['n_features'] != second['n_features']:
        raise ValueError('Absolute dictionary capacities must match')
    excess_a = first['ontology_all_features']-first['ontology_frequency_floor_all_features']
    excess_b = second['ontology_all_features']-second['ontology_frequency_floor_all_features']
    return {'2b': first, '27b_matched': second,
            'ontology_gap_27b_minus_2b': second['ontology_all_features']-first['ontology_all_features'],
            'ontology_excess_gap_27b_minus_2b': excess_b-excess_a,
            'heldout_inactive_gap_27b_minus_2b': second['heldout_inactive_fraction']-first['heldout_inactive_fraction'],
            'all_row_inactive_gap_27b_minus_2b': second['all_row_inactive_fraction']-first['all_row_inactive_fraction']}


def model_row(root, model, layer, universe, reference, audit):
    audit.context = f'{model}/L{layer}: '
    run = root/'runs'/('2b_atlas' if model == '2b' else '27b_matched')
    directory = root/'results/annotations'/('2b' if model == '2b' else '27b_matched')
    annotation = audit.read(directory/f'ann_v2_L{layer:02d}.json')
    catalogue = audit.read(run/f'feature_catalog_L{layer:02d}.json')
    training = audit.read(run/'sae'/f'train_L{layer:02d}.json')
    evaluation = audit.read(root/'results'/f'2b_eval_L{layer:02d}.json') if model == '2b' else None
    counts = annotation_counts(annotation, audit)
    catalogue_checks(catalogue, annotation, audit, 614400, evaluation)
    sample = dense_sample(catalogue, annotation, universe, reference, audit)
    audit.equal('full library flag', annotation['summary']['legacy_libraries'], False)
    audit.equal('model universe size', annotation['summary']['universe_size'], len(universe))
    for key, value in [('n_rows', 614400), ('n_features', 9216), ('d_model', 2304 if model == '2b' else 4608),
                       ('k', 32), ('epochs', 6)]: audit.equal('training '+key, training[key], value)
    audit.equal('catalogue absolute capacity', counts['n_features'], 9216)
    measurement = evaluation if model == '2b' else training
    if model != '2b':
        audit.equal('training seed', training['seed'], 0)
        audit.equal('input nonfinite rejection', training['n_nonfinite_clipped'], 0)
        audit.check('six finite epoch losses', len(training['epoch_train_mse']) == 6 and np.isfinite(training['epoch_train_mse']).all())
    frequencies = np.asarray([row['activation_freq'] for row in catalogue['features']])
    firing = {'dead_all_rows': int((frequencies == 0).sum()), 'dead_frac_all_rows': float((frequencies == 0).mean()),
              'mean_active_per_position': float(frequencies.sum()),
              'firing_freq_quantiles_1_10_50_90_99': np.quantile(frequencies[frequencies > 0], [.01, .1, .5, .9, .99]).tolist(),
              'share_features_firing_gt_10pct': float((frequencies > .1).mean()),
              'share_features_firing_lt_1e-4': float(((frequencies < 1e-4) & (frequencies > 0)).mean())}
    for key, value in firing.items(): audit.equal('metric/catalogue '+key, measurement[key], value)
    heldout_dead = measurement['dead_val_reextracted'] if model == '2b' else training['dead_features']
    audit.check('heldout inactivity at least all-row inactivity', firing['dead_all_rows'] <= heldout_dead <= 9216)
    variance = measurement['variance_explained_reextracted'] if model == '2b' else training['variance_explained']
    for filename in ['row_gene_names.json', 'row_cell_ids.npy']:
        audit.hash(run/filename); audit.hash(root/'runs/2b_atlas'/filename)
        audit.equal('same exact atlas rows '+filename, audit.sources[str(run/filename)], audit.sources[str(root/'runs/2b_atlas'/filename)])
    return {'model': model, 'layer': layer, 'relative_depth': layer/(25 if model == '2b' else 45),
            'd_model': training['d_model'], 'n_features': counts['n_features'], 'expansion': counts['n_features']/training['d_model'],
            'n_catalogue_active': counts['n_alive'], 'n_catalogue_inactive': firing['dead_all_rows'],
            'heldout_inactive': int(heldout_dead), 'heldout_inactive_fraction': heldout_dead/9216,
            'all_row_inactive_fraction': firing['dead_frac_all_rows'],
            'heldout_variance_explained': variance,
            'variance_metric_source': 'fresh 2B re-evaluation' if model == '2b' else 'new matched-capacity training heldout',
            'mean_decoder_cos': training['mean_decoder_cos'], 'firing': firing,
            'n_ontology': counts['n_ontology'], 'n_any_evidence': counts['n_annotated'],
            'ontology_all_features': counts['ontology_rate_all_features'], 'ontology_catalogue_active': counts['ontology_rate_alive'],
            'any_evidence_all_features': counts['any_rate_alive']*counts['n_alive']/9216,
            'ontology_frequency_floor_all_features': counts['floors']['freq']['ontology_all_features'],
            'ontology_frequency_floor_catalogue_active': counts['floors']['freq']['ontology_alive'],
            'floors': counts['floors'], 'full_tests_per_feature': annotation['summary']['tests_per_feature'],
            'dense_sample_feature_ids': sample,
            'training_configuration': {key: training.get(key) for key in ['epochs', 'seed', 'lr', 'batch_size', 'k', 'n_rows']}}


def optional_e4(root, audit):
    path = root/'results/dropmassive_comparison.json'
    if not path.exists(): return {'status': 'not available; optional experiment'}
    saved = audit.read(path)
    # Keep this reporting stage bounded: the scientific projection comparator
    # remains the producer, and only completed saved metric arithmetic is checked.
    records = []
    for layer in [24, 38]:
        original_path = root/'runs/27b_matched/sae'/f'train_L{layer:02d}.json'
        projected_path = root/'runs/27b_matched/sae_dropmassive'/f'train_L{layer:02d}.json'
        if not projected_path.exists():
            records.append({'layer': layer, 'status': 'pending'}); continue
        original, projected = audit.read(original_path), audit.read(projected_path)
        if projected.get('status') == 'skipped_no_dimensions':
            records.append({'layer': layer, 'status': 'skip: no prespecified dimensions'}); continue
        audit.equal('E4 absolute capacity', projected['n_features'], original['n_features'])
        audit.equal('E4 projected variance reference', projected['input_projection']['variance_reference'], 'projected input after zeroing selected columns')
        records.append({'layer': layer, 'status': 'complete', 'removed_dimensions': projected['input_projection']['removed_dimensions'],
                        'dead_heldout_original': original['dead_frac'], 'dead_heldout_projected': projected['dead_frac'],
                        'dead_heldout_difference_projected_minus_original': projected['dead_frac']-original['dead_frac'],
                        'dead_all_rows_original': original['dead_frac_all_rows'], 'dead_all_rows_projected': projected['dead_frac_all_rows'],
                        'dead_all_rows_difference_projected_minus_original': projected['dead_frac_all_rows']-original['dead_frac_all_rows'],
                        'variance_explained_projected_reference': projected['variance_explained']})
    return {'status': 'saved optional comparator read', 'records': records,
            'producer_comparison': saved, 'variance_caveat': 'Projected-input VE and original-input VE use different variance references.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True, help='Source-hashed scale_reporting.json')
    args = parser.parse_args(); audit = Audit()
    universe = {gene.upper() for gene in audit.read(args.root/'runs/2b_atlas/row_gene_names.json') if gene != '<CELL>'}
    reference = references(args.root/'ontologies_v2', universe, 5, 500)
    for path in reference[-1]: audit.hash(path)
    pairs = []
    for first, second in zip(LAYERS_2B, LAYERS_27B):
        a = model_row(args.root, '2b', first, universe, reference[:4], audit); gc.collect()
        b = model_row(args.root, '27b_matched', second, universe, reference[:4], audit); gc.collect()
        pairs.append(paired_contrast(a, b))
        print('audited matched pair', first, second, flush=True)
    e4 = optional_e4(args.root, audit)
    audit.hash(Path(__file__)); audit.hash(Path(__file__).with_name('audit_sensitivity.py'))
    report = {'schema': 'c2s-scale-reporting-v1', 'audit': 'Independent seven-layer matched-capacity annotation/metric audit',
              'passed': audit.passed, 'n_checks': audit.n_checks, 'findings': audit.failed, 'n_pairs': len(pairs),
              'pairs': pairs, 'optional_e4': e4, 'source_sha256': audit.sources,
              'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform == 'darwin' else 1024),
              'units': {'capacity': '9,216 dictionary features; expansion 4 for 2B, expansion 2 for 27B.',
                        'training': 'Six saved epochs; original 2B optimizer/seed fields remain unreported where absent.',
                        'ontology_primary': 'Complete named ontology/regulon IDs divided by all 9,216 features; active-only denominators are separately saved.',
                        'floor': 'Saved frequency-null ontology rate rescaled by active/all-feature ratio.',
                        'inactivity': 'Heldout inactive and never active across all 614,400 positions are distinct.',
                        'variance': 'Heldout VE uses the residual-activation variance of each model; projected E4 uses projected-input variance.'},
              'limitations': ['Six deterministic features per annotation receive independent dense full-family p/q checks; all other rows receive complete-ID/count/denominator checks.',
                              'No SAE fits or reconstruction forwards are rerun; saved reconstruction statistics are reported.',
                              'Fixed absolute capacity, approximate relative depths and six epochs do not match every aspect of optimization or checkpoint training.',
                              'Floor draws and single model seeds do not establish between-training-run uncertainty.']}
    args.out.parent.mkdir(parents=True, exist_ok=True); args.out.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({key: report[key] for key in ['passed', 'n_checks', 'findings', 'n_pairs', 'peak_rss_bytes']}))
    raise SystemExit(0 if audit.passed else 1)


if __name__ == '__main__': main()
