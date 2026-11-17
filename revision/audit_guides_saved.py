#!/usr/bin/env python3
"""Independent small arithmetic audit of descriptive saved guide-pair summaries."""
import argparse
from collections import Counter
from itertools import combinations
import json
from pathlib import Path

import numpy as np

from audit_experimental_saved import Audit


def pair_values(first, second, responding_first, responding_second):
    first, second = np.asarray(first, float), np.asarray(second, float)
    a, b = np.asarray(responding_first, bool), np.asarray(responding_second, bool)
    joint, union = a & b, a | b
    supported = (np.abs(first) > .5) & (np.abs(second) > .5)
    agreement = np.sign(first) == np.sign(second)
    return {'n_features': len(first), 'effect_pearson': float(np.corrcoef(first, second)[0, 1]) if first.std() and second.std() else None,
            'n_joint_responses': int(joint.sum()), 'n_union_responses': int(union.sum()),
            'response_jaccard': float(joint.sum()/union.sum()) if union.any() else None,
            'joint_response_sign_agreement': float(agreement[joint].mean()) if joint.any() else None,
            'n_both_abs_effect_above_threshold': int(supported.sum()),
            'effect_supported_sign_agreement': float(agreement[supported].mean()) if supported.any() else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    root, audit = args.root, Audit()
    directory = root/'results/phase8_thp1_pooled/statistics'
    saved = audit.read(directory/'guide_concordance.json')
    design = audit.read(root/'data/papalexi/c6_pooled_rep1tx/design.json')
    rows = audit.read(root/'results/phase8_thp1_pooled/rows.json')
    for source, digest in saved['source_sha256'].items():
        audit.hash(source); audit.equal('source SHA '+source, audit.sources[source], digest)
    controls = [index for index, row in enumerate(rows) if row['kind'] == 'control']
    audit.equal('fixed control row order', saved['control_extracted_row_indices'], controls)
    audit.equal('fixed control cell IDs', saved['control_h5ad_cell_indices'], [rows[index]['cell_index'] for index in controls])
    audit.equal('fixed control denominator', saved['n_fixed_control_cells'], len(controls))
    audit.equal('single recorded batch', saved['batch'], ['rep1-tx'])
    groups = {group['group']: group for group in design['groups'] if group['kind'] == 'tf'}
    layers = sorted({record['layer'] for record in saved['records']})
    audit.equal('all TF/layer profiles retained', len(saved['records']), len(groups)*len(layers))
    summary = []
    for layer in layers:
        path = directory/f'guide_statistics_L{layer:02d}.npz'; audit.hash(path)
        with np.load(path, allow_pickle=False) as archive:
            statistics = {(group, guide): (archive['effect'][index], archive['responding'][index])
                          for index, (group, guide) in enumerate(zip(archive['groups'].astype(str), archive['guides'].astype(str)))}
        pair_rows, selected_count, evaluable_count = [], 0, 0
        unevaluable = []
        for record in [row for row in saved['records'] if row['layer'] == layer]:
            name, expected = record['group'], groups[record['group']]
            audit.equal(name+' group cell count', record['n_cells'], len(expected['cells']))
            audit.equal(name+' selected guides', record['n_guides_selected'], len(expected['guide_counts_selected']))
            available = []
            for guide in record['guides']:
                identity = guide['guide']; count = expected['guide_counts_selected'][identity]
                audit.equal(identity+' cell count', guide['n_cells'], count)
                status = 'evaluable' if count >= saved['minimum_cells_per_guide'] else 'insufficient cells'
                audit.equal(identity+' availability status', guide['status'], status)
                selected_count += 1
                if status == 'evaluable':
                    available.append(identity); evaluable_count += 1
                    effect, response = statistics[name, identity]
                    audit.equal(identity+' response count', guide['n_responding'], int(response.sum()))
                    retrieval = guide['gene_retrieval']
                    audit.equal(identity+' prevalence arithmetic', retrieval['prevalence'], retrieval['n_positive']/retrieval['n'])
                    if retrieval['average_precision'] is not None:
                        audit.equal(identity+' prevalence lift', retrieval['ap_lift'], retrieval['average_precision']/retrieval['prevalence'])
                        audit.equal(identity+' exact-null lift', retrieval['lift_over_random_expected_ap'], retrieval['average_precision']/retrieval['random_label_expected_ap'])
                else:
                    unevaluable.append({'guide': identity, 'n_cells': count})
                    audit.check(identity+' unavailable guide not scored', (name, identity) not in statistics)
            audit.equal(name+' evaluable guides', record['n_guides_evaluable'], len(available))
            expected_pairs = {tuple(pair) for pair in combinations(sorted(available), 2)}
            pairs = {(pair['first_guide'], pair['second_guide']): pair for pair in record['pairs']}
            audit.equal(name+' unique pairs', len(pairs), len(record['pairs']))
            audit.equal(name+' complete evaluable-pair set', set(pairs), expected_pairs)
            for identities, pair in pairs.items():
                a, ar = statistics[name, identities[0]]; b, br = statistics[name, identities[1]]
                recomputed = pair_values(a, b, ar, br)
                for key, value in recomputed.items():
                    # Effect arrays are stored float32; original correlations use
                    # float64 before saving. Compare within that storage precision.
                    audit.equal(name+'/'+str(identities)+' '+key, pair[key], value, atol=1e-7)
                pair_rows.append(pair)
        descriptive = {'layer': layer, 'n_selected_guide_records': selected_count, 'n_evaluable_guide_records': evaluable_count,
                       'n_pairs': len(pair_rows), 'unevaluable': unevaluable,
                       'n_pairs_with_joint_responses': sum(row['n_joint_responses'] > 0 for row in pair_rows),
                       'n_pairs_with_empty_response_union': sum(row['n_union_responses'] == 0 for row in pair_rows)}
        for key in ['effect_pearson', 'response_jaccard', 'effect_supported_sign_agreement']:
            values = [row[key] for row in pair_rows if row[key] is not None]
            descriptive['median_'+key] = float(np.median(values)) if values else None
        summary.append(descriptive)
    audit.hash(Path(__file__))
    report = {'audit': 'Independent descriptive guide-pair saved-array arithmetic', 'passed': audit.passed,
              'n_checks': audit.n_checks, 'findings': audit.failed, 'summary': summary, 'source_sha256': audit.sources,
              'limitations': ['Stored float32 guide effects and exact saved response masks are checked; raw per-guide MW/BH tests are not rerun.',
                              'Guide pairs share controls, factors and feature columns; no independent-pair significance test or interval is calculated.',
                              'Guide-level gene AP ratios are checked arithmetically; gene score rankings are not regenerated.']}
    args.out.parent.mkdir(parents=True, exist_ok=True); args.out.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({key: report[key] for key in ['passed', 'n_checks', 'findings', 'summary']}))
    raise SystemExit(0 if audit.passed else 1)


if __name__ == '__main__': main()
