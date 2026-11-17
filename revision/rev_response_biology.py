"""E-C5: biological descriptions of measured knockdown feature responses.

Use the already-saved response masks; this script performs no new hypothesis tests. A feature is
"target labelled" when its catalogue top genes overlap that factor's TRRUST downstream targets, excluding the knocked-down factor itself. This is the
same associative label used for target retrieval, not evidence that the feature regulates a gene.
"Generic programme" is an explicit category label (below), independent of the target label.
Categories overlap. The mutually exclusive three-way summary gives target labels first priority.
Named features are selected by absolute response effect, with feature index breaking ties; no
activation/response direction or regulatory sign is inferred. PHB2, GATA1 and MYC are predefined
worked examples where present, and absent factors are recorded explicitly.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
from rev_cpu_analysis import CATEGORIES, annotation_terms, category_for_term


GENERIC_CATEGORIES = {'translation and ribosome', 'RNA processing', 'protein folding and degradation',
                      'cell cycle and mitosis', 'DNA replication and repair',
                      'energy and other metabolism', 'stress and apoptosis'}


def gene_names(feature):
    return {str(g['gene_name'] if isinstance(g, dict) else g).upper()
            for g in feature.get('top_genes', [])}


def feature_categories(annotations):
    categories = {}
    for feature, annotation in annotations.items():
        if 'significant_term_ids' not in annotation:
            raise ValueError('Response biology requires complete significant_term_ids from the current scorer')
        categories[feature] = {category_for_term(term) for _, term in annotation_terms(annotation)}
    return categories


def response_profile(responding, effect, q, catalogue, annotations, known_targets, max_examples=3,
                     categories=None, *, perturbed_gene):
    """Describe saved responses with downstream labels after excluding the intervention target.

    ``perturbed_gene`` is required so callers cannot accidentally include a curated
    autoregulatory edge as downstream evidence. The exclusion applies to counts and
    named examples, and matches the specificity/retrieval label definition.
    """
    responding, effect, q = np.asarray(responding), np.asarray(effect, float), np.asarray(q, float)
    if responding.ndim!=1 or responding.shape!=effect.shape or q.shape!=effect.shape:
        raise ValueError('Response mask, effect and q vectors must have equal one-dimensional shapes')
    if not np.isfinite(effect).all() or not np.isfinite(q).all() or not np.isin(responding, [0, 1]).all():
        raise ValueError('Invalid response masks or nonfinite feature statistics')
    if (q<0).any() or (q>1).any():
        raise ValueError('Feature response q-values must be in [0,1]')
    catalogue = {int(row['feature_idx']): row for row in catalogue}
    if set(catalogue)!=set(range(len(effect))):
        raise ValueError('Catalogue must cover exactly the extracted feature columns')
    categories = feature_categories(annotations) if categories is None else categories
    known_targets = {gene.upper() for gene in known_targets} - {perturbed_gene.upper()}
    response_ids = np.flatnonzero(responding.astype(bool)).tolist()
    target_ids, generic_ids, unannotated = set(), set(), 0
    category_counts, target_category_counts = Counter(), Counter()
    for feature in response_ids:
        labels = categories.get(feature, set())
        category_counts.update(labels)
        if not labels:
            unannotated += 1
        if labels & GENERIC_CATEGORIES:
            generic_ids.add(feature)
        if gene_names(catalogue[feature]) & known_targets:
            target_ids.add(feature)
            target_category_counts.update(labels)
    examples = []
    for feature in sorted(response_ids, key=lambda index: (-abs(effect[index]), index))[:max_examples]:
        annotation = annotations.get(feature, {})
        examples.append({'feature_idx': feature, 'response_strength_abs_d': float(abs(effect[feature])),
                         'response_q': float(q[feature]), 'categories': sorted(categories.get(feature, set())),
                         'top_genes': [str(g['gene_name'] if isinstance(g, dict) else g)
                                       for g in catalogue[feature].get('top_genes', [])[:10]],
                         'matching_trrust_targets': sorted(gene_names(catalogue[feature]) & known_targets),
                         'significant_term_ids': annotation.get('significant_term_ids', []),
                         'displayed_significant_terms': [term for term in annotation.get('terms', [])
                                                        if term.get('source') in ('GO_BP', 'KEGG', 'Reactome', 'TRRUST')
                                                        and term.get('q', 1)<.05]})
    return {'n_responding': len(response_ids), 'n_target_labelled_responses': len(target_ids),
            'n_generic_programme_responses': len(generic_ids),
            'n_both_generic_and_target_labelled': len(generic_ids & target_ids),
            'n_ontology_unannotated_responses': unannotated,
            'category_counts': {category: category_counts[category] for category in CATEGORIES},
            'target_labelled_category_counts': {category: target_category_counts[category] for category in CATEGORIES},
            'exclusive_response_counts': {'target_labelled': len(target_ids),
                  'generic_programme_without_target_label': len(generic_ids-target_ids),
                  'other_or_unannotated_without_target_label': len(set(response_ids)-generic_ids-target_ids)},
            'named_features': examples}


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1<<20), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stats-dir', type=Path, required=True)
    parser.add_argument('--annotations-dir', type=Path, required=True)
    parser.add_argument('--catalog-dir', type=Path, required=True)
    parser.add_argument('--trrust', type=Path, required=True)
    parser.add_argument('--layers', nargs='+', type=int, default=[8, 16, 22])
    parser.add_argument('--example-tfs', nargs='+', default=['PHB2', 'GATA1', 'MYC'])
    parser.add_argument('--max-examples', type=int, default=3)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    targets = {}
    for line in args.trrust.read_text().splitlines():
        parts = line.split('\t')
        if len(parts)>=2:
            targets.setdefault(parts[0].upper(), set()).add(parts[1].upper())
    records_path = args.stats_dir/'regulatory_statistics.json'
    statistics = json.loads(records_path.read_text())
    source_paths, results = [records_path, args.trrust, Path(__file__), Path(__file__).with_name('rev_cpu_analysis.py')], []
    for layer in args.layers:
        array_path = args.stats_dir/f'feature_statistics_L{layer:02d}.npz'
        annotation_path = args.annotations_dir/f'ann_v2_L{layer:02d}.json'
        catalogue_path = args.catalog_dir/f'feature_catalog_L{layer:02d}.json'
        source_paths.extend([array_path, annotation_path, catalogue_path])
        annotation = {int(record['feature_idx']): record for record in json.loads(annotation_path.read_text())['features']}
        categories = feature_categories(annotation)
        catalogue = json.loads(catalogue_path.read_text())['features']
        family = {record['group']: record for record in statistics['groups']
                  if record['representation']=='model' and record['layer']==layer}
        with np.load(array_path, allow_pickle=False) as arrays:
            groups = arrays['groups'].astype(str).tolist()
            responding, effect, q = arrays['responding'], arrays['effect'], arrays['q']
        if len(set(groups))!=len(groups) or set(groups)!=set(family):
            raise ValueError(f'Feature-array groups and regulatory records differ at layer {layer}')
        if responding.shape!=effect.shape or q.shape!=effect.shape or effect.shape[0]!=len(groups):
            raise ValueError('Saved feature response matrices have incompatible shapes')
        for index, group in enumerate(groups):
            measured = family[group]
            known = targets.get(group.upper(), set()) if measured['kind']=='tf' else set()
            profile = response_profile(responding[index], effect[index], q[index], catalogue, annotation,
                                       known, args.max_examples, categories, perturbed_gene=group)
            if profile['n_responding']!=measured['n_responding']:
                raise ValueError(f'Response count differs from saved statistical record: {group}, L{layer}')
            results.append({'group': group, 'kind': measured['kind'], 'layer': layer,
                            'n_cells': measured['n_cells'], 'n_control_cells': measured['n_control_cells'],
                            'specificity': measured['specificity'], **profile})
    predefined = {factor: {'status': 'present' if any(r['group'].upper()==factor.upper() and r['kind']=='tf' for r in results) else 'not present among tested TFs',
                           'records': [r for r in results if r['group'].upper()==factor.upper() and r['kind']=='tf']}
                  for factor in args.example_tfs}
    output = {'experiment': 'E-C5', 'responses': results, 'predefined_worked_examples': predefined,
              'generic_programme_categories': sorted(GENERIC_CATEGORIES), 'category_order': CATEGORIES,
              'definitions': {'response_mask': 'Saved feature response decisions; no new tests',
                              'target_label': 'At least one catalogue top gene is a TRRUST downstream target of the knocked-down factor, excluding that factor itself',
                              'generic_programme': 'At least one named ontology category in the predefined generic category list',
                              'category_counts': 'Overlapping; a feature is counted once per category',
                              'exclusive_counts': 'Target-labelled first, generic-only second, all remaining responses third',
                              'examples': 'Largest absolute response d; ties broken by feature index',
                              'interpretation': 'Associative biological descriptions; no response-direction or regulatory-sign claim'},
              'source_sha256': {str(path): file_hash(path) for path in source_paths}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2, allow_nan=False))
    print(f'{args.out}: {len(results)} group/layer response profiles', flush=True)


if __name__=='__main__':
    main()
