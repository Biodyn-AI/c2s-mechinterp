#!/usr/bin/env python3
"""Recheck the named proteasome circuit edge and both endpoint annotations."""
import argparse
import json
from pathlib import Path

import numpy as np

from audit_experimental_saved import Audit
from audit_annotation_dense import references, dense_tests, EXTRA


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(); audit = Audit()
    baseline = audit.read(args.root/'results/circuit_baseline.json')
    example = baseline['strong_shared_examples'][0]
    identity = (7, 254, 9, 254)
    audit.equal('prespecified edge identity', [example[k] for k in ['source_layer', 'source_feature', 'target_layer', 'target_feature']], list(identity))
    graph = audit.read(args.root/'runs/2b_atlas/circuit_graph.json')
    matches = [row for row in graph['edges'] if tuple(row[k] for k in ['source_layer', 'source_feature', 'target_layer', 'target_feature']) == identity]
    audit.equal('original graph edge unique', len(matches), 1)
    original = matches[0]
    for key in ['cohens_d', 'consistency', 'mean_delta']:
        audit.equal('original saved edge '+key, example[key], original[key], atol=1e-300)
    universe = {gene.upper() for gene in audit.read(args.root/'runs/2b_atlas/row_gene_names.json') if gene != '<CELL>'}
    terms, string, regulatory, factors, files = references(args.root/'ontologies_v2', universe, 5, 500)
    for path in files: audit.hash(path)
    identifiers = [source+'::'+name for source, name, _ in terms]+[source+'::'+name for source, name in EXTRA]
    positions = {identifier: index for index, identifier in enumerate(identifiers)}
    term_id = 'GO_BP::Proteasomal Protein Catabolic Process (GO:0010498)'
    endpoints = []
    sets = []
    for layer, feature in [(7, 254), (9, 254)]:
        annotation = audit.read(args.root/'results/annotations/2b'/f'ann_v2_L{layer:02d}.json')
        catalogue = audit.read(args.root/'runs/2b_atlas'/f'feature_catalog_L{layer:02d}.json')
        row = next(row for row in annotation['features'] if row['feature_idx'] == feature)
        catalogued = next(row for row in catalogue['features'] if row['feature_idx'] == feature)
        genes = {gene['gene_name'].upper() for gene in catalogued['top_genes']} & universe
        p, q, _, _ = dense_tests(genes, universe, terms, string, regulatory, factors, annotation['summary']['min_overlap'])
        audit.equal('full tested family', annotation['summary']['tests_per_feature'], len(identifiers))
        expected_ids = {identifiers[index] for index in np.flatnonzero(q[:len(terms)] < annotation['summary']['alpha'])}
        audit.equal(f'L{layer} complete significant IDs', set(row['significant_term_ids']), expected_ids)
        selected = next(term for term in row['terms'] if term['source']+'::'+term['term'] == term_id)
        index = positions[term_id]
        audit.equal(f'L{layer} proteasome p', selected['p'], p[index], atol=1e-300)
        audit.equal(f'L{layer} proteasome q', selected['q'], q[index], atol=1e-300)
        audit.equal(f'L{layer} proteasome fold', selected['fold'], (len(genes & terms[index][2])/len(genes))/(len(terms[index][2])/len(universe)))
        audit.check(f'L{layer} named term significant', q[index] < .05)
        endpoints.append({'layer': layer, 'feature': feature, 'top_genes': catalogued['top_genes'],
                          'proteasome_term': selected, 'full_tests_per_feature': len(identifiers)})
        sets.append(expected_ids)
    shared = sets[0] & sets[1]
    audit.equal('full current shared term IDs', {source+'::'+term for source, term in example['shared_terms']}, shared)
    audit.check('proteasome is shared', term_id in shared)
    audit.hash(Path(__file__))
    report = {'audit': 'Named proteasome circuit saved-edge and dense endpoint annotation audit',
              'passed': audit.passed, 'n_checks': audit.n_checks, 'findings': audit.failed,
              'edge': original, 'endpoints': endpoints, 'n_complete_shared_named_terms': len(shared),
              'source_sha256': audit.sources,
              'scope': 'Model feature ablation in a saved capped graph; ontology labels describe endpoint gene sets. No external pathway, TF-direction or organism-level mechanism is established.'}
    args.out.parent.mkdir(parents=True, exist_ok=True); args.out.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({key: report[key] for key in ['passed', 'n_checks', 'findings', 'n_complete_shared_named_terms']}))
    raise SystemExit(0 if audit.passed else 1)


if __name__ == '__main__': main()
