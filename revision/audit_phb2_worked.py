#!/usr/bin/env python3
"""Check the PHB2 L16 worked example against arrays, C5 and a dense annotation test."""
import argparse
import ast
import json
from pathlib import Path

import numpy as np

from audit_experimental_saved import Audit
from audit_annotation_dense import references, dense_tests


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--claims', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    root, audit = args.root, Audit()
    claims = audit.read(args.claims)
    factor, layer, feature = claims['factor'], claims['layer'], claims['feature']
    statistics = root/'results/phase8_k562/statistics'
    rows = audit.read(root/'results/phase8_k562/rows.json')
    biology = audit.read(statistics/'response_biology.json')
    catalogue = audit.read(root/'runs/2b_atlas'/f'feature_catalog_L{layer:02d}.json')['features']
    catalog_by_id = {row['feature_idx']: row for row in catalogue}
    annotation = audit.read(root/'results/annotations/2b'/f'ann_v2_L{layer:02d}.json')
    annotation_by_id = {row['feature_idx']: row for row in annotation['features']}
    genes_path, feature_path = statistics/'gene_statistics.npz', statistics/f'feature_statistics_L{layer:02d}.npz'
    audit.hash(genes_path); audit.hash(feature_path)
    with np.load(genes_path, allow_pickle=False) as arrays:
        group_index = arrays['groups'].astype(str).tolist().index(factor)
        genes = arrays['genes'].astype(str).tolist()
        for record in claims['gene_records']:
            position = genes.index(record['gene'])
            for field, key in [('expression_d', 'expression_effect'), ('expression_q', 'expression_q'),
                               ('expression_responding', 'expression_responding'), ('rank_d', 'input_effect'), ('rank_q', 'input_q')]:
                audit.equal(record['gene']+' '+field, record[field], arrays[key][group_index, position], atol=1e-300)
    with np.load(feature_path, allow_pickle=False) as arrays:
        group_index = arrays['groups'].astype(str).tolist().index(factor)
        for key, value in claims['feature_response'].items():
            audit.equal('feature '+key, value, arrays[key][group_index, feature], atol=1e-300)
        responding = arrays['responding'][group_index]
        effect, q = arrays['effect'][group_index, feature], arrays['q'][group_index, feature]
    c5 = next(row for row in biology['responses'] if row['group'] == factor and row['layer'] == layer)
    example = next(row for row in c5['named_features'] if row['feature_idx'] == feature)
    audit.equal('C5 absolute effect', example['response_strength_abs_d'], abs(float(effect)))
    audit.equal('C5 response q', example['response_q'], float(q))
    top_genes = [row['gene_name'] if isinstance(row, dict) else row for row in catalog_by_id[feature]['top_genes']]
    audit.equal('C5 top genes join catalogue', example['top_genes'], top_genes[:10])
    audit.equal('C5 complete IDs join annotation', example['significant_term_ids'], annotation_by_id[feature]['significant_term_ids'])
    audit.equal('25knockdown cells', sum(row['group'] == factor for row in rows), 25)
    audit.equal('400control cells', sum(row['kind'] == 'control' for row in rows), 400)
    universe_path = root/'runs/2b_atlas/row_gene_names.json'
    universe = {gene.upper() for gene in audit.read(universe_path) if gene and gene != '<CELL>'}
    summary = annotation['summary']
    terms, string, regulatory, factors, source_files = references(root/'ontologies_v2', universe, summary['min_term'], summary['max_term'])
    for path in source_files:
        audit.hash(path)
    pvalues, qvalues, _, _ = dense_tests(set(top_genes) & universe, universe, terms, string, regulatory, factors, summary['min_overlap'])
    identity = 'FASTK Family Proteins Regulate Processing and Stability of Mitochondrial RNAs'
    position = next(i for i, term in enumerate(terms) if term[0] == 'Reactome' and term[1] == identity)
    term = next(row for row in annotation_by_id[feature]['terms'] if row['source'] == 'Reactome' and row['term'] == identity)
    audit.equal('FASTK full-family dense p', term['p'], pvalues[position], atol=1e-300)
    audit.equal('FASTK full-family dense q', term['q'], qvalues[position], atol=1e-300)
    n, k, K, N = len(set(top_genes) & universe), len(set(top_genes) & terms[position][2]), len(terms[position][2]), len(universe)
    audit.equal('FASTK overlap count', term['k'], k)
    audit.equal('FASTK background size', term['K'], K)
    audit.equal('FASTK fold', term['fold'], (k/n)/(K/N))
    displayed = next(row for row in example['displayed_significant_terms'] if row['source'] == 'Reactome' and row['term'] == identity)
    audit.equal('C5 FASTK detail equals source annotation', displayed, term)
    # Evaluate the prespecified category rules directly, without importing the
    # experiment's category/scoring implementation.
    rules_path = root/'scripts/rev_cpu_analysis.py'; audit.hash(rules_path)
    rules = next(ast.literal_eval(node.value) for node in ast.parse(rules_path.read_text()).body
                 if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == 'CATEGORY_RULES' for target in node.targets))
    known = set()
    for line in (root/'ontologies_v2/trrust_human.tsv').read_text().splitlines():
        fields = line.split('\t')
        if len(fields) >= 2 and fields[0].upper() == factor:
            known.add(fields[1].upper())
    known.discard(factor)
    generic, target, unannotated = np.zeros(len(catalogue), bool), np.zeros(len(catalogue), bool), np.ones(len(catalogue), bool)
    for index, row in catalog_by_id.items():
        names = {gene['gene_name'] if isinstance(gene, dict) else gene for gene in row['top_genes']}
        target[index] = bool(names & known)
        ids = annotation_by_id.get(index, {}).get('significant_term_ids', [])
        unannotated[index] = not bool(ids)
        # The documented ordered rules assign each term its first matching
        # category; a feature can collect several categories across its terms.
        categories = {next((category for category, stems in rules
                            if any(stem in identity.split('::', 1)[-1].lower() for stem in stems)), 'other')
                      for identity in ids}
        generic[index] = bool(categories & set(biology['generic_programme_categories']))
    counts = {'n_responding': int(responding.sum()), 'n_target_labelled_responses': int((responding & target).sum()),
              'n_generic_programme_responses': int((responding & generic).sum()),
              'n_both_generic_and_target_labelled': int((responding & generic & target).sum()),
              'n_ontology_unannotated_responses': int((responding & unannotated).sum())}
    for key, value in counts.items():
        audit.equal('C5 '+key, c5[key], value)
    audit.equal('worked feature has no known-target label', example['matching_trrust_targets'], [])
    audit.hash(Path(__file__)); audit.hash(Path(__file__).with_name('audit_annotation_dense.py'))
    report = {'audit': 'PHB2 L16 worked example independent saved-data check', 'passed': audit.passed,
              'n_checks': audit.n_checks, 'findings': audit.failed, 'counts': counts,
              'feature_response': claims['feature_response'], 'gene_records': claims['gene_records'],
              'best_annotation': term, 'source_sha256': audit.sources,
              'scope': 'Same 25 knockdown and 400 control cells; no independent guide or external PHB2 mechanism claim. Annotation names describe the feature gene set, not a PHB2 regulatory pathway.'}
    args.out.parent.mkdir(parents=True, exist_ok=True); args.out.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({key: report[key] for key in ['passed', 'n_checks', 'findings', 'counts']}))
    raise SystemExit(0 if audit.passed else 1)


if __name__ == '__main__':
    main()
