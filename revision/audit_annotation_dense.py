"""Independent V1 annotation audit: dense SciPy tests and statsmodels BH, never Scorer.

Only a deterministic sample of real features is rescored. Summary counts/denominators are
checked over the saved per-feature records. Null gene sets were not saved, so chance-floor
checks cover seed-rate arithmetic and denominators, not independent regeneration of all nulls.
"""
from __future__ import annotations

import argparse
from collections import Counter
from itertools import combinations
import gzip
import hashlib
import json
from pathlib import Path
import resource
import sys

import numpy as np
from scipy.stats import hypergeom, poisson
from statsmodels.stats.multitest import multipletests


NAMED = ('GO_BP', 'KEGG', 'Reactome', 'TRRUST')
EXTRA = [('STRING', 'interaction enrichment'), ('TRRUST_pairs', 'regulatory pair enrichment'),
         ('TF_membership', 'transcription factor enrichment')]


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1<<20), b''): digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def references(directory, universe, minimum, maximum):
    directory = Path(directory)
    terms, files = [], []
    for source, filename in [('GO_BP', 'go_bp.json'), ('KEGG', 'kegg.json'), ('Reactome', 'reactome.json')]:
        path = directory/filename
        files.append(path)
        for name, members in read(path).items():
            genes = {gene.upper() for gene in members} & universe
            if minimum<=len(genes)<=maximum: terms.append((source, name, genes))
    trrust, factors = set(), set()
    regulons = {}
    path = directory/'trrust_human.tsv'; files.append(path)
    for line in path.read_text().splitlines():
        fields = line.split('\t')
        if len(fields)>=2:
            factor, target = fields[0].upper(), fields[1].upper()
            factors.add(factor)
            regulons.setdefault(factor, set()).add(target)
            if factor!=target and factor in universe and target in universe:
                trrust.add(tuple(sorted((factor, target))))
    for factor, targets in regulons.items():
        genes = targets & universe
        if minimum<=len(genes)<=maximum: terms.append(('TRRUST', f'{factor} targets', genes))
    paths = sorted(directory.glob('string_v*_pairs.tsv.gz'))
    if len(paths)!=1: raise ValueError('Audit needs one unambiguous saved STRING release')
    files.append(paths[0]); string = set()
    with gzip.open(paths[0], 'rt') as handle:
        for line in handle:
            a, b = line.rstrip('\n').split('\t')
            a, b = a.upper(), b.upper()
            if a!=b and a in universe and b in universe: string.add(tuple(sorted((a, b))))
    return terms, string, trrust, factors & universe, files


def dense_tests(genes, universe, terms, string, regulatory, factors, minimum_overlap):
    """The explicit p=1 vector includes every retained term and all three extra tests."""
    n, population = len(genes), len(universe)
    values = np.ones(len(terms)+3)
    observed, expected = {}, {}
    eligible = n>=3
    for position, (source, term, members) in enumerate(terms):
        overlap = len(genes & members)
        if eligible and overlap>=minimum_overlap:
            values[position] = hypergeom.sf(overlap-1, population, len(members), n)
    pairs = {tuple(sorted(pair)) for pair in combinations(genes, 2)}
    possible_universe_pairs = population*(population-1)/2
    possible_feature_pairs = n*(n-1)/2
    for offset, (label, edges) in enumerate([('STRING', string), ('TRRUST_pairs', regulatory)]):
        observed[label] = len(pairs & edges)
        expected[label] = possible_feature_pairs*len(edges)/possible_universe_pairs
        if eligible and observed[label]>0:
            values[len(terms)+offset] = poisson.sf(observed[label]-1, expected[label])
    observed['TF_membership'] = len(genes & factors)
    expected['TF_membership'] = n*len(factors)/population
    if eligible:
        values[-1] = hypergeom.sf(observed['TF_membership']-1, population, len(factors), n)
    adjusted = multipletests(values, method='fdr_bh')[1]
    return values, adjusted, observed, expected


def choose_features(records, number, seed):
    """Include named/network signals and weak positive/negative boundaries before random fill."""
    reasons = {}
    def pick(rows, reason, count):
        for row in rows[:count]: reasons.setdefault(int(row['feature_idx']), []).append(reason)
    best_q = lambda row: min((term['q'] for term in row.get('terms', [])), default=1.)
    stable = lambda rows, key: sorted(rows, key=lambda row: (key(row), int(row['feature_idx'])))
    pick(stable(records, lambda row: row['min_p']), 'strong minimum p', 3)
    pick(stable([row for row in records if row['significant_term_ids']], best_q), 'strong named ontology signal', 3)
    pick(stable([row for row in records if row['annotated']], lambda row: abs(best_q(row)-.05)), 'weak positive boundary', 5)
    pick(stable([row for row in records if not row['annotated']], lambda row: abs(best_q(row)-.05)), 'weak negative boundary', 5)
    for source, _ in EXTRA:
        pick(stable([row for row in records if any(t['source']==source and t['q']<.05 for t in row.get('terms', []))], best_q), f'{source} signal', 2)
    pick(stable([row for row in records if row['annotated'] and not row['significant_term_ids']], best_q), 'network or membership only', 3)
    if len(reasons)>number:
        reasons = dict(list(reasons.items())[:number])
    remaining = sorted(int(row['feature_idx']) for row in records if int(row['feature_idx']) not in reasons)
    for feature in np.random.default_rng(seed).choice(remaining, min(number-len(reasons), len(remaining)), replace=False):
        reasons[int(feature)] = ['deterministic random fill']
    return reasons


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--annotation', type=Path, required=True)
    parser.add_argument('--catalog', type=Path, required=True)
    parser.add_argument('--universe', type=Path, required=True)
    parser.add_argument('--ont-dir', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--n-features', type=int, default=30)
    parser.add_argument('--seed', type=int, default=20261117)
    args = parser.parse_args()
    artifact = read(args.annotation)
    summary, records = artifact['summary'], artifact['features']
    if summary['scorer_version']!='3-plan-aligned' or summary['legacy_libraries']:
        raise ValueError('This audit requires new full-library v3 annotations, not legacy/pilot files')
    universe = {str(gene).upper() for gene in read(args.universe) if gene and gene!='<CELL>'}
    terms, string, regulatory, factors, sources = references(args.ont_dir, universe, summary['min_term'], summary['max_term'])
    catalogue = read(args.catalog)['features']
    catalogue_by_id = {int(row['feature_idx']): row for row in catalogue}
    alive = {int(row['feature_idx']) for row in catalogue if not row.get('is_dead', False)}
    records_by_id = {int(row['feature_idx']): row for row in records}
    checks = []
    def check(label, actual, expected, *, tolerance=False):
        passed = bool(np.isclose(actual, expected, rtol=1e-8, atol=1e-300)) if tolerance else actual==expected
        checks.append({'check': label, 'passed': bool(passed), 'saved': actual, 'independent': expected})
    check('catalogue feature IDs unique', len(catalogue_by_id), len(catalogue))
    check('annotation feature IDs unique', len(records_by_id), len(records))
    check('annotation IDs equal live catalogue IDs', sorted(records_by_id), sorted(alive))
    check('universe size', summary['universe_size'], len(universe))
    check('full per-feature family size including p=1 terms', summary['tests_per_feature'], len(terms)+3)
    check('named term counts after universe restriction', summary['terms_by_source'], {s: sum(t[0]==s for t in terms) for s in NAMED})
    check('STRING universe edge count', summary['string_pairs_in_universe'], len(string))
    check('TRRUST undirected nonself universe edge count', summary['trrust_pairs_in_universe'], len(regulatory))
    check('total feature denominator', summary['n_features_total'], len(catalogue))
    check('live feature denominator', summary['n_alive'], len(alive))
    check('scored feature denominator', summary['n_sets'], len(records))
    n_annotated = sum(bool(row['annotated']) for row in records)
    n_ontology = sum(bool(row['significant_term_ids']) for row in records)
    n_testable = sum(row['n_genes']>=3 for row in records)
    for key, expected in [('n_annotated', n_annotated), ('n_ontology_annotated', n_ontology),
                          ('n_network_or_membership_only', n_annotated-n_ontology), ('n_testable', n_testable)]:
        check(key, summary[key], expected)
    for key, expected in [('annotation_rate', n_annotated/len(records)),
                          ('ontology_annotation_rate_alive', n_ontology/len(records)),
                          ('annotation_rate_all_features', n_annotated/len(catalogue)),
                          ('ontology_annotation_rate_all_features', n_ontology/len(catalogue))]:
        check(key, summary[key], expected, tolerance=True)
    sources_per_feature = Counter()
    for row in records:
        check(f'complete named IDs unique F{row["feature_idx"]}', len(row['significant_term_ids']), len(set(row['significant_term_ids'])))
        named_sources = {term.split('::', 1)[0] for term in row['significant_term_ids']}
        extra_sources = {term['source'] for term in row.get('terms', []) if term['source'] not in NAMED and term['q']<summary['alpha']}
        sources_per_feature.update(named_sources | extra_sources)
    for source, saved in summary['rate_by_source'].items():
        check(f'{source} annotation rate denominator', saved, sources_per_feature[source]/len(records), tolerance=True)
    floor_checks = {}
    for kind, floor in summary['floors'].items():
        if 'ontology_values' not in floor or 'ontology_mean' not in floor:
            checks.append({'check': f'{kind} ontology-only chance floor exists', 'passed': False})
            continue
        combined, ontology = np.asarray(floor['values']), np.asarray(floor['ontology_values'])
        check(f'{kind} floor seed alignment', len(combined), len(ontology))
        check(f'{kind} combined floor mean', floor['mean'], float(combined.mean()), tolerance=True)
        check(f'{kind} combined floor SD', floor['sd'], float(combined.std()), tolerance=True)
        check(f'{kind} ontology-only floor mean', floor['ontology_mean'], float(ontology.mean()), tolerance=True)
        check(f'{kind} ontology floor is subset of combined', bool(np.all(ontology<=combined)), True)
        for label, values in [('combined', combined), ('ontology', ontology)]:
            check(f'{kind} {label} rates represent integer counts / n_alive', bool(np.allclose(values*len(alive), np.rint(values*len(alive)), rtol=0, atol=1e-6)), True)
            check(f'{kind} {label} valid rate range', bool(np.all((values>=0)&(values<=1))), True)
        floor_checks[kind] = {'combined_alive': floor['mean'], 'ontology_alive': floor['ontology_mean'],
                             'combined_all_features': floor['mean']*len(alive)/len(catalogue),
                             'ontology_all_features': floor['ontology_mean']*len(alive)/len(catalogue)}
    threshold = summary['global_bh']['p_threshold']
    saved_global_rate = sum(row['min_p']<=threshold for row in records)/len(records)
    check('global BH rate from saved minimum p and threshold', summary['global_bh']['annotation_rate'], saved_global_rate, tolerance=True)
    chosen = choose_features(records, args.n_features, args.seed)
    identities = [f'{source}::{name}' for source, name, _ in terms]+[f'{source}::{name}' for source, name in EXTRA]
    index_by_identity = {name: i for i, name in enumerate(identities)}
    dense_p, dense_q, sampled = [], [], []
    for feature, reasons in chosen.items():
        row, catalogued = records_by_id[feature], catalogue_by_id[feature]
        genes = {str(g['gene_name'] if isinstance(g, dict) else g).upper() for g in catalogued['top_genes']} & universe
        p, q, observed, expected = dense_tests(genes, universe, terms, string, regulatory, factors, summary['min_overlap'])
        dense_p.append(p); dense_q.append(q)
        significant = q<summary['alpha']
        term_ids = {identities[i] for i in np.flatnonzero(significant[:len(terms)])}
        check(f'F{feature} restricted gene count', row['n_genes'], len(genes))
        check(f'F{feature} any significant test', row['annotated'], bool(significant.any()))
        check(f'F{feature} number of significant tests', row['n_sig'], int(significant.sum()))
        check(f'F{feature} complete significant named IDs', sorted(row['significant_term_ids']), sorted(term_ids))
        check(f'F{feature} minimum p', row['min_p'], float(p.min()), tolerance=True)
        for saved in row.get('terms', []):
            identity = f'{saved["source"]}::{saved["term"]}'
            if identity not in index_by_identity:
                checks.append({'check': f'F{feature} displayed identity is in test family: {identity}', 'passed': False}); continue
            position = index_by_identity[identity]
            check(f'F{feature} {identity} p', saved['p'], float(p[position]), tolerance=True)
            check(f'F{feature} {identity} q from full dense BH', saved['q'], float(q[position]), tolerance=True)
            if position<len(terms):
                source, name, members = terms[position]
                overlap = genes & members
                for key, value in [('k', len(overlap)), ('K', len(members)), ('n', len(genes))]:
                    check(f'F{feature} {identity} {key}', saved[key], value)
                fold = (len(overlap)/len(genes))/(len(members)/len(universe))
                check(f'F{feature} {identity} fold', saved['fold'], fold, tolerance=True)
                check(f'F{feature} {identity} overlapping genes', sorted(saved['genes']), sorted(overlap))
            else:
                check(f'F{feature} {identity} count', saved['k'], observed[saved['source']], tolerance=True)
                check(f'F{feature} {identity} expected count', saved['expected'], expected[saved['source']], tolerance=True)
        sampled.append({'feature_idx': feature, 'selection_reasons': reasons, 'n_genes': len(genes),
                        'independent_min_p': float(p.min()), 'independent_min_q': float(q.min()),
                        'independent_n_significant': int(significant.sum()), 'independent_named_term_ids': sorted(term_ids),
                        'independent_pair_or_membership_counts': observed})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    array_path = args.out.with_suffix('.dense_pq.npz')
    np.savez_compressed(array_path, feature_ids=np.array(list(chosen)), test_ids=np.asarray(identities),
                        p=np.asarray(dense_p), q=np.asarray(dense_q))
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    output = {'audit': 'V1 independent dense annotation audit', 'all_checks_passed': all(c['passed'] for c in checks),
              'n_checks': len(checks), 'n_sampled_features': len(chosen), 'n_catalogue_features': len(catalogue),
              'n_alive_features': len(alive), 'n_tests_per_feature': len(terms)+3, 'universe_size': len(universe),
              'alpha': summary['alpha'], 'selection_seed': args.seed, 'sampled_features': sampled,
              'floor_denominators': floor_checks, 'checks': checks,
              'dense_array_artifact': {'file': array_path.name, 'sha256': sha256(array_path)},
              'peak_rss_bytes': int(rss if sys.platform=='darwin' else rss*1024),
              'limitations': ['Only sampled real features were independently rescored; all saved feature flags were counted.',
                  'Null gene sets were not saved; chance-floor arithmetic and denominators were checked, not all null rescoring.',
                  'Global BH rate was checked against saved min-p/threshold; its threshold was not regenerated from every feature/test pair.',
                  'Poisson pair-test calculations are audited as specified; this does not establish exact calibration under dependent graph edges.'],
              'source_sha256': {str(path): sha256(path) for path in [args.annotation, args.catalog, args.universe, *sources, Path(__file__)]}}
    args.out.write_text(json.dumps(output, indent=2, allow_nan=False))
    failures = [check for check in checks if not check['passed']]
    print(json.dumps({'audit': str(args.out), 'passed': output['all_checks_passed'], 'n_checks': len(checks),
                      'n_sampled_features': len(chosen), 'peak_rss_bytes': output['peak_rss_bytes'],
                      'failures': failures[:20]}, indent=2))
    if failures: raise SystemExit(2)


if __name__=='__main__': main()
