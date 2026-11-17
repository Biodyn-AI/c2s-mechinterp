#!/usr/bin/env python3
"""Rejoin capped circuit edges to complete named annotations; no model forwards."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import resource
import sys

import numpy as np

from audit_experimental_saved import Audit, circuit_from_source_counts


def popcount(bits):
    return bits.bit_count() if hasattr(bits, 'bit_count') else bin(bits).count('1')


def exact_join(edges, source_pools, target_loader):
    clusters = dict.fromkeys((int(e['source_layer']), int(e['source_feature'])) for e in edges)
    clusters = {key: np.zeros(3) for key in clusters}
    by_target = defaultdict(list)
    for edge in edges:
        by_target[int(edge['target_layer'])].append(edge)
    eligible_by_pair = Counter()
    fully_random_weighted_sum = 0.
    for target_layer, layer_edges in sorted(by_target.items()):
        targets = {feature: ids for feature, ids in target_loader(target_layer).items() if ids}
        inverted = defaultdict(int)
        for feature, terms in targets.items():
            for term in terms:
                inverted[term] |= 1 << feature

        def chance(terms):
            bits = 0
            for term in terms:
                bits |= inverted.get(term, 0)
            return popcount(bits)/len(targets) if targets else 0.

        for edge in layer_edges:
            source_layer, feature = int(edge['source_layer']), int(edge['source_feature'])
            terms = source_pools[source_layer].get(feature, set())
            target_terms = targets.get(int(edge['target_feature']), set())
            if terms and target_terms:
                clusters[source_layer, feature] += [bool(terms & target_terms), 1, chance(terms)]
                eligible_by_pair[source_layer, target_layer] += 1
        for source_layer in source_pools:
            number = eligible_by_pair[source_layer, target_layer]
            if number:
                annotated_sources = [terms for terms in source_pools[source_layer].values() if terms]
                probability = sum(chance(terms) for terms in annotated_sources)/len(annotated_sources)
                fully_random_weighted_sum += number*probability
    shared, eligible, expected = np.asarray(list(clusters.values())).sum(axis=0)
    return {'n_edges': len(edges), 'n_annotated_edges': int(eligible), 'n_shared': int(shared),
            'n_source_clusters': len(clusters), 'observed': float(shared/eligible),
            'same_source_random_target': float(expected/eligible),
            'fully_random_annotated_pair': float(fully_random_weighted_sum/eligible),
            'excess': float((shared-expected)/eligible),
            'source_counts': [{'source_layer': layer, 'source_feature': feature, 'shared': int(values[0]),
                               'eligible': int(values[1]), 'expected_shared': float(values[2])}
                              for (layer, feature), values in clusters.items()]}


def power_curve(saved, audit):
    n, simulations, seed = saved['n_active_cells'], saved['n_sim'], saved['seed']
    noise = np.random.default_rng(seed).normal(size=(simulations, n))
    means, standard_deviations = noise.mean(axis=1), noise.std(axis=1, ddof=1)
    first_crossing, null_probability = None, None
    for row in saved['curve']:
        effect = row['population_standardized_effect']
        positive_fraction = (noise+effect > 0).mean(axis=1)
        effect_sizes = (means+effect)/standard_deviations
        called = (np.abs(effect_sizes) > .5) & (np.maximum(positive_fraction, 1-positive_fraction) > .7)
        probability = float(called.mean())
        audit.equal(f'normal sensitivity n={n} delta={effect}', row['call_probability'], probability)
        if effect == 0:
            null_probability = probability
        if first_crossing is None and probability >= .8:
            first_crossing = effect
    audit.equal(f'normal conditional n={n} first 80% grid crossing', saved['minimum_grid_effect_at_80pct'], first_crossing)
    audit.equal(f'normal conditional n={n} null call probability', saved['null_call_probability'], null_probability)
    return {'n_active_cells': n, 'minimum_grid_effect_at_80pct': first_crossing,
            'null_call_probability': null_probability,
            'interpretation': 'Conditional independent-normal sensitivity; not real-data MDE or FDR.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    audit = Audit()
    saved = audit.read(args.root/'results/circuit_baseline.json')
    graph_path = args.root/'runs/2b_atlas/circuit_graph.json'
    graph = audit.read(graph_path)
    edges = graph['edges']
    audit.equal('graph own edge count', graph['n_edges'], len(edges))
    source_layers = sorted({int(edge['source_layer']) for edge in edges})

    def load_pool(layer):
        annotation = audit.read(args.root/'results/annotations/2b'/f'ann_v2_L{layer:02d}.json')
        audit.equal(f'L{layer} full annotation version', annotation['summary']['scorer_version'], '3-plan-aligned')
        return {int(row['feature_idx']): {term for term in row['significant_term_ids']
                if term.split('::', 1)[0] in {'GO_BP', 'KEGG', 'Reactome', 'TRRUST'}} for row in annotation['features']}

    sources = {layer: load_pool(layer) for layer in source_layers}
    joined = exact_join(edges, sources, lambda layer: sources[layer] if layer in sources else load_pool(layer))
    for key, value in joined.items():
        if key != 'source_counts': audit.equal('actual graph/annotation '+key, saved[key], value)
    audit.equal('source-count rows', len(saved['source_counts']), len(joined['source_counts']))
    for original, expected in zip(saved['source_counts'], joined['source_counts']):
        for key, value in expected.items(): audit.equal(f'source {expected["source_layer"]}/{expected["source_feature"]} {key}', original[key], value)
    bootstrap = circuit_from_source_counts(joined['source_counts'], saved['n_boot'], saved['seed'])
    for key, value in bootstrap['source_bootstrap'].items():
        for field in ['estimate', 'ci95']:
            audit.equal('rejoined source bootstrap '+key+' '+field, saved['source_bootstrap'][key][field], value[field])
    edge_counts = Counter((edge['source_layer'], edge['source_feature']) for edge in edges)
    counts = np.asarray(list(edge_counts.values()))
    rng = np.random.default_rng(saved['seed'])
    totals = [int(np.sum(counts[rng.integers(0, len(counts), size=len(counts))])) for _ in range(saved['n_boot'])]
    total_ci = np.quantile(totals, [.025, .975]).tolist()
    audit.equal('total capped-edge-count source-bootstrap CI', saved['total_edge_count_source_bootstrap_ci95'], total_ci)
    downstream_counts = Counter((edge['source_layer'], edge['source_feature'], edge['target_layer']) for edge in edges)
    audit.check('capped graph at most50 edges per source/downstream layer', max(downstream_counts.values()) <= 50)
    sensitivity = [power_curve(row, audit) for row in saved['edge_rule_sensitivity']]
    audit.hash(Path(__file__))
    report = {'audit': 'Independent exact circuit graph/complete-annotation join and source bootstrap',
              'passed': audit.passed, 'n_checks': audit.n_checks, 'findings': audit.failed,
              'joined': joined, 'source_bootstrap': bootstrap['source_bootstrap'],
              'total_edge_count_source_bootstrap_ci95': total_ci, 'conditional_normal_sensitivity': sensitivity,
              'source_sha256': audit.sources,
              'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform == 'darwin' else 1024),
              'limitations': ['The saved capped graph is rejoined; omitted edges and model interventions are not regenerated.',
                              'Source features are bootstrap units; shared-source edges are not independent observations.',
                              'Normal sensitivity assumes every source is active and independent normal deltas; it is not a multiple-testing or empirical-power calculation.']}
    args.out.parent.mkdir(parents=True, exist_ok=True); args.out.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({key: report[key] for key in ['passed', 'n_checks', 'findings', 'peak_rss_bytes']}))
    raise SystemExit(0 if audit.passed else 1)


if __name__ == '__main__': main()
