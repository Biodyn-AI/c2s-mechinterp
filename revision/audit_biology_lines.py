#!/usr/bin/env python3
"""Independent category/persistence/line-design audit; CPU reads only, no model fits."""
from collections import Counter
import argparse
import ast
import gc
import hashlib
import json
from pathlib import Path
import resource
import sys

import h5py
import numpy as np
from scipy.stats import hypergeom

from audit_experimental_saved import Audit
from audit_c6_merge import h5_strings
from audit_annotation_dense import references, dense_tests


NAMED = {'GO_BP', 'KEGG', 'Reactome', 'TRRUST'}


def feature_categories(identifiers, rules):
    categories = set()
    for identifier in identifiers:
        source, term = identifier.split('::', 1)
        if source in NAMED:
            categories.add(next((category for category, stems in rules if any(stem in term.lower() for stem in stems)), 'other'))
    return categories


def persistence_summary(first, last, source_ids, target_ids, cosines, threshold, bootstrap, seed):
    target_counts = Counter(frozenset(value) for value in last.values() if value)
    probabilities = {}
    pairs = []
    for source, target, cosine in zip(source_ids, target_ids, cosines):
        a, b = first.get(int(source), set()), last.get(int(target), set())
        if cosine < threshold or not a or not b: continue
        key = frozenset(a)
        if key not in probabilities:
            probabilities[key] = sum(count for terms, count in target_counts.items() if a & terms)/sum(target_counts.values())
        pairs.append([bool(a & b), probabilities[key]])
    pairs = np.asarray(pairs, float)
    observed, baseline = pairs.mean(axis=0)
    rng = np.random.default_rng(seed)
    differences = [np.diff(pairs[rng.integers(0, len(pairs), len(pairs))].mean(axis=0))[0]*-1 for _ in range(bootstrap)]
    return {'n_matched_annotated_pairs': len(pairs), 'observed_same_category': float(observed),
            'random_target_baseline': float(baseline), 'excess': float(observed-baseline),
            'excess_ci95': np.quantile(differences, [.025, .975]).tolist(), 'cosine_threshold': threshold}


def selected_sparse_rows(group, selected):
    selected = np.asarray(selected, int)
    shape = tuple(group.attrs['shape']) if isinstance(group, h5py.Group) else group.shape
    if len(set(selected)) != len(selected) or np.any(selected < 0) or np.any(selected >= shape[0]):
        raise ValueError('Invalid/duplicate selected source rows')
    if isinstance(group, h5py.Dataset):
        order = np.argsort(selected); return np.asarray(group[selected[order], :], np.float32)[np.argsort(order)]
    output = np.zeros((len(selected), shape[1]), np.float32)
    encoding = group.attrs['encoding-type']
    if encoding == 'csr_matrix':
        for position, row in enumerate(selected):
            start, stop = map(int, group['indptr'][row:row+2])
            np.add.at(output[position], group['indices'][start:stop], group['data'][start:stop])
    elif encoding == 'csc_matrix':
        lookup = np.full(shape[0], -1, int); lookup[selected] = np.arange(len(selected))
        pointers = group['indptr'][:]
        for column in range(shape[1]):
            start, stop = int(pointers[column]), int(pointers[column+1])
            rows = group['indices'][start:stop]; positions = lookup[rows]; keep = positions >= 0
            if keep.any(): np.add.at(output[:, column], positions[keep], group['data'][start:stop][keep])
    else: raise ValueError('Unsupported count matrix storage: '+str(encoding))
    return output


def audit_biology(root, audit):
    saved = audit.read(root/'results/biology.json')
    rules = saved['category_rules']
    script = root/'scripts/rev_cpu_analysis.py'; audit.hash(script)
    source_rules = next(ast.literal_eval(node.value) for node in ast.parse(script.read_text()).body if isinstance(node, ast.Assign)
                        and any(isinstance(target, ast.Name) and target.id == 'CATEGORY_RULES' for target in node.targets))
    audit.equal('category rule source identity', rules, json.loads(json.dumps(source_rules)))
    audit.equal('all26 biology layers', sorted(row['layer'] for row in saved['layers']), list(range(26)))
    all_categories, distributions = {}, []
    for layer_row in saved['layers']:
        layer = layer_row['layer']; audit.context = f'biology/L{layer}: '
        ann = audit.read(root/'results/annotations/2b'/f'ann_v2_L{layer:02d}.json')
        catalog = audit.read(root/'runs/2b_atlas'/f'feature_catalog_L{layer:02d}.json')
        records = {row['feature_idx']: row for row in ann['features']}
        catalogue = {row['feature_idx']: row for row in catalog['features']}
        categories = {feature: feature_categories(row['significant_term_ids'], rules) for feature, row in records.items()}
        all_categories[layer] = categories
        annotated = sum(bool(value) for value in categories.values())
        counts = Counter(category for value in categories.values() for category in value)
        audit.equal('feature denominator', layer_row['n_features'], len(catalogue))
        audit.equal('ontology denominator', layer_row['n_ontology_annotated'], annotated)
        for category in saved['categories']:
            audit.equal('category count '+category, layer_row['category_counts'].get(category, 0), counts[category])
            audit.equal('category share/all '+category, layer_row['category_share_all_features'][category], counts[category]/len(catalogue))
            audit.equal('category share/annotated '+category, layer_row['category_share_annotated_features'][category], counts[category]/annotated)
        minima = {}
        for feature, row in records.items():
            for term in row.get('terms', []):
                if term['source'] in NAMED and term['q'] < .05:
                    category = next(iter(feature_categories([term['source']+'::'+term['term']], rules)))
                    minima[category] = min(minima.get(category, 1.), term['q'])
        for category, example in layer_row['examples'].items():
            feature, term = example['feature_idx'], example['term']
            audit.check('example category supported by complete IDs', category in categories[feature])
            audit.check('example numeric term in source feature', term in records[feature]['terms'])
            audit.equal('example strongest saved q '+category, term['q'], minima[category], atol=1e-300)
            audit.equal('example top genes '+category, example['top_genes'], catalogue[feature]['top_genes'][:10])
            audit.equal('example hypergeometric p', term['p'], float(hypergeom.sf(term['k']-1, ann['summary']['universe_size'], term['K'], term['n'])), atol=1e-300)
            audit.equal('example fold', term['fold'], (term['k']/term['n'])/(term['K']/ann['summary']['universe_size']))
        modules = audit.read(root/'runs/2b_atlas'/f'coactivation_modules_L{layer:02d}.json')['modules']
        indexed = {module['module_id']: module for module in layer_row['modules']}
        audit.equal('all module IDs', set(indexed), {module['module_id'] for module in modules})
        for module in modules:
            expected = dict(Counter(category for feature in module['features'] for category in categories.get(feature, set())))
            audit.equal('module category counts', indexed[module['module_id']]['category_counts'], expected)
            audit.equal('module feature count', indexed[module['module_id']]['n_features'], len(module['features']))
        if layer == 13:
            universe = {gene.upper() for gene in audit.read(root/'runs/2b_atlas/row_gene_names.json') if gene != '<CELL>'}
            terms, string, regulatory, factors, files = references(root/'ontologies_v2', universe, 5, 500)
            for path in files: audit.hash(path)
            identities = [source+'::'+name for source, name, _ in terms]+[
                'STRING::interaction enrichment', 'TRRUST_pairs::regulatory pair enrichment',
                'TF_membership::transcription factor enrichment']
            positions = {identity: index for index, identity in enumerate(identities)}
            for feature in [9106, 1089, 254]:
                genes = {gene['gene_name'].upper() for gene in catalogue[feature]['top_genes']} & universe
                p, q, _, _ = dense_tests(genes, universe, terms, string, regulatory, factors, ann['summary']['min_overlap'])
                complete = {identities[index] for index in np.flatnonzero(q[:len(terms)] < ann['summary']['alpha'])}
                audit.equal(f'F{feature} dense full-family named IDs', set(records[feature]['significant_term_ids']), complete)
                for term in records[feature]['terms']:
                    index = positions[term['source']+'::'+term['term']]
                    audit.equal(f'F{feature} dense p', term['p'], p[index], atol=1e-300)
                    audit.equal(f'F{feature} dense q', term['q'], q[index], atol=1e-300)
            del terms, string, regulatory, factors
        distributions.append({'layer': layer, 'n_ontology': annotated, 'median_categories_per_annotated_feature':
                              float(np.median([len(value) for value in categories.values() if value])),
                              'other_share_annotated': counts['other']/annotated})
        del ann, catalog, records, catalogue; gc.collect()
        print('audited biology layer', layer, flush=True)
    return saved, all_categories, distributions


def audit_persistence(root, saved, categories, audit):
    import torch
    torch.set_num_threads(4)
    match_path = root/'results/biology.json.matches.npz'; audit.hash(match_path)
    with np.load(match_path) as archive:
        sources, targets, cosines = [archive[key] for key in ['source_ids', 'target_ids', 'cosines']]
    directions = []
    for layer in [0, 25]:
        path = root/'runs/2b_atlas/sae'/f'sae_L{layer:02d}.pt'; audit.hash(path)
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        values = checkpoint['model_state_dict']['W_dec.weight'].numpy().T.copy()
        values /= np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)
        directions.append(values)
        del checkpoint; gc.collect()
    audit.equal('source matching covers full dictionary', sources, np.arange(len(directions[0])))
    for start in range(0, len(sources), 256):
        similarity = np.abs(directions[0][start:start+256] @ directions[1].T)
        best = similarity.argmax(axis=1)
        audit.equal('recomputed best decoder targets', targets[start:start+len(best)], best)
        audit.equal('recomputed best decoder absolute cosines', cosines[start:start+len(best)], similarity[np.arange(len(best)), best], atol=1e-6)
    stage = audit.read(root/'stage_logs/biology-summary.complete.json')
    command = stage['command']
    bootstrap = int(command[command.index('--n-boot')+1]) if '--n-boot' in command else 2000
    seed = int(command[command.index('--seed')+1]) if '--seed' in command else 0
    expected = persistence_summary(categories[0], categories[25], sources, targets, cosines,
                                   saved['persistent_categories']['cosine_threshold'], bootstrap, seed)
    for key, value in expected.items(): audit.equal('persistent category '+key, saved['persistent_categories'][key], value)
    expected['n_distinct_target_features_among_all_best_matches'] = len(set(targets))
    expected['interpretation'] = 'Dependent source-feature bootstrap; broad overlapping categories and target reuse. Interval includes zero excess.'
    return expected


def audit_lines(root, audit):
    directory = root/'designs/transfer_controls'; run = root/'runs/lines2'
    sources, cells = audit.read(directory/'SOURCES.json'), audit.read(directory/'cells.json')
    names = audit.read(directory/'row_gene_names.json'); row_path = directory/'row_cell_ids.npy'; audit.hash(row_path)
    row_cells = np.load(row_path)
    for filename in ['cells.json', 'row_gene_names.json', 'row_cell_ids.npy']:
        audit.hash(run/filename)
        audit.equal('exact extracted design file '+filename, audit.sources[str(run/filename)], audit.sources[str(directory/filename)])
    saved = audit.read(root/'results/lines2.json')
    audit.equal('exactly two intended lines', saved['lines'], ['k562', 'rpe1'])
    line_of = np.asarray([cell['line'] for cell in cells])[row_cells]
    stage = audit.read(root/'stage_logs/lines-eval.complete.json'); command = stage['command']
    audit.equal('same trained K562 dictionary path', str(Path(command[command.index('--sae-run')+1]).resolve()), str((root/'runs/2b_atlas').resolve()))
    audit.equal('dictionary subdirectory', command[command.index('--sae-subdir')+1] if '--sae-subdir' in command else 'sae', 'sae')
    design_hashes = {}
    for path in (root/'designs').glob('phase8_*.json'):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        design_hashes[digest] = path
    raw_verification = []
    for line in saved['lines']:
        chosen = [cell for cell in cells if cell['line'] == line]
        audit.equal(line+' 300 controls', len(chosen), 300)
        audit.equal(line+' verified design count', sources[line]['n_verified_non_targeting_cells'], len(chosen))
        parent_design = audit.read(design_hashes[sources[line]['design_sha256']])
        parent_control = next(group for group in parent_design['groups'] if group['kind'] == 'control')
        parent_sentences = dict(zip(parent_control['cells'], parent_control['sentences']))
        path = root/'data'/sources[line]['source']
        with h5py.File(path, 'r') as file:
            source_names = h5_strings(file['var'][file['var'].attrs['_index']]).tolist()
            indices = [cell['h5_index'] for cell in chosen]
            targets = h5_strings(file['obs/gene'])
            audit.check(line+' raw source labels all non-targeting', all(targets[index] == 'non-targeting' for index in indices))
            counts = selected_sparse_rows(file['X'], indices)
            audit.check(line+' raw counts finite nonnegative integers', np.isfinite(counts).all() and np.all(counts >= 0) and np.array_equal(counts, np.rint(counts)))
            for position, cell in enumerate(chosen):
                values = counts[position]; nonzero = np.flatnonzero(values > 0)
                ranked = nonzero[np.argsort(-values[nonzero], kind='stable')][:512]
                expected = [source_names[index] for index in ranked]
                sentence = [names[index] for index in np.flatnonzero(row_cells == cell['row_cell'])]
                audit.equal(line+' exact raw-count ranked sentence', sentence, expected)
                audit.equal(line+' parent-control sentence', sentence, parent_sentences[cell['h5_index']][:512])
        raw_verification.append({'line': line, 'n_cells': len(chosen), 'source': str(path), 'selected_counts_sha256': hashlib.sha256(counts.astype('<f4').tobytes()).hexdigest(),
                                 'source_shape': [len(targets), len(source_names)]})
        del counts, targets; gc.collect()
        print('audited raw line source', line, flush=True)
    path = root/'results/lines2_freq.npz'; audit.hash(path)
    with np.load(path) as frequencies:
        audit.equal('exact intended layer/line frequency keys', set(frequencies.files), {f'L{layer:02d}_{line}' for layer in [0, 13, 22] for line in saved['lines']})
        for layer in [0, 13, 22]:
            activation_path = run/f'layer_{layer:02d}_activations.npy'; audit.hash(activation_path)
            matrix = np.load(activation_path, mmap_mode='r')
            mean_path = root/'runs/2b_atlas/sae'/f'mu_L{layer:02d}.npy'; audit.hash(mean_path)
            audit.hash(root/'runs/2b_atlas/sae'/f'sae_L{layer:02d}.pt')
            mu = np.load(mean_path).reshape(-1)
            for line in saved['lines']:
                vector = frequencies[f'L{layer:02d}_{line}']; row = saved['layers'][str(layer)][line]
                positions = np.flatnonzero(line_of == line)
                audit.equal('line row count', row['n_rows'], len(positions))
                audit.equal('line dictionary width', len(vector), 9216)
                audit.check('line frequency finite and within probability range', np.all(np.isfinite(vector) & (vector >= 0) & (vector <= 1)))
                audit.equal('share firing over all dictionary features', row['share_features_firing'], float((vector > 0).mean()))
                audit.check('frequency counts integral within float32 precision', np.max(np.abs(vector*len(positions)-np.rint(vector*len(positions)))) < .02)
                total = np.zeros(matrix.shape[1])
                for start in range(0, len(positions), 2048): total += np.asarray(matrix[positions[start:start+2048]], float).sum(axis=0)
                shift = float(np.linalg.norm(total/len(positions)-mu)/np.linalg.norm(mu))
                audit.equal('line mean shift from raw activation rows', row['mu_shift_rel'], shift, atol=1e-4)
    return {'raw_source_verification': raw_verification, 'saved_firing_metrics': saved,
            'reconstruction_scope': 'Saved all-position line reconstruction values are retained; SAE reconstruction forwards are not repeated.'}


def circuit_category_counts(root, categories, audit):
    graph = audit.read(root/'runs/2b_atlas/circuit_graph.json')
    eligible, shared, by_category = 0, 0, Counter()
    for edge in graph['edges']:
        a, b = categories[edge['source_layer']].get(edge['source_feature'], set()), categories[edge['target_layer']].get(edge['target_feature'], set())
        if a and b:
            eligible += 1; intersection = a & b; shared += bool(intersection); by_category.update(intersection)
    return {'n_capped_edges': len(graph['edges']), 'n_ontology_annotated_edges': eligible,
            'n_edges_sharing_at_least_one_broad_category': shared, 'shared_category_counts_overlapping': dict(by_category),
            'interpretation': 'Descriptive broad category overlap; no chance correction, causal/pathway or independent-edge claim.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(); audit = Audit()
    saved, categories, distributions = audit_biology(args.root, audit)
    persistence = audit_persistence(args.root, saved, categories, audit)
    lines = audit_lines(args.root, audit)
    circuits = circuit_category_counts(args.root, categories, audit)
    audit.hash(Path(__file__))
    report = {'audit': 'Independent biological categories, decoder persistence and two-line source/firing audit',
              'passed': audit.passed, 'n_checks': audit.n_checks, 'findings': audit.failed,
              'category_distributions': distributions, 'persistence': persistence, 'lines': lines, 'circuit_categories_descriptive': circuits,
              'source_sha256': audit.sources, 'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform == 'darwin' else 1024),
              'limitations': ['Category labels are coarse overlapping descriptions; ontology labels do not prove infection or a unique cell programme.',
                              'Example hypergeometric p/fold and saved q/detail joins are checked; F9106/F1089/F254 receive independent dense full-family BH, other example q-values are joined to saved source annotations.',
                              'Raw counts verify every selected line sentence. Selected count matrices are hashed; whole large source h5ads are not rehashed.',
                              'Line firing arrays and mean shifts are checked with fixed dictionary/mean provenance; line SAE reconstruction forwards are not rerun.']}
    args.out.parent.mkdir(parents=True, exist_ok=True); args.out.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({key: report[key] for key in ['passed', 'n_checks', 'findings', 'peak_rss_bytes']}))
    raise SystemExit(0 if audit.passed else 1)


if __name__ == '__main__': main()
