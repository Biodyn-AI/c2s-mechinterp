#!/usr/bin/env python3
"""Independent saved-data sensitivity audit; CPU only, no training/model forward pass.

Checks full-family BH on six deterministic features per distinct annotation file,
all saved annotation decisions/counts/denominators, catalogue firing, report joins,
pooling geometry and checkpoint/mean identities. Large residual arrays are memmapped.
"""
import argparse
from collections import Counter
import gc
import json
from pathlib import Path
import resource
import re
import sys

import numpy as np

from audit_experimental_saved import Audit, annotation_counts, catalogue_checks
from audit_annotation_dense import references, dense_tests, EXTRA


def stream_moments(matrix, chunk=2048):
    total, square = np.zeros(matrix.shape[1]), 0.
    for start in range(0, len(matrix), chunk):
        block = np.asarray(matrix[start:start+chunk], float)
        total += block.sum(axis=0)
        square += np.einsum('ij,ij->', block, block)
    mean = total/len(matrix)
    return mean, float(square/len(matrix)-np.dot(mean, mean))


def selected_gene_similarity(matrix, gene_rows):
    means = np.array([np.mean(matrix[indices], axis=0, dtype=np.float64) for indices in gene_rows])
    means -= means.mean(axis=0)
    means /= np.linalg.norm(means, axis=1, keepdims=True)+1e-12
    return means @ means.T


def legacy_references(directory, universe, minimum, maximum, audit):
    terms, regulatory, factors, regulons = [], set(), set(), {}
    for source, filename in [('GO_BP', 'go_bp_gene_sets.json'), ('KEGG', 'kegg_gene_sets.json'), ('Reactome', 'reactome_gene_sets.json')]:
        for name, values in audit.read(directory/filename).items():
            genes = {gene.upper() for gene in values} & universe
            if minimum <= len(genes) <= maximum: terms.append((source, name, genes))
    path = directory/'trrust_human.tsv'; audit.hash(path)
    for line in path.read_text().splitlines():
        fields = line.split('\t')
        if len(fields) >= 2:
            tf, target = fields[0].upper(), fields[1].upper()
            factors.add(tf); regulons.setdefault(tf, set()).add(target)
            if tf != target and tf in universe and target in universe: regulatory.add(tuple(sorted([tf, target])))
    for tf, values in regulons.items():
        genes = values & universe
        if minimum <= len(genes) <= maximum: terms.append(('TRRUST', tf+' targets', genes))
    string = {tuple(sorted([a.upper(), b.upper()])) for a, b in audit.read(directory/'string_ppi_edges.json')['pairs_700']
              if a.upper() != b.upper() and a.upper() in universe and b.upper() in universe}
    return terms, string, regulatory, factors & universe


def chosen_features(rows):
    ordered = sorted(rows, key=lambda row: row['feature_idx'])
    choices = {min(ordered, key=lambda r: (r['n_genes'], r['feature_idx']))['feature_idx'],
               min(ordered, key=lambda r: (r['min_p'], r['feature_idx']))['feature_idx'],
               min(ordered, key=lambda r: (abs(min((t['q'] for t in r.get('terms', [])), default=1)-.05), r['feature_idx']))['feature_idx']}
    remaining = [row['feature_idx'] for row in ordered if row['feature_idx'] not in choices]
    choices.update(np.random.default_rng(20261117).choice(remaining, min(6-len(choices), len(remaining)), replace=False).tolist())
    return sorted(choices)


def dense_sample(catalogue, annotation, universe, reference, audit):
    terms, string, regulatory, factors = reference
    identities = [source+'::'+name for source, name, _ in terms]+[source+'::'+name for source, name in EXTRA]
    positions = {identity: index for index, identity in enumerate(identities)}
    summary = annotation['summary']
    audit.equal('full test family including p=1 tests', summary['tests_per_feature'], len(identities))
    audit.equal('named families after universe restriction', summary['terms_by_source'], dict(Counter(source for source, _, _ in terms)))
    audit.equal('STRING background', summary['string_pairs_in_universe'], len(string))
    audit.equal('TRRUST pair background', summary['trrust_pairs_in_universe'], len(regulatory))
    catalogs = {row['feature_idx']: row for row in catalogue['features']}
    selected = chosen_features(annotation['features'])
    for row in annotation['features']:
        if row['feature_idx'] not in selected: continue
        genes = {str(gene['gene_name'] if isinstance(gene, dict) else gene).upper()
                 for gene in catalogs[row['feature_idx']]['top_genes']} & universe
        p, q, _, _ = dense_tests(genes, universe, terms, string, regulatory, factors, summary['min_overlap'])
        called = q < summary['alpha']
        complete = {identities[index] for index in np.flatnonzero(called[:len(terms)])}
        audit.equal(f'F{row["feature_idx"]} dense significant count', row['n_sig'], int(called.sum()))
        audit.equal(f'F{row["feature_idx"]} restricted genes', row['n_genes'], len(genes))
        audit.equal(f'F{row["feature_idx"]} full significant named IDs', set(row['significant_term_ids']), complete)
        for term in row.get('terms', []):
            index = positions[term['source']+'::'+term['term']]
            audit.equal('dense reported p', term['p'], p[index], atol=1e-300)
            audit.equal('dense reported q', term['q'], q[index], atol=1e-300)
            if term['source'] in ['GO_BP', 'KEGG', 'Reactome', 'TRRUST']:
                audit.equal('dense reported fold', term['fold'], (term['k']/len(genes))/(len(terms[index][2])/len(universe)))
    return selected


def audit_entries(root, saved, audit):
    atlas = root/'runs/2b_atlas'
    names = audit.read(atlas/'row_gene_names.json'); universe = {gene.upper() for gene in names if gene != '<CELL>'}
    audit.equal('atlas gene-token positions', len(names), 614400)
    full_reference = references(root/'ontologies_v2', universe, 5, 500)
    for path in full_reference[-1]: audit.hash(path)
    legacy = legacy_references(root/'data/ontologies', universe, 5, 500, audit)
    entries = list(saved['sweeps'])+list(saved['pooling']['entries'])+list(saved['untrained'])
    entries += [pair[kind] for pair in saved['library_sensitivity'] for kind in ['full', 'legacy']]
    checked, contrasts, mean_h = {}, [], None
    for entry in entries:
        audit.context = entry['name']+f'/L{entry["layer"]}: '
        audit.equal('report entry complete', entry['status'], 'complete')
        if entry['status'] != 'complete': continue
        catalogue_path, annotation_path = [root/path for path in entry['sources'][:2]]
        if str(annotation_path) not in checked:
            catalogue, annotation = audit.read(catalogue_path), audit.read(annotation_path)
            recounted = annotation_counts(annotation, audit)
            catalogue_checks(catalogue, annotation, audit, len(names))
            sampled = dense_sample(catalogue, annotation, universe,
                legacy if annotation['summary']['legacy_libraries'] else full_reference[:4], audit)
            extra = {'n_testable': sum(row['n_genes'] >= 3 for row in annotation['features']),
                     'n_network': sum(any(t['source'] in ['STRING', 'TRRUST_pairs', 'TF_membership'] and t['q'] < annotation['summary']['alpha']
                                          for t in row.get('terms', [])) for row in annotation['features'])}
            checked[str(annotation_path)] = {**recounted, **extra, 'sampled_dense_feature_ids': sampled}
            if catalogue_path == atlas/'feature_catalog_L13.json':
                mean_h = np.asarray([row['mean_activation'] for row in sorted(catalogue['features'], key=lambda row: row['feature_idx'])])
            frequencies = np.asarray([row['activation_freq'] for row in catalogue['features']])
            checked[str(annotation_path)]['firing'] = {
                'dead_all_rows': int((frequencies == 0).sum()), 'dead_frac_all_rows': float((frequencies == 0).mean()),
                'mean_active_per_position': float(frequencies.sum()),
                'firing_freq_quantiles_1_10_50_90_99': np.quantile(frequencies[frequencies > 0], [.01, .1, .5, .9, .99]).tolist(),
                'share_features_firing_gt_10pct': float((frequencies > .1).mean()),
                'share_features_firing_lt_1e-4': float(((frequencies < 1e-4) & (frequencies > 0)).mean())}
            del catalogue, annotation; gc.collect()
        counts, report = checked[str(annotation_path)], entry['annotation']
        for key, expected in [('n_features', counts['n_features']), ('n_catalogue_active', counts['n_alive']),
                              ('n_catalogue_inactive', counts['n_features']-counts['n_alive']), ('n_any', counts['n_annotated']),
                              ('n_ontology', counts['n_ontology']), ('n_testable', counts['n_testable']), ('n_network_evidence', counts['n_network'])]:
            audit.equal('report '+key, report[key], expected)
        for family, actual in [('any', counts['any_rate_alive']), ('ontology', counts['ontology_rate_alive'])]:
            audit.equal('report active '+family, report['rates_catalogue_active'][family], actual)
            audit.equal('report all-feature '+family, report['rates_all_features'][family], actual*counts['n_alive']/counts['n_features'])
        for kind, floor in counts['floors'].items():
            for definition, target in [('catalogue_active', 'alive'), ('all_features', 'all_features')]:
                for family in ['any', 'ontology']:
                    audit.equal(f'report {kind}/{definition}/{family} floor', report['floors'][kind][definition][family+'_mean'], floor[family+'_'+target])
        if 'metrics' in entry:
            training = audit.read(root/entry['sources'][2]); baseline = '_k32_x4' in entry['name']
            measurement = audit.read(root/entry['sources'][3]) if baseline else training
            metrics = entry['metrics']; total = counts['n_features']
            audit.equal('reported held-out VE', metrics['variance_explained'], measurement['variance_explained_reextracted'] if baseline else training['variance_explained'])
            dead = measurement['dead_val_reextracted'] if baseline else training['dead_features']
            audit.equal('reported held-out inactive count', metrics['heldout_inactive'], dead)
            audit.equal('reported held-out inactive fraction', metrics['heldout_inactive_fraction'], dead/total)
            audit.equal('reported fixed checkpoint decoder cosine', metrics['mean_decoder_cos'], training['mean_decoder_cos'])
            for key, value in counts['firing'].items(): audit.equal('reported all-row '+key, metrics['firing'][key], value)
            audit.check('held-out/all-row inactivity denominators distinct and bounded', counts['n_features']-counts['n_alive'] <= dead <= total)
            setting = re.fullmatch(r'sweep_L(\d+)_k(\d+)_x(\d+)', entry['name'])
            layer, k, expansion = map(int, setting.groups())
            audit.equal('setting layer', entry['layer'], layer)
            audit.equal('setting sparsity', training['k'], k)
            audit.equal('setting dictionary width', total, expansion*2304)
            for key, value in metrics['training_configuration'].items():
                audit.equal('reported training configuration '+key, value, training.get(key))
            if not baseline:
                run = root/'runs'/entry['name']
                audit.check('same saved activation input inode', (run/f'layer_{layer:02d}_activations.npy').samefile(atlas/f'layer_{layer:02d}_activations.npy'))
                for filename in ['row_gene_names.json', 'row_cell_ids.npy']:
                    audit.check('same source-row input inode '+filename, (run/filename).samefile(atlas/filename))
                for key, value in [('epochs', 6), ('seed', 0), ('n_rows', len(names)), ('d_model', 2304), ('n_features', total)]:
                    audit.equal('variant configuration '+key, training[key], value)
        contrasts.append({'name': entry['name'], 'layer': entry['layer'], 'n_features': counts['n_features'],
                          'ontology_all_features': report['rates_all_features']['ontology'],
                          'frequency_floor_all_features': counts['floors']['freq']['ontology_all_features'],
                          'ontology_excess_all_features': report['rates_all_features']['ontology']-counts['floors']['freq']['ontology_all_features'],
                          'heldout_ve': entry.get('metrics', {}).get('variance_explained'),
                          'n_dense_features_checked': len(counts['sampled_dense_feature_ids'])})
        print('audited', entry['name'], entry['layer'], flush=True)
    return contrasts, mean_h


def audit_weight_identities(root, audit):
    import torch
    torch.set_num_threads(4)
    atlas, records = root/'runs/2b_atlas', []
    for pool in ['first', 'last']:
        run = root/'runs'/('pool_'+pool)
        audit.context = 'pool_'+pool+': '
        audit.check('same trained checkpoint inode', (run/'sae/sae_L13.pt').samefile(atlas/'sae/sae_L13.pt'))
        audit.check('same frozen mean inode', (run/'sae/mu_L13.npy').samefile(atlas/'sae/mu_L13.npy'))
        for name in ['row_gene_names.json', 'row_cell_ids.npy']:
            audit.hash(run/name); audit.hash(atlas/name)
            audit.equal('exact source-row file '+name, audit.sources[str(run/name)], audit.sources[str(atlas/name)])
        manifest = audit.read(run/'manifest.json'); mean_manifest = audit.read(atlas/'manifest.json')
        audit.equal('pooling rule', manifest['pool'], pool)
        for key in ['model', 'model_commit', 'n_rows', 'n_cells', 'max_genes', 'prompt', 'd_model']:
            audit.equal('same extraction '+key, manifest[key], mean_manifest[key])
    for layer in [0, 13, 22]:
        audit.context = f'untrained/L{layer}: '
        run = root/'runs'/f'untrained_L{layer:02d}'
        paths = [atlas/'sae'/f'sae_L{layer:02d}.pt', run/'sae'/f'sae_L{layer:02d}.pt']
        for path in paths: audit.hash(path)
        trained, random = [torch.load(path, map_location='cpu', weights_only=False) for path in paths]
        a, b = trained['model_state_dict'], random['model_state_dict']
        for key in ['d_model', 'n_features', 'k']: audit.equal('dictionary '+key, random[key], trained[key])
        norm_errors = []
        for start in range(0, trained['n_features'], 512):
            x = a['W_enc.weight'][start:start+512].double().norm(dim=1)
            y = b['W_enc.weight'][start:start+512].double().norm(dim=1)
            norm_errors.append(float(((x-y).abs()/x.clamp(min=1e-12)).max()))
        maximum = max(norm_errors)
        audit.check('encoder row norms matched within float32 precision', maximum < 1e-6, maximum, '<1e-6')
        for key in ['W_enc.bias', 'W_dec.bias']: audit.check('trained bias retained '+key, torch.equal(a[key], b[key]))
        audit.check('encoder directions changed', not torch.equal(a['W_enc.weight'], b['W_enc.weight']))
        means = [path.parent/f'mu_L{layer:02d}.npy' for path in paths]
        for path in means: audit.hash(path)
        audit.check('same centering mean values', np.array_equal(np.load(means[0]), np.load(means[1])))
        audit.check('same activation input inode', (run/f'layer_{layer:02d}_activations.npy').samefile(atlas/f'layer_{layer:02d}_activations.npy'))
        records.append({'layer': layer, 'max_relative_encoder_norm_difference': maximum, 'trained_biases_retained': True})
        del trained, random, a, b; gc.collect()
    return records


def audit_pool_geometry(root, saved, mean_h, audit):
    import torch
    atlas = root/'runs/2b_atlas'
    names = np.asarray(audit.read(atlas/'row_gene_names.json'))
    unique, inverse, counts = np.unique(names, return_inverse=True, return_counts=True)
    candidates = np.flatnonzero(counts >= 20)
    chosen = np.random.RandomState(0).choice(candidates, min(600, len(candidates)), replace=False)
    order = np.argsort(inverse, kind='stable'); ends = np.r_[0, np.cumsum(counts)]
    gene_rows = [order[ends[index]:ends[index+1]] for index in chosen]
    mean_path = atlas/'layer_13_activations.npy'; audit.hash(mean_path)
    mean = np.load(mean_path, mmap_mode='r')
    reference = selected_gene_similarity(mean, gene_rows); upper = np.triu_indices(len(chosen), 1)
    output = {'n_genes': len(chosen), 'n_gene_positions': len(mean), 'comparison': saved}
    for pool in ['first', 'last']:
        audit.context = 'pool geometry/'+pool+': '
        path = root/'runs'/('pool_'+pool)/'layer_13_activations.npy'; audit.hash(path)
        matrix = np.load(path, mmap_mode='r'); audit.equal('same array shape', list(matrix.shape), list(mean.shape))
        cosine = np.empty(len(mean))
        for start in range(0, len(mean), 2048):
            a, b = np.asarray(mean[start:start+2048], float), np.asarray(matrix[start:start+2048], float)
            cosine[start:start+len(a)] = np.einsum('ij,ij->i', a, b)/(np.linalg.norm(a, axis=1)*np.linalg.norm(b, axis=1)+1e-12)
        similarity = selected_gene_similarity(matrix, gene_rows)
        audit.equal('row cosine mean', saved[pool]['row_cosine_with_mean_pool_mean'], float(cosine.mean()), atol=1e-6)
        audit.equal('row cosine fifth percentile', saved[pool]['row_cosine_with_mean_pool_p05'], float(np.quantile(cosine, .05)), atol=1e-6)
        audit.equal('gene similarity correlation', saved[pool]['gene_similarity_matrix_correlation'], float(np.corrcoef(similarity[upper], reference[upper])[0, 1]), atol=1e-8)
        del matrix, a, b, cosine
    # E||error||² - ||E(error)||² converts saved all-row FUV to the
    # centered-error VE definition in pool_comparison. It is not held-out VE.
    average, trace = stream_moments(mean)
    path = atlas/'sae/sae_L13.pt'; audit.hash(path)
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)['model_state_dict']
    mu_path = atlas/'sae/mu_L13.npy'; audit.hash(mu_path)
    mu = np.load(mu_path).reshape(-1)
    reconstruction_mean = checkpoint['W_dec.weight'].numpy().astype(float) @ mean_h + checkpoint['W_dec.bias'].numpy()
    error_mean = average-mu-reconstruction_mean
    evaluation = audit.read(root/'results/2b_eval_L13.json')
    reconstructed_ve = 1-evaluation['fraction_variance_unexplained_all_rows']+np.dot(error_mean, error_mean)/trace
    audit.equal('mean-pool all-row centered-error VE from independent moments', saved['mean']['sae_variance_explained'], float(reconstructed_ve), atol=1e-5)
    output['independent_mean_all_row_ve'] = float(reconstructed_ve)
    output['fresh_heldout_ve'] = evaluation['variance_explained_reextracted']
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(); audit = Audit()
    saved = audit.read(args.root/'results/sensitivity_reporting.json')
    audit.equal('all sweeps plus baselines complete', saved['n_complete_sweeps_including_baselines'], 12)
    for section in saved['library_sensitivity']+saved['untrained']+[saved['pooling']]: audit.equal('all sensitivity components complete', section['status'], 'complete')
    contrasts, mean_h = audit_entries(args.root, saved, audit)
    identities = audit_weight_identities(args.root, audit)
    pooling = audit_pool_geometry(args.root, saved['pooling']['comparison'], mean_h, audit)
    # Verify every reporting source checksum, including files consumed repeatedly.
    for relative, source in saved['sources'].items():
        path = args.root/relative
        if str(path) not in audit.sources: audit.hash(path)
        audit.equal('report source hash '+relative, source['sha256'], audit.sources[str(path)])
        audit.equal('report source byte count '+relative, source['bytes'], path.stat().st_size)
    audit.hash(Path(__file__))
    report = {'audit': 'Independent sensitivity saved-data/dictionary/geometry audit', 'passed': audit.passed,
              'n_checks': audit.n_checks, 'findings': audit.failed, 'contrasts': contrasts,
              'untrained_weight_checks': identities, 'pool_geometry': pooling, 'source_sha256': audit.sources,
              'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform == 'darwin' else 1024),
              'limitations': ['Six deterministic strong/boundary/short-set/random features per distinct annotation file receive dense full-family p/q checks; other features receive complete-ID/count/denominator checks.',
                              'Saved seeded floor arithmetic is checked; random gene sets are not regenerated.',
                              'Pooling geometry is recomputed from all saved rows. Mean-pool VE is cross-checked using all-row error/input moments; first/last SAE reconstruction outputs are not rerun.',
                              'Untrained controls retain trained encoder norms, encoder/decoder biases, centering mean and base-model activations; they are not a wholly untrained model.',
                              'Single saved seeds are not independent trained-model replicates. Baseline training JSONs omit seed/lr/batch-size fields; missing values are not imputed.']}
    args.out.parent.mkdir(parents=True, exist_ok=True); args.out.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({key: report[key] for key in ['passed', 'n_checks', 'findings', 'peak_rss_bytes']}))
    raise SystemExit(0 if audit.passed else 1)


if __name__ == '__main__': main()
