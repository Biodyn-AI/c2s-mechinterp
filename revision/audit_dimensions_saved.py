#!/usr/bin/env python3
"""Independent raw-coordinate geometry audit. Reads saved arrays; never fits a model.

Run large cohorts on the approved pod. Distances use SciPy's direct coordinate
differences, and PR uses row-Gram blocks, independently of the reporting code.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import gc
import hashlib
import json
from pathlib import Path
import resource
import sys

import h5py
import numpy as np
from scipy.spatial.distance import cdist

from audit_experimental_saved import Audit
from audit_biology_lines import selected_sparse_rows
from audit_c6_merge import h5_strings


def raw_metrics(vectors, block_size=128, workers=1):
    vectors = np.asarray(vectors, np.float64)
    if vectors.ndim != 2 or min(vectors.shape) < 2 or len(vectors) < 4 or not np.isfinite(vectors).all():
        raise ValueError('Need finite coordinates for at least four genes')
    centered = vectors - vectors.mean(axis=0)
    trace = float(np.square(centered).sum())
    squared_trace = 0.
    for start in range(0, len(vectors), block_size):
        gram = centered[start:start+block_size] @ centered.T
        squared_trace += float(np.square(gram).sum())
    del centered
    # Explicitly remove only the query's own row, retaining duplicate other genes.
    def nearest(start):
        stop = min(start+block_size, len(vectors))
        distances = cdist(vectors[start:stop], vectors, metric='sqeuclidean')
        distances[np.arange(stop-start), np.arange(start, stop)] = np.inf
        pair = np.partition(distances, 1, axis=1)[:, :2]
        pair.sort(axis=1)
        return pair
    starts = range(0, len(vectors), block_size)
    if workers == 1:
        pairs = list(map(nearest, starts))
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            pairs = list(pool.map(nearest, starts))
    first, second = np.sqrt(np.vstack(pairs)).T
    valid = (first > 1e-12) & (second > first)
    log_ratios = np.log(second[valid] / first[valid])
    return {'n_genes': len(vectors), 'n_coordinates': vectors.shape[1],
            'participation_ratio': trace*trace/squared_trace if trace else 0.,
            'twonn_id': float(len(log_ratios)/log_ratios.sum()) if len(log_ratios) and log_ratios.sum() > 0 else None,
            'twonn_valid_rows': int(valid.sum()), 'twonn_excluded_rows': int((~valid).sum()),
            'zero_nearest_rows': int((first <= 1e-12).sum()),
            'positive_tied_nearest_rows': int(((first > 1e-12) & (second <= first)).sum())}


def select_cells(n_observations, count, rng, explicit=None):
    if explicit is None:
        return np.sort(rng.choice(n_observations, min(count, n_observations), replace=False))
    if not isinstance(explicit, list) or not explicit or any(type(value) is not int for value in explicit):
        raise ValueError('Explicit source rows must be a nonempty integer list')
    indices = np.asarray(explicit, dtype=np.int64)
    if len(np.unique(indices)) != len(indices) or np.any(indices < 0) or np.any(indices >= n_observations):
        raise ValueError('Duplicate/out-of-bounds source rows')
    return indices


def log_cp10k(counts):
    counts = np.asarray(counts, np.float32)
    if not np.isfinite(counts).all() or np.any(counts < 0):
        raise ValueError('Counts must be finite and nonnegative')
    return np.log1p(counts / np.maximum(counts.sum(axis=1, keepdims=True), 1) * 1e4)


def grouped_gene_means(names, activations, minimum=20):
    genes, inverse, counts = np.unique(names, return_inverse=True, return_counts=True)
    order = np.argsort(inverse, kind='stable')
    stops = np.cumsum(counts); starts = stops-counts
    kept = np.flatnonzero(counts >= minimum)
    means = np.empty((len(kept), activations.shape[1]), np.float32)
    for destination, group in enumerate(kept):
        occurrence_rows = order[starts[group]:stops[group]]
        values = np.asarray(activations[occurrence_rows], np.float64)
        if not np.isfinite(values).all(): raise ValueError('Nonfinite source activations')
        means[destination] = values.mean(axis=0)
    return genes[kept], counts[kept], means


def array_sha(values):
    values = np.ascontiguousarray(values)
    digest = hashlib.sha256(str(values.dtype).encode()+str(values.shape).encode())
    digest.update(memoryview(values).cast('B'))
    return digest.hexdigest()


def source_path(root, original):
    value = str(original)
    if value.startswith('/workspace/rev1/'):
        return root/value[len('/workspace/rev1/'):]
    return Path(value)


def audit_cohort(root, cohort, audit, workers):
    saved = audit.read(root/'results'/f'{cohort}_dimensions.json')
    inputs = saved['inputs']; rng = np.random.default_rng(inputs['seed'])
    run = root/'runs'/f'geometry_{cohort}'
    design = root/'designs'/('geometry_cd34' if cohort == 'cd34' else 'atlas_k562')
    manifest = audit.read(run/'manifest.json')
    h5ad = source_path(root, inputs['h5ad'])
    explicit = audit.read(source_path(root, inputs['cell_indices'])) if inputs['cell_indices'] else None
    audit.equal('seed is prespecified zero', inputs['seed'], 0)
    audit.equal('no gene cap', inputs['gene_cap'], 0)
    audit.equal('three prespecified layers', inputs['layers'], [9, 13, 17])
    audit.equal('correct source layer', inputs['counts_layer'], 'spliced' if cohort == 'cd34' else '')
    audit.equal('extractor model', manifest['model'], 'vandijklab/C2S-Scale-Gemma-2-2B')
    audit.equal('extractor revision', manifest['model_commit'], '5ddf28b8f1c81b7ab7a9be192924da82b6c5d512')
    audit.equal('mean gene-token pooling', manifest['pool'], 'mean')
    audit.equal('gene-token unit', manifest['unit'], 'gene-token')
    audit.equal('no truncated sentences', manifest['token_stats']['n_truncated_sentences'], 0)
    with h5py.File(h5ad) as source:
        names = h5_strings(source['var'][source['var'].attrs.get('_index', '_index')])
        observation_names = h5_strings(source['obs'][source['obs'].attrs.get('_index', '_index')])
        indices = select_cells(len(observation_names), inputs['n_cells'], rng, explicit)
        count_node = source['layers'][inputs['counts_layer']] if inputs['counts_layer'] else source['X']
        counts = selected_sparse_rows(count_node, indices)
    audit.equal('exact source cell indices/order', saved['cell_indices'], indices)
    audit.equal('expected cohort size', len(indices), 4000 if cohort == 'cd34' else 1200)
    audit.equal('manifest cell count', manifest['n_cells'], len(indices))
    audit.check('unique source gene symbols', len(np.unique(names)) == len(names))
    audit.check('unique source cell identifiers', len(np.unique(observation_names[indices])) == len(indices))
    audit.equal('selected source count columns', counts.shape[1], len(names))
    row_names = np.asarray(audit.read(run/'row_gene_names.json'))
    row_cells = np.load(run/'row_cell_ids.npy')
    design_names = audit.read(design/'row_gene_names.json')
    design_cells = np.load(design/'row_cell_ids.npy')
    audit.equal('extractor names equal design', row_names, np.asarray(design_names))
    audit.equal('extractor IDs equal design', row_cells, design_cells)
    audit.equal('aligned occurrence names and IDs', len(row_names), len(row_cells))
    audit.equal('manifest occurrence count', manifest['n_rows'], len(row_names))
    # Original CD34 row IDs are source indices; atlas K562 IDs index the explicit design.
    expected_ids = indices if cohort == 'cd34' else np.arange(len(indices))
    expected_names, expected_cells = [], []
    for position, (cell_id, values) in enumerate(zip(expected_ids, counts)):
        positive = np.flatnonzero(values > 0)
        chosen = positive[np.argsort(-values[positive], kind='stable')[:512]]
        expected_names.extend(names[chosen].tolist())
        expected_cells.extend([int(cell_id)]*len(chosen))
    audit.equal('all raw-count sentence genes, stable ties', row_names, np.asarray(expected_names))
    audit.equal('all raw-count sentence cell IDs/order', row_cells, np.asarray(expected_cells))
    audit.equal('extraction unique source cells', np.unique(row_cells), np.sort(expected_ids))
    del expected_names, expected_cells, design_names, design_cells
    expression = log_cp10k(counts)
    count_sha = array_sha(counts); del counts
    model_path = source_path(root, inputs['model_vectors'])
    model = np.load(model_path)
    genes, occurrence_counts = np.unique(row_names, return_counts=True)
    keep = occurrence_counts >= 20
    audit.equal('complete occurrence>=20 model gene universe', model['genes'], genes[keep])
    audit.equal('all model occurrence counts', model['counts'], occurrence_counts[keep])
    lookup = {gene: row for row, gene in enumerate(names)}
    model_rows = np.asarray([row for row, gene in enumerate(model['genes']) if str(gene) in lookup])
    gene_rows = np.asarray([lookup[str(model['genes'][row])] for row in model_rows])
    audit.equal('no missing model genes in source panel', len(model_rows), len(model['genes']))
    audit.equal('complete shared gene order', saved['genes'], names[gene_rows].tolist())
    vectors = expression[:, gene_rows].T.copy(); del expression
    metrics_rows, aggregation = [], []
    saved_rows = {row['representation']: row for row in saved['rows']}
    audit.equal('one saved row per representation', len(saved_rows), len(saved['rows']))
    def metric(name, values):
        print('auditing', cohort, name, values.shape, flush=True)
        independent = raw_metrics(values, workers=workers)
        report = saved_rows[name]
        for key in ['n_genes', 'n_coordinates', 'participation_ratio', 'twonn_valid_rows', 'twonn_excluded_rows']:
            audit.equal(name+'/'+key, report[key], independent[key])
        if independent['twonn_id'] is None:
            audit.equal(name+'/twonn_id', report['twonn_id'], None)
        else:
            audit.check(name+'/twonn_id direct-distance agreement',
                        bool(np.isclose(report['twonn_id'], independent['twonn_id'], rtol=1e-6, atol=1e-8)),
                        report['twonn_id'], independent['twonn_id'])
        metrics_rows.append({'representation': name, 'saved': report, 'independent': independent,
                             'matrix_sha256': array_sha(values)})
    metric('expression_log1p_CP10k', vectors)
    shuffled = vectors.copy()
    for row in shuffled: rng.shuffle(row)
    metric('expression_cells_shuffled_within_gene', shuffled); del shuffled
    metric('expression_shape_iid_normal', rng.normal(size=vectors.shape)); del vectors
    for layer in inputs['layers']:
        activation_path = run/f'layer_{layer:02d}_activations.npy'
        activations = np.load(activation_path, mmap_mode='r')
        audit.equal(f'L{layer} source occurrence shape', list(activations.shape), [len(row_names), manifest['d_model']])
        independent_genes, independent_counts, means = grouped_gene_means(row_names, activations)
        values = model[f'L{layer:02d}']
        audit.equal(f'L{layer} aggregation genes', model['genes'], independent_genes)
        audit.equal(f'L{layer} aggregation counts', model['counts'], independent_counts)
        max_error = float(np.max(np.abs(values-means)))
        audit.check(f'L{layer} all gene means match raw occurrence aggregation',
                    bool(np.allclose(values, means, rtol=1e-6, atol=2e-6)), max_error, 'float32 tolerance rtol1e-6/atol2e-6')
        aggregation.append({'layer': layer, 'n_genes': len(means), 'max_abs_gene_mean_error': max_error,
                            'source_shape': list(activations.shape)})
        del means, activations
        metric(f'model_L{layer:02d}', values[model_rows])
        metric(f'model_L{layer:02d}_shape_iid_normal', rng.normal(size=values[model_rows].shape))
        audit.hash(activation_path); gc.collect()
    audit.equal('all nine complete representations', sorted(saved_rows), sorted(row['representation'] for row in metrics_rows))
    for path in [h5ad, model_path, run/'row_cell_ids.npy', design/'row_cell_ids.npy',
                 root/'scripts/rev_cpu_analysis.py', root/'scripts/rev_misc.py', root/'scripts/rev_extract.py']:
        audit.hash(path)
    return {'cohort': cohort, 'n_cells': len(indices), 'n_source_genes': len(names), 'n_shared_genes': len(gene_rows),
            'n_gene_occurrences': len(row_names), 'selected_counts_sha256': count_sha,
            'selected_cell_indices_sha256': array_sha(indices),
            'selected_cell_names_sha256': hashlib.sha256('\n'.join(observation_names[indices]).encode()).hexdigest(),
            'source_counts_layer': inputs['counts_layer'] or 'X',
            'cell_id_semantics': 'original h5 row index' if cohort == 'cd34' else 'ordinal mapped through explicit h5ad_cell_indices.json',
            'aggregation': aggregation, 'rows': metrics_rows,
            'rng': {'seed': inputs['seed'], 'algorithm': 'NumPy default_rng/PCG64',
                    'sampling_consumes_rng': explicit is None,
                    'sequence': 'sample cells if implicit; optional gene cap (zero here); shuffle each shared gene over cells; expression iid normal; layer9 iid normal; layer13 iid normal; layer17 iid normal',
                    'final_state': rng.bit_generator.state},
            'coordinate_definition': 'Rows are genes. Raw model gene means; expression log1p(CP10k) normalized over the entire source gene panel, then projected to the shared genes. No gene standardization or dimensionality reduction.',
            'reference_distinction': 'Earlier standardized geometry (PR approximately94–132; TwoNN approximately5–6.2) uses different coordinates and is not a matched numerical baseline.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--cohort', choices=['cd34', 'k562_replication'], required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--distance-workers', type=int, default=4)
    args = parser.parse_args(); audit = Audit()
    report = audit_cohort(args.root, args.cohort, audit, args.distance_workers)
    for name in ['audit_dimensions_saved.py', 'audit_biology_lines.py', 'audit_c6_merge.py', 'audit_experimental_saved.py']:
        audit.hash(Path(__file__).with_name(name))
    report.update({'schema': 'c2s-dimension-independent-audit-v1', 'passed': audit.passed,
                   'complete': not audit.pending, 'n_checks': audit.n_checks, 'findings': audit.failed,
                   'source_sha256': audit.sources,
                   'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform == 'darwin' else 1024),
                   'limits': ['Saved activations and their exact per-gene aggregation are audited; no model forward is repeated.',
                              'TwoNN is a Euclidean nearest-neighbour estimator, sensitive to coordinate scaling, ties and high-dimensional distance concentration; the nulls are representation-shape controls, not uncertainty estimates.']})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({key: report[key] for key in ['cohort', 'passed', 'complete', 'n_checks', 'findings', 'peak_rss_bytes']}))
    raise SystemExit(0 if audit.passed else 1)


if __name__ == '__main__': main()
