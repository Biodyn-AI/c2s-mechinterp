"""CPU statistics for rev_phase8_extract.py, using exactly the extracted cells.

Run on the pod (h5ad positive controls are too large for routine local work)::

  python rev_phase8_stats.py --features-dir out/phase8_k562 --design designs/phase8_k562.json \
    --catalog-dir runs/2b_atlas --h5ad data/ReplogleWeissman2022_K562_essential.h5ad \
    --trrust ont/trrust_human.tsv --layers 8 16 22 --out out/phase8_k562_stats

The expression positive control is log1p(counts / cell library size * 10000). Input scores are
descending rank weights in the first 512 names (absent genes have weight zero). All feature/gene
response tests are two-sided Mann--Whitney tests, BH across ALL columns, and |d| > 0.5. To retain
the original pipeline's effect-size convention, d divides by sqrt((var(P)+var(C))/2), ddof=0.
Specificity uses one-sided Fisher enrichment of known targets among responses, followed by BH
across TFs, separately for each representation/layer. The fixed Fisher universe is every column
with any nonzero observation across extracted groups. No TF is selected using its outcome.
All screened TFs remain in the specificity BH family. Separate coverage counts show which
factors have at least 1, 3 or 5 eligible known target genes. The perturbed gene is excluded
from gene-level specificity/retrieval and from known-target labels, including autoregulation.

Average precision (stepwise AUPRC) and AUROC use absolute response effect size. Gene scores from
the model average absolute feature effects across features listing that gene in their top genes;
genes without a catalogue membership get score zero. Gene retrieval uses the h5ad gene universe
for all representations, except the directly perturbed gene. Prevalence lift and lift over the exact
tie-aware random-label AP expectation are descriptive ratios, not significance tests.
Bootstrap intervals resample positives/negatives separately: they describe
gene/feature retrieval conditional on the observed cell sample, not uncertainty over new cells.
The optional in-silico file uses its paired cell differences and keeps that analysis separate from
measured knockdowns; deleting a name is an intervention on model text, not a biological intervention.
"""
from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import time

import numpy as np
from scipy import sparse
from scipy.stats import fisher_exact, mannwhitneyu, t as student_t
from sklearn.metrics import average_precision_score, roc_auc_score


def bh_adjust(pvalues):
    """BH adjustment retaining all supplied tests, including p=1 null columns."""
    p = np.asarray(pvalues, dtype=float)
    if not len(p):
        return p.copy()
    if np.any(~np.isfinite(p)) or np.any((p < 0) | (p > 1)):
        raise ValueError('p-values must be finite and between zero and one')
    order = np.argsort(p, kind='stable')
    ranked = np.minimum.accumulate((p[order] * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
    adjusted = np.empty_like(p)
    adjusted[order] = np.minimum(ranked, 1.)
    return adjusted


def response_statistics(perturbed, control, alpha=.05, effect_threshold=.5, chunk=512):
    """Column-wise tests with a bounded temporary allocation; inputs may be sparse."""
    if perturbed.shape[1] != control.shape[1] or min(len(perturbed.shape), len(control.shape)) != 2:
        raise ValueError('response matrices must have the same columns')
    if min(perturbed.shape[0], control.shape[0]) < 3:
        raise ValueError('at least three cells per group are required')
    n_columns = control.shape[1]
    p = np.ones(n_columns)
    effect = np.zeros(n_columns)
    mean_difference = np.zeros(n_columns)
    for start in range(0, n_columns, chunk):
        stop = min(start + chunk, n_columns)
        a, b = perturbed[:, start:stop], control[:, start:stop]
        a = a.toarray() if sparse.issparse(a) else np.asarray(a)
        b = b.toarray() if sparse.issparse(b) else np.asarray(b)
        a, b = a.astype(float), b.astype(float)
        difference = a.mean(0) - b.mean(0)
        scale = np.sqrt((a.var(0) + b.var(0)) / 2.)
        varying = (scale > 0) | (difference != 0)
        block_p = np.ones(stop - start)
        if varying.any():
            block_p[varying] = mannwhitneyu(a[:, varying], b[:, varying], axis=0,
                                            alternative='two-sided', method='asymptotic').pvalue
        p[start:stop] = block_p
        effect[start:stop] = difference / np.maximum(scale, 1e-9)
        mean_difference[start:stop] = difference
    q = bh_adjust(p)
    return {'p': p, 'q': q, 'effect': effect, 'mean_difference': mean_difference,
            'responding': (q < alpha) & (np.abs(effect) > effect_threshold)}


def specificity_test(responding, targets, eligible):
    responding, targets, eligible = (np.asarray(x, bool) for x in (responding, targets, eligible))
    r, target = responding & eligible, targets & eligible
    table = [[int((r & target).sum()), int((r & ~target).sum())],
             [int((~r & target & eligible).sum()), int((~r & ~target & eligible).sum())]]
    p = float(fisher_exact(table, alternative='greater')[1]) if target.any() and r.any() else 1.
    n_target, n_other = int(target.sum()), int((eligible & ~target).sum())
    reason = ('no eligible known targets' if not n_target else
              'no eligible non-target columns' if not n_other else None)
    return {'table': table, 'p': p, 'n_eligible': int(eligible.sum()),
            'n_target_columns': n_target, 'n_non_target_columns': n_other,
            'n_target_responses': table[0][0], 'n_responding_eligible': int(r.sum()),
            'evaluable': reason is None, 'unevaluable_reason': reason}


def downstream_target_labels(genes, known_targets, perturbed_gene):
    """Exclude the directly perturbed gene, including curated autoregulatory edges.

    Its expression loss is an expected direct intervention effect and cannot validate
    retrieval of downstream targets. Apply the same exclusion to every representation.
    """
    factor = perturbed_gene.upper()
    known = {gene.upper() for gene in known_targets} - {factor}
    candidates = np.array([gene.upper() != factor for gene in genes])
    labels = np.array([gene.upper() in known for gene in genes])
    return known, labels, candidates


def random_label_expected_ap(scores, n_positive):
    """Exact AP expectation for uniformly permuted labels, respecting score ties.

    Average precision evaluates precision at each tied group's end. Conditioning on
    an item being positive gives the expected number of other positives in that
    prefix; summing over groups avoids an expensive permutation loop.
    """
    scores = np.asarray(scores, float)
    n_items = len(scores)
    if not n_items or not 0 <= n_positive <= n_items:
        raise ValueError('AP expectation needs a nonempty score vector and valid positive count')
    if not n_positive:
        return 0.
    if n_items == 1:
        return 1.
    _, group_sizes = np.unique(scores, return_counts=True)
    group_sizes = group_sizes[::-1]
    prefix_sizes = np.cumsum(group_sizes)
    return float(np.sum(group_sizes / (n_items * prefix_sizes) *
                        (1 + (prefix_sizes - 1) * (n_positive - 1) / (n_items - 1))))


def retrieval_metrics(labels, scores, bootstrap=200, seed=0):
    labels, scores = np.asarray(labels, bool), np.asarray(scores, float)
    if labels.shape != scores.shape or labels.ndim != 1 or not np.isfinite(scores).all():
        raise ValueError('retrieval labels/scores must be aligned finite vectors')
    positives, negatives = np.flatnonzero(labels), np.flatnonzero(~labels)
    prevalence = float(labels.mean()) if len(labels) else 0.
    expected_ap = random_label_expected_ap(scores, len(positives)) if len(labels) else 0.
    record = {'n': len(labels), 'n_positive': len(positives), 'prevalence': prevalence,
              'average_precision': None, 'auroc': None, 'ap_lift': None,
              'prevalence_lift': None, 'random_label_expected_ap': expected_ap,
              'lift_over_random_expected_ap': None,
              'average_precision_ci95': None, 'auroc_ci95': None, 'ap_lift_ci95': None,
              'lift_over_random_expected_ap_ci95': None,
              'bootstrap': int(bootstrap), 'bootstrap_unit': 'stratified retrieval items'}
    if not len(positives) or not len(negatives):
        return record
    ap = float(average_precision_score(labels, scores))
    record.update(average_precision=ap, auroc=float(roc_auc_score(labels, scores)),
                  ap_lift=ap / prevalence, prevalence_lift=ap / prevalence,
                  lift_over_random_expected_ap=ap / expected_ap)
    if bootstrap:
        rng = np.random.default_rng(seed)
        replicates = np.empty((bootstrap, 2))
        for i in range(bootstrap):
            index = np.concatenate([rng.choice(positives, len(positives), replace=True),
                                    rng.choice(negatives, len(negatives), replace=True)])
            replicates[i] = (average_precision_score(labels[index], scores[index]),
                             roc_auc_score(labels[index], scores[index]))
        intervals = np.quantile(replicates, [.025, .975], axis=0)
        record['average_precision_ci95'] = intervals[:, 0].tolist()
        record['auroc_ci95'] = intervals[:, 1].tolist()
        record['ap_lift_ci95'] = (intervals[:, 0] / prevalence).tolist()
        record['lift_over_random_expected_ap_ci95'] = (intervals[:, 0] / expected_ap).tolist()
    return record


def feature_gene_scores(feature_scores, catalog, genes):
    """Average effects over catalogue memberships; never use TF target labels as weights."""
    gene_index = {g.upper(): i for i, g in enumerate(genes)}
    total, count = np.zeros(len(genes)), np.zeros(len(genes), int)
    for position, feature in enumerate(catalog):
        feature_index = int(feature.get('feature_idx', position))
        if feature_index >= len(feature_scores):
            raise ValueError('catalogue feature index exceeds extracted feature width')
        names = {str(g['gene_name'] if isinstance(g, dict) else g).upper()
                 for g in feature.get('top_genes', [])}
        for name in names:
            if name in gene_index:
                index = gene_index[name]
                total[index] += float(feature_scores[feature_index])
                count[index] += 1
    return total / np.maximum(count, 1), count


def input_rank_matrix(sentences, genes, max_genes=512):
    gene_index = {g.upper(): i for i, g in enumerate(genes)}
    matrix = np.zeros((len(sentences), len(genes)), np.float32)
    for row, sentence in enumerate(sentences):
        for rank, name in enumerate(sentence[:max_genes]):
            column = gene_index.get(name.upper())
            if column is not None:
                matrix[row, column] = max(matrix[row, column], (max_genes - rank) / max_genes)
    return matrix


def insilico_statistics(arrays, min_cells=5):
    """Paired t summary from sufficient statistics; BH per TF/layer over all genes."""
    count = np.asarray(arrays['count'], float)[..., None]
    difference_sum = np.asarray(arrays['sum_tf']) - np.asarray(arrays['sum_ctrl'])
    mean = difference_sum / np.maximum(count, 1)
    variance = np.maximum(np.asarray(arrays['sum_d2']) - difference_sum**2 / np.maximum(count, 1), 0)
    variance /= np.maximum(count - 1, 1)
    se = np.sqrt(variance / np.maximum(count, 1))
    statistic = mean / np.maximum(se, 1e-12)
    eligible = np.broadcast_to(count >= min_cells, mean.shape)
    p = np.where(eligible, 2 * student_t.sf(np.abs(statistic), np.maximum(count - 1, 1)), 1.)
    q = np.ones_like(p)
    for tf in range(p.shape[0]):
        for layer in range(p.shape[2]):
            q[tf, :, layer] = bh_adjust(p[tf, :, layer])
    return {'mean_difference': mean, 'standard_error': se, 't': statistic, 'p': p, 'q': q,
            'eligible': eligible}


def load_targets(path):
    targets = {}
    with open(path) as handle:
        for line in handle:
            fields = line.rstrip('\n').split('\t')
            if len(fields) >= 2:
                targets.setdefault(fields[0].upper(), set()).add(fields[1].upper())
    return targets


def write_json(path, record):
    with open(path, 'w') as handle:
        json.dump(record, handle, indent=2, allow_nan=False)


def summarize_retrieval(records, field, bootstrap, seed):
    """Mean retrieval across factors, with factors (not genes) as the resampling unit."""
    selected = [record[field] for record in records
                if field in record and record[field]['average_precision'] is not None]
    summary = {'n_tfs_evaluable': len(selected), 'bootstrap_unit': 'transcription factors'}
    rng = np.random.default_rng(seed)
    for metric in ('average_precision', 'auroc', 'ap_lift', 'prevalence', 'prevalence_lift',
                   'random_label_expected_ap', 'lift_over_random_expected_ap'):
        values = np.array([record[metric] for record in selected], float)
        summary[f'mean_{metric}'] = float(values.mean()) if len(values) else None
        summary[f'mean_{metric}_ci95'] = None
        if len(values) and bootstrap:
            means = rng.choice(values, size=(bootstrap, len(values)), replace=True).mean(1)
            summary[f'mean_{metric}_ci95'] = np.quantile(means, [.025, .975]).tolist()
    return summary


def summarize_specificity(factors, alpha):
    """Retain the full screened BH family; expose prespecified coverage sensitivities.

    Thresholds count distinct known target genes, not the potentially much larger
    number of features containing those genes. Zero-coverage factors are screened
    but cannot provide interpretable evidence about downstream target specificity.
    """
    sensitivity = {}
    for minimum in (1, 3, 5):
        eligible = [record for record in factors if record['specificity']['evaluable'] and
                    record['n_eligible_known_target_genes'] >= minimum]
        sensitivity[f'min_{minimum}_eligible_target_genes'] = {
            'n_tfs_evaluable': len(eligible),
            'n_specific_full_family_bh': sum(r['specificity'].get('q', 1) < alpha for r in eligible)}
    return {'n_tfs_screened': len(factors),
            'n_tfs_evaluable': sensitivity['min_1_eligible_target_genes']['n_tfs_evaluable'],
            'n_tfs_unevaluable': len(factors) - sensitivity['min_1_eligible_target_genes']['n_tfs_evaluable'],
            'target_coverage_sensitivity': sensitivity}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--features-dir', required=True)
    parser.add_argument('--design', required=True)
    parser.add_argument('--catalog-dir', required=True)
    parser.add_argument('--h5ad', required=True)
    parser.add_argument('--trrust', required=True)
    parser.add_argument('--layers', type=int, nargs='+', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--expression-layer', default='', help='default: h5ad.X, which must contain raw counts')
    parser.add_argument('--bootstrap', type=int, default=200)
    parser.add_argument('--seed', type=int, default=20261117)
    parser.add_argument('--alpha', type=float, default=.05)
    parser.add_argument('--effect-threshold', type=float, default=.5)
    parser.add_argument('--max-genes', type=int, default=512)
    args = parser.parse_args()
    import anndata
    started = time.time()
    out, feature_dir = Path(args.out), Path(args.features_dir)
    out.mkdir(parents=True, exist_ok=True)
    design = json.loads(Path(args.design).read_text())
    rows = json.loads((feature_dir / 'rows.json').read_text())
    design_cells = {(g['group'], int(c)): s for g in design['groups']
                    for c, s in zip(g['cells'], g['sentences'])}
    if len({(r['group'], int(r['cell_index'])) for r in rows}) != len(rows):
        raise ValueError('duplicate extracted group/cell rows')
    sentences = [design_cells[(r['group'], int(r['cell_index']))] for r in rows]
    groups = list(dict.fromkeys(r['group'] for r in rows))
    group_indices = {g: np.array([i for i, r in enumerate(rows) if r['group'] == g]) for g in groups}
    group_kind = {r['group']: r['kind'] for r in rows}
    controls = [g for g in groups if group_kind[g] == 'control']
    if len(controls) != 1:
        raise ValueError('exactly one extracted control group is required')
    control_group = controls[0]
    targets = load_targets(args.trrust)
    adata = anndata.read_h5ad(args.h5ad, backed='r')
    genes = [str(g).upper() for g in adata.var_names]
    if len(set(genes)) != len(genes):
        raise ValueError('h5ad gene symbols must be unique after uppercasing; resolve aliases explicitly')
    source_matrix = adata.layers[args.expression_layer] if args.expression_layer else adata.X

    def expression_for(indices):
        cell_indices = np.array([int(rows[i]['cell_index']) for i in indices])
        order = np.argsort(cell_indices)
        selected = source_matrix[cell_indices[order], :]
        selected = sparse.csr_matrix(selected, dtype=np.float32)
        inverse = np.argsort(order)
        selected = selected[inverse]
        if selected.data.size and (np.any(selected.data < 0) or
                                  np.max(np.abs(selected.data - np.rint(selected.data))) > 1e-3):
            raise ValueError('expression control requires nonnegative raw counts; select the correct --expression-layer')
        library_size = np.asarray(selected.sum(1)).ravel()
        selected = sparse.diags(10000 / np.maximum(library_size, 1)) @ selected
        selected.data = np.log1p(selected.data)
        return selected.tocsr()

    control_indices = group_indices[control_group]
    expression_control = expression_for(control_indices)
    input_control = input_rank_matrix([sentences[i] for i in control_indices], genes, args.max_genes)
    # Fixed per-representation backgrounds. Genes can respond even if absent in controls.
    expression_eligible = np.asarray((expression_control != 0).sum(0)).ravel() > 0
    input_eligible = np.any(input_control != 0, axis=0)
    for group in groups:
        if group == control_group:
            continue
        expression_eligible |= np.asarray((expression_for(group_indices[group]) != 0).sum(0)).ravel() > 0
        input_eligible |= np.any(input_rank_matrix([sentences[i] for i in group_indices[group]],
                                                  genes, args.max_genes) != 0, axis=0)
    records = []
    arrays = {'genes': np.asarray(genes), 'cell_indices': np.array([r['cell_index'] for r in rows]),
              'expression_eligible': expression_eligible, 'input_eligible': input_eligible}
    conditions = [g for g in groups if g != control_group and len(group_indices[g]) >= 3]
    arrays['groups'] = np.asarray(conditions)
    for representation, eligible, control in [('expression', expression_eligible, expression_control),
                                               ('input', input_eligible, input_control)]:
        columns = {key: [] for key in ('p', 'q', 'effect', 'mean_difference', 'responding')}
        for index, group in enumerate(conditions):
            indices = group_indices[group]
            matrix = (expression_for(indices) if representation == 'expression' else
                      input_rank_matrix([sentences[i] for i in indices], genes, args.max_genes))
            stats = response_statistics(matrix, control, args.alpha, args.effect_threshold)
            known = targets.get(group.upper(), set()) if group_kind[group] == 'tf' else set()
            known, labels, candidates = downstream_target_labels(genes, known, group)
            specificity = specificity_test(stats['responding'], labels, eligible & candidates)
            retrieval = retrieval_metrics(labels[candidates], np.abs(stats['effect'][candidates]),
                                          args.bootstrap, args.seed + index)
            records.append({'representation': representation, 'layer': None, 'group': group,
                            'kind': group_kind[group], 'n_cells': len(indices),
                            'n_control_cells': len(control_indices),
                            'n_responding': int(stats['responding'].sum()),
                            'n_eligible_known_target_genes': int((labels & eligible).sum()),
                            'self_target_excluded': True,
                            'perturbed_gene_excluded_from_gene_retrieval': True,
                            'specificity': specificity, 'gene_retrieval': retrieval})
            for key in columns:
                columns[key].append(stats[key])
            print(f'{representation} {group}: {stats["responding"].sum()} responses, specificity p={specificity["p"]:.3g}', flush=True)
        for key, values in columns.items():
            arrays[f'{representation}_{key}'] = np.asarray(values)
    np.savez_compressed(out / 'gene_statistics.npz', **arrays)
    del input_control, expression_control, arrays

    for layer in args.layers:
        matrix = np.load(feature_dir / f'feat_L{layer:02d}.npz')['X']
        if matrix.ndim != 2 or len(matrix) != len(rows) or not np.isfinite(matrix).all():
            raise ValueError(f'feature matrix L{layer} is not aligned with rows.json or has nonfinite values')
        catalog_json = json.loads((Path(args.catalog_dir) / f'feature_catalog_L{layer:02d}.json').read_text())
        catalog = catalog_json['features'] if isinstance(catalog_json, dict) else catalog_json
        feature_targets = []
        for position, feature in enumerate(catalog):
            fi = int(feature.get('feature_idx', position))
            feature_targets.append((fi, {str(g['gene_name'] if isinstance(g, dict) else g).upper()
                                         for g in feature.get('top_genes', [])}))
        eligible = np.any(matrix != 0, axis=0)
        active_catalog_genes = set().union(*(gene_set for fi, gene_set in feature_targets if eligible[fi]))
        values = {key: [] for key in ('p', 'q', 'effect', 'mean_difference', 'responding')}
        gene_score_rows = []
        for index, group in enumerate(conditions):
            indices = group_indices[group]
            stats = response_statistics(matrix[indices], matrix[control_indices], args.alpha, args.effect_threshold)
            known = targets.get(group.upper(), set()) if group_kind[group] == 'tf' else set()
            known, gene_labels, gene_candidates = downstream_target_labels(genes, known, group)
            feature_labels = np.zeros(matrix.shape[1], bool)
            for fi, gene_set in feature_targets:
                feature_labels[fi] = bool(gene_set & known)
            gene_scores, membership = feature_gene_scores(np.abs(stats['effect']), catalog, genes)
            specificity = specificity_test(stats['responding'], feature_labels, eligible)
            records.append({'representation': 'model', 'layer': layer, 'group': group,
                            'kind': group_kind[group], 'n_cells': len(indices),
                            'n_control_cells': len(control_indices), 'n_responding': int(stats['responding'].sum()),
                            'n_eligible_known_target_genes': len(known & active_catalog_genes),
                            'self_target_excluded': True,
                            'perturbed_gene_excluded_from_gene_retrieval': True,
                            'specificity': specificity,
                            'feature_retrieval': retrieval_metrics(feature_labels[eligible], np.abs(stats['effect'][eligible]),
                                                                   args.bootstrap, args.seed + index),
                            'gene_retrieval': retrieval_metrics(gene_labels[gene_candidates], gene_scores[gene_candidates],
                                                               args.bootstrap, args.seed + index),
                            'gene_catalog_coverage': float((membership > 0).mean()),
                            'target_gene_catalog_coverage': float((membership[gene_labels] > 0).mean()) if gene_labels.any() else None})
            for key in values:
                values[key].append(stats[key])
            gene_score_rows.append(gene_scores)
            print(f'model L{layer} {group}: {stats["responding"].sum()} responses, specificity p={specificity["p"]:.3g}', flush=True)
        np.savez_compressed(out / f'feature_statistics_L{layer:02d}.npz', groups=np.asarray(conditions),
                            eligible=eligible, gene_scores=np.asarray(gene_score_rows),
                            **{key: np.asarray(value) for key, value in values.items()})

    # Correct each prespecified representation/layer family across all tested TFs.
    families = list(dict.fromkeys((r['representation'], r['layer']) for r in records))
    summaries = []
    for representation, layer in families:
        family = [r for r in records if (r['representation'], r['layer']) == (representation, layer)]
        factors = [r for r in family if r['kind'] == 'tf']
        for record, q in zip(factors, bh_adjust([r['specificity']['p'] for r in factors])):
            record['specificity']['q'] = float(q)
        summaries.append({'representation': representation, 'layer': layer, 'n_tfs': len(factors),
                          **summarize_specificity(factors, args.alpha),
                          'n_specific': sum(r['specificity'].get('q', 1) < args.alpha for r in factors),
                          'n_responding_groups': sum(r['n_responding'] > 0 for r in family),
                          'n_groups': len(family),
                          'gene_retrieval': summarize_retrieval(factors, 'gene_retrieval', args.bootstrap, args.seed),
                          'feature_retrieval': summarize_retrieval(factors, 'feature_retrieval', args.bootstrap, args.seed),
                          'pseudo_controls': [{'group': r['group'], 'n_responding': r['n_responding']}
                                              for r in family if r['kind'] == 'pseudo']})
    insilico_path = feature_dir / 'insilico.npz'
    if insilico_path.exists():
        insilico = np.load(insilico_path)
        meta = json.loads((feature_dir / 'insilico_meta.json').read_text())
        stats = insilico_statistics(insilico)
        np.savez_compressed(out / 'insilico_statistics.npz', **stats)
        retrieval = []
        for tf_index, tf in enumerate(meta['tfs']):
            _, labels, candidates = downstream_target_labels(meta['genes'], targets.get(tf.upper(), set()), tf)
            for layer_index, layer in enumerate(meta['layers']):
                eligible = stats['eligible'][tf_index, :, layer_index] & candidates
                retrieval.append({'tf': tf, 'layer': layer,
                                  **retrieval_metrics(labels[eligible], stats['mean_difference'][tf_index, eligible, layer_index],
                                                      args.bootstrap, args.seed + tf_index)})
        write_json(out / 'insilico_retrieval.json', {'meta': meta, 'retrieval': retrieval,
                   'interpretation': 'text-name deletion; nearby deletion control does not match subword token length'})
    adata.file.close()
    write_json(out / 'regulatory_statistics.json', {'summary': summaries, 'groups': records})
    write_json(out / 'manifest.json', {'arguments': vars(args), 'source': design.get('source'),
               'n_extracted_cells': len(rows), 'n_control_cells': len(control_indices), 'n_genes': len(genes),
               'gene_universe': 'all h5ad gene symbols except the perturbed gene; unavailable input/model scores zero',
               'feature_fisher_background': 'features nonzero in any extracted group (fixed across factors)',
               'response_bh_family': 'all genes/features per condition and representation/layer',
               'specificity_bh_family': 'all TFs per representation/layer',
               'effect_size': 'mean difference / sqrt((population variance P + population variance C)/2)',
               'bootstrap_interpretation': 'conditional retrieval-item uncertainty, not biological cell uncertainty',
               'auprc_definition': 'average precision (stepwise precision-recall integral)',
               'retrieval_lift_definition': 'ap_lift is an alias for prevalence_lift; lift_over_random_expected_ap uses the exact fixed-score tie-aware random-label AP expectation; both are descriptive ratios, not significance tests',
               'target_definition': 'TRRUST downstream targets with the perturbed factor itself excluded; coverage sensitivities count distinct eligible known target genes at thresholds 1, 3 and 5; all screened TFs remain in the specificity BH family',
               'source_sha256': {str(path): hashlib.sha256(Path(path).read_bytes()).hexdigest()
                                 for path in (args.design, args.trrust, feature_dir / 'rows.json', Path(__file__))},
               'seconds': round(time.time() - started, 1),
               'software': {package: version(package) for package in ('numpy', 'scipy', 'scikit-learn', 'anndata')}})


if __name__ == '__main__':
    main()
