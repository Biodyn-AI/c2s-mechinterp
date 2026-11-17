#!/usr/bin/env python3
"""Stage a small, auditable THP1 contingency cohort; never run model inference.

Pooled data default to one HTO partition (rep1-tx), keeping control and targeted
cells in the same recorded batch. Arrayed HTO values encode interventions, so
arrayed data remain a separate experiment with no inferred batch matching.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import resource
import sys

import numpy as np
from scipy import sparse

ZENODO = 'https://zenodo.org/records/13350497'
EXPECTED_MD5 = {
    'PapalexiSatija2021_eccite_RNA.h5ad': 'd9e8bfe20b0e7b0919be042e8fcf6d03',
    'PapalexiSatija2021_eccite_arrayed_RNA.h5ad': '843820d48b024348d6132cd53be0da91',
}


def hashes(path):
    algorithms = {name: hashlib.new(name) for name in ('sha256', 'md5')}
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            for digest in algorithms.values(): digest.update(block)
    return {name: digest.hexdigest() for name, digest in algorithms.items()}


def parse_guide_target(guide):
    """Strip only a terminal guide suffix; never guess aliases or multi-guide labels."""
    if re.fullmatch(r'NTg\d+', str(guide), flags=re.I): return 'control'
    match = re.fullmatch(r'([A-Za-z0-9][A-Za-z0-9_.-]*)g([1-9]\d*)', str(guide))
    if not match: raise ValueError(f'Unrecognised single-guide identifier: {guide!r}')
    return match.group(1).upper()


def checked_target(guide, existing_label):
    target = parse_guide_target(guide)
    expected = str(existing_label)
    if target == 'control':
        if expected.lower() != 'control': raise ValueError(f'Control/guide disagreement: {guide}, {expected}')
    elif expected.upper() not in (str(guide).upper(), target):
        raise ValueError(f'Target/guide disagreement: {guide}, {expected}')
    return target


def gene_collapse_plan(symbols, ensembl_ids):
    """Merge case-colliding columns only with one identical, nonempty Ensembl ID.

    Distinct or missing IDs are explicit exclusions of every conflicting column.
    Unique symbols keep their original spelling. A merged symbol is uppercased,
    making the case policy explicit without claiming a current HGNC canonical name.
    """
    if len(symbols) != len(ensembl_ids): raise ValueError('Gene symbols and identifiers differ in length')
    groups = defaultdict(list)
    for index, symbol in enumerate(symbols): groups[str(symbol).upper()].append(index)
    mapping = np.full(len(symbols), -1, dtype=np.int64)
    output_symbols, output_ids, original_symbols, collisions = [], [], [], []
    for upper, indices in groups.items():
        ids = [str(ensembl_ids[index]).strip() for index in indices]
        valid_ids = set(ids)
        verified = len(valid_ids) == 1 and not valid_ids.intersection({'', 'nan', 'None', '<NA>'})
        if len(indices) > 1:
            action = 'sum_same_ensembl' if verified else 'drop_ambiguous_identity'
            collisions.append({'uppercase_symbol': upper, 'source_indices': indices,
                               'source_symbols': [str(symbols[index]) for index in indices],
                               'ensembl_ids': ids, 'action': action})
            if not verified: continue
        canonical = upper if len(indices) > 1 else str(symbols[indices[0]])
        mapping[indices] = len(output_symbols)
        output_symbols.append(canonical)
        output_ids.append(ids[0])
        original_symbols.append('|'.join(str(symbols[index]) for index in indices))
    return {'symbols': output_symbols, 'ensembl_ids': output_ids,
            'source_symbols': original_symbols, 'column_map': mapping, 'collisions': collisions}


def collapse_counts(counts, plan):
    """Sparse column summation preserves raw counts and avoids a dense cell-by-gene copy."""
    mapping = np.asarray(plan['column_map'])
    retained = np.flatnonzero(mapping >= 0)
    projection = sparse.csr_matrix((np.ones(len(retained), dtype=np.float32),
                                     (retained, mapping[retained])),
                                    shape=(len(mapping), len(plan['symbols'])))
    return (sparse.csr_matrix(counts) @ projection).tocsr()


def sample_across_guides(indices, guides, count, rng):
    """Sample in shuffled round-robin guide order so rare guides remain represented."""
    indices = np.asarray(indices, int)
    pools = [list(rng.permutation(indices[guides[indices] == guide]))
             for guide in sorted(set(guides[indices]))]
    chosen = []
    while len(chosen) < min(count, len(indices)):
        for pool in pools:
            if pool and len(chosen) < min(count, len(indices)): chosen.append(int(pool.pop()))
    return np.asarray(sorted(chosen), dtype=int)


def select_groups(targets, guides, eligible, factors, seed=20261117, n_control=400,
                  cells_per_target=50, min_cells=30, n_nontf=4, n_pseudo=4, pseudo_cells=50):
    rng = np.random.default_rng(seed)
    targets, guides, eligible = np.asarray(targets), np.asarray(guides), np.asarray(eligible, bool)
    if not (targets.shape == guides.shape == eligible.shape): raise ValueError('Cell metadata are not aligned')
    controls = np.flatnonzero(eligible & (targets == 'control'))
    requested_controls = n_control + n_pseudo * pseudo_cells
    if len(controls) < requested_controls:
        raise ValueError(f'Need {requested_controls} disjoint controls; only {len(controls)} available')
    sampled_controls = sample_across_guides(controls, guides, requested_controls, rng)
    sampled_controls = rng.permutation(sampled_controls)
    groups = [{'group': 'control', 'kind': 'control', 'source_cells': sorted(sampled_controls[:n_control].tolist())}]
    target_counts = Counter(targets[eligible])
    available = sorted(target for target, count in target_counts.items() if target != 'control' and count >= min_cells)
    tf_targets = [target for target in available if target in factors]
    others = [target for target in available if target not in factors]
    nontf_targets = sorted(rng.permutation(others)[:n_nontf].tolist())
    if not tf_targets: raise ValueError('No TRRUST factors meet the prespecified cell minimum')
    for target in tf_targets + nontf_targets:
        indices = np.flatnonzero(eligible & (targets == target))
        selected = sample_across_guides(indices, guides, cells_per_target, rng)
        groups.append({'group': target, 'kind': 'tf' if target in factors else 'nontf',
                       'source_cells': selected.tolist()})
    for index in range(n_pseudo):
        start = n_control + index * pseudo_cells
        groups.append({'group': f'pseudo_{index:02d}', 'kind': 'pseudo',
                       'source_cells': sorted(sampled_controls[start:start+pseudo_cells].tolist())})
    for group in groups:
        selected_guides = Counter(guides[group['source_cells']].tolist())
        group['guide_counts_selected'] = dict(selected_guides)
        group['n_guides_selected'] = len(selected_guides)
    return groups


def raw_summary(counts):
    values = counts.data if sparse.issparse(counts) else np.asarray(counts).ravel()
    if not len(values): return {'n_nonzero': 0, 'min': 0., 'max': 0., 'max_integer_residual': 0.}
    finite = np.isfinite(values)
    residual = float(np.max(np.abs(values - np.rint(values)))) if finite.all() else None
    summary = {'n_nonzero': len(values), 'min': float(values.min()), 'max': float(values.max()),
               'n_nonfinite': int((~finite).sum()), 'max_integer_residual': residual}
    if not finite.all() or np.min(values) < 0 or residual > 1e-6:
        raise ValueError(f'Raw-count validation failed: {summary}')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--h5ad', type=Path, required=True)
    parser.add_argument('--trrust', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--batch', default=None, help='Pooled default rep1-tx; arrayed default all; pooled all is refused')
    parser.add_argument('--seed', type=int, default=20261117)
    parser.add_argument('--n-control', type=int, default=400)
    parser.add_argument('--cells-per-target', type=int, default=50)
    parser.add_argument('--min-cells', type=int, default=30)
    parser.add_argument('--n-nontf', type=int, default=4)
    parser.add_argument('--n-pseudo', type=int, default=4)
    parser.add_argument('--pseudo-cells', type=int, default=50)
    args = parser.parse_args()
    import anndata
    import pandas as pd
    source_hashes = hashes(args.h5ad)
    expected_md5 = EXPECTED_MD5.get(args.h5ad.name)
    if expected_md5 is None or source_hashes['md5'] != expected_md5:
        raise ValueError('Source does not match the verified scPerturb deposition filename/checksum')
    adata = anndata.read_h5ad(args.h5ad, backed='r')
    obs = adata.obs
    guides = obs['guide_id'].astype(str).to_numpy()
    existing = obs['perturbation'].astype(str).to_numpy()
    targets = np.array([checked_target(guide, label) for guide, label in zip(guides, existing)])
    nperts = np.asarray(obs['nperts'], int)
    if np.any(nperts != np.where(targets == 'control', 0, 1)):
        raise ValueError('nperts disagrees with the single-guide/control assignment')
    pooled = '_arrayed_' not in args.h5ad.name
    batch = args.batch or ('rep1-tx' if pooled else 'all')
    if pooled and batch == 'all': raise ValueError('Pooled experiment requires one explicit HTO partition per design')
    batches = obs['hto'].astype(str).to_numpy()
    eligible = batches == batch if batch != 'all' else np.ones(adata.n_obs, bool)
    if not eligible.any(): raise ValueError(f'No rows in HTO partition {batch!r}')
    factors = {line.split('\t')[0].upper() for line in args.trrust.read_text().splitlines() if line}
    groups = select_groups(targets, guides, eligible, factors, args.seed, args.n_control,
                           args.cells_per_target, args.min_cells, args.n_nontf, args.n_pseudo, args.pseudo_cells)
    source_rows = np.asarray(sorted(cell for group in groups for cell in group['source_cells']))
    if len(set(source_rows)) != len(source_rows): raise ValueError('Selected groups overlap')
    if len(source_rows) > 2000: raise ValueError('Local staging is capped at 2000 selected cells; prepare a smaller design')
    if 'ensembl_id' not in adata.var: raise ValueError('Ensembl identifiers are required to assess symbol collisions')
    plan = gene_collapse_plan(adata.var_names.tolist(), adata.var['ensembl_id'].tolist())
    sample_rows = np.linspace(0, adata.n_obs-1, min(128, adata.n_obs), dtype=int)
    sample = raw_summary(adata.X[sample_rows, :])
    chunks, selected_nnz = [], 0
    for start in range(0, len(source_rows), 256):
        counts = adata.X[source_rows[start:start+256], :]
        raw_summary(counts)
        selected_nnz += counts.nnz if sparse.issparse(counts) else np.count_nonzero(counts)
        chunks.append(collapse_counts(counts, plan))
    cleaned_counts = sparse.vstack(chunks, format='csr')
    cleaned_obs = obs.iloc[source_rows].copy()
    cleaned_obs['c6_source_cell_index'] = source_rows
    cleaned_obs['c6_source_obs_name'] = adata.obs_names[source_rows].astype(str)
    cleaned_obs['c6_gene'] = targets[source_rows]
    cleaned_obs['c6_recorded_batch'] = batches[source_rows]
    clean_var = pd.DataFrame({'ensembl_id': plan['ensembl_ids'], 'source_symbols': plan['source_symbols']},
                             index=plan['symbols'])
    cleaned = anndata.AnnData(X=cleaned_counts, obs=cleaned_obs, var=clean_var)
    args.out.mkdir(parents=True, exist_ok=True)
    cleaned_path = args.out / 'counts_cleaned.h5ad'
    cleaned.write_h5ad(cleaned_path, compression='gzip')
    source_to_clean = {int(source): index for index, source in enumerate(source_rows)}
    for group in groups:
        group['cells'] = [source_to_clean[source] for source in group['source_cells']]
        group['sentences'] = []
        cap = 520 if group['kind'] == 'control' else 512
        for cell in group['cells']:
            values = cleaned_counts[cell].toarray().ravel()
            # Stable ties use the cleaned source gene order; no normalized counts enter ranking.
            nonzero = np.flatnonzero(values > 0)
            ranked = nonzero[np.argsort(-values[nonzero], kind='stable')][:cap]
            group['sentences'].append([plan['symbols'][index] for index in ranked])
    source_shape = [adata.n_obs, adata.n_vars]
    adata.file.close()
    provenance = {'status': 'staged contingency only; no model inference or outcome selection',
        'source_file': args.h5ad.name, 'source_local_path': str(args.h5ad.resolve()),
        'source_shape': source_shape, 'source_hashes': source_hashes,
        'source_deposition': ZENODO,
        'download_url': f'https://zenodo.org/records/13350497/files/{args.h5ad.name}?download=1',
        'paper_doi': '10.1038/s41588-021-00778-2', 'source_geo': 'GSE153056',
        'cleaned_file': cleaned_path.name, 'cleaned_shape': list(cleaned.shape),
        'cleaned_hashes': hashes(cleaned_path), 'sampled_raw_counts': sample,
        'selected_raw_counts_validated': True, 'selected_source_nnz': int(selected_nnz),
        'collision_policy': 'sum exact identical nonempty Ensembl IDs; otherwise drop all conflicting columns; unique symbol spelling retained',
        'collisions': plan['collisions'], 'batch': batch,
        'batch_policy': 'one recorded HTO partition for pooled data; arrayed HTO encodes interventions and is not treated as a shared batch',
        'guide_policy': 'validated terminal g-number parsing; NTg-number is control; guide-balanced cell sampling; gene-level grouping retains guide identities',
        'control_and_target_labels': dict(Counter(targets[eligible])),
        'all_hto_counts': dict(Counter(batches)), 'all_guide_to_target': dict(zip(guides, targets)),
        'trrust_file': str(args.trrust.resolve()), 'trrust_hashes': hashes(args.trrust),
        'arguments': {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        'source_cell_indices': source_rows.tolist(),
        'script_sha256': hashes(Path(__file__))['sha256'],
        'peak_rss_bytes': int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024))}
    design = {'source': cleaned_path.name, 'seed': args.seed, 'max_genes': 512,
              'source_provenance': 'provenance.json', 'cohort': 'Papalexi THP1 contingency', 'groups': groups}
    (args.out / 'design.json').write_text(json.dumps(design, indent=2))
    (args.out / 'provenance.json').write_text(json.dumps(provenance, indent=2))
    print(json.dumps({'out': str(args.out), 'source': args.h5ad.name, 'batch': batch,
                      'shape': list(cleaned.shape), 'groups': dict(Counter(g['kind'] for g in groups)),
                      'collisions': plan['collisions'], 'peak_rss_bytes': provenance['peak_rss_bytes']}, indent=2))

if __name__ == '__main__': main()
