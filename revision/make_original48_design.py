"""Reconstruct the original Phase-8 cell selection, without loading the whole h5ad.

The selection follows code/phase8_perturbation_v2.py exactly: RandomState(42), control choice,
non-TF permutation, then knockdown-cell choices in descending-abundance TF order. Defaults encode
the recorded run flags: 48 TFs, 12 non-TFs, 400 controls and 25 cells per knockdown. Comparison with
an old result validates target order/counts, but cannot prove cell identity if old cell IDs were
not saved. Sentences use stable descending raw-count ranking and the same gene-name order.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy import sparse


def select_groups(gene, tfs, seed=42, n_control=400, n_tf=48, n_nontf=12,
                  cells_per_target=25, min_cells=15, control_label='non-targeting'):
    rng = np.random.RandomState(seed)
    control = np.flatnonzero(gene == control_label)
    if not len(control):
        raise ValueError(f'no controls labelled {control_label!r}')
    if len(control) > n_control:
        control = rng.choice(control, n_control, replace=False)
    unique, counts = np.unique(gene, return_counts=True)
    present = {g: int(c) for g, c in zip(unique, counts) if g != control_label and c >= min_cells}
    factors = sorted([g for g in present if g.upper() in tfs], key=lambda g: -present[g])[:n_tf]
    nonfactors = list(rng.permutation(sorted(g for g in present if g.upper() not in tfs)))[:n_nontf]
    groups = [{'group': 'control', 'kind': 'control', 'cells': control.tolist()}]
    for target in factors + nonfactors:
        indices = np.flatnonzero(gene == target)
        if len(indices) > cells_per_target:
            indices = rng.choice(indices, cells_per_target, replace=False)
        groups.append({'group': str(target), 'kind': 'tf' if target.upper() in tfs else 'nontf',
                       'cells': indices.tolist()})
    return groups


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--h5ad', required=True)
    parser.add_argument('--trrust', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--old-result', default='')
    parser.add_argument('--perturbation-col', default='gene')
    parser.add_argument('--control-label', default='non-targeting')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n-tf', type=int, default=48)
    parser.add_argument('--n-nontf', type=int, default=12)
    parser.add_argument('--n-control-cells', type=int, default=400)
    parser.add_argument('--cells-per-target', type=int, default=25)
    parser.add_argument('--min-cells', type=int, default=15)
    parser.add_argument('--max-genes', type=int, default=512)
    args = parser.parse_args()
    import anndata
    targets = {line.split('\t')[0].upper() for line in Path(args.trrust).read_text().splitlines() if line}
    adata = anndata.read_h5ad(args.h5ad, backed='r')
    gene = adata.obs[args.perturbation_col].astype(str).values
    groups = select_groups(gene, targets, args.seed, args.n_control_cells, args.n_tf, args.n_nontf,
                           args.cells_per_target, args.min_cells, args.control_label)
    old_agreement = None
    if args.old_result:
        old = json.loads(Path(args.old_result).read_text())
        old_targets = old['targets']
        chosen_targets = groups[1:]
        old_agreement = ([r['target'] for r in old_targets] == [g['group'] for g in chosen_targets] and
                         [r['n_cells'] for r in old_targets] == [len(g['cells']) for g in chosen_targets])
        if not old_agreement:
            raise ValueError('old result target order/cell counts differ from reconstructed design')
    names = np.asarray(adata.var_names).astype(str)
    for group in groups:
        indices = np.asarray(group['cells'])
        order = np.argsort(indices)
        matrix = adata.X[indices[order], :]
        inverse = np.argsort(order)
        matrix = matrix[inverse]
        sentences = []
        for row in range(len(indices)):
            values = matrix[row].toarray().ravel() if sparse.issparse(matrix) else np.asarray(matrix[row]).ravel()
            if not np.isfinite(values).all() or (values < 0).any():
                raise ValueError('raw-count row contains nonfinite or negative values')
            nonzero = np.flatnonzero(values > 0)
            ranked = nonzero[np.argsort(-values[nonzero], kind='stable')][:args.max_genes]
            sentences.append(names[ranked].tolist())
        group['sentences'] = sentences
    adata.file.close()
    output = {'source': Path(args.h5ad).name, 'seed': args.seed, 'max_genes': args.max_genes,
              'arguments': vars(args), 'groups': groups, 'old_result_target_count_agreement': old_agreement,
              'cell_identity_basis': 'deterministic original RNG recipe; original cell IDs were not recorded',
              'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output))
    print(f'{path}: {sum(g["kind"] == "tf" for g in groups)} TF groups, '
          f'{sum(len(g["cells"]) for g in groups)} cells', flush=True)


if __name__ == '__main__':
    main()
