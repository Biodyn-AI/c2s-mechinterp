#!/usr/bin/env python3
"""Independent C6 preparation checks: identities, batch, guides and source-row joins."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re

import numpy as np
from scipy import sparse

from audit_experimental_saved import Audit


def guide_target(guide):
    if re.fullmatch(r'NTg[1-9][0-9]*', guide, flags=re.IGNORECASE):
        return 'control'
    match = re.fullmatch(r'([A-Za-z0-9][A-Za-z0-9_.-]*)g[1-9][0-9]*', guide)
    if not match:
        raise ValueError('Ambiguous guide identity: '+guide)
    return match.group(1).upper()


def collision_is_identity_merge(collision):
    symbols, identifiers = collision['source_symbols'], collision['ensembl_ids']
    return (len(symbols) == len(identifiers) >= 2 and
            {str(symbol).upper() for symbol in symbols} == {collision['uppercase_symbol']} and
            len(set(identifiers)) == 1 and identifiers[0] not in ['', 'nan', 'None', '<NA>'] and
            collision['action'] == 'sum_same_ensembl')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    import anndata
    audit = Audit()
    design, provenance = audit.read(args.prepared/'design.json'), audit.read(args.prepared/'provenance.json')
    counts_path = args.prepared/'counts_cleaned.h5ad'
    audit.hash(counts_path)
    audit.equal('cleaned count file hash equals recorded staging', audit.sources[str(counts_path)], provenance['cleaned_hashes']['sha256'])
    cleaned = anndata.read_h5ad(counts_path, backed='r')
    obs, var = cleaned.obs, cleaned.var
    audit.equal('cleaned shape', list(cleaned.shape), provenance['cleaned_shape'])
    audit.equal('design/preparation seed', design['seed'], provenance['arguments']['seed'])
    audit.equal('prespecified preparation seed', design['seed'], 20261117)
    source_indices = obs['c6_source_cell_index'].to_numpy().astype(int)
    audit.equal('source cell indices match preparation', source_indices, provenance['source_cell_indices'])
    audit.equal('source cell indices unique', len(set(source_indices)), len(source_indices))
    audit.check('source indices within original row bounds', np.all((source_indices >= 0) & (source_indices < provenance['source_shape'][0])))
    audit.equal('source obs names retained exactly', obs['c6_source_obs_name'].astype(str).tolist(), cleaned.obs_names.astype(str).tolist())
    audit.check('cleaned symbols unique after uppercasing', len({str(gene).upper() for gene in cleaned.var_names}) == cleaned.n_vars)
    audit.check('one recorded pooled batch', provenance['batch'] != 'all')
    audit.check('selected cells all in prespecified HTO batch', np.all(obs['hto'].astype(str).to_numpy() == provenance['batch']))
    audit.equal('saved recorded batches equal HTO', obs['c6_recorded_batch'].astype(str).tolist(), obs['hto'].astype(str).tolist())
    for collision in provenance['collisions']:
        audit.check('case collision uses identical nonempty Ensembl identity', collision_is_identity_merge(collision))
        if collision_is_identity_merge(collision):
            row = var.loc[collision['uppercase_symbol']]
            audit.equal('merged Ensembl identity', row['ensembl_id'], collision['ensembl_ids'][0])
            audit.equal('merged original symbol names', row['source_symbols'], '|'.join(collision['source_symbols']))
    expected_width = provenance['source_shape'][1]-sum(len(row['source_symbols'])-1 for row in provenance['collisions'])
    audit.equal('column width after identity-preserving merge', cleaned.n_vars, expected_width)
    values = cleaned.X[:]
    observed_values = values.data if sparse.issparse(values) else np.asarray(values).ravel()
    audit.check('all prepared values are finite nonnegative integer counts',
                np.isfinite(observed_values).all() and np.all(observed_values >= 0) and
                np.all(np.abs(observed_values-np.rint(observed_values)) < 1e-7))
    guides = obs['guide_id'].astype(str).to_numpy()
    inferred_targets = np.asarray([guide_target(guide) for guide in guides])
    audit.equal('gene labels match independent terminal-guide parsing', obs['c6_gene'].astype(str).tolist(), inferred_targets.tolist())
    audit.equal('single perturbation count matches control/target guide assignment', obs['nperts'].to_numpy().astype(int), (inferred_targets != 'control').astype(int))
    used, groups = [], []
    for group in design['groups']:
        cells = np.asarray(group['cells'], int)
        used.extend(cells.tolist())
        wanted = 'control' if group['kind'] in ['control', 'pseudo'] else group['group'].upper()
        audit.check(group['group']+' selected target assignment', np.all(inferred_targets[cells] == wanted))
        audit.equal(group['group']+' source row mapping', source_indices[cells], group['source_cells'])
        counts = dict(Counter(guides[cells]))
        audit.equal(group['group']+' recorded selected-guide counts', group['guide_counts_selected'], counts)
        audit.equal(group['group']+' selected guide denominator', group['n_guides_selected'], len(counts))
        groups.append({'group': group['group'], 'kind': group['kind'], 'n_cells': len(cells), 'n_guides': len(counts)})
    audit.equal('groups disjoint', len(set(used)), len(used))
    audit.check('groups cover exactly prepared rows', set(used) == set(range(cleaned.n_obs)))
    cleaned.file.close()
    audit.hash(Path(__file__))
    report = {'audit': 'C6 pooled preparation identity/batch/guide audit', 'passed': audit.passed,
              'n_checks': audit.n_checks, 'findings': audit.failed, 'groups': groups,
              'batch': provenance['batch'], 'shape': provenance['cleaned_shape'], 'collisions': provenance['collisions'],
              'source_sha256': audit.sources,
              'limitations': ['Prepared count hash, full count validity and selected-cell metadata are checked.',
                              'Original deposition count-column summation is not rerun; source collision identities/provenance are checked.',
                              'Several guides per factor are represented; this is not an independent biological replicate or donor design.',
                              'The recorded seed and design/source joins are verified; preparation chronology is not inferred from outcome statistics.']}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({key: report[key] for key in ['passed', 'n_checks', 'findings', 'batch', 'shape']}))
    raise SystemExit(0 if audit.passed else 1)


if __name__ == '__main__': main()
