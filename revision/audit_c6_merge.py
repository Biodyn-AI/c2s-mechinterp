#!/usr/bin/env python3
"""Read only two original sparse columns and verify their prepared-cohort sum."""
import argparse
import json
from pathlib import Path
import resource
import sys

import h5py
import numpy as np

from audit_experimental_saved import Audit


def h5_strings(node):
    if isinstance(node, h5py.Group):
        categories, codes = node['categories'].asstr()[:], node['codes'][:]
        return np.asarray([categories[code] if code >= 0 else '' for code in codes])
    return node.asstr()[:]


def sparse_columns(group, columns, chunk_rows=128):
    """Bounded CSR/CSC reads; repeated stored coordinates retain sparse sum semantics."""
    n_rows, n_columns = map(int, group.attrs['shape'])
    if any(type(column) is not int or not 0 <= column < n_columns for column in columns):
        raise ValueError('Invalid requested column')
    output = np.zeros((n_rows, len(columns)), float)
    encoding = group.attrs['encoding-type']
    if encoding == 'csc_matrix':
        for position, column in enumerate(columns):
            start, stop = map(int, group['indptr'][column:column+2])
            np.add.at(output[:, position], group['indices'][start:stop], group['data'][start:stop])
    elif encoding == 'csr_matrix':
        for first in range(0, n_rows, chunk_rows):
            last = min(n_rows, first+chunk_rows)
            pointers = group['indptr'][first:last+1]
            start, stop = int(pointers[0]), int(pointers[-1])
            column_ids, values = group['indices'][start:stop], group['data'][start:stop]
            row_ids = np.repeat(np.arange(first, last), np.diff(pointers))
            for position, column in enumerate(columns):
                keep = column_ids == column
                np.add.at(output[:, position], row_ids[keep], values[keep])
    else:
        raise ValueError('Unsupported sparse encoding: '+str(encoding))
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    audit = Audit()
    provenance = audit.read(args.prepared/'provenance.json')
    prepared_path = args.prepared/'counts_cleaned.h5ad'
    for path, expected in [(args.source, provenance['source_hashes']['sha256']),
                           (prepared_path, provenance['cleaned_hashes']['sha256'])]:
        audit.hash(path)
        audit.equal(path.name+' staged SHA256', audit.sources[str(path)], expected)
    with h5py.File(args.source, 'r') as source, h5py.File(prepared_path, 'r') as prepared:
        source_names = source['var'][source['var'].attrs['_index']].asstr()[:].tolist()
        prepared_names = prepared['var'][prepared['var'].attrs['_index']].asstr()[:].tolist()
        columns = [source_names.index(name) for name in ['C2orf15', 'C2ORF15']]
        merged_column = prepared_names.index('C2ORF15')
        original = sparse_columns(source['X'], columns)
        merged = sparse_columns(prepared['X'], [merged_column])[:, 0]
        source_rows = prepared['obs/c6_source_cell_index'][:]
        audit.equal('prepared source-row indices match recorded staging', source_rows, provenance['source_cell_indices'])
        ensembl = h5_strings(source['var/ensembl_id'])
        audit.check('original common Ensembl identity is nonempty', ensembl[columns[0]] not in ['', 'nan', 'None', '<NA>'])
        audit.equal('original two columns same Ensembl ID', ensembl[columns[0]], ensembl[columns[1]])
        audit.equal('prepared Ensembl ID', h5_strings(prepared['var/ensembl_id'])[merged_column], ensembl[columns[0]])
        expected = original[source_rows].sum(axis=1)
        audit.check('prepared merged column exactly equals two source columns at recorded rows', np.array_equal(merged, expected))
        audit.check('all extracted column values finite nonnegative integers', np.isfinite(original).all() and np.isfinite(merged).all() and
                    np.all(original >= 0) and np.all(merged >= 0) and np.array_equal(original, np.rint(original)) and np.array_equal(merged, np.rint(merged)))
    def describe(values):
        return {'n_cells': len(values), 'identical_vectors': bool(np.array_equal(values[:, 0], values[:, 1])),
                'n_cells_different': int(np.sum(values[:, 0] != values[:, 1])),
                'n_cells_both_nonzero': int(np.sum((values[:, 0] != 0) & (values[:, 1] != 0))),
                'n_nonzero_by_column': np.count_nonzero(values, axis=0).tolist(),
                'total_counts_by_column': values.sum(axis=0).tolist()}
    changed = np.flatnonzero(expected != 0)
    audit.hash(Path(__file__))
    report = {'audit': 'Independent C6 C2ORF15 raw-column identity and merge check',
              'passed': audit.passed, 'n_checks': audit.n_checks, 'findings': audit.failed,
              'source_symbols': ['C2orf15', 'C2ORF15'], 'source_columns': columns,
              'ensembl_id': str(ensembl[columns[0]]), 'prepared_column': merged_column,
              'all_original_cells': describe(original), 'selected_prepared_cohort': describe(original[source_rows]),
              'merged_total_count': float(merged.sum()), 'maximum_absolute_merge_error': float(np.max(np.abs(merged-expected))),
              'nonzero_prepared_cells': [{'prepared_cell': int(index), 'source_cell': int(source_rows[index]),
                                          'source_counts': original[source_rows[index]].tolist(),
                                          'prepared_count': float(merged[index])} for index in changed],
              'source_sha256': audit.sources,
              'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform == 'darwin' else 1024),
              'limitations': ['Equality or inequality of observed count vectors does not establish molecular independence.',
                              'The explicit summation policy is checked; no model fits or experimental outputs were changed.']}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({key: report[key] for key in ['passed', 'n_checks', 'findings', 'all_original_cells',
                                                  'selected_prepared_cohort', 'maximum_absolute_merge_error', 'peak_rss_bytes']}))
    raise SystemExit(0 if audit.passed else 1)


if __name__ == '__main__': main()
