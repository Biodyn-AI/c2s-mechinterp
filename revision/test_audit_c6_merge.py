"""Sparse access checks use both storage layouts and duplicate stored coordinates."""
from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np
from scipy import sparse

from audit_c6_merge import sparse_columns, h5_strings


class SparseMergeAuditTests(unittest.TestCase):
    def test_categorical_strings_and_missing_codes_are_read_without_guessing(self):
        with tempfile.TemporaryDirectory() as temporary:
            with h5py.File(Path(temporary)/'fixture.h5', 'w') as handle:
                group = handle.create_group('labels')
                group.create_dataset('categories', data=np.array(['ENSG1'], dtype=h5py.string_dtype()))
                group.create_dataset('codes', data=[0, -1])
                self.assertEqual(h5_strings(group).tolist(), ['ENSG1', ''])

    def test_csr_csc_column_reads_match_dense_values_with_small_chunks(self):
        expected = np.array([[1, 0, 2, 0], [0, 3, 0, 1], [4, 0, 0, 0], [0, 5, 6, 0]])
        for constructor, encoding in [(sparse.csr_matrix, 'csr_matrix'), (sparse.csc_matrix, 'csc_matrix')]:
            matrix = constructor(expected)
            with tempfile.TemporaryDirectory() as temporary:
                with h5py.File(Path(temporary)/'fixture.h5', 'w') as handle:
                    group = handle.create_group('X'); group.attrs['shape'] = matrix.shape
                    group.attrs['encoding-type'] = encoding
                    for key in ['data', 'indices', 'indptr']: group.create_dataset(key, data=getattr(matrix, key))
                    np.testing.assert_array_equal(sparse_columns(group, [3, 0], chunk_rows=2), expected[:, [3, 0]])

    def test_duplicate_stored_entries_are_summed_and_invalid_columns_fail(self):
        with tempfile.TemporaryDirectory() as temporary:
            with h5py.File(Path(temporary)/'fixture.h5', 'w') as handle:
                group = handle.create_group('X'); group.attrs['shape'] = (3, 1)
                group.attrs['encoding-type'] = 'csc_matrix'
                group.create_dataset('indptr', data=[0, 2]); group.create_dataset('indices', data=[1, 1])
                group.create_dataset('data', data=[2., 3.])
                np.testing.assert_array_equal(sparse_columns(group, [0])[:, 0], [0., 5., 0.])
                with self.assertRaises(ValueError): sparse_columns(group, [1])


if __name__ == '__main__': unittest.main()
