from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np
from scipy import sparse

from audit_biology_lines import feature_categories, persistence_summary, selected_sparse_rows


class BiologyLinesAuditTests(unittest.TestCase):
    def test_ordered_term_categories_then_feature_overlap_excludes_network_labels(self):
        rules = [('translation', ['translation']), ('RNA', ['rna'])]
        ids = ['GO_BP::RNA translation', 'Reactome::RNA processing', 'STRING::translation']
        self.assertEqual(feature_categories(ids, rules), {'translation', 'RNA'})
        self.assertEqual(feature_categories(['STRING::translation'], rules), set())

    def test_persistence_counts_source_pairs_when_targets_are_reused(self):
        result = persistence_summary({0: {'A'}, 1: {'A'}, 2: {'B'}}, {0: {'A'}, 1: {'B'}},
                                     [0, 1, 2], [0, 0, 1], [.8, .8, .6], .7, 100, 0)
        self.assertEqual(result['n_matched_annotated_pairs'], 2)
        self.assertEqual(result['observed_same_category'], 1.)
        self.assertEqual(result['random_target_baseline'], .5)
        self.assertEqual(result['excess_ci95'], [.5, .5])

    def test_selected_raw_rows_preserve_requested_order_for_csr_and_csc(self):
        expected = np.array([[1, 0, 2], [0, 3, 0], [4, 0, 5], [0, 6, 0]], np.float32)
        for constructor, encoding in [(sparse.csr_matrix, 'csr_matrix'), (sparse.csc_matrix, 'csc_matrix')]:
            matrix = constructor(expected)
            with tempfile.TemporaryDirectory() as directory:
                with h5py.File(Path(directory)/'tiny.h5', 'w') as handle:
                    group = handle.create_group('X'); group.attrs.update({'shape': matrix.shape, 'encoding-type': encoding})
                    for key in ['data', 'indices', 'indptr']: group.create_dataset(key, data=getattr(matrix, key))
                    np.testing.assert_array_equal(selected_sparse_rows(group, [3, 0]), expected[[3, 0]])


if __name__ == '__main__': unittest.main()
