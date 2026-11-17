import unittest

from audit_e4_saved import count_contrasts


class E4SavedAuditTests(unittest.TestCase):
    def test_count_contrast_separates_heldout_from_all_rows_and_rejects_capacity_change(self):
        original = {'n_features': 4, 'dead_features': 3, 'dead_all_rows': 1}
        projected = {'n_features': 4, 'dead_features': 2, 'dead_all_rows': 2}
        result = count_contrasts(original, projected)
        self.assertEqual(result['dead_fraction_heldout_delta'], -.25)
        self.assertEqual(result['dead_fraction_all_rows_delta'], .25)
        self.assertNotIn('variance_explained_delta', result)
        with self.assertRaises(ValueError): count_contrasts(original, {**projected, 'n_features': 8})


if __name__ == '__main__': unittest.main()
