"""Row identities and cache checks are tested without loading a model or using a GPU."""
import unittest

import numpy as np

from rev_validate_cell_states import compare_state, sample_rows


class CellStateValidationTests(unittest.TestCase):
    def test_sampled_state_rows_map_to_nonidentity_h5ad_ids(self):
        selected = sample_rows(np.array([19, 3, 11]), n_observations=20, n_cells=3, seed=0)
        self.assertEqual(selected, [(0, 19), (1, 3), (2, 11)])

    def test_sampling_is_reproducible_and_rejects_duplicate_or_out_of_range_ids(self):
        first = sample_rows(np.arange(30)[::-1], 30, 5, 4)
        self.assertEqual(first, sample_rows(np.arange(30)[::-1], 30, 5, 4))
        for ids in (np.array([1, 1]), np.array([1, 30])):
            with self.assertRaises(ValueError):
                sample_rows(ids, 30, 1, 0)

    def test_exact_vector_recovers_own_cached_row(self):
        cache = np.eye(3)
        record = compare_state(cache[1], cache, 1)
        self.assertAlmostEqual(record['cosine'], 1.)
        self.assertEqual(record['relative_l2'], 0.)
        self.assertEqual(record['max_abs_difference'], 0.)
        self.assertEqual(record['nearest_stored_row'], 1)
        self.assertTrue(record['passes_sample_criteria'])

    def test_wrong_row_is_detected_by_nearest_match_and_vector_error(self):
        cache = np.eye(3)
        record = compare_state(cache[2], cache, 1)
        self.assertEqual(record['nearest_stored_row'], 2)
        self.assertFalse(record['nearest_is_expected'])
        self.assertFalse(record['passes_sample_criteria'])

    def test_nonfinite_or_zero_fresh_state_is_rejected(self):
        for vector in (np.zeros(3), np.array([np.nan, 1., 2.])):
            with self.assertRaises(ValueError):
                compare_state(vector, np.eye(3), 0)


if __name__ == '__main__':
    unittest.main()
