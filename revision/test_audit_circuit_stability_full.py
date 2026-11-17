import unittest

import numpy as np

from audit_circuit_stability_full import aligned_vector, source_summary


class FullCircuitStabilityTests(unittest.TestCase):
    def test_nonpassing_effects_change_full_correlation_while_shared_only_is_perfect(self):
        a = np.array([10., 20., 0., 0., 0., 0.])
        b = np.array([10., 20., -10., 10., -10., 10.])
        mask = np.array([True, True, False, False, False, False])
        result = source_summary(a, b, mask, mask, 10, 20)
        self.assertEqual(result['all_tested']['n_effect_pairs'], 6)
        self.assertEqual(result['shared_passing_only']['n_effect_pairs'], 2)
        self.assertAlmostEqual(result['shared_passing_only']['pearson'], 1.)
        self.assertLess(result['all_tested']['pearson'], .9)

    def test_source_needs_two_active_cells_in_each_run(self):
        result = source_summary([1., 2.], [1., 2.], [True, True], [True, True], 2, 1)
        self.assertFalse(result['common_eligible'])
        self.assertIsNone(result['all_tested'])

    def test_alignment_uses_layer_identity_not_array_position(self):
        vector = aligned_vector(np.array([[10, 20], [30, 40]]), [9, 8], [8, 9])
        np.testing.assert_array_equal(vector, [30, 40, 10, 20])


if __name__ == '__main__': unittest.main()
