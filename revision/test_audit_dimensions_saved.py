import unittest

import numpy as np

from audit_dimensions_saved import raw_metrics, select_cells, log_cp10k, grouped_gene_means


class RawDimensionAuditTests(unittest.TestCase):
    def test_self_is_excluded_but_duplicate_and_tied_neighbours_are_rejected(self):
        values = np.array([[0., 0.], [0., 0.], [1., 0.], [3., 0.], [7., 0.]])
        metrics = raw_metrics(values, block_size=2)
        self.assertAlmostEqual(metrics['participation_ratio'], 1.)
        self.assertEqual(metrics['twonn_valid_rows'], 2)
        self.assertEqual(metrics['twonn_excluded_rows'], 3)
        self.assertAlmostEqual(metrics['twonn_id'], 1 / np.log(1.5))

    def test_constant_and_isotropic_geometry(self):
        constant = raw_metrics(np.ones((4, 2)))
        self.assertEqual(constant['participation_ratio'], 0.)
        self.assertEqual(constant['twonn_excluded_rows'], 4)
        self.assertIsNone(constant['twonn_id'])
        self.assertAlmostEqual(raw_metrics(np.vstack([np.eye(3), -np.eye(3)]))['participation_ratio'], 3.)

    def test_explicit_cell_order_does_not_consume_sampling_rng(self):
        rng = np.random.default_rng(0)
        np.testing.assert_array_equal(select_cells(10, 4, rng, [8, 1, 5]), [8, 1, 5])
        self.assertEqual(rng.normal(), np.random.default_rng(0).normal())
        rng = np.random.default_rng(0)
        expected = np.random.default_rng(0)
        np.testing.assert_array_equal(select_cells(10, 4, rng), np.sort(expected.choice(10, 4, replace=False)))
        self.assertEqual(rng.normal(), expected.normal())
        with self.assertRaises(ValueError): select_cells(10, 4, rng, [1, 1])

    def test_expression_normalizes_complete_panel_before_gene_projection(self):
        counts = np.array([[1, 1, 0], [3, 0, 1]], np.float32)
        expected = np.log1p(np.array([[5000, 5000, 0], [7500, 0, 2500]], np.float32))
        np.testing.assert_allclose(log_cp10k(counts), expected)
        np.testing.assert_allclose(log_cp10k(counts * 5), expected)
        self.assertNotEqual(log_cp10k(counts)[:, 0][0], np.log1p(10000))

    def test_gene_means_preserve_magnitude_and_occurrence_threshold(self):
        names = np.array(['b', 'a', 'b', 'a', 'rare'])
        values = np.array([[10., 1.], [100., 4.], [20., 3.], [200., 8.], [1., 1.]])
        genes, counts, means = grouped_gene_means(names, values, minimum=2)
        np.testing.assert_array_equal(genes, ['a', 'b'])
        np.testing.assert_array_equal(counts, [2, 2])
        np.testing.assert_array_equal(means, [[150., 6.], [15., 2.]])


if __name__ == '__main__': unittest.main()
