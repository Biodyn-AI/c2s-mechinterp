import unittest
import json
from pathlib import Path
import tempfile
import numpy as np
from rev_residual_summary import layer_summary, rank_correlation, summarize_model


class ResidualSummaryTests(unittest.TestCase):
    def test_isotropic_covariance_has_full_width_pr(self):
        record = layer_summary(np.ones(4), np.array([1, 1, 1, 20]), np.zeros(4), [100, 3, 40, .5])
        self.assertEqual(record['centered_covariance_pr'], 4)
        self.assertEqual(record['centered_covariance_pr_fraction_width'], 1)
        self.assertEqual(record['top1_variance_share'], .25)
        self.assertEqual(record['top10_variance_share'], 1)
        self.assertEqual(record['rms_max_over_median'], 20)
        self.assertEqual(record['n_dimensions_rms_gt_10x_median'], 1)
        self.assertEqual(record['median_of_cell_median_abs_activation'], .5)
        self.assertEqual(record['max_abs_over_median_of_cell_medians'], 80)

    def test_rank_one_and_legacy_scalars(self):
        record = layer_summary([0, 0, 0, 9], np.ones(4), np.ones(4), [200, 3, 7])
        self.assertEqual(record['centered_covariance_pr'], 1)
        self.assertEqual(record['top1_variance_share'], 1)
        self.assertIsNone(record['median_of_cell_median_abs_activation'])

    def test_descriptive_rank_correlation_handles_constant_dead_fraction(self):
        self.assertAlmostEqual(rank_correlation([3, 2, 1], [1, 2, 3])['spearman_rho'], -1)
        self.assertIsNone(rank_correlation([1, 2, 3], [0, 0, 0])['spearman_rho'])
        self.assertNotIn('p', rank_correlation([1, 2, 3], [1, 2, 3]))

    def test_rejects_nonfinite_and_negative_covariance_values(self):
        for eigenvalues in ([1, np.inf], [1, -1]):
            with self.assertRaises(ValueError):
                layer_summary(eigenvalues, [1, 1], [0, 0], [10, 1, 1])

    def test_join_keeps_original_and_reevaluated_dead_fractions_distinct(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            np.savez(directory / 'statistics.npz', eig_L00=[1., 1.], rms_L00=[1., 1.],
                     mean_L00=[0., 0.], scalars_L00=[100., 1., 2.])
            (directory / 'train_L00.json').write_text(json.dumps({'n_features': 10, 'dead_features': 2,
                                                                 'variance_explained': .8}))
            (directory / '2b_eval_L00.json').write_text(json.dumps({'n_features': 10, 'dead_val_reextracted': 1,
                     'dead_frac_all_rows': 0., 'variance_explained_reextracted': .79}))
            model = summarize_model(directory / 'statistics.npz', directory, directory, expected_layers=[0])
            record = model['layers'][0]
            self.assertEqual(record['dead_fraction_heldout_original'], .2)
            self.assertEqual(record['dead_fraction_heldout_reextracted'], .1)
            self.assertEqual(record['dead_fraction_heldout'], .1)
            self.assertEqual(record['dead_fraction_all_rows'], 0)


if __name__ == '__main__':
    unittest.main()
