"""Lightweight guard and sampling checks, without a GPU or model download."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from rev_extract import load_sentences, storage_array
from rev_sae_tools import _load_acts
from rev_phase8_extract import reconstruction_summary
from make_original48_design import select_groups


class GPUUtilityTests(unittest.TestCase):
    def test_storage_rejects_nonfinite_and_overflow_without_clipping(self):
        for values in ([0, np.inf], [0, np.nan], [0, 70000]):
            with self.assertRaises(ValueError):
                storage_array(np.array(values, np.float32), np.float16)
        np.testing.assert_array_equal(storage_array(np.array([61000], np.float32), np.float32), [61000])

    def test_loader_refuses_nonfinite_arrays(self):
        with tempfile.TemporaryDirectory() as temporary:
            np.save(Path(temporary) / 'layer_00_activations.npy', [[0, np.inf]])
            with self.assertRaises(ValueError):
                _load_acts(temporary, 0)

    def test_sentence_rows_must_be_nonempty_contiguous_cells(self):
        with tempfile.TemporaryDirectory() as temporary:
            (Path(temporary) / 'row_gene_names.json').write_text(json.dumps(['A', 'B', 'C']))
            np.save(Path(temporary) / 'row_cell_ids.npy', [0, 0, 1])
            self.assertEqual(load_sentences(temporary), ([['A', 'B'], ['C']], [0, 1]))
            np.save(Path(temporary) / 'row_cell_ids.npy', [0, 1, 0])
            with self.assertRaises(ValueError):
                load_sentences(temporary)

    def test_reconstruction_uses_centered_variance_and_real_firings(self):
        x = np.array([[1., 2.], [3., 6.], [5., 10.]])
        error = np.ones_like(x) * 10  # constant reconstruction offset has zero residual variance
        record = reconstruction_summary(len(x), x.sum(0), (x*x).sum(0), error.sum(0),
                                        (error*error).sum(0), np.array([3, 0, 1]))
        self.assertAlmostEqual(record['variance_explained'], 1.)
        self.assertEqual(record['dead_features'], 1)
        self.assertAlmostEqual(record['mse'], 100.)

    def test_original_selection_matches_original_rng_recipe(self):
        gene = np.array(['non-targeting'] * 20 + ['TF1'] * 8 + ['TF2'] * 9 + ['X'] * 10 + ['Y'] * 10)
        groups = select_groups(gene, {'TF1', 'TF2'}, seed=42, n_control=10, n_tf=2,
                               n_nontf=1, cells_per_target=5, min_cells=3)
        rng = np.random.RandomState(42)
        control = rng.choice(np.arange(20), 10, replace=False)
        non = list(rng.permutation(['X', 'Y']))[:1]
        expected = [('control', control)]
        for target in ['TF2', 'TF1'] + non:
            expected.append((target, rng.choice(np.flatnonzero(gene == target), 5, replace=False)))
        self.assertEqual([g['group'] for g in groups], [g for g, _ in expected])
        for group, (_, indices) in zip(groups, expected):
            np.testing.assert_array_equal(group['cells'], indices)


if __name__ == '__main__':
    unittest.main()
