"""Small selection and paired-bootstrap checks; no large expression matrices or models."""
import unittest
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

from rev_selection_matched_transfer import encode_selected_shared, encode_shared_first, paired_contrasts, ranked_positive_indices, resultant_agreement


class SelectionMatchedTransferTests(unittest.TestCase):
    def test_full_panel_selection_precedes_shared_gene_projection(self):
        # Gene0 exists only in one panel and consumes its first sentence position.
        expression = np.array([[10., 8., 7.]])
        shared = np.array([1, 2])
        full_rank, full_magnitude, diagnostics = encode_selected_shared(expression, shared, 2)
        shared_rank, _, _ = encode_selected_shared(expression[:, shared], np.arange(2), 2)
        np.testing.assert_array_equal(full_rank, [[1., 0.]])
        np.testing.assert_array_equal(full_magnitude, [[8., 0.]])
        np.testing.assert_array_equal(shared_rank, [[2., 1.]])
        self.assertEqual(diagnostics['n_nonshared_selected'], [1])

    def test_positive_gene_ties_follow_original_panel_order(self):
        np.testing.assert_array_equal(ranked_positive_indices(np.array([4., 4., 4., 0., -2.]), 2), [0, 1])
        np.testing.assert_array_equal(ranked_positive_indices(np.array([0., 4., 4., 1.]), 8), [1, 2, 3])

    def test_shared_first_diagnostic_preserves_panel_ties_when_no_genes_are_removed(self):
        expression = np.array([[5., 5., 5.]])
        # Output shared columns are alphabetical, which need not match the panel order.
        shared = np.array([1, 2, 0])
        full_rank, _, _ = encode_selected_shared(expression, shared, 2)
        diagnostic_rank, _, _ = encode_shared_first(expression, shared, 2)
        np.testing.assert_array_equal(full_rank, [[1., 0., 2.]])
        np.testing.assert_array_equal(diagnostic_rank, full_rank)

    def test_identical_predictions_have_zero_paired_difference_but_no_equivalence_claim(self):
        truth = np.linspace(0, 2 * np.pi, 30, endpoint=False)
        predictions = {'model': truth + 0.2, 'rank': truth + 0.2}
        contrast = paired_contrasts(predictions, truth, [('model', 'rank')], n_boot=100, seed=3)['model_minus_rank']
        self.assertEqual(contrast['point_delta'], 0.0)
        self.assertEqual(contrast['ci95'], [0.0, 0.0])
        self.assertEqual(contrast['interpretation'], 'unresolved_difference')

    def test_constant_floor_uses_measured_target_phase_concentration(self):
        truth = np.array([0.0, 0.2, 0.4, 0.6])
        expected = abs(np.exp(1j * truth).mean())
        self.assertAlmostEqual(resultant_agreement(np.ones(4) * 2.3, truth), expected)

    def test_small_end_to_end_run_saves_auditable_full_precision_output(self):
        import anndata
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            route = root / 'route'
            route.mkdir()
            (route / 'cc_phase.py').write_text('''import numpy as np
def phase_angle_oriented(adata, seed=0):
    phi = np.asarray(adata.obs['phase'], float)
    return phi, phi * 0, phi * 0, {'n_cc_genes': 3, 'flipped': False}
''')
            phase = np.linspace(0, 2 * np.pi, 12, endpoint=False)
            rng = np.random.default_rng(5)
            for label, names in [('source', ['A', 'B', 'C', 'D']), ('target', ['A', 'B', 'C', 'E'])]:
                adata = anndata.AnnData(rng.uniform(1, 5, (12, 4)))
                adata.var_names = names
                adata.obs['phase'] = phase
                adata.write_h5ad(root / f'{label}.h5ad')
                states = root / label
                states.mkdir()
                (states / 'manifest.json').write_text(json.dumps({'unit': 'cell-summary', 'max_genes': 2, 'model': 'fixture'}))
                np.save(states / 'row_cell_ids.npy', np.arange(12))
                np.save(states / 'layer_21_activations.npy', np.column_stack([np.cos(phase), np.sin(phase)]))
            output = root / 'result.json'
            command = [sys.executable, str(Path(__file__).with_name('rev_selection_matched_transfer.py')),
                '--source-h5ad', str(root / 'source.h5ad'), '--target-h5ad', str(root / 'target.h5ad'),
                '--source-states', str(root / 'source'), '--target-states', str(root / 'target'),
                '--route-dir', str(route), '--k', '2', '--n-boot', '30', '--out', str(output)]
            completed = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            saved = json.loads(output.read_text())
            self.assertEqual(saved['n_shared_genes'], 3)
            self.assertEqual(saved['primary_contrast'], 'model_minus_expression_topk_rank_fullpanel')
            self.assertEqual(len(saved['predicted_target_phases']['model']), 12)
            self.assertEqual(len(saved['contrasts'][saved['primary_contrast']]['bootstrap_delta_draws']), 30)
            self.assertIn('diagnostic_shared_first_rank', saved['point'])
            self.assertIn('sha256', saved['inputs']['phase_code'])


if __name__ == '__main__':
    unittest.main()
