"""Small fixtures for the archived cell-cycle ruler; no large matrices or model loading."""
import unittest
import json
from pathlib import Path
import tempfile
from unittest.mock import patch
import numpy as np
from rev_within_cellcycle import encode_topk_inputs, circular_r2, paired_cell_bootstrap, check_cache_validation


class WithinCellCycleTests(unittest.TestCase):
    def test_topk_uses_positive_genes_and_stable_full_panel_ties(self):
        encoded = encode_topk_inputs(np.array([[0., 5., 5., 2., 0.], [1., 0., 0., 0., 0.]]), k=2)
        np.testing.assert_array_equal(encoded['expression_topk_rank'], [[0, 2, 1, 0, 0], [2, 0, 0, 0, 0]])
        np.testing.assert_array_equal(encoded['expression_topk_binary'], [[0, 1, 1, 0, 0], [1, 0, 0, 0, 0]])
        np.testing.assert_array_equal(encoded['expression_topk_magnitude'], [[0, 5, 5, 0, 0], [1, 0, 0, 0, 0]])

    def test_perfect_phase_r_squared_and_constant_reference(self):
        phase = np.linspace(0, 2*np.pi, 60, endpoint=False)
        self.assertEqual(circular_r2(phase, phase), 1.)
        self.assertAlmostEqual(circular_r2(np.zeros_like(phase), phase), 0., places=10)

    def test_identical_fixed_predictions_have_zero_paired_delta_not_equivalence(self):
        phase = np.linspace(0, 2*np.pi, 40, endpoint=False)
        result = paired_cell_bootstrap({'model': phase, 'rank': phase}, phase,
                                       [('model', 'rank')], n_boot=100, seed=4)
        contrast = result['contrasts']['model_minus_rank']
        np.testing.assert_allclose(contrast['ci95'], [0., 0.])
        self.assertEqual(contrast['interpretation'], 'unresolved_difference')
        self.assertIn('not equivalence', contrast['interpretation_limit'])

    def test_cli_preserves_nontrivial_state_row_order_and_records_provenance(self):
        import anndata
        from scipy import sparse
        from rev_within_cellcycle import main
        phase = np.linspace(0, 2*np.pi, 20, endpoint=False)
        order = np.roll(np.arange(20), 3)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            states, route = root/'states', root/'route'
            states.mkdir(); route.mkdir()
            counts = np.random.default_rng(7).poisson(5, size=(20, 6)).astype(np.float32)
            cohort = anndata.AnnData(X=sparse.csr_matrix(counts))
            cohort.obs_names = [f'cell-{i}' for i in range(20)]
            cohort.obs['phase'] = phase
            cohort.write_h5ad(root/'cohort.h5ad')
            (route/'cc_phase.py').write_text('import numpy as np\ndef phase_angle_oriented(a,seed=0):\n return a.obs["phase"].to_numpy(),None,None,{"synthetic_fixture":True}\n')
            np.save(states/'row_cell_ids.npy', order)
            np.save(states/'layer_21_activations.npy', np.column_stack([np.cos(phase), np.sin(phase)])[order])
            (states/'manifest.json').write_text(json.dumps({'unit': 'cell-summary', 'model': 'vandijklab/C2S-Scale-Gemma-2-2B',
                'layers': [21], 'max_genes': 3, 'prompt': True, 'n_rows': 20, 'n_cells': 20, 'd_model': 2}))
            arguments = ['rev_within_cellcycle.py', '--h5ad', str(root/'cohort.h5ad'), '--states', str(states),
                         '--route-dir', str(route), '--out', str(root/'out.json'), '--expected-cells', '20', '--k', '3', '--n-boot', '20']
            with patch('sys.argv', arguments), patch('builtins.print'):
                main()
            output = json.loads((root/'out.json').read_text())
            self.assertEqual(output['cell_indices'], order.tolist())
            self.assertEqual(output['cell_ids'], [f'cell-{i}' for i in order])
            np.testing.assert_allclose(output['phase'], phase[order])
            self.assertEqual(set(output['point']), {'expression_full', 'expression_topk_magnitude',
                             'expression_topk_binary', 'expression_topk_rank', 'model', 'constant', 'random'})
            self.assertIn('sha256', output['inputs']['h5ad'])

    def test_optional_cache_validation_rejects_a_different_cohort_hash(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory/'cohort').write_text('current cohort')
            (directory/'validation.json').write_text(json.dumps({'schema': 'sampled-cell-summary-cache-validation-v1',
                'model': 'model', 'layer': 21, 'all_sampled_rows_pass': True, 'n_sampled': 20,
                'inputs': {'h5ad': {'sha256': 'wrong hash'}}, 'coverage_limit': 'sampled rows only'}))
            with self.assertRaises(ValueError):
                check_cache_validation(directory/'validation.json', {'h5ad': directory/'cohort'}, 'model', 21)


if __name__=='__main__':
    unittest.main()
