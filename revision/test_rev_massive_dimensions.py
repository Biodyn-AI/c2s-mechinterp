"""Lightweight E4 checks; no model training or GPU dependency."""
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from rev_sae_tools import projection_spec, zero_columns_in_place, cmd_train
from rev_massive_dimensions import high_rms_preset, compare_records


class MassiveDimensionTests(unittest.TestCase):
    def test_projection_changes_only_selected_ram_columns_and_preserves_disk(self):
        original = np.arange(24, dtype=np.float32).reshape(6, 4)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'activations.npy'
            np.save(path, original)
            activations = np.load(path).astype(np.float32)
            pointer = activations.__array_interface__['data'][0]
            zero_columns_in_place(activations, [1, 3])
            self.assertEqual(activations.__array_interface__['data'][0], pointer)
            np.testing.assert_array_equal(activations[:, [0, 2]], original[:, [0, 2]])
            np.testing.assert_array_equal(activations[:, [1, 3]], 0)
            np.testing.assert_array_equal(np.load(path), original)

    def test_projection_validates_integer_unique_in_bounds_and_layer(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'dimensions.json'
            for invalid in ([1, 1], [-1], [4], [1.5], [True]):
                path.write_text(json.dumps({'layer': 24, 'indices': invalid}))
                with self.assertRaises(ValueError): projection_spec(str(path), 4, 24)
            path.write_text(json.dumps({'layer': 38, 'indices': [1]}))
            with self.assertRaises(ValueError): projection_spec(str(path), 4, 24)

    def test_high_rms_selection_is_strict_and_records_empty_skip(self):
        self.assertEqual(high_rms_preset([1, 1, 1, 11], 24)['indices'], [3])
        record = high_rms_preset([1, 1, 1, 10], 24)
        self.assertEqual(record['indices'], [])
        self.assertEqual(record['status'], 'skip_no_high_rms_dimensions')
        self.assertEqual(high_rms_preset([0, 0, 0], 24)['status'], 'skip_zero_median_rms')

    def test_comparison_does_not_subtract_variance_from_different_inputs(self):
        baseline = {'n_features': 9216, 'k': 32, 'epochs': 6, 'seed': 0,
                    'dead_frac': .4, 'dead_frac_all_rows': .3, 'variance_explained': .98}
        projected = {**baseline, 'dead_frac': .2, 'dead_frac_all_rows': .1,
                     'variance_explained': .9, 'input_projection': {'mode': 'zero_selected_columns', 'removed_dimensions': [2]}}
        record = compare_records(baseline, projected, 24)
        self.assertAlmostEqual(record['dead_fraction_heldout_delta'], -.2)
        self.assertEqual(record['variance_explained_projected_input'], .9)
        self.assertNotIn('variance_explained_delta', record)
        skip = compare_records(baseline, {'status': 'skipped_no_dimensions', 'input_projection': {'removed_dimensions': []}}, 24)
        self.assertEqual(skip['status'], 'skipped_no_dimensions')

    def test_empty_preset_writes_skip_without_loading_activations_or_training(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            np.save(directory/'layer_24_activations.npy', np.ones((6, 4), np.float32))
            (directory/'dimensions.json').write_text(json.dumps({'layer': 24, 'indices': []}))
            arguments = SimpleNamespace(seed=0, run=str(directory), layer=24,
                                        drop_dims=str(directory/'dimensions.json'), sae_subdir='sae_dropmassive')
            fake_torch = SimpleNamespace(manual_seed=lambda seed: None,
                                          cuda=SimpleNamespace(is_available=lambda: False))
            fake_sae = SimpleNamespace(SAETrainer=None, TopKSAE=None)
            with patch.dict('sys.modules', {'torch': fake_torch, 'sae_topk': fake_sae}), \
                 patch('rev_sae_tools._load_acts', side_effect=AssertionError('full activations must not load')):
                cmd_train(arguments)
            record = json.loads((directory/'sae_dropmassive/train_L24.json').read_text())
            self.assertEqual(record['status'], 'skipped_no_dimensions')
            self.assertFalse((directory/'sae_dropmassive/sae_L24.pt').exists())

    def test_queue_places_optional_ablation_after_completed_core(self):
        from run_revision import build_queue
        with tempfile.TemporaryDirectory() as temporary:
            stages = build_queue(Path(temporary), Path(__file__).parent)
            names = [stage.name for stage in stages]
            selected = [stage for stage in stages if stage.name.startswith('scale-dropmassive')]
            self.assertEqual(len(selected), 4)
            self.assertTrue(all(stage.optional for stage in selected))
            self.assertGreater(names.index('scale-dropmassive-presets'), names.index('scale-residual-summary'))
            for stage in selected:
                if '-train-' in stage.name:
                    self.assertEqual(len(stage.outputs), 1)
                    self.assertEqual(stage.outputs[0].suffix, '.json')  # Empty selections legitimately have no checkpoint.


if __name__=='__main__':
    unittest.main()
