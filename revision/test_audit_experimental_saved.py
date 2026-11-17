"""Independent audit invariants: denominators, full IDs, bootstrap units and moments."""
import unittest
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from audit_experimental_saved import Audit, annotation_counts, circuit_from_source_counts, residual_metrics, audit_residual


def tiny_annotation():
    return {'summary': {'alpha': .05, 'n_features_total': 4, 'n_alive': 3, 'n_sets': 3,
                       'n_testable': 2, 'n_annotated': 2, 'n_ontology_annotated': 1,
                       'n_network_or_membership_only': 1, 'annotation_rate': 2/3,
                       'annotation_rate_all_features': .5, 'ontology_annotation_rate_alive': 1/3,
                       'ontology_annotation_rate_all_features': .25, 'rate_by_source': {'GO_BP': 1/3},
                       'global_bh': {'p_threshold': .001, 'annotation_rate': 1/3},
                       'median_set_size': 3., 'floors': {'freq': {'values': [1/3, 0, 1/3],
                       'mean': 2/9, 'sd': np.std([1/3, 0, 1/3]), 'ontology_values': [0, 0, 0], 'ontology_mean': 0.}}},
            'features': [{'feature_idx': 0, 'n_genes': 3, 'n_sig': 1, 'annotated': True, 'min_p': .0001,
                          'significant_term_ids': ['GO_BP::Complete but not displayed'], 'terms': []},
                         {'feature_idx': 1, 'n_genes': 3, 'n_sig': 1, 'annotated': True, 'min_p': .01,
                          'significant_term_ids': [], 'terms': [{'source': 'STRING', 'q': .02}]},
                         {'feature_idx': 2, 'n_genes': 0, 'n_sig': 0, 'annotated': False, 'min_p': 1.,
                          'significant_term_ids': [], 'terms': []}]}


class SavedExperimentAuditTests(unittest.TestCase):
    def test_all_residual_taps_do_not_require_untrained_27b_dictionaries(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root/'results').mkdir()
            summary = {}
            for model, run, prefix, depth in [('2b','2b_atlas','atlas-extract',26),
                                              ('27b_matched','27b_matched','scale-extract',46)]:
                trained_layers = set(range(26)) if model=='2b' else {0,7,14,24,31,38,45}
                directory = root/'runs'/run/'sae'; directory.mkdir(parents=True)
                arrays, rows = {}, []
                for layer in range(depth):
                    moments = {'eig':[1.,1.], 'rms':[1.,1.], 'mean':[0.,0.], 'scalars':[10.,1.,2.,1.]}
                    arrays.update({f'{key}_L{layer:02d}':value for key,value in moments.items()})
                    row = residual_metrics(*(moments[key] for key in ['eig','rms','mean','scalars']))
                    row.pop('moment_centered_trace')
                    row.update(layer=layer, dead_fraction_heldout=None, dead_fraction_heldout_original=None,
                               dead_fraction_all_rows=None, variance_explained_original=None,
                               variance_explained_reextracted=None)
                    if layer in trained_layers:
                        training = {'n_features':4, 'dead_frac':.25, 'variance_explained':.9, 'dead_frac_all_rows':0.}
                        (directory/f'train_L{layer:02d}.json').write_text(json.dumps(training))
                        row.update(dead_fraction_heldout=.25, dead_fraction_heldout_original=.25,
                                   dead_fraction_all_rows=0., variance_explained_original=.9)
                        if model=='2b':
                            evaluation={'n_features':4,'dead_val_reextracted':1,'dead_frac_all_rows':0.,'variance_explained_reextracted':.91}
                            (root/'results'/f'2b_eval_L{layer:02d}.json').write_text(json.dumps(evaluation))
                            row['variance_explained_reextracted']=.91
                    rows.append(row)
                np.savez(root/'results'/f'{prefix}_residual_stats.npz', **arrays)
                summary[model]={'layers':rows,'descriptive_rank_correlations':{}}
            path=root/'results/residual_summary.json';path.write_text(json.dumps(summary))
            audit=Audit();audit_residual(root,audit)
            self.assertTrue(audit.passed,audit.failed)
            self.assertEqual(audit.pending,[])
            # A required trained tap must still be reported missing.
            (root/'runs/27b_matched/sae/train_L24.json').unlink()
            audit=Audit();audit_residual(root,audit)
            self.assertTrue(any('train_L24.json' in name for name in audit.pending))
            # An untrained tap still requires its measured residual spectrum.
            summary['27b_matched']['layers'].pop(1);path.write_text(json.dumps(summary))
            audit=Audit();audit_residual(root,audit)
            self.assertTrue(any('coverage' in row['check'] for row in audit.failed))

    def test_tiny_p_values_need_relative_agreement_and_nonfinite_failures_serialize(self):
        audit = Audit(); audit.equal('tiny q', 1e-20, 1e-19, atol=1e-300)
        self.assertFalse(audit.passed)
        audit.equal('nonfinite vector', np.array([np.nan]), np.array([1.]))
        json.dumps(audit.failed, allow_nan=False)

    def test_counts_use_complete_ids_and_all_vs_alive_denominators(self):
        audit = Audit(); result = annotation_counts(tiny_annotation(), audit)
        self.assertTrue(audit.passed, audit.failed)
        self.assertEqual(result['n_ontology'], 1)
        self.assertEqual(result['ontology_rate_all_features'], .25)
        self.assertAlmostEqual(result['frequency_ontology_floor_all_features'], 0.)

    def test_wrong_denominator_and_missing_complete_ids_are_detected(self):
        annotation = tiny_annotation()
        annotation['summary']['ontology_annotation_rate_all_features'] = 1/3
        del annotation['features'][0]['significant_term_ids']
        audit = Audit(); annotation_counts(annotation, audit)
        self.assertFalse(audit.passed)
        self.assertTrue(any('complete' in row['check'] for row in audit.failed))
        self.assertTrue(any('all_features' in row['check'] for row in audit.failed))

    def test_duplicate_significant_ids_are_not_silently_counted(self):
        annotation = tiny_annotation()
        annotation['features'][0]['significant_term_ids'] *= 2
        audit = Audit(); annotation_counts(annotation, audit)
        self.assertFalse(audit.passed)

    def test_circuit_resamples_sources_including_empty_source_clusters(self):
        rows = [{'shared': 2, 'eligible': 2, 'expected_shared': .5},
                {'shared': 0, 'eligible': 1, 'expected_shared': .25},
                {'shared': 0, 'eligible': 0, 'expected_shared': 0}]
        result = circuit_from_source_counts(rows, 1000, 9)
        self.assertAlmostEqual(result['observed'], 2/3)
        self.assertAlmostEqual(result['same_source_random_target'], .25)
        self.assertAlmostEqual(result['excess'], 5/12)
        self.assertEqual(result['n_source_clusters'], 3)
        # With these three independent source units, some nonempty draws contain
        # exclusively zero-sharing sources and others only perfect-sharing sources.
        self.assertEqual(result['source_bootstrap']['observed']['ci95'], [0., 1.])

    def test_residual_centered_pr_is_distinct_from_uncentered_rms(self):
        result = residual_metrics([3., 1.], [np.sqrt(12.), 2.], [3., np.sqrt(3.)], [12, 4., 9., .5])
        self.assertAlmostEqual(result['centered_covariance_pr'], 1.6)
        self.assertAlmostEqual(result['centered_covariance_pr_fraction_width'], .8)
        self.assertAlmostEqual(result['mean_vector_l2_norm'], np.sqrt(12.))
        self.assertEqual(result['max_abs_over_median_of_cell_medians'], 18.)
        self.assertAlmostEqual(result['moment_centered_trace'], 4.)
        with self.assertRaises(ValueError):
            residual_metrics([3., np.nan], [1., 2.], [0., 0.], [12, 1, 3])


if __name__ == '__main__':
    unittest.main()
