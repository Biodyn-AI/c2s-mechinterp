"""Independent checks against exhaustive retrieval and full-family BH examples."""
import itertools
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from sklearn.metrics import average_precision_score

from audit_phase8_saved import random_ap, check_response, metric_summary, audit_phase8, check_raw_feature_sample
from audit_experimental_saved import Audit


class Phase8SavedAuditTests(unittest.TestCase):
    def test_random_ap_matches_exhaustive_labels_with_ties(self):
        scores = np.array([3., 3., 1., 0.])
        measured = []
        for positives in itertools.combinations(range(4), 2):
            labels = np.zeros(4); labels[list(positives)] = 1
            measured.append(average_precision_score(labels, scores))
        self.assertAlmostEqual(random_ap(scores, 2), np.mean(measured))

    def test_response_bh_includes_null_columns_and_strict_effect_boundary(self):
        # A p=.02 response cannot survive BH if three p=1 null columns
        # are discarded; its q must be .08, not .02.
        arrays = {'p': np.array([.02, 1, 1, 1]), 'q': np.array([.08, 1, 1, 1]),
                  'effect': np.array([2., 0, .5, 1]), 'responding': np.zeros(4, bool)}
        audit = Audit(); check_response(arrays, audit, .05, .5)
        self.assertTrue(audit.passed, audit.failed)
        arrays['q'][0] = .02
        audit = Audit(); check_response(arrays, audit, .05, .5)
        self.assertFalse(audit.passed)

    def test_family_means_resample_factors_and_exclude_undefined_ap(self):
        records = [{'average_precision': .2, 'auroc': .3, 'ap_lift': 2., 'prevalence': .1,
                    'prevalence_lift': 2., 'random_label_expected_ap': .15, 'lift_over_random_expected_ap': 4/3},
                   {'average_precision': .8, 'auroc': .7, 'ap_lift': 4., 'prevalence': .2,
                    'prevalence_lift': 4., 'random_label_expected_ap': .4, 'lift_over_random_expected_ap': 2.},
                   {'average_precision': None}]
        result = metric_summary(records, 1000, 4)
        self.assertEqual(result['n_tfs_evaluable'], 2)
        self.assertAlmostEqual(result['mean_average_precision'], .5)
        self.assertEqual(result['mean_average_precision_ci95'], [.2, .8])

    def test_raw_feature_sample_catches_wrong_effects_and_uses_exact_cell_groups(self):
        matrix = np.array([[0., 0.], [1., 0.], [2., 0.], [5., 0.], [6., 0.], [7., 0.]])
        rows = [{'group': 'control', 'kind': 'control'}]*3 + [{'group': 'TF', 'kind': 'tf'}]*3
        # Equal population variances=2/3 in the two groups; mean difference=5.
        from scipy.stats import mannwhitneyu
        p = mannwhitneyu(matrix[3:], matrix[:3], axis=0, method='asymptotic').pvalue
        arrays = {'p': np.array([p]), 'effect': np.array([[5/np.sqrt(2/3), 0.]]),
                  'q': np.ones((1, 2)), 'responding': np.zeros((1, 2), bool)}
        audit = Audit(); check_raw_feature_sample(matrix, rows, ['TF'], arrays, audit)
        self.assertTrue(audit.passed, audit.failed)
        arrays['effect'][0, 0] = 0
        audit = Audit(); check_raw_feature_sample(matrix, rows, ['TF'], arrays, audit)
        self.assertFalse(audit.passed)

    def test_complete_tied_score_fixture_checks_self_exclusion_and_pseudo_groups(self):
        # Every score is tied: AP must equal prevalence, AUROC=.5 and lift=1.
        # Each TF has one downstream target plus an autoregulatory record that
        # must not contribute to either gene or feature labels.
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            stats, catalogs = folder/'statistics', folder/'catalogs'
            stats.mkdir(); catalogs.mkdir()
            trrust = folder/'trrust.tsv'
            trrust.write_text('TF1\tTF1\nTF1\tA\nTF2\tTF2\nTF2\tB\n')
            groups = ['TF1', 'TF2', 'pseudo']
            np.savez(stats/'gene_statistics.npz', genes=['TF1', 'TF2', 'A', 'B'], groups=groups,
                     expression_p=np.ones((3, 4)), expression_q=np.ones((3, 4)),
                     expression_effect=np.zeros((3, 4)), expression_responding=np.zeros((3, 4), bool),
                     expression_eligible=np.ones(4, bool))
            np.savez(stats/'feature_statistics_L08.npz', groups=groups, eligible=np.ones(2, bool),
                     p=np.ones((3, 2)), q=np.ones((3, 2)), effect=np.zeros((3, 2)),
                     responding=np.zeros((3, 2), bool), gene_scores=np.zeros((3, 4)))
            (catalogs/'feature_catalog_L08.json').write_text(json.dumps({'features': [
                {'feature_idx': 0, 'top_genes': ['A', 'TF1']},
                {'feature_idx': 1, 'top_genes': ['B', 'TF2']}]}))
            arguments = {'alpha': .05, 'effect_threshold': .5, 'bootstrap': 20, 'seed': 2}
            (stats/'manifest.json').write_text(json.dumps({'arguments': arguments, 'n_genes': 4}))

            def retrieval(n, positive):
                prevalence = positive/n
                valid = positive > 0
                return {'n': n, 'n_positive': positive, 'prevalence': prevalence,
                        'random_label_expected_ap': prevalence,
                        'average_precision': prevalence if valid else None, 'auroc': .5 if valid else None,
                        'ap_lift': 1. if valid else None, 'prevalence_lift': 1. if valid else None,
                        'lift_over_random_expected_ap': 1. if valid else None,
                        'average_precision_ci95': [prevalence]*2 if valid else None,
                        'auroc_ci95': [.5]*2 if valid else None, 'ap_lift_ci95': [1.]*2 if valid else None,
                        'lift_over_random_expected_ap_ci95': [1.]*2 if valid else None}

            records, summaries = [], []
            for representation, layer, width in [('expression', None, 3), ('model', 8, 2)]:
                for group in groups:
                    pseudo = group == 'pseudo'
                    n = (4 if pseudo else 3) if representation == 'expression' else width
                    specificity = {'table': [[0, 0], [0 if pseudo else 1, n if pseudo else n-1]],
                                   'p': 1., 'n_eligible': n, 'n_target_columns': 0 if pseudo else 1,
                                   'n_non_target_columns': n if pseudo else n-1, 'n_target_responses': 0,
                                   'n_responding_eligible': 0, 'evaluable': not pseudo}
                    if not pseudo:
                        specificity['q'] = 1.
                    record = {'representation': representation, 'layer': layer, 'group': group,
                              'kind': 'pseudo' if pseudo else 'tf', 'n_responding': 0,
                              'self_target_excluded': True, 'perturbed_gene_excluded_from_gene_retrieval': True,
                              'n_eligible_known_target_genes': 0 if pseudo else 1, 'specificity': specificity,
                              'gene_retrieval': retrieval(4 if pseudo else 3, 0 if pseudo else 1)}
                    if representation == 'model':
                        record['feature_retrieval'] = retrieval(2, 0 if pseudo else 1)
                    records.append(record)
                summary = {'representation': representation, 'layer': layer, 'n_tfs': 2,
                           'n_tfs_screened': 2, 'n_tfs_evaluable': 2, 'n_tfs_unevaluable': 0,
                           'n_specific': 0, 'n_groups': 3, 'n_responding_groups': 0,
                           'pseudo_controls': [{'group': 'pseudo', 'n_responding': 0}],
                           'target_coverage_sensitivity': {f'min_{k}_eligible_target_genes': {
                               'n_tfs_evaluable': 2 if k == 1 else 0, 'n_specific_full_family_bh': 0} for k in [1, 3, 5]}}
                for kind in ['gene_retrieval', 'feature_retrieval']:
                    metric = retrieval(3 if kind == 'gene_retrieval' else 2, 1)
                    aggregate = {'n_tfs_evaluable': 2}
                    if kind == 'feature_retrieval' and representation == 'expression':
                        aggregate['n_tfs_evaluable'] = 0
                    for name in ['average_precision', 'auroc', 'ap_lift', 'prevalence', 'prevalence_lift',
                                 'random_label_expected_ap', 'lift_over_random_expected_ap']:
                        value = metric[name] if aggregate['n_tfs_evaluable'] else None
                        aggregate['mean_'+name] = value
                        aggregate['mean_'+name+'_ci95'] = [value]*2 if value is not None else None
                    summary[kind] = aggregate
                summaries.append(summary)
            (stats/'regulatory_statistics.json').write_text(json.dumps({'groups': records, 'summary': summaries}))
            audit = Audit(); audit_phase8(stats, catalogs, trrust, audit)
            self.assertTrue(audit.passed, audit.failed)
            # A target count that includes autoregulation is rejected by the
            # actual label/score reconstruction, not only an exclusion flag.
            records[0]['gene_retrieval']['n_positive'] = 2
            (stats/'regulatory_statistics.json').write_text(json.dumps({'groups': records, 'summary': summaries}))
            audit = Audit(); audit_phase8(stats, catalogs, trrust, audit)
            self.assertFalse(audit.passed)


if __name__ == '__main__':
    unittest.main()
