"""Small synthetic checks for the regulatory-test statistics; no model or h5ad needed."""
import unittest
import json
from itertools import combinations
from pathlib import Path
import tempfile
from unittest.mock import patch

import numpy as np

from rev_phase8_stats import (bh_adjust, feature_gene_scores, retrieval_metrics,
                             response_statistics, specificity_test, input_rank_matrix,
                             insilico_statistics, downstream_target_labels,
                             random_label_expected_ap, summarize_specificity)


class RegulatoryStatisticsTests(unittest.TestCase):
    def test_bh_counts_all_tests_and_preserves_order(self):
        np.testing.assert_allclose(bh_adjust([1, .01, .04, .02]), [1, .04, .053333333, .04])

    def test_positive_and_null_responses(self):
        rng = np.random.default_rng(3)
        control = rng.normal(size=(100, 12))
        perturbed = control.copy()
        perturbed[:, :3] += 4
        stats = response_statistics(perturbed, control)
        np.testing.assert_array_equal(stats['responding'], [True] * 3 + [False] * 9)
        null = response_statistics(control, control)
        self.assertFalse(null['responding'].any())

    def test_fisher_detects_target_specific_response(self):
        response = np.array([True] * 10 + [False] * 90)
        target = response.copy()
        record = specificity_test(response, target, np.ones(100, bool))
        self.assertEqual(record['table'], [[10, 0], [0, 90]])
        self.assertLess(record['p'], 1e-10)

    def test_ap_roc_and_chance_prevalence_with_ties(self):
        labels = np.array([1, 1, 0, 0, 0, 0], bool)
        perfect = retrieval_metrics(labels, [1, 1, 0, 0, 0, 0], bootstrap=20)
        self.assertEqual(perfect['average_precision'], 1)
        self.assertEqual(perfect['auroc'], 1)
        self.assertAlmostEqual(perfect['prevalence'], 1 / 3)
        chance = retrieval_metrics(labels, np.zeros(6), bootstrap=0)
        self.assertAlmostEqual(chance['average_precision'], 1 / 3)
        self.assertAlmostEqual(chance['auroc'], .5)
        self.assertIsNone(retrieval_metrics(np.ones(3, bool), [0, 1, 2])['auroc'])

    def test_exact_random_label_ap_matches_exhaustive_permutations(self):
        from sklearn.metrics import average_precision_score
        for scores in ([5, 4, 3, 2, 1], [3, 3, 2, 1, 1], [1, 1, 1, 1, 1]):
            for n_positive in (1, 2, 4):
                enumerated = []
                for positive_indices in combinations(range(5), n_positive):
                    labels = np.zeros(5, bool)
                    labels[list(positive_indices)] = True
                    enumerated.append(average_precision_score(labels, scores))
                self.assertAlmostEqual(random_label_expected_ap(scores, n_positive), np.mean(enumerated))
        single_positive = retrieval_metrics([1, 0, 0, 0, 0], [5, 4, 3, 2, 1], bootstrap=0)
        self.assertAlmostEqual(single_positive['random_label_expected_ap'], sum(1/i for i in range(1, 6)) / 5)
        self.assertLess(single_positive['lift_over_random_expected_ap'], single_positive['prevalence_lift'])
        tied = retrieval_metrics([1, 0, 0, 0, 0], np.zeros(5), bootstrap=0)
        self.assertAlmostEqual(tied['lift_over_random_expected_ap'], 1)

    def test_direct_factor_and_autoregulatory_target_are_excluded(self):
        known, labels, candidates = downstream_target_labels(['TF', 'A', 'B', 'C'], {'TF', 'A', 'B'}, 'tf')
        self.assertEqual(known, {'A', 'B'})
        np.testing.assert_array_equal(labels, [False, True, True, False])
        np.testing.assert_array_equal(candidates, [False, True, True, True])
        specificity = specificity_test([True, False, False, False], labels, candidates)
        self.assertEqual(specificity['n_responding_eligible'], 0)
        self.assertEqual(specificity['p'], 1)

    def test_screened_bh_family_is_separate_from_target_coverage(self):
        records = []
        for count, q in [(0, 1), (1, .01), (3, .04), (5, .2)]:
            test = specificity_test(np.zeros(10, bool), np.arange(10) < count, np.ones(10, bool))
            test['q'] = q
            records.append({'specificity': test, 'n_eligible_known_target_genes': count})
        summary = summarize_specificity(records, .05)
        self.assertEqual(summary['n_tfs_screened'], 4)
        self.assertEqual(summary['n_tfs_evaluable'], 3)
        self.assertEqual(summary['n_tfs_unevaluable'], 1)
        self.assertEqual(summary['target_coverage_sensitivity']['min_3_eligible_target_genes'],
                         {'n_tfs_evaluable': 2, 'n_specific_full_family_bh': 1})
        self.assertEqual(records[0]['specificity']['unevaluable_reason'], 'no eligible known targets')

    def test_gene_scores_average_only_feature_gene_memberships(self):
        genes = ['A', 'B', 'C', 'D']
        catalog = [{'feature_idx': 0, 'top_genes': [{'gene_name': 'A'}, {'gene_name': 'B'}]},
                   {'feature_idx': 1, 'top_genes': ['B', 'C']}]
        scores, coverage = feature_gene_scores(np.array([2., 4.]), catalog, genes)
        np.testing.assert_allclose(scores, [2, 3, 4, 0])
        np.testing.assert_array_equal(coverage, [1, 2, 1, 0])

    def test_input_rank_and_absence_use_same_gene_universe(self):
        matrix = input_rank_matrix([['B', 'A', 'C'], ['A']], ['A', 'B', 'C', 'D'], max_genes=3)
        np.testing.assert_allclose(matrix, [[2/3, 1, 1/3, 0], [1, 0, 0, 0]])

    def test_insilico_uses_paired_difference_and_minimum_count(self):
        arrays = {'sum_tf': np.array([[[12.], [5.]]]),
                  'sum_ctrl': np.array([[[2.], [4.]]]),
                  'sum_d2': np.array([[[12.], [1.]]]),
                  'count': np.array([[10., 1.]])}
        stats = insilico_statistics(arrays, min_cells=5)
        self.assertAlmostEqual(stats['mean_difference'][0, 0, 0], 1.)
        self.assertTrue(stats['eligible'][0, 0, 0])
        self.assertFalse(stats['eligible'][0, 1, 0])
        self.assertEqual(stats['q'][0, 1, 0], 1.)

    def test_complete_command_uses_exact_extracted_cells(self):
        import anndata
        from scipy import sparse
        from rev_phase8_stats import main
        rng = np.random.default_rng(7)
        genes = ['A', 'B', 'C', 'D', 'TF', 'F', 'G', 'H']
        counts = rng.poisson(20, size=(24, 8))
        counts[8:16, :3] += 30
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            feature_dir, catalog_dir = base / 'features', base / 'catalog'
            feature_dir.mkdir()
            catalog_dir.mkdir()
            adata = anndata.AnnData(X=sparse.csr_matrix(counts))
            adata.var_names = genes
            adata.write_h5ad(base / 'counts.h5ad')
            design, rows, feature_rows = [], [], []
            for group, kind, start in [('control', 'control', 0), ('TF', 'tf', 8), ('pseudo_00', 'pseudo', 16)]:
                cells = list(range(start, start + 8))
                sentences = [[genes[i] for i in np.argsort(-counts[cell], kind='stable')] for cell in cells]
                design.append({'group': group, 'kind': kind, 'cells': cells, 'sentences': sentences})
                # Extractor order need not match numerical h5ad row order.
                for cell in reversed(cells):
                    rows.append({'group': group, 'kind': kind, 'cell_index': cell})
                    feature_rows.append(counts[cell])
            (base / 'design.json').write_text(json.dumps({'source': 'synthetic', 'groups': design}))
            (feature_dir / 'rows.json').write_text(json.dumps(rows))
            np.savez_compressed(feature_dir / 'feat_L01.npz', X=np.asarray(feature_rows, float))
            catalog = [{'feature_idx': i, 'top_genes': [gene]} for i, gene in enumerate(genes)]
            (catalog_dir / 'feature_catalog_L01.json').write_text(json.dumps({'features': catalog}))
            (base / 'trrust.tsv').write_text('TF\tA\tActivation\t1\nTF\tB\tActivation\t1\n')
            arguments = ['rev_phase8_stats.py', '--features-dir', str(feature_dir), '--design', str(base / 'design.json'),
                         '--catalog-dir', str(catalog_dir), '--h5ad', str(base / 'counts.h5ad'),
                         '--trrust', str(base / 'trrust.tsv'), '--layers', '1', '--out', str(base / 'out'),
                         '--bootstrap', '0', '--max-genes', '8']
            with patch('sys.argv', arguments):
                main()
            manifest = json.loads((base / 'out/manifest.json').read_text())
            self.assertEqual(manifest['n_extracted_cells'], 24)
            output = json.loads((base / 'out/regulatory_statistics.json').read_text())
            self.assertEqual(len(output['summary']), 3)
            self.assertTrue(all(family['n_tfs'] == 1 for family in output['summary']))
            stats = np.load(base / 'out/gene_statistics.npz')
            self.assertEqual(stats['expression_p'].shape, (2, 8))
            np.testing.assert_array_equal(stats['cell_indices'], [row['cell_index'] for row in rows])
            expected = response_statistics(np.log1p(counts[8:16] / counts[8:16].sum(1, keepdims=True) * 10000),
                                           np.log1p(counts[:8] / counts[:8].sum(1, keepdims=True) * 10000))
            np.testing.assert_allclose(stats['expression_effect'][0], expected['effect'], rtol=1e-5)


if __name__ == '__main__':
    unittest.main()
