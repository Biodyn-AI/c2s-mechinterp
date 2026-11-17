"""Synthetic tests for response categories and explicit cell selection."""
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from rev_response_biology import response_profile
from rev_cpu_analysis import choose_cell_indices


class ResponseBiologyTests(unittest.TestCase):
    def setUp(self):
        self.catalogue = [{'feature_idx': index, 'top_genes': [gene]}
                          for index, gene in enumerate(['A', 'B', 'C', 'D'])]
        self.annotation = {0: {'significant_term_ids': ['GO_BP::cytoplasmic translation']},
                           1: {'significant_term_ids': ['GO_BP::mitotic cell cycle', 'Reactome::cell cycle']},
                           2: {'significant_term_ids': []}, 3: {'significant_term_ids': []}}

    def test_target_label_and_generic_category_are_separate_axes(self):
        record = response_profile([1, 1, 1, 0], [2., -3., .8, 0.], [.01, .02, .03, 1.],
                                  self.catalogue, self.annotation, {'A'}, perturbed_gene='TF')
        self.assertEqual(record['n_responding'], 3)
        self.assertEqual(record['n_target_labelled_responses'], 1)
        self.assertEqual(record['n_generic_programme_responses'], 2)
        self.assertEqual(record['exclusive_response_counts'], {'target_labelled': 1,
                         'generic_programme_without_target_label': 1, 'other_or_unannotated_without_target_label': 1})
        self.assertEqual(record['category_counts']['cell cycle and mitosis'], 1)
        self.assertEqual(record['n_ontology_unannotated_responses'], 1)
        self.assertEqual(record['named_features'][0]['feature_idx'], 1)
        self.assertNotIn('direction', record['named_features'][0])

    def test_autoregulation_cannot_label_a_self_only_feature_as_downstream(self):
        catalogue = [{'feature_idx': 0, 'top_genes': ['TF']},
                     {'feature_idx': 1, 'top_genes': ['TARGET']},
                     {'feature_idx': 2, 'top_genes': ['OTHER']}]
        annotations = {index: {'significant_term_ids': []} for index in range(3)}
        record = response_profile([1, 1, 0], [3., 2., 0.], [.01, .02, 1.],
                                  catalogue, annotations, {'TF', 'TARGET'}, perturbed_gene='tf')
        self.assertEqual(record['n_target_labelled_responses'], 1)
        self.assertEqual(record['exclusive_response_counts']['target_labelled'], 1)
        self.assertEqual(record['named_features'][0]['matching_trrust_targets'], [])
        self.assertEqual(record['named_features'][1]['matching_trrust_targets'], ['TARGET'])

    def test_partial_annotation_ids_and_nonfinite_effects_are_rejected(self):
        with self.assertRaises(ValueError):
            response_profile([1, 0, 0, 0], [np.inf, 0, 0, 0], [.01, 1, 1, 1],
                             self.catalogue, self.annotation, {'A'}, perturbed_gene='TF')
        self.annotation[0] = {'terms': []}
        with self.assertRaises(ValueError):
            response_profile([1, 0, 0, 0], [2, 0, 0, 0], [.01, 1, 1, 1],
                             self.catalogue, self.annotation, {'A'}, perturbed_gene='TF')

    def test_explicit_cell_selection_preserves_order_and_rejects_duplicates(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'cells.json'
            path.write_text(json.dumps([7, 2, 5]))
            np.testing.assert_array_equal(choose_cell_indices(10, 4, np.random.default_rng(0), str(path)), [7, 2, 5])
            for invalid in ([1, 1], [2, 10], [1, 2.5], []):
                path.write_text(json.dumps(invalid))
                with self.assertRaises(ValueError):
                    choose_cell_indices(10, 4, np.random.default_rng(0), str(path))

    def test_default_cell_selection_preserves_previous_sampling(self):
        expected = np.sort(np.random.default_rng(5).choice(20, 7, replace=False))
        np.testing.assert_array_equal(choose_cell_indices(20, 7, np.random.default_rng(5)), expected)

    def test_dimensions_use_the_exact_explicit_cells(self):
        import anndata
        from scipy import sparse
        from rev_cpu_analysis import cmd_dimensions, dimension_metrics
        counts = np.random.default_rng(7).poisson(10, size=(9, 6)).astype(np.float32)
        cells, genes = [7, 2, 5, 1], ['A', 'B', 'C', 'D', 'E', 'F']
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            adata = anndata.AnnData(X=sparse.csr_matrix(counts))
            adata.var_names = genes
            adata.write_h5ad(directory/'counts.h5ad')
            (directory/'cells.json').write_text(json.dumps(cells))
            np.savez(directory/'model.npz', genes=np.asarray(genes), L09=np.random.default_rng(2).normal(size=(6, 3)))
            args = SimpleNamespace(h5ad=str(directory/'counts.h5ad'), counts_layer='', seed=0, n_cells=8,
                                   cell_indices=str(directory/'cells.json'), model_vectors=str(directory/'model.npz'),
                                   layers=[9], gene_cap=0, out=str(directory/'out.json'))
            cmd_dimensions(args)
            record = json.loads((directory/'out.json').read_text())
            self.assertEqual(record['cell_indices'], cells)
            expected = np.log1p(counts[cells]/counts[cells].sum(1, keepdims=True)*1e4).T
            self.assertAlmostEqual(record['rows'][0]['participation_ratio'], dimension_metrics(expected)['participation_ratio'], places=5)

    def test_response_biology_command_aligns_masks_records_and_named_examples(self):
        from rev_response_biology import main
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            stats, annotations, catalogues = [directory/name for name in ('statistics', 'annotations', 'catalogues')]
            for path in (stats, annotations, catalogues): path.mkdir()
            groups = ['PHB2', 'pseudo_00']
            np.savez(stats/'feature_statistics_L08.npz', groups=np.asarray(groups),
                     responding=[[1, 1, 1, 0], [0, 0, 0, 0]], effect=[[2., -3., .8, 0.], [0., 0., 0., 0.]],
                     q=[[.01, .02, .03, 1.], [1., 1., 1., 1.]])
            records = [{'representation': 'model', 'layer': 8, 'group': group, 'kind': kind,
                        'n_cells': 5, 'n_control_cells': 10, 'n_responding': responding,
                        'specificity': {'p': .5}} for group, kind, responding in [('PHB2', 'tf', 3), ('pseudo_00', 'pseudo', 0)]]
            (stats/'regulatory_statistics.json').write_text(json.dumps({'groups': records}))
            (annotations/'ann_v2_L08.json').write_text(json.dumps({'features': [{'feature_idx': key, **value} for key, value in self.annotation.items()]}))
            command_catalogue = [dict(row) for row in self.catalogue]
            command_catalogue[1] = {'feature_idx': 1, 'top_genes': ['PHB2']}
            (catalogues/'feature_catalog_L08.json').write_text(json.dumps({'features': command_catalogue}))
            (directory/'trrust.tsv').write_text('PHB2\tA\tActivation\t1\nPHB2\tPHB2\tActivation\t2\n')
            arguments = ['rev_response_biology.py', '--stats-dir', str(stats), '--annotations-dir', str(annotations),
                         '--catalog-dir', str(catalogues), '--trrust', str(directory/'trrust.tsv'), '--layers', '8',
                         '--out', str(directory/'out.json')]
            with patch('sys.argv', arguments): main()
            output = json.loads((directory/'out.json').read_text())
            self.assertEqual(len(output['responses']), 2)
            self.assertEqual(output['predefined_worked_examples']['PHB2']['status'], 'present')
            self.assertEqual(output['predefined_worked_examples']['GATA1']['status'], 'not present among tested TFs')
            self.assertEqual(output['responses'][0]['n_target_labelled_responses'], 1)


if __name__ == '__main__':
    unittest.main()
