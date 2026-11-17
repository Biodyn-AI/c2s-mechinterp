"""Small fixtures for strict supplementary assembly; no analysis is executed."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from assemble_supplement import Assembler, table, tex_escape


class SupplementAssemblyTests(unittest.TestCase):
    def write_scaling_fixture(self, root):
        results = root/'results'
        results.mkdir(exist_ok=True)
        pairs = []
        for small, large in zip((0, 4, 8, 13, 17, 21, 25), (0, 7, 14, 24, 31, 38, 45)):
            pair = {}
            for model, layer, active, annotated in [('2b', small, 9216, 4608), ('27b_matched', large, 4608, 2304)]:
                pair[model] = {'model': model, 'layer': layer, 'n_features': 9216,
                    'n_catalogue_active': active, 'n_catalogue_inactive': 9216-active,
                    'n_ontology': annotated, 'heldout_inactive_fraction': .125 if model=='2b' else .625,
                    'all_row_inactive_fraction': (9216-active)/9216, 'heldout_variance_explained': .876543,
                    'ontology_all_features': annotated/9216, 'ontology_catalogue_active': annotated/active,
                    'ontology_frequency_floor_all_features': .2*active/9216,
                    'ontology_frequency_floor_catalogue_active': .2}
            pairs.append(pair)
        report = {'schema': 'c2s-scale-reporting-v1', 'passed': True, 'n_pairs': 7,
                  'pairs': pairs, 'source_sha256': {'source.json': 'a'*64}}
        (results/'scale_reporting.json').write_text(json.dumps(report))
        return results, report

    def test_scaling_retains_all_and_active_denominators_and_rejects_incomplete_pairs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            results, report = self.write_scaling_fixture(root)
            assembly = Assembler(root, results, root/'figures', root/'logs', root/'versions.json', root/'cost.json')
            text = assembly.scaling()
            for saved in ('87.6543', '12.5', '62.5', '4~608/9~216', '25', '50', '10', '20'):
                self.assertIn(saved, text)
            self.assertIn('all dictionary features', text)
            self.assertIn('catalogue-active', text)
            self.assertEqual(assembly.hashes[str(results/'scale_reporting.json')],
                             hashlib.sha256((results/'scale_reporting.json').read_bytes()).hexdigest())
            report['pairs'][1]['27b_matched']['ontology_catalogue_active'] = .25
            (results/'scale_reporting.json').write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError, 'denominator'):
                assembly.scaling()
            report['pairs'].pop()
            (results/'scale_reporting.json').write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError, 'seven'):
                assembly.scaling()

    def test_residual_table_requires_every_tap_and_never_substitutes_old_sae_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); results = root/'results'; results.mkdir()
            record = {'source_sha256': {'moments.npz': 'b'*64}}
            for model, width, count in [('2b', 2304, 26), ('27b_matched', 4608, 46)]:
                record[model] = {'layers': [{'layer': layer, 'width': width, 'n_gene_positions': 102400,
                    'centered_covariance_pr': 12.3456, 'centered_covariance_pr_fraction_width': 12.3456/width,
                    'top1_variance_share': .123456, 'top10_variance_share': .456789,
                    'rms_max_over_median': 98.7654, 'mean_vector_norm_over_mean_row_norm': .654321,
                    'variance_explained_original': .999999, 'dead_fraction_heldout': None} for layer in range(count)]}
            path = results/'residual_summary.json'; path.write_text(json.dumps(record))
            assembly = Assembler(root, results, root/'figures', root/'logs', root/'versions.json', root/'cost.json')
            text = assembly.residuals()
            self.assertIn('27B L45', text)
            self.assertIn('2B L25', text)
            self.assertIn('12.3456', text)
            self.assertIn('98.7654', text)
            self.assertNotIn('0.999999', text)
            self.assertNotIn('not evaluable', text)
            record['27b_matched']['layers'].pop()
            path.write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError, '(?i)complete.*tap'):
                assembly.residuals()

    def test_dimension_ablation_keeps_distinct_variance_references_and_signed_inactivity_delta(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); results, scale = self.write_scaling_fixture(root)
            comparison = {'experiment': 'E4', 'source_sha256': {'train.json': 'c'*64}, 'layers': []}
            for layer in (24, 38):
                comparison['layers'].append({'layer': layer, 'status': 'compared',
                    'input_projection': {'mode': 'zero_selected_columns', 'original_width': 4608,
                        'removed_dimensions': [2, 3], 'retained_input_dimensions': 4606,
                        'variance_reference': 'projected input after zeroing selected columns'},
                    'training_recipe': {'n_features': 9216, 'k': 32, 'epochs': 6, 'seed': 0},
                    'dead_fraction_heldout_original': .6, 'dead_fraction_heldout_projected': .4,
                    'dead_fraction_heldout_delta': -.2, 'dead_fraction_all_rows_original': 0.,
                    'dead_fraction_all_rows_projected': .0001, 'dead_fraction_all_rows_delta': .0001,
                    'variance_explained_original_input': .987654,
                    'variance_explained_projected_input': .876543})
            scale['optional_e4'] = {'producer_comparison': comparison}
            (results/'scale_reporting.json').write_text(json.dumps(scale))
            (results/'dropmassive_comparison.json').write_text(json.dumps(comparison))
            assembly = Assembler(root, results, root/'figures', root/'logs', root/'versions.json', root/'cost.json')
            text = assembly.dimension_ablation()
            self.assertIn('98.7654', text)
            self.assertIn('87.6543', text)
            self.assertIn('different variance references', text)
            self.assertIn('0.01', text)
            self.assertNotIn('not evaluable', text)
            comparison['layers'][0]['variance_explained_projected_input'] = .1
            (results/'dropmassive_comparison.json').write_text(json.dumps(comparison))
            with self.assertRaisesRegex(ValueError, 'audited.*comparison'):
                assembly.dimension_ablation()

    def test_circuit_stability_separates_all_tested_and_shared_passing_and_checks_source_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); results = root/'results'; results.mkdir()
            edge_comparison = {'n_first': 5, 'n_second': 4, 'n_shared': 3, 'jaccard': .5}
            baseline = results/'circuit_stability.json'
            baseline.write_text(json.dumps({'comparison': edge_comparison}))
            source = {'source_layer': 7, 'source_feature': 11, 'common_eligible': True,
                'n_active_first': 4, 'n_active_second': 8, 'n_first_passing': 5, 'n_second_passing': 4,
                'n_shared_passing': 3, 'passing_jaccard': .5, 'n_target_layers': 2,
                'n_target_features_per_layer': 6, 'all_tested': {'n_effect_pairs': 12, 'pearson': .123456, 'spearman': .234567},
                'shared_passing_only': {'n_effect_pairs': 3, 'pearson': .987654, 'spearman': .876543}}
            record = {'passed': True, 'status': 'complete', 'source_layers': [7], 'n_source_features': 1,
                'n_common_eligible_source_features': 1, 'per_source': [source],
                'pooled_all_tested': source['all_tested'], 'selection_conditioned_shared_passing': {
                    'correlations': source['shared_passing_only'], 'edge_comparison': edge_comparison},
                'source_sha256': {'/workspace/rev1/results/circuit_stability.json': hashlib.sha256(baseline.read_bytes()).hexdigest()}}
            path = results/'circuit_stability_full.json'; path.write_text(json.dumps(record))
            assembly = Assembler(root, results, root/'figures', root/'logs', root/'versions.json', root/'cost.json')
            text = assembly.circuit_stability()
            self.assertIn('All common tested effects & 12 & 0.123456 & 0.234567', text)
            self.assertIn('Shared passing effects only & 3 & 0.987654 & 0.876543', text)
            self.assertIn('including zeros and nonpassing', text)
            self.assertIn('11', text)
            record['pooled_all_tested'] = {**source['all_tested'], 'n_effect_pairs': 99}
            path.write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError, 'tested.*count'):
                assembly.circuit_stability()
            record['pooled_all_tested'] = source['all_tested']; path.write_text(json.dumps(record))
            baseline.write_text('{}')
            with self.assertRaisesRegex(ValueError, 'source hash'):
                assembly.circuit_stability()

    def test_dimensions_require_both_cohorts_common_genes_and_valid_excluded_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); results = root/'results'; results.mkdir()
            representations = ['expression_log1p_CP10k', 'expression_cells_shuffled_within_gene',
                'expression_shape_iid_normal'] + [f'model_L{layer:02d}{suffix}' for layer in (9, 13, 17)
                for suffix in ('', '_shape_iid_normal')]
            payload = {'genes': ['A', 'B', 'C', 'D'], 'cell_indices': [1, 2, 3],
                'rows': [{'representation': name, 'n_genes': 4, 'n_coordinates': 3 if name.startswith('expression') else 5,
                    'participation_ratio': 2.34567, 'twonn_id': 6.54321,
                    'twonn_valid_rows': 3, 'twonn_excluded_rows': 1} for name in representations]}
            (results/'cd34_dimensions.json').write_text(json.dumps(payload))
            assembly = Assembler(root, results, root/'figures', root/'logs', root/'versions.json', root/'cost.json')
            with self.assertRaises(FileNotFoundError):
                assembly.dimensions()
            other = results/'k562_replication_dimensions.json'; other.write_text(json.dumps(payload))
            text = assembly.dimensions()
            self.assertIn('2.34567', text)
            self.assertIn('6.54321', text)
            self.assertIn('Valid / excluded', text)
            self.assertIn('Common genes', text)
            payload['rows'][0]['twonn_excluded_rows'] = 0; other.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, 'valid.*excluded'):
                assembly.dimensions()
            payload['rows'][0]['twonn_excluded_rows'] = 1
            payload['rows'][4]['n_coordinates'] = 6; other.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, 'shape'):
                assembly.dimensions()
            payload['rows'][4]['n_coordinates'] = 5; payload['rows'][1]['n_genes'] = 3
            other.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, 'common.*gene'):
                assembly.dimensions()
            payload['rows'][1]['n_genes'] = 4
            payload['rows'][3]['n_coordinates'] = payload['rows'][4]['n_coordinates'] = None
            other.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, 'coordinate'):
                assembly.dimensions()

    def write_phase_fixture(self, root):
        results = root/'results'
        results.mkdir()
        within = {'schema': 'within-k562-input-ladder-v1', 'n_cells': 44, 'k': 512, 'layer': 21, 'n_boot': 25,
            'pca_dimensions': {name: 50 for name in ('expression_full', 'expression_topk_magnitude',
                'expression_topk_binary', 'expression_topk_rank', 'model')},
            'point': {name: {'circular_r2': 0.123456, 'ci95': [-0.012345, 0.234567]} for name in
                ('expression_full', 'expression_topk_magnitude', 'expression_topk_binary', 'expression_topk_rank', 'model', 'constant', 'random')},
            'contrasts': {name: {'point_delta': -0.045678, 'ci95': [-0.098765, 0.012345], 'interpretation': 'unresolved_difference'}
                for name in ('model_minus_expression_topk_rank', 'model_minus_expression_topk_binary', 'model_minus_expression_full')}}
        transfer = {'schema': 'selection-matched-frozen-transfer-v1', 'n_source': 44, 'n_target': 55,
            'n_shared_genes': 66, 'k': 512, 'ridge_alpha': 1000, 'layer': 21, 'n_boot': 25,
            'point': {name: {'R_diff': 0.345678} for name in ('model', 'expression_full_shared',
                'expression_topk_magnitude_fullpanel', 'expression_topk_rank_fullpanel', 'constant', 'random')},
            'primary_contrast': 'model_minus_expression_topk_rank_fullpanel',
            'contrasts': {'model_minus_expression_topk_rank_fullpanel': {
                'point_delta': -0.001234, 'ci95': [-0.023456, 0.034567], 'interpretation': 'unresolved_difference'}}}
        transfer['point']['diagnostic_shared_first_rank'] = {'R_diff': 0.999999}
        for name, payload in [('within_k562_cellcycle.json', within), ('selection_matched_transfer.json', transfer)]:
            (results/name).write_text(json.dumps(payload))
        return results, transfer

    def test_tex_cells_escape_data_without_changing_values(self):
        self.assertEqual(tex_escape('GO_BP 5% & X#1'), r'GO\_BP 5\% \& X\#1')

    def test_table_keeps_number_groups_and_scientific_exponents_together(self):
        text = table('Saved 5 000 draws', ['Count', 'Interval'], [('3 000', '-0.1 to -1.234e-06')], [.3, .5])
        self.assertIn('5~000', text)
        self.assertIn('3~000', text)
        self.assertIn(r'\mbox{-1.234e-06}', text)
        # A reserved last-page rule cannot spill onto a header-only page.
        self.assertIn(r'\endlastfoot', text)
        self.assertLess(text.index(r'\bottomrule'), text.index('3~000'))
        self.assertTrue(text.rstrip().endswith(r'\end{longtable}'))

    def test_strict_missing_inputs_do_not_create_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            assembly = Assembler(root, root/'results', root/'figures', root/'logs', root/'versions.json', root/'cost.json')
            with self.assertRaisesRegex(ValueError, 'Incomplete supplementary assembly'):
                assembly.write(root/'generated', draft=False)
            self.assertFalse((root/'generated').exists())

    def test_draft_uses_saved_measurements_and_marks_missing_blocks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            results = root/'results'
            results.mkdir()
            environment = {'python': '3.12', 'packages': {'numpy': '2.1'}, 'cuda': '12.8', 'gpu': 'device & name'}
            source = results/'environment.json'
            source.write_text(json.dumps(environment))
            assembly = Assembler(root, results, root/'figures', root/'logs', root/'versions.json', root/'cost.json')
            manifest = assembly.write(root/'generated', draft=True)
            self.assertEqual(manifest['status'], 'draft_incomplete')
            self.assertIn('software', manifest['complete_blocks'])
            self.assertIn('tokenization', manifest['pending_blocks'])
            self.assertIn('device \\& name', (root/'generated/software.tex').read_text())
            self.assertIn(r'\SupplementPending', (root/'generated/tokenization.tex').read_text())
            self.assertEqual(manifest['input_sha256'][str(source)], hashlib.sha256(source.read_bytes()).hexdigest())

    def test_synthetic_figure_and_stale_generator_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            figures = root/'figures'
            figures.mkdir()
            (figures/'Figure_S1.pdf').write_bytes(b'%PDF-test-fixture')
            generator = root/'make_revision_figures.py'
            generator.write_text('# fixture generator\n')
            sidecar = figures/'Figure_S1.provenance.json'
            payload = {'figure': 'S1', 'synthetic_preview': True, 'plotted_values': {'a': 1},
                       'source_sha256': {'fixture.json': '0'*64}, 'generator_sha256': '0'*64}
            sidecar.write_text(json.dumps(payload))
            assembly = Assembler(root, root/'results', figures, root/'logs', root/'versions.json', root/'cost.json')
            with self.assertRaisesRegex(ValueError, 'synthetic'):
                assembly.figure('S1')
            payload['synthetic_preview'] = False
            sidecar.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, 'generator hash'):
                assembly.figure('S1')

    def test_provenance_splits_long_source_lists_without_losing_names(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            assembly = Assembler(root, root/'results', root/'figures', root/'logs', root/'versions.json', root/'cost.json')
            names = [f'stage-name-{index:03d}.complete.json' for index in range(70)]
            assembly.figure = lambda name: {'source_sha256': {name: 'a'*64 for name in names}}
            with patch('assemble_supplement.FIGURE_UNITS', {'S6': ('Wall time', 'Completed executable stages')}):
                text = assembly.provenance()
            for name in names:
                self.assertEqual(text.count(name), 1)
            self.assertGreater(sum(line.startswith('S6 &') for line in text.splitlines()), 1)
            self.assertEqual(text.count('Wall time'), 1)
            self.assertEqual(text.count('Completed executable stages'), 1)

    def test_unknown_billing_does_not_become_a_complete_cost_table(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            accounting = root/'cost.json'
            accounting.write_text(json.dumps({'allocations': [], 'total_cost_usd': None, 'basis': 'billing pending'}))
            assembly = Assembler(root, root/'results', root/'figures', root/'logs', root/'versions.json', accounting)
            with self.assertRaisesRegex(KeyError, 'accounting'):
                assembly.costs()

    def test_costs_keep_reported_charges_separate_from_lifetime_estimate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            accounting = root/'cost.json'
            accounting.write_text(json.dumps({'allocations': [{'gpu': 'GPU', 'hourly_usd': 2,
                'billed_seconds': 1800, 'cost_usd': 1.125}], 'total_cost_usd': 1.125,
                'conservative_combined_lifetime_estimate_usd': 1.875,
                'basis': 'Reported charges as of a saved timestamp; billing lag remains.'}))
            assembly = Assembler(root, root/'results', root/'figures', root/'logs', root/'versions.json', accounting)
            text = assembly.costs()
            self.assertIn('Reported total, all allocations & -- & -- & 1.125', text)
            self.assertIn('Conservative full-lifetime estimate & -- & -- & 1.875', text)
            self.assertIn('GPU-billed hours', text)
            self.assertIn('GPU & 2 & 0.5 & 1.125', text)

    def test_phase_tables_preserve_saved_units_intervals_and_primary_arms(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            results, transfer = self.write_phase_fixture(root)
            assembly = Assembler(root, results, root/'figures', root/'logs', root/'versions.json', root/'cost.json')
            text = assembly.phase_probes()
            for saved in ('0.123456', '-0.012345 to 0.234567', '-0.045678', '-0.098765 to 0.012345',
                          '0.345678', '-0.001234', '-0.023456 to 0.034567'):
                self.assertIn(saved, text)
            self.assertIn('circular R-squared', text)
            self.assertIn('R-diff', text)
            self.assertNotIn('diagnostic', text)
            self.assertNotIn('0.999999', text)
            self.assertEqual(len(assembly.hashes), 2)
            transfer['primary_contrast'] = 'model_minus_diagnostic_shared_first_rank'
            (results/'selection_matched_transfer.json').write_text(json.dumps(transfer))
            with self.assertRaisesRegex(ValueError, 'full-panel'):
                assembly.phase_probes()

    def test_sampled_cache_and_guide_coverage_use_saved_counts_and_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            results, _ = self.write_phase_fixture(root)
            for line in ('k562', 'rpe1'):
                (results/f'cache_validation_{line}.json').write_text(json.dumps({
                    'n_cached_rows': 3000, 'n_sampled': 20, 'n_passed': 20, 'all_sampled_rows_pass': True,
                    'criteria': {'minimum_cosine': 0.9999, 'maximum_relative_l2': 0.02,
                                 'expected_row_must_be_nearest_or_tied': True}}))
            (results/'audits').mkdir()
            guide_file = results/'audits/phase8_independent_checks.json'
            guide_file.write_text(json.dumps({'designs': {'k562_original48': {'n_tfs': 48,
                'n_tf_groups_with_single_guide': 43, 'n_tf_groups_with_two_guides': 5,
                'n_cells_total': 1900, 'n_controls': 400, 'control_guide_count': 93}}}))
            assembly = Assembler(root, results, root/'figures', root/'logs', root/'versions.json', root/'cost.json')
            cache_text = assembly.sampled_cache()
            self.assertIn('3~000', cache_text)
            self.assertIn('20', cache_text)
            self.assertIn('unsampled', cache_text)
            self.assertIn('0.9999', cache_text)
            guide_text = assembly.guide_coverage()
            self.assertIn('43', guide_text)
            self.assertIn('48', guide_text)
            self.assertIn('guide replication', guide_text)
            manifest = assembly.write(root/'generated', draft=True)
            self.assertTrue({'phase_probes', 'sampled_cache', 'guide_coverage'} <= set(manifest['complete_blocks']))
            self.assertEqual(manifest['input_sha256'][str(guide_file)], hashlib.sha256(guide_file.read_bytes()).hexdigest())
            (results/'cache_validation_rpe1.json').unlink()
            with self.assertRaises(FileNotFoundError):
                assembly.sampled_cache()

    def test_regulatory_tables_keep_both_ap_references_and_saved_auroc(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            results = root/'results'
            retrieval = {'n_tfs_evaluable': 3, 'mean_average_precision': 0.123456,
                'mean_average_precision_ci95': [0.111111, 0.222222], 'mean_prevalence': 0.012345,
                'mean_random_label_expected_ap': 0.034567, 'mean_auroc': 0.678901,
                'mean_auroc_ci95': [0.56789, 0.789012]}
            family = {'representation': 'model', 'layer': 16, 'n_specific': 1, 'n_tfs': 5,
                      'gene_retrieval': retrieval, 'feature_retrieval': retrieval}
            for line in ('k562', 'rpe1', 'thp1_pooled'):
                directory = results/f'phase8_{line}/statistics'
                directory.mkdir(parents=True)
                (directory/'regulatory_statistics.json').write_text(json.dumps({'summary': [family]}))
            assembly = Assembler(root, results, root/'figures', root/'logs', root/'versions.json', root/'cost.json')
            text = assembly.regulatory()
            for saved in ('0.123456', '0.012345', '0.034567', '0.678901', '0.56789 to 0.789012', '1/5'):
                self.assertIn(saved, text)
            self.assertIn('Target prevalence', text)
            self.assertIn('Exact random-label expected AP', text)
            self.assertIn('AUROC', text)
            self.assertIn('THP1 (pooled guides)', text)
            thp1_path = results/'phase8_thp1_pooled/statistics/regulatory_statistics.json'
            thp1_path.unlink()
            with self.assertRaises(FileNotFoundError):
                assembly.regulatory()
            thp1_path.write_text(json.dumps({'summary': [family]}))
            del retrieval['mean_random_label_expected_ap']
            (results/'phase8_k562/statistics/regulatory_statistics.json').write_text(json.dumps({'summary': [family]}))
            with self.assertRaisesRegex(KeyError, 'mean_random_label_expected_ap'):
                assembly.regulatory()

    def test_reconstruction_requires_the_activated_thp1_cohort(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            results = root/'results'
            for line in ('k562', 'rpe1', 'thp1_pooled'):
                path = results/f'phase8_{line}/meta.json'
                path.parent.mkdir(parents=True)
                path.write_text(json.dumps({'n_cells': 42, 'reconstruction_by_group_layer': [
                    {'layer': 8, 'variance_explained': 0.8, 'dead_fraction': 0.1},
                    {'layer': 8, 'variance_explained': 0.9, 'dead_fraction': 0.2}]}))
            assembly = Assembler(root, results, root/'figures', root/'logs', root/'versions.json', root/'cost.json')
            text = assembly.reconstruction()
            self.assertIn('THP1 (pooled guides)', text)
            self.assertIn('0.8 to 0.9', text)
            (results/'phase8_thp1_pooled/meta.json').unlink()
            with self.assertRaises(FileNotFoundError):
                assembly.reconstruction()

    def test_text_deletion_table_requires_both_lines_and_exact_saved_source_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            results = root/'results'
            report = {'schema': 'text-name-deletion-report-v1', 'status': 'complete', 'bootstrap': 5000,
                      'source_sha256': {}, 'lines': {}}
            for line in ('k562', 'rpe1'):
                relative = f'phase8_{line}/statistics/insilico_retrieval.json'
                source = results/relative
                source.parent.mkdir(parents=True)
                source.write_text(json.dumps({'meta': {'tfs': ['TF'], 'n_pairs': 34}}))
                report['source_sha256'][relative] = hashlib.sha256(source.read_bytes()).hexdigest()
                report['lines'][line] = {'n_screened_named_tfs': 1, 'n_pairs': 34, 'n_named_gene_universe': 99,
                    'max_pre_deletion_cosdist_mean': 0.000012345, 'max_pre_deletion_cosdist_max': 0.00006789,
                    'noise_check_layer': 8, 'layers': [{'layer': 16, 'n_tfs_screened': 1, 'n_tfs_evaluable': 1,
                        'mean_average_precision': 0.123456, 'mean_average_precision_ci95': [0.111111, 0.222222],
                        'mean_auroc': 0.654321, 'mean_auroc_ci95': [0.555555, 0.777777],
                        'mean_prevalence': 0.012345, 'mean_prevalence_ci95': [0.001234, 0.023456],
                        'mean_random_label_expected_ap': 0.034567,
                        'mean_random_label_expected_ap_ci95': [0.022222, 0.044444],
                        'evaluable_downstream_genes_min': 10, 'evaluable_downstream_genes_max': 98}]}
            destination = results/'text_deletion_reporting.json'
            destination.write_text(json.dumps(report))
            assembly = Assembler(root, results, root/'figures', root/'logs', root/'versions.json', root/'cost.json')
            text = assembly.text_deletion()
            for value in ('K562', 'RPE1', '0.123456', '0.654321', '0.034567', '0.111111 to 0.222222'):
                self.assertIn(value, text)
            self.assertIn('text', text)
            self.assertIn('subword', text)
            self.assertEqual(len(assembly.hashes), 3)
            (results/'phase8_rpe1/statistics/insilico_retrieval.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'source hash'):
                assembly.text_deletion()

    def test_surface_form_table_uses_saved_intervals_and_finite_ap_reference(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            results = root/'results'
            results.mkdir()
            forms = {name: {'n': 87, 'n_positive': 50, 'average_precision': 0.123456,
                'average_precision_ci95': [0.111111, 0.222222], 'auroc': 0.654321,
                'auroc_ci95': [0.555555, 0.777777], 'prevalence': 0.012345,
                'random_label_expected_ap': 0.034567} for name in ('human','nearmiss','lower','mouse','anagram')}
            source = results/'surface_form_retrieval.json'
            source.write_text(json.dumps({'forms': forms, 'positive_class': 'G2/M'}))
            assembly = Assembler(root, results, root/'figures', root/'logs', root/'versions.json', root/'cost.json')
            text = assembly.surface_forms()
            for value in ('0.123456', '0.654321', '0.034567', '0.111111 to 0.222222', '87', '50'):
                self.assertIn(value, text)
            del forms['mouse']['auroc_ci95']
            source.write_text(json.dumps({'forms': forms, 'positive_class': 'G2/M'}))
            with self.assertRaises(KeyError):
                assembly.surface_forms()


if __name__ == '__main__':
    unittest.main()
