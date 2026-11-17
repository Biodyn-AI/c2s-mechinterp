"""Small semantic tests for guide aggregation, gene identity and batch selection."""
import unittest
import numpy as np
from scipy import sparse
# Discovery imports this as a package; direct execution keeps the local-script path.
if __package__:
    from .prepare_papalexi import (parse_guide_target, checked_target, gene_collapse_plan,
                                  collapse_counts, sample_across_guides, select_groups)
else:
    from prepare_papalexi import (parse_guide_target, checked_target, gene_collapse_plan,
                                 collapse_counts, sample_across_guides, select_groups)

class PapalexiPreparationTests(unittest.TestCase):
    def test_guide_suffix_is_anchored_and_preserves_target_numbers(self):
        self.assertEqual(parse_guide_target('STAT5Ag2'), 'STAT5A')
        self.assertEqual(parse_guide_target('C2ORF15g12'), 'C2ORF15')
        self.assertEqual(parse_guide_target('NTg10'), 'control')
        for invalid in ('IRF1', 'IRF1g1+STAT1g2', 'g1', 'IRF1g1_extra'):
            with self.assertRaises(ValueError): parse_guide_target(invalid)

    def test_existing_gene_and_guide_labels_must_agree(self):
        self.assertEqual(checked_target('IRF1g2', 'IRF1g2'), 'IRF1')
        self.assertEqual(checked_target('IRF1g2', 'IRF1'), 'IRF1')
        self.assertEqual(checked_target('NTg7', 'control'), 'control')
        with self.assertRaises(ValueError): checked_target('IRF1g2', 'STAT1')
        with self.assertRaises(ValueError): checked_target('NTg7', 'IRF1')

    def test_case_collision_merges_only_verified_equal_ensembl_ids(self):
        plan = gene_collapse_plan(['A', 'C2orf15', 'C2ORF15', 'B'], ['EA', 'E2', 'E2', 'EB'])
        self.assertEqual(plan['symbols'], ['A', 'C2ORF15', 'B'])
        np.testing.assert_array_equal(plan['column_map'], [0, 1, 1, 2])
        self.assertEqual(plan['collisions'][0]['action'], 'sum_same_ensembl')
        values = sparse.csr_matrix([[1, 2, 3, 4], [0, 5, 0, 6]], dtype=np.float32)
        cleaned = collapse_counts(values, plan)
        np.testing.assert_array_equal(cleaned.toarray(), [[1, 5, 4], [0, 5, 6]])
        np.testing.assert_array_equal(np.asarray(cleaned.sum(1)), np.asarray(values.sum(1)))

    def test_ambiguous_collision_drops_every_member_without_choosing_one(self):
        for ids in (['E1', 'E2'], ['E1', '']):
            plan = gene_collapse_plan(['a', 'A', 'Other'], ids + ['E3'])
            self.assertEqual(plan['symbols'], ['Other'])
            np.testing.assert_array_equal(plan['column_map'], [-1, -1, 0])
            self.assertEqual(plan['collisions'][0]['action'], 'drop_ambiguous_identity')
            cleaned = collapse_counts(sparse.csr_matrix([[1, 2, 3]]), plan)
            np.testing.assert_array_equal(cleaned.toarray(), [[3]])

    def test_guide_sampling_preserves_multiple_guides_and_is_reproducible(self):
        guides = np.array(['g1'] * 7 + ['g2'] * 3 + ['g3'])
        first = sample_across_guides(np.arange(11), guides, 8, np.random.default_rng(0))
        second = sample_across_guides(np.arange(11), guides, 8, np.random.default_rng(0))
        np.testing.assert_array_equal(first, second)
        self.assertEqual(len(set(first)), 8)
        self.assertEqual(set(guides[first]), {'g1', 'g2', 'g3'})

    def test_groups_stay_inside_batch_and_pseudo_controls_are_disjoint(self):
        targets = np.array(['control'] * 18 + ['TF'] * 8 + ['OTHER'] * 8 + ['TF'] * 5)
        guides = np.array(['NTg1'] * 9 + ['NTg2'] * 9 + ['TFg1'] * 4 + ['TFg2'] * 4 +
                          ['OTHERg1'] * 8 + ['TFg1'] * 5)
        eligible = np.arange(len(targets)) < 34
        groups = select_groups(targets, guides, eligible, {'TF'}, seed=7, n_control=8,
                               cells_per_target=6, min_cells=5, n_nontf=1,
                               n_pseudo=2, pseudo_cells=4)
        selected = [cell for group in groups for cell in group['source_cells']]
        self.assertEqual(len(selected), len(set(selected)))
        self.assertTrue(all(eligible[selected]))
        tf = next(group for group in groups if group['kind'] == 'tf')
        self.assertEqual(tf['group'], 'TF')
        self.assertEqual(tf['n_guides_selected'], 2)
        self.assertEqual(sum(group['kind'] == 'pseudo' for group in groups), 2)

if __name__ == '__main__': unittest.main()
