"""Small synthetic checks for revision statistics; no model or dataset required."""
import unittest

import numpy as np

from rev_cpu_analysis import (
    annotation_terms, category_for_term, circuit_baseline, cluster_bootstrap,
    dimension_metrics, edge_mask, persistence_categories, compare_edges, best_decoder_matches,
)


class RevisionStatisticsTests(unittest.TestCase):
    def test_only_significant_real_terms_are_shared(self):
        row = {"terms": [{"source": "GO_BP", "term": "mitosis", "q": 0.01},
                         {"source": "GO_BP", "term": "repair", "q": 0.2},
                         {"source": "STRING", "term": "interaction enrichment", "q": 0.001}]}
        self.assertEqual(annotation_terms(row), {("GO_BP", "mitosis")})

    def test_complete_term_ids_take_precedence_over_display_cap(self):
        row = {"significant_term_ids": ["GO_BP::mitosis", "Reactome::DNA repair", "TRRUST::TP53 targets", "STRING::interaction enrichment"],
               "terms": [{"source": "GO_BP", "term": "mitosis", "q": 0.01}]}
        self.assertEqual(annotation_terms(row), {("GO_BP", "mitosis"), ("Reactome", "DNA repair"), ("TRRUST", "TP53 targets")})

    def test_layer_matched_baseline_is_exact_and_keeps_sources(self):
        terms = {0: {0: {("GO_BP", "a")}, 1: {("GO_BP", "b")}},
                 1: {0: {("GO_BP", "a")}, 1: {("GO_BP", "a")}, 2: {("GO_BP", "b")}}}
        edges = [{"source_layer": 0, "source_feature": 0, "target_layer": 1, "target_feature": 0},
                 {"source_layer": 0, "source_feature": 1, "target_layer": 1, "target_feature": 2}]
        summary = circuit_baseline(edges, terms, n_boot=100, seed=0)
        self.assertEqual(summary["n_annotated_edges"], 2)
        self.assertEqual(summary["observed"], 1.0)
        self.assertAlmostEqual(summary["same_source_random_target"], 0.5)
        self.assertAlmostEqual(summary["fully_random_annotated_pair"], 0.5)
        self.assertAlmostEqual(summary["excess"], 0.5)

    def test_bootstrap_resamples_source_clusters_not_individual_edges(self):
        # Clusters have unequal edge counts, so a bootstrap of edge proportions is wrong.
        out = cluster_bootstrap(np.array([[9, 9, 0], [0, 1, 0.0]]), n_boot=3000, seed=1)
        self.assertAlmostEqual(out["observed"]["estimate"], 0.9)
        self.assertEqual(out["observed"]["ci95"], [0.0, 1.0])

    def test_edge_rule_uses_strict_thresholds_and_actual_activity(self):
        self.assertEqual(edge_mask(np.array([0.5, 0.6, 0.7]), np.array([1, 0.7, 0.8]), 1).sum(), 0)
        np.testing.assert_array_equal(edge_mask(np.array([0.5, -0.6, 0.7]), np.array([1, 0.8, 0.8]), 50), [False, True, True])

    def test_participation_ratio_distinguishes_rank_one_and_equal_axes(self):
        rng = np.random.default_rng(2)
        rank_one = rng.normal(size=(400, 1)) @ np.array([[1, 2, 3]])
        self.assertAlmostEqual(dimension_metrics(rank_one)["participation_ratio"], 1.0, places=7)
        equal = np.vstack([np.eye(3), -np.eye(3)])
        self.assertAlmostEqual(dimension_metrics(equal)["participation_ratio"], 3.0, places=7)

    def test_twonn_detects_one_dimensional_random_curve(self):
        rng = np.random.default_rng(2)
        t = rng.uniform(0, 2 * np.pi, 2000)
        estimate = dimension_metrics(np.column_stack([np.cos(t), np.sin(t)]))["twonn_id"]
        self.assertTrue(0.85 < estimate < 1.2)

    def test_categories_are_explicit_and_deterministic(self):
        self.assertEqual(category_for_term("Erythrocyte differentiation"), "haem and erythroid")
        self.assertEqual(category_for_term("mitotic spindle assembly"), "cell cycle and mitosis")
        self.assertEqual(category_for_term("unrecognised pathway"), "other")

    def test_persistence_compares_matched_feature_ids(self):
        first = {10: {"cell cycle and mitosis"}, 11: {"RNA processing"}}
        last = {21: {"cell cycle and mitosis"}, 22: {"RNA processing"}}
        out = persistence_categories(first, last, np.array([10, 11]), np.array([21, 22]), np.ones(2), 0.7, 100, 0)
        self.assertEqual(out["observed_same_category"], 1.0)
        self.assertAlmostEqual(out["random_target_baseline"], 0.5)

    def test_edge_overlap_is_compared_on_layer_and_feature_ids(self):
        first = [{"source_layer": 0, "source_feature": 4, "target_layer": 1,
                  "target_feature": target, "cohens_d": float(target)} for target in (1, 2)]
        second = [{"source_layer": 0, "source_feature": 4, "target_layer": 1,
                   "target_feature": target, "cohens_d": float(target * 2)} for target in (2, 3)]
        summary = compare_edges(first, second)
        self.assertEqual(summary["n_shared"], 1)
        self.assertAlmostEqual(summary["jaccard"], 1 / 3)
        self.assertEqual(summary["fraction_first_recovered"], 0.5)

    def test_decoder_matches_keep_argmax_and_absolute_cosine(self):
        target, cosine = best_decoder_matches(np.eye(3), -np.eye(3), block_size=1)
        np.testing.assert_array_equal(target, [0, 1, 2])
        np.testing.assert_allclose(cosine, 1.0)


if __name__ == "__main__":
    unittest.main()
