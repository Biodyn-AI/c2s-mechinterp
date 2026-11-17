"""Synthetic checks for the full test family and pair-count annotation tests."""
import unittest

import numpy as np
from scipy.stats import hypergeom, poisson

from annotate_v2 import Scorer


class AnnotationTests(unittest.TestCase):
    def scorer(self, extra_terms=0):
        universe = [f"G{i}" for i in range(100)]
        libraries = {source: {} for source in ["GO_BP", "KEGG", "Reactome", "TRRUST"]}
        libraries["GO_BP"] = {"signal": set(universe[:5])}
        libraries["GO_BP"].update({f"null{i}": set(universe[50:55]) for i in range(extra_terms)})
        pairs = {(universe[i], universe[j]) for i in range(5) for j in range(i + 1, 5)}
        return Scorer(universe, libraries, pairs, 5, 500, 2, .05,
                      trrust_pairs={("G0", "G1")}, trrust_tfs={"G0", "G1"})

    def test_full_family_counts_zero_overlap_terms(self):
        scorer = self.scorer(extra_terms=99)
        self.assertEqual(scorer.m_tests, 103)
        scored = scorer.score([["G0", "G1", "G2"]], keep_terms=1000)
        term = next(t for t in scored["details"][0] if t["term"] == "signal")
        expected_p = hypergeom.sf(2, 100, 5, 3)
        self.assertAlmostEqual(term["p"], expected_p)
        self.assertGreaterEqual(term["q"], expected_p)

    def test_pair_density_and_tf_membership(self):
        scorer = self.scorer()
        scored = scorer.score([["G0", "G1", "G2"]], keep_terms=1000)
        self.assertEqual(scored["k_string"][0], 3)
        self.assertEqual(scored["k_trrust"][0], 1)
        expected = 3 * 10 / (100 * 99 / 2)
        self.assertAlmostEqual(scored["expected_string"][0], expected)
        self.assertAlmostEqual(scored["p_string"][0], poisson.sf(2, expected))
        self.assertAlmostEqual(scored["p_trrust_pairs"][0], poisson.sf(0, 3 / 4950))

    def test_complete_significant_ids_are_not_display_capped(self):
        scorer = self.scorer()
        scorer.term_name[0] = "signal"
        scored = scorer.score([["G0", "G1", "G2", "G3", "G4"]], keep_terms=1)
        self.assertIn("GO_BP::signal", scored["significant_term_ids"][0])
        self.assertLessEqual(len(scored["details"][0]), 1)

    def test_empty_and_duplicate_sets(self):
        scored = self.scorer().score([[], ["G0", "G0", "outside"]])
        np.testing.assert_array_equal(scored["n"], [0, 1])
        self.assertFalse(scored["annotated"].any())


if __name__ == "__main__":
    unittest.main()
