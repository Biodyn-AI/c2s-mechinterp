import unittest

from audit_scale import paired_contrast


class ScaleReportingAuditTests(unittest.TestCase):
    def test_paired_contrast_keeps_denominators_and_variance_references_explicit(self):
        a = {'layer': 0, 'n_features': 4, 'ontology_all_features': .5, 'ontology_frequency_floor_all_features': .1,
             'heldout_inactive_fraction': .5, 'all_row_inactive_fraction': .25, 'heldout_variance_explained': .9}
        b = {**a, 'layer': 7, 'ontology_all_features': .25, 'ontology_frequency_floor_all_features': .2,
             'heldout_inactive_fraction': .75, 'all_row_inactive_fraction': .5}
        result = paired_contrast(a, b)
        self.assertAlmostEqual(result['ontology_excess_gap_27b_minus_2b'], -.35)
        self.assertEqual(result['all_row_inactive_gap_27b_minus_2b'], .25)
        self.assertEqual(result['heldout_inactive_gap_27b_minus_2b'], .25)
        self.assertEqual(result['2b'], a)
        with self.assertRaises(ValueError): paired_contrast(a, {**b, 'n_features': 8})


if __name__ == '__main__': unittest.main()
