import unittest

from audit_guides_saved import pair_values


class GuideSavedAuditTests(unittest.TestCase):
    def test_empty_response_union_is_undefined_not_perfect_agreement(self):
        result = pair_values([1., -.5], [-1., .5], [False, False], [False, False])
        self.assertIsNone(result['response_jaccard'])
        self.assertIsNone(result['joint_response_sign_agreement'])
        self.assertEqual(result['n_both_abs_effect_above_threshold'], 1)
        self.assertEqual(result['effect_supported_sign_agreement'], 0.)

    def test_joint_and_union_counts_match_explicit_sets(self):
        result = pair_values([1., -1., 2.], [2., 1., 3.], [True, True, False], [True, False, True])
        self.assertEqual(result['n_joint_responses'], 1)
        self.assertEqual(result['n_union_responses'], 3)
        self.assertAlmostEqual(result['response_jaccard'], 1/3)
        self.assertEqual(result['joint_response_sign_agreement'], 1.)


if __name__ == '__main__': unittest.main()
