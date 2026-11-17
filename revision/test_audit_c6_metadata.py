import unittest

from audit_c6_metadata import guide_target, collision_is_identity_merge


class C6MetadataAuditTests(unittest.TestCase):
    def test_only_terminal_guide_suffix_is_removed(self):
        self.assertEqual(guide_target('STAT5Ag2'), 'STAT5A')
        self.assertEqual(guide_target('NTg8'), 'control')
        for ambiguous in ['STAT1', 'STAT1g1+STAT2g2', 'STAT1g0']:
            with self.assertRaises(ValueError): guide_target(ambiguous)

    def test_case_match_alone_does_not_justify_merging_distinct_genes(self):
        collision = {'source_symbols': ['C2orf15', 'C2ORF15'], 'uppercase_symbol': 'C2ORF15',
                     'ensembl_ids': ['ENSG00000273045', 'ENSG00000273045'], 'action': 'sum_same_ensembl'}
        self.assertTrue(collision_is_identity_merge(collision))
        for ids in [['ENSG1', 'ENSG2'], ['', ''], ['nan', 'nan']]:
            self.assertFalse(collision_is_identity_merge({**collision, 'ensembl_ids': ids}))


if __name__ == '__main__': unittest.main()
