import unittest

from audit_circuits_saved import exact_join


class CircuitJoinAuditTests(unittest.TestCase):
    def test_only_named_annotated_endpoints_enter_denominator_and_chance_pool(self):
        pools = {0: {0: {'GO_BP::A'}, 1: {'GO_BP::B'}, 2: set()},
                 1: {0: {'GO_BP::A'}, 1: {'GO_BP::B'}, 2: {'GO_BP::A', 'GO_BP::B'}, 3: set()}}
        edges = [{'source_layer': 0, 'source_feature': source, 'target_layer': 1, 'target_feature': target}
                 for source, target in [(0, 0), (0, 1), (2, 2), (1, 3)]]
        result = exact_join(edges, {0: pools[0]}, lambda layer: pools[layer])
        self.assertEqual(result['n_edges'], 4)
        self.assertEqual(result['n_annotated_edges'], 2)
        self.assertEqual(result['n_shared'], 1)
        self.assertEqual(result['n_source_clusters'], 3)
        self.assertAlmostEqual(result['observed'], .5)
        self.assertAlmostEqual(result['same_source_random_target'], 2/3)
        self.assertAlmostEqual(result['fully_random_annotated_pair'], 2/3)
        self.assertEqual(sum(row['eligible'] == 0 for row in result['source_counts']), 2)


if __name__ == '__main__': unittest.main()
