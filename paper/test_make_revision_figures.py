"""Focused semantic tests for the saved circuit graph count panel."""
import unittest
from unittest.mock import MagicMock
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from make_revision_figures import circuit_edge_profile, Figures, required_paths


class CircuitEdgeProfileTests(unittest.TestCase):
    def graph(self):
        pairs = [(0, 2), (0, 2), (0, 8), (7, 8), (7, 12), (16, 17), (16, 17)]
        return {'n_edges':len(pairs), 'edges':[
            {'source_layer':source, 'source_feature':index, 'target_layer':target, 'target_feature':100+index}
            for index,(source,target) in enumerate(pairs)]}

    def test_source_target_counts_sum_to_saved_total_and_retain_zero_layers(self):
        profile = circuit_edge_profile(self.graph())
        self.assertEqual(profile['n_edges'],7)
        self.assertEqual(profile['target_layers'],list(range(26)))
        self.assertEqual(profile['counts_by_source_layer']['0'][2],2)
        self.assertEqual(profile['counts_by_source_layer']['0'][8],1)
        self.assertEqual(profile['counts_by_source_layer']['7'][8],1)
        self.assertEqual(profile['counts_by_source_layer']['7'][12],1)
        self.assertEqual(profile['counts_by_source_layer']['16'][17],2)
        self.assertEqual(profile['counts_by_source_layer']['7'][7],0)
        self.assertEqual(sum(sum(row) for row in profile['counts_by_source_layer'].values()),profile['n_edges'])
        self.assertEqual(profile['total_by_source_layer'],{'0':3,'7':2,'16':2})
        self.assertEqual(profile['storage_cap_per_source_feature_per_target_layer'],50)

    def test_empty_misaligned_duplicate_and_noncausal_graphs_fail(self):
        for broken in ({'n_edges':0,'edges':[]}, {'n_edges':8, 'edges':self.graph()['edges']},
                       {'n_edges':1,'edges':[{'source_layer':7,'source_feature':0,'target_layer':3,'target_feature':1}]},
                       {'n_edges':2,'edges':[self.graph()['edges'][0]]*2},
                       {'edges':self.graph()['edges']}):
            with self.subTest(graph=broken):
                with self.assertRaises(ValueError):circuit_edge_profile(broken)


    def test_missing_graph_or_empty_statistics_cannot_write_figure4(self):
        with TemporaryDirectory() as temporary:
            base = Path(temporary)
            args = SimpleNamespace(results_dir=base/'results', runs_dir=base/'runs',
                                   route_dir=base/'route', geometry_dir=base/'geometry',
                                   synthetic_preview=None, out_dir=base/'final')
            graph_path = args.runs_dir/'2b_atlas/circuit_graph.json'
            self.assertIn(graph_path, required_paths(args,'4'))
            with self.assertRaises(FileNotFoundError):Figures(args).figure4()
            graph_path.parent.mkdir(parents=True)
            graph_path.write_text(json.dumps(self.graph()))
            args.results_dir.mkdir()
            (args.results_dir/'circuit_baseline.json').write_text('{}')
            for line in ('k562','rpe1'):
                statistics = args.results_dir/f'phase8_{line}/statistics'
                statistics.mkdir(parents=True)
                (statistics/'regulatory_statistics.json').write_text('{"summary":[]}')
            with self.assertRaises(ValueError):Figures(args).figure4()
            self.assertFalse(args.out_dir.exists())


class RetrievalReferenceTests(unittest.TestCase):
    def test_ap_panel_shows_finite_random_expectation_separately_from_prevalence(self):
        axis = MagicMock()
        figures = Figures(SimpleNamespace(results_dir=None, runs_dir=None, route_dir=None, geometry_dir=None))
        records = [{'representation':'model', 'gene_retrieval':{
            'mean_average_precision':.02, 'mean_average_precision_ci95':[.01,.03],
            'mean_prevalence':.001, 'mean_random_label_expected_ap':.005}}]
        figures.retrieval_panel(axis, records, 'gene_retrieval', 'average_precision',
                                ['Model'], 'a', 'Retrieval')
        plotted = {call.kwargs.get('label'):call.args[1] for call in axis.plot.call_args_list
                   if call.kwargs.get('label')}
        self.assertEqual(plotted['Target prevalence'], [.001])
        self.assertEqual(plotted['Finite random-label AP'], [.005])


if __name__=='__main__':unittest.main()
