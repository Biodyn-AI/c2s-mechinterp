"""Exercise the public annotation CLI against tiny, fully specified gene libraries."""
import gzip
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


REPOSITORY = Path(__file__).resolve().parents[1]


class AnnotationEntrypointTests(unittest.TestCase):
    def fixture(self, root, versioned=False):
        run, ontology = root / 'run', root / 'ontology'
        run.mkdir()
        ontology.mkdir()
        genes = [f'G{i}' for i in range(30)]
        (run / 'row_gene_names.json').write_text(json.dumps(genes + genes[:5]))
        (run / 'feature_catalog_L00.json').write_text(json.dumps({'features': [
            {'feature_idx': 0, 'is_dead': False, 'top_genes': [{'gene_name': gene} for gene in genes[1:4]]}]}))
        libraries = {'go_bp': {'strong': genes[1:6], 'zero overlap': genes[10:15]},
                     'kegg': {'pathway': genes[1:6]}, 'reactome': {'process': genes[1:6]}}
        for source, content in libraries.items():
            filename = source + ('.json' if versioned else '_gene_sets.json')
            (ontology / filename).write_text(json.dumps(content))
        if versioned:
            with gzip.open(ontology / 'string_v12.5_score700_pairs.tsv.gz', 'wt') as handle:
                handle.write('G1\tG2\n')
        else:
            (ontology / 'string_ppi_edges.json').write_text(json.dumps({'pairs_700': [['G1', 'G2']]}))
        (ontology / 'trrust_human.tsv').write_text(''.join(f'G0\t{gene}\tActivation\t1\n' for gene in genes[1:6]))
        return run, ontology

    def invoke(self, run, ontology, *extra):
        return subprocess.run([sys.executable, '-m', 'c2s_mechinterp.annotate_features',
            '--run', str(run), '--layer', '0', '--ontology-dir', str(ontology), *extra],
            cwd=REPOSITORY, capture_output=True, text=True)

    def test_default_entrypoint_uses_full_universe_family_and_floors(self):
        with tempfile.TemporaryDirectory() as temporary:
            run, ontology = self.fixture(Path(temporary))
            completed = self.invoke(run, ontology)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            output = json.loads((run / 'feature_annotations_L00.json').read_text())
            self.assertEqual(output['summary']['scorer_version'], '3-plan-aligned')
            self.assertEqual(output['summary']['universe_size'], 30)
            # All four ontology terms, one TRRUST regulon and three network/membership tests.
            self.assertEqual(output['summary']['tests_per_feature'], 8)
            self.assertEqual(output['metadata']['library_layout'], 'snapshot')
            self.assertIn('uniform', output['summary']['floors'])
            self.assertIn('freq', output['summary']['floors'])
            self.assertTrue(output['features'][0]['significant_term_ids'])

    def test_versioned_layout_keeps_network_evidence_out_of_ontology_compatibility(self):
        with tempfile.TemporaryDirectory() as temporary:
            run, ontology = self.fixture(Path(temporary), versioned=True)
            completed = self.invoke(run, ontology, '--library-layout', 'versioned', '--n-floor-seeds', '1')
            self.assertEqual(completed.returncode, 0, completed.stderr)
            output = json.loads((run / 'feature_annotations_L00.json').read_text())
            self.assertEqual(output['metadata']['library_layout'], 'versioned')
            self.assertTrue(output['network_evidence']['0'])
            self.assertTrue(all(row['ontology'] in {'GO_BP', 'KEGG', 'Reactome', 'TRRUST'}
                                for row in output['feature_annotations']['0']))
            self.assertTrue(all('fold_enrichment' in row and 'odds_ratio' not in row
                                for row in output['feature_annotations']['0']))

    def test_missing_full_universe_is_an_error_instead_of_top_gene_fallback(self):
        with tempfile.TemporaryDirectory() as temporary:
            run, ontology = self.fixture(Path(temporary))
            (run / 'row_gene_names.json').unlink()
            completed = self.invoke(run, ontology)
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn('universe', completed.stderr.lower())
            self.assertFalse((run / 'feature_annotations_L00.json').exists())

    def test_ambiguous_layout_requires_an_explicit_choice(self):
        with tempfile.TemporaryDirectory() as temporary:
            run, ontology = self.fixture(Path(temporary))
            for source in ('go_bp', 'kegg', 'reactome'):
                (ontology / f'{source}.json').write_text((ontology / f'{source}_gene_sets.json').read_text())
            with gzip.open(ontology / 'string_v12.5_pairs.tsv.gz', 'wt') as handle:
                handle.write('G1\tG2\n')
            completed = self.invoke(run, ontology)
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn('ambiguous', completed.stderr.lower())
            self.assertFalse((run / 'feature_annotations_L00.json').exists())


if __name__ == '__main__':
    unittest.main()
