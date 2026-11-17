"""Small fixtures for safe collection; never access a pod or real run outputs."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location("collector", Path(__file__).with_name("collect_completed_artifacts.py"))
collector = importlib.util.module_from_spec(SPEC)
if SPEC.loader:
    SPEC.loader.exec_module(collector)


class CollectionTests(unittest.TestCase):
    def test_stage_requires_successful_latest_attempt_and_existing_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'stage_logs').mkdir()
            output = root/'results/output.json'
            output.parent.mkdir()
            output.write_text('{}')
            record = {'name': 'done', 'returncode': 0, 'fingerprint': 'old',
                'start_utc': '2026-11-17T10:00:00+00:00', 'end_utc': '2026-11-17T10:01:00+00:00',
                'outputs': [str(output)]}
            (root/'stage_logs/done.complete.json').write_text(json.dumps(record))
            self.assertTrue(collector.stage_ready(root, 'done'))
            (root/'stage_logs/done.attempt.json').write_text(json.dumps({**record,
                'fingerprint': 'new', 'start_utc': '2026-11-17T11:00:00+00:00'}))
            self.assertFalse(collector.stage_ready(root, 'done'))

    def test_prior_dedup_is_by_hash_and_forbidden_files_never_enter(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            good = root/'results/summary.json'
            good.parent.mkdir()
            good.write_text('{"new":1}')
            prior = {'results/summary.json': {'sha256': hashlib.sha256(good.read_bytes()).hexdigest()}}
            selected, repeated = collector.snapshot_files(root, [good], prior)
            self.assertFalse(selected)
            self.assertEqual(repeated, ['results/summary.json'])
            good.write_text('{"new":2}')
            selected, repeated = collector.snapshot_files(root, [good], prior)
            self.assertIn('results/summary.json', selected)
            self.assertFalse(repeated)
            for name in ('counts.h5ad', 'layer_13_activations.npy', 'model.safetensors', 'credentials.json'):
                path = root/name
                path.write_bytes(b'not collected')
                with self.subTest(name=name), self.assertRaises(ValueError):
                    collector.snapshot_files(root, [path], {})

    def test_asset_partition_limit_and_archive_roundtrip(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = []
            for index in range(3):
                path = root/f'file{index}.json'
                path.write_text(json.dumps({'index': index, 'value': 'x'*100}))
                paths.append(path)
            files, _ = collector.snapshot_files(root, paths, {})
            parts = collector.partition_files(files, max_payload_bytes=260)
            self.assertEqual(len(parts), 2)
            out = root/'package.tar.gz'
            descriptor = collector.write_archive(root, files, out, 'scientific', {})
            verified = collector.verify_archive(out, descriptor['sha256'])
            self.assertEqual(verified['n_files'], 3)
            self.assertEqual(verified['bytes'], sum(p.stat().st_size for p in paths))

    def test_archive_refuses_a_file_changed_after_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root/'saved.json'
            path.write_text('{"before":1}')
            files, _ = collector.snapshot_files(root, [path], {})
            path.write_text('{"after":123456}')
            with self.assertRaisesRegex(ValueError, 'changed'):
                collector.write_archive(root, files, root/'package.tar.gz', 'scientific', {})
            self.assertFalse((root/'package.tar.gz').exists())

    def test_weights_and_symlinks_are_never_in_scientific_payload(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            weights = root/'runs/sweep/sae/sae_L00.pt'
            weights.parent.mkdir(parents=True)
            weights.write_bytes(b'weights')
            with self.assertRaisesRegex(ValueError, 'weights'):
                collector.snapshot_files(root, [weights], {}, category='scientific')
            selected, _ = collector.snapshot_files(root, [weights], {}, category='local_only_weights')
            self.assertEqual(len(selected), 1)
            link = root/'alias.json'
            link.symlink_to(weights)
            with self.assertRaisesRegex(ValueError, 'symlink'):
                collector.snapshot_files(root, [link], {})

    def test_scientific_install_preserves_existing_files_and_routes_conflicts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)/'source'
            (root/'results').mkdir(parents=True)
            source = root/'results/value.json'
            source.write_text('{"saved":1}')
            files, _ = collector.snapshot_files(root, [source], {})
            archive = Path(directory)/'science.tar.gz'
            asset = collector.write_archive(root, files, archive, 'scientific', {})
            destination = Path(directory)/'results'
            destination.mkdir()
            existing = destination/'value.json'
            existing.write_text('{"newer_local":2}')
            conflicts = Path(directory)/'conflicts'
            result = collector.install_scientific(archive, asset['sha256'], {'results': destination}, conflicts)
            self.assertEqual(json.loads(existing.read_text()), {'newer_local': 2})
            self.assertEqual((conflicts/'results/value.json').read_bytes(), source.read_bytes())
            self.assertEqual(len(result['conflicts']), 1)
            self.assertEqual(result['n_verified'], 1)

    def test_local_only_weight_archive_cannot_be_installed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)/'source'
            root.mkdir()
            source = root/'sae_L00.pt'
            source.write_bytes(b'weights')
            files, _ = collector.snapshot_files(root, [source], {}, category='local_only_weights')
            archive = Path(directory)/'weights.tar.gz'
            asset = collector.write_archive(root, files, archive, 'local_only_weights', {})
            destination = Path(directory)/'destination'
            with self.assertRaisesRegex(ValueError, 'local-only'):
                collector.install_scientific(archive, asset['sha256'], {'runs': destination}, Path(directory)/'conflicts')
            self.assertFalse(destination.exists())

    def test_annotation_priority_scope_requires_all_seven_completed_layers(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            (root/'stage_logs').mkdir()
            ann = root/'results/annotations/27b_matched'
            ann.mkdir(parents=True)
            for layer in (0,7,14,24,31,38,45):
                output = ann/f'ann_v2_L{layer:02d}.json'
                output.write_text('{}')
                stage = f'scale-annotate-L{layer:02d}'
                (root/'stage_logs'/f'{stage}.complete.json').write_text(json.dumps({'name':stage,
                    'returncode':0,'fingerprint':'saved','start_utc':'2026-11-17T10:00:00Z',
                    'end_utc':'2026-11-17T10:01:00Z','outputs':[str(output)]}))
            plan = collector.collect_plan(root, {}, scope='annotations')
            self.assertFalse(plan['pending_groups'])
            self.assertEqual(len([name for name in plan['scientific'] if '/annotations/' in name]),7)
            self.assertFalse(plan['local_only_weights'])
            (root/'stage_logs/scale-annotate-L45.complete.json').unlink()
            plan = collector.collect_plan(root, {}, scope='annotations')
            self.assertIn('matched_scale_annotations',plan['pending_groups'])
            self.assertFalse(plan['scientific'])

    def test_final_scope_stays_pending_without_geometry_and_rejects_unfinished_audits(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            plan = collector.collect_plan(root, {}, scope='final')
            self.assertTrue({'geometry_cd34','geometry_k562_replication'} <= set(plan['pending_groups']))
            path = root/'audit.json'
            path.write_text(json.dumps({'passed':True,'pending_files':['still_running']}))
            self.assertFalse(collector.audit_ready(path))
            path.write_text(json.dumps({'passed':True,'pending_files':[],'findings':[]}))
            self.assertTrue(collector.audit_ready(path))
            plan = collector.collect_plan(root, {}, scope='final', required_audits=['audit.json'])
            self.assertIn('audit.json',plan['scientific'])
            editorial = root/'atlas_v4_roundtrip.json'
            editorial.write_text(path.read_text())
            self.assertFalse(collector.science_audit_ready(editorial))
            science = root/'annotation_2b_L00.json'
            science.write_text(path.read_text())
            self.assertTrue(collector.science_audit_ready(science))
            for payload in ({'all_checks_passed':True},{'all_arithmetic_checks_passed':True,'findings':[]},
                            {'checks':{'first':True,'second':True}}):
                science.write_text(json.dumps(payload))
                self.assertTrue(collector.science_audit_ready(science))
            science.write_text(json.dumps({'passed':False,'all_checks_passed':True}))
            self.assertFalse(collector.science_audit_ready(science))


if __name__ == '__main__':
    unittest.main()
