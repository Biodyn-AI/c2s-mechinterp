"""Check queue resumption and failure handling without running scientific workloads."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

from run_revision import Stage, execute_stage


class RevisionQueueTests(unittest.TestCase):
    def test_output_inside_linked_directory_does_not_create_real_destination(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'atlas/sae'
            source.mkdir(parents=True)
            checkpoint = source / 'model.pt'
            checkpoint.write_bytes(b'original checkpoint')
            source_names = sorted(path.relative_to(source) for path in source.rglob('*'))
            source_mtime = source.stat().st_mtime_ns
            checkpoint_mtime = checkpoint.stat().st_mtime_ns
            destination = root / 'pool_first/sae'
            stage = Stage('linked-output', outputs=[destination / 'model.pt'], links=[(destination, source)])
            self.assertEqual(execute_stage(stage, root), 'completed')
            self.assertTrue(destination.is_symlink())
            self.assertEqual(destination.resolve(), source.resolve())
            self.assertEqual((destination / 'model.pt').read_bytes(), b'original checkpoint')
            self.assertEqual(sorted(path.relative_to(source) for path in source.rglob('*')), source_names)
            self.assertEqual(source.stat().st_mtime_ns, source_mtime)
            self.assertEqual(checkpoint.stat().st_mtime_ns, checkpoint_mtime)
            # A matching link is also safe when a separate stage encounters it.
            again = Stage('existing-link', outputs=stage.outputs, links=stage.links)
            self.assertEqual(execute_stage(again, root), 'completed')

    def test_existing_unrelated_file_or_directory_is_rejected_unchanged(self):
        for kind in ['file', 'directory']:
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                source = root / 'atlas/sae'
                source.mkdir(parents=True)
                (source / 'model.pt').write_bytes(b'original checkpoint')
                destination = root / 'pool_first/sae'
                destination.parent.mkdir(parents=True)
                if kind == 'file':
                    destination.write_bytes(b'keep this file')
                    preserved = destination
                else:
                    destination.mkdir()
                    preserved = destination / 'unrelated.txt'
                    preserved.write_bytes(b'keep this file')
                stage = Stage('refuse-existing', outputs=[destination / 'model.pt'], links=[(destination, source)])
                with self.assertRaisesRegex(ValueError, 'Refusing to replace an existing file with a link'):
                    execute_stage(stage, root)
                self.assertFalse(destination.is_symlink())
                self.assertEqual(preserved.read_bytes(), b'keep this file')
                self.assertEqual((source / 'model.pt').read_bytes(), b'original checkpoint')
                self.assertFalse((root / 'stage_logs/refuse-existing.complete.json').exists())

    def test_success_sentinel_is_written_and_skips_repeat(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "output.json"
            stage = Stage("smoke", [sys.executable, "-c", "import pathlib; pathlib.Path('output.json').write_text('{}')"], [output])
            self.assertEqual(execute_stage(stage, root), "completed")
            stamp = output.stat().st_mtime_ns
            self.assertEqual(execute_stage(stage, root), "skipped")
            self.assertEqual(output.stat().st_mtime_ns, stamp)
            record = json.loads((root / "stage_logs/smoke.complete.json").read_text())
            self.assertEqual(record["returncode"], 0)

    def test_failed_command_has_no_success_sentinel(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(RuntimeError):
                execute_stage(Stage("failure", [sys.executable, "-c", "raise SystemExit(3)"], []), root)
            self.assertFalse((root / "stage_logs/failure.complete.json").exists())
            self.assertEqual(json.loads((root / "stage_logs/failure.attempt.json").read_text())["returncode"], 3)

    def test_missing_output_is_a_failure_even_for_exit_zero(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(RuntimeError):
                execute_stage(Stage("missing", [sys.executable, "-c", "pass"], [root / "absent.json"]), root)
            self.assertFalse((root / "stage_logs/missing.complete.json").exists())


if __name__ == "__main__":
    unittest.main()
