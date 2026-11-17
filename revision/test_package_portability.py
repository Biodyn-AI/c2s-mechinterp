"""Public command imports must resolve inside this checkout, including from another cwd."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


REPOSITORY = Path(__file__).resolve().parents[1]


class PackagePortabilityTests(unittest.TestCase):
    def run_elsewhere(self, arguments):
        with tempfile.TemporaryDirectory() as directory:
            environment = dict(os.environ)
            environment['PYTHONPATH'] = str(REPOSITORY)
            return subprocess.run([sys.executable, *arguments], cwd=directory, env=environment,
                                  capture_output=True, text=True)

    def test_coactivation_help_does_not_need_torch_or_an_author_checkout(self):
        completed = self.run_elsewhere(['-m', 'c2s_mechinterp.coactivation', '--help'])
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn('--run', completed.stdout)

    def test_coactivation_loads_the_package_sae_instead_of_a_top_level_copy(self):
        # Tiny stubs stop at SAE loading, so no model, tensor allocation or fixture data is needed.
        code = '''
import sys, types
torch = types.ModuleType('torch')
torch.cuda = types.SimpleNamespace(is_available=lambda: False)
sys.modules['torch'] = torch
sae = types.ModuleType('c2s_mechinterp.sae_topk')
class PackageSAE:
    @staticmethod
    def load(*args, **kwargs):
        print('PACKAGE_LOCAL_SAE')
        raise SystemExit(0)
sae.TopKSAE = PackageSAE
sys.modules['c2s_mechinterp.sae_topk'] = sae
from c2s_mechinterp.coactivation import main
sys.argv = ['coactivation', '--run', 'unused', '--layer', '0']
main()
'''
        completed = self.run_elsewhere(['-c', code])
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn('PACKAGE_LOCAL_SAE', completed.stdout)


if __name__ == '__main__':
    unittest.main()
