from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


class RemoteSetupTests(unittest.TestCase):
    def test_authorized_policy_is_limited_to_kaggle(self):
        import tomllib
        policy = tomllib.loads((ROOT / 'context/agent-policy.toml').read_text())
        self.assertEqual(policy['scope']['external_write_access'], 'approved_services')
        self.assertEqual(policy['scope']['external_write_services'], ['kaggle'])
        self.assertEqual(policy['meta']['autonomy_level'], 'L2_human_verified_ai_execution')

    def test_doctor_does_not_require_or_expose_credentials(self):
        result = subprocess.run([sys.executable, '-m', 'src.research_pipeline.remote', 'doctor'],
                                cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        self.assertEqual(report['authentication'], 'deferred')
        self.assertFalse(report['network_checked'])
        self.assertFalse(report['remote_execution_ready'])
        self.assertIn('local_dependencies_ready', report)

    def test_adapter_prefers_repository_environment_without_path_activation(self):
        from src.research_pipeline.remote.kaggle import executable_path
        base = ROOT / 'var/agent_runs/tests'
        base.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=base) as directory:
            root = Path(directory)
            import os
            expected = root / 'var/agent_runs/remote-env' / ('Scripts/kaggle.exe' if os.name == 'nt' else 'bin/kaggle')
            expected.parent.mkdir(parents=True)
            expected.touch()
            with patch('src.research_pipeline.remote.kaggle.ROOT', root), \
                 patch('src.research_pipeline.remote.kaggle.shutil.which', return_value=None):
                self.assertEqual(executable_path(), expected)

    def test_mcp_configuration_uses_local_launcher_and_stays_disabled(self):
        import tomllib
        config = tomllib.loads((ROOT / '.codex/config.toml').read_text())
        self.assertFalse(config['mcp_servers']['kaggle']['enabled'])
        colab = config['mcp_servers']['colab']
        self.assertFalse(colab['enabled'])
        self.assertIn('src.research_pipeline.remote.colab_server', colab['args'])
        self.assertNotIn('uvx', colab['command'])


if __name__ == '__main__':
    unittest.main()
