import unittest
import ast
from pathlib import Path
from src.sciml.rad_campaign import job_spec


class JobMappingTests(unittest.TestCase):
    def test_five_pairs_share_actual_training_seeds(self):
        for seed in range(201,206):
            self.assertEqual(job_spec(seed),('sobol',seed))
            self.assertEqual(job_spec(seed+1000),('rad',seed))

    def test_undeclared_job_is_rejected(self):
        for job in [101,1206,-1]:
            with self.assertRaises(ValueError): job_spec(job)

    def test_unsigned_manifest_rejected_before_creating_output(self):
        import json,os,tempfile
        from pathlib import Path
        from unittest.mock import patch
        from src.sciml.rad_campaign import main
        with tempfile.TemporaryDirectory(dir='var/agent_runs') as tmp:
            source=Path(tmp)/'manifest.json'; out=Path(tmp)/'not-created'
            source.write_text(json.dumps({'research_status':{'scientific_preregistration_signed':False},
                                         'comparison_design':{'launch_ready':False}}))
            with patch.dict(os.environ,{'HARNESS_MANIFEST':str(source),'HARNESS_RUN_DIR':str(out),'HARNESS_SEED':'201'}):
                with self.assertRaisesRegex(RuntimeError,'scientific'):
                    main()
            self.assertFalse(out.exists())

    def test_burgers_campaign_import_does_not_require_allen_cahn_module(self):
        source = Path(__file__).resolve().parents[1] / 'src/sciml/campaign.py'
        tree = ast.parse(source.read_text(encoding='utf-8'))
        top_level_imports = {
            node.module
            for node in tree.body
            if isinstance(node, ast.ImportFrom) and node.module
        }
        self.assertNotIn('src.sciml.allen_cahn_evaluation', top_level_imports)
