import unittest
from src.research_pipeline.remote.notebook import render_notebook


class RADNotebookTests(unittest.TestCase):
    def test_paired_notebook_has_adaptation_and_comparison_cells(self):
        manifest={'iteration':{'model_id':'rbf_fixed','problem_id':'burgers','hypothesis_id':'rad',
                    'protocol_phase':'exploratory'},'model':{'id':'rbf_fixed'},
                  'comparison_design':{'paired_seeds':[201,202,203,204,205]}}
        spec={'bundle_id':'a'*64,'evidence_kind':'research','runtime_dependencies':[],
              'setup_timeout_sec':900}
        nb=render_notebook(spec,{},manifest)
        ids={c['id'] for c in nb['cells']}
        self.assertIn('adaptive-points',ids)
        self.assertIn('paired-comparison',ids)
        for c in nb['cells']:
            if c['cell_type']=='code': compile(''.join(c['source']),c['id'],'exec')
        sampling=''.join(next(c for c in nb['cells'] if c['id']=='sampling-signals')['source'])
        self.assertIn('job_spec(seed)',sampling)

    def test_parallel_cell_checks_gate_before_runtime_dispatch(self):
        manifest={'iteration':{'model_id':'rbf_fixed','problem_id':'burgers','hypothesis_id':'rad'},
                  'comparison_design':{'paired_seeds':[201,202,203,204,205]}}
        spec={'bundle_id':'a'*64,'evidence_kind':'research','runtime_dependencies':[],'setup_timeout_sec':900}
        nb=render_notebook(spec,{},manifest)
        source=''.join(next(c for c in nb['cells'] if c['id']=='parallel-run')['source'])
        self.assertIn('assert_launch_ready(',source)
        self.assertLess(source.index('assert_launch_ready('),source.index('campaign_state = run_bundle'))
