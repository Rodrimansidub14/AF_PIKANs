import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from src.research_pipeline.sciml.train import train,make_problem
from src.research_pipeline.sciml.sampling import pools
from src.research_pipeline.sciml.model import build_model
from src.mechanisms.rad_callback import RADCallback


class Sink:
    def log(self,*args,**kwargs): pass


class RADPinaTests(unittest.TestCase):
    def test_live_condition_data_replaced_boundary_untouched(self):
        torch.set_default_dtype(torch.float64)
        data=pools('burgers',201,64,16)
        problem=make_problem('burgers',data)
        boundary=problem.conditions['left'].input.clone()
        model=build_model('burgers',{'id':'rbf_fixed'}).double()
        with tempfile.TemporaryDirectory(dir='var/agent_runs') as tmp:
            cb=RADCallback(model,problem,data,Path(tmp),{'seed':201,'steps_per_problem':8,
                'rad':{'candidate_points':128,'chunk_size':32,'warmup_steps':2,'refresh_every':2}},Sink(),{'nu':.01/np.pi})
            cb.refresh(2)
            with np.load(Path(tmp)/'rad-000002.npz') as saved:
                np.testing.assert_array_equal(problem.conditions['physics'][list(range(64))].input.detach().numpy(),saved['points'])
                torch.testing.assert_close(problem.conditions['left'].input,boundary)
                np.testing.assert_array_equal(data['interior'],saved['points'])

    def test_training_consumes_adaptive_pool_and_preserves_step_budget(self):
        cfg={'seed':201,'cpu_threads':2,'interior_points':64,'boundary_points':16,
             'learning_rate':.001,'batch_size':16,'steps_per_problem':8,'seconds_per_problem':120,
             'accelerator':'cpu','diagnostic_points':16,'sensitivity_points':2,'diagnostic_every':4,
             'metric_log_every':2,'sampling':'rad','rad':{'candidate_points':128,'chunk_size':32,
             'warmup_steps':2,'refresh_every':2,'power':2,'uniform_fraction':.5}}
        with tempfile.TemporaryDirectory(dir='var/agent_runs') as tmp:
            out=Path(tmp)/'run'
            checkpoint,summary=train('burgers',cfg,out,Sink(),{'id':'rbf_fixed'},
                {'nu':.01/np.pi,'definition':'raissi_pinnacle_1d','residual_scale':2+.01/np.pi})
            self.assertEqual(summary['steps'],8)
            updates=[json.loads(line) for line in (out/'rad-history.jsonl').read_text().splitlines()]
            self.assertEqual([u['step'] for u in updates],[2,4,6])
            self.assertTrue(checkpoint.is_file())
            self.assertEqual(summary['adaptive_sampling']['refresh_count'],3)
            self.assertGreater(summary['adaptive_sampling']['consumed_changed_batches'],0)
            original=np.load(out/'pools.npz')['interior']
            final=np.load(out/'rad-000006.npz')['points']
            self.assertFalse(np.array_equal(original,final))
