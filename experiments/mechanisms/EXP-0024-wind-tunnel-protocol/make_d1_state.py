"""EXP-0024: estado de Burgers (RBF48, semilla 1) tras 50 pasos de Adam en float32 -- el punto exacto donde D1 reubica en la receta reducida (25 % de 200) --
guardado en float64 (state_dict completo y los puntos de colocacion), para comparar la reubicacion en CPU y en GPU sobre pesos identicos.
Uso: python make_d1_state.py  ->  kaggle/dataset/d1_state_f64.pt
"""
import importlib.util
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.argv = [sys.argv[0]]
spec = importlib.util.spec_from_file_location("t1_trainer_state", str(HERE / "t1_trainer.py"))
t1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(t1)
torch.set_num_threads(2)

cfg = t1.PROBLEMS["burgers"]
prob = t1.pc.make_problem("burgers", str(cfg["ref"]))
model = prob.build("rbf", cfg["width"], 11, 1).to(dtype=torch.float32)
P = prob.pools(1, torch.device("cpu"), torch.float32)
t1.adam_with_hook(prob, model, P, 50)
torch.save({"state_dict": {k: v.detach().double() for k, v in model.state_dict().items()}, "points": P["residual"].detach().double(), "adam_steps": 50, "seed": 1,
            "note": "Burgers RBF48 semilla 1, 50 pasos de Adam en float32, convertido a float64"}, HERE / "kaggle/dataset/d1_state_f64.pt")
print("ok", sum(v.numel() for v in model.state_dict().values()))
