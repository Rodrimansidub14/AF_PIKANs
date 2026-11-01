from __future__ import annotations

import base64
import hashlib
import io
import json
import re
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
BUNDLE = ROOT / "bundles" / "heat"
notebook = json.loads((BUNDLE / "heat-tier2-t4x2.ipynb").read_text(encoding="utf-8"))
source = "".join(notebook["cells"][2].get("source", []))
match = re.search(r'PAYLOAD_B64\s*=\s*"([^"]+)"', source)
if not match:
    raise RuntimeError("Saved Heat v4 payload archive is missing.")

original_zip = base64.b64decode(match.group(1))
with zipfile.ZipFile(io.BytesIO(original_zip)) as bundle:
    files = {name: bundle.read(name) for name in bundle.namelist()}
runner = OUT / "heat_stepmatched_runner.py"
files["heat_stepmatched_runner.py"] = runner.read_bytes()
payload_manifest = json.loads(files["payload-manifest.json"].decode("utf-8"))
payload_manifest.setdefault("files", {})["heat_stepmatched_runner.py"] = hashlib.sha256(
    files["heat_stepmatched_runner.py"]
).hexdigest()
payload_manifest["status"] = "PENDING_HUMAN_SIGNATURE"
payload_manifest["training"] = (
    "float64; exact saved v4 Sobol pools; 20,000 Adam updates plus "
    "DeepXDE 1.12.2 PyTorch L-BFGS defaults; RBF and B-spline only"
)
files["payload-manifest.json"] = (json.dumps(payload_manifest, indent=2) + "\n").encode("utf-8")

packed = io.BytesIO()
with zipfile.ZipFile(packed, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as bundle:
    for name, data in files.items():
        bundle.writestr(name, data)
payload_zip = packed.getvalue()
campaign_template = (OUT / "heat_stepmatched_campaign.py").read_text(encoding="utf-8")
if campaign_template.count("__S2_PAYLOAD_B64__") != 1:
    raise RuntimeError("S2 payload marker missing or duplicated.")
entrypoint = campaign_template.replace(
    "__S2_PAYLOAD_B64__", base64.b64encode(payload_zip).decode("ascii")
)
entrypoint_path = OUT / "heat-stepmatched-s2.py"
entrypoint_path.write_text(entrypoint, encoding="utf-8", newline="\n")

metadata = {
    "id": "rodrimandub/af-pikans-heat-step-matched-s2-20260928",
    "title": "AF PIKANs Heat Step Matched S2 20260928",
    "code_file": entrypoint_path.name,
    "language": "python",
    "kernel_type": "script",
    "is_private": True,
    "enable_gpu": True,
    "enable_tpu": False,
    "enable_internet": False,
    "machine_shape": "NvidiaTeslaT4",
    "dataset_sources": [],
    "competition_sources": [],
    "kernel_sources": ["rodrimandub/af-pikans-heat-tier2-torch-pilot-t4x2"],
    "model_sources": [],
}
(OUT / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

pools = {
    "901": "318cdd951dc76a19ce0be65dd170beaffdf40741d4e11301abda5a8b2fe18f51",
    "902": "c89cd6ae60703dcb8cdf04448e2d50f8cdc02324446c1807dcfb3a2145b926d6",
    "903": "0e3557a470006da65b348dd066ba73ccad768c6682e3c337c5392c1e8bc6d4e7",
    "904": "f3067f1997a1b17947fc3910fd4f70db0e5fded96cb87ef3f87a12f43ea05825",
    "905": "dc567fc082828a68ac5483975659f155644d9040cf01916147758e1b5e6736b0",
}
optimizer = {
    "adam_updates": 20000,
    "learning_rate": 0.001,
    "lbfgs_backend": "PyTorch L-BFGS using DeepXDE 1.12.2 defaults",
    "maxiter": 15000,
    "maxfun": 18750,
    "history_size": 100,
    "tolerance_grad": 1e-8,
    "tolerance_change": 0.0,
    "line_search_fn": "strong_wolfe",
    "chunk_iter": 25,
    "chunk_max_eval": 32,
}
local_manifest = {
    "session": "S2",
    "status": "prepared_pending_quota_refresh",
    "kernel_id": metadata["id"],
    "kernel_url": "https://www.kaggle.com/code/rodrimandub/af-pikans-heat-step-matched-s2-20260928",
    "session_cap_hours": 1.25,
    "session_cap_seconds": 4500,
    "per_run_cap_seconds": 680,
    "arms": ["rbf", "bspline"],
    "seeds": [901, 902, 903, 904, 905],
    "pool_hashes": pools,
    "optimizer": optimizer,
    "evaluator": "heat-confirmatory-512x256-v1, CUDA after all training",
    "reference_seed_pairing": "Unpaired to DeepXDE alpha=0.1 seeds 11,22,33,44,55; H-RBF/H-BS paired by seed 901-905",
    "pool_source": "rodrimandub/af-pikans-heat-tier2-torch-pilot-t4x2 output pools",
    "human_preregistration_signature": "PENDING",
    "scientific_accepted": False,
    "reference_accepted": False,
    "source_hashes": {
        "campaign_source_sha256": hashlib.sha256(
            (OUT / "heat_stepmatched_campaign.py").read_bytes()
        ).hexdigest(),
        "runner_sha256": hashlib.sha256(runner.read_bytes()).hexdigest(),
        "entrypoint_sha256": hashlib.sha256(entrypoint_path.read_bytes()).hexdigest(),
        "payload_zip_sha256": hashlib.sha256(payload_zip).hexdigest(),
    },
}
(OUT / "manifest.json").write_text(json.dumps(local_manifest, indent=2) + "\n", encoding="utf-8")

experiment = Path("experiments/EXP-0007-heat-stepmatched-deepxde")
experiment.mkdir(parents=True, exist_ok=True)
spec = """# EXP-0007 — Heat step-matched comparison (v5)

Status: PENDING_HUMAN_SIGNATURE. This contract is unsigned and unaccepted.

## Question and scope

Under a fixed Heat PDE and saved per-seed collocation pools, what field error do the AF RBF and cubic B-spline arms reach after the DeepXDE tutorial optimizer budget? This is an exploratory comparison, not a superiority or SOTA claim.

## Problem

Solve u_t = 0.1 u_xx on [0,1] x [0,1], with u(x,0)=sin(pi*x), homogeneous Dirichlet boundaries, and exact solution sin(pi*x) exp(-0.1*pi^2*t). Residual is scaled by 1.1. Traces use sin(pi*x)+t*x*(1-x)*raw(x,t).

## Arms and seeds

RBF (rbf_fixed) and cubic B-spline (bspline_cubic), five paired seeds 901-905. The FNN arm is omitted before launch under the plan runtime rule because the 15-arm estimate exceeds 65 minutes. DeepXDE alpha=0.1 uses seeds 11,22,33,44,55, so comparisons to it are unpaired.

## Frozen pools and training

Reuse the exact saved v4 Torch scrambled Sobol NPZ pools for seeds 901-905; expected hashes are recorded in the run manifest and verified by the runner. Each arm uses the same pool for its seed, float64, full batch of 4,096 interior points, 1,024 initial points, and 512 boundary points per side. Adam lr 1e-3 for exactly 20,000 updates, then PyTorch L-BFGS with DeepXDE v1.12.2 defaults: maxiter=15000, maxfun=18750, history size 100, gradient tolerance 1e-8, function-change tolerance 0, and strong-Wolfe line search. Calls are chunked at 25 optimizer iterations to persist checkpoints and respect time caps. Per-run cap 680 seconds.

## Evaluation and resources

Train only on fixed pools. After all runs finish, score final checkpoints with the common CUDA evaluator on heat-confirmatory-512x256-v1. Session: private Kaggle T4x2, hard cap 4,500 seconds (1.25 quota-hours). Atomic progress and checkpoints are retained. Completion is summarized by Adam steps, L-BFGS steps/stopping reason, and confirmation score count.

## Pre-registration and decision boundary

This contract remains PENDING_HUMAN_SIGNATURE; no agent signs it. scientific_accepted=false and reference_accepted=false in every artifact. Results are evidence for the human scientific_claim gate only. DeepXDE comparisons are descriptive and unpaired; collocation, network, trace, and implementation differ.
"""
(experiment / "spec.md").write_text(spec, encoding="utf-8")
experiment_manifest = {
    "experiment_id": "EXP-0007",
    "version": "heat-stepmatched-v5",
    "status": "PENDING_HUMAN_SIGNATURE",
    "protocol_phase": "exploratory",
    "problem": "heat_1d_alpha_0.1",
    "arms": ["rbf_fixed", "bspline_cubic"],
    "seeds": [901, 902, 903, 904, 905],
    "training": optimizer,
    "pools": pools,
    "evaluation": "heat-confirmatory-512x256-v1",
    "resource_cap": {"platform": "Kaggle private T4x2", "session_hours": 1.25, "session_seconds": 4500},
    "scientific_accepted": False,
    "reference_accepted": False,
    "human_signature": None,
}
(experiment / "manifest.json").write_text(json.dumps(experiment_manifest, indent=2) + "\n", encoding="utf-8")
