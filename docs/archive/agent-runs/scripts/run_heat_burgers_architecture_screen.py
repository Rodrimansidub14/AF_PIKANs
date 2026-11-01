"""Prepare, submit, inspect and fetch the canonical Heat+Burgers screen."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from src.research_pipeline.remote.bundle import prepare, verify_transport
from src.research_pipeline.remote.evaluation import evaluate
from src.research_pipeline.remote.kaggle import fetch, invoke, status, submit
from src.research_pipeline.remote.postprocess import (
    extract_results_archive,
    sync_offline_wandb,
    verify_results_hashes,
)
from src.research_pipeline.remote.runtime import write_json


BASE = ROOT / "var" / "agent_runs" / "heat-burgers-architecture-screen"
NOTEBOOKS = ROOT / "notebooks" / "experiments"
BUNDLE_DIRNAME = "bundle-logscale-v1"
SEEDS = [101, 102, 103]
MODEL_SPECS = {
    "rbf_fixed": {
        "label": "RBF-PIKAN fija",
        "id": "rbf_fixed",
        "basis_count": 11,
        "centers": [-1.05, 1.05],
        "width": 0.21,
        "geometry_trainable": False,
        "edge_base": "SiLU",
        "hidden_activation": "tanh",
        "architecture": [2, 16, 16, 1],
        "expected_parameters": 3648,
        "initialization": "Normal(0, 0.1/sqrt(input_width)) for edge coefficients and base weights",
    },
    "mlp_tanh": {
        "label": "PINN MLP tanh (58 units)",
        "id": "mlp_tanh",
        "width": 58,
        "architecture": [2, 58, 58, 1],
        "hidden_activation": "tanh",
        "output_bias": True,
        "initialization": "Xavier uniform with tanh gain on hidden layers; Xavier uniform output; zero biases",
        "expected_parameters": 3655,
    },
    "bspline_cubic": {
        "label": "B-spline PIKAN cúbica fija",
        "id": "bspline_cubic",
        "basis_count": 11,
        "degree": 3,
        "knots": "open-uniform, fixed over [-1.05, 1.05]",
        "geometry_trainable": False,
        "edge_base": "SiLU",
        "hidden_activation": "tanh",
        "architecture": [2, 16, 16, 1],
        "expected_parameters": 3648,
        "initialization": "Normal(0, 0.1/sqrt(input_width)) for spline coefficients and base weights",
    },
}

SOURCE_FILES = [
    "src/research_pipeline/sciml/__init__.py",
    "src/research_pipeline/sciml/model.py",
    "src/research_pipeline/sciml/sampling.py",
    "src/research_pipeline/sciml/train.py",
    "src/research_pipeline/sciml/campaign.py",
    "src/research_pipeline/remote/__init__.py",
    "src/research_pipeline/remote/reporting.py",
    "evals/__init__.py",
    "evals/checkpoint_validation.py",
    "evals/problems/__init__.py",
    "evals/problems/_grids.py",
    "evals/problems/heat_1d/__init__.py",
    "evals/problems/heat_1d/reference.py",
    "evals/problems/burgers_1d/__init__.py",
    "evals/problems/burgers_1d/reference.py",
    "evals/problems/burgers_pinnacle_1d/__init__.py",
    "evals/problems/burgers_pinnacle_1d/reference.py",
]

ARTIFACTS = [
    "metrics.json", "decision_note.md", "problem_status.json", "versions.json",
    "config.json", "evidence_hashes.json", "hardware.json", "gpu_telemetry.csv",
    "gpu_telemetry_status.json",
]
for _problem in ("heat", "burgers"):
    ARTIFACTS.extend(
        f"{_problem}/{_name}"
        for _name in (
            "pools.npz", "sampling_signals.json", "history.jsonl",
            "diagnostic_signals.json", "training_summary.json", "validation.json",
            "validation.npz", "validation.html", "training-losses-log.html",
            "training-signals-log.html", "signals.json",
        )
    )


def load_local_credentials():
    allowed = {
        "KAGGLE_API_TOKEN", "KAGGLE_USERNAME", "WANDB_API_KEY",
        "WANDB_ENTITY", "WANDB_PROJECT",
    }
    env_file = ROOT / ".env"
    if not env_file.is_file():
        return
    for line in env_file.read_text(encoding="utf-8-sig").splitlines():
        if "=" not in line or line.lstrip().startswith("#"):
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip().strip("\"'")
        if key in allowed and value:
            os.environ.setdefault(key, value)


def decisions_markdown(model):
    return f"""# Decisiones de parámetros — screening Heat + Burgers canónico

**Estado:** exploratorio; tres semillas piloto; no establece SOTA ni acepta un claim.
**Modelo de este paquete:** {model['label']} (`{model['id']}`).

| Decisión | Configuración | Justificación |
|---|---|---|
| Heat | Conducción 1D en varilla homogénea: `u_t=0.1 u_xx`, x,t en [0,1], IC `sin(pi*x)`, extremos a cero. | Modo seno fundamental con solución exacta `sin(pi*x)*exp(-0.1*pi^2*t)`; control suave y analítico. |
| Burgers | `u_t + u*u_x - nu*u_xx=0`, dominio [-1,1]×[0,1], `nu=0.01/pi`, IC `-sin(pi*x)`, fronteras cero. | Benchmark PINN estándar Raissi 2019 / Burgers1d de PINNacle NeurIPS 2024. Es transporte no lineal escalar viscoso, puente hacia NS; no incluye presión ni geometría. |
| Referencia Burgers | Transformación Cole–Hopf; convolución gaussiana por Gauss–Hermite orden128; evaluación independiente en 1025×129. | La variante canónica no tiene la referencia tanh estacionaria de EXP-0002. Orden96 vs128 difirió ≤3.2e-15 en la grilla de refinamiento documentada; IC/fronteras verificadas. |
| Región del frente | `abs(x)<=5*nu` y `t>=1/pi`. | Métrica secundaria definida por el proyecto: frente centrado después del tiempo de empinamiento invíscido, dentro de cinco escalas viscosas. No es una métrica oficial de PINNacle. |
| Geometría y muestreo | Scrambled Sobol fijo; mismos streams/pools, semillas y conteos que EXP-0002. | Aísla arquitectura. Se verificaron arrays idénticos para interior, diagnóstico, fronteras e IC en seeds 101–103. No se compara el algoritmo de muestreo aún. |
| Arquitectura | {model['label']}; {model['architecture']}; `{model['expected_parameters']}` parámetros. | Frente a RBF-PIKAN fija de 3.648 parámetros; la MLP tiene 3.655 (diferencia <0.2%). Geometrías B-spline/RBF fijas. |
| Activaciones/base | Hidden tanh; SiLU en camino base de KAN; salida lineal. | Igual función de salida y condiciones físicas para todos los modelos. Heat impone IC y Dirichlet exactamente con `sin(pi*x)+t*x*(1-x)*N`. |
| Escalas físicas | Heat residual 1.1; Burgers residual `2+nu`; amplitud de campo 1. | Escalas explícitas con `U=T=L=1` en Burgers; misma normalización entre modelos. |
| Precisión y optimización | float64, PINA/PyTorch, Adam lr=0.001, sin scheduler ni L-BFGS. | Conserva el protocolo de derivadas segundas del baseline; comparación de una sola variable principal. |
| Presupuesto | Heat: 4.096 interiores, 5.000 pasos/300 s máximo. Burgers: 8.192 interiores, 20.000 pasos/1.200 s máximo. Batch 1.024. | Mismos límites para cada arquitectura; se reportan pasos y tiempo reales y cuál límite terminó el run. Tres semillas son screening. |
| Diagnóstico | Sobol independiente, 256 puntos por snapshot cada250 pasos; sensibilidad `du/dtheta` en32 puntos; métricas cada100 pasos. | Instrumenta muestreo, pérdidas, activaciones/bases, gradientes, derivadas, residual, sensibilidad, hardware y costo sin adaptar puntos. |
| Semillas y cómputo | 101,102,103; dos Tesla T4; máximo dos semillas concurrentes; un proceso por GPU. | Pares por semilla entre arquitecturas; cada proceso entrena Heat y luego Burgers. Timeout exterior 2.100 s por semilla. |
| W&B y artefactos | Un run offline por problema/semilla, grupo compartido `{"heat-burgers-architecture-screen-EXP-0005"}`; CSV/JSON/NPZ/Plotly/HTML y notebook reproducible. | La clave W&B solo se usa al sincronizar desde el entorno local; no se incluye en notebook, manifiesto o bundle. |
| Escala visual | Pérdidas, gradientes, residuales y error absoluto positivo en escala log; solución firmada lineal; ceros del error sin color en el mapa log. | Permite leer magnitudes separadas por órdenes sin cambiar los valores registrados ni añadir un epsilon artificial. |
| Estado estadístico | Sin umbrales después de resultados; evaluación final bloqueada. | Informa las tres observaciones y su dispersión descriptiva; no infiere significancia, causalidad ni SOTA. |

**Fuentes:** Raissi, Perdikaris & Karniadakis, *Physics-informed neural networks*, JCP 2019, DOI 10.1016/j.jcp.2018.10.045; Hao et al., *PINNacle*, NeurIPS 2024, apéndice Burgers1d; separación de variables de la conducción en una varilla, MIT OCW. Los enlaces y el contrato están en `problem-definition-proposal.md`.
"""


def build_manifest(model_id):
    model = MODEL_SPECS[model_id]
    model = {**model, "implementation": "src/research_pipeline/sciml/campaign.py"}
    return {
        "project": {"name": "AF_PIKANs", "ontology": "pikan-research"},
        "iteration": {
            "id": f"heat-burgers-arch-screen-{model_id}-exp005",
            "experiment_id": "EXP-0005",
            "protocol_phase": "exploratory",
            "domain_level": "canonical_pde",
            "problem_id": "heat_and_burgers_standard",
            "model_id": model_id,
            "focalization_id": "none_fixed_sobol",
            "hypothesis_id": "matched_architecture_screen",
        },
        "hypothesis": {
            "statement": "Under canonical Heat and Burgers definitions, fixed scrambled Sobol pools and matched parameter/compute budgets, compare fixed RBF-PIKAN, tanh MLP PINN and fixed cubic B-spline PIKAN.",
            "expected_effect": "Quantify differences in field error, physical residual, front-local error, training signals and cost; no direction of superiority is assumed.",
            "falsification_condition": "The comparison is inconclusive if any architecture lacks finite independent validation, uses different pools/physics, or fails to preserve its declared parameter and budget contract.",
        },
        "problem": {
            "sequence": ["heat", "burgers"],
            "heat": {
                "alpha": 0.1, "x": [0.0, 1.0], "t": [0.0, 1.0],
                "evaluation_id": "heat", "residual_scale": 1.1,
                "initial_condition": "sin(pi*x)", "boundary_values": [0.0, 0.0],
                "reference": "sin(pi*x)*exp(-alpha*pi^2*t)",
            },
            "burgers": {
                "definition": "raissi_pinnacle_1d",
                "evaluation_id": "burgers_pinnacle_1d",
                "nu": 0.01 / 3.141592653589793,
                "x": [-1.0, 1.0], "t": [0.0, 1.0],
                "initial_condition": "-sin(pi*x)",
                "boundary_values": [0.0, 0.0],
                "field_scale": 1.0,
                "residual_scale": 2.0 + 0.01 / 3.141592653589793,
                "reference": "Cole-Hopf Gaussian convolution; Gauss-Hermite order 128",
                "validation_grid": {"nx": 1025, "nt": 129},
                "front_region": "abs(x)<=5*nu and t>=1/pi",
            },
        },
        "model": model,
        "training": {
            "seeds": SEEDS,
            "steps_per_problem": 20000,
            "seconds_per_problem": 1200,
            "interior_points": 8192,
            "boundary_points": 512,
            "batch_size": 1024,
            "learning_rate": 0.001,
            "optimizer": "Adam",
            "sampling": "scrambled_sobol_fixed",
            "seed_offsets": {"interior": 1, "diagnostic": 2, "boundary": 3, "initial": 4},
            "diagnostic_every": 250,
            "diagnostic_points": 256,
            "sensitivity_points": 32,
            "cpu_threads": 2,
            "accelerator": "gpu",
            "problem_overrides": {
                "heat": {"interior_points": 4096, "steps_per_problem": 5000, "seconds_per_problem": 300},
                "burgers": {"interior_points": 8192, "steps_per_problem": 20000, "seconds_per_problem": 1200},
            },
            "metric_log_every": 100,
        },
        "tracking": {
            "entity": os.environ.get("WANDB_ENTITY", "man22611-uvg"),
            "project": os.environ.get("WANDB_PROJECT", "AF_PIKANS"),
            "group": "heat-burgers-architecture-screen-EXP-0005",
            "mode": "offline",
        },
        "remote": {
            "enable_internet": True,
            "runtime_dependencies": [
                "pina-mathlab==0.3.2.post2609",
                "wandb==0.30.0",
                "plotly==7.1.0",
                "scipy==1.15.3",
                "pandas==2.2.3",
            ],
            "setup_timeout_sec": 900,
            "machine_shape": "NvidiaTeslaT4",
            "max_parallel_seeds": 2,
            "run_timeout_sec": 2100,
        },
        "focalization": {"enabled": False, "bounded_edit_surface": []},
        "acceptance_gates": {
            "metrics": {
                "heat/field_relative_l2": {"required": True, "direction": "minimize"},
                "burgers/field_relative_l2": {"required": True, "direction": "minimize"},
                "burgers/front_field_rmse_normalized": {"required": True, "direction": "minimize"},
            }
        },
        "artifacts": {"expected": ARTIFACTS},
        "parameter_decisions_markdown": decisions_markdown(model),
    }


def prepare_all(model_ids=None):
    NOTEBOOKS.mkdir(parents=True, exist_ok=True)
    results = []
    owner = os.environ.get("KAGGLE_USERNAME", "rodrigomansilla")
    selected_models = model_ids or MODEL_SPECS.keys()
    for model_id in selected_models:
        model = MODEL_SPECS[model_id]
        model_root = BASE / "models" / model_id
        model_root.mkdir(parents=True, exist_ok=True)
        manifest_path = model_root / "manifest.json"
        manifest = build_manifest(model_id)
        write_json(manifest_path, manifest)
        kernel = f"{owner}/af-pikans-arch-screen-{model_id.replace('_', '-')}-exp005"
        bundle = model_root / BUNDLE_DIRNAME
        spec = prepare(
            manifest_path,
            bundle,
            kernel,
            SOURCE_FILES,
            timeout=2100,
            evidence_kind="research",
            gpu=True,
        )
        approval_path = model_root / f"approval-{BUNDLE_DIRNAME}.json"
        write_json(
            approval_path,
            {
                "bundle_id": spec["bundle_id"],
                "approved_by": "Rodrigo Mansilla",
                "scope": "approved",
                "hypothesis": "approved",
                "code_surface": "approved",
                "remote_execution": True,
                "authorization_source": "Conversation: researcher directed implementation and retraining of Heat+Burgers, requested canonical literature definitions, and has prior approved Kaggle/T4x2 execution.",
                "interpretation": "Exploratory paired architecture screen; three seeds; no SOTA claim or scientific acceptance.",
            },
        )
        verify_transport(bundle)
        notebook_copy = NOTEBOOKS / f"heat-burgers-architecture-screen-{model_id}-{BUNDLE_DIRNAME}.ipynb"
        shutil.copy2(bundle / "experiment.ipynb", notebook_copy)
        results.append(
            {
                "model_id": model_id,
                "bundle_id": spec["bundle_id"],
                "kernel": kernel,
                "max_parallel_seeds": spec["max_parallel_seeds"],
                "timeout_sec_per_seed": spec["timeout_sec"],
                "notebook": str(notebook_copy.relative_to(ROOT)),
                "source_files": len(spec["files"]),
            }
        )
    write_json(BASE / f"prepared-campaigns-{BUNDLE_DIRNAME}.json", results)
    return results


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("action", choices=("prepare", "submit", "status", "fetch", "check"))
parser.add_argument("--model", choices=tuple(MODEL_SPECS))
args = parser.parse_args()

if args.action == "prepare":
    print(json.dumps(prepare_all([args.model] if args.model else None), indent=2))
else:
    load_local_credentials()
    if args.action == "check":
        reply = invoke(["kernels", "list", "--mine", "--page-size", "1", "--csv"])
        print(json.dumps({"authenticated_listing": "ref" in reply or "No kernels found" in reply}))
    elif args.action == "submit":
        results = []
        for model_id in ([args.model] if args.model else MODEL_SPECS):
            model_root = BASE / "models" / model_id
            bundle = model_root / BUNDLE_DIRNAME
            state_file = bundle / "submission.json"
            if state_file.exists():
                results.append({"model_id": model_id, "submission": "already-recorded"})
                continue
            approval = model_root / f"approval-{BUNDLE_DIRNAME}.json"
            results.append({"model_id": model_id, **submit(bundle, approval)})
        print(json.dumps(results, indent=2))
    elif args.action == "status":
        print(json.dumps([
            {"model_id": model_id, **status(BASE / "models" / model_id / BUNDLE_DIRNAME)}
            for model_id in ([args.model] if args.model else MODEL_SPECS)
        ], indent=2))
    elif args.action == "fetch":
        results = []
        for model_id in ([args.model] if args.model else MODEL_SPECS):
            model_root = BASE / "models" / model_id
            bundle = model_root / BUNDLE_DIRNAME
            download_dir = model_root / f"download-{BUNDLE_DIRNAME}"
            if download_dir.exists():
                fetch_result = {"download_reused": True}
            else:
                try:
                    fetch_result = fetch(bundle, download_dir)
                except subprocess.TimeoutExpired:
                    # Kaggle can time out after saving its complete result ZIP.
                    # The archive checks below decide whether the local copy is usable.
                    fetch_result = {
                        "transport_warning": "Kaggle output retrieval timed out; validating the saved result archive."
                    }
            archives = list(download_dir.rglob("heat-burgers-t4x2-results.zip"))
            if len(archives) != 1:
                raise ValueError(f"Expected one results archive for {model_id}")
            extracted_dir = model_root / f"download-extracted-{BUNDLE_DIRNAME}"
            if extracted_dir.exists():
                extracted = extracted_dir
            else:
                extracted = extract_results_archive(archives[0], extracted_dir)
            fetch_result["result_file_verification"] = verify_results_hashes(extracted)
            loose_tree_validity = fetch_result.get("execution_valid")
            archive_evaluation = evaluate(bundle, extracted / "runs")
            fetch_result["loose_tree_execution_valid"] = loose_tree_validity
            fetch_result["execution_valid"] = archive_evaluation["execution_valid"]
            fetch_result["archive_evaluation"] = {
                key: archive_evaluation[key]
                for key in ("execution_valid", "thresholds_pass", "eligible_for_review", "failures", "metrics")
            }
            write_json(download_dir / "archive-evaluation.json", archive_evaluation)
            if not archive_evaluation["execution_valid"]:
                raise ValueError(f"Archived campaign failed execution validation for {model_id}")
            candidates = list(extracted.rglob("heat-burgers-results.ipynb"))
            if len(candidates) != 1:
                raise ValueError(f"Expected one results notebook for {model_id}")
            notebook_copy = NOTEBOOKS / f"heat-burgers-architecture-screen-{model_id}-{BUNDLE_DIRNAME}-results.ipynb"
            shutil.copy2(candidates[0], notebook_copy)
            fetch_result["results_notebook"] = str(notebook_copy.relative_to(ROOT))
            sync_result = sync_offline_wandb(extracted / "runs", os.environ)
            write_json(model_root / "wandb-sync.json", sync_result)
            fetch_result["wandb_sync"] = sync_result
            results.append({"model_id": model_id, **fetch_result})
        print(json.dumps(results, indent=2))
