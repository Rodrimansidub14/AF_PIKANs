"""Reproducible registry and report builder for exploratory RBF-PIKAN signals.

This module only reads already-produced artifacts under ``var/agent_runs``.
It never imports or reads the protected ``evals/`` verifier and does not start
training. Literature scores are only marked direct when the entire PDE and
experimental contract matches.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
from pathlib import Path
from typing import Any, Iterable


PDE_CONTRACT_FIELDS = (
    "equation",
    "domain",
    "initial_condition",
    "boundary_condition",
    "physical_parameters",
)
PROTOCOL_FIELDS = (
    "primary_metric",
    "evaluation_protocol",
    "architecture",
    "optimizer",
    "training_budget",
    "training_points",
)
REQUIRED_COMPARISON_FIELDS = PDE_CONTRACT_FIELDS + PROTOCOL_FIELDS


ARCHITECTURE_PROFILES: dict[str, dict[str, Any]] = {
    "rbf_fixed": {
        "label": "RBF-PIKAN fija",
        "role": "modelo primario para exploración de señales",
        "explicit_basis": True,
        "fixed_centers_and_widths": True,
        "directly_observable": [
            "activación por base y capa",
            "ocupación y cobertura de centros/ancho fijo",
            "conteo efectivo de bases",
            "coeficientes y norma de contribución por base",
            "sensibilidad espacial y estabilidad entre semillas",
        ],
        "claim_limit": (
            "La ventaja interpretativa frente a otras arquitecturas no está establecida; "
            "es una hipótesis que se debe medir con estabilidad, localización y ablación."
        ),
        "interpretability_claim_status": "hypothesis_to_measure",
    },
    "bspline_cubic": {
        "label": "PIKAN B-spline cúbica",
        "role": "control de representación con base explícita",
        "explicit_basis": True,
        "fixed_centers_and_widths": False,
        "directly_observable": [
            "activación por base y capa",
            "soporte local y ocupación de bases",
            "coeficientes y norma de contribución por base",
        ],
        "claim_limit": (
            "Las B-splines también ofrecen bases locales explícitas; no se asume que RBF sea más interpretable."
        ),
        "interpretability_claim_status": "control_required",
    },
    "mlp_tanh": {
        "label": "PINN MLP tanh",
        "role": "referencia contextual solicitada por el investigador",
        "explicit_basis": False,
        "fixed_centers_and_widths": False,
        "directly_observable": [
            "activaciones por capa",
            "gradientes y sensibilidad de entrada",
            "pesos y normas agregadas",
        ],
        "claim_limit": (
            "La ausencia de una base explícita no impide análisis de sensibilidad; "
            "solo hace que el diagnóstico geométrico por centro/ancho no exista de forma nativa."
        ),
        "interpretability_claim_status": "context_only",
    },
}


# Signal definitions distinguish universal instrumentation from features tied
# to the physics. These are measurements to collect, not candidate loss terms.
PROBLEM_SIGNAL_REGISTRY: dict[str, dict[str, Any]] = {
    "heat": {
        "contract": "heat_1d_alpha_0.1_sin_pi",
        "problem_specific": ["modal_decay", "thermal_energy_decay", "u_xx_scale", "initial_boundary_trace"],
        "question": "¿La cobertura de bases y los derivados predicen pérdida de modos o error térmico?",
        "next_diagnostic": "Descomponer u(x,t) en modos seno y correlacionar la cola modal con activación RBF en validación fija.",
    },
    "burgers": {
        "contract": "burgers_1d_raissi_pinnacle_transient",
        "problem_specific": ["front_position_and_width", "shock_gradient", "viscous_dissipation", "initial_boundary_trace"],
        "question": "¿Qué combinación de residual, derivadas y cobertura de bases localiza error alrededor del frente?",
        "next_diagnostic": "Mantener la evaluación independiente y medir residual/error por bandas del frente, además de correlación y enriquecimiento por semilla.",
    },
    "allen_cahn": {
        "contract": "allen_cahn_deepxde_1_12_2_dirichlet",
        "problem_specific": ["interface_and_energy_dissipation", "interface_position_and_width", "double_well_balance", "initial_boundary_trace"],
        "question": "¿La actividad de bases y la tasa de disipación predicen la localización y evolución de la interfaz?",
        "next_diagnostic": "Converger y desbloquear la referencia de campo; medir error de interfaz y energía frente a Sobol en puntos held-out.",
    },
    "dfg_ns": {
        "contract": "dfg_2d1_re20_incompressible",
        "problem_specific": ["divergence_and_momentum_balance", "cylinder_wall_shear", "drag_lift_pressure_drop", "wake_length"],
        "question": "¿Qué señales detectan defectos de conservación y error en magnitudes integrales de flujo?",
        "next_diagnostic": "Primero fijar solver FEM convergido, malla, normalización, QoIs DFG y descomposición de residuo por ecuación.",
    },
    "poisson": {
        "contract": "scope_pending",
        "problem_specific": ["flux_balance", "corner_singularity_or_smoothness", "boundary_trace"],
        "question": "¿Las bases fijas resuelven modos suaves y singularidades de esquina con señales distintas?",
        "next_diagnostic": "Conservar como problema candidato; reabrir solo tras decidir el alcance de la tesis.",
    },
}


COMMON_SIGNAL_FAMILIES = {
    "sampling": ["Sobol discrepancy", "nearest-neighbor distances", "domain/BC/IC coverage", "train-diagnostic separation"],
    "training": ["component losses", "gradient norm/quantiles", "gradient signal-to-noise by spatial partition", "fitting/transition/diffusion phase", "steps, throughput and stopping reason"],
    "autodiff": ["u and PDE derivatives by quantile", "term-wise residual scales", "finite/nonfinite derivative counts"],
    "rbf_basis": ["activation footprint", "occupancy", "effective basis count", "coefficient norm", "seed stability"],
    "outcome": ["independent held-out residual", "field error when reference is unlocked", "problem-specific physics/QoIs"],
    "cost": ["train and diagnostic seconds", "memory", "GPU utilization provenance", "diagnostic overhead"],
}


LITERATURE_COMPARATORS: list[dict[str, str]] = [
    {
        "problem": "Heat 1D",
        "method": "Free-RBF-KAN",
        "year": "2026",
        "source": "https://arxiv.org/abs/2601.07760",
        "evidence": "Preprint; includes a high-frequency 2D heat problem and compares adaptive RBF shapes/grids with fixed RBF and spline KAN variants.",
        "published_primary_score": "No score copied into the local leaderboard: the paper's high-frequency 2D heat contract differs from the project's smooth 1D heat equation.",
        "local_metric": "EXP-0005 fixed-RBF relative field L2 = 8.88e-5 ± 4.54e-5 (n=3); the B-spline screen mean is 2.30e-5 ± 9.49e-6.",
        "contract_status": "different_problem_contract",
        "signal_lesson": "Band widths/grid adaptation and high-frequency modal coverage are candidate signals; the project's smooth sin(pi*x), alpha=0.1 heat run is a different PDE instance.",
        "local_result_comparison": "context_only",
    },
    {
        "problem": "Burgers 1D",
        "method": "RAMS",
        "year": "2026",
        "source": "https://doi.org/10.1002/aidi.202500214",
        "evidence": "Paper reports 10 independent runs and a random-sampling mean relative-L2 reduction from 0.181 to 0.010 after residual-adversarial sample movement; its MLP, training budget and protocol differ.",
        "published_primary_score": "relative L2: 0.181 (random) → 0.010 (RAMS), 10 independent runs in the paper.",
        "local_metric": "EXP-0005 fixed-RBF relative field L2 = 5.40e-2 ± 3.93e-2 (n=3).",
        "contract_status": "same_problem_family_protocol_unmatched",
        "signal_lesson": "Measure whether points moved by residual gradients add information beyond residual magnitude, while recording movement overhead and independent held-out field error.",
        "local_result_comparison": "context_only_no_delta",
    },
    {
        "problem": "Burgers 1D",
        "method": "vRBA",
        "year": "2026",
        "source": "https://doi.org/10.1038/s44387-026-00084-4",
        "evidence": "For the canonical nu=1/(100*pi) Burgers contract, Table 4 reports relative L2=8.25e-9 for vRBA+SSBroyden (2.90e-8 for prior RAD+Fourier Features+SSBroyden). The paper uses 5,000 Adam warm-up plus 60,000 SSBroyden iterations, Fourier/periodic encoding and adaptive importance sampling; the local run uses fixed RBF-PIKAN and a different training protocol.",
        "published_primary_score": "relative L2: 8.25e-9 (vRBA+SSBroyden); prior RAD+Fourier Features+SSBroyden: 2.90e-8.",
        "local_metric": "EXP-0005 fixed-RBF relative field L2 = 5.40e-2 ± 3.93e-2 (n=3). Same canonical equation, domain, initial condition and zero endpoint data; protocol/evaluation details are not aligned.",
        "contract_status": "same_pde_different_protocol",
        "signal_lesson": "Residual weighting selects a functional: log/linear/quadratic/exponential potentials test different error-tail behavior. Track distribution weights and gradient signal-to-noise, not only mean residual.",
        "local_result_comparison": "same_problem_context_not_head_to_head",
    },
    {
        "problem": "Allen-Cahn 1D",
        "method": "vRBA",
        "year": "2026",
        "source": "https://doi.org/10.1038/s44387-026-00084-4",
        "evidence": "Table 4 reports relative L2=1.88e-6 for vRBA+SSBroyden. Its Allen-Cahn benchmark uses k=1e-4 with periodic spatial boundaries; EXP-0003 uses k=1e-3 and Dirichlet u(+/-1,t)=-1. The project pilot has no field-L2 while its reference is locked.",
        "published_primary_score": "relative L2: 1.88e-6 (best listed vRBA+SSBroyden result for its Allen-Cahn contract).",
        "local_metric": "No local field error: reference is locked pending convergence. EXP-0003 has only held-out PDE residual and energy signals.",
        "contract_status": "different_problem_contract",
        "signal_lesson": "Use energy dissipation, interface position/width, residual concentration and reference error together; residual alone cannot certify interface accuracy.",
        "local_result_comparison": "not_comparable",
    },
    {
        "problem": "DFG cylinder Re=20",
        "method": "FEATFLOW DFG 2D-1 benchmark plus PINNacle/cPIKAN methods as future learned comparators",
        "year": "benchmark + literature",
        "source": "https://wwwold.mathematik.tu-dortmund.de/~featflow/en/benchmarks/cfdbenchmarking/flow/dfg_benchmark1_re20.html",
        "evidence": "Official reference values and quantities of interest exist for DFG 2D-1; no local AF_PIKANs run or contract-matched learned SOTA score is yet recorded.",
        "published_primary_score": "Official FEM reference/QoIs exist; learned-model score pending contract-matched selection.",
        "local_metric": "No local run.",
        "contract_status": "reference_available_learned_comparator_pending",
        "signal_lesson": "Prioritize continuity, momentum terms, wall shear, pressure and drag/lift errors; validate against converged FEM QoIs before comparing architectures.",
        "local_result_comparison": "no_local_run",
    },
]


def classify_literature_comparability(
    local: dict[str, Any], literature: dict[str, Any]
) -> dict[str, Any]:
    """Classify whether a published score supports a numeric head-to-head.

    ``directly_comparable`` requires the PDE, its full physical contract, the
    metric, evaluation protocol, architecture, optimizer and budget all match.
    Matching only the PDE receives a context-only status.
    """
    missing = [
        key
        for key in REQUIRED_COMPARISON_FIELDS
        if key not in local or key not in literature or local[key] is None or literature[key] is None
    ]
    if missing:
        return {"status": "contract_unverified", "missing_fields": missing, "mismatched_fields": []}
    pde_mismatches = [key for key in PDE_CONTRACT_FIELDS if local[key] != literature[key]]
    if pde_mismatches:
        return {
            "status": "different_problem_contract",
            "missing_fields": [],
            "mismatched_fields": pde_mismatches,
        }
    protocol_mismatches = [key for key in PROTOCOL_FIELDS if local[key] != literature[key]]
    if protocol_mismatches:
        return {
            "status": "same_pde_different_protocol",
            "missing_fields": [],
            "mismatched_fields": protocol_mismatches,
        }
    return {"status": "directly_comparable", "missing_fields": [], "mismatched_fields": []}


def summarize_independent_runs(values: Iterable[float]) -> dict[str, float | int]:
    """Summarize run-level values; callers must pass one value per independent seed/run."""
    samples = [float(value) for value in values]
    if not samples or not all(math.isfinite(value) for value in samples):
        raise ValueError("one or more finite values per independent run are required")
    return {
        "n_runs": len(samples),
        "mean": statistics.fmean(samples),
        "sample_std": statistics.stdev(samples) if len(samples) > 1 else 0.0,
        "median": statistics.median(samples),
        "min": min(samples),
        "max": max(samples),
    }


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _float(value: str | None) -> float | None:
    if value in (None, ""):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _metric_record(
    *,
    problem: str,
    architecture: str,
    metric: str,
    values: list[float],
    provenance: str,
    unit: str = "",
) -> dict[str, Any]:
    record = {
        "problem": problem,
        "architecture": architecture,
        "metric": metric,
        "unit": unit,
        "provenance": provenance,
        **summarize_independent_runs(values),
    }
    return record


def collect_observed_signals(repo_root: Path) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Compile selected, already-measured signals and hashes from local campaign artifacts."""
    root = repo_root.resolve()
    records: list[dict[str, Any]] = []
    source_paths = {
        "screen_primary": root / "var/agent_runs/heat-burgers-architecture-screen/analysis/primary-summary.csv",
        "screen_signals": root / "var/agent_runs/heat-burgers-architecture-screen/analysis/signal-summary.csv",
        "burgers_residual_error": root / "var/agent_runs/burgers-rad-exp006/audit/localization.csv",
        "allen_cahn": root / "var/agent_runs/allen-cahn-exp003/analysis/quantitative-signals.csv",
    }
    missing = [str(path) for path in source_paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("required completed campaign artifact missing: " + ", ".join(missing))

    primary = _read_csv(source_paths["screen_primary"])
    selected_metrics = {
        "heat/field_relative_l2": ("relative field L2", "dimensionless"),
        "heat/residual_rmse_abs": ("PDE residual RMSE", "PDE residual units"),
        "burgers/field_relative_l2": ("relative field L2", "dimensionless"),
        "burgers/front_field_rmse_normalized": ("front-region field RMSE", "dimensionless"),
        "burgers/residual_rmse_abs": ("PDE residual RMSE", "PDE residual units"),
    }
    for row in primary:
        metric = row.get("signal", "")
        if metric not in selected_metrics:
            continue
        records.append(
            {
                "problem": metric.split("/", 1)[0],
                "architecture": row.get("architecture", row.get("model_id", "unknown")),
                "metric": selected_metrics[metric][0],
                "unit": row.get("unit") or selected_metrics[metric][1],
                "provenance": "EXP-0005 primary-summary.csv; mean/std are precomputed over seed-level values",
                "n_runs": int(row["n"]),
                "mean": float(row["mean"]),
                "sample_std": float(row["std"]),
                "median": float(row["median"]),
                "min": float(row["min"]),
                "max": float(row["max"]),
            }
        )

    # Keep instrument coverage and RBF basis statistics as explicit measured signals.
    screen_signals = _read_csv(source_paths["screen_signals"])
    for row in screen_signals:
        signal = row.get("signal", "")
        if not any(term in signal for term in ("effective_basis_count", "occupancy_below_0_01_count", "sensitivity/", "sampling/")):
            continue
        problem = signal.split("/", 1)[0]
        records.append(
            {
                "problem": problem,
                "architecture": row.get("architecture", row.get("model_id", "unknown")),
                "metric": signal,
                "unit": row.get("unit", ""),
                "provenance": "EXP-0005 signal-summary.csv; mean/std over three independent seeds",
                "n_runs": int(row["n"]),
                "mean": float(row["mean"]),
                "sample_std": float(row["std"]),
                "median": float(row["median"]),
                "min": float(row["min"]),
                "max": float(row["max"]),
            }
        )

    ac_rows = _read_csv(source_paths["allen_cahn"])
    ac_signal_columns = [
        "diagnostic_rbf_layer_0_effective_basis_count",
        "diagnostic_rbf_layer_1_effective_basis_count",
        "diagnostic_rbf_layer_2_effective_basis_count",
        "diagnostic_rbf_layer_1_occupancy_below_0_01_count",
        "diagnostic_rbf_layer_2_occupancy_below_0_01_count",
        "diagnostic_autodiff_uxx_rms",
        "diagnostic_autodiff_uxx_abs_p95",
        "diagnostic_residual_normalized_rms",
        "diagnostic_residual_normalized_abs_p95",
        "diagnostic_sensitivity_l2_p95",
        "sampling_sobol_centered_discrepancy",
        "sampling_heldout_to_training_ratio",
        "training_training_and_diagnostics_sec",
        "training_diagnostic_sec",
        "training_steps",
        "training_parameters",
        "validation_heldout_normalized_residual_rmse",
        "validation_heldout_normalized_residual_abs_p95",
        "validation_energy_relative_change",
        "validation_energy_increase_count",
        "validation_boundary_trace_max_abs_error",
        "resource_seed_campaign_gpu_utilization_pct_mean",
    ]
    ac_units = {
        "diagnostic_rbf_layer_0_effective_basis_count": "bases efectivas",
        "diagnostic_rbf_layer_1_effective_basis_count": "bases efectivas",
        "diagnostic_rbf_layer_2_effective_basis_count": "bases efectivas",
        "diagnostic_rbf_layer_1_occupancy_below_0_01_count": "bases",
        "diagnostic_rbf_layer_2_occupancy_below_0_01_count": "bases",
        "diagnostic_autodiff_uxx_rms": "u/unidad_espacial²",
        "diagnostic_autodiff_uxx_abs_p95": "u/unidad_espacial²",
        "diagnostic_residual_normalized_rms": "adimensional",
        "diagnostic_residual_normalized_abs_p95": "adimensional",
        "diagnostic_sensitivity_l2_p95": "norma L2",
        "sampling_sobol_centered_discrepancy": "adimensional",
        "sampling_heldout_to_training_ratio": "adimensional",
        "training_training_and_diagnostics_sec": "s",
        "training_diagnostic_sec": "s",
        "training_steps": "pasos",
        "training_parameters": "parámetros",
        "validation_heldout_normalized_residual_rmse": "adimensional",
        "validation_heldout_normalized_residual_abs_p95": "adimensional",
        "validation_energy_relative_change": "adimensional",
        "validation_energy_increase_count": "intervalos",
        "validation_boundary_trace_max_abs_error": "adimensional",
        "resource_seed_campaign_gpu_utilization_pct_mean": "% de campaña",
    }
    for column in ac_signal_columns:
        values = [_float(row.get(column)) for row in ac_rows]
        finite_values = [value for value in values if value is not None]
        if finite_values:
            records.append(
                _metric_record(
                    problem="allen_cahn",
                    architecture="RBF-PIKAN fija",
                    metric=column,
                    values=finite_values,
                    provenance="EXP-0003 quantitative-signals.csv; n is independent seeds",
                    unit=ac_units.get(column, ""),
                )
            )

    residual_rows = [
        row
        for row in _read_csv(source_paths["burgers_residual_error"])
        if row.get("model") == "rbf_fixed"
        and row.get("problem") == "burgers"
        and row.get("scope") == "interior"
    ]
    for metric, label in (
        ("spearman", "abs residual vs absolute field error Spearman"),
        ("error_enrichment", "top residual decile field-error enrichment"),
        ("top_decile_overlap", "top residual decile overlap with top error decile"),
    ):
        vals = [_float(row.get(metric)) for row in residual_rows]
        finite_values = [value for value in vals if value is not None]
        if finite_values:
            records.append(
                _metric_record(
                    problem="burgers",
                    architecture="RBF-PIKAN fija",
                    metric=label,
                    values=finite_values,
                    provenance="EXP-0005 held-out post-hoc localization audit; 3 seeds, points are not replicates",
                    unit="dimensionless",
                )
            )

    hashes = {
        key: hashlib.sha256(path.read_bytes()).hexdigest()
        for key, path in source_paths.items()
    }
    return records, hashes


def _format_metric(value: Any) -> str:
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def render_markdown(records: list[dict[str, Any]], hashes: dict[str, str]) -> str:
    lines = [
        "# Baseline exploratorio de señales RBF-PIKAN frente a métodos frontera",
        "",
        "**Corte:** 2026-09-24. **Alcance:** análisis de artefactos ya ejecutados; no inicia entrenamiento.",
        "**Autoridad:** descriptivo. No hay aceptación científica ni claim SOTA.",
        "",
        "## Propósito del baseline",
        "",
        "Este baseline caracteriza señales observables en RBF-PIKAN y determina qué hipótesis de focalización o diagnóstico merecen una corrida controlada. El MLP tanh es una referencia contextual. B-spline se conserva como control de otra base explícita, necesario para probar si la geometría RBF añade valor interpretativo más allá de que el modelo tenga cualquier base local.",
        "",
        "La motivación de RBF es estructural y medible: con centros y anchos fijos, cada activación puede enlazarse con una región del dominio y con distancias normalizadas a centros. Esto vuelve medibles ocupación, huella, redundancia, sensibilidad y estabilidad entre semillas. No demuestra superioridad de precisión ni interpretación. B-spline también tiene bases explícitas y soporte local; un MLP puede analizarse por sensibilidad aunque no exponga bases geométricas equivalentes.",
        "",
        "## Resultados locales por semilla",
        "",
        "Los valores de EXP-0005 son medias y desviaciones estándar entre tres semillas; las celdas y tiempos dentro de una corrida no cuentan como muestras independientes. Allen–Cahn reporta tres semillas, pero el error de campo sigue bloqueado por la referencia numérica pendiente; su residual y energía no son sustitutos del error de solución.",
        "",
        "| Problema | Arquitectura | Señal medida | n semillas | Media | DE muestral | Rango | Unidad |",
        "|---|---|---|---:|---:|---:|---:|---|",
    ]
    for row in records:
        metric = row["metric"]
        if metric not in {
            "relative field L2",
            "front-region field RMSE",
            "PDE residual RMSE",
            "abs residual vs absolute field error Spearman",
            "top residual decile field-error enrichment",
            "validation_heldout_normalized_residual_rmse",
            "validation_heldout_normalized_residual_abs_p95",
            "validation_energy_relative_change",
            "training_training_and_diagnostics_sec",
        }:
            continue
        lines.append(
            "| {problem} | {architecture} | {metric} | {n} | {mean} | {std} | {low}–{high} | {unit} |".format(
                problem=row["problem"],
                architecture=row["architecture"],
                metric=metric,
                n=row["n_runs"],
                mean=_format_metric(row["mean"]),
                std=_format_metric(row["sample_std"]),
                low=_format_metric(row["min"]),
                high=_format_metric(row["max"]),
                unit=row["unit"],
            )
        )
    lines += [
        "",
        "### Lectura de las señales actuales",
        "",
        "- En el screening pareado de tres semillas y parámetros casi iguales, B-spline logra menor error relativo L2 medio en Heat (2.30e-5 frente a 8.88e-5 de RBF), mientras RBF queda delante en Burgers (5.40e-2 frente a 1.16e-1 del MLP contextual y 2.21e-1 de B-spline). Es evidencia exploratoria por contrato, no una clasificación universal ni la razón única para elegir la arquitectura.",
        "- En Burgers, el piloto estima Spearman entre residual absoluto y error absoluto de 0.118–0.296; el decil de mayor residual contiene 2.60–4.84 veces el error medio. Son asociaciones descriptivas con n=3 semillas, no una prueba causal ni señal confirmatoria para guiar puntos.",
        "- En Allen–Cahn, la energía baja en las cinco muestras temporales de cada semilla y la energía relativa cambia aproximadamente −0.88; el residual held-out RMS varía entre semillas. Eso valida instrumentación física básica, pero no localiza errores de interfaz porque no hay campo de referencia habilitado.",
        "- La pantalla de arquitecturas igualó parámetros aproximadamente (RBF 3,648; B-spline 3,648; MLP 3,655). RBF y B-spline registran bases/ocupación explícitas; MLP queda como orientación contextual. Estas métricas describen lo que el sistema puede inspeccionar, no prueban que RBF explique mejor una salida.",
        "- Un resultado útil de interpretabilidad requerirá tres chequeos: repetibilidad entre semillas, predicción de error en validación independiente, y ablación/perturbación de las bases señaladas con cambio de campo/residual en la dirección predicha.",
        "",
        "## Contraste con resultados SOTA/frontera publicados",
        "",
        "Las cifras publicadas se conservan en filas separadas. No se calculan deltas entre artículos y AF_PIKANs salvo que coincidan ecuación, dominio, condiciones, parámetros, métrica, malla de evaluación, arquitectura, optimizador, puntos y presupuesto. Esta puerta evita comparar el valor de una métrica bajo contratos diferentes.",
        "",
        "| Problema | Método publicado | Score publicado y señal local | Comparabilidad con AF_PIKANs | Señal que motiva explorar |",
        "|---|---|---|---|---|",
    ]
    for entry in LITERATURE_COMPARATORS:
        lines.append(
            f"| {entry['problem']} | [{entry['method']}]({entry['source']}) ({entry['year']}) | Publicado: {entry.get('published_primary_score', entry['evidence'])} Local: {entry.get('local_metric', 'sin medida')} | `{entry['contract_status']}`; {entry['local_result_comparison']} | {entry['signal_lesson']} |"
        )
    lines += [
        "",
        "**Lectura numérica del SOTA:** el artículo vRBA publica 8.25e-9 de error relativo L2 para Burgers canónico con vRBA+SSBroyden; AF_PIKANs midió 5.40e-2 ± 3.93e-2 con RBF fija. Ambas cifras usan la misma familia de PDE y métrica, pero difieren en optimizador, features, muestreo, presupuesto y protocolo de evaluación; el cociente entre ellas no es un efecto experimental ni un head-to-head. RAMS reporta 0.181→0.010 con movimiento de muestras en diez repeticiones, igualmente como referencia de método, no como delta frente al RBF local. Para Allen–Cahn, el artículo publica 1.88e-6, pero usa otra difusividad/condición de frontera y el piloto no habilita error de campo. Heat tiene solución analítica local; DFG carece aún de corrida local.",
        "",
        "## Señales que pasan a la siguiente exploración",
        "",
        "| PDE | Fenómeno dominante | Señales prioritarias | Decisión habilitante |",
        "|---|---|---|---|",
    ]
    for problem, definition in PROBLEM_SIGNAL_REGISTRY.items():
        features = ", ".join(definition["problem_specific"])
        lines.append(f"| {problem} | `{definition['contract']}` | {features} | {definition['next_diagnostic']} |")
    lines += [
        "",
        "## Recomendación de la siguiente campaña",
        "",
        "1. Cerrar el baseline RBF fijo y su scorecard descriptiva con los pools Sobol ya documentados; preservar RBF como arquitectura primaria para observar señales, MLP como referencia y B-spline como control de base local.",
        "2. En Burgers, preregistrar una sola intervención contra Sobol. Priorizar una reproducción de RAD/RAR-D o RAMS/vRBA bajo presupuesto definido y conservar arquitectura RBF; medir cobertura de bases, residuales/error por banda del frente, gradiente/SNR por partición, error held-out, overhead y distribución de puntos.",
        "3. En Allen–Cahn, completar convergencia y revisión de la referencia de campo; medir error y anchura/posición de interfaz junto a energía y residual. Añadir SNR/estados de entrenamiento porque la literatura informa fases de fitting, transición y difusión que el resumen agregado actual no mide.",
        "4. Pasar a DFG/NS después de fijar FEM, malla y QoIs. No usar el score de calor o Burgers para transferir umbrales a fluidos.",
        "5. Confirmar cualquier señal elegida con al menos cinco semillas y un presupuesto igualado antes de afirmar generalidad o ventaja frente a resultados publicados.",
        "",
        "## Medición y linaje",
        "",
        "Los artefactos tabulares se generan con `python -m src.sciml.rbf_signal_baseline`. `observed-signals.csv` conserva métricas medidas por run; `literature-crosswalk.json` conserva contexto y estatus de comparabilidad; `exploratory-signals.html` muestra solo resultados locales, sin superponer puntuaciones incompatibles de papers.",
        "",
        "Entradas verificadas por SHA-256:",
        "",
    ]
    for key, digest in hashes.items():
        lines.append(f"- `{key}`: `{digest}`")
    lines += ["", "No se leyó `evals/`, no se cambió ningún grader o umbral y `scientific_accepted=false`.", ""]
    return "\n".join(lines)


def _write_plotly_report(path: Path, records: list[dict[str, Any]]) -> None:
    """Write local-only Plotly charts. Literature values are intentionally excluded."""
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        path.write_text(
            "<!doctype html><meta charset='utf-8'><h1>Señales locales</h1><p>Instale Plotly para generar gráficos interactivos.</p>",
            encoding="utf-8",
        )
        return

    models = ["RBF-PIKAN fija", "PIKAN B-spline cúbica", "PINN MLP tanh"]
    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=("Heat: error relativo L2", "Burgers: error relativo L2", "Burgers: error del frente"),
        horizontal_spacing=0.12,
    )
    specs = [
        ("heat", "relative field L2", 1),
        ("burgers", "relative field L2", 2),
        ("burgers", "front-region field RMSE", 3),
    ]
    for problem, metric, col in specs:
        for model in models:
            row = next(
                (item for item in records if item["problem"] == problem and item["metric"] == metric and item["architecture"] == model),
                None,
            )
            if row is None:
                continue
            fig.add_trace(
                go.Bar(
                    x=[model],
                    y=[row["mean"]],
                    name=model,
                    legendgroup=model,
                    showlegend=col == 1,
                    error_y={"type": "data", "array": [row["sample_std"]], "visible": True},
                    customdata=[[row["n_runs"], row["min"], row["max"]]],
                    hovertemplate="%{x}<br>media=%{y:.3g}<br>DE=%{error_y.array:.3g}<br>n=%{customdata[0]}<br>rango=%{customdata[1]:.3g}–%{customdata[2]:.3g}<extra></extra>",
                ),
                row=1,
                col=col,
            )
    fig.update_yaxes(type="log", title_text="Error (escala logarítmica)")
    fig.update_layout(
        title="Screening local EXP-0005 · métricas descriptivas por semilla",
        barmode="group",
        height=540,
        template="plotly_white",
        annotations=list(fig.layout.annotations) + [
            {"text": "3 semillas · DE muestral · no es score SOTA directo", "xref": "paper", "yref": "paper", "x": 0.5, "y": -0.19, "showarrow": False}
        ],
    )
    html_doc = "<!doctype html><html><head><meta charset='utf-8'><title>Baseline de señales RBF-PIKAN</title></head><body>"
    html_doc += "<h1>Baseline exploratorio: resultados locales</h1><p>El MLP se muestra como referencia contextual. Las cifras de literatura están en el informe con su etiqueta de comparabilidad; no se mezclan en esta escala.</p>"
    html_doc += fig.to_html(full_html=False, include_plotlyjs=True)

    basis_rows = []
    for row in records:
        metric = row["metric"]
        if "effective_basis_count" not in metric or row["architecture"] != "RBF-PIKAN fija":
            continue
        layer = next((part for part in metric.split("/") if part.startswith("layer_")), None)
        if layer is None:
            match = re.search(r"layer_(\d+)", metric)
            layer = f"layer_{match.group(1)}" if match else None
        if layer:
            basis_rows.append((row["problem"], layer, row))
    if basis_rows:
        basis_fig = go.Figure()
        colors = {"heat": "#1f77b4", "burgers": "#d62728", "allen_cahn": "#2ca02c"}
        for problem in sorted({problem for problem, _, _ in basis_rows}):
            items = [(layer, row) for prob, layer, row in basis_rows if prob == problem]
            basis_fig.add_trace(
                go.Bar(
                    x=[layer for layer, _ in items],
                    y=[row["mean"] for _, row in items],
                    name=problem,
                    marker_color=colors.get(problem),
                    error_y={"type": "data", "array": [row["sample_std"] for _, row in items], "visible": True},
                    customdata=[[row["n_runs"]] for _, row in items],
                    hovertemplate="%{fullData.name} · %{x}<br>conteo efectivo=%{y:.3g}<br>n=%{customdata[0]}<extra></extra>",
                )
            )
        basis_fig.update_layout(
            title="RBF: cantidad efectiva de bases por capa y PDE",
            xaxis_title="Capa",
            yaxis_title="Bases efectivas (media ± DE entre semillas)",
            barmode="group",
            template="plotly_white",
            height=480,
        )
        html_doc += basis_fig.to_html(full_html=False, include_plotlyjs=False)

    ac_residual = next((row for row in records if row["problem"] == "allen_cahn" and row["metric"] == "validation_heldout_normalized_residual_rmse"), None)
    ac_energy = next((row for row in records if row["problem"] == "allen_cahn" and row["metric"] == "validation_energy_relative_change"), None)
    if ac_residual and ac_energy:
        ac_fig = make_subplots(rows=1, cols=2, subplot_titles=("Allen–Cahn: residual held-out RMS", "Allen–Cahn: cambio relativo de energía"))
        ac_fig.add_trace(
            go.Bar(x=["RBF fija"], y=[ac_residual["mean"]], error_y={"type": "data", "array": [ac_residual["sample_std"]], "visible": True}, name="Residual RMS"),
            row=1, col=1,
        )
        ac_fig.add_trace(
            go.Bar(x=["RBF fija"], y=[ac_energy["mean"]], error_y={"type": "data", "array": [ac_energy["sample_std"]], "visible": True}, name="Cambio energía"),
            row=1, col=2,
        )
        ac_fig.update_yaxes(type="log", row=1, col=1)
        ac_fig.update_yaxes(title_text="Fracción respecto a energía inicial", row=1, col=2)
        ac_fig.update_layout(title="Allen–Cahn: métricas físicas disponibles (n=3 semillas)", template="plotly_white", showlegend=False, height=430)
        html_doc += ac_fig.to_html(full_html=False, include_plotlyjs=False)
        html_doc += "<p>Allen–Cahn: residual y disipación son diagnósticos físicos; no representan error de campo, que sigue bloqueado hasta converger la referencia.</p>"
    html_doc += "</body></html>"
    path.write_text(html_doc, encoding="utf-8")


def build_signal_baseline(repo_root: Path, output_dir: Path) -> dict[str, Any]:
    records, hashes = collect_observed_signals(repo_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    observed_path = output_dir / "observed-signals.csv"
    fields = [
        "problem", "architecture", "metric", "unit", "n_runs", "mean", "sample_std", "median", "min", "max", "provenance"
    ]
    with observed_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in records)

    (output_dir / "literature-crosswalk.json").write_text(
        json.dumps(
            {
                "policy": "Published numeric outcomes are context only unless the complete PDE and experimental contract matches.",
                "required_match_fields": list(REQUIRED_COMPARISON_FIELDS),
                "comparators": LITERATURE_COMPARATORS,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (output_dir / "architecture-observability.json").write_text(
        json.dumps(ARCHITECTURE_PROFILES, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output_dir / "signal-registry.json").write_text(
        json.dumps(
            {"common_families": COMMON_SIGNAL_FAMILIES, "problem_signals": PROBLEM_SIGNAL_REGISTRY},
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    report_path = output_dir / "README.md"
    report_path.write_text(render_markdown(records, hashes), encoding="utf-8")
    _write_plotly_report(output_dir / "exploratory-signals.html", records)

    result = {
        "status": "descriptive_exploration",
        "records": len(records),
        "problems_with_local_runs": sorted({row["problem"] for row in records}),
        "scientific_accepted": False,
        "direct_literature_head_to_head": False,
        "input_sha256": hashes,
        "artifacts": [
            "README.md",
            "observed-signals.csv",
            "literature-crosswalk.json",
            "architecture-observability.json",
            "signal-registry.json",
            "exploratory-signals.html",
        ],
    }
    (output_dir / "build-manifest.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("var/agent_runs/rbf-signal-baseline-20260924"),
        help="output path relative to repo root unless absolute",
    )
    args = parser.parse_args()
    out = args.out if args.out.is_absolute() else args.repo_root / args.out
    print(json.dumps(build_signal_baseline(args.repo_root, out), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
