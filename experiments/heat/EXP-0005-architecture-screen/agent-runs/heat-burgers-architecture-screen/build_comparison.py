"""Build a reproducible EXP-0005 architecture comparison from validated runs."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.utils import PlotlyJSONEncoder

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "var/agent_runs/heat-burgers-architecture-screen"
OUTPUT = BASE / "analysis"
NOTEBOOK = ROOT / "notebooks/experiments/heat-burgers-architecture-screen-comparison.ipynb"
MODELS = {
    "rbf_fixed": "RBF-PIKAN fija",
    "mlp_tanh": "PINN MLP tanh",
    "bspline_cubic": "PIKAN B-spline cúbica",
}
PRIMARY = {
    "Heat: error relativo L2": ("heat/field_relative_l2", "dimensionless"),
    "Heat: RMSE residual PDE": ("heat/residual_rmse_abs", "PDE residual units"),
    "Burgers: error relativo L2": ("burgers/field_relative_l2", "dimensionless"),
    "Burgers: RMSE del frente normalizado": (
        "burgers/front_field_rmse_normalized", "dimensionless"
    ),
    "Burgers: RMSE residual PDE": ("burgers/residual_rmse_abs", "PDE residual units"),
    "Heat: discrepancia Sobol centrada": (
        "heat/signals/sampling/sobol_centered_discrepancy", "dimensionless"
    ),
    "Heat: distancia vecino más cercano mediana": (
        "heat/signals/sampling/nearest_neighbor_median", "normalized coordinate distance"
    ),
    "Burgers: discrepancia Sobol centrada": (
        "burgers/signals/sampling/sobol_centered_discrepancy", "dimensionless"
    ),
    "Burgers: distancia vecino más cercano mediana": (
        "burgers/signals/sampling/nearest_neighbor_median", "normalized coordinate distance"
    ),
    "Campaña (Heat + Burgers): utilización GPU media": (
        "burgers/signals/resource_seed_campaign/utilization_gpu_pct_mean", "%"
    ),
    "Campaña (Heat + Burgers): memoria GPU máxima": (
        "burgers/signals/resource_seed_campaign/memory_used_mib_max", "MiB"
    ),
}
PLOT_METRICS = (
    ("heat/field_relative_l2", "Heat — error relativo L2", "Error relativo L2"),
    ("burgers/field_relative_l2", "Burgers — error relativo L2", "Error relativo L2"),
    (
        "burgers/front_field_rmse_normalized",
        "Burgers — RMSE del frente normalizado",
        "RMSE normalizado del frente",
    ),
)


def _report_path(model_id: str) -> Path:
    root = BASE / "models" / model_id
    if model_id == "rbf_fixed":
        reports = list((root / "download-bundle-logscale-v1").glob("pikan-*/evaluation.json"))
        if len(reports) != 1:
            raise ValueError(f"Expected one validated RBF evaluation report; found {len(reports)}")
        return reports[0]
    return root / "download-bundle-logscale-v1/archive-evaluation.json"


def collect_results() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    seed_rows = []
    signal_rows = []
    provenance = {}
    for model_id, label in MODELS.items():
        model_root = BASE / "models" / model_id
        report = json.loads(_report_path(model_id).read_text(encoding="utf-8"))
        if report.get("execution_valid") is not True or len(report.get("runs", [])) != 3:
            raise ValueError(f"{model_id} does not have three valid completed seeds")
        bundle_root = model_root / "bundle-logscale-v1"
        bundle = json.loads((bundle_root / "bundle.json").read_text(encoding="utf-8"))
        manifest = json.loads((bundle_root / "manifest.json").read_text(encoding="utf-8"))
        model_spec = manifest["model"]
        runs_root = model_root / "download-extracted-bundle-logscale-v1/runs"
        sync = json.loads((model_root / "wandb-sync.json").read_text(encoding="utf-8"))
        if sync.get("status") != "synced" or sync.get("synced_count") != 6:
            raise ValueError(f"W&B did not sync all six runs for {model_id}")
        provenance[model_id] = {
            "label": label,
            "bundle_id": bundle["bundle_id"],
            "parameters": int(model_spec["expected_parameters"]),
            "execution_valid": report["execution_valid"],
            "wandb_status": sync["status"],
            "wandb_runs_synced": sync["synced_count"],
            "evaluator_thresholds_pass": report.get("thresholds_pass"),
            "scientific_accepted": report.get("scientific_accepted", False),
        }
        for seed in (101, 102, 103):
            seed_root = runs_root / f"seed-{seed}"
            metrics = json.loads((seed_root / "metrics.json").read_text(encoding="utf-8"))
            summary = json.loads((seed_root / "run_summary.json").read_text(encoding="utf-8"))
            if summary.get("result", {}).get("status") != "completed":
                raise ValueError(f"{model_id} seed {seed} did not complete")
            row = {
                "model_id": model_id,
                "architecture": label,
                "parameters": int(model_spec["expected_parameters"]),
                "seed": seed,
                "gpu_slot": summary.get("environment", {}).get("gpu_slot"),
                "elapsed_sec": float(summary["result"]["elapsed_sec"]),
                **metrics,
            }
            seed_rows.append(row)
            for name, value in metrics.items():
                if isinstance(value, (int, float)) and np.isfinite(value):
                    problem, _, _ = name.partition("/")
                    if "resource_seed_campaign" in name:
                        problem = "campaign"
                    signal_rows.append({
                        "model_id": model_id,
                        "architecture": label,
                        "parameters": int(model_spec["expected_parameters"]),
                        "seed": seed,
                        "problem": problem,
                        "signal": name,
                        "value": float(value),
                        "unit": _unit(name),
                    })
            seed_rows[-1]["run_status"] = summary["result"]["status"]
    return pd.DataFrame(seed_rows), pd.DataFrame(signal_rows), provenance


def _unit(name: str) -> str:
    key = name.lower()
    if "_count" in key or key.endswith("_points") or "parameters" in key:
        return "count"
    if "elapsed_sec" in key or key.endswith("_sec"):
        return "s"
    if "memory_used_mib" in key or key.endswith("_mib"):
        return "MiB"
    if "utilization_gpu_pct" in key or key.endswith("_pct"):
        return "%"
    if "nearest_neighbor" in key:
        return "normalized coordinate distance"
    if "sobol_centered_discrepancy" in key:
        return "dimensionless"
    if "residual" in key:
        return "PDE residual units"
    return "dimensionless"


def summarize(seed_results: pd.DataFrame, signal_rows: pd.DataFrame):
    primary_rows = []
    for model_id, label in MODELS.items():
        model_rows = seed_results[seed_results["model_id"] == model_id]
        for label_metric, (metric, unit) in PRIMARY.items():
            values = pd.to_numeric(model_rows[metric], errors="coerce").dropna().to_numpy()
            primary_rows.append({
                "architecture": label,
                "model_id": model_id,
                "signal": metric,
                "description": label_metric,
                "n": int(values.size),
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)) if values.size > 1 else np.nan,
                "median": float(np.median(values)),
                "min": float(values.min()),
                "max": float(values.max()),
                "unit": unit,
            })
    summary = pd.DataFrame(primary_rows)
    grouped = signal_rows.groupby(
        ["model_id", "architecture", "parameters", "problem", "signal", "unit"],
        as_index=False,
    )["value"].agg(n="count", mean="mean", std="std", median="median", min="min", max="max")
    return summary, grouped


def _performance_figure(seed_results: pd.DataFrame) -> go.Figure:
    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=tuple(item[1] for item in PLOT_METRICS),
        horizontal_spacing=0.11,
    )
    colors = {"rbf_fixed": "#3366cc", "mlp_tanh": "#dc3912", "bspline_cubic": "#109618"}
    for col, (metric, _, y_title) in enumerate(PLOT_METRICS, start=1):
        for model_id, label in MODELS.items():
            values = seed_results[seed_results["model_id"] == model_id]
            fig.add_trace(go.Scatter(
                x=[label] * len(values),
                y=values[metric].astype(float),
                text=[f"seed {seed}" for seed in values["seed"]],
                mode="markers",
                name=label,
                legendgroup=model_id,
                marker={"color": colors[model_id], "size": 10},
                showlegend=(col == 1),
                hovertemplate="%{x}<br>%{text}<br>%{y:.3g}<extra></extra>",
            ), row=1, col=col)
            numeric = values[metric].astype(float)
            fig.add_trace(go.Scatter(
                x=[label], y=[numeric.mean()], mode="markers",
                marker={"color": colors[model_id], "size": 15, "symbol": "diamond"},
                error_y={"type": "data", "array": [numeric.std(ddof=1)], "visible": True},
                name=f"{label}: mean ± SD", legendgroup=model_id,
                showlegend=False,
                hovertemplate="%{x}<br>mean ± SD<br>%{y:.3g}<extra></extra>",
            ), row=1, col=col)
        fig.update_yaxes(type="log", title_text=y_title, row=1, col=col)
        fig.update_xaxes(tickangle=-20, row=1, col=col)
    fig.update_layout(
        title="EXP-0005 · errores de validación independiente (tres semillas)",
        template="plotly_white", height=570, width=1550,
        yaxis_type="log", hovermode="closest",
    )
    return fig


def _summary_table_html(summary: pd.DataFrame) -> str:
    view = summary[summary["signal"].isin([item[0] for item in PLOT_METRICS])].copy()
    view["media ± SD"] = view.apply(lambda r: f"{r['mean']:.4g} ± {r['std']:.2g}", axis=1)
    view = view[["architecture", "description", "n", "media ± SD", "min", "max", "unit"]]
    return view.to_html(index=False, border=0, float_format=lambda x: f"{x:.4g}", classes="data")


def _make_notebook(seed_results: pd.DataFrame, summary: pd.DataFrame, signals: pd.DataFrame, fig: go.Figure):
    def markdown(text):
        return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}

    def data_output(html: str, plain: str):
        return {"output_type": "display_data", "data": {"text/html": html, "text/plain": plain}, "metadata": {}}

    def plot_output(figure):
        return {
            "output_type": "display_data",
            "data": {
                "application/vnd.plotly.v1+json": json.loads(figure.to_json()),
                "text/html": figure.to_html(include_plotlyjs="cdn", full_html=False),
                "text/plain": "Plotly Figure — validation errors on logarithmic y scales",
            },
            "metadata": {},
        }

    setup = """from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

ROOT = Path.cwd()
if not (ROOT / 'var/agent_runs/heat-burgers-architecture-screen/analysis').exists():
    ROOT = ROOT.parents[1]
ANALYSIS = ROOT / 'var/agent_runs/heat-burgers-architecture-screen/analysis'
seed_results = pd.read_csv(ANALYSIS / 'seed-results.csv')
primary_summary = pd.read_csv(ANALYSIS / 'primary-summary.csv')
signal_summary = pd.read_csv(ANALYSIS / 'signal-summary.csv')
"""
    summary_code = "primary_summary[primary_summary['signal'].isin([\n" + ",\n".join(
        repr(metric) for metric, _, _ in PLOT_METRICS
    ) + "])][['architecture', 'description', 'n', 'mean', 'std', 'min', 'max', 'unit']]"
    summary_view = summary[summary["signal"].isin([item[0] for item in PLOT_METRICS])]
    signals_filter = signals[
        signals["signal"].str.contains("sampling|residual|resource_seed_campaign", regex=True)
    ][["architecture", "problem", "signal", "n", "mean", "std", "min", "max", "unit"]]
    plot_code = """colors = {'rbf_fixed':'#3366cc', 'mlp_tanh':'#dc3912', 'bspline_cubic':'#109618'}
metrics = [
    ('heat/field_relative_l2', 'Heat — error relativo L2'),
    ('burgers/field_relative_l2', 'Burgers — error relativo L2'),
    ('burgers/front_field_rmse_normalized', 'Burgers — RMSE del frente normalizado'),
]
fig = make_subplots(rows=1, cols=3, subplot_titles=[label for _, label in metrics])
for col, (metric, _) in enumerate(metrics, start=1):
    for model_id, group in seed_results.groupby('model_id', sort=False):
        for row in group.to_dict('records'):
            fig.add_trace(go.Scatter(x=[row['architecture']], y=[row[metric]],
                mode='markers', name=f\"{row['architecture']} · semilla {int(row['seed'])}\",
                legendgroup=model_id, marker={'color':colors[model_id], 'size':10}),
                row=1, col=col)
        values = group[metric]
        fig.add_trace(go.Scatter(x=[group.iloc[0]['architecture']], y=[values.mean()],
            mode='markers', showlegend=False, marker={'color':colors[model_id], 'size':15, 'symbol':'diamond'},
            error_y={'type':'data', 'array':[values.std(ddof=1)], 'visible':True}), row=1, col=col)
    fig.update_yaxes(type='log', title_text='Error (escala log)', row=1, col=col)
    fig.update_xaxes(tickangle=-20, row=1, col=col)
fig.update_layout(template='plotly_white', height=570, width=1550,
    title='EXP-0005 · errores de validación independiente', hovermode='closest')
fig
"""
    nb = {
        "cells": [
            markdown(
                "# EXP-0005 — comparación exploratoria Heat + Burgers\n\n"
                "Tres arquitecturas con parámetros cercanos (3.648–3.655), tres semillas y los mismos pools Sobol scrambled fijos. "
                "Heat es conducción 1D en una varilla con referencia analítica; Burgers es el benchmark viscoso transitorio de Raissi/PINNacle. "
                "Las tablas son mediciones por semilla; los puntos diamante muestran media ± desviación estándar.\n\n"
                "Esta evaluación valida ejecución y artefactos. No se preregistraron umbrales numéricos, por lo que no cuenta como aceptación científica ni como claim SOTA. "
                "Campos con signo permanecen lineales; pérdidas, magnitudes de gradiente/residual y errores positivos se muestran en escala logarítmica. "
                "El error absoluto usa `log10(|error|)`; los ceros exactos no reciben epsilon.\n\n"
                "Datos completos: [`quantified-signals.csv`](../../var/agent_runs/heat-burgers-architecture-screen/analysis/quantified-signals.csv), "
                "[`signal-summary.csv`](../../var/agent_runs/heat-burgers-architecture-screen/analysis/signal-summary.csv), "
                "[`results.html`](../../var/agent_runs/heat-burgers-architecture-screen/analysis/architecture-screen-comparison.html)."
            ),
            {"cell_type": "code", "execution_count": 1, "metadata": {},
             "source": setup.splitlines(keepends=True), "outputs": []},
            markdown("## Métricas principales\n\nErrores positivos graficados en eje log. `n=3` semillas por arquitectura."),
            {"cell_type": "code", "execution_count": 2, "metadata": {},
             "source": summary_code.splitlines(keepends=True),
             "outputs": [data_output(summary_view.to_html(index=False, border=0), summary_view.to_string(index=False))]},
            {"cell_type": "code", "execution_count": 3, "metadata": {},
             "source": plot_code.splitlines(keepends=True), "outputs": [plot_output(fig)]},
            markdown("## Señales medidas de puntos, residual y recursos\n\nLa tabla completa conserva una fila por señal, problema, arquitectura y semilla; aquí se muestra el agregado descriptivo."),
            {"cell_type": "code", "execution_count": 4, "metadata": {},
             "source": "signal_summary[signal_summary['signal'].str.contains('sampling|residual|resource_seed_campaign', regex=True)][['architecture','problem','signal','n','mean','std','min','max','unit']]".splitlines(keepends=True),
             "outputs": [data_output(signals_filter.to_html(index=False, border=0), signals_filter.head(80).to_string(index=False))]},
            markdown(
                "## Lectura limitada\n\n"
                "El screening favorece B-spline para el error de Heat y RBF para los dos errores de Burgers seleccionados. "
                "Es una señal de especialización por problema, no una conclusión general: cada estimación usa tres semillas y no se realizó ajuste de hiperparámetros ni comparación con métodos SOTA externos. "
                "La utilización y memoria GPU se informan como señales agregadas de campaña (Heat + Burgers), no por PDE; la campaña usó slots GPU 0/1 asignados por semilla."
            ),
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
            "research": {"role": "exploration", "campaign": "EXP-0005"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    for index, cell in enumerate(nb["cells"], start=1):
        cell["id"] = f"exp0005-cell-{index}"
    NOTEBOOK.parent.mkdir(parents=True, exist_ok=True)
    NOTEBOOK.write_text(json.dumps(nb, ensure_ascii=False, cls=PlotlyJSONEncoder), encoding="utf-8")


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    seed_results, signal_rows, provenance = collect_results()
    primary, all_signals = summarize(seed_results, signal_rows)
    seed_results.to_csv(OUTPUT / "seed-results.csv", index=False)
    signal_rows.to_csv(OUTPUT / "quantified-signals.csv", index=False)
    primary.to_csv(OUTPUT / "primary-summary.csv", index=False)
    all_signals.to_csv(OUTPUT / "signal-summary.csv", index=False)
    (OUTPUT / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")

    figure = _performance_figure(seed_results)
    _make_notebook(seed_results, primary, all_signals, figure)
    primary_html = _summary_table_html(primary)
    sample_columns = [
        "heat/signals/sampling/interior_count",
        "burgers/signals/sampling/interior_count",
        "heat/signals/sampling/sobol_centered_discrepancy",
        "burgers/signals/sampling/sobol_centered_discrepancy",
        "burgers/signals/resource_seed_campaign/utilization_gpu_pct_mean",
        "burgers/signals/resource_seed_campaign/memory_used_mib_max",
    ]
    signal_view = all_signals[all_signals["signal"].isin(sample_columns)]
    html = """<!doctype html><html lang=\"es\"><head><meta charset=\"utf-8\"><title>EXP-0005 Architecture screen</title>
<style>body{font:16px system-ui,sans-serif;max-width:1600px;margin:2rem auto;padding:0 1rem;color:#18202a}table{border-collapse:collapse;width:100%;margin:1rem 0 2rem}th,td{border:1px solid #d7dde5;padding:.45rem;text-align:left}th{background:#edf2f7}.note{background:#f1f5f9;padding:1rem;border-radius:8px}</style></head><body>
<h1>EXP-0005 · screening exploratorio Heat + Burgers</h1>
<p class=\"note\">Tres arquitecturas × dos PDE × tres semillas. Pools Sobol scrambled fijos y pareados. La ejecución y los artefactos son válidos; no se definieron umbrales de aceptación. Esto no constituye claim SOTA.</p>
<h2>Errores independientes por semilla</h2>""" + primary_html + figure.to_html(include_plotlyjs="cdn", full_html=False) + """
<h2>Señales cuantificadas de puntos y GPU</h2>""" + signal_view.to_html(index=False, border=0, float_format=lambda x: f"{x:.4g}", classes="data") + """
<p>Campos con signo en lineal; pérdidas y magnitudes positivas en log; error absoluto en escala de color log10. Las señales de recurso se agregan a nivel de campaña (Heat + Burgers). Datos completos y metodología en los CSV/JSON del mismo directorio.</p>
</body></html>"""
    (OUTPUT / "architecture-screen-comparison.html").write_text(html, encoding="utf-8")
    print(json.dumps({
        "seed_rows": len(seed_results),
        "quantified_signal_rows": len(signal_rows),
        "summary_rows": len(primary),
        "notebook": str(NOTEBOOK.relative_to(ROOT)),
        "html": str((OUTPUT / "architecture-screen-comparison.html").relative_to(ROOT)),
        "wandb_synced_runs": sum(item["wandb_runs_synced"] for item in provenance.values()),
    }, indent=2))


if __name__ == "__main__":
    main()
