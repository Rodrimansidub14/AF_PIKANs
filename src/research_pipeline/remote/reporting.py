"""Quantitative tables and Plotly figures built from immutable run artifacts."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _add_prefixed(row, family, values):
    for name, value in values.items():
        if type(value) in (int, float) and np.isfinite(value):
            if name.startswith(family + "/"):
                name = name[len(family) + 1:]
            row[f"{family}_{name.replace('/', '_')}"] = float(value)


def signal_unit(name, problem=None):
    """Map a measured scalar name to an explicit SI/nondimensional unit label."""
    key = str(name).lower().replace("/", "_")
    if "energy" in key:
        return "nondimensional Allen–Cahn energy"
    if "per_sec" in key:
        return "steps/s"
    if "stop_reason_step_limit" in key:
        return "boolean flag"
    if any(token in key for token in ("_count", "_steps", "_points", "examples_seen", "parameters", "samples")):
        return "count"
    if key.endswith("_sec") or "elapsed_sec" in key:
        return "s"
    if key.endswith("_bytes"):
        return "bytes"
    if key.endswith("_mib") or "_memory_" in key and "_mib" in key:
        return "MiB"
    if ("utilization" in key and "pct" in key) or key.endswith("_pct"):
        return "%"
    if "power_draw_w" in key or key.endswith("_w"):
        return "W"
    if "temperature_c" in key or key.endswith("_c"):
        return "°C"
    if "nearest_neighbor" in key:
        return "normalized coordinate distance"
    if "coordinate_x" in key:
        return "x coordinate"
    if "coordinate_t" in key:
        return "t coordinate"
    if "sobol_centered_discrepancy" in key:
        return "dimensionless"
    if "autodiff_uxx" in key:
        return "U/L²"
    if "autodiff_ux" in key:
        return "U/L"
    if "autodiff_ut" in key:
        return "U/T"
    if "autodiff_u_" in key or key.endswith("autodiff_u_mean"):
        return "U"
    if "residual_normalized" in key or "relative_l2" in key or "normalized" in key:
        return "dimensionless"
    if "residual" in key:
        return "PDE residual units"
    if "field_rmse" in key or "field_p95" in key or "field_max" in key:
        return "U"
    if "max_abs_error" in key:
        return "U"
    if "sensitivity" in key:
        return "U per parameter (L2)"
    if "gradient" in key or "coefficient_l2" in key or "parameter_l2" in key:
        return "model parameter units"
    if "occupancy" in key or "saturation" in key or "activation" in key:
        return "dimensionless"
    if "loss" in key:
        return "dimensionless normalized loss"
    return "dimensionless"


def long_quantitative_table(table):
    """Convert the per-run wide table to measured value + unit records."""
    import pandas as pd

    records = []
    metadata = {"seed", "problem", "seed_status", "problem_status", "gpu_slot", "gpu_name"}
    for _, row in table.iterrows():
        for name, value in row.items():
            if name in metadata or type(value) not in (int, float) or not np.isfinite(value):
                continue
            family, separator, signal = name.partition("_")
            if not separator:
                family, signal = "run", name
            records.append({
                "problem": row["problem"],
                "seed": int(row["seed"]),
                "family": family,
                "signal": signal,
                "value": float(value),
                "unit": signal_unit(name, row["problem"]),
            })
    return pd.DataFrame.from_records(records)


def _history(path):
    file = Path(path)
    if not file.is_file():
        return []
    return [json.loads(line) for line in file.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def quantitative_table(runs_root):
    """One row per seed/problem, with measured scalars and unit-bearing names."""
    import pandas as pd

    root = Path(runs_root)
    records = []
    for seed_dir in sorted(root.glob("seed-*")):
        try:
            seed = int(seed_dir.name.split("-", 1)[1])
        except ValueError:
            continue
        run_summary_path = seed_dir / "run_summary.json"
        run_summary = _read_json(run_summary_path) if run_summary_path.exists() else {}
        hardware_path = seed_dir / "hardware.json"
        hardware = _read_json(hardware_path) if hardware_path.exists() else {}
        status_path = seed_dir / "problem_status.json"
        statuses = _read_json(status_path) if status_path.exists() else {}
        for problem in ("heat", "burgers", "allen_cahn"):
            folder = seed_dir / problem
            if not folder.is_dir():
                continue
            row = {
                "seed": seed,
                "problem": problem,
                "seed_status": run_summary.get("result", {}).get("status", "not available"),
                "problem_status": statuses.get(problem, {}).get("status", "not available"),
                "gpu_slot": run_summary.get("environment", {}).get("gpu_slot"),
                "gpu_name": hardware.get("gpu_name"),
            }
            sample_path = folder / "sampling_signals.json"
            diagnostic_path = folder / "diagnostic_signals.json"
            summary_path = folder / "training_summary.json"
            validation_path = folder / "validation.json"
            if sample_path.exists():
                _add_prefixed(row, "sampling", _read_json(sample_path))
            if diagnostic_path.exists():
                _add_prefixed(row, "diagnostic", _read_json(diagnostic_path))
            if summary_path.exists():
                summary = _read_json(summary_path)
                for key in ("steps", "training_and_diagnostics_sec", "diagnostic_sec", "parameters",
                            "peak_gpu_bytes"):
                    value = summary.get(key)
                    if type(value) in (int, float) and np.isfinite(value):
                        row[f"training_{key}"] = float(value)
                elapsed = summary.get("training_and_diagnostics_sec")
                steps = summary.get("steps")
                if type(elapsed) in (int, float) and elapsed > 0 and type(steps) is int:
                    row["training_steps_per_sec"] = steps / elapsed
                if summary.get("stop_reason") in ("step_limit", "time_limit"):
                    row["training_stop_reason_step_limit"] = float(summary["stop_reason"] == "step_limit")
            if type(hardware.get("gpu_total_memory_bytes")) in (int, float):
                row["resource_gpu_total_memory_mib"] = hardware["gpu_total_memory_bytes"] / 2**20
            if summary_path.exists():
                value = _read_json(summary_path).get("peak_gpu_bytes")
                if type(value) in (int, float):
                    row["resource_peak_gpu_mib"] = value / 2**20
            telemetry_path = seed_dir / "gpu_telemetry.csv"
            if telemetry_path.is_file():
                telemetry = pd.read_csv(telemetry_path)
                row["resource_seed_campaign_gpu_telemetry_samples"] = len(telemetry)
                for source, target in (
                    ("utilization_gpu_pct", "gpu_utilization_pct"),
                    ("memory_used_mib", "gpu_memory_used_mib"),
                    ("power_draw_w", "gpu_power_draw_w"),
                    ("temperature_c", "gpu_temperature_c"),
                ):
                    if source in telemetry and telemetry[source].notna().any():
                        values = telemetry[source].dropna().to_numpy(dtype=float)
                        row[f"resource_seed_campaign_{target}_mean"] = float(values.mean())
                        row[f"resource_seed_campaign_{target}_p95"] = float(np.quantile(values, .95))
                        row[f"resource_seed_campaign_{target}_max"] = float(values.max())
            histories = _history(folder / "history.jsonl")
            if histories:
                gradient = np.asarray([item["gradient_norm"] for item in histories
                                       if type(item.get("gradient_norm")) in (int, float)], dtype=float)
                if gradient.size:
                    row["training_gradient_norm_median"] = float(np.median(gradient))
                    row["training_gradient_norm_p95"] = float(np.quantile(gradient, .95))
                    row["training_gradient_norm_max"] = float(np.max(gradient))
                loss_names = sorted({key for item in histories for key in item
                                     if key == "train_loss" or key.endswith("_loss")})
                for name in loss_names:
                    values = np.asarray([item[name] for item in histories
                                         if type(item.get(name)) in (int, float)], dtype=float)
                    if values.size:
                        safe_name = name.replace("/", "_")
                        row[f"training_{safe_name}_first_recorded"] = float(values[0])
                        row[f"training_{safe_name}_last_recorded"] = float(values[-1])
            if validation_path.exists():
                validation = _read_json(validation_path).get("metrics", {})
                for key, value in validation.items():
                    if type(value) in (int, float) and np.isfinite(value):
                        row[f"validation_{key}"] = float(value)
            records.append(row)
    if not records:
        raise ValueError(f"No seed/problem result artifacts found under {root}")
    return pd.DataFrame.from_records(records).sort_values(["problem", "seed"]).reset_index(drop=True)


def aggregate_quantitative_table(table):
    """Per-problem descriptive statistics across available seeds, with sample n."""
    numeric = [name for name in table.select_dtypes(include="number").columns if name != "seed"]
    grouped = table.groupby("problem", sort=True)[list(numeric)]
    result = grouped.agg(["count", "mean", "std", "median", "min", "max"])
    return result


def training_figures(runs_root):
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go

    root = Path(runs_root)
    known_problems = ("heat", "burgers", "allen_cahn")
    seed_dirs = sorted(root.glob("seed-*"))
    present = {problem for problem in known_problems
               if any((seed_dir / problem).is_dir() for seed_dir in seed_dirs)}
    # Preserve the historical empty-campaign layout, but avoid misleading blank
    # panels when a campaign contains only one or two declared problems.
    problems = tuple(problem for problem in known_problems if problem in present)
    if not problems:
        problems = ("heat", "burgers")
    loss = make_subplots(rows=len(problems), cols=1, shared_xaxes=True,
                         subplot_titles=tuple(f"{problem.replace('_', ' ').title()}: recorded losses"
                                               for problem in problems))
    signals = make_subplots(rows=len(problems), cols=1, shared_xaxes=True,
                            subplot_titles=tuple(f"{problem.replace('_', ' ').title()}: gradient and residual signals"
                                                 for problem in problems))
    for seed_dir in sorted(root.glob("seed-*")):
        seed = seed_dir.name.replace("seed-", "")
        for row_index, problem in enumerate(problems, start=1):
            history = _history(seed_dir / problem / "history.jsonl")
            if not history:
                continue
            for name in ("train_loss", "physics_loss", "left_loss", "right_loss", "initial_loss"):
                values = _positive_log_points(history, name)
                if not values or (name != "train_loss" and problem != "burgers"):
                    continue
                x_values, y_values = zip(*values)
                loss.add_trace(go.Scatter(x=x_values, y=y_values, mode="lines",
                    name=f"seed {seed} {name}", legendgroup=f"{seed}-{name}"),
                    row=row_index, col=1)
            for name, label in (("gradient_norm", "gradient L2"),
                                ("residual/normalized_rms", "normalized residual RMS"),
                                ("residual/normalized_abs_p95", "normalized residual abs p95")):
                values = _positive_log_points(history, name)
                if not values:
                    continue
                x_values, y_values = zip(*values)
                signals.add_trace(go.Scatter(x=x_values, y=y_values, mode="lines",
                    name=f"seed {seed} {label}", legendgroup=f"{seed}-{label}"),
                    row=row_index, col=1)
    loss.update_yaxes(type="log", title_text="loss (log scale)")
    signals.update_yaxes(type="log", title_text="measured magnitude (log scale)")
    for figure in (loss, signals):
        figure.update_xaxes(title_text="training and diagnostic wall time (s)", row=len(problems), col=1)
        figure.update_layout(template="plotly_white", height=1050, hovermode="x unified")
    loss.update_layout(title="Training losses by problem and seed")
    signals.update_layout(title="Gradient and independent diagnostic residual signals")
    return {"training-loss-curves": loss, "training-internal-signals": signals}


def _positive_log_points(history, metric):
    """Keep finite positive magnitudes; zeros and signed values have no log coordinate."""
    points = []
    for item in history:
        x_value, y_value = item.get("elapsed_sec"), item.get(metric)
        if (type(x_value) in (int, float) and np.isfinite(x_value)
                and type(y_value) in (int, float) and np.isfinite(y_value)
                and x_value >= 0 and y_value > 0):
            points.append((float(x_value), float(y_value)))
    return points


def _single_history_figure(history, problem, metrics, title, y_title):
    import plotly.graph_objects as go

    figure = go.Figure()
    for metric, label in metrics:
        values = _positive_log_points(history, metric)
        if not values:
            continue
        x_values, y_values = zip(*values)
        figure.add_trace(go.Scatter(
            x=x_values, y=y_values, mode="lines", name=label,
            connectgaps=False,
        ))
    figure.update_yaxes(type="log", title_text=y_title)
    figure.update_xaxes(title_text="training and diagnostic wall time (s)")
    figure.update_layout(
        title=f"{problem.title()}: {title}", template="plotly_white",
        hovermode="x unified",
    )
    return figure


def _log_error_coloraxis(error):
    positive_mask = np.isfinite(error) & (error > 0)
    finite_positive = error[positive_mask]
    if finite_positive.size:
        log_values = np.full(error.shape, np.nan, dtype=np.float64)
        finite_log_values = np.log10(finite_positive)
        log_values[positive_mask] = finite_log_values
        lower, upper = float(finite_log_values.min()), float(finite_log_values.max())
        if lower == upper:
            lower, upper = lower - 0.5, upper + 0.5
        first_tick, last_tick = int(np.ceil(lower)), int(np.floor(upper))
        ticks = list(range(first_tick, last_tick + 1))
        if not ticks:
            ticks = [0.5 * (lower + upper)]
        ticktext = [f"{10.0 ** exponent:.1e}" for exponent in ticks]
    else:
        # All values are exact zeros: use an empty map with an explicit decade range.
        log_values = np.full(error.shape, np.nan, dtype=np.float64)
        lower, upper, ticks, ticktext = -16.0, -15.0, [-16.0], ["≤1.0e-16"]
    return log_values, {
        "cmin": lower,
        "cmax": upper,
        "colorscale": "Viridis",
        "colorbar": {
            "title": "|error| (log scale)",
            "tickmode": "array",
            "tickvals": ticks,
            "ticktext": ticktext,
        },
    }


def validation_log_figure(prediction, reference, absolute_error, xs, ts, title):
    """Plot signed solution fields linearly and absolute error in log10 color."""
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go

    error = np.asarray(absolute_error, dtype=np.float64)
    plotted_log_error, error_coloraxis = _log_error_coloraxis(error)
    figure = make_subplots(
        rows=1, cols=3,
        subplot_titles=("Prediction u(x,t)", "Reference u(x,t)",
                       "Absolute error (log color scale)"),
    )
    for column, (values, coloraxis, name) in enumerate((
        (prediction, "coloraxis", "prediction"),
        (reference, "coloraxis", "reference"),
        (plotted_log_error, "coloraxis2", "log10 absolute error"),
    ), start=1):
        figure.add_trace(go.Heatmap(
            x=ts, y=xs, z=values, coloraxis=coloraxis, name=name,
        ), row=1, col=column)
        figure.update_xaxes(title_text="time t", row=1, col=column)
        figure.update_yaxes(title_text="space x", row=1, col=column)
    low = float(min(np.nanmin(prediction), np.nanmin(reference)))
    high = float(max(np.nanmax(prediction), np.nanmax(reference)))
    figure.update_layout(
        title=title, template="plotly_white", width=1450, height=520,
        coloraxis={"cmin": low, "cmax": high,
                   "colorbar": {"title": "u (linear scale)"}},
        coloraxis2=error_coloraxis,
    )
    return figure


def exploratory_physics_figure(prediction, residual, xs, ts, title):
    """Plot a signed field and log-scaled residual without requiring truth data."""
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go

    prediction = np.asarray(prediction, dtype=np.float64)
    residual = np.asarray(residual, dtype=np.float64)
    figure = make_subplots(rows=1, cols=2,
                           subplot_titles=("Prediction u(x,t)", "|PDE residual| (log scale)"))
    figure.add_trace(go.Heatmap(x=ts, y=xs, z=prediction, colorbar={"title": "u"}), row=1, col=1)
    figure.add_trace(go.Heatmap(
        x=ts, y=xs, z=np.log10(np.maximum(np.abs(residual), 1e-16)),
        colorbar={"title": "log10 |residual|"},
    ), row=1, col=2)
    for column in (1, 2):
        figure.update_xaxes(title_text="time t", row=1, col=column)
        figure.update_yaxes(title_text="space x", row=1, col=column)
    figure.update_layout(title=title, template="plotly_white", width=1250, height=520)
    return figure


def energy_figure(times, energy, title="Allen–Cahn energy"):
    import plotly.graph_objects as go

    figure = go.Figure(go.Scatter(x=times, y=energy, mode="lines+markers", name="E[u(t)]"))
    figure.update_layout(title=title, template="plotly_white", xaxis_title="time t",
                         yaxis_title="energy E[u(t)]", width=900, height=480)
    return figure


def log_wandb_figures(run, history, validation_path, problem, output_dir=None):
    """Log explicit Plotly charts with log magnitudes to W&B and local artifacts."""
    loss_figure = _single_history_figure(
        history, problem,
        [(name, name.replace("_", " ")) for name in
         ("train_loss", "physics_loss", "left_loss", "right_loss", "initial_loss")],
        "training losses", "loss (log scale)",
    )
    signal_figure = _single_history_figure(
        history, problem,
        [("gradient_norm", "gradient L2"),
         ("residual/normalized_rms", "normalized residual RMS"),
         ("residual/normalized_abs_p95", "normalized residual abs p95")],
        "gradient and residual signals", "measured magnitude (log scale)",
    )
    figures = {"plots/training-losses-log": loss_figure,
               "plots/training-signals-log": signal_figure}
    with np.load(validation_path) as data:
        if problem == "allen_cahn":
            validation_figure = exploratory_physics_figure(
                data["prediction"], data["residual"], data["x"], data["t"],
                "Allen–Cahn exploratory field and held-out residual",
            )
            energy = energy_figure(data["energy_times"], data["energy"])
            figures["plots/validation-physics-only"] = validation_figure
            figures["plots/energy-trajectory"] = energy
        else:
            validation_figure = validation_log_figure(
                data["prediction"], data["reference"], data["absolute_error"],
                data["x"], data["t"],
                f"{problem.title()} independent validation: signed fields linear, error logarithmic",
            )
            figures["plots/validation-error-log"] = validation_figure
    if output_dir is not None:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        for name, figure in figures.items():
            filename = {
                "plots/training-losses-log": "training-losses-log.html",
                "plots/training-signals-log": "training-signals-log.html",
                "plots/validation-error-log": "validation.html",
                "plots/validation-physics-only": "validation.html",
                "plots/energy-trajectory": "energy.html",
            }[name]
            figure.write_html(output / filename, include_plotlyjs="cdn", full_html=True)
    run.log(figures)
    return figures


def field_figures(runs_root):
    figures = {}
    for seed_dir in sorted(Path(runs_root).glob("seed-*")):
        seed = seed_dir.name.replace("seed-", "")
        for problem in ("heat", "burgers", "allen_cahn"):
            path = seed_dir / problem / "validation.npz"
            if not path.is_file():
                continue
            with np.load(path) as data:
                if problem == "allen_cahn":
                    figure = exploratory_physics_figure(
                        data["prediction"], data["residual"], data["x"], data["t"],
                        f"Allen–Cahn field and residual, seed {seed}",
                    )
                else:
                    figure = validation_log_figure(
                        data["prediction"], data["reference"], data["absolute_error"],
                        data["x"], data["t"],
                        f"{problem.title()} validation fields, seed {seed}",
                    )
                figures[f"validation-fields-{problem}-seed-{seed}"] = figure
    return figures


def write_results_notebook(runs_root, analysis_dir, bundle_id):
    """Write an output-bearing analysis notebook from this campaign's saved artifacts."""
    import plotly.express as px

    runs = Path(runs_root)
    analysis = Path(analysis_dir)
    analysis.mkdir(parents=True, exist_ok=True)
    table = quantitative_table(runs)
    long_table = long_quantitative_table(table)
    model_ids = set()
    for config_path in runs.glob("seed-*/config.json"):
        model_id = _read_json(config_path).get("model", {}).get("id")
        if model_id:
            model_ids.add(str(model_id))
    model_ids = sorted(model_ids)
    model_description = ", ".join(model_ids) if model_ids else "see per-run configuration"

    def markdown(source):
        return {"cell_type": "markdown", "metadata": {},
                "source": source.splitlines(keepends=True)}

    def display_output(html, plain):
        return {"output_type": "display_data",
                "data": {"text/html": html, "text/plain": plain},
                "metadata": {}}

    def plot_output(figure, plain):
        return {"output_type": "display_data",
                "data": {"application/vnd.plotly.v1+json": json.loads(figure.to_json()),
                         "text/html": figure.to_html(include_plotlyjs="cdn", full_html=False),
                         "text/plain": plain},
                "metadata": {}}

    problems = sorted(table["problem"].unique().tolist())
    problems_label = " + ".join(problem.replace("_", " ").title() for problem in problems)
    seeds_count = int(table["seed"].nunique())
    if problems == ["allen_cahn"]:
        validation_description = (
            "residuo físico held-out, trazas duras, energía y recursos. "
            "Error de campo y L2 permanecen bloqueados; no hay referencia numérica convergida."
        )
        visual_description = (
            "Residuo absoluto y pérdidas positivos se muestran en escala logarítmica; "
            "el campo con signo y la energía conservan su escala física."
        )
    else:
        validation_description = "validación y recursos"
        visual_description = (
            "Pérdidas, gradientes y residuales positivos usan ejes log; "
            "el error absoluto usa color logarítmico. Los campos con signo conservan escala lineal."
        )
    cells = [markdown(
        f"# Resultados medidos: {problems_label}\n\n"
        f"Bundle {bundle_id}. {seeds_count} semillas exploratorias; no hay afirmación SOTA ni aceptación científica. "
        "Las tablas contienen mediciones de pools, entrenamiento, autodiferenciación, bases, "
        f"{validation_description} "
        "Cada señal numérica incluye su unidad; los datos de entrada permanecen junto a este notebook.\n\n"
        f"Arquitectura(s): {model_description}. Muestreo: Sobol scrambled fijo. "
        "Entrenamiento: PINA + PyTorch. "
        f"{visual_description} Curvas y campos: Plotly. "
        "PyVista no aplica a los campos 1D de esta campaña."
    )]

    table_code = (
        "import pandas as pd\n"
        "quantitative_signals = pd.read_csv(BASELINE_ANALYSIS / 'quantitative-signals-with-units.csv')\n"
        "quantitative_signals"
    )
    table_html = long_table.to_html(index=False, border=0)
    cells.append({
        "cell_type": "code", "execution_count": 1, "metadata": {},
        "source": table_code.splitlines(keepends=True),
        "outputs": [display_output(table_html, long_table.to_string(index=False))],
    })

    seed = int(table["seed"].iloc[0])
    for problem in problems:
        pool_path = runs / f"seed-{seed}" / problem / "pools.npz"
        with np.load(pool_path) as pool:
            points = pool["interior"].copy()
            if problem == "allen_cahn":
                heldout = pool["heldout"].copy()
        if problem == "allen_cahn":
            import pandas as pd
            plotted = pd.DataFrame({
                "x": np.r_[points[::16, 0], heldout[::32, 0]],
                "t": np.r_[points[::16, 1], heldout[::32, 1]],
                "pool": (["training"] * len(points[::16])
                         + ["held-out diagnostic"] * len(heldout[::32])),
            })
            figure = px.scatter(
                data_frame=plotted, x="x", y="t", color="pool", opacity=.55,
                labels={"x": "space coordinate x", "t": "time coordinate t"},
                title=f"Allen–Cahn independent training and held-out Sobol pools, seed {seed}",
            )
            code = (
                "import numpy as np, pandas as pd, plotly.express as px\n"
                "pools = np.load(BASELINE_RUNS / "
                f"\"seed-{seed}\" / \"{problem}\" / \"pools.npz\")\n"
                "training = pools['interior']; heldout = pools['heldout']\n"
                "points = pd.DataFrame({'x': np.r_[training[::16,0], heldout[::32,0]],\n"
                "    't': np.r_[training[::16,1], heldout[::32,1]],\n"
                "    'pool': ['training']*len(training[::16]) + ['held-out diagnostic']*len(heldout[::32])})\n"
                "figure = px.scatter(points, x='x', y='t', color='pool', opacity=.55,\n"
                "    labels={'x':'space coordinate x','t':'time coordinate t'},\n"
                f"    title='Allen–Cahn independent training and held-out Sobol pools, seed {seed}')\n"
                "figure"
            )
            section = "Puntos generados: Allen–Cahn, entrenamiento vs. held-out independiente"
        else:
            figure = px.scatter(
                x=points[::4, 0], y=points[::4, 1], opacity=.55,
                labels={"x": "space coordinate x", "y": "time coordinate t"},
                title=f"{problem.title()} interior Sobol pool, seed {seed}",
            )
            code = (
                "import numpy as np, plotly.express as px\n"
                "from pathlib import Path\n"
                "points = np.load(BASELINE_RUNS / "
                f"\"seed-{seed}\" / \"{problem}\" / \"pools.npz\")['interior']\n"
                "figure = px.scatter(x=points[::4, 0], y=points[::4, 1], opacity=.55,\n"
                "    labels={'x':'space coordinate x','y':'time coordinate t'},\n"
                f"    title='{problem.title()} interior Sobol pool, seed {seed}')\n"
                "figure"
            )
            section = f"Puntos generados: {problem.title()}"
        cells.append(markdown(f"## {section}"))
        cells.append({
            "cell_type": "code", "execution_count": len(cells) + 1, "metadata": {},
            "source": code.splitlines(keepends=True),
            "outputs": [plot_output(figure, f"Plotly scatter: {problem}, seed {seed}")],
        })

    for name, figure in training_figures(runs).items():
        source = (
            "from src.research_pipeline.remote.reporting import training_figures\n"
            "for name, figure in training_figures(BASELINE_RUNS).items():\n"
            f"    if name == {name!r}:\n"
            "        display(figure)\n"
        )
        cells.append(markdown(f"## {name.replace('-', ' ').title()}"))
        cells.append({
            "cell_type": "code", "execution_count": len(cells) + 1, "metadata": {},
            "source": source.splitlines(keepends=True),
            "outputs": [plot_output(figure, f"Plotly figure: {name}")],
        })

    for name, figure in field_figures(runs).items():
        source = (
            "from src.research_pipeline.remote.reporting import field_figures\n"
            "for name, figure in field_figures(BASELINE_RUNS).items():\n"
            f"    if name == {name!r}:\n"
            "        display(figure)\n"
        )
        cells.append(markdown(f"## {name.replace('-', ' ').title()}"))
        cells.append({
            "cell_type": "code", "execution_count": len(cells) + 1, "metadata": {},
            "source": source.splitlines(keepends=True),
            "outputs": [plot_output(figure, f"Plotly field map: {name}")],
        })

    cells.append(markdown(
        "## Interpretación\n\n"
        "Las cifras anteriores son medidas directas. Las estadísticas entre semillas son descriptivas (n=3). "
        "No se abre la evaluación final, no se declaran umbrales y no se convierte una correlación de señales "
        "en explicación causal. PyVista se reserva para futuras geometrías o mallas explícitas."
    ))

    base_path_code = '''import os
from pathlib import Path
if os.environ.get("AF_PIKAN_RUNS"):
    BASELINE_RUNS = Path(os.environ["AF_PIKAN_RUNS"]).resolve()
else:
    matches = []
    for parent in (Path.cwd(), *Path.cwd().parents):
        matches.extend(parent.glob("var/agent_runs/*/download-extracted/runs"))
        matches.extend(parent.glob("var/agent_runs/*/download/*/pikan-*/runs"))
    BASELINE_RUNS = max(matches, key=lambda path: path.stat().st_mtime) if matches else Path("../runs").resolve()
BASELINE_ANALYSIS = BASELINE_RUNS.parent / "analysis"
'''
    cells.insert(1, {
        "cell_type": "code", "execution_count": None, "metadata": {},
        "source": base_path_code.splitlines(keepends=True),
        "outputs": [],
    })
    for index, cell in enumerate(cells):
        if cell["cell_type"] == "code" and cell["outputs"]:
            cell["execution_count"] = index + 1
    notebook = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
            "language_info": {"name": "python", "version": "3"},
            "research": {"notebook_id": f"results.{bundle_id[:16]}", "role": "exploration",
                         "source_of_truth": "manifest"},
        },
        "cells": cells,
    }
    path = analysis / ("-".join(problems) + "-results.ipynb")
    path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return path
