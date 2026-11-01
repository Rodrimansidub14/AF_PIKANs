"""Panel app for quantitative PIKAN run analysis and optional off-screen meshes."""
from __future__ import annotations

import os
import sys
from pathlib import Path


def _runs_path(repository_root=None):
    root = Path(repository_root).resolve() if repository_root else Path(__file__).resolve().parents[3]
    args = sys.argv
    if "--runs" in args:
        index = args.index("--runs")
        if index + 1 < len(args):
            return Path(args[index + 1]).resolve()
    if os.environ.get("AF_PIKAN_RUNS"):
        return Path(os.environ["AF_PIKAN_RUNS"]).resolve()
    candidates = list((root / "var" / "agent_runs").glob(
        "heat-burgers-*-t4x2/download-extracted/runs"
    ))
    candidates.extend((root / "var" / "agent_runs").glob(
        "heat-burgers-*-t4x2/download/*/pikan-*/runs"
    ))
    candidates.extend((root / "var" / "agent_runs").glob(
        "heat-burgers-architecture-screen/models/*/download-extracted-bundle-logscale-v1/runs"
    ))
    candidates = [path for path in candidates if path.is_dir()]
    if not candidates:
        raise FileNotFoundError(
            "Set AF_PIKAN_RUNS to a downloaded runs/ directory or pass --runs to panel serve."
        )
    return max(candidates, key=lambda path: path.stat().st_mtime)


def export_mesh_offscreen(mesh_path, image_path, scalar=None):
    """Export a VTK-family mesh to PNG without opening a display window."""
    import pyvista as pv

    mesh = pv.read(str(mesh_path))
    plotter = pv.Plotter(off_screen=True, window_size=(1400, 900))
    plotter.add_mesh(mesh, scalars=scalar, show_edges=False)
    plotter.add_axes()
    plotter.screenshot(str(image_path), return_img=False)
    plotter.close()
    return Path(image_path)


def build_app():
    try:
        import panel as pn
        import pandas as pd
        import plotly.graph_objects as go
    except ImportError as exc:
        raise RuntimeError(
            "Install requirements-analysis.txt before launching the Panel analysis app."
        ) from exc

    from src.research_pipeline.remote.reporting import quantitative_table, training_figures

    runs = _runs_path()
    table = quantitative_table(runs)
    pn.extension("plotly", "tabulator")
    problem_select = pn.widgets.Select(
        label="Problem", options=sorted(table["problem"].unique()), value="heat"
    )
    metric_options = [
        name for name in table.columns
        if name.startswith(("training_", "diagnostic_", "sampling_", "validation_", "resource_"))
        and pd.api.types.is_numeric_dtype(table[name])
    ]
    preferred = "validation_field_relative_l2"
    metric_select = pn.widgets.Select(
        label="Measured signal", options=metric_options,
        value=preferred if preferred in metric_options else metric_options[0]
    )

    @pn.depends(problem_select.param.value)
    def measured_table(problem):
        subset = table.loc[table["problem"] == problem]
        return pn.widgets.Tabulator(subset, height=380, pagination="local", page_size=6,
                                    sizing_mode="stretch_width", disabled=True)

    @pn.depends(problem_select.param.value, metric_select.param.value)
    def metric_plot(problem, metric):
        import numpy as np

        subset = table.loc[table["problem"] == problem].sort_values("seed")
        values = np.asarray(subset[metric], dtype=float)
        log_axis = bool(values.size and np.isfinite(values).all() and (values > 0).all())
        figure = go.Figure()
        figure.add_trace(go.Scatter(
            x=subset["seed"], y=subset[metric], mode="lines+markers",
            name=metric, text=[f"seed {seed}" for seed in subset["seed"]],
        ))
        figure.update_layout(
            title=f"{problem.title()}: {metric} ({'log' if log_axis else 'linear'} scale)",
            xaxis_title="random seed",
            yaxis_title="measured value (see metric name for units)",
            template="plotly_white",
            height=400,
        )
        figure.update_yaxes(type="log" if log_axis else "linear")
        return pn.pane.Plotly(figure, sizing_mode="stretch_width")

    curves = training_figures(runs)
    curve_view = pn.Column(
        *[pn.pane.Plotly(figure, sizing_mode="stretch_width")
          for _, figure in curves.items()],
        sizing_mode="stretch_width",
    )

    mesh_paths = sorted(
        path for extension in ("*.vtu", "*.vtk", "*.vtp", "*.vts", "*.vtr")
        for path in runs.rglob(extension)
    )
    if mesh_paths:
        mesh_select = pn.widgets.Select(name="Mesh", options=[str(path) for path in mesh_paths])

        @pn.depends(mesh_select.param.value)
        def mesh_preview(selected):
            try:
                import pyvista  # noqa: F401
                image_path = runs / "analysis" / (Path(selected).stem + "-offscreen.png")
                image_path.parent.mkdir(parents=True, exist_ok=True)
                export_mesh_offscreen(selected, image_path)
                return pn.pane.PNG(str(image_path), sizing_mode="scale_width")
            except Exception as exc:
                return pn.pane.Alert(f"Off-screen mesh export failed: {exc}", alert_type="danger")

        mesh_view = pn.Column("### PyVista mesh export (off-screen)", mesh_select, mesh_preview)
    else:
        mesh_view = pn.pane.Markdown(
            "Heat/Burgers contain no explicit VTK mesh, so PyVista field rendering is not applicable."
        )

    app = pn.Column(
        "# AF_PIKANs · post-run analysis",
        pn.pane.Markdown(f"**Evidence directory:** {runs} · {len(table)} measured seed/problem rows"),
        pn.Row(problem_select, metric_select),
        measured_table,
        metric_plot,
        "## Training curves",
        curve_view,
        "## Geometry and meshes",
        mesh_view,
        sizing_mode="stretch_width",
    )
    return app.servable()


app = build_app()
