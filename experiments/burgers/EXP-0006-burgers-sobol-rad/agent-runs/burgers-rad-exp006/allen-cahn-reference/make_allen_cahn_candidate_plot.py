#!/usr/bin/env python
"""Build an offline Plotly dashboard for the candidate Allen-Cahn study.

Requires NumPy and the adjacent official plotly-4.0.0.min.js bundle. This
script only reads the candidate NPZ/metrics and writes a standalone HTML file;
it does not rerun the PDE solver or edit the reference artifacts.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PLOTLY_JS = HERE / "plotly-4.0.0.min.js"
DEFAULT_STUDY = HERE / "study"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def json_for_script(value) -> str:
    """Encode strict JSON and prevent values from closing an HTML script tag."""
    return json.dumps(value, separators=(",", ":"), allow_nan=False).replace("</", "<\\/")


def chart_specs(metrics, archive):
    x = archive["x"]
    t = archive["t"]
    u = archive["u_primary"]
    if u.shape != (len(x), len(t)):
        raise ValueError(f"unexpected finest field shape {u.shape}")

    field_pairs = metrics["spatial_refinement_differences"]
    pair_labels = [
        f"{item['coarse'].split('_')[1]}→{item['fine'].split('_')[1]}"
        for item in field_pairs
    ]
    field_delta = [item["weighted_relative_l2_field_change"] for item in field_pairs]
    energy_delta = [item["max_relative_energy_change"] for item in field_pairs]
    event_times = metrics["contract"]["evaluation_times"]
    energy_by_run = metrics["energy_by_run_at_evaluation_times"]
    spatial_runs = metrics["runs"]
    colors = ["#2266aa", "#d97706", "#18845a"]
    energy_traces = []
    for i, run in enumerate(spatial_runs):
        label = run["label"]
        energy_traces.append({
            "type": "scatter", "mode": "lines+markers",
            "name": f"{run['node_count']} nodes",
            "x": event_times, "y": energy_by_run[label],
            "line": {"color": colors[i % len(colors)], "width": 2},
            "marker": {"size": 8},
            "hovertemplate": "t=%{x:.2f}<br>E=%{y:.8g}<extra>%{fullData.name}</extra>",
        })

    delta_fine = np.abs(archive["u_common_n1025"] - archive["u_common_n2049"])
    delta_floor = 1e-16
    log_delta = np.log10(np.maximum(delta_fine, delta_floor)).T
    log_delta_min = float(np.nanmin(log_delta))
    log_delta_max = float(np.nanmax(log_delta))

    figures = [
        {
            "div": "field-surface",
            "data": [{
                "type": "surface", "x": x.tolist(), "y": t.tolist(),
                "z": u.T.tolist(), "surfacecolor": u.T.tolist(),
                "colorscale": "RdBu", "cmin": -1.0, "cmax": 1.0,
                "colorbar": {"title": "u", "len": 0.72},
                "hovertemplate": "x=%{x:.4f}<br>t=%{y:.4f}<br>u=%{z:.6f}<extra></extra>",
                "name": "2049-node candidate field",
                "showscale": True,
            }],
            "layout": {
                "title": {"text": "Finest-grid candidate solution u(x,t) — signed linear scale"},
                "scene": {
                    "xaxis": {"title": "x", "range": [-1, 1]},
                    "yaxis": {"title": "t", "range": [0, 1]},
                    "zaxis": {"title": "u", "range": [-1, 1]},
                    "camera": {"eye": {"x": 1.55, "y": 1.65, "z": 1.15}},
                    "aspectratio": {"x": 1.4, "y": 1.2, "z": 0.8},
                },
                "margin": {"l": 10, "r": 20, "t": 55, "b": 10},
                "height": 650,
            },
        },
        {
            "div": "refinement-deltas",
            "data": [
                {"type": "scatter", "mode": "lines+markers", "name": "Weighted relative field change",
                 "x": pair_labels, "y": field_delta, "line": {"color": "#2266aa", "width": 3},
                 "marker": {"size": 10}, "hovertemplate": "%{x}<br>field Δ=%{y:.6g}<extra></extra>"},
                {"type": "scatter", "mode": "lines+markers", "name": "Maximum relative energy change",
                 "x": pair_labels, "y": energy_delta, "line": {"color": "#d97706", "width": 3},
                 "marker": {"size": 10}, "hovertemplate": "%{x}<br>energy Δ=%{y:.6g}<extra></extra>"},
                {"type": "scatter", "mode": "lines", "name": "Field threshold 0.005",
                 "x": pair_labels, "y": [metrics["convergence_contract"]["field_change_limit"]]*len(pair_labels),
                 "line": {"color": "#2266aa", "dash": "dash", "width": 1},
                 "hoverinfo": "skip"},
                {"type": "scatter", "mode": "lines", "name": "Energy threshold 0.001",
                 "x": pair_labels, "y": [metrics["convergence_contract"]["qoi_change_limit"]]*len(pair_labels),
                 "line": {"color": "#d97706", "dash": "dash", "width": 1},
                 "hoverinfo": "skip"},
            ],
            "layout": {
                "title": {"text": "Spatial refinement changes (not ground-truth error)"},
                "xaxis": {"title": "Consecutive spatial resolutions"},
                "yaxis": {"title": "Positive relative change", "type": "log"},
                "legend": {"orientation": "h", "y": -0.28},
                "margin": {"l": 75, "r": 20, "t": 55, "b": 90},
                "height": 450,
                "annotations": [{
                    "text": "513→1025 energy change = 1.301e-3 at t=1 (above 1e-3); 1025→2049 = 3.257e-4",
                    "xref": "paper", "yref": "paper", "x": 0.5, "y": 1.12,
                    "showarrow": False, "font": {"size": 11},
                }],
            },
        },
        {
            "div": "energy-history",
            "data": energy_traces,
            "layout": {
                "title": {"text": "Allen–Cahn energy at exact evaluation times — linear scale"},
                "xaxis": {"title": "t", "range": [0.2, 1.03]},
                "yaxis": {"title": "E[u]", "rangemode": "tozero"},
                "legend": {"orientation": "h", "y": -0.25},
                "margin": {"l": 70, "r": 20, "t": 55, "b": 85},
                "height": 430,
            },
        },
        {
            "div": "field-difference-map",
            "data": [{
                "type": "heatmap", "x": x.tolist(), "y": t.tolist(),
                "z": log_delta.tolist(), "colorscale": "Viridis",
                "zmin": log_delta_min, "zmax": log_delta_max,
                "colorbar": {"title": "log₁₀ |Δu|", "len": 0.84},
                "hovertemplate": "x=%{x:.4f}<br>t=%{y:.4f}<br>log₁₀|Δu|=%{z:.3f}<extra></extra>",
                "name": "1025 vs 2049 nodes",
            }],
            "layout": {
                "title": {"text": "Spatial refinement difference |u₁₀₂₅ − u₂₀₄₉| (log₁₀ scale)"},
                "xaxis": {"title": "x"}, "yaxis": {"title": "t"},
                "margin": {"l": 65, "r": 20, "t": 55, "b": 55},
                "height": 450,
            },
        },
    ]
    return figures


def render(study: Path, output: Path) -> None:
    metrics_path = study / "metrics.json"
    npz_path = study / "allen_cahn_candidate_reference.npz"
    for path in (metrics_path, npz_path, PLOTLY_JS):
        if not path.is_file():
            raise FileNotFoundError(path)
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    # At x=±1, u0=-1 and u0_xx=pi²-2; the reaction term is zero.
    # Thus the PDE-side initial u_t is 0.001*(pi²-2), unlike the fixed trace's 0.
    corner_ut = 0.001 * (np.pi**2 - 2.0) + 5.0 * (-1.0 - (-1.0)**3)
    with np.load(npz_path) as archive:
        figures = chart_specs(metrics, archive)

    js = PLOTLY_JS.read_text(encoding="utf-8")
    plotly_hash = sha256(PLOTLY_JS)
    js = js.replace("</script", "<\\/script")
    containers = "\n".join(
        f'<div id="{fig["div"]}" class="plot"></div>' for fig in figures
    )
    init = "\n".join(
        "Plotly.newPlot(%s, %s, %s, {responsive:true, displaylogo:false});" % (
            json_for_script(fig["div"]), json_for_script(fig["data"]), json_for_script(fig["layout"])
        )
        for fig in figures
    )
    c = metrics["convergence_contract"]
    s = metrics["sanity_checks"]
    body = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Allen–Cahn candidate reference diagnostics</title>
  <style>
    :root {{ color-scheme: light; font-family: system-ui, -apple-system, Segoe UI, sans-serif; }}
    body {{ margin: 0; background: #f4f6f8; color: #17212b; }}
    main {{ max-width: 1500px; margin: 0 auto; padding: 24px; }}
    h1 {{ margin: 0 0 8px; font-size: 1.8rem; }}
    .badge {{ display: inline-block; border: 1px solid #b54708; border-radius: 999px; padding: 4px 10px; color: #8a2c0d; background: #fff5eb; font-weight: 650; }}
    .note {{ line-height: 1.55; background: white; border-left: 4px solid #d97706; padding: 12px 16px; margin: 14px 0; box-shadow: 0 1px 3px #0001; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(430px, 1fr)); gap: 14px; }}
    .plot {{ background: white; border-radius: 8px; box-shadow: 0 1px 4px #0002; min-width: 0; }}
    #field-surface {{ grid-column: 1 / -1; }}
    .meta {{ color: #53616c; font-size: .86rem; overflow-wrap: anywhere; }}
    table {{ border-collapse: collapse; background: white; margin: 12px 0; }}
    th,td {{ border: 1px solid #d5dbe0; padding: 6px 10px; text-align: left; }}
    @media(max-width: 700px) {{ main {{ padding: 10px; }} .grid {{ grid-template-columns: 1fr; }} #field-surface {{ grid-column: auto; }} }}
  </style>
</head>
<body>
<main>
  <h1>Allen–Cahn CPU reference study</h1>
  <span class="badge">Candidate only · not canonical · not scientifically accepted</span>
  <p class="note"><b>How to read this:</b> refinement deltas are solver-to-solver differences, not error against ground truth. The 2049-node BDF field is a candidate visualization only; this page does not unlock field evaluation.</p>
  <p class="note"><b>Early-time compatibility:</b> the IC value traces match the fixed boundary exactly (max error {s['max_boundary_abs_error']:.3g}). However, the initial PDE limit gives u<sub>t</sub>={corner_ut:.8f} at each endpoint while the constant Dirichlet trace prescribes u<sub>t</sub>=0. This first-order parabolic corner condition is not met. The value trace is valid, but near-zero-time regularity remains a caution; the study has no separate very-early-time refinement ladder.</p>
  <p class="note"><b>Convergence:</b> last spatial field change {c['last_spatial_field_change']:.3e} (limit {c['field_change_limit']:.3g}); last spatial energy change {c['last_spatial_energy_change']:.3e} (limit {c['qoi_change_limit']:.3g}); tight-time field change {c['temporal_tolerance_difference']['weighted_relative_l2_field_change']:.3e}. The first spatial-pair energy change is {metrics['spatial_refinement_differences'][0]['max_relative_energy_change']:.3e}, slightly above 1e-3; the finer pair is below. The “10% of effect” rule remains unresolved.</p>
  <div class="grid">{containers}</div>
  <p class="meta">Python {html.escape(metrics['provenance']['python'].split()[0])} · NumPy {html.escape(metrics['provenance']['numpy'])} · SciPy {html.escape(metrics['provenance']['scipy'])} · Plotly.js 4.0.0. Generated {html.escape(metrics['provenance']['generated_at_utc'])}.</p>
  <p class="meta">Reference script SHA-256: <code>{html.escape(metrics['provenance']['source_script_sha256'])}</code><br>
  Candidate NPZ SHA-256: <code>{html.escape(metrics['provenance']['candidate_npz_sha256'])}</code><br>
  Plotly.js bundle SHA-256: <code>{plotly_hash}</code></p>
</main>
<script>{js}</script>
<script>
  const config = {{responsive: true, displaylogo: false, scrollZoom: true}};
  {init}
</script>
</body>
</html>
"""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(body, encoding="utf-8", newline="\n")
    print(json.dumps({
        "output": str(output.resolve()), "bytes": output.stat().st_size,
        "plotly_js_sha256": plotly_hash, "figures": [f["div"] for f in figures],
        "external_script_references": False,
        "candidate_only": True,
    }, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=DEFAULT_STUDY)
    parser.add_argument("--output", type=Path, default=DEFAULT_STUDY / "allen_cahn_candidate_diagnostics.html")
    args = parser.parse_args()
    render(args.study.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
