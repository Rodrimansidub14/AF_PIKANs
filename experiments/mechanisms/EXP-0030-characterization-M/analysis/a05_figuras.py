"""EXP-0030: figuras (Altair + vl-convert, paleta de la tesis). (1) error robusto por semilla con el hardware visible; (2) cuotas de la traza del bloque por grupo a lo largo del entrenamiento, M."""
import json

import altair as alt
import numpy as np
import pandas as pd
import vl_convert as vlc

from common import EXP, SALIDA, cargar

NAVY, SLATE, BRICK, STEEL, INK, GRID, TEAL = "#1F3A5F", "#6B7787", "#9B3D2F", "#5B84B1", "#1B2430", "#D9DCE1", "#237D75"
FUENTE = "Georgia, 'Latin Modern Roman', serif"
FIG = EXP / "figures"


def guardar(chart, nombre):
    spec = chart.to_dict()
    (FIG / f"{nombre}.pdf").write_bytes(vlc.vegalite_to_pdf(spec, vl_version="5.20"))
    (FIG / f"{nombre}.png").write_bytes(vlc.vegalite_to_png(spec, vl_version="5.20", scale=3))


# --- 1. error robusto por semilla, con el hardware
filas = json.load(open(SALIDA / "A_corridas.json", encoding="utf-8"))
df = pd.DataFrame(filas)
df["x"] = df.brazo.map({"B": 0, "E": 1, "M": 2}) + ((df.semilla % 5) - 2) * 0.09
hw_col = alt.Scale(domain=["T4", "H100", "A100-80", "A100-40"], range=[SLATE, STEEL, BRICK, "#C98B7F"])
hw_forma = alt.Scale(domain=["T4", "H100", "A100-80", "A100-40"], range=["triangle-up", "square", "circle", "diamond"])


def panel(pr, titulo):
    d = df[df.problema == pr]
    pts = alt.Chart(d, width=230, height=190, title=alt.TitleParams(titulo, anchor="start", fontSize=12, fontWeight="bold", color=INK)).mark_point(filled=True, size=90, opacity=0.95).encode(
        x=alt.X("x:Q", title=None, scale=alt.Scale(domain=[-0.5, 2.5]), axis=alt.Axis(values=[0, 1, 2], labelExpr="['B','E','M'][datum.value]", labelFontSize=12)),
        y=alt.Y("u_rob:Q", title="error típico del tramo final", scale=alt.Scale(type="log", domain=[0.025, 0.5], nice=False), axis=alt.Axis(values=[0.03, 0.05, 0.1, 0.2, 0.4], format="~g", grid=True, gridColor=GRID)),
        color=alt.Color("hw:N", scale=hw_col, legend=alt.Legend(title="hardware", orient="bottom")), shape=alt.Shape("hw:N", scale=hw_forma, legend=None),
        tooltip=["brazo", "semilla", "hw", "u_rob"])
    return pts


g = alt.hconcat(panel("converse", "a  Converso"), panel("direct", "b  Directo"), spacing=30).configure(font=FUENTE).configure_axis(
    labelColor=SLATE, titleColor=INK, domainColor=SLATE, tickColor=SLATE, labelFontSize=10, titleFontSize=10).configure_view(stroke=None).configure_legend(labelFontSize=10, titleFontSize=10)
guardar(g, "fig1_error_hardware")

# --- 2. cuotas de la traza por grupo, M converso
R = cargar()
rows = []
for (pr, a, s), r in sorted(R.items()):
    if a != "M":
        continue
    z = np.load(r["npz"], allow_pickle=True)
    n = sorted({int(k[7:9]) for k in z.keys() if k.startswith("refresh") and k.endswith("/step")})
    for i in n:
        L = sorted({int(k.split("/layer")[1][0]) for k in z.keys() if k.startswith(f"refresh{i:02d}/layer") and "gtrace_mec" in k})
        gt = {nm: sum(float(z[f"refresh{i:02d}/layer{l}/gtrace_{nm}"].astype(float).sum()) for l in L) for nm in ("mec", "elec", "equil", "borde")}
        tot = sum(gt.values())
        for nm, nombre in (("mec", "mecánica"), ("elec", "eléctrica"), ("equil", "equilibrio")):
            rows.append(dict(problema=pr, semilla=s, paso=int(z[f"refresh{i:02d}/step"]), grupo=nombre, cuota=gt[nm] / tot))
dq = pd.DataFrame(rows)


def panel2(pr, titulo):
    d = dq[dq.problema == pr]
    return alt.Chart(d, width=240, height=180, title=alt.TitleParams(titulo, anchor="start", fontSize=12, fontWeight="bold", color=INK)).mark_line(point=True, strokeWidth=1.6).encode(
        x=alt.X("paso:Q", title="paso de entrenamiento", axis=alt.Axis(format="d")), y=alt.Y("mean(cuota):Q", title="cuota de la traza", scale=alt.Scale(domain=[0, 1])),
        color=alt.Color("grupo:N", scale=alt.Scale(domain=["mecánica", "equilibrio", "eléctrica"], range=[NAVY, TEAL, BRICK]), legend=alt.Legend(title="grupo", orient="bottom")))


g2 = alt.hconcat(panel2("converse", "a  Converso (media de 3 semillas)"), panel2("direct", "b  Directo (media de 3 semillas)"), spacing=30).configure(font=FUENTE).configure_axis(
    labelColor=SLATE, titleColor=INK, domainColor=SLATE, tickColor=SLATE, labelFontSize=10, titleFontSize=10, grid=True, gridColor=GRID).configure_view(stroke=None)
guardar(g2, "fig2_cuotas_M")
print("ok")
