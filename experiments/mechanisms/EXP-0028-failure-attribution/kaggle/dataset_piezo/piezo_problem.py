"""PiezoProblem (EXP-0028, fase de piezo): bimorfo PVDF 2D, DOS problemas del paper (Gonzalez Carrillo et al., MCA 31, 214): CONVERSO (100 V arriba, 0 V abajo, sin carga
mecanica) y DIRECTO (0.1 N hacia abajo en el extremo libre, electrodo superior flotante con carga neta cero). Puerto propio, linea por linea, del codigo de los autores
(PINNs_piezoelectricity @ e7856f1: src/pinn_piezo/{indirect,direct}/{model,losses,sampling}.py, scaling.py, materials.py, geometry.py); ese repo NO se importa aqui
(solo las pruebas lo importan para comparar). Ver piezo/DIFERENCIAS.md y piezo/PROCEDENCIA.md.

Interfaz del banco (como NSProblem de pikan_campaign.py):  prob.build(arm, width, bases, seed) -> modelo;  prob.pools(seed, device, dtype, batch=0) -> dict de puntos;
prob.loss(model, P) -> (total, {termino: peso*media(r^2)});  prob.score(model, device, dtype) -> dict;  prob.residual_terms(model_fn, X) -> [(r, w, nombre)] (adaptador
`piezo_residual_terms` de src/mechanisms/residual_terms.py; la suma  sum_t w_t mean(r_t^2)  es prob.loss).

Formulacion (paper): 8 salidas (u, v, phi, sx, sz, tau, Dx, Dy) en formulacion MIXTA; entradas (xi=x/L, eta=y/H, |eta-1/2|, fase p=+-1); construccion condicionada por fase
(se evaluan N(+1) y N(-1): continuos = promedio, sx y Dx = promedio + p (N+ - N-)/2); escalas por campo; bordes duros por levantamiento; PESOS FIJOS (converso: todos 1;
directo: lambda_BC = 10, lambda_PDE = lambda_int = 1); L_int es identicamente 0 en el modelo mixto (se verifica con `interface_gap`, no entra a la perdida ni al gradiente).
Converso: `mechanical gradient routing` (las tensiones independientes de los residuos constitutivos mecanicos son constantes al derivar: stop-gradient). Directo: gradientes
ordinarios, normalizacion constitutiva 'dominant', tracciones y aislamiento incorporados en las salidas, termino de carga neta cero L_Q = (mean D2/sD en el electrodo)^2.
float64, Adam 1e-3, 20 000 epocas, sin L-BFGS, 2048 interiores POR CAPA + 256 por frontera/interfaz, remuestreo cada 200 epocas: eso lo hace el lazo (run_piezo.py).

El modulo no lee la referencia FEM salvo en `score` (fem_path); `loss`, `pools` y `residual_terms` no la tocan.
"""
from __future__ import annotations

import math
import os
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

# ------------------------------------------------------------------ geometria y material (config.py, materials.py)
WIDTH, HEIGHT = 100e-3, 1e-3
CENTER = HEIGHT / 2
REFERENCE_VOLTAGE = 100.0
REFERENCE_FORCE = 0.1
PAPER_SEED = 20260728


def _material():
    E, NU, G = 2.0e9, 0.29, 0.775e9
    d31, d33 = 2.2e-11, -3.0e-11
    rel_perm, eps0 = 12, 8.854187817620e-12
    C11 = E / (1 - NU ** 2)
    C12 = NU * C11
    C22 = C11
    c2d = np.array([[C11, C12, 0], [C12, C11, 0], [0, 0, G]])
    pze_D = np.array([[0, d31], [0, d33], [0, 0]])
    pze_E = c2d @ pze_D
    D_const_stress = np.array([[rel_perm, 0], [0, rel_perm]]) * eps0
    D_const_strain = D_const_stress - pze_D.T @ pze_E
    c13 = np.array([[pze_E[0, 0], pze_E[0, 1]], [pze_E[2, 0], pze_E[2, 1]]])
    c31 = c13.T
    c23 = np.array([[pze_E[2, 0], pze_E[2, 1]], [pze_E[1, 0], pze_E[1, 1]]])
    c32 = c23.T
    c33 = np.array([D_const_strain[0, 0], D_const_strain[1, 0], D_const_strain[0, 1], D_const_strain[1, 1]])
    return dict(C11=float(C11), C12=float(C12), C22=float(C22), G=float(G), eps1=float(c33[0]), eps2=float(c33[-1]),
                e31_bottom=float(c31[1, 0]), e31_top=float(-c31[1, 0]), e33_bottom=float(c32[-1, -1]), e33_top=float(-c32[-1, -1]),
                coupling=float(max(abs(pze_E[0, 1]), abs(pze_E[1, 1]))), permittivity=float(max(c33[0], c33[-1])), NU=NU, center=CENTER)


MAT = _material()


# ------------------------------------------------------------------ escalas (scaling.py)
def _scales(strain, phi, stress_x, stress_y, shear, d_scale, traction):
    return dict(length=WIDTH, height=HEIGHT, u=strain * WIDTH, v=strain * WIDTH ** 2 / HEIGHT, phi=phi, stress_x=stress_x, stress_y=stress_y, shear=shear,
                electric_displacement=d_scale, traction=traction, equilibrium_x=max(stress_x / WIDTH, shear / HEIGHT), equilibrium_y=shear / WIDTH,
                gauss=d_scale / HEIGHT)


def converse_scales(voltage=REFERENCE_VOLTAGE):
    m = MAT
    phi = max(abs(float(voltage)), 1.0)
    ef = phi / HEIGHT
    strain = max(m["coupling"] * ef / m["C11"], 1e-16)
    stress = m["C11"] * strain
    d_scale = max(m["permittivity"] * ef, m["coupling"] * strain)
    return _scales(strain, phi, stress, stress, stress, d_scale, stress)


def direct_scales(force=REFERENCE_FORCE):
    m = MAT
    f = max(abs(float(force)), 1e-16)
    traction = f / HEIGHT
    stress = 6.0 * f * WIDTH / HEIGHT ** 2
    strain = stress / m["C11"]
    phi = max(m["coupling"] * strain * HEIGHT / m["permittivity"], 1e-12)
    ef = phi / HEIGHT
    d_scale = max(m["coupling"] * strain, m["permittivity"] * ef)
    return _scales(strain, phi, stress, traction, traction, d_scale, traction)


def represented_scales(s):
    return dict(sxx=s["stress_x"], syy=s["stress_y"], tau=s["shear"], dx=s["electric_displacement"], dy=s["electric_displacement"])


def dominant_scales(s):
    m = MAT
    axial = s["u"] / s["length"]
    ef = s["phi"] / s["height"]
    return dict(sxx=max(s["stress_x"], m["C11"] * axial), syy=max(s["stress_y"], m["C12"] * axial), tau=max(s["shear"], m["G"] * s["u"] / s["height"]),
                dx=max(s["electric_displacement"], m["eps1"] * ef), dy=max(s["electric_displacement"], m["eps2"] * ef, m["coupling"] * axial))


# ------------------------------------------------------------------ modelo
class RBFEdge(nn.Module):
    """Arista RBF-PIKAN (misma familia y atributos que pikan_campaign.RBFEdge / daniel_arm.RBFNet): coefficients (O, I, G), centers, width, base."""

    def __init__(self, inputs, outputs, bases=11):
        super().__init__()
        spacing = 2.0 / (bases - 1)
        self.register_buffer("centers", torch.linspace(-1.0, 1.0, bases))
        self.register_buffer("width", torch.tensor(1.75 * spacing))
        self.coefficients = nn.Parameter(torch.randn(outputs, inputs, bases) * (0.1 / math.sqrt(inputs)))
        self.base = nn.Parameter(torch.randn(outputs, inputs) * (0.1 / math.sqrt(inputs)))

    def forward(self, x):
        basis = torch.exp(-((x.unsqueeze(-1) - self.centers) / self.width).square())
        return torch.einsum("big,oig->bo", basis, self.coefficients) + F.linear(F.silu(x), self.base)


def _mlp_trunk(hidden=(50, 50, 50), n_in=4, n_out=8, activation=nn.SiLU):
    layers = [("input", nn.Linear(n_in, hidden[0])), ("act0", activation())]
    for i in range(1, len(hidden)):
        layers += [(f"hidden_{i - 1}", nn.Linear(hidden[i - 1], hidden[i])), (f"act_{i}", activation())]
    layers.append(("output", nn.Linear(hidden[-1], n_out)))
    return nn.Sequential(OrderedDict(layers))


def _init_mlp(net, output_gain=None):
    """init_weights del repo (Xavier normal, sesgos 0) y, en el directo, la reinicializacion de la capa de salida (--output-init-gain 1.0): consume la misma secuencia del RNG."""
    for m in net.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0)
    if output_gain is not None:
        nn.init.xavier_normal_(net.output.weight, gain=output_gain)
        nn.init.zeros_(net.output.bias)
    return net


class PiezoNet(nn.Module):
    """Modelo completo del paper: tronco (MLP 3x50 SiLU del paper o RBF-PIKAN) + construccion por fase + escalas + bordes duros. forward(x[, phase]) -> (n, 8) en unidades fisicas.
    x: (n, 2) en metros (o (n, 3) con la fase explicita en la tercera columna: se usa para las trazas de la interfaz). `layers` (solo RBF) lo ve GramPreconditioner."""

    def __init__(self, mode, trunk, width=16, bases=11, force=-REFERENCE_FORCE, voltage=REFERENCE_VOLTAGE):
        super().__init__()
        assert mode in ("converse", "direct") and trunk in ("mlp", "rbf")
        self.mode, self.trunk = mode, trunk
        self.force = float(force)
        self.voltage = float(voltage)
        self.scales = converse_scales(voltage) if mode == "converse" else direct_scales(abs(force))
        if mode == "direct":
            self.floating_voltage = nn.Parameter(torch.zeros(1))        # V_f adimensional (ec. 75); se aprende con L_Q
        if trunk == "mlp":
            self.net = _mlp_trunk()
        else:
            self.layers = nn.ModuleList([RBFEdge(4, width, bases), RBFEdge(width, width, bases), RBFEdge(width, 8, bases)])
        self.register_buffer("_jump", torch.tensor([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0]]), persistent=False)

    # -- tronco
    def raw(self, z):
        if self.trunk == "mlp":
            return self.net(z)
        h = torch.cat([2 * z[:, 0:1] - 1, 2 * z[:, 1:2] - 1, 4 * z[:, 2:3] - 1, z[:, 3:4]], dim=1)
        for layer in self.layers[:-1]:
            h = torch.tanh(layer(h))
        return self.layers[-1](h)

    # -- construccion por fase (ecs. 30-35)
    @staticmethod
    def material_phase(x):
        return torch.where(x[:, 1:2] >= CENTER, torch.ones_like(x[:, 1:2]), -torch.ones_like(x[:, 1:2]))

    def _outputs(self, x, phase, center=False):
        phase = self.material_phase(x) if phase is None else phase.to(dtype=x.dtype, device=x.device)
        plus = torch.ones_like(phase)
        if center:
            xi = torch.cat((x[:, 0:1] / WIDTH, torch.full_like(x[:, 1:2], 0.5)), dim=1)
            cusp = torch.zeros_like(xi[:, 1:2])
        else:
            xi = torch.cat((x[:, 0:1] / WIDTH, x[:, 1:2] / HEIGHT), dim=1)
            cusp = torch.abs(xi[:, 1:2] - 0.5)
        n_plus = self.raw(torch.cat((xi, cusp, plus), dim=1))
        n_minus = self.raw(torch.cat((xi, cusp, -plus), dim=1))
        average = 0.5 * (n_plus + n_minus)
        selected = average + 0.5 * phase * (n_plus - n_minus)
        return average + self._jump.to(average.dtype) * (selected - average)

    def forward(self, x, phase=None):
        if x.shape[1] == 3:
            phase, x = x[:, 2:3], x[:, 0:2]
        outputs = self._outputs(x, phase)
        center = self._outputs(x, phase, center=True)
        return self._constrain(x, outputs, center)

    def forward_trace(self, x, phase):
        return self.forward(x, phase=phase)

    # -- bordes duros (ecs. 72-80)
    def _constrain(self, x, outputs, center):
        s = self.scales
        xn, yn = x[:, 0:1] / WIDTH, x[:, 1:2] / HEIGHT
        slender = HEIGHT / WIDTH
        v_dimless = center[:, 1:2] + slender ** 2 * (outputs[:, 1:2] - center[:, 1:2])
        u = s["u"] * xn * outputs[:, 0:1]
        v = s["v"] * xn * v_dimless
        phi = outputs[:, 2:3]
        if self.mode == "converse":
            phi_f = self.voltage * (yn + yn * (yn - 1.0) * phi)
            stress = torch.cat([s["stress_x"] * outputs[:, 3:4], s["stress_y"] * outputs[:, 4:5], s["shear"] * outputs[:, 5:6]], dim=1)
            d = s["electric_displacement"] * torch.cat((outputs[:, 6:7], outputs[:, 7:8]), dim=1)
            fields = torch.cat([u, v, phi_f, stress, d], dim=1)
            return (self.voltage / self.voltage) * fields
        phi_f = s["phi"] * yn * (self.floating_voltage + (1.0 - yn) * phi)
        load = self.force / (HEIGHT * s["shear"])
        face = yn * (1.0 - yn)
        tip_distance = (1.0 - xn) ** 2
        interior_lift = face / (face + tip_distance + 1e-30)
        on_tip = torch.isclose(xn, torch.ones_like(xn))
        lift = torch.where(on_tip, torch.where(face > 0.0, torch.ones_like(face), torch.zeros_like(face)), interior_lift)
        tip_traction = torch.full_like(yn, load)
        stress = torch.cat([s["stress_x"] * (1.0 - xn) * outputs[:, 3:4], s["stress_y"] * face * outputs[:, 4:5],
                            s["shear"] * (tip_traction * lift + face * (1.0 - xn) * outputs[:, 5:6])], dim=1)
        d = torch.cat([s["electric_displacement"] * xn * (1.0 - xn) * outputs[:, 6:7], s["electric_displacement"] * outputs[:, 7:8]], dim=1)
        return torch.cat((torch.cat((u, v, phi_f), dim=1), torch.cat((stress, d), dim=1)), dim=1)


# ------------------------------------------------------------------ muestreo (indirect/sampling.py, valores por defecto)
def sample_pools(seed, n_interior_per_layer=2048, n_boundary=256, n_interface=256, device="cpu", dtype=torch.float64):
    """Dos capas muestreadas por separado (fondo primero, tope despues), 256 por frontera y por interfaz; semilla `seed` (el lazo usa seed + k en el bloque k de 200 epocas)."""
    rng = np.random.default_rng(seed)
    margin = 1e-8 * HEIGHT

    def interior(y0, y1):
        accepted, remaining = [], n_interior_per_layer
        while remaining:
            count = max(remaining * 2, 64)
            cand = np.column_stack((rng.uniform(0.0, WIDTH, count), rng.uniform(y0, y1, count)))
            take = cand[:remaining]
            accepted.append(take)
            remaining -= len(take)
        return np.vstack(accepted)

    xy = np.vstack((interior(0.0 + margin, CENTER - margin), interior(CENTER + margin, HEIGHT - margin)))
    xb = rng.uniform(0.0, WIDTH, n_boundary)
    yb = rng.uniform(0.0, HEIGHT, n_boundary)
    xi = rng.uniform(0.0, WIDTH, n_interface)
    t = lambda a: torch.tensor(a, dtype=dtype, device=device)
    return {"interior": t(xy), "top": t(np.column_stack((xb, np.full(n_boundary, HEIGHT)))), "bottom": t(np.column_stack((xb, np.zeros(n_boundary)))),
            "right": t(np.column_stack((np.full(n_boundary, WIDTH), yb))), "left": t(np.column_stack((np.zeros(n_boundary), yb))),
            "interface": t(np.column_stack((xi, np.full(n_interface, CENTER))))}


# ------------------------------------------------------------------ el problema
class PiezoProblem:
    n_in, n_out = 2, 8
    adam_steps_default = 20000
    resample_every = 200

    def __init__(self, mode="converse", fem_path=None, poling_sign=-1.0, interior_per_layer=2048, boundary=256, interface=256, routing=None, weights=None):
        assert mode in ("converse", "direct")
        assert poling_sign == -1.0, "el PINN del paper usa +e en la capa inferior: su FEM de referencia es poling_sign = -1 (CANONICAL_POLING_SIGN)"
        self.mode = mode
        self.name = "piezo_bimorph_indirect_2d" if mode == "converse" else "piezo_bimorph_direct_2d"
        self.fem_path = fem_path
        self.n_layer, self.n_boundary, self.n_interface = interior_per_layer, boundary, interface
        self.force = -REFERENCE_FORCE
        self.voltage = REFERENCE_VOLTAGE
        self.scales = converse_scales(self.voltage) if mode == "converse" else direct_scales(abs(self.force))
        self.cscales = represented_scales(self.scales) if mode == "converse" else dominant_scales(self.scales)
        self.routing = (mode == "converse") if routing is None else bool(routing)           # 'mechanical gradient routing' (solo converso)
        self.w = dict(weights or ({"pde": 1.0, "bc": 1.0} if mode == "converse" else {"pde": 1.0, "bc": 10.0}), nw=1.0)
        self._fem = None
        self.spec = dict(mode=mode, mat=MAT, scales=self.scales, cscales=self.cscales, w=self.w, stopgrad=self.routing, force=self.force)

    # -- construccion
    def build(self, arm, width=16, bases=11, seed=PAPER_SEED, dtype=torch.float64):
        """arm 'mlp' (MLP 3x50 SiLU del paper) o 'rbf'. Misma secuencia de RNG que el repo (torch.manual_seed + np.random.seed, luego el MLP con su init, y para 'rbf' las
        aristas RBF DESPUES de consumir el RNG del MLP, como daniel_arm.py): la semilla da los mismos parametros que la linea base congelada."""
        old = torch.get_default_dtype()
        torch.set_default_dtype(torch.float64)
        try:
            torch.manual_seed(seed)
            np.random.seed(seed)
            gain = 1.0 if self.mode == "direct" else None
            if arm == "mlp":
                model = PiezoNet(self.mode, "mlp", force=self.force, voltage=self.voltage)
                _init_mlp(model.net, gain)
            else:
                burn = _init_mlp(_mlp_trunk(), gain)                                   # solo consume el RNG
                del burn
                model = PiezoNet(self.mode, "rbf", width, bases, force=self.force, voltage=self.voltage)
        finally:
            torch.set_default_dtype(old)
        return model.to(dtype)

    def pools(self, seed, device="cpu", dtype=torch.float64, batch=0):
        return sample_pools(seed + batch, self.n_layer, self.n_boundary, self.n_interface, device, dtype)

    # -- perdida (copia del repo; mismo orden de operaciones)
    @staticmethod
    def _coef(y):
        m = MAT
        bot = y < CENTER
        wh = lambda b, t: torch.where(bot, torch.full_like(y, b), torch.full_like(y, t))
        return wh(m["e31_bottom"], m["e31_top"]), wh(m["e33_bottom"], m["e33_top"])

    def _pde(self, model, interior):
        m, cs, sc = MAT, self.cscales, self.scales
        x_data = interior[:, 0:1].detach().clone().requires_grad_(True)
        y_data = interior[:, 1:2].detach().clone().requires_grad_(True)
        data = torch.hstack((x_data, y_data))
        y_hat = model(data)
        sx_p, sz_p, ta_p, dx_p, dy_p = (y_hat[:, i:i + 1] for i in range(3, 8))
        if self.mode == "converse":
            u_p, v_p, phi_p = y_hat[:, 0:1], y_hat[:, 1:2], y_hat[:, 2:3]

            def g(out, wrt):
                return torch.autograd.grad(outputs=out, inputs=wrt, grad_outputs=torch.ones_like(out), create_graph=True, retain_graph=True)[0]

            ux, uy, vx, vy, phix, phiy = g(u_p, x_data), g(u_p, y_data), g(v_p, x_data), g(v_p, y_data), g(phi_p, x_data), g(phi_p, y_data)
            sx_x, ta_y, ta_x, sz_y, dx_x, dy_y = g(sx_p, x_data), g(ta_p, y_data), g(ta_p, x_data), g(sz_p, y_data), g(dx_p, x_data), g(dy_p, y_data)
        else:
            grads = [torch.autograd.grad(y_hat[:, i].sum(), data, create_graph=True)[0] for i in range(y_hat.shape[1])]
            ux, uy, vx, vy, phix, phiy = grads[0][:, 0:1], grads[0][:, 1:2], grads[1][:, 0:1], grads[1][:, 1:2], grads[2][:, 0:1], grads[2][:, 1:2]
            sx_x, sz_y, ta_x, ta_y, dx_x, dy_y = grads[3][:, 0:1], grads[4][:, 1:2], grads[5][:, 0:1], grads[5][:, 1:2], grads[6][:, 0:1], grads[7][:, 1:2]
        e31, e33 = self._coef(y_data)
        exx, eyy, exy = ux, vy, (uy + vx)
        Ex, Ey = -phix, -phiy
        sigmax = (m["C11"] * exx + m["C12"] * eyy - e31 * Ey)
        sigmaz = (m["C12"] * exx + m["C22"] * eyy - e33 * Ey)
        tauxz = m["G"] * exy
        Dx = m["eps1"] * Ex
        Dy = (e31 * exx + e33 * eyy + m["eps2"] * Ey)
        if self.mode == "converse" and self.routing:
            sx_c, sz_c, ta_c = sx_p.detach(), sz_p.detach(), ta_p.detach()
        else:
            sx_c, sz_c, ta_c = sx_p, sz_p, ta_p
        r_sx, r_sz, r_ta = (sx_c - sigmax) / cs["sxx"], (sz_c - sigmaz) / cs["syy"], (ta_c - tauxz) / cs["tau"]
        r_dx, r_dy = (dx_p - Dx) / cs["dx"], (dy_p - Dy) / cs["dy"]
        div1, div2, divD = (sx_x + ta_y) / sc["equilibrium_x"], (ta_x + sz_y) / sc["equilibrium_y"], (dx_x + dy_y) / sc["gauss"]
        return dict(constitutive_sigma_xx=torch.mean(r_sx ** 2), constitutive_sigma_yy=torch.mean(r_sz ** 2), constitutive_tau_xy=torch.mean(r_ta ** 2),
                    constitutive_D_x=torch.mean(r_dx ** 2), constitutive_D_y=torch.mean(r_dy ** 2), equilibrium_x=torch.mean(div1 ** 2),
                    equilibrium_y=torch.mean(div2 ** 2), gauss=torch.mean(divD ** 2))

    def _bc(self, model, P):
        sc = self.scales
        top, bot, right, left = model(P["top"]), model(P["bottom"]), model(P["right"]), model(P["left"])
        t = {}
        if self.mode == "converse":
            t["bc_top_sigma_yy"] = torch.mean((top[:, 4:5] / sc["stress_y"]) ** 2)
            t["bc_top_tau_xy"] = torch.mean((top[:, 5:6] / sc["shear"]) ** 2)
            t["bc_bottom_sigma_yy"] = torch.mean((bot[:, 4:5] / sc["stress_y"]) ** 2)
            t["bc_bottom_tau_xy"] = torch.mean((bot[:, 5:6] / sc["shear"]) ** 2)
            t["bc_right_sigma_xx"] = torch.mean((right[:, 3:4] / sc["stress_x"]) ** 2)
            t["bc_right_tau_xy"] = torch.mean((right[:, 5:6] / sc["shear"]) ** 2)
            t["bc_right_Dn"] = torch.mean(((right[:, 6:7] * 1.0 + right[:, 7:8] * 0.0) / sc["electric_displacement"]) ** 2)
            t["bc_left_Dn"] = torch.mean(((left[:, 6:7] * -1.0 + left[:, 7:8] * 0.0) / sc["electric_displacement"]) ** 2)
            return t
        t["bc_top_sigma_yy"] = torch.mean((top[:, 4:5] / sc["stress_y"]) ** 2)
        t["bc_top_tau_xy"] = torch.mean((top[:, 5:6] / sc["shear"]) ** 2)
        t["bc_bottom_sigma_yy"] = torch.mean((bot[:, 4:5] / sc["stress_y"]) ** 2)
        t["bc_bottom_tau_xy"] = torch.mean((bot[:, 5:6] / sc["shear"]) ** 2)
        tx = right[:, 3:4] * 1.0 + right[:, 5:6] * 0.0
        ty = right[:, 5:6] * 1.0 + right[:, 4:5] * 0.0
        t["bc_right_traction_x"] = torch.mean((tx / sc["stress_x"]) ** 2)
        t["bc_right_traction_y"] = torch.mean(((ty - torch.full_like(ty, self.force / HEIGHT)) / sc["traction"]) ** 2)
        t["bc_right_Dn"] = torch.mean(((right[:, 6:7] * 1.0 + right[:, 7:8] * 0.0) / sc["electric_displacement"]) ** 2)
        t["bc_left_Dn"] = torch.mean(((left[:, 6:7] * -1.0 + left[:, 7:8] * 0.0) / sc["electric_displacement"]) ** 2)
        d_top = top[:, 7:8] / sc["electric_displacement"]
        phi_top = top[:, 2:3] / sc["phi"]
        t["bc_top_equipotential"] = torch.mean((phi_top - torch.mean(phi_top)) ** 2)     # identicamente 0 con el electrodo flotante duro
        t["bc_top_zero_net_charge"] = torch.mean(d_top) ** 2                              # L_Q (ec. 67)
        return t

    def loss(self, model, P):
        """(total, {termino: peso * media(r^2)}). Total en el MISMO orden de sumas del repo (lambda_BC * BC + lambda_PDE * PDE)."""
        pde = self._pde(model, P["interior"])
        bc = self._bc(model, P)
        nw, w = self.w["nw"], self.w
        if self.mode == "converse":
            total_pde = sum(pde.values()) + (nw - 1.0) * (pde["constitutive_sigma_xx"] + pde["constitutive_sigma_yy"])
            names = ["bc_top_sigma_yy", "bc_top_tau_xy", "bc_bottom_sigma_yy", "bc_bottom_tau_xy", "bc_right_sigma_xx", "bc_right_tau_xy"]
            stress = sum(bc[k] for k in names)
            electric = bc["bc_right_Dn"] + bc["bc_left_Dn"]
            BC = stress + electric
        else:
            mech = nw * (pde["constitutive_sigma_xx"] + pde["constitutive_sigma_yy"]) + pde["constitutive_tau_xy"]
            elec = pde["constitutive_D_x"] + pde["constitutive_D_y"]
            div = pde["equilibrium_x"] + pde["equilibrium_y"] + pde["gauss"]
            total_pde = 1.0 * (1 * mech + 1 * elec) + 1 * div
            loss_top = bc["bc_top_sigma_yy"] + bc["bc_top_tau_xy"]
            loss_bottom = bc["bc_bottom_sigma_yy"] + bc["bc_bottom_tau_xy"]
            loss_right = bc["bc_right_traction_x"] + bc["bc_right_traction_y"]
            side = bc["bc_right_Dn"] + bc["bc_left_Dn"]
            stress = loss_top + loss_right + loss_bottom
            electric = side + bc["bc_top_equipotential"] + bc["bc_top_zero_net_charge"]
            BC = stress + electric
        total = w["bc"] * BC + w["pde"] * total_pde
        terms = {k: w["pde"] * v * (nw if k in ("constitutive_sigma_xx", "constitutive_sigma_yy") else 1.0) for k, v in pde.items()}
        terms.update({k: w["bc"] * v for k, v in bc.items()})
        return total, terms

    @torch.no_grad()
    def interface_gap(self, model, P):
        """Maximo |N+ - N-| de los seis campos continuos en la interfaz (L_int del paper): 0 por construccion en el modelo mixto."""
        xy = P["interface"]
        top = model.forward_trace(xy, torch.ones_like(xy[:, 0:1]))
        bot = model.forward_trace(xy, -torch.ones_like(xy[:, 0:1]))
        return float((top[:, [0, 1, 2, 4, 5, 7]] - bot[:, [0, 1, 2, 4, 5, 7]]).abs().max())

    # -- residuo vectorial para el Gauss-Newton (adaptador en src/mechanisms/residual_terms.py)
    def residual_terms(self, model_fn, X):
        return _residual_terms_module().piezo_residual_terms(self, model_fn, X)

    # -- contrato de artefactos (artifacts.py): malla de evaluacion, FEM, bordes de la malla y consistencia constitutiva
    field_names = ("u", "v", "phi", "sigma_xx", "sigma_yy", "tau_xy", "D_x", "D_y")

    def grid_points(self):
        return self._reference()["xy"]

    def reference_fields(self):
        return dict(self._reference())

    @staticmethod
    def boundary_sets(xy):
        tol = 1e-12
        return {"top": xy[xy[:, 1] >= HEIGHT * (1 - tol)], "bottom": xy[xy[:, 1] <= tol], "right": xy[xy[:, 0] >= WIDTH * (1 - tol)], "left": xy[xy[:, 0] <= tol]}

    def consistency(self, model, xy):
        """Discrepancia (unidades fisicas) entre las 5 salidas mixtas predichas y las reconstruidas desde du, dv, dphi con la ley constitutiva (la limitacion que declara el paper), y su L2 relativo por campo."""
        m = MAT
        _p = next(model.parameters())
        x = torch.as_tensor(np.asarray(xy), dtype=_p.dtype, device=_p.device).clone().requires_grad_(True)
        y = model(x)
        g = [torch.autograd.grad(y[:, i].sum(), x, retain_graph=True)[0] for i in range(3)]
        ux, uy, vx, vy, phix, phiy = g[0][:, 0:1], g[0][:, 1:2], g[1][:, 0:1], g[1][:, 1:2], g[2][:, 0:1], g[2][:, 1:2]
        e31, e33 = self._coef(x[:, 1:2].detach())
        Ex, Ey = -phix, -phiy
        recon = torch.cat([m["C11"] * ux + m["C12"] * vy - e31 * Ey, m["C12"] * ux + m["C22"] * vy - e33 * Ey, m["G"] * (uy + vx), m["eps1"] * Ex,
                           e31 * ux + e33 * vy + m["eps2"] * Ey], dim=1).detach()
        pred = y[:, 3:8].detach()
        disc = pred - recon
        rel = (disc.norm(dim=0) / recon.norm(dim=0).clamp_min(1e-300)).cpu().numpy()
        return dict(discrepancy=disc.cpu().numpy().astype(np.float32), rel_l2=rel, names=("sigma_xx", "sigma_yy", "tau_xy", "D_x", "D_y"))

    # -- tensiones y desplazamiento electrico contra el FEM P3 refinado (paper, sec. 2.6): 0.001 L < x < 0.999 L, pesos de area, sin la fila de la interfaz (sx y Dx saltan)
    def _flux_reference(self):
        if getattr(self, "_flux", None) is None:
            path = (Path(os.environ["BUNDLE_DIR"]) / f"fem_flux_{self.mode}.npz" if os.environ.get("BUNDLE_DIR") else Path(__file__).resolve().parent / "reference" / f"fem_flux_{self.mode}.npz")
            z = np.load(path)
            self._flux = {k: z[k] for k in z.files}
        return self._flux

    @torch.no_grad()
    def score_flux(self, model, device="cpu"):
        ref = self._flux_reference()
        xy = ref["xy"]
        keep = ref["inside"] & (np.abs(xy[:, 1] - CENTER) > 1e-12)
        xs_, ys_ = np.unique(xy[:, 0]), np.unique(xy[:, 1])
        trap = lambda g: np.diff(g, prepend=g[0], append=g[-1]).reshape(-1)  # noqa: E731
        wx_, wy_ = (trap(xs_)[:-1] + trap(xs_)[1:]) / 2, (trap(ys_)[:-1] + trap(ys_)[1:]) / 2
        w = np.outer(wy_, wx_).ravel()[keep]                                  # pesos de area (trapecio) en la rejilla 21 x 201, orden de make_fem_reference
        o = model(torch.as_tensor(xy, dtype=torch.float64, device=torch.device(device))).double().cpu().numpy()[:, 3:8]
        names = ("sigma_xx", "sigma_yy", "tau_xy", "D_x", "D_y")
        res, s_ref, d_ref = {}, float(np.abs(ref["sigma_xx"][ref["inside"]]).max()), float(np.abs(ref["D_y"][ref["inside"]]).max())
        for i, n in enumerate(names):
            e = o[keep, i] - ref[n][keep]
            rms = float(np.sqrt((w * e ** 2).sum() / w.sum()))
            res[n] = dict(rms=rms, nrms_pct=100.0 * rms / (s_ref if i < 3 else d_ref), rel_l2=float(np.sqrt((w * e ** 2).sum() / (w * ref[n][keep] ** 2).sum())))
        res["sigma_ref_max"], res["D_y_ref_max"] = s_ref, d_ref
        return res

    # -- puntuacion contra el FEM del paper (doble precision)
    def _reference(self):
        if self._fem is None:
            path = self.fem_path or (Path(os.environ["BUNDLE_DIR"]) / f"fem_{self.mode}.npz" if os.environ.get("BUNDLE_DIR")
                                     else Path(__file__).resolve().parent / "reference" / f"fem_{self.mode}.npz")
            z = np.load(path)
            self._fem = {k: z[k] for k in ("xy", "u", "v", "phi")}
        return self._fem

    @torch.no_grad()
    def score(self, model, device="cpu", dtype=torch.float64):
        ref = self._reference()
        dev = torch.device(device)
        m = model if next(model.parameters()).dtype == torch.float64 else __import__("copy").deepcopy(model).double()
        xy = torch.as_tensor(ref["xy"], dtype=torch.float64, device=dev)
        o = m(xy).double().cpu().numpy()
        tip_xy = np.array([[0.1, 0.0]])
        tip = float(m(torch.as_tensor(tip_xy, dtype=torch.float64, device=dev))[0, 1])
        rel = lambda a, b: float(np.linalg.norm(a - b) / np.linalg.norm(b))
        top = ref["xy"][:, 1] >= HEIGHT * (1 - 1e-12)
        tip_i = int(np.argmin(np.abs(ref["xy"][:, 0] - 0.1) + ref["xy"][:, 1]))
        eu, ev, ep = rel(o[:, 0], ref["u"]), rel(o[:, 1], ref["v"]), rel(o[:, 2], ref["phi"])
        ext = {}
        for i, n in enumerate(("u", "v", "phi")):                                   # RMSE, MAE y maximo absoluto sobre la rejilla (paper, sec. 2.6)
            e = o[:, i] - ref[n]
            ext.update({f"rmse_{n}": float(np.sqrt((e ** 2).mean())), f"mae_{n}": float(np.abs(e).mean()), f"max_abs_{n}": float(np.abs(e).max())})
        return dict(**ext, l2re_u=eu, l2re_v=ev, l2re_phi=ep, l2re_mean_components=(eu + ev + ep) / 3.0, l2re_mean_uv=(eu + ev) / 2.0,
                    l2re_flat=rel(np.concatenate([o[:, 0], o[:, 1], o[:, 2]]), np.concatenate([ref["u"], ref["v"], ref["phi"]])),
                    tip_v=tip, tip_v_fem=float(ref["v"][tip_i]), phi_top_mean=float(o[top, 2].mean()), phi_top_mean_fem=float(ref["phi"][top].mean()))


def _residual_terms_module():
    """src/mechanisms/residual_terms.py: junto a este archivo (bundle plano), o en el repo; mismo nombre de modulo que gram_preconditioning (una sola instancia)."""
    if "mechanisms_residual_terms" in sys.modules:
        return sys.modules["mechanisms_residual_terms"]
    here = Path(__file__).resolve().parent
    cands = [here / "residual_terms.py"] + [p / "src/mechanisms/residual_terms.py" for p in here.parents]
    import importlib.util
    for c in cands:
        if c.exists():
            spec = importlib.util.spec_from_file_location("mechanisms_residual_terms", str(c))
            mod = importlib.util.module_from_spec(spec)
            sys.modules["mechanisms_residual_terms"] = mod
            spec.loader.exec_module(mod)
            return mod
    raise ImportError("residual_terms.py no encontrado")
