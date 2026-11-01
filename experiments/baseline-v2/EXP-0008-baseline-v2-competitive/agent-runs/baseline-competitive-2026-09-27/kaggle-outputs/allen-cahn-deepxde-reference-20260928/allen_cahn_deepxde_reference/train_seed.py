import argparse, datetime, hashlib, json, os, sys, time
os.environ['DDE_BACKEND'] = 'pytorch'
import numpy as np
import torch
import deepxde as dde
from scipy.io import loadmat

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, required=True)
parser.add_argument('--mat', required=True)
parser.add_argument('--expected-sha256', required=True)
parser.add_argument('--deadline', type=float, required=True)
parser.add_argument('--progress', required=True)
parser.add_argument('--result', required=True)
parser.add_argument('--gpu-count', type=int, required=True)
args = parser.parse_args()
started = time.monotonic()

def atomic_json(path, obj):
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)

def base_result():
    return {
        'seed': args.seed, 'status': 'running', 'recipe_complete': False,
        'adam_steps': 0, 'lbfgs_completed': False,
        'field_l2_relative': None, 'mean_abs_residual': None,
        'ic_max_abs_error': None, 'bc_max_abs_error': None,
        'energy_initial': None, 'energy_final': None, 'energy_delta': None,
        'energy_mean': None, 'elapsed_seconds': 0.0,
        'active_gpu_seconds': 0.0, 'allocated_gpu_seconds': 0.0,
        'allocated_gpu_hours': 0.0, 'cost_usd_estimate': None,
        'cost_basis': 'Kaggle does not expose monetary billing; GPU time is recorded.',
        'stop_reason': None,
    }

result = base_result()
def finish(status, reason=None):
    elapsed = time.monotonic() - started
    result['status'] = status
    result['stop_reason'] = reason
    result['elapsed_seconds'] = round(elapsed, 3)
    result['active_gpu_seconds'] = round(elapsed, 3)
    result['allocated_gpu_seconds'] = round(elapsed * args.gpu_count, 3)
    result['allocated_gpu_hours'] = round(elapsed * args.gpu_count / 3600.0, 6)
    atomic_json(args.result, result)
    atomic_json(args.progress, result)

try:
    mat_sha = hashlib.sha256(open(args.mat, 'rb').read()).hexdigest()
    if mat_sha != args.expected_sha256:
        raise RuntimeError('Dataset SHA-256 changed between setup and worker.')
    raw = loadmat(args.mat)
    t = raw['t'].reshape(-1).astype(np.float64)
    u_true = raw['u'].astype(np.float64)
    if 'x' in raw:
        x = raw['x'].reshape(-1).astype(np.float64)
    else:
        x = np.linspace(-1.0, 1.0, u_true.shape[1], dtype=np.float64)
    if u_true.shape != (t.size, x.size):
        raise ValueError(f'Incompatible reference shapes: t={t.shape}, x={x.shape}, u={u_true.shape}')

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    dde.config.set_random_seed(args.seed)
    torch.cuda.set_device(0)

    geom = dde.geometry.Interval(-1, 1)
    timedomain = dde.geometry.TimeDomain(0, 1)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)
    d = 0.001

    def pde(inputs, y):
        dy_t = dde.grad.jacobian(y, inputs, i=0, j=1)
        dy_xx = dde.grad.hessian(y, inputs, i=0, j=0)
        return dy_t - d * dy_xx - 5 * (y - y**3)

    def output_transform(inputs, y):
        xcoord = inputs[:, 0:1]
        tcoord = inputs[:, 1:2]
        return xcoord**2 * torch.cos(np.pi * xcoord) + tcoord * (1 - xcoord**2) * y

    class Watchdog(dde.callbacks.Callback):
        def __init__(self, phase):
            super().__init__()
            self.phase = phase
            self.triggered = False
            self.last_write = 0
        def on_epoch_end(self):
            now = time.monotonic()
            state = self.model.train_state
            step = int(getattr(state, 'step', 0))
            if now - self.last_write >= 20:
                result['phase'] = self.phase
                result['phase_step'] = step
                result['elapsed_seconds'] = round(now - started, 3)
                atomic_json(args.progress, result)
                self.last_write = now
            if now >= args.deadline:
                self.triggered = True
                self.model.stop_training = True

    data = dde.data.TimePDE(
        geomtime, pde, [], num_domain=8000, num_boundary=400, num_initial=800
    )
    net = dde.nn.FNN([2] + [20] * 3 + [1], 'tanh', 'Glorot normal')
    net.apply_output_transform(output_transform)
    model = dde.Model(data, net)
    model.compile('adam', lr=1e-3)
    adam_watchdog = Watchdog('adam')
    adam_step_before = int(getattr(model.train_state, 'step', 0))
    model.train(iterations=40000, callbacks=[adam_watchdog], display_every=10000)
    adam_steps = max(0, int(getattr(model.train_state, 'step', 0)) - adam_step_before)
    result['adam_steps'] = adam_steps
    result['training'] = {'adam_requested': 40000, 'adam_completed': adam_steps, 'lbfgs_requested': True, 'lbfgs_completed': False}
    if adam_watchdog.triggered or adam_steps < 40000:
        finish('truncated', 'Global watchdog stopped Adam before 40000 steps.')
        sys.exit(0)

    lbfgs_watchdog = Watchdog('lbfgs')
    model.compile('L-BFGS')
    model.train(callbacks=[lbfgs_watchdog], display_every=1000)
    if lbfgs_watchdog.triggered:
        finish('truncated', 'Global watchdog stopped L-BFGS before completion.')
        sys.exit(0)
    result['lbfgs_completed'] = True
    result['training']['lbfgs_completed'] = True

    # Conserva margen para cerrar archivos y consolidar las cinco filas.
    if time.monotonic() + 90 >= args.deadline:
        finish('truncated_evaluation', 'Entrenamiento terminó, pero no quedó margen para completar la evaluación.')
        sys.exit(0)

    xx, tt = np.meshgrid(x, t)
    X = np.column_stack((xx.ravel(), tt.ravel())).astype(np.float32)
    y_flat = u_true.reshape(-1, 1).astype(np.float32)
    pred_parts, residual_parts, energy_parts = [], [], []

    def energy_density(inputs, y):
        ux = dde.grad.jacobian(y, inputs, i=0, j=0)
        return 0.5 * d * ux**2 + 1.25 * (y**2 - 1.0)**2

    for start in range(0, len(X), 2048):
        if time.monotonic() + 30 >= args.deadline:
            finish('truncated_evaluation', 'Watchdog durante la evaluación del campo.')
            sys.exit(0)
        batch = X[start:start + 2048]
        pred_parts.append(model.predict(batch))
        residual_parts.append(model.predict(batch, operator=pde))
        energy_parts.append(model.predict(batch, operator=energy_density))

    y_pred = np.vstack(pred_parts)
    residual = np.vstack(residual_parts)
    energy_density_grid = np.vstack(energy_parts).reshape(u_true.shape)
    field_l2 = float(dde.metrics.l2_relative_error(y_flat, y_pred))
    mean_abs_residual = float(np.mean(np.abs(residual)))

    x_ic = np.column_stack((x, np.zeros_like(x))).astype(np.float32)
    ic_pred = model.predict(x_ic).reshape(-1)
    ic_target = x**2 * np.cos(np.pi * x)
    ic_error = float(np.max(np.abs(ic_pred - ic_target)))
    t_bc = t.astype(np.float32)
    left = np.column_stack((np.full_like(t_bc, -1.0), t_bc))
    right = np.column_stack((np.full_like(t_bc, 1.0), t_bc))
    bc_pred = np.vstack((model.predict(left), model.predict(right))).reshape(-1)
    bc_error = float(np.max(np.abs(bc_pred + 1.0)))

    dx = np.diff(x)
    energy_by_time = np.sum(
        0.5 * (energy_density_grid[:, :-1] + energy_density_grid[:, 1:]) * dx[None, :],
        axis=1,
    )
    result.update({
        'field_l2_relative': field_l2,
        'mean_abs_residual': mean_abs_residual,
        'ic_max_abs_error': ic_error,
        'bc_max_abs_error': bc_error,
        'energy_initial': float(energy_by_time[0]),
        'energy_final': float(energy_by_time[-1]),
        'energy_delta': float(energy_by_time[-1] - energy_by_time[0]),
        'energy_mean': float(np.mean(energy_by_time)),
        'energy_definition': 'integral_x [0.5*d*u_x^2 + 1.25*(u^2-1)^2] dx; trapezoid on official x raster',
        'metrics_grid_shape': list(u_true.shape),
        'recipe_complete': True,
        'training': {
            'adam_requested': 40000,
            'adam_completed': adam_steps,
            'lbfgs_requested': True,
            'lbfgs_completed': True,
        },
        'stop_reason': None,
    })
    finish('complete')
except SystemExit:
    raise
except Exception as exc:
    finish('worker_failed', f'{type(exc).__name__}: {exc}')
    raise
