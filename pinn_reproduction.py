from __future__ import annotations

import json
import math
import random
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import scipy.io
from scipy.optimize import minimize
import torch
from torch import nn


ROOT = Path(__file__).resolve().parent
REFERENCE_ROOT = ROOT / "reference_official"
DATA_ROOT = REFERENCE_ROOT / "data"
IRK_ROOT = REFERENCE_ROOT / "irk"
RESULTS_ROOT = ROOT / "results"

DEFAULT_DTYPE = torch.float32
NU_BURGERS = 0.01 / np.pi

PAPER_MAIN_METRICS = {
    "ct_burgers": 6.7e-4,
    "ct_schrodinger": 1.97e-3,
    "dt_burgers": 8.2e-4,
    "dt_ac": 6.99e-3,
}

PAPER_TABLE_1 = np.array(
    [
        [2.9e-01, 4.4e-01, 8.9e-01, 1.2e00, 9.9e-02, 4.2e-02],
        [6.5e-02, 1.1e-02, 5.0e-01, 9.6e-03, 4.6e-01, 7.5e-02],
        [3.6e-01, 1.2e-02, 1.7e-01, 5.9e-03, 1.9e-03, 8.2e-03],
        [5.5e-03, 1.0e-03, 3.2e-03, 7.8e-03, 4.9e-02, 4.5e-03],
        [6.6e-02, 2.7e-01, 7.2e-03, 6.8e-04, 2.2e-03, 6.7e-04],
        [1.5e-01, 2.3e-03, 8.2e-04, 8.9e-04, 6.1e-04, 4.9e-04],
    ],
    dtype=np.float64,
)

PAPER_TABLE_2 = np.array(
    [
        [7.4e-02, 5.3e-02, 1.0e-01],
        [3.0e-03, 9.4e-04, 6.4e-04],
        [9.6e-03, 1.3e-03, 6.1e-04],
        [2.5e-03, 9.6e-04, 5.6e-04],
    ],
    dtype=np.float64,
)

PAPER_TABLE_3 = np.array(
    [
        [4.1e-02, 4.1e-02, 1.5e-01],
        [2.7e-03, 5.0e-03, 2.4e-03],
        [3.6e-03, 1.9e-03, 9.5e-04],
    ],
    dtype=np.float64,
)

PAPER_TABLE_4 = np.array(
    [
        [3.5e-02, 1.1e-01, 2.3e-01, 3.8e-01],
        [5.4e-03, 5.1e-02, 9.3e-02, 2.2e-01],
        [1.2e-03, 1.5e-02, 3.6e-02, 5.4e-02],
        [6.7e-04, 1.8e-03, 8.7e-03, 5.8e-02],
        [5.1e-04, 7.6e-02, 8.4e-04, 1.1e-03],
        [7.4e-04, 5.2e-04, 4.2e-04, 7.0e-04],
        [4.5e-04, 4.8e-04, 1.2e-03, 7.8e-04],
        [5.1e-04, 5.7e-04, 1.8e-02, 1.2e-03],
        [4.1e-04, 3.8e-04, 4.2e-04, 8.2e-04],
    ],
    dtype=np.float64,
)


@dataclass
class TrainConfig:
    adam_steps: int = 0
    adam_lr: float = 1e-3
    lbfgs_maxiter: int = 50000
    lbfgs_maxfun: int = 50000
    lbfgs_maxcor: int = 50
    lbfgs_maxls: int = 50
    lbfgs_ftol: float = float(np.finfo(float).eps)
    adam_log_every: int = 1000


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(device: str | None = None) -> torch.device:
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def to_tensor(array: np.ndarray, device: torch.device, dtype: torch.dtype = DEFAULT_DTYPE) -> torch.Tensor:
    return torch.as_tensor(array, dtype=dtype, device=device)


def relative_l2_error(target: np.ndarray, pred: np.ndarray) -> float:
    return float(np.linalg.norm(target - pred, 2) / np.linalg.norm(target, 2))


def lhs(dim: int, samples: int, lb: np.ndarray, ub: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
    cut = np.linspace(0.0, 1.0, samples + 1)
    u = rng.rand(samples, dim)
    a = cut[:-1]
    b = cut[1:]
    points = np.zeros((samples, dim), dtype=np.float64)
    for j in range(dim):
        points[:, j] = u[:, j] * (b - a) + a
        rng.shuffle(points[:, j])
    return lb + (ub - lb) * points


def ensure_results_dir() -> None:
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


@lru_cache(maxsize=1)
def load_burgers_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = scipy.io.loadmat(DATA_ROOT / "burgers_shock.mat")
    t = data["t"].flatten()[:, None]
    x = data["x"].flatten()[:, None]
    exact = np.real(data["usol"]).T
    return x.astype(np.float64), t.astype(np.float64), exact.astype(np.float64)


@lru_cache(maxsize=1)
def load_nls_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = scipy.io.loadmat(DATA_ROOT / "NLS.mat")
    x = data["x"].flatten()[:, None]
    t = data["tt"].flatten()[:, None]
    exact = data["uu"]
    return x.astype(np.float64), t.astype(np.float64), exact


@lru_cache(maxsize=1)
def load_ac_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = scipy.io.loadmat(DATA_ROOT / "AC.mat")
    x = data["x"].flatten()[:, None]
    t = data["tt"].flatten()[:, None]
    exact = np.real(data["uu"]).T
    return x.astype(np.float64), t.astype(np.float64), exact.astype(np.float64)


@lru_cache(maxsize=None)
def load_irk_weights(q: int) -> np.ndarray:
    raw = np.loadtxt(IRK_ROOT / f"Butcher_IRK{q}.txt").astype(np.float64).reshape(-1)
    return raw[: q * q + q].reshape(q + 1, q)


class ManualTanhMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_layers: int,
        width: int,
        lb: np.ndarray,
        ub: np.ndarray,
        dtype: torch.dtype = DEFAULT_DTYPE,
    ) -> None:
        super().__init__()
        sizes = [input_dim] + [width] * hidden_layers + [output_dim]
        self.layers = nn.ModuleList(
            [nn.Linear(sizes[i], sizes[i + 1], bias=True, dtype=dtype) for i in range(len(sizes) - 1)]
        )
        self.register_buffer("lb", torch.tensor(lb, dtype=dtype))
        self.register_buffer("ub", torch.tensor(ub, dtype=dtype))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for layer in self.layers:
            std = math.sqrt(2.0 / (layer.in_features + layer.out_features))
            nn.init.trunc_normal_(layer.weight, std=std)
            nn.init.zeros_(layer.bias)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return 2.0 * (x - self.lb) / (self.ub - self.lb) - 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.normalize(x)
        for layer in self.layers[:-1]:
            h = torch.tanh(layer(h))
        return self.layers[-1](h)

    def forward_with_grads_1d(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.normalize(x)
        scale = (2.0 / (self.ub - self.lb)).reshape(1, -1)
        dh = scale.expand_as(h)
        d2h = torch.zeros_like(h)

        for layer in self.layers[:-1]:
            wt = layer.weight.t()
            z = h @ wt + layer.bias
            dz = dh @ wt
            d2z = d2h @ wt
            h = torch.tanh(z)
            sech2 = 1.0 - h.pow(2)
            dh = sech2 * dz
            d2h = sech2 * d2z - 2.0 * h * sech2 * dz.pow(2)

        out = self.layers[-1](h)
        wt = self.layers[-1].weight.t()
        out_x = dh @ wt
        out_xx = d2h @ wt
        return out, out_x, out_xx

    def forward_with_grads_2d(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.normalize(x)
        scale = 2.0 / (self.ub - self.lb)
        dh_dx = torch.zeros_like(h)
        dh_dt = torch.zeros_like(h)
        dh_dx[:, 0] = scale[0]
        dh_dt[:, 1] = scale[1]
        d2h_dxx = torch.zeros_like(h)

        for layer in self.layers[:-1]:
            wt = layer.weight.t()
            z = h @ wt + layer.bias
            dz_dx = dh_dx @ wt
            dz_dt = dh_dt @ wt
            d2z_dxx = d2h_dxx @ wt
            h = torch.tanh(z)
            sech2 = 1.0 - h.pow(2)
            dh_dx = sech2 * dz_dx
            dh_dt = sech2 * dz_dt
            d2h_dxx = sech2 * d2z_dxx - 2.0 * h * sech2 * dz_dx.pow(2)

        out = self.layers[-1](h)
        wt = self.layers[-1].weight.t()
        out_x = dh_dx @ wt
        out_t = dh_dt @ wt
        out_xx = d2h_dxx @ wt
        return out, out_x, out_t, out_xx


def flatten_parameters(parameters: Iterable[torch.nn.Parameter]) -> np.ndarray:
    return np.concatenate([p.detach().cpu().numpy().ravel().astype(np.float64) for p in parameters])


def flatten_gradients(parameters: Iterable[torch.nn.Parameter]) -> np.ndarray:
    chunks: list[np.ndarray] = []
    for p in parameters:
        grad = p.grad if p.grad is not None else torch.zeros_like(p)
        chunks.append(grad.detach().cpu().numpy().ravel().astype(np.float64))
    return np.concatenate(chunks)


def assign_flat_parameters(parameters: list[torch.nn.Parameter], flat: np.ndarray) -> None:
    offset = 0
    with torch.no_grad():
        for p in parameters:
            size = p.numel()
            value = torch.from_numpy(flat[offset : offset + size]).to(device=p.device, dtype=p.dtype).view_as(p)
            p.copy_(value)
            offset += size


def train_with_scipy(
    model: nn.Module,
    loss_fn: Callable[[], torch.Tensor],
    config: TrainConfig,
    log_prefix: str = "",
) -> dict[str, Any]:
    parameters = [p for p in model.parameters()]
    history: dict[str, Any] = {"adam_losses": []}

    if config.adam_steps > 0:
        optimizer = torch.optim.Adam(parameters, lr=config.adam_lr)
        start = time.time()
        for step in range(config.adam_steps):
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn()
            loss.backward()
            optimizer.step()
            if step == 0 or (step + 1) % config.adam_log_every == 0 or step + 1 == config.adam_steps:
                value = float(loss.detach().cpu().item())
                history["adam_losses"].append({"step": step + 1, "loss": value})
                print(f"{log_prefix}Adam step {step + 1}/{config.adam_steps}: loss={value:.3e}")
        history["adam_seconds"] = time.time() - start

    x0 = flatten_parameters(parameters)

    def objective(flat: np.ndarray) -> tuple[float, np.ndarray]:
        assign_flat_parameters(parameters, flat)
        model.zero_grad(set_to_none=True)
        loss = loss_fn()
        loss.backward()
        return float(loss.detach().cpu().item()), flatten_gradients(parameters)

    start = time.time()
    result = minimize(
        objective,
        x0=x0,
        jac=True,
        method="L-BFGS-B",
        options={
            "maxiter": config.lbfgs_maxiter,
            "maxfun": config.lbfgs_maxfun,
            "maxcor": config.lbfgs_maxcor,
            "maxls": config.lbfgs_maxls,
            "ftol": config.lbfgs_ftol,
        },
    )
    history["lbfgs_seconds"] = time.time() - start
    history["lbfgs_result"] = {
        "success": bool(result.success),
        "status": int(result.status),
        "message": str(result.message),
        "nit": int(result.nit),
        "nfev": int(result.nfev),
        "fun": float(result.fun),
    }
    assign_flat_parameters(parameters, result.x)
    return history


def make_ct_burgers_loss(
    model: ManualTanhMLP,
    x_u: torch.Tensor,
    u: torch.Tensor,
    x_f: torch.Tensor,
) -> Callable[[], torch.Tensor]:
    zeros_f = torch.zeros((x_f.shape[0], 1), device=x_f.device, dtype=x_f.dtype)

    def loss_fn() -> torch.Tensor:
        u_pred = model(x_u)
        u_f, u_x_f, u_t_f, u_xx_f = model.forward_with_grads_2d(x_f)
        f_pred = u_t_f + u_f * u_x_f - NU_BURGERS * u_xx_f
        return torch.mean((u_pred - u) ** 2) + torch.mean((f_pred - zeros_f) ** 2)

    return loss_fn


def make_ct_schrodinger_loss(
    model: ManualTanhMLP,
    x0: torch.Tensor,
    u0: torch.Tensor,
    v0: torch.Tensor,
    x_lb: torch.Tensor,
    x_ub: torch.Tensor,
    x_f: torch.Tensor,
) -> Callable[[], torch.Tensor]:
    zeros_f = torch.zeros((x_f.shape[0], 1), device=x_f.device, dtype=x_f.dtype)

    def loss_fn() -> torch.Tensor:
        uv0 = model(x0)
        uv_lb, uv_x_lb, _, _ = model.forward_with_grads_2d(x_lb)
        uv_ub, uv_x_ub, _, _ = model.forward_with_grads_2d(x_ub)
        uv_f, uv_x_f, uv_t_f, uv_xx_f = model.forward_with_grads_2d(x_f)

        u0_pred = uv0[:, 0:1]
        v0_pred = uv0[:, 1:2]

        u_lb = uv_lb[:, 0:1]
        v_lb = uv_lb[:, 1:2]
        u_ub = uv_ub[:, 0:1]
        v_ub = uv_ub[:, 1:2]

        u_x_lb = uv_x_lb[:, 0:1]
        v_x_lb = uv_x_lb[:, 1:2]
        u_x_ub = uv_x_ub[:, 0:1]
        v_x_ub = uv_x_ub[:, 1:2]

        u_f = uv_f[:, 0:1]
        v_f = uv_f[:, 1:2]
        u_t_f = uv_t_f[:, 0:1]
        v_t_f = uv_t_f[:, 1:2]
        u_xx_f = uv_xx_f[:, 0:1]
        v_xx_f = uv_xx_f[:, 1:2]
        modulus = u_f**2 + v_f**2

        f_u = u_t_f + 0.5 * v_xx_f + modulus * v_f
        f_v = v_t_f - 0.5 * u_xx_f - modulus * u_f

        return (
            torch.mean((u0_pred - u0) ** 2)
            + torch.mean((v0_pred - v0) ** 2)
            + torch.mean((u_lb - u_ub) ** 2)
            + torch.mean((v_lb - v_ub) ** 2)
            + torch.mean((u_x_lb - u_x_ub) ** 2)
            + torch.mean((v_x_lb - v_x_ub) ** 2)
            + torch.mean((f_u - zeros_f) ** 2)
            + torch.mean((f_v - zeros_f) ** 2)
        )

    return loss_fn


def make_dt_burgers_loss(
    model: ManualTanhMLP,
    x0: torch.Tensor,
    u0: torch.Tensor,
    x1: torch.Tensor,
    dt: float,
    irk: torch.Tensor,
) -> Callable[[], torch.Tensor]:
    def loss_fn() -> torch.Tensor:
        u1, u1_x, u1_xx = model.forward_with_grads_1d(x0)
        u = u1[:, :-1]
        u_x = u1_x[:, :-1]
        u_xx = u1_xx[:, :-1]
        f = -u * u_x + NU_BURGERS * u_xx
        u0_pred = u1 - dt * (f @ irk.t())
        u1_boundary = model(x1)
        return torch.sum((u0_pred - u0) ** 2) + torch.sum(u1_boundary**2)

    return loss_fn


def make_dt_ac_loss(
    model: ManualTanhMLP,
    x0: torch.Tensor,
    u0: torch.Tensor,
    x1: torch.Tensor,
    dt: float,
    irk: torch.Tensor,
) -> Callable[[], torch.Tensor]:
    def loss_fn() -> torch.Tensor:
        u1, u1_x, u1_xx = model.forward_with_grads_1d(x0)
        u = u1[:, :-1]
        u_xx = u1_xx[:, :-1]
        f = 5.0 * u - 5.0 * u**3 + 0.0001 * u_xx
        u0_pred = u1 - dt * (f @ irk.t())

        u1_boundary, u1_x_boundary, _ = model.forward_with_grads_1d(x1)
        return (
            torch.sum((u0_pred - u0) ** 2)
            + torch.sum((u1_boundary[0, :] - u1_boundary[1, :]) ** 2)
            + torch.sum((u1_x_boundary[0, :] - u1_x_boundary[1, :]) ** 2)
        )

    return loss_fn


def _common_payload(name: str, seed: int, train_history: dict[str, Any], error: float, paper_value: float) -> dict[str, Any]:
    return {
        "experiment": name,
        "seed": seed,
        "error": error,
        "paper_error": paper_value,
        "absolute_delta": abs(error - paper_value),
        "relative_ratio": error / paper_value,
        "train_history": train_history,
    }


def run_ct_burgers(
    seed: int,
    device: str | None = None,
    n_u: int = 100,
    n_f: int = 10000,
    hidden_layers: int = 8,
    width: int = 20,
    train_config: TrainConfig | None = None,
) -> dict[str, Any]:
    device_t = get_device(device)
    set_seed(seed)
    rng = np.random.RandomState(seed)
    x, t, exact = load_burgers_data()

    x_grid, t_grid = np.meshgrid(x, t)
    x_star = np.hstack((x_grid.flatten()[:, None], t_grid.flatten()[:, None]))
    u_star = exact.flatten()[:, None]
    lb = x_star.min(axis=0)
    ub = x_star.max(axis=0)

    xx1 = np.hstack((x_grid[0:1, :].T, t_grid[0:1, :].T))
    uu1 = exact[0:1, :].T
    xx2 = np.hstack((x_grid[:, 0:1], t_grid[:, 0:1]))
    uu2 = exact[:, 0:1]
    xx3 = np.hstack((x_grid[:, -1:], t_grid[:, -1:]))
    uu3 = exact[:, -1:]

    x_u_full = np.vstack([xx1, xx2, xx3])
    u_full = np.vstack([uu1, uu2, uu3])
    x_f = np.vstack([lhs(2, n_f, lb, ub, rng), x_u_full])
    idx = rng.choice(x_u_full.shape[0], n_u, replace=False)
    x_u = x_u_full[idx, :]
    u = u_full[idx, :]

    model = ManualTanhMLP(2, 1, hidden_layers, width, lb, ub).to(device_t)
    loss_fn = make_ct_burgers_loss(
        model,
        to_tensor(x_u, device_t),
        to_tensor(u, device_t),
        to_tensor(x_f, device_t),
    )
    history = train_with_scipy(model, loss_fn, train_config or TrainConfig(), "[ct_burgers] ")
    u_pred = model(to_tensor(x_star, device_t)).detach().cpu().numpy()
    error = relative_l2_error(u_star, u_pred)

    payload = _common_payload("ct_burgers", seed, history, error, PAPER_MAIN_METRICS["ct_burgers"])
    payload["config"] = {"n_u": n_u, "n_f": n_f, "hidden_layers": hidden_layers, "width": width}
    return payload


def run_ct_schrodinger(
    seed: int,
    device: str | None = None,
    n0: int = 50,
    n_b: int = 50,
    n_f: int = 20000,
    hidden_layers: int = 4,
    width: int = 100,
    train_config: TrainConfig | None = None,
) -> dict[str, Any]:
    device_t = get_device(device)
    set_seed(seed)
    rng = np.random.RandomState(seed)
    x, t, exact = load_nls_data()

    exact_u = np.real(exact)
    exact_v = np.imag(exact)
    exact_h = np.sqrt(exact_u**2 + exact_v**2)
    x_grid, t_grid = np.meshgrid(x, t)
    x_star = np.hstack((x_grid.flatten()[:, None], t_grid.flatten()[:, None]))
    h_star = exact_h.T.flatten()[:, None]

    lb = np.array([-5.0, 0.0], dtype=np.float64)
    ub = np.array([5.0, np.pi / 2.0], dtype=np.float64)

    idx_x = rng.choice(x.shape[0], n0, replace=False)
    x0 = x[idx_x, :]
    u0 = exact_u[idx_x, 0:1]
    v0 = exact_v[idx_x, 0:1]

    idx_t = rng.choice(t.shape[0], n_b, replace=False)
    tb = t[idx_t, :]

    x0_full = np.concatenate((x0, np.zeros_like(x0)), axis=1)
    x_lb = np.concatenate((np.full_like(tb, lb[0]), tb), axis=1)
    x_ub = np.concatenate((np.full_like(tb, ub[0]), tb), axis=1)
    x_f = lhs(2, n_f, lb, ub, rng)

    model = ManualTanhMLP(2, 2, hidden_layers, width, lb, ub).to(device_t)
    loss_fn = make_ct_schrodinger_loss(
        model,
        to_tensor(x0_full, device_t),
        to_tensor(u0, device_t),
        to_tensor(v0, device_t),
        to_tensor(x_lb, device_t),
        to_tensor(x_ub, device_t),
        to_tensor(x_f, device_t),
    )
    history = train_with_scipy(
        model,
        loss_fn,
        train_config or TrainConfig(adam_steps=50000),
        "[ct_schrodinger] ",
    )

    uv_pred = model(to_tensor(x_star, device_t)).detach().cpu().numpy()
    h_pred = np.sqrt(uv_pred[:, 0:1] ** 2 + uv_pred[:, 1:2] ** 2)
    error = relative_l2_error(h_star, h_pred)

    payload = _common_payload("ct_schrodinger", seed, history, error, PAPER_MAIN_METRICS["ct_schrodinger"])
    payload["config"] = {"n0": n0, "n_b": n_b, "n_f": n_f, "hidden_layers": hidden_layers, "width": width}
    return payload


def run_dt_burgers(
    seed: int,
    device: str | None = None,
    q: int = 500,
    skip: int = 80,
    n_points: int = 250,
    hidden_layers: int = 3,
    width: int = 50,
    train_config: TrainConfig | None = None,
) -> dict[str, Any]:
    device_t = get_device(device)
    set_seed(seed)
    rng = np.random.RandomState(seed)
    x, t, exact = load_burgers_data()

    idx_t0 = 10
    idx_t1 = idx_t0 + skip
    dt = float(t[idx_t1] - t[idx_t0])
    lb = np.array([-1.0], dtype=np.float64)
    ub = np.array([1.0], dtype=np.float64)

    idx_x = rng.choice(exact.shape[1], n_points, replace=False)
    x0 = x[idx_x, :]
    u0 = exact[idx_t0 : idx_t0 + 1, idx_x].T
    x1 = np.vstack((lb, ub))
    irk = load_irk_weights(q)

    model = ManualTanhMLP(1, q + 1, hidden_layers, width, lb, ub).to(device_t)
    loss_fn = make_dt_burgers_loss(
        model,
        to_tensor(x0, device_t),
        to_tensor(u0, device_t),
        to_tensor(x1, device_t),
        dt,
        to_tensor(irk, device_t),
    )
    history = train_with_scipy(
        model,
        loss_fn,
        train_config or TrainConfig(adam_steps=10000),
        "[dt_burgers] ",
    )

    u1_pred = model(to_tensor(x, device_t)).detach().cpu().numpy()
    error = relative_l2_error(exact[idx_t1, :], u1_pred[:, -1])

    payload = _common_payload("dt_burgers", seed, history, error, PAPER_MAIN_METRICS["dt_burgers"])
    payload["config"] = {
        "q": q,
        "skip": skip,
        "dt": dt,
        "n_points": n_points,
        "hidden_layers": hidden_layers,
        "width": width,
    }
    return payload


def run_dt_ac(
    seed: int,
    device: str | None = None,
    q: int = 100,
    n_points: int = 200,
    hidden_layers: int = 4,
    width: int = 200,
    idx_t0: int = 20,
    idx_t1: int = 180,
    train_config: TrainConfig | None = None,
) -> dict[str, Any]:
    device_t = get_device(device)
    set_seed(seed)
    rng = np.random.RandomState(seed)
    x, t, exact = load_ac_data()

    dt = float(t[idx_t1] - t[idx_t0])
    lb = np.array([-1.0], dtype=np.float64)
    ub = np.array([1.0], dtype=np.float64)

    idx_x = rng.choice(exact.shape[1], n_points, replace=False)
    x0 = x[idx_x, :]
    u0 = exact[idx_t0 : idx_t0 + 1, idx_x].T
    x1 = np.vstack((lb, ub))
    irk = load_irk_weights(q)

    model = ManualTanhMLP(1, q + 1, hidden_layers, width, lb, ub).to(device_t)
    loss_fn = make_dt_ac_loss(
        model,
        to_tensor(x0, device_t),
        to_tensor(u0, device_t),
        to_tensor(x1, device_t),
        dt,
        to_tensor(irk, device_t),
    )
    history = train_with_scipy(model, loss_fn, train_config or TrainConfig(adam_steps=10000), "[dt_ac] ")

    u1_pred = model(to_tensor(x, device_t)).detach().cpu().numpy()
    error = relative_l2_error(exact[idx_t1, :], u1_pred[:, -1])

    payload = _common_payload("dt_ac", seed, history, error, PAPER_MAIN_METRICS["dt_ac"])
    payload["config"] = {
        "q": q,
        "dt": dt,
        "n_points": n_points,
        "hidden_layers": hidden_layers,
        "width": width,
        "idx_t0": idx_t0,
        "idx_t1": idx_t1,
    }
    return payload


def run_multi_seed(
    run_fn: Callable[..., dict[str, Any]],
    seeds: list[int],
    **kwargs: Any,
) -> dict[str, Any]:
    runs = [run_fn(seed=seed, **kwargs) for seed in seeds]
    errors = np.array([run["error"] for run in runs], dtype=np.float64)
    paper_error = float(runs[0]["paper_error"])
    return {
        "runs": runs,
        "summary": {
            "count": int(len(runs)),
            "mean_error": float(errors.mean()),
            "std_error": float(errors.std(ddof=0)),
            "min_error": float(errors.min()),
            "max_error": float(errors.max()),
            "paper_error": paper_error,
            "mean_over_paper": float(errors.mean() / paper_error),
        },
    }


def _checkpoint_table_payload(path: Path | None, payload: dict[str, Any]) -> None:
    if path is not None:
        save_json(path, payload)


def run_ct_burgers_tables(
    seed: int,
    device: str | None = None,
    train_config: TrainConfig | None = None,
    checkpoint_path: Path | None = None,
) -> dict[str, Any]:
    n_u_values = [20, 40, 60, 80, 100, 200]
    n_f_values = [2000, 4000, 6000, 7000, 8000, 10000]
    hidden_values = [2, 4, 6, 8]
    width_values = [10, 20, 40]

    table1 = np.zeros((len(n_u_values), len(n_f_values)), dtype=np.float64)
    for i, n_u in enumerate(n_u_values):
        for j, n_f in enumerate(n_f_values):
            print(f"[ct_burgers_table1] seed={seed} cell=({i + 1},{j + 1}) N_u={n_u} N_f={n_f}")
            result = run_ct_burgers(
                seed=seed,
                device=device,
                n_u=n_u,
                n_f=n_f,
                hidden_layers=8,
                width=20,
                train_config=train_config,
            )
            table1[i, j] = result["error"]
            _checkpoint_table_payload(
                checkpoint_path,
                {
                    "seed": seed,
                    "table_1": table1.tolist(),
                    "table_2": None,
                    "paper_table_1": PAPER_TABLE_1.tolist(),
                    "paper_table_2": PAPER_TABLE_2.tolist(),
                    "stage": f"table_1_{i + 1}_{j + 1}",
                },
            )

    table2 = np.zeros((len(hidden_values), len(width_values)), dtype=np.float64)
    for i, hidden in enumerate(hidden_values):
        for j, width in enumerate(width_values):
            print(f"[ct_burgers_table2] seed={seed} cell=({i + 1},{j + 1}) hidden_layers={hidden} width={width}")
            result = run_ct_burgers(
                seed=seed,
                device=device,
                n_u=100,
                n_f=10000,
                hidden_layers=hidden,
                width=width,
                train_config=train_config,
            )
            table2[i, j] = result["error"]
            _checkpoint_table_payload(
                checkpoint_path,
                {
                    "seed": seed,
                    "table_1": table1.tolist(),
                    "table_2": table2.tolist(),
                    "paper_table_1": PAPER_TABLE_1.tolist(),
                    "paper_table_2": PAPER_TABLE_2.tolist(),
                    "stage": f"table_2_{i + 1}_{j + 1}",
                },
            )

    return {
        "seed": seed,
        "table_1": table1.tolist(),
        "table_2": table2.tolist(),
        "paper_table_1": PAPER_TABLE_1.tolist(),
        "paper_table_2": PAPER_TABLE_2.tolist(),
        "table_1_abs_diff_mean": float(np.mean(np.abs(table1 - PAPER_TABLE_1))),
        "table_2_abs_diff_mean": float(np.mean(np.abs(table2 - PAPER_TABLE_2))),
    }


def run_dt_burgers_tables(
    seed: int,
    device: str | None = None,
    train_config: TrainConfig | None = None,
    checkpoint_path: Path | None = None,
) -> dict[str, Any]:
    q_values = [1, 2, 4, 8, 16, 32, 64, 100, 500]
    skip_values = [20, 40, 60, 80]
    hidden_values = [1, 2, 3]
    width_values = [10, 25, 50]

    table4 = np.zeros((len(q_values), len(skip_values)), dtype=np.float64)
    for i, q in enumerate(q_values):
        for j, skip in enumerate(skip_values):
            print(f"[dt_burgers_table4] seed={seed} cell=({i + 1},{j + 1}) q={q} skip={skip}")
            result = run_dt_burgers(
                seed=seed,
                device=device,
                q=q,
                skip=skip,
                hidden_layers=3,
                width=50,
                train_config=train_config,
            )
            table4[i, j] = result["error"]
            _checkpoint_table_payload(
                checkpoint_path,
                {
                    "seed": seed,
                    "table_3": None,
                    "table_4": table4.tolist(),
                    "paper_table_3": PAPER_TABLE_3.tolist(),
                    "paper_table_4": PAPER_TABLE_4.tolist(),
                    "stage": f"table_4_{i + 1}_{j + 1}",
                },
            )

    table3 = np.zeros((len(hidden_values), len(width_values)), dtype=np.float64)
    for i, hidden in enumerate(hidden_values):
        for j, width in enumerate(width_values):
            print(f"[dt_burgers_table3] seed={seed} cell=({i + 1},{j + 1}) hidden_layers={hidden} width={width}")
            result = run_dt_burgers(
                seed=seed,
                device=device,
                q=500,
                skip=80,
                hidden_layers=hidden,
                width=width,
                train_config=train_config,
            )
            table3[i, j] = result["error"]
            _checkpoint_table_payload(
                checkpoint_path,
                {
                    "seed": seed,
                    "table_3": table3.tolist(),
                    "table_4": table4.tolist(),
                    "paper_table_3": PAPER_TABLE_3.tolist(),
                    "paper_table_4": PAPER_TABLE_4.tolist(),
                    "stage": f"table_3_{i + 1}_{j + 1}",
                },
            )

    return {
        "seed": seed,
        "table_3": table3.tolist(),
        "table_4": table4.tolist(),
        "paper_table_3": PAPER_TABLE_3.tolist(),
        "paper_table_4": PAPER_TABLE_4.tolist(),
        "table_3_abs_diff_mean": float(np.mean(np.abs(table3 - PAPER_TABLE_3))),
        "table_4_abs_diff_mean": float(np.mean(np.abs(table4 - PAPER_TABLE_4))),
    }


def write_result_file(name: str, payload: dict[str, Any]) -> Path:
    ensure_results_dir()
    path = RESULTS_ROOT / f"{name}.json"
    save_json(path, payload)
    return path
