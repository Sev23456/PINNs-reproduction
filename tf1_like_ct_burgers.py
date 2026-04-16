from __future__ import annotations

import argparse
import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.io
from scipy.optimize import minimize
from pyDOE import lhs

os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import tensorflow as tf


tf.compat.v1.disable_eager_execution()

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "reference_official" / "data" / "burgers_shock.mat"
RESULTS_DIR = ROOT / "results"
PAPER_ERROR = 6.7e-4
NU = 0.01 / np.pi


@dataclass
class Config:
    seed: int = 1234
    n_u: int = 100
    n_f: int = 10000
    layers: tuple[int, ...] = (2, 20, 20, 20, 20, 20, 20, 20, 20, 1)
    maxiter: int = 50000
    maxfun: int = 50000
    maxcor: int = 50
    maxls: int = 50
    ftol: float = float(np.finfo(float).eps)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.compat.v1.set_random_seed(seed)


def xavier_init(size: tuple[int, int]) -> tf.Variable:
    in_dim, out_dim = size
    std = np.sqrt(2.0 / (in_dim + out_dim))
    return tf.Variable(tf.random.truncated_normal([in_dim, out_dim], stddev=std, dtype=tf.float32), dtype=tf.float32)


class PhysicsInformedNN:
    def __init__(self, x_u: np.ndarray, u: np.ndarray, x_f: np.ndarray, lb: np.ndarray, ub: np.ndarray, layers: tuple[int, ...]):
        self.lb = lb.astype(np.float32)
        self.ub = ub.astype(np.float32)
        self.x_u = x_u[:, 0:1].astype(np.float32)
        self.t_u = x_u[:, 1:2].astype(np.float32)
        self.x_f = x_f[:, 0:1].astype(np.float32)
        self.t_f = x_f[:, 1:2].astype(np.float32)
        self.u = u.astype(np.float32)
        self.layers = layers

        self.weights, self.biases = self.initialize_nn(layers)
        self.trainables = self.weights + self.biases

        session_config = tf.compat.v1.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=False,
            inter_op_parallelism_threads=1,
            intra_op_parallelism_threads=1,
        )
        self.sess = tf.compat.v1.Session(config=session_config)

        self.x_u_tf = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])
        self.t_u_tf = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])
        self.u_tf = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])
        self.x_f_tf = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])
        self.t_f_tf = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])

        self.u_pred = self.net_u(self.x_u_tf, self.t_u_tf)
        self.f_pred = self.net_f(self.x_f_tf, self.t_f_tf)
        self.loss = tf.reduce_mean(tf.square(self.u_tf - self.u_pred)) + tf.reduce_mean(tf.square(self.f_pred))
        self.grads = tf.gradients(self.loss, self.trainables)

        self.assign_placeholders = [
            tf.compat.v1.placeholder(tf.float32, shape=var.shape, name=f"assign_{idx}")
            for idx, var in enumerate(self.trainables)
        ]
        self.assign_ops = [tf.compat.v1.assign(var, ph) for var, ph in zip(self.trainables, self.assign_placeholders)]

        self.sess.run(tf.compat.v1.global_variables_initializer())

    def initialize_nn(self, layers: tuple[int, ...]) -> tuple[list[tf.Variable], list[tf.Variable]]:
        weights = []
        biases = []
        for in_dim, out_dim in zip(layers[:-1], layers[1:]):
            weights.append(xavier_init((in_dim, out_dim)))
            biases.append(tf.Variable(tf.zeros([1, out_dim], dtype=tf.float32), dtype=tf.float32))
        return weights, biases

    def neural_net(self, x: tf.Tensor) -> tf.Tensor:
        h = 2.0 * (x - self.lb) / (self.ub - self.lb) - 1.0
        for w, b in zip(self.weights[:-1], self.biases[:-1]):
            h = tf.tanh(tf.add(tf.matmul(h, w), b))
        return tf.add(tf.matmul(h, self.weights[-1]), self.biases[-1])

    def net_u(self, x: tf.Tensor, t: tf.Tensor) -> tf.Tensor:
        return self.neural_net(tf.concat([x, t], axis=1))

    def net_f(self, x: tf.Tensor, t: tf.Tensor) -> tf.Tensor:
        u = self.net_u(x, t)
        u_t = tf.gradients(u, t)[0]
        u_x = tf.gradients(u, x)[0]
        u_xx = tf.gradients(u_x, x)[0]
        return u_t + u * u_x - NU * u_xx

    def _pack_trainables(self) -> np.ndarray:
        values = self.sess.run(self.trainables)
        return np.concatenate([value.astype(np.float64).ravel() for value in values])

    def _unpack_to_feed(self, flat: np.ndarray) -> dict[tf.Tensor, np.ndarray]:
        feed: dict[tf.Tensor, np.ndarray] = {}
        offset = 0
        for placeholder, var in zip(self.assign_placeholders, self.trainables):
            shape = tuple(int(dim) for dim in var.shape)
            size = int(np.prod(shape))
            feed[placeholder] = flat[offset : offset + size].reshape(shape).astype(np.float32)
            offset += size
        return feed

    def _assign_flat(self, flat: np.ndarray) -> None:
        self.sess.run(self.assign_ops, feed_dict=self._unpack_to_feed(flat))

    def _train_feed(self) -> dict[tf.Tensor, np.ndarray]:
        return {
            self.x_u_tf: self.x_u,
            self.t_u_tf: self.t_u,
            self.u_tf: self.u,
            self.x_f_tf: self.x_f,
            self.t_f_tf: self.t_f,
        }

    def train(self, config: Config) -> dict[str, float | int | str | bool]:
        feed = self._train_feed()
        x0 = self._pack_trainables()

        def objective(flat: np.ndarray) -> tuple[float, np.ndarray]:
            self._assign_flat(flat)
            loss_value, grad_values = self.sess.run([self.loss, self.grads], feed_dict=feed)
            grad_flat = np.concatenate([g.astype(np.float64).ravel() for g in grad_values])
            return float(loss_value), grad_flat

        start = time.time()
        result = minimize(
            objective,
            x0=x0,
            jac=True,
            method="L-BFGS-B",
            options={
                "maxiter": config.maxiter,
                "maxfun": config.maxfun,
                "maxcor": config.maxcor,
                "maxls": config.maxls,
                "ftol": config.ftol,
            },
        )
        self._assign_flat(result.x)
        return {
            "seconds": time.time() - start,
            "success": bool(result.success),
            "status": int(result.status),
            "message": str(result.message),
            "nit": int(result.nit),
            "nfev": int(result.nfev),
            "fun": float(result.fun),
        }

    def predict(self, x_star: np.ndarray) -> np.ndarray:
        return self.sess.run(
            self.u_pred,
            feed_dict={
                self.x_u_tf: x_star[:, 0:1].astype(np.float32),
                self.t_u_tf: x_star[:, 1:2].astype(np.float32),
            },
        )


def build_dataset(config: Config) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    data = scipy.io.loadmat(DATA_PATH)
    t = data["t"].flatten()[:, None]
    x = data["x"].flatten()[:, None]
    exact = np.real(data["usol"]).T

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

    x_u_train = np.vstack([xx1, xx2, xx3])
    u_train = np.vstack([uu1, uu2, uu3])
    x_f_train = lb + (ub - lb) * lhs(2, config.n_f)
    x_f_train = np.vstack((x_f_train, x_u_train))

    idx = np.random.choice(x_u_train.shape[0], config.n_u, replace=False)
    x_u_train = x_u_train[idx, :]
    u_train = u_train[idx, :]
    return x_u_train, u_train, x_f_train, x_star, u_star, lb, ub


def run(config: Config) -> dict[str, object]:
    set_seed(config.seed)
    x_u_train, u_train, x_f_train, x_star, u_star, lb, ub = build_dataset(config)
    model = PhysicsInformedNN(x_u_train, u_train, x_f_train, lb, ub, config.layers)
    train_result = model.train(config)
    u_pred = model.predict(x_star)
    error = float(np.linalg.norm(u_star - u_pred, 2) / np.linalg.norm(u_star, 2))
    return {
        "experiment": "ct_burgers_tf1_like",
        "seed": config.seed,
        "error": error,
        "paper_error": PAPER_ERROR,
        "absolute_delta": abs(error - PAPER_ERROR),
        "relative_ratio": error / PAPER_ERROR,
        "train_result": train_result,
        "config": {
            "n_u": config.n_u,
            "n_f": config.n_f,
            "layers": list(config.layers),
            "tensorflow_version": tf.__version__,
        },
        "devices": [device.name for device in tf.config.list_physical_devices()],
        "gpu_devices": [device.name for device in tf.config.list_physical_devices("GPU")],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="TF1-like continuous Burgers reproduction")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed for data sampling and initialization")
    parser.add_argument("--output-name", type=str, default="", help="Optional JSON filename inside the results directory")
    args = parser.parse_args()

    config = Config(seed=args.seed)
    result = run(config)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_name = args.output_name or f"ct_burgers_tf1_like_seed{config.seed}.json"
    output = RESULTS_DIR / output_name
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"Saved results to {output}")


if __name__ == "__main__":
    main()
