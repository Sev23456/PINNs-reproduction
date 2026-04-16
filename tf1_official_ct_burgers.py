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
import tensorflow as tf
from pyDOE import lhs
from tensorflow.python.client import device_lib


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


def make_layers(hidden_layers: int, width: int) -> tuple[int, ...]:
    return tuple([2] + [width] * hidden_layers + [1])


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)


class PhysicsInformedNN:
    def __init__(
        self,
        x_u: np.ndarray,
        u: np.ndarray,
        x_f: np.ndarray,
        layers: tuple[int, ...],
        lb: np.ndarray,
        ub: np.ndarray,
        nu: float,
        config: Config,
    ) -> None:
        self.lb = lb
        self.ub = ub

        self.x_u = x_u[:, 0:1]
        self.t_u = x_u[:, 1:2]
        self.x_f = x_f[:, 0:1]
        self.t_f = x_f[:, 1:2]
        self.u = u

        self.layers = layers
        self.nu = nu
        self.train_losses: list[float] = []

        self.weights, self.biases = self.initialize_nn(layers)

        self.sess = tf.Session(
            config=tf.ConfigProto(
                allow_soft_placement=True,
                log_device_placement=False,
            )
        )

        self.x_u_tf = tf.placeholder(tf.float32, shape=[None, self.x_u.shape[1]])
        self.t_u_tf = tf.placeholder(tf.float32, shape=[None, self.t_u.shape[1]])
        self.u_tf = tf.placeholder(tf.float32, shape=[None, self.u.shape[1]])

        self.x_f_tf = tf.placeholder(tf.float32, shape=[None, self.x_f.shape[1]])
        self.t_f_tf = tf.placeholder(tf.float32, shape=[None, self.t_f.shape[1]])

        self.u_pred = self.net_u(self.x_u_tf, self.t_u_tf)
        self.f_pred = self.net_f(self.x_f_tf, self.t_f_tf)

        self.loss = tf.reduce_mean(tf.square(self.u_tf - self.u_pred)) + tf.reduce_mean(tf.square(self.f_pred))

        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(
            self.loss,
            method="L-BFGS-B",
            options={
                "maxiter": config.maxiter,
                "maxfun": config.maxfun,
                "maxcor": config.maxcor,
                "maxls": config.maxls,
                "ftol": config.ftol,
            },
        )

        self.sess.run(tf.global_variables_initializer())

    def initialize_nn(self, layers: tuple[int, ...]) -> tuple[list[tf.Variable], list[tf.Variable]]:
        weights = []
        biases = []
        num_layers = len(layers)
        for idx in range(num_layers - 1):
            weights.append(self.xavier_init(size=[layers[idx], layers[idx + 1]]))
            biases.append(tf.Variable(tf.zeros([1, layers[idx + 1]], dtype=tf.float32), dtype=tf.float32))
        return weights, biases

    @staticmethod
    def xavier_init(size: list[int]) -> tf.Variable:
        in_dim = size[0]
        out_dim = size[1]
        xavier_stddev = np.sqrt(2.0 / (in_dim + out_dim))
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)

    def neural_net(self, x: tf.Tensor, weights: list[tf.Variable], biases: list[tf.Variable]) -> tf.Tensor:
        num_layers = len(weights) + 1
        h = 2.0 * (x - self.lb) / (self.ub - self.lb) - 1.0
        for idx in range(num_layers - 2):
            h = tf.tanh(tf.add(tf.matmul(h, weights[idx]), biases[idx]))
        return tf.add(tf.matmul(h, weights[-1]), biases[-1])

    def net_u(self, x: tf.Tensor, t: tf.Tensor) -> tf.Tensor:
        return self.neural_net(tf.concat([x, t], 1), self.weights, self.biases)

    def net_f(self, x: tf.Tensor, t: tf.Tensor) -> tf.Tensor:
        u = self.net_u(x, t)
        u_t = tf.gradients(u, t)[0]
        u_x = tf.gradients(u, x)[0]
        u_xx = tf.gradients(u_x, x)[0]
        return u_t + u * u_x - self.nu * u_xx

    def callback(self, loss: float) -> None:
        self.train_losses.append(float(loss))

    def train(self) -> dict[str, object]:
        tf_dict = {
            self.x_u_tf: self.x_u,
            self.t_u_tf: self.t_u,
            self.u_tf: self.u,
            self.x_f_tf: self.x_f,
            self.t_f_tf: self.t_f,
        }

        start = time.time()
        self.optimizer.minimize(
            self.sess,
            feed_dict=tf_dict,
            fetches=[self.loss],
            loss_callback=self.callback,
        )
        elapsed = time.time() - start
        final_loss = float(self.sess.run(self.loss, feed_dict=tf_dict))

        return {
            "seconds": elapsed,
            "callback_evals": len(self.train_losses),
            "final_loss": final_loss,
            "loss_tail": self.train_losses[-5:],
        }

    def predict(self, x_star: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        u_star = self.sess.run(self.u_pred, {self.x_u_tf: x_star[:, 0:1], self.t_u_tf: x_star[:, 1:2]})
        f_star = self.sess.run(self.f_pred, {self.x_f_tf: x_star[:, 0:1], self.t_f_tf: x_star[:, 1:2]})
        return u_star, f_star


def build_dataset(config: Config) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    data = scipy.io.loadmat(DATA_PATH)

    t = data["t"].flatten()[:, None]
    x = data["x"].flatten()[:, None]
    exact = np.real(data["usol"]).T

    x_grid, t_grid = np.meshgrid(x, t)
    x_star = np.hstack((x_grid.flatten()[:, None], t_grid.flatten()[:, None]))
    u_star = exact.flatten()[:, None]

    lb = x_star.min(0)
    ub = x_star.max(0)

    xx1 = np.hstack((x_grid[0:1, :].T, t_grid[0:1, :].T))
    uu1 = exact[0:1, :].T
    xx2 = np.hstack((x_grid[:, 0:1], t_grid[:, 0:1]))
    uu2 = exact[:, 0:1]
    xx3 = np.hstack((x_grid[:, -1:], t_grid[:, -1:]))
    uu3 = exact[:, -1:]

    x_u_train = np.vstack([xx1, xx2, xx3])
    x_f_train = lb + (ub - lb) * lhs(2, config.n_f)
    x_f_train = np.vstack((x_f_train, x_u_train))
    u_train = np.vstack([uu1, uu2, uu3])

    idx = np.random.choice(x_u_train.shape[0], config.n_u, replace=False)
    x_u_train = x_u_train[idx, :]
    u_train = u_train[idx, :]
    return x_u_train, u_train, x_f_train, x_star, u_star, lb, ub


def local_devices() -> list[dict[str, str]]:
    devices = []
    for device in device_lib.list_local_devices():
        devices.append(
            {
                "name": device.name,
                "device_type": device.device_type,
                "physical_device_desc": device.physical_device_desc,
            }
        )
    return devices


def run(config: Config) -> dict[str, object]:
    set_seed(config.seed)
    x_u_train, u_train, x_f_train, x_star, u_star, lb, ub = build_dataset(config)

    model = PhysicsInformedNN(x_u_train, u_train, x_f_train, config.layers, lb, ub, NU, config)
    train_result = model.train()
    u_pred, _ = model.predict(x_star)
    error = float(np.linalg.norm(u_star - u_pred, 2) / np.linalg.norm(u_star, 2))

    return {
        "experiment": "ct_burgers_tf1_official",
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
            "numpy_version": np.__version__,
            "protobuf_impl": os.environ.get("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", ""),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        },
        "devices": local_devices(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Official TF1-style continuous Burgers reproduction")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--n-u", type=int, default=100)
    parser.add_argument("--n-f", type=int, default=10000)
    parser.add_argument("--hidden-layers", type=int, default=8)
    parser.add_argument("--width", type=int, default=20)
    parser.add_argument("--maxiter", type=int, default=50000)
    parser.add_argument("--maxfun", type=int, default=50000)
    parser.add_argument("--output-name", type=str, default="")
    args = parser.parse_args()

    config = Config(
        seed=args.seed,
        n_u=args.n_u,
        n_f=args.n_f,
        layers=make_layers(args.hidden_layers, args.width),
        maxiter=args.maxiter,
        maxfun=args.maxfun,
    )
    result = run(config)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_name = args.output_name or f"ct_burgers_tf1_official_seed{config.seed}.json"
    output_path = RESULTS_DIR / output_name
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"Saved results to {output_path}")


if __name__ == "__main__":
    main()
