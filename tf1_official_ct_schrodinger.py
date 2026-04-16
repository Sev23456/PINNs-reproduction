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
DATA_PATH = ROOT / "reference_official" / "data" / "NLS.mat"
RESULTS_DIR = ROOT / "results"
PAPER_ERROR = 1.97e-3


@dataclass
class Config:
    seed: int = 1234
    n0: int = 50
    n_b: int = 50
    n_f: int = 20000
    layers: tuple[int, ...] = (2, 100, 100, 100, 100, 2)
    adam_iters: int = 50000
    adam_log_every: int = 1000
    maxiter: int = 50000
    maxfun: int = 50000
    maxcor: int = 50
    maxls: int = 50
    ftol: float = float(np.finfo(float).eps)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)


class PhysicsInformedNN:
    def __init__(
        self,
        x0: np.ndarray,
        u0: np.ndarray,
        v0: np.ndarray,
        tb: np.ndarray,
        x_f: np.ndarray,
        layers: tuple[int, ...],
        lb: np.ndarray,
        ub: np.ndarray,
        config: Config,
    ) -> None:
        x0_t = np.concatenate((x0, 0 * x0), 1)
        x_lb = np.concatenate((0 * tb + lb[0], tb), 1)
        x_ub = np.concatenate((0 * tb + ub[0], tb), 1)

        self.lb = lb
        self.ub = ub

        self.x0 = x0_t[:, 0:1]
        self.t0 = x0_t[:, 1:2]

        self.x_lb = x_lb[:, 0:1]
        self.t_lb = x_lb[:, 1:2]

        self.x_ub = x_ub[:, 0:1]
        self.t_ub = x_ub[:, 1:2]

        self.x_f = x_f[:, 0:1]
        self.t_f = x_f[:, 1:2]

        self.u0 = u0
        self.v0 = v0
        self.layers = layers

        self.weights, self.biases = self.initialize_nn(layers)
        self.adam_history: list[dict[str, float]] = []
        self.lbfgs_losses: list[float] = []

        self.x0_tf = tf.placeholder(tf.float32, shape=[None, self.x0.shape[1]])
        self.t0_tf = tf.placeholder(tf.float32, shape=[None, self.t0.shape[1]])
        self.u0_tf = tf.placeholder(tf.float32, shape=[None, self.u0.shape[1]])
        self.v0_tf = tf.placeholder(tf.float32, shape=[None, self.v0.shape[1]])

        self.x_lb_tf = tf.placeholder(tf.float32, shape=[None, self.x_lb.shape[1]])
        self.t_lb_tf = tf.placeholder(tf.float32, shape=[None, self.t_lb.shape[1]])
        self.x_ub_tf = tf.placeholder(tf.float32, shape=[None, self.x_ub.shape[1]])
        self.t_ub_tf = tf.placeholder(tf.float32, shape=[None, self.t_ub.shape[1]])

        self.x_f_tf = tf.placeholder(tf.float32, shape=[None, self.x_f.shape[1]])
        self.t_f_tf = tf.placeholder(tf.float32, shape=[None, self.t_f.shape[1]])

        self.u0_pred, self.v0_pred, _, _ = self.net_uv(self.x0_tf, self.t0_tf)
        self.u_lb_pred, self.v_lb_pred, self.u_x_lb_pred, self.v_x_lb_pred = self.net_uv(self.x_lb_tf, self.t_lb_tf)
        self.u_ub_pred, self.v_ub_pred, self.u_x_ub_pred, self.v_x_ub_pred = self.net_uv(self.x_ub_tf, self.t_ub_tf)
        self.f_u_pred, self.f_v_pred = self.net_f_uv(self.x_f_tf, self.t_f_tf)

        self.loss = (
            tf.reduce_mean(tf.square(self.u0_tf - self.u0_pred))
            + tf.reduce_mean(tf.square(self.v0_tf - self.v0_pred))
            + tf.reduce_mean(tf.square(self.u_lb_pred - self.u_ub_pred))
            + tf.reduce_mean(tf.square(self.v_lb_pred - self.v_ub_pred))
            + tf.reduce_mean(tf.square(self.u_x_lb_pred - self.u_x_ub_pred))
            + tf.reduce_mean(tf.square(self.v_x_lb_pred - self.v_x_ub_pred))
            + tf.reduce_mean(tf.square(self.f_u_pred))
            + tf.reduce_mean(tf.square(self.f_v_pred))
        )

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
        self.optimizer_adam = tf.train.AdamOptimizer()
        self.train_op_adam = self.optimizer_adam.minimize(self.loss)

        self.sess = tf.Session(
            config=tf.ConfigProto(
                allow_soft_placement=True,
                log_device_placement=False,
            )
        )
        self.sess.run(tf.global_variables_initializer())

    def initialize_nn(self, layers: tuple[int, ...]) -> tuple[list[tf.Variable], list[tf.Variable]]:
        weights = []
        biases = []
        for idx in range(len(layers) - 1):
            weights.append(self.xavier_init([layers[idx], layers[idx + 1]]))
            biases.append(tf.Variable(tf.zeros([1, layers[idx + 1]], dtype=tf.float32), dtype=tf.float32))
        return weights, biases

    @staticmethod
    def xavier_init(size: list[int]) -> tf.Variable:
        in_dim = size[0]
        out_dim = size[1]
        xavier_stddev = np.sqrt(2.0 / (in_dim + out_dim))
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)

    def neural_net(self, x: tf.Tensor, weights: list[tf.Variable], biases: list[tf.Variable]) -> tf.Tensor:
        h = 2.0 * (x - self.lb) / (self.ub - self.lb) - 1.0
        for idx in range(len(weights) - 1):
            h = tf.tanh(tf.add(tf.matmul(h, weights[idx]), biases[idx]))
        return tf.add(tf.matmul(h, weights[-1]), biases[-1])

    def net_uv(self, x: tf.Tensor, t: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        uv = self.neural_net(tf.concat([x, t], 1), self.weights, self.biases)
        u = uv[:, 0:1]
        v = uv[:, 1:2]
        u_x = tf.gradients(u, x)[0]
        v_x = tf.gradients(v, x)[0]
        return u, v, u_x, v_x

    def net_f_uv(self, x: tf.Tensor, t: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        u, v, u_x, v_x = self.net_uv(x, t)
        u_t = tf.gradients(u, t)[0]
        u_xx = tf.gradients(u_x, x)[0]
        v_t = tf.gradients(v, t)[0]
        v_xx = tf.gradients(v_x, x)[0]
        f_u = u_t + 0.5 * v_xx + (u**2 + v**2) * v
        f_v = v_t - 0.5 * u_xx - (u**2 + v**2) * u
        return f_u, f_v

    def callback(self, loss: float) -> None:
        self.lbfgs_losses.append(float(loss))

    def train(self, config: Config) -> dict[str, object]:
        tf_dict = {
            self.x0_tf: self.x0,
            self.t0_tf: self.t0,
            self.u0_tf: self.u0,
            self.v0_tf: self.v0,
            self.x_lb_tf: self.x_lb,
            self.t_lb_tf: self.t_lb,
            self.x_ub_tf: self.x_ub,
            self.t_ub_tf: self.t_ub,
            self.x_f_tf: self.x_f,
            self.t_f_tf: self.t_f,
        }

        adam_start = time.time()
        tick = time.time()
        for it in range(config.adam_iters):
            self.sess.run(self.train_op_adam, tf_dict)
            if it % config.adam_log_every == 0:
                loss_value = float(self.sess.run(self.loss, tf_dict))
                now = time.time()
                self.adam_history.append(
                    {
                        "iter": float(it),
                        "loss": loss_value,
                        "seconds_since_last_log": now - tick,
                    }
                )
                print(f"[ct_schrodinger_tf1_adam] iter={it} loss={loss_value:.6e} dt={now - tick:.2f}s")
                tick = now

        adam_seconds = time.time() - adam_start
        lbfgs_start = time.time()
        self.optimizer.minimize(
            self.sess,
            feed_dict=tf_dict,
            fetches=[self.loss],
            loss_callback=self.callback,
        )
        lbfgs_seconds = time.time() - lbfgs_start
        final_loss = float(self.sess.run(self.loss, tf_dict))

        return {
            "adam_seconds": adam_seconds,
            "lbfgs_seconds": lbfgs_seconds,
            "adam_logs": self.adam_history,
            "lbfgs_callback_evals": len(self.lbfgs_losses),
            "lbfgs_loss_tail": self.lbfgs_losses[-5:],
            "final_loss": final_loss,
        }

    def predict(self, x_star: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        tf_dict_uv = {self.x0_tf: x_star[:, 0:1], self.t0_tf: x_star[:, 1:2]}
        u_star = self.sess.run(self.u0_pred, tf_dict_uv)
        v_star = self.sess.run(self.v0_pred, tf_dict_uv)

        tf_dict_f = {self.x_f_tf: x_star[:, 0:1], self.t_f_tf: x_star[:, 1:2]}
        f_u_star = self.sess.run(self.f_u_pred, tf_dict_f)
        f_v_star = self.sess.run(self.f_v_pred, tf_dict_f)
        return u_star, v_star, f_u_star, f_v_star


def build_dataset(config: Config) -> tuple[np.ndarray, ...]:
    lb = np.array([-5.0, 0.0], dtype=np.float64)
    ub = np.array([5.0, np.pi / 2.0], dtype=np.float64)

    data = scipy.io.loadmat(DATA_PATH)
    t = data["tt"].flatten()[:, None]
    x = data["x"].flatten()[:, None]
    exact = data["uu"]
    exact_u = np.real(exact)
    exact_v = np.imag(exact)
    exact_h = np.sqrt(exact_u**2 + exact_v**2)

    x_grid, t_grid = np.meshgrid(x, t)
    x_star = np.hstack((x_grid.flatten()[:, None], t_grid.flatten()[:, None]))
    u_star = exact_u.T.flatten()[:, None]
    v_star = exact_v.T.flatten()[:, None]
    h_star = exact_h.T.flatten()[:, None]

    idx_x = np.random.choice(x.shape[0], config.n0, replace=False)
    x0 = x[idx_x, :]
    u0 = exact_u[idx_x, 0:1]
    v0 = exact_v[idx_x, 0:1]

    idx_t = np.random.choice(t.shape[0], config.n_b, replace=False)
    tb = t[idx_t, :]

    x_f = lb + (ub - lb) * lhs(2, config.n_f)
    return x0, u0, v0, tb, x_f, x_star, u_star, v_star, h_star, lb, ub


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
    x0, u0, v0, tb, x_f, x_star, u_star, v_star, h_star, lb, ub = build_dataset(config)
    model = PhysicsInformedNN(x0, u0, v0, tb, x_f, config.layers, lb, ub, config)
    train_result = model.train(config)
    u_pred, v_pred, _, _ = model.predict(x_star)
    h_pred = np.sqrt(u_pred**2 + v_pred**2)

    error_u = float(np.linalg.norm(u_star - u_pred, 2) / np.linalg.norm(u_star, 2))
    error_v = float(np.linalg.norm(v_star - v_pred, 2) / np.linalg.norm(v_star, 2))
    error_h = float(np.linalg.norm(h_star - h_pred, 2) / np.linalg.norm(h_star, 2))

    return {
        "experiment": "ct_schrodinger_tf1_official",
        "seed": config.seed,
        "error_h": error_h,
        "error_u": error_u,
        "error_v": error_v,
        "paper_error_h": PAPER_ERROR,
        "absolute_delta_h": abs(error_h - PAPER_ERROR),
        "relative_ratio_h": error_h / PAPER_ERROR,
        "train_result": train_result,
        "config": {
            "n0": config.n0,
            "n_b": config.n_b,
            "n_f": config.n_f,
            "layers": list(config.layers),
            "adam_iters": config.adam_iters,
            "tensorflow_version": tf.__version__,
            "numpy_version": np.__version__,
            "protobuf_impl": os.environ.get("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", ""),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        },
        "devices": local_devices(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Official TF1-style continuous Schrodinger reproduction")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--n0", type=int, default=50)
    parser.add_argument("--n-b", type=int, default=50)
    parser.add_argument("--n-f", type=int, default=20000)
    parser.add_argument("--adam-iters", type=int, default=50000)
    parser.add_argument("--adam-log-every", type=int, default=1000)
    parser.add_argument("--maxiter", type=int, default=50000)
    parser.add_argument("--maxfun", type=int, default=50000)
    parser.add_argument("--output-name", type=str, default="")
    args = parser.parse_args()

    config = Config(
        seed=args.seed,
        n0=args.n0,
        n_b=args.n_b,
        n_f=args.n_f,
        adam_iters=args.adam_iters,
        adam_log_every=args.adam_log_every,
        maxiter=args.maxiter,
        maxfun=args.maxfun,
    )
    result = run(config)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_name = args.output_name or f"ct_schrodinger_tf1_official_seed{config.seed}.json"
    output_path = RESULTS_DIR / output_name
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"Saved results to {output_path}")


if __name__ == "__main__":
    main()
