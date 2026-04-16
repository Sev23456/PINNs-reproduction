from __future__ import annotations

import json

import tensorflow as tf


def main() -> None:
    payload: dict[str, object] = {
        "tensorflow_version": tf.__version__,
        "built_with_cuda": tf.test.is_built_with_cuda(),
        "physical_devices": [device.name for device in tf.config.list_physical_devices()],
        "gpu_devices": [device.name for device in tf.config.list_physical_devices("GPU")],
        "logical_gpus": [device.name for device in tf.config.list_logical_devices("GPU")],
    }

    if payload["gpu_devices"]:
        with tf.device("/GPU:0"):
            a = tf.random.uniform((1024, 1024), dtype=tf.float32)
            b = tf.random.uniform((1024, 1024), dtype=tf.float32)
            c = tf.matmul(a, b)
            payload["gpu_test_device"] = c.device
            payload["gpu_test_mean"] = float(tf.reduce_mean(c).numpy())
    else:
        payload["gpu_test_device"] = None
        payload["gpu_test_mean"] = None

    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
