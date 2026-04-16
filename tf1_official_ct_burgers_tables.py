from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from tf1_official_ct_burgers import Config, RESULTS_DIR, make_layers, run


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


def checkpoint_path_for(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.stem}.partial.json")


def load_checkpoint(path: Path) -> tuple[np.ndarray, np.ndarray] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    table_1 = np.array(payload.get("table_1", []), dtype=np.float64)
    table_2 = np.array(payload.get("table_2", []), dtype=np.float64)
    return table_1, table_2


def save_checkpoint(path: Path, table_1: np.ndarray, table_2: np.ndarray, seed: int, stage: str) -> None:
    payload = {
        "seed": seed,
        "stage": stage,
        "table_1": table_1.tolist(),
        "table_2": table_2.tolist(),
        "paper_table_1": PAPER_TABLE_1.tolist(),
        "paper_table_2": PAPER_TABLE_2.tolist(),
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Official TF1 continuous Burgers sensitivity tables")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output-name", type=str, default="ct_burgers_tf1_official_tables_seed1234.json")
    parser.add_argument("--maxiter", type=int, default=50000)
    parser.add_argument("--maxfun", type=int, default=50000)
    args = parser.parse_args()

    n_u_values = [20, 40, 60, 80, 100, 200]
    n_f_values = [2000, 4000, 6000, 7000, 8000, 10000]
    hidden_values = [2, 4, 6, 8]
    width_values = [10, 20, 40]

    output_path = RESULTS_DIR / args.output_name
    partial_path = checkpoint_path_for(output_path)

    table_1 = np.full((len(n_u_values), len(n_f_values)), np.nan, dtype=np.float64)
    table_2 = np.full((len(hidden_values), len(width_values)), np.nan, dtype=np.float64)
    checkpoint = load_checkpoint(partial_path)
    if checkpoint is not None:
        loaded_table_1, loaded_table_2 = checkpoint
        if loaded_table_1.shape == table_1.shape:
            table_1 = loaded_table_1
        if loaded_table_2.shape == table_2.shape:
            table_2 = loaded_table_2

    for i, n_u in enumerate(n_u_values):
        for j, n_f in enumerate(n_f_values):
            if np.isfinite(table_1[i, j]):
                print(f"[ct_burgers_tf1_table1] seed={args.seed} cell=({i + 1},{j + 1}) already checkpointed, skipping")
                continue
            print(f"[ct_burgers_tf1_table1] seed={args.seed} cell=({i + 1},{j + 1}) n_u={n_u} n_f={n_f}")
            result = run(
                Config(
                    seed=args.seed,
                    n_u=n_u,
                    n_f=n_f,
                    layers=make_layers(8, 20),
                    maxiter=args.maxiter,
                    maxfun=args.maxfun,
                )
            )
            table_1[i, j] = result["error"]
            save_checkpoint(partial_path, table_1, table_2, args.seed, f"table_1_{i + 1}_{j + 1}")

    for i, hidden_layers in enumerate(hidden_values):
        for j, width in enumerate(width_values):
            if np.isfinite(table_2[i, j]):
                print(
                    f"[ct_burgers_tf1_table2] seed={args.seed} cell=({i + 1},{j + 1}) "
                    "already checkpointed, skipping"
                )
                continue
            print(
                f"[ct_burgers_tf1_table2] seed={args.seed} cell=({i + 1},{j + 1}) "
                f"hidden_layers={hidden_layers} width={width}"
            )
            result = run(
                Config(
                    seed=args.seed,
                    n_u=100,
                    n_f=10000,
                    layers=make_layers(hidden_layers, width),
                    maxiter=args.maxiter,
                    maxfun=args.maxfun,
                )
            )
            table_2[i, j] = result["error"]
            save_checkpoint(partial_path, table_1, table_2, args.seed, f"table_2_{i + 1}_{j + 1}")

    payload = {
        "seed": args.seed,
        "table_1": table_1.tolist(),
        "table_2": table_2.tolist(),
        "paper_table_1": PAPER_TABLE_1.tolist(),
        "paper_table_2": PAPER_TABLE_2.tolist(),
        "table_1_abs_diff_mean": float(np.mean(np.abs(table_1 - PAPER_TABLE_1))),
        "table_2_abs_diff_mean": float(np.mean(np.abs(table_2 - PAPER_TABLE_2))),
        "maxiter": args.maxiter,
        "maxfun": args.maxfun,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    if partial_path.exists():
        partial_path.unlink()
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"Saved results to {output_path}")


if __name__ == "__main__":
    main()
