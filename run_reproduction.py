from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from pinn_reproduction import (
    TrainConfig,
    run_ct_burgers,
    run_ct_burgers_tables,
    run_ct_schrodinger,
    run_dt_ac,
    run_dt_burgers,
    run_dt_burgers_tables,
    run_multi_seed,
    write_result_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reproduce PINN experiments from Raissi et al. (2017).")
    parser.add_argument(
        "--task",
        required=True,
        choices=[
            "ct_burgers",
            "ct_burgers_tables",
            "ct_schrodinger",
            "dt_burgers",
            "dt_burgers_tables",
            "dt_ac",
            "headline_suite",
        ],
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1234])
    parser.add_argument("--adam-steps", type=int, default=None)
    parser.add_argument("--lbfgs-maxiter", type=int, default=None)
    parser.add_argument("--lbfgs-maxfun", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def make_train_config(args: argparse.Namespace) -> TrainConfig | None:
    if args.adam_steps is None and args.lbfgs_maxiter is None and args.lbfgs_maxfun is None:
        return None
    config = TrainConfig()
    if args.adam_steps is not None:
        config.adam_steps = args.adam_steps
    if args.lbfgs_maxiter is not None:
        config.lbfgs_maxiter = args.lbfgs_maxiter
    if args.lbfgs_maxfun is not None:
        config.lbfgs_maxfun = args.lbfgs_maxfun
    return config


def write_payload(task: str, payload: dict[str, Any], output: Path | None) -> None:
    path = output if output is not None else write_result_file(task, payload)
    if output is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(__import__("json").dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved results to {path}")


def main() -> None:
    args = parse_args()
    config = make_train_config(args)

    if args.task == "ct_burgers":
        payload = run_multi_seed(run_ct_burgers, seeds=args.seeds, device=args.device, train_config=config)
    elif args.task == "ct_burgers_tables":
        payload = run_ct_burgers_tables(
            seed=args.seeds[0],
            device=args.device,
            train_config=config,
            checkpoint_path=args.output,
        )
    elif args.task == "ct_schrodinger":
        payload = run_multi_seed(run_ct_schrodinger, seeds=args.seeds, device=args.device, train_config=config)
    elif args.task == "dt_burgers":
        payload = run_multi_seed(run_dt_burgers, seeds=args.seeds, device=args.device, train_config=config)
    elif args.task == "dt_burgers_tables":
        payload = run_dt_burgers_tables(
            seed=args.seeds[0],
            device=args.device,
            train_config=config,
            checkpoint_path=args.output,
        )
    elif args.task == "dt_ac":
        payload = run_multi_seed(run_dt_ac, seeds=args.seeds, device=args.device, train_config=config)
    else:
        payload = {
            "ct_burgers": run_multi_seed(run_ct_burgers, seeds=args.seeds, device=args.device, train_config=config),
            "ct_schrodinger": run_multi_seed(
                run_ct_schrodinger, seeds=args.seeds, device=args.device, train_config=config
            ),
            "dt_burgers": run_multi_seed(run_dt_burgers, seeds=args.seeds, device=args.device, train_config=config),
            "dt_ac": run_multi_seed(run_dt_ac, seeds=args.seeds, device=args.device, train_config=config),
        }
    write_payload(args.task, payload, args.output)


if __name__ == "__main__":
    main()
