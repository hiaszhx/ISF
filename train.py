from __future__ import annotations

import argparse
from pathlib import Path

from src.train.pipeline import run_experiment
from src.utils.config import load_config
from src.utils.seed import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Image/Spectrum/Fusion classification training")
    parser.add_argument("--config", type=str, default="configs/base.yaml", help="Path to yaml config")
    parser.add_argument("--task", type=str, default=None, choices=["image", "spectrum", "fusion"], help="Override task")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg_obj = load_config(args.config)
    cfg = cfg_obj.raw

    seed = cfg.get("seed", 42)
    set_seed(seed)

    data_cfg = cfg["data"]
    train_cfg = cfg["train"]
    exp_cfg = cfg["experiment"]
    model_cfg = cfg["model"].copy()

    task = args.task if args.task is not None else exp_cfg["task"]
    model_cfg["task"] = task

    root = Path(args.config).parent.parent
    data_cfg["root_dir"] = str(root / data_cfg["root_dir"])

    train_result, test_loss, test_acc, save_dir = run_experiment(
        data_cfg,
        train_cfg,
        model_cfg,
        seed,
        output_dir=exp_cfg.get("output_dir", "outputs"),
        run_name=exp_cfg.get("run_name", "run"),
    )

    print("=" * 80)
    print(f"Task: {task}")
    print(f"Best Val Acc: {train_result.best_val_acc:.4f}")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Acc: {test_acc:.4f}")
    print(f"Results saved to: {save_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
