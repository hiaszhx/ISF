from __future__ import annotations

from pathlib import Path
from typing import Dict

import torch
import pandas as pd
import yaml
from torch.utils.data import DataLoader

from src.data.dataset_builder import (
    FusionDataset,
    ImageOnlyDataset,
    SpectrumOnlyDataset,
    build_samples,
    split_samples,
)
from src.models.fusion_models import build_fusion_model
from src.models.image_models import build_image_model
from src.models.spectrum_models import build_spectrum_model
from src.train.trainer import (
    build_confusion_matrix,
    evaluate_model,
    save_confusion_matrix_figure,
    save_results_figure,
    train_model,
)
from src.utils.config import ensure_dir, get_class_names


def build_loaders(cfg: Dict, samples, batch_size: int, num_workers: int):
    image_size = cfg["image_size"]
    spectrum_length = cfg["spectrum_length"]

    def loader(ds, shuffle=False):
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    return image_size, spectrum_length, loader


def prepare_datasets(cfg: Dict, task: str, samples, train_samples, val_samples, test_samples):
    image_size = cfg["image_size"]
    spectrum_length = cfg["spectrum_length"]

    if task == "image":
        train_ds = ImageOnlyDataset(train_samples, image_size=image_size, train=True)
        val_ds = ImageOnlyDataset(val_samples, image_size=image_size, train=False)
        test_ds = ImageOnlyDataset(test_samples, image_size=image_size, train=False)
    elif task == "spectrum":
        train_ds = SpectrumOnlyDataset(train_samples, spectrum_length=spectrum_length)
        val_ds = SpectrumOnlyDataset(val_samples, spectrum_length=spectrum_length)
        test_ds = SpectrumOnlyDataset(test_samples, spectrum_length=spectrum_length)
    elif task == "fusion":
        train_ds = FusionDataset(train_samples, image_size=image_size, spectrum_length=spectrum_length, train=True)
        val_ds = FusionDataset(val_samples, image_size=image_size, spectrum_length=spectrum_length, train=False)
        test_ds = FusionDataset(test_samples, image_size=image_size, spectrum_length=spectrum_length, train=False)
    else:
        raise ValueError(f"Unsupported task: {task}")

    return train_ds, val_ds, test_ds


def save_experiment_config_snapshot(
    save_dir: Path,
    cfg: Dict,
    train_cfg: Dict,
    model_cfg: Dict,
    seed: int,
    run_name: str,
    test_loss: float,
    test_acc: float,
    best_val_acc: float,
    full_cfg: Dict | None = None,
) -> None:
    split_cfg = cfg.get("split", {})
    snapshot = {
        "run": {
            "run_name": run_name,
            "task": model_cfg.get("task"),
            "seed": seed,
        },
        "data": {
            "root_dir": cfg.get("root_dir"),
            "image_size": cfg.get("image_size"),
            "spectrum_length": cfg.get("spectrum_length"),
            "val_ratio": cfg.get("val_ratio"),
            "test_ratio": cfg.get("test_ratio"),
            "strict_pair": cfg.get("strict_pair"),
            "split": {
                "mode": split_cfg.get("mode", "time_order"),
                "shuffle_before_split": split_cfg.get("shuffle_before_split", False),
            },
        },
        "model": {
            "image_model": model_cfg.get("image_model"),
            "spectrum_model": model_cfg.get("spectrum_model"),
            "fusion_model": model_cfg.get("fusion_model"),
            "num_classes": model_cfg.get("num_classes"),
        },
        "train": {
            "epochs": train_cfg.get("epochs"),
            "batch_size": train_cfg.get("batch_size"),
            "num_workers": train_cfg.get("num_workers"),
            "lr": train_cfg.get("lr"),
            "weight_decay": train_cfg.get("weight_decay"),
            "device": train_cfg.get("device"),
            "optimizer": train_cfg.get("optimizer", "adamw"),
            "scheduler": train_cfg.get("scheduler", "cosine"),
            "scheduler_params": train_cfg.get("scheduler_params"),
        },
        "result": {
            "best_val_acc": float(best_val_acc),
            "test_loss": float(test_loss),
            "test_acc": float(test_acc),
        },
    }

    if full_cfg is not None:
        snapshot["full_config"] = full_cfg

    config_path = save_dir / "experiment_config.yaml"
    with config_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(snapshot, f, allow_unicode=True, sort_keys=False)


def run_experiment(
    cfg: Dict,
    train_cfg: Dict,
    model_cfg: Dict,
    seed: int,
    output_dir: str | Path | None = None,
    run_name: str = "run",
    full_cfg: Dict | None = None,
):
    data_root = cfg["root_dir"]
    class_names = get_class_names(data_root)
    samples, _ = build_samples(
        data_root,
        cfg["image_extensions"],
        strict_pair=cfg.get("strict_pair", False),
    )

    split_cfg = cfg.get("split", {})

    train_samples, val_samples, test_samples = split_samples(
        samples,
        val_ratio=cfg["val_ratio"],
        test_ratio=cfg["test_ratio"],
        seed=seed,
        split_mode=split_cfg.get("mode", "time_order"),
        shuffle_before_split=split_cfg.get("shuffle_before_split", False),
    )

    task = model_cfg["task"]

    if task in ["spectrum", "fusion"]:
        # 找到第一个包含光谱路径的样本
        for s in samples:
            if s.spectrum_path is not None:
                # 读取该样本的原始光谱长度
                df = pd.read_csv(s.spectrum_path)
                orig_len = len(df.iloc[:, 1])
                target_len = cfg["spectrum_length"]
                print(f"\n[*] 训练初始化: 光谱长度将由 {orig_len} 转换为目标长度 {target_len}\n")
                break

    train_ds, val_ds, test_ds = prepare_datasets(cfg, task, samples, train_samples, val_samples, test_samples)

    _, _, loader = build_loaders(cfg, samples, train_cfg["batch_size"], train_cfg["num_workers"])

    loaders = {
        "train": loader(train_ds, shuffle=True),
        "val": loader(val_ds, shuffle=False),
        "test": loader(test_ds, shuffle=False),
    }

    num_classes = len(class_names)

    if task == "image":
        model = build_image_model(model_cfg["image_model"], num_classes)
    elif task == "spectrum":
        model = build_spectrum_model(model_cfg["spectrum_model"], cfg["spectrum_length"], num_classes)
    else:
        image_model = build_image_model(model_cfg["image_model"], num_classes)
        spectrum_model = build_spectrum_model(model_cfg["spectrum_model"], cfg["spectrum_length"], num_classes)
        model = build_fusion_model(model_cfg["fusion_model"], image_model, spectrum_model, num_classes)

    device = torch.device("cuda" if torch.cuda.is_available() and train_cfg["device"] == "auto" else "cpu")
    model.to(device)

    train_result = train_model(
        model,
        loaders,
        device,
        epochs=train_cfg["epochs"],
        lr=train_cfg["lr"],
        weight_decay=train_cfg["weight_decay"],
        optimizer_name=train_cfg.get("optimizer", "adamw"),
        scheduler_name=train_cfg.get("scheduler", "cosine"),
        scheduler_params=train_cfg.get("scheduler_params"),
    )

    test_loss, test_acc, y_pred, y_true = evaluate_model(model, loaders["test"], device)

    save_dir = ensure_dir(Path(output_dir or "outputs") / run_name / task)
    cm = build_confusion_matrix(y_true, y_pred, num_classes)
    save_confusion_matrix_figure(cm, class_names, save_dir / "confusion_matrix.png", normalize=False)
    save_confusion_matrix_figure(cm, class_names, save_dir / "confusion_matrix_normalized.png", normalize=True)
    save_results_figure(train_result.history, save_dir / "results.png")
    save_experiment_config_snapshot(
        save_dir=save_dir,
        cfg=cfg,
        train_cfg=train_cfg,
        model_cfg=model_cfg,
        seed=seed,
        run_name=run_name,
        test_loss=test_loss,
        test_acc=test_acc,
        best_val_acc=train_result.best_val_acc,
        full_cfg=full_cfg,
    )

    return train_result, test_loss, test_acc, save_dir
