from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
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
    spectrum_left = cfg.get("spectrum_left")
    spectrum_right = cfg.get("spectrum_right")

    if task == "image":
        train_ds = ImageOnlyDataset(train_samples, image_size=image_size, train=True)
        val_ds = ImageOnlyDataset(val_samples, image_size=image_size, train=False)
        test_ds = ImageOnlyDataset(test_samples, image_size=image_size, train=False)
    elif task == "spectrum":
        train_ds = SpectrumOnlyDataset(train_samples, spectrum_length=spectrum_length,
                                       spectrum_left=spectrum_left, spectrum_right=spectrum_right)
        val_ds = SpectrumOnlyDataset(val_samples, spectrum_length=spectrum_length,
                                     spectrum_left=spectrum_left, spectrum_right=spectrum_right)
        test_ds = SpectrumOnlyDataset(test_samples, spectrum_length=spectrum_length,
                                      spectrum_left=spectrum_left, spectrum_right=spectrum_right)
    elif task == "fusion":
        train_ds = FusionDataset(train_samples, image_size=image_size, spectrum_length=spectrum_length, train=True,
                                 spectrum_left=spectrum_left, spectrum_right=spectrum_right)
        val_ds = FusionDataset(val_samples, image_size=image_size, spectrum_length=spectrum_length, train=False,
                               spectrum_left=spectrum_left, spectrum_right=spectrum_right)
        test_ds = FusionDataset(test_samples, image_size=image_size, spectrum_length=spectrum_length, train=False,
                                spectrum_left=spectrum_left, spectrum_right=spectrum_right)
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
    all_test_results: Dict | None = None,
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
            "spectrum_left": cfg.get("spectrum_left"),
            "spectrum_right": cfg.get("spectrum_right"),
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
            "multiscale_info": model_cfg.get("multiscale_info"),
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
            "label_smoothing": train_cfg.get("label_smoothing", 0.0),
        },
        "result": {
            "best_val_acc": float(best_val_acc),
            "test_loss": float(test_loss),
            "test_acc": float(test_acc),
        },
    }

    if all_test_results:
        snapshot["all_test_results"] = {
            k: {kk: float(vv) for kk, vv in v.items()}
            for k, v in all_test_results.items()
        }

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
        spectrum_left = cfg.get("spectrum_left")
        spectrum_right = cfg.get("spectrum_right")
        # 找到第一个包含光谱路径的样本
        for s in samples:
            if s.spectrum_path is not None:
                # 读取该样本的原始光谱长度
                df = pd.read_csv(s.spectrum_path)
                x_col = df.iloc[:, 0].to_numpy()
                orig_len = len(df.iloc[:, 1])
                target_len = cfg["spectrum_length"]
                bound_info = ""
                if spectrum_left is not None or spectrum_right is not None:
                    l_str = str(spectrum_left) if spectrum_left is not None else str(x_col.min())
                    r_str = str(spectrum_right) if spectrum_right is not None else str(x_col.max())
                    mask = np.ones(orig_len, dtype=bool)
                    if spectrum_left is not None:
                        mask &= x_col >= spectrum_left
                    if spectrum_right is not None:
                        mask &= x_col <= spectrum_right
                    cropped_len = int(mask.sum())
                    bound_info = f", 截取范围=[{l_str}, {r_str}], 截取后长度={cropped_len}"
                print(f"\n[*] 训练初始化: 原始光谱长度={orig_len}{bound_info}, 目标长度={target_len}\n")
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

    # 打印多尺度融合信息
    multiscale_info = getattr(model, "multiscale_info", None)
    if multiscale_info:
        print(f"\n[*] {multiscale_info}\n")
        model_cfg["multiscale_info"] = multiscale_info

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
        label_smoothing=train_cfg.get("label_smoothing", 0.0),
    )

    save_dir = ensure_dir(Path(output_dir or "outputs") / run_name / task)

    # ---- 保存权重文件 ----
    weights_dir = ensure_dir(save_dir / "weights")
    if train_result.best_state_dict is not None:
        torch.save(train_result.best_state_dict, weights_dir / "best_acc.pth")
    if train_result.best_loss_state_dict is not None:
        torch.save(train_result.best_loss_state_dict, weights_dir / "best_loss.pth")
    if train_result.last_state_dict is not None:
        torch.save(train_result.last_state_dict, weights_dir / "last.pth")
    print(f"\n[*] 权重已保存至 {weights_dir}")

    # ---- 保存训练曲线 ----
    save_results_figure(train_result.history, save_dir / "results.png")

    # ---- 对三组权重分别测试 ----
    weights_to_test = {
        "best_acc": (train_result.best_state_dict, f"best_val_acc={train_result.best_val_acc:.4f}"),
        "best_loss": (train_result.best_loss_state_dict, f"best_val_loss={train_result.best_val_loss:.4f}"),
        "last": (train_result.last_state_dict, "最后一轮权重"),
    }

    all_test_results = {}
    for weight_name, (state_dict, desc) in weights_to_test.items():
        if state_dict is None:
            print(f"\n权重 '{weight_name}' 不可用，跳过")
            continue
        model.load_state_dict(state_dict)
        print(f"\n测试权重: {weight_name} ({desc})")
        t_loss, t_acc, y_pred, y_true = evaluate_model(model, loaders["test"], device)
        print(f"  test_loss={t_loss:.4f}, test_acc={t_acc:.4f}")
        all_test_results[weight_name] = {"test_loss": t_loss, "test_acc": t_acc}

        # 保存混淆矩阵
        cm = build_confusion_matrix(y_true, y_pred, num_classes)
        save_confusion_matrix_figure(cm, class_names, save_dir / f"confusion_matrix_{weight_name}.png", normalize=False)
        save_confusion_matrix_figure(cm, class_names, save_dir / f"confusion_matrix_{weight_name}_normalized.png", normalize=True)

    # 找出测试准确率最高的权重
    best_weight = max(all_test_results, key=lambda k: all_test_results[k]["test_acc"])
    best_info = all_test_results[best_weight]
    print(f"\n最优测试权重: {best_weight} (test_acc={best_info['test_acc']:.4f}, test_loss={best_info['test_loss']:.4f})")

    # ---- 选择主测试结果（用于 experiment_config） ----
    primary = train_cfg.get("test_weights", "best")
    if primary == "best":
        primary = "best_acc"
    elif primary == "best_loss":
        primary = "best_loss"
    primary_result = all_test_results.get(primary, next(iter(all_test_results.values())))
    test_loss = primary_result["test_loss"]
    test_acc = primary_result["test_acc"]

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
        all_test_results=all_test_results,
    )

    return train_result, test_loss, test_acc, save_dir
