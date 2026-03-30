from __future__ import annotations

from pathlib import Path
from typing import Dict

import torch
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


def run_experiment(cfg: Dict, train_cfg: Dict, model_cfg: Dict, seed: int, output_dir: str | Path | None = None, run_name: str = "run"):
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
    )

    test_loss, test_acc, y_pred, y_true = evaluate_model(model, loaders["test"], device)

    save_dir = ensure_dir(Path(output_dir or "outputs") / run_name / task)
    cm = build_confusion_matrix(y_true, y_pred, num_classes)
    save_confusion_matrix_figure(cm, class_names, save_dir / "confusion_matrix.png", normalize=False)
    save_confusion_matrix_figure(cm, class_names, save_dir / "confusion_matrix_normalized.png", normalize=True)
    save_results_figure(train_result.history, save_dir / "results.png")

    return train_result, test_loss, test_acc, save_dir
