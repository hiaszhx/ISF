from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm


@dataclass
class TrainResult:
    best_val_acc: float
    history: Dict[str, list]


def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    correct = (preds == labels).sum().item()
    return correct / max(1, labels.size(0))


def train_model(
    model: nn.Module,
    loaders: Dict[str, DataLoader],
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
) -> TrainResult:
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        print(f"\n[Epoch {epoch}/{epochs}]")

        train_loss, train_acc, _, _ = _run_epoch(
            model,
            loaders["train"],
            criterion,
            optimizer,
            device,
            train=True,
            epoch=epoch,
            total_epochs=epochs,
        )
        val_loss, val_acc, _, _ = _run_epoch(
            model,
            loaders["val"],
            criterion,
            None,
            device,
            train=False,
            epoch=epoch,
            total_epochs=epochs,
        )

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        best_val_acc = max(best_val_acc, val_acc)

        print(
            f"Epoch {epoch}/{epochs} | "
            f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
            f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}, "
            f"best_val_acc={best_val_acc:.4f}"
        )

    return TrainResult(best_val_acc=best_val_acc, history=history)


def evaluate_model(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float, List[int], List[int]]:
    criterion = nn.CrossEntropyLoss()
    loss, acc, preds, labels = _run_epoch(model, loader, criterion, None, device, train=False, collect=True)
    return loss, acc, preds, labels


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    train: bool,
    epoch: int | None = None,
    total_epochs: int | None = None,
    collect: bool = False,
) -> Tuple[float, float, List[int], List[int]]:
    if train:
        model.train()
        phase = "train"
    else:
        model.eval()
        phase = "val" if not collect else "test"

    total_loss = 0.0
    total_acc = 0.0
    batches = 0

    pred_list: List[int] = []
    label_list: List[int] = []

    if epoch is not None and total_epochs is not None:
        desc = f"{phase} {epoch}/{total_epochs}"
    else:
        desc = phase

    for batch in tqdm(loader, leave=False, desc=desc):
        if len(batch) == 2:
            inputs, labels = batch
            inputs = inputs.to(device)
        else:
            img, spec, labels = batch
            inputs = (img.to(device), spec.to(device))

        labels = labels.to(device)

        if train:
            assert optimizer is not None
            optimizer.zero_grad()
            logits = model(*inputs) if isinstance(inputs, tuple) else model(inputs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                logits = model(*inputs) if isinstance(inputs, tuple) else model(inputs)
                loss = criterion(logits, labels)

        total_loss += loss.item()
        total_acc += accuracy(logits.detach(), labels)
        batches += 1

        if collect:
            pred_list.extend(logits.argmax(dim=1).detach().cpu().tolist())
            label_list.extend(labels.detach().cpu().tolist())

    avg_loss = total_loss / max(1, batches)
    avg_acc = total_acc / max(1, batches)

    if collect:
        return avg_loss, avg_acc, pred_list, label_list

    return avg_loss, avg_acc, [], []


def build_confusion_matrix(y_true: List[int], y_pred: List[int], num_classes: int) -> np.ndarray:
    return confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))


def save_confusion_matrix_figure(
    cm: np.ndarray,
    class_names: List[str],
    save_path: str | Path,
    normalize: bool = True,
) -> None:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    plot_cm = cm.astype(np.float32)
    if normalize:
        row_sum = plot_cm.sum(axis=1, keepdims=True)
        plot_cm = np.divide(plot_cm, np.maximum(row_sum, 1.0))

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(plot_cm, interpolation="nearest", cmap=plt.cm.Blues)
    fig.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True Label",
        xlabel="Predicted Label",
        title="Confusion Matrix" + (" (Normalized)" if normalize else ""),
    )

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = plot_cm.max() / 2.0 if plot_cm.size > 0 else 0.0
    for i in range(plot_cm.shape[0]):
        for j in range(plot_cm.shape[1]):
            text_val = f"{plot_cm[i, j]:.2f}" if normalize else str(int(plot_cm[i, j]))
            ax.text(j, i, text_val, ha="center", va="center", color="white" if plot_cm[i, j] > thresh else "black")

    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
