from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
from torchvision import transforms


@dataclass
class SampleItem:
    image_path: Optional[Path]
    spectrum_path: Optional[Path]
    label: int
    class_name: str
    sample_id: str


def _collect_files_by_stem(class_dir: Path, suffixes: List[str]) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    for p in class_dir.iterdir():
        if p.is_file() and p.suffix.lower() in suffixes:
            out[p.stem] = p
    return out


def build_samples(
    root_dir: str | Path,
    image_extensions: List[str],
    strict_pair: bool = False,
) -> Tuple[List[SampleItem], List[str]]:
    root_dir = Path(root_dir)
    class_dirs = sorted([p for p in root_dir.iterdir() if p.is_dir()])

    samples: List[SampleItem] = []
    class_names: List[str] = [c.name for c in class_dirs]

    image_suffixes = [e.lower() for e in image_extensions]

    for label, class_dir in enumerate(class_dirs):
        img_map = _collect_files_by_stem(class_dir, image_suffixes)
        csv_map = _collect_files_by_stem(class_dir, [".csv"])

        keys = sorted(set(img_map.keys()) | set(csv_map.keys()))
        for k in keys:
            img_p = img_map.get(k)
            csv_p = csv_map.get(k)

            if strict_pair and (img_p is None or csv_p is None):
                continue

            samples.append(
                SampleItem(
                    image_path=img_p,
                    spectrum_path=csv_p,
                    label=label,
                    class_name=class_dir.name,
                    sample_id=k,
                )
            )

    return samples, class_names


def split_samples(
    samples: List[SampleItem],
    val_ratio: float,
    test_ratio: float,
    seed: int,
    split_mode: str = "time_order",
    shuffle_before_split: bool = False,
) -> Tuple[List[SampleItem], List[SampleItem], List[SampleItem]]:
    if split_mode in {"time_order", "temporal"}:
        from collections import defaultdict

        class_samples = defaultdict(list)
        for s in samples:
            class_samples[s.label].append(s)

        rng = np.random.default_rng(seed)
        train_samples, val_samples, test_samples = [], [], []

        for _, items in class_samples.items():
            items = sorted(items, key=lambda x: x.sample_id)
            if shuffle_before_split:
                rng.shuffle(items)

            n = len(items)
            test_size = int(n * test_ratio)
            val_size = int(n * val_ratio)
            train_size = n - test_size - val_size

            train_samples.extend(items[:train_size])
            val_samples.extend(items[train_size : train_size + val_size])
            test_samples.extend(items[train_size + val_size :])

        return train_samples, val_samples, test_samples

    if split_mode in {"stratified_random", "random"}:
        labels = [s.label for s in samples]

        train_samples, test_samples = train_test_split(
            samples,
            test_size=test_ratio,
            random_state=seed,
            stratify=labels,
        )

        train_labels = [s.label for s in train_samples]
        val_size = val_ratio / (1.0 - test_ratio)

        train_samples, val_samples = train_test_split(
            train_samples,
            test_size=val_size,
            random_state=seed,
            stratify=train_labels,
        )

        return train_samples, val_samples, test_samples

    raise ValueError(f"Unsupported split_mode: {split_mode}")


class ImageOnlyDataset(Dataset):
    def __init__(self, samples: List[SampleItem], image_size: int, train: bool = True):
        self.samples = [s for s in samples if s.image_path is not None]
        if train:
            self.tf = transforms.Compose(
                [
                    transforms.CenterCrop((image_size, image_size)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                ]
            )
        else:
            self.tf = transforms.Compose(
                [
                    transforms.CenterCrop((image_size, image_size)),
                    transforms.ToTensor(),
                ]
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        img = Image.open(s.image_path).convert("RGB")
        x = self.tf(img)
        y = torch.tensor(s.label, dtype=torch.long)
        return x, y


class SpectrumOnlyDataset(Dataset):
    def __init__(self, samples: List[SampleItem], spectrum_length: int):
        self.samples = [s for s in samples if s.spectrum_path is not None]
        self.spectrum_length = spectrum_length

    def __len__(self) -> int:
        return len(self.samples)

    def _read_spectrum(self, path: Path) -> torch.Tensor:
        df = pd.read_csv(path)
        arr = df.iloc[:, 1].to_numpy(dtype=np.float32)

        if len(arr) >= self.spectrum_length:
            arr = arr[: self.spectrum_length]
        else:
            pad = np.zeros((self.spectrum_length - len(arr),), dtype=np.float32)
            arr = np.concatenate([arr, pad], axis=0)

        mean = arr.mean() if arr.size > 0 else 0.0
        std = arr.std() if arr.size > 0 else 1.0
        arr = (arr - mean) / (std + 1e-6)

        return torch.tensor(arr, dtype=torch.float32)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        x = self._read_spectrum(s.spectrum_path)
        y = torch.tensor(s.label, dtype=torch.long)
        return x, y


class FusionDataset(Dataset):
    def __init__(self, samples: List[SampleItem], image_size: int, spectrum_length: int, train: bool = True):
        self.samples = [s for s in samples if s.image_path is not None and s.spectrum_path is not None]
        self.spectrum_length = spectrum_length

        if train:
            self.img_tf = transforms.Compose(
                [
                    transforms.CenterCrop((image_size, image_size)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                ]
            )
        else:
            self.img_tf = transforms.Compose(
                [
                    transforms.CenterCrop((image_size, image_size)),
                    transforms.ToTensor(),
                ]
            )

    def __len__(self) -> int:
        return len(self.samples)

    def _read_spectrum(self, path: Path) -> torch.Tensor:
        df = pd.read_csv(path)
        arr = df.iloc[:, 1].to_numpy(dtype=np.float32)

        if len(arr) >= self.spectrum_length:
            arr = arr[: self.spectrum_length]
        else:
            pad = np.zeros((self.spectrum_length - len(arr),), dtype=np.float32)
            arr = np.concatenate([arr, pad], axis=0)

        mean = arr.mean() if arr.size > 0 else 0.0
        std = arr.std() if arr.size > 0 else 1.0
        arr = (arr - mean) / (std + 1e-6)

        return torch.tensor(arr, dtype=torch.float32)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        img = Image.open(s.image_path).convert("RGB")
        img = self.img_tf(img)
        spec = self._read_spectrum(s.spectrum_path)
        y = torch.tensor(s.label, dtype=torch.long)
        return img, spec, y
