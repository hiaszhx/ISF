from __future__ import annotations

import torch
import torch.nn as nn


class SpectrumMLP(nn.Module):
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.feature_dim = 128
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Linear(self.feature_dim, num_classes)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.forward_features(x))


class SpectrumCNN(nn.Module):
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.feature_dim = 64
        self.net = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Linear(self.feature_dim, num_classes)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        x = self.net(x)
        x = x.squeeze(-1)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.forward_features(x))


def build_spectrum_model(name: str, input_dim: int, num_classes: int) -> nn.Module:
    name = name.lower()
    if name == "spectrum_mlp":
        return SpectrumMLP(input_dim, num_classes)
    if name == "spectrum_cnn":
        return SpectrumCNN(input_dim, num_classes)
    raise ValueError(f"Unsupported spectrum model: {name}")
