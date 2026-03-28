from __future__ import annotations

import torch
import torch.nn as nn


class ConcatFusion(nn.Module):
    def __init__(self, image_model: nn.Module, spectrum_model: nn.Module, num_classes: int):
        super().__init__()
        self.image_model = image_model
        self.spectrum_model = spectrum_model

        self.image_feat_dim = 512 if hasattr(image_model, "model") else 128
        self.spec_feat_dim = 128 if hasattr(spectrum_model, "net") and spectrum_model.__class__.__name__ == "SpectrumMLP" else 64

        self.fusion_head = nn.Sequential(
            nn.Linear(self.image_feat_dim + self.spec_feat_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

    def forward(self, image: torch.Tensor, spectrum: torch.Tensor) -> torch.Tensor:
        img_feat = self.image_model.forward_features(image)
        spec_feat = self.spectrum_model.forward_features(spectrum)
        fused = torch.cat([img_feat, spec_feat], dim=1)
        return self.fusion_head(fused)


def build_fusion_model(name: str, image_model: nn.Module, spectrum_model: nn.Module, num_classes: int) -> nn.Module:
    name = name.lower()
    if name == "concat_fusion":
        return ConcatFusion(image_model, spectrum_model, num_classes)
    raise ValueError(f"Unsupported fusion model: {name}")
