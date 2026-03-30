from __future__ import annotations

import torch
import torch.nn as nn


def _get_feature_dim(model: nn.Module, branch_name: str) -> int:
    if not hasattr(model, "forward_features"):
        raise ValueError(f"{branch_name} model must implement forward_features for fusion")

    feature_dim = getattr(model, "feature_dim", None)
    if not isinstance(feature_dim, int) or feature_dim <= 0:
        raise ValueError(
            f"{branch_name} model must define a positive integer feature_dim for fusion extensibility"
        )

    return feature_dim


class ConcatFusion(nn.Module):
    def __init__(self, image_model: nn.Module, spectrum_model: nn.Module, num_classes: int):
        super().__init__()
        self.image_model = image_model
        self.spectrum_model = spectrum_model

        self.image_feat_dim = _get_feature_dim(image_model, "image")
        self.spec_feat_dim = _get_feature_dim(spectrum_model, "spectrum")

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
