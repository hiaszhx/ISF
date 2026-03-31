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


class CrossAttentionFusion(nn.Module):
    def __init__(
        self,
        image_model: nn.Module,
        spectrum_model: nn.Module,
        num_classes: int,
        hidden_dim: int = 256,
        num_heads: int = 8,
        num_tokens: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.image_model = image_model
        self.spectrum_model = spectrum_model

        self.image_feat_dim = _get_feature_dim(image_model, "image")
        self.spec_feat_dim = _get_feature_dim(spectrum_model, "spectrum")

        self.hidden_dim = hidden_dim
        self.num_tokens = num_tokens

        self.image_token_proj = nn.Linear(self.image_feat_dim, hidden_dim * num_tokens)
        self.spec_token_proj = nn.Linear(self.spec_feat_dim, hidden_dim * num_tokens)

        self.img_to_spec_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.spec_to_img_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.img_norm1 = nn.LayerNorm(hidden_dim)
        self.spec_norm1 = nn.LayerNorm(hidden_dim)
        self.img_norm2 = nn.LayerNorm(hidden_dim)
        self.spec_norm2 = nn.LayerNorm(hidden_dim)

        self.img_ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.spec_ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

        self.fusion_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def _to_tokens(self, feat: torch.Tensor, projector: nn.Module) -> torch.Tensor:
        b = feat.size(0)
        tokens = projector(feat)
        return tokens.view(b, self.num_tokens, self.hidden_dim)

    def forward(self, image: torch.Tensor, spectrum: torch.Tensor) -> torch.Tensor:
        img_feat = self.image_model.forward_features(image)
        spec_feat = self.spectrum_model.forward_features(spectrum)

        img_tokens = self._to_tokens(img_feat, self.image_token_proj)
        spec_tokens = self._to_tokens(spec_feat, self.spec_token_proj)

        # Bidirectional cross-attention: each modality queries the other.
        img_cross, _ = self.img_to_spec_attn(query=img_tokens, key=spec_tokens, value=spec_tokens)
        spec_cross, _ = self.spec_to_img_attn(query=spec_tokens, key=img_tokens, value=img_tokens)

        img_tokens = self.img_norm1(img_tokens + img_cross)
        spec_tokens = self.spec_norm1(spec_tokens + spec_cross)

        img_tokens = self.img_norm2(img_tokens + self.img_ffn(img_tokens))
        spec_tokens = self.spec_norm2(spec_tokens + self.spec_ffn(spec_tokens))

        img_repr = img_tokens.mean(dim=1)
        spec_repr = spec_tokens.mean(dim=1)
        fused = torch.cat([img_repr, spec_repr], dim=1)
        return self.fusion_head(fused)


def build_fusion_model(name: str, image_model: nn.Module, spectrum_model: nn.Module, num_classes: int) -> nn.Module:
    name = name.lower()
    if name == "concat_fusion":
        return ConcatFusion(image_model, spectrum_model, num_classes)
    if name in {"cross_attention_fusion", "cross_attn_fusion"}:
        return CrossAttentionFusion(image_model, spectrum_model, num_classes)
    raise ValueError(f"Unsupported fusion model: {name}")
