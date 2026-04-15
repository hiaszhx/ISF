from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


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


class MultiScaleCrossAttentionFusion(nn.Module):
    """多尺度交叉注意力融合。

    从图像和光谱分支各提取多尺度（低层/中层/高层 CNN）特征，
    在每个尺度上进行双向跨模态交叉注意力融合，
    再通过可学习的尺度权重聚合各尺度的融合表征进行分类。

    若子模型未实现 forward_multiscale_features，则自动退化为
    单尺度（等同于 CrossAttentionFusion）。
    """

    def __init__(
        self,
        image_model: nn.Module,
        spectrum_model: nn.Module,
        num_classes: int,
        hidden_dim: int = 128,
        num_heads: int = 4,
        num_tokens: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.image_model = image_model
        self.spectrum_model = spectrum_model
        self.hidden_dim = hidden_dim
        self.num_tokens = num_tokens

        # 获取多尺度维度；没有则退化为单尺度
        img_has_ms = hasattr(image_model, "multiscale_dims")
        spec_has_ms = hasattr(spectrum_model, "multiscale_dims")
        img_ms_dims: List[int] = getattr(
            image_model, "multiscale_dims",
            [_get_feature_dim(image_model, "image")],
        )
        spec_ms_dims: List[int] = getattr(
            spectrum_model, "multiscale_dims",
            [_get_feature_dim(spectrum_model, "spectrum")],
        )
        self.num_scales = min(len(img_ms_dims), len(spec_ms_dims))

        # 退化信息
        degraded_branches: List[str] = []
        if not img_has_ms:
            degraded_branches.append("image")
        if not spec_has_ms:
            degraded_branches.append("spectrum")
        if degraded_branches:
            self.multiscale_info = (
                f"多尺度融合退化: {', '.join(degraded_branches)} 分支不支持多尺度，"
                f"已退化为单尺度 (num_scales=1)"
            )
        else:
            self.multiscale_info = (
                f"多尺度融合: num_scales={self.num_scales}, "
                f"image_dims={img_ms_dims[:self.num_scales]}, "
                f"spectrum_dims={spec_ms_dims[:self.num_scales]}"
            )

        # 每个尺度：投影 → 交叉注意力 → LayerNorm → FFN
        self.img_projectors = nn.ModuleList()
        self.spec_projectors = nn.ModuleList()
        self.img_to_spec_attns = nn.ModuleList()
        self.spec_to_img_attns = nn.ModuleList()
        self.img_norms1 = nn.ModuleList()
        self.spec_norms1 = nn.ModuleList()
        self.img_norms2 = nn.ModuleList()
        self.spec_norms2 = nn.ModuleList()
        self.img_ffns = nn.ModuleList()
        self.spec_ffns = nn.ModuleList()

        for i in range(self.num_scales):
            self.img_projectors.append(nn.Linear(img_ms_dims[i], hidden_dim * num_tokens))
            self.spec_projectors.append(nn.Linear(spec_ms_dims[i], hidden_dim * num_tokens))

            self.img_to_spec_attns.append(
                nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
            )
            self.spec_to_img_attns.append(
                nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
            )

            self.img_norms1.append(nn.LayerNorm(hidden_dim))
            self.spec_norms1.append(nn.LayerNorm(hidden_dim))
            self.img_norms2.append(nn.LayerNorm(hidden_dim))
            self.spec_norms2.append(nn.LayerNorm(hidden_dim))

            self.img_ffns.append(nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 2, hidden_dim),
            ))
            self.spec_ffns.append(nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 2, hidden_dim),
            ))

        # 可学习的尺度重要性权重
        self.scale_weights = nn.Parameter(torch.ones(self.num_scales))

        # 分类头
        self.fusion_head = nn.Sequential(
            nn.Linear(hidden_dim * 2 * self.num_scales, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def _to_tokens(self, feat: torch.Tensor, projector: nn.Module) -> torch.Tensor:
        b = feat.size(0)
        return projector(feat).view(b, self.num_tokens, self.hidden_dim)

    def forward(self, image: torch.Tensor, spectrum: torch.Tensor) -> torch.Tensor:
        # 提取多尺度特征
        if hasattr(self.image_model, "forward_multiscale_features"):
            img_feats = self.image_model.forward_multiscale_features(image)
        else:
            img_feats = [self.image_model.forward_features(image)]

        if hasattr(self.spectrum_model, "forward_multiscale_features"):
            spec_feats = self.spectrum_model.forward_multiscale_features(spectrum)
        else:
            spec_feats = [self.spectrum_model.forward_features(spectrum)]

        # 尺度权重归一化
        scale_w = F.softmax(self.scale_weights, dim=0)

        fused_scales = []
        for i in range(self.num_scales):
            img_tokens = self._to_tokens(img_feats[i], self.img_projectors[i])
            spec_tokens = self._to_tokens(spec_feats[i], self.spec_projectors[i])

            # 双向交叉注意力
            img_cross, _ = self.img_to_spec_attns[i](
                query=img_tokens, key=spec_tokens, value=spec_tokens,
            )
            spec_cross, _ = self.spec_to_img_attns[i](
                query=spec_tokens, key=img_tokens, value=img_tokens,
            )

            img_tokens = self.img_norms1[i](img_tokens + img_cross)
            spec_tokens = self.spec_norms1[i](spec_tokens + spec_cross)

            img_tokens = self.img_norms2[i](img_tokens + self.img_ffns[i](img_tokens))
            spec_tokens = self.spec_norms2[i](spec_tokens + self.spec_ffns[i](spec_tokens))

            img_repr = img_tokens.mean(dim=1)   # (B, hidden_dim)
            spec_repr = spec_tokens.mean(dim=1)

            fused_scales.append(
                torch.cat([img_repr, spec_repr], dim=1) * scale_w[i]
            )

        fused = torch.cat(fused_scales, dim=1)  # (B, hidden_dim * 2 * num_scales)
        return self.fusion_head(fused)


def build_fusion_model(name: str, image_model: nn.Module, spectrum_model: nn.Module, num_classes: int) -> nn.Module:
    name = name.lower()
    if name == "concat_fusion":
        return ConcatFusion(image_model, spectrum_model, num_classes)
    if name in {"cross_attention_fusion", "cross_attn_fusion"}:
        return CrossAttentionFusion(image_model, spectrum_model, num_classes)
    if name in {"multiscale_fusion", "multiscale_cross_attention_fusion"}:
        return MultiScaleCrossAttentionFusion(image_model, spectrum_model, num_classes)
    raise ValueError(f"Unsupported fusion model: {name}")
