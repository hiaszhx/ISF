from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectrumAttention(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        hidden = max(1, in_channels // 8)
        self.query_conv = nn.Conv1d(in_channels, hidden, kernel_size=1)
        self.key_conv = nn.Conv1d(in_channels, hidden, kernel_size=1)
        self.value_conv = nn.Conv1d(in_channels, in_channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        query = self.query_conv(x).permute(0, 2, 1)
        key = self.key_conv(x)
        value = self.value_conv(x)

        energy = torch.bmm(query, key)
        weights = self.softmax(energy)
        out = torch.bmm(value, weights.permute(0, 2, 1))
        return out + x


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


class SpectraNetClassifier(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, dropout_rate: float = 0.3):
        super().__init__()
        self.feature_dim = 256

        self.conv1 = nn.Conv1d(1, 64, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(128)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        self.relu3 = nn.ReLU(inplace=True)
        self.pool3 = nn.AdaptiveAvgPool1d(50)

        self.attention = SpectrumAttention(in_channels=256)

        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(256 * 50, 512)
        self.bn_fc1 = nn.BatchNorm1d(512)
        self.relu_fc1 = nn.ReLU(inplace=True)

        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(512, self.feature_dim)
        self.bn_fc2 = nn.BatchNorm1d(self.feature_dim)
        self.relu_fc2 = nn.ReLU(inplace=True)

        self.dropout3 = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(self.feature_dim, num_classes)

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)

        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu3(self.bn3(self.conv3(x))))

        x = self.attention(x)
        x = x.view(x.size(0), -1)

        x = self.dropout1(x)
        x = self.relu_fc1(self.bn_fc1(self.fc1(x)))
        x = self.dropout2(x)
        x = self.relu_fc2(self.bn_fc2(self.fc2(x)))
        x = self.dropout3(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.forward_features(x))


# ---------------------------------------------------------------------------
# GADF (Gramian Angular Difference Field) 光谱图像转换
# ---------------------------------------------------------------------------

class GADFTransform(nn.Module):
    """将 1D 光谱转换为 Gramian Angular Difference Field (GADF) 图像。

    步骤：
    1. 将光谱重采样到 image_size 个点
    2. 逐样本归一化到 [-1, 1]
    3. 角度编码：phi_i = arccos(x_i)
    4. GADF 矩阵：G[i,j] = sin(phi_i - phi_j)

    输入：(B, L)  -> 输出：(B, 1, image_size, image_size)
    """

    def __init__(self, image_size: int) -> None:
        super().__init__()
        self.image_size = image_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        N = self.image_size

        # 1. 重采样
        x_3d = x.unsqueeze(1)  # (B, 1, L)
        x_res = F.interpolate(x_3d, size=N, mode="linear", align_corners=False).squeeze(1)  # (B, N)

        # 2. 逐样本归一化到 [-1, 1]
        x_min = x_res.min(dim=1, keepdim=True).values
        x_max = x_res.max(dim=1, keepdim=True).values
        denom = (x_max - x_min).clamp(min=1e-6)
        x_norm = (2.0 * (x_res - x_min) / denom - 1.0).clamp(-1.0 + 1e-6, 1.0 - 1e-6)

        # 3. 角度编码
        phi = torch.arccos(x_norm)  # (B, N)

        # 4. GADF 矩阵：sin(phi_i - phi_j)
        phi_i = phi.unsqueeze(2)  # (B, N, 1)
        phi_j = phi.unsqueeze(1)  # (B, 1, N)
        gadf = torch.sin(phi_i - phi_j)  # (B, N, N)

        return gadf.unsqueeze(1)  # (B, 1, N, N)


class _SpatialAttention2D(nn.Module):
    """轻量级空间自注意力，供 GADF 分类器内部使用。"""

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        hidden = max(1, in_channels // 8)
        self.query_conv = nn.Conv2d(in_channels, hidden, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, hidden, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        q = self.query_conv(x).reshape(b, -1, h * w).permute(0, 2, 1)
        k = self.key_conv(x).reshape(b, -1, h * w)
        v = self.value_conv(x).reshape(b, -1, h * w)
        attn = self.softmax(torch.bmm(q, k))
        out = torch.bmm(v, attn.permute(0, 2, 1)).reshape(b, -1, h, w)
        return out + x


class GADFSpectrumClassifier(nn.Module):
    """基于 GADF 图像的光谱分类模型。

    将 1D 光谱在线转换为 GADF 图像，再经 2D CNN + 注意力进行分类。
    与现有接口完全兼容（forward_features / feature_dim），
    可直接作为 spectrum_model 接入 ConcatFusion / CrossAttentionFusion。

    Args:
        input_dim:   原始光谱长度（运行时动态重采样，不影响网络参数）
        num_classes: 类别数
        image_size:  GADF 图像边长，64 或 128
        dropout_rate: Dropout 比例
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        image_size: int = 64,
        dropout_rate: float = 0.3,
    ) -> None:
        super().__init__()
        self.feature_dim = 256
        self.image_size = image_size

        self.gadf = GADFTransform(image_size)

        # 2D CNN 骨干（单通道输入）
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.relu3 = nn.ReLU(inplace=True)
        self.pool3 = nn.AdaptiveAvgPool2d((8, 8))

        self.attention = _SpatialAttention2D(in_channels=256)

        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(256 * 8 * 8, 512)
        self.bn_fc1 = nn.BatchNorm1d(512)
        self.relu_fc1 = nn.ReLU(inplace=True)

        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(512, self.feature_dim)
        self.bn_fc2 = nn.BatchNorm1d(self.feature_dim)
        self.relu_fc2 = nn.ReLU(inplace=True)

        self.dropout3 = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(self.feature_dim, num_classes)

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """输入 (B, L) 光谱，输出 (B, feature_dim) 特征向量。"""
        img = self.gadf(x)                                         # (B, 1, N, N)
        x = self.pool1(self.relu1(self.bn1(self.conv1(img))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu3(self.bn3(self.conv3(x))))
        x = self.attention(x)
        x = x.view(x.size(0), -1)
        x = self.dropout1(x)
        x = self.relu_fc1(self.bn_fc1(self.fc1(x)))
        x = self.dropout2(x)
        x = self.relu_fc2(self.bn_fc2(self.fc2(x)))
        x = self.dropout3(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.forward_features(x))


def build_spectrum_model(name: str, input_dim: int, num_classes: int) -> nn.Module:
    name = name.lower()
    if name == "spectrum_mlp":
        return SpectrumMLP(input_dim, num_classes)
    if name == "spectrum_cnn":
        return SpectrumCNN(input_dim, num_classes)
    if name == "spectranet_attn":
        return SpectraNetClassifier(input_dim, num_classes)
    if name == "gadf_cnn_64":
        return GADFSpectrumClassifier(input_dim, num_classes, image_size=64)
    if name == "gadf_cnn_128":
        return GADFSpectrumClassifier(input_dim, num_classes, image_size=128)
    raise ValueError(f"Unsupported spectrum model: {name}")
