from __future__ import annotations

import torch
import torch.nn as nn


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


def build_spectrum_model(name: str, input_dim: int, num_classes: int) -> nn.Module:
    name = name.lower()
    if name == "spectrum_mlp":
        return SpectrumMLP(input_dim, num_classes)
    if name == "spectrum_cnn":
        return SpectrumCNN(input_dim, num_classes)
    if name == "spectranet_attn":
        return SpectraNetClassifier(input_dim, num_classes)
    raise ValueError(f"Unsupported spectrum model: {name}")
