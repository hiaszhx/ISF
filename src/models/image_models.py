from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class ImageAttention(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        hidden = max(1, in_channels // 8)
        self.query_conv = nn.Conv2d(in_channels, hidden, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, hidden, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        query = self.query_conv(x).reshape(b, -1, h * w).permute(0, 2, 1)
        key = self.key_conv(x).reshape(b, -1, h * w)
        value = self.value_conv(x).reshape(b, -1, h * w)

        energy = torch.bmm(query, key)
        weights = self.softmax(energy)
        out = torch.bmm(value, weights.permute(0, 2, 1)).reshape(b, -1, h, w)
        return out + x


class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.feature_dim = 128
        self.multiscale_dims = [32, 64, 128]
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(self.feature_dim, num_classes)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        return x

    def forward_multiscale_features(self, x: torch.Tensor) -> list[torch.Tensor]:
        feats = []
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i == 2:   # 32 channels after first pool
                feats.append(F.adaptive_avg_pool2d(x, (1, 1)).flatten(1))
            elif i == 5: # 64 channels after second pool
                feats.append(F.adaptive_avg_pool2d(x, (1, 1)).flatten(1))
            elif i == 8: # 128 channels after adaptive pool
                feats.append(x.flatten(1))
        return feats

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.forward_features(x))


class ResNet18Classifier(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        # m = models.resnet18(weights=None)
        m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        in_f = m.fc.in_features
        self.feature_dim = in_f
        self.multiscale_dims = [64, 128, 512]
        m.fc = nn.Linear(in_f, num_classes)
        self.model = m

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        return x

    def forward_multiscale_features(self, x: torch.Tensor) -> list[torch.Tensor]:
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        f1 = F.adaptive_avg_pool2d(x, (1, 1)).flatten(1)  # 64
        x = self.model.layer2(x)
        f2 = F.adaptive_avg_pool2d(x, (1, 1)).flatten(1)  # 128
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        f3 = F.adaptive_avg_pool2d(x, (1, 1)).flatten(1)  # 512
        return [f1, f2, f3]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.forward_features(x)
        return self.model.fc(feat)


class MobileNetV2Classifier(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        # m = models.mobilenet_v2(weights=None)
        m = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        in_f = m.classifier[1].in_features
        self.feature_dim = in_f
        self.multiscale_dims = [24, 64, 1280]
        m.classifier[1] = nn.Linear(in_f, num_classes)
        self.model = m

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model.features(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        return x

    def forward_multiscale_features(self, x: torch.Tensor) -> list[torch.Tensor]:
        feats = []
        for i, block in enumerate(self.model.features):
            x = block(x)
            if i == 3:    # 24 channels
                feats.append(F.adaptive_avg_pool2d(x, (1, 1)).flatten(1))
            elif i == 10: # 64 channels
                feats.append(F.adaptive_avg_pool2d(x, (1, 1)).flatten(1))
        feats.append(F.adaptive_avg_pool2d(x, (1, 1)).flatten(1))  # 1280
        return feats

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.forward_features(x)
        return self.model.classifier(feat)


class ImageSpectraStyleClassifier(nn.Module):
    def __init__(self, num_classes: int, dropout_rate: float = 0.3):
        super().__init__()
        self.feature_dim = 256
        self.multiscale_dims = [64, 128, 256]

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, padding=3)
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

        self.attention = ImageAttention(in_channels=256)

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
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
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

    def forward_multiscale_features(self, x: torch.Tensor) -> list[torch.Tensor]:
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        f1 = F.adaptive_avg_pool2d(x, (1, 1)).flatten(1)  # 64
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        f2 = F.adaptive_avg_pool2d(x, (1, 1)).flatten(1)  # 128
        x = self.pool3(self.relu3(self.bn3(self.conv3(x))))
        x = self.attention(x)
        f3 = F.adaptive_avg_pool2d(x, (1, 1)).flatten(1)  # 256
        return [f1, f2, f3]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.forward_features(x))


def build_image_model(name: str, num_classes: int) -> nn.Module:
    name = name.lower()
    if name == "simple_cnn":
        return SimpleCNN(num_classes)
    if name == "resnet18":
        return ResNet18Classifier(num_classes)
    if name in {"mobilenetv2", "mobilenet_v2"}:
        return MobileNetV2Classifier(num_classes)
    if name in {"image_attn"}:
        return ImageSpectraStyleClassifier(num_classes)
    raise ValueError(f"Unsupported image model: {name}")
