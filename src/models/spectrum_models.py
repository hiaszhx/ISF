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
        self.multiscale_dims = [16, 32, 64]
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

    def forward_multiscale_features(self, x: torch.Tensor) -> list[torch.Tensor]:
        x = x.unsqueeze(1)
        feats = []
        for i, layer in enumerate(self.net):
            x = layer(x)
            if i == 2:   # 16 channels after first pool
                feats.append(F.adaptive_avg_pool1d(x, 1).flatten(1))
            elif i == 5: # 32 channels after second pool
                feats.append(F.adaptive_avg_pool1d(x, 1).flatten(1))
            elif i == 8: # 64 channels after adaptive pool
                feats.append(x.flatten(1))
        return feats

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.forward_features(x))


class SpectraNetClassifier(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, dropout_rate: float = 0.3):
        super().__init__()
        self.feature_dim = 256
        self.multiscale_dims = [64, 128, 256]

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

    def forward_multiscale_features(self, x: torch.Tensor) -> list[torch.Tensor]:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        f1 = F.adaptive_avg_pool1d(x, 1).flatten(1)  # 64
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        f2 = F.adaptive_avg_pool1d(x, 1).flatten(1)  # 128
        x = self.pool3(self.relu3(self.bn3(self.conv3(x))))
        x = self.attention(x)
        f3 = F.adaptive_avg_pool1d(x, 1).flatten(1)  # 256
        return [f1, f2, f3]

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


# ---------------------------------------------------------------------------
# Mamba (Selective State Space Model) — 纯 PyTorch 实现
# ---------------------------------------------------------------------------


class _MambaSSM(nn.Module):
    """选择性状态空间模型核心。

    实现 Mamba (Gu & Dao, 2023) 的选择性扫描机制。
    输入依赖的 B, C, dt 使模型能够选择性地保留或遗忘序列信息。
    """

    def __init__(self, d_inner: int, d_state: int = 16) -> None:
        super().__init__()
        self.d_inner = d_inner
        self.d_state = d_state

        # 可学习 SSM 参数
        A = torch.arange(1, d_state + 1, dtype=torch.float32)
        A = A.unsqueeze(0).expand(d_inner, -1).clone()
        self.A_log = nn.Parameter(torch.log(A))       # (D, N)
        self.D = nn.Parameter(torch.ones(d_inner))     # (D,) skip

        # 输入依赖投影: dt(D) + B(N) + C(N)
        self.x_proj = nn.Linear(d_inner, d_inner + d_state * 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, D) → (B, L, D)"""
        B_sz, L, D = x.shape
        N = self.d_state

        x_proj = self.x_proj(x)                                       # (B, L, D+2N)
        dt, B_ssm, C_ssm = x_proj.split([D, N, N], dim=-1)
        dt = F.softplus(dt)                                            # 保证正值

        A = -torch.exp(self.A_log)                                     # (D, N) 负数保持稳定

        # 顺序扫描（特征图 L≤256 时足够高效）
        h = x.new_zeros(B_sz, D, N)
        ys = []
        for t in range(L):
            dt_t = dt[:, t]                                            # (B, D)
            A_bar = torch.exp(dt_t.unsqueeze(-1) * A)                  # (B, D, N)
            B_bar = dt_t.unsqueeze(-1) * B_ssm[:, t].unsqueeze(1)      # (B, D, N)
            h = A_bar * h + B_bar * x[:, t].unsqueeze(-1)              # (B, D, N)
            y_t = (h * C_ssm[:, t].unsqueeze(1)).sum(-1)               # (B, D)
            y_t = y_t + self.D * x[:, t]                               # skip connection
            ys.append(y_t)

        return torch.stack(ys, dim=1)                                  # (B, L, D)


class _MambaBlock(nn.Module):
    """单个 Mamba 块：LayerNorm → in_proj → [Conv1d + SiLU → SSM] ⊗ [SiLU gate] → out_proj + 残差。"""

    def __init__(self, d_model: int, d_state: int = 16,
                 d_conv: int = 4, expand: int = 2) -> None:
        super().__init__()
        d_inner = d_model * expand

        self.norm = nn.LayerNorm(d_model)
        self.in_proj = nn.Linear(d_model, d_inner * 2, bias=False)

        # 深度可分离卷积用于局部上下文
        self.conv1d = nn.Conv1d(
            d_inner, d_inner,
            kernel_size=d_conv, padding=d_conv - 1,
            groups=d_inner, bias=True,
        )

        self.ssm = _MambaSSM(d_inner, d_state)
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, D) → (B, L, D)"""
        residual = x
        x = self.norm(x)

        xz = self.in_proj(x)                                          # (B, L, 2*D_inner)
        x_branch, z = xz.chunk(2, dim=-1)

        # 局部卷积分支
        x_branch = x_branch.transpose(1, 2)                           # (B, D_inner, L)
        x_branch = self.conv1d(x_branch)[:, :, :residual.size(1)]     # 因果裁剪
        x_branch = x_branch.transpose(1, 2)
        x_branch = F.silu(x_branch)

        # 选择性 SSM
        x_branch = self.ssm(x_branch)

        # 门控输出
        out = x_branch * F.silu(z)
        out = self.out_proj(out)

        return out + residual


class _SpatialMamba2D(nn.Module):
    """双向 Mamba 用于 2D 特征图，可替代空间自注意力。

    将 (B, C, H, W) 展平为序列，正向与反向各运行 Mamba 块，
    合并双向输出后恢复为 2D 特征图。
    """

    def __init__(self, in_channels: int, d_state: int = 16,
                 d_conv: int = 4, expand: int = 2,
                 n_layers: int = 1) -> None:
        super().__init__()
        self.fwd_blocks = nn.ModuleList([
            _MambaBlock(in_channels, d_state, d_conv, expand)
            for _ in range(n_layers)
        ])
        self.bwd_blocks = nn.ModuleList([
            _MambaBlock(in_channels, d_state, d_conv, expand)
            for _ in range(n_layers)
        ])
        self.merge = nn.Linear(in_channels * 2, in_channels, bias=False)
        self.norm = nn.LayerNorm(in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, H, W) → (B, C, H, W)"""
        B, C, H, W = x.shape

        # 展平为序列 (B, L, C)，L = H*W
        seq = x.reshape(B, C, H * W).permute(0, 2, 1)

        # 正向扫描
        fwd = seq
        for blk in self.fwd_blocks:
            fwd = blk(fwd)

        # 反向扫描
        bwd = seq.flip(1)
        for blk in self.bwd_blocks:
            bwd = blk(bwd)
        bwd = bwd.flip(1)

        # 合并双向
        merged = self.merge(torch.cat([fwd, bwd], dim=-1))            # (B, L, C)
        merged = self.norm(merged)

        # 恢复 2D
        out = merged.permute(0, 2, 1).reshape(B, C, H, W)
        return out + x                                                 # 残差


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
        self.multiscale_dims = [64, 128, 256]
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

    def forward_multiscale_features(self, x: torch.Tensor) -> list[torch.Tensor]:
        img = self.gadf(x)
        x = self.pool1(self.relu1(self.bn1(self.conv1(img))))
        f1 = F.adaptive_avg_pool2d(x, (1, 1)).flatten(1)  # 64
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        f2 = F.adaptive_avg_pool2d(x, (1, 1)).flatten(1)  # 128
        x = self.pool3(self.relu3(self.bn3(self.conv3(x))))
        x = self.attention(x)
        f3 = F.adaptive_avg_pool2d(x, (1, 1)).flatten(1)  # 256
        return [f1, f2, f3]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.forward_features(x))


class GADFMambaSpectrumClassifier(nn.Module):
    """基于 GADF + Mamba 的光谱分类模型。

    用双向 Mamba SSM 替代自注意力层，对 GADF 图像的 CNN 特征进行序列建模。
    可选保留一个轻量注意力层构成 Mamba+Attention 混合模式。

    Args:
        input_dim:      原始光谱长度
        num_classes:    类别数
        image_size:     GADF 图像边长，64 或 128
        use_attention:  是否在 Mamba 之后追加空间注意力（混合模式）
        d_state:        SSM 隐状态维度
        d_conv:         Mamba 局部卷积核大小
        expand:         内部维度扩展倍数
        dropout_rate:   Dropout 比例
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        image_size: int = 64,
        use_attention: bool = False,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout_rate: float = 0.3,
    ) -> None:
        super().__init__()
        self.feature_dim = 256
        self.multiscale_dims = [64, 128, 256]
        self.image_size = image_size

        self.gadf = GADFTransform(image_size)

        # 2D CNN 骨干
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

        # Mamba 替代自注意力
        self.mamba = _SpatialMamba2D(
            in_channels=256, d_state=d_state, d_conv=d_conv, expand=expand,
        )

        # 可选：Mamba 之后追加一个轻量注意力层（混合模式）
        self.attention = _SpatialAttention2D(256) if use_attention else None

        # MLP 分类头
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
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """输入 (B, L) 光谱，输出 (B, feature_dim) 特征向量。"""
        img = self.gadf(x)                                         # (B, 1, N, N)
        x = self.pool1(self.relu1(self.bn1(self.conv1(img))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu3(self.bn3(self.conv3(x))))
        x = self.mamba(x)
        if self.attention is not None:
            x = self.attention(x)
        x = x.reshape(x.size(0), -1)
        x = self.dropout1(x)
        x = self.relu_fc1(self.bn_fc1(self.fc1(x)))
        x = self.dropout2(x)
        x = self.relu_fc2(self.bn_fc2(self.fc2(x)))
        x = self.dropout3(x)
        return x

    def forward_multiscale_features(self, x: torch.Tensor) -> list[torch.Tensor]:
        img = self.gadf(x)
        x = self.pool1(self.relu1(self.bn1(self.conv1(img))))
        f1 = F.adaptive_avg_pool2d(x, (1, 1)).flatten(1)  # 64
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        f2 = F.adaptive_avg_pool2d(x, (1, 1)).flatten(1)  # 128
        x = self.pool3(self.relu3(self.bn3(self.conv3(x))))
        x = self.mamba(x)
        if self.attention is not None:
            x = self.attention(x)
        f3 = F.adaptive_avg_pool2d(x, (1, 1)).flatten(1)  # 256
        return [f1, f2, f3]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.forward_features(x))


class GADFMobileNetV2SpectrumClassifier(nn.Module):
    """GADF + MobileNetV2 光谱分类器。

    将 1D 光谱在线转换为 GADF 图像（单通道），扩展为 3 通道后
    送入预训练 MobileNetV2 进行特征提取和分类。

    Args:
        input_dim:   原始光谱长度
        num_classes: 类别数
        image_size:  GADF 图像边长，64 或 128
    """

    def __init__(self, input_dim: int, num_classes: int, image_size: int = 64) -> None:
        super().__init__()
        from torchvision import models

        self.gadf = GADFTransform(image_size)

        m = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        self.features = m.features
        in_f = m.classifier[1].in_features  # 1280
        self.feature_dim = in_f
        self.multiscale_dims = [24, 64, 1280]
        self.classifier = nn.Linear(in_f, num_classes)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        img = self.gadf(x)                        # (B, 1, N, N)
        img = img.expand(-1, 3, -1, -1)           # (B, 3, N, N)
        x = self.features(img)
        x = F.adaptive_avg_pool2d(x, (1, 1)).flatten(1)
        return x

    def forward_multiscale_features(self, x: torch.Tensor) -> list[torch.Tensor]:
        img = self.gadf(x).expand(-1, 3, -1, -1)
        feats = []
        for i, block in enumerate(self.features):
            img = block(img)
            if i == 3:     # 24 channels
                feats.append(F.adaptive_avg_pool2d(img, (1, 1)).flatten(1))
            elif i == 10:  # 64 channels
                feats.append(F.adaptive_avg_pool2d(img, (1, 1)).flatten(1))
        feats.append(F.adaptive_avg_pool2d(img, (1, 1)).flatten(1))  # 1280
        return feats

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
    if name == "gadf_mamba_64":
        return GADFMambaSpectrumClassifier(input_dim, num_classes, image_size=64, use_attention=False)
    if name == "gadf_mamba_128":
        return GADFMambaSpectrumClassifier(input_dim, num_classes, image_size=128, use_attention=False)
    if name == "gadf_mamba_attn_64":
        return GADFMambaSpectrumClassifier(input_dim, num_classes, image_size=64, use_attention=True)
    if name == "gadf_mamba_attn_128":
        return GADFMambaSpectrumClassifier(input_dim, num_classes, image_size=128, use_attention=True)
    if name == "gadf_mobilenetv2_64":
        return GADFMobileNetV2SpectrumClassifier(input_dim, num_classes, image_size=64)
    if name == "gadf_mobilenetv2_128":
        return GADFMobileNetV2SpectrumClassifier(input_dim, num_classes, image_size=128)
    raise ValueError(f"Unsupported spectrum model: {name}")
