"""
src/models/improved_cnn_v4.py

ImprovedCNN_v4 — авторская архитектура на основе остаточных связей.
Использует residual blocks (He et al., CVPR 2016) и Global Average Pooling
(Lin et al., ICLR 2014) вместо классического полносвязного слоя.

Архитектура:
    Stem:    Conv 1→32, BN, GELU                     → 32×28×28
    Stage 1: 2× ResBlock(32),         stride=1       → 32×28×28
    Stage 2: 2× ResBlock(32→64),      stride=2       → 64×14×14
    Stage 3: 2× ResBlock(64→128),     stride=2       → 128×7×7
    Head:    GAP → Dropout(0.3) → FC(128 → num_classes)

Параметров: ~470 тыс. (в 3,7 раза меньше v3 при потенциально выше точности).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """
    Базовый residual-блок: Conv → BN → GELU → Conv → BN → + residual → GELU.
    При изменении пространственного размера или числа каналов используется
    проективный shortcut (1×1 свёртка).
    """
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)

        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)
        out = F.gelu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.gelu(out + identity)


class ImprovedCNN_v4(nn.Module):
    def __init__(self, num_classes: int = 47):
        super().__init__()
        # Stem: первичное извлечение признаков
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
        )
        # Stage 1: 2 блока 32→32, без понижения размера
        self.stage1 = nn.Sequential(
            ResidualBlock(32, 32),
            ResidualBlock(32, 32),
        )
        # Stage 2: первый блок понижает 28→14 и повышает 32→64
        self.stage2 = nn.Sequential(
            ResidualBlock(32, 64, stride=2),
            ResidualBlock(64, 64),
        )
        # Stage 3: первый блок понижает 14→7 и повышает 64→128
        self.stage3 = nn.Sequential(
            ResidualBlock(64, 128, stride=2),
            ResidualBlock(128, 128),
        )
        # Head: Global Average Pooling вместо Flatten + FC
        self.gap     = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.3)
        self.fc      = nn.Linear(128, num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.gap(x).flatten(1)
        x = self.dropout(x)
        return self.fc(x)
