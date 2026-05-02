"""
src/models/improved_cnn_v5.py

ImprovedCNN_v5 — авторская архитектура с механизмом внимания
Squeeze-and-Excitation (Hu et al., CVPR 2018). SE-блоки выполняют
адаптивное перевзвешивание каналов признаков, что позволяет сети
концентрироваться на наиболее информативных каналах.

Архитектура: residual-блоки + SE-attention + Global Average Pooling.

    Stem:    Conv 1→32, BN, GELU                     → 32×28×28
    Stage 1: 2× SEResBlock(32)                       → 32×28×28
    Stage 2: 2× SEResBlock(32→64),    stride=2       → 64×14×14
    Stage 3: 2× SEResBlock(64→128),   stride=2       → 128×7×7
    Head:    GAP → Dropout(0.3) → FC(128 → num_classes)

Параметров: ~520 тыс. (на ~50K больше чем v4 — за счёт SE-модулей).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation: channel attention.
      Squeeze:  GAP по пространственным осям → вектор C
      Excite:   2 FC-слоя со sigmoid → веса каналов [0, 1]
      Scale:    исходные карты признаков × веса каналов
    """
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.fc1 = nn.Linear(channels, hidden, bias=False)
        self.fc2 = nn.Linear(hidden, channels, bias=False)

    def forward(self, x):
        b, c, _, _ = x.shape
        s = x.mean(dim=(2, 3))                  # squeeze: (B, C)
        s = F.gelu(self.fc1(s))                 # excite, шаг 1
        s = torch.sigmoid(self.fc2(s))          # excite, шаг 2 → веса
        return x * s.view(b, c, 1, 1)           # scale


class SEResidualBlock(nn.Module):
    """Residual-блок с SE-механизмом перед residual-сложением."""
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)
        self.se    = SEBlock(out_ch)

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
        out = self.se(out)                       # ← attention перед residual
        return F.gelu(out + identity)


class ImprovedCNN_v5(nn.Module):
    def __init__(self, num_classes: int = 47):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
        )
        self.stage1 = nn.Sequential(
            SEResidualBlock(32, 32),
            SEResidualBlock(32, 32),
        )
        self.stage2 = nn.Sequential(
            SEResidualBlock(32, 64, stride=2),
            SEResidualBlock(64, 64),
        )
        self.stage3 = nn.Sequential(
            SEResidualBlock(64, 128, stride=2),
            SEResidualBlock(128, 128),
        )
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
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.gap(x).flatten(1)
        x = self.dropout(x)
        return self.fc(x)
