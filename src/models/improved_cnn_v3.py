import torch
import torch.nn as nn
import torch.nn.functional as F


class ImprovedCNN_v3(nn.Module):
    """
    Трёхслойная CNN с BatchNorm + Dropout.

    Архитектура:
        Input: 1×28×28
        Conv1: 1→32,  BN, ReLU, MaxPool → 32×13×13
        Conv2: 32→64, BN, ReLU, MaxPool → 64×5×5
        Conv3: 64→128, BN, ReLU         → 128×3×3  (без пулинга — картинка маленькая)
        Flatten → 1152
        FC1: 1152→256, ReLU, Dropout(0.5)
        FC2: 256→128,  ReLU, Dropout(0.3)
        FC3: 128→num_classes
    """

    def __init__(self, num_classes: int = 47):
        super().__init__()

        # ── Свёрточные блоки ──────────────────────────────────────────
        self.conv1 = nn.Conv2d(1,  32,  kernel_size=3, padding=1)  # padding=1 → размер не уменьшается
        self.bn1   = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64,  kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm2d(128)

        # ── Полносвязные слои ─────────────────────────────────────────
        # После 2 пулингов: 28 → 14 → 7, каналов 128 → 128×7×7 = 6272
        self.fc1      = nn.Linear(128 * 7 * 7, 256)
        self.dropout1 = nn.Dropout(0.5)

        self.fc2      = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(0.3)

        self.fc3      = nn.Linear(128, num_classes)

    def forward(self, x):
        # Блок 1: conv → bn → relu → pool
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)                # 28×28 → 14×14

        # Блок 2: conv → bn → relu → pool
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)                # 14×14 → 7×7

        # Блок 3: conv → bn → relu (без пулинга)
        x = F.relu(self.bn3(self.conv3(x)))   # 7×7 → 7×7

        # Классификатор
        x = x.view(x.size(0), -1)             # 128×7×7 = 6272

        x = F.relu(self.fc1(x))
        x = self.dropout1(x)

        x = F.relu(self.fc2(x))
        x = self.dropout2(x)

        x = self.fc3(x)

        return x
