import torch.nn as nn
import torch.nn.functional as F

class ImprovedCNN_v2(nn.Module):

    def __init__(self, num_classes=47):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 32, 3)
        self.bn1   = nn.BatchNorm2d(32)    # ← стабилизирует обучение

        self.conv2 = nn.Conv2d(32, 64, 3)
        self.bn2   = nn.BatchNorm2d(64)    # ← стабилизирует обучение

        self.fc1     = nn.Linear(1600, 256)
        self.dropout = nn.Dropout(0.5)     # ← борется с переобучением
        self.fc2     = nn.Linear(256, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)

        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)

        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)                # ← применяем dropout
        x = self.fc2(x)

        return x