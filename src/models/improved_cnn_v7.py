"""
src/models/improved_cnn_v7.py

ImprovedCNN_v7 — гибридная архитектура CNN + Transformer.
Объединяет преимущества свёрточных сетей (локальная индуктивная
смещённость, иерархия признаков) и Transformer-механизма
(глобальное внимание между всеми позициями).

Архитектура:
    CNN-backbone (как у v4 без head'а):
        Stem + Stage1 + Stage2 + Stage3                → 128×7×7

    Преобразование в последовательность токенов:
        Reshape (B, 128, 7, 7) → (B, 49, 128)
        Добавление обучаемых позиционных эмбеддингов  → 128

    Transformer Encoder (1 блок):
        Multi-Head Self-Attention (4 heads, dim=128)
        FeedForward (128→256→128) с GELU + Dropout

    Classification head:
        Mean pooling по 49 токенам → 128
        Dropout → FC(128 → num_classes)

Параметров: ~700 тыс.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Тот же ResBlock, что в v4."""
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


class TransformerBlock(nn.Module):
    """
    Один блок Transformer Encoder в pre-norm стиле:
        x + MHSA(LN(x))
        x + FFN(LN(x))
    """
    def __init__(self, dim: int = 128, num_heads: int = 4,
                 mlp_ratio: int = 2, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = nn.MultiheadAttention(dim, num_heads,
                                            dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mlp_ratio, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # Self-attention с residual
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + attn_out
        # FFN с residual
        x = x + self.mlp(self.norm2(x))
        return x


class ImprovedCNN_v7(nn.Module):
    def __init__(self, num_classes: int = 47):
        super().__init__()
        # ── CNN-backbone ─────────────────────────────────────────
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
        )
        self.stage1 = nn.Sequential(
            ResidualBlock(32, 32),
            ResidualBlock(32, 32),
        )
        self.stage2 = nn.Sequential(
            ResidualBlock(32, 64, stride=2),
            ResidualBlock(64, 64),
        )
        self.stage3 = nn.Sequential(
            ResidualBlock(64, 128, stride=2),
            ResidualBlock(128, 128),
        )

        # ── Transformer-голова ────────────────────────────────────
        # 7×7=49 токенов размерности 128
        self.num_tokens = 49
        self.dim        = 128
        # Обучаемые позиционные эмбеддинги
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_tokens, self.dim)
        )
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.transformer = TransformerBlock(
            dim=self.dim, num_heads=4, mlp_ratio=2, dropout=0.1
        )
        self.norm = nn.LayerNorm(self.dim)

        # ── Classification head ──────────────────────────────────
        self.dropout = nn.Dropout(0.3)
        self.fc      = nn.Linear(self.dim, num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # CNN-часть
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)                       # (B, 128, 7, 7)

        # → последовательность токенов
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)         # (B, 49, 128)
        x = x + self.pos_embed                   # позиционные эмбеддинги

        # Transformer
        x = self.transformer(x)
        x = self.norm(x)

        # Mean pooling по токенам
        x = x.mean(dim=1)                        # (B, 128)
        x = self.dropout(x)
        return self.fc(x)
