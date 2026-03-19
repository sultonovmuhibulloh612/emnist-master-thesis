"""
src/dataset/emnist_loader.py

Модуль загрузки и аугментации данных EMNIST.
Вынесен отдельно чтобы легко менять аугментацию не трогая train.py.

Использование:
    from src.dataset.emnist_loader import get_dataloaders, get_augmentation_info

    train_loader, test_loader = get_dataloaders(cfg)
"""

import torch
from torchvision import datasets, transforms
from typing import Tuple


# ══════════════════════════════════════════════════════════════════════
# ПРЕСЕТЫ АУГМЕНТАЦИИ
# Легко переключаться между ними через --augmentation в train.py
# ══════════════════════════════════════════════════════════════════════

def _get_train_transform(augmentation: str) -> transforms.Compose:
    """
    Возвращает transform для обучающей выборки.

    Пресеты:
        none   — только нормализация (для baseline, честное сравнение)
        light  — лёгкая аугментация (рекомендуется)
        strong — сильная аугментация (если модель сильно переобучается)
    """

    # Базовые шаги — всегда одинаковые
    base = [
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ]

    if augmentation == "none":
        # Без аугментации — для честного baseline
        return transforms.Compose(base)

    elif augmentation == "light":
        # Лёгкая аугментация — хороший баланс
        return transforms.Compose([
            transforms.RandomRotation(10),              # поворот ±10°
            transforms.RandomAffine(
                degrees=0,
                translate=(0.1, 0.1),                  # сдвиг до 10%
            ),
        ] + base)

    elif augmentation == "strong":
        # Сильная аугментация — если модель сильно переобучается
        return transforms.Compose([
            transforms.RandomRotation(15),              # поворот ±15°
            transforms.RandomAffine(
                degrees=0,
                translate=(0.15, 0.15),                # сдвиг до 15%
                scale=(0.85, 1.15),                    # масштаб 85–115%
                shear=10,                              # наклон ±10°
            ),
            transforms.RandomPerspective(
                distortion_scale=0.2,
                p=0.3,                                 # 30% шанс искажения
            ),
        ] + base)

    else:
        raise ValueError(
            f"Неизвестный пресет аугментации: '{augmentation}'. "
            f"Доступные: none, light, strong"
        )


def _get_test_transform() -> transforms.Compose:
    """
    Transform для тестовой выборки.
    Никогда не аугментируем тест — только нормализация.
    """
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])


# ══════════════════════════════════════════════════════════════════════
# ГЛАВНАЯ ФУНКЦИЯ
# ══════════════════════════════════════════════════════════════════════

def get_dataloaders(
    data_dir:    str = "data/raw",
    split:       str = "balanced",
    batch_size:  int = 64,
    num_workers: int = 2,
    augmentation: str = "light",
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Загружает EMNIST и возвращает train/test DataLoader'ы.

    Args:
        data_dir:     Папка с данными
        split:        Сплит EMNIST (balanced / letters / digits / byclass)
        batch_size:   Размер батча
        num_workers:  Потоков для загрузки (0 = без многопоточности)
        augmentation: Пресет аугментации (none / light / strong)

    Returns:
        train_loader, test_loader
    """
    train_transform = _get_train_transform(augmentation)
    test_transform  = _get_test_transform()

    train_dataset = datasets.EMNIST(
        root=data_dir, split=split,
        train=True,  download=True,
        transform=train_transform,
    )
    test_dataset = datasets.EMNIST(
        root=data_dir, split=split,
        train=False, download=True,
        transform=test_transform,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, test_loader


def get_augmentation_info(augmentation: str) -> str:
    """
    Возвращает описание аугментации для логгера.
    Используется в train.py при записи model_params.
    """
    info = {
        "none":   "Без аугментации",
        "light":  "Поворот ±10°, сдвиг 10%",
        "strong": "Поворот ±15°, сдвиг 15%, масштаб, наклон, перспектива",
    }
    return info.get(augmentation, "Неизвестно")