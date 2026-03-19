"""
src/training/train.py

Главный модуль обучения для дипломной работы.
Поддерживает: EMNIST, любую модель из src/models/, ExperimentLogger + TensorBoard.

Запуск:
    python src/training/train.py
    python src/training/train.py --experiment dropout --epochs 20 --lr 0.001
"""

import argparse
import sys
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import confusion_matrix

# Добавляем корень проекта в путь (чтобы работали импорты src/...)
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from src.utils.logger import ExperimentLogger, Timer


# ══════════════════════════════════════════════════════════════════════
# 1. КОНФИГУРАЦИЯ
# ══════════════════════════════════════════════════════════════════════

def get_config() -> argparse.Namespace:
    """Параметры запуска через командную строку."""
    parser = argparse.ArgumentParser(description="Обучение модели на EMNIST")

    # Эксперимент
    parser.add_argument("--experiment", type=str, default="baseline",
                        help="Название эксперимента (папка в results/)")

    # Данные
    parser.add_argument("--data_dir",   type=str, default="data/raw",
                        help="Папка с датасетом EMNIST")
    parser.add_argument("--split",      type=str, default="balanced",
                        choices=["balanced", "letters", "digits", "byclass"],
                        help="Сплит EMNIST")
    parser.add_argument("--num_classes",type=int, default=47,
                        help="Число классов (47 для balanced)")

    # Обучение
    parser.add_argument("--epochs",     type=int,   default=15)
    parser.add_argument("--batch_size", type=int,   default=64)
    parser.add_argument("--lr",         type=float, default=0.001)
    parser.add_argument("--weight_decay",type=float,default=1e-4)

    # Scheduler: lr умножается на gamma каждые step_size эпох
    parser.add_argument("--step_size",  type=int,   default=5)
    parser.add_argument("--gamma",      type=float, default=0.5)

    # Сохранение
    parser.add_argument("--save_every", type=int, default=5,
                        help="Сохранять чекпоинт каждые N эпох (0 = только лучшую)")

    # TensorBoard
    parser.add_argument("--no_tb", action="store_true",
                        help="Отключить TensorBoard")

    # Воспроизводимость
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


# ══════════════════════════════════════════════════════════════════════
# 2. ДАННЫЕ
# ══════════════════════════════════════════════════════════════════════

def get_dataloaders(cfg: argparse.Namespace):
    """
    Загружает EMNIST и возвращает train/test DataLoader'ы.
    Данные скачиваются автоматически при первом запуске.
    """
    from torchvision import datasets, transforms

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),   # среднее/std EMNIST
    ])

    train_dataset = datasets.EMNIST(
        root=cfg.data_dir,
        split=cfg.split,
        train=True,
        download=True,
        transform=transform,
    )
    test_dataset = datasets.EMNIST(
        root=cfg.data_dir,
        split=cfg.split,
        train=False,
        download=True,
        transform=transform,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    return train_loader, test_loader


# ══════════════════════════════════════════════════════════════════════
# 3. ОДНА ЭПОХА ОБУЧЕНИЯ
# ══════════════════════════════════════════════════════════════════════

def train_one_epoch(
    model:      nn.Module,
    loader:     torch.utils.data.DataLoader,
    optimizer:  torch.optim.Optimizer,
    criterion:  nn.Module,
    device:     torch.device,
) -> tuple[float, float]:
    """
    Одна эпоха обучения.

    Returns:
        avg_loss: средний loss за эпоху
        accuracy: точность в процентах
    """
    model.train()

    total_loss    = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        outputs = model(images)
        loss    = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        # Статистика
        total_loss    += loss.item() * images.size(0)
        preds          = outputs.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += images.size(0)

    avg_loss = total_loss / total_samples
    accuracy = 100.0 * total_correct / total_samples

    return avg_loss, accuracy


# ══════════════════════════════════════════════════════════════════════
# 4. ОЦЕНКА НА ТЕСТОВОЙ ВЫБОРКЕ
# ══════════════════════════════════════════════════════════════════════

def evaluate(
    model:     nn.Module,
    loader:    torch.utils.data.DataLoader,
    criterion: nn.Module,
    device:    torch.device,
    collect_preds: bool = False,
) -> tuple:
    """
    Оценка модели на тестовой выборке.

    Args:
        collect_preds: если True — возвращает также images, labels, preds
                       (нужно для confusion matrix и примеров)

    Returns:
        avg_loss, accuracy  — всегда
        images, labels, preds  — только если collect_preds=True
    """
    model.eval()

    total_loss    = 0.0
    total_correct = 0
    total_samples = 0

    all_labels = []
    all_preds  = []
    sample_images = []  # небольшой буфер для примеров

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            loss    = criterion(outputs, labels)

            total_loss    += loss.item() * images.size(0)
            preds          = outputs.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += images.size(0)

            if collect_preds:
                all_labels.extend(labels.cpu().tolist())
                all_preds.extend(preds.cpu().tolist())

                # Сохраняем первые 10 изображений для visualize
                if len(sample_images) < 10:
                    n = min(10 - len(sample_images), images.size(0))
                    sample_images.extend(images[:n].cpu())

    avg_loss = total_loss / total_samples
    accuracy = 100.0 * total_correct / total_samples

    if collect_preds:
        return avg_loss, accuracy, sample_images, all_labels, all_preds

    return avg_loss, accuracy


# ══════════════════════════════════════════════════════════════════════
# 5. ГЛАВНАЯ ФУНКЦИЯ ОБУЧЕНИЯ
# ══════════════════════════════════════════════════════════════════════

def train(cfg: argparse.Namespace):
    """Полный цикл обучения: инициализация → обучение → сохранение итогов."""

    # ── Воспроизводимость ──────────────────────────────────────────────
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    # ── Устройство ────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Логгер ────────────────────────────────────────────────────────
    exp_logger = ExperimentLogger(
        experiment_name=cfg.experiment,
        use_tensorboard=not cfg.no_tb,
    )
    log = exp_logger.logger   # удобный псевдоним

    log.info(f"Устройство: {device}")
    log.info(f"Seed: {cfg.seed}")

    # ── Данные ────────────────────────────────────────────────────────
    log.info("Загрузка EMNIST...")
    train_loader, test_loader = get_dataloaders(cfg)
    log.info(f"Train: {len(train_loader.dataset):,} | "
             f"Test: {len(test_loader.dataset):,} | "
             f"Батч: {cfg.batch_size}")

    # ── Модель ────────────────────────────────────────────────────────
    # Импортируем здесь — легко поменять на другую модель
    from src.models.simple_cnn import SimpleCNN
    model = SimpleCNN(num_classes=cfg.num_classes).to(device)

    # ── Параметры обучения ────────────────────────────────────────────
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    scheduler = StepLR(optimizer, step_size=cfg.step_size, gamma=cfg.gamma)

    # ── Инфо о модели → логгер + TensorBoard ──────────────────────────
    model_params = {
        "optimizer":    "Adam",
        "lr":           cfg.lr,
        "weight_decay": cfg.weight_decay,
        "batch_size":   cfg.batch_size,
        "epochs":       cfg.epochs,
        "scheduler":    f"StepLR(step={cfg.step_size}, gamma={cfg.gamma})",
        "split":        cfg.split,
        "num_classes":  cfg.num_classes,
        "seed":         cfg.seed,
    }
    sample_input = torch.zeros(1, 1, 28, 28).to(device)
    exp_logger.log_model_info(model, model_params, sample_input)

    # ══════════════════════════════════════════════════════════════════
    # ГЛАВНЫЙ ЦИКЛ ОБУЧЕНИЯ
    # ══════════════════════════════════════════════════════════════════
    log.info("\nСТАРТ ОБУЧЕНИЯ\n" + "-" * 60)

    for epoch in range(1, cfg.epochs + 1):

        with Timer() as t:
            train_loss, train_acc = train_one_epoch(
                model, train_loader, optimizer, criterion, device
            )
            test_loss, test_acc = evaluate(
                model, test_loader, criterion, device
            )

        # Обновляем lr
        scheduler.step()
        exp_logger.log_learning_rate(optimizer, epoch)

        # Логируем эпоху (файл + CSV + TensorBoard)
        exp_logger.log_epoch(
            epoch, cfg.epochs,
            train_loss, train_acc,
            test_loss, test_acc,
            t.duration,
        )

        # Гистограммы весов раз в 5 эпох
        if epoch % 5 == 0:
            exp_logger.log_gradients(model, epoch)

        # Сохраняем лучшую модель
        is_best = (test_acc >= exp_logger.best_acc)
        if is_best:
            exp_logger.save_model(model, epoch, is_best=True)

        # Чекпоинт каждые N эпох
        if cfg.save_every > 0 and epoch % cfg.save_every == 0:
            exp_logger.save_model(model, epoch, is_best=False)

    # ══════════════════════════════════════════════════════════════════
    # ФИНАЛЬНАЯ ОЦЕНКА + ВИЗУАЛИЗАЦИЯ
    # ══════════════════════════════════════════════════════════════════
    log.info("\nФИНАЛЬНАЯ ОЦЕНКА...")

    _, _, sample_images, all_labels, all_preds = evaluate(
        model, test_loader, criterion, device, collect_preds=True
    )

    # Матрица ошибок
    cm = confusion_matrix(all_labels, all_preds)
    exp_logger.log_confusion_matrix(cm)

    # Примеры предсказаний
    exp_logger.log_example_predictions(
        images=sample_images,
        labels=all_labels[:10],
        predictions=all_preds[:10],
    )

    # Итоговый отчёт
    exp_logger.log_final_summary()
    exp_logger.close()


# ══════════════════════════════════════════════════════════════════════
# ТОЧКА ВХОДА
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    cfg = get_config()
    train(cfg)