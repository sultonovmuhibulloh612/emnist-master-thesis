"""
src/training/train.py
"""

import argparse
import sys
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from sklearn.metrics import confusion_matrix

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from src.utils.logger import ExperimentLogger, Timer
from src.dataset.emnist_loader import get_dataloaders, get_augmentation_info  
from torch.optim.lr_scheduler import ReduceLROnPlateau

# ══════════════════════════════════════════════════════════════════════
# 1. РЕЕСТР МОДЕЛЕЙ
# ══════════════════════════════════════════════════════════════════════

def get_model(model_name: str, num_classes: int) -> nn.Module:
    models = {
        "simple_cnn":      "src.models.simple_cnn.SimpleCNN",
        "improved_cnn_v2": "src.models.improved_cnn_v2.ImprovedCNN_v2",
        "improved_cnn_v3": "src.models.improved_cnn_v3.ImprovedCNN_v3",
        "improved_cnn_v4": "src.models.improved_cnn_v4.ImprovedCNN_v4",
    }
    if model_name not in models:
        raise ValueError(f"Модель '{model_name}' не найдена. Доступные: {list(models.keys())}")

    module_path, class_name = models[model_name].rsplit(".", 1)
    module = __import__(module_path, fromlist=[class_name])
    return getattr(module, class_name)(num_classes=num_classes)


# ══════════════════════════════════════════════════════════════════════
# 2. КОНФИГУРАЦИЯ
# ══════════════════════════════════════════════════════════════════════

def get_config() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Модель
    parser.add_argument("--model", type=str, default="simple_cnn",
                        choices=["simple_cnn", "improved_cnn_v2",
                                 "improved_cnn_v3", "improved_cnn_v4"])
    parser.add_argument("--experiment", type=str, default=None)

    # Данные
    parser.add_argument("--data_dir",    type=str, default="data/raw")
    parser.add_argument("--split",       type=str, default="balanced",
                        choices=["balanced", "letters", "digits", "byclass"])
    parser.add_argument("--num_classes", type=int, default=47)
    parser.add_argument("--num_workers", type=int, default=2)

    # ← Аугментация теперь управляется отсюда
    parser.add_argument("--augmentation", type=str, default="light",
                        choices=["none", "light", "strong"],
                        help="Пресет аугментации из emnist_loader.py")

    # Обучение
    parser.add_argument("--epochs",       type=int,   default=15)
    parser.add_argument("--batch_size",   type=int,   default=64)
    parser.add_argument("--lr",           type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=1e-4)

    # Scheduler
    parser.add_argument("--scheduler", type=str, default="step",
                    choices=["step", "cosine", "plateau", "none"])
    parser.add_argument("--step_size", type=int,   default=5)
    parser.add_argument("--gamma",     type=float, default=0.5)

    # Early stopping
    parser.add_argument("--patience", type=int, default=0)

    # Сохранение
    parser.add_argument("--save_every", type=int, default=5)

    # TensorBoard
    parser.add_argument("--no_tb", action="store_true")

    # Воспроизводимость
    parser.add_argument("--seed", type=int, default=42)

    cfg = parser.parse_args()
    if cfg.experiment is None:
        cfg.experiment = cfg.model
    return cfg


# ══════════════════════════════════════════════════════════════════════
# 3. SCHEDULER
# ══════════════════════════════════════════════════════════════════════

def get_scheduler(cfg, optimizer):
    if cfg.scheduler == "step":
        return StepLR(optimizer, step_size=cfg.step_size, gamma=cfg.gamma)
    elif cfg.scheduler == "cosine":
        return CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=1e-6)
    elif cfg.scheduler == "plateau":
        return ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=0.5,
            patience=2,
            min_lr=1e-6,   # ← чтобы lr не ушёл в 0
            )
    return None


# ══════════════════════════════════════════════════════════════════════
# 4. ОДНА ЭПОХА ОБУЧЕНИЯ
# ══════════════════════════════════════════════════════════════════════

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, total_correct, total_samples = 0.0, 0, 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss    += loss.item() * images.size(0)
        total_correct += (outputs.argmax(1) == labels).sum().item()
        total_samples += images.size(0)

    return total_loss / total_samples, 100.0 * total_correct / total_samples


# ══════════════════════════════════════════════════════════════════════
# 5. ОЦЕНКА НА ТЕСТЕ
# ══════════════════════════════════════════════════════════════════════

def evaluate(model, loader, criterion, device, collect_preds=False):
    model.eval()
    total_loss, total_correct, total_samples = 0.0, 0, 0
    all_labels, all_preds, sample_images = [], [], []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            loss    = criterion(outputs, labels)
            preds   = outputs.argmax(1)

            total_loss    += loss.item() * images.size(0)
            total_correct += (preds == labels).sum().item()
            total_samples += images.size(0)

            if collect_preds:
                all_labels.extend(labels.cpu().tolist())
                all_preds.extend(preds.cpu().tolist())
                if len(sample_images) < 10:
                    n = min(10 - len(sample_images), images.size(0))
                    sample_images.extend(images[:n].cpu())

    avg_loss = total_loss / total_samples
    accuracy = 100.0 * total_correct / total_samples

    if collect_preds:
        return avg_loss, accuracy, sample_images, all_labels, all_preds
    return avg_loss, accuracy


# ══════════════════════════════════════════════════════════════════════
# 6. ГЛАВНАЯ ФУНКЦИЯ
# ══════════════════════════════════════════════════════════════════════

def train(cfg: argparse.Namespace):

    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    exp_logger = ExperimentLogger(
        experiment_name=cfg.experiment,
        use_tensorboard=not cfg.no_tb,
    )
    log = exp_logger.logger
    log.info(f"Устройство: {device} | Модель: {cfg.model} | Seed: {cfg.seed}")

    # ── Данные ← теперь из отдельного файла ───────────────────────────
    log.info(f"Загрузка EMNIST (аугментация: {cfg.augmentation})...")
    train_loader, test_loader = get_dataloaders(
        data_dir=cfg.data_dir,
        split=cfg.split,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        augmentation=cfg.augmentation,        # ← передаём пресет
    )
    log.info(f"Train: {len(train_loader.dataset):,} | "
             f"Test:  {len(test_loader.dataset):,} | "
             f"Батч:  {cfg.batch_size}")

    # ── Модель ────────────────────────────────────────────────────────
    model     = get_model(cfg.model, cfg.num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),
                           lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = get_scheduler(cfg, optimizer)

    # ── Инфо о модели ─────────────────────────────────────────────────
    model_params = {
        "model":        cfg.model,
        "optimizer":    "Adam",
        "lr":           cfg.lr,
        "weight_decay": cfg.weight_decay,
        "batch_size":   cfg.batch_size,
        "epochs":       cfg.epochs,
        "scheduler":    cfg.scheduler,
        "augmentation": get_augmentation_info(cfg.augmentation),  # ← описание
        "split":        cfg.split,
        "num_classes":  cfg.num_classes,
        "seed":         cfg.seed,
    }
    sample_input = torch.zeros(1, 1, 28, 28).to(device)
    exp_logger.log_model_info(model, model_params, sample_input)

    # ══════════════════════════════════════════════════════════════════
    # ГЛАВНЫЙ ЦИКЛ
    # ══════════════════════════════════════════════════════════════════
    log.info("\nСТАРТ ОБУЧЕНИЯ\n" + "-" * 60)

    epochs_no_improve = 0

    for epoch in range(1, cfg.epochs + 1):

        with Timer() as t:
            train_loss, train_acc = train_one_epoch(
                model, train_loader, optimizer, criterion, device
            )
            test_loss, test_acc = evaluate(
                model, test_loader, criterion, device
            )

        if scheduler:
            if cfg.scheduler == "plateau":
                scheduler.step(test_acc)
            else:
                scheduler.step()
            exp_logger.log_learning_rate(optimizer, epoch)  

        exp_logger.log_epoch(
            epoch, cfg.epochs,
            train_loss, train_acc,
            test_loss,  test_acc,
            t.duration,
        )

        if epoch % 5 == 0:
            exp_logger.log_gradients(model, epoch)

        is_best = test_acc >= exp_logger.best_acc
        if is_best:
            exp_logger.save_model(model, epoch, is_best=True)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if cfg.save_every > 0 and epoch % cfg.save_every == 0:
            exp_logger.save_model(model, epoch, is_best=False)

        if cfg.patience > 0 and epochs_no_improve >= cfg.patience:
            log.info(f"\nEarly stopping на эпохе {epoch} "
                     f"(нет улучшений {cfg.patience} эпох)")
            break

    # ══════════════════════════════════════════════════════════════════
    # ФИНАЛЬНАЯ ОЦЕНКА
    # ══════════════════════════════════════════════════════════════════
    log.info("\nФИНАЛЬНАЯ ОЦЕНКА...")
    _, _, sample_images, all_labels, all_preds = evaluate(
        model, test_loader, criterion, device, collect_preds=True
    )

    cm = confusion_matrix(all_labels, all_preds)
    exp_logger.log_confusion_matrix(cm)
    exp_logger.log_example_predictions(
        images=sample_images,
        labels=all_labels[:10],
        predictions=all_preds[:10],
    )
    exp_logger.log_final_summary()
    exp_logger.close()


# ══════════════════════════════════════════════════════════════════════
# ТОЧКА ВХОДА
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    cfg = get_config()
    train(cfg)