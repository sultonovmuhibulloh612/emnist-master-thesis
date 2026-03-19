"""
Модуль логирования для дипломной работы.
Содержит инструменты для отслеживания экспериментов и сохранения результатов.
Поддерживает: файловые логи, CSV, JSON, TensorBoard.
"""

import logging
import csv
import json
import time
import torch
import matplotlib.pyplot as plt
import seaborn as sns

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# TensorBoard — необязательная зависимость
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False


class ExperimentLogger:
    """
    Профессиональный логгер для дипломной работы.
    Сохраняет всё: метрики, параметры, графики, лучшие модели, TensorBoard.

    Запуск TensorBoard:
        tensorboard --logdir results/
    """

    def __init__(
        self,
        experiment_name: str,
        base_dir: str = "results",
        use_tensorboard: bool = True,       # ← включить/выключить TB
    ):
        """
        Args:
            experiment_name: Название эксперимента (baseline, dropout, …)
            base_dir:        Базовая папка для всех результатов
            use_tensorboard: Записывать ли события в TensorBoard
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_name = experiment_name
        self.exp_dir = Path(base_dir) / f"{timestamp}_{experiment_name}"
        self.exp_dir.mkdir(parents=True, exist_ok=True)

        # Подпапки
        self.models_dir  = self.exp_dir / "models"
        self.logs_dir    = self.exp_dir / "logs"
        self.metrics_dir = self.exp_dir / "metrics"
        self.tb_dir      = self.exp_dir / "tensorboard"   # ← папка для TB-событий

        self.models_dir.mkdir(exist_ok=True)
        self.logs_dir.mkdir(exist_ok=True)
        self.metrics_dir.mkdir(exist_ok=True)
        self.tb_dir.mkdir(exist_ok=True)

        # CSV открывается один раз
        self._csv_file   = open(self.metrics_dir / "training_metrics.csv",
                                "w", newline="", encoding="utf-8")
        self._csv_writer = csv.writer(self._csv_file)
        self._csv_writer.writerow(
            ["epoch", "train_loss", "train_acc", "test_loss", "test_acc", "time"]
        )

        # ── TensorBoard Writer ──────────────────────────────────────────
        self.tb: Optional["SummaryWriter"] = None
        if use_tensorboard:
            if TENSORBOARD_AVAILABLE:
                self.tb = SummaryWriter(log_dir=str(self.tb_dir))
            else:
                print("[Logger] tensorboard не установлен → pip install tensorboard")
        # ───────────────────────────────────────────────────────────────

        self.logger = self._setup_logger()

        self.history = {
            "train_loss":  [],
            "train_acc":   [],
            "test_loss":   [],
            "test_acc":    [],
            "epoch_times": [],
        }

        self.best_acc   = 0.0
        self.best_epoch = 0
        self.model_params = {}

        self.logger.info("=" * 60)
        self.logger.info(f"НАЧАЛО ЭКСПЕРИМЕНТА: {experiment_name}")
        self.logger.info(f"Папка результатов: {self.exp_dir}")
        self.logger.info(f"Время старта: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        if self.tb:
            self.logger.info(f"TensorBoard: tensorboard --logdir {self.exp_dir.parent}")
        self.logger.info("=" * 60)

    # ──────────────────────────────────────────────────────────────────
    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger(f"exp_{self.exp_dir.name}")
        logger.setLevel(logging.INFO)

        if logger.handlers:
            logger.handlers.clear()

        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        fh = logging.FileHandler(self.logs_dir / "training.log", encoding="utf-8")
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        mh = logging.FileHandler(self.logs_dir / "metrics.log", encoding="utf-8")
        mh.setLevel(logging.WARNING)
        mh.setFormatter(formatter)
        logger.addHandler(mh)

        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        logger.propagate = False
        return logger

    # ──────────────────────────────────────────────────────────────────
    def log_model_info(
        self,
        model: torch.nn.Module,
        model_params: Optional[Dict] = None,
        sample_input: Optional[torch.Tensor] = None,    # ← для графа в TB
    ):
        """
        Сохраняет информацию о модели.

        Args:
            sample_input: тензор-пример, чтобы TensorBoard нарисовал граф.
                          Пример: torch.zeros(1, 1, 28, 28)
        """
        total_params     = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        self.logger.info("\nИНФОРМАЦИЯ О МОДЕЛИ:")
        self.logger.info(f"   Архитектура: {model.__class__.__name__}")
        self.logger.info(f"   Всего параметров: {total_params:,}")
        self.logger.info(f"   Обучаемых параметров: {trainable_params:,}")

        if model_params:
            self.logger.info("\nПАРАМЕТРЫ ОБУЧЕНИЯ:")
            for k, v in model_params.items():
                self.logger.info(f"   {k}: {v}")

        # ── TB: граф модели ───────────────────────────────────────────
        self.model_params = model_params or {}
        if self.tb and sample_input is not None:
            try:
                self.tb.add_graph(model, sample_input)
                self.logger.info("   TB: граф модели записан")
            except Exception as e:
                self.logger.warning(f"   TB граф пропущен: {e}")
        # ─────────────────────────────────────────────────────────────

        config = {
            "model_name":           model.__class__.__name__,
            "total_parameters":     total_params,
            "trainable_parameters": trainable_params,
            "model_params":         model_params or {},
            "timestamp":            datetime.now().isoformat(),
        }
        with open(self.exp_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

    # ──────────────────────────────────────────────────────────────────
    def log_epoch(
        self,
        epoch:      int,
        num_epochs: int,
        train_loss: float,
        train_acc:  float,
        test_loss:  float,
        test_acc:   float,
        epoch_time: float,
    ):
        """Логирует результаты одной эпохи (файл + CSV + TensorBoard)."""
        self.history["train_loss"].append(train_loss)
        self.history["train_acc"].append(train_acc)
        self.history["test_loss"].append(test_loss)
        self.history["test_acc"].append(test_acc)
        self.history["epoch_times"].append(epoch_time)

        is_best = test_acc > self.best_acc
        if is_best:
            self.best_acc   = test_acc
            self.best_epoch = epoch

        suffix = " >>> НОВЫЙ РЕКОРД!" if is_best else ""
        
        msg = (
            f"[{epoch}/{num_epochs}] "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
            f"Test Loss: {test_loss:.4f}  | Test Acc: {test_acc:.2f}% | "
            f"Время: {epoch_time:.2f}с{suffix}"
        )
        
        self.logger.info(msg)     
        self.logger.warning(msg)  

        # CSV
        self._csv_writer.writerow(
            [epoch, train_loss, train_acc, test_loss, test_acc, epoch_time]
        )
        self._csv_file.flush()

        # ── TensorBoard ───────────────────────────────────────────────
        if self.tb:
            # Loss: Train vs Test на одном графике
            self.tb.add_scalars(
                "Loss",
                {"train": train_loss, "test": test_loss},
                epoch,
            )
            # Accuracy: Train vs Test на одном графике
            self.tb.add_scalars(
                "Accuracy",
                {"train": train_acc, "test": test_acc},
                epoch,
            )
            # Время эпохи
            self.tb.add_scalar("Epoch_Time_sec", epoch_time, epoch)
        # ─────────────────────────────────────────────────────────────

    # ──────────────────────────────────────────────────────────────────
    def log_learning_rate(self, optimizer: torch.optim.Optimizer, epoch: int):
        """
        Записывает текущий learning rate в TensorBoard.
        Вызывать после каждого шага scheduler'а.

        Пример:
            scheduler.step()
            exp_logger.log_learning_rate(optimizer, epoch)
        """
        if self.tb:
            for i, pg in enumerate(optimizer.param_groups):
                self.tb.add_scalar(f"LR/group_{i}", pg["lr"], epoch)

    # ──────────────────────────────────────────────────────────────────
    def log_gradients(self, model: torch.nn.Module, epoch: int):
        """
        Записывает гистограммы градиентов и весов в TensorBoard.
        Вызывать ПОСЛЕ loss.backward(), ДО optimizer.step().

        Позволяет обнаружить затухающие / взрывные градиенты.

        Пример:
            loss.backward()
            exp_logger.log_gradients(model, epoch)
            optimizer.step()
        """
        if self.tb:
            for name, param in model.named_parameters():
                if param.grad is not None:
                    self.tb.add_histogram(f"Gradients/{name}", param.grad, epoch)
                    self.tb.add_histogram(f"Weights/{name}",   param.data,  epoch)

    # ──────────────────────────────────────────────────────────────────
    def save_model(self, model: torch.nn.Module, epoch: int, is_best: bool = False):
        """Сохраняет веса модели."""
        path = (
            self.models_dir / "best_model.pth"
            if is_best
            else self.models_dir / f"model_epoch_{epoch}.pth"
        )
        torch.save(
            {
                "epoch":            epoch,
                "model_state_dict": model.state_dict(),
                "best_acc":         self.best_acc,
                "history":          self.history,
            },
            path,
        )
        if is_best:
            self.logger.info(
                f"Сохранена лучшая модель (эпоха {epoch}, точность: {self.best_acc:.2f}%)"
            )

    # ──────────────────────────────────────────────────────────────────
    def log_confusion_matrix(self, cm, class_names: Optional[List[str]] = None):
        """Сохраняет матрицу ошибок в PNG и в TensorBoard."""
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(cm, annot=False, fmt="d", cmap="Blues", ax=ax)
        ax.set_title(f"Confusion Matrix — {self.experiment_name}")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

        if class_names and len(class_names) <= 47:
            ax.set_xticklabels(class_names, rotation=90)
            ax.set_yticklabels(class_names, rotation=0)

        fig.tight_layout()
        out = self.metrics_dir / "confusion_matrix.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")

        # ── TB: матрица как Figure ────────────────────────────────────
        if self.tb:
            self.tb.add_figure("Confusion_Matrix", fig)
        # ─────────────────────────────────────────────────────────────

        plt.close(fig)
        self.logger.info(f"Матрица ошибок сохранена: {out}")

    # ──────────────────────────────────────────────────────────────────
    def log_example_predictions(
        self, images, labels, predictions, num_examples: int = 10
    ):
        """Сохраняет примеры предсказаний в PNG и в TensorBoard."""
        n = min(num_examples, len(images))
        fig, axes = plt.subplots(2, 5, figsize=(15, 6))
        axes = axes.ravel()

        for i in range(n):
            img = images[i].cpu().numpy().squeeze()
            axes[i].imshow(img, cmap="gray")
            color = "green" if predictions[i] == labels[i] else "red"
            axes[i].set_title(f"True: {labels[i]}\nPred: {predictions[i]}", color=color)
            axes[i].axis("off")

        fig.tight_layout()
        out = self.metrics_dir / "example_predictions.png"
        fig.savefig(out, dpi=150)

        # ── TB: примеры как Figure ────────────────────────────────────
        if self.tb:
            self.tb.add_figure("Example_Predictions", fig)
        # ─────────────────────────────────────────────────────────────

        plt.close(fig)
        correct = sum(1 for i in range(n) if predictions[i] == labels[i])
        self.logger.info(f"Примеры сохранены. Точность на примерах: {correct}/{n}")

    # ──────────────────────────────────────────────────────────────────
    def log_final_summary(self):
        """Итоговый отчёт по эксперименту."""
        acc_list  = self.history["test_acc"]
        time_list = self.history["epoch_times"]

        avg_acc        = sum(acc_list)  / len(acc_list)
        avg_epoch_time = sum(time_list) / len(time_list)
        total_time     = sum(time_list)

        self.logger.info("\n" + "=" * 60)
        self.logger.info("ИТОГИ ЭКСПЕРИМЕНТА")
        self.logger.info("=" * 60)
        self.logger.info(f"   Лучшая эпоха:        {self.best_epoch}")
        self.logger.info(f"   Лучшая точность:     {self.best_acc:.2f}%")
        self.logger.info(f"   Средняя точность:    {avg_acc:.2f}%")
        self.logger.info(f"   Среднее время эпохи: {avg_epoch_time:.2f}с")
        self.logger.info(f"   Общее время:         {total_time:.2f}с ({total_time/60:.2f} мин)")
        self.logger.info(f"   Результаты в:        {self.exp_dir}")
        if self.tb:
            self.logger.info(
                f"   TensorBoard:         tensorboard --logdir {self.exp_dir.parent}"
            )
        self.logger.info("=" * 60)

        # ── TB: итоговые гиперпараметры + метрика ────────────────────

        if self.model_params:
            from torch.utils.tensorboard import SummaryWriter
            hparams_writer = SummaryWriter(
                log_dir=str(self.exp_dir.parent / "hparams")
            )
            hparams_writer.add_hparams(
                hparam_dict={k: str(v) for k, v in self.model_params.items()},
                metric_dict={
                    "best_acc":   float(self.best_acc),
                    "avg_acc":    float(avg_acc),
                    "best_epoch": float(self.best_epoch),
                },
            )
            hparams_writer.flush()
            hparams_writer.close()




    # ──────────────────────────────────────────────────────────────────
    def close(self):
        """Закрывает все открытые ресурсы. Вызывать в конце обучения."""
        self._csv_file.close()
        if self.tb:
            self.tb.close()
            self.logger.info("TensorBoard writer закрыт.")


# ======================================================================
class Timer:
    """Контекстный менеджер для замера времени эпох."""

    def __enter__(self):
        self._start = time.time()
        return self

    def __exit__(self, *args):
        self.duration = time.time() - self._start

    def get_elapsed(self) -> float:
        """Прошедшее время с момента старта (сек)."""
        return time.time() - self._start