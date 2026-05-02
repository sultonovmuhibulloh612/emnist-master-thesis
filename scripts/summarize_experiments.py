#!/usr/bin/env python3
"""
Скрипт для сбора и группировки результатов экспериментов из training.log файлов.
Запускать из папки scripts/ или из корня проекта.
"""

import re
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import json


def get_project_root() -> Path:
    """Определяет корень проекта (на уровень выше scripts/)."""
    script_dir = Path(__file__).resolve().parent
    
    # Если скрипт в scripts/, то корень на уровень выше
    if script_dir.name == 'scripts':
        return script_dir.parent
    # Если скрипт в корне, то это и есть корень
    else:
        return script_dir


def parse_training_log(log_path: Path) -> dict:
    """Парсит один training.log и возвращает словарь с результатами."""
    
    if not log_path.exists():
        return None
    
    with open(log_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Извлекаем информацию
    info = {
        'experiment_name': None,
        'results_folder': None,
        'start_time': None,
        'device': None,
        'model': None,
        'architecture': None,
        'total_params': None,
        'trainable_params': None,
        'optimizer': None,
        'learning_rate': None,
        'weight_decay': None,
        'batch_size': None,
        'epochs': None,
        'scheduler': None,
        'augmentation': None,
        'num_classes': None,
        'seed': None,
        'best_epoch': None,
        'best_accuracy': None,
        'avg_accuracy': None,
        'avg_epoch_time': None,
        'total_time': None,
        'path': str(log_path)
    }
    
    # Начало эксперимента
    match = re.search(r'НАЧАЛО ЭКСПЕРИМЕНТА: (.+)', content)
    if match:
        info['experiment_name'] = match.group(1)
    
    match = re.search(r'Папка результатов: (.+)', content)
    if match:
        info['results_folder'] = match.group(1)
    
    match = re.search(r'Время старта: (.+)', content)
    if match:
        info['start_time'] = match.group(1)
    
    # Устройство и модель
    match = re.search(r'Устройство: (\w+) \| Модель: (\w+)', content)
    if match:
        info['device'] = match.group(1)
        info['model'] = match.group(2)
    
    # Архитектура
    match = re.search(r'Архитектура: (.+)', content)
    if match:
        info['architecture'] = match.group(1)
    
    # Параметры
    match = re.search(r'Всего параметров: ([\d,]+)', content)
    if match:
        info['total_params'] = match.group(1)
    
    match = re.search(r'Обучаемых параметров: ([\d,]+)', content)
    if match:
        info['trainable_params'] = match.group(1)
    
    # Гиперпараметры
    match = re.search(r'optimizer: (.+)', content)
    if match:
        info['optimizer'] = match.group(1)
    
    match = re.search(r'lr: ([\d.]+)', content)
    if match:
        info['learning_rate'] = match.group(1)
    
    match = re.search(r'weight_decay: ([\d.]+)', content)
    if match:
        info['weight_decay'] = match.group(1)
    
    match = re.search(r'batch_size: (\d+)', content)
    if match:
        info['batch_size'] = match.group(1)
    
    match = re.search(r'epochs: (\d+)', content)
    if match:
        info['epochs'] = match.group(1)
    
    match = re.search(r'scheduler: (.+)', content)
    if match:
        info['scheduler'] = match.group(1)
    
    match = re.search(r'augmentation: (.+)', content)
    if match:
        info['augmentation'] = match.group(1)
    
    match = re.search(r'num_classes: (\d+)', content)
    if match:
        info['num_classes'] = match.group(1)
    
    match = re.search(r'seed: (\d+)', content)
    if match:
        info['seed'] = match.group(1)
    
    # Итоги
    match = re.search(r'Лучшая эпоха:\s+(\d+)', content)
    if match:
        info['best_epoch'] = int(match.group(1))
    
    match = re.search(r'Лучшая точность:\s+([\d.]+)%', content)
    if match:
        info['best_accuracy'] = float(match.group(1))
    
    match = re.search(r'Средняя точность:\s+([\d.]+)%', content)
    if match:
        info['avg_accuracy'] = float(match.group(1))
    
    match = re.search(r'Среднее время эпохи:\s+([\d.]+)с', content)
    if match:
        info['avg_epoch_time'] = float(match.group(1))
    
    match = re.search(r'Общее время:\s+([\d.]+)с', content)
    if match:
        info['total_time'] = float(match.group(1))
    
    return info

def find_all_experiments(results_dir: Path) -> list:
    """Находит все training.log в директории results и парсит их."""
    
    if not results_dir.exists():
        print(f"Директория {results_dir} не найдена!")
        return []
    
    experiments = []
    
    for log_file in results_dir.glob("*/logs/training.log"):
        print(f"Обработка: {log_file}")
        info = parse_training_log(log_file)
        if info:
            experiments.append(info)
    
    return experiments


def group_experiments(experiments: list) -> dict:
    """Группирует эксперименты по разным критериям."""
    
    groups = {
        'by_model': defaultdict(list),
        'by_augmentation': defaultdict(list),
        'by_optimizer': defaultdict(list),
        'by_device': defaultdict(list),
        'best_overall': None,
        'best_by_model': {},
    }
    
    for exp in experiments:
        # Группировка по модели
        model = exp.get('model', 'unknown')
        groups['by_model'][model].append(exp)
        
        # Группировка по аугментации
        aug = exp.get('augmentation', 'unknown')
        groups['by_augmentation'][aug].append(exp)
        
        # Группировка по оптимизатору
        opt = exp.get('optimizer', 'unknown')
        groups['by_optimizer'][opt].append(exp)
        
        # Группировка по устройству
        device = exp.get('device', 'unknown')
        groups['by_device'][device].append(exp)
        
        # Лучший в каждой модели
        if exp.get('best_accuracy'):
            if model not in groups['best_by_model'] or \
               exp['best_accuracy'] > groups['best_by_model'][model]['best_accuracy']:
                groups['best_by_model'][model] = exp
    
    # Лучший общий
    best = None
    for exp in experiments:
        if exp.get('best_accuracy'):
            if best is None or exp['best_accuracy'] > best['best_accuracy']:
                best = exp
    groups['best_overall'] = best
    
    return groups


def print_summary_table(experiments: list, sort_by: str = 'best_accuracy'):
    """Выводит сводную таблицу экспериментов."""
    
    # Сортируем
    sorted_exps = sorted(experiments, 
                        key=lambda x: x.get(sort_by, 0) if x.get(sort_by) else 0, 
                        reverse=True)
    
    print("\n" + "="*120)
    print(f"{'ID':<4} {'Эксперимент':<35} {'Модель':<15} {'Лучшая':<8} {'Средняя':<8} {'Время':<10} {'Аугментация':<25}")
    print("-"*120)
    
    for i, exp in enumerate(sorted_exps, 1):
        name = exp.get('experiment_name', 'N/A') or 'N/A'
        name = name[:33]
        
        model = exp.get('model', 'N/A') or 'N/A'
        model = model[:13]
        
        best = f"{exp.get('best_accuracy', 0):.2f}%" if exp.get('best_accuracy') else 'N/A'
        avg = f"{exp.get('avg_accuracy', 0):.2f}%" if exp.get('avg_accuracy') else 'N/A'
        time = f"{exp.get('total_time', 0):.1f}с" if exp.get('total_time') else 'N/A'
        
        aug = exp.get('augmentation', 'N/A')
        aug = aug[:23] if aug else 'N/A'
        
        print(f"{i:<4} {name:<35} {model:<15} {best:<8} {avg:<8} {time:<10} {aug:<25}")
    
    print("="*120)

def export_to_json(experiments: list, groups: dict, filename: str = "experiments_summary.json"):
    """Экспортирует результаты в JSON."""
    
    data = {
        'total_experiments': len(experiments),
        'experiments': experiments,
        'summary': {
            'best_overall': groups['best_overall']['experiment_name'] if groups['best_overall'] else None,
            'best_accuracy': groups['best_overall']['best_accuracy'] if groups['best_overall'] else None,
            'models_count': {k: len(v) for k, v in groups['by_model'].items()},
            'best_by_model': {k: v['best_accuracy'] for k, v in groups['best_by_model'].items()}
        }
    }
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"\nРезультаты экспортированы в {filename}")


def main():
    """Основная функция."""
    
    # Определяем корень проекта
    project_root = get_project_root()
    results_dir = project_root / 'results'
    
    print(f"Корень проекта: {project_root}")
    print(f"Поиск логов в: {results_dir}")
    
    # Находим все эксперименты
    experiments = find_all_experiments(results_dir)
    
    if not experiments:
        print("Эксперименты не найдены!")
        return
    
    print(f"\nНайдено экспериментов: {len(experiments)}")
    
    # Группируем
    groups = group_experiments(experiments)
    
    # Выводим сводную таблицу
    print_summary_table(experiments)
    
    # Статистика по группам
    print("\n" + "="*80)
    print("СТАТИСТИКА ПО ГРУППАМ")
    print("="*80)
    
    print("\nПо моделям:")
    for model, exps in groups['by_model'].items():
        accs = [e['best_accuracy'] for e in exps if e.get('best_accuracy') is not None]
        if accs:
            best_acc = max(accs)
            avg_acc = sum(accs) / len(accs)
            print(f"  {model}: {len(exps)} эксп. | Лучшая: {best_acc:.2f}% | Средняя: {avg_acc:.2f}%")
        else:
            print(f"  {model}: {len(exps)} эксп. | Нет данных о точности")
    
    print("\nПо аугментации:")
    for aug, exps in groups['by_augmentation'].items():
        accs = [e['best_accuracy'] for e in exps if e.get('best_accuracy') is not None]
        if accs:
            best_acc = max(accs)
            print(f"  {aug}: {len(exps)} эксп. | Лучшая: {best_acc:.2f}%")
        else:
            print(f"  {aug}: {len(exps)} эксп. | Нет данных о точности")
    # Лучший эксперимент
    if groups['best_overall']:
        print("\n" + "="*80)
        print("ЛУЧШИЙ ЭКСПЕРИМЕНТ:")
        print("="*80)
        best = groups['best_overall']
        print(f"  Название: {best['experiment_name']}")
        print(f"  Точность: {best['best_accuracy']}%")
        print(f"  Модель: {best['model']} ({best['architecture']})")
        print(f"  Параметры: {best['total_params']}")
        print(f"  Аугментация: {best['augmentation']}")
        print(f"  Время: {best['total_time']:.1f}с")
    
    # Экспорт в JSON (сохраняем тоже в корень проекта)
    json_path = project_root / "experiments_summary.json"
    export_to_json(experiments, groups, str(json_path))


if __name__ == "__main__":
    main()