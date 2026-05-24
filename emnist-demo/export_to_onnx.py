"""
Экспорт обученной модели improved_cnn_v5 в формат ONNX.
Запускается однократно в среде с PyTorch и обученными весами.
"""

import torch
import sys
import os

# Добавляем путь к исходному коду проекта
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models.improved_cnn_v5 import ImprovedCNN_v5

def export_model(weights_path='best_model.pth', output_path='model_v5.onnx'):
    """
    Конвертирует обученную PyTorch-модель в ONNX формат.
    
    Args:
        weights_path: путь к файлу с весами обученной модели
        output_path: путь для сохранения ONNX-файла
    """
    
    print("Загрузка модели improved_cnn_v5...")
    model = ImprovedCNN_v5(num_classes=47)
    
    # Загружаем обученные веса
    try:
        # Безопасная загрузка с обработкой чекпоинтов
        try:
            checkpoint = torch.load(weights_path, map_location='cpu', weights_only=True)
        except TypeError:
            # Для старых версий PyTorch, где нет параметра weights_only
            checkpoint = torch.load(weights_path, map_location='cpu')
        
        # Проверяем, является ли загруженный объект чекпоинтом обучения
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            # Извлекаем веса из чекпоинта
            state_dict = checkpoint['model_state_dict']
            epoch = checkpoint.get('epoch', 'неизвестно')
            best_acc = checkpoint.get('best_acc', 'неизвестно')
            print(f"✓ Обнаружен чекпоинт обучения:")
            print(f"  - Эпоха: {epoch}")
            print(f"  - Лучшая точность: {best_acc}")
            if 'history' in checkpoint:
                print(f"  - История обучения: сохранена")
        else:
            # Загружен чистый state_dict
            state_dict = checkpoint
            print("✓ Загружен чистый словарь весов модели")
        
        # Загружаем веса в модель
        model.load_state_dict(state_dict)
        print(f"✓ Веса успешно загружены из {weights_path}")
        
    except Exception as e:
        print(f"✗ Ошибка загрузки весов: {e}")
        print("  Убедитесь, что файл best_model.pth находится в текущей директории")
        return False
    
    model.eval()
    
    # Создаем фиктивный входной тензор для трассировки
    dummy_input = torch.randn(1, 1, 28, 28)
    
    print(f"Экспорт модели в ONNX (opset_version=18)...")
    
    try:
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            input_names=['input'],
            output_names=['output'],
            opset_version=18,  # КРИТИЧНО: должен быть ≥ 18 из-за операции mean() в SE-блоке
            dynamic_axes=None,  # Фиксированный размер входа 28×28
            do_constant_folding=True,  # Оптимизация: предвычисление констант
        )
        
        # Проверяем размер полученного файла
        file_size = os.path.getsize(output_path)
        print(f"✓ Модель экспортирована в {output_path}")
        print(f"  Размер файла: {file_size:,} байт (~{file_size/1024:.1f} КБ)")
        
        # Верификация: сравнение выходов PyTorch и ONNX
        verify_export(model, output_path, dummy_input)
        
        return True
        
    except Exception as e:
        print(f"✗ Ошибка экспорта: {e}")
        print("  Проверьте версию PyTorch и onnx: pip install --upgrade torch onnx")
        return False


def verify_export(pytorch_model, onnx_path, sample_input):
    """
    Сравнивает выходы PyTorch-модели и ONNX-модели.
    """
    import onnx
    import onnxruntime
    import numpy as np
    
    print("\nВерификация экспорта...")
    
    # Получаем выход PyTorch-модели
    with torch.no_grad():
        pytorch_output = pytorch_model(sample_input).numpy()
    
    # Загружаем и запускаем ONNX-модель
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    
    ort_session = onnxruntime.InferenceSession(onnx_path)
    ort_inputs = {ort_session.get_inputs()[0].name: sample_input.numpy()}
    ort_output = ort_session.run(None, ort_inputs)[0]
    
    # Вычисляем расхождение
    max_diff = np.max(np.abs(pytorch_output - ort_output))
    print(f"  Максимальное расхождение: {max_diff:.2e}")
    
    if max_diff < 1e-6:
        print("✓ Экспорт корректен: расхождение незначительное (< 1e-6)")
    else:
        print("⚠ Расхождение заметное, но может быть допустимым")
        print(f"  {max_diff:.2e}")
    
    return max_diff


if __name__ == "__main__":
    print("=" * 60)
    print("ЭКСПОРТ МОДЕЛИ improved_cnn_v5 → ONNX")
    print("=" * 60)
    
    success = export_model('best_model.pth', 'model_v5.onnx')
    
    if success:
        print("\n" + "=" * 60)
        print("✓ ГОТОВО! Файл model_v5.onnx создан.")
        print("  Скопируйте его в директорию веб-приложения emnist-demo/")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("✗ Экспорт не удался. Проверьте ошибки выше.")
        print("=" * 60)