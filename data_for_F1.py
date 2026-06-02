import torch, json
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from src.models.improved_cnn_v5 import ImprovedCNN_v5
from src.dataset.emnist_loader import get_dataloaders

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Загрузка модели
model = ImprovedCNN_v5(num_classes=47).to(device)

# === ИСПРАВЛЕНИЕ: извлекаем model_state_dict ===
checkpoint = torch.load("results/20260503_141958_v5_light_cosine/models/best_model.pth",
                        map_location=device)
state_dict = checkpoint.get("model_state_dict", checkpoint)  # безопасное извлечение
model.load_state_dict(state_dict)
# ==============================================

model.eval()

# Тестовые данные
_, test_loader = get_dataloaders(data_dir="data/raw/EMNIST/raw", split="balanced",
                                 batch_size=256, num_workers=2, augmentation="none")

# Инференс
all_labels, all_preds = [], []
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        preds = model(images).argmax(1).cpu()
        all_labels.extend(labels.tolist())
        all_preds.extend(preds.tolist())

# Сохранение результатов
classes = list('0123456789') + list('ABCDEFGHIJKLMNOPQRSTUVWXYZ') + ['a','b','d','e','f','g','h','n','q','r','t']
report = classification_report(all_labels, all_preds, target_names=classes, output_dict=True, digits=4)
json.dump(report, open("classification_report.json", "w"), indent=2, ensure_ascii=False)

cm = confusion_matrix(all_labels, all_preds)
np.save("confusion_matrix.npy", cm)

print("✓ Сохранено: classification_report.json и confusion_matrix.npy")
print(f"Overall accuracy: {report['accuracy']*100:.2f}%")