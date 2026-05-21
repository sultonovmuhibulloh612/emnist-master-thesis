import os
import sys
from torchvision import datasets, transforms

root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "raw")
print(f"Путь: {os.path.abspath(root)}")
print(f"Папка существует: {os.path.exists(root)}")

os.makedirs(root, exist_ok=True)

try:
    print("Загрузка train...")
    train = datasets.EMNIST(
        root=root,
        split="balanced",
        train=True,
        download=True,
        transform=transforms.ToTensor()
    )
    print(f"Train OK: {len(train):,} примеров")
except Exception as e:
    print(f"ОШИБКА train: {e}", file=sys.stderr)

try:
    print("Загрузка test...")
    test = datasets.EMNIST(
        root=root,
        split="balanced",
        train=False,
        download=True,
        transform=transforms.ToTensor()
    )
    print(f"Test OK: {len(test):,} примеров")
except Exception as e:
    print(f"ОШИБКА test: {e}", file=sys.stderr)

print("Готово.")