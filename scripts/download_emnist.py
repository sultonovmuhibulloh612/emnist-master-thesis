from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.ToTensor()
])

train_dataset = datasets.EMNIST(
    root="data/raw",
    split="balanced",
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.EMNIST(
    root="data/raw",
    split="balanced",
    train=False,
    download=True,
    transform=transform
)

print("Dataset downloaded")