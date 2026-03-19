from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_dataloaders(batch_size=64):

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    train_dataset = datasets.EMNIST(
        root="data/raw",
        split="balanced",
        train=True,
        transform=transform
    )

    test_dataset = datasets.EMNIST(
        root="data/raw",
        split="balanced",
        train=False,
        transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, test_loader