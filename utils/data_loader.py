import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import random
import os

def get_cifar10_loaders(batch_size=16, subset_size=1000):
    # Dynamically get root path (works from any script)
    ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    DATA_DIR = os.path.join(ROOT, "data")

    # Image preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5),
                             (0.5, 0.5, 0.5))
    ])

    # Load full dataset
    full_trainset = torchvision.datasets.CIFAR10(root=DATA_DIR, train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root=DATA_DIR, train=False, download=True, transform=transform)

    # Use a small random subset (quick testing)
    indices = random.sample(range(len(full_trainset)), subset_size)
    trainset = Subset(full_trainset, indices)

    # Data loaders
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)

    return trainloader, testloader, full_trainset.classes


if __name__ == "__main__":
    trainloader, testloader, class_names = get_cifar10_loaders()
    print("Classes:", class_names)
    dataiter = iter(trainloader)
    images, labels = next(dataiter)
    print(f"Batch size: {images.shape}")  # Should be [64, 3, 224, 224]
