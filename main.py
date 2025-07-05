import torch
from models.resnet_custom import get_resnet18
from utils.data_loader import get_cifar10_loaders
from train import train_model

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Load smaller dataset for fast testing
    trainloader, testloader, class_names = get_cifar10_loaders(batch_size=16, subset_size=1000)
    model = get_resnet18(num_classes=len(class_names))

    # Train for 1 epoch only
    trained_model = train_model(model, trainloader, device, epochs=1)

    # Save model
    torch.save(trained_model.state_dict(), "model.pth")
    print("Model saved as model.pth")

if __name__ == "__main__":
    main()
