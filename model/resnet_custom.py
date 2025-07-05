import torch.nn as nn
from torchvision import models

def get_resnet18(num_classes=10):
    model = models.resnet18(pretrained=True)
    # Replace the final fully connected layer
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
