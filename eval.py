import torch
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
import cv2

from models.resnet_custom import get_resnet18
from utils.gradcam import GradCAM
from utils.data_loader import get_cifar10_loaders
from sklearn.metrics import accuracy_score, f1_score

def evaluate_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = get_resnet18(num_classes=10)
    model.load_state_dict(torch.load("model.pth", map_location=device))
    model.to(device)
    model.eval()

    _, testloader, class_names = get_cifar10_loaders(batch_size=64)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in testloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')

    print(f"\n Evaluation Metrics:")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score (macro): {f1:.4f}")


def apply_gradcam():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = get_resnet18(num_classes=10)
    model.load_state_dict(torch.load("model.pth", map_location=device))
    model.to(device)

    # Get a sample batch
    _, testloader, class_names = get_cifar10_loaders(batch_size=1)
    images, labels = next(iter(testloader))

    img = images[0].unsqueeze(0).to(device)

    # Hook the final conv layer
    target_layer = model.layer4[-1]

    gradcam = GradCAM(model, target_layer)
    cam = gradcam.generate(img)

    # Prepare original image
    original_img = images[0].permute(1, 2, 0).cpu().numpy()
    original_img = (original_img * 0.5 + 0.5) * 255  # unnormalize
    original_img = np.uint8(original_img)

    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    overlayed_img = heatmap + np.float32(original_img) / 255
    overlayed_img = overlayed_img / np.max(overlayed_img)

    # Save to output
    os.makedirs("output/gradcam_vis", exist_ok=True)
    output_path = f"output/gradcam_vis/cam_class_{class_names[labels[0]]}.jpg"
    plt.imsave(output_path, overlayed_img)
    print(f"Grad-CAM saved at: {output_path}")

if __name__ == "__main__":
    apply_gradcam()
    evaluate_model()
