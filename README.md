# interpretable-cnn
CNN classifier on CIFAR-10 with Grad-CAM interpretability

# 🧠 Interpretable CNN with Grad-CAM on CIFAR-10

A deep learning project demonstrating **explainable image classification** using a custom-built **Convolutional Neural Network (CNN)** in **PyTorch**, trained on the **CIFAR-10 dataset**, and enhanced with **Grad-CAM visualizations** to provide insights into the model's decision-making process.

---

## 🚀 Project Highlights

- ✅ End-to-end CNN built from scratch using PyTorch  
- ✅ Trained on CIFAR-10 (10-class natural image dataset)  
- ✅ Grad-CAM integration for pixel-level interpretability  
- ✅ Visual outputs showing which regions the model "attends to"

---

## 🧠 Why Interpretability Matters

Modern CNNs often act like "black boxes" — even when they perform well, we don’t know *why* they predict what they predict.

This project integrates **Grad-CAM (Gradient-weighted Class Activation Mapping)** to:
- Visualize what part of the image influenced the model’s prediction  
- Debug and validate model behavior  
- Improve **trust and accountability** of the model  

---

## 📦 Dataset

- Dataset: [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)  
- 60,000 32x32 color images in 10 classes (cats, dogs, planes, ships, etc.)  
- Automatically downloaded using:
```python
torchvision.datasets.CIFAR10(download=True)
