import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

def train_model(model, trainloader, device, epochs=1):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model.to(device)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        loop = tqdm(trainloader, desc=f"Epoch {epoch+1}/{epochs}")

        for inputs, labels in loop:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            loop.set_postfix(loss=loss.item(), acc=100.*correct/total)

        print(f"Epoch {epoch+1}: Loss={running_loss/len(trainloader):.4f}, Accuracy={100.*correct/total:.2f}%")

    return model
