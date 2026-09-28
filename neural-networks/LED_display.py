import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Subset
import numpy as np
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

# Function to load and prepare the dataset
def load_data(filepath):
    data = np.loadtxt(filepath, delimiter=',', dtype=np.float32)
    features = torch.from_numpy(data[:, :-1])
    targets = torch.from_numpy(data[:, -1]).long()
    return features, targets

# Define the neural network model
class LEDNet(nn.Module):
    def __init__(self):
        super(LEDNet, self).__init__()
        self.fc1 = nn.Linear(7, 16)
        self.fc2 = nn.Linear(16, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Training and evaluation function
def train_and_evaluate(model, train_loader, val_loader, num_epochs=500):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    train_losses, val_losses, val_accuracies = [], [], []

    for epoch in range(num_epochs):
        model.train()
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss, correct, total = 0, 0, 0
            for inputs, labels in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_loss /= total
        accuracy = correct / total

        train_losses.append(loss.item())
        val_losses.append(val_loss)
        val_accuracies.append(accuracy)

    return train_losses, val_losses, val_accuracies

# Load data
features, targets = load_data(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'led-data', '500_N5.txt'))

# 10-fold cross-validation
kf = KFold(n_splits=10)
fold_results = []

for fold, (train_idx, val_idx) in enumerate(kf.split(features)):
    train_features, train_targets = features[train_idx], targets[train_idx]
    val_features, val_targets = features[val_idx], targets[val_idx]

    train_dataset = TensorDataset(train_features, train_targets)
    val_dataset = TensorDataset(val_features, val_targets)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    model = LEDNet()
    train_losses, val_losses, val_accuracies = train_and_evaluate(model, train_loader, val_loader)

    fold_results.append((val_losses[-1], val_accuracies[-1]))

# Calculate average validation loss and accuracy over all folds
average_loss = np.mean([result[0] for result in fold_results])
average_accuracy = np.mean([result[1] for result in fold_results])

print(f'Average Validation Loss: {average_loss:.4f}')
print(f'Average Validation Accuracy: {average_accuracy:.4f}%')

# Plotting the results for the last fold
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Loss during Training')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(val_accuracies, label='Validation Accuracy')
plt.title('Validation Accuracy during Training')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.show()
