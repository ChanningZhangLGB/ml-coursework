import matplotlib.pyplot as plt
import numpy as np
from model import MLP

def average_metrics(num_runs, num_epochs, train_data, target_data, num_input, num_output, hidden_layers, lr, momentum):
    all_losses = np.zeros((num_runs, num_epochs))
    all_accuracies = np.zeros((num_runs, num_epochs))

    for run in range(num_runs):
        mlp = MLP(num_input, num_output, hidden_layers, lr, momentum)
        losses, accuracies, _ = mlp.train(train_data, target_data, num_epochs,activation_type='sigmoid', use_softmax=False)
        all_losses[run] = losses
        all_accuracies[run] = accuracies
       

    mean_losses = np.mean(all_losses, axis=0)
    mean_accuracies = np.mean(all_accuracies, axis=0)

    # Plotting
    epochs = range(num_epochs)
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, mean_losses, label='Average Loss')
    plt.title('Average Loss during Training (c = 5)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, mean_accuracies, label='Average Accuracy')
    plt.title('Average Accuracy during Training (c = 5)')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()

    return mean_losses, mean_accuracies


# XOR
num_input = 2
num_output = 1
hidden_layers = [3]
lr = 0.1
momentums = [0.5, 0.9, 0.99]
num_runs = 10
num_epochs = 1000
train_data = np.array([[0,0], [0,1], [1,0], [1,1]])
target_data = np.array([[0], [1], [1], [0]])

# Initialize dictionaries to store the results
losses_dict = {}
accuracies_dict = {}

# Iterate over learning rates and store the results
for i, momentum in enumerate(momentums):
    mean_losses, mean_accuracies = average_metrics(num_runs, num_epochs, train_data, target_data, num_input, num_output, hidden_layers, lr, momentum)


