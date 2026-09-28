import numpy as np
import matplotlib.pyplot as plt
import itertools
import time
from model import MLP

# M-of-N
def generate_k_input_M_of_N_dataset(k):
    M = k // 2
    inputs = np.array(list(itertools.product([0, 1], repeat=k)))
    outputs = np.array([[1] if sum(x) == M else [0] for x in inputs])
    return inputs, outputs

def average_metrics(num_runs, num_epochs, train_data, target_data, num_input, num_output, hidden_layers, lr, momentum):
    all_losses = np.zeros((num_runs, num_epochs))
    all_accuracies = np.zeros((num_runs, num_epochs))
    all_times = []

    for run in range(num_runs):
        mlp = MLP(num_input, num_output, hidden_layers, lr, momentum)
        start_time = time.time()
        losses, accuracies, _ = mlp.train(train_data, target_data, num_epochs)
        end_time = time.time()

        all_losses[run] = losses
        all_accuracies[run] = accuracies
        all_times.append(end_time - start_time)

    mean_losses = np.mean(all_losses, axis=0)
    mean_accuracies = np.mean(all_accuracies, axis=0)
    average_time = np.mean(all_times)

    return mean_losses, mean_accuracies, average_time

def train_and_plot(num_runs, num_epochs, k, hidden_layers_configs, lr, momentum):
    train_data, target_data = generate_k_input_M_of_N_dataset(k)
    
    epochs = range(num_epochs)
    plt.figure(figsize=(14, 6))
    training_times = []

    # Plot for Losses and Accuracies
    for hidden_layers in hidden_layers_configs:
        mean_losses, mean_accuracies, average_time = average_metrics(
            num_runs, num_epochs, train_data, target_data, k, 1, 
            hidden_layers, lr, momentum)
        
        training_times.append((hidden_layers, average_time))
        
        # Plotting Losses
        plt.subplot(1, 2, 1)
        plt.plot(epochs, mean_losses, label=f'Layers {hidden_layers}')
        
        # Plotting Accuracies
        plt.subplot(1, 2, 2)
        plt.plot(epochs, mean_accuracies, label=f'Layers {hidden_layers}')

    plt.subplot(1, 2, 1)
    plt.title('Training Loss for Different Configurations (c = 10)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.title('Training Accuracy for Different Configurations (c = 10)')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()

    # Print training times for each configuration
    for layers, time in training_times:
        print(f"Average training time for hidden layers {layers}: {time:.2f} seconds")

# Define parameters and call the training function
num_runs = 10
num_epochs = 1000
k = [4, 16, 32]
hidden_layers_configs = [[5], [5, 5], [10, 10]]
lr = 0.1
momentum = 0.9

train_and_plot(num_runs, num_epochs, k[0], hidden_layers_configs, lr, momentum)
