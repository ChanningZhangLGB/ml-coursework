import numpy as np
import matplotlib.pyplot as plt

class MLP:
    def __init__(self, num_input, num_output, hidden_layers, lr=0.1, momentum=0.9):
        self.num_input = num_input
        self.num_output = num_output
        self.hidden_layers = hidden_layers
        self.lr = lr
        self.momentum = momentum

    def initialize_network(self):
        """Initialize the network weights and biases."""
        self.layers = [self.num_input] + self.hidden_layers + [self.num_output]
        self.weights = []
        self.biases = []
        self.velocity_weights = []
        self.velocity_biases = []

        for i in range(len(self.layers) - 1):
            self.weights.append(np.random.randn(self.layers[i], self.layers[i + 1]) * np.sqrt(2. / self.layers[i]))
            self.biases.append(np.zeros((1, self.layers[i + 1])))
            self.velocity_weights.append(np.zeros((self.layers[i], self.layers[i + 1])))
            self.velocity_biases.append(np.zeros((1, self.layers[i + 1])))

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def _delsigmoid(self, x):
        return x * (1 - x)

    def forward(self, batch):
        activation = batch
        activations = [batch]
        for i in range(len(self.weights)):
            z = np.dot(activation, self.weights[i]) + self.biases[i]
            activation = self._sigmoid(z)
            activations.append(activation)
        return activations

    def backpropagate(self, activations, target):
        error = activations[-1] - target
        loss = np.mean((error) ** 2)
        deltas = [error * self._delsigmoid(activations[-1])]

        for i in reversed(range(len(self.weights) - 1)):
            delta = np.dot(deltas[0], self.weights[i + 1].T) * self._delsigmoid(activations[i + 1])
            deltas.insert(0, delta)

        for i in range(len(self.weights)):
            layer_input = np.atleast_2d(activations[i])
            delta = np.atleast_2d(deltas[i])
            gradient_weight = np.dot(layer_input.T, delta)
            gradient_bias = np.sum(delta, axis=0, keepdims=True)

            self.velocity_weights[i] = self.momentum * self.velocity_weights[i] + self.lr * gradient_weight
            self.velocity_biases[i] = self.momentum * self.velocity_biases[i] + self.lr * gradient_bias

            self.weights[i] -= self.velocity_weights[i]
            self.biases[i] -= self.velocity_biases[i]

        return loss, activations[1]  # Return activations from the first hidden layer

    def train(self, train_data, target_data, num_epochs):
        self.initialize_network()
        losses = []
        accuracies = []
        hidden_layer_outputs = np.zeros((num_epochs, len(train_data), self.hidden_layers[0]))  # Store hidden layer outputs

        for epoch in range(num_epochs):
            activations = self.forward(train_data)
            predictions = (activations[-1] > 0.5).astype(int)
            accuracy = np.mean(predictions == target_data)
            loss, hidden_outputs = self.backpropagate(activations, target_data)

            losses.append(loss)
            accuracies.append(accuracy)
            hidden_layer_outputs[epoch] = hidden_outputs

        return losses, accuracies, hidden_layer_outputs

def average_metrics(num_runs, num_epochs, train_data, target_data, num_input, num_output, hidden_layers, lr, momentum):
    all_losses = np.zeros((num_runs, num_epochs))
    all_accuracies = np.zeros((num_runs, num_epochs))
    all_hidden_outputs = None

    for run in range(num_runs):
        mlp = MLP(num_input, num_output, hidden_layers, lr, momentum)
        losses, accuracies, hidden_outputs = mlp.train(train_data, target_data, num_epochs)
        all_losses[run] = losses
        all_accuracies[run] = accuracies
        if all_hidden_outputs is None:
            all_hidden_outputs = np.zeros((num_runs, num_epochs, len(train_data), hidden_layers[0]))
        all_hidden_outputs[run] = hidden_outputs

    # Plotting hidden layer outputs for each training instance over epochs
    epochs = range(num_epochs)
    num_instances = len(train_data)
    for instance_index in range(num_instances):
        plt.figure(figsize=(12, 5))
        for neuron_index in range(hidden_layers[0]):
            plt.plot(epochs, all_hidden_outputs[:, :, instance_index, neuron_index].T, label=f'Neuron {neuron_index}')
        plt.title(f'Hidden Layer Outputs for Training Instance {instance_index}')
        plt.xlabel('Epoch')
        plt.ylabel('Output')
        plt.legend()
        plt.show()

# Example usage
num_input = 4
num_output = 4
hidden_layers = [4]
lr = 0.1
momentum = 0.9
num_runs = 1  # For detailed individual run analysis
num_epochs = 200
train_data = np.array([[(i >> j) & 1 for j in range(4)] for i in range(16)])
target_data = np.array(train_data)

average_metrics(num_runs, num_epochs, train_data, target_data, num_input, num_output, hidden_layers, lr, momentum)
