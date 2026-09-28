import numpy as np
import matplotlib.pyplot as plt
import time

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

    def _relu(self, x):
        return np.maximum(0, x)

    def _drelu(self, x):
        return (x > 0).astype(float)

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def _delsigmoid(self, x):
        return x * (1 - x)

    def forward_sig(self, batch):
        activation = batch
        activations = [batch]
        for i in range(len(self.weights)):
            z = np.dot(activation, self.weights[i]) + self.biases[i]
            activation = self._sigmoid(z)
            activations.append(activation)
        return activations

    def forward_relu(self, batch):
        activation = batch
        activations = [batch]
        for i in range(len(self.weights) - 1):
            z = np.dot(activation, self.weights[i]) + self.biases[i]
            activation = self._relu(z)  # Using ReLU in hidden layers to avoid vanishing gradients
            activations.append(activation)
        # Pre-clipping step before sigmoid in the output layer
        z = np.dot(activation, self.weights[-1]) + self.biases[-1]
        z_clipped = np.clip(z, -10, 10)  # Additional clipping step before sigmoid to prevent overflow
        activation = self._sigmoid(z_clipped)
        activations.append(activation)
        return activations


    def backpropagate_sig(self, activations, target):
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

        return loss

    def backpropagate_relu(self, activations, target):
        error = activations[-1] - target
        loss = np.mean((error) ** 2)
        deltas = [error * self._delsigmoid(activations[-1])]

        for i in reversed(range(len(self.weights) - 1)):
            delta = np.dot(deltas[0], self.weights[i + 1].T) * self._drelu(activations[i + 1])
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

        return loss
    
    def train(self, train_data, target_data, num_epochs):
        self.initialize_network()
        losses = []
        accuracies = []
        start_time = time.time()

        for epoch in range(num_epochs):
            activations = self.forward(train_data)
            predictions = (activations[-1] > 0.5).astype(int)
            accuracy = np.mean(predictions == target_data)
            loss = self.backpropagate(activations, target_data)

            losses.append(loss)
            accuracies.append(accuracy)

            if epoch % 10 == 0:  # Optional: report time every 100 epochs
                elapsed_time = time.time() - start_time
                print(f"Epoch {epoch}: Loss = {loss:.4f}, Accuracy = {accuracy:.4f}, Time = {elapsed_time:.2f} sec")

        total_time = time.time() - start_time
        print(f"Total training time: {total_time:.2f} seconds")
        return losses, accuracies, total_time
    

# Prepare data
def prepare_identity_data(n):
    x = np.array([np.binary_repr(i, width=int(np.ceil(np.log2(n)))) for i in range(n)])
    x = np.array([[int(b) for b in num] for num in x])
    y = np.copy(x)
    return x, y

def train_identity_mapping(mlp, train_data, target_data, num_epochs, activation_type):
    # Choose the forward and backpropagation method based on activation type
    if activation_type == 'relu':
        forward = mlp.forward_relu
        backpropagate = mlp.backpropagate_relu
    else:
        forward = mlp.forward_sig
        backpropagate = mlp.backpropagate_sig

    mlp.initialize_network()
    losses = []
    accuracies = []
    start_time = time.time()

    for epoch in range(num_epochs):
        activations = forward(train_data)
        predictions = (activations[-1] > 0.5).astype(int)
        accuracy = np.mean(predictions == target_data)
        loss = backpropagate(activations, target_data)

        losses.append(loss)
        accuracies.append(accuracy)

        if epoch % 10 == 0:
            elapsed_time = time.time() - start_time
            print(f"Epoch {epoch}: Loss = {loss:.4f}, Accuracy = {accuracy:.4f}, Time = {elapsed_time:.2f} sec")

    total_time = time.time() - start_time
    print(f"Total training time: {total_time:.2f} seconds")
    return losses, accuracies, total_time

def average_results(num_runs, num_epochs, train_data, target_data, activation_type):
    all_losses = []
    all_accuracies = []
    all_times = []
    
    for _ in range(num_runs):
        mlp = MLP(num_input=4, num_output=4, hidden_layers=[4], lr=0.1, momentum=0.9)
        losses, accuracies, training_time = train_identity_mapping(mlp, train_data, target_data, num_epochs, activation_type)
        
        all_losses.append(losses)
        all_accuracies.append(accuracies)
        all_times.append(training_time)
    
    avg_losses = np.mean(all_losses, axis=0)
    avg_accuracies = np.mean(all_accuracies, axis=0)
    avg_time = np.mean(all_times)
    
    return avg_losses, avg_accuracies, avg_time

train_data, target_data = prepare_identity_data(16)

# Running experiments for both activation functions
avg_losses_relu, avg_accuracies_relu, _ = average_results(5, 200, train_data, target_data, 'relu')
avg_losses_sig, avg_accuracies_sig, _ = average_results(5, 200, train_data, target_data, 'sigmoid')

# Plotting results
epochs = range(200)
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(epochs, avg_losses_relu, label='ReLU Loss')
plt.plot(epochs, avg_losses_sig, label='Sigmoid Loss')
plt.title('Average Loss per Epoch for ReLU vs. Sigmoid')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs, avg_accuracies_relu, label='ReLU Accuracy')
plt.plot(epochs, avg_accuracies_sig, label='Sigmoid Accuracy')
plt.title('Average Accuracy per Epoch for ReLU vs. Sigmoid')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.show()