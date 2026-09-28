import numpy as np
import time

class MLP:
    def __init__(self, num_input, num_output, hidden_layers, lr=0.1, momentum=0.9):
        self.num_input = num_input
        self.num_output = num_output
        self.hidden_layers = hidden_layers
        self.lr = lr
        self.momentum = momentum
        self.initialize_network()

    def initialize_network(self):
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

    def _softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def forward(self, batch, activation_type='relu', use_softmax=False):
        activation = batch
        activations = [batch]
        for i in range(len(self.weights) - 1):
            z = np.dot(activation, self.weights[i]) + self.biases[i]
            if activation_type == 'relu':
                activation = self._relu(z)
            elif activation_type == 'sigmoid':
                activation = self._sigmoid(z)
            activations.append(activation)
        z = np.dot(activation, self.weights[-1]) + self.biases[-1]
        if use_softmax:
            activation = self._softmax(z)
        else:
            activation = self._sigmoid(z)
        activations.append(activation)
        return activations

    def backpropagate(self, activations, target, activation_type='relu', use_softmax=False):
        if use_softmax:
            error = activations[-1] - target
            deltas = [error]  # For softmax, gradient already accounts for the derivative
        else:
            error = (activations[-1] - target) * self._delsigmoid(activations[-1])
            deltas = [error]

        loss = np.mean(np.sum(-target * np.log(np.clip(activations[-1], 1e-10, 1)), axis=1)) if use_softmax else np.mean(error**2)

        for i in reversed(range(len(self.weights) - 1)):
            if activation_type == 'relu':
                delta = np.dot(deltas[0], self.weights[i + 1].T) * self._drelu(activations[i + 1])
            elif activation_type == 'sigmoid':
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

    def train(self, train_data, target_data, num_epochs, activation_type='relu', use_softmax=False):
        losses = []
        accuracies = []
        start_time = time.time()

        for epoch in range(num_epochs):
            activations = self.forward(train_data, activation_type, use_softmax)
            predictions = np.argmax(activations[-1], axis=1) if use_softmax else (activations[-1] > 0.5).astype(int)
            accuracy = np.mean(predictions == np.argmax(target_data, axis=1) if use_softmax else target_data)
            loss = self.backpropagate(activations, target_data, activation_type, use_softmax)

            losses.append(loss)
            accuracies.append(accuracy)

            if epoch % 10 == 0:
                elapsed_time = time.time() - start_time
                print(f"Epoch {epoch}: Loss = {loss:.4f}, Accuracy = {accuracy:.4f}, Time = {elapsed_time:.2f} sec")

        total_time = time.time() - start_time
        print(f"Total training time: {total_time:.2f} seconds")
        return losses, accuracies, total_time



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

        return loss

    def train(self, train_data, target_data, num_epochs):
        self.initialize_network()
        losses = []
        accuracies = []

        for epoch in range(num_epochs):
            activations = self.forward(train_data)
            predictions = (activations[-1] > 0.5).astype(int)
            accuracy = np.mean(predictions == target_data)
            loss = self.backpropagate(activations, target_data)

            losses.append(loss)
            accuracies.append(accuracy)

        return losses, accuracies