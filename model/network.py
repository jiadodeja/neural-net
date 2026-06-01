import numpy as np

class NeuralNetwork:
    def __init__(self, layer_sizes):
        self.layer_sizes = layer_sizes
        self.weights = []
        self.biases = []

        for i in range(len(layer_sizes) - 1):
            w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * 0.01
            b = np.zeros((1, layer_sizes[i+1]))
            self.weights.append(w)
            self.biases.append(b)

    def relu(self, z):
        return np.maximum(0, z)

    def relu_derivative(self, z):
        return (z > 0).astype(float)

    def softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def forward(self, X):
        self.activations = [X]
        self.z_values = []

        for i in range(len(self.weights) - 1):
            z = self.activations[-1] @ self.weights[i] + self.biases[i]
            self.z_values.append(z)
            self.activations.append(self.relu(z))

        z = self.activations[-1] @ self.weights[-1] + self.biases[-1]
        self.z_values.append(z)
        self.activations.append(self.softmax(z))

        return self.activations[-1]

    def compute_loss(self, y_pred, y_true):
        n = y_true.shape[0]
        log_likelihood = -np.log(y_pred[range(n), y_true] + 1e-8)
        return np.mean(log_likelihood)

    def backward(self, y_true, learning_rate):
        n = y_true.shape[0]
        d_weights = [None] * len(self.weights)
        d_biases = [None] * len(self.biases)

        delta = self.activations[-1].copy()
        delta[range(n), y_true] -= 1
        delta /= n

        for i in reversed(range(len(self.weights))):
            d_weights[i] = self.activations[i].T @ delta
            d_biases[i] = np.sum(delta, axis=0, keepdims=True)
            if i > 0:
                delta = delta @ self.weights[i].T * self.relu_derivative(self.z_values[i-1])

        for i in range(len(self.weights)):
            self.weights[i] -= learning_rate * d_weights[i]
            self.biases[i] -= learning_rate * d_biases[i]

    def train(self, X, y, epochs=100, learning_rate=0.01, batch_size=64):
        for epoch in range(epochs):
            indices = np.random.permutation(X.shape[0])
            X, y = X[indices], y[indices]

            for start in range(0, X.shape[0], batch_size):
                X_batch = X[start:start+batch_size]
                y_batch = y[start:start+batch_size]
                self.forward(X_batch)
                self.backward(y_batch, learning_rate)

            if epoch % 10 == 0:
                y_pred = self.forward(X)
                loss = self.compute_loss(y_pred, y)
                accuracy = np.mean(np.argmax(y_pred, axis=1) == y)
                print(f"Epoch {epoch}: loss={loss:.4f}, accuracy={accuracy:.4f}")

    def save(self, path):
        np.savez(path,
                 weights=np.array(self.weights, dtype=object),
                 biases=np.array(self.biases, dtype=object),
                 layer_sizes=np.array(self.layer_sizes))

    def load(self, path):
        data = np.load(path, allow_pickle=True)
        self.weights = list(data['weights'])
        self.biases = list(data['biases'])
        self.layer_sizes = list(data['layer_sizes'])
