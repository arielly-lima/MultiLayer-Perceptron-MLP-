import numpy as np
from .activations import relu, relu_derivative, softmax
from .losses import cross_entropy


class NeuralNetwork:
    def __init__(self, layers, seed=None):
        self.layers = layers
        if seed is not None:
            np.random.seed(seed)

        self.weights = []
        self.biases = []

        for i in range(len(layers) - 1):
            W = np.random.randn(layers[i], layers[i + 1]) * np.sqrt(2 / layers[i])
            b = np.zeros((1, layers[i + 1]), dtype=np.float32)
            self.weights.append(W.astype(np.float32))
            self.biases.append(b)

    def forward(self, X):
        A = X.astype(np.float32)
        self.activations = [A]
        self.zs = []

        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            Z = A @ W + b
            self.zs.append(Z)

            if i == len(self.weights) - 1:
                A = softmax(Z)
            else:
                A = relu(Z)

            self.activations.append(A)

        return A

    def backward(self, y_true, lr=0.01):
        m = y_true.shape[0]
        dZ = self.activations[-1] - y_true

        for i in reversed(range(len(self.weights))):
            A_prev = self.activations[i]
            dW = A_prev.T @ dZ / m
            db = np.sum(dZ, axis=0, keepdims=True) / m

            self.weights[i] -= lr * dW
            self.biases[i] -= lr * db

            if i > 0:
                dA_prev = dZ @ self.weights[i].T
                dZ = dA_prev * relu_derivative(self.zs[i - 1])

    def predict_proba(self, X):
        return self.forward(X)

    def predict(self, X):
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)

    def evaluate(self, X, y_true):
        proba = self.predict_proba(X)
        loss = cross_entropy(y_true, proba)
        accuracy = np.mean(np.argmax(proba, axis=1) == np.argmax(y_true, axis=1))
        return loss, accuracy