import numpy as np
from keras.datasets import mnist
from activations import relu, softmax


# Carrega o dataset MNIST
(X_train, y_train), (X_test, y_test) = mnist.load_data()

print(X_train.shape)
print(y_train.shape)

print(X_test.shape)
print(y_test.shape)


# Pré-processamento
X_train = X_train.reshape(-1, 784).astype(np.float32)
X_test = X_test.reshape(-1, 784).astype(np.float32)

X_train /= 255.0
X_test /= 255.0


# array de camadas da rede (input, hidden1, hidden2, output)
layers = [784, 128, 64, 10]

# Classe da Rede Neural
class NeuralNetwork:
    # Inicialização da rede com pesos e biases
    def __init__(self, layers):

        self.layers = layers
        self.weights = []
        self.biases = []

        # inicialização dos pesos e biases
        for i in range(len(layers) - 1):

            W = np.random.randn(
                layers[i],
                layers[i + 1]
            ) * np.sqrt(2 / layers[i])

            b = np.zeros((1, layers[i + 1]))

            self.weights.append(W)
            self.biases.append(b)

    # Propagação para frente (forward pass)
    def forward(self, X):

        A = X
        self.activations = [X]
        self.zs = []

        # Para cada camada, calculamos Z = A @ W + b e depois aplicamos a função de ativação
        for i in range(len(self.weights)):

            Z = A @ self.weights[i] + self.biases[i]
            self.zs.append(Z)

            if i == len(self.weights) - 1:
                A = softmax(Z)
            else:
                A = relu(Z)

            self.activations.append(A)

        return A