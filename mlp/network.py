import numpy as np
from .activations import relu, relu_derivative, softmax
from .losses import cross_entropy

# Classe para a rede neural MLP
class NeuralNetwork:
    # Inicialização da rede com camadas especificadas
    def __init__(self, layers, seed=None):
        self.layers = layers
        if seed is not None:
            np.random.seed(seed)

        # Inicializar pesos e vieses para cada camada
        self.weights = []
        self.biases = []

        # Inicialização de He para camadas ocultas e zeros para vieses
        for i in range(len(layers) - 1):
            W = np.random.randn(layers[i], layers[i + 1]) * np.sqrt(2 / layers[i])
            b = np.zeros((1, layers[i + 1]), dtype=np.float32)
            self.weights.append(W.astype(np.float32))
            self.biases.append(b)
            
    # Propagação para frente da rede
    def forward(self, X):
        # Converter entrada para float32 para consistência. A = entrada normalizada
        A = X.astype(np.float32)
        self.activations = [A]
        # Armazenar as saídas lineares (Z) para uso na retropropagação
        self.zs = []
        
        # Propagação para trás e atualização dos pesos
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            # Calcular a saída linear da camada
            Z = A @ W + b
            self.zs.append(Z)

            # Aplicar função de ativação: ReLU para camadas ocultas
            if i == len(self.weights) - 1:
                # Para a última camada, usamos softmax para obter probabilidades
                A = softmax(Z)
            else:
                A = relu(Z)

            self.activations.append(A)

        return A

    # Retropropagação para atualizar pesos e vieses
    def backward(self, y_true, lr=0.01):
        # Número de amostras no batch (conjunto de treinamento)
        m = y_true.shape[0]
        dZ = self.activations[-1] - y_true
        
        # Atualizar pesos e vieses para cada camada, começando da última para a primeira
        for i in reversed(range(len(self.weights))):
            A_prev = self.activations[i]
            
            # Calcular gradientes para pesos e vieses
            dW = A_prev.T @ dZ / m
            # Gradiente para os vieses é a média dos gradientes de saída
            db = np.sum(dZ, axis=0, keepdims=True) / m

            # Atualizar pesos e vieses usando o gradiente descendente
            self.weights[i] -= lr * dW
            self.biases[i] -= lr * db

            # Para camadas ocultas, calcular o gradiente para a próxima camada usando a derivada da função de ativação ReLU
            if i > 0:
                dA_prev = dZ @ self.weights[i].T
                dZ = dA_prev * relu_derivative(self.zs[i - 1])

    # Função para prever probabilidades e classes
    def predict_proba(self, X):
        return self.forward(X)

    # Função para prever classes com base nas probabilidades
    def predict(self, X):
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)

    # Função para avaliar a perda e a acurácia do modelo
    def evaluate(self, X, y_true):
        proba = self.predict_proba(X)
        loss = cross_entropy(y_true, proba)
        accuracy = np.mean(np.argmax(proba, axis=1) == np.argmax(y_true, axis=1))
        return loss, accuracy