import numpy as np
from torch import softmax
from keras.datasets import mnist

# Array com o número de neurônios em cada camada (input, hidden1, hidden2, output)
layers = [784, 128, 64, 10]

# Listas para armazenar os pesos e biases de cada camada
weights = []
biases = []

# Inicialização dos pesos e biases
for i in range(len(layers) - 1):    
    # Inicialização dos pesos para a camada i usando a técnica de He (He initialization)
    W = np.random.randn(
        layers[i],
        layers[i+1]
    ) * np.sqrt(2 / layers[i])

    # Inicialização dos biases para a camada i como zeros
    b = np.zeros((1, layers[i+1]))

    weights.append(W)
    biases.append(b)
    

# Foward pass
def forward(X, weights, biases):

    activations = [X]
    zs = []

    A = X
    # Para cada camada, calculamos a saída linear (Z) e aplicamos a função de ativação
    for i in range(len(weights)):

        Z = A @ weights[i] + biases[i]

        zs.append(Z)
        # Para a última camada, usamos softmax para gerar probabilidades, caso contrário, usamos ReLU
        if i == len(weights) - 1:
            A = softmax(Z)
        else:
            A = relu(Z)

        activations.append(A)

    return activations, zs