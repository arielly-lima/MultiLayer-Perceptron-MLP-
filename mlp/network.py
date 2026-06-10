import numpy as np
import 

# Array com a quantidade de neurônios em cada camada da rede
layers = [784, 128, 64, 10]

# Lista para armazenar os pesos e bias de cada camada
weights = []
biases = []

# Criando os pesos e bias para cada camada
for i in range(len(layers) - 1):
    