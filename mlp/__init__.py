import numpy as np
from keras.datasets import mnist
from network import NeuralNetwork
from losses import cross_entropy

# Carrega o dataset MNIST
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(-1, 784) / 255.0
X_test = X_test.reshape(-1, 784) / 255.0


# one-hot 
def one_hot(y, num_classes=10):
    out = np.zeros((len(y), num_classes))
    out[np.arange(len(y)), y] = 1
    return out

# Convertendo os rótulos para one-hot encoding
y_train = one_hot(y_train)
y_test = one_hot(y_test)


nn = NeuralNetwork([784, 128, 64, 10])

# Treinamento
epochs = 10
lr = 0.01

# Loop de treinamento
for epoch in range(epochs):

    # forward
    y_pred = nn.forward(X_train)

    # loss
    loss = cross_entropy(y_train, y_pred)

    # backward
    nn.backward(X_train, y_train, lr)

    print(f"Epoch {epoch} | Loss: {loss}")