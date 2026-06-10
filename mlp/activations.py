# Ativação ReLU
def relu(x):
    return np.maximum(0, x)


def relu_derivative(x):
    return (x > 0).astype(float)


# Função softmax para a camada de saída - gerar probabilidades
def softmax(x):

    exp_x = np.exp(
        x - np.max(x, axis=1, keepdims=True)
    )

    return exp_x / np.sum(
        exp_x,
        axis=1,
        keepdims=True
    )
