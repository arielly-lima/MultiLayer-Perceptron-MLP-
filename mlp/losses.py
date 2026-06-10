# Função de perda: Cross-Entropy
def cross_entropy(y_true, y_pred):
    m = y_true.shape[0]

    return -np.sum(
        y_true * np.log(y_pred + 1e-8)
    ) / m


# derivada simplificada (softmax + CE)
def cross_entropy_derivative(y_true, y_pred):
    return (y_pred - y_true) / y_true.shape[0]