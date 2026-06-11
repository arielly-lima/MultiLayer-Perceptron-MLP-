import numpy as np

# Otimizadores para atualização dos pesos e biases da rede neural
# Suporte a SGD e Adam.

def sgd(param, grad, lr):
    return param - lr * grad


def adam(param, grad, m, v, t, lr, beta1=0.9, beta2=0.999, epsilon=1e-8):
    m = beta1 * m + (1 - beta1) * grad
    v = beta2 * v + (1 - beta2) * (grad ** 2)
    m_hat = m / (1 - beta1 ** t)
    v_hat = v / (1 - beta2 ** t)
    param = param - lr * m_hat / (np.sqrt(v_hat) + epsilon)
    return param, m, v