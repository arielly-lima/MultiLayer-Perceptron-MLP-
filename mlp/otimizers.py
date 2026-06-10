# Otimizadores para atualização dos pesos e biases da rede neural
# Atualmente, apenas o Stochastic Gradient Descent (SGD) é implementado, mas outros otimizadores como Adam ou RMSprop podem ser adicionados posteriormente.
def sgd(param, grad, lr):
    return param - lr * grad