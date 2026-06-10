# MLP - MultiLayer Perceptron

**Como rodar:** comandos para instalar dependências e executar o treinamento
**Arquitetura escolhida:** quantas camadas, quantos neurônios, quais ativações, e por que essas escolhas
**Resultados:** acurácia final, curva de loss, tabela comparativa de experimentos
**Decisões e dificuldades:** seção obrigatória (detalhes abaixo)


Decisões:
- 1. fiz um teste de mlp para o caso XOR
O problema do perceptron e a necessidade de adicionar camadas à rede, torna o MLP uma solução para isso. O problema central está não não-lineariedade entre estes pontos:
``(0,1) -> 1
(1,0) -> 1

(0,0) -> 0
(1,1) -> 0``

- 2. decisão das camadas:
Como as imagens do MNIST possuem 28x28 pixels, foram escolhidos 784 pixels de entrada
As dimensões que são criadas com arrays numpy, ficam assim:
W.shape = (neurônios da camada atual,
           neurônios da próxima camada)

Dessa forma, podemos achatar cada imagem assim:
``(28,28)``

A primeira camada oculta foi selecionado 128 neurônios/pesos
``W1.shape = (784,128)
b1.shape = (1,128)``

Já a segunda camada possui 64 neurônios/pesos
``W2.shape = (128,64)
b2.shape = (1,64)``

E, para saída:
``W3.shape = (64,10)
b3.shape = (1,10)``

- 3. Para inicialização foi usado Inicialização de He
- Mantém a variância estável ao longo das camadas.
- Por isso é a escolha padrão para ReLU.

- 4. Decisão pelo uso de classes