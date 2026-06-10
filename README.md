# MLP do Zero — Classificação de Dígitos (MNIST)

Implementação completa de um Multi-Layer Perceptron usando apenas NumPy, sem frameworks de deep learning. O objetivo não foi só obter uma boa acurácia: foi entender cada operação que acontece dentro da rede — o que cada linha de código realmente significa e por que a rede aprende.

---

## Visão geral

| Item | Detalhe |
|---|---|
| Dataset | MNIST (70.000 imagens, 28×28 px) |
| Arquitetura principal | 784 → 128 → 64 → 10 |
| Ativações | ReLU (ocultas) + Softmax (saída) |
| Loss | Cross-Entropy |
| Otimizadores | SGD, Adam |
| Melhor acurácia no teste | **97,22%** |

---

## Como rodar

### Instalação

```bash
pip install -r requirements.txt
```

O `requirements.txt` lista apenas `numpy>=1.24` e `matplotlib>=3.8`. Nenhuma dependência de framework de deep learning é necessária.

### Treinamento

```bash
python train.py
```

O `train.py` tenta carregar o MNIST via `keras.datasets.mnist`. Se `keras` não estiver disponível, ele faz o download automático dos arquivos IDX brutos diretamente do Google Storage e processa tudo em NumPy puro:

```python
# Carrega MNIST via keras se disponível, senão baixa os arquivos IDX brutos
try:
    from keras.datasets import mnist as keras_mnist
except Exception:
    keras_mnist = None
```

Os arquivos são lidos com `gzip` e `struct`, os pixels normalizados para `[0, 1]` e as imagens convertidas de `(N, 28, 28)` para vetores de `(N, 784)` — tudo sem depender de nenhum framework.

### Notebooks de análise

```bash
jupyter notebook notebooks/experimentos.ipynb
```

O notebook contém os experimentos comparativos, os gráficos de loss e acurácia por época, e a validação do MLP no problema XOR antes de escalar para o MNIST.

---

## Metodologia de desenvolvimento

Antes de atacar o MNIST, construí uma versão simplificada do MLP para resolver o problema XOR.

O XOR foi escolhido porque é o exemplo clássico que uma rede linear não consegue resolver — ele exige pelo menos uma camada oculta com não linearidade. Se o forward pass e o backpropagation estivessem errados, a rede simplesmente não aprenderia o XOR.

O objetivo foi validar de forma isolada, em um problema pequeno o suficiente para depurar linha a linha:

- Que o forward pass produzia saídas coerentes
- Que o backpropagation calculava gradientes corretos
- Que a atualização de pesos fazia a loss cair de forma consistente

Somente após confirmar que a rede aprendia o XOR, expandi a implementação para múltiplas camadas ocultas e apliquei ao MNIST. Essa abordagem foi essencial: quando algo dava errado no MNIST, eu sabia que o mecanismo de gradientes já estava validado e o problema tinha outra causa (escala dos pesos, learning rate, dimensões das matrizes).

---

## Arquitetura da rede

### Estrutura principal

```
Entrada (784)
   ↓
Camada oculta 1 — 128 neurônios, ReLU
   ↓
Camada oculta 2 — 64 neurônios, ReLU
   ↓
Saída — 10 neurônios, Softmax
```

A `NeuralNetwork` aceita uma lista de camadas de tamanho arbitrário, então todas as configurações testadas usam a mesma classe:

```python
model = NeuralNetwork([784, 128, 64, 10], seed=42, optimizer="sgd")
model = NeuralNetwork([784, 256, 128, 10], seed=42, optimizer="sgd")
model = NeuralNetwork([784, 128, 64, 10], seed=42, optimizer="adam")
```

### Justificativa das escolhas

**Número de camadas:** duas camadas ocultas são suficientes para capturar a não linearidade do MNIST sem exagerar no número de parâmetros. Uma única camada ainda classificaria bem, mas duas permitem que a rede aprenda representações hierárquicas: a primeira camada combina pixels brutos em padrões locais (bordas, curvas), a segunda combina esses padrões em estruturas mais complexas (partes de dígitos).

**Número de neurônios (128 e 64):** valores moderados que oferecem expressividade sem tornar o treinamento inviável. A redução progressiva de 128 para 64 força a rede a comprimir a representação antes da classificação final — funciona como um gargalo suave que desfavorece memorização e incentiva generalização.

**ReLU nas camadas ocultas:** não satura para valores positivos, o gradiente flui melhor do que com sigmoid ou tanh, e se encaixa diretamente com a inicialização de He. Sua derivada é trivial — `(x > 0).astype(float)` — o que simplifica muito a implementação do backpropagation manual:

```python
# activations.py
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)
```

**Softmax na saída:** converte os logits em uma distribuição de probabilidade sobre as 10 classes. O truque numérico de subtrair o máximo antes da exponencial evita overflow:

```python
# activations.py
def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)
```

**One-Hot Encoding:** os rótulos originais do MNIST são inteiros de 0 a 9. Para que a Cross-Entropy possa comparar a saída do Softmax diretamente com o rótulo, cada inteiro é convertido para um vetor de 10 posições com `1.0` na posição correta:

```
7  →  [0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
```

```python
# train.py
def one_hot(labels, num_classes=10):
    encoded = np.zeros((labels.shape[0], num_classes), dtype=np.float32)
    encoded[np.arange(labels.shape[0]), labels] = 1.0
    return encoded
```

---

## Detalhes de implementação

### Inicialização dos pesos

Usei **He Initialization** para todas as camadas ocultas:

```
W ~ N(0, sqrt(2 / n_in))
```

```python
# network.py
W = np.random.randn(layers[i], layers[i + 1]) * np.sqrt(2 / layers[i])
b = np.zeros((1, layers[i + 1]), dtype=np.float32)
```

Quando os pesos são inicializados com valores muito pequenos, os gradientes desaparecem nas camadas anteriores. Quando são muito grandes, as ativações explodem e o Softmax produz `inf` ou `NaN`. He Initialization mantém a variância das ativações estável ao longo das camadas — e é especificamente recomendada para ReLU porque leva em conta que metade das ativações são zeradas (os valores negativos).

Os vieses começam em zero porque não sofrem do problema de simetria dos pesos. Pesos zerados forçam todos os neurônios a aprenderem a mesma função (o gradiente é igual para todos). Vieses zerados não têm esse problema — eles são ajustados individualmente durante o treinamento.

---

### Forward pass

Cada camada executa duas etapas. Os valores intermediários são armazenados em `self.zs` e `self.activations` para uso no backward:

```python
# network.py
for i, (W, b) in enumerate(zip(self.weights, self.biases)):
    Z = A @ W + b          # combinação linear
    self.zs.append(Z)

    if i == len(self.weights) - 1:
        A = softmax(Z)     # camada de saída
    else:
        A = relu(Z)        # camadas ocultas

    self.activations.append(A)
```

`Z` guarda a saída linear antes da ativação. `A` guarda as ativações. Esses dois caches são indispensáveis no backward: `A_prev` fornece o gradiente de `W`, e `Z` é necessário para calcular a derivada da ReLU na camada anterior.

---

### Backpropagation

O objetivo é descobrir quanto cada peso contribuiu para o erro e ajustá-lo na direção que reduz a loss.

**Gradiente na camada de saída** (derivada combinada de Softmax + Cross-Entropy):

```python
dZ = self.activations[-1] - y_true
```

Esse resultado simplificado é uma das razões pelas quais Softmax e Cross-Entropy são sempre usadas juntas: os termos se cancelam e o gradiente de saída vira apenas a diferença entre a previsão e o rótulo. Isso só acontece porque a derivada da Cross-Entropy em relação ao logit pré-softmax simplifica para `y_pred - y_true`.

**Propagação para as camadas ocultas** (regra da cadeia, iterando de trás para frente):

```python
# network.py
for i in reversed(range(len(self.weights))):
    A_prev = self.activations[i]

    dW = A_prev.T @ dZ / m          # gradiente dos pesos
    db = np.sum(dZ, axis=0, keepdims=True) / m   # gradiente dos vieses

    # ... atualização dos pesos ...

    if i > 0:
        dA_prev = dZ @ self.weights[i].T
        dZ = dA_prev * relu_derivative(self.zs[i - 1])
```

A divisão por `m` (tamanho do batch) normaliza os gradientes, garantindo que o learning rate não precise ser reajustado quando o tamanho do mini-batch muda.

**Atualização com SGD:**

```python
self.weights[i] -= lr * dW
self.biases[i]  -= lr * db
```

---

### Adam

Além do SGD, implementei o otimizador Adam dentro da própria classe `NeuralNetwork`. Adam mantém médias móveis dos gradientes (`m`) e dos gradientes ao quadrado (`v`), com correção de viés para as primeiras iterações:

```python
# network.py (trecho do backward com Adam)
self.m_weights[i] = self.beta1 * self.m_weights[i] + (1 - self.beta1) * dW
self.v_weights[i] = self.beta2 * self.v_weights[i] + (1 - self.beta2) * (dW ** 2)
m_hat = self.m_weights[i] / (1 - self.beta1 ** self.t)
v_hat = self.v_weights[i] / (1 - self.beta2 ** self.t)
self.weights[i] -= lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
```

Os valores padrão usados foram `beta1=0.9`, `beta2=0.999` e `epsilon=1e-8`, que são os recomendados no paper original.

---

### Estrutura dos experimentos

Cada configuração é definida como um dicionário e passada para a mesma função `train_model()`:

```python
# train.py
experiments = [
    {"label": "config-1", "layers": [784, 128, 64, 10],  "lr": 0.02, "epochs": 20},
    {"label": "config-2", "layers": [784, 256, 128, 10], "lr": 0.02, "epochs": 20},
    {"label": "config-3", "layers": [784, 128, 64, 10],  "lr": 0.03, "epochs": 20},
    {"label": "config-4-adam", "layers": [784, 128, 64, 10], "lr": 0.001,
     "epochs": 20, "optimizer": "adam"},
]
```

---

## Resultados

Os experimentos foram executados em duas rodadas. A primeira revelou dois bugs que distorceram os resultados. A segunda, após as correções, produziu resultados válidos e comparáveis.

---

### Rodada 1 — antes da correção (com bugs)

#### Acurácia final (rodada 1)

| Modelo | Camadas | Otimizador | LR real usado | Acurácia (teste) |
|---|---|---|---|---|
| config-1 | 784 → 128 → 64 → 10 | SGD | 0.02 | 96.45% |
| config-2 | 784 → 256 → 128 → 10 | SGD | 0.02 | 97.22% |
| config-3 | 784 → 128 → 64 → 10 | SGD | **0.04** ← bug | colapsou (NaN) |
| config-4-adam | 784 → 128 → 64 → 10 | Adam | **0.03** ← bug | 94.31% |

**Bug 1 — config-3 com lr=0.04:** o experimento estava configurado com `lr=0.04`, alto demais para essa arquitetura. O treinamento parecia estável nas primeiras 14 épocas, chegando a ~97% de acurácia, e então colapsou abruptamente na época 15.

**Bug 2 — config-4-adam com lr errado:** o dicionário de configuração definia `lr=0.001` para o Adam, mas a função `train_model()` não recebia `lr=config["lr"]` corretamente — o Adam treinava com a taxa padrão `lr=0.03`, que é alta demais para esse otimizador.

#### Curvas — rodada 1 (com divergência do config-3)

**Loss por época:**

![Loss por época — rodada 1](results/loss_comparacao.png)

O `config-3` (roxo) apresenta oscilação crescente a partir da época 10 — sinal inicial da instabilidade acumulada antes do colapso. O Adam (rosa) mostra queda de loss extremamente rápida com train loss próxima de zero, mas val loss bem acima, o que já indicava que o lr=0.03 era inadequado para Adam.

**Acurácia por época:**

![Acurácia por época — rodada 1](results/acc_comparacao.png)

O efeito do `lr=0.04` no `config-3` é visível na queda vertical de ~97% para ~10% na época 15 — o equivalente a chute aleatório entre 10 classes. Quando os pesos explodem, o Softmax passa a receber logits com magnitudes enormes, produz `NaN`, e a rede trava em predições degeneradas. A queda comprime o eixo Y do gráfico inteiro e faz as curvas do Adam parecerem anômalas, mesmo o Adam não sendo o problema nesse momento.

---

### Rodada 2 — após as correções

Duas correções foram aplicadas:

1. `config-3` ajustado de `lr=0.04` para `lr=0.03`
2. `train_model()` corrigido para receber `lr=config["lr"]` explicitamente
3. Proteção adicionada para interromper o treinamento automaticamente se a loss virar `NaN` ou `inf`

#### Acurácia final (rodada 2)

| Modelo | Camadas | Otimizador | LR | Acurácia (teste) |
|---|---|---|---|---|
| config-1 | 784 → 128 → 64 → 10 | SGD | 0.02 | 96.51% |
| config-2 | 784 → 256 → 128 → 10 | SGD | 0.02 | 96.73% |
| config-3 | 784 → 128 → 64 → 10 | SGD | 0.03 | 97.05% |
| config-4-adam | 784 → 128 → 64 → 10 | Adam | 0.001 | **97.77%** |

#### Curvas — rodada 2 (estável)

**Loss por época:**

![Loss por época — rodada 2](results/loss_comparacao_corrigido.png)

Com os parâmetros corrigidos, todas as configurações convergem sem oscilação. O Adam (rosa) ainda desce muito mais rápido que o SGD — train loss próxima de zero antes da época 10 — mas agora a val loss acompanha de forma mais coerente, com gap menor que na rodada 1.

**Acurácia por época:**

![Acurácia por época — rodada 2](results/acc_comparacao_corrigido.png)

Sem o colapso do config-3, o gráfico revela o comportamento real de cada configuração. O Adam sobe para ~99% de acurácia no treino rapidamente, mas a val acc fica em torno de 97–98%, indicando que a rede memorizou mais do que generalizou. As configurações com SGD sobem de forma mais gradual e a diferença entre train e val é menor, o que é um sinal mais saudável.

#### Análise comparativa

| Modelo | LR | Otimizador | Acurácia | Observação |
|---|---|---|---|---|
| config-1 | 0.02 | SGD | 96.51% | baseline, arquitetura menor, lr conservador |
| config-2 | 0.02 | SGD | 96.73% | mais parâmetros (256→128), ganho pequeno com mesmo lr |
| config-3 | 0.03 | SGD | 97.05% | mesma arquitetura do config-1, lr maior acelera convergência |
| config-4-adam | 0.001 | Adam | **97.77%** | melhor resultado, mas gap train/val maior indica overfitting |

O resultado mais interessante foi o do Adam após a correção do lr. Com `lr=0.001`, ele superou todas as configurações de SGD — mas o gráfico de acurácia deixa claro que a train acc (~99%) está bem acima da val acc (~97–98%), o que não acontece com o SGD. Para o MNIST sem regularização, o Adam converge mais rápido mas overfita mais.

A comparação entre config-1 e config-3 (mesma arquitetura, lr diferente) mostra que o lr foi mais determinante que o tamanho da rede: config-3 com `lr=0.03` superou config-2 com arquitetura maior e `lr=0.02`. Isso reforça que, dentro de uma faixa estável, uma taxa de aprendizado maior acelera a convergência mais do que adicionar neurônios.

---

## Verificação dos gradientes (Gradient Check)

Para validar o backpropagation, implementei um gradient check numérico comparando o gradiente analítico com uma estimativa pela definição de derivada:

```
gradiente_numérico ≈ (f(x + ε) - f(x - ε)) / 2ε
```

Quando a diferença relativa entre os dois gradientes ficou abaixo de `1e-5`, considerei os gradientes corretos. Essa verificação foi o que me deu confiança para escalar do XOR para o MNIST: com os gradientes validados no problema simples, qualquer problema no MNIST tinha outra causa — e eu sabia exatamente onde procurar.

---

## Decisões e dificuldades

### A decisão técnica mais difícil

A decisão mais difícil foi escolher a taxa de aprendizado — e entender por que ela importa tanto.

Comecei com `lr=0.05`. Nas primeiras épocas a loss despencava, o que parecia promissor. Mas a partir de um certo ponto, ela começava a oscilar e logo se tornava `NaN`. Isso acontece porque um passo muito grande faz os pesos ultrapassarem o mínimo, o gradiente inverte de sinal, os pesos vão para o outro lado com força ainda maior, e o processo diverge.

Tentei `lr=0.1` por curiosidade — o resultado foi instantaneamente `NaN`. Fui reduzindo progressivamente: `0.05` → `0.04` → `0.03` → `0.02`. Com `lr=0.02` e `lr=0.03` a convergência ficou estável para as arquiteturas testadas.

A lição foi que a taxa de aprendizado não é só um parâmetro de velocidade. Ela define o regime de treinamento inteiro: abaixo de um certo valor a rede aprende devagar mas converge; acima de outro valor ela diverge e o treino é inútil. A janela de valores úteis é surpreendentemente estreita.

---

### O que não funcionou

**Inicializar todos os pesos com zero:** minha primeira tentativa usava pesos zerados. O resultado foi que todos os neurônios de cada camada calculavam exatamente a mesma combinação linear, recebiam o mesmo gradiente e atualizavam para o mesmo valor. A rede tinha 128 neurônios na primeira camada, mas se comportava como se tivesse um só — nenhuma representação diferente emergia porque todos evoluíam de forma idêntica. A quebra de simetria na inicialização não é opcional.

**Inicializar com valores muito grandes:** o efeito oposto. Com pesos fora da escala, os logits chegavam ao Softmax com magnitudes enormes. A operação `exp(x)` saturava e produzia `inf`. O Softmax dividia `inf / inf`, resultando em `NaN` em toda a camada de saída, que propagava `NaN` para os gradientes e inviabilizava o treinamento.

**Problemas de dimensão no backward:** a multiplicação correta para o gradiente dos pesos é `A_prev.T @ dZ`, não `dZ @ A_prev`. Errar a ordem gerava dois tipos de problema: no melhor caso, um erro de `shape` incompatível que era fácil de identificar. No pior caso, as dimensões coincidiam por acidente e o resultado era numericamente errado sem nenhuma mensagem de erro — a loss simplesmente não diminuía. Aprendi a checar as dimensões esperadas antes de qualquer multiplicação matricial.

**Bug no experimento com Adam:** o `config-4-adam` estava definido com `lr=0.001`, mas na primeira versão do código a função `train_model()` não recebia `lr=config["lr"]` corretamente — o Adam estava treinando com a taxa padrão `0.03`, que é alta demais para esse otimizador. Isso produzia curvas estranhas e acurácia abaixo do esperado. O bug só ficou visível ao revisar os parâmetros que chegavam na função. Após a correção, o Adam treinou de forma estável.

**Config-3 com lr=0.04 e colapso do gráfico:** durante os testes intermediários, eu havia configurado `config-3` com `lr=0.04`. O treinamento parecia estável nas primeiras épocas, mas em torno da época 14–15 a loss explodiu para `NaN` e a acurácia despencou de ~97% para ~10% (equivalente a chute aleatório entre 10 classes). Isso acontece porque pesos que parecem estáveis podem acumular gradientes que crescem ao longo das épocas até ultrapassar um limiar crítico — e a partir daí o processo diverge de forma abrupta, não gradual.

O efeito nos gráficos foi dramático: a curva de acurácia do `config-3` caía verticalmente de ~97% para ~10% na época 15 e ficava travada lá. Isso distorcia o eixo Y e criava a falsa impressão de que o Adam também estava com problema — visualmente as curvas do `config-4-adam` pareciam anômalas porque o gráfico inteiro estava comprimido pela queda do `config-3`.

A correção foi dupla: reduzi o `config-3` para `lr=0.03` (que é estável), e adicionei uma proteção no `train_model()` para interromper o experimento automaticamente se a loss virar `NaN` ou `inf`:

```python
# train.py
if not np.isfinite(train_loss) or not np.isfinite(val_loss):
    print("Perda divergiu para NaN/inf; interrompendo o treinamento deste experimento.")
    break
```

Essa proteção evita que um experimento ruim distorça os gráficos dos demais e torna o diagnóstico muito mais rápido.

**Carregar o MNIST sem depender de keras:** a solução óbvia era `from keras.datasets import mnist`, mas isso exige TensorFlow instalado. Para manter o projeto independente de frameworks de deep learning, implementei o carregamento dos arquivos IDX brutos. O formato IDX é binário: os primeiros 4 bytes são um número mágico que codifica o tipo de dado e o número de dimensões, seguidos pelas dimensões e depois pelos dados. Implementei a leitura com `gzip` e `struct`, converti os pixels para `float32` e normalizei para `[0, 1]`.

---

### O que faria diferente

Implementaria **Batch Normalization** desde o início. Normalizar as ativações entre camadas torna o treinamento muito mais estável e permitiria testar learning rates maiores com segurança.

Adicionaria **regularização L2** ou **Dropout** para investigar se conseguia generalizar além de 97% sem overfitting — o Adam com 20 épocas claramente começou a overfitar.

Centralizaria todos os parâmetros dos experimentos em um único dicionário desde o começo, e passaria sempre `**config` para `train_model()`. O bug do Adam — `lr` errado chegando na função — seria impossível com essa estrutura.

---

## Lições aprendidas

Redes neurais são operações matriciais encadeadas mais funções de ativação. O "aprendizado" é inteiramente a regra da cadeia aplicada à loss, repetida muitas vezes. Antes dessa implementação, eu usava PyTorch como caixa-preta; agora consigo mapear cada operação de alto nível para o que ela faz matricialmente.

As partes que parecem triviais — inicialização dos pesos, ordem das dimensões nas multiplicações, escala do learning rate — são exatamente onde os bugs se escondem. E quando algo falha sem mensagem de erro clara (loss que não cai, acurácia presa em 10%), o problema quase sempre está nessas partes "simples".

---

## Estrutura do repositório

```
.
├── README.md
├── mlp/
│   ├── __init__.py          ← exporta NeuralNetwork, relu, softmax, cross_entropy, sgd, adam
│   ├── network.py           ← classe NeuralNetwork (forward, backward, predict, evaluate)
│   ├── activations.py       ← relu, relu_derivative, softmax
│   ├── losses.py            ← cross_entropy, cross_entropy_derivative
│   └── otimizers.py         ← funções sgd e adam standalone
├── notebooks/
│   └── experimentos.ipynb   ← XOR, MNIST, gráficos, análises comparativas
├── results/
│   ├── loss_comparacao.png
│   └── acc_comparacao.png
├── train.py                 ← carrega MNIST, roda experimentos, gera gráficos
└── requirements.txt         ← numpy>=1.24, matplotlib>=3.8
```

---

## Critérios atendidos

- [x] Forward pass genérico para número arbitrário de camadas
- [x] Backpropagation com gradientes corretos (loss converge)
- [x] SGD com learning rate configurável e mini-batches (batch_size=128)
- [x] Acurácia ≥ 92% no teste (melhor resultado: **97,22%**)
- [x] Curva de loss e acurácia por época geradas automaticamente
- [x] Comparação de 4 configurações diferentes (2 arquiteturas, 2 learning rates, 2 otimizadores)
- [x] Gradient check numérico implementado
- [x] Otimizador adicional (Adam com beta1, beta2, epsilon configuráveis)
- [x] README completo com todas as seções obrigatórias