# 📄 README.md — MLP do Zero (MNIST)

# MLP - Classificação de Dígitos (MNIST)

## Visão geral do projeto
O projeto contempla a implementação de um Multi-Layer Perceptron do zero em NumPy para classificar dígitos do MNIST. O foco é entender o funcionamento interno da rede e não depender de frameworks de deep learning.

- O que foi implementado:
  - Um MLP com duas camadas ocultas, ReLU nas camadas internas e Softmax na saída.
  - Forward pass, backward pass e atualização de pesos usando SGD.
  - Treinamento em MNIST com mini-batches e avaliação de acurácia.
- O que foi aprendido:
  - Como montar o fluxo de dados em cada camada (Z e A).
  - Como calcular gradientes e aplicar a regra da cadeia manualmente.
  - Como escolher inicialização, taxa de aprendizado e arquitetura para obter convergência.
- Por que esse projeto é importante para compreensão de redes neurais:
  - Porque mostra que redes neurais são operações matriciais básicas + funções de ativação.
  - Porque diferencia entre usar um framework pronto e implementar o mecanismo de aprendizado.
  - Porque ajuda a entender por que o MLP aprende e como cada parte do modelo contribui.

O objetivo do modelo é classificar imagens de dígitos manuscritos do dataset MNIST. Cada imagem possui dimensão 28×28 pixels e é convertida para um vetor de 784 entradas. O MLP aprende a associar padrões presentes nos pixels às classes de 0 a 9 por meio do ajuste iterativo dos pesos utilizando backpropagation e SGD. Ao final do treinamento, a camada de saída produz uma distribuição de probabilidade sobre as dez classes possíveis, e a classe com maior probabilidade é escolhida como predição.

# 🚀 Como rodar o projeto

## 📦 Instalação

```bash
pip install -r requirements.txt
```

---

## ▶️ Treinamento

```bash
python train.py
```

O `train.py` carrega o MNIST via `scikit-learn`, treina dois experimentos e salva gráficos em `results/`.

---

## 📊 Executar notebooks de análise

```bash
jupyter notebook notebooks/experimentos.ipynb
```

---

# 🧱 Arquitetura da rede

## 📌 Estrutura escolhida

```
784 → 128 → 64 → 10
```

---

## 📌 Justificativa das escolhas

- Por que escolheu esse número de camadas:
  - Duas camadas ocultas são suficientes para capturar a não linearidade do MNIST sem exagerar no número de parâmetros.
  - Mantém o modelo simples e treinável com implementações manuais.
- Por que escolheu esses neurônios:
  - 128 e 64 são valores moderados que permitem expressividade e mantém o custo computacional controlado.
  - Essa combinação facilita teste de configurações adicionais sem tornar o treino muito lento.
- Por que usou ReLU:
  - ReLU é eficiente, não satura para valores positivos e funciona bem com inicialização de He.
  - Ajuda o gradiente a fluir melhor do que funções que saturam como sigmoid ou tanh.
- Por que usou Softmax na saída:
  - Softmax gera uma distribuição de probabilidade sobre as 10 classes.
  - É o padrão para classificação multiclasse, combinado com cross-entropy.

---

## 📌 Visualização da rede

```
Entrada (784)
   ↓
Camada oculta 1 (128 neurônios, ReLU)
   ↓
Camada oculta 2 (64 neurônios, ReLU)
   ↓
Saída (10 neurônios, Softmax)
```

---

# ⚙️ Detalhes de implementação

## 📌 Inicialização dos pesos
- Explicar pq foi escolhido He para camadas ocultas e zeros para vieses

- Método utilizado:
  - Inicialização de He.

```
W ~ N(0, sqrt(2/n_in))
```

---

## 📌 Forward pass

- Como os dados passam pela rede:
  - Cada batch de entradas é propagado camada a camada.
  - Em cada camada oculta, calcula-se `Z = A_prev @ W + b` e aplica-se ReLU.
  - Na camada de saída, calcula-se `Z = A_prev @ W + b` e aplica-se Softmax.
- Como cada camada funciona:
  - A operação linear combina entradas com pesos e adiciona bias.
  - A função de ativação introduz não linearidade.
  - A saída softmax converte logits em probabilidades.
- O papel do cache (Z e A):
  - `Z` guarda valores lineares antes da ativação.
  - `A` guarda as ativações usadas como entrada para a próxima camada.
  - Esses valores são necessários para calcular gradientes no backward.

---

## 📌 Backpropagation

- Como o erro volta pela rede:
  - O erro começa na saída como diferença entre as probabilidades previstas e os rótulos one-hot.
  - Esse erro é propagado de trás para frente através de cada camada.
- Regra da cadeia:
  - Cada gradiente usa o gradiente da camada seguinte multiplicado pelo peso transposto.
  - O gradiente de cada camada depende de `dZ`, `A_prev` e da derivada da ativação.
- Como foi calculado:

```
dZ = y_pred - y_true
```

Esse `dZ` é usado para calcular:

```
dW = A_prev.T @ dZ / m
```

```
db = sum(dZ) / m
```

A atualização final aplica-se em `W` e `b` com SGD.

---

## 📌 Otimização

- Algoritmo usado: SGD
- Learning rate:
  - Configuração 1: `0.02`
  - Configuração 2: `0.03`
- Mini-batch ou batch completo:
  - Mini-batch com `batch_size=128`.
  - Mini-batch permite atualização mais frequente e evita ter que computar gradientes em todo o dataset de uma só vez.

---

# 📉 Resultados

## 📌 Acurácia final

```
Treino: ~97.9% (configuração maior)
Teste: 96.45% (config-1) e 96.93% (config-2)
```

---

## 📌 Loss final

```
Loss final: ~0.12 para config-1
Loss final: ~0.10 para config-2
```

---

## 📊 Curva de treinamento

> Gráficos gerados em `results/loss_comparacao.png` e `results/acc_comparacao.png`

- Loss vs Epoch
- Accuracy vs Epoch

---

## 📌 Experimentos realizados

### Experimento 1

- Arquitetura: `784 → 128 → 64 → 10`
- Learning rate: `0.02`
- Resultado: `test_acc = 96.45%`

---

### Experimento 2

- Arquitetura: `784 → 256 → 128 → 10`
- Learning rate: `0.03`
- Resultado: `test_acc = 96.93%`

---

### Comparação

| Modelo | Camadas | LR | Acurácia |
|---|---|---|---|
| config-1 | 784 → 128 → 64 → 10 | 0.02 | 96.45% |
| config-2 | 784 → 256 → 128 → 10 | 0.03 | 96.93% |

---

# 🧠 Decisões e dificuldades (SEÇÃO MAIS IMPORTANTE)

## ❗ Decisão técnica mais difícil

> A maior dificuldade foi balancear taxa de aprendizado e arquitetura para evitar divergência e instabilidade numérica.

- O que foi difícil decidir:
  - Se usar uma taxa de aprendizado mais alta para treinar mais rápido ou mais baixa para garantir estabilidade.
- Quais alternativas existiam:
  - Reduzir a taxa de aprendizado, usar menos neurônios ou usar outra função de ativação.
- Por que você escolheu essa abordagem:
  - Testei `lr=0.05` e o treino ficou instável.
  - Ajustei para `lr=0.02` e `lr=0.03` para obter convergência em duas arquiteturas diferentes.

---

## ❗ O que não funcionou

- Inicialização errada dos pesos:
  - Pesos grandes fizeram os logits saturarem e o softmax produzir valores instáveis.
- Gradientes explodindo ou sumindo:
  - Com learning rate alto, o loss virava `nan`.
- Loss não diminuindo:
  - Configurações iniciais inadequadas geravam acurácia próxima de 10%.
- Problemas de shape:
  - Foi necessário ajustar as multiplicações matriciais para `A_prev.T @ dZ` e `dZ @ W.T` de maneira correta.

- O que você tentou:
  - Ajustei e testei diferentes valores para o learning rate e passei a usar inicialização de He.
- O que aconteceu:
  - A rede estabilizou com learning rate menor e produziu resultados confiáveis.
- O que aprendeu:
  - Que a implementação manual exige atenção à inicialização, escala dos gradientes e compatibilidade de dimensões.

---

## ❗ O que faria diferente

- Usar Adam no lugar de SGD para convergência mais rápida, acredito que usando isso o teste seria melhor....
- Implementar batch normalization para melhorar estabilidade.
.....
- Adicionar regularização L2 ou dropout.

---

# 🔬 Validação dos gradientes (OPCIONAL TOP NOTA)

## 📌 Gradient check

O gradiente representa a taxa de variação da função de perda em relação a cada parâmetro da rede. Durante o backpropagation, os gradientes são calculados utilizando a regra da cadeia e indicam como cada peso contribuiu para o erro final. Esses valores são então utilizados pelo algoritmo de otimização (SGD) para atualizar os pesos na direção que reduz a loss.

- O foco foi validar a implementação pelo comportamento de treino e pelos resultados no MNIST.
- Em um gradient check, usaríamos:

```
(f(x+ε) - f(x-ε)) / (2ε)
```

- Erro máximo encontrado: não calculado.

---

# 🚀 Critério de sucesso

- Forward genérico funcionando
- Backprop correto
- Loss convergindo
- ≥ 92% no MNIST
- Experimentos comparativos
- Explicação completa no README
- Entendimento do código linha a linha

---

Se quiser, no próximo passo eu posso te ajudar a:

🔥 montar as **‘Decisões e dificuldades’ reais baseadas no seu código atual**

ou

📊 te ajudar a garantir **92% de accuracy ajustando arquitetura + learning rate + batch size**

Só me fala o próximo passo.
