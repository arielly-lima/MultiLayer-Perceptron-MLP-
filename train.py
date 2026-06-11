import gzip
import os
import ssl
import struct
import time
import urllib.request

import numpy as np
from mlp import NeuralNetwork
from mlp.losses import cross_entropy

# Verificar se matplotlib está disponível para plotagem
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# Função para converter rótulos inteiros em codificação one-hot
def one_hot(labels, num_classes=10):
    encoded = np.zeros((labels.shape[0], num_classes), dtype=np.float32)
    # Atribuir 1.0 na posição correspondente ao rótulo de cada amostra
    encoded[np.arange(labels.shape[0]), labels] = 1.0
    return encoded

# Função para baixar arquivos do MNIST se não estiverem presentes localmente, garantindo que os dados estejam disponíveis para treinamento
def download_file(url, path):
    if os.path.exists(path):
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        urllib.request.urlretrieve(url, path)
    except Exception:
        context = ssl._create_unverified_context()
        urllib.request.urlretrieve(url, path, context=context)

# Função para ler arquivos IDX compactados do MNIST, convertendo-os em arrays numpy para uso no treinamento e avaliação do modelo
def read_idx_gz(path):
    with gzip.open(path, "rb") as f:
        magic = int.from_bytes(f.read(4), "big")
        num_dims = magic & 0xFF
        shape = tuple(int.from_bytes(f.read(4), "big") for _ in range(num_dims))
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.reshape(shape)


# Função para carregar o dataset MNIST via keras.datasets.mnist ou arquivo raw
# Retorna X_train, X_test, y_train, y_test no formato esperado pelo modelo.
def load_mnist(test_size=10000, random_state=42):
    try:
        from keras.datasets import mnist as keras_mnist
    except Exception:
        keras_mnist = None

    if keras_mnist is not None:
        (X_train, y_train), (X_test, y_test) = keras_mnist.load_data()
    else:
        # Baixar e ler os arquivos IDX compactados do MNIST se keras não estiver disponível
        base_url = "https://storage.googleapis.com/cvdf-datasets/mnist/"
        data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "mnist")
        files = {
            "train-images-idx3-ubyte.gz": base_url + "train-images-idx3-ubyte.gz",
            "train-labels-idx1-ubyte.gz": base_url + "train-labels-idx1-ubyte.gz",
            "t10k-images-idx3-ubyte.gz": base_url + "t10k-images-idx3-ubyte.gz",
            "t10k-labels-idx1-ubyte.gz": base_url + "t10k-labels-idx1-ubyte.gz",
        }
        
        # Baixar os arquivos do MNIST se não estiverem presentes localmente
        for filename, url in files.items():
            download_file(url, os.path.join(data_dir, filename))

        # Ler os arquivos IDX compactados do MNIST usando a função read_idx_gz para obter os arrays de imagens e rótulos para treino e teste
        X_train = read_idx_gz(os.path.join(data_dir, "train-images-idx3-ubyte.gz"))
        y_train = read_idx_gz(os.path.join(data_dir, "train-labels-idx1-ubyte.gz"))
        X_test = read_idx_gz(os.path.join(data_dir, "t10k-images-idx3-ubyte.gz"))
        y_test = read_idx_gz(os.path.join(data_dir, "t10k-labels-idx1-ubyte.gz"))

    # Normalizar os dados de imagem para o intervalo [0, 1] e converter os rótulos para o tipo inteiro, preparando os dados para o treinamento do modelo MLP
    X_train = X_train.reshape(-1, 28 * 28).astype(np.float32) / 255.0
    X_test = X_test.reshape(-1, 28 * 28).astype(np.float32) / 255.0
    y_train = y_train.astype(np.int64)
    y_test = y_test.astype(np.int64)
    return X_train, X_test, y_train, y_test

# Separar conjunto de validação a partir do conjunto de treino
def create_validation_split(
    X_train,
    y_train,
    validation_size=5000
):

    X_val = X_train[-validation_size:]
    y_val = y_train[-validation_size:]

    X_train = X_train[:-validation_size]
    y_train = y_train[:-validation_size]

    return X_train, X_val, y_train, y_val

# Gerador de batches para treinamento em mini-batches
def batch_generator(X, y, batch_size=128):
    n = X.shape[0]
    perm = np.random.permutation(n)
    # Gerar batches de dados embaralhados para treinamento
    for i in range(0, n, batch_size):
        indices = perm[i : i + batch_size]
        yield X[indices], y[indices]


# Função para treinar o modelo MLP usando o dataset MNIST, avaliando a perda e a acurácia em cada época
def train_model(
    model,
    X_train,
    y_train,
    X_val,
    y_val,
    epochs=20,
    lr=0.03,
    batch_size=128,
):
    # Dicionário para armazenar o histórico de perda e acurácia durante o treinamento, permitindo a análise do desempenho do modelo ao longo das épocas
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    # Loop de treinamento principal, iterando por um número definido de épocas, realizando a passagem forward e backward para cada batch de dados, e avaliando o desempenho do modelo no conjunto de treino e validação ao final de cada época
    for epoch in range(1, epochs + 1):
        # Medir o tempo gasto para cada época para monitorar a eficiência do treinamento
        start = time.time()
        for X_batch, y_batch in batch_generator(X_train, y_train, batch_size):
            model.forward(X_batch)
            model.backward(y_batch, lr=lr)

        # Avaliar a perda e a acurácia do modelo no conjunto de treino e validação ao final de cada época, armazenando os resultados no histórico para análise posterior
        train_loss, train_acc = model.evaluate(X_train, y_train)
        val_loss, val_acc = model.evaluate(X_val, y_val)

        # Armazenar os resultados de perda e acurácia para treino e validação no histórico, permitindo a visualização do desempenho do modelo ao longo das épocas
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        elapsed = time.time() - start
        print(
            f"Epoch {epoch:02}/{epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | "
            f"{elapsed:.1f}s"
        )

        # Verificar se a perda divergiu para NaN ou infinito, interrompendo o treinamento se isso ocorrer para evitar desperdício de tempo em experimentos que não estão convergindo
        if not np.isfinite(train_loss) or not np.isfinite(val_loss):
            print("Perda divergiu para NaN/inf; interrompendo o treinamento deste experimento.")
            break

    return history

# Função para plotar os gráficos de perda e acurácia para comparação entre diferentes configurações de treinamento
def plot_history(history_list, labels, output_dir="results"):
    os.makedirs(output_dir, exist_ok=True)
    if not HAS_MATPLOTLIB:
        print("matplotlib não está instalado; pulando geração de gráficos.")
        return

    plt.figure(figsize=(10, 4))
    for history, label in zip(history_list, labels):
        plt.plot(history["train_loss"], label=f"{label} - train loss")
        plt.plot(history["val_loss"], label=f"{label} - val loss", linestyle="--")

    plt.title("Loss por época")
    plt.xlabel("Época")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "loss_comparacao.png"))
    plt.close()

    plt.figure(figsize=(10, 4))
    for history, label in zip(history_list, labels):
        plt.plot(history["train_acc"], label=f"{label} - train acc")
        plt.plot(history["val_acc"], label=f"{label} - val acc", linestyle="--")

    plt.title("Acurácia por época")
    plt.xlabel("Época")
    plt.ylabel("Acurácia")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "acc_comparacao.png"))
    plt.close()
    print(f"Gráficos salvos em {output_dir}")

# Função principal para executar os experimentos de treinamento com diferentes configurações de camadas, taxa de aprendizado e número de épocas, avaliando o desempenho final do modelo em cada configuração
def run_experiments():
    print("Carregando MNIST...")

    X_train, X_test, y_train, y_test = load_mnist()

    # Separar conjunto de validação
    X_train, X_val, y_train, y_val = create_validation_split(
        X_train,
        y_train,
        validation_size=5000
    )

    # Converter rótulos para one-hot
    y_train_onehot = one_hot(y_train, num_classes=10)
    y_val_onehot = one_hot(y_val, num_classes=10)
    y_test_onehot = one_hot(y_test, num_classes=10)

    print(f"Treino: {X_train.shape}")
    print(f"Validação: {X_val.shape}")
    print(f"Teste: {X_test.shape}")
    experiments = [
        {
            "label": "config-1",
            "layers": [784, 128, 64, 10],
            "lr": 0.02,
            "epochs": 20,
            "batch_size": 128,
        },
        {
            "label": "config-2",
            "layers": [784, 256, 128, 10],
            "lr": 0.02,
            "epochs": 20,
            "batch_size": 128,
        },
        {
            "label": "config-3",
            "layers": [784, 128, 64, 10],
            "lr": 0.03,
            "epochs": 20,
            "batch_size": 128,
        },
        {
            "label": "config-4-adam",
            "layers": [784, 128, 64, 10],
            "lr": 0.001,
            "epochs": 20,
            "batch_size": 128,
            "optimizer": "adam",
        },
    ]

    histories = []
    final_scores = []
    # Executar cada configuração de experimento, treinando o modelo e avaliando o desempenho final em termos de perda e acurácia no conjunto de teste
    for config in experiments:
        print("\n" + "=" * 60)
        print(f"Treinando {config['label']} | camadas={config['layers']} lr={config['lr']} optimizer={config.get('optimizer', 'sgd')}\n")
        model = NeuralNetwork(
            config["layers"],
            seed=42,
            optimizer=config.get("optimizer", "sgd"),
        )
        # Treinar o modelo com a configuração atual e armazenar o histórico de perda e acurácia para análise posteriorvaliar o desempenho final do modelo no conjunto de teste e armazenar os resultados para comparação entre as diferentes configurações
        history = train_model(
            model,
            X_train,
            y_train_onehot,
            X_val,
            y_val_onehot,
            epochs=config.get("epochs", 20),
            lr=config.get("lr", 0.03),
            batch_size=config.get("batch_size", 128),
        )
        histories.append(history)
        test_loss, test_acc = model.evaluate(X_test, y_test_onehot)
        final_scores.append(
            {
                "label": config["label"],
                "layers": config["layers"],
                "lr": config["lr"],
                "epochs": config["epochs"],
                "test_loss": test_loss,
                "test_acc": test_acc,
            }
        )
        print(
            f"Resultado final {config['label']}: test_loss={test_loss:.4f} test_acc={test_acc:.4f}"
        )

    plot_history(histories, [cfg["label"] for cfg in experiments])

    print("\nComparação final:")
    for score in final_scores:
        print(
            f"{score['label']}: layers={score['layers']} lr={score['lr']} "
            f"epochs={score['epochs']} test_acc={score['test_acc']:.4f}"
        )

    return final_scores


if __name__ == "__main__":
    run_experiments()