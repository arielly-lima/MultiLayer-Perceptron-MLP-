import os
import time
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from mlp import NeuralNetwork
from mlp.losses import cross_entropy

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def one_hot(labels, num_classes=10):
    encoded = np.zeros((labels.shape[0], num_classes), dtype=np.float32)
    encoded[np.arange(labels.shape[0]), labels] = 1.0
    return encoded


def load_mnist(test_size=10000, random_state=42):
    mnist = fetch_openml("mnist_784", version=1, as_frame=False)
    X = mnist["data"].astype(np.float32) / 255.0
    y = mnist["target"].astype(np.int64)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        train_size=60000,
        stratify=y,
        random_state=random_state,
    )
    return X_train, X_test, y_train, y_test


def batch_generator(X, y, batch_size=128):
    n = X.shape[0]
    perm = np.random.permutation(n)
    for i in range(0, n, batch_size):
        indices = perm[i : i + batch_size]
        yield X[indices], y[indices]


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
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    for epoch in range(1, epochs + 1):
        start = time.time()
        for X_batch, y_batch in batch_generator(X_train, y_train, batch_size):
            model.forward(X_batch)
            model.backward(y_batch, lr=lr)

        train_loss, train_acc = model.evaluate(X_train, y_train)
        val_loss, val_acc = model.evaluate(X_val, y_val)

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
    return history


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


def run_experiments():
    print("Carregando MNIST...")
    X_train, X_test, y_train, y_test = load_mnist()

    y_train_onehot = one_hot(y_train, num_classes=10)
    y_test_onehot = one_hot(y_test, num_classes=10)

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
            "lr": 0.03,
            "epochs": 20,
            "batch_size": 128,
        },
    ]

    histories = []
    final_scores = []

    for config in experiments:
        print("\n" + "=" * 60)
        print(f"Treinando {config['label']} | camadas={config['layers']} lr={config['lr']}\n")
        model = NeuralNetwork(config["layers"], seed=42)
        history = train_model(
            model,
            X_train,
            y_train_onehot,
            X_test,
            y_test_onehot,
            epochs=config["epochs"],
            lr=config["lr"],
            batch_size=config["batch_size"],
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
