from datasets.adult import (
    X_train,
    X_test,
    y_train,
    y_test
)

from mlp.network import NeuralNetwork
from mlp.layers import Dense

from mlp.activations import (
    ReLU,
    Tanh,
    Sigmoid
)

from mlp.losses import MSE

from mlp.optimizers import (
    Adam
)

from mlp.grid_search import GridSearch

from mlp.metrics import Accuracy


def build_network(params):
    """
    Constrói uma rede com os parâmetros
    recebidos pelo Grid Search.
    """

    network = NeuralNetwork()

    activation = params["activation"]()

    network.add(
        Dense(
            input_size=X_train.shape[1],
            output_size=params["neurons"],
            activation=activation
        )
    )

    network.add(
        Dense(
            input_size=params["neurons"],
            output_size=1,
            activation=Sigmoid()
        )
    )

    return network


def main():

    print("=" * 60)
    print("GRID SEARCH - ADULT INCOME")
    print("=" * 60)

    print("\nDados:")
    print(f"X_train: {X_train.shape}")
    print(f"y_train: {y_train.shape}")
    print(f"X_test:  {X_test.shape}")
    print(f"y_test:  {y_test.shape}")

    param_grid = {

        "neurons": [
            8,
            16
        ],

        "activation": [
            ReLU,
            Tanh
        ],

        "optimizer": [
            lambda: Adam(
                lr=0.001
            )
        ],

        "epochs": [
            100
        ],

        "loss": [
            MSE
        ]
    }

    search = GridSearch(
        network_builder=build_network,
        param_grid=param_grid
    )

    print("\n")
    print("=" * 60)
    print("INICIANDO GRID SEARCH")
    print("=" * 60)

    search.fit(
        X_train,
        y_train
    )

    print("\n")
    print("=" * 60)
    print("MELHOR CONFIGURAÇÃO")
    print("=" * 60)

    print(
        f"Accuracy de treino: "
        f"{search.best_score:.4f}"
    )

    print(
        f"Loss de treino: "
        f"{search.best_loss:.6f}"
    )

    print(
        f"Parâmetros: "
        f"{search.best_params}"
    )

    # ==========================================
    # Avaliação no conjunto de teste
    # ==========================================

    print("\n")
    print("=" * 60)
    print("AVALIAÇÃO NO CONJUNTO DE TESTE")
    print("=" * 60)

    best_network = search.best_model

    predictions = best_network.forward(
        X_test
    )

    test_accuracy = Accuracy.calculate(
        y_test,
        predictions
    )

    print(
        f"\nAccuracy no teste: "
        f"{test_accuracy:.4f}"
    )

    classes = (
        predictions > 0.5
    ).astype(int)

    print("\nPrimeiras 20 predições:")
    print(predictions[:20])

    print("\nPrimeiras 20 classes previstas:")
    print(classes[:20])

    print("\nPrimeiras 20 classes reais:")
    print(y_test[:20])


if __name__ == "__main__":
    main()