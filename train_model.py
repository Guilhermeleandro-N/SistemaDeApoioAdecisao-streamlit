import pickle

from datasets.adult import load_and_prepare_data

from mlp.network import NeuralNetwork
from mlp.layers import Dense

from mlp.activations import (
    ReLU,
    Sigmoid
)

from mlp.losses import MSE

from mlp.optimizers import Adam

from mlp.metrics import Accuracy

from mlp.checkpoint import ModelCheckpoint


# ============================================================
# CONFIGURAÇÕES DO MODELO
# ============================================================

NEURONS = 16

EPOCHS = 100

LEARNING_RATE = 0.001


# ============================================================
# CONSTRUÇÃO DA REDE
# ============================================================

def build_network(input_size):

    network = NeuralNetwork()

    # Camada oculta
    network.add(
        Dense(
            input_size=input_size,
            output_size=NEURONS,
            activation=ReLU()
        )
    )

    # Camada de saída
    network.add(
        Dense(
            input_size=NEURONS,
            output_size=1,
            activation=Sigmoid()
        )
    )

    return network


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("TREINAMENTO DO MODELO ADULT INCOME")
    print("=" * 60)

    # --------------------------------------------------------
    # Carregar e preparar dados
    # --------------------------------------------------------

    (
        X_train,
        X_test,
        y_train,
        y_test,
        preprocessing
    ) = load_and_prepare_data()

    print("\n")
    print("=" * 60)
    print("DADOS")
    print("=" * 60)

    print(
        f"X_train: {X_train.shape}"
    )

    print(
        f"X_test:  {X_test.shape}"
    )

    # --------------------------------------------------------
    # Criar modelo
    # --------------------------------------------------------

    network = build_network(
        input_size=X_train.shape[1]
    )

    print("\n")
    print("=" * 60)
    print("ARQUITETURA")
    print("=" * 60)

    print(
        f"Entrada: {X_train.shape[1]}"
    )

    print(
        f"Camada oculta: {NEURONS} neurônios - ReLU"
    )

    print(
        "Saída: 1 neurônio - Sigmoid"
    )

    print(
        f"Épocas: {EPOCHS}"
    )

    print(
        f"Learning rate: {LEARNING_RATE}"
    )

    print(
        "Otimizador: Adam"
    )

    print(
        "Loss: MSE"
    )

    # --------------------------------------------------------
    # Treinamento
    # --------------------------------------------------------

    print("\n")
    print("=" * 60)
    print("INICIANDO TREINAMENTO")
    print("=" * 60)

    optimizer = Adam(
        lr=LEARNING_RATE
    )

    network.fit(
        X_train,
        y_train,
        epochs=EPOCHS,
        loss_function=MSE(),
        optimizer=optimizer
    )

    # --------------------------------------------------------
    # Avaliação no treino
    # --------------------------------------------------------

    train_predictions = network.forward(
        X_train
    )

    train_accuracy = Accuracy.calculate(
        y_train,
        train_predictions
    )

    # --------------------------------------------------------
    # Avaliação no teste
    # --------------------------------------------------------

    test_predictions = network.forward(
        X_test
    )

    test_accuracy = Accuracy.calculate(
        y_test,
        test_predictions
    )

    # --------------------------------------------------------
    # Resultados
    # --------------------------------------------------------

    print("\n")
    print("=" * 60)
    print("RESULTADOS")
    print("=" * 60)

    print(
        f"Accuracy treino: "
        f"{train_accuracy:.4f}"
    )

    print(
        f"Accuracy teste: "
        f"{test_accuracy:.4f}"
    )

    # --------------------------------------------------------
    # Salvar modelo
    # --------------------------------------------------------

    print("\n")
    print("=" * 60)
    print("SALVANDO MODELO")
    print("=" * 60)

    ModelCheckpoint.save(
        network,
        "models/adult_mlp.pkl"
    )

    # --------------------------------------------------------
    # Salvar pré-processamento
    # --------------------------------------------------------

    with open(
        "models/preprocessing.pkl",
        "wb"
    ) as file:

        pickle.dump(
            preprocessing,
            file
        )

    print(
        "Modelo salvo em:"
    )

    print(
        "models/adult_mlp.pkl"
    )

    print(
        "\nPré-processamento salvo em:"
    )

    print(
        "models/preprocessing.pkl"
    )

    print("\nTreinamento concluído.")


if __name__ == "__main__":
    main()