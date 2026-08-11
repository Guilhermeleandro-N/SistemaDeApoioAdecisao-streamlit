#Representar uma camada da rede
import numpy as np
from mlp.initializers import HeInitializer

class Dense:
    """
    Representa uma camada totalmente conectada (Fully Connected Layer).

    Cada neurônio da camada recebe como entrada todas as saídas
    da camada anterior.

    A camada é responsável por:
    - armazenar pesos e bias;
    - realizar a propagação direta (forward);
    - calcular gradientes durante o backpropagation.
    """

    def __init__(
        self,
        input_size,
        output_size,
        activation
    ):
        """
        Inicializa a camada.

        Parâmetros:
        ----------
        input_size : int
            Quantidade de neurônios da camada anterior.

        output_size : int
            Quantidade de neurônios desta camada.

        activation : Activation
            Função de ativação utilizada pela camada.
        """

        self.weights = (
            HeInitializer.initialize(
                input_size,
                output_size
            )
        )

        self.bias = np.zeros(
            (1, output_size)
        )

        self.activation = activation

    def forward(self, x):
        """
        Executa a propagação direta (Forward Propagation).

        Entrada:
            x

        Cálculos:

            Z = XW + b

            A = f(Z)

        Onde:

            X = entradas
            W = pesos
            b = bias
            f = função de ativação

        Retorna:
            saída da camada após a ativação.
        """

        self.input = x

        self.z = (
            np.dot(
                x,
                self.weights
            )
            + self.bias
        )

        self.output = (
            self.activation.forward(
                self.z
            )
        )

        return self.output
    
    def backward(self, grad_output):
        """
        Executa o Backpropagation.

        Entrada:
            grad_output

        Representa:

            dL/dA

        ou seja, o gradiente da função de perda
        em relação à saída da camada.

        Objetivo:
            calcular:

            dL/dW
            dL/db
            dL/dX

        Retorna:
            grad_input (dL/dX)
        """
        
        dz = (
            grad_output *
            self.activation.derivative(
                self.z
            )
        )

        self.grad_weights = np.dot(
            self.input.T,
            dz
        )

        self.grad_bias = np.sum(
            dz,
            axis=0,
            keepdims=True
        )

        grad_input = np.dot(
            dz,
            self.weights.T
        )

        return grad_input