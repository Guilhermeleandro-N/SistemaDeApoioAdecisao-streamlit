import numpy as np
"""
Módulo responsável pela inicialização dos pesos da rede neural.
"""

class HeInitializer:
    """
    Inicialização de He (He et al., 2015) - Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification.

    Recomendada principalmente para:

    - ReLU
    - Leaky ReLU
    - ELU
    - GELU
    - Mish
    - Swish

    A ideia é manter a variância das ativações
    aproximadamente constante ao longo das camadas.

    Fórmula:

        W ~ N(0, 2 / n)

    onde:

        n = número de entradas do neurônio
    """

    @staticmethod
    def initialize(input_size, output_size):

        return np.random.randn(
            input_size,
            output_size
        ) * np.sqrt(2 / input_size)
    
class XavierInitializer:
    """
    Inicialização de Xavier
    (Glorot & Bengio, 2010) - Understanding the difficulty of training deep feedforward neural networks.

    Recomendada principalmente para:

    - Sigmoid
    - Tanh

    O objetivo é evitar que os sinais
    aumentem ou diminuam excessivamente
    ao atravessar várias camadas.

    Fórmula simplificada:

        W ~ N(0, 1 / n)

    onde:

        n = número de entradas da camada
    """

    @staticmethod
    def initialize(input_size, output_size):

        return np.random.randn(
            input_size,
            output_size
        ) * np.sqrt(
            1 / input_size
        )