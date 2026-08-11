"""
Classe principal da Rede Neural.
"""

class NeuralNetwork:

    def __init__(self):

        self.layers = []

    def add(self, layer):

        self.layers.append(layer)
    
    def forward(self, x):
        """
        Executa a propagação direta
        (Forward Propagation).

        A saída de uma camada torna-se
        a entrada da próxima.

        Fluxo:

            X
             ↓
            Layer 1
             ↓
            Layer 2
             ↓
            Layer N
             ↓
            ŷ

        Parâmetros
        ----------
        x : numpy.ndarray

            Dados de entrada.

        Retorna
        -------
        numpy.ndarray

            Predição da rede.
        """

        for layer in self.layers:
            x = layer.forward(x)

        return x
    
    def backward(self, grad):
        """
        Executa o Backpropagation.

        Percorre as camadas na ordem inversa
        propagando o gradiente do erro.

        Fluxo:

            dL/dY
              ↑
            Layer N
              ↑
            Layer N-1
              ↑
            Layer 1

        Parâmetros
        ----------
        grad : numpy.ndarray

            Gradiente inicial proveniente
            da função de perda.
        """

        for layer in reversed(
            self.layers
        ):
            grad = layer.backward(grad)
    
    def fit(
            self,
            X,
            y,
            epochs,
            loss_function,
            optimizer
        ):
        """
        Treina a rede neural.

        Processo executado em cada época:

        1. Forward Propagation
        2. Cálculo da Loss
        3. Cálculo do Gradiente da Loss
        4. Backpropagation
        5. Atualização dos Pesos

        Parâmetros
        ----------
        X : numpy.ndarray

            Dados de entrada.

        y : numpy.ndarray

            Saídas esperadas.

        epochs : int

            Número de épocas de treinamento.

        loss_function : Loss

            Função de perda.

            Exemplos:

                MSE
                CrossEntropy

        optimizer : Optimizer

            Algoritmo de otimização.

            Exemplos:

                SGD
                Momentum
                RMSProp
                Adam
                AdamW
                Nadam
        """
        for epoch in range(epochs):

            y_pred = self.forward(X)

            loss = loss_function.forward(
                y,
                y_pred
            )

            grad = (
                loss_function.derivative(
                    y,
                    y_pred
                )
            )

            self.backward(grad)

            for layer in self.layers:
                optimizer.update(layer)

            if epoch % 100 == 0:

                print(
                    f"Epoch {epoch} Loss={loss}"
                )