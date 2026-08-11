#Calcular erro
import numpy as np

class MSE:

    def forward(
        self,
        y_true,
        y_pred
    ):

        return np.mean(
            (y_true - y_pred)**2
        )

    def derivative(
        self,
        y_true,
        y_pred
    ):

        return (
            2 *
            (y_pred - y_true)
            / y_true.shape[0]
        )
    
class CrossEntropy:

    def forward(
        self,
        y_true,
        y_pred
    ):

        y_pred = np.clip(
            y_pred,
            1e-15,
            1 - 1e-15
        )

        return -np.mean(
            np.sum(
                y_true *
                np.log(y_pred),
                axis=1
            )
        )

    def derivative(
        self,
        y_true,
        y_pred
    ):

        return y_pred - y_true