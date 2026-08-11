import numpy as np


class Accuracy:

    @staticmethod
    def calculate(
        y_true,
        y_pred
    ):

        predictions = (
            y_pred > 0.5
        ).astype(int)

        return np.mean(
            predictions == y_true
        )