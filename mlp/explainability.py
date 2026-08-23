import shap
import matplotlib.pyplot as plt
import numpy as np


def generate_shap_summary(
    model,
    X_background,
    X_explain,
    feature_names
):
    """
    Gera o SHAP Summary Plot (Beeswarm).

    Parâmetros
    ----------
    model : NeuralNetwork
        Modelo MLP treinado.

    X_background : numpy.ndarray
        Amostras utilizadas como referência pelo SHAP.

    X_explain : numpy.ndarray
        Amostras que serão explicadas.

    feature_names : list
        Nomes das features.
    """

    #função que o SHAP utilizará para realizar as previsões
    def predict(data):

        predictions = model.forward(data)

        return predictions.reshape(-1)

    explainer = shap.KernelExplainer(
        predict,
        X_background
    )

    shap_values = explainer.shap_values(
        X_explain
    )

    fig, ax = plt.subplots(
        figsize=(7, 4)
    )

    shap.summary_plot(
        shap_values,
        X_explain,
        feature_names=feature_names,
        show=False
    )

    plt.tight_layout()

    return fig