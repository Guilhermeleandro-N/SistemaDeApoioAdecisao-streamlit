import itertools

from mlp.metrics import Accuracy


class ExperimentResult:

    def __init__(
        self,
        params,
        score,
        loss
    ):

        self.params = params
        self.score = score
        self.loss = loss


class GridSearch:
    """
    Grid Search para Redes Neurais.

    Testa todas as combinações possíveis
    dos hiperparâmetros informados.
    """

    def __init__(
        self,
        network_builder,
        param_grid,
        metric=Accuracy
    ):

        self.network_builder = (
            network_builder
        )

        self.param_grid = param_grid

        self.metric = metric

        self.results = []

        self.best_score = -1

        self.best_loss = float("inf")

        self.best_params = None

        self.best_model = None

    def generate_combinations(self):

        keys = list(
            self.param_grid.keys()
        )

        values = list(
            self.param_grid.values()
        )

        combinations = itertools.product(
            *values
        )

        for combo in combinations:

            yield dict(
                zip(
                    keys,
                    combo
                )
            )

    def fit(
        self,
        X,
        y
    ):

        experiment_number = 1

        for params in self.generate_combinations():

            print(
                f"\nExperimento {experiment_number}"
            )

            experiment_number += 1

            current_params = (
                params.copy()
            )

            network = (
                self.network_builder(
                    current_params
                )
            )

            optimizer = (
                current_params[
                    "optimizer"
                ]()
            )

            loss_function = (
                current_params[
                    "loss"
                ]()
            )

            network.fit(
                X,
                y,
                epochs=current_params[
                    "epochs"
                ],
                loss_function=loss_function,
                optimizer=optimizer
            )

            predictions = (
                network.forward(X)
            )

            score = (
                self.metric.calculate(
                    y,
                    predictions
                )
            )

            loss = (
                loss_function.forward(
                    y,
                    predictions
                )
            )

            self.results.append(
                ExperimentResult(
                    self.pretty_params(
                        current_params
                    ),
                    score,
                    loss
                )
            )

            print(
                f"Accuracy = {score:.4f}"
            )

            print(
                f"Loss = {loss:.6f}"
            )

            #
            # Critério de desempate
            #
            # 1) maior accuracy
            # 2) menor loss
            #

            if (
                score > self.best_score
                or
                (
                    score == self.best_score
                    and loss < self.best_loss
                )
            ):

                self.best_score = score

                self.best_loss = loss

                self.best_params = (
                    current_params.copy()
                )

                #
                # Guarda a rede treinada
                #
                self.best_model = network

        return self

    def summary(self):

        print("\n")
        print("=" * 60)
        print("RESULTADOS")
        print("=" * 60)

        for result in self.results:

            print(
                f"Accuracy={result.score:.4f}"
            )

            print(
                f"Loss={result.loss:.6f}"
            )

            print(
                result.params
            )

            print()

        print("=" * 60)

        print(
            f"Melhor Accuracy = "
            f"{self.best_score:.4f}"
        )

        print(
            f"Melhor Loss = "
            f"{self.best_loss:.6f}"
        )

        print(
            self.pretty_params(
                self.best_params
            )
        )

    def pretty_params(
        self,
        params
    ):

        pretty = {}

        for key, value in params.items():

            if key == "activation":

                pretty[key] = (
                    value.__name__
                )

            elif key == "optimizer":

                pretty[key] = (
                    value().__class__.__name__
                )

            elif key == "loss":

                pretty[key] = (
                    value.__name__
                )

            else:

                pretty[key] = value

        return pretty