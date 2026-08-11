import pickle


class ModelCheckpoint:

    @staticmethod
    def save(
        model,
        filepath
    ):

        with open(
            filepath,
            "wb"
        ) as file:

            pickle.dump(
                model,
                file
            )

    @staticmethod
    def load(filepath):

        with open(
            filepath,
            "rb"
        ) as file:

            return pickle.load(
                file
            )