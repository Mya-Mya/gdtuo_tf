from .types import TFNumeric


class Hyperedableopt():
    def __init__(self) -> None:
        pass

    def step(self, gradients: TFNumeric, variables: TFNumeric) -> TFNumeric:
        raise NotImplementedError()

    def get_hyperparameters(self) -> TFNumeric:
        raise NotImplementedError()

    def set_hyperparameters(self, hyperparameters: TFNumeric) -> None:
        raise NotImplementedError()
