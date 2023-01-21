from .hyperedableopt import Hyperedableopt
from .types import TFNumeric


class SGD(Hyperedableopt):
    def __init__(self, alpha: TFNumeric) -> None:
        super().__init__()
        self.set_hyperparameters(alpha)

    def step(self, gradients: TFNumeric, variables: TFNumeric) -> TFNumeric:
        new_variables = variables - gradients * self.alpha
        return new_variables

    def set_hyperparameters(self, hyperparameters: TFNumeric):
        self.alpha = hyperparameters

    def get_hyperparameters(self) -> TFNumeric:
        return self.alpha
