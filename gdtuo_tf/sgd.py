from .optimizableoptimizer import OptimizableOptimizer
from .types import TFNumeric
from tensorflow import Variable


class SGD(OptimizableOptimizer):
    def __init__(self, alpha: float = 0.01) -> None:
        super().__init__()
        alpha = Variable(float(alpha))
        self.set_hyperparameters(alpha)

    def step(self, gradients: TFNumeric, variables: TFNumeric) -> TFNumeric:
        new_variables = variables - gradients * self.alpha
        return new_variables

    def set_hyperparameters(self, hyperparameters: TFNumeric):
        self.alpha = hyperparameters

    def get_hyperparameters(self) -> TFNumeric:
        return self.alpha
