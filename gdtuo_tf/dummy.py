from .optimizableoptimizer import OptimizableOptimizer
from .types import TFNumeric


class Dummy(OptimizableOptimizer):
    """
    Does not optimize anything, be optimized anything.
    Useful for hopt in Hyperoptimizer when you want to run single optimizer.
    """
    def __init__(self) -> None:
        super().__init__()

    def step(self, gradients: TFNumeric, variables: TFNumeric) -> TFNumeric:
        return variables

    def set_hyperparameters(self, hyperparameters: TFNumeric):
        pass

    def get_hyperparameters(self) -> TFNumeric:
        return None
