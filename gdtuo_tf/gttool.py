from tensorflow import GradientTape
from .types import TFNumeric


def start_watch(variable: TFNumeric) -> GradientTape:
    gt = GradientTape(persistent=True)
    gt.__enter__()
    gt.watch(variable)
    return gt


def get_grad(gt: GradientTape, y: TFNumeric, x: TFNumeric) -> TFNumeric:
    grad = gt.gradient(y, x)
    gt.__exit__(None, None, None)
    return grad
