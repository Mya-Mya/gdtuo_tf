from .hyperedableopt import Hyperedableopt
from .types import TFNumeric
from tensorflow import Variable, zeros_like, sqrt


class Adam(Hyperedableopt):
    def __init__(self, alpha: TFNumeric, beta1: TFNumeric, beta2: TFNumeric) -> None:
        super().__init__()
        h = Variable([alpha, beta1, beta2])
        self.set_hyperparameters(h)
        self.momentums = None
        self.velocities = None
        self.epsilon = 1e-7

    def step(self, gradients: TFNumeric, variables: TFNumeric) -> TFNumeric:
        h = self.get_hyperparameters()
        alpha = h[0]
        beta1 = h[1]
        beta2 = h[2]

        if self.momentums is None:
            self.momentums = zeros_like(gradients)
        else:
            self.momentums = beta1*self.momentums + (1.-beta1)*gradients

        if self.velocities is None:
            self.velocities = zeros_like(gradients)
        else:
            self.velocities = beta2*self.velocities + (1.-beta2)*(gradients**2)

        hatm = self.momentums/(1.-beta1)
        hatv = self.velocities/(1.-beta2)
        new_variables = variables - alpha*hatm/(sqrt(hatv)+self.epsilon)
        return new_variables

    def set_hyperparameters(self, hyperparameters: TFNumeric) -> None:
        self.h = hyperparameters

    def get_hyperparameters(self) -> TFNumeric:
        return self.h
