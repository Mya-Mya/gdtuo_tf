from .hyperedableopt import Hyperedableopt
from .types import TFNumeric
from .gttool import start_watch, get_grad
from tensorflow import GradientTape, zeros_like


class Hyperoptimizer:
    """
    The optimizer composites 2 optimizers;
    the optimizer of variables, and the optimizer of hyper parameters.
    """
    def __init__(self, vopt: Hyperedableopt, hopt: Hyperedableopt) -> None:
        """
        Parameters
        ----------
        vopt : Hyperedableopt
            Optimizes the variable you want to optimize.
        hopt : Hyperedableopt
            Optimizes vopt's hyper parameters. 
        """
        self.vopt = vopt
        self.hopt = hopt
        self.vgt = None
        self.hgt = None

    def get_hyperparameters_grad(self, fv: TFNumeric, h: TFNumeric) -> TFNumeric:
        if self.hgt is None:
            return zeros_like(h)
        return get_grad(self.hgt, fv, h)

    def get_variables_grad(self, fv: TFNumeric, v: TFNumeric) -> TFNumeric:
        if self.vgt is None:
            return zeros_like(v)
        return get_grad(self.vgt, fv, v)

    def step(self, fv: TFNumeric, v: TFNumeric) -> TFNumeric:
        """
        Updates your variables and hyper parameters of the optimizer according to given cost function.
        Parameters
        ----------
        fv : Tensor | Variable
            The values of your cost function, fv = f(v).
        v : Tensor | Variable
            The variable you want to optimize.
        
        Returns
        -------
        new_v : Tensor | Variable
            Updated variables.
        """
        h = self.vopt.get_hyperparameters()
        gh = self.get_hyperparameters_grad(fv, h)  # gh←fvのhに対する微分
        new_h = self.hopt.step(gh, h)  # ghに基づいてhを更新する
        self.vopt.set_hyperparameters(new_h)
        self.hgt = start_watch(new_h)  # hの追跡を始める
        gv = self.get_variables_grad(fv, v)  # gv←fvのvに対する微分
        new_v = self.vopt.step(gv, v)  # gvに基づいてvを更新する
        self.vgt = start_watch(new_v)  # vの追跡を始める
        return new_v
