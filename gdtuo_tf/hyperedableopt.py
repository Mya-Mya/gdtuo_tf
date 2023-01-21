from .types import TFNumeric


class Hyperedableopt():
    """
    An abstract optimizer whose hyper parameters can be optimized.
    In order to calculate the gradient of updated variables on hyper parameters,
    there are some notes on implementation not to disconnect some necessary computation nodes.
    """
    def __init__(self) -> None:
        pass

    def step(self, gradients: TFNumeric, variables: TFNumeric) -> TFNumeric:
        """
        Updates variables according to given gradients.

        Parameters
        ----------
        gradients : Tensor | Variable
            The gradients of your cost function on variables.
        variables : Tensor | Variable
            The variables you want to optimize.
        
        Returns
        -------
        new_variables : Tensor | Variable
            Updated variables.

        Note that the given variables MUST NOT BE CHANGED DESTRUCTIVELY.
        new_variables must be the different object than given variables.
        And the hyper parameters used in this method MUST BE THE SAME OBJECT GIVEN IN set_hyperparameters.
        """
        raise NotImplementedError()

    def get_hyperparameters(self) -> TFNumeric:
        """
        Retrieves the hyper parameters object.
        
        Returns
        -------
        hyperparameters : Tensor | Variable
            Hyper parameters.
        
        Note that the hyperparameters MUST BE THE SAME OBJECT USED IN step. 
        """
        raise NotImplementedError()

    def set_hyperparameters(self, hyperparameters: TFNumeric) -> None:
        """
        Set the hyper parameters object.

        Parameters
        ----------
        hyperparameters : Tensor | Variable
            New hyper parameters.
        """
        raise NotImplementedError()
