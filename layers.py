import numpy as np
from typing import Union, Callable, Tuple

__all__ = ['LossLayer', 'FCLayer']

class Layer:
    def __init__(self, activation: Union[str, Tuple[Callable[[np.array], np.array],
                                                    Callable[[np.array], np.array]]]) -> None:
        """
        Abstract class defining various methods and attributes shared by different Layer types:w

        :param activation:
        Activation function, one out of ['sigmoid', ...] or a tuple of callables with the first being
        the activation function and the second its derivative
        """

        self._sigmoid = np.vectorize(lambda x: 1 / (1 + np.exp(-x)))
        self._sigmoid_der = np.vectorize(lambda x: np.exp(-x) / (1 + np.exp(-x)) ** 2)

        if isinstance(activation, str):
            activations = {'sigmoid': (self._sigmoid, self._sigmoid_der)}
            self.activation = activations[activation]
        else:
            self.activation = activation

    def forward_pass(self, inputs: Tuple[np.array]) -> np.array:
        pass

    def reverse_pass(self, inputs: Tuple[np.array]) -> np.array:
        pass


class LossLayer(Layer):

    def __init__(self, loss: Union[str, Tuple[Callable[[np.array], np.array],
                                              Callable[[np.array], np.array]]]) -> None:
        """
        Class implementing various loss functions
        :param loss:
        Loss function, one out of ['MAE', 'RMSE', ...]
        """

        super().__init(activation=loss)
        self.loss_function = {
            'RMSE': np.vectorize(lambda x, y: np.sqrt((x - y) ** 2))
        }[loss]

    def forward_pass(self, inputs: Tuple[np.array]):
        return self.lossfunction[0](inputs[0], inputs[1])


class FCLayer(Layer):
    def __init__(self, n_inputs: int, n_outputs: int, activation='sigmoid') -> None:
        super().__init__(activation)
        self.weights = np.random.rand(shape=(n_inputs + 1, n_outputs))
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs

    def forward_pass(self, inputs: Tuple[np.array]) -> np.array:
        return self.activation(np.concat([np.ones(shape=self.n_outputs), inputs]) @ self.weights)

    def reverse_pass(self, inputs: Tuple[np.array]) -> np.array:
        pass
