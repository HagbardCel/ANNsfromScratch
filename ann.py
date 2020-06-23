import numpy as np
from .layers import Layer
from typing import List, Type, Optional

__all__ = ['ANN']


class ANN:
    def __init__(self, layers: List[Type[Layer]]) -> None:
        """
        Class implementing an artificial neural network
        :param Layers:
        """
        self.layers = layers
        self.first = self.layers[0]
        self.last = self.layers[-1]

    def train(self, data: np.array, target: np.array, batch_size: int, epochs: int,
              verbose: Optional[int] = True) -> None:
        """
        Train neural network

        :param data:
        :param target:
        :param batch_size:
        :param epochs:
        :param verbose:
        :return:
        """
        n = data.shape[0]
        n_batches = n // batch_size + 1
        for epoch in range(epochs):
            for batch in range(n_batches):
                # Beginning and end of respective batch
                indices = (batch * batch_size, min((batch + 1) * batch_size, n))

                # Stack for forward-pass outputs:
                forward_stack = list()
                # Perform forward pass
                for layer in self.layers:
                    if layer == self.first:
                        inputs = tuple(data[indices[0]:indices[1], :])
                    elif layer == self.last:
                        inputs = tuple(forward_out, target[indices[0]:indices[1], :])
                    else:
                        inputs = tuple(forward_out)
                    forward_out = layer.forward_pass(*inputs)
                    forward_stack.append(forward_out)

                # Perform backward pass:
                reverse_out = tuple()
                for layer in reversed(self.layers):
                    reverse_out = tuple(layer.reverse_pass(reverse_out))

            if verbose:
                print("Epoch {}/{}, Loss: {}".format(epoch, epochs, forward_out))
