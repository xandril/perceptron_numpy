from abc import abstractmethod

import numpy as np

from .module import Module


class ActivationFunction(Module):
    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def backward(self, x: np.ndarray) -> np.ndarray:
        pass


class ReLU(ActivationFunction):
    def __init__(self):
        self._activation_value = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._activation_value = np.maximum(0, x)
        return self._activation_value

    def backward(self, x: np.ndarray) -> np.ndarray:
        return x * (self._activation_value > 0).astype(float)
