from abc import abstractmethod

import numpy as np

from .module import Module


class LossFunction(Module):

    @abstractmethod
    def forward(self, y: np.ndarray, ygt: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def backward(self, y: np.ndarray, ygt: np.ndarray) -> np.ndarray:
        pass


class SoftMaxCrossEntropy(LossFunction):

    def forward(self, y: np.ndarray, ygt: np.ndarray) -> np.ndarray:
        softmax = np.exp(y) / (np.exp(y).sum(axis=1, keepdims=True) + 1e-6)
        return -np.log(softmax[np.arange(y.shape[0]), ygt] + 1e-6).mean()

    def backward(self, y: np.ndarray, ygt: np.ndarray) -> np.ndarray:
        softmax = np.exp(y) / (np.exp(y).sum(axis=1, keepdims=True) + 1e-6)
        softmax[np.arange(y.shape[0]), ygt] -= 1
        return softmax / y.shape[0]
