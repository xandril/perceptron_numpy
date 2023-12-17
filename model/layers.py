from abc import abstractmethod

import numpy as np

from .module import Module


class Layer(Module):
    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def backward(self, x: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def step(self, lr: float) -> None:
        pass


class DenseLayer(Layer):
    @staticmethod
    def _init_bias_data(bias: bool, layer_size: int) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
        bias_values = None
        bias_grads = None
        if bias:
            bias_values = np.random.randn(layer_size) / 10.
            bias_grads = np.empty_like(bias_values)
        return bias_values, bias_grads

    @staticmethod
    def _init_weights_data(input_size: int, output_size: int) -> tuple[np.ndarray, np.ndarray]:
        weights = np.random.randn(input_size, output_size) / 10.  # shape (input, output)
        weights_grads = np.empty_like(weights)  # shape (input, output)
        return weights, weights_grads

    def __init__(self, input_size: int, output_size: int, bias: bool):
        self._weights, self._weights_grads = self._init_weights_data(input_size, output_size)
        self._layer_input = None
        self._biases, self._biases_grads = self._init_bias_data(bias, output_size)

    def _is_bias(self) -> bool:
        return self._biases is not None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._layer_input = x
        linear_value = self._layer_input @ self._weights
        if self._is_bias():
            linear_value += self._biases
        return linear_value

    def backward(self, x: np.ndarray) -> np.ndarray:
        output_grad = x
        if self._is_bias():
            self._biases_grads = (np.ones(shape=(1, x.shape[0])) @ output_grad).squeeze()
        self._weights_grads = self._layer_input.T @ output_grad
        return output_grad @ self._weights.T

    def step(self, lr: float) -> None:
        self._weights -= self._weights_grads * lr
        if self._biases is not None:
            self._biases -= self._biases_grads * lr
