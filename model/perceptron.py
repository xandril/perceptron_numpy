import pickle
from dataclasses import dataclass
from functools import reduce
from pathlib import Path

import numpy as np

from .activation_functions import ActivationFunction
from .layers import DenseLayer
from .module import Module


class Perceptron(Module):
    def __init__(self, layers: list[ActivationFunction | DenseLayer]):
        self._layers = layers

    def store_model(self, store_path: Path):
        with open(store_path, 'wb') as f:
            pickle.dump(self._layers, f)

    @classmethod
    def load_model(cls, store_path: Path) -> 'Module':
        with open(store_path, 'rb') as f:
            layers = pickle.load(f)
        return cls(layers)

    def forward(self, x: np.ndarray) -> np.ndarray:
        layers = (layer.forward for layer in self._layers)

        res = reduce(lambda layer_input, layer_forward: layer_forward(layer_input),
                     layers,
                     x)
        return res

    def backward(self, output_grad: np.ndarray) -> np.ndarray:
        layers_grads = (layer.backward for layer in self._layers[::-1])

        res = reduce(lambda layer_output_grad, layer_backward: layer_backward(layer_output_grad),
                     layers_grads,
                     output_grad)
        return res

    def step(self, lr: float):
        for layer in self._layers:
            if hasattr(layer, 'step'):
                layer.step(lr)

    def predict(self, x: np.ndarray):
        return self.forward(x)


@dataclass(frozen=True)
class ClassifierOutput:
    predictions: np.ndarray
    probas: np.ndarray


class PerceptronClassifier(Perceptron):
    def predict(self, x: np.ndarray) -> ClassifierOutput:
        probas = self.predict_proba(x)
        predictions = np.argmax(probas, axis=1)
        return ClassifierOutput(predictions, probas)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)
