from abc import ABCMeta, abstractmethod

import numpy as np


class Module(metaclass=ABCMeta):

    def __call__(self, *args, **kwargs) -> np.ndarray:
        return self.forward(*args, **kwargs)

    @abstractmethod
    def forward(self, *args, **kwargs) -> np.ndarray:
        pass

    @abstractmethod
    def backward(self, *args, **kwargs) -> np.ndarray:
        pass
