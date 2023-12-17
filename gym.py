from enum import Enum

import numpy as np
from sklearn.metrics import classification_report
from tqdm import tqdm

from model.loss_functions import LossFunction
from model.perceptron import ClassifierOutput


class TaskType(Enum):
    CLASSIFICATION = 'classification'
    REGRESSION = 'regression'


def split_reminder(x, chunk_size, axis=0):
    indices = np.arange(chunk_size, x.shape[axis], chunk_size)
    return np.array_split(x, indices, axis)


def test_classificator(model, X_test: np.ndarray, y_test: np.ndarray) -> str:
    batched_X = split_reminder(X_test, 1)
    preds = []
    for x in batched_X:
        res: ClassifierOutput = model.predict(x)
        preds.append(float(res.predictions))
    preds_array = np.array(preds)
    return classification_report(y_pred=preds_array, y_true=y_test)


class Gym:
    def __init__(self, model, loss_function: LossFunction):
        self.model = model
        self.loss_function = loss_function

    def fit(self,
            X_train: np.ndarray,
            y_train: np.ndarray,
            lr: float,
            epoch_count: int,
            batch_size: int) -> list[float]:
        train_size = X_train.shape[0]
        batched_X = split_reminder(X_train, batch_size)
        batched_y = split_reminder(y_train, batch_size)
        train_loss = []
        pbar = tqdm(range(epoch_count), desc='training')
        for _ in pbar:
            epoch_loss = []
            for X_batch, y_batch in zip(batched_X, batched_y):
                predicted = self.model.forward(X_batch)
                loss = self.loss_function.forward(predicted, y_batch)
                loss_grad = self.loss_function.backward(predicted, y_batch)
                self.model.backward(loss_grad)
                self.model.step(lr)
                epoch_loss.append(loss)
            avg_epoch_loss = np.sum(np.stack(epoch_loss)) / train_size

            tqdm_desc = f'avg_epoch_loss: {avg_epoch_loss}'
            pbar.set_description(desc=tqdm_desc)

            train_loss.append(avg_epoch_loss)
        return train_loss
