"""Нейросеть с одним скрытым слоем, реализованная на NumPy без фреймворков."""

from __future__ import annotations

import numpy as np


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


class NeuralNetwork:
    """Бинарный классификатор: вход -> скрытый слой (sigmoid) -> выход (sigmoid).

    Данные подаются в формате scikit-learn: X размера (n_samples, n_features),
    y из нулей и единиц размера (n_samples,). Внутри примеры лежат по столбцам.
    """

    def __init__(
        self,
        hidden_units: int = 13,
        learning_rate: float = 0.1,
        epochs: int = 5000,
        random_state: int | None = None,
    ) -> None:
        self.hidden_units = hidden_units
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.random_state = random_state
        self.loss_history: list[float] = []

    def _init_params(self, n_features: int) -> None:
        rng = np.random.default_rng(self.random_state)
        self.w1 = rng.standard_normal((self.hidden_units, n_features)) * 0.01
        self.b1 = np.zeros((self.hidden_units, 1))
        self.w2 = rng.standard_normal((1, self.hidden_units)) * 0.01
        self.b2 = np.zeros((1, 1))

    def _forward(self, X: np.ndarray) -> np.ndarray:
        self.z1 = self.w1 @ X + self.b1
        self.a1 = sigmoid(self.z1)
        self.z2 = self.w2 @ self.a1 + self.b2
        self.a2 = sigmoid(self.z2)
        return self.a2

    @staticmethod
    def _loss(y_hat: np.ndarray, y: np.ndarray, sample_weight: np.ndarray) -> float:
        eps = 1e-9
        y_hat = np.clip(y_hat, eps, 1 - eps)
        losses = -(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))
        return float(np.sum(losses * sample_weight))

    def _backward(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray) -> None:
        # Для sigmoid на выходе и бинарной кросс-энтропии dL/dz2 = y_hat - y
        delta2 = (self.a2 - y) * sample_weight
        self.dw2 = delta2 @ self.a1.T
        self.db2 = np.sum(delta2, axis=1, keepdims=True)

        delta1 = (self.w2.T @ delta2) * self.a1 * (1 - self.a1)
        self.dw1 = delta1 @ X.T
        self.db1 = np.sum(delta1, axis=1, keepdims=True)

    def _step(self) -> None:
        self.w1 -= self.learning_rate * self.dw1
        self.b1 -= self.learning_rate * self.db1
        self.w2 -= self.learning_rate * self.dw2
        self.b2 -= self.learning_rate * self.db2

    def fit(
        self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None
    ) -> NeuralNetwork:
        X = np.asarray(X, dtype=float).T
        y = np.asarray(y, dtype=float).reshape(1, -1)
        m = y.shape[1]
        # Веса примеров нормируются к сумме 1; без весов это обычное усреднение по выборке
        w = np.full(m, 1.0 / m) if sample_weight is None else np.asarray(sample_weight, float)
        w = (w / w.sum()).reshape(1, -1)

        self._init_params(X.shape[0])
        self.loss_history = []
        for epoch in range(self.epochs):
            y_hat = self._forward(X)
            if epoch % 100 == 0:
                self.loss_history.append(self._loss(y_hat, y, w))
            self._backward(X, y, w)
            self._step()
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._forward(np.asarray(X, dtype=float).T).ravel()

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X) >= 0.5).astype(int)
