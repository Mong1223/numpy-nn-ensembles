"""Бэггинг и бустинг (AdaBoost) поверх собственной нейросети."""

from __future__ import annotations

import numpy as np

from .network import NeuralNetwork


class BaggingNN:
    """Бэггинг: каждая сеть учится на бутстрэп-выборке, ответ — голосование большинством."""

    def __init__(self, n_estimators: int = 5, random_state: int | None = None, **nn_params) -> None:
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.nn_params = nn_params

    def fit(self, X: np.ndarray, y: np.ndarray) -> BaggingNN:
        X, y = np.asarray(X, float), np.asarray(y, int)
        rng = np.random.default_rng(self.random_state)
        self.estimators_ = []
        for _ in range(self.n_estimators):
            idx = rng.integers(0, len(X), size=len(X))
            seed = int(rng.integers(1 << 31))
            model = NeuralNetwork(random_state=seed, **self.nn_params).fit(X[idx], y[idx])
            self.estimators_.append(model)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return np.mean([m.predict(X) for m in self.estimators_], axis=0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X) >= 0.5).astype(int)


class AdaBoostNN:
    """AdaBoost: следующая сеть сильнее учитывает примеры, на которых ошиблись предыдущие.

    Веса примеров передаются в функцию потерь сети, итоговый ответ — голосование,
    взвешенное по качеству каждой сети (alpha).
    """

    def __init__(self, n_estimators: int = 5, random_state: int | None = None, **nn_params) -> None:
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.nn_params = nn_params

    def fit(self, X: np.ndarray, y: np.ndarray) -> AdaBoostNN:
        X, y = np.asarray(X, float), np.asarray(y, int)
        rng = np.random.default_rng(self.random_state)
        weights = np.full(len(X), 1.0 / len(X))
        self.estimators_, self.alphas_ = [], []
        for _ in range(self.n_estimators):
            seed = int(rng.integers(1 << 31))
            model = NeuralNetwork(random_state=seed, **self.nn_params).fit(X, y, weights)
            miss = model.predict(X) != y
            error = np.clip(weights[miss].sum(), 1e-10, 1 - 1e-10)
            if error >= 0.5:
                break
            alpha = 0.5 * np.log((1 - error) / error)
            weights *= np.exp(np.where(miss, alpha, -alpha))
            weights /= weights.sum()
            self.estimators_.append(model)
            self.alphas_.append(alpha)
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        pairs = zip(self.alphas_, self.estimators_, strict=True)
        votes = [alpha * (2 * model.predict(X) - 1) for alpha, model in pairs]
        return np.sum(votes, axis=0) / np.sum(self.alphas_)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.decision_function(X) >= 0).astype(int)
