"""Сравнение собственных моделей с моделями scikit-learn на датасете Heart Disease.

Запуск: python -m nn_ensembles.evaluate [--data data/heart.csv] [--repeats 3]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from . import AdaBoostNN, BaggingNN, NeuralNetwork

DEFAULT_DATA = Path(__file__).resolve().parents[2] / "data" / "heart.csv"


def load_data(path: Path) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(path)
    # В исходном файле 1025 строк, но уникальных только 302: без удаления дублей
    # одни и те же пациенты попадают и в обучение, и в тест, и точность завышается.
    df = df.drop_duplicates()
    return df.drop(columns="target").to_numpy(float), df["target"].to_numpy(int)


def scores(model, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        return proba[:, 1] if proba.ndim == 2 else proba
    return model.decision_function(X)


def make_models(seed: int) -> dict:
    return {
        "NeuralNetwork (своя)": NeuralNetwork(random_state=seed),
        "BaggingNN (свой, 5 сетей)": BaggingNN(n_estimators=5, random_state=seed),
        "AdaBoostNN (свой, 5 сетей)": AdaBoostNN(n_estimators=5, random_state=seed),
        "LogisticRegression (sklearn)": LogisticRegression(max_iter=1000),
        "MLPClassifier (sklearn)": MLPClassifier(
            hidden_layer_sizes=(13,), activation="logistic", max_iter=5000, random_state=seed
        ),
        "GradientBoosting (sklearn)": GradientBoostingClassifier(random_state=seed),
    }


def evaluate(X: np.ndarray, y: np.ndarray, repeats: int = 3, seed: int = 42) -> pd.DataFrame:
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=repeats, random_state=seed)
    rows = []
    for fold, (train, test) in enumerate(cv.split(X, y)):
        scaler = StandardScaler().fit(X[train])
        X_train, X_test = scaler.transform(X[train]), scaler.transform(X[test])
        for name, model in make_models(seed + fold).items():
            model.fit(X_train, y[train])
            rows.append({
                "model": name,
                "accuracy": accuracy_score(y[test], model.predict(X_test)),
                "roc_auc": roc_auc_score(y[test], scores(model, X_test)),
            })
    table = pd.DataFrame(rows).groupby("model", sort=False).agg(["mean", "std"])
    return table


def to_markdown(table: pd.DataFrame) -> str:
    lines = ["| Модель | Accuracy | ROC-AUC |", "|---|---|---|"]
    for name, r in table.iterrows():
        lines.append(
            f"| {name} | {r[('accuracy', 'mean')]:.3f} ± {r[('accuracy', 'std')]:.3f} "
            f"| {r[('roc_auc', 'mean')]:.3f} ± {r[('roc_auc', 'std')]:.3f} |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    X, y = load_data(args.data)
    print(f"Уникальных пациентов: {len(y)}, признаков: {X.shape[1]}")
    print(f"Кросс-валидация: 5 фолдов x {args.repeats} повтора\n")
    print(to_markdown(evaluate(X, y, args.repeats)))


if __name__ == "__main__":
    main()
