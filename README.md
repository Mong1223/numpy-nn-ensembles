# numpy-nn-ensembles

[![CI](https://github.com/Mong1223/numpy-nn-ensembles/actions/workflows/ci.yml/badge.svg)](https://github.com/Mong1223/numpy-nn-ensembles/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

Нейросеть и ансамбли **бэггинг** и **AdaBoost**, написанные с нуля на NumPy, без
PyTorch и без `sklearn.ensemble`. Задача — предсказать болезнь сердца по 13
медицинским признакам. Проект вырос из лабораторной работы по дисциплине «Методы
вычислительного интеллекта» (ТПУ, магистратура, 2022).

## Что реализовано

- **`NeuralNetwork`** — сеть с одним скрытым слоем: прямой проход, бинарная
  кросс-энтропия, обратное распространение ошибки и градиентный спуск вручную.
  Поддерживает веса примеров, они нужны для бустинга.
- **`BaggingNN`** — каждая сеть учится на своей бутстрэп-выборке, ответ —
  голосование большинством.
- **`AdaBoostNN`** — следующая сеть сильнее учитывает примеры, на которых ошиблись
  предыдущие. Итог — голосование, взвешенное по качеству каждой сети.
- Интерфейс как у scikit-learn: `fit`, `predict`, `predict_proba`.

Правильность обратного распространения проверяется тестом: аналитический градиент
сравнивается с численным.

## Результаты

5-кратная стратифицированная кросс-валидация, 3 повтора, масштабирование признаков
внутри каждого фолда:

| Модель | Accuracy | ROC-AUC |
|---|---|---|
| NeuralNetwork (своя) | 0.831 ± 0.036 | 0.897 ± 0.029 |
| BaggingNN (свой, 5 сетей) | 0.827 ± 0.048 | 0.874 ± 0.039 |
| AdaBoostNN (свой, 5 сетей) | 0.822 ± 0.040 | 0.860 ± 0.027 |
| LogisticRegression (sklearn) | 0.820 ± 0.028 | 0.901 ± 0.023 |
| MLPClassifier (sklearn) | 0.829 ± 0.036 | 0.906 ± 0.021 |
| GradientBoosting (sklearn) | 0.801 ± 0.044 | 0.884 ± 0.033 |

Выводы:

- Своя сеть работает на уровне `MLPClassifier` из scikit-learn с той же архитектурой
  и немного точнее логистической регрессии.
- Ансамбли не дают прироста: данных мало, а одиночная сеть уже стабильна. ROC-AUC
  у ансамблей ниже, потому что их уверенность — доля голосов, а не вероятность.
- **Про данные.** В исходном `heart.csv` 1025 строк, но уникальных только 302.
  Без удаления дублей одни и те же пациенты попадают и в обучение, и в тест, и
  точность завышается: на таком разбиении бэггинг из 3 сетей даёт 0.90 вместо 0.83.
  Здесь дубли удаляются до разбиения.

## Быстрый старт

```bash
git clone https://github.com/Mong1223/numpy-nn-ensembles.git
cd numpy-nn-ensembles
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e .
nn-evaluate                        # или python -m nn_ensembles.evaluate
```

Пример использования:

```python
from nn_ensembles import AdaBoostNN, NeuralNetwork

net = NeuralNetwork(hidden_units=13, learning_rate=0.1, epochs=5000).fit(X_train, y_train)
boost = AdaBoostNN(n_estimators=5).fit(X_train, y_train)
print(net.predict(X_test), boost.predict(X_test))
```

## Разработка

```bash
pip install -e ".[dev]"
ruff check .
pytest
```

## Структура проекта

```
numpy-nn-ensembles/
├── src/nn_ensembles/
│   ├── network.py      # нейросеть на NumPy
│   ├── ensembles.py    # BaggingNN и AdaBoostNN
│   └── evaluate.py     # сравнение с моделями scikit-learn
├── tests/              # проверка градиента, обучения и ансамблей
├── data/heart.csv      # датасет Heart Disease
├── legacy/             # исходный код лабораторной работы 2022 года
└── pyproject.toml
```

## Данные

[Heart Disease Dataset](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset)
на основе [UCI Heart Disease](https://archive.ics.uci.edu/dataset/45/heart+disease)
(лицензия CC BY 4.0): 13 признаков (возраст, давление, холестерин, ЭКГ и др.) и
целевая переменная — наличие болезни сердца.

## Лицензия

[MIT](LICENSE)
