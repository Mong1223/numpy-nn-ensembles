import numpy as np
import pytest

from nn_ensembles import AdaBoostNN, BaggingNN, NeuralNetwork
from nn_ensembles.evaluate import DEFAULT_DATA, load_data


@pytest.fixture
def blobs():
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(-2, 1, (60, 4)), rng.normal(2, 1, (60, 4))])
    y = np.array([0] * 60 + [1] * 60)
    return X, y


def test_backprop_matches_numerical_gradient():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(5, 8))
    y = rng.integers(0, 2, size=(1, 8)).astype(float)
    w = np.full((1, 8), 1 / 8)

    net = NeuralNetwork(hidden_units=4, random_state=1)
    net._init_params(n_features=5)
    net.w1 *= 100  # крупные веса, чтобы градиент был заметно ненулевым
    net._forward(X)
    net._backward(X, y, w)

    eps = 1e-6
    for name, grad in [("w1", net.dw1), ("b1", net.db1), ("w2", net.dw2), ("b2", net.db2)]:
        param = getattr(net, name)
        numerical = np.zeros_like(param)
        for idx in np.ndindex(param.shape):
            old = param[idx]
            param[idx] = old + eps
            plus = net._loss(net._forward(X), y, w)
            param[idx] = old - eps
            minus = net._loss(net._forward(X), y, w)
            param[idx] = old
            numerical[idx] = (plus - minus) / (2 * eps)
        np.testing.assert_allclose(grad, numerical, rtol=1e-4, atol=1e-8)


def test_network_learns_separable_data(blobs):
    X, y = blobs
    net = NeuralNetwork(epochs=500, random_state=0).fit(X, y)
    assert (net.predict(X) == y).mean() > 0.95
    assert net.loss_history[-1] < net.loss_history[0]
    assert net.predict_proba(X).shape == (len(X),)


@pytest.mark.parametrize("cls", [BaggingNN, AdaBoostNN])
def test_ensembles_fit_and_predict(cls, blobs):
    X, y = blobs
    model = cls(n_estimators=3, random_state=0, epochs=300).fit(X, y)
    pred = model.predict(X)
    assert set(np.unique(pred)) <= {0, 1}
    assert (pred == y).mean() > 0.95


def test_load_data_drops_duplicates():
    X, y = load_data(DEFAULT_DATA)
    assert X.shape == (302, 13)
    assert len(np.unique(np.column_stack([X, y]), axis=0)) == len(y)
