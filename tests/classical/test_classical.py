"""Tests for tabkernels.classical (Chapter 2)."""
import numpy as np
import pytest
from tabkernels.classical import (
    KNNRegressor, KNNClassifier, NadarayaWatson, KernelRidgeRegression, rbf_kernel,
)


@pytest.fixture
def reg_data():
    rng = np.random.RandomState(0)
    X = rng.randn(50, 3).astype(np.float32)
    y = (X[:, 0] + X[:, 1] ** 2 + 0.1 * rng.randn(50)).astype(np.float32)
    return X, y


@pytest.fixture
def cla_data():
    rng = np.random.RandomState(0)
    X = rng.randn(60, 2).astype(np.float32)
    y = (X[:, 0] + X[:, 1] > 0).astype(np.int64)
    return X, y


def test_knn_regressor_perfect_recall(reg_data):
    """1-NN at training points returns exact training labels."""
    X, y = reg_data
    m = KNNRegressor(k=1).fit(X, y)
    pred = m.predict(X)
    assert np.allclose(pred, y, atol=1e-5)


def test_knn_regressor_average(reg_data):
    """Predictions are bounded by min/max of training y."""
    X, y = reg_data
    m = KNNRegressor(k=10).fit(X, y)
    pred = m.predict(X)
    assert pred.min() >= y.min() - 1e-5
    assert pred.max() <= y.max() + 1e-5


def test_knn_regressor_distance_weights(reg_data):
    """Distance-weighted KNN at training point exactly recovers training label."""
    X, y = reg_data
    m = KNNRegressor(k=5, weights="distance").fit(X, y)
    # At training points, distance to self is 0; weight is infinite.
    pred = m.predict(X)
    assert np.allclose(pred, y, atol=1e-3)


def test_knn_classifier_perfect_recall(cla_data):
    X, y = cla_data
    m = KNNClassifier(k=1).fit(X, y)
    assert (m.predict(X) == y).all()


def test_knn_invalid_k():
    with pytest.raises(ValueError):
        KNNRegressor(k=0)
    with pytest.raises(ValueError):
        KNNRegressor(k=10).fit(np.zeros((5, 2)), np.zeros(5))  # k > N


def test_nw_kernel_matrix_symmetric(reg_data):
    X, _ = reg_data
    m = NadarayaWatson(bandwidth=1.0)
    K = m.kernel_matrix(X, X)
    assert np.allclose(K, K.T, atol=1e-5)
    assert np.allclose(np.diag(K), 1.0, atol=1e-5)


def test_nw_kernel_psd(reg_data):
    """RBF Gram matrix is PSD."""
    X, _ = reg_data
    m = NadarayaWatson(bandwidth=1.0)
    K = m.kernel_matrix(X, X)
    eigvals = np.linalg.eigvalsh(K)
    assert eigvals.min() > -1e-4


def test_nw_predict_bandwidth_effect(reg_data):
    """Wide bandwidth -> prediction approaches mean of y."""
    X, y = reg_data
    m_wide = NadarayaWatson(bandwidth=100.0).fit(X, y)
    m_narrow = NadarayaWatson(bandwidth=0.05).fit(X, y)
    pred_wide = m_wide.predict(X)
    pred_narrow = m_narrow.predict(X)
    # Wide bandwidth: predictions tightly clustered around y.mean().
    assert pred_wide.std() < y.std() / 5
    # Narrow bandwidth: predictions track y closely (1-NN-like).
    assert ((pred_narrow - y) ** 2).mean() < ((pred_wide - y) ** 2).mean()


def test_krr_dual_form(reg_data):
    """KRR alpha = (K + N*lambda*I)^-1 y."""
    X, y = reg_data
    m = KernelRidgeRegression(bandwidth=1.0, ridge=0.01).fit(X, y)
    K = m.kernel_matrix(X, X)
    N = X.shape[0]
    expected_alpha = np.linalg.solve(K + N * 0.01 * np.eye(N, dtype=K.dtype), y)
    assert np.allclose(m.dual_coefs, expected_alpha, atol=1e-4)


def test_krr_low_ridge_high_train_fit(reg_data):
    """Very small ridge -> KRR fits training set closely."""
    X, y = reg_data
    m = KernelRidgeRegression(bandwidth=1.0, ridge=1e-6).fit(X, y)
    pred = m.predict(X)
    train_mse = ((pred - y) ** 2).mean()
    assert train_mse < 0.1


def test_krr_high_ridge_smoothing(reg_data):
    """Very large ridge -> predictions shrink toward zero (or training mean for centered y)."""
    X, y = reg_data
    m = KernelRidgeRegression(bandwidth=1.0, ridge=1e3).fit(X, y)
    pred = m.predict(X)
    assert np.abs(pred).max() < np.abs(y).max() + 1e-3


def test_predict_before_fit_raises():
    m = KNNRegressor(k=3)
    with pytest.raises(RuntimeError):
        m.predict(np.zeros((1, 2)))
    m = NadarayaWatson()
    with pytest.raises(RuntimeError):
        m.predict(np.zeros((1, 2)))
    m = KernelRidgeRegression()
    with pytest.raises(RuntimeError):
        m.predict(np.zeros((1, 2)))


def test_rbf_kernel_self_one():
    """RBF kernel value at the same point is 1."""
    X = np.array([[1.0, 2.0]], dtype=np.float32)
    K = rbf_kernel(X, X, sigma=1.0)
    assert np.allclose(K, [[1.0]])
