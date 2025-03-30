import pytest
import numpy as np
from scipy.sparse import random
from ipcas import StackedLinearOperator, MeanCenterOperator  # adjust as needed

@pytest.fixture
def stacked_test_data():
    np.random.seed(42)
    X = random(200, 30, density=0.2).tocsr()
    mu = np.random.random(30)
    std = np.random.random(30) + 0.1  # avoid divide by zero
    mco = MeanCenterOperator(X, mu, std)

    B = np.random.random((20, 30))  # bottom block, dense

    l_M = np.random.random((30, 4))
    l_1 = np.random.random((30, 1))
    l_v = np.random.random(30)

    r_M = np.random.random((220, 4))
    r_1 = np.random.random((220, 1))
    r_v = np.random.random(220)

    return mco, B, l_M, l_1, l_v, r_M, r_1, r_v

def test_initialization(stacked_test_data):
    A, B, *_ = stacked_test_data
    op = StackedLinearOperator(A, B)
    assert op.shape == (A.shape[0] + B.shape[0], A.shape[1])
    assert np.issubdtype(op.dtype, np.floating)

def test_matmat(stacked_test_data):
    A, B, l_M, *_ = stacked_test_data
    op = StackedLinearOperator(A, B)
    result = op @ l_M
    expected = np.vstack([A @ l_M, B @ l_M])
    np.testing.assert_allclose(result, expected, rtol=1e-6)

def test_matvec_1(stacked_test_data):
    A, B, _, l_1, *_ = stacked_test_data
    op = StackedLinearOperator(A, B)
    result = op @ l_1
    expected = np.vstack([A @ l_1, B @ l_1])
    np.testing.assert_allclose(result, expected, rtol=1e-6)

def test_matvec(stacked_test_data):
    A, B, _, _, l_v, *_ = stacked_test_data
    op = StackedLinearOperator(A, B)
    result = op @ l_v
    expected = np.concatenate([A @ l_v, B @ l_v])
    np.testing.assert_allclose(result, expected, rtol=1e-6)

def test_rmatmat(stacked_test_data):
    A, B, *_, r_M, _, _ = stacked_test_data
    op = StackedLinearOperator(A, B)
    result = op.T @ r_M
    expected = A.rmatmat(r_M[:A.shape[0], :]) + B.T @ r_M[A.shape[0]:, :]
    np.testing.assert_allclose(result, expected, rtol=1e-6)

def test_rmatvec_1(stacked_test_data):
    A, B, *_, r_1, _ = stacked_test_data
    op = StackedLinearOperator(A, B)
    result = np.ravel(op.T @ r_1)
    expected = A.rmatvec(r_1[:A.shape[0], :].ravel()) + B.T @ r_1[A.shape[0]:, :].ravel()
    np.testing.assert_allclose(result, expected, rtol=1e-6)

def test_rmatvec(stacked_test_data):
    A, B, *_, r_v = stacked_test_data
    op = StackedLinearOperator(A, B)
    result = op.T @ r_v
    expected = A.rmatvec(r_v[:A.shape[0]]) + B.T @ r_v[A.shape[0]:]
    np.testing.assert_allclose(result, expected, rtol=1e-6)

def test_invalid_dimensions():
    X = random(10, 20, density=0.2).tocsr()
    mu = np.random.random(20)
    std = np.random.random(20) + 0.1
    mco = MeanCenterOperator(X, mu, std)
    bad_B = np.random.random((5, 15))  # mismatched number of columns
    with pytest.raises(ValueError):
        StackedLinearOperator(mco, bad_B)
