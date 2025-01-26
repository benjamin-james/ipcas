import pytest
import numpy as np
from scipy.stats import zscore
from scipy.sparse import random, diags
from scipy.sparse.linalg import LinearOperator
from ipcas import MeanCenterOperator 

@pytest.fixture
def test_data():
    np.random.seed(42)
    X = random(20, 30, 0.17).tocsr()
    mu = np.ravel(X.mean(0))
    std = np.std(X.toarray(), axis=0) + 1e-300  # Avoid division by zero
    l_M = np.random.random((X.shape[1], 3)) 
    l_1 = np.random.random((X.shape[1], 1))
    l_v = np.random.random(X.shape[1])
    r_M = np.random.random((X.shape[0], 3))
    r_1 = np.random.random((X.shape[0], 1))
    r_v = np.random.random(X.shape[0])
    return X, mu, std, l_M, l_1, l_v, r_M, r_1, r_v

def test_initialization(test_data):
    X, mu, std, _, _, _, _, _, _ = test_data
    operator = MeanCenterOperator(X, mu, std)
    assert operator.shape == X.shape, "Shape mismatch during initialization."
    assert np.issubdtype(operator.dtype, np.floating), "Operator dtype must be floating point."

def test_matmat(test_data):
    X, mu, std, l_M, _, _, _, _, _ = test_data
    operator = MeanCenterOperator(X, mu, std)
    result = operator @ l_M 
    expected = ((X.toarray() - mu[None, :]) / std[None, :]) @ l_M
    np.testing.assert_allclose(result, expected, rtol=1e-6, err_msg="Matrix-matrix multiplication failed.")

def test_matvec_1(test_data):
    X, mu, std, _, l_1, _, _, _, _ = test_data
    operator = MeanCenterOperator(X, mu, std)
    result = operator @ l_1 
    expected = ((X.toarray() - mu[None, :]) / std[None, :]) @ l_1
    np.testing.assert_allclose(result, expected, rtol=1e-6, err_msg="Matrix-matrix1 multiplication failed.")
    
def test_matvec(test_data):
    X, mu, std, _, _, l_v, _, _, _ = test_data
    operator = MeanCenterOperator(X, mu, std)
    result = operator @ l_v  
    expected = ((X.toarray() - mu[None, :]) / std[None, :]) @ l_v
    np.testing.assert_allclose(result, expected, rtol=1e-6, err_msg="Matrix-vector multiplication failed.")

def test_rmatmat(test_data):
    X, mu, std, _, _, _, r_M, _, _ = test_data
    operator = MeanCenterOperator(X, mu, std)
    result = operator.T @ r_M  
    expected = ((X.toarray() - mu[None, :]) / std[None, :]).T @ r_M
    np.testing.assert_allclose(result, expected, rtol=1e-6, err_msg="Matrix-matrix multiplication failed.")

def test_rmatvec_1(test_data):
    X, mu, std, _, _, _, _, r_1, _ = test_data
    operator = MeanCenterOperator(X, mu, std)
    result = operator.T @ r_1 
    expected = ((X.toarray() - mu[None, :]) / std[None, :]).T @ r_1
    np.testing.assert_allclose(result, expected, rtol=1e-6, err_msg="Matrix-matrix1 multiplication failed.")
    
def test_rmatvec(test_data):
    X, mu, std, _, _, _, _, _, r_v = test_data
    operator = MeanCenterOperator(X, mu, std)
    result = operator.T @ r_v  
    expected = ((X.toarray() - mu[None, :]) / std[None, :]).T @ r_v
    np.testing.assert_allclose(result, expected, rtol=1e-6, err_msg="Matrix-vector multiplication failed.")

def test_invalid_inputs():
    X = random(30, 20, 0.2)
    mu = np.array([1.0, 2.0, 3.0])  # Wrong shape
    std = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    with pytest.raises(ValueError):
        MeanCenterOperator(X, mu, std)  # Should fail due to mismatched dimensions
