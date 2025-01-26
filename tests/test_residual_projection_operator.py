import pytest
import numpy as np
from scipy.stats import zscore
from scipy.sparse import random, diags
from scipy.sparse.linalg import LinearOperator
from ipcas import MeanCenterOperator, ResidualProjectionOperator

@pytest.fixture
def test_data():
    np.random.seed(42)
    X = random(20, 30, 0.17).tocsr()
    V = np.random.random((30, 18))
    l_M = np.random.random((X.shape[1], 3)) 
    l_1 = np.random.random((X.shape[1], 1))
    l_v = np.random.random(X.shape[1])
    r_M = np.random.random((X.shape[0], 3))
    r_1 = np.random.random((X.shape[0], 1))
    r_v = np.random.random(X.shape[0])
    return X, V, l_M, l_1, l_v, r_M, r_1, r_v

def test_initialization(test_data):
    X, V, _, _, _, _, _, _ = test_data
    operator = ResidualProjectionOperator(X, V)
    assert operator.shape == X.shape, "Shape mismatch during initialization."
    assert np.issubdtype(operator.dtype, np.floating), "Operator dtype must be floating point."

def test_matmat(test_data):
    X, V, l_M, _, _, _, _, _ = test_data
    operator = ResidualProjectionOperator(X, V)
    result = operator @ l_M 
    expected = (X - X @ V @ V.T) @ l_M
    np.testing.assert_allclose(result, expected, rtol=1e-6, err_msg="Matrix-matrix multiplication failed.")

def test_matvec_1(test_data):
    X, V, _, l_1, _, _, _, _ = test_data
    operator = ResidualProjectionOperator(X, V)
    result = operator @ l_1 
    expected = (X - X @ V @ V.T) @ l_1
    np.testing.assert_allclose(result, expected, rtol=1e-6, err_msg="Matrix-matrix1 multiplication failed.")
    
def test_matvec(test_data):
    X, V, _, _, l_v, _, _, _ = test_data
    operator = ResidualProjectionOperator(X, V)
    result = operator @ l_v  
    expected = np.ravel((X - X @ V @ V.T) @ l_v)
    np.testing.assert_allclose(result, expected, rtol=1e-6, err_msg="Matrix-vector multiplication failed.")

def test_rmatmat(test_data):
    X, V, _, _, _, r_M, _, _ = test_data
    operator = ResidualProjectionOperator(X, V)
    result = operator.T @ r_M  
    expected = (X - X @ V @ V.T).T @ r_M
    np.testing.assert_allclose(result, expected, rtol=1e-6, err_msg="Matrix-matrix multiplication failed.")

def test_rmatvec_1(test_data):
    X, V, _, _, _, _, r_1, _ = test_data
    operator = ResidualProjectionOperator(X, V)
    result = operator.T @ r_1 
    expected = (X - X @ V @ V.T).T @ r_1
    np.testing.assert_allclose(result, expected, rtol=1e-6, err_msg="Matrix-matrix1 multiplication failed.")
    
def test_rmatvec(test_data):
    X, V, _, _, _, _, _, r_v = test_data
    operator = ResidualProjectionOperator(X, V)
    result = operator.T @ r_v  
    expected = np.ravel((X - X @ V @ V.T).T @ r_v)
    np.testing.assert_allclose(result, expected, rtol=1e-6, err_msg="Matrix-vector multiplication failed.")

def test_invalid_inputs():
    X = random(30, 20, 0.2).tocsr()
    V = np.random.random((30, 19))
    with pytest.raises(ValueError):
        ResidualProjectionOperator(X, V)  # Should fail due to mismatched dimensions

