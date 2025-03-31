import numpy as np
from numpy.typing import NDArray
from scipy.sparse import spmatrix, diags
from scipy.sparse.linalg import svds, LinearOperator
from typing import Union
from .mean_center_operator import MeanCenterOperator
from .residual_projection_operator import ResidualProjectionOperator
from .stacked_linear_operator import StackedLinearOperator
class IPCAS:
    def __init__(self,
                 n_components:int,
                 mean:NDArray[np.float64],
                 std:NDArray[np.float64]):
        self.n_components_ = n_components
        self.mean_ = mean
        self.std_ = std
        self.n_samples_seen_ = 0
        self.singular_values_ = None
        self.components_ = None  ## = V', (k x p)
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
    def partial_fit(self, X:Union[spmatrix, LinearOperator], use_stack:bool=True):
        mco = MeanCenterOperator(X, mu=self.mean_, std=self.std_)
        if self.components_ is None:
            _, s, Vt = svds(mco, self.n_components_, which="LM")
        elif use_stack:
            _, s, Vt = svds(StackedLinearOperator(mco,
                                                  diags(self.singular_values_) @ self.components_),
                            self.n_components_, which="LM")
        else:
            _, R_s, R_Vt = svds(ResidualProjectionOperator(mco, self.components_.T),
                                self.n_components_, which="LM")
            _, s, Vt = svds(np.vstack((diags(R_s) @ R_Vt,
                                       diags(self.singular_values_) @ self.components_)),
                            self.n_components_, which="LM")
        idx = np.argsort(-s)
        self.n_samples_seen_ += X.shape[0]
        self.singular_values_ = s[idx]
        self.components_ = Vt[idx, :]
        self.explained_variance_ = self.singular_values_ ** 2 / (self.n_samples_seen_ - 1)
        self.explained_variance_ratio_ = self.singular_values_ ** 2 / np.sum(np.ones_like(self.std_) ** 2 * self.n_samples_seen_)
    def transform(self, X:spmatrix):
        if self.components_ is None:
            raise ValueError("Error: Not fitted yet")
        mco = MeanCenterOperator(X, mu=self.mean_, std=self.std_)
        return mco @ self.components_.T
