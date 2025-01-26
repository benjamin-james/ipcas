
from scipy.sparse.linalg import LinearOperator
class ResidualProjectionOperator(LinearOperator):
    def __init__(self, X, V):
        self.X = X
        self.V = V
        if X.shape[1] != V.shape[0]:
            raise ValueError("Incompatible shapes passed")
        super().__init__(dtype=X.dtype, shape=X.shape)
    def _matvec(self, v):
        X_v = self.X @ v
        VT_v = self.V.T @ v
        X_V_VT_v = self.X @ (self.V @ VT_v)
        return X_v - X_V_VT_v
    def _matmat(self, M):
        X_M = self.X @ M
        VT_M = self.V.T @ M
        X_V_VT_M = self.X @ (self.V @ VT_M)
        return X_M - X_V_VT_M
    def _rmatvec(self, v):
        XT_v = self.X.T @ v
        VT_XT_v = self.V.T @ XT_v
        return XT_v - self.V @ VT_XT_v
    def _rmatmat(self, M):
        XT_M = self.X.T @ M
        VT_XT_M = self.V.T @ XT_M
        return XT_M - self.V @ VT_XT_M
