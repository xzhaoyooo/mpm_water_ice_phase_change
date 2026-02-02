import taichi as ti
import taichi.math as tm
import numpy as np
import scipy as sp

@ti.data_oriented
class MatrixPropertiesValidator:
    def __init__(self, A: ti.template(), b: ti.template()) -> None:
        self.A = A
        self.b = b

    def make_snapshot(self) -> None:
        self.A_np = self.A.to_numpy()
        self.b_np = self.b.to_numpy()

    def is_matrix_symmetric(self) -> bool:
        return sp.linalg.issymmetric(self.A_np)
    
    def get_b_norm(self) -> float:
        return np.linalg.norm(self.b_np) # type: ignore

    def is_matrix_positive_semidefinite(self, tol=-1e-10) -> bool:
        U, S, Vt = np.linalg.svd(self.A_np)
        return np.all(S >= tol)  # type: ignore
    
    def dump_matrix(self, filename: str) -> None:
        np.save(filename, self.A_np)