from taichi.linalg import MatrixFreeBICGSTAB, MatrixFreeCG, LinearOperator
import taichi as ti

@ti.data_oriented
class MatrixFreeCGSolver:
    def __init__(self, A: LinearOperator, b: ti.template(), x: ti.template(), maxiter: int = 1000, tol: float = 1e-10, quiet: bool): # type: ignore
        self.A = A
        self.b = b
        self.x = x
        self.maxiter = maxiter
        self.tol = tol
        self.quiet = quiet

    def solve(self):
        MatrixFreeCG(self.A, self.b, self.x, maxiter=self.maxiter, tol=self.tol, quiet=self.quiet)