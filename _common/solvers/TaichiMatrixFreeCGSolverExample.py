import taichi as ti
import math
from taichi.linalg import MatrixFreeCG, LinearOperator

ti.init(arch=ti.cpu)

GRID = 4
x = ti.field(dtype=ti.f32, shape=(GRID, GRID))
b = ti.field(dtype=ti.f32, shape=(GRID, GRID))

@ti.kernel
def init():
    for i, j in ti.ndrange(GRID, GRID):
        xl = i / (GRID - 1)
        yl = j / (GRID - 1)
        b[i, j] = ti.sin(2 * math.pi * xl) * ti.sin(2 * math.pi * yl)
        x[i, j] = 0.0

@ti.kernel
def compute_Ax(v: ti.template(), mv: ti.template()):
    for i, j in v:
        l = v[i - 1, j] if i - 1 >= 0 else 0.0
        r = v[i + 1, j] if i + 1 <= GRID - 1 else 0.0
        t = v[i, j + 1] if j + 1 <= GRID - 1 else 0.0
        b = v[i, j - 1] if j - 1 >= 0 else 0.0
        # Avoid ill-conditioned matrix A
        mv[i, j] = 20 * v[i, j] - l - r - t - b

A = LinearOperator(compute_Ax)
init()
MatrixFreeCG(A, b, x, maxiter=10 * GRID * GRID, tol=1e-18, quiet=False)
print(x.to_numpy())