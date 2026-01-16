import taichi as ti
import taichi.math as tm

@ti.data_oriented
class CGSolver:
    def __init__(self, A: ti.template(), b: ti.template(), x: ti.template()) -> None: # type: ignore
        self.A = A
        self.b = b
        self.x = x

        self.r = ti.field(ti.f32, shape=A.shape)  # residual
        self.p = ti.field(ti.f32, shape=A.shape)  # conjugate gradient
        self.Ap = ti.field(ti.f32, shape=A.shape) # matrix-vector product
        self.Ax = ti.field(ti.f32, shape=A.shape) # matrix-vector product

    @ti.kernel
    def reset(self):
        self.r.fill(0)
        self.p.fill(0)
        self.Ap.fill(0)
        self.Ax.fill(0)

    @ti.kernel
    def compute_Ax(self):
        for i in self.Ax:
            self.Ax[i] = 0.0
        for i, j in ti.ndrange(self.A.shape[0], self.A.shape[1]):
            self.Ax[i] += self.A[i, j] * self.x[j]