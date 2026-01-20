from taichi.linalg import SparseMatrixBuilder, SparseCG, LinearOperator
from _common.solvers.matrix_free_cg_solver import MatrixFreeCGSolver

import taichi as ti


@ti.data_oriented
class HeatSolver:
    def __init__(self, solver) -> None:
        self.w_cells = solver.w_grid * solver.w_grid
        self.solver = solver
        self.A = LinearOperator(self.compute_Ax)
        self.x = ti.field(dtype=ti.f32, shape=self.w_cells)
        self.b = ti.field(dtype=ti.f32, shape=self.w_cells)
        self.mat_free_cg_solver = MatrixFreeCGSolver(self.A, self.b, self.x, maxiter=10, tol=1e-10, quiet=False)
        self.iter_count = 0

    @ti.kernel
    def fill_linear_system(self, A: ti.types.sparse_matrix_builder(), b: ti.types.ndarray()):  # pyright: ignore
        for i, j in ti.ndrange(self.solver.w_grid, self.solver.w_grid):
            idx = (i * self.solver.w_grid) + j  # raveled index
            b[idx] = self.solver.temperature_c[i, j]  # right-hand side

            # We enforce Dirichlet temperature boundary conditions at CELLS that are in contact with fixed
            # temperature bodies (like a heated pan (-> boundary cells in our case) or air (-> empty cells)),
            # i.e we keep the currently recorded cell temperatures for empty (air) cells.
            if not self.solver.is_interior(i, j):
                A[idx, idx] += 1.0
                continue

            # Compute (1 / dx^2) * ((dt * dx^d) / (m_c * c_c)) [Jiang 2016, Ch. 5.8],
            # NOTE: dx^d is cancelled out by 1 / dx^2 because d == 2.
            dt_inv_mass_capacity = self.solver.dt[None] / (self.solver.mass_c[i, j] * self.solver.capacity_c[i, j])
            inv_dx_sqrd = self.solver.inv_dx * self.solver.inv_dx
            diagonal = 1.0  # to keep max_num_triplets as low as possible

            # We enforce homogeneous Neumann boundary conditions at FACES adjacent to cells that are corresponding
            # to insulated objects, i.e we set the conductivity to zero for faces adjacent to insulated cells
            # (or by simply just not incorporating them).
            if not self.solver.is_insulated(i + 1, j):  # homogeneous Neumann
                diagonal += dt_inv_mass_capacity * self.solver.conductivity_x[i + 1, j]
                if self.solver.is_empty(i + 1, j):  # non-homogeneous Dirichlet
                    A[idx, idx + self.solver.w_grid] -= dt_inv_mass_capacity * self.solver.conductivity_x[i + 1, j]
                    b[idx] += inv_dx_sqrd * self.solver.conductivity_x[i + 1, j] * self.solver.temperature_c[i + 1, j]

            if not self.solver.is_insulated(i - 1, j):  # homogeneous Neumann
                diagonal += dt_inv_mass_capacity * self.solver.conductivity_x[i, j]
                if self.solver.is_empty(i - 1, j):  # non-homogeneous Dirichlet
                    A[idx, idx - self.solver.w_grid] -= dt_inv_mass_capacity * self.solver.conductivity_x[i, j]
                    b[idx] += inv_dx_sqrd * self.solver.conductivity_x[i, j] * self.solver.temperature_c[i - 1, j]

            if not self.solver.is_insulated(i, j + 1):  # homogeneous Neumann
                diagonal += dt_inv_mass_capacity * self.solver.conductivity_y[i, j + 1]
                if self.solver.is_empty(i, j + 1):  # non-homogeneous Dirichlet
                    A[idx, idx + 1] -= dt_inv_mass_capacity * self.solver.conductivity_y[i, j + 1]
                    b[idx] += inv_dx_sqrd * self.solver.conductivity_y[i, j + 1] * self.solver.temperature_c[i, j + 1]

            if not self.solver.is_insulated(i, j - 1):  # homogeneous Neumann
                diagonal += dt_inv_mass_capacity * self.solver.conductivity_y[i, j]
                if self.solver.is_empty(i, j - 1):  # non-homogeneous Dirichlet
                    A[idx, idx - 1] -= dt_inv_mass_capacity * self.solver.conductivity_y[i, j]
                    b[idx] += inv_dx_sqrd * self.solver.conductivity_y[i, j] * self.solver.temperature_c[i, j - 1]

            A[idx, idx] += diagonal  # add value from variable, to keep max_num_triplets as low as possible

    @ti.kernel
    def fill_temperature_field(self, T: ti.template()):  # pyright: ignore
        for i, j in ti.ndrange(self.solver.w_grid, self.solver.w_grid):
            self.solver.temperature_c[i, j] = T[(i * self.solver.w_grid) + j]

    @ti.kernel
    def fill_b(self):  # pyright: ignore
        for i, j in ti.ndrange(self.solver.w_grid, self.solver.w_grid):
            idx = (i * self.solver.w_grid) + j  # raveled index
            self.b[idx] = self.solver.temperature_c[i, j]  # right-hand side

            if not self.solver.is_interior(i, j): continue

            inv_dx_sqrd = self.solver.inv_dx * self.solver.inv_dx

            # We enforce homogeneous Neumann boundary conditions at FACES adjacent to cells that are corresponding
            # to insulated objects, i.e we set the conductivity to zero for faces adjacent to insulated cells
            # (or by simply just not incorporating them).
            if not self.solver.is_insulated(i + 1, j):  # homogeneous Neumann
                if self.solver.is_empty(i + 1, j):  # non-homogeneous Dirichlet
                    self.b[idx] += inv_dx_sqrd * self.solver.conductivity_x[i + 1, j] * self.solver.temperature_c[i + 1, j]

            if not self.solver.is_insulated(i - 1, j):  # homogeneous Neumann
                if self.solver.is_empty(i - 1, j):  # non-homogeneous Dirichlet
                    self.b[idx] += inv_dx_sqrd * self.solver.conductivity_x[i, j] * self.solver.temperature_c[i - 1, j]

            if not self.solver.is_insulated(i, j + 1):  # homogeneous Neumann
                if self.solver.is_empty(i, j + 1):  # non-homogeneous Dirichlet
                    self.b[idx] += inv_dx_sqrd * self.solver.conductivity_y[i, j + 1] * self.solver.temperature_c[i, j + 1]

            if not self.solver.is_insulated(i, j - 1):  # homogeneous Neumann
                if self.solver.is_empty(i, j - 1):  # non-homogeneous Dirichlet
                    self.b[idx] += inv_dx_sqrd * self.solver.conductivity_y[i, j] * self.solver.temperature_c[i, j - 1]
    @ti.kernel
    def compute_Ax(self, x: ti.template(), Ax: ti.template()):  # pyright: ignore
        for i, j in ti.ndrange(self.solver.w_grid, self.solver.w_grid):
            idx = (i * self.solver.w_grid) + j  # raveled index

            if not self.solver.is_interior(i, j):
                Ax[idx] = x[idx]
                continue

            dt_inv_mass_capacity = self.solver.dt[None] / (self.solver.mass_c[i, j] * self.solver.capacity_c[i, j])
            diagonal = 1.0  # to keep max_num_triplets as low as possible
            l, r, b, t = 0.0, 0.0, 0.0, 0.0
            l_idx, r_idx, b_idx, t_idx = idx - 1, idx + 1, idx + self.solver.w_grid, idx - self.solver.w_grid

            if not self.solver.is_insulated(i + 1, j):  # homogeneous Neumann
                diagonal += dt_inv_mass_capacity * self.solver.conductivity_x[i + 1, j]
                if self.solver.is_empty(i + 1, j):  # non-homogeneous Dirichlet
                    b -= dt_inv_mass_capacity * self.solver.conductivity_x[i + 1, j]

            if not self.solver.is_insulated(i - 1, j):  # homogeneous Neumann
                diagonal += dt_inv_mass_capacity * self.solver.conductivity_x[i, j]
                if self.solver.is_empty(i - 1, j):  # non-homogeneous Dirichlet
                    t -= dt_inv_mass_capacity * self.solver.conductivity_x[i, j]

            if not self.solver.is_insulated(i, j + 1):  # homogeneous Neumann
                diagonal += dt_inv_mass_capacity * self.solver.conductivity_y[i, j + 1]
                if self.solver.is_empty(i, j + 1):  # non-homogeneous Dirichlet
                    r -= dt_inv_mass_capacity * self.solver.conductivity_y[i, j + 1]

            if not self.solver.is_insulated(i, j - 1):  # homogeneous Neumann
                diagonal += dt_inv_mass_capacity * self.solver.conductivity_y[i, j]
                if self.solver.is_empty(i, j - 1):  # non-homogeneous Dirichlet
                    l -= dt_inv_mass_capacity * self.solver.conductivity_y[i, j]

            Ax[idx] = diagonal * x[idx] + l * x[l_idx] + r * x[r_idx] + b * x[b_idx] + t * x[t_idx]


    def solve(self):
        A = SparseMatrixBuilder(
            max_num_triplets=(5 * self.w_cells),
            num_rows=self.w_cells,
            num_cols=self.w_cells,
            dtype=ti.f32,
        )
        b = ti.ndarray(ti.f32, shape=self.w_cells)
        self.fill_linear_system(A, b)

        # Solve the linear system.
        solver = SparseCG(A.build(), b, atol=1e-5, max_iter=500)
        T, _ = solver.solve()

        self.fill_temperature_field(T)

    def matrix_free_cg_solve(self):
        self.iter_count += 1
        print(f'Heat Solve Iteration: {self.iter_count}')
        self.fill_b()
        self.mat_free_cg_solver.solve()
        self.fill_temperature_field(self.x)
