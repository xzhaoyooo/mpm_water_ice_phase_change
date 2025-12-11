from abc import ABC, abstractmethod
import taichi as ti


@ti.data_oriented
class BaseSolver(ABC):
    def __init__(self, quality: int, max_particles: int):
        self.n_particles = ti.field(dtype=ti.int32, shape=())
        self.max_particles = max_particles
        self.n_grid = 128 * quality
        self.n_cells = self.n_grid * self.n_grid
        self.dx = 1 / self.n_grid
        self.inv_dx = float(self.n_grid)
        self.dt = 3e-4 / quality
        self.vol_0_p = (self.dx * 0.5) ** 2
        self.n_dimensions = 2

        
        self.position_p = ti.Vector.field(2, dtype=ti.f32, shape=max_particles)
        self.color_p = ti.Vector.field(3, dtype=ti.f32, shape=max_particles)

    @abstractmethod
    def substep(self):
        pass
