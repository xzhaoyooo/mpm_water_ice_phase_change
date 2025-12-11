from src.parsing import arguments, should_use_cuda_backend, should_use_collocated

from sampler import PoissonDiskSampler
from apic import APIC

from presets import configuration_list
from simulation import Simulation

import taichi as ti


def main():
    # Initialize Taichi on the chosen architecture:
    ti.init(arch=ti.cuda if should_use_cuda_backend else ti.cpu, debug=False)

    initial_configuration = arguments.configuration % len(configuration_list)
    simulation_name = f"Affine Particle-In-Cell Method"

    # The radius for the particles and the Poisson-Disk Sampler:
    max_particles = 500_000  # TODO: this could be computed from radius
    n_grid = 128 * arguments.quality
    radius = 1 / (8 * float(n_grid))  # dx / 4
    dt = 1e-3 / arguments.quality

    solver = (CollocatedAPIC(max_particles, n_grid, dt) if should_use_collocated else APIC(max_particles, n_grid, dt))
    sampler = PoissonDiskSampler(apic_solver=solver, r=radius, k=50)

    simulation = Simulation(
        initial_configuration=initial_configuration,
        configurations=configuration_list,
        poisson_disk_sampler=sampler,
        apic_solver=solver,
        name=simulation_name,
        res=(720, 720),
        radius=radius,
    )
    simulation.run()

    print("\n", "#" * 100, sep="")
    print("###", simulation_name)
    print("#" * 100)
    print(">>> R        -> [R]eset the simulation.")
    print(">>> P|SPACE  -> [P]ause/Un[P]ause the simulation.")
    print()


if __name__ == "__main__":
    main()
