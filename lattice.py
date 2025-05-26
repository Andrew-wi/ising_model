from helpers import random_lattices_generation, metropolis_algorithm, \
    thermalization_visualization
from classes import IsingModel

if __name__ == "__main__":
    # # visualize
    # model = IsingModel(size=50, beta=1.0, h=0.1)
    # print(f"Energy: {model.calculate_energy()}")
    # model.visualize()

    # # random lattice generation
    # random_lattices_generation()

    # thermalization
    thermalization_visualization(25, 1000)
    