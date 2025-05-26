import numpy as np
import matplotlib.pyplot as plt

# lattice site class
class LatticeSite():
    def __init__(self, indices, spin):
        self.indices = indices
        self.spin = spin
        self.connections = []
    
    # add connections
    def add_connections(self, neighboring_lattice_sites):
        self.connections.append(neighboring_lattice_sites)

class IsingModel():
    def __init__(self, beta, h):
        self.beta = beta
        self.h = h

# create a lattice of spins with some initialization
def create_lattice():
    # dimensions
    dims = [5, 5]
    lattice = np.empty(dims, dtype=object)

    for i in range(dims[0]):
        for j in range(dims[1]):
            # randomize spins
            spin = int(np.round(np.random.uniform(0, 1)))
            lattice[i, j] = LatticeSite((i, j), spin)
            # Add connections to nearest neighbors with periodic boundary conditions
            neighbors = []
            for di, dj in [(-1,0), (1,0), (0,-1), (0,1)]:
                ni = (i + di) % dims[0]
                nj = (j + dj) % dims[1]
                neighbors.append(lattice[ni, nj])
            lattice[i, j].connections = neighbors
    return lattice

# calculate the energy
def calculate_energy(lattice, model):
    energy = 0.0
    dims = lattice.shape
    for i in range(dims[0]):
        for j in range(dims[1]):
            site = lattice[i, j]
            spin = 1 if site.spin == 1 else -1
            # Nearest neighbors (periodic boundary conditions)
            neighbors = site.connections
            for neighbor in neighbors:
                if neighbor is not None:
                    neighbor_spin = 1 if neighbor.spin == 1 else -1
                    energy -= model.beta * spin * neighbor_spin / 2  # divide by 2 to avoid double counting
            energy -= model.h * spin
    return energy

def visualize_lattice(lattice):
    spins = [[2*lattice[i,j].spin-1 for j in range(lattice.shape[1])] 
             for i in range(lattice.shape[0])]
    plt.figure(figsize=(5,5))
    plt.imshow(spins, cmap='coolwarm')
    plt.colorbar(label='Spin (↑/↓)')
    plt.title('Ising Model Lattice')
    plt.show()

# boilerplate
if __name__ == "__main__":
    # Create a test Model instance with reasonable parameters
    test_model = IsingModel(beta=1.0, h=0.1)
    lattice = create_lattice()

    print(calculate_energy(lattice, test_model))
    visualize_lattice(lattice)