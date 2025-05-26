import numpy as np
import matplotlib.pyplot as plt

class IsingModel:
    def __init__(self, size=5, beta=1.0, h=0.1):
        self.size = size
        self.beta = beta
        self.h = h
        self.spins = 2 * np.random.randint(0, 2, (size, size)) - 1
    
    def calculate_energy(self):
        energy = 0.0
        # Vectorized nearest-neighbor interaction calculation
        # Horizontal interactions
        energy -= self.beta * np.sum(self.spins[:, :-1] * self.spins[:, 1:])
        # Wrap-around horizontal
        energy -= self.beta * np.sum(self.spins[:, -1] * self.spins[:, 0])
        # Vertical interactions
        energy -= self.beta * np.sum(self.spins[:-1, :] * self.spins[1:, :])
        # Wrap-around vertical
        energy -= self.beta * np.sum(self.spins[-1, :] * self.spins[0, :])
        # External field contribution
        energy -= self.h * np.sum(self.spins)
        return energy
    
    def visualize(self):
        plt.figure(figsize=(5, 5))
        plt.imshow(self.spins, cmap='coolwarm')
        plt.colorbar(label='Spin (↑/↓)')
        plt.title('Ising Model Lattice')
        plt.show()

def random_lattices_generation():
    trials = 10000
    energies = np.zeros(trials)
    for i in range(trials):
        model = IsingModel(size=50, beta=1.0, h=0.1)
        energies[i] = model.calculate_energy()
    
    # plot energies
    plt.hist(energies, bins=20, color='skyblue', edgecolor='black')
    plt.xlabel('Energy')
    plt.ylabel('Frequency')
    plt.title('Histogram of Random Lattice Energies')
    plt.show()

# boilerplate
if __name__ == "__main__":
    # model = IsingModel(size=50, beta=1.0, h=0.1)
    # print(f"Energy: {model.calculate_energy()}")
    # model.visualize()
    random_lattices_generation()