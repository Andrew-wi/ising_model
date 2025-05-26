import numpy as np
import matplotlib.pyplot as plt
from classes import IsingModel

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

def metropolis_algorithm(model):
    # choose random spin
    size = model.spins.shape[0]
    i, j = np.random.randint(0, size, 2)

    # current spin that is flipped
    current_spin = model.spins[i, j]

    # Energy contribution from neighbors (with periodic boundaries)
    # For each spin flip, the change in energy ΔE is:
    # ΔE = 2 * current_spin * (sum of neighbor spins * beta + h)
    # This accounts for the fact that flipping the spin changes 
    # its interaction with each neighbor,
    # and the factor of 2 comes from the difference between the initial 
    # and final state.
    neighbors = (model.spins[(i-1)%size, j] + model.spins[(i+1)%size, j] + 
                model.spins[i, (j-1)%size] + model.spins[i, (j+1)%size])
    
    delta_E = 2 * current_spin * (model.beta * neighbors + model.h)
    
    # Accept or reject the flip
    if delta_E <= 0 or np.random.random() < np.exp(-delta_E):
        model.spins[i, j] *= -1

# thermalization visualization
def thermalization_visualization(lambda_size=25, number_of_trials=100):
    energies = np.zeros(number_of_trials)

    for i in range(number_of_trials):
        model = IsingModel(size=lambda_size, beta=1.0, h=0.1)
        thermalization_time = 1000
        for _ in range(thermalization_time):
            metropolis_algorithm(model)
        # calculate energy, place into array
        energies[i] = model.calculate_energy()
    
    # Plot histogram of distribution
    plt.figure(figsize=(8, 6))
    plt.hist(energies, bins=20, color='lightcoral', edgecolor='black', alpha=0.7)
    plt.xlabel('Energy')
    plt.ylabel('Frequency')
    plt.title('Histogram of Thermalized Lattice Energies (100 trials)')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    print(f"Mean energy: {np.mean(energies):.2f}")
    print(f"Standard deviation: {np.std(energies):.2f}")
    print(f"Energy range: [{np.min(energies):.2f}, {np.max(energies):.2f}]")
    