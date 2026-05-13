import numpy as np
import numba
import matplotlib.pyplot as plt

## Setup
print("Setting up simulation...")

# Parameters
genome_size = 16 
total_simulations = 100000
K_range = list(range(9))   # Intragenomic partners
K2_range = list(range(9))  # Intergenomic partners
beta_a, beta_b = 0.5, 0.5

##################################################################################################################

## Functions for fitness calculation

@numba.njit
def calculate_fitness_from_int(genome_int, genome_size, epistasis_matrix, fitness_matrix, total_partners):
    total_fitness = 0.0
    for locus in range(genome_size):
        focal_bit = (genome_int >> (genome_size - 1 - locus)) & 1
        idx = focal_bit
        for p_idx in range(total_partners):
            partner_locus = epistasis_matrix[locus, p_idx]
            partner_bit = (genome_int >> (genome_size - 1 - partner_locus)) & 1
            idx |= (partner_bit << (p_idx + 1))
        total_fitness += fitness_matrix[locus, idx]
    return total_fitness / genome_size

@numba.njit
def find_optimal_fitness(genome_size, epistasis_matrix, fitness_matrix, total_partners):
    best_f = 0.0
    num_combinations = 1 << genome_size
    for i in range(num_combinations):
        f = calculate_fitness_from_int(i, genome_size, epistasis_matrix, fitness_matrix, total_partners)
        if f > best_f:
            best_f = f
    return best_f

##################################################################################################################

## Main simulation loop

# Create empty matrix
results_matrix = np.zeros((len(K_range), len(K2_range)))
std_matrix = np.zeros((len(K_range), len(K2_range)))

# Loop over K and K2 values
for i, K in enumerate(K_range):
    for j, K2 in enumerate(K2_range):
        total_partners = K + K2
        print(f"Running K={K}, K2={K2}")
        
        # Run simulations for this combination and store each optimalfitness
        sim_optimal_fitnesses = []
        for sim in range(total_simulations):
            # For every simulation:
            # Create empty epistasis matrix
            epistasis_matrix = np.zeros((genome_size, total_partners), dtype=np.int32)
            for locus in range(genome_size):
                # Pick K intragenomic partners (except itself)
                available_intra = np.delete(np.arange(genome_size), locus)
                intra_partners = np.random.choice(available_intra, size=K, replace=False)
                
                # Pick K2 intergenomic partners (including itself)
                available_inter = np.arange(genome_size)
                inter_partners = np.random.choice(available_inter, size=K2, replace=False)
                
                # Fill the epistasis matrix
                epistasis_matrix[locus, :K] = intra_partners
                epistasis_matrix[locus, K:] = inter_partners

            # Generate fitness matrix
            fitness_matrix = np.random.beta(beta_a, beta_b, size=(genome_size, 2**(1 + total_partners)))

            # Find optimal fitness
            best_fitness = find_optimal_fitness(genome_size, epistasis_matrix, fitness_matrix, total_partners)
            sim_optimal_fitnesses.append(best_fitness)
        
        # Store the average and SD of this combination in the matrix
        results_matrix[i, j] = np.mean(sim_optimal_fitnesses)
        std_matrix[i, j] = np.std(sim_optimal_fitnesses)
        

##################################################################################################################

## Data collection

# Plotting heatmap
plt.figure(figsize=(10, 8))
im = plt.imshow(results_matrix, origin='lower', aspect='auto', cmap='viridis')

# Add colorbar
cbar = plt.colorbar(im)
cbar.set_label('Average Optimal Fitness', rotation=270, labelpad=15)

# Add labels
plt.xticks(ticks=np.arange(len(K2_range)), labels=K2_range)
plt.yticks(ticks=np.arange(len(K_range)), labels=K_range)
plt.xlabel('$K_2$')
plt.ylabel('$K$')
plt.title(f'NK Fitness Landscape: Optimal Fitness Averages\n(Genome Size $N={genome_size}$, {total_simulations} sims per cell)')

# Include values in cells
for i in range(len(K_range)):
    for j in range(len(K2_range)):
        plt.text(j, i, f'{results_matrix[i, j]:.3f}', 
                 ha="center", va="center", color="w" if results_matrix[i, j] < results_matrix.max()*0.7 else "black")

plt.tight_layout()
plt.savefig(f"Fitness_Heatmap_N{genome_size} {total_simulations} sims.png")
plt.show()

# Save raw data
np.savez(f"Optimal_Fitness_Matrix_N{genome_size} {total_simulations} sims.npz", matrix=results_matrix, std_matrix=std_matrix, K=K_range, K2=K2_range)
print("Simulation complete. Heatmap saved.")