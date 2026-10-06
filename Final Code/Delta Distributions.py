import numpy as np
import random
import numba
import matplotlib.pyplot as plt

## Core functions

def mobius_transform(fitness_values):
    num_coeffs = len(fitness_values)
    coefficients = np.zeros(num_coeffs, dtype=float)
    coefficients[0] = fitness_values[0]
    for j in range(1, num_coeffs):
        subset_sum = 0.0
        for l in range(j):
            if l == (l & j):
                subset_sum += coefficients[l]
        coefficients[j] = fitness_values[j] - subset_sum
    return coefficients

def calculateCoefficients(fitness_matrix, K, K2, genome_size):
    num_IESI = 2 ** (K + 1)
    num_multilinear_coeffs = 2 ** K2
    coefficients = np.zeros((genome_size, num_IESI, num_multilinear_coeffs), dtype=float)
    for locus in range(genome_size):
        for iesi in range(num_IESI):
            sub_hypercube = np.zeros(num_multilinear_coeffs, dtype=float)
            for group_corner in range(num_multilinear_coeffs):
                fitness_index = (iesi << K2) | group_corner
                sub_hypercube[group_corner] = fitness_matrix[locus, fitness_index]
            coefficients[locus, iesi, :] = mobius_transform(sub_hypercube)
    return coefficients

@numba.njit
def evaluate_multilinear(coefficients, group_avg_values):
    total = 0.0
    compensation = 0.0
    for j in range(len(coefficients)):
        term = coefficients[j]
        for k in range(len(group_avg_values)):
            if j & (1 << k):
                term *= group_avg_values[k]
        y = term - compensation
        t = total + y
        compensation = (t - total) - y
        total = t
    return total

@numba.njit
def calculateLocusFitness(locus, genome, group_avg_genome, epistasis_matrix, coefficients, K, K2):
    intragenomic_partners = epistasis_matrix[locus, :K]
    intergenomic_partners = epistasis_matrix[locus, K:]
    iesi = genome[locus]
    for i in range(K):
        partner = intragenomic_partners[i]
        iesi |= (genome[partner] << (i + 1))
    group_avg_values = group_avg_genome[intergenomic_partners]
    locus_coeffs = coefficients[locus, iesi, :]
    return evaluate_multilinear(locus_coeffs, group_avg_values)

@numba.njit
def calculateFitness(genome, group_avg_genome, epistasis_matrix, coefficients, K, K2, genome_size):
    total_fitness = 0.0
    for locus in range(genome_size):
        fitness_contribution = calculateLocusFitness(locus, genome, group_avg_genome, epistasis_matrix, coefficients, K, K2)
        total_fitness += fitness_contribution
    return total_fitness / genome_size

# Modified to skip the focal
@numba.njit
def calculateGroupFitnessExcludingFocal(pop_input, group_size, group_avg_genome, epistasis_matrix, coefficients, K, K2, genome_size, group_id, focal_id):
    total_group_fitness = 0.0
    count = 0
    for ind_id in range(group_size):
        if ind_id == focal_id:
            continue
        genome = pop_input[group_id, ind_id]
        total_group_fitness += calculateFitness(genome, group_avg_genome, epistasis_matrix, coefficients, K, K2, genome_size)
        count += 1
    return total_group_fitness / count

@numba.njit
def determineDominanceSample(pop_input, group_sizes_input, group_averages_input, epistasis_matrix, coefficients, K, K2, genome_size, group_number, max_group_size, sample_size):
    individual_deltas = np.zeros(sample_size)
    group_deltas = np.zeros(sample_size)
    total_possible_flips = group_number * max_group_size * genome_size
    actual_sample_count = min(sample_size, total_possible_flips)
    
    for i in range(actual_sample_count):
        group_id = np.random.randint(0, group_number)
        while group_sizes_input[group_id] <= 1:
            group_id = np.random.randint(0, group_number)
            
        focal_id = np.random.randint(0, group_sizes_input[group_id])
        locus_id = np.random.randint(0, genome_size)
        
        n = group_sizes_input[group_id]
        original_average_genome = group_averages_input[group_id]
        genome = pop_input[group_id, focal_id]
        
        # Calculate pre-mutation fitness values (excluding focal individual)
        original_individual_fitness = calculateFitness(genome, original_average_genome, epistasis_matrix, coefficients, K, K2, genome_size)
        original_group_fitness = calculateGroupFitnessExcludingFocal(pop_input, n, original_average_genome, epistasis_matrix, coefficients, K, K2, genome_size, group_id, focal_id)
        
        old_bit = genome[locus_id]
        new_bit = 1 - old_bit
        
        # Compute new group average genome vector
        new_average_genome = original_average_genome.copy()
        new_average_genome[locus_id] += (new_bit - old_bit) / n
        
        # Mutate the focal individual in a temporary copy of the population slice
        pop_input_copy = pop_input[group_id].copy()
        pop_input_copy[focal_id, locus_id] = new_bit
        
        # Wrapped structure to pass the modified group array to helper function
        temp_pop = np.zeros((1, max_group_size, genome_size), dtype=np.int8)
        temp_pop[0] = pop_input_copy
        
        # Calculate post-mutation fitness values (focal individual is still skipped, but others encounter the updated group average)
        new_focal_fitness = calculateFitness(pop_input_copy[focal_id], new_average_genome, epistasis_matrix, coefficients, K, K2, genome_size)
        new_group_fitness = calculateGroupFitnessExcludingFocal(temp_pop, n, new_average_genome, epistasis_matrix, coefficients, K, K2, genome_size, 0, focal_id)
        
        individual_deltas[i] = new_focal_fitness - original_individual_fitness
        group_deltas[i] = new_group_fitness - original_group_fitness
                
    return individual_deltas, group_deltas

## Function to collect data for multiple population for one K, K2 combination
def collect_deltas(K_val, K2_val, num_populations=1000, samples_per_pop=100, seed=1):
    genome_size = 16 
    max_group_size = 40 
    starting_group_size = 20 
    group_number = 500 
    dominance_sample_size = num_populations * samples_per_pop
    
    np.random.seed(seed)
    random.seed(seed)
    
    # Store results
    all_individual_deltas = np.zeros(dominance_sample_size)
    all_group_deltas = np.zeros(dominance_sample_size)
    
    idx = 0
    # For each population
    for p in range(num_populations):
        # Create a unique homogeneous genome
        genome = np.random.randint(0, 2, size=genome_size, dtype=np.int8)
        
        # Initialize population
        population = np.zeros((group_number, max_group_size, genome_size), dtype=np.int8)
        
        # Tile the genome
        for g in range(group_number):
            population[g, :starting_group_size] = genome

        group_sizes = np.full((group_number), starting_group_size, dtype = np.int8)
        group_averages = np.zeros((group_number, genome_size))
        
        for g in range(group_number):
            mask = (np.arange(max_group_size) < group_sizes[g])[:, np.newaxis]
            group_sums = np.sum(population[g] * mask, axis=0)
            group_averages[g] = group_sums / group_sizes[g]

        # Generate unique epistasis and coefficients
        epistasis_matrix = np.zeros((genome_size, K_val + K2_val), dtype = int)
        for i in range(genome_size):
            available_loci = np.delete(np.arange(genome_size), i)
            epistasis_matrix[i, :K_val] = np.random.choice(available_loci, size = K_val, replace = False)
            epistasis_matrix[i, K_val:] = np.random.choice(genome_size, size = K2_val, replace = False)

        fitness_matrix = np.random.beta(0.5, 0.5, size=(genome_size, 2**(K_val+K2_val+1)))
        coefficients = calculateCoefficients(fitness_matrix, K_val, K2_val, genome_size)
        
        # Sample from this population
        ind_d, grp_d = determineDominanceSample(
            population, group_sizes, group_averages, epistasis_matrix, 
            coefficients, K_val, K2_val, genome_size, group_number, max_group_size, samples_per_pop
        )
        
        # Save results
        all_individual_deltas[idx:idx+samples_per_pop] = ind_d
        all_group_deltas[idx:idx+samples_per_pop] = grp_d
        idx += samples_per_pop
        
    return all_individual_deltas, all_group_deltas

## Plotting

if __name__ == "__main__":
    # Dimensions
    num_pops = 1000
    samples_per = 100

    # Baseline (K=1, K2=1)
    print(f"Sampling deltas across {num_pops} populations for K=1, K2=1...")
    ind_k1_k2_1, grp_k1_k2_1 = collect_deltas(K_val=1, K2_val=1, num_populations=num_pops, samples_per_pop=samples_per)

    # Increase K (Individual Epistasis)
    print(f"Sampling deltas across {num_pops} populations for K=8, K2=1...")
    ind_k8_k2_1, grp_k8_k2_1 = collect_deltas(K_val=8, K2_val=1, num_populations=num_pops, samples_per_pop=samples_per)

    # Increase K2 (Group Epistasis)
    print(f"Sampling deltas across {num_pops} populations for K=1, K2=8...")
    ind_k1_k2_8, grp_k1_k2_8 = collect_deltas(K_val=1, K2_val=8, num_populations=num_pops, samples_per_pop=samples_per)

    # 2x3 plot layout
    fig, axes = plt.subplots(2, 3, figsize=(17, 10))

    # Row 0: Individual Deltas
    axes[0, 0].hist(ind_k1_k2_1, bins=50, color='salmon', edgecolor='black', alpha=0.75)
    axes[0, 0].axvline(0, color='red', linestyle='--', linewidth=1)
    axes[0, 0].set_xlim(-0.4, 0.4)
    axes[0, 0].set_title(f'Individual Deltas ($K=1, K_2=1$)', fontsize=12)
    axes[0, 0].set_ylabel('Frequency Count', fontsize=11)
    axes[0, 0].grid(True, linestyle=':', alpha=0.6)
    axes[0, 0].text(0.02, 0.95, f'Std Dev: {np.std(ind_k1_k2_1):.4f}', transform=axes[0, 0].transAxes, fontsize=10, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))

    axes[0, 1].hist(ind_k8_k2_1, bins=50, color='darkred', edgecolor='black', alpha=0.75)
    axes[0, 1].axvline(0, color='red', linestyle='--', linewidth=1)
    axes[0, 1].set_title(f'Individual Deltas ($K=8, K_2=1$)', fontsize=12)
    axes[0, 1].set_xlim(-0.4, 0.4)
    axes[0, 1].grid(True, linestyle=':', alpha=0.6)
    axes[0, 1].text(0.02, 0.95, f'Std Dev: {np.std(ind_k8_k2_1):.4f}', transform=axes[0, 1].transAxes, fontsize=10, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))

    axes[0, 2].hist(ind_k1_k2_8, bins=50, color='coral', edgecolor='black', alpha=0.75)
    axes[0, 2].axvline(0, color='red', linestyle='--', linewidth=1)
    axes[0, 2].set_title(f'Individual Deltas ($K=1, K_2=8$)', fontsize=12)
    axes[0, 2].set_xlim(-0.4, 0.4)
    axes[0, 2].grid(True, linestyle=':', alpha=0.6)
    axes[0, 2].text(0.02, 0.95, f'Std Dev: {np.std(ind_k1_k2_8):.4f}', transform=axes[0, 2].transAxes, fontsize=10, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))

    # Row 1: Group Deltas
    axes[1, 0].hist(grp_k1_k2_1, bins=50, color='skyblue', edgecolor='black', alpha=0.75)
    axes[1, 0].axvline(0, color='red', linestyle='--', linewidth=1)
    axes[1, 0].set_xlim(-0.02, 0.02)
    axes[1, 0].set_title(f'Group Deltas ($K=1, K_2=1$)', fontsize=12)
    axes[1, 0].set_xlabel('Delta Fitness ($\Delta$)', fontsize=11)
    axes[1, 0].set_ylabel('Frequency Count', fontsize=11)
    axes[1, 0].grid(True, linestyle=':', alpha=0.6)
    axes[1, 0].text(0.02, 0.95, f'Std Dev: {np.std(grp_k1_k2_1):.4f}', transform=axes[1, 0].transAxes, fontsize=10, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))

    axes[1, 1].hist(grp_k8_k2_1, bins=50, color='skyblue', edgecolor='black', alpha=0.75)
    axes[1, 1].axvline(0, color='red', linestyle='--', linewidth=1)
    axes[1, 1].set_title(f'Group Deltas ($K=8, K_2=1$)', fontsize=12)
    axes[1, 1].set_xlabel('Delta Fitness ($\Delta$)', fontsize=11)
    axes[1, 1].set_xlim(-0.02, 0.02)
    axes[1, 1].grid(True, linestyle=':', alpha=0.6)
    axes[1, 1].text(0.02, 0.95, f'Std Dev: {np.std(grp_k8_k2_1):.4f}', transform=axes[1, 1].transAxes, fontsize=10, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))

    axes[1, 2].hist(grp_k1_k2_8, bins=50, color='teal', edgecolor='black', alpha=0.75)
    axes[1, 2].axvline(0, color='red', linestyle='--', linewidth=1)
    axes[1, 2].set_title(f'Group Deltas ($K=1, K_2=8$)', fontsize=12)
    axes[1, 2].set_xlabel('Delta Fitness ($\Delta$)', fontsize=11)
    axes[1, 2].set_xlim(-0.02, 0.02)
    axes[1, 2].grid(True, linestyle=':', alpha=0.6)
    axes[1, 2].text(0.02, 0.95, f'Std Dev: {np.std(grp_k1_k2_8):.4f}', transform=axes[1, 2].transAxes, fontsize=10, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))

    plt.tight_layout()

    output_image = 'Delta Distributions.png'
    plt.savefig(output_image, dpi=300)
    print(f"Process complete. Image grid saved to '{output_image}'.")