import numpy as np
import random
import numba
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from collections import Counter

## Parameters
genome_size = 16 
max_group_size = 40 
starting_group_size = 20 
group_number = 500 
mutation_rate = 0.001 
group_split_rate = 1
K = 8
K2 = 8
endtime = 200000000

##################################################################################################################

### Functions

## Fitness calculation functions

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

## Recalculations

def recalculateGroupAverage(group_id, population, group_sizes, group_averages, max_group_size):
    mask = (np.arange(max_group_size) < group_sizes[group_id])[:, np.newaxis]
    group_sums = np.sum(population[group_id] * mask, axis=0)
    group_averages[group_id] = group_sums / group_sizes[group_id]

def recalculateGroupFitness(pop_input, group_id, group_sizes, group_averages, fitness_array, epistasis_matrix, coefficients, K, K2, genome_size, max_group_size):
    updated_group_fitness = []
    fitness_cache = {}
    for i in range(group_sizes[group_id]):
        genome = pop_input[group_id, i]
        genome_key = genome.tobytes()
        if genome_key in fitness_cache:
            fit = fitness_cache[genome_key]
        else:
            fit = calculateFitness(genome, group_averages[group_id], epistasis_matrix, coefficients, K, K2, genome_size)
            fitness_cache[genome_key] = fit
        updated_group_fitness.append(fit)

    start_idx = group_id * max_group_size
    end_idx = start_idx + max_group_size    
    group_slice = np.zeros(max_group_size)
    group_slice[:group_sizes[group_id]] = updated_group_fitness
    fitness_array[start_idx:end_idx] = group_slice

## Reproduction events

def replaceMemberEvent(pop_input, group_id, newborn_genome, group_sizes, group_averages, fitness_array, epistasis_matrix, coefficients, K, K2, genome_size, max_group_size):
    unlucky_index = np.random.choice(max_group_size)
    pop_input[group_id, unlucky_index] = newborn_genome
    recalculateGroupAverage(group_id, pop_input, group_sizes, group_averages, max_group_size)
    recalculateGroupFitness(pop_input, group_id, group_sizes, group_averages, fitness_array, epistasis_matrix, coefficients, K, K2, genome_size, max_group_size)

def groupSplitEvent(pop_input, group_id, newborn_genome, group_sizes, group_averages, fitness_array, epistasis_matrix, coefficients, K, K2, genome_size, max_group_size, group_number):
    left_over_groups = [i for i in range(group_number) if i != group_id]
    unlucky_group_index = np.random.choice(left_over_groups)
    group_sizes[unlucky_group_index] = 0

    move_mask = np.random.choice([True, False], size=max_group_size, p=[0.5, 0.5])
    if not np.any(move_mask): move_mask[np.random.randint(0, max_group_size)] = True
    elif np.all(move_mask): move_mask[np.random.randint(0, max_group_size)] = False
            
    moving_members = pop_input[group_id, move_mask, :]
    num_moving = len(moving_members)
    pop_input[unlucky_group_index, 0:num_moving, :] = moving_members
    group_sizes[unlucky_group_index] = num_moving
        
    staying_members = pop_input[group_id, ~move_mask, :]
    num_staying = len(staying_members)
    pop_input[group_id, 0:num_staying, :] = staying_members
    group_sizes[group_id] = num_staying
    
    pop_input[group_id, group_sizes[group_id]] = newborn_genome
    group_sizes[group_id] += 1  
   
    for gid in [group_id, unlucky_group_index]:
        recalculateGroupAverage(gid, pop_input, group_sizes, group_averages, max_group_size)
        recalculateGroupFitness(pop_input, gid, group_sizes, group_averages, fitness_array, epistasis_matrix, coefficients, K, K2, genome_size, max_group_size)

def birthEvent(pop_input, group_id, newborn_genome, group_sizes, group_averages, fitness_array, epistasis_matrix, coefficients, K, K2, genome_size, max_group_size):
    pop_input[group_id, group_sizes[group_id]] = newborn_genome
    group_sizes[group_id] += 1
    recalculateGroupAverage(group_id, pop_input, group_sizes, group_averages, max_group_size)
    recalculateGroupFitness(pop_input, group_id, group_sizes, group_averages, fitness_array, epistasis_matrix, coefficients, K, K2, genome_size, max_group_size)

def mutationEvent(newborn_genome, mutation_rate, genome_size):
    mutation_mask = np.random.choice([True, False], size=genome_size, p=[mutation_rate, 1-mutation_rate])
    newborn_genome[mutation_mask] = 1 - newborn_genome[mutation_mask]

def reproductionEvent(pop_input, group_sizes, group_averages, fitness_array, epistasis_matrix, coefficients, K, K2, genome_size, max_group_size, group_number, mutation_rate, group_split_rate):
    probabilities = fitness_array / np.sum(fitness_array)
    selected_index = np.random.choice(len(fitness_array), p=probabilities)
    group_id = selected_index // max_group_size
    member_id = selected_index % max_group_size

    genome_copy = pop_input[group_id, member_id].copy()
    mutationEvent(genome_copy, mutation_rate, genome_size)
    
    if group_sizes[group_id] == max_group_size:
        if random.random() > group_split_rate:
            replaceMemberEvent(pop_input, group_id, genome_copy, group_sizes, group_averages, fitness_array, epistasis_matrix, coefficients, K, K2, genome_size, max_group_size)
        else:
            groupSplitEvent(pop_input, group_id, genome_copy, group_sizes, group_averages, fitness_array, epistasis_matrix, coefficients, K, K2, genome_size, max_group_size, group_number)
    else:
        birthEvent(pop_input, group_id, genome_copy, group_sizes, group_averages, fitness_array, epistasis_matrix, coefficients, K, K2, genome_size, max_group_size)

## Analysis functions

def calculate_shannon_diversities(population, group_sizes, group_number):
    total_population_size = np.sum(group_sizes)
    if total_population_size == 0:
        return 0.0, 0.0, 0.0

    all_genomes = []
    H_g_list = []
    w_g_list = []

    for g in range(group_number):
        size = group_sizes[g]
        if size == 0:
            continue
        
        # Relative group size weight w_g
        w_g = size / total_population_size
        w_g_list.append(w_g)
        
        # Convert genomes to bytes
        group_genomes_bytes = [genome.tobytes() for genome in population[g, :size]]
        all_genomes.extend(group_genomes_bytes)
        
        # Entropy H_g within group g
        counts_g = Counter(group_genomes_bytes)
        H_g = 0.0
        for count in counts_g.values():
            f_ig = count / size
            H_g -= f_ig * np.log2(f_ig)
            
        H_g_list.append(H_g)

    # Total diversity H
    total_counts = Counter(all_genomes)
    H = 0.0
    for count in total_counts.values():
        f_i = count / total_population_size
        H -= f_i * np.log2(f_i)

    # Within-group diversity W
    W = np.sum(np.array(w_g_list) * np.array(H_g_list))

    # Among-group diversity I
    I = H - W

    return H, W, I

################################################################################################################################

### Simulation logic

def run_simulation(sim_id):
    np.random.seed(sim_id)
    random.seed(sim_id)
    
    population = np.random.randint(0, 2, size = (group_number, max_group_size, genome_size), dtype = np.int8)
    group_sizes = np.full((group_number), starting_group_size, dtype = np.int8)
    group_averages = np.zeros((group_number, genome_size))
    
    for g in range(group_number):
        recalculateGroupAverage(g, population, group_sizes, group_averages, max_group_size)

    epistasis_matrix = np.zeros((genome_size, K + K2), dtype = int)
    for i in range(genome_size):
        available_loci = np.delete(np.arange(genome_size), i)
        epistasis_matrix[i, :K] = np.random.choice(available_loci, size = K, replace = False)
        epistasis_matrix[i, K:] = np.random.choice(genome_size, size = K2, replace = False)

    fitness_matrix = np.random.beta(0.5, 0.5, size=(genome_size, 2**(K+K2+1)))
    coefficients = calculateCoefficients(fitness_matrix, K, K2, genome_size)
    
    fitness_array = np.zeros(group_number * max_group_size)
    for g in range(group_number):
        recalculateGroupFitness(population, g, group_sizes, group_averages, fitness_array, epistasis_matrix, coefficients, K, K2, genome_size, max_group_size)

    print(f"Simulation {sim_id}")

    # Data collection settings
    temporal_resolution = 1000000

    # Metric tracking over time
    total_diversity_H_over_time = []
    within_diversity_W_over_time = []
    among_diversity_I_over_time = []

    for i in range(endtime + 1):
        if i % temporal_resolution == 0:
            print(f"Simulation {sim_id}, Timestep {i}/{endtime}")
            
            H, W, I = calculate_shannon_diversities(population, group_sizes, group_number)
            total_diversity_H_over_time.append(H)
            within_diversity_W_over_time.append(W)
            among_diversity_I_over_time.append(I)
            
            print(f"Total H: {H:.4f} | Within W: {W:.4f} | Among I: {I:.4f}")

        reproductionEvent(population, group_sizes, group_averages, fitness_array, epistasis_matrix, coefficients, K, K2, genome_size, max_group_size, group_number, mutation_rate, group_split_rate)

    # Save data
    data_filename = f"{genome_size}-{max_group_size}-{group_number} K={K} K2={K2} S={group_split_rate} M={mutation_rate} T={endtime/1000000}M Sim={sim_id} Top2Genomes.npz"
    np.savez(
        data_filename, 
        timesteps = np.arange(0, endtime+1, temporal_resolution),
        total_diversity_H = total_diversity_H_over_time,
        within_diversity_W = within_diversity_W_over_time,
        among_diversity_I = among_diversity_I_over_time,
        population = population,
        group_sizes = group_sizes,
        epistasis_matrix = epistasis_matrix,
        coefficients = coefficients,
        K = K,
        K2 = K2,
        genome_size = genome_size,
        group_number = group_number
    )
    print(f"Simulation {sim_id} complete. Saved to {data_filename}")

if __name__ == "__main__":
    num_simulations = 40
    cores_to_use = min(multiprocessing.cpu_count(), num_simulations)
    print(f"Starting {num_simulations} simulations on {cores_to_use} cores...")
    with ProcessPoolExecutor(max_workers=cores_to_use) as executor:
        list(executor.map(run_simulation, range(num_simulations)))