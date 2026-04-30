import numpy as np
import random
import numba
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

## Parameters
genome_size = 16 
max_group_size = 40 
starting_group_size = 20 
group_number = 500 
mutation_rate = 0.001 
group_split_rate = 1 
K = 1
K2 = 1 
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
    # Recalculate group-wide data
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

def calculateGroupFitnesses(fitness_array, group_number, max_group_size):
    average_fitness_per_group = np.zeros(group_number)
    for i in range(group_number):
        group_fitness = fitness_array[i*max_group_size:(i+1)*max_group_size]
        filled_fitness = group_fitness[group_fitness > 0]
        average_fitness_per_group[i] = np.mean(filled_fitness) if len(filled_fitness) > 0 else 0
    return average_fitness_per_group

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
        
        original_individual_fitness = calculateFitness(genome, original_average_genome, epistasis_matrix, coefficients, K, K2, genome_size)
        
        sum_others_orig = 0.0
        for m_id in range(n):
            if m_id != focal_id:
                sum_others_orig += calculateFitness(pop_input[group_id, m_id], original_average_genome, epistasis_matrix, coefficients, K, K2, genome_size)
        original_others_mean = sum_others_orig / (n-1)

        old_bit = genome[locus_id]
        new_bit = 1 - old_bit
        new_average_genome = original_average_genome.copy()
        new_average_genome[locus_id] += (new_bit - old_bit) / n
        
        new_genome = genome.copy()
        new_genome[locus_id] = new_bit
        new_focal_fitness = calculateFitness(new_genome, new_average_genome, epistasis_matrix, coefficients, K, K2, genome_size)
        
        sum_others_new = 0.0
        for m_id in range(n):
            if m_id != focal_id:
                sum_others_new += calculateFitness(pop_input[group_id, m_id], new_average_genome, epistasis_matrix, coefficients, K, K2, genome_size)
        new_others_mean = sum_others_new / (n-1)
        
        individual_deltas[i] = new_focal_fitness - original_individual_fitness
        group_deltas[i] = new_others_mean - original_others_mean
                
    return individual_deltas, group_deltas

def createTrackedSample(group_number, max_group_size, genome_size, sample_size):
    all_possible_flips = []
    for g_id in range(group_number):
        for m_id in range(max_group_size):
            for l_id in range(genome_size):
                all_possible_flips.append((g_id, m_id, l_id))
    actual_sample_count = min(sample_size, len(all_possible_flips))
    return random.sample(all_possible_flips, actual_sample_count)

@numba.njit
def locusAnalysis(pop_input, group_sizes_input, group_averages_input, epistasis_matrix, coefficients, K, K2, track_sampled_indices, genome_size):
    results = []
    for group_id, focal_id, locus_id in track_sampled_indices:
        n = group_sizes_input[group_id]
        if focal_id >= n:
            results.append("REMOVED")
            continue

        focal_genome = pop_input[group_id, focal_id]
        original_avg = group_averages_input[group_id]

        genome_0 = focal_genome.copy(); genome_0[locus_id] = 0
        genome_1 = focal_genome.copy(); genome_1[locus_id] = 1

        current_bit = focal_genome[locus_id]
        sum_others_locus = (original_avg[locus_id] * n) - current_bit 
        
        group_avg_0 = original_avg.copy(); group_avg_0[locus_id] = sum_others_locus / n
        group_avg_1 = original_avg.copy(); group_avg_1[locus_id] = (sum_others_locus + 1) / n

        fit_ind_0 = calculateFitness(genome_0, group_avg_0, epistasis_matrix, coefficients, K, K2, genome_size)
        fit_ind_1 = calculateFitness(genome_1, group_avg_1, epistasis_matrix, coefficients, K, K2, genome_size)

        sum_others_0 = 0.0; sum_others_1 = 0.0
        for m_id in range(n):
            if m_id != focal_id:
                member_genome = pop_input[group_id, m_id]
                sum_others_0 += calculateFitness(member_genome, group_avg_0, epistasis_matrix, coefficients, K, K2, genome_size)
                sum_others_1 += calculateFitness(member_genome, group_avg_1, epistasis_matrix, coefficients, K, K2, genome_size)

        mean_others_0 = sum_others_0 / (n - 1) if n > 1 else 0.0
        mean_others_1 = sum_others_1 / (n - 1) if n > 1 else 0.0

        ind_01 = "+" if (fit_ind_1 - fit_ind_0) > 0 else "-"
        grp_01 = "+" if (mean_others_1 - mean_others_0) > 0 else "-"
        results.append(ind_01 + grp_01)
    return results

def calculateAlignment(individual_deltas, group_deltas):
    ind_signs = np.sign(individual_deltas)
    group_signs = np.sign(group_deltas)
    consensus = np.sum(ind_signs == group_signs)
    conflict = np.sum(ind_signs == -group_signs)
    return consensus / conflict if conflict != 0 else np.inf

def calculateWeightedAlignment(individual_deltas, group_deltas):
    consensus = np.mean(np.maximum(0, individual_deltas) * np.maximum(0, group_deltas)) + np.mean(np.maximum(0, -individual_deltas) * np.maximum(0, -group_deltas))
    conflict = np.mean(np.maximum(0, -individual_deltas) * np.maximum(0, group_deltas)) + np.mean(np.maximum(0, individual_deltas) * np.maximum(0, -group_deltas))
    return consensus / conflict if conflict != 0 else np.inf

def calculateDominanceRatio(individual_deltas, group_deltas):
    ind_signs = np.sign(individual_deltas)
    group_signs = np.sign(group_deltas)
    group_dom = np.sum((ind_signs > 0) & (group_signs < 0))
    ind_dom = np.sum((ind_signs < 0) & (group_signs > 0))
    return group_dom / ind_dom if ind_dom != 0 else np.inf

def weightedDominanceRatio(individual_deltas, group_deltas):
    group_dom = np.sum(np.maximum(0, individual_deltas) * np.maximum(0, -group_deltas))
    ind_dom = np.sum(np.maximum(0, -individual_deltas) * np.maximum(0, group_deltas))
    return group_dom / ind_dom if ind_dom != 0 else np.inf

def findOptimalFitness(epistasis_matrix, coefficients, K, K2, genome_size):
    best_fitness = 0.0
    for i in range(2**genome_size):
        genome = np.array([int(x) for x in np.binary_repr(i, width = genome_size)], dtype = np.int8)
        fitness = calculateFitness(genome, genome, epistasis_matrix, coefficients, K, K2, genome_size)
        if fitness > best_fitness: best_fitness = fitness
    return best_fitness

################################################################################################################################

### Simulation logic

def run_simulation(sim_id):
    # LOCAL INITIALIZATION
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

    optimal_fitness = findOptimalFitness(epistasis_matrix, coefficients, K, K2, genome_size)
    print(f"Simulation {sim_id} - Optimal Fitness: {optimal_fitness:.4f}")

    # Data collection settings
    dominance_sample_size = 5000
    track_amount = 5000
    temporal_resolution = 10000
    locus_analysis_resolution = 50

    average_fitness_over_time = []
    suboptimal_counts, group_dominance_counts, individual_dominance_counts, agreement_counts = [], [], [], []
    alignments_over_time, weighted_alignments_over_time, dominance_ratios_over_time, weighted_dominance_ratios_over_time = [], [], [], []
    accommodation_history, enforcement_history, unchanged_agreement_history, unchanged_conflict_history = [], [], [], []
    com_group_history, com_individual_history, agreed_swap_history, conflicted_swap_history, removed_locus_history = [], [], [], [], []
    
    checkpoints = [0, endtime//5, 2*endtime//5, 3*endtime//5, 4*endtime//5, endtime]
    dominance_plot_snapshots = []

    tracked_indices = createTrackedSample(group_number, max_group_size, genome_size, track_amount)
    initial_locus_results = locusAnalysis(population, group_sizes, group_averages, epistasis_matrix, coefficients, K, K2, tracked_indices, genome_size)

    for i in range(endtime+1):
        if i % temporal_resolution == 0:
            print(f"Simulation {sim_id}, Timestep {i}/{endtime}")
            avg_fit = np.mean(fitness_array[fitness_array > 0])
            average_fitness_over_time.append(avg_fit)
            ind_d, grp_d = determineDominanceSample(population, group_sizes, group_averages, epistasis_matrix, coefficients, K, K2, genome_size, group_number, max_group_size, dominance_sample_size)
            
            suboptimal_counts.append(((ind_d >= 0) & (grp_d >= 0)).sum())
            group_dominance_counts.append(((ind_d > 0) & (grp_d < 0)).sum())
            individual_dominance_counts.append(((ind_d < 0) & (grp_d > 0)).sum())
            agreement_counts.append(((ind_d <= 0) & (grp_d <= 0)).sum())
            
            alignments_over_time.append(calculateAlignment(ind_d, grp_d))
            weighted_alignments_over_time.append(calculateWeightedAlignment(ind_d, grp_d))
            dominance_ratios_over_time.append(calculateDominanceRatio(ind_d, grp_d))
            weighted_dominance_ratios_over_time.append(weightedDominanceRatio(ind_d, grp_d))

        if i in checkpoints:
            d_ind, d_grp = determineDominanceSample(population, group_sizes, group_averages, epistasis_matrix, coefficients, K, K2, genome_size, group_number, max_group_size, dominance_sample_size)
            dominance_plot_snapshots.append((i, d_ind, d_grp))
        
        if i % (endtime//locus_analysis_resolution) == 0:
            current_locus_results = locusAnalysis(population, group_sizes, group_averages, epistasis_matrix, coefficients, K, K2, tracked_indices, genome_size)
            acc, enf, un_ag, un_co, com_g, com_i, a_sw, c_sw, rem = 0, 0, 0, 0, 0, 0, 0, 0, 0
            for prev, curr in zip(initial_locus_results, current_locus_results):
                if curr == "REMOVED": rem += 1; continue
                if ((prev == "+-" or prev == "REMOVED") and curr == "++") or ((prev == "-+" or prev == "REMOVED") and curr == "--"): acc += 1
                elif ((prev == "+-" or prev == "REMOVED") and curr == "--") or ((prev == "-+" or prev == "REMOVED") and curr == "++"): enf += 1
                elif ((prev == "++" or prev == "REMOVED") and curr == "++") or ((prev == "--" or prev == "REMOVED") and curr == "--"): un_ag += 1
                elif ((prev == "+-" or prev == "REMOVED") and curr == "+-") or ((prev == "-+" or prev == "REMOVED") and curr == "-+"): un_co += 1
                elif ((prev == "--" or prev == "REMOVED") and curr == "+-") or ((prev == "++" or prev == "REMOVED") and curr == "-+"): com_g += 1
                elif ((prev == "--" or prev == "REMOVED") and curr == "-+") or ((prev == "++" or prev == "REMOVED") and curr == "+-"): com_i += 1
                elif ((prev == "--" or prev == "REMOVED") and curr == "++") or ((prev == "++" or prev == "REMOVED") and curr == "--"): a_sw += 1
                elif ((prev == "-+" or prev == "REMOVED") and curr == "+-") or ((prev == "+-" or prev == "REMOVED") and curr == "-+"): c_sw += 1
                else: rem += 1
            accommodation_history.append(acc); enforcement_history.append(enf); unchanged_agreement_history.append(un_ag); unchanged_conflict_history.append(un_co)
            com_group_history.append(com_g); com_individual_history.append(com_i); agreed_swap_history.append(a_sw); conflicted_swap_history.append(c_sw); removed_locus_history.append(rem)

        reproductionEvent(population, group_sizes, group_averages, fitness_array, epistasis_matrix, coefficients, K, K2, genome_size, max_group_size, group_number, mutation_rate, group_split_rate)

    # Save data
    data_filename = f"{genome_size}-{max_group_size}-{group_number} K={K} K2={K2} S={group_split_rate} M={mutation_rate} T={endtime/1000000}M Sim_{sim_id}.npz"
    np.savez(data_filename, 
            timesteps = np.arange(0, endtime+1, temporal_resolution),
            dominance_sample_size = dominance_sample_size,
            track_amount = track_amount,
            average_fitness_over_time = np.array(average_fitness_over_time),
            suboptimal_counts = np.array(suboptimal_counts),
            group_dominance_counts = np.array(group_dominance_counts),
            individual_dominance_counts = np.array(individual_dominance_counts),
            agreement_counts = np.array(agreement_counts),
            alignments_over_time = np.array(alignments_over_time),
            weighted_alignments_over_time = np.array(weighted_alignments_over_time),
            dominance_ratios_over_time = np.array(dominance_ratios_over_time),
            weighted_dominance_ratios_over_time = np.array(weighted_dominance_ratios_over_time),
            snapshot_times = np.array([s[0] for s in dominance_plot_snapshots]),
            snapshot_ind_deltas = np.array([s[1] for s in dominance_plot_snapshots], dtype=object),
            snapshot_grp_deltas = np.array([s[2] for s in dominance_plot_snapshots], dtype=object),
            analysis_timesteps = np.array([i for i in range(endtime+1) if i % (endtime//locus_analysis_resolution) == 0]),
            accommodation_history = np.array(accommodation_history),
            enforcement_history = np.array(enforcement_history),
            unchanged_agreement_history = np.array(unchanged_agreement_history),
            unchanged_conflict_history = np.array(unchanged_conflict_history),
            com_group_history = np.array(com_group_history),
            com_individual_history = np.array(com_individual_history),
            agreed_swap_history = np.array(agreed_swap_history),
            conflicted_swap_history = np.array(conflicted_swap_history),
            removed_locus_history = np.array(removed_locus_history),
            optimal_fitness = optimal_fitness
        )
    print(f"Simulation {sim_id} complete. Saved to {data_filename}")

if __name__ == "__main__":
    num_simulations = 10
    cores_to_use = min(multiprocessing.cpu_count(), num_simulations)
    print(f"Starting {num_simulations} simulations on {cores_to_use} cores...")
    with ProcessPoolExecutor(max_workers=cores_to_use) as executor:
        list(executor.map(run_simulation, range(num_simulations)))