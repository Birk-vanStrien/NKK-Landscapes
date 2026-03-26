import numpy as np
import random
import matplotlib.pyplot as plt
import numba

## Set up

# Show that simulation is setting up
print("Setting up simulation...")

# Parameters
genome_size = 16 # Number of loci in one genome
max_group_size = 40 # Maximum members in a group
starting_group_size = 20 # Group size at the start of the siumulation
group_number = 500 # Number of groups in population
mutation_rate = 0.001 # Mutation rate at each locus
group_split_rate = 0.1 # Chance group splits or one individual dies instead
K = 1 # Number of intragenomic partners
K2 = 8 # Number of intergenomic partners

# Simulation settings
endtime = 3000000

# Create initial population
initial_population = np.random.randint(0, 2, size = (group_number, max_group_size, genome_size), dtype = np.int8)
population = initial_population.copy()

# Keep track of group sizes
group_sizes = np.full((group_number), starting_group_size, dtype = np.int8)

# Keep track of group averages
# Mask 'True' only for indices within the actual group size
mask = np.arange(max_group_size) < group_sizes[:, np.newaxis]
# Sum values of each locus
group_sums = np.sum(population * mask[:, :, np.newaxis], axis=1)
# Get average at each locus
group_averages = group_sums / group_sizes[:, np.newaxis]



## Epistasis matrix

# Make empty matrix
epistasis_matrix = np.zeros((genome_size, K + K2), dtype = int)

# For each locus
for i in range(genome_size):
    # Pick K intragenomic partners (excluding itself)
    available_loci = np.delete(np.arange(genome_size), i) 
    intragenomic_partners = np.random.choice(available_loci, size = K, replace = False)

    # Pick K2 intergenomic partners (including itself)
    intergenomic_partners = np.random.choice(genome_size, size = K2, replace = False)

    # Store in epistasis matrix
    epistasis_matrix[i, :K] = intragenomic_partners
    epistasis_matrix[i, K:] = intergenomic_partners


## Fitness matrix

# Beta distribution
beta_a = 0.5
beta_b = 0.5

# Give every possible combination of binary values of relevant loci a random fitness value
fitness_matrix = np.random.beta(beta_a, beta_b, size=(genome_size, 2**(K+K2+1)))

##################################################################################################################

### Functions

## Fitness calculation functions

# Mobius transformation
def mobius_transform(fitness_values):
    num_coeffs = len(fitness_values)
    coefficients = np.zeros(num_coeffs, dtype=float)

    # Base case: coefficient for all-zeros corner (constant term)
    coefficients[0] = fitness_values[0]

    # Calculate remaining coefficients using inclusion-exclusion
    # Process in increasing order so dependencies are already computed
    for j in range(1, num_coeffs):
        subset_sum = 0.0

        # Sum coefficients for all proper subsets of j
        # A subset means: all bits set in l are also set in j
        for l in range(j):  # Only check l < j (proper subsets)
            # Bitwise AND: if l & j == l, then l is a subset of j
            if l == (l & j):
                subset_sum += coefficients[l]

        # Möbius inversion formula
        coefficients[j] = fitness_values[j] - subset_sum

    return coefficients

# Coefficients calculation
def calculateCoefficients(fitness_matrix, K, K2):
    num_IESI = 2 ** (K + 1)
    num_multilinear_coeffs = 2 ** K2

    # Initialize coefficient tensor
    coefficients = np.zeros((genome_size, num_IESI, num_multilinear_coeffs), dtype=float)

    # Process each locus
    for locus in range(genome_size):
        # Process each IESI (Intragenomic Epistatic State Index)
        for iesi in range(num_IESI):
            # Extract the 2^K2 fitness values for this sub-hypercube
            # These are the fitness values for all possible states of the
            # K' group-average partners, given this IESI

            # INDEXING SCHEME:
            # fitness_matrix column index is a binary number with:
            # - High-order bits (positions K' to K+K'): IESI (focal + K partners)
            # - Low-order bits (positions 0 to K'-1): group corner state

            sub_hypercube = np.zeros(num_multilinear_coeffs, dtype=float)
            for group_corner in range(num_multilinear_coeffs):
                # Combine IESI (high bits) with group_corner (low bits)
                # Left shift IESI by K' positions, then OR with group_corner
                fitness_index = (iesi << K2) | group_corner
                sub_hypercube[group_corner] = fitness_matrix[locus, fitness_index]

            # Apply Möbius transform to get multilinear coefficients
            coefficients[locus, iesi, :] = mobius_transform(sub_hypercube)

    return coefficients

# Multilinear evaluation
@numba.njit
def evaluate_multilinear(coefficients, group_avg_values):
    total = 0.0
    compensation = 0.0

    # Calculate each term in the multilinear expansion
    for j in range(len(coefficients)):
        # Start with the coefficient
        term = coefficients[j]

        # Multiply by group_avg_values[k] for each bit k that is set in j
        # Example: if j=5 (binary 101), multiply by values[0] and values[2]
        for k in range(len(group_avg_values)):
            # Check if bit k is set in j using bitwise operations
            if j & (1 << k):  # (1 << k) creates a mask with bit k set
                term *= group_avg_values[k]
        y = term - compensation
        t = total + y
        compensation = (t - total) - y
        total = t
    return total

# Calculate the fitness of a single locus
@numba.njit
def calculateLocusFitness(locus, genome, group_avg_genome, epistasis_matrix, coefficients, K, K2):
    # Look up epistatic partners for this locus from epistasis matrix
    intragenomic_partners = epistasis_matrix[locus, :K]      # First K columns
    intergenomic_partners = epistasis_matrix[locus, K:]      # Last K' columns

    # Calculate IESI: binary index from focal locus + K intragenomic partners
    # Start with the focal locus bit (0 or 1)
    iesi = genome[locus]

    # Add bits from the K intragenomic partners
    # Partner i goes in bit position (i+1)
    for i, partner in enumerate(intragenomic_partners):
        iesi |= (genome[partner] << (i + 1))  # OR operation to set bits

    # Get group-average values at the K2 intergenomic partner loci
    # These are floats in [0,1], not binary!
    group_avg_values = group_avg_genome[intergenomic_partners]

    # Get the appropriate coefficients for this locus and IESI
    # This selects the right K'-dimensional sub-hypercube
    locus_coeffs = coefficients[locus, iesi, :]

    # Evaluate the multilinear expansion
    # This computes the interpolated fitness for this locus
    fitness_contribution = evaluate_multilinear(locus_coeffs, group_avg_values)

    return fitness_contribution

# Calculate the fitness of the whole genome
@numba.njit
def calculateFitness(genome, group_avg_genome, epistasis_matrix, coefficients, K, K2):
    total_fitness = 0.0
    # Sum fitness contributions from all loci
    for locus in range(genome_size):
        fitness_contribution = calculateLocusFitness(
            locus, genome, group_avg_genome, epistasis_matrix, coefficients, K, K2
        )
        total_fitness += fitness_contribution

    # Return mean fitness (average across loci)
    return total_fitness / genome_size


## Recalculations after reproduction event

# Recalculate group average
def recalculateGroupAverage(group_id_input):
    # Mask 'True' only for indices less than the actual group size
    mask = (np.arange(max_group_size) < group_sizes[group_id_input])[:, np.newaxis]
    group_sums = np.sum(population[group_id_input] * mask, axis=0)
    group_averages[group_id_input] = group_sums / group_sizes[group_id_input]


# Recalculate fitness in one group
def recalculateGroupFitness(pop_input, group_id_input):
    updated_group_fitness = []
    
    # Store fitness of unique genomes encountered in this group
    fitness_cache = {}

    # For each filled slot in group
    for i in range(group_sizes[group_id_input]):
        # Get the genome
        genome = pop_input[group_id_input, i]
        genome_key = genome.tobytes() # Convert to hashable format
        
        # Check if fitness has already been calculated for this sequence
        if genome_key in fitness_cache:
            fit = fitness_cache[genome_key]
        else:
            # Calculate the fitness of this genome
            fit = calculateFitness(genome, group_averages[group_id_input], epistasis_matrix, coefficients, K, K2)
            # Store in cache for other identical individuals in this group
            fitness_cache[genome_key] = fit
            
        updated_group_fitness.append(fit)

    # Place this array in global fitness array
    start_idx = group_id_input * max_group_size
    end_idx = start_idx + max_group_size    
    
    group_slice = np.zeros(max_group_size)
    group_slice[:group_sizes[group_id_input]] = updated_group_fitness
    
    # Update the global array
    fitness_array[start_idx:end_idx] = group_slice


## Reproduction events

# Group is full and one member dies
def replaceMemberEvent(pop_input, group_id_input, newborn_genome_input):
    
    # Replace one unlucky member with newborn
    unlucky_index = np.random.choice(max_group_size)
    pop_input[group_id_input, [unlucky_index]] = newborn_genome_input
    # print(f"Individual {unlucky_index} in group {group_id_input} died")

    # If newborn is the same as the one it replaced, skip recalculations
    if np.array_equal(pop_input[group_id_input, unlucky_index], newborn_genome_input):
        return pop_input
    
    # Recalculate group average genome
    recalculateGroupAverage(group_id_input)

    # Recalculate fitness
    recalculateGroupFitness(pop_input, group_id_input)

    return pop_input

# Group is full and splits
def groupSplitEvent(pop_input, group_id_input, newborn_genome_input):
    # All groups except the current one
    left_over_groups = [i for i in range(group_number) if i != group_id_input]
    unlucky_group_index = np.random.choice(left_over_groups)
    # Set group size of unlucky group to 0
    group_sizes[unlucky_group_index] = 0

    # Boolean mask to split the group in a random way
    move_mask = np.random.choice([True, False], size = max_group_size, p = [0.5, 0.5])
        
    if not np.any(move_mask): # If all false, select 1 random to be true
        move_mask[np.random.randint(0, max_group_size)] = True
    elif np.all(move_mask):   # Vice versa
        move_mask[np.random.randint(0, max_group_size)] = False
            
    # Move those marked 'True' to new group
    moving_members = pop_input[group_id_input, move_mask, :]
    num_moving = len(moving_members)
    # Place them at the top of the new group
    pop_input[unlucky_group_index, 0:num_moving, :] = moving_members
    # Set group size to new amount
    group_sizes[unlucky_group_index] = num_moving
        
    # Move those marked 'False' to top of old group
    staying_members = pop_input[group_id_input, ~move_mask, :]
    num_staying = len(staying_members)
    # Place on top
    pop_input[group_id_input, 0:num_staying, :] = staying_members
    # Set group size to new amount
    group_sizes[group_id_input] = num_staying
    
    # Add newborn to old group
    # Place genone copy below the taken spaces, overwriting the 'garbage' data
    pop_input[group_id_input, group_sizes[group_id_input]] = newborn_genome_input
    # print(f"Reproduced member in group {group_id_input} at slot {group_sizes[group_id_input]}")

    # Increase group size so new individual is considered as 'actual' data
    group_sizes[group_id_input] += 1  
   
    # Recalculate group average genome of both groups
    recalculateGroupAverage(group_id_input)
    recalculateGroupAverage(unlucky_group_index)

    # Recalculate fitness in both groups
    recalculateGroupFitness(pop_input, group_id_input)  
    recalculateGroupFitness(pop_input, unlucky_group_index)

    return pop_input

# Group has space and one member is born
def birthEvent(pop_input, group_id_input, newborn_genome_input):

    # Place gemone copy below the taken spaces, overwriting the 'garbage' data
    pop_input[group_id_input, group_sizes[group_id_input]] = newborn_genome_input
    # print(f"Reproduced member in group {group_id_input} at slot {group_sizes[group_id_input]}")

    # Increase group size so new individual is considered as 'actual' data
    group_sizes[group_id_input] += 1
    
    # Recalculate group average genome
    recalculateGroupAverage(group_id_input)

    # Recalculate fitness
    recalculateGroupFitness(pop_input, group_id_input)

    return pop_input

# Mutation function
def mutationEvent(newborn_genome_input):
    #
    mutation_mask = np.random.choice([True, False], size = genome_size, p = [mutation_rate, 1-mutation_rate])

    # Flip those marked true
    newborn_genome_input[mutation_mask] = 1 - newborn_genome_input[mutation_mask]
    
    return newborn_genome_input


## Reproduction function

def reproductionEvent(pop_input):
    # Select a random individual from fitness array based on weight
    probabilities = fitness_array / np.sum(fitness_array)
    selected_index = np.random.choice(len(fitness_array), p=probabilities)
    # Get group and member id
    group_id = selected_index//max_group_size
    member_id = selected_index%max_group_size

    # Get genome of selected individual
    selected_genome = pop_input[group_id, member_id]
    genome_copy = selected_genome.copy()

    # Mutation event (has a chance within the function)
    mutationEvent(genome_copy)
    
    # Check if group is full
    if group_sizes[group_id] == max_group_size:
        # Either split or replace
        if random.random() > group_split_rate:
            replaceMemberEvent(pop_input, group_id, genome_copy)
        else:
            groupSplitEvent(pop_input, group_id, genome_copy)
    # Else just add new member to group
    else:
        birthEvent(pop_input, group_id, genome_copy)

## Analysis functions

# Calculate average fitness for each group
def calculateGroupFitnesses(fitness_array):
    # Make array
    average_fitness_per_group = np.zeros(group_number)
    # Go over each group
    for i in range(group_number):
        # Get the fitness values for this group
        group_fitness = fitness_array[i*max_group_size:(i+1)*max_group_size]
        # Only consider filled slots (fitness > 0)
        filled_fitness = group_fitness[group_fitness > 0]
        # Calculate average
        average_fitness_per_group[i] = np.mean(filled_fitness)
    return average_fitness_per_group

# Determine dominance
def determineDominance(pop_input, group_sizes_input, group_averages_input, epistasis_matrix, coefficients, K, K2):
    # Lists to store the fitness changes for every flip
    individual_deltas = []
    group_deltas = []
    
    # Go over each group
    for group_id in range(group_number):
        # Get the number of members in this group
        n = group_sizes_input[group_id]
        
        # Calculate original mean fitness of this group
        original_fitness = [calculateFitness(pop_input[group_id, j], group_averages_input[group_id], epistasis_matrix, coefficients, K, K2) for j in range(n)]
        original_mean = np.mean(original_fitness)
        original_average_genome = group_averages_input[group_id]
        
        for focal_id in range(n):
            original_individual_fitness = original_fitness[focal_id]
            genome = pop_input[group_id, focal_id]
            
            # Flip every locus in the individual's genome
            for locus_id in range(genome_size):
                old_bit = genome[locus_id]
                new_bit = 1 - old_bit
                
                # Update group average for this flip
                new_average_genome = original_average_genome.copy()
                new_average_genome[locus_id] += (new_bit - old_bit) / n
                
                # Calculate new fitness for all group members under the new environment except the focal individual, who has the flipped bit
                new_fitnesses = []
                for member_id in range(n):
                    member_genome = pop_input[group_id, member_id]
                    # If focal individual:
                    if member_id == focal_id:
                        # Calculate fitness with flipped bit
                        new_genome = member_genome.copy()
                        new_genome[locus_id] = new_bit
                        focal_fitness = calculateFitness(new_genome, new_average_genome, epistasis_matrix, coefficients, K, K2)
                    else:
                        # Recalculate finess
                        other_fit = calculateFitness(member_genome, new_average_genome, epistasis_matrix, coefficients, K, K2)
                        new_fitnesses.append(other_fit)
                
                # Store the deltas
                individual_deltas.append(focal_fitness - original_individual_fitness)
                group_deltas.append(np.mean(new_fitnesses) - original_mean)
                
    return individual_deltas, group_deltas

def determineDominanceSample(pop_input, group_sizes_input, group_averages_input, epistasis_matrix, coefficients, K, K2, sample_size):
    individual_deltas = []
    group_deltas = []
    
    # Create a list of all possible combinations (group, member, locus)
    all_possible_flips = []
    for g_id in range(group_number):
        for m_id in range(group_sizes_input[g_id]):
            for l_id in range(genome_size):
                all_possible_flips.append((g_id, m_id, l_id))
    
    # Randomly sample from the available triplets
    # Use min() to ensure that sample i not larger than the total number of possible flips
    actual_sample_count = min(sample_size, len(all_possible_flips))
    sampled_indices = random.sample(all_possible_flips, actual_sample_count)
    
    for group_id, focal_id, locus_id in sampled_indices:
        n = group_sizes_input[group_id]
        original_average_genome = group_averages_input[group_id]
        genome = pop_input[group_id, focal_id]
        
        # Calculate original fitnesses for the group
        original_individual_fitness = calculateFitness(genome, original_average_genome, epistasis_matrix, coefficients, K, K2)
        
        # Calculate mean fitness of other members in the group
        other_fitnesses = []
        for m_id in range(n):
            if m_id != focal_id:
                fit = calculateFitness(pop_input[group_id, m_id], original_average_genome, epistasis_matrix, coefficients, K, K2)
                other_fitnesses.append(fit)
        
        original_others_mean = np.mean(other_fitnesses) if other_fitnesses else 0

        # Flip
        old_bit = genome[locus_id]
        new_bit = 1 - old_bit
        
        # Update group average for this specific flip
        new_average_genome = original_average_genome.copy()
        new_average_genome[locus_id] += (new_bit - old_bit) / n
        
        # Calculate new fitnesses
        new_genome = genome.copy()
        new_genome[locus_id] = new_bit
        new_focal_fitness = calculateFitness(new_genome, new_average_genome, epistasis_matrix, coefficients, K, K2)
        
        # Others in the new group environment
        new_others_fitnesses = []
        for m_id in range(n):
            if m_id != focal_id:
                fit = calculateFitness(pop_input[group_id, m_id], new_average_genome, epistasis_matrix, coefficients, K, K2)
                new_others_fitnesses.append(fit)
        
        new_others_mean = np.mean(new_others_fitnesses) if new_others_fitnesses else 0
        
        # Store deltas
        # Effect on indidual
        individual_deltas.append(new_focal_fitness - original_individual_fitness)
        # Effect on the group
        group_deltas.append(new_others_mean - original_others_mean)
                
    return individual_deltas, group_deltas


# Calculate average deltas of the flips for both individual and group
def averageDeltas(individual_deltas, group_deltas):
    individual_deltas = np.array(individual_deltas)
    group_deltas = np.array(group_deltas)
    
    average_individual_delta = np.mean(individual_deltas)
    average_group_delta = np.mean(group_deltas)
    
    print(f"Average Individual Delta: {average_individual_delta:.4f}")
    print(f"Average Group Delta: {average_group_delta:.4f}")
    
    return average_individual_delta, average_group_delta

# Calculate ratio between consensus and conflict
def calculateAlignment(individual_deltas, group_deltas):
    individual_deltas = np.array(individual_deltas)
    group_deltas = np.array(group_deltas)
    
    # Get signs: positive = 1, negative = -1, zero = 0
    ind_signs = np.sign(individual_deltas)
    group_signs = np.sign(group_deltas)
    
    # Consensus: same sign (both 1 or both -1)
    consensus = np.sum(ind_signs == group_signs)
    # Conflict: opposite signs
    conflict = np.sum(ind_signs == -group_signs)

    if conflict == 0:
        return np.inf
    
    alignment = consensus / conflict
    return alignment

# Calculate weighted alignment ratio
def calculateWeightedAlignment(individual_deltas, group_deltas):
    individual_deltas = np.array(individual_deltas)
    group_deltas = np.array(group_deltas)

    consensus = np.mean(np.maximum(0, individual_deltas) * np.maximum(0, group_deltas)) + np.mean(np.maximum(0, -individual_deltas) * np.maximum(0, -group_deltas))
    conflict = np.mean(np.maximum(0, -individual_deltas) * np.maximum(0, group_deltas)) + np.mean(np.maximum(0, individual_deltas) * np.maximum(0, -group_deltas))

    if conflict == 0:
        return float('inf') if consensus > 0 else 0.0
    
    weighted_alignment = consensus / conflict
    return weighted_alignment

# Calculate dominance ratio (group dominance / individual dominance)
def calculateDominanceRatio(individual_deltas, group_deltas):
    individual_deltas = np.array(individual_deltas)
    group_deltas = np.array(group_deltas)
    
    # Get signs: positive = 1, negative = -1, zero = 0
    ind_signs = np.sign(individual_deltas)
    group_signs = np.sign(group_deltas)

    group_dominance = np.sum((ind_signs > 0) & (group_signs < 0))
    individual_dominance = np.sum((ind_signs < 0) & (group_signs > 0))

    if individual_dominance == 0:
        return np.inf

    return group_dominance / individual_dominance

# Calculate weighted dominance ratio
def weightedDominanceRatio(individual_deltas, group_deltas):
    individual_deltas = np.array(individual_deltas)
    group_deltas = np.array(group_deltas)

    group_dominance = np.sum(np.maximum(0, individual_deltas) * np.maximum(0, -group_deltas))
    individual_dominance = np.sum(np.maximum(0, -individual_deltas) * np.maximum(0, group_deltas))

    if individual_dominance == 0:
        return float('inf') 

    return group_dominance / individual_dominance

################################################################################################################################

### Main

## Data collection settings
dominance_sample_size = 5000
temporal_resolution = 10000

## Calculate coefficients

coefficients = calculateCoefficients(fitness_matrix, K, K2)

## Make fitness array with fitness of each individual

fitness_table = np.zeros((group_number, max_group_size))

# Calculate fitness for filled slots
# For each group...
for i in range(group_number):
    # Calculate fitness of each genome
    for ii in range(group_sizes[i]):
        fitness_table[i, ii] = calculateFitness(
            population[i, ii], group_averages[i], epistasis_matrix, coefficients, K, K2
        )

# Flatten to 1D array
fitness_array = fitness_table.flatten()

## Simulation loop

# Average fitness of initial population (excluding empty slots)
initial_average_fitness = np.mean(fitness_array[fitness_array > 0])

# Track average fitness across timesteps
average_fitness_over_time = []

# Track dominance plot snapshots at specific timesteps
dominance_plot_snapshots = []
checkpoints = [0, endtime//5, 2*endtime//5, 3*endtime//5, 4*endtime//5, endtime]

# Track quadrant counts over time
suboptimal_counts = []
group_dominance_counts = []
individual_dominance_counts = []
agreement_counts = []
unchanged_counts = []

# Track average deltas over time
average_individual_deltas_over_time = []
average_group_deltas_over_time = []

# Track ratios over time
alignments_over_time = []
weighted_alignments_over_time = []
dominance_ratios_over_time = []
weighted_dominance_ratios_over_time = []

# Loop
for i in range(endtime+1):
    reproductionEvent(population)
    if i % temporal_resolution == 0: # Print every x timesteps
        average_fitness = np.mean(fitness_array[fitness_array > 0])
        average_fitness_over_time.append(average_fitness)
        print(f"Timestep {i}/{endtime} - Average Fitness: {average_fitness:.4f}")
        # calculateGroupFitnesses(fitness_array)
        # individual_delta, group_delta = averageDeltas(*determineDominanceSample(population, group_sizes, group_averages, epistasis_matrix, coefficients, K, K2, dominance_sample_size))
        # average_individual_deltas_over_time.append(individual_delta)
        # average_group_deltas_over_time.append(group_delta)

        # get individual and group deltas
        ind_d, grp_d = determineDominanceSample(population, group_sizes, group_averages, epistasis_matrix, coefficients, K, K2, dominance_sample_size)
        ind_d = np.array(ind_d)
        grp_d = np.array(grp_d)
        # Calculate quadrant counts
        suboptimal = ((ind_d >= 0) & (grp_d >= 0)).sum()
        group_dominance = ((ind_d > 0) & (grp_d < 0)).sum()
        individual_dominance = ((ind_d < 0) & (grp_d > 0)).sum()
        agree = ((ind_d <= 0) & (grp_d <= 0)).sum()
        # unchanged = ((ind_d == 0) & (grp_d == 0)).sum()
        suboptimal_counts.append(suboptimal)
        group_dominance_counts.append(group_dominance)
        individual_dominance_counts.append(individual_dominance)
        agreement_counts.append(agree)
        # unchanged_counts.append(unchanged)
        print(f"Quadrant Counts:")
        print(f"suboptimal (0+/0+): {suboptimal}")
        print(f"Group Dominance (+/-): {group_dominance}")
        print(f"Individual Dominance (-/+): {individual_dominance}")
        print(f"Agreement (0-/0-): {agree}")
        # print(f"Unchanged (0/0): {unchanged}")
        # print(f"Alignment (consensus/conflict): {calculateAlignment(ind_d, grp_d):.4f}")
        # print(f"Weighted alignment (consensus/conflict): {calculateWeightedAlignment(ind_d, grp_d):.4f}")
        # print(f"Dominance ratio (group/individual): {calculateDominanceRatio(ind_d, grp_d):.4f}")
        # print(f"Weighted dominance ratio (group/individual): {weightedDominanceRatio(ind_d, grp_d):.4f}")
        
        # Store ratios over time
        alignments_over_time.append(calculateAlignment(ind_d, grp_d))
        weighted_alignments_over_time.append(calculateWeightedAlignment(ind_d, grp_d))
        dominance_ratios_over_time.append(calculateDominanceRatio(ind_d, grp_d))
        weighted_dominance_ratios_over_time.append(weightedDominanceRatio(ind_d, grp_d))

    # Capture snapshots for dominance plot at specific checkpoints
    if i in checkpoints:
        print(f"Snapshot captured at T={i}")
        d_ind, d_grp = determineDominanceSample(population, group_sizes, group_averages, epistasis_matrix, coefficients, K, K2, dominance_sample_size)
        dominance_plot_snapshots.append((i, d_ind, d_grp))


## Save data

# Define the file name
data_filename = f"{genome_size}-{max_group_size}-{group_number} K={K} K2={K2} S={group_split_rate} M={mutation_rate} T={endtime/1000000}M.npz"

# define timesteps
timesteps = np.arange(0, endtime+1, temporal_resolution) 

# Save all relevant tracking lists and snapshot data
np.savez(
    data_filename,
    timesteps=timesteps,
    average_fitness_over_time=np.array(average_fitness_over_time),
    suboptimal_counts=np.array(suboptimal_counts),
    group_dominance_counts=np.array(group_dominance_counts),
    individual_dominance_counts=np.array(individual_dominance_counts),
    agreement_counts=np.array(agreement_counts),
    alignments_over_time=np.array(alignments_over_time),
    weighted_alignments_over_time=np.array(weighted_alignments_over_time),
    dominance_ratios_over_time=np.array(dominance_ratios_over_time),
    weighted_dominance_ratios_over_time=np.array(weighted_dominance_ratios_over_time),
    snapshot_times=np.array([s[0] for s in dominance_plot_snapshots]),
    snapshot_ind_deltas=np.array([s[1] for s in dominance_plot_snapshots], dtype=object),
    snapshot_grp_deltas=np.array([s[2] for s in dominance_plot_snapshots], dtype=object)
)

print(f"Data saved to {data_filename}")