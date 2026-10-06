import numpy as np
import matplotlib.pyplot as plt

### Setup
num_simulations = 40 # Number of simulations to process
filename_base = "16-40-500 K=8 K2=2 S=1 M=0.001 T=200.0M Sim_{}.npz" # Determines parameters to process

# Create lists to store extracted data from all simulations
all_alignment_fractions = []
all_weighted_alignment_fractions = []
all_weighted_alignment_fractions2 = []
all_dominance_fractions = []
all_weighted_dominance_fractions = []
all_individual_dominance_fractions = []
all_weighted_individual_dominance_fractions = []
all_fitness = []
optimal_fitness_fractions = []

# Extract data from each simulation file
for sim_id in range(num_simulations):
    data = np.load(filename_base.format(sim_id), allow_pickle=True)
    
    all_alignment_fractions.append(data['alignment_fractions_over_time'])
    all_weighted_alignment_fractions.append(data['weighted_alignment_fractions_over_time'])
    all_weighted_alignment_fractions2.append(data['weighted_alignment_fractions_over_time2'])
    all_dominance_fractions.append(data['dominance_fractions_over_time'])
    all_weighted_dominance_fractions.append(data['weighted_dominance_fractions_over_time'])
    
    fitness_over_time = data['average_fitness_over_time']
    all_fitness.append(fitness_over_time)
    
    optimal_fitness = data['optimal_fitness']
    timesteps = data['timesteps']
    all_individual_dominance_fractions.append(data['individual_dominance_fractions'])
    all_weighted_individual_dominance_fractions.append(data['weighted_individual_dominance_fractions'])

    # Calculate fraction of optimal fitness for this run
    start_idx = int(len(fitness_over_time) * 0.75)
    avg_last_quarter_fitness = np.mean(fitness_over_time[start_idx:])
    sim_fraction_of_optimal = avg_last_quarter_fitness / optimal_fitness
    optimal_fitness_fractions.append(sim_fraction_of_optimal)


# Convert to numpy arrays
all_alignment_fractions = np.array(all_alignment_fractions)
all_weighted_alignment_fractions = np.array(all_weighted_alignment_fractions)
all_weighted_alignment_fractions2 = np.array(all_weighted_alignment_fractions2)
all_dominance_fractions = np.array(all_dominance_fractions)
all_weighted_dominance_fractions = np.array(all_weighted_dominance_fractions)
all_individual_dominance_fractions = np.array(all_individual_dominance_fractions)
all_weighted_individual_dominance_fractions = np.array(all_weighted_individual_dominance_fractions)
all_fitness = np.array(all_fitness)
optimal_fitness_fractions = np.array(optimal_fitness_fractions) 

# Calculate averages
avg_alignment = np.mean(all_alignment_fractions, axis=0)
avg_weighted_alignment = np.mean(all_weighted_alignment_fractions, axis=0)
avg_weighted_alignment2 = np.mean(all_weighted_alignment_fractions2, axis=0)
avg_dominance = np.mean(all_dominance_fractions, axis=0)
avg_weighted_dominance = np.mean(all_weighted_dominance_fractions, axis=0)
avg_individual_dominance = np.mean(all_individual_dominance_fractions, axis=0)
avg_weighted_individual_dominance = np.mean(all_weighted_individual_dominance_fractions, axis=0)
avg_fitness = np.mean(all_fitness, axis=0)
avg_optimal_fitness_fraction = np.mean(optimal_fitness_fractions)


# Standard deviation of last 25%
start_idx = int(all_alignment_fractions.shape[1] * 0.75)

std_alignment = np.std(np.mean(all_alignment_fractions[:, start_idx:], axis=1))
std_weighted_alignment = np.std(np.mean(all_weighted_alignment_fractions[:, start_idx:], axis=1))
std_weighted_alignment2 = np.std(np.mean(all_weighted_alignment_fractions2[:, start_idx:], axis=1))
std_dominance = np.std(np.mean(all_dominance_fractions[:, start_idx:], axis=1))
std_weighted_dominance = np.std(np.mean(all_weighted_dominance_fractions[:, start_idx:], axis=1))
std_individual_dominance = np.std(np.mean(all_individual_dominance_fractions[:, start_idx:], axis=1))
std_weighted_individual_dominance = np.std(np.mean(all_weighted_individual_dominance_fractions[:, start_idx:], axis=1))
std_fitness = np.std(np.mean(all_fitness[:, start_idx:], axis=1))
std_optimal_fitness_fraction = np.std(optimal_fitness_fractions) 


### Plotting
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
plt.subplots_adjust(hspace=0.3, wspace=0.25, top=0.95, bottom=0.07, left=0.05, right=0.97)

## Alignment Plots
# Normal Alignment
for i in range(num_simulations):
    axes[0, 0].plot(timesteps, all_alignment_fractions[i], alpha=0.3, linewidth=1) # Reduced alpha for 40 lines
    axes[0, 0].set_title("Alignment Fraction", fontweight='bold')
    axes[0, 0].set_ylabel("Consensus / Conflict")
    axes[0, 0].set_xlabel("Timestep")
    axes[0, 0].axhline(0.5, color='black', linestyle='--', label='Parity Line')
    axes[0, 0].set_ylim(0, 1)

# Weighted Alignment
for i in range(num_simulations):
    axes[0, 1].plot(timesteps, all_weighted_alignment_fractions[i], alpha=0.3, linewidth=1)
    axes[0, 1].set_title("Weighted Alignment Fraction", fontweight='bold')
    axes[0, 1].set_xlabel("Timestep")
    axes[0, 1].axhline(0.5, color='black', linestyle='--', label='Parity Line')
    axes[0, 1].set_ylim(0, 1)

# Average Alignment
axes[0, 2].plot(timesteps, avg_alignment, color='black', linewidth=1, label='Normal Mean')
axes[0, 2].plot(timesteps, avg_weighted_alignment, color='red', linewidth=1, label='Weighted Mean')
axes[0, 2].set_title("Alignment Average Fraction", fontweight='bold')
axes[0, 2].set_xlabel("Timestep")
axes[0, 2].legend(fontsize=9)
axes[0, 2].axhline(0.5, color='black', linestyle='--', label='Parity Line')
axes[0, 2].set_ylim(0, 1)

## Dominance Plots
# Normal Dominance Fraction
for i in range(num_simulations):
    axes[1, 0].plot(timesteps, all_dominance_fractions[i], alpha=0.3, linewidth=1)
    axes[1, 0].set_title("Dominance Fraction", fontweight='bold')
    axes[1, 0].set_ylabel("Group / (Group + Individual)")
    axes[1, 0].set_xlabel("Timestep")
    axes[1, 0].axhline(0.5, color='black', linestyle='--', label='Parity Line')
    axes[1, 0].set_ylim(0, 1)

# Weighted Dominance Fraction
for i in range(num_simulations):
    axes[1, 1].plot(timesteps, all_weighted_dominance_fractions[i], alpha=0.3, linewidth=1)
    axes[1, 1].set_title("Weighted Dominance Fraction", fontweight='bold')
    axes[1, 1].set_xlabel("Timestep")
    axes[1, 1].axhline(0.5, color='black', linestyle='--', label='Parity Line')
    axes[1, 1].set_ylim(0, 1)

# Average Dominance Fraction (arithmetic average of fractions)
axes[1, 2].plot(timesteps, avg_dominance, color='black', linewidth=1, label='Normal Mean')
axes[1, 2].plot(timesteps, avg_weighted_dominance, color='red', linewidth=1, label='Weighted Mean')
axes[1, 2].plot(timesteps, avg_individual_dominance, color='blue', linewidth=1, label='Ind Dom Mean')
axes[1, 2].plot(timesteps, avg_weighted_individual_dominance, color='green', linewidth=1, label='Weighted Ind Dom Mean')
axes[1, 2].set_title("Dominance Fraction Average", fontweight='bold')
axes[1, 2].set_xlabel("Timestep")
axes[1, 2].legend(fontsize=9)
axes[1, 2].axhline(0.5, color='black', linestyle='--', label='Parity Line')
axes[1, 2].set_ylim(0, 1)

# Global styling
for ax in axes.flatten():
    ax.grid(True, alpha=0.3)

## Fitness plots
fig_fit, axes_fit = plt.subplots(1, 3, figsize=(18, 10))
plt.subplots_adjust(hspace=0.3, wspace=0.25, top=0.95, bottom=0.07, left=0.05, right=0.97)

# Fitness of all runs
for i in range(num_simulations):
    axes_fit[0].plot(timesteps, all_fitness[i], alpha=0.3, linewidth=1, label=f"Run {i}" if num_simulations <= 10 else None)

axes_fit[0].set_ylim(0.4, 1)
axes_fit[0].set_title("Fitness", fontweight='bold')
axes_fit[0].set_ylabel("Average Fitness")
axes_fit[0].set_xlabel("Timestep")
if num_simulations <= 10:
    axes_fit[0].legend(fontsize=8, loc='upper left', bbox_to_anchor=(1, 1))

# Average Fitness
axes_fit[1].plot(timesteps, avg_fitness, color='black', linewidth=1, label='Average Fitness')
axes_fit[1].set_ylim(0.4, 1)
axes_fit[1].set_title("Average Fitness", fontweight='bold')
axes_fit[1].set_ylabel("Average Fitness")
axes_fit[1].set_xlabel("Timestep")
axes_fit[1].legend(fontsize=9)

# Optimal Fitness Fraction
axes_fit[2].bar(range(num_simulations), optimal_fitness_fractions, color='skyblue', edgecolor='black')
axes_fit[2].set_ylim(0, 1)
axes_fit[2].set_title("Optimal Fitness Fraction", fontweight='bold')
axes_fit[2].set_ylabel("Optimal Fitness Fraction")
axes_fit[2].set_xlabel("Simulation")
axes_fit[2].axhline(avg_optimal_fitness_fraction, color='red', linestyle='--', label='Average Fraction')
axes_fit[2].legend(fontsize=9)

# Save data (Added optimal_fitness_fractions)
np.savez(filename_base.format("summary v1.7"),
        avg_alignment=avg_alignment,
        avg_weighted_alignment=avg_weighted_alignment,
        avg_weighted_alignment2=avg_weighted_alignment2,
        avg_dominance=avg_dominance,
        avg_weighted_dominance=avg_weighted_dominance,
        avg_individual_dominance=avg_individual_dominance,
        avg_weighted_individual_dominance=avg_weighted_individual_dominance,    
        avg_optimal_fitness_fraction=avg_optimal_fitness_fraction,
        avg_fitness=avg_fitness,
        std_alignment=std_alignment,
        std_weighted_alignment=std_weighted_alignment,
        std_weighted_alignment2=std_weighted_alignment2,
        std_dominance=std_dominance,
        std_weighted_dominance=std_weighted_dominance,
        std_individual_dominance=std_individual_dominance,
        std_weighted_individual_dominance=std_weighted_individual_dominance,
        std_fitness=std_fitness,
        std_optimal_fitness_fraction=std_optimal_fitness_fraction,
        timesteps=timesteps)

print("Summary data saved to:", filename_base.format("summary v1.7"))

## Show plots
# plt.show()