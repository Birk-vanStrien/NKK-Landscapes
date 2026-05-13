import numpy as np
import matplotlib.pyplot as plt

# --- 1. Setup and Data Loading ---
num_simulations = 10
filename_base = "16-40-500 K=8 K2=1 S=1 M=0.001 T=300.0M Shared Landscape Sim_{}.npz"

all_alignments = []
all_weighted_alignments = []
all_dominance_ratios = []
all_weighted_dominance_ratios = []
all_fitness = []

timesteps = None



for sim_id in range(num_simulations):
    data = np.load(filename_base.format(sim_id), allow_pickle=True)
    
    all_alignments.append(data['alignments_over_time'])
    all_weighted_alignments.append(data['weighted_alignments_over_time'])
    all_dominance_ratios.append(data['dominance_ratios_over_time'])
    all_weighted_dominance_ratios.append(data['weighted_dominance_ratios_over_time'])
    all_fitness.append(data['average_fitness_over_time'])
    optimal_fitness = data['optimal_fitness']
    
    if timesteps is None:
        timesteps = data['timesteps']

# Convert to numpy arrays
all_alignments = np.array(all_alignments)
all_weighted_alignments = np.array(all_weighted_alignments)
all_dominance_ratios = np.array(all_dominance_ratios)
all_weighted_dominance_ratios = np.array(all_weighted_dominance_ratios)

# Calculate averages
avg_alignment = np.mean(all_alignments, axis=0)
avg_weighted_alignment = np.mean(all_weighted_alignments, axis=0)
avg_dominance = np.mean(all_dominance_ratios, axis=0)
avg_weighted_dominance = np.mean(all_weighted_dominance_ratios, axis=0)

# --- 2. Plotting ---
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
plt.subplots_adjust(hspace=0.3, wspace=0.25, top=0.95, bottom=0.07, left=0.05, right=0.97)

## Alignment Plots
# Normal Alignment
for i in range(num_simulations):
    axes[0, 0].plot(timesteps, all_alignments[i], alpha=1, linewidth=1)
    # Get log scale for y-axis
    axes[0, 0].set_yscale('log')
    axes[0, 0].set_title("Alignment", fontweight='bold')
    axes[0, 0].set_ylabel("Consensus / Conflict")
    axes[0, 0].set_xlabel("Timestep")
    axes[0, 0].axhline(1, color='black', linestyle='--', label='Parity Line')

# Weighted Alignment
for i in range(num_simulations):
    axes[0, 1].plot(timesteps, all_weighted_alignments[i], alpha=1, linewidth=1)
    axes[0, 1].set_yscale('log')
    axes[0, 1].set_title("Weighted Alignment", fontweight='bold')
    axes[0, 1].set_xlabel("Timestep")
    axes[0, 1].axhline(1, color='black', linestyle='--', label='Parity Line')

# Average Alignment
axes[0, 2].plot(timesteps, avg_alignment, color='black', linewidth=1, label='Normal Mean')
axes[0, 2].plot(timesteps, avg_weighted_alignment, color='red', linewidth=1, label='Weighted Mean')
axes[0, 2].set_yscale('log')
axes[0, 2].set_title("Alignment Average", fontweight='bold')
axes[0, 2].set_xlabel("Timestep")
axes[0, 2].legend(fontsize=9)
axes[0, 2].axhline(1, color='black', linestyle='--', label='Parity Line')

## Dominance Plots
# Normal Dominance Ratio
for i in range(num_simulations):
    axes[1, 0].plot(timesteps, all_dominance_ratios[i], alpha=1, linewidth=1)
    axes[1, 0].set_yscale('log')
    axes[1, 0].set_title("Dominance Ratio", fontweight='bold')
    axes[1, 0].set_ylabel("Group / Individual Ratio")
    axes[1, 0].set_xlabel("Timestep")
    axes[1, 0].axhline(1, color='black', linestyle='--', label='Parity Line')

# Weighted Dominance Ratio
for i in range(num_simulations):
    axes[1, 1].plot(timesteps, all_weighted_dominance_ratios[i], alpha=1, linewidth=1)
    axes[1, 1].set_yscale('log')
    axes[1, 1].set_title("Weighted Dominance", fontweight='bold')
    axes[1, 1].set_xlabel("Timestep")
    axes[1, 1].axhline(1, color='black', linestyle='--', label='Parity Line')

# Average Dominance Ratio
axes[1, 2].plot(timesteps, avg_dominance, color='black', linewidth=1, label='Normal Mean')
axes[1, 2].plot(timesteps, avg_weighted_dominance, color='red', linewidth=1, label='Weighted Mean')
axes[1, 2].set_yscale('log')
axes[1, 2].set_title("Dominance Ratio Average", fontweight='bold')
axes[1, 2].set_xlabel("Timestep")
axes[1, 2].legend(fontsize=9)
axes[1, 2].axhline(1, color='black', linestyle='--', label='Parity Line')

# Global styling
for ax in axes.flatten():
    ax.grid(True, alpha=0.3)

# plt.show()


## Fitness plots

fig, axes = plt.subplots(1, 2, figsize=(18, 10))
plt.subplots_adjust(hspace=0.3, wspace=0.25, top=0.95, bottom=0.07, left=0.05, right=0.97)

# Fitness of all runs
for i in range(num_simulations):
    axes[0].plot(timesteps, all_fitness[i], alpha=1, linewidth=1)
    axes[0].set_ylim(0.4, 1)
    axes[0].set_title("Fitness", fontweight='bold')
    axes[0].set_ylabel("Average Fitness")
    axes[0].set_xlabel("Timestep")
    axes[0].axhline(optimal_fitness, color='black', linestyle='--', label='Optimal Fitness')


# Average Fitness
avg_fitness = np.mean(all_fitness, axis=0)
axes[1].plot(timesteps, avg_fitness, color='black', linewidth=1, label='Average Fitness')
axes[1].set_ylim(0.4, 1)
axes[1].set_title("Average Fitness", fontweight='bold')
axes[1].set_ylabel("Average Fitness")
axes[1].set_xlabel("Timestep")
axes[1].legend(fontsize=9)
axes[1].axhline(optimal_fitness, color='black', linestyle='--', label='Optimal Fitness')

plt.show()