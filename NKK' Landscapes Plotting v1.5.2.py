import numpy as np
import matplotlib.pyplot as plt

### Setup
num_simulations = 40
filename_base = "16-40-500 K=2 K2=1 S=1 M=0.001 T=200.0M Sim_{}.npz"

all_alignments = []
all_weighted_alignments = []
all_dominance_fractions = []
all_weighted_dominance_fractions = []
all_cheater_fractions = []
all_fitness = []

for sim_id in range(num_simulations):
    data = np.load(filename_base.format(sim_id), allow_pickle=True)
    
    all_alignments.append(data['alignments_over_time'])
    all_weighted_alignments.append(data['weighted_alignments_over_time'])
    all_dominance_fractions.append(data['dominance_fractions_over_time'])
    all_weighted_dominance_fractions.append(data['weighted_dominance_fractions_over_time'])
    all_cheater_fractions.append(data['cheater_fraction'])
    all_fitness.append(data['average_fitness_over_time'])
    optimal_fitness = data['optimal_fitness']
    timesteps = data['timesteps']

# Convert to numpy arrays
all_alignments = np.array(all_alignments)
all_weighted_alignments = np.array(all_weighted_alignments)
all_dominance_fractions = np.array(all_dominance_fractions)
all_weighted_dominance_fractions = np.array(all_weighted_dominance_fractions)
all_cheater_fractions = np.array(all_cheater_fractions)
all_fitness = np.array(all_fitness)

# Calculate averages
avg_dominance = np.mean(all_dominance_fractions, axis=0)
avg_weighted_dominance = np.mean(all_weighted_dominance_fractions, axis=0)
avg_cheater_fraction = np.mean(all_cheater_fractions, axis=0)
avg_fitness = np.mean(all_fitness, axis=0)

# Calculate logs for alignment metrics to match the log-average plotting logic
all_alignments_log = np.log10(all_alignments)
all_weighted_alignments_log = np.log10(all_weighted_alignments)
avg_alignment_log = np.mean(all_alignments_log, axis=0)
avg_weighted_alignment_log = np.mean(all_weighted_alignments_log, axis=0)

# Standard deviation
std_alignment_log = np.std(all_alignments_log, axis=0)
std_weighted_alignment_log = np.std(all_weighted_alignments_log, axis=0)
std_dominance = np.std(all_dominance_fractions, axis=0)
std_weighted_dominance = np.std(all_weighted_dominance_fractions, axis=0)
std_cheater_fraction = np.std(all_cheater_fractions, axis=0)
std_fitness = np.std(all_fitness, axis=0)


### Plotting
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
plt.subplots_adjust(hspace=0.3, wspace=0.25, top=0.95, bottom=0.07, left=0.05, right=0.97)

## Alignment Plots
# Normal Alignment
for i in range(num_simulations):
    axes[0, 0].plot(timesteps, all_alignments[i], alpha=1, linewidth=1)
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

# Average Alignment (average of logs)
axes[0, 2].plot(timesteps, avg_alignment_log, color='black', linewidth=1, label='Normal Mean')
axes[0, 2].plot(timesteps, avg_weighted_alignment_log, color='red', linewidth=1, label='Weighted Mean')
axes[0, 2].set_title("Alignment Average of Logs", fontweight='bold')
axes[0, 2].set_xlabel("Timestep")
axes[0, 2].legend(fontsize=9)
axes[0, 2].axhline(0, color='black', linestyle='--', label='Parity Line')

## Dominance Plots
# Normal Dominance Fraction
for i in range(num_simulations):
    axes[1, 0].plot(timesteps, all_dominance_fractions[i], alpha=1, linewidth=1)
    axes[1, 0].set_title("Dominance Fraction", fontweight='bold')
    axes[1, 0].set_ylabel("Group / (Group + Individual)")
    axes[1, 0].set_xlabel("Timestep")
    axes[1, 0].axhline(0.5, color='black', linestyle='--', label='Parity Line')
    axes[1, 0].set_ylim(0, 1)

# Weighted Dominance Fraction
for i in range(num_simulations):
    axes[1, 1].plot(timesteps, all_weighted_dominance_fractions[i], alpha=1, linewidth=1)
    axes[1, 1].set_title("Weighted Dominance Fraction", fontweight='bold')
    axes[1, 1].set_xlabel("Timestep")
    axes[1, 1].axhline(0.5, color='black', linestyle='--', label='Parity Line')
    axes[1, 1].set_ylim(0, 1)

# Average Dominance Fraction (arithmetic average of fractions)
axes[1, 2].plot(timesteps, avg_dominance, color='black', linewidth=1, label='Normal Mean')
axes[1, 2].plot(timesteps, avg_weighted_dominance, color='red', linewidth=1, label='Weighted Mean')
axes[1, 2].plot(timesteps, avg_cheater_fraction, color='blue', linewidth=1, label='Cheater Mean')
axes[1, 2].set_title("Dominance Fraction Average", fontweight='bold')
axes[1, 2].set_xlabel("Timestep")
axes[1, 2].legend(fontsize=9)
axes[1, 2].axhline(0.5, color='black', linestyle='--', label='Parity Line')
axes[1, 2].set_ylim(0, 1)

# Global styling
for ax in axes.flatten():
    ax.grid(True, alpha=0.3)

## Fitness plots
fig_fit, axes_fit = plt.subplots(1, 2, figsize=(18, 10))
plt.subplots_adjust(hspace=0.3, wspace=0.25, top=0.95, bottom=0.07, left=0.05, right=0.97)

# Fitness of all runs
for i in range(num_simulations):
    axes_fit[0].plot(timesteps, all_fitness[i], alpha=1, linewidth=1, label=f"Run {i}")

axes_fit[0].set_ylim(0.4, 1)
axes_fit[0].set_title("Fitness", fontweight='bold')
axes_fit[0].set_ylabel("Average Fitness")
axes_fit[0].set_xlabel("Timestep")
axes_fit[0].legend(fontsize=8, loc='upper left', bbox_to_anchor=(1, 1))

# Average Fitness
axes_fit[1].plot(timesteps, avg_fitness, color='black', linewidth=1, label='Average Fitness')
axes_fit[1].set_ylim(0.4, 1)
axes_fit[1].set_title("Average Fitness", fontweight='bold')
axes_fit[1].set_ylabel("Average Fitness")
axes_fit[1].set_xlabel("Timestep")
axes_fit[1].legend(fontsize=9)

# Save data
np.savez(filename_base.format("summary v1.5.2"),
        avg_alignment=avg_alignment_log,
        avg_weighted_alignment=avg_weighted_alignment_log,
        avg_dominance=avg_dominance,
        avg_weighted_dominance=avg_weighted_dominance,
        avg_cheater_fraction=avg_cheater_fraction,
        avg_fitness=avg_fitness,
        std_alignment=std_alignment_log,
        std_weighted_alignment=std_weighted_alignment_log,
        std_dominance=std_dominance,
        std_weighted_dominance=std_weighted_dominance,
        std_cheater_fraction=std_cheater_fraction,
        std_fitness=std_fitness,
        timesteps=timesteps)

print("Summary data saved to:", filename_base.format("summary v1.5.2"))

# plt.show()