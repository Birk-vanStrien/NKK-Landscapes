import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

### Setup
num_simulations = 40
K_range = [1, 2, 4, 8]
K2_range = [1, 2, 4, 8]
filename_base = "16-40-500 K={} K2={} S=1 M=0.001 T=200.0M Sim_summary v1.5.2.npz"

# Plotting options
std_fill = False
ste_fill = True

# Store averages of runs across all combinations
all_avg_alignments = []
all_avg_weighted_alignments = []
all_avg_dominance_fractions = []
all_avg_weighted_dominance_fractions = []
all_avg_cheater_fractions = []
all_avg_fitness = []

# Store standard deviations
all_std_alignments = []
all_std_weighted_alignments = []
all_std_dominance_fractions = []
all_std_weighted_dominance_fractions = []
all_std_cheater_fractions = []
all_std_fitness = []

loaded_params = [] 

# Load data. Skip if not found
for K in K_range:
    for K2 in K2_range:
        try:
            data = np.load(filename_base.format(K, K2), allow_pickle=True)
        except FileNotFoundError:
            print(f"File not found: {filename_base.format(K, K2)}")
            continue

        # Load and store data
        all_avg_alignments.append(data['avg_alignment'])
        all_avg_weighted_alignments.append(data['avg_weighted_alignment'])
        all_avg_dominance_fractions.append(data['avg_dominance'])
        all_avg_weighted_dominance_fractions.append(data['avg_weighted_dominance'])
        all_avg_cheater_fractions.append(data['avg_cheater_fraction'])
        all_avg_fitness.append(data['avg_fitness'])
        
        all_std_alignments.append(data['std_alignment'])
        all_std_weighted_alignments.append(data['std_weighted_alignment'])
        all_std_dominance_fractions.append(data['std_dominance'])
        all_std_weighted_dominance_fractions.append(data['std_weighted_dominance'])
        all_std_cheater_fractions.append(data['std_cheater_fraction'])
        all_std_fitness.append(data['std_fitness'])
        
        loaded_params.append((K, K2))
        timesteps = data['timesteps']

# Convert to numpy arrays
all_avg_alignments = np.array(all_avg_alignments)
all_avg_weighted_alignments = np.array(all_avg_weighted_alignments)
all_avg_dominance_fractions = np.array(all_avg_dominance_fractions)
all_avg_weighted_dominance_fractions = np.array(all_avg_weighted_dominance_fractions)
all_avg_cheater_fractions = np.array(all_avg_cheater_fractions)
all_avg_fitness = np.array(all_avg_fitness)

all_std_alignments = np.array(all_std_alignments)
all_std_weighted_alignments = np.array(all_std_weighted_alignments)
all_std_dominance_fractions = np.array(all_std_dominance_fractions)
all_std_weighted_dominance_fractions = np.array(all_std_weighted_dominance_fractions)
all_std_cheater_fractions = np.array(all_std_cheater_fractions)
all_std_fitness = np.array(all_std_fitness)


### Styling
def get_color(K, K2):
    if K > K2:
        return 'tab:red'
    elif K == K2:
        return '#000000'
    else:
        return 'tab:blue'

# Legend
custom_legend_lines = [
    Line2D([0], [0], color='tab:red', lw=2, label='K > K2'),
    Line2D([0], [0], color='#000000', lw=2, label='K = K2'),
    Line2D([0], [0], color='tab:blue', lw=2, label='K < K2'),
]

# Create space for text
if len(timesteps) > 0:
    x_max = max(timesteps)
    x_offset = (x_max - min(timesteps)) * 0.15 # add 15% padding to the right
    xlim_right = x_max + x_offset
else:
    xlim_right = None


### Plotting

## Alignment Plots
# Plot of normal alignment for all K and K2
plt.figure(figsize=(12, 8))
for i in range(len(all_avg_alignments)):
    K, K2 = loaded_params[i]
    color = get_color(K, K2)
    y_data = all_avg_alignments[i]
    std = all_std_alignments[i]
 
    plt.plot(timesteps, y_data, color=color, alpha=1, linewidth=1)
    plt.text(timesteps[-1], y_data[-1], f" K={K}, K2={K2}", color=color, fontsize=9, va='center')
    if std_fill == True:
        plt.fill_between(timesteps, y_data - std, y_data + std, color=color, alpha=0.15)
    if ste_fill == True:
        ste = all_std_alignments[i] / np.sqrt(num_simulations)
        plt.fill_between(timesteps, y_data - ste, y_data + ste, color=color, alpha=0.15)

plt.legend(handles=custom_legend_lines, bbox_to_anchor=(1, 1), loc='upper left', fontsize=9) 
plt.xlim(right=xlim_right)
plt.xlabel("Timestep")
plt.ylabel("Average Alignment")
plt.title("Alignment Plots")
plt.axhline(0, color='black', linestyle='--')

# Plot of weighted alignment for all K and K2
plt.figure(figsize=(12, 8))
for i in range(len(all_avg_weighted_alignments)):
    K, K2 = loaded_params[i]
    color = get_color(K, K2)
    y_data = all_avg_weighted_alignments[i]
    std = all_std_weighted_alignments[i] 
    
    plt.plot(timesteps, y_data, color=color, alpha=1, linewidth=1)
    plt.text(timesteps[-1], y_data[-1], f" K={K}, K2={K2}", color=color, fontsize=9, va='center')
    if std_fill == True:
        plt.fill_between(timesteps, y_data - std, y_data + std, color=color, alpha=0.15)
    if ste_fill == True:
        ste = all_std_weighted_alignments[i] / np.sqrt(num_simulations) 
        plt.fill_between(timesteps, y_data - ste, y_data + ste, color=color, alpha=0.15)

plt.legend(handles=custom_legend_lines, bbox_to_anchor=(1, 1), loc='upper left', fontsize=9)
plt.xlim(right=xlim_right)
plt.xlabel("Timestep")
plt.ylabel("Average Weighted Alignment")
plt.title("Weighted Alignment Plots")
plt.axhline(0, color='black', linestyle='--')


## Dominance Fraction Plots
# Plot of dominance fraction for all K and K2
plt.figure(figsize=(12, 8))
for i in range(len(all_avg_dominance_fractions)):
    K, K2 = loaded_params[i]
    color = get_color(K, K2)
    y_data = all_avg_dominance_fractions[i]  
    std = all_std_dominance_fractions[i]
    
    plt.plot(timesteps, y_data, color=color, alpha=1, linewidth=1)
    plt.text(timesteps[-1], y_data[-1], f" K={K}, K2={K2}", color=color, fontsize=9, va='center')
    if std_fill == True:
        plt.fill_between(timesteps, y_data - std, y_data + std, color=color, alpha=0.15)
    if ste_fill == True:
        ste = all_std_dominance_fractions[i] / np.sqrt(num_simulations) 
        plt.fill_between(timesteps, y_data - ste, y_data + ste, color=color, alpha=0.15)

plt.legend(handles=custom_legend_lines, bbox_to_anchor=(1, 1), loc='upper left', fontsize=9)
plt.xlim(right=xlim_right)
plt.ylim(-0.05, 1.05)
plt.xlabel("Timestep")
plt.ylabel("Average Group Dominance Fraction")
plt.title("Group Dominance Fraction Plots")
plt.axhline(0.5, color='black', linestyle='--', alpha=0.7)

# Plot of weighted dominance fraction for all K and K2
plt.figure(figsize=(12, 8))
for i in range(len(all_avg_weighted_dominance_fractions)):
    K, K2 = loaded_params[i]
    color = get_color(K, K2)
    y_data = all_avg_weighted_dominance_fractions[i]  
    std = all_std_weighted_dominance_fractions[i] # Using saved standard deviation
    
    plt.plot(timesteps, y_data, color=color, alpha=1, linewidth=1)
    plt.text(timesteps[-1], y_data[-1], f" K={K}, K2={K2}", color=color, fontsize=9, va='center')
    if std_fill == True:
        plt.fill_between(timesteps, y_data - std, y_data + std, color=color, alpha=0.15)
    if ste_fill == True:
        ste = all_std_weighted_dominance_fractions[i] / np.sqrt(num_simulations) 
        plt.fill_between(timesteps, y_data - ste, y_data + ste, color=color, alpha=0.15)

plt.legend(handles=custom_legend_lines, bbox_to_anchor=(1, 1), loc='upper left', fontsize=9)
plt.xlim(right=xlim_right)
plt.ylim(-0.05, 1.05)
plt.xlabel("Timestep")
plt.ylabel("Average Weighted Group Dominance Fraction")
plt.title("Weighted Group Dominance Fraction Plots")
plt.axhline(0.5, color='black', linestyle='--', alpha=0.7)


## Fitness Plots
# Plot of fitness for all K and K2
plt.figure(figsize=(12, 8))
for i in range(len(all_avg_fitness)):
    K, K2 = loaded_params[i]
    color = get_color(K, K2)
    y_data = all_avg_fitness[i]
    std = all_std_fitness[i] # Using saved standard deviation
    
    plt.plot(timesteps, y_data, color=color, alpha=1, linewidth=1)
    plt.text(timesteps[-1], y_data[-1], f" K={K}, K2={K2}", color=color, fontsize=9, va='center')
    if std_fill == True:
        plt.fill_between(timesteps, y_data - std, y_data + std, color=color, alpha=0.15)
    if ste_fill == True:
        ste = all_std_fitness[i] / np.sqrt(num_simulations)
        plt.fill_between(timesteps, y_data - ste, y_data + ste, color=color, alpha=0.15)

plt.legend(handles=custom_legend_lines, bbox_to_anchor=(1, 1), loc='upper left', fontsize=9)
plt.xlim(right=xlim_right)
plt.xlabel("Timestep")
plt.ylabel("Average Fitness")
plt.title("Fitness Plots")


## Matrix plots
matrix_shape = (len(K2_range), len(K_range))
average_matrices = {
    "Alignment": np.full(matrix_shape, np.nan),
    "Weighted Alignment": np.full(matrix_shape, np.nan),
    "Fitness": np.full(matrix_shape, np.nan),
    "Dominance Fraction": np.full(matrix_shape, np.nan),
    "Weighted Dominance Frac": np.full(matrix_shape, np.nan),
    "Cheater Fraction": np.full(matrix_shape, np.nan)
}
standard_deviation_matrices = {
    "Alignment": np.full(matrix_shape, np.nan),
    "Weighted Alignment": np.full(matrix_shape, np.nan),
    "Fitness": np.full(matrix_shape, np.nan),
    "Dominance Fraction": np.full(matrix_shape, np.nan),
    "Weighted Dominance Frac": np.full(matrix_shape, np.nan),
    "Cheater Fraction": np.full(matrix_shape, np.nan)
}

K_map = {val: i for i, val in enumerate(K_range)}
K2_map = {val: i for i, val in enumerate(K2_range)}

# Calculate the index slice for the last quarter of timesteps
start_idx = int(len(timesteps) * 0.75)

for i, (K, K2) in enumerate(loaded_params):
    row, col = K2_map[K2], K_map[K] 
    
    # Averages slices
    align_slice = all_avg_alignments[i][start_idx:]
    w_align_slice = all_avg_weighted_alignments[i][start_idx:]
    fitness_slice = all_avg_fitness[i][start_idx:]
    dom_slice = all_avg_dominance_fractions[i][start_idx:]
    w_dom_slice = all_avg_weighted_dominance_fractions[i][start_idx:]
    cheater_slice = all_avg_cheater_fractions[i][start_idx:]
    # Saved standard deviation slices
    std_align_slice = all_std_alignments[i][start_idx:]
    std_w_align_slice = all_std_weighted_alignments[i][start_idx:]
    std_fitness_slice = all_std_fitness[i][start_idx:]
    std_dom_slice = all_std_dominance_fractions[i][start_idx:]
    std_w_dom_slice = all_std_weighted_dominance_fractions[i][start_idx:]
    std_cheater_slice = all_std_cheater_fractions[i][start_idx:]

    # Alignment
    average_matrices["Alignment"][row, col] = np.mean(align_slice)
    standard_deviation_matrices["Alignment"][row, col] = np.mean(std_align_slice)
    
    # Weighted Alignment
    average_matrices["Weighted Alignment"][row, col] = np.mean(w_align_slice)
    standard_deviation_matrices["Weighted Alignment"][row, col] = np.mean(std_w_align_slice)
    
    # Fitness
    average_matrices["Fitness"][row, col] = np.mean(fitness_slice)
    standard_deviation_matrices["Fitness"][row, col] = np.mean(std_fitness_slice)
    
    # Dominance Fraction
    valid_dom = dom_slice[np.isfinite(dom_slice)]
    valid_std_dom = std_dom_slice[np.isfinite(std_dom_slice)]
    average_matrices["Dominance Fraction"][row, col] = np.mean(valid_dom) if len(valid_dom) > 0 else np.nan
    standard_deviation_matrices["Dominance Fraction"][row, col] = np.mean(valid_std_dom) if len(valid_std_dom) > 0 else np.nan
    
    # Weighted Dominance Fraction
    valid_w_dom = w_dom_slice[np.isfinite(w_dom_slice)]
    valid_std_w_dom = std_w_dom_slice[np.isfinite(std_w_dom_slice)]
    average_matrices["Weighted Dominance Frac"][row, col] = np.mean(valid_w_dom) if len(valid_w_dom) > 0 else np.nan
    standard_deviation_matrices["Weighted Dominance Frac"][row, col] = np.mean(valid_std_w_dom) if len(valid_std_w_dom) > 0 else np.nan

    # Cheater Fraction
    valid_cheater = cheater_slice[np.isfinite(cheater_slice)]
    valid_std_cheater = std_cheater_slice[np.isfinite(std_cheater_slice)]
    average_matrices["Cheater Fraction"][row, col] = np.mean(valid_cheater) if len(valid_cheater) > 0 else np.nan
    standard_deviation_matrices["Cheater Fraction"][row, col] = np.mean(valid_std_cheater) if len(valid_std_cheater) > 0 else np.nan

# Plotting Loop
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

current_cmap = plt.colormaps['viridis'].copy()
current_cmap.set_bad(color='lightgrey')

for i, (label, matrix) in enumerate(average_matrices.items()):
    ax = axes[i]
    
    im = ax.imshow(matrix, cmap=current_cmap, origin='lower')
    
    ax.set_title(f"Avg of {label} (Last 25%)", fontweight='bold')
    ax.set_xticks(np.arange(len(K_range)))
    ax.set_yticks(np.arange(len(K2_range)))
    ax.set_xticklabels(K_range)
    ax.set_yticklabels(K2_range)
    ax.set_xlabel("K")
    ax.set_ylabel("K2")
    
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    for r in range(len(K2_range)):
        for c in range(len(K_range)):
            val = matrix[r, c]
            if not np.isnan(val):
                std_val = standard_deviation_matrices[label][r, c]
                ax.text(c, r, f'{val:.3f}\n(±{std_val:.3f})', ha='center', va='center', 
                        color='white' if val < np.nanmean(matrix) else 'black')
            else:
                ax.text(c, r, 'N/A', ha='center', va='center', color='dimgrey', fontsize=8)


plt.tight_layout()
plt.show()