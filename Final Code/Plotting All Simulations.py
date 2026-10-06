## NOTES
# Requires the summary files from Single simulation plots

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

### Setup
num_simulations = 40
K_range = [0, 1, 2, 4, 8]
K2_range = [0, 1, 2, 4, 8]
filename_base = "16-40-500 K={} K2={} S=0.0001 M=0.001 T=200.0M Sim_summary.npz"

# Plotting options
std_fill = False
ste_fill = False
matrix_use_std = False
matrix_use_ste = True
use_alternative_weighted_alignment = False
remove_K2_zero = True

# Store averages of runs across all KK' values
all_avg_alignment_fractions = []
all_avg_weighted_alignment_fractions = []
all_avg_weighted_alignment_fractions2 = []
all_avg_dominance_fractions = []
all_avg_weighted_dominance_fractions = []
all_avg_individual_dominance_fractions = []
all_avg_weighted_individual_dominance_fractions = []
all_avg_fitness = []
all_avg_optimal_fitness_fractions = []

# Store standard deviations
all_std_alignment_fractions = []
all_std_weighted_alignment_fractions = []
all_std_weighted_alignment_fractions2 = []
all_std_dominance_fractions = []
all_std_weighted_dominance_fractions = []
all_std_individual_dominance_fractions = []
all_std_weighted_individual_dominance_fractions = []
all_std_fitness = []
all_std_optimal_fitness_fractions = []

loaded_params = [] 

# Load data. Skip if not found
for K in K_range:
    for K2 in K2_range:
        try:
            data = np.load(filename_base.format(K, K2), allow_pickle = True)
        except FileNotFoundError:
            print(f"File not found: {filename_base.format(K, K2)}")
            continue

        # Load and store data
        all_avg_alignment_fractions.append(data['avg_alignment'])
        all_avg_weighted_alignment_fractions.append(data['avg_weighted_alignment'])
        all_avg_weighted_alignment_fractions2.append(data['avg_weighted_alignment2'])
        all_avg_dominance_fractions.append(data['avg_dominance'])
        all_avg_weighted_dominance_fractions.append(data['avg_weighted_dominance'])
        all_avg_individual_dominance_fractions.append(data['avg_individual_dominance'])
        all_avg_weighted_individual_dominance_fractions.append(data['avg_weighted_individual_dominance'])
        all_avg_fitness.append(data['avg_fitness'])
        all_avg_optimal_fitness_fractions.append(data['avg_optimal_fitness_fraction'])
        
        all_std_alignment_fractions.append(float(data['std_alignment']))
        all_std_weighted_alignment_fractions.append(float(data['std_weighted_alignment']))
        all_std_weighted_alignment_fractions2.append(float(data['std_weighted_alignment2']))
        all_std_dominance_fractions.append(float(data['std_dominance']))
        all_std_weighted_dominance_fractions.append(float(data['std_weighted_dominance']))
        all_std_individual_dominance_fractions.append(float(data['std_individual_dominance']))
        all_std_weighted_individual_dominance_fractions.append(float(data['std_weighted_individual_dominance']))
        all_std_fitness.append(float(data['std_fitness']))
        all_std_optimal_fitness_fractions.append(float(data['std_optimal_fitness_fraction']))
        
        loaded_params.append((K, K2)) # Keep track which KK' values have data
        timesteps = data['timesteps']

# Convert to numpy arrays
all_avg_alignment_fractions = np.array(all_avg_alignment_fractions)
all_avg_weighted_alignment_fractions = np.array(all_avg_weighted_alignment_fractions)
all_avg_weighted_alignment_fractions2 = np.array(all_avg_weighted_alignment_fractions2)
all_avg_dominance_fractions = np.array(all_avg_dominance_fractions)
all_avg_weighted_dominance_fractions = np.array(all_avg_weighted_dominance_fractions)
all_avg_individual_dominance_fractions = np.array(all_avg_individual_dominance_fractions)
all_avg_weighted_individual_dominance_fractions = np.array(all_avg_weighted_individual_dominance_fractions)
all_avg_fitness = np.array(all_avg_fitness)
all_avg_optimal_fitness_fractions = np.array(all_avg_optimal_fitness_fractions)

all_std_alignment_fractions = np.array(all_std_alignment_fractions)
all_std_weighted_alignment_fractions = np.array(all_std_weighted_alignment_fractions)
all_std_weighted_alignment_fractions2 = np.array(all_std_weighted_alignment_fractions2)
all_std_dominance_fractions = np.array(all_std_dominance_fractions)
all_std_weighted_dominance_fractions = np.array(all_std_weighted_dominance_fractions)
all_std_individual_dominance_fractions = np.array(all_std_individual_dominance_fractions)
all_std_weighted_individual_dominance_fractions = np.array(all_std_weighted_individual_dominance_fractions)
all_std_fitness = np.array(all_std_fitness)
all_std_optimal_fitness_fractions = np.array(all_std_optimal_fitness_fractions)


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
    Line2D([0], [0], color = 'tab:red', lw = 2, label = 'K > K2'),
    Line2D([0], [0], color = '#000000', lw = 2, label = 'K = K2'),
    Line2D([0], [0], color = 'tab:blue', lw = 2, label = 'K < K2'),
]

# Create space for text
x_max = max(timesteps)
x_offset = (x_max - min(timesteps)) * 0.15
xlim_right = x_max + x_offset



### Plotting

## Alignment Plots
# Plot of normal alignment fraction for all K and K2
plt.figure(figsize = (12, 8))
for i in range(len(all_avg_alignment_fractions)):
    K, K2 = loaded_params[i]
    color = get_color(K, K2)
    y_data = all_avg_alignment_fractions[i]
    std = all_std_alignment_fractions[i]
 
    plt.plot(timesteps, y_data, color = color, alpha = 1, linewidth = 1)
    plt.text(timesteps[-1], y_data[-1], f" K={K}, K2={K2}", color = color, fontsize = 9, va='center')
    if std_fill == True:
        plt.fill_between(timesteps, y_data - std, y_data + std, color = color, alpha =  0.15)
    if ste_fill == True:
        ste = all_std_alignment_fractions[i] / np.sqrt(num_simulations)
        plt.fill_between(timesteps, y_data - ste, y_data + ste, color = color, alpha = 0.15)

plt.legend(handles = custom_legend_lines, bbox_to_anchor = (1, 1), loc = 'upper left', fontsize = 9) 
plt.xlim(right = xlim_right)
plt.ylim(-0.05, 1.05)
plt.xlabel("Timestep")
plt.ylabel("Average Alignment Fraction")
plt.title("Alignment Plots")
plt.axhline(0.5, color = 'black', linestyle='--')

# Plot of weighted alignment fraction for all K and K2
plt.figure(figsize=(12, 8))
if use_alternative_weighted_alignment:
    for i in range(len(all_avg_weighted_alignment_fractions2)):
        K, K2 = loaded_params[i]
        color = get_color(K, K2)
        y_data = all_avg_weighted_alignment_fractions2[i]
        std = all_std_weighted_alignment_fractions2[i]
        plt.plot(timesteps, y_data, color = color, alpha = 1, linewidth = 1)
        plt.text(timesteps[-1], y_data[-1], f" K={K}, K2={K2}", color = color, fontsize = 9, va = 'center')
        if std_fill == True:
            plt.fill_between(timesteps, y_data - std, y_data + std, color = color, alpha = 0.15)
        if ste_fill == True:
            ste = all_std_weighted_alignment_fractions[i] / np.sqrt(num_simulations) 
            plt.fill_between(timesteps, y_data - ste, y_data + ste, color = color, alpha = 0.15)
else:        
    for i in range(len(all_avg_weighted_alignment_fractions)):
        K, K2 = loaded_params[i]
        color = get_color(K, K2)
        y_data = all_avg_weighted_alignment_fractions[i]
        std = all_std_weighted_alignment_fractions[i] 
        plt.plot(timesteps, y_data, color = color, alpha = 1, linewidth = 1)
        plt.text(timesteps[-1], y_data[-1], f" K={K}, K2={K2}", color = color, fontsize = 9, va = 'center')
        if std_fill == True:
            plt.fill_between(timesteps, y_data - std, y_data + std, color = color, alpha = 0.15)
        if ste_fill == True:
            ste = all_std_weighted_alignment_fractions[i] / np.sqrt(num_simulations) 
            plt.fill_between(timesteps, y_data - ste, y_data + ste, color = color, alpha = 0.15)

plt.legend(handles = custom_legend_lines, bbox_to_anchor = (1, 1), loc = 'upper left', fontsize = 9)
plt.xlim(right = xlim_right)
plt.xlabel("Timestep")
plt.ylim(-0.05, 1.05)
plt.ylabel("Average Weighted Alignment Fraction")
plt.title("Weighted Alignment Plots")
plt.axhline(0.5, color = 'black', linestyle='--')


## Dominance Fraction Plots
# Plot of dominance fraction for all K and K2
plt.figure(figsize = (12, 8))
for i in range(len(all_avg_dominance_fractions)):
    K, K2 = loaded_params[i]
    color = get_color(K, K2)
    y_data = all_avg_dominance_fractions[i]  
    std = all_std_dominance_fractions[i]
    
    plt.plot(timesteps, y_data, color = color, alpha = 1, linewidth = 1)
    plt.text(timesteps[-1], y_data[-1], f" K={K}, K2={K2}", color = color, fontsize = 9, va='center')
    if std_fill == True:
        plt.fill_between(timesteps, y_data - std, y_data + std, color = color, alpha = 0.15)
    if ste_fill == True:
        ste = all_std_dominance_fractions[i] / np.sqrt(num_simulations) 
        plt.fill_between(timesteps, y_data - ste, y_data + ste, color = color, alpha = 0.15)

plt.legend(handles=custom_legend_lines, bbox_to_anchor=(1, 1), loc = 'upper left', fontsize = 9)
plt.xlim(right = xlim_right)
plt.ylim(-0.05, 1.05)
plt.xlabel("Timestep")
plt.ylabel("Average Group Group Dominance")
plt.title("Group Dominance Fraction Plots")
plt.axhline(0.5, color = 'black', linestyle='--', alpha = 0.7)

# Plot of weighted dominance fraction for all K and K2
plt.figure(figsize=(12, 8))
for i in range(len(all_avg_weighted_dominance_fractions)):
    K, K2 = loaded_params[i]
    color = get_color(K, K2)
    y_data = all_avg_weighted_dominance_fractions[i]  
    std = all_std_weighted_dominance_fractions[i] 
    
    plt.plot(timesteps, y_data, color = color, alpha = 1, linewidth = 1)
    plt.text(timesteps[-1], y_data[-1], f" K={K}, K2={K2}", color = color, fontsize = 9, va = 'center')
    if std_fill == True:
        plt.fill_between(timesteps, y_data - std, y_data + std, color = color, alpha = 0.15)
    if ste_fill == True:
        ste = all_std_weighted_dominance_fractions[i] / np.sqrt(num_simulations) 
        plt.fill_between(timesteps, y_data - ste, y_data + ste, color = color, alpha = 0.15)

plt.legend(handles = custom_legend_lines, bbox_to_anchor = (1, 1), loc = 'upper left', fontsize = 9)
plt.xlim(right = xlim_right)
plt.ylim(-0.05, 1.05)
plt.xlabel("Timestep")
plt.ylabel("Average Weighted Group Group Dominance")
plt.title("Weighted Group Dominance Fraction Plots")
plt.axhline(0.5, color = 'black', linestyle = '--', alpha = 0.7)


## Fitness Plot for all KK' values
plt.figure(figsize=(12, 8))
for i in range(len(all_avg_fitness)):
    K, K2 = loaded_params[i]
    color = get_color(K, K2)
    y_data = all_avg_fitness[i]
    std = all_std_fitness[i] 
    
    plt.plot(timesteps, y_data, color = color, alpha = 1, linewidth = 1)
    plt.text(timesteps[-1], y_data[-1], f" K={K}, K2={K2}", color = color, fontsize = 9, va = 'center')
    if std_fill == True:
        plt.fill_between(timesteps, y_data - std, y_data + std, color = color, alpha = 0.15)
    if ste_fill == True:
        ste = all_std_fitness[i] / np.sqrt(num_simulations)
        plt.fill_between(timesteps, y_data - ste, y_data + ste, color = color, alpha = 0.15)

plt.legend(handles = custom_legend_lines, bbox_to_anchor = (1, 1), loc='upper left', fontsize = 9)
plt.xlim(right = xlim_right)
plt.xlabel("Timestep")
plt.ylabel("Average Fitness")
plt.title("Fitness Plots")


## Matrix plots for alignment, dominance, and cheater locus frequency
K2_range_matrix = [val for val in K2_range if not (remove_K2_zero and val == 0)]
matrix_shape = (len(K2_range_matrix), len(K_range))

# Set up matrices
average_matrices = {
    "Alignment": np.full(matrix_shape, np.nan),
    "Group Dominance": np.full(matrix_shape, np.nan),
    "Cheater Loci Fraction": np.full(matrix_shape, np.nan),
    "Weighted Alignment": np.full(matrix_shape, np.nan),
    "Weighted Group Dominance": np.full(matrix_shape, np.nan),
    "Weighted Cheater Loci Fraction": np.full(matrix_shape, np.nan)
}
standard_deviation_matrices = {
    "Alignment": np.full(matrix_shape, np.nan),
    "Group Dominance": np.full(matrix_shape, np.nan),
    "Cheater Loci Fraction": np.full(matrix_shape, np.nan),
    "Weighted Alignment": np.full(matrix_shape, np.nan),
    "Weighted Group Dominance": np.full(matrix_shape, np.nan),
    "Weighted Cheater Loci Fraction": np.full(matrix_shape, np.nan)
}

## Matrix plots for fitness and fraction of optimal fitness
fitness_matrix_shape = (len(K2_range), len(K_range))
fitness_matrix = np.full(fitness_matrix_shape, np.nan)
std_fitness_matrix = np.full(fitness_matrix_shape, np.nan)

optimal_fitness_matrix = np.full(fitness_matrix_shape, np.nan)
std_optimal_fitness_matrix = np.full(fitness_matrix_shape, np.nan)

K_map = {val: i for i, val in enumerate(K_range)}
K2_map_matrix = {val: i for i, val in enumerate(K2_range_matrix)}
K2_map_fitness = {val: i for i, val in enumerate(K2_range)}

# Calculate the index slice for the last quarter of timesteps
start_idx = int(len(timesteps) * 0.75)

for i, (K, K2) in enumerate(loaded_params):
    # original maps for fitness block
    row_fit, col_fit = K2_map_fitness[K2], K_map[K]
    
    # Averages slices
    align_slice = all_avg_alignment_fractions[i][start_idx:]
    w_align_slice = all_avg_weighted_alignment_fractions[i][start_idx:]
    w_align_slice2 = all_avg_weighted_alignment_fractions2[i][start_idx:]
    fitness_slice = all_avg_fitness[i][start_idx:]
    dom_slice = all_avg_dominance_fractions[i][start_idx:]
    w_dom_slice = all_avg_weighted_dominance_fractions[i][start_idx:]
    i_dom_slice = all_avg_individual_dominance_fractions[i][start_idx:]
    w_i_dom_slice = all_avg_weighted_individual_dominance_fractions[i][start_idx:]

    # Fill matrices
    if K2 in K2_map_matrix:
        row, col = K2_map_matrix[K2], K_map[K]
        
        # Alignment
        average_matrices["Alignment"][row, col] = np.mean(align_slice)
        if matrix_use_std:
            standard_deviation_matrices["Alignment"][row, col] = all_std_alignment_fractions[i]
        if matrix_use_ste:
            standard_deviation_matrices["Alignment"][row, col] = all_std_alignment_fractions[i] / np.sqrt(num_simulations)
        
        # Weighted Alignment
        if use_alternative_weighted_alignment:
            average_matrices["Weighted Alignment"][row, col] = np.mean(w_align_slice2)
            if matrix_use_std:
                standard_deviation_matrices["Weighted Alignment"][row, col] = all_std_weighted_alignment_fractions2[i]
            if matrix_use_ste:
                standard_deviation_matrices["Weighted Alignment"][row, col] = all_std_weighted_alignment_fractions2[i] / np.sqrt(num_simulations)
        else:    
            average_matrices["Weighted Alignment"][row, col] = np.mean(w_align_slice)
            if matrix_use_std:
                standard_deviation_matrices["Weighted Alignment"][row, col] = all_std_weighted_alignment_fractions[i]
            if matrix_use_ste:
                standard_deviation_matrices["Weighted Alignment"][row, col] = all_std_weighted_alignment_fractions[i] / np.sqrt(num_simulations)

        # Dominance Fraction
        valid_dom = dom_slice[np.isfinite(dom_slice)]
        average_matrices["Group Dominance"][row, col] = np.mean(valid_dom) if len(valid_dom) > 0 else np.nan
        if matrix_use_std:
            standard_deviation_matrices["Group Dominance"][row, col] = all_std_dominance_fractions[i]
        if matrix_use_ste:
            standard_deviation_matrices["Group Dominance"][row, col] = all_std_dominance_fractions[i] / np.sqrt(num_simulations)
        
        # Weighted Dominance Fraction
        valid_w_dom = w_dom_slice[np.isfinite(w_dom_slice)]
        average_matrices["Weighted Group Dominance"][row, col] = np.mean(valid_w_dom) if len(valid_w_dom) > 0 else np.nan
        if matrix_use_std:
            standard_deviation_matrices["Weighted Group Dominance"][row, col] = all_std_weighted_dominance_fractions[i]
        if matrix_use_ste:
            standard_deviation_matrices["Weighted Group Dominance"][row, col] = all_std_weighted_dominance_fractions[i] / np.sqrt(num_simulations)

        # Cheater Loci Fraction
        valid_i_dom = i_dom_slice[np.isfinite(i_dom_slice)]
        average_matrices["Cheater Loci Fraction"][row, col] = np.mean(valid_i_dom) if len(valid_i_dom) > 0 else np.nan
        if matrix_use_std:
            standard_deviation_matrices["Cheater Loci Fraction"][row, col] = all_std_individual_dominance_fractions[i]
        if matrix_use_ste:
            standard_deviation_matrices["Cheater Loci Fraction"][row, col] = all_std_individual_dominance_fractions[i] / np.sqrt(num_simulations)

        # Weighted Cheater Loci Fraction
        valid_w_i_dom = w_i_dom_slice[np.isfinite(w_i_dom_slice)]
        average_matrices["Weighted Cheater Loci Fraction"][row, col] = np.mean(valid_w_i_dom) if len(valid_w_i_dom) > 0 else np.nan
        if matrix_use_std:
            standard_deviation_matrices["Weighted Cheater Loci Fraction"][row, col] = all_std_weighted_individual_dominance_fractions[i]
        if matrix_use_ste:
            standard_deviation_matrices["Weighted Cheater Loci Fraction"][row, col] = all_std_weighted_individual_dominance_fractions[i] / np.sqrt(num_simulations)

    # Fitness
    fitness_matrix[row_fit, col_fit] = np.mean(fitness_slice)
    if matrix_use_std:
        std_fitness_matrix[row_fit, col_fit] = all_std_fitness[i]
    if matrix_use_ste:
        std_fitness_matrix[row_fit, col_fit] = all_std_fitness[i] / np.sqrt(num_simulations)

    # Optimal Fitness Fraction
    optimal_fitness_matrix[row_fit, col_fit] = all_avg_optimal_fitness_fractions[i]
    if matrix_use_std:
        std_optimal_fitness_matrix[row_fit, col_fit] = all_std_optimal_fitness_fractions[i]
    if matrix_use_ste:
        std_optimal_fitness_matrix[row_fit, col_fit] = all_std_optimal_fitness_fractions[i] / np.sqrt(num_simulations)

## Plotting loop
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

current_cmap = plt.colormaps['viridis'].copy() 
current_cmap.set_bad(color = 'lightgrey') # N/A color

for i, (label, matrix) in enumerate(average_matrices.items()):
    ax = axes[i]
    
    im = ax.imshow(matrix, cmap = current_cmap, origin = 'lower')
    
    ax.set_title(f"{label}", fontweight = 'bold')
    ax.set_xticks(np.arange(len(K_range)))
    ax.set_yticks(np.arange(len(K2_range_matrix)))
    ax.set_xticklabels(K_range)
    ax.set_yticklabels(K2_range_matrix)
    ax.set_xlabel("K")
    ax.set_ylabel("K2")
    
    plt.colorbar(im, ax = ax, fraction = 0.05, pad = 0.05)
    
    # Add values to matrix plot
    for r in range(len(K2_range_matrix)):
        for c in range(len(K_range)):
            val = matrix[r, c]
            if not np.isnan(val): 
                std_val = standard_deviation_matrices[label][r, c] # Add standard deviation
                ax.text(c, r, f'{val:.3f}\n(±{std_val:.3f})', ha = 'center', va = 'center', 
                        color = 'white' if val < np.nanmean(matrix) else 'black') # Contrasting
            else:
                ax.text(c, r, 'N/A', ha = 'center', va = 'center', color = 'dimgrey', fontsize = 8)


plt.tight_layout()


## Plotting fitness matrices
fig_fit, axes_fit = plt.subplots(1, 2, figsize = (13, 5)) 

# Plot standard fitness matrix
ax_fit = axes_fit[0]
im_fit = ax_fit.imshow(fitness_matrix, cmap = current_cmap, origin = 'lower')
ax_fit.set_title("Avg of Fitness (Last 25%)", fontweight = 'bold')
ax_fit.set_xticks(np.arange(len(K_range)))
ax_fit.set_yticks(np.arange(len(K2_range)))
ax_fit.set_xticklabels(K_range)
ax_fit.set_yticklabels(K2_range)
ax_fit.set_xlabel("K")
ax_fit.set_ylabel("K2")
plt.colorbar(im_fit, ax = ax_fit, fraction = 0.05, pad = 0.05)

# Add values to the plot
for r in range(len(K2_range)):
    for c in range(len(K_range)):
        val = fitness_matrix[r, c]
        if not np.isnan(val):
            std_val = std_fitness_matrix[r, c]
            ax_fit.text(c, r, f'{val:.3f}\n(±{std_val:.3f})', ha = 'center', va = 'center', 
                        color = 'white' if val < np.nanmean(fitness_matrix) else 'black')
        else:
            ax_fit.text(c, r, 'N/A', ha = 'center', va = 'center', color = 'dimgrey', fontsize = 8)

# Plot optimal fitness fraction
ax_opt = axes_fit[1]
im_opt = ax_opt.imshow(optimal_fitness_matrix, cmap = current_cmap, origin = 'lower')
ax_opt.set_title("Fraction of Optimal Fitness", fontweight='bold')
ax_opt.set_xticks(np.arange(len(K_range)))
ax_opt.set_yticks(np.arange(len(K2_range)))
ax_opt.set_xticklabels(K_range)
ax_opt.set_yticklabels(K2_range)
ax_opt.set_xlabel("K")
ax_opt.set_ylabel("K2")
plt.colorbar(im_opt, ax=ax_opt, fraction = 0.05, pad = 0.05)

# Add values to plot
for r in range(len(K2_range)):
    for c in range(len(K_range)):
        val = optimal_fitness_matrix[r, c]
        if not np.isnan(val):
            std_val = std_optimal_fitness_matrix[r, c]
            ax_opt.text(c, r, f'{val:.3f}\n(±{std_val:.3f})', ha = 'center', va = 'center', 
                        color = 'white' if val < np.nanmean(optimal_fitness_matrix) else 'black')
        else:
            ax_opt.text(c, r, 'N/A', ha = 'center', va = 'center', color = 'dimgrey', fontsize = 8)

plt.tight_layout()
plt.show()
