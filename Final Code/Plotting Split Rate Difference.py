import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t as student_t

### Setup & Configuration
num_simulations = 40
K_range = [0, 1, 2, 4, 8]
K2_range = [0, 1, 2, 4, 8]
filename_template = "16-40-500 K={{}} K2={{}} S={} M=0.001 T=200.0M Sim_summary.npz"

# Input the S values to compare
s_val_a = input("Enter first S value (e.g. 0.01): ").strip()
s_val_b = input("Enter second S value to compare against (e.g. 1): ").strip()

use_alternative_weighted_alignment = True # Correct for cases where group benefit
remove_K2_zero = True

K2_range_matrix = [val for val in K2_range if not (remove_K2_zero and val == 0)]
matrix_shape = (len(K2_range_matrix), len(K_range))
fitness_matrix_shape = (len(K2_range), len(K_range))

K_map = {val: i for i, val in enumerate(K_range)}
K2_map_matrix = {val: i for i, val in enumerate(K2_range_matrix)}
K2_map_fitness = {val: i for i, val in enumerate(K2_range)}

# Load summary averages and standard deviations for a given S value string
def load_summary_and_std_matrices(s_val_str):
    filename_base = filename_template.format(s_val_str)
    
    avg_matrices = {
        "Alignment": np.full(matrix_shape, np.nan),
        "Group Dominance": np.full(matrix_shape, np.nan),
        "Cheater Loci Fraction": np.full(matrix_shape, np.nan),
        "Weighted Alignment": np.full(matrix_shape, np.nan),
        "Weighted Group Dominance": np.full(matrix_shape, np.nan),
        "Weighted Cheater Loci Fraction": np.full(matrix_shape, np.nan)
    }
    std_matrices = {key: np.full(matrix_shape, np.nan) for key in avg_matrices}

    fitness_mat = np.full(fitness_matrix_shape, np.nan)
    std_fitness_mat = np.full(fitness_matrix_shape, np.nan)

    optimal_fitness_mat = np.full(fitness_matrix_shape, np.nan)
    std_optimal_fitness_mat = np.full(fitness_matrix_shape, np.nan)

    for K in K_range:
        for K2 in K2_range:
            try:
                data = np.load(filename_base.format(K, K2), allow_pickle=True)
            except FileNotFoundError:
                continue

            timesteps = data['timesteps']
            start_idx = int(len(timesteps) * 0.75)

            # Metric slices
            align_slice = data['avg_alignment'][start_idx:]
            w_align_slice = data['avg_weighted_alignment'][start_idx:]
            w_align_slice2 = data['avg_weighted_alignment2'][start_idx:]
            fitness_slice = data['avg_fitness'][start_idx:]
            dom_slice = data['avg_dominance'][start_idx:]
            w_dom_slice = data['avg_weighted_dominance'][start_idx:]
            i_dom_slice = data['avg_individual_dominance'][start_idx:]
            w_i_dom_slice = data['avg_weighted_individual_dominance'][start_idx:]

            row_fit, col_fit = K2_map_fitness[K2], K_map[K]
            fitness_mat[row_fit, col_fit] = np.mean(fitness_slice)
            std_fitness_mat[row_fit, col_fit] = float(data['std_fitness'])

            optimal_fitness_mat[row_fit, col_fit] = data['avg_optimal_fitness_fraction']
            std_optimal_fitness_mat[row_fit, col_fit] = float(data['std_optimal_fitness_fraction'])

            if K2 in K2_map_matrix:
                row, col = K2_map_matrix[K2], K_map[K]
                
                # Alignment
                avg_matrices["Alignment"][row, col] = np.mean(align_slice)
                std_matrices["Alignment"][row, col] = float(data['std_alignment'])

                # Weighted Alignment
                if use_alternative_weighted_alignment:
                    avg_matrices["Weighted Alignment"][row, col] = np.mean(w_align_slice2)
                    std_matrices["Weighted Alignment"][row, col] = float(data['std_weighted_alignment2'])
                else:
                    avg_matrices["Weighted Alignment"][row, col] = np.mean(w_align_slice)
                    std_matrices["Weighted Alignment"][row, col] = float(data['std_weighted_alignment'])

                # Group Dominance
                valid_dom = dom_slice[np.isfinite(dom_slice)]
                avg_matrices["Group Dominance"][row, col] = np.mean(valid_dom) if len(valid_dom) > 0 else np.nan
                std_matrices["Group Dominance"][row, col] = float(data['std_dominance'])

                # Weighted Group Dominance
                valid_w_dom = w_dom_slice[np.isfinite(w_dom_slice)]
                avg_matrices["Weighted Group Dominance"][row, col] = np.mean(valid_w_dom) if len(valid_w_dom) > 0 else np.nan
                std_matrices["Weighted Group Dominance"][row, col] = float(data['std_weighted_dominance'])

                # Cheater Loci
                valid_i_dom = i_dom_slice[np.isfinite(i_dom_slice)]
                avg_matrices["Cheater Loci Fraction"][row, col] = np.mean(valid_i_dom) if len(valid_i_dom) > 0 else np.nan
                std_matrices["Cheater Loci Fraction"][row, col] = float(data['std_individual_dominance'])

                # Weighted Cheater Loci
                valid_w_i_dom = w_i_dom_slice[np.isfinite(w_i_dom_slice)]
                avg_matrices["Weighted Cheater Loci Fraction"][row, col] = np.mean(valid_w_i_dom) if len(valid_w_i_dom) > 0 else np.nan
                std_matrices["Weighted Cheater Loci Fraction"][row, col] = float(data['std_weighted_individual_dominance'])

    return (avg_matrices, std_matrices, 
            fitness_mat, std_fitness_mat, 
            optimal_fitness_mat, std_optimal_fitness_mat)


# Perform Welch's t-test to determine significance
def check_significance(mean1, std1, mean2, std2, n=40, alpha=0.05):
    alpha_corrected = alpha / (len(K_range) * len(K2_range))

    if np.isnan(mean1) or np.isnan(mean2) or np.isnan(std1) or np.isnan(std2):
        return False

    se1 = std1 / np.sqrt(n)
    se2 = std2 / np.sqrt(n)
    se_diff = np.sqrt(se1**2 + se2**2)
    
    if se_diff == 0:
        return False

    t_stat = abs(mean1 - mean2) / se_diff
    
    df_denom = (se1**4 / (n - 1)) + (se2**4 / (n - 1))
    df = (se_diff**4) / df_denom if df_denom > 0 else 2 * (n - 1)
    
    p_val = 2 * (1 - student_t.cdf(t_stat, df=df))
    return p_val < alpha_corrected


# Load datasets for the user-defined S values
(mA, stdA, fitA, std_fitA, opt_fitA, std_opt_fitA) = load_summary_and_std_matrices(s_val_a)
(mB, stdB, fitB, std_fitB, opt_fitB, std_opt_fitB) = load_summary_and_std_matrices(s_val_b)

# Difference Matrices (S_A minus S_B)
diff_matrices = {key: mA[key] - mB[key] for key in mA}
diff_fitness = fitA - fitB
diff_optimal_fitness = opt_fitA - opt_fitB

# Swapped Colormap
diverging_cmap = plt.colormaps['bwr_r'].copy()
diverging_cmap.set_bad(color='lightgrey')

# Plotting Metric Difference Matrices
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for i, (label, matrix) in enumerate(diff_matrices.items()):
    ax = axes[i]
    
    max_val = np.nanmax(np.abs(matrix)) if not np.all(np.isnan(matrix)) else 1.0
    im = ax.imshow(matrix, cmap=diverging_cmap, origin='lower', vmin=-max_val, vmax=max_val)
    
    ax.set_title(f"Δ {label}\n(S={s_val_a} - S={s_val_b})", fontweight='bold')
    ax.set_xticks(np.arange(len(K_range)))
    ax.set_yticks(np.arange(len(K2_range_matrix)))
    ax.set_xticklabels(K_range)
    ax.set_yticklabels(K2_range_matrix)
    ax.set_xlabel("K")
    ax.set_ylabel("K2")
    
    plt.colorbar(im, ax=ax, fraction=0.05, pad=0.05)
    
    # Annotate values with significance markers
    for r in range(len(K2_range_matrix)):
        for c in range(len(K_range)):
            val = matrix[r, c]
            if not np.isnan(val): 
                is_sig = check_significance(
                    mA[label][r, c], stdA[label][r, c],
                    mB[label][r, c], stdB[label][r, c],
                    n=num_simulations
                )
                
                star = "*" if is_sig else ""
                weight = 'bold' if is_sig else 'normal'
                text_color = 'white' if abs(val) > max_val * 0.6 else 'black'
                
                ax.text(c, r, f'{val:+.3f}{star}', ha='center', va='center', 
                        color=text_color, fontweight=weight)
            else:
                ax.text(c, r, 'N/A', ha='center', va='center', color='dimgrey', fontsize=8)

plt.tight_layout()

## Plotting Fitness Difference Matrices
fig_fit, axes_fit = plt.subplots(1, 2, figsize=(13, 5)) 

fit_diff_list = [
    (f"Δ Fitness (Last 25%)\n(S={s_val_a} - S={s_val_b})", diff_fitness, std_fitA, std_fitB, fitA, fitB, K2_range),
    (f"Δ Optimal Fitness Fraction\n(S={s_val_a} - S={s_val_b})", diff_optimal_fitness, std_opt_fitA, std_opt_fitB, opt_fitA, opt_fitB, K2_range)
]

for idx, (title, mat, s_matA, s_matB, raw_matA, raw_matB, y_ticks) in enumerate(fit_diff_list):
    ax = axes_fit[idx]
    max_val = np.nanmax(np.abs(mat)) if not np.all(np.isnan(mat)) else 1.0
    im = ax.imshow(mat, cmap=diverging_cmap, origin='lower', vmin=-max_val, vmax=max_val)
    
    ax.set_title(title, fontweight='bold')
    ax.set_xticks(np.arange(len(K_range)))
    ax.set_yticks(np.arange(len(y_ticks)))
    ax.set_xticklabels(K_range)
    ax.set_yticklabels(y_ticks)
    ax.set_xlabel("K")
    ax.set_ylabel("K2")
    plt.colorbar(im, ax=ax, fraction=0.05, pad=0.05)

    for r in range(len(y_ticks)):
        for c in range(len(K_range)):
            val = mat[r, c]
            if not np.isnan(val):
                is_sig = check_significance(
                    raw_matA[r, c], s_matA[r, c],
                    raw_matB[r, c], s_matB[r, c],
                    n=num_simulations
                )
                
                star = "*" if is_sig else ""
                weight = 'bold' if is_sig else 'normal'
                text_color = 'white' if abs(val) > max_val * 0.6 else 'black'
                
                ax.text(c, r, f'{val:+.3f}{star}', ha='center', va='center', 
                        color=text_color, fontweight=weight)
            else:
                ax.text(c, r, 'N/A', ha='center', va='center', color='dimgrey', fontsize=8)

plt.tight_layout()
plt.show()