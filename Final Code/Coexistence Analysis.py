import glob
import os
import re
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import numba
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns
import statsmodels.api as sm

## Settings
max_groups_to_plot = 500 # Number of groups to analyze per sim file
target_K = 1       
target_K2 = 8
target_s = 1  
export_individual_plots = False # Save plots showing actual graphs for each group

## Core fucntions

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
def calculateLocusFitness(
    locus, genome, group_avg_genome, epistasis_matrix, coefficients, K, K2
):
    intragenomic_partners = epistasis_matrix[locus, :K]
    intergenomic_partners = epistasis_matrix[locus, K:]
    iesi = genome[locus]
    for i in range(K):
        partner = intragenomic_partners[i]
        iesi |= genome[partner] << (i + 1)
    group_avg_values = group_avg_genome[intergenomic_partners]
    locus_coeffs = coefficients[locus, iesi, :]
    return evaluate_multilinear(locus_coeffs, group_avg_values)

@numba.njit
def calculateFitness(
    genome,
    group_avg_genome,
    epistasis_matrix,
    coefficients,
    K,
    K2,
    genome_size,
):
    total_fitness = 0.0
    for locus in range(genome_size):
        fitness_contribution = calculateLocusFitness(
            locus,
            genome,
            group_avg_genome,
            epistasis_matrix,
            coefficients,
            K,
            K2,
        )
        total_fitness += fitness_contribution
    return total_fitness / genome_size

## Analysis functions

def plot_single_group_analysis(
    group_id,
    counts,
    group_population,
    epistasis_matrix,
    coefficients,
    K,
    K2,
    genome_size,
    sim_id,
    output_dir,
    save_plot=True,
):
    group_avg_genome = np.mean(group_population, axis=0)
    group_size = len(group_population)
    shannon_diversity = -sum(
        (count / group_size) * np.log2(count / group_size)
        for count in counts.values()
    )
    unique_genomes_count = len(counts)

    top_two = counts.most_common(2)
    mcg_bytes, mcg_count = top_two[0]
    smcg_bytes, smcg_count = top_two[1]

    p_current = smcg_count / group_size

    mcg_genome = np.frombuffer(mcg_bytes, dtype=np.int8)
    smcg_genome = np.frombuffer(smcg_bytes, dtype=np.int8)

    p_vals = np.linspace(0.0, 1.0, 5000)
    mcg_fitnesses, smcg_fitnesses, group_fitnesses = [], [], []

    for p in p_vals:
        group_avg = p * smcg_genome + (1.0 - p) * mcg_genome
        fit_mcg = calculateFitness(
            mcg_genome, group_avg, epistasis_matrix, coefficients, K, K2, genome_size
        )
        fit_smcg = calculateFitness(
            smcg_genome, group_avg, epistasis_matrix, coefficients, K, K2, genome_size
        )
        fit_group = (1.0 - p) * fit_mcg + p * fit_smcg

        mcg_fitnesses.append(fit_mcg)
        smcg_fitnesses.append(fit_smcg)
        group_fitnesses.append(fit_group)

    p_vals = np.array(p_vals)
    mcg_fitnesses = np.array(mcg_fitnesses)
    smcg_fitnesses = np.array(smcg_fitnesses)
    group_fitnesses = np.array(group_fitnesses)

    dG_dp = np.gradient(group_fitnesses, p_vals)
    G_increasing = dG_dp > 0
    G_decreasing = dG_dp < 0

    SMCG_favored = smcg_fitnesses > mcg_fitnesses
    MCG_favored = mcg_fitnesses > smcg_fitnesses

    yellow_region = MCG_favored & G_increasing
    blue_region = SMCG_favored & G_decreasing

    has_yellow_region = bool(np.any(yellow_region))
    has_blue_region = bool(np.any(blue_region))

    diffs = smcg_fitnesses - mcg_fitnesses

    # Determine if stable crossing point exists(difference changes positive -> negative)
    mutant_starts_higher = diffs[0] > 0
    stable_crossings = np.where((diffs[:-1] > 0) & (diffs[1:] <= 0))[0]
    interior_stable_crossings = [idx for idx in stable_crossings if 0 < idx < len(p_vals) - 1]

    has_stable_crossing = len(interior_stable_crossings) > 0
    has_internal_coexistence = mutant_starts_higher and has_stable_crossing

    # Determine if vertex is between 0 and 1 (unstable coexistence)
    vertex_idx = np.argmax(group_fitnesses)
    p_vertex = p_vals[vertex_idx]
    has_group_optimum_coexistence = 0.0 < p_vertex < 1.0

    # Classification
    if has_internal_coexistence and has_group_optimum_coexistence:
        classification = "both_coexistence"
    elif has_internal_coexistence:
        classification = "internal_coexistence_only"
    elif has_group_optimum_coexistence:
        classification = "group_optimum_only"
    else:
        classification = "no_coexistence"

    p_crossing = None
    if has_stable_crossing:
        idx = interior_stable_crossings[0]
        p1, p2 = p_vals[idx], p_vals[idx + 1]
        d1, d2 = diffs[idx], diffs[idx + 1]
        p_crossing = p1 - d1 * (p2 - p1) / (d2 - d1) if (d2 - d1) != 0 else p1

    idx_current = np.argmin(np.abs(p_vals - p_current))
    in_yellow = bool(yellow_region[idx_current])
    in_blue = bool(blue_region[idx_current])

    if save_plot:
        fig, (ax, ax_text) = plt.subplots(
            1, 2, figsize=(14, 6.5), gridspec_kw={"width_ratios": [3, 1.2]}
        )

        ax.plot(
            p_vals,
            mcg_fitnesses,
            color="blue",
            linewidth=2.5,
            label=r"Most Common Genome ($\beta_1$)",
        )
        ax.plot(
            p_vals,
            smcg_fitnesses,
            color="red",
            linewidth=2.5,
            label=r"Second Most Common Genome ($\beta_2$)",
        )
        ax.plot(
            p_vals,
            group_fitnesses,
            color="black",
            linestyle="--",
            linewidth=2.0,
            label=r"Group Fitness ($W_G$)",
        )

        if 0 <= p_vertex <= 1:
            ax.axvline(p_vertex, color="gray", linestyle=":", linewidth=1.5, label=f"Max Group Fitness ({p_vertex:.2f})")
        if p_crossing is not None and 0 <= p_crossing <= 1:
            ax.axvline(p_crossing, color="purple", linestyle="--", linewidth=1.5, label=f"Stable Crossing ({p_crossing:.2f})")

        ax.axvline(
            p_current,
            color="green",
            linestyle="-.",
            linewidth=2.0,
            label=f"Current $p$ ({p_current:.2f})"
        )

        ax.fill_between(p_vals, mcg_fitnesses, smcg_fitnesses, 
                        where=yellow_region, 
                        color='yellow', alpha=0.3, label='Spread possible with group selection',
                        interpolate=True)

        ax.fill_between(p_vals, mcg_fitnesses, smcg_fitnesses, 
                        where=MCG_favored & G_decreasing, 
                        color='red', alpha=0.3, label='No spread',
                        interpolate=True)

        ax.fill_between(p_vals, mcg_fitnesses, smcg_fitnesses, 
                        where=SMCG_favored & G_increasing, 
                        color='green', alpha=0.3, label='Spread supported',
                        interpolate=True)

        ax.fill_between(p_vals, mcg_fitnesses, smcg_fitnesses, 
                        where=blue_region, 
                        color='blue', alpha=0.3, label='Spread limited by group selection',
                        interpolate=True)

        ax.set_xlim(0, 1.0)
        y_min = min(mcg_fitnesses.min(), smcg_fitnesses.min(), group_fitnesses.min())
        y_max = max(mcg_fitnesses.max(), smcg_fitnesses.max(), group_fitnesses.max())
        pad = (y_max - y_min) * 0.15 if y_max != y_min else 0.1
        ax.set_ylim(max(0, y_min - pad), y_max + pad)

        ax.set_xlabel(r"Proportion of Second Most Common Genome ($p$)", fontsize=12)
        ax.set_ylabel("Fitness", fontsize=12)
        ax.set_title(f"Sim {sim_id} - Group {group_id} ({classification})", fontsize=12)

        ax.legend(
            bbox_to_anchor=(-0.18, 1.0),
            loc="upper right",
            frameon=True,
            fontsize=9,
        )

        ax_text.axis("off")
        y_pos = 0.95
        ax_text.text(0.0, y_pos, "Group Metrics", fontsize=11, fontweight="bold")
        y_pos -= 0.07
        ax_text.text(0.0, y_pos, f"Group Size: {group_size}", fontsize=10)
        y_pos -= 0.06
        ax_text.text(0.0, y_pos, f"Unique Genomes: {unique_genomes_count}", fontsize=10)
        y_pos -= 0.06
        ax_text.text(0.0, y_pos, f"Shannon Diversity: {shannon_diversity:.4f}", fontsize=10)
        y_pos -= 0.06
        ax_text.text(0.0, y_pos, f"Current Proportion (p): {p_current:.4f}", fontsize=10)

        y_pos -= 0.10
        ax_text.text(0.0, y_pos, "Most Common Genome:", fontsize=10, fontweight="bold")
        y_pos -= 0.06
        mcg_str = "".join(str(b) for b in mcg_genome)
        ax_text.text(0.0, y_pos, mcg_str, fontfamily="monospace", fontsize=9, color="black")

        y_pos -= 0.08
        ax_text.text(0.0, y_pos, "Second Most Common Genome:", fontsize=10, fontweight="bold")
        y_pos -= 0.06

        x_pos = 0.0
        for bit_m, bit_s in zip(mcg_genome, smcg_genome):
            char = str(bit_s)
            color = "red" if bit_m != bit_s else "black"
            t = ax_text.text(
                x_pos,
                y_pos,
                char,
                fontfamily="monospace",
                fontsize=9,
                color=color,
                fontweight="bold" if color == "red" else "normal",
            )
            fig.canvas.draw()
            bbox = t.get_window_extent(renderer=fig.canvas.get_renderer())
            bbox_data = ax_text.transData.inverted().transform(bbox)
            char_width = bbox_data[1][0] - bbox_data[0][0]
            x_pos += char_width + 0.005

        y_pos -= 0.10
        ax_text.text(0.0, y_pos, "Group Average Genome:", fontsize=10, fontweight="bold")
        y_pos -= 0.06

        avg_str_1 = "[" + ", ".join(f"{val:.2f}" for val in group_avg_genome[:8]) + ","
        avg_str_2 = " " + ", ".join(f"{val:.2f}" for val in group_avg_genome[8:]) + "]"
        ax_text.text(0.0, y_pos, avg_str_1, fontfamily="monospace", fontsize=8)
        y_pos -= 0.05
        ax_text.text(0.0, y_pos, avg_str_2, fontfamily="monospace", fontsize=8)

        plt.tight_layout()

        plot_filename = os.path.join(
            output_dir, f"Sim_{sim_id}_Group_{group_id}_K={K}_K2={K2}.png"
        )
        plt.savefig(plot_filename, dpi=300, bbox_inches="tight")
        plt.close()

    return (
        classification, 
        shannon_diversity, 
        unique_genomes_count, 
        p_current, 
        has_yellow_region,
        has_blue_region,
        in_yellow, 
        in_blue,
        has_stable_crossing,
        p_crossing,
        has_internal_coexistence,
        has_group_optimum_coexistence,
        p_vertex
    )

def plot_diversity_analysis(df, k_suffix, output_dir="output_plots"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
    
    palette = {
        "both_coexistence": "#2ecc71", 
        "internal_coexistence_only": "#3498db",
        "group_optimum_only": "#f39c12",
        "no_coexistence": "#e74c3c"
    }
    categories = ["both_coexistence", "internal_coexistence_only", "group_optimum_only", "no_coexistence"]
    plot_df = df[df["classification"].isin(categories)].copy()
    
    sns.violinplot(
        data=plot_df, 
        x="classification", 
        y="shannon_diversity", 
        order=categories,
        palette=palette,
        inner=None, 
        ax=ax1, 
        cut=0,
        alpha=0.4
    )
    sns.boxplot(
        data=plot_df, 
        x="classification", 
        y="shannon_diversity", 
        order=categories,
        width=0.15, 
        ax=ax1, 
        boxprops=dict(facecolor='white', edgecolor='black'),
        medianprops=dict(color='black', linewidth=2),
        whiskerprops=dict(color='black'),
        capprops=dict(color='black')
    )
    
    ax1.set_xticks(range(4))
    ax1.set_xticklabels(["Both", "Internal\nOnly", "Group Opt\nOnly", "Neither"], fontsize=9)
    ax1.set_xlabel("Coexistence Mechanisms", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Shannon Diversity", fontsize=12, fontweight="bold")
    ax1.set_title("Diversity Distribution across Mechanisms", fontsize=13)
    
    x_seq = np.linspace(df["shannon_diversity"].min(), df["shannon_diversity"].max(), 300)
    X_seq = sm.add_constant(x_seq)
    n_bins = 8

    if df["shannon_diversity"].nunique() > 1 and len(df) > 5:
        y = df["internal_coexistence"].values
        X = sm.add_constant(df["shannon_diversity"])

        try:
            logit_model = sm.Logit(y, X).fit(disp=0)
            pred = logit_model.predict(X_seq)
            
            cov = logit_model.cov_params()
            linear_pred = np.dot(X_seq, logit_model.params)
            se = np.sqrt(np.sum(np.dot(X_seq, cov) * X_seq, axis=1))
            ci_lower = 1 / (1 + np.exp(-(linear_pred - 1.96 * se)))
            ci_upper = 1 / (1 + np.exp(-(linear_pred + 1.96 * se)))

            ax2.plot(x_seq, pred, color="#3498db", linewidth=2.5, label="Internal Coexistence Fit")
            ax2.fill_between(x_seq, ci_lower, ci_upper, color="#3498db", alpha=0.15)
        except Exception:
            pass

        df_temp = df.copy()
        df_temp['target_binary'] = y
        df_temp['div_bin'] = pd.qcut(df_temp['shannon_diversity'], q=n_bins, duplicates='drop')
        
        binned_stats = df_temp.groupby('div_bin', observed=False).agg(
            mean_div=('shannon_diversity', 'mean'),
            prop_target=('target_binary', 'mean'),
            count=('target_binary', 'count')
        ).reset_index()

        binned_stats['se'] = np.sqrt(
            binned_stats['prop_target'] * (1 - binned_stats['prop_target']) / binned_stats['count']
        )

        ax2.errorbar(
            binned_stats['mean_div'], 
            binned_stats['prop_target'], 
            yerr=1.96 * binned_stats['se'], 
            fmt='o', 
            color='#2c3e50', 
            ecolor='#2c3e50', 
            elinewidth=1.5, 
            capsize=4, 
            markersize=6, 
            label='Observed Internal %'
        )

        sns.rugplot(
            data=df, 
            x="shannon_diversity", 
            ax=ax2, 
            color="#2c3e50", 
            alpha=0.3, 
            height=0.04
        )

        ax2.set_xlabel("Shannon Diversity", fontsize=12, fontweight="bold")
        ax2.set_ylabel("P(Internal Coexistence)", fontsize=12, fontweight="bold")
        ax2.set_ylim(-0.02, 1.02)
        ax2.set_title("Probability of Internal Dynamics Coexistence", fontsize=13)
        ax2.legend(loc="upper left", frameon=True, fontsize=9)

    plt.suptitle(f"Separated Coexistence Analysis ({k_suffix})", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    
    fig_path = os.path.join(output_dir, f"diversity_coexistence_crossing_plots_{k_suffix}.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Visualization saved successfully to '{fig_path}'.")

# Extract data from file

def process_npz_file(npz_file_path, group_records, sim_stats, output_dir="output_plots"):
    filename = os.path.basename(npz_file_path)
    if target_s is not None:
        file_s = None

        s_match = re.search(r"\b[Ss]\s*[=_:-]?\s*([0-9]*\.?[0-9]+)", filename)
        if s_match:
            file_s = float(s_match.group(1))

        if file_s is not None:
            target_list = target_s if isinstance(target_s, (list, tuple, set)) else [target_s]
            if not any(np.isclose(file_s, t) for t in target_list):
                print(f"Skipping {filename}: S={file_s} (Target S={target_s})")
                return
        else:
            print(f"Warning: Could not extract 'S' or 's' from filename '{filename}'. Processing anyway...")

    data = np.load(npz_file_path)

    K = int(data["K"])
    K2 = int(data["K2"])

    if target_K is not None and K != target_K:
        print(f"Skipping {filename}: K={K} (Target K={target_K})")
        return
    if target_K2 is not None and K2 != target_K2:
        print(f"Skipping {filename}: K2={K2} (Target K2={target_K2})")
        return

    print(f"Currently processing file: {filename} (K={K}, K2={K2})")

    population = data["population"]
    group_sizes = data["group_sizes"]
    epistasis_matrix = data["epistasis_matrix"]
    coefficients = data["coefficients"]
    genome_size = int(data["genome_size"])
    group_number = int(data["group_number"])

    sim_id_match = re.search(r"Sim=(\d+)", npz_file_path)
    sim_id = sim_id_match.group(1) if sim_id_match else "unknown"

    os.makedirs(output_dir, exist_ok=True)
    eligible_groups = 0

    for g in range(group_number):
        if eligible_groups >= max_groups_to_plot:
            break

        size = group_sizes[g]
        if size < 2:
            print(f"Skipping group {g} (Sim {sim_id}): Population size < 2")
            continue

        group_pop = population[g, :size]
        genomes_bytes = [genome.tobytes() for genome in group_pop]
        counts = Counter(genomes_bytes)

        if len(counts) < 2:
            print(f"Skipping group {g} (Sim {sim_id}): Monomorphic (fewer than 2 unique genomes)")
            continue

        print(f"Processing group {g} of simulation {sim_id} (Eligible count: {eligible_groups + 1})...")

        (
            line_type, 
            shannon_div, 
            unique_cnt, 
            p_current, 
            has_yellow,
            has_blue,
            in_yellow, 
            in_blue,
            has_stable_crossing,
            p_crossing,
            has_internal_coexist,
            has_group_optimum,
            p_vertex
        ) = plot_single_group_analysis(
            g,
            counts,
            group_pop,
            epistasis_matrix,
            coefficients,
            K,
            K2,
            genome_size,
            sim_id,
            output_dir,
            save_plot=export_individual_plots
        )
        sim_stats[sim_id][line_type] += 1
        eligible_groups += 1

        group_records.append({
            "sim_id": sim_id,
            "group_id": g,
            "K": K,
            "K2": K2,
            "classification": line_type,
            "internal_coexistence": 1 if has_internal_coexist else 0,
            "group_optimum_coexistence": 1 if has_group_optimum else 0,
            "has_stable_crossing": 1 if has_stable_crossing else 0,
            "p_crossing": p_crossing if p_crossing is not None else np.nan,
            "p_vertex": p_vertex,
            "shannon_diversity": shannon_div,
            "unique_genomes": unique_cnt,
            "group_size": size,
            "p_current": p_current,
            "has_yellow_region": has_yellow,
            "has_blue_region": has_blue,
            "in_yellow_region": in_yellow,
            "in_blue_region": in_blue
        })

    print(
        f"File {filename}: Processed {eligible_groups} eligible groups."
    )

def export_summary_file(sim_stats, k_suffix, output_dir="output_plots"):
    summary_filepath = os.path.join(output_dir, f"simulation_summary_{k_suffix}.txt")
    with open(summary_filepath, "w", encoding="utf-8") as f:
        f.write(f"Simulation Coexistence Potential Summary ({k_suffix})\n")
        f.write("=" * 80 + "\n")
        f.write(f"{'Sim ID':<10} | {'Internal Only':<15} | {'Group Opt Only':<15} | {'Both':<10} | {'Neither':<10}\n")
        f.write("-" * 80 + "\n")
        
        for sim_id, counts in sorted(sim_stats.items(), key=lambda x: str(x[0])):
            f.write(
                f"{sim_id:<10} | "
                f"{counts['internal_coexistence_only']:<15} | "
                f"{counts['group_optimum_only']:<15} | "
                f"{counts['both_coexistence']:<10} | "
                f"{counts['no_coexistence']:<10}\n"
            )
            
    print(f"\nSimulation summary successfully saved to '{summary_filepath}'.")

def analyze_diversity_vs_coexistence(group_records, k_suffix, output_dir="output_plots"):
    if not group_records:
        print("\nNo matching group records available for diversity analysis.")
        return

    df = pd.DataFrame(group_records)

    csv_path = os.path.join(output_dir, f"group_diversity_coexistence_analysis_{k_suffix}.csv")
    df.to_csv(csv_path, index=False)
    txt_path = os.path.join(output_dir, f"diversity_coexistence_analysis_{k_suffix}.txt")

    plot_diversity_analysis(df, k_suffix, output_dir)

    lines = []
    lines.append("=" * 65)
    lines.append(f"    SEPARATED COEXISTENCE DYNAMICS ANALYSIS ({k_suffix})")
    lines.append("=" * 65)
    lines.append(f"Total Groups Analyzed:                {len(df)}")
    lines.append(f"Internal Coexistence Only:            {len(df[df['classification'] == 'internal_coexistence_only'])}")
    lines.append(f"Group Optimum Coexistence Only:       {len(df[df['classification'] == 'group_optimum_only'])}")
    lines.append(f"Both Mechanisms Present:              {len(df[df['classification'] == 'both_coexistence'])}")
    lines.append(f"Neither Mechanism Present:           {len(df[df['classification'] == 'no_coexistence'])}")
    lines.append("-" * 65)

    lines.append("DESCRIPTIVE STATISTICS (Shannon Diversity)")
    for cat in ["both_coexistence", "internal_coexistence_only", "group_optimum_only", "no_coexistence"]:
        sub_df = df[df["classification"] == cat]["shannon_diversity"]
        if len(sub_df) > 0:
            lines.append(f"  {cat:<26} -> Mean: {sub_df.mean():.4f} | Median: {sub_df.median():.4f} (n={len(sub_df)})")
        else:
            lines.append(f"  {cat:<26} -> N/A")
    lines.append("-" * 65)

    lines.append("1. MANN-WHITNEY U TEST (Internal Coexist vs No Internal Coexist)")
    int_yes = df[df["internal_coexistence"] == 1]["shannon_diversity"]
    int_no = df[df["internal_coexistence"] == 0]["shannon_diversity"]
    if len(int_yes) > 0 and len(int_no) > 0:
        stat, p_val = stats.mannwhitneyu(int_yes, int_no, alternative="two-sided")
        lines.append(f"   U = {stat:.2f}, p-value = {p_val:.4e}")
        lines.append(f"     -> {'SIGNIFICANT DIFFERENCE' if p_val < 0.05 else 'No significant difference'}")
    else:
        lines.append("   Insufficient data for test")
    lines.append("-" * 65)

    lines.append("2. BINARY LOGISTIC REGRESSION (Internal Coexistence)")
    if df["shannon_diversity"].nunique() > 1 and len(df["internal_coexistence"].unique()) > 1:
        try:
            y = df["internal_coexistence"].values
            X = sm.add_constant(df["shannon_diversity"])
            logit = sm.Logit(y, X).fit(disp=0)
            
            coeff = logit.params["shannon_diversity"]
            p_val = logit.pvalues["shannon_diversity"]
            lines.append(f"   Coeff (Shannon Div): {coeff:.4f} | Odds Ratio: {np.exp(coeff):.4f}")
            lines.append(f"   p-value:             {p_val:.4e}")
            lines.append(f"   Pseudo R-squared:    {logit.prsquared:.4f}")
        except Exception as e:
            lines.append(f"   Failed to fit logistic regression: {e}")
    else:
        lines.append("   Insufficient data/variation.")

    lines.append("=" * 65)

    report_text = "\n".join(lines)
    print("\n" + report_text)

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(report_text + "\n")

    print(f"Analysis successfully written to '{txt_path}'.")

if __name__ == "__main__":
    output_directory = "output_plots_final"
    npz_files = glob.glob("*Top2Genomes.npz")
    sim_stats = defaultdict(lambda: {
        "internal_coexistence_only": 0,
        "group_optimum_only": 0,
        "both_coexistence": 0,
        "no_coexistence": 0
    })
    group_records = []
    
    k_str = f"K_{target_K if target_K is not None else 'Any'}"
    k2_str = f"K2_{target_K2 if target_K2 is not None else 'Any'}"
    s_str = f"s_{target_s if target_s is not None else 'Any'}"
    k_suffix = f"{k_str}_{k2_str}_{s_str}"

    for file in npz_files:
        process_npz_file(file, group_records, sim_stats, output_dir=output_directory)

    if sim_stats:
        export_summary_file(sim_stats, k_suffix, output_dir=output_directory)

    analyze_diversity_vs_coexistence(group_records, k_suffix, output_dir=output_directory)