
### Setup

import numpy as np
import matplotlib.pyplot as plt

# Load the data
data = np.load("16-40-500 K=1 K2=1 S=1 M=0.001 T=0.3M Sim_0.npz", allow_pickle=True)

# Extract the data
timesteps = data['timesteps']
average_fitness_over_time = data['average_fitness_over_time']
dominance_sample_size = data['dominance_sample_size']
track_amount = data['track_amount']
agreement_counts = data['agreement_counts']
individual_dominance_counts = data['individual_dominance_counts']
group_dominance_counts = data['group_dominance_counts']
suboptimal_counts = data['suboptimal_counts']
alignments_over_time = data['alignments_over_time']
weighted_alignments_over_time = data['weighted_alignments_over_time']
dominance_ratios_over_time = data['dominance_ratios_over_time']
weighted_dominance_ratios_over_time = data['weighted_dominance_ratios_over_time']
snapshot_times = data['snapshot_times']
snapshot_ind = data['snapshot_ind_deltas']
snapshot_grp = data['snapshot_grp_deltas']
analysis_timesteps = data['analysis_timesteps']
accommodation_history = data['accommodation_history']
enforcement_history = data['enforcement_history']
unchanged_agreement_history = data['unchanged_agreement_history']
unchanged_conflict_history = data['unchanged_conflict_history']
com_group_history = data['com_group_history']
com_individual_history = data['com_individual_history']
agreed_swap_history = data['agreed_swap_history']
conflicted_swap_history = data['conflicted_swap_history']
removed_locus_history = data['removed_locus_history']
optimal_fitness = data['optimal_fitness']

### Plotting

## Plotting funtions
def plotDominanceCombined(individual_deltas, group_deltas, ax=None, title='Dominance Plot'):
    # If no axis is provided (like the final standalone plot), create one
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.scatter(individual_deltas, group_deltas, alpha=0.4, s=5, c='blue')
    
    # Add center axes
    ax.axhline(0, color='black', linewidth=1)
    ax.axvline(0, color='black', linewidth=1)
    
    # Scale for labels
    xlimit = max(np.abs(individual_deltas)) * 1.1
    ylimit = max(np.abs(group_deltas)) * 1.1
    limit = max(max(np.abs(individual_deltas)), max(np.abs(group_deltas)), 0.001) * 1.1
    
    # Quadrant Labels (smaller font for grid view)
    ax.text(limit*0.5, limit*0.5, '+/+', ha='center', color='grey', fontsize=8)
    ax.text(limit*0.5, -limit*0.5, 'Grp Dom\n(+/-)', ha='center', color='red', fontsize=8)
    ax.text(-limit*0.5, limit*0.5, 'Ind Dom\n(-/+)', ha='center', color='red', fontsize=8)
    ax.text(-limit*0.5, -limit*0.5, 'Agree\n(-/-)', ha='center', color='green', fontsize=8)
    
    ax.set_xlabel('$\Delta F_{ind}$', fontsize=9)
    ax.set_ylabel('$\Delta \\bar{F}_{group}$', fontsize=9)
    ax.set_title(title, fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.5)
    # ax.set_xlim(-limit, limit)
    # ax.set_ylim(-limit/(max_group_size-1)*1.1, limit/(max_group_size-1)*1.1)
    ax.set_xlim(-xlimit, xlimit)
    ax.set_ylim(-ylimit, ylimit)

## Plotting directly

# Create a nxn grid of subplots
fig, axes = plt.subplots(2, 3, figsize=(10, 9))
plt.subplots_adjust(left=0.04, bottom=0.07, right=0.895, top=0.965, wspace=0.2, hspace=0.5)

# Cumulative Quadrant Counts (Stacked Area Plot)
# Define the order of stacking
y1 = np.array(agreement_counts)/dominance_sample_size
y2 = y1 + np.array(individual_dominance_counts)/dominance_sample_size
y3 = y2 + np.array(group_dominance_counts)/dominance_sample_size
y4 = y3 + np.array(suboptimal_counts)/dominance_sample_size

axes[0, 0].fill_between(timesteps, 0, y1, label='Agreement', color='#2ca02c', alpha=0.7)
axes[0, 0].fill_between(timesteps, y1, y2, label='Indiv. Dominance', color='#d62728', alpha=0.7)
axes[0, 0].fill_between(timesteps, y2, y3, label='Group Dominance', color='#1f77b4', alpha=0.7)
axes[0, 0].fill_between(timesteps, y3, y4, label='Suboptimal', color='#7f7f7f', alpha=0.7)
axes[0, 0].set_title('Cumulative Quadrant Fractions')
axes[0, 0].set_ylabel('Fraction of Samples')
axes[0, 0].set_xlabel('Timestep')
axes[0, 0].legend(loc='upper left', prop = {'size':8}, bbox_to_anchor=(0, -0.1)) # Moved legend outside to see clearly
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].set_xlim(timesteps[0], timesteps[-1])
axes[0, 0].set_ylim(0, 1)

# Fitness Over Time
axes[0, 1].plot(timesteps, average_fitness_over_time, label='Average Fitness', color='black')
axes[0, 1].set_title('Fitness Over Time')
axes[0, 1].set_xlabel('Timestep')
axes[0, 1].set_ylabel('Average Fitness')
axes[0, 1].grid(True, alpha=0.3)
# Draw optimal fitness line
axes[0, 1].axhline(optimal_fitness, color='red', linestyle='--', label='Optimal Fitness')

# Conflict resolution history
y1 = accommodation_history/track_amount
y2 = y1 + enforcement_history/track_amount
y3 = y2 + unchanged_agreement_history/track_amount
y4 = y3 + agreed_swap_history/track_amount
y5 = y4 + removed_locus_history/track_amount
y6 = y5 + conflicted_swap_history/track_amount
y7 = y6 + unchanged_conflict_history/track_amount
y8 = y7 + com_group_history/track_amount
y9 = y8 + com_individual_history/track_amount
axes[0, 2].fill_between(analysis_timesteps, 0, y1, label='Accomodation', color="blue", alpha=0.7)
axes[0, 2].fill_between(analysis_timesteps, y1, y2, label='Enforcement', color="#726bfd", alpha=0.7)
axes[0, 2].fill_between(analysis_timesteps, y2, y3, label='Unchanged Agree', color='#7f7f7f', alpha=0.7)
axes[0, 2].fill_between(analysis_timesteps, y3, y4, label='Agree Swap', color='#4d4d4d', alpha=0.7)
axes[0, 2].fill_between(analysis_timesteps, y4, y5, label='Removed Locus', color='black', alpha=0.7)
axes[0, 2].fill_between(analysis_timesteps, y5, y6, label='Conflict Swap', color='#4d4d4d', alpha=0.7)
axes[0, 2].fill_between(analysis_timesteps, y6, y7, label='Unchanged Conflict', color="#7f7f7f", alpha=0.7)
axes[0, 2].fill_between(analysis_timesteps, y7, y8, label='CoM Group', color='#FF6347', alpha=0.7)
axes[0, 2].fill_between(analysis_timesteps, y8, y9, label='CoM Individual', color='red', alpha=0.7)


axes[0, 2].set_title('Conflict Changes')
axes[0, 2].set_xlabel('Timestep')
axes[0, 2].set_ylabel('Fraction of Samples')
axes[0, 2].set_xlim(analysis_timesteps[0], analysis_timesteps[-1])
axes[0, 2].set_ylim(0, 1)
axes[0, 2].legend(bbox_to_anchor=(1, 1), prop = {'size':8}, reverse = True)
axes[0, 2].grid(True, alpha=0.3)

# Only enforment and accommodation vs conflict creation

res_cre_total = (accommodation_history + enforcement_history + com_group_history + com_individual_history)

y1_p = np.divide(accommodation_history, res_cre_total, out=np.zeros_like(res_cre_total, dtype=float), where=res_cre_total!=0)
y2_p = np.divide(enforcement_history, res_cre_total, out=np.zeros_like(res_cre_total, dtype=float), where=res_cre_total!=0)
y3_p = np.divide(com_group_history, res_cre_total, out=np.zeros_like(res_cre_total, dtype=float), where=res_cre_total!=0)
y4_p = np.divide(com_individual_history, res_cre_total, out=np.zeros_like(res_cre_total, dtype=float), where=res_cre_total!=0)

stack1 = y1_p
stack2 = stack1 + y2_p
stack3 = stack2 + y3_p
stack4 = stack3 + y4_p

axes[1, 2].fill_between(analysis_timesteps, 0, stack1, label='Accommodation', color="blue", alpha=0.7)
axes[1, 2].fill_between(analysis_timesteps, stack1, stack2, label='Enforcement', color="#726bfd", alpha=0.7)
axes[1, 2].fill_between(analysis_timesteps, stack2, stack3, label='CoM Group', color='#FF6347', alpha=0.7)
axes[1, 2].fill_between(analysis_timesteps, stack3, stack4, label='CoM Individual', color='red', alpha=0.7)

axes[1, 2].set_title('Resolution vs Creation')
axes[1, 2].set_xlabel('Timestep')
axes[1, 2].set_ylabel('Fraction of Samples')
axes[1, 2].set_xlim(analysis_timesteps[0], analysis_timesteps[-1])
axes[1, 2].set_ylim(0, 1)
axes[1, 2].legend(bbox_to_anchor=(1, 1), prop = {'size':8}, reverse = True)
axes[1, 2].grid(True, alpha=0.3)

# Alignment Over Time
axes[1, 0].plot(timesteps, alignments_over_time, label='Alignment', color='black')
axes[1, 0].plot(timesteps, weighted_alignments_over_time, label='Weighted', color='red')
axes[1, 0].set_title('Alignment Over Time')
axes[1, 0].set_ylabel('Consensus / Conflict')
axes[1, 0].set_xlabel('Timestep')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Dominance Ratio Over Time
axes[1, 1].plot(timesteps, dominance_ratios_over_time, label='Dominance Ratio', color='black')
axes[1, 1].plot(timesteps, weighted_dominance_ratios_over_time, label='Weighted', color='red')
axes[1, 1].set_title('Dominance Ratio Over Time')
axes[1, 1].set_xlabel('Timestep')
axes[1, 1].set_ylabel('Group / Individual')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

# Average Deltas Over Time
# axes[0, 1].plot(timesteps, average_individual_deltas_over_time, label='Avg Indiv Delta', color='red')
# axes[0, 1].plot(timesteps, average_group_deltas_over_time, label='Avg Group Delta', color='blue')
# axes[0, 1].set_title('Average Deltas Over Time')
# axes[0, 1].set_ylabel('Average Delta')
# axes[0, 1].legend()
# axes[0, 1].grid(True, alpha=0.3)

# plt.tight_layout()
plt.show()

# Plot dominance snapshots
fig_snapshots, axes_snapshots = plt.subplots(2, 3, figsize=(20, 12)) # Adjusted height for better aspect ratio
axes_snapshots = axes_snapshots.flatten()
plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.4)

# Merge separate snapshot arrays
for idx, (ts, d_ind, d_grp) in enumerate(zip(snapshot_times, snapshot_ind, snapshot_grp)):
    if idx < len(axes_snapshots):
        plotDominanceCombined(d_ind, d_grp, ax=axes_snapshots[idx], title=f"T = {ts}")

# Hide any unused subplots if checkpoints < 6
for j in range(len(snapshot_times), len(axes_snapshots)):
    axes_snapshots[j].axis('off')
    
# plt.tight_layout()
plt.show()


