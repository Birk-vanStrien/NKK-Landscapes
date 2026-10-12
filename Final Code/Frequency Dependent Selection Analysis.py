import numpy as np
import matplotlib.pyplot as plt

# Options
show_optimal_mutant_proportion = True
show_crossing_point = True
show_nodes = True  # Show stable and unstable nodes on the x-axis

# Wildtype values
Wt0 = 0.4 # Wildtype starting value (at p=0)
Wt1 = 0.6 # Wildtype ending value (at p=1)

# Mutant values
Mt0 = 0.6 # Mutant starting value (at p=0)
Mt1 = 0.4 # Mutant ending value (at p=1)

# X-axis: Proportion of mutants from 0 to 1
x = np.linspace(0, 1, 5000)  # High resolution for precise shading boundaries

# Wildtype formula: linear interpolation between Wt0 and Wt1
Beta_Wt = Wt0 + x * (Wt1 - Wt0)

# Mutant formula: linear interpolation between Mt0 and Mt1
Beta_Mt = Mt0 + x * (Mt1 - Mt0)

# Group fitness based on average of wildtype and mutant fitnesses, weighted by their proportions
Group_Fitness = (1 - x) * Beta_Wt + x * Beta_Mt

## Vertex Calculation
# Coefficients of the quadratic equation G(x) = A*x^2 + B*x + C
A = (Mt1 - Mt0) - (Wt1 - Wt0)
B = Wt1 - 2*Wt0 + Mt0

# Calculate the precise vertex x-coordinate
if A != 0:
    vertex_x = -B / (2 * A)
else:
    vertex_x = None  # Linear relationship if A is 0

# Calculate derivative of Group Fitness at each x
dG_dx = 2 * A * x + B

# Define conditions based on slope (increasing vs decreasing group fitness)
G_increasing = dG_dx > 0
G_decreasing = dG_dx < 0

# Natural selection conditions
Mt_favored = Beta_Mt > Beta_Wt
Wt_favored = Beta_Wt > Beta_Mt

## Plotting
fig, ax = plt.subplots(figsize=(9, 7))

# Plot lines
ax.plot(x, Beta_Wt, label=r'Wildtype ($\beta_{Wt}$)', color='blue', linewidth=2.5)
ax.plot(x, Beta_Mt, label=r'Mutant ($\beta_{Mt}$)', color='red', linewidth=2.5)
ax.plot(x, Group_Fitness, label=r'Group Fitness ($W_{G}$)', color='black', linestyle='--', linewidth=2)

# Mark the vertex
if show_optimal_mutant_proportion and vertex_x is not None and 0 <= vertex_x <= 1:
    vertex_label = "Vertex"
    ax.axvline(x=vertex_x, color='gray', linestyle=':', linewidth=1.5, label=vertex_label)

## Crossing Point Calculation
crossing_x = None
denom = (Wt1 - Wt0) - (Mt1 - Mt0)
if denom != 0:
    crossing_x = (Mt0 - Wt0) / denom

if show_crossing_point and crossing_x is not None and 0 <= crossing_x <= 1:
    ax.axvline(x=crossing_x, color='grey', linestyle='--', linewidth=1.5, label='Crossing Point')

## Nodes calculation and plotting on the x-axis (y = 0)
if show_nodes:
    nodes = []
    
    # Boundary x = 0 stability: Stable if mutant is disfavored when rare (Mt0 < Wt0)
    stability_0 = 'Stable' if (Mt0 - Wt0) < 0 else 'Unstable'
    nodes.append((0.0, stability_0))
    
    # Boundary x = 1 stability: Stable if wildtype is disfavored when rare (Mt1 > Wt1)
    stability_1 = 'Stable' if (Mt1 - Wt1) > 0 else 'Unstable'
    nodes.append((1.0, stability_1))
    
    # Internal crossing stability: Stable if fitness difference slope is negative
    if crossing_x is not None and 0 <= crossing_x <= 1:
        slope_diff = (Mt1 - Mt0) - (Wt1 - Wt0)
        stability_cross = 'Stable' if slope_diff < 0 else 'Unstable'
        nodes.append((crossing_x, stability_cross))
    
    # Sort nodes by their x-coordinate to correctly place phase arrows between them
    nodes.sort(key=lambda n: n[0])
    
    # Plot nodes on the x-axis (y = 0)
    stable_plotted = False
    unstable_plotted = False
    
    for nx, st in nodes:
        if st == 'Stable':
            label = 'Internal Stable Node' if not stable_plotted else ""
            ax.scatter([nx], [0.0], color='black', s=140, zorder=6, marker='o', edgecolors='black', label=label)
            stable_plotted = True
        else:
            label = 'Internal Unstable Node' if not unstable_plotted else ""
            ax.scatter([nx], [0.0], color='white', s=140, zorder=6, marker='o', edgecolors='black', label=label)
            unstable_plotted = True

    # Add fancy selection arrows (phase line arrows) between intervals
    for i in range(len(nodes) - 1):
        x_start = nodes[i][0]
        x_end = nodes[i+1][0]
        x_mid = (x_start + x_end) / 2.0
        
        # Evaluate selection direction at the midpoint
        # If Beta_Mt > Beta_Wt at midpoint, selection pushes right (mutants increase)
        val_wt = Wt0 + x_mid * (Wt1 - Wt0)
        val_mt = Mt0 + x_mid * (Mt1 - Mt0)
        
        arrow_dx = 0.04 if val_mt > val_wt else -0.04
        
        ax.annotate('', 
                    xy=(x_mid + arrow_dx, 0.0), 
                    xytext=(x_mid, 0.0),
                    arrowprops=dict(arrowstyle="->", color='black', lw=2.5, mutation_scale=15),
                    zorder=7)

## Region Shading
# Mutant is individually selected against (Wt > Mt), but spreading them BENEFITS the group (dG/dx > 0)
ax.fill_between(x, Beta_Wt, Beta_Mt, where=Wt_favored & G_increasing, 
                color='yellow', alpha=0.3, label='Spread possible with group selection',
                interpolate=True)

# Mutant is individually selected against (Wt > Mt), and spreading them HARMS the group (dG/dx < 0)
ax.fill_between(x, Beta_Wt, Beta_Mt, where=Wt_favored & G_decreasing, 
                color='red', alpha=0.3, label='No spread',
                interpolate=True)

# Mutant is individually favored (Mt > Wt), and spreading them BENEFITS the group (dG/dx > 0)
ax.fill_between(x, Beta_Wt, Beta_Mt, where=Mt_favored & G_increasing, 
                color='green', alpha=0.3, label='Spread supported',
                interpolate=True)

# Mutant is individually favored (Mt > Wt), but spreading them HARMS the group (dG/dx < 0)
ax.fill_between(x, Beta_Wt, Beta_Mt, where=Mt_favored & G_decreasing, 
                color='blue', alpha=0.3, label='Spread limited by group selection',
                interpolate=True)

# Formatting the plot
ax.set_xlabel('Proportion of Mutants ($p$)', fontsize=12)
ax.set_ylabel('Fitness', fontsize=12)
ax.set_xlim(0, 1)
ax.set_ylim(-0.03, 1)  
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10, framealpha=0.9)
# x-axis line at y=0
ax.axhline(0, color='black', linewidth=1.5, zorder=0)
ax.grid(False)

plt.tight_layout()
plt.show()
