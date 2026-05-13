import numpy as np
import pandas as pd

# Load data
filename = "Optimal_Fitness_Matrix_N16 100 sims.npz"
matrix_data = np.load(filename, allow_pickle=True)
K_range = matrix_data['K']
K2_range = matrix_data['K2']
results_matrix = matrix_data['matrix']
std_matrix = matrix_data['std_matrix']

# Make table dataset
table_data = []
for i, K in enumerate(K_range):
    for j, K2 in enumerate(K2_range):
        table_data.append({'K': K, 'K2': K2, 'Optimal Fitness': results_matrix[i, j], 'Standard Deviation': std_matrix[i, j]})
table_df = pd.DataFrame(table_data)
print(table_df)