import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.load_MIMIC import prepare_dataloader


# Assuming X, Y come from prepare_dataloader(True)
# X is a numpy array and Y contains labels
X, Y = prepare_dataloader(False)

# X shape: (n_samples, n_timesteps, n_features)
n_samples, n_timesteps, n_features = X.shape

# 1. Compute basic statistics for each feature at each time step
# Reshape the data so that we can analyze each feature across all time steps
reshaped_data = X.reshape(-1, n_features)  # Combining samples and time steps

# Create a DataFrame for easy manipulation
df = pd.DataFrame(reshaped_data, columns=[f'Feature_{i+1}' for i in range(n_features)])

# Compute summary statistics (mean, std, min, max, etc.)
statistics = df.describe()

# 2. Compute mean and standard deviation over time steps for each feature
mean_per_timestep = np.mean(X, axis=0)  # Mean across samples, shape (n_timesteps, n_features)
std_per_timestep = np.std(X, axis=0)    # Standard deviation across samples, shape (n_timesteps, n_features)

# Convert mean and std to DataFrame for visualization
mean_df = pd.DataFrame(mean_per_timestep, columns=[f'Feature_{i+1}' for i in range(n_features)], index=[f'TimeStep_{i+1}' for i in range(n_timesteps)])
std_df = pd.DataFrame(std_per_timestep, columns=[f'Feature_{i+1}' for i in range(n_features)], index=[f'TimeStep_{i+1}' for i in range(n_timesteps)])

# 3. Analyze class distribution in Y
unique, counts = np.unique(Y, return_counts=True)
class_distribution = dict(zip(unique, counts))

# Save the statistical analysis and temporal analysis to CSV files
statistics.to_csv('statistics_summary.csv', index=True)
mean_df.to_csv('mean_per_timestep.csv', index=True)
std_df.to_csv('std_per_timestep.csv', index=True)

# Print class distribution
print(class_distribution)