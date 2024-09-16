import pymc3 as pm
import matplotlib.pyplot as plt
import numpy as np
n_samples = 1000

# Using the real loss data (after standardization)
# Generate synthetic loss data for each set (in range [0, 1])
loss_confident = np.random.beta(5, 2, size=int(n_samples * 0.5))  # Low loss for confident samples
loss_hard = np.random.beta(2, 2, size=int(n_samples * 0.3))       # Medium loss for hard samples
loss_noisy = np.random.beta(2, 5, size=int(n_samples * 0.2))      # High loss for noisy samples

# Combine the data
loss_data = np.concatenate([loss_confident, loss_hard, loss_noisy])

# Standardize the real loss values to the range [0, 1]
def min_max_scale(loss):
    return (loss - np.min(loss)) / (np.max(loss) - np.min(loss))

loss_real_standardized = min_max_scale(loss_data)

# Plot the distribution of the real loss data
plt.hist(loss_data, bins=50, density=True, alpha=0.5, color='g')
plt.title("Real Loss Data (Standardized to [0, 1])")
plt.show()

# Define the Bayesian Beta Mixture Model using PyMC3
with pm.Model() as loss_model:
    # Priors for alpha and beta of each Beta distribution (one for each set)
    alpha_confident = pm.Gamma('alpha_confident', 2.0, 2.0)
    beta_confident = pm.Gamma('beta_confident', 2.0, 2.0)
    
    alpha_hard = pm.Gamma('alpha_hard', 2.0, 2.0)
    beta_hard = pm.Gamma('beta_hard', 2.0, 2.0)
    
    alpha_noisy = pm.Gamma('alpha_noisy', 2.0, 2.0)
    beta_noisy = pm.Gamma('beta_noisy', 2.0, 2.0)
    
    # Mixture weights (proportions of each set)
    weights = pm.Dirichlet('weights', a=np.array([1, 1, 1]))

    # Define the Beta distributions for each set
    beta_confident_dist = pm.Beta.dist(alpha_confident, beta_confident)
    beta_hard_dist = pm.Beta.dist(alpha_hard, beta_hard)
    beta_noisy_dist = pm.Beta.dist(alpha_noisy, beta_noisy)

    # Mixture of Beta distributions
    mixture = pm.Mixture('mixture', w=weights, comp_dists=[beta_confident_dist, beta_hard_dist, beta_noisy_dist], observed=loss_data)
    
    # Inference using MCMC
    trace = pm.sample(2000, return_inferencedata=True)

# Plot the posterior distribution of the parameters
pm.plot_trace(trace)
plt.show()

# Print summary of the posterior estimates
pm.summary(trace)