import numpy as np
import pandas as pd

def generate_clean_dataset(n_samples = 100):
    # # Number of samples
    n_samples = 100

    # Generate synthetic features
    np.random.seed(42)  # For reproducibility
    feature_1 = np.random.normal(loc=0, scale=1, size=n_samples)  # Feature 1
    feature_2 = np.random.normal(loc=5, scale=2, size=n_samples)  # Feature 2

    # Assign labels based on a rule to ensure 100% certainty
    # For example, if feature_1 > 0 and feature_2 > 5, label = 1, else label = 0
    labels = np.where((feature_1 > 0) & (feature_2 > 5), 1, 0)

    # Create a DataFrame
    df = pd.DataFrame({'Feature_1': feature_1, 'Feature_2': feature_2, 'Label': labels})

    print(df.head())
    

    return(df)

# Function to introduce class-dependent noise based on the transition matrix
def introduce_noise(labels, transition_matrix):
    noisy_labels = []
    for label in labels:
        # The current label determines the row in the transition matrix
        # We then choose the new label based on the probabilities in that row
        noisy_label = np.random.choice([0, 1], p=transition_matrix[label])
        noisy_labels.append(noisy_label)
    return np.array(noisy_labels)

def generate_noisy_dataset(n_samples=100, noise_level = 0.1):

    # Seed for reproducibility
    np.random.seed(42)

    # Generate synthetic features
    feature_1 = np.random.normal(loc=0, scale=1, size=n_samples)  # Feature 1
    feature_2 = np.random.normal(loc=5, scale=2, size=n_samples)  # Feature 2

    # Assign clean labels based on a deterministic rule
    clean_labels = np.where((feature_1 > 0) & (feature_2 > 5), 1, 0)

    # Define a class-dependent noise transition matrix
    # For simplicity, let's consider a binary classification with the following matrix:
    # | From\To | 0   | 1   |
    # |---------|-----|-----|
    # | 0       | 0.6 | 0.4 |  # 40% chance to flip from 0 to 1
    # | 1       | 0.3 | 0.7 |  # 30% chance to flip from 1 to 0

    transition_matrix = np.array([[
        1-noise_level, noise_level],  # Probability of staying in class 0 or flipping to class 1
        [noise_level, 1-noise_level]   # Probability of flipping to class 0 or staying in class 1
    ])

    # Apply noise to the clean labels
    noisy_labels = introduce_noise(clean_labels, transition_matrix)

    # Create a DataFrame with clean and noisy labels for comparison
    df = pd.DataFrame({
        'Feature_1': feature_1,
        'Feature_2': feature_2,
        'Clean_Label': clean_labels,
        'Noisy_Label': noisy_labels
    })
    
    
    df.to_csv(f'../data/synthetic_noisy_data_{noise_level}.csv', index=False)

    print(df.head())
    




if __name__ == "__main__":
    generate_noisy_dataset(n_samples=1000)