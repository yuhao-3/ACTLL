import numpy as np
import os

def generate_patient_data(n_patients=1000, n_features=2, time_steps = 1, n_outcomes=2,feature_params = [(0, 1), (5, 1)], noise_level=0.4):
    np.random.seed(42)  # For reproducibility
    
    # Generate synthetic data with only one timepoint
    # Initialize an empty array to hold the data
    X = np.empty((n_patients, time_steps, n_features))

    # Generate data for each feature based on provided loc and scale
    for i, (loc, scale) in enumerate(feature_params):
        X[:, :, i] = np.random.normal(loc=loc, scale=scale, size=(n_patients, time_steps))

    
    # Generate labels for each patient with a specified number of outcomes
    # Here y is binary, thus only one column is enough to describe it; no need for n_outcomes in this case
    y = np.array([(1 if row[0, 0] > 0 and row[0, 1] > 5 else 0) for row in X]).reshape(-1, 1)  # Shape (N, 1)
    
    # Count class distribution
    unique_values, counts = np.unique(y, return_counts=True)
    print("Counts:", counts)
    
    # Convert y to one-hot encoding
    num_classes = 2  # Assuming binary classification for simplicity
    y_one_hot = np.eye(num_classes)[y.flatten()]  # Shape (N, num_classes)
    
    # Transition matrix for noisy labels
    transition_matrix = {
        0: [1-noise_level, noise_level],  # The chance that label stays 0 or flips to 1
        1: [noise_level, 1-noise_level]   # The chance that label stays 1 or flips to 0
    }
    
    # Generate noisy labels using the transition matrix
    y_noisy = np.array([np.random.choice([0, 1], p=transition_matrix[label[0]]) for label in y])
    
    
    
    # Convert y_noisy to one-hot encoding
    y_noisy_one_hot = np.eye(num_classes)[y_noisy.flatten()]  # Shape (N, num_classes)
    
    return X, y_one_hot, y_noisy_one_hot

def save_data_to_numpy(X, y_one_hot, y_noisy_one_hot, output_dir='../data/synthetic_camelot', noise_level=0.4):
    noise_level_str = f"{noise_level:.1f}".replace('.', '_')
    output_dir = os.path.join(output_dir, f'noise_{noise_level_str}')
    os.makedirs(output_dir, exist_ok=True)
    
    np.save(os.path.join(output_dir, 'X.npy'), X)
    np.save(os.path.join(output_dir, 'y.npy'), y_one_hot)
    np.save(os.path.join(output_dir, f'y_noisy_{noise_level_str}.npy'), y_noisy_one_hot)
    print(f'Saved X, y, and y_noisy to {output_dir}')



if __name__ == "__main__":
    noise_level = 0.9
    output_dir = '../data/synthetic_camelot'
    noise_level_str = f"{noise_level:.1f}".replace('.', '_')
    x_path = os.path.join(output_dir, f'noise_{noise_level_str}', 'X.npy')
    y_path = os.path.join(output_dir, f'noise_{noise_level_str}', 'y.npy')
    y_noisy_path = os.path.join(output_dir, f'noise_{noise_level_str}', f'y_noisy_{noise_level_str}.npy')
    
    if not os.path.exists(x_path) or not os.path.exists(y_path):
        feature_params = [(0, 1), (5, 1)]
        X, y_one_hot, y_noisy_one_hot = generate_patient_data(n_patients=1000, n_features=2, time_steps=1, feature_params=feature_params, noise_level=noise_level)
        save_data_to_numpy(X, y_one_hot, y_noisy_one_hot, output_dir, noise_level)
    else:
        print("Data already exists.")

    # Optionally load and print a sample to verify
    X = np.load(x_path)
    y_one_hot = np.load(y_path)
    y_noisy_one_hot = np.load(y_noisy_path)
    print("Sample X data (first patient):")
    print(X[0])
    print("\nSample y data (one-hot, first patient):")
    print(y_one_hot[0])
    print("\nSample y_noisy data (one-hot, first patient):")
    print(y_noisy_one_hot[0])