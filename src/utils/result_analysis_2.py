# Directory containing the CSV files
csv_dir = "././statistic_results/ACTLL_ablation_3_epoch300"

import os
import pandas as pd


# Iterate through all CSV files in the directory
for file_name in os.listdir(csv_dir):
    if file_name.endswith('.csv'):
        file_path = os.path.join(csv_dir, file_name)
        try:
            # Read the CSV file
            df = pd.read_csv(file_path)
            
            # Ensure the CSV has the required columns before proceeding
            if 'avg_five_test_f1' in df.columns and 'std_five_test_f1' in df.columns:
                # Calculate the mean of avg_five_test_f1 and std_five_test_f1
                avg_f1_score = df['avg_five_test_f1'].mean()
                std_f1_score = df['std_five_test_f1'].mean()
                
                # Print the results for each file in the specified format (weighted F1 ± SD)
                print(f"File: {file_name}")
                print(f"(Weighted F1 ± SD): ({avg_f1_score:.3f} ± {std_f1_score:.3f})\n")
        except Exception as e:
            print(f"Error reading {file_path}: {e}")