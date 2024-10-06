import os
import pandas as pd

# Directory containing the CSV files
csv_dir = "././statistic_results"

# Output CSV file name for overall summary
output_csv = os.path.join(csv_dir, "summary.csv")

# Output CSV file name for EHR summary (average of MIMIC and eICU)
ehr_output_csv = os.path.join(csv_dir, "EHR_summary.csv")

# Get a list of all CSV files in the directory and sort them alphabetically
csv_files = sorted([file for file in os.listdir(csv_dir) if file.endswith('.csv')])

# Initialize lists to store summary data for all datasets and EHR-specific datasets
summary_data = []
ehr_summary_data = []

# Iterate through the sorted CSV files and process each file
for file_name in csv_files:
    file_path = os.path.join(csv_dir, file_name)
    try:
        # Read the CSV file
        df = pd.read_csv(file_path)
        
        # Ensure the CSV has the required columns before proceeding
        if 'avg_five_test_f1' in df.columns and 'std_five_test_f1' in df.columns:
            # Calculate the mean of avg_five_test_f1 and std_five_test_f1 for the entire dataset
            avg_f1_score = round(df['avg_five_test_f1'].mean(), 3)
            std_f1_score = round(df['std_five_test_f1'].mean(), 3)
            
            # Append the results for this file to the overall summary list
            summary_data.append({
                'File Name': file_name,
                'Weighted F1': avg_f1_score,
                'Standard Deviation': std_f1_score
            })
            
            # Check if the dataset has 'dataset_name' column to identify EHR-related datasets
            if 'dataset_name' in df.columns:
                # Filter rows for MIMIC and eICU datasets only
                mimic_df = df[df['dataset_name'].str.contains('MIMIC', case=False, na=False)]
                eicu_df = df[df['dataset_name'].str.contains('eICU', case=False, na=False)]

                # Calculate F1 score for MIMIC rows only
                if not mimic_df.empty:
                    mimic_avg_f1 = round(mimic_df['avg_five_test_f1'].mean(), 3)
                    mimic_std_f1 = round(mimic_df['std_five_test_f1'].mean(), 3)
                
                # Calculate F1 score for eICU rows only
                if not eicu_df.empty:
                    eicu_avg_f1 = round(eicu_df['avg_five_test_f1'].mean(), 3)
                    eicu_std_f1 = round(eicu_df['std_five_test_f1'].mean(), 3)

                # Calculate EHR (average of MIMIC and eICU) if both MIMIC and eICU data are present
                if not mimic_df.empty and not eicu_df.empty:
                    # Calculate weighted F1 and standard deviation for EHR
                    ehr_avg_f1 = round((mimic_avg_f1 + eicu_avg_f1) / 2, 3)
                    ehr_std_f1 = round((mimic_std_f1 + eicu_std_f1) / 2, 3)
                    ehr_summary_data.append({
                        'File Name': file_name,
                        'EHR Dataset': 'MIMIC + eICU',
                        'Weighted F1': ehr_avg_f1,
                        'Standard Deviation': ehr_std_f1
                    })
                    # Print formatted F1 score for EHR dataset
                    print(f"EHR Dataset: {file_name} (Weighted F1 ± SD): {ehr_avg_f1:.3f} ± {ehr_std_f1:.3f}")

            print(f"Processed: {file_name} (Weighted F1 ± SD): ({avg_f1_score:.3f} ± {std_f1_score:.3f})")

    except Exception as e:
        print(f"Error reading {file_path}: {e}")

# Create a DataFrame from the overall summary data
summary_df = pd.DataFrame(summary_data)

# Save the summary DataFrame to a new CSV file
summary_df.to_csv(output_csv, index=False)

# Create a DataFrame from the EHR summary data
ehr_summary_df = pd.DataFrame(ehr_summary_data)

# Save the EHR summary DataFrame to a new CSV file
ehr_summary_df.to_csv(ehr_output_csv, index=False)

print(f"\nOverall Summary CSV saved to: {output_csv}")
print(f"EHR Summary CSV saved to: {ehr_output_csv}")