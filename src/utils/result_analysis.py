import os
import pandas as pd

# Directory containing the CSV files
csv_dir = "././statistic_results/Benchmark_MIMIC"

# List of model names you want to analyze
model_names = ['ACTLL_TimeAtteCNNv3', 'ACTLL_DiffusionCNNv3', 'ACTLL_CNNv3','SREA',
               'ACTLL_TimeAtteCNNv3_GMM','CTW','MixUp_BMM','co_teaching','dividemix',
               'sigua','vanilla']

# model_names=['CTW']


# Function to analyze each CSV file and extract performance by noise level and noise rate for both MIMIC and overall dataset
def analyze_csv(file_path, model_name):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_path)
    
    # Extract file name components
    file_info = os.path.basename(file_path).split('_')
    
    # Try to match the model_name from the beginning of the file name
    # Create a joined string of the file_info parts starting from the front until a match is found
    matched_model_name = None
    for i in range(1, len(file_info)):
        # Join the first i elements to form a potential model name
        potential_model_name = '_'.join(file_info[:i])
        if potential_model_name == model_name:
            matched_model_name = potential_model_name
            break

    # If the model name is found in the file name, continue processing
    if matched_model_name:
        # Retrieve noise_type and noise_rate, assuming they come immediately after the model name
        noise_type = file_info[i]    # e.g., 'asym' (the next part after the matched model name)
        noise_rate = file_info[i + 1]  # e.g., '30' (the part after noise_type)
        
        # Add these as columns to the DataFrame
        df['model_name'] = matched_model_name
        df['noise_type'] = noise_type
        df['noise_rate'] = noise_rate

        # Split into MIMIC and overall datasets based on the dataset_name column
        if 'dataset_name' in df.columns:
            # Filter rows that contain 'MIMIC' in the dataset_name column
            df_mimic = df[df['dataset_name'].str.contains('MIMIC', case=False, na=False)]
            return df_mimic, df
        else:
            return None, df
    else:
        # If the model name doesn't match, return None or raise an exception based on preference
        print(f"Model name '{model_name}' does not match any part of the file name '{os.path.basename(file_path)}'")
        return None, None


# Function to process the DataFrame and calculate the average metrics across all datasets
def process_results(df):
    # Sort by noise type and noise rate (ascending order)
    df = df.sort_values(by=['noise_type', 'noise_rate'], ascending=[True, True])

    print(df)
    # Initialize a result list to store the output for each noise type
    result = []

    # Loop over each noise type in the order sym -> asym -> inst
    for noise_type in ['sym', 'asym', 'inst']:
        # Filter rows for the current noise type
        noise_df = df[df['noise_type'] == noise_type]

        # Group by noise rate and calculate the mean metrics across all datasets
        avg_metrics = noise_df.groupby('noise_rate').agg({
            'avg_five_test_f1': 'mean',
            'std_five_test_f1': 'mean'
        }).reset_index()

        # Create a list to store results for this noise type
        noise_results = []

        # Loop over rows and append the formatted string for each noise rate
        for _, row in avg_metrics.iterrows():
            avg_f1 = row['avg_five_test_f1']
            std_f1 = row['std_five_test_f1']
            noise_results.append(f"{avg_f1:.3f}({std_f1:.3f})")

        # Append the results for this noise type to the final result list
        result.append(" & ".join(noise_results))  # Use & as separator

    # Combine the results for all noise types into a single row
    return " & ".join(result)


# Collect summaries to print after all datasets are processed
summaries = []

# Data to store in the summary CSV files
mimic_summary_data = []
overall_summary_data = []

# Analyze all CSV files in the directory for the specified model names
for model_name in model_names:
    mimic_results = []
    overall_results = []
    
    # Iterate over files in the directory, assuming the model name is part of the filename
    for filename in os.listdir(csv_dir):
        if model_name in filename and filename.endswith(".csv"):
            file_path = os.path.join(csv_dir, filename)
            
            # Always check for MIMIC dataset inside the dataset_name column
            df_mimic, df_overall = analyze_csv(file_path, model_name)
            

            # Collect MIMIC-specific rows
            if df_mimic is not None and not df_mimic.empty:
                mimic_results.append(df_mimic)
            
            # Collect overall rows (entire dataset, including MIMIC)
            if df_overall is not None and not df_overall.empty:
                overall_results.append(df_overall)

    # Ensure the output directory for the model exists
    model_output_dir = os.path.join(csv_dir, model_name)
    os.makedirs(model_output_dir, exist_ok=True)

    # If there are results for MIMIC datasets, concatenate and save them
    if mimic_results:
        # Concatenate all MIMIC-specific DataFrames
        df_mimic_final = pd.concat(mimic_results, ignore_index=True)

        # Sort by noise type and noise rate for readability
        df_mimic_final = df_mimic_final.sort_values(by=['noise_type', 'noise_rate'])

        # Generate summary for MIMIC and store it for printing later
        mimic_summary = process_results(df_mimic_final)
        summaries.append(f"MIMIC Summary for {model_name}: {mimic_summary}")

        # Save the results for MIMIC to a new CSV file inside the model's folder
        output_path_mimic = os.path.join(model_output_dir, f"{model_name}_MIMIC_performance_summary.csv")
        df_mimic_final.to_csv(output_path_mimic, index=False)

        # Add to MIMIC summary data for CSV
        mimic_summary_data.append({'model_name': model_name, 'summary': mimic_summary})
    else:
        summaries.append(f"No results found for {model_name} on MIMIC dataset.")

    # If there are results for overall datasets, calculate the average statistics and save them
    if overall_results:
        # Concatenate all overall DataFrames
        df_overall_final = pd.concat(overall_results, ignore_index=True)

        # Sort by noise type and noise rate for readability
        df_overall_final = df_overall_final.sort_values(by=['noise_type', 'noise_rate'])

        # Generate summary for overall results and store it for printing later
        overall_summary = process_results(df_overall_final)
        summaries.append(f"Overall Summary for {model_name}: {overall_summary}")

        # Save the overall average results for the model inside the model's folder
        output_path_overall = os.path.join(model_output_dir, f"{model_name}_Overall_performance_summary.csv")
        df_overall_final.to_csv(output_path_overall, index=False)

        # Add to overall summary data for CSV
        overall_summary_data.append({'model_name': model_name, 'summary': overall_summary})
    else:
        summaries.append(f"No overall results found for {model_name}.")

# Print all summaries after processing all models
for summary in summaries:
    print(summary)

# Create DataFrames for the summary data
mimic_summary_df = pd.DataFrame(mimic_summary_data)
overall_summary_df = pd.DataFrame(overall_summary_data)

# Save the MIMIC summary to a separate CSV file
mimic_summary_output_path = os.path.join(csv_dir, "MIMIC_summary.csv")
mimic_summary_df.to_csv(mimic_summary_output_path, index=False)

# Save the overall summary to a separate CSV file
overall_summary_output_path = os.path.join(csv_dir, "Overall_summary.csv")
overall_summary_df.to_csv(overall_summary_output_path, index=False)

print(f"MIMIC summary saved to {mimic_summary_output_path}")
print(f"Overall summary saved to {overall_summary_output_path}")