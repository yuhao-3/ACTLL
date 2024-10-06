import os
import pandas as pd

# Define paths for the input folders and output folder
input_folder_mimic = '././statistic_results/Benchmark_MIMIC'
input_folder_eicu = '././statistic_results/eICU'
output_folder = '././statistic_results/Combined'
ehr_output_folder = '././statistic_results/EHR_summary'

# Create the output folders if they don't exist
os.makedirs(output_folder, exist_ok=True)
os.makedirs(ehr_output_folder, exist_ok=True)

# Function to analyze each CSV file and extract performance by noise level and noise rate
def analyze_csv(file_path, model_name):
    df = pd.read_csv(file_path)
    file_info = os.path.basename(file_path).split('_')

    matched_model_name = None
    for i in range(1, len(file_info)):
        potential_model_name = '_'.join(file_info[:i])
        if potential_model_name == model_name:
            matched_model_name = potential_model_name
            break

    if matched_model_name:
        noise_type = file_info[i]
        noise_rate = file_info[i + 1]
        df['model_name'] = matched_model_name
        df['noise_type'] = noise_type
        df['noise_rate'] = noise_rate
        return df
    else:
        print(f"Model name '{model_name}' does not match any part of the file name '{os.path.basename(file_path)}'")
        return None

# Function to process the DataFrame and calculate average metrics across all datasets
def process_results(df):
    df = df.sort_values(by=['noise_type', 'noise_rate'], ascending=[True, True])
    result = []
    for noise_type in ['sym', 'asym', 'inst']:
        noise_df = df[df['noise_type'] == noise_type]
        avg_metrics = noise_df.groupby('noise_rate').agg({
            'avg_five_test_f1': 'mean',
            'std_five_test_f1': 'mean'
        }).reset_index()
        noise_results = [f"{row['avg_five_test_f1']:.3f}({row['std_five_test_f1']:.3f})" for _, row in avg_metrics.iterrows()]
        result.append(" & ".join(noise_results))
    return " & ".join(result)

# Define model names for processing
model_names = ['ACTLL_TimeAtteCNNv3', 'ACTLL_DiffusionCNNv3', 'ACTLL_CNNv3','SREA',
               'ACTLL_TimeAtteCNNv3_GMM','ACTLL_TimeAtteCNNv3_noAug','ACTLL_TimeAtteCNNv3_noCorr',
               'ACTLL_TimeAtteCNNv3_noAugnoCorr','ACTLL_TimeAtteCNNv3_SLoss',
               'CTW','MixUp_BMM','co_teaching','dividemix',
               'sigua','vanilla']

# Read and analyze CSV files
overall_summary_data = []
ehr_summary_data = []


# Iterate over model names and process files for each model
for model_name in model_names:
    mimic_results = []
    eicu_results = []
    combined_results = []

    # Read CSV files from both MIMIC and eICU folders and combine them
    for folder, dataset_type in [(input_folder_mimic, 'MIMIC'), (input_folder_eicu, 'eICU')]:
        for filename in os.listdir(folder):
            if model_name in filename and filename.endswith(".csv"):
                file_path = os.path.join(folder, filename)
                df = analyze_csv(file_path, model_name)

                # Collect all results and separate MIMIC and eICU results
                if df is not None and not df.empty:
                    combined_results.append(df)
                    if dataset_type == 'MIMIC':
                        mimic_results.append(df[df['dataset_name'].str.contains('MIMIC', case=False, na=False)])
                    elif dataset_type == 'eICU':
                        eicu_results.append(df)

    
    # Create a subfolder for each model in the combined folder
    model_output_dir = os.path.join(output_folder, model_name)
    os.makedirs(model_output_dir, exist_ok=True)

    # Concatenate and save combined results across all datasets
    if combined_results:
        df_combined = pd.concat(combined_results, ignore_index=True)
        df_combined = df_combined.sort_values(by=['noise_type', 'noise_rate'])
        overall_summary = process_results(df_combined)
        
        # Save the combined summary for this model
        output_path_combined = os.path.join(model_output_dir, f"{model_name}_Overall_performance_summary.csv")
        df_combined.to_csv(output_path_combined, index=False)
        overall_summary_data.append({'model_name': model_name, 'summary': overall_summary})

    # Create EHR summary by combining MIMIC and eICU results
    if mimic_results and eicu_results:
        # Concatenate MIMIC and eICU results
        df_ehr_combined = pd.concat([pd.concat(mimic_results, ignore_index=True), pd.concat(eicu_results, ignore_index=True)], ignore_index=True)

        # Calculate average metrics for EHR data
        ehr_combined_summary = process_results(df_ehr_combined)
        
        # Save EHR summary for this model
        ehr_output_path = os.path.join(ehr_output_folder, f"{model_name}_EHR_performance_summary.csv")
        df_ehr_combined.to_csv(ehr_output_path, index=False)
        
        # Append EHR summary
        ehr_summary_data.append({'model_name': model_name, 'summary': ehr_combined_summary})

# Save overall and EHR summaries as separate CSV files
overall_summary_df = pd.DataFrame(overall_summary_data)
ehr_summary_df = pd.DataFrame(ehr_summary_data)

overall_summary_output_path = os.path.join(output_folder, "Overall_summary_across_all_datasets.csv")
ehr_summary_output_path = os.path.join(ehr_output_folder, "EHR_summary_across_all_datasets.csv")

overall_summary_df.to_csv(overall_summary_output_path, index=False)
ehr_summary_df.to_csv(ehr_summary_output_path, index=False)

print(f"Overall summary saved to {overall_summary_output_path}")
print(f"EHR summary saved to {ehr_summary_output_path}")