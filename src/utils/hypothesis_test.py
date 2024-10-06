# Assuming the script has already created and saved `overall_summary.csv`
import pandas as pd
from scipy.stats import mannwhitneyu

# Define the directory and the summary file path
summary_file_path = "././statistic_results/Combined/Overall_summary_across_all_datasets.csv"

# Read the overall summary CSV file
overall_summary = pd.read_csv(summary_file_path)

# Check the content of the overall summary file
print("Overall Summary Data:")
print(overall_summary)

# Define the reference model for comparison
reference_model = 'ACTLL_TimeAtteCNNv3'  # You can change this to your desired reference model

# Ensure that the summary contains the required columns
if 'model_name' not in overall_summary.columns or 'summary' not in overall_summary.columns:
    raise ValueError("The summary CSV file must contain 'model_name' and 'summary' columns.")

# Extract performance metrics from the 'summary' column for each model
# Note: 'summary' should contain formatted values like "0.798(0.025)", which we need to parse
def extract_f1_scores(summary_string):
    """
    Extract F1-score values from the summary string format "0.798(0.025)".
    Assumes that F1-scores are separated by '&' for different noise rates.
    Handles cases where there are empty or irregular entries.
    """
    # Initialize a list to store valid F1-scores
    f1_scores = []

    # Split the summary string by '&' and extract only the F1-scores
    parts = summary_string.split('&')
    
    for part in parts:
        # Remove leading and trailing whitespaces
        part = part.strip()
        
        # Skip empty parts
        if not part:
            continue

        # Check if the part has a numeric value before '(' and can be split correctly
        if '(' in part and len(part.split('(')[0].strip()) > 0:
            try:
                # Extract the value before '(' and convert to float
                f1_score = float(part.split('(')[0].strip())
                f1_scores.append(f1_score)
            except ValueError:
                # Print a warning if there's a value that cannot be converted
                print(f"Warning: Could not convert '{part.split('(')[0]}' to float. Skipping this entry.")
        else:
            print(f"Warning: Irregular entry '{part}' found. Skipping this entry.")

    # Return the list of valid F1 scores
    return f1_scores

# Extract reference model's F1 scores
reference_summary_row = overall_summary[overall_summary['model_name'] == reference_model]
if reference_summary_row.empty:
    raise ValueError(f"Reference model '{reference_model}' not found in the summary file.")
reference_f1_scores = extract_f1_scores(reference_summary_row['summary'].values[0])

# Perform Mann-Whitney U test for each model against the reference model
test_results = []

for _, row in overall_summary.iterrows():
    model_name = row['model_name']
    if model_name == reference_model:
        continue  # Skip comparison with itself
    
    # Extract F1-scores for the comparison model
    comparison_f1_scores = extract_f1_scores(row['summary'])

    # Perform Mann-Whitney U test
    u_statistic, p_value = mannwhitneyu(reference_f1_scores, comparison_f1_scores, alternative='two-sided')

    # Store results
    test_results.append({
        'Model': model_name,
        'U-Statistic': u_statistic,
        'P-Value': p_value
    })

    # Print results for each comparison
    print(f"Comparison of {model_name} vs. {reference_model}: U-statistic = {u_statistic:.3f}, p-value = {p_value:.5f}")

# Create a DataFrame for the test results and save it to a CSV
test_results_df = pd.DataFrame(test_results)
test_results_output_path = "././statistic_results/Combined/MannWhitneyU_Test_Results.csv"
test_results_df.to_csv(test_results_output_path, index=False)

print(f"\nMann-Whitney U test results saved to: {test_results_output_path}")