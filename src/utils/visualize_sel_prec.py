import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Get the current working directory
current_directory = os.getcwd()
print(f"Current Directory: {current_directory}")

# Define file paths for the different sample selection methods
dataset_name = "MIMIC"
file_paths = {
    'Small Loss (1)': f"./src/bar_info/{dataset_name}1.csv",
    'GMM (2)': f"./src/bar_info/{dataset_name}2.csv",
    'BMM (5)': f"./src/bar_info/{dataset_name}5.csv"
}

# Create a DataFrame to store all data
all_data = []

# Read each CSV file and calculate precision
for method, file_path in file_paths.items():
    if os.path.exists(file_path):
        # Load the CSV file
        df = pd.read_csv(file_path)

        # Convert 'TP' and 'FP' columns to numeric data types, in case they are stored as strings
        df['TP'] = pd.to_numeric(df['TP'], errors='coerce')
        df['FP'] = pd.to_numeric(df['FP'], errors='coerce')

        # Ensure 'epoch' and 'seed' columns are also numeric, if needed
        df['epoch'] = pd.to_numeric(df['epoch'], errors='coerce')
        df['seed'] = pd.to_numeric(df['seed'], errors='coerce')

        # Calculate Precision for each row
        df['Precision'] = df['TP'] / (df['TP'] + df['FP'])

        # Calculate the average precision over 5 seeds for each epoch
        df_grouped = df.groupby(['epoch']).agg({'Precision': 'mean'}).reset_index()
        
        # Add method information
        df_grouped['Method'] = method

        # Append to all_data list
        all_data.append(df_grouped)
    else:
        print(f"File not found: {file_path}")

# Concatenate all data into a single DataFrame
all_df = pd.concat(all_data, ignore_index=True)

# Set up a seaborn plot for better visualization
plt.figure(figsize=(14, 7))

# Use a specific color palette and line styles
palette = sns.color_palette("husl", len(file_paths))  # Choose a visually distinct color palette
line_styles = ['-', '--', '-.']  # Different line styles

# Plot each method with specific styling (remove markers)
for (method, data), linestyle in zip(all_df.groupby('Method'), line_styles):
    sns.lineplot(
        data=data, 
        x='epoch', 
        y='Precision', 
        linestyle=linestyle, 
        label=method, 
        palette=palette,
        linewidth=2.5  # Make lines thicker for better clarity
    )

# Customize the plot
plt.xlabel('Epoch', fontsize=16, weight='bold')  # Make text bold
plt.ylabel('Average Precision Over 5 Seeds', fontsize=16, weight='bold')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Customize legend for better readability
plt.legend(
    title='Sample Selection Method', 
    loc='upper right', 
    fontsize=14,  # Increase font size
    title_fontsize=14,  # Increase title font size
    frameon=True,  # Add frame around legend
    shadow=True  # Add shadow to the legend for emphasis
)

# Add grid lines with more distinction
plt.grid(True, linestyle='--', alpha=0.7)  # Increase alpha for clearer grid

# Adjust layout to ensure everything fits well
plt.tight_layout()

# Save the plot
plt.savefig(f"{dataset_name}_precision_comparison_no_dots.png", format='png', dpi=300, bbox_inches='tight')

# Show the plot
plt.show()