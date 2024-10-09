import matplotlib.pyplot as plt


# # Data based on provided values
# data = [
#     {"correction_num": 50, "EHR_F1": 0.552, "EHR_SD": 0.126, "All_F1": 0.664, "All_SD": 0.063},    # Correction 50
#     {"correction_num": 100, "EHR_F1": 0.437, "EHR_SD": 0.106, "All_F1": 0.655, "All_SD": 0.058},
#     {"correction_num": 150, "EHR_F1": 0.515, "EHR_SD": 0.117, "All_F1": 0.665, "All_SD": 0.057},   # Correction 150,   # Correction 250
#     {"correction_num": 200, "EHR_F1": 0.597, "EHR_SD": 0.066, "All_F1": 0.686, "All_SD": 0.061} ,
#     {"correction_num": 250, "EHR_F1": 0.500, "EHR_SD": 0.092, "All_F1": 0.668, "All_SD": 0.055}# BMM Correction 50
# ]

# # Extract values
# correction_nums = [d["correction_num"] for d in data]
# ehr_f1 = [d["EHR_F1"] for d in data]
# ehr_sd = [d["EHR_SD"] for d in data]
# all_f1 = [d["All_F1"] for d in data]
# all_sd = [d["All_SD"] for d in data]

# # Plotting EHR and All dataset F1 scores
# plt.figure(figsize=(10, 6))

# # Plot EHR dataset scores
# plt.errorbar(correction_nums, ehr_f1, yerr=ehr_sd, fmt='-o', label='EHR Dataset F1 Score', color='blue', capsize=5)

# # Plot All dataset scores
# plt.errorbar(correction_nums, all_f1, yerr=all_sd, fmt='-s', label='All Dataset F1 Score', color='green', capsize=5)

# # Formatting
# plt.xlabel('T_corr')
# plt.ylabel('F1 Score')
# plt.title('EHR and All F1 Score(Sym. 50) change when increasing correction starting time')
# plt.legend()
# plt.grid(True)
# plt.xticks(correction_nums)

# plt.savefig('T_corr.png', format='png', dpi=300, bbox_inches='tight')

# # Display plot
# plt.show()



import matplotlib.pyplot as plt
import numpy as np

# Define the function for L_unce coefficient
def l_unce_coefficient(k, T, N, epsilon_maxcorr=1, T_corr=200):
    return epsilon_maxcorr / (1 + np.exp(-k * (N - T_corr - T)))

# Set up values for plotting
epochs = np.arange(200, 301)  # From epoch 200 to 300
k_values = [0.1, 0.5]  # Different steepness values
T_values = [1, 10, 100]  # Different temperature scaling values
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']  # Blue, Orange, Green, Red, Purple, Brown  # Colors for different lines
line_styles = ['-', '--']  # Different line styles for k values

# Create a plot with a white background and increased size
plt.figure(figsize=(14, 8))
plt.axhline(y=1, color='gray', linestyle='--', linewidth=1)  # Horizontal line at y=1

# Plot L_unce coefficient for each combination of k and T with different styles
for i, k in enumerate(k_values):
    for j, T in enumerate(T_values):
        l_unce_values = l_unce_coefficient(k, T, epochs)
        plt.plot(
            epochs, l_unce_values, 
            label=f"$k={k}, T={T}$", 
            color=colors[i * len(T_values) + j], 
            linestyle=line_styles[i],
            linewidth=2,
            marker='o' if j == 1 else 's',  # Different markers for different T values
            markersize=5
        )

# Set labels and title with increased font size and math text formatting
plt.xlabel("Epoch", fontsize=16)
plt.ylabel("$L_{unce}$ Coefficient Value", fontsize=16)
plt.title("$L_{unce}$ Coefficient for Different $k$ and $T$ Values", fontsize=18, fontweight='bold')

# Set grid and legend with better visibility and spacing
plt.grid(True, which='both', linestyle=':', linewidth=0.5)
plt.legend(
    fontsize=14, 
    loc='upper right', 
    bbox_to_anchor=(1.25, 0.5),  # Shift the legend to the right and center it vertically
    frameon=True, 
    shadow=True
)

# Set axis limits and tick sizes
plt.ylim(0, 1.1)
plt.xlim(200, 300)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Save the figure with tight bounding box and high resolution
plt.savefig('L_unce.png', format='png', dpi=300, bbox_inches='tight')

# Show the plot
plt.show()