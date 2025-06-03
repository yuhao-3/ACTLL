# import matplotlib.pyplot as plt
# import numpy as np

# # Sample x-values to represent the distribution
# x = np.linspace(0, 1, 500)

# # Define the clean and noisy distributions to match the exact original shape
# clean_dist = 2.5 * np.exp(-((x - 0.2) ** 2) / (2 * 0.05 ** 2))  # Clean distribution centered at 0.2
# noisy_dist = 1.5 * np.exp(-((x - 0.6) ** 2) / (2 * 0.1 ** 2))  # Noisy distribution centered at 0.6

# # Plot the distributions
# plt.figure(figsize=(10, 6))
# plt.plot(x, clean_dist, 'b-', linewidth=2, label="Clean Distribution")
# plt.plot(x, noisy_dist, 'r--', linewidth=2, label="Noisy Distribution")

# # Add shaded regions with updated colors
# plt.fill_between(x, 0, clean_dist, where=(x < 0.3), color='darkblue', alpha=0.5, label="Certain Set")  # Dark blue
# plt.fill_between(x, 0, noisy_dist, where=(x > 0.7), color='red', alpha=0.5, label="Hard Set")     # Purple/magenta
# plt.fill_between(x, 0, np.maximum(clean_dist, noisy_dist), where=((x >= 0.3) & (x <= 0.7)), 
#                  color='lightblue', alpha=0.5, label="Uncertain Set")  # Teal/cyan

# # Set axis limits and labels
# plt.ylim(0, 2.5)
# plt.xlabel("Loss Value", fontsize=14)
# plt.ylabel("Density", fontsize=14)
# plt.title("Loss Distribution Modelled By Beta Mixture Model", fontsize=16, weight='bold')

# # Add vertical lines and labels for mu_clean and mu_noisy below the x-axis
# plt.axvline(0.3, color='blue', linestyle='--', linewidth=1)
# plt.axvline(0.7, color='red', linestyle='--', linewidth=1)
# plt.text(0.3, -0.1, r"$\mu_{\mathrm{clean}}$", color="blue", fontsize=12, ha='center', va='top')
# plt.text(0.7, -0.1, r"$\mu_{\mathrm{noisy}}$", color="red", fontsize=12, ha='center', va='top')

# # Display the legend with larger font size
# plt.legend(loc="upper right", fontsize=13)  # Larger font for legend


# # Save the plot as an image file
# plt.savefig("Loss_Distribution_BMM_Color_Matched.png", dpi=300, bbox_inches="tight")

# # Show the plot
# plt.show()




import matplotlib.pyplot as plt
import numpy as np

# Sample x-values to represent the distribution
x = np.linspace(0, 1, 500)

# Define the clean and noisy distributions to match the exact original shape
clean_dist = 2.5 * np.exp(-((x - 0.2) ** 2) / (2 * 0.05 ** 2))  # Clean distribution centered at 0.2
noisy_dist = 1.5 * np.exp(-((x - 0.6) ** 2) / (2 * 0.1 ** 2))  # Noisy distribution centered at 0.6

# Plot the distributions
plt.figure(figsize=(10, 6))
plt.plot(x, clean_dist, 'b-', linewidth=2, label="Clean Distribution")
plt.plot(x, noisy_dist, 'r--', linewidth=2, label="Noisy Distribution")

# Add shaded regions with updated colors
plt.fill_between(x, 0, clean_dist, where=(x < 0.3), color='darkblue', alpha=0.5, label="Clean Set")  # Dark blue
plt.fill_between(x, 0, noisy_dist, where=(x > 0.7), color='red', alpha=0.5, label="Noisy Set")     # Purple/magenta
plt.fill_between(x, 0, np.maximum(clean_dist, noisy_dist), where=((x >= 0.3) & (x <= 0.7)), 
                 color='lightblue', alpha=0.5, label="Uncertain Set")  # Teal/cyan

# Set axis limits and labels
plt.ylim(0, 2.5)
plt.xlabel("Loss Value", fontsize=14)
plt.ylabel("Density", fontsize=14)
plt.title("Loss Distribution Modelled By Gaussian Mixture Model", fontsize=16, weight='bold')

# Add vertical lines and labels for mu_clean and mu_noisy below the x-axis
plt.axvline(0.3, color='blue', linestyle='--', linewidth=1)
plt.axvline(0.7, color='red', linestyle='--', linewidth=1)
plt.text(0.3, -0.1, r"$\mu_{\mathrm{clean}}$", color="blue", fontsize=12, ha='center', va='top')
plt.text(0.7, -0.1, r"$\mu_{\mathrm{noisy}}$", color="red", fontsize=12, ha='center', va='top')

# Display the legend with larger font size
plt.legend(loc="upper right", fontsize=13)  # Larger font for legend


# Save the plot as an image file
plt.savefig("Loss_Distribution_GMM_Color_Matched.png", dpi=300, bbox_inches="tight")

# Show the plot
plt.show()