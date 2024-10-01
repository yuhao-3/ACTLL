import numpy as np
from scipy.stats import mannwhitneyu

# Data extracted from the table for symmetric noise
# Replace these lists with the values from your table
actll = [0.7106, 0.6967, 0.6811, 0.6502, 0.4843]  # ACTLL values for symmetric noise levels
vanilla = [0.768, 0.667, 0.543, 0.398, 0.793]      # Vanilla values for symmetric noise levels
ctw = [0.819, 0.799, 0.770, 0.711, 0.611]          # CTW values for symmetric noise levels

# Perform Mann-Whitney U Test comparing ACTLL with Vanilla
u_stat_vanilla, p_value_vanilla = mannwhitneyu(actll, vanilla, alternative='two-sided')
print(f"Mann-Whitney U Test comparing ACTLL with Vanilla: U statistic = {u_stat_vanilla}, p-value = {p_value_vanilla}")

# Perform Mann-Whitney U Test comparing ACTLL with CTW
u_stat_ctw, p_value_ctw = mannwhitneyu(actll, ctw, alternative='two-sided')
print(f"Mann-Whitney U Test comparing ACTLL with CTW: U statistic = {u_stat_ctw}, p-value = {p_value_ctw}")