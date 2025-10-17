import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math

# Load data
file_path = "/Users/tarahousen/Documents/Aromatics_removal_desulfurization_refining/excel_files/nitrogen distribution.xlsx"
df = pd.read_excel(file_path, sheet_name='Sheet 1 - wpd_datasets(9)')

temp_lines = ["lower temp", "higher temp"]
column_indices = [(i*2, i*2+1) for i in range(len(temp_lines))]

# Extract data
x_lower = df.iloc[2:, column_indices[0][0]].astype(float).values
y_lower = df.iloc[2:, column_indices[0][1]].astype(float).values

x_upper = df.iloc[2:, column_indices[1][0]].astype(float).values
y_upper = df.iloc[2:, column_indices[1][1]].astype(float).values

# Plotting
fig, ax1 = plt.subplots(figsize=(8, 6))
line1, = ax1.plot(x_lower, y_lower, label="lower temp")
line2, = ax1.plot(x_upper, y_upper, label="higher temp")

ax1.set_xlabel("TBP TEMP F")
ax1.set_ylabel("CUMULATIVE PERCENT OF ORIGINAL NITROGEN", color=line1.get_color())
ax1.set_title("Nitrogen distribution of crude oil fractions (Gary and Handwerk, 2007)")
ax1.grid(True)
ax1.tick_params(axis='y', labelcolor=line1.get_color())

# Right Y-axis
ax2 = ax1.twinx()
ax2.set_ylabel("CUMULATIVE PERCENT OF ORIGINAL NITROGEN", color=line2.get_color())
ax2.set_ylim(ax1.get_ylim()[0] * 100, ax1.get_ylim()[1] * 100)
ax2.tick_params(axis='y', labelcolor=line2.get_color())

# Legend and layout
plt.tight_layout()
plt.savefig('outputs/nitrogen_distributions.png', dpi=300)


def interpolate_N_content(lower_temp_value, upper_temp_value):
    # Check bounds for lower_temp_value
    if lower_temp_value < np.min(x_lower):
        lower_interp = math.nan
    else:
        lower_interp = np.interp(lower_temp_value, x_lower, y_lower)

    upper_interp = np.interp(upper_temp_value, x_upper, y_upper) * 100

    # Compute N_content only if both values are valid
    if math.isnan(lower_interp) or math.isnan(upper_interp):
        N_content = math.nan
    else:
        N_content = upper_interp - lower_interp

    return N_content