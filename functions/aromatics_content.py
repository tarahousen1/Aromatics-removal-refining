import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

# Load Excel data
file_path = "/Users/tarahousen/Documents/Aromatics_removal_desulfurization_refining/excel_files/total_aromatic_content.xlsx"
df = pd.read_excel(file_path, sheet_name='Sheet 1 - wpd_datasets(14)', header=1)

# Manually define the temperature labels and corresponding column indices
pressure = ["P = 20 bar", "P = 30 bar", "P = 40 bar", "P = 50 bar"]
column_indices = [(i*2, i*2+1) for i in range(len(pressure))]

# Plotting
plt.figure(figsize=(8, 6))

for (x_col, y_col), p_label in zip(column_indices, pressure):
    x = df.iloc[2:, x_col].astype(float)
    y = df.iloc[2:, y_col].astype(float)
    p_val = p_label.split('=')[1].strip()
    plt.plot(x, y, label=p_label)

plt.xlabel("Temp")
plt.ylabel("Total aromatics content")
plt.legend(title="Pressure")
plt.grid(True)
plt.tight_layout()
plt.savefig('outputs/total_aromatics_content_post_hydrotreatment.png', dpi=300)


def interpolate_total_aromatics_content(temp, pressure_feed):
    pressure_labels = ["P = 20 bar", "P = 30 bar", "P = 40 bar", "P = 50 bar"]
    column_indices = [(i * 2, i * 2 + 1) for i in range(len(pressure_labels))]

    # Collect curves of temp vs total aromatic content for each pressure
    curves = {}
    for label, (x_col, y_col) in zip(pressure_labels, column_indices):
        x = df.iloc[2:, x_col].astype(float).dropna()
        y = df.iloc[2:, y_col].astype(float).dropna()
        # Extract numeric pressure value from label
        p_val = float(label.split('=')[1].strip().split()[0])
        curves[p_val] = (x.values, y.values)

    # If exact pressure match
    if pressure_feed in curves:
        x_vals, y_vals = curves[pressure_feed]
        return np.interp(temp, x_vals, y_vals)

    # Otherwise, interpolate between the two nearest pressure curves
    pressures = sorted(curves.keys())
    lower_p = [p for p in pressures if p < pressure_feed]
    upper_p = [p for p in pressures if p > pressure_feed]

    if lower_p and upper_p:
        p1 = max(lower_p)
        p2 = min(upper_p)

        x1, y1 = curves[p1]
        x2, y2 = curves[p2]

        # Interpolate within each curve by temp
        arom_1 = np.interp(temp, x1, y1)
        arom_2 = np.interp(temp, x2, y2)

        # Linear interpolation between the two pressures
        arom_interp = arom_1 + (arom_2 - arom_1) * (pressure_feed - p1) / (p2 - p1)
        return arom_interp

    elif lower_p:
        p = max(lower_p)
        x, y = curves[p]
        return np.interp(temp, x, y)

    elif upper_p:
        p = min(upper_p)
        x, y = curves[p]
        return np.interp(temp, x, y)

    else:
        return np.nan  # pressure out of known range
    
