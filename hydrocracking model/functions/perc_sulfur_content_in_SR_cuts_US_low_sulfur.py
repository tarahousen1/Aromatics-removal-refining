import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from xlsxwriter import Workbook

# Gary and Handwerk, 2007 pg 75 Figure 4.3A
file_path = "/Users/tarahousen/Documents/Aromatics_removal_desulfurization_refining/excel_files/perc_sulfur_content_in_SR_cuts_US_low_sulfur.xlsx"
df = pd.read_excel(file_path, sheet_name='Sheet 1 - wpd_datasets(1)')

# Manually define the temperature labels and corresponding column indices
temperatures = ["240F", "300F", "350F", "400F", "450F", "500F", "600F", "800F", "1000F"]
column_indices = [(i*2, i*2+1) for i in range(len(temperatures))]

# Plotting
plt.figure(figsize=(10, 6))

for (x_col, y_col), temp in zip(column_indices, temperatures):
    x = df.iloc[2:, x_col].astype(float)
    y = df.iloc[2:, y_col].astype(float)
    plt.plot(x, y, label=temp)

plt.yscale('log')  # <-- Logarithmic Y-axis
plt.xlabel("PERCENT  SULFUR  IN  CRUDE  OIL")
plt.ylabel("PERCENT  SULFUR  IN  STRAIGHT–RUN  PRODUCT")
plt.title("Sulfur content of products from miscellaneous U.S. crude oils (Gary and Handwerk, 2007)") 
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('outputs/sulfur_perc_in_products_distillation.png', dpi=300)
plt.show()

def interpolate_sulfur_in_product(crude_sulfur, cut_temp):
    """
    Interpolates sulfur wt% in straight-run product for a given crude sulfur wt% and cut temperature (°F),
    based on Gary & Handwerk SR product sulfur plot data.
    """

    temp_labels = ["240F", "300F", "350F", "400F", "450F", "500F", "600F", "800F", "1000F"]
    temp_values = [int(t.strip('F')) for t in temp_labels]
    column_indices = [(i * 2, i * 2 + 1) for i in range(len(temp_labels))]

    # Collect sulfur vs crude_sulfur curves at all temp points
    curves = {}
    for label, (x_col, y_col) in zip(temp_labels, column_indices):
        x = df.iloc[2:, x_col].astype(float).dropna()
        y = df.iloc[2:, y_col].astype(float).dropna()
        curves[int(label.strip('F'))] = (x.values, y.values)

    # If exact temperature match
    if cut_temp in curves:
        x_vals, y_vals = curves[cut_temp]
        return np.interp(crude_sulfur, x_vals, y_vals)

    # Otherwise, interpolate between the two nearest temperature curves
    temps = sorted(curves.keys())
    lower_temps = [t for t in temps if t < cut_temp]
    upper_temps = [t for t in temps if t > cut_temp]

    if not lower_temps or not upper_temps:
        return 0  # Outside temp range

    t1 = max(lower_temps)
    t2 = min(upper_temps)

    x1, y1 = curves[t1]
    x2, y2 = curves[t2]

    # Interpolate sulfur at crude_sulfur for each temperature
    s1 = np.interp(crude_sulfur, x1, y1)
    s2 = np.interp(crude_sulfur, x2, y2)

    # Linear interpolation in temperature
    sulfur_interp = s1 + (s2 - s1) * (cut_temp - t1) / (t2 - t1)
    return sulfur_interp
