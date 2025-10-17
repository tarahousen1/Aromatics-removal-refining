import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

# Gary and Handwerk, 2007 pg 173 Figure 7.5
file_path = "/Users/tarahousen/Documents/Aromatics_removal_desulfurization_refining/excel_files/hydrocracking_H2_content_hydrocarbons.xlsx"
df = pd.read_excel(file_path, sheet_name='Sheet 1 - wpd_datasets(10)')

# Manually define the temperature labels and corresponding column indices
volume_avg_boiling_pt_temps = ["100°F", "400°F", "500°F", "600°F", "700°F", "800°F", "900°F", "1000°F"]
column_indices = [(i*2, i*2+1) for i in range(len(volume_avg_boiling_pt_temps))]


# Plotting
plt.figure(figsize=(10, 6))

for (x_col, y_col), Kw in zip(column_indices, volume_avg_boiling_pt_temps):
    x = df.iloc[2:, x_col].astype(float)
    y = df.iloc[2:, y_col].astype(float)
    plt.plot(x, y, label=Kw)

plt.xlabel("CHARACTERIZATION FACTOR")
plt.ylabel("PERCENT H2 BY WEIGHT")
plt.title("Hydrogen Content of Hydrocarbons (Gary and Handwerk, 2007)") 
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('outputs/hydrocracking_H2_content_hydrocarbons.png', dpi=300)


def interpolate_H2_content(Kw_value, volume_avg_boiling_pt):
    available_temps = [float(temp.strip("°F")) for temp in volume_avg_boiling_pt_temps]

    closest_temp = min(available_temps, key=lambda t: abs(t - volume_avg_boiling_pt))
    temp_index = available_temps.index(closest_temp)
    x_col, y_col = column_indices[temp_index]

    x = df.iloc[2:, x_col].astype(float)
    y = df.iloc[2:, y_col].astype(float)

    # Drop rows with NaNs
    valid = ~(x.isna() | y.isna())
    x = x[valid]
    y = y[valid]

    if x.empty or y.empty:
        raise ValueError("Interpolation failed: x or y is empty after removing NaNs.")

    # Sort for interpolation
    if not np.all(np.diff(x) >= 0):
        sorted_indices = np.argsort(x)
        x = x.iloc[sorted_indices]
        y = y.iloc[sorted_indices]

    interp_fn = interp1d(x, y, kind='linear', fill_value='extrapolate')
    return float(interp_fn(Kw_value))
