import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Gary and Handwerk, 2007 pg 172 Figure 7.4
file_path = "/Users/tarahousen/Documents/Aromatics_removal_desulfurization_refining/excel_files/yields_C5_180_and_180_400_hydrocrackates.xlsx"
df = pd.read_excel(file_path, sheet_name='Sheet 1 - wpd_datasets(4)')

# Manually define the temperature labels and corresponding column indices
Kw_values = ["Kw 12.1", "Kw 11.75", "Kw 11.30", "Kw 10.90"]
column_indices = [(i*2, i*2+1) for i in range(len(Kw_values))]

# Plotting
plt.figure(figsize=(10, 6))

for (x_col, y_col), Kw in zip(column_indices, Kw_values):
    x = df.iloc[2:, x_col].astype(float)
    y = df.iloc[2:, y_col].astype(float)
    plt.plot(x, y, label=Kw)

plt.xlabel("VOL % YIELD C5-180F NAPHTHA")
plt.ylabel("VOL % YIELD 180-400F NAPHTHA")
plt.title("Relationship between yields of C5-180°F and 180–400°F hydrocrackates (Gary and Handwerk, 2007)") 
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('outputs/hydrocracking_C5_to_180_and_180_to_400_yield.png', dpi=300)


def interpolate_C5_180(vol_C5_to_180, Kw):

    Kw_labels = ["Kw 12.1", "Kw 11.75", "Kw 11.30", "Kw 10.90"]
    column_indices = [(i * 2, i * 2 + 1) for i in range(len(Kw_values))]

    # Collect sulfur vs crude_sulfur curves at all temp points
    curves = {}
    for label, (x_col, y_col) in zip(Kw_labels, column_indices):
        x = df.iloc[2:, x_col].astype(float).dropna()
        y = df.iloc[2:, y_col].astype(float).dropna()
        curves[float(label.strip('Kw '))] = (x.values, y.values)

    # If exact temperature match
    if Kw in curves:
        x_vals, y_vals = curves[Kw]
        return np.interp(vol_C5_to_180, x_vals, y_vals)

    # Otherwise, interpolate between the two nearest temperature curves
    Kws = sorted(curves.keys())
    lower_kw = [kw for kw in Kws if kw < Kw]
    upper_kw = [kw for kw in Kws if kw > Kw]


    if lower_kw and upper_kw:
        kw1 = max(lower_kw)
        kw2 = min(upper_kw)

        x1, y1 = curves[kw1]
        x2, y2 = curves[kw2]

        # Interpolate between the two curves
        vol1 = np.interp(vol_C5_to_180, x1, y1)
        vol2 = np.interp(vol_C5_to_180, x2, y2)

        vol_C5_to_180_interp = vol1 + (vol2 - vol1) * (Kw - kw1) / (kw2 - kw1)
        return vol_C5_to_180_interp

    elif lower_kw:
        kw = max(lower_kw)
        x, y = curves[kw]
        return np.interp(vol_C5_to_180, x, y)

    elif upper_kw:
        kw = min(upper_kw)
        x, y = curves[kw]
        return np.interp(vol_C5_to_180, x, y)

    else:
        return np.nan  # Kw is out of known range
