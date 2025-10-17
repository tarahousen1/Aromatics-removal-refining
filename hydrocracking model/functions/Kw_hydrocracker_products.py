
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Gary and Handwerk, 2007 pg 173 Figure 7.5
file_path = "/Users/tarahousen/Documents/Aromatics_removal_desulfurization_refining/excel_files/Kw_hydrocracker_products.xlsx"
df = pd.read_excel(file_path, sheet_name='Sheet 1 - wpd_datasets(5)')

# Manually define the temperature labels and corresponding column indices
Kw_values = ["Kw 10.5", "Kw 11", "Kw 11.5", "Kw 12", "Kw 12.5"]
column_indices = [(i*2, i*2+1) for i in range(len(Kw_values))]

# Plotting
plt.figure(figsize=(8, 6))

for (x_col, y_col), Kw in zip(column_indices, Kw_values):
    x = df.iloc[2:, x_col].astype(float)
    y = df.iloc[2:, y_col].astype(float)
    plt.plot(x, y, label=Kw)

plt.xlabel("F")
plt.ylabel("Kw OF PRODUCTS")
plt.title("Kw of hydrocracker products (Gary and Handwerk, 2007)") 
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('outputs/characterization_factor_hydrocracker_products.png', dpi=300)


def interpolate_Kw_product(temp, Kw_feed):
    Kw_labels = ["Kw 10.5", "Kw 11", "Kw 11.5", "Kw 12", "Kw 12.5"]
    column_indices = [(i * 2, i * 2 + 1) for i in range(len(Kw_values))]

    # Collect sulfur vs crude_sulfur curves at all temp points
    curves = {}
    for label, (x_col, y_col) in zip(Kw_labels, column_indices):
        x = df.iloc[2:, x_col].astype(float).dropna()
        y = df.iloc[2:, y_col].astype(float).dropna()
        curves[float(label.strip('Kw '))] = (x.values, y.values)

    # If exact temperature match
    if Kw_feed in curves:
        x_vals, y_vals = curves[Kw_feed]
        return np.interp(temp, x_vals, y_vals)

    # Otherwise, interpolate between the two nearest temperature curves
    Kws = sorted(curves.keys())
    lower_kw = [kw for kw in Kws if kw < Kw_feed]
    upper_kw = [kw for kw in Kws if kw > Kw_feed]

    if lower_kw and upper_kw:
        kw1 = max(lower_kw)
        kw2 = min(upper_kw)

        x1, y1 = curves[kw1]
        x2, y2 = curves[kw2]

        # Interpolate between the two curves
        kw_prod_1 = np.interp(temp, x1, y1)
        kw_prod_2 = np.interp(temp, x2, y2)

        kw_prod_interp = kw_prod_1 + (kw_prod_2 - kw_prod_1) * (Kw_feed - kw1) / (kw2 - kw1)
        return kw_prod_interp

    elif lower_kw:
        kw = max(lower_kw)
        x, y = curves[kw]
        return np.interp(temp, x, y)

    elif upper_kw:
        kw = min(upper_kw)
        x, y = curves[kw]
        return np.interp(temp, x, y)

    else:
        return np.nan  # Kw is out of known range
