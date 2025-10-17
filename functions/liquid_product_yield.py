
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

# Load data
file_path = "/Users/tarahousen/Documents/Aromatics_removal_desulfurization_refining/excel_files/hydrotreatment_liquid_product_yield.xlsx"
df = pd.read_excel(file_path, sheet_name='Sheet 1 - wpd_datasets(13)')


# Define LHSV and Pressure conditions as per their column positions
conditions = [
    (1, 50),
    (1, 20),
    (2, 50),
    (2, 20),
    (3, 50),
    (3, 20),
]

curves = {}
for i, (lhs, pres) in enumerate(conditions):
    x_col = i * 2
    y_col = x_col + 1
    # Skip rows with non-numeric entries (e.g., headers like "LHSV = 1")
    x = pd.to_numeric(df.iloc[:, x_col], errors="coerce").dropna().values
    y = pd.to_numeric(df.iloc[:, y_col], errors="coerce").dropna().values
    curves[(lhs, pres)] = (x, y)

def interpolate_liquid_yield(temp, LHSV_input, Pressure_input):
    available_LHSVs = sorted(set(lhs for lhs, _ in curves))
    available_pressures = sorted(set(p for _, p in curves))

    def nearest_two(values, target):
        below = [v for v in values if v <= target]
        above = [v for v in values if v >= target]
        if below and above:
            return max(below), min(above)
        elif below:
            return max(below), max(below)
        elif above:
            return min(above), min(above)
        else:
            return None, None

    lhs1, lhs2 = nearest_two(available_LHSVs, LHSV_input)
    p1, p2 = nearest_two(available_pressures, Pressure_input)

    def get_yield(lhs, pres):
        if (lhs, pres) in curves:
            x_vals, y_vals = curves[(lhs, pres)]
            return np.interp(temp, x_vals, y_vals)
        return np.nan

    y11 = get_yield(lhs1, p1)
    y12 = get_yield(lhs1, p2)
    y21 = get_yield(lhs2, p1)
    y22 = get_yield(lhs2, p2)

    if p1 != p2:
        y1 = y11 + (y12 - y11) * (Pressure_input - p1) / (p2 - p1)
        y2 = y21 + (y22 - y21) * (Pressure_input - p1) / (p2 - p1)
    else:
        y1 = y11
        y2 = y21

    if lhs1 != lhs2:
        final_yield = y1 + (y2 - y1) * (LHSV_input - lhs1) / (lhs2 - lhs1)
    else:
        final_yield = y1

    print(f"Estimated yield at Temp={temp}°C, LHSV={LHSV_input}, Pressure={Pressure_input} bar: {final_yield:.2f}%")
    return final_yield
