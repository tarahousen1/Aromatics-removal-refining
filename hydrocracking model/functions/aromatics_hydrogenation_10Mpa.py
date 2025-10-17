
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Gary and Handwerk, 2007 Figure 9.2
file_path = "/Users/tarahousen/Documents/Aromatics_removal_desulfurization_refining/excel_files/aromatics_hydrogenation_10MPa.xlsx"
df = pd.read_excel(file_path, sheet_name='Sheet 1 - wpd_datasets(11)')

# Manually define the characterization factor and H2 input labels and corresponding column indices
aromatics_hydrogenation = ["LHSV 1.5", "LHSV 1.0", "LHSV 0.5"]
column_indices = [(i*2, i*2+1) for i in range(len(aromatics_hydrogenation))]

# Interpolation function
def interpolate_perc_aromatics_hydrogenation(temp, LHSV):
    if LHSV not in aromatics_hydrogenation:
        raise ValueError(f"LHSV must be one of {aromatics_hydrogenation}")
    
    idx = aromatics_hydrogenation.index(LHSV)
    temp_col, perc_col = column_indices[idx]
    
    temp_values = pd.to_numeric(df.iloc[:, temp_col], errors='coerce').dropna().values
    perc_values = pd.to_numeric(df.iloc[:, perc_col], errors='coerce').dropna().values

    if not (min(temp_values) <= temp <= max(temp_values)):
        raise ValueError(f"Temperature {temp}°C is out of interpolation range for {LHSV}")

    interpolated_value = np.interp(temp, temp_values, perc_values)
    return interpolated_value

result = interpolate_perc_aromatics_hydrogenation(temp=360, LHSV="LHSV 1.0")
print(f"Interpolated % aromatics at 360°C and LHSV 1.0: {result:.2f}")