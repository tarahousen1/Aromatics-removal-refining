import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Gary and Handwerk, 2007 pg 168 Figure 7.3
file_path = "/Users/tarahousen/Documents/Aromatics_removal_desulfurization_refining/excel_files/Approx_H2_required_hydrocracking_API_vs_vol_C5_to_180.xlsx"
df = pd.read_excel(file_path, sheet_name='Sheet 1 - wpd_datasets(2)')

# Manually define the characterization factor and H2 input labels and corresponding column indices
Kw_H2_req = ["10.9 characterization factor; 500 H2 cu ft/bbl feed", "12.1 characterization factor; 500 H2 cu ft/bbl feed", "12.1 characterization factor; 1000 H2 cu ft/bbl feed", "12.1 characterization factor; 1250 H2 cu ft/bbl feed", "10.9 characterization factor; 1500 H2 cu ft/bbl feed", "12.1 characterization factor; 1500 H2 cu ft/bbl feed", "12.1 characterization factor; 1750 H2 cu ft/bbl feed", "12.1 characterization factor; 2000 H2 cu ft/bbl feed", "12.1 characterization factor; 2250 H2 cu ft/bbl feed", "10.9 characterization factor; 2500 H2 cu ft/bbl feed", "12.1 characterization factor; 2500 H2 cu ft/bbl feed", "12.1 characterization factor; 2750 H2 cu ft/bbl feed"]
column_indices = [(i*2, i*2+1) for i in range(len(Kw_H2_req))]

def interpolate_naphtha_vol(api_gravity, h2_input, Kw):
    """
    Interpolates naphtha volume based on API gravity, hydrogen input, and characterization factor (Kw).
    Falls back to interpolation between closest available Kw values if exact label is not found.
    """
    Kw_H2_req = ["10.9 characterization factor; 500 H2 cu ft/bbl feed", "12.1 characterization factor; 500 H2 cu ft/bbl feed", "12.1 characterization factor; 1000 H2 cu ft/bbl feed", "12.1 characterization factor; 1250 H2 cu ft/bbl feed", "10.9 characterization factor; 1500 H2 cu ft/bbl feed", "12.1 characterization factor; 1500 H2 cu ft/bbl feed", "12.1 characterization factor; 1750 H2 cu ft/bbl feed", "12.1 characterization factor; 2000 H2 cu ft/bbl feed", "12.1 characterization factor; 2250 H2 cu ft/bbl feed", "10.9 characterization factor; 2500 H2 cu ft/bbl feed", "12.1 characterization factor; 2500 H2 cu ft/bbl feed", "12.1 characterization factor; 2750 H2 cu ft/bbl feed"]

    Kw = float(Kw)  # Ensure it's numeric

    
    # Parse Kw_H2_req into usable tuples
    parsed_labels = []
    for label in Kw_H2_req:
        try:
            parts = label.split(" characterization factor; ")
            label_kw = float(parts[0])
            label_h2 = int(parts[1].split()[0])
            parsed_labels.append((label_kw, label_h2, label))
        except (ValueError, IndexError):
            continue

    # Filter to matching H2 rate
    matching_h2 = [tup for tup in parsed_labels if tup[1] == h2_input]

    if not matching_h2:
        raise ValueError(f"No H2 input match found for {h2_input} scf/bbl")

    # Try exact match
    exact_match = next((label for label_kw, _, label in matching_h2 if abs(label_kw - Kw) < 1e-4), None)

    if exact_match:
        idx = Kw_H2_req.index(exact_match)
        x_col, y_col = column_indices[idx]
        x = df.iloc[2:, x_col].astype(float)
        y = df.iloc[2:, y_col].astype(float)
        if not (x.min() <= api_gravity <= x.max()):
            raise ValueError(f"API gravity {api_gravity} out of range: {x.min()} - {x.max()}")
        return np.interp(api_gravity, x, y)

    # No exact match – find nearest lower and upper Kw
    lower = max([tup for tup in matching_h2 if tup[0] < Kw], default=None)
    upper = min([tup for tup in matching_h2 if tup[0] > Kw], default=None)

    if not lower and upper:
        lower = upper
    elif not upper and lower:
        upper = lower
    elif not lower and not upper:
        raise ValueError(f"No valid lower/upper Kw match for interpolation with Kw={Kw}")

    # Interpolate between lower and upper
    interpolated_results = []
    for label_kw, _, label in [lower, upper]:
        idx = Kw_H2_req.index(label)
        x_col, y_col = column_indices[idx]
        x = df.iloc[2:, x_col].astype(float)
        y = df.iloc[2:, y_col].astype(float)
        interpolated = np.interp(api_gravity, x, y)
        interpolated_results.append((label_kw, interpolated))

    # Linear interpolation in Kw dimension
    kw1, val1 = interpolated_results[0]
    kw2, val2 = interpolated_results[1]
    final_val = val1 + (Kw - kw1) * (val2 - val1) / (kw2 - kw1) if kw2 != kw1 else val1

    print(f"Warning: {Kw} characterization factor; {h2_input} H2 cu ft/bbl feed not found. Using fallback interpolation.")

    return final_val

# Plotting
plt.figure(figsize=(10, 6))

for (x_col, y_col), Kw_H2 in zip(column_indices, Kw_H2_req):
    x = df.iloc[2:, x_col].astype(float)
    y = df.iloc[2:, y_col].astype(float)
    plt.plot(x, y, label=Kw_H2)

plt.yscale('log')  # <-- Logarithmic Y-axis
plt.xlabel("API GRAVITY OF FEED")
plt.ylabel("VOL% (C5-180 F) NAPHTHA")
plt.title("Approx H2 required for hydrocracking (Gary and Handwerk, 2007)") 
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('outputs/hydrocracking_H2_requirement_vol_C5_180.png', dpi=300)
