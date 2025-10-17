import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the Excel file
file_path = "/Users/tarahousen/Documents/Aromatics_removal_desulfurization_refining/excel_files/molecule_type_boiling_pt.xlsx"
df = pd.read_excel(file_path, sheet_name='Sheet 1 - wpd_datasets(7)')

# Drop completely empty columns
df = df.dropna(axis=1, how='all')

# Define the number of column pairs (each component has X and Y)
num_columns = df.shape[1]
num_pairs = num_columns // 2

# Define product cut ranges
cuts = {
    "Butanes and lighter": (0, 90),
    "LSR gasoline (90-190°F)": (90, 190),
    "HSR gasoline (190-330°F)": (190, 330),
    "Kerosine (330-480°F)": (330, 480),
    "Light gas oil (480-610°F)": (480, 610),
    "Heavy gas oil (610-800°F)": (610, 800),
    "Vacuum gas oil (800-1050°F)": (800, 1050),
    "1050°F +": (1050, 1300),
}

# Store midpoint values
midpoints = {label: (start + end) / 2 for label, (start, end) in cuts.items()}

# Mapping from original column names to desired molecular type labels
desired_labels = [
    "Normal paraffins - Isoparaffins", 
    "Isoparaffins - Naphthenes", 
    "Naphthenes - Aromatics", 
    "Aromatics - Naphthenoaromatics", 
    "Naphthenoaromatics - Nitrogen, sulfur, and oxygen compounds"
]

# Create the plot
plt.figure(figsize=(12, 8))

# Dictionary to store interpolated Y-values at midpoints
interpolated_values = {cut: {} for cut in cuts}

# Plot each X/Y pair and perform interpolation
for i in range(num_pairs):
    x_col = df.columns[i * 2]
    y_col = df.columns[i * 2 + 1]
    
    # Assign label from the predefined list if available
    label = desired_labels[i] if i < len(desired_labels) else x_col.split('/')[0].strip()

    valid_rows = df[[x_col, y_col]].dropna()
    x_vals = pd.to_numeric(valid_rows[x_col], errors='coerce')
    y_vals = pd.to_numeric(valid_rows[y_col], errors='coerce')
    valid = ~(x_vals.isna() | y_vals.isna())

    x_vals = x_vals * 9 / 5 + 32

    x_vals = x_vals[valid]

    y_vals = y_vals[valid]

    # Plot the line
    plt.plot(x_vals, y_vals, label=None)  # label shown manually later

    # Interpolate Y-values at each cut midpoint
    for cut_label, midpoint in midpoints.items():
        try:
            y_interp = np.interp(midpoint, x_vals, y_vals)
            interpolated_values[cut_label][label] = y_interp
        except:
            interpolated_values[cut_label][label] = np.nan

    # Annotate line at its last valid point
    #if not x_vals.empty and not y_vals.empty:
    #    plt.text(x_vals.values[-1], y_vals.values[-1], label, fontsize=9,
    #             verticalalignment='center', horizontalalignment='left', color='black')

# Add shaded regions for product cuts
colors = plt.cm.Pastel1.colors
for i, (label, (start, end)) in enumerate(cuts.items()):
    plt.axvspan(start, end, color=colors[i % len(colors)], alpha=0.3, label=label if i == 0 else "")

# Add cut labels as annotations above plot
for i, (label, (start, end)) in enumerate(cuts.items()):
    plt.text((start + end) / 2, 102, label, rotation=90, ha='center', va='bottom', fontsize=8)

plt.xlabel("Boiling Point (°C)")
plt.ylabel("Percentage of Molecular Type")
plt.grid(True)
plt.xlim(0, 1250)
plt.ylim(0, 100)
plt.tight_layout()
plt.show()

print(interpolated_values)


# Create an empty dataframe for storing molecule percentages
molecule_percentages = pd.DataFrame(columns=['Normal paraffins', 'Isoparaffins', 'Naphthenes', 'Aromatics', 'Naphthenoaromatics', 'NSO'])

# Function to calculate molecule percentages for each cut
def calculate_molecule_percentages(interpolated_values):
    cuts = list(interpolated_values.keys())
    for cut in cuts:
        transition = interpolated_values[cut]
        
        # Initial molecule percentages
        normal_paraffins = 100 - transition['Normal paraffins - Isoparaffins']
        isoparaffins = transition['Normal paraffins - Isoparaffins'] - transition['Isoparaffins - Naphthenes']
        naphthenes = transition['Isoparaffins - Naphthenes'] - transition['Naphthenes - Aromatics']
        aromatics = transition['Naphthenes - Aromatics'] - transition['Aromatics - Naphthenoaromatics']
        naphthenoaromatics = transition['Aromatics - Naphthenoaromatics'] - transition['Naphthenoaromatics - Nitrogen, sulfur, and oxygen compounds']
        nso = transition['Naphthenoaromatics - Nitrogen, sulfur, and oxygen compounds']
        
        molecule_percentages.loc[cut] = [normal_paraffins, isoparaffins, naphthenes, aromatics, naphthenoaromatics, nso]
    
    molecule_percentages['Total'] = molecule_percentages.sum(axis=1)
    return molecule_percentages

# Calculate the molecule percentages
molecule_percentages = calculate_molecule_percentages(interpolated_values)

# Display the molecule percentages
print(molecule_percentages)