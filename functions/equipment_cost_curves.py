
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import os

# Convert from M USD to USD and CEPCI index
scaling_factor = 1000000 * 816.0 / 468.2

# Helper function for log-log cubic interpolation
def create_log_log_interp(x_vals, y_vals, kind, tol=1):
    log_x = np.log10(x_vals)
    log_y = np.log10(y_vals)
    log_interp_func = interp1d(log_x, log_y, kind=kind, fill_value='extrapolate')
    
    def interpolator(x):
        x = np.asarray(x)  # allow vector input
        min_x, max_x = np.min(x_vals), np.max(x_vals)
        lower_bound = (1 - tol) * min_x
        upper_bound = (1 + tol) * max_x
        
        if np.any((x < lower_bound) | (x > upper_bound)):
            raise ValueError(
                f"x={x} is outside allowed range "
                f"[{lower_bound:.4g}, {upper_bound:.4g}]"
            )
        
        return 10 ** log_interp_func(np.log10(x))
    
    return interpolator

def create_log_x_linear_y_interp(x_vals, y_vals):
    log_x = np.log10(x_vals)
    interp_func = interp1d(log_x, y_vals, kind='cubic', fill_value='extrapolate')
    
    def interpolator(x):
        return interp_func(np.log10(x))
    
    return interpolator

# ===============================
# 1. Atmospheric Distillation Unit
# ===============================
atmospheric_path = "/Users/tarahousen/Documents/Aromatics_removal_desulfurization_refining/excel_files/atmospheric_distillation_unit_cost_data.xlsx"
atmospheric_df = pd.read_excel(atmospheric_path, header=2)
atmospheric_df.columns = atmospheric_df.columns.str.strip()
atmospheric_x = atmospheric_df['X'].values
atmospheric_y = atmospheric_df['Y'].values
atmospheric_distillation_unit_cost_interp = create_log_log_interp(atmospheric_x, atmospheric_y, kind='linear')

# ==========================
# 2. Vacuum Distillation Unit
# ==========================
vacuum_path = "/Users/tarahousen/Documents/Aromatics_removal_desulfurization_refining/excel_files/vacuum_distillation_unit_cost_data.xlsx"
vacuum_df = pd.read_excel(vacuum_path, header=2)
vacuum_df.columns = vacuum_df.columns.str.strip()
vacuum_x = vacuum_df['X'].values
vacuum_y = vacuum_df['Y'].values
vacuum_distillation_unit_cost_interp = create_log_log_interp(vacuum_x, vacuum_y, kind='linear')

# ===============
# 3. Hydrotreator
# ===============
hydrotreator_path = "/Users/tarahousen/Documents/Aromatics_removal_desulfurization_refining/excel_files/hydrotreatment_cost_data.xlsx"
hydrotreator_df = pd.read_excel(hydrotreator_path, header=2)
hydrotreator_df.columns = hydrotreator_df.columns.str.strip()

# Split into lower and upper bound data
hydrotreator_x_lower = hydrotreator_df['X_lower'].dropna().values
hydrotreator_y_lower = hydrotreator_df['Y_lower'].dropna().values

hydrotreator_x_upper = hydrotreator_df['X_upper'].dropna().values
hydrotreator_y_upper = hydrotreator_df['Y_upper'].dropna().values

# Create two interpolation functions (log-log)
hydrotreator_cost_interp_lower = create_log_log_interp(hydrotreator_x_lower,
                                                      hydrotreator_y_lower,
                                                      kind='linear')

hydrotreator_cost_interp_upper = create_log_log_interp(hydrotreator_x_upper,
                                                      hydrotreator_y_upper,
                                                      kind='linear')

# =========================
# 4. Amine Gas Treating Unit
# =========================
amine_path = "/Users/tarahousen/Documents/Aromatics_removal_desulfurization_refining/excel_files/amine_gas_treating_unit_cost_data.xlsx"
amine_df = pd.read_excel(amine_path, header=2)
amine_df.columns = amine_df.columns.str.strip()
amine_x = amine_df['X'].values
amine_y = amine_df['Y'].values
amine_gas_treating_unit_cost_interp = create_log_log_interp(amine_x, amine_y,  kind='linear')

# ================
# 5. Claus Unit
# ================
claus_path = "/Users/tarahousen/Documents/Aromatics_removal_desulfurization_refining/excel_files/claus_unit_equipment_cost_data.xlsx"
claus_df = pd.read_excel(claus_path, header=2)
claus_df.columns = claus_df.columns.str.strip()
claus_x = claus_df['X'].values
claus_y = claus_df['Y'].values
claus_unit_cost_interp = create_log_log_interp(claus_x, claus_y, kind='linear')

# ================
# 6. SMR 
# ================
SMR_path = "/Users/tarahousen/Documents/Aromatics_removal_desulfurization_refining/excel_files/SMR_equip_cost_data.xlsx"
SMR_df = pd.read_excel(SMR_path, header=2)
SMR_df.columns = SMR_df.columns.str.strip()
SMR_x = SMR_df['X'].values
SMR_y = SMR_df['Y'].values
SMR_cost_interp = create_log_log_interp(SMR_x, SMR_y,  kind='linear')

# ================
# 7. Electrolyzer 
# ================
electrolyzer_path = "/Users/tarahousen/Documents/Aromatics_removal_desulfurization_refining/excel_files/PEM_electrolyzer_cost_data.xlsx"
electrolyzer_df = pd.read_excel(electrolyzer_path, header=2)
electrolyzer_df.columns = electrolyzer_df.columns.str.strip()
electrolyzer_x = electrolyzer_df['X'].values
electrolyzer_y = electrolyzer_df['Y'].values
electrolyzer_cost_interp = create_log_x_linear_y_interp(electrolyzer_x, electrolyzer_y)

# ================
# 8. Packed column 
# ================
packed_column_path = "/Users/tarahousen/Documents/Aromatics_removal_desulfurization_refining/excel_files/packed_column_cost_data.xlsx"
packed_column_df = pd.read_excel(packed_column_path, header=2)
packed_column_df.columns = packed_column_df.columns.str.strip()
packed_column_x = packed_column_df['X'].values
packed_column_y = packed_column_df['Y'].values
packed_column_cost_interp = create_log_log_interp(packed_column_x, packed_column_y, kind='linear')


# ================
# 9. Distillation column 
# ================
column_path = "/Users/tarahousen/Documents/Aromatics_removal_desulfurization_refining/excel_files/distillation_column_cost_data.xlsx"

# Skip the first row (0-indexed), use second row as header
column_df = pd.read_excel(column_path, header=[1,2])

# Flatten MultiIndex
column_df.columns = ["_".join([str(c) for c in col if str(c) != 'nan']).strip() for col in column_df.columns]

# Prepare dictionary for column data
column_data = {}
diameters = []

# Loop through column pairs (X/Y)
for i in range(0, len(column_df.columns), 2):
    col_x = column_df.columns[i]
    col_y = column_df.columns[i+1]

    # Extract diameter from the column name
    try:
        diam_str = col_x.split('D =')[1].split('m')[0].strip()
        diam = float(diam_str)
        diameters.append(diam)
        
        x_values = column_df[col_x].dropna().values
        y_values = column_df[col_y].dropna().values
        column_data[diam] = {'X': x_values, 'Y': y_values}
    except Exception as e:
        print(f"Skipping columns {col_x}, {col_y}: {e}")

diameters = sorted(diameters)

# Interpolation function
def distillation_column_cost_interp(diameter, height):
    if diameter <= diameters[0]:
        d_low, d_high = diameters[0], diameters[1]
    elif diameter >= diameters[-1]:
        d_low, d_high = diameters[-2], diameters[-1]
    else:
        d_low = max(d for d in diameters if d <= diameter)
        d_high = min(d for d in diameters if d >= diameter)

    f_low = interp1d(column_data[d_low]['X'], column_data[d_low]['Y'], kind='linear', fill_value='extrapolate')
    f_high = interp1d(column_data[d_high]['X'], column_data[d_high]['Y'], kind='linear', fill_value='extrapolate')

    y_low = f_low(height)
    y_high = f_high(height)

    cost = y_low + (y_high - y_low) * (diameter - d_low) / (d_high - d_low)
    return float(cost)

# ================
# 9. Trays 
# ================
tray_path = "/Users/tarahousen/Documents/Aromatics_removal_desulfurization_refining/excel_files/tray_cost_data.xlsx"

# Read with first row as header (skip "X Y" row later)
tray_df = pd.read_excel(tray_path, header=0)

def tray_cost_interp(tray_type):
    if tray_type == 'valve':
        tray_column_x = tray_df['Valve trays X'].dropna().values
        tray_column_y = tray_df['Valve trays Y'].dropna().values
    if tray_type == 'sieve':
        tray_column_x = tray_df['Sieve trays X'].dropna().values
        tray_column_y = tray_df['Sieve trays Y'].dropna().values

    interp_func = create_log_log_interp(tray_column_x, tray_column_y, kind='linear')
    
    return interp_func

valve_x = pd.to_numeric(tray_df['Valve trays X'].dropna(), errors='coerce').values
sieve_x = pd.to_numeric(tray_df['Sieve trays X'].dropna(), errors='coerce').values

valve_tray_cost_interp = tray_cost_interp('valve')
sieve_tray_cost_interp = tray_cost_interp('sieve')


# ================
# 10. Heat exchanger 
# ================
heat_exchanger_path = "/Users/tarahousen/Documents/Aromatics_removal_desulfurization_refining/excel_files/heat_exchanger_cost_data.xlsx"

# Read with first row as header (skip "X Y" row later)
heat_exchanger_df = pd.read_excel(heat_exchanger_path, header=2)
heat_exchanger_df.columns = heat_exchanger_df.columns.str.strip()
heat_exchanger_x = heat_exchanger_df['X'].values
heat_exchanger_y = heat_exchanger_df['Y'].values
heat_exchanger_cost_interp = create_log_log_interp(heat_exchanger_x, heat_exchanger_y, kind='linear')


# Create output folder
output_folder = "equipment_cost_plots"
os.makedirs(output_folder, exist_ok=True)

# Equipment list with (name, interpolator, x_values)
equipment_list = [
    ("Atmospheric Distillation Unit", atmospheric_distillation_unit_cost_interp, atmospheric_x),
    ("Vacuum Distillation Unit", vacuum_distillation_unit_cost_interp, vacuum_x),
    ("Hydrotreator lower bound", hydrotreator_cost_interp_lower, hydrotreator_x_lower),
    ("Hydrotreator upper bound", hydrotreator_cost_interp_upper, hydrotreator_x_upper),
    ("Amine Gas Treating Unit", amine_gas_treating_unit_cost_interp, amine_x),
    ("Claus Unit", claus_unit_cost_interp, claus_x),
    ("SMR", SMR_cost_interp, SMR_x),
    ("Electrolyzer", electrolyzer_cost_interp, electrolyzer_x),
    ("Packed Column", packed_column_cost_interp, packed_column_x),
    ("Valve Trays", tray_cost_interp('valve'), valve_x),
    ("Sieve Trays", tray_cost_interp('sieve'), sieve_x),
    ("Heat Exchanger", heat_exchanger_cost_interp, heat_exchanger_x)
]

# Generate individual plots
for name, interp_func, x_vals in equipment_list:
    x_plot = np.logspace(np.log10(np.min(x_vals)), np.log10(np.max(x_vals)), 200)
    y_plot = interp_func(x_plot)

    plt.figure(figsize=(8, 6))
    plt.plot(x_plot, y_plot, 'b-', linewidth=2)
    plt.scatter(x_vals, interp_func(x_vals), color='red', label='Data points')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Capacity (units consistent with Excel data)')
    plt.ylabel('Equipment Cost (USD)')
    plt.title(f'{name} Cost Scaling')
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()

    # Save figure
    file_name = os.path.join(output_folder, f"{name.replace(' ', '_').lower()}.png")
    plt.tight_layout()
    plt.savefig(file_name, dpi=300)
    plt.close()