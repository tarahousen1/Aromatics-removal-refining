import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import toml
from functions.functions import find_lbhr_conversion
from functions.functions import compute_hydrogen_consumption
from functions.functions import product_height_diameter, tray_stack_height
from functions.experimental_hydrotreatment_kerosene_data import interpolate_aromatics_saturation_efficiency, interpolate_sulfur_removal_perc
from functions.liquid_product_yield import interpolate_liquid_yield
import os
from functions.equipment_cost_curves import atmospheric_distillation_unit_cost_interp, vacuum_distillation_unit_cost_interp, hydrotreator_cost_interp_lower, hydrotreator_cost_interp_upper, amine_gas_treating_unit_cost_interp, claus_unit_cost_interp, SMR_cost_interp, electrolyzer_cost_interp, packed_column_cost_interp, valve_tray_cost_interp, sieve_tray_cost_interp
from functions.dcfror import discounted_cash_flow
from functions.cost_functions import calculate_additional_capital_costs, calculate_additional_operating_costs
from functions.CEPCI_index import get_cepci_value, cepci_values
import math
from functions.LLE_calculate_num_stages import interpolate_num_stages, interpolate_SF_ratio

# Create output directory if it doesn't exist
output_dir = "/Users/tarahousen/Documents/Aromatics_removal_desulfurization_refining/outputs"
os.makedirs(output_dir, exist_ok=True)

# Load density conversion table
file_path = "/Users/tarahousen/Documents/Aromatics_removal_desulfurization_refining/excel_files/density_conv_table.xlsx"
density_conv_table = pd.read_excel(file_path, sheet_name='Sheet1')

# Load data from TOML files
user_inputs = toml.load('input_text_files/user_inputs.toml')
input_price_data = toml.load('input_text_files/input_price_data.toml')
lca_data = toml.load('input_text_files/lca_data.toml')
financial_data = toml.load('input_text_files/financial_assumptions.toml')
crude_selection = user_inputs['Refinery']['crude_oil']
crude_oil_BPCD = user_inputs['Refinery']['input_bpcd']

hydrogen_source = user_inputs['Refinery']['hydrogen_source']
refinery_type = user_inputs['Refinery']['refinery_type']

refinery_utility_inputs = toml.load('input_text_files/refinery_utility_inputs.toml')
material_data = toml.load('input_text_files/construction_materials.toml')
atmospheric_distillation_utility_inputs = refinery_utility_inputs['atmospheric_distillation_utility_data']
vacuum_distillation_utility_inputs = refinery_utility_inputs['vacuum_distillation_utility_data']
desalter_utility_inputs = refinery_utility_inputs['desalter_utility_data']
hydrotreatment_utility_data = refinery_utility_inputs['hydrotreatment_utility_data']
amine_gas_treating_utility_data = refinery_utility_inputs['amine_gas_treating_utility_data']
claus_process_utility_data = refinery_utility_inputs['claus_sulfur_recovery_utility_data']
SCOT_utility_data = refinery_utility_inputs['SCOT_utility_data']
SMR_utility_data = refinery_utility_inputs['SMR_utility_data']
electrolyzer_utility_data = refinery_utility_inputs['electrolyzer_utility_data']
sulfolane_utility_data = refinery_utility_inputs['sulfolane_utility_data']

fixed_parameters = toml.load('input_text_files/fixed_parameters.toml')
conversion_parameters = fixed_parameters['conversion_parameters']
solvent_extraction_data = toml.load('input_text_files/solvent_extraction_data.toml')

aromatics_removal_technique = user_inputs['Refinery']['aromatics_removal_technique']
solvent_choice = user_inputs['solvent_extraction_parameters']['solvent_choice']
electricity_choice = user_inputs['Refinery']['electricity_choice']

hydrotreatment_operating_parameters = user_inputs['hydrotreatment_operating_parameters']

# Replace this with your actual Excel file path
file_path = f'/Users/tarahousen/Documents/Aromatics_removal_desulfurization_refining/BP crude assays/{crude_selection}.xlsx'

df = pd.read_excel(file_path, header=None)

def round_dataframe(df):
    df_rounded = df.copy()
    for col in df_rounded.select_dtypes(include='number').columns:
        if col == 'Specific Gravity':
            df_rounded[col] = df_rounded[col].round(4)
        else:
            df_rounded[col] = df_rounded[col].round(2)
    return df_rounded

output_file = os.path.join(output_dir, f"refinery_outputs_{aromatics_removal_technique}_{crude_selection}_{crude_oil_BPCD}.xlsx")

# Find data rows in crude assay
volume_row = df[df.iloc[:, 1] == "Yield on crude (% vol)"].index[0]
density_row = df[df.iloc[:, 1] == "Density at 15°C (kg/litre)"].index[0]
boiling_row = df[df.iloc[:, 1] == "End (°C API) "].index[0]

vol_percent = df.iloc[volume_row, 3:12].astype(float).values
density = df.iloc[density_row, 3:12].astype(float).values
boiling_labels = df.iloc[boiling_row, 3:12].values
cum_vol_percent = np.cumsum(vol_percent)

density_water = 0.999 # kg/L
specific_gravity = density/density_water
API = 141.5/specific_gravity - 131.5

fig, ax1 = plt.subplots(figsize=(10, 6))

ax1.plot(cum_vol_percent, boiling_labels, 'b-o', label='Boiling Point (°C)')
ax1.set_xlabel('Cumulative Volume % on Crude')
ax1.set_ylabel('Boiling Point (°C)', color='b')
ax1.tick_params(axis='y', labelcolor='b')

ax2 = ax1.twinx()
ax2.plot(cum_vol_percent, API, 'r--s', label='API Gravity')
ax2.set_ylabel('API Gravity (°API)', color='r')
ax2.tick_params(axis='y', labelcolor='r')

plt.title('TBP Curve with API Gravity vs Cumulative Volume %')
plt.grid(True)
fig.tight_layout()

distillation_inputs = {
    'Volume %': {},
    'API': {},
    'Specific Gravity': {},
    'Characterization Factor': {},
    'MW': {},
    'BPCD': {},
    'lb/hr from bbl/day': {},
    'lb/hr': {},
    'wt% S': {},
    'lb/hr S': {},
    'wt %': {},
    'wppm N': {},
    'lb/hr N': {},
}

distillation_outputs = {
    'Volume %': {},
    'API': {},
    'Specific Gravity': {},
    'Characterization Factor': {},
    'MW': {},
    'BPCD': {},
    'lb/hr from bbl/day': {},
    'lb/hr': {},
    'MJ/yr': {},
    'wt% S': {},
    'lb/hr S': {},
    'wt %': {},
    'wt% H': {},
    'lb/hr H': {},
    'wppm N': {},
    'lb/hr N': {},
    'vol% paraffins': {},
    'vol% naphthenes': {},
    'vol% aromatics': {},
    'wt% paraffins': {},
    'wt% naphthenes': {},
    'wt% aromatics': {},
    'wt% naphthalenes': {},
}

distillation_inputs['API']['Crude oil'] = df.iloc[8, 14]
distillation_inputs['wt% S']['Crude oil'] = df.iloc[9, 14]
distillation_inputs['BPCD']['Crude oil'] = crude_oil_BPCD
distillation_inputs['wppm N']['Crude oil'] = df.iloc[40, 2]

distillation_inputs['Specific Gravity']['Crude oil'] =  141.5 / (distillation_inputs['API']['Crude oil'] + 131.5) 

distillation_inputs['lb/hr from bbl/day']['Crude oil'] = find_lbhr_conversion(distillation_inputs['Specific Gravity']['Crude oil'], density_conv_table)             # Obtain BPCD to lb/hr conversion from density conversion table

distillation_inputs['lb/hr']['Crude oil'] = distillation_inputs['BPCD']['Crude oil'] * distillation_inputs['lb/hr from bbl/day']['Crude oil'] 
distillation_inputs['lb/hr S']['Crude oil'] = distillation_inputs['lb/hr']['Crude oil'] * distillation_inputs['wt% S']['Crude oil']/100
distillation_inputs['lb/hr N']['Crude oil'] = distillation_inputs['lb/hr']['Crude oil'] * distillation_inputs['wppm N']['Crude oil']/1000000

start_bp_row = df[df.iloc[:, 1] == "Start (°C API)"].index[0]
end_bp_row = df[df.iloc[:, 1] == "End (°C API) "].index[0]
vol_percent_row = df[df.iloc[:, 1] == "Yield on crude (% vol)"].index[0]
wt_percent_row = df[df.iloc[:, 1] == "Yield on crude (% wt)"].index[0]
density_row = df[df.iloc[:, 1] == "Density at 15°C (kg/litre)"].index[0]
sulfur_wt_row = df[df.iloc[:, 1] == "Total Sulphur (% wt)"].index[0]
nitrogen_wppm_row = df[df.iloc[:, 1] == "Total Nitrogen (ppm wt)"].index[0]
paraffins_wt_row = df[df.iloc[:, 1] == "Paraffins (%wt)"].index[0]
naphthenes_wt_row = df[df.iloc[:, 1] == "Naphthenes (%wt)"].index[0]
aromatics_wt_row = df[df.iloc[:, 1] == "Aromatics (%wt)"].index[0]
aromatics_vol_row = df[df.iloc[:, 1] == "Aromatics (%vol)"].index[0]
naphthalenes_wt_row = df[df.iloc[:, 1] == "Naphthalenes (%wt)"].index[0]

# Determine product cuts
if refinery_type == 'hydroskimming':
    raw_cut_names = pd.concat([
        df.iloc[27, 3:9],
        df.iloc[27, 12:13]
    ], axis=0).astype(str).replace(['nan', 'NaN', 'None'], '', regex=True)
    cols = list(range(3,9)) + [12]
elif refinery_type == 'cracking':
    raw_cut_names = pd.concat([
    df.iloc[27, 3:12],
    df.iloc[27, 15:16]
    ], axis=0).astype(str).replace(['nan', 'NaN', 'None'], '', regex=True)
    cols = list(range(3,12)) + [15]
else:
    print('Refinery type not specified')

cut_names = []
last_name = ""
for name in raw_cut_names:
    if name.strip() == '':
        if last_name:
            cut_names.append('... ' + last_name)
        else:
            cut_names.append("Unknown")  # fallback if first is empty
    else:
        cut_names.append(name)
        last_name = name


start_bp = (
    df.iloc[start_bp_row, cols]
    .astype(str)
    .str.strip()
    .replace('C5', 50)
    .astype(float)
    .values
)

end_bp = (
    df.iloc[end_bp_row, cols]
    .astype(str)
    .str.strip()
    .replace({'FBP': '1050', '-': np.nan})
    .astype(float)
    .values
)

vol_percent = df.iloc[vol_percent_row, cols].astype(float).values
wt_percent = df.iloc[wt_percent_row, cols].astype(float).values
density = df.iloc[density_row, cols].astype(float).values

sulfur_wt = (
    df.iloc[sulfur_wt_row, cols]
    .replace('-', np.nan)
    .astype(float)
    .values
)

nitrogen_wppm = (
    df.iloc[nitrogen_wppm_row, cols]
    .replace('-', np.nan)
    .astype(float)
    .values
)

paraffins_wt = (
    df.iloc[paraffins_wt_row, cols]
    .replace('-', np.nan)
    .astype(float)
    .values
)

naphthenes_wt = (
    df.iloc[naphthenes_wt_row, cols]
    .replace('-', np.nan)
    .astype(float)
    .values
)

aromatics_wt = (
    df.iloc[aromatics_wt_row, cols]
    .replace('-', np.nan)
    .astype(float)
    .values
)

aromatics_vol = (
    df.iloc[aromatics_vol_row, cols]
    .replace('-', np.nan)
    .astype(float)
    .values
)

naphthalenes_wt = (
    df.iloc[naphthalenes_wt_row, cols]
    .replace('-', np.nan)
    .astype(float)
    .values
)

# Calculate specific gravity
specific_gravity = density/density_water

# Calculate API gravity
api_gravity = 141.5 / specific_gravity - 131.5

for i, cut in enumerate(cut_names):
    if cut == '':
        continue  # Skip empty cut names

    distillation_outputs['wt %'][cut] = wt_percent[i]
    distillation_outputs['Volume %'][cut] = vol_percent[i]
    distillation_outputs['wt% S'][cut] = sulfur_wt[i]
    distillation_outputs['wppm N'][cut] = nitrogen_wppm[i]
    distillation_outputs['wt% paraffins'][cut] = paraffins_wt[i]
    distillation_outputs['wt% naphthenes'][cut] = naphthenes_wt[i]
    distillation_outputs['wt% aromatics'][cut] = aromatics_wt[i]
    distillation_outputs['vol% aromatics'][cut] = aromatics_vol[i]
    distillation_outputs['wt% naphthalenes'][cut] = naphthalenes_wt[i]

    # Mass & volumetric flow rates – these are safe since we have wt%
    distillation_outputs['BPCD'][cut] = vol_percent[i]/100 * distillation_inputs['BPCD']['Crude oil']
    distillation_outputs['lb/hr'][cut] = wt_percent[i]/100 * distillation_inputs['lb/hr']['Crude oil']
    distillation_outputs['lb/hr S'][cut] = sulfur_wt[i]/100 * distillation_outputs['lb/hr'][cut]
    distillation_outputs['lb/hr N'][cut] = nitrogen_wppm[i]/1e6 * distillation_outputs['lb/hr'][cut]

    if np.isnan(start_bp[i]) or np.isnan(density[i]):
        # Don’t try to calculate API, SG, CF, H%, MW
        distillation_outputs['API'][cut] = None
        distillation_outputs['Specific Gravity'][cut] = None
        distillation_outputs['Characterization Factor'][cut] = None
        distillation_outputs['wt% H'][cut] = None
        distillation_outputs['lb/hr H'][cut] = None
        distillation_outputs['MW'][cut] = None
        continue

    # --- normal TBP cuts with BP & density ---
    specific_gravity = density[i] / density_water
    api_gravity = 141.5 / specific_gravity - 131.5
    distillation_outputs['Specific Gravity'][cut] = specific_gravity
    distillation_outputs['API'][cut] = api_gravity

    T_mean_avg_bp_C = (start_bp[i] + end_bp[i]) / 2
    T_mean_avg_bp_R = T_mean_avg_bp_C * 9/5 + 491.67

    distillation_outputs['Characterization Factor'][cut] = (
        T_mean_avg_bp_R ** (1/3) / specific_gravity
    )

    # Hydrogen balance
    if not np.isnan(paraffins_wt[i]):
        hydrogen_wt = (
            paraffins_wt[i] * 15.4 +
            naphthenes_wt[i] * 13.6 +
            aromatics_wt[i] * 8.5
        ) / 100
    else:
        T_mean_avg_bp_R_kerosene = T_mean_avg_bp_R
        # ASTM D-3343
        hydrogen_wt = (
        (5.2407 + 0.01448 * T_mean_avg_bp_C - 7.018 * aromatics_vol[i]/100)
        / (specific_gravity - 0.901 * aromatics_vol[i]/100)
        + 0.01298 * aromatics_vol[i]/100 * T_mean_avg_bp_C
        - 0.01345 * T_mean_avg_bp_C
        + 5.6879
        )
    distillation_outputs['wt% H'][cut] = hydrogen_wt
    distillation_outputs['lb/hr H'][cut] = (
        hydrogen_wt/100 * distillation_outputs['lb/hr'][cut]
        if hydrogen_wt is not None else None
    )

    # MW correlation
    distillation_outputs['MW'][cut] = (
        0.000045673 * T_mean_avg_bp_R**2.1962 *
        specific_gravity**(-1.0164)
    )

light_hc_keys = ["Methane", "Ethane", "Propane", "Isobutane", "n-Butane"]
light_hc_lbhr_from_bpcd = {"Methane": 6.194, "Ethane": 7.976, "Propane": 7.417, "Isobutane": 8.223, "n-Butane": 8.528}
light_hc_hwt = {"Methane": 25.0, "Ethane": 18.3, "Propane": 15.4, "Isobutane": 14.7, "n-Butane": 14.7}
light_hc_specific_energy = {"Methane": fixed_parameters['specific_energy']['CH4'], "Ethane": fixed_parameters['specific_energy']['C2H6'], "Propane": fixed_parameters['specific_energy']['C3H8'], "Isobutane": fixed_parameters['specific_energy']['butane'], "n-Butane": fixed_parameters['specific_energy']['butane']}

valid_light_hc_keys = []
for key in light_hc_keys:
    row = df.index[df.iloc[:, 6].astype(str).str.strip() == key]
    if not row.empty:
        light_hc_wt_perc = df.iloc[row[0], 9]
        try:
            light_hc_wt_perc = float(light_hc_wt_perc)
            if light_hc_wt_perc != 0:   # only keep if non-zero
                distillation_outputs['wt %'][key] = light_hc_wt_perc
                lb_hr = (light_hc_wt_perc /100) * distillation_inputs['lb/hr']['Crude oil']
                distillation_outputs['lb/hr'][key] = lb_hr
                lb_hr_from_bbl_day = light_hc_lbhr_from_bpcd[key]
                distillation_outputs['lb/hr from bbl/day'][key] = lb_hr_from_bbl_day
                BPCD = float (lb_hr / lb_hr_from_bbl_day)
                distillation_outputs['BPCD'][key] = BPCD 
                valid_light_hc_keys.append(key)
                distillation_outputs['wt% H'][key] = light_hc_hwt[key]
                distillation_outputs['lb/hr H'][key] = light_hc_hwt[key] / 100 * lb_hr
                distillation_outputs['MJ/yr'][key] = lb_hr * conversion_parameters['lb_to_kg'] * 8760 * light_hc_specific_energy[key]

        except:
            pass

# Normalize mass flow rate of light hydrocarbons to acheive mass balance
# Step 1: collect lb/hr of distillate cuts
distillate_cuts = [cut for cut in distillation_outputs['lb/hr'] if cut not in light_hc_keys]
distillate_cuts_lb_hr_total = sum(distillation_outputs['lb/hr'][cut] for cut in distillate_cuts)

# Step 2: calculate remaining lb/hr for light HC
remaining_lb_hr = distillation_inputs['lb/hr']['Crude oil'] - distillate_cuts_lb_hr_total

if remaining_lb_hr <= 0:
    raise ValueError("TBP cuts already exceed total crude lb/hr, cannot assign to light HC")

# Step 3: get current sum of light HC lb/hr
light_hc_lb_hr_sum = sum(distillation_outputs['lb/hr'][cut] for cut in light_hc_keys if cut in distillation_outputs['lb/hr'])

# Step 4: calculate normalization factor for light HC only
if light_hc_lb_hr_sum > 0:
    factor = remaining_lb_hr / light_hc_lb_hr_sum

    for cut in light_hc_keys:
        if cut in distillation_outputs['lb/hr']:
            # scale lb/hr
            distillation_outputs['lb/hr'][cut] *= factor
            # update wt %
            distillation_outputs['wt %'][cut] = 100 * distillation_outputs['lb/hr'][cut] / distillation_inputs['lb/hr']['Crude oil']
            # update BPCD, lb/hr H, MJ/yr
            lb_hr_from_bbl_day = distillation_outputs['lb/hr from bbl/day'][cut]
            distillation_outputs['BPCD'][cut] = distillation_outputs['lb/hr'][cut] / lb_hr_from_bbl_day
            distillation_outputs['lb/hr H'][key] = light_hc_hwt[key] / 100 * distillation_outputs['lb/hr'][cut]
            distillation_outputs['MJ/yr'][key] = distillation_outputs['lb/hr'][cut] * conversion_parameters['lb_to_kg'] * 8760 * light_hc_specific_energy[key]

distillation_outputs['MJ/yr']['Light Naphtha'] = distillation_outputs['lb/hr']['Light Naphtha'] * conversion_parameters['lb_to_kg'] * 8760 * fixed_parameters['specific_energy']['naphtha']
distillation_outputs['MJ/yr']['Heavy Naphtha'] = distillation_outputs['lb/hr']['Heavy Naphtha'] * conversion_parameters['lb_to_kg'] * 8760 * fixed_parameters['specific_energy']['naphtha']
distillation_outputs['MJ/yr']['...Heavy Naphtha'] = distillation_outputs['lb/hr']['...Heavy Naphtha'] * conversion_parameters['lb_to_kg'] * 8760 * fixed_parameters['specific_energy']['naphtha']
distillation_outputs['MJ/yr']['Kerosine'] = distillation_outputs['lb/hr']['Kerosine'] * conversion_parameters['lb_to_kg'] * 8760 * fixed_parameters['specific_energy']['kerosene']
distillation_outputs['MJ/yr']['Light Gas Oil'] = distillation_outputs['lb/hr']['Light Gas Oil'] * conversion_parameters['lb_to_kg'] * 8760 * fixed_parameters['specific_energy']['gas_oil']
distillation_outputs['MJ/yr']['Heavy Gas Oil'] = distillation_outputs['lb/hr']['Heavy Gas Oil'] * conversion_parameters['lb_to_kg'] * 8760 * fixed_parameters['specific_energy']['gas_oil']
distillation_outputs['MJ/yr']['Light Vacuum Gas Oil'] = distillation_outputs['lb/hr']['Light Vacuum Gas Oil'] * conversion_parameters['lb_to_kg'] * 8760 * fixed_parameters['specific_energy']['gas_oil']
distillation_outputs['MJ/yr']['Heavy Vacuum Gas Oil'] = distillation_outputs['lb/hr']['Heavy Vacuum Gas Oil'] * conversion_parameters['lb_to_kg'] * 8760 * fixed_parameters['specific_energy']['gas_oil']
distillation_outputs['MJ/yr']['...Heavy Vacuum Gas Oil'] = distillation_outputs['lb/hr']['...Heavy Vacuum Gas Oil'] * conversion_parameters['lb_to_kg'] * 8760 * fixed_parameters['specific_energy']['gas_oil']
distillation_outputs['MJ/yr']['Vacuum Residue'] = distillation_outputs['lb/hr']['Vacuum Residue'] * conversion_parameters['lb_to_kg'] * 8760 * fixed_parameters['specific_energy']['residual_fuel_oil']

df_distillation_inputs = pd.DataFrame(distillation_inputs).reset_index().rename(columns={'index': 'Cut'})
df_distillation_outputs = pd.DataFrame(distillation_outputs).reset_index().rename(columns={'index': 'Cut'})

output_order = valid_light_hc_keys + [cut for cut in df_distillation_outputs['Cut'] if cut not in valid_light_hc_keys]
df_distillation_outputs = df_distillation_outputs.set_index('Cut').loc[output_order].reset_index()

inputs_label = pd.DataFrame([['Inputs'] + [''] * (df_distillation_inputs.shape[1] - 1)],
                            columns=df_distillation_inputs.columns)
outputs_label = pd.DataFrame([['Outputs'] + [''] * (df_distillation_inputs.shape[1] - 1)],
                             columns=df_distillation_inputs.columns)

# Calculate total inputs and outputs for BPCD, lb/hr, and sulfur lb/hr
total_dist_inputs_bpcd = df_distillation_inputs['BPCD'].apply(pd.to_numeric, errors='coerce').sum()
total_dist_outputs_bpcd = df_distillation_outputs['BPCD'].apply(pd.to_numeric, errors='coerce').sum()

total_dist_inputs_lbhr = df_distillation_inputs['lb/hr'].replace(',', '', regex=True).apply(pd.to_numeric, errors='coerce').sum()
total_dist_outputs_lbhr = df_distillation_outputs['lb/hr'].replace(',', '', regex=True).apply(pd.to_numeric, errors='coerce').sum()

total_dist_inputs_sulfur = df_distillation_inputs['lb/hr S'].apply(pd.to_numeric, errors='coerce').sum()
total_dist_outputs_sulfur = df_distillation_outputs['lb/hr S'].apply(pd.to_numeric, errors='coerce').sum()

total_dist_outputs_energy_content = df_distillation_outputs['MJ/yr'].apply(pd.to_numeric, errors='coerce').sum()

total_dist_inputs_row = {
    'Cut': 'Total',
    'BPCD': total_dist_inputs_bpcd,
    'lb/hr': total_dist_inputs_lbhr,
    'lb/hr S': total_dist_inputs_sulfur
}
total_dist_outputs_row = {
    'Cut': 'Total',
    'BPCD': total_dist_outputs_bpcd,
    'lb/hr': total_dist_outputs_lbhr,
    'MJ/yr': total_dist_outputs_energy_content,
    'lb/hr S': total_dist_outputs_sulfur
}

# Convert to DataFrames
df_total_atm_inputs_row = pd.DataFrame([total_dist_inputs_row], columns=df_distillation_inputs.columns)
df_total_atm_outputs_row = pd.DataFrame([total_dist_outputs_row], columns=df_distillation_outputs.columns)

# Insert totals after inputs and outputs
df_distillation = pd.concat([
    inputs_label,
    df_distillation_inputs,
    df_total_atm_inputs_row,
    outputs_label,
    df_distillation_outputs,
    df_total_atm_outputs_row
], ignore_index=True)

df_distillation.fillna('', inplace=True)
print(df_distillation)

# Determine utilities for atmospheric distillation
annual_atmospheric_dist_utilities = {
    key: (
        utility_data["amount"] * total_dist_inputs_bpcd * 365,
        utility_data["unit"]
    )
    for key, utility_data in atmospheric_distillation_utility_inputs.items()
}

# Determine utilities for vacuum distillation
if refinery_type == 'cracking':
    vacuum_distillation_flow_rate_BPCD = distillation_outputs['BPCD']['Light Vacuum Gas Oil'] + distillation_outputs['BPCD']['Heavy Vacuum Gas Oil'] + distillation_outputs['BPCD']['...Heavy Vacuum Gas Oil'] + distillation_outputs['BPCD']['Vacuum Residue']

    annual_vacuum_dist_utilities = {
        key: (
            utility_data["amount"] * vacuum_distillation_flow_rate_BPCD * 365,
            utility_data["unit"]
        )
        for key, utility_data in vacuum_distillation_utility_inputs.items()
    }


# KEROSENE HYDROTREATMENT -------------------------------------------------------------------------------------------------------------------
if aromatics_removal_technique == 'hydrotreatment':

    # Initialize hydrocracking inputs and outputs
    hydrotreatment_inputs = {
    'API': {},
    'Specific Gravity': {},
    'Characterization Factor': {},
    'MW': {},
    'BPCD': {},
    'lb/hr from bbl/day': {},
    'lb/hr': {},
    'MJ/yr': {},
    'wt% S': {},
    'wppm S': {},
    'lb/hr S': {},
    'wt %': {},
    'wt% H': {},
    'lb/hr H': {},
    'wppm N': {},
    'lb/hr N': {},
    'vol% paraffins': {},
    'vol% naphthenes': {},
    'vol% aromatics': {},
    'wt% paraffins': {},
    'wt% naphthenes': {},
    'wt% aromatics': {},
    'wt% naphthalenes': {},
    'wt% polyaromatics': {},
    'wt% monoaromatics': {},
    'vol% polyaromatics': {},
    'vol% monoaromatics': {},
    }

    hydrotreatment_outputs = {
    'API': {},
    'Specific Gravity': {},
    'Characterization Factor': {},
    'MW': {},
    'BPCD': {},
    'lb/hr from bbl/day': {},
    'lb/hr': {},
    'MJ/yr': {},
    'wt% S': {},
    'wppm S': {},
    'lb/hr S': {},
    'wt %': {},
    'wt% H': {},
    'lb/hr H': {},
    'wppm N': {},
    'lb/hr N': {},
    'vol% paraffins': {},
    'vol% naphthenes': {},
    'vol% aromatics': {},
    'wt% paraffins': {},
    'wt% naphthenes': {},
    'wt% aromatics': {},
    'wt% naphthalenes': {},
    'wt% polyaromatics': {},
    'wt% monoaromatics': {},
    'vol% polyaromatics': {},
    'vol% monoaromatics': {},
    }

    # Feed for hydrotreatment is kerosine
    hydrotreatment_cuts = {
        'Kerosine': distillation_outputs
    }

    properties = ['BPCD', 'API', 'Specific Gravity', 'Characterization Factor', 'MW', 'lb/hr', 'MJ/yr', 'wt% S', 'lb/hr S', 'wt% H', 'lb/hr H', 'wppm N', 'lb/hr N', 'vol% aromatics', 'wt% naphthalenes']

    for prop in properties:
        for cut, source_df in hydrotreatment_cuts.items():
            hydrotreatment_inputs[prop][cut] = source_df[prop][cut]

    total_hydrotreatment_inputs_bpcd = sum(hydrotreatment_inputs['BPCD'].values())
    total_hydrotreatment_inputs_lb_hr_sulfur = pd.to_numeric(pd.Series(hydrotreatment_inputs['lb/hr S'].values()), errors='coerce').fillna(0).sum()
    total_hydrotreatment_inputs_wppm_nitrogen = sum(hydrotreatment_inputs['wppm N'].values())

    # Determine liquid hydrocarbon product yield from hydrotreatment
    liquid_product_yield = interpolate_liquid_yield(hydrotreatment_operating_parameters['reactor_temp'], hydrotreatment_operating_parameters['LHSV'], hydrotreatment_operating_parameters['reactor_pressure'])
    print(f'liquid_product_yield: {liquid_product_yield}')
    hydrotreatment_outputs['lb/hr']['HT Kerosine'] = liquid_product_yield / 100 * hydrotreatment_inputs['lb/hr']['Kerosine']
    
    mean_avg_boiling_pt = (330 + 480) / 2
    mean_avg_boiling_pt_Rankine = mean_avg_boiling_pt + 459.67

    # Determine aromatic saturation efficiency of hydrotreatment
    monoaromatics_saturation_efficiency, polyaromatics_saturation_efficiency = interpolate_aromatics_saturation_efficiency(hydrotreatment_operating_parameters['reactor_temp'], hydrotreatment_operating_parameters['reactor_pressure'])

    bulk_density = hydrotreatment_inputs['Specific Gravity']['Kerosine'] * density_water    # kg/L

    aromatics_density = 0.88
    polyaromatics_density = 1.02  # kg/L
    monoaromatics_density = 0.869 # kg/L

    hydrotreatment_inputs['wt% polyaromatics']['Kerosine'] = hydrotreatment_inputs['wt% naphthalenes']['Kerosine']
    hydrotreatment_inputs['vol% polyaromatics']['Kerosine'] = hydrotreatment_inputs['wt% polyaromatics']['Kerosine'] * bulk_density / polyaromatics_density

    hydrotreatment_inputs['vol% monoaromatics']['Kerosine'] = hydrotreatment_inputs['vol% aromatics']['Kerosine'] - hydrotreatment_inputs['vol% polyaromatics']['Kerosine']
    hydrotreatment_inputs['wt% monoaromatics']['Kerosine'] =  hydrotreatment_inputs['vol% monoaromatics']['Kerosine'] * monoaromatics_density / bulk_density

    hydrotreatment_inputs['wt% aromatics']['Kerosine'] =  hydrotreatment_inputs['vol% aromatics']['Kerosine'] * aromatics_density / bulk_density

    # ----- Aromatics saturation -------------------------------------------------------------------
    hydrotreatment_outputs['wt% monoaromatics']['HT Kerosine'] =  hydrotreatment_inputs['vol% monoaromatics']['Kerosine'] * (1 - monoaromatics_saturation_efficiency/100)
    hydrotreatment_outputs['wt% polyaromatics']['HT Kerosine'] = hydrotreatment_inputs['wt% polyaromatics']['Kerosine'] * (1 - polyaromatics_saturation_efficiency/100)
    hydrotreatment_outputs['wt% aromatics']['HT Kerosine'] = hydrotreatment_outputs['wt% monoaromatics']['HT Kerosine'] + hydrotreatment_outputs['wt% polyaromatics']['HT Kerosine']

    # ----- Naphthalene removal -------------------------------------------------------------------
    #hydrotreatment_outputs['wt% monoaromatics']['HT Kerosine'] =  hydrotreatment_inputs['vol% monoaromatics']['Kerosine']
    #hydrotreatment_outputs['wt% polyaromatics']['HT Kerosine'] = 0
    #hydrotreatment_outputs['wt% aromatics']['HT Kerosine'] = hydrotreatment_outputs['wt% monoaromatics']['HT Kerosine'] + hydrotreatment_outputs['wt% polyaromatics']['HT Kerosine']
    # ---------------------------------------------------------------------------------------------


    # Determine sulfur removal efficiency
    sulfur_removal_efficiency = interpolate_sulfur_removal_perc(hydrotreatment_operating_parameters['reactor_temp'], hydrotreatment_operating_parameters['reactor_pressure'])

    hydrotreatment_outputs['wt% S']['HT Kerosine'] = hydrotreatment_inputs['wt% S']['Kerosine'] * (1 - sulfur_removal_efficiency/100)

    # HT Kerosene properties
    # Component densities in kg/L
    
    # Compute initial non-aromatics fraction
    non_aromatics_initial_pct = 100 - hydrotreatment_inputs['wt% aromatics']['Kerosine']
    non_aromatics_final_pct = 100 -  hydrotreatment_outputs['wt% aromatics']['HT Kerosine']

    # Estimate change in density due to aromatics saturation
    delta_density_aromatics = (
    (hydrotreatment_inputs['wt% aromatics']['Kerosine'] / 100) * fixed_parameters['density']['aromatics']
    + ((100 - hydrotreatment_inputs['wt% aromatics']['Kerosine']) / 100) * fixed_parameters['density']['alkanes']
    ) - (
    (hydrotreatment_outputs['wt% aromatics']['HT Kerosine'] / 100) * fixed_parameters['density']['aromatics']
    + ((100 - hydrotreatment_outputs['wt% aromatics']['HT Kerosine']) / 100) * fixed_parameters['density']['alkanes']
    )

    # Correct initial density to match given actual density
    # So we apply delta to actual density
    density_intermediate = bulk_density - delta_density_aromatics

    density_sulfur = 1.8
    delta_density_sulfur = (
    (hydrotreatment_inputs['wt% S']['Kerosine'] / 100) * density_sulfur
    + ((100 - hydrotreatment_inputs['wt% S']['Kerosine']) / 100) * bulk_density
    ) - (
    (hydrotreatment_outputs['wt% S']['HT Kerosine'] / 100) * density_sulfur
    + ((100 - hydrotreatment_outputs['wt% S']['HT Kerosine']) / 100) * bulk_density
    )

    density_final = density_intermediate - delta_density_sulfur

    # Estimate change in specific energy due to aromatics saturation
    delta_specific_energy_aromatics = (
    (hydrotreatment_inputs['wt% aromatics']['Kerosine'] / 100) * fixed_parameters['specific_energy']['aromatics']
    + ((100 - hydrotreatment_inputs['wt% aromatics']['Kerosine']) / 100) * fixed_parameters['specific_energy']['alkanes']
    ) - (
    (hydrotreatment_outputs['wt% aromatics']['HT Kerosine'] / 100) * fixed_parameters['specific_energy']['aromatics']
    + ((100 - hydrotreatment_outputs['wt% aromatics']['HT Kerosine']) / 100) * fixed_parameters['specific_energy']['alkanes']
    )

    specific_energy_HT_kerosene = fixed_parameters['specific_energy']['kerosene'] - delta_specific_energy_aromatics 
    print(f'specific_energy_HT_kerosene: {specific_energy_HT_kerosene}')

    # Convert to API gravity
    # Convert kg/m3 to specific gravity at 60F (15.56C)
    hydrotreatment_outputs['Specific Gravity']['HT Kerosine'] = density_final / density_water
    hydrotreatment_outputs['API']['HT Kerosine'] = (141.5 / hydrotreatment_outputs['Specific Gravity']['HT Kerosine']) - 131.5

    hydrotreatment_outputs['Characterization Factor']['HT Kerosine'] = T_mean_avg_bp_R_kerosene ** (1/3) / hydrotreatment_outputs['Specific Gravity']['HT Kerosine']
    hydrotreatment_outputs['lb/hr from bbl/day']['HT Kerosine'] = find_lbhr_conversion(hydrotreatment_outputs['Specific Gravity']['HT Kerosine'], density_conv_table)

    hydrotreatment_outputs['BPCD']['HT Kerosine'] = hydrotreatment_outputs['lb/hr']['HT Kerosine'] / hydrotreatment_outputs['lb/hr from bbl/day']['HT Kerosine']
    hydrotreatment_outputs['MW']['HT Kerosine'] = 0.000045673 * T_mean_avg_bp_R_kerosene**(2.1962) * hydrotreatment_outputs['Specific Gravity']['HT Kerosine']**(-1.0164)


    # Interpolate sulfur content 
    hydrotreatment_outputs['lb/hr S']['HT Kerosine'] = hydrotreatment_outputs['wt% S']['HT Kerosine'] / 100 * hydrotreatment_outputs['lb/hr']['HT Kerosine']

    hydrotreatment_outputs['wppm N']['HT Kerosine'] = 0
    hydrotreatment_outputs['lb/hr N']['HT Kerosine'] = 0

    # Calculate output of H2S to remove sulfur
    hydrotreatment_outputs['lb/hr S']['H2S'] = total_hydrotreatment_inputs_lb_hr_sulfur - hydrotreatment_outputs['lb/hr S']['HT Kerosine']
    hydrotreatment_outputs['lb/hr']['H2S'] = hydrotreatment_outputs['lb/hr S']['H2S'] * fixed_parameters['molecular_weight']['H2S'] / fixed_parameters['molecular_weight']['S']
    hydrotreatment_outputs['lb/hr H']['H2S'] = hydrotreatment_outputs['lb/hr']['H2S'] * 2 * fixed_parameters['molecular_weight']['H'] / fixed_parameters['molecular_weight']['H2S']

    # Calculate output of H2S to remove nitrogen
    hydrotreatment_outputs['lb/hr N']['NH3'] = hydrotreatment_inputs['lb/hr N']['Kerosine'] - hydrotreatment_outputs['lb/hr N']['HT Kerosine']
    hydrotreatment_outputs['lb/hr']['NH3'] = hydrotreatment_outputs['lb/hr N']['NH3'] * fixed_parameters['molecular_weight']['NH3'] / fixed_parameters['molecular_weight']['N']
    hydrotreatment_outputs['lb/hr H']['NH3'] = hydrotreatment_outputs['lb/hr']['NH3'] * 3 * fixed_parameters['molecular_weight']['H'] / fixed_parameters['molecular_weight']['NH3']

    hydrogen_partial_pressure = hydrotreatment_operating_parameters['reactor_pressure'] / 10        # Assume hydrogen partial pressure is reactor pressure convert bar to MPa (divide by 10)

    hydrotreatment_inputs['wppm S']['Kerosine'] = hydrotreatment_inputs['wt% S']['Kerosine'] * 10000
    hydrotreatment_outputs['wppm S']['HT Kerosine'] = hydrotreatment_outputs['wt% S']['HT Kerosine'] * 10000

    # Calculate chemical hydrogen consumption and dissolved hydrogen
    H_HDS, H_HDN, H_HDA, H_chem_total_Nm3_per_m3, H_chem_lb_per_hr, H_chem_scf_per_bbl, H_diss_lb_per_hr, H_diss_scf_per_bbl = compute_hydrogen_consumption(
    hydrotreatment_inputs['wppm S']['Kerosine'], hydrotreatment_outputs['wppm S']['HT Kerosine'],
    hydrotreatment_inputs['wppm N']['Kerosine'], hydrotreatment_outputs['wppm N']['HT Kerosine'],
    hydrotreatment_inputs['wt% polyaromatics']['Kerosine'], hydrotreatment_outputs['wt% polyaromatics']['HT Kerosine'],
    hydrotreatment_inputs['wt% monoaromatics']['Kerosine'], hydrotreatment_outputs['wt% monoaromatics']['HT Kerosine'],
    hydrotreatment_inputs['Specific Gravity']['Kerosine'], hydrotreatment_outputs['Specific Gravity']['HT Kerosine'],
    hydrotreatment_inputs['MW']['Kerosine'], hydrotreatment_outputs['MW']['HT Kerosine'],
    liquid_product_yield, total_hydrotreatment_inputs_bpcd,
    hydrotreatment_operating_parameters['H_purity'], hydrogen_partial_pressure, hydrotreatment_operating_parameters['gas_to_oil'])  
    H_HDA_scf_per_bbl = H_HDA / conversion_parameters['H2_scf_to_Nm3'] * conversion_parameters['bbl_to_m3']
    print(f"HDS: {H_HDS:.3f} Nm³ H₂/m³ oil")
    print(f"HDN: {H_HDN:.3f} Nm³ H₂/m³ oil")
    print(f"HDA: {H_HDA:.3f} Nm³ H₂/m³ oil")
    print(f"HDA: {H_HDA_scf_per_bbl:.3f} scf H₂/m³ oil")
    print(f"Total chemical H₂ consumption: {H_chem_total_Nm3_per_m3:.3f} Nm³ H₂/m³ oil")
    print(f"Total chemical H₂ consumption: {H_chem_lb_per_hr:.3f} lb/hr")
    print(f"Total chemical H₂ consumption: {H_chem_scf_per_bbl:.3f} scf/bbl")
    print(f"Total H₂ dissolved: {H_diss_lb_per_hr:.3f} lb/hr")
    print(f"Total H₂ dissolved: {H_diss_scf_per_bbl:.3f} scf/bbl")

    H_cons_lb_per_hr = H_chem_lb_per_hr + H_diss_lb_per_hr
    H_cons_scf_per_bbl = H_chem_scf_per_bbl + H_diss_scf_per_bbl

    print(f"Total H₂ consumption: {H_cons_lb_per_hr:.3f} lb/hr")
    print(f"Total H₂ consumption: {H_cons_scf_per_bbl:.3f} scf/bbl")
    if H_cons_scf_per_bbl > 800:
        print(f'Hydrogen consumption exceeds 800 scf/bbl --> indicative of hydrocracking (adjust operating conditions!)')

    # Set a range for H2 consumption
    #H2_consumed = np.array([200, 800])
    H2_consumed = H_cons_scf_per_bbl

    hydrotreatment_operaring_hydrogen_input_Nm3_per_hr = hydrotreatment_operating_parameters['gas_to_oil'] * hydrotreatment_inputs['BPCD']['Kerosine'] * conversion_parameters['BPCD_to_m3_per_hr']  

    hydrotreatment_operaring_hydrogen_input_lb_per_hr = hydrotreatment_operaring_hydrogen_input_Nm3_per_hr * conversion_parameters['H2_Nm3_to_lb']
    hydrotreatment_operaring_hydrogen_input_scf_per_bbl = hydrotreatment_operaring_hydrogen_input_lb_per_hr * conversion_parameters['H2_lb_to_scf'] / (hydrotreatment_inputs['BPCD']['Kerosine'] / 24)

    print(f"Operating H₂ input (makeup + recycle): {hydrotreatment_operaring_hydrogen_input_lb_per_hr:.3f} lb/hr")
    print(f"Operating H₂ input (makeup + recycle): {hydrotreatment_operaring_hydrogen_input_scf_per_bbl:.3f} scf/bbl")

    hydrotreatment_inputs['lb/hr']['Makeup Hydrogen'] = H2_consumed
    hydrotreatment_inputs['lb/hr']['Recycle Hydrogen'] = hydrotreatment_operaring_hydrogen_input_lb_per_hr - hydrotreatment_inputs['lb/hr']['Makeup Hydrogen']

    H2_scf_per_yr = H2_consumed * hydrotreatment_inputs['BPCD']['Kerosine'] * 365

    hydrotreatment_outputs['lb/hr']['H2'] = hydrotreatment_inputs['lb/hr']['Recycle Hydrogen']
    
    # Determine HC gas output from mass balance
    hydrotreatment_outputs['lb/hr']['HC gas'] = hydrotreatment_inputs['lb/hr']['Kerosine'] + hydrotreatment_inputs['lb/hr']['Makeup Hydrogen'] - hydrotreatment_outputs['lb/hr']['HT Kerosine'] - hydrotreatment_outputs['lb/hr']['H2S']

    # Molecular breakdown of HC gas
    hydrotreatment_outputs_wt_perc_CH4 = 0.0588 
    hydrotreatment_outputs_wt_perc_C2H6 = 0.0588
    hydrotreatment_outputs_wt_perc_C3H8 = 0.2124
    hydrotreatment_outputs_wt_perc_iC4 = 0.4425 
    hydrotreatment_outputs_wt_perc_nC4 = 0.2274

    # Weight percent of H2 in HC gas components
    wt_perc_H_CH4 = 4 * fixed_parameters['molecular_weight']['H'] / fixed_parameters['molecular_weight']['CH4']
    wt_perc_H_C2H6 = 6 * fixed_parameters['molecular_weight']['H'] / fixed_parameters['molecular_weight']['C2H6']
    wt_perc_H_C3H8 = 8 * fixed_parameters['molecular_weight']['H'] / fixed_parameters['molecular_weight']['C3H8']
    wt_perc_H_iC4 = 17.38/100
    wt_perc_H_nC4 = 17.38/100

    hydrotreatment_outputs['lb/hr']['C3 and lighter'] = (hydrotreatment_outputs_wt_perc_CH4 + hydrotreatment_outputs_wt_perc_C2H6 + hydrotreatment_outputs_wt_perc_C3H8) * hydrotreatment_outputs['lb/hr']['HC gas']
    hydrotreatment_outputs['lb/hr']['iC4'] = hydrotreatment_outputs_wt_perc_iC4 * hydrotreatment_outputs['lb/hr']['HC gas']
    hydrotreatment_outputs['lb/hr']['nC4'] = hydrotreatment_outputs_wt_perc_nC4 * hydrotreatment_outputs['lb/hr']['HC gas']

    hydrotreatment_outputs['wt% H']['C3 and lighter'] = 100 * (wt_perc_H_CH4 * hydrotreatment_outputs_wt_perc_CH4 + wt_perc_H_C2H6 * hydrotreatment_outputs_wt_perc_C2H6  + wt_perc_H_C3H8 * hydrotreatment_outputs_wt_perc_C3H8) / (hydrotreatment_outputs_wt_perc_CH4 + hydrotreatment_outputs_wt_perc_C2H6 + hydrotreatment_outputs_wt_perc_C3H8)
    hydrotreatment_outputs['lb/hr H']['C3 and lighter'] = hydrotreatment_outputs['wt% H']['C3 and lighter']/100 * hydrotreatment_outputs['lb/hr']['C3 and lighter']
    hydrotreatment_outputs['wt% H']['iC4'] = wt_perc_H_iC4 *100
    hydrotreatment_outputs['lb/hr H']['iC4'] = hydrotreatment_outputs['wt% H']['iC4'] * hydrotreatment_outputs['lb/hr']['iC4'] /100
    hydrotreatment_outputs['wt% H']['nC4'] = wt_perc_H_nC4 * 100
    hydrotreatment_outputs['lb/hr H']['nC4'] = hydrotreatment_outputs['wt% H']['nC4'] * hydrotreatment_outputs['lb/hr']['nC4'] /100

    hydrotreatment_outputs['wt% H']['HC gas'] = (hydrotreatment_outputs_wt_perc_CH4 * wt_perc_H_CH4 + hydrotreatment_outputs_wt_perc_C2H6 * wt_perc_H_C2H6 + hydrotreatment_outputs_wt_perc_C3H8 * wt_perc_H_C3H8 + hydrotreatment_outputs_wt_perc_iC4 * wt_perc_H_iC4 + hydrotreatment_outputs_wt_perc_nC4 * wt_perc_H_nC4) * 100
    
    hydrotreatment_outputs['lb/hr H']['HC gas'] = hydrotreatment_outputs['lb/hr']['HC gas'] * hydrotreatment_outputs['wt% H']['HC gas'] / 100

    hydrogen_added_to_liquid_molecules = H_chem_lb_per_hr - hydrotreatment_outputs['lb/hr H']['H2S'] - hydrotreatment_outputs['lb/hr H']['HC gas']

    hydrotreatment_outputs['lb/hr H']['HT Kerosine'] = hydrogen_added_to_liquid_molecules + H_diss_lb_per_hr + hydrotreatment_inputs['lb/hr H']['Kerosine']
    hydrotreatment_outputs['wt% H']['HT Kerosine'] = hydrotreatment_outputs['lb/hr H']['HT Kerosine'] / hydrotreatment_outputs['lb/hr']['HT Kerosine'] * 100

    hydrotreatment_outputs['lb/hr from bbl/day']['C3 and lighter'] = 7.42
    hydrotreatment_outputs['lb/hr from bbl/day']['iC4'] = 8.22
    hydrotreatment_outputs['lb/hr from bbl/day']['nC4'] = 8.51

    hydrotreatment_outputs['BPCD']['C3 and lighter'] = hydrotreatment_outputs['lb/hr']['C3 and lighter']/hydrotreatment_outputs['lb/hr from bbl/day']['C3 and lighter'] 
    hydrotreatment_outputs['BPCD']['iC4'] = hydrotreatment_outputs['lb/hr']['iC4']/hydrotreatment_outputs['lb/hr from bbl/day']['iC4'] 
    hydrotreatment_outputs['BPCD']['nC4'] = hydrotreatment_outputs['lb/hr']['nC4']/hydrotreatment_outputs['lb/hr from bbl/day']['nC4'] 

    specific_energy_C3_and_lighter =  hydrotreatment_outputs_wt_perc_CH4 * fixed_parameters['specific_energy']['CH4'] + hydrotreatment_outputs_wt_perc_C2H6 * fixed_parameters['specific_energy']['C2H6'] + hydrotreatment_outputs_wt_perc_C3H8 * fixed_parameters['specific_energy']['C3H8']

    hydrotreatment_outputs['MJ/yr']['C3 and lighter'] = hydrotreatment_outputs['lb/hr']['C3 and lighter'] * conversion_parameters['lb_to_kg'] * 8760 * specific_energy_C3_and_lighter
    hydrotreatment_outputs['MJ/yr']['iC4'] = hydrotreatment_outputs['lb/hr']['iC4'] * conversion_parameters['lb_to_kg'] * 8760 * fixed_parameters['specific_energy']['butane']
    hydrotreatment_outputs['MJ/yr']['nC4'] = hydrotreatment_outputs['lb/hr']['nC4'] * conversion_parameters['lb_to_kg'] * 8760 * fixed_parameters['specific_energy']['butane']
    hydrotreatment_outputs['MJ/yr']['HT Kerosine'] = hydrotreatment_outputs['lb/hr']['HT Kerosine'] * conversion_parameters['lb_to_kg'] * 8760 * specific_energy_HT_kerosene

    df_hydrotreatment_inputs = pd.DataFrame(hydrotreatment_inputs).reset_index().rename(columns={'index': 'Cut'})
    df_hydrotreatment_outputs = pd.DataFrame(hydrotreatment_outputs).reset_index().rename(columns={'index': 'Cut'})
    df_hydrotreatment_outputs = df_hydrotreatment_outputs[df_hydrotreatment_outputs['Cut'] != 'total gas flow']
    df_hydrotreatment_outputs = df_hydrotreatment_outputs[df_hydrotreatment_outputs['Cut'] != 'HC gas']

    desired_order = ['H2S', 'NH3', 'H2', 'C3 and lighter', 'iC4', 'nC4', 'HT Kerosine']

    # Sort the dataframe by the 'Cut' column using categorical ordering
    df_hydrotreatment_outputs['Cut'] = pd.Categorical(df_hydrotreatment_outputs['Cut'], categories=desired_order, ordered=True)
    df_hydrotreatment_outputs = df_hydrotreatment_outputs.sort_values('Cut').reset_index(drop=True)
    
    inputs_label = pd.DataFrame([['Inputs'] + [''] * (df_hydrotreatment_inputs.shape[1] - 1)],
                            columns=df_hydrotreatment_inputs.columns)
    outputs_label = pd.DataFrame([['Outputs'] + [''] * (df_hydrotreatment_inputs.shape[1] - 1)],
                             columns=df_hydrotreatment_inputs.columns)

    total_hydrotreatmentinputs_bpcd = df_hydrotreatment_inputs['BPCD'].apply(pd.to_numeric, errors='coerce').sum()
    total_hydrotreatment_outputs_bpcd = df_hydrotreatment_outputs['BPCD'].apply(pd.to_numeric, errors='coerce').sum()

    total_hydrotreatment_inputs_lbhr = df_hydrotreatment_inputs['lb/hr'].replace(',', '', regex=True).apply(pd.to_numeric, errors='coerce').sum()
    total_hydrotreatment_outputs_lbhr = df_hydrotreatment_outputs['lb/hr'].replace(',', '', regex=True).apply(pd.to_numeric, errors='coerce').sum()

    total_hydrotreatment_inputs_lbhr_H2 = df_hydrotreatment_inputs['lb/hr H'].replace(',', '', regex=True).apply(pd.to_numeric, errors='coerce').sum()
    total_hydrotreatment_outputs_lbhr_H2 = df_hydrotreatment_outputs['lb/hr H'].replace(',', '', regex=True).apply(pd.to_numeric, errors='coerce').sum()

    total_hydrotreatment_inputs_sulfur = df_hydrotreatment_inputs['lb/hr S'].apply(pd.to_numeric, errors='coerce').sum()
    total_hydrotreatment_outputs_sulfur = df_hydrotreatment_outputs['lb/hr S'].apply(pd.to_numeric, errors='coerce').sum()

    total_hydrotreatment_inputs_nitrogen = df_hydrotreatment_inputs['lb/hr N'].apply(pd.to_numeric, errors='coerce').sum()
    total_hydrocracking_inputs_wppm_nitrogen = total_hydrotreatment_inputs_nitrogen/total_hydrotreatment_inputs_lbhr * 1000000

    total_hydrotreatment_outputs_energy_content = df_hydrotreatment_outputs['MJ/yr'].apply(pd.to_numeric, errors='coerce').sum()

    total_inputs_row = {
    'Cut': 'Total',
    'BPCD': total_hydrotreatmentinputs_bpcd,
    'lb/hr': total_hydrotreatment_inputs_lbhr,
    'lb/hr H': total_hydrotreatment_inputs_lbhr_H2,
    'lb/hr S': total_hydrotreatment_inputs_sulfur,
    'lb/hr N': total_hydrotreatment_inputs_nitrogen,
    'wppm N': total_hydrotreatment_inputs_wppm_nitrogen,
    }

    total_outputs_row = {
    'Cut': 'Total',
    'BPCD': total_hydrotreatment_outputs_bpcd,
    'lb/hr': total_hydrotreatment_outputs_lbhr,
    'MJ/yr': total_hydrotreatment_outputs_energy_content,
    'lb/hr H': total_hydrotreatment_outputs_lbhr_H2,
    'lb/hr S': total_hydrotreatment_outputs_sulfur
    }

    df_total_inputs_row = pd.DataFrame([total_inputs_row], columns=df_hydrotreatment_inputs.columns)
    df_total_outputs_row = pd.DataFrame([total_outputs_row], columns=df_hydrotreatment_inputs.columns)

    df_hydrotreatment = pd.concat([
    inputs_label,
    df_hydrotreatment_inputs,
    df_total_inputs_row,
    outputs_label,
    df_hydrotreatment_outputs,
    df_total_outputs_row
    ], ignore_index=True)

    df_hydrotreatment.fillna('', inplace=True)

    print(df_hydrotreatment)

    # Determine utilities for hydrotreatment
    hydrotreatment_utility_data['H2'] = {
    "amount": float(H2_consumed),
    "unit": "scf"
    }
    print(hydrotreatment_utility_data)
    
    annual_hydrotreatment_utilities = {
        key: (
        np.array(
            [a * hydrotreatment_inputs['BPCD']['Kerosine'] * 365 for a in utility_data["amount"]]
        ) if isinstance(utility_data["amount"], (list, np.ndarray)) else
        float(utility_data["amount"] * hydrotreatment_inputs['BPCD']['Kerosine'] * 365),
        utility_data["unit"]
        )
        for key, utility_data in hydrotreatment_utility_data.items()
    }

    print(annual_hydrotreatment_utilities)

    # AMINE GAS TREATING

    # Initialize amine gas treating inputs and outputs
    amine_contactor_inputs = {
    'lb/hr': {},
    }

    amine_contactor_outputs = {
    'lb/hr': {},
    }

    amine_gas_treating_cuts = {
        'H2S': hydrotreatment_outputs,
        'NH3': hydrotreatment_outputs,
        'H2': hydrotreatment_outputs
    }

    properties = ['lb/hr']

    for prop in properties:
        for cut, source_df in amine_gas_treating_cuts.items():
            amine_contactor_inputs[prop][f'{cut} in sour gas'] = source_df[prop][cut]


    amine_contactor_outputs['lb/hr']['H2S in sweet gas'] = amine_contactor_inputs['lb/hr']['H2S in sour gas'] * (1 - 0.99)
    amine_contactor_outputs['lb/hr']['H2 in sweet gas'] = amine_contactor_inputs['lb/hr']['H2 in sour gas']

    # amine_contactor_outputs['lb/hr']['NH3 in sweet gas'] = amine_contactor_inputs['lb/hr']['NH3 in sour gas']

    H2S_absorbed_by_amine_lb_hr = amine_contactor_inputs['lb/hr']['H2S in sour gas'] * 0.99

    H2S_absorbed_by_amine_mole_hr = H2S_absorbed_by_amine_lb_hr * conversion_parameters['lb_to_g'] / fixed_parameters['molecular_weight']['H2S']

    MEA_mole_hr = H2S_absorbed_by_amine_mole_hr * 3

    MEA_lb_hr = MEA_mole_hr * fixed_parameters['molecular_weight']['MEA'] / conversion_parameters['lb_to_g']

    amine_contactor_inputs['lb/hr']['MEA in lean amine'] = MEA_lb_hr
    amine_contactor_outputs['lb/hr']['MEA in rich amine'] = amine_contactor_inputs['lb/hr']['MEA in lean amine']

    residual_H2S_in_lean_amine_mole_hr = 0.09 * MEA_mole_hr # 0.09 moles per mole of MEA
    residual_H2S_in_lean_amine_lb_hr = residual_H2S_in_lean_amine_mole_hr / conversion_parameters['lb_to_g'] * fixed_parameters['molecular_weight']['H2S']

    solution_circulation_rate_lb_hr = MEA_lb_hr /0.20 # MEA is 20% of solution
    amine_contactor_inputs['lb/hr']['H2O in lean amine'] = solution_circulation_rate_lb_hr - MEA_lb_hr
    amine_contactor_outputs['lb/hr']['H2O in rich amine'] = amine_contactor_inputs['lb/hr']['H2O in lean amine']
    amine_contactor_inputs['lb/hr']['H2S in lean amine'] = residual_H2S_in_lean_amine_lb_hr

    amine_contactor_outputs['lb/hr']['H2S in rich amine'] = amine_contactor_inputs['lb/hr']['H2S in lean amine'] + H2S_absorbed_by_amine_lb_hr
    amine_contactor_outputs['lb/hr']['NH3 in rich amine'] = amine_contactor_inputs['lb/hr']['NH3 in sour gas']

    amine_solution_circulation_rate_GPM = solution_circulation_rate_lb_hr * 0.119826427317 / 60
    amine_solution_circulation_rate_GPD = amine_solution_circulation_rate_GPM * (24*60)

    def sort_key(cut_name):
        return ('amine' in cut_name.lower(), cut_name.lower())

    df_amine_contactor_inputs = (
    pd.DataFrame(amine_contactor_inputs)
    .reset_index()
    .rename(columns={'index': 'Cut'})
    .sort_values(by='Cut', key=lambda col: col.map(sort_key))
    .reset_index(drop=True)
    )

    df_amine_contactor_outputs = (
    pd.DataFrame(amine_contactor_outputs)
    .reset_index()
    .rename(columns={'index': 'Cut'})
    .sort_values(by='Cut', key=lambda col: col.map(sort_key))
    .reset_index(drop=True)
    )

    total_amine_contactor_inputs_lbhr = df_amine_contactor_inputs['lb/hr'].replace(',', '', regex=True).apply(pd.to_numeric, errors='coerce').sum()
    total_amine_contactor_outputs_lbhr = df_amine_contactor_outputs['lb/hr'].replace(',', '', regex=True).apply(pd.to_numeric, errors='coerce').sum()

    total_inputs_row = {
    'Cut': 'Total',
    'lb/hr': total_amine_contactor_inputs_lbhr
    }

    total_outputs_row = {
    'Cut': 'Total',
    'lb/hr': total_amine_contactor_outputs_lbhr,
    }

    df_total_inputs_row = pd.DataFrame([total_inputs_row], columns=df_amine_contactor_inputs.columns)
    df_total_outputs_row = pd.DataFrame([total_outputs_row], columns=df_amine_contactor_inputs.columns)

    inputs_label = pd.DataFrame([['Inputs'] + [''] * (df_amine_contactor_inputs.shape[1] - 1)],
                            columns=df_amine_contactor_inputs.columns)
    outputs_label = pd.DataFrame([['Outputs'] + [''] * (df_amine_contactor_outputs.shape[1] - 1)],
                             columns=df_amine_contactor_outputs.columns)

    df_amine_gas_treating = pd.concat([
    inputs_label,
    df_amine_contactor_inputs,
    df_total_inputs_row,
    outputs_label,
    df_amine_contactor_outputs,
    df_total_outputs_row
    ], ignore_index=True)

    print(f'Amine Contactor:')
    print(df_amine_gas_treating)

    recycle_gas_purity = amine_contactor_outputs['lb/hr']['H2 in sweet gas'] / (amine_contactor_outputs['lb/hr']['H2 in sweet gas'] + amine_contactor_outputs['lb/hr']['H2S in sweet gas']) * 100
    if not isinstance(recycle_gas_purity, (list, tuple, np.ndarray)):
        recycle_gas_purity = [recycle_gas_purity]

    for i, purity in enumerate(recycle_gas_purity):
        print(f'H2 recycle gas purity (case {i+1}) = {purity:.2f}%')

    amine_stripper_inputs = {
    'lb/hr': {},
    }

    amine_stripper_outputs = {
    'lb/hr': {},
    }

    amine_stripper_cuts = {
        'H2O in rich amine': amine_contactor_outputs,
        'H2S in rich amine': amine_contactor_outputs,
        'MEA in rich amine': amine_contactor_outputs,
        'NH3 in rich amine': amine_contactor_outputs
    }

    properties = ['lb/hr']

    for prop in properties:
        for cut, source_df in amine_stripper_cuts.items():
            amine_stripper_inputs[prop][cut] = source_df[prop][cut]

    amine_stripper_outputs['lb/hr']['H2O in lean amine'] = amine_contactor_inputs['lb/hr']['H2O in lean amine']
    amine_stripper_outputs['lb/hr']['H2S in lean amine'] = amine_contactor_inputs['lb/hr']['H2S in lean amine']
    amine_stripper_outputs['lb/hr']['MEA in lean amine'] = amine_contactor_inputs['lb/hr']['MEA in lean amine']
    amine_stripper_outputs['lb/hr']['H2S in acid gas'] = amine_stripper_inputs['lb/hr']['H2S in rich amine'] - amine_contactor_inputs['lb/hr']['H2S in lean amine']
    amine_stripper_outputs['lb/hr']['NH3 in acid gas'] = amine_stripper_inputs['lb/hr']['NH3 in rich amine']
    
    df_amine_stripper_inputs = (
    pd.DataFrame(amine_stripper_inputs)
    .reset_index()
    .rename(columns={'index': 'Cut'})
    .sort_values(by='Cut', key=lambda col: col.map(sort_key))
    .reset_index(drop=True)
    )

    df_amine_stripper_outputs = (
    pd.DataFrame(amine_stripper_outputs)
    .reset_index()
    .rename(columns={'index': 'Cut'})
    .sort_values(by='Cut', key=lambda col: col.map(sort_key))
    .reset_index(drop=True)
    )

    total_amine_stripper_inputs_lbhr = df_amine_stripper_inputs['lb/hr'].replace(',', '', regex=True).apply(pd.to_numeric, errors='coerce').sum()
    total_amine_stripper_outputs_lbhr = df_amine_stripper_outputs['lb/hr'].replace(',', '', regex=True).apply(pd.to_numeric, errors='coerce').sum()

    total_inputs_row = {
    'Cut': 'Total',
    'lb/hr': total_amine_stripper_inputs_lbhr
    }

    total_outputs_row = {
    'Cut': 'Total',
    'lb/hr': total_amine_stripper_outputs_lbhr,
    }

    df_total_inputs_row = pd.DataFrame([total_inputs_row], columns=df_amine_stripper_inputs.columns)
    df_total_outputs_row = pd.DataFrame([total_outputs_row], columns=df_amine_stripper_outputs.columns)

    inputs_label = pd.DataFrame([['Inputs'] + [''] * (df_amine_stripper_inputs.shape[1] - 1)],
                            columns=df_amine_stripper_inputs.columns)
    outputs_label = pd.DataFrame([['Outputs'] + [''] * (df_amine_stripper_outputs.shape[1] - 1)],
                             columns=df_amine_stripper_outputs.columns)

    df_amine_stripper = pd.concat([
    inputs_label,
    df_amine_stripper_inputs,
    df_total_inputs_row,
    outputs_label,
    df_amine_stripper_outputs,
    df_total_outputs_row
    ], ignore_index=True)

    print(f'Amine Stripper:')
    print(df_amine_stripper)

    # Determine utilities for amine gas treating
    annual_amine_gas_treating_utilities = {
    key: (
        utility_data["amount"] * amine_solution_circulation_rate_GPD * 365,
        utility_data["unit"]
    )
    for key, utility_data in amine_gas_treating_utility_data.items()
    }

    # CLAUS PROCESS
    claus_process_inputs = {
    'lb/hr': {},
    }

    claus_process_outputs = {
    'lb/hr': {},
    'MJ/yr': {}
    }

    claus_process_cuts = {
        'H2S in acid gas': amine_stripper_outputs,
        'NH3 in acid gas': amine_stripper_outputs
    }

    properties = ['lb/hr']

    for prop in properties:
        for cut, source_df in claus_process_cuts.items():
            claus_process_inputs[prop][cut] = source_df[prop][cut]

    moles_hr_H2S_in_acid_gas = claus_process_inputs['lb/hr']['H2S in acid gas'] * conversion_parameters['lb_to_g'] / fixed_parameters['molecular_weight']['H2S']
    moles_hr_O2 = moles_hr_H2S_in_acid_gas / 2
    moles_hr_H2O = moles_hr_H2S_in_acid_gas 

    claus_process_inputs['lb/hr']['O2'] = moles_hr_O2 * fixed_parameters['molecular_weight']['O2'] / conversion_parameters['lb_to_g']
    claus_process_outputs['lb/hr']['S'] = claus_process_inputs['lb/hr']['H2S in acid gas'] * fixed_parameters['molecular_weight']['S'] / fixed_parameters['molecular_weight']['H2S'] 
    claus_process_outputs['lb/hr']['H2O'] = moles_hr_H2O * fixed_parameters['molecular_weight']['H2O'] / conversion_parameters['lb_to_g']

    claus_process_outputs['MJ/yr']['S'] = claus_process_outputs['lb/hr']['S'] * conversion_parameters['lb_to_kg'] * 8760 * fixed_parameters['specific_energy']['sulfur']

    df_claus_process_inputs = (
    pd.DataFrame(claus_process_inputs)
    .reset_index()
    .rename(columns={'index': 'Cut'})
    .sort_values(by='Cut', key=lambda col: col.map(sort_key))
    .reset_index(drop=True)
    )

    df_claus_process_outputs = (
    pd.DataFrame(claus_process_outputs)
    .reset_index()
    .rename(columns={'index': 'Cut'})
    .sort_values(by='Cut', key=lambda col: col.map(sort_key))
    .reset_index(drop=True)
    )

    total_claus_process_inputs_lbhr = df_claus_process_inputs['lb/hr'].replace(',', '', regex=True).apply(pd.to_numeric, errors='coerce').sum()
    total_claus_process_outputs_lbhr = df_claus_process_outputs['lb/hr'].replace(',', '', regex=True).apply(pd.to_numeric, errors='coerce').sum()

    total_inputs_row = {
    'Cut': 'Total',
    'lb/hr': total_claus_process_inputs_lbhr
    }

    total_outputs_row = {
    'Cut': 'Total',
    'lb/hr': total_claus_process_outputs_lbhr,
    }

    df_total_inputs_row = pd.DataFrame([total_inputs_row], columns=df_claus_process_inputs.columns)
    df_total_outputs_row = pd.DataFrame ([total_outputs_row], columns=df_claus_process_outputs.columns)

    inputs_label = pd.DataFrame([['Inputs'] + [''] * (df_claus_process_inputs.shape[1] - 1)],
                            columns=df_claus_process_inputs.columns)
    outputs_label = pd.DataFrame([['Outputs'] + [''] * (df_claus_process_outputs.shape[1] - 1)],
                             columns=df_claus_process_outputs.columns)

    df_claus_process = pd.concat([
    inputs_label,
    df_claus_process_inputs,
    df_total_inputs_row,
    outputs_label,
    df_claus_process_outputs,
    df_total_outputs_row
    ], ignore_index=True)

    print(f'Claus process:')
    print(df_claus_process)

    sulfur_production_long_tons_day = claus_process_outputs['lb/hr']['S'] * 24 /2240

    # Determine utilities claus process
    annual_claus_process_utilities = {
    key: (
        utility_data["amount"] * sulfur_production_long_tons_day * 365,
        utility_data["unit"]
    )
    for key, utility_data in claus_process_utility_data.items()
    }


    # Determine utilities claus process
    annual_SCOT_utilities = {
    key: (
        utility_data["amount"] * sulfur_production_long_tons_day * 365,
        utility_data["unit"]
    )
    for key, utility_data in SCOT_utility_data.items()
    }

def column_diameter(flow_rate, throughput):

    diameter = math.sqrt((4 * flow_rate) / (math.pi * throughput))
    return diameter

# KEROSENE SOLVENT EXTRACTION -------------------------------------------------------------------------------------------------------------------
if aromatics_removal_technique == 'solvent_extraction':
    
    # Initialize solvent extraction inputs and outputs
    solvent_extraction_inputs = {
    'API': {},
    'Specific Gravity': {},
    'Characterization Factor': {},
    'MW': {},
    'BPCD': {},
    'lb/hr from bbl/day': {},
    'lb/hr': {},
    'MJ/yr': {},
    'wt% S': {},
    'wppm S': {},
    'lb/hr S': {},
    'wt %': {},
    'wt% H': {},
    'lb/hr H': {},
    'wppm N': {},
    'lb/hr N': {},
    'vol% paraffins': {},
    'vol% naphthenes': {},
    'vol% aromatics': {},
    'wt% paraffins': {},
    'wt% naphthenes': {},
    'wt% aromatics': {},
    'wt% naphthalenes': {},
    'wt% polyaromatics': {},
    'wt% monoaromatics': {},
    'vol% polyaromatics': {},
    'vol% monoaromatics': {},
    }

    solvent_extraction_outputs = {
    'API': {},
    'Specific Gravity': {},
    'Characterization Factor': {},
    'MW': {},
    'BPCD': {},
    'lb/hr from bbl/day': {},
    'lb/hr': {},
    'MJ/yr': {},
    'wt% S': {},
    'wppm S': {},
    'lb/hr S': {},
    'wt %': {},
    'wt% H': {},
    'lb/hr H': {},
    'wppm N': {},
    'lb/hr N': {},
    'vol% paraffins': {},
    'vol% naphthenes': {},
    'vol% aromatics': {},
    'wt% paraffins': {},
    'wt% naphthenes': {},
    'wt% aromatics': {},
    'wt% naphthalenes': {},
    'wt% polyaromatics': {},
    'wt% monoaromatics': {},
    'vol% polyaromatics': {},
    'vol% monoaromatics': {},

    }

    # Feed for extractor is kerosine
    solvent_extraction_cuts = {
        'Kerosine': distillation_outputs
    }

    properties = ['BPCD', 'API', 'Specific Gravity', 'Characterization Factor', 'MW', 'lb/hr', 'MJ/yr', 'wt% S', 'lb/hr S', 'wt% H', 'lb/hr H', 'wppm N', 'lb/hr N', 'vol% aromatics', 'wt% naphthalenes']

    for prop in properties:
        for cut, source_df in solvent_extraction_cuts.items():
            solvent_extraction_inputs[prop][cut] = source_df[prop][cut]

    total_solvent_extraction_bpcd = sum(solvent_extraction_inputs['BPCD'].values())
    total_solvent_extraction_inputs_lb_hr_sulfur = pd.to_numeric(pd.Series(solvent_extraction_inputs['lb/hr S'].values()), errors='coerce').fillna(0).sum()
    total_solvent_extraction_inputs_wppm_nitrogen = sum(solvent_extraction_inputs['wppm N'].values())

    if solvent_choice == 'sulfolane':
        solvent_data = solvent_extraction_data['sulfolane_solvent_extraction']

    # Determing solvent input
    solvent_to_feed_wt_ratio = user_inputs['solvent_extraction_parameters']['solvent_to_feed_ratio']

    kerosene_density = solvent_extraction_inputs['Specific Gravity']['Kerosine'] * density_water    # kg/L
    solvent_density = solvent_data['density']  # kg/L

    # Base kerosene flow in gal/hr (scalar)
    kerosene_flow_gal_hr = solvent_extraction_inputs['lb/hr']['Kerosine'] * conversion_parameters['lb_to_kg'] * kerosene_density * conversion_parameters['L_to_gal']

    # Extractor dimensions
    tray_eff = 0.75
    tray_spacing = 0.6096
    throughput = 1000 # gal/hr per ft2

    def max_solvent_to_feed_ratio(
        max_extractor_prod_height_diam,
        tray_spacing,
        tray_eff,
        kerosene_flow_gal_hr,
        solvent_density,
        solvent_extraction_inputs,
        throughput,
        conversion_parameters
    ):
        """
        Calculate the maximum solvent-to-feed ratio that keeps
        extractor dimensions within the allowed maximum.

        Returns
        -------
        tuple
        (min_SF_ratio, max_SF_ratio)
        """
        # Step 1: Start with a guess range of SF ratio
        SF_min_guess, SF_max_guess = 0.1, 10.0  # reasonable bounds

        def compute_prod_height_diam(SF_ratio):
            # Solvent recycle flow
            solvent_recycle_lb_hr = solvent_extraction_inputs['lb/hr']['Kerosine'] * SF_ratio

            # Column flow rate including solvent recycle
            extractor_column_gal_hr = kerosene_flow_gal_hr + solvent_recycle_lb_hr * conversion_parameters['lb_to_kg'] * solvent_density * conversion_parameters['L_to_gal']

            # Column diameter in ft
            diameter_ft = column_diameter(extractor_column_gal_hr, throughput)
            diameter_m = diameter_ft * conversion_parameters['ft_to_m']

            # Number of stages from SF ratio
            num_stages = interpolate_num_stages(SF_ratio)

            # Column height
            column_height_m = tray_spacing * num_stages / tray_eff

            # Product height × diameter
            prod_height_diam = column_height_m * diameter_m ** 1.5
            return prod_height_diam

        # Step 2: Use a simple bisection method to find SF_ratio where prod_height_diam = max limit
        tol = 1e-3
        low, high = solvent_to_feed_wt_ratio[0], solvent_to_feed_wt_ratio[1]
        SF_solution = SF_max_guess

        while high - low > tol:
            mid = (low + high) / 2
            prod = compute_prod_height_diam(mid)
            if prod > max_extractor_prod_height_diam:
                high = mid
            else:
                low = mid
                SF_solution = mid

        return SF_solution
    
    max_SF = max_solvent_to_feed_ratio(
    max_extractor_prod_height_diam=150,
    tray_spacing=0.6096,
    tray_eff=0.75,
    kerosene_flow_gal_hr=kerosene_flow_gal_hr,
    solvent_density=solvent_density,
    solvent_extraction_inputs=solvent_extraction_inputs,
    throughput=throughput,
    conversion_parameters=conversion_parameters
    )

    print(f"Maximum solvent-to-feed ratio = {max_SF:.2f}")


    if isinstance(solvent_to_feed_wt_ratio, (list, tuple)) and len(solvent_to_feed_wt_ratio) == 2:
        # Interpolate for both values
        stages_sf = [interpolate_num_stages(solvent_to_feed_wt_ratio[i]) for i in (0, 1)]
        extractor_num_stages = (min(stages_sf), max(stages_sf))

        solvent_recycle_lb_hr = (
        solvent_extraction_inputs['lb/hr']['Kerosine'] * solvent_to_feed_wt_ratio[0],
        solvent_extraction_inputs['lb/hr']['Kerosine'] * solvent_to_feed_wt_ratio[1])

        extractor_column_gal_hr_flow_rate = (
        kerosene_flow_gal_hr + solvent_recycle_lb_hr[0] * conversion_parameters['lb_to_kg'] * solvent_density * conversion_parameters['L_to_gal'],
        kerosene_flow_gal_hr + solvent_recycle_lb_hr[1] * conversion_parameters['lb_to_kg'] * solvent_density * conversion_parameters['L_to_gal']
        )

        # Column diameter as a range
        extractor_column_diameter_ft = (
        column_diameter(extractor_column_gal_hr_flow_rate[0], throughput),
        column_diameter(extractor_column_gal_hr_flow_rate[1], throughput)
        )
        extractor_column_diameter_m = (
        extractor_column_diameter_ft[0] * conversion_parameters['ft_to_m'],
        extractor_column_diameter_ft[1] * conversion_parameters['ft_to_m']
        )

        # Column height as a range
        extractor_column_height_m = (
        tray_spacing * extractor_num_stages[0] / tray_eff,
        tray_spacing * extractor_num_stages[1] / tray_eff
        )

        print(f"Solvent to feed ratio: {solvent_to_feed_wt_ratio[0]:.2f} – {solvent_to_feed_wt_ratio[1]:.2f}")
        print(f"Extractor number of stages: {extractor_num_stages[0]:.2f} – {extractor_num_stages[1]:.2f}")
        print(f"Extractor: diameter = {extractor_column_diameter_m[0]:.2f} – {extractor_column_diameter_m[1]:.2f} m, "
          f"height = {extractor_column_height_m[0]:.2f} – {extractor_column_height_m[1]:.2f} m")
        
        extractor_stack_height_m = []
        for h in extractor_column_height_m:
            available_height = h - (2 * tray_spacing)
            n_trays = math.floor(available_height / tray_spacing)
            stack_height = n_trays * tray_spacing
            extractor_stack_height_m.append(stack_height)
            
            extractor_prod_height_diam_min, extractor_prod_height_diam_max = product_height_diameter(extractor_column_height_m, extractor_column_diameter_m)

        extractor_prod_height_diam = [extractor_prod_height_diam_min, extractor_prod_height_diam_max]
        max_extractor_prod_height_diam = 150

        extractor_column_height_m = list(extractor_column_height_m)

        for i in (0, 1):
            if extractor_prod_height_diam[i] > max_extractor_prod_height_diam:
                print('Recalculate solvent-feed ratio and number of stages to obtain reasonable extractor dimensions')
        
                # Calculate max allowable column height for this index
                max_extractor_column_height_m = max_extractor_prod_height_diam / (extractor_column_diameter_m[i] ** 1.5)
                extractor_column_height_m[i] = max_extractor_column_height_m

                # Update number of stages (upper bound)
                extractor_num_stages = list(extractor_num_stages)  # convert tuple to list
                extractor_num_stages[i] = extractor_column_height_m[i] * tray_eff / tray_spacing
        
                # Update solvent-to-feed ratio corresponding to new number of stages
                solvent_to_feed_wt_ratio = list(solvent_to_feed_wt_ratio)  # convert tuple to list
                solvent_to_feed_wt_ratio[i] = interpolate_SF_ratio(extractor_num_stages[i])
                
                solvent_recycle_lb_hr[i] = solvent_extraction_inputs['lb/hr']['Kerosine'] * solvent_to_feed_wt_ratio[i]

        print(f"Solvent to feed ratio: {solvent_to_feed_wt_ratio[0]:.2f} – {solvent_to_feed_wt_ratio[1]:.2f}")
        print(f"Extractor number of stages: {extractor_num_stages[0]:.2f} – {extractor_num_stages[1]:.2f}")
        print(f"Extractor: diameter = {extractor_column_diameter_m[0]:.2f} – {extractor_column_diameter_m[1]:.2f} m, "
          f"height = {extractor_column_height_m[0]:.2f} – {extractor_column_height_m[1]:.2f} m")
        


    else:
        print('Single value case')
        extractor_num_stages = interpolate_num_stages(solvent_to_feed_wt_ratio)
        print(f"Extractor number of stages: {extractor_num_stages:.2f}")

        extractor_column_diameter_ft = column_diameter(extractor_column_gal_hr_flow_rate, throughput)
        extractor_column_diameter_m = extractor_column_diameter_ft * conversion_parameters['ft_to_m']
        extractor_column_height_m = tray_spacing * extractor_num_stages / tray_eff

        print(f"Extractor: diameter = {extractor_column_diameter_m:.2f} m, "
            f"height = {extractor_column_height_m:.2f} m")
        
        

    if isinstance(solvent_to_feed_wt_ratio, (list, tuple)) and len(solvent_to_feed_wt_ratio) == 2:

        # solvent_recycle_lb_hr as a range (min, max)
        solvent_recycle_lb_hr = (
        solvent_extraction_inputs['lb/hr']['Kerosine'] * solvent_to_feed_wt_ratio[0],
        solvent_extraction_inputs['lb/hr']['Kerosine'] * solvent_to_feed_wt_ratio[1]
        )
    else:
        # Single value
        solvent_recycle_lb_hr = solvent_extraction_inputs['lb/hr']['Kerosine'] * solvent_to_feed_wt_ratio


    if isinstance(solvent_recycle_lb_hr, (list, tuple)):
        # Flow rate as a range (min, max)
        extractor_column_gal_hr_flow_rate = (
        kerosene_flow_gal_hr + solvent_recycle_lb_hr[0] * conversion_parameters['lb_to_kg'] * solvent_density * conversion_parameters['L_to_gal'],
        kerosene_flow_gal_hr + solvent_recycle_lb_hr[1] * conversion_parameters['lb_to_kg'] * solvent_density * conversion_parameters['L_to_gal']
        )
    else:
        # Single value
        extractor_column_gal_hr_flow_rate = kerosene_flow_gal_hr + solvent_recycle_lb_hr * conversion_parameters['lb_to_kg'] * solvent_density * conversion_parameters['L_to_gal']


    # Determine aromatics content before and after extraction
    bulk_density = solvent_extraction_inputs['Specific Gravity']['Kerosine'] * density_water    # kg/L

    aromatics_density = 0.88
    polyaromatics_density = 1.02  # kg/L
    monoaromatics_density = 0.869 # kg/L

    solvent_extraction_inputs['wt% polyaromatics']['Kerosine'] = solvent_extraction_inputs['wt% naphthalenes']['Kerosine']
    solvent_extraction_inputs['vol% polyaromatics']['Kerosine'] = solvent_extraction_inputs['wt% polyaromatics']['Kerosine'] * bulk_density / polyaromatics_density

    solvent_extraction_inputs['vol% monoaromatics']['Kerosine'] = solvent_extraction_inputs['vol% aromatics']['Kerosine'] - solvent_extraction_inputs['vol% polyaromatics']['Kerosine']
    solvent_extraction_inputs['wt% monoaromatics']['Kerosine'] =  solvent_extraction_inputs['vol% monoaromatics']['Kerosine'] * monoaromatics_density / bulk_density
    
    solvent_extraction_inputs['wt% aromatics']['Kerosine'] =  solvent_extraction_inputs['vol% aromatics']['Kerosine'] * aromatics_density / bulk_density

    solvent_extraction_inputs['wt% monoaromatics']['Kerosine'] =  solvent_extraction_inputs['vol% monoaromatics']['Kerosine'] * monoaromatics_density / bulk_density
    solvent_extraction_inputs['wt% aromatics']['Kerosine'] =  solvent_extraction_inputs['vol% aromatics']['Kerosine'] * aromatics_density / bulk_density

    # ----- Aromatics saturation -------------------------------------------------------------------

    #solvent_extraction_outputs['lb/hr']['Aromatics Product'] = solvent_data['recovery_ratio_aromatics'] * solvent_extraction_inputs['wt% aromatics']['Kerosine'] / 100 * solvent_extraction_inputs['lb/hr']['Kerosine'] 
    #solvent_extraction_outputs['MJ/yr']['Aromatics Product'] = solvent_extraction_outputs['lb/hr']['Aromatics Product'] * conversion_parameters['lb_to_kg'] * 8760 * fixed_parameters['specific_energy']['aromatics']

    #solvent_extraction_outputs['lb/hr']['Raffinate'] = solvent_extraction_inputs['lb/hr']['Kerosine']  - solvent_extraction_outputs['lb/hr']['Aromatics Product'] 

    #solvent_extraction_outputs['wt% aromatics']['Raffinate'] = (1 - solvent_data['mass_purity_nonaromatics']) * 100


    # ----- Naphthalene removal -------------------------------------------------------------------
    solvent_extraction_outputs['wt% polyaromatics']['Raffinate'] = 0
    solvent_extraction_outputs['wt% monoaromatics']['Raffinate'] = solvent_extraction_inputs['wt% monoaromatics']['Kerosine']
    solvent_extraction_outputs['wt% aromatics']['Raffinate'] = solvent_extraction_outputs['wt% monoaromatics']['Raffinate'] + solvent_extraction_outputs['wt% polyaromatics']['Raffinate']

    solvent_extraction_outputs['lb/hr']['Raffinate'] = (
    solvent_extraction_inputs['lb/hr']['Kerosine'] *
    (1 - solvent_extraction_inputs['wt% polyaromatics']['Kerosine'] / 100))

    solvent_extraction_outputs['lb/hr']['Aromatics Product'] = solvent_extraction_inputs['lb/hr']['Kerosine'] * solvent_extraction_inputs['wt% polyaromatics']['Kerosine'] / 100
    solvent_extraction_outputs['MJ/yr']['Aromatics Product'] = solvent_extraction_outputs['lb/hr']['Aromatics Product'] * conversion_parameters['lb_to_kg'] * 8760 * fixed_parameters['specific_energy']['aromatics']
    # ---------------------------------------------------------------------------------------------

    # Estimate change in specific energy due to aromatics saturation
    delta_specific_energy_aromatics = (
    (solvent_extraction_inputs['wt% aromatics']['Kerosine'] / 100) * fixed_parameters['specific_energy']['aromatics']
    + ((100 - solvent_extraction_inputs['wt% aromatics']['Kerosine']) / 100) * fixed_parameters['specific_energy']['alkanes']
    ) - (
    (solvent_extraction_outputs['wt% aromatics']['Raffinate'] / 100) * fixed_parameters['specific_energy']['aromatics']
    + ((100 - solvent_extraction_outputs['wt% aromatics']['Raffinate']) / 100) * fixed_parameters['specific_energy']['alkanes']
    )

    specific_energy_raffinate = fixed_parameters['specific_energy']['kerosene'] - delta_specific_energy_aromatics 

    solvent_extraction_outputs['MJ/yr']['Raffinate'] = solvent_extraction_outputs['lb/hr']['Raffinate'] * conversion_parameters['lb_to_kg'] * 8760 * specific_energy_raffinate

    # Compute initial non-aromatics fraction CHECKK
    non_aromatics_initial_pct = 100 - solvent_extraction_inputs['wt% aromatics']['Kerosine']
    non_aromatics_final_pct = 100 - solvent_extraction_outputs['wt% aromatics']['Raffinate']

    # Estimate delta density due to aromatics removal
    delta_density_aromatics = (
    (solvent_extraction_inputs['wt% aromatics']['Kerosine'] / 100) * fixed_parameters['density']['aromatics']
    + ((100 - solvent_extraction_inputs['wt% aromatics']['Kerosine']) / 100) * fixed_parameters['density']['alkanes']
    ) - (
    (solvent_extraction_outputs['wt% aromatics']['Raffinate'] / 100) * fixed_parameters['density']['aromatics']
    + ((100 - solvent_extraction_outputs['wt% aromatics']['Raffinate']) / 100) * fixed_parameters['density']['alkanes']
    )

    # Correct initial density to match given actual density
    # So we apply delta to actual density
    density_raffinate = bulk_density - delta_density_aromatics

    # Convert to API gravity
    # Convert kg/m3 to specific gravity at 60F (15.56C)
    solvent_extraction_outputs['Specific Gravity']['Raffinate'] = density_raffinate / density_water

    solvent_extraction_outputs['lb/hr from bbl/day']['Raffinate'] = find_lbhr_conversion(solvent_extraction_outputs['Specific Gravity']['Raffinate'], density_conv_table)             # Obtain BPCD to lb/hr conversion from density conversion table
    solvent_extraction_outputs['BPCD']['Raffinate'] = solvent_extraction_outputs['lb/hr']['Raffinate'] / solvent_extraction_outputs['lb/hr from bbl/day']['Raffinate']

    raffinate_gal_hr = solvent_extraction_outputs['lb/hr']['Raffinate'] * conversion_parameters['lb_to_kg'] * density_raffinate * conversion_parameters['L_to_gal'] 
    solvent_extraction_outputs['lb/hr']['Aromatics Product'] = solvent_extraction_inputs['lb/hr']['Kerosine'] - solvent_extraction_outputs['lb/hr']['Raffinate']
    solvent_extraction_outputs['MJ/yr']['Aromatics Product'] = solvent_extraction_outputs['lb/hr']['Aromatics Product'] * conversion_parameters['lb_to_kg'] * 8760 * fixed_parameters['specific_energy']['aromatics']
    

    # --- Solvent regenerator and stripper calculations ---
    # Aromatics in extract (scalar)
    aromatics_in_extract_lb_hr = (
    solvent_extraction_inputs['lb/hr']['Kerosine'] * solvent_extraction_inputs['wt% aromatics']['Kerosine'] -
    solvent_extraction_outputs['lb/hr']['Raffinate'] * solvent_extraction_outputs['wt% aromatics']['Raffinate']
    )

        # Extract lb/hr including recycled solvent
    if isinstance(solvent_recycle_lb_hr, (list, tuple)):
        extract_lb_hr = (
        solvent_extraction_outputs['lb/hr']['Aromatics Product'] + solvent_recycle_lb_hr[0],
        solvent_extraction_outputs['lb/hr']['Aromatics Product'] + solvent_recycle_lb_hr[1]
        )
    else:
        extract_lb_hr = solvent_extraction_outputs['lb/hr']['Aromatics Product'] + solvent_recycle_lb_hr

    # Extract weight fraction of aromatics
    if isinstance(extract_lb_hr, (list, tuple)):
        extract_wt_perc_aromatics = tuple(aromatics_in_extract_lb_hr / x for x in extract_lb_hr)
    else:
        extract_wt_perc_aromatics = aromatics_in_extract_lb_hr / extract_lb_hr

    # Extract density
    if isinstance(extract_wt_perc_aromatics, (list, tuple)):
        extract_density = tuple(p * fixed_parameters['density']['aromatics'] + (1 - p) * fixed_parameters['density']['alkanes'] for p in extract_wt_perc_aromatics)
    else:
        extract_density = extract_wt_perc_aromatics * fixed_parameters['density']['aromatics'] + (1 - extract_wt_perc_aromatics) * fixed_parameters['density']['alkanes']

    # Extract flow in gal/hr
    if isinstance(extract_density, (list, tuple)):
        extract_gal_hr = tuple(
            lb_hr * conversion_parameters['lb_to_kg'] * density * conversion_parameters['L_to_gal'] +
            (solvent_recycle_lb_hr[i] if isinstance(solvent_recycle_lb_hr, (list, tuple)) else solvent_recycle_lb_hr) *
            conversion_parameters['lb_to_kg'] * solvent_density * conversion_parameters['L_to_gal']
            for i, (lb_hr, density) in enumerate(zip(extract_lb_hr, extract_density))
        )
    else:
        extract_gal_hr = extract_lb_hr * conversion_parameters['lb_to_kg'] * extract_density * conversion_parameters['L_to_gal'] + \
                     (solvent_recycle_lb_hr if not isinstance(solvent_recycle_lb_hr, (list, tuple)) else solvent_recycle_lb_hr[0]) * \
                     conversion_parameters['lb_to_kg'] * solvent_density * conversion_parameters['L_to_gal']

    # Solvent regenerator geometry
    solvent_regenerator_num_trays = solvent_data['solvent_regenerator_num_stages'] / solvent_data['tray_efficiency']
    solvent_regenerator_column_height_m = tray_spacing * solvent_regenerator_num_trays

    if isinstance(extract_gal_hr, (list, tuple)):
        solvent_regenerator_column_diameter_m = tuple(column_diameter(flow, throughput) * conversion_parameters['ft_to_m'] for flow in extract_gal_hr)
    else:
        solvent_regenerator_column_diameter_m = column_diameter(extract_gal_hr, throughput) * conversion_parameters['ft_to_m']

    print(f'Solvent regenerator: diameter = {solvent_regenerator_column_diameter_m if isinstance(solvent_regenerator_column_diameter_m, (int,float)) else f"{solvent_regenerator_column_diameter_m[0]:.2f} - {solvent_regenerator_column_diameter_m[1]:.2f}"} m, '
      f'height = {solvent_regenerator_column_height_m:.2f} m')

    # Stripper geometry
    stripper_num_trays = solvent_data['stripper_num_stages'] / solvent_data['tray_efficiency']
    stripper_column_height_m = tray_spacing * stripper_num_trays

    if isinstance(extract_gal_hr, (list, tuple)):
        stripper_column_diameter_m = tuple(column_diameter(flow, throughput) * conversion_parameters['ft_to_m'] for flow in extract_gal_hr)
    else:
        stripper_column_diameter_m = column_diameter(extract_gal_hr, throughput) * conversion_parameters['ft_to_m']

    print(f'Stripper: diameter = {stripper_column_diameter_m if isinstance(stripper_column_diameter_m, (int,float)) else f"{stripper_column_diameter_m[0]:.2f} - {stripper_column_diameter_m[1]:.2f}"} m, '
      f'height = {stripper_column_height_m:.2f} m')

    raffinate_wash_column_diameter_ft = column_diameter(raffinate_gal_hr, throughput)
    raffinate_wash_column_diameter_m = raffinate_wash_column_diameter_ft * conversion_parameters['ft_to_m']
    washer_num_trays = solvent_data['washer_num_stages'] / solvent_data['tray_efficiency']
    raffinate_wash_column_height_m = tray_spacing * washer_num_trays
    print(f'Raffinate wash column: diameter = {raffinate_wash_column_diameter_m:,.2f} m, height = {raffinate_wash_column_height_m:,.2f} m')

    df_solvent_extraction_inputs = pd.DataFrame(solvent_extraction_inputs).reset_index().rename(columns={'index': 'Cut'})
    df_solvent_extractiont_outputs = pd.DataFrame(solvent_extraction_outputs).reset_index().rename(columns={'index': 'Cut'})
    df_solvent_extraction_outputs = df_solvent_extractiont_outputs[df_solvent_extractiont_outputs['Cut'] != 'total gas flow']
    df_solvent_extraction_outputs = df_solvent_extraction_outputs[df_solvent_extractiont_outputs['Cut'] != 'HC gas']
    
    inputs_label = pd.DataFrame([['Inputs'] + [''] * (df_solvent_extraction_inputs.shape[1] - 1)],
                            columns=df_solvent_extraction_inputs.columns)
    outputs_label = pd.DataFrame([['Outputs'] + [''] * (df_solvent_extraction_outputs.shape[1] - 1)],
                             columns=df_solvent_extraction_outputs.columns)

    total_solvent_extraction_inputs_bpcd = df_solvent_extraction_inputs['BPCD'].apply(pd.to_numeric, errors='coerce').sum()
    total_solvent_extraction_outputs_bpcd = df_solvent_extraction_outputs['BPCD'].apply(pd.to_numeric, errors='coerce').sum()

    total_solvent_extraction_inputs_lbhr = df_solvent_extraction_inputs['lb/hr'].replace(',', '', regex=True).apply(pd.to_numeric, errors='coerce').sum()
    total_solvent_extraction_outputs_lbhr = df_solvent_extraction_outputs['lb/hr'].replace(',', '', regex=True).apply(pd.to_numeric, errors='coerce').sum()

    total_solvent_extraction_inputs_sulfur = df_solvent_extraction_inputs['lb/hr S'].apply(pd.to_numeric, errors='coerce').sum()
    total_solvent_extraction_outputs_sulfur = df_solvent_extraction_outputs['lb/hr S'].apply(pd.to_numeric, errors='coerce').sum()
    total_solvent_extraction_inputs_nitrogen = df_solvent_extraction_inputs['lb/hr N'].apply(pd.to_numeric, errors='coerce').sum()

    total_solvent_extraction_outputs_energy_content = df_solvent_extraction_outputs['MJ/yr'].apply(pd.to_numeric, errors='coerce').sum()

    total_inputs_row = {
    'Cut': 'Total',
    'BPCD': total_solvent_extraction_inputs_bpcd,
    'lb/hr': total_solvent_extraction_inputs_lbhr,
    'lb/hr S': total_solvent_extraction_inputs_sulfur,
    'lb/hr N': total_solvent_extraction_inputs_nitrogen,
    }

    total_outputs_row = {
    'Cut': 'Total',
    'BPCD': total_solvent_extraction_outputs_bpcd,
    'lb/hr': total_solvent_extraction_outputs_lbhr,
    'MJ/yr': total_solvent_extraction_outputs_energy_content,
    'lb/hr S': total_solvent_extraction_outputs_sulfur
    }

    df_total_inputs_row = pd.DataFrame([total_inputs_row], columns=df_solvent_extraction_inputs.columns)
    df_total_outputs_row = pd.DataFrame([total_outputs_row], columns=df_solvent_extraction_outputs.columns)

    df_solvent_extraction = pd.concat([
    inputs_label,
    df_solvent_extraction_inputs,
    df_total_inputs_row,
    outputs_label,
    df_solvent_extraction_outputs,
    df_total_outputs_row
    ], ignore_index=True)

    df_solvent_extraction.fillna('', inplace=True)

    print(df_solvent_extraction)

    feed_ton_yr = solvent_extraction_inputs['lb/hr']['Kerosine'] * conversion_parameters['lb_to_ton'] * 8760

    if solvent_choice == 'sulfolane':
        # Determine utilities for extraction

        annual_solvent_extraction_utilities = {
        key: (
        (
            [a * feed_ton_yr for a in utility_data["amount"]]
            if isinstance(utility_data["amount"], list)
            else utility_data["amount"] * feed_ton_yr
            ),
            utility_data["unit"]
        )
        for key, utility_data in hydrotreatment_utility_data.items()
    }
        
# Evaluate equipment costs
CEPCI_2022 = get_cepci_value(2022, cepci_values)
CEPCI_2016 = get_cepci_value(2016, cepci_values)
CEPCI_2002 = get_cepci_value(2002, cepci_values)
CEPCI_2000 = get_cepci_value(2000, cepci_values)
CEPCI_2005 = get_cepci_value(2005, cepci_values)
CEPCI_2020 = get_cepci_value(2020, cepci_values)

# CHANGE!!!!!

# Always calculate atmospheric distillation cost
atm_flow_MBPCD = distillation_inputs['BPCD']['Crude oil'] / 1000
atm_cost = atmospheric_distillation_unit_cost_interp(atm_flow_MBPCD) * 1000000 * CEPCI_2022 / CEPCI_2005

# Initialize equipment and costs
equipments_SR_kerosene = ['Atmospheric Distillation Unit']
costs_SR_kerosene = [atm_cost]
equipments_treated_kerosene = ['Atmospheric Distillation Unit']
costs_treated_kerosene = [atm_cost]

# Conditionally add vacuum distillation
if refinery_type == 'cracking':
    vac_flow_MBPCD = sum(distillation_outputs['BPCD'][key] for key in [
        'Light Vacuum Gas Oil', 'Heavy Vacuum Gas Oil', '...Heavy Vacuum Gas Oil', 'Vacuum Residue'
    ]) / 1000

    vac_cost = vacuum_distillation_unit_cost_interp(vac_flow_MBPCD) * 1000000 * CEPCI_2022 / CEPCI_2005
    equipments_SR_kerosene.append('Vacuum Distillation Unit')
    costs_SR_kerosene.append(vac_cost)

    equipments_treated_kerosene.append('Vacuum Distillation Unit')
    costs_treated_kerosene.append(vac_cost)


# Conditionally add hydrotreatment and amine gas treating units
Cost_HydrogenStart = 0
if aromatics_removal_technique == "hydrotreatment":
    kerosene_MBPCD = distillation_outputs['BPCD']['Kerosine'] / 1000
    hydro_cost_lower = hydrotreator_cost_interp_lower(kerosene_MBPCD) * 1000000 * CEPCI_2022 / CEPCI_2005
    hydro_cost_upper = hydrotreator_cost_interp_upper(kerosene_MBPCD) * 1000000 * CEPCI_2022 / CEPCI_2005
    hydro_cost = (hydro_cost_lower, hydro_cost_upper)
    amine_cost = amine_gas_treating_unit_cost_interp(amine_solution_circulation_rate_GPM) * 1000000 * CEPCI_2022 / CEPCI_2005

    claus_cost = claus_unit_cost_interp(sulfur_production_long_tons_day) * 1000000 * CEPCI_2022 / CEPCI_2005  
    SCOT_cost = claus_unit_cost_interp(sulfur_production_long_tons_day) * 1000000 * CEPCI_2022 / CEPCI_2005      # Multiply by a factor of 2 to include SCOT unit for >99% recovery
    equipments_treated_kerosene.extend(['Kerosene Hydrotreator', 'Amine Gas Treating Unit', 'Claus Unit', 'SCOT'])
    costs_treated_kerosene.extend([hydro_cost, amine_cost, claus_cost, SCOT_cost])

    if hydrogen_source == 'SMR':
        Cost_HydrogenStart = 0

        H2_input_mscf_day = H2_scf_per_yr / 1000000 /365 

        print('H2 input million scf per day')
        print(H2_input_mscf_day)
        SMR_cost = SMR_cost_interp(H2_input_mscf_day) * 1000000 * CEPCI_2022 / CEPCI_2005
        equipments_treated_kerosene.extend(['SMR'])
        costs_treated_kerosene.extend([SMR_cost])

        H2_production_lb_per_yr = H_cons_lb_per_hr * 8760
        H2_production_kg_per_yr = H2_production_lb_per_yr * conversion_parameters['lb_to_kg']

        print('H2_production_lb_per_yr')
        print(H2_production_lb_per_yr)

        annual_SMR_utilities = {
        key: (
            utility_data["amount"] * H2_production_lb_per_yr,
            utility_data["unit"]
            )
        for key, utility_data in SMR_utility_data.items()}

    if hydrogen_source == 'electrolysis':

        Cost_HydrogenStart = 0
        electrolyzer_energy = 0.1135 # kWh per scf of H2

        electrolyzer_power_kWh_per_yr = H2_scf_per_yr * electrolyzer_energy 
        print('Electrolyzer power (kWh/yr)')
        print(electrolyzer_power_kWh_per_yr)
        electrolyzer_power_kW = electrolyzer_power_kWh_per_yr /8760
        electrolyzer_power_MW = electrolyzer_power_kW / 1000
        print('Electrolyzer power (MW)')
        print(electrolyzer_power_MW)
        electrolyzer_cost_per_kW = electrolyzer_cost_interp( electrolyzer_power_MW )
        print('Electrolyzer cost ($/kW)')
        print(electrolyzer_cost_per_kW)
        electrolyzer_cost = electrolyzer_cost_per_kW * electrolyzer_power_kW * CEPCI_2022 / CEPCI_2020
        print(electrolyzer_cost)
        equipments_treated_kerosene.extend(['Electrolyzer'])
        costs_treated_kerosene.extend([electrolyzer_cost])

        H2_production_lb_per_yr = H2_scf_per_yr * conversion_parameters['H2_scf_to_lb']

        annual_electrolyzer_utilities = {
        key: (
            utility_data["amount"] * H2_production_lb_per_yr,
            utility_data["unit"]
            )
        for key, utility_data in electrolyzer_utility_data.items()}

        print(annual_electrolyzer_utilities)


if aromatics_removal_technique == "solvent_extraction":
    # --- Extractor costs ---
    if 'extractor_column_height_m' in locals():
    #    # Determine if inputs are ranges
    #    def distillation_column_cost_range(diameter, height, num_stages, column_cost_func, tray_cost_func, tray_eff):
    #        # Helper to extract min/max from scalar or tuple
    #        def get_min_max(val):
    #            if isinstance(val, (list, tuple)):
    #                return val[0], val[1]
    #            else:
    #                return val, val

    #        d0, d1 = get_min_max(diameter)
    #        h0, h1 = get_min_max(height)
    #        n0, n1 = get_min_max(num_stages)

            # If everything is scalar, return scalar
    #        if d0 == d1 and h0 == h1 and n0 == n1:
    #            return column_cost_func(d0) * h0 * CEPCI_2022 / CEPCI_2002 + \
    #                tray_cost_func(d0) * n0 / tray_eff * CEPCI_2022 / CEPCI_2002
    #        else:
    #            return (
    #                column_cost_func(d0) * h0 * CEPCI_2022 / CEPCI_2002 + tray_cost_func(d0) * n0 / tray_eff * CEPCI_2022 / CEPCI_2002,
    #                column_cost_func(d1) * h1 * CEPCI_2022 / CEPCI_2002 + tray_cost_func(d1) * n1 / tray_eff * CEPCI_2022 / CEPCI_2002
    #            )

    #    extractor_cost = distillation_column_cost_range(extractor_column_diameter_m, extractor_column_height_m, extractor_num_stages, 
    #                                packed_column_cost_interp, valve_tray_cost_interp, tray_eff)
        
    #    solvent_regenerator_cost = distillation_column_cost_range(solvent_regenerator_column_diameter_m, solvent_regenerator_column_height_m,
    #                                         solvent_data['solvent_regenerator_num_stages'], 
    #                                         packed_column_cost_interp, valve_tray_cost_interp, tray_eff)

    #    raffinate_washer_cost = packed_column_cost_interp(raffinate_wash_column_diameter_m) * raffinate_wash_column_height_m * CEPCI_2022/CEPCI_2002 + \
    #                           valve_tray_cost_interp(raffinate_wash_column_diameter_m) * washer_num_trays * CEPCI_2022/CEPCI_2002

    #    stripper_cost = distillation_column_cost_range(stripper_column_diameter_m, stripper_column_height_m, stripper_num_trays,
    #                               packed_column_cost_interp, sieve_tray_cost_interp, tray_eff)


        extractor_prod_height_diam_min = min(extractor_prod_height_diam)
        extractor_prod_height_diam_max = max(extractor_prod_height_diam)

       # extractor_stack_height_m = tray_stack_height(extractor_column_height_m, 0.6)
     #  print(extractor_stack_height_m)

        extractor_tray_stack_prod_height_diam = product_height_diameter(extractor_stack_height_m, extractor_column_diameter_m)
        extractor_tray_stack_prod_height_diam_min = min(extractor_tray_stack_prod_height_diam)
        extractor_tray_stack_prod_height_diam_max = max(extractor_tray_stack_prod_height_diam)


        print(f'extractor_prod_height_diam, extractor_prod_height_diam_max')
        print(extractor_prod_height_diam_min, extractor_prod_height_diam_max)

        print(f'extractor_tray_stack_prod_height_diam_min, extractor_tray_stack_prod_height_diam_max')
        print(extractor_tray_stack_prod_height_diam_min, extractor_tray_stack_prod_height_diam_max)

        # Function to compute cost for a single value
        def carbon_steel_column_cost(prod):
            if 4 < prod < 150:
                return 545000 * (prod / 100)**0.57
            else:
                print(f'Value {prod} is outside the valid range')
                return None

        def carbon_steel_tray_cost(prod):
            if 1.5  < prod < 66:
                return 167000 * (prod / 66)**0.39
            elif 66 < prod < 250:
                return 167000 * (prod / 66)**0.78
            else:
                print(f'Value {prod} is outside the valid range')
                return None
            
        # Calculate min and max
        extractor_column_cost_min = carbon_steel_column_cost(extractor_prod_height_diam_min)
        extractor_column_cost_max = carbon_steel_column_cost(extractor_prod_height_diam_max)
        extractor_column_cost = [extractor_column_cost_min, extractor_column_cost_max]

        extractor_trays_cost_min = carbon_steel_column_cost(extractor_tray_stack_prod_height_diam_min)
        extractor_trays_cost_max = carbon_steel_column_cost(extractor_tray_stack_prod_height_diam_max)
        extractor_trays_cost = [extractor_trays_cost_min, extractor_trays_cost_max]

        print('extractor_column_cost')

        print(extractor_column_cost)
        print(extractor_trays_cost)

        # Total equipment cost with 6% added for heat exchangers
        if isinstance(extractor_cost, tuple):
            total_equip_cost = tuple((sum(cost) * 1.06 for cost in zip(extractor_cost, stripper_cost, solvent_regenerator_cost, (raffinate_washer_cost, raffinate_washer_cost))))
            heat_exchangers_cost = tuple(total - sum(cost) for total, cost in zip(total_equip_cost, zip(extractor_cost, stripper_cost, solvent_regenerator_cost, (raffinate_washer_cost, raffinate_washer_cost))))
        else:
            total_equip_cost = (extractor_cost + stripper_cost + solvent_regenerator_cost + raffinate_washer_cost) * 1.06
            heat_exchangers_cost = total_equip_cost - (extractor_cost + stripper_cost + solvent_regenerator_cost + raffinate_washer_cost)

        equipments_treated_kerosene.extend(['Extractor', 'Stripper', 'Solvent regenerator', 'Raffinate wash column', 'Heat exchangers'])
        costs_treated_kerosene.extend([extractor_cost, stripper_cost, solvent_regenerator_cost, raffinate_washer_cost, heat_exchangers_cost])

# Number of processing steps
if aromatics_removal_technique == 'none':
    if refinery_type == 'hydroskimming':
        processing_steps = 1
    if refinery_type == 'cracking':
        processing_steps = 2
else:
    if refinery_type == 'hydroskimming':
        processing_steps = 1 #2
    if refinery_type == 'cracking':
        processing_steps = 2 #3


annual_crude_input_cost = distillation_inputs['BPCD']['Crude oil'] * input_price_data['crude_oil_price'][crude_selection] * 365
print(f'\nAnnual crude oil input cost = $ {annual_crude_input_cost:,.2f}')

# Determine which utility sets to include

distillation_keys = set(annual_atmospheric_dist_utilities.keys())
if refinery_type == 'cracking':
    distillation_keys |= set(annual_vacuum_dist_utilities.keys())
if aromatics_removal_technique == "hydrotreatment":
    treatment_keys = distillation_keys
    treatment_keys |= set(annual_hydrotreatment_utilities.keys())
    treatment_keys |= set(annual_amine_gas_treating_utilities.keys())
    treatment_keys |= set(annual_claus_process_utilities.keys())
    if hydrogen_source == 'SMR':
        treatment_keys |= set(annual_SMR_utilities.keys())
    if hydrogen_source == 'electrolysis':
        treatment_keys |= set(annual_electrolyzer_utilities.keys())
if aromatics_removal_technique == 'solvent_extraction':
    treatment_keys = distillation_keys
    treatment_keys |= set(annual_solvent_extraction_utilities.keys())
if aromatics_removal_technique == 'none':
    treatment_keys = distillation_keys


# Define all relevant utility sources based on configuration

utility_sources = {
    'Atmospheric Distillation': annual_atmospheric_dist_utilities,
    'Vacuum Distillation': annual_vacuum_dist_utilities if refinery_type == 'cracking' else {},
    'Hydrotreatment': annual_hydrotreatment_utilities if aromatics_removal_technique == "hydrotreatment" else {},
    'Amine Gas Treating': annual_amine_gas_treating_utilities if aromatics_removal_technique == "hydrotreatment" else {},
    'Claus Process': annual_claus_process_utilities if aromatics_removal_technique == "hydrotreatment" else {},
    'SCOT Process': annual_SCOT_utilities if aromatics_removal_technique == "hydrotreatment" else {},
    'SMR': annual_SMR_utilities if aromatics_removal_technique == "hydrotreatment" and hydrogen_source == 'SMR' else {},
    'Electrolysis': annual_electrolyzer_utilities if aromatics_removal_technique == "hydrotreatment" and hydrogen_source == 'electrolysis' else {},
    'Solvent extraction': annual_solvent_extraction_utilities if aromatics_removal_technique == "solvent_extraction" else {},
}

def is_nonzero(val):
    if isinstance(val, np.ndarray):
        return (val != 0).any()      # collapse NumPy arrays
    elif isinstance(val, (list, tuple)):
        return any(is_nonzero(x) for x in val)  # recursive check for nested lists/arrays
    else:
        return val != 0              # scalar case

utility_sources = {
    k: v for k, v in utility_sources.items()
    if any(is_nonzero(val) for val in v.values())
}

def get_utility_description(key):
    for source in utility_sources.values():
        if key in source:
            return source[key][1]
    return ''


# Build utility data dictionary for SR kerosene
SR_kero_utility_data = {
    'Utility': [f"{key} ({get_utility_description(key)})" for key in distillation_keys]
}


SR_units = ["Atmospheric Distillation", "Vacuum Distillation"]

for name, source in utility_sources.items():
    if name in SR_units:
        # Keep the same: take the first element if scalar/range is not needed
        SR_kero_utility_data[name] = [source.get(key, (0, ''))[0] for key in distillation_keys]

# Build utility data dictionary for treated kerosene
treated_kero_utility_data = {
    'Utility': [f"{key} ({get_utility_description(key)})" for key in treatment_keys]
}

for name, source in utility_sources.items():
    # Keep the same: take the first element if scalar/range is not needed
    treated_kero_utility_data[name] = [source.get(key, (0, ''))[0] for key in treatment_keys]


# --- Add NG for steam production as a *new row*, not new column ---

# Conversion: 2.71–3.44 GJ NG / ton steam → MMBtu NG / lb steam
GJ_to_MMBtu = 0.947817
GJ_range = np.array([2.71, 3.44])
MMBtu_per_ton = GJ_range * GJ_to_MMBtu
MMBtu_per_lb = MMBtu_per_ton / 2000  # convert to per lb steam
NG_range = list(MMBtu_per_lb)  # [0.001282, 0.00163]


# Append new utility row
SR_kero_utility_data['Utility'].append('NG for steam production (MMBtu/lb steam)')

# For each process, calculate total NG use based on its steam use
for process in [k for k in SR_kero_utility_data.keys() if k != 'Utility']:
    steam_index = SR_kero_utility_data['Utility'].index('steam (lb)')
    steam_usage = SR_kero_utility_data[process][steam_index]  # lb steam
    total_NG = np.outer(steam_usage, NG_range)
    SR_kero_utility_data[process].append(total_NG)

# Append new utility row
treated_kero_utility_data['Utility'].append('NG for steam production (MMBtu/lb steam)')

# For each process, calculate total NG use based on its steam use
for process in [k for k in treated_kero_utility_data.keys() if k != 'Utility']:
    steam_index = treated_kero_utility_data['Utility'].index('steam (lb)')
    steam_usage = treated_kero_utility_data[process][steam_index]  # lb steam
    total_NG = np.outer(steam_usage, NG_range)
    treated_kero_utility_data[process].append(total_NG)


# Compute totals handling ranges
def sum_row_with_ranges(row):
    total_min = 0.0
    total_max = 0.0

    for val in row:
        if val is None or (isinstance(val, float) and np.isnan(val)):
            continue  # skip missing
        
        if isinstance(val, (list, tuple, np.ndarray)):
            arr = np.array(val, dtype=float).flatten()

            if arr.size == 1:
                total_min += arr.item()
                total_max += arr.item()
            elif arr.size >= 2:
                total_min += arr[0]
                total_max += arr[1]
            else:
                # fallback if array is empty
                continue

        else:
            # treat single number as both min and max
            total_min += float(val)
            total_max += float(val)

    return [total_min, total_max]

df_utilities = pd.DataFrame(treated_kero_utility_data)
totals = df_utilities.drop(columns='Utility').apply(sum_row_with_ranges, axis=1)
df_utilities['Total'] = totals
treated_kero_utility_data['Total'] = {key: total for key, total in zip(treatment_keys, totals)}

df_utilities_SR_kerosene = pd.DataFrame(SR_kero_utility_data)
totals = df_utilities_SR_kerosene.drop(columns='Utility').apply(sum_row_with_ranges, axis=1)
df_utilities_SR_kerosene['Total'] = totals
SR_kero_utility_data['Total'] = {key: total for key, total in zip(distillation_keys, totals)}

formatted_df_utilities = df_utilities.copy()
for col in formatted_df_utilities.select_dtypes(include='number').columns:
    formatted_df_utilities[col] = formatted_df_utilities[col].apply(lambda x: f"{round(x, 2):,}")

# Calculate steam system cost for total treated kerosene system
total_steam = treated_kero_utility_data['Total'].get('steam', 0)

if isinstance(total_steam, (list, tuple, np.ndarray)):
    # Convert each end of range to lb/hr
    steam_lb_hr = [val / 8760 for val in total_steam]
else:
    steam_lb_hr = total_steam / 8760
if isinstance(steam_lb_hr, list):
    steam_system_cost = [105 * lb_hr * CEPCI_2022 / CEPCI_2005 for lb_hr in steam_lb_hr]
else:
    steam_system_cost = 105 * steam_lb_hr * CEPCI_2022 / CEPCI_2005

equipments_treated_kerosene.append('Steam system')
costs_treated_kerosene.append(steam_system_cost)

# Calculate steam system cost for distillation units
total_distillation_steam = SR_kero_utility_data['Total'].get('steam', 0)

# Handle both single values and ranges
if isinstance(total_distillation_steam, (list, tuple, np.ndarray)):
    # Convert each end of range to lb/hr
    steam_distillation_lb_hr = [val / 8760 for val in total_distillation_steam]
else:
    steam_distillation_lb_hr = total_distillation_steam / 8760

# Compute steam system cost for scalar or range
if isinstance(steam_distillation_lb_hr, list):
    distillation_steam_system_cost = [105 * lb_hr * CEPCI_2022 / CEPCI_2005 for lb_hr in steam_distillation_lb_hr]
else:
    distillation_steam_system_cost = 105 * steam_distillation_lb_hr * CEPCI_2022 / CEPCI_2005

equipments_SR_kerosene.append('Distillation steam system')
costs_SR_kerosene.append(distillation_steam_system_cost)

# Calculate cooling water circ. system cost
total_cooling_water = treated_kero_utility_data['Total'].get('cooling_water_crclt', 0)

if isinstance(total_cooling_water, (list, tuple, np.ndarray)):
    # Convert each end of range to gpm
    cooling_water_circ_gpm = [val / 60 / 8760 for val in total_cooling_water]
else:
    cooling_water_circ_gpm = total_cooling_water / 60 / 8760

# Apply same cost formula to range or scalar
if isinstance(cooling_water_circ_gpm, list):
    cooling_water_system_cost = [
        130 * gpm * CEPCI_2022 / CEPCI_2005 for gpm in cooling_water_circ_gpm
    ]
else:
    cooling_water_system_cost = 130 * cooling_water_circ_gpm * CEPCI_2022 / CEPCI_2005

equipments_treated_kerosene.append('Coling water circulation system')
costs_treated_kerosene.append(cooling_water_system_cost)

# Calculate cooling water circ. for distillation units
distillation_cooling_water = SR_kero_utility_data['Total'].get('cooling_water_crclt', 0)

if isinstance(distillation_cooling_water, (list, tuple, np.ndarray)):
    # Convert each end of range to gpm
    cooling_water_circ_gpm = [val / 60 / 8760 for val in distillation_cooling_water]
else:
    cooling_water_circ_gpm = distillation_cooling_water / 60 / 8760

# Apply same cost formula to range or scalar
if isinstance(cooling_water_circ_gpm, list):
    distillation_cooling_water_system_cost = [
        130 * gpm * CEPCI_2022 / CEPCI_2005 for gpm in cooling_water_circ_gpm
    ]
else:
    distillation_cooling_water_system_cost = 130 * cooling_water_circ_gpm * CEPCI_2022 / CEPCI_2005

equipments_SR_kerosene.append('Distillation cooling water circulation system')
costs_SR_kerosene.append(distillation_cooling_water_system_cost)

equip_range = any(isinstance(c, (tuple, list, np.ndarray)) for c in costs_treated_kerosene)

if equip_range:
    costs_min, costs_max = [], []
    for c in costs_treated_kerosene:
        if isinstance(c, (tuple, list, np.ndarray)):  # Range
            c_min, c_max = c[0], c[1]  # Safe for arrays/lists/tuples
        else:  # Single value - use same for min/max
            c_min = c_max = c
        costs_min.append(float(c_min))
        costs_max.append(float(c_max))

    # Build range-aware DataFrame
    treated_kerosene_equipment_costs = pd.DataFrame({
        'Equipment': equipments_treated_kerosene,
        'Cost Min (USD)': costs_min,
        'Cost Max (USD)': costs_max
    })

    # Calculate totals
    total_purchase_equipment_cost_min = sum(costs_min)
    total_purchase_equipment_cost_max = sum(costs_max)
    total_purchase_equipment_cost = (total_purchase_equipment_cost_min, total_purchase_equipment_cost_max)

else:
    # No ranges - single column
    equipment_costs = pd.DataFrame({
        'Equipment': equipments_treated_kerosene,
        'Cost (USD)': costs_treated_kerosene
    })

    total_purchase_equipment_cost = sum(costs_treated_kerosene)

    equipment_costs['Cost (USD)'] = equipment_costs['Cost (USD)'].apply(lambda x: f"${x:,.0f}")
    print(equipment_costs)
    print(f"\nTotal Purchase Equipment Cost: ${total_purchase_equipment_cost:,.0f}")

SR_kerosene_equip_range = any(isinstance(c, (tuple, list, np.ndarray)) for c in costs_SR_kerosene)

if SR_kerosene_equip_range:
    costs_min, costs_max = [], []
    for c in costs_SR_kerosene:
        if isinstance(c, (tuple, list, np.ndarray)):  # Range
            c_min, c_max = c[0], c[1]  # Safe for arrays/lists/tuples
        else:  # Single value - use same for min/max
            c_min = c_max = c
        costs_min.append(float(c_min))
        costs_max.append(float(c_max))

    # Build range-aware DataFrame
    equipment_costs = pd.DataFrame({
        'Equipment': equipments_SR_kerosene,
        'Cost Min (USD)': costs_min,
        'Cost Max (USD)': costs_max
    })

    # Calculate totals
    total_purchase_equipment_cost_SR_min = sum(costs_min)
    total_purchase_equipment_cost_SR_max = sum(costs_max)
    total_purchase_equipment_cost_SR = (total_purchase_equipment_cost_SR_min, total_purchase_equipment_cost_SR_max)
else:
    # No ranges - single column
    equipment_costs = pd.DataFrame({
        'Equipment': equipments_SR_kerosene,
        'Cost (USD)': costs_SR_kerosene
    })

    total_purchase_equipment_cost_SR = sum(costs_SR_kerosene)

OutputLightNaphtha = distillation_outputs['BPCD']['Light Naphtha'] * 365
OutputHeavyNaphtha = (distillation_outputs['BPCD']['Heavy Naphtha'] + distillation_outputs['BPCD']['...Heavy Naphtha']) * 365
OutputLightGasOil = distillation_outputs['BPCD']['Light Gas Oil'] * 365
OutputHeavyGasOil = distillation_outputs['BPCD']['Heavy Gas Oil'] * 365
OutputAtmResidue = distillation_outputs.get('BPCD', {}).get('Atm Residuee', 0) * 365
OutputLightVacuumGasOil = distillation_outputs.get('BPCD', {}).get('Light Vacuum Gas Oil', 0) * 365
OutputHeavyVacuumGasOil = (
    distillation_outputs.get('BPCD', {}).get('Heavy Vacuum Gas Oil', 0) +
    distillation_outputs.get('BPCD', {}).get('...Heavy Vacuum Gas Oil', 0)
) * 365
OutputVacuumResidues = distillation_outputs.get('BPCD', {}).get('Vacuum Residue', 0) * 365
OutputLightends = 0

distillation_kerosene_AF = distillation_outputs['MJ/yr']['Kerosine'] / total_dist_outputs_energy_content
crude_oil_kerosene_AF = distillation_outputs['MJ/yr']['Kerosine'] / (distillation_inputs['lb/hr']['Crude oil'] * 8760 * conversion_parameters['lb_to_kg'] * fixed_parameters['specific_energy']['crude'])

if aromatics_removal_technique == "hydrotreatment":
    OutputKerosene = hydrotreatment_outputs['BPCD']['HT Kerosine'] * 365
    MJ_per_yr_OutputKerosene = hydrotreatment_outputs['MJ/yr']['HT Kerosine']
    ton_per_yr_OutputKerosene = hydrotreatment_outputs['lb/hr']['HT Kerosine'] * 8760 * conversion_parameters['lb_to_ton']

    OutputPropane = hydrotreatment_outputs['BPCD']['C3 and lighter'] * 365 * hydrotreatment_outputs_wt_perc_C3H8 / (hydrotreatment_outputs_wt_perc_CH4 + hydrotreatment_outputs_wt_perc_C2H6 + hydrotreatment_outputs_wt_perc_C3H8)
    OutputSulfur = claus_process_outputs['lb/hr']['S'] * conversion_parameters['lb_to_kg'] * 8760
    OutputBTX = 0

    OutputSRKerosene = distillation_outputs['BPCD']['Kerosine'] * 365
    
    hydrotreatment_HT_kerosene_AF = hydrotreatment_outputs['MJ/yr']['HT Kerosine'] / (hydrotreatment_outputs['MJ/yr']['HT Kerosine'] + hydrotreatment_outputs['MJ/yr']['C3 and lighter'] + hydrotreatment_outputs['MJ/yr']['iC4'] + hydrotreatment_outputs['MJ/yr']['nC4'] + claus_process_outputs['MJ/yr']['S'])

    crude_oil_HT_kerosene_AF = crude_oil_kerosene_AF * hydrotreatment_outputs['MJ/yr']['HT Kerosine'] / distillation_outputs['MJ/yr']['Kerosine'] 

    distillation_HT_kerosene_AF = distillation_kerosene_AF * hydrotreatment_outputs['MJ/yr']['HT Kerosine'] / distillation_outputs['MJ/yr']['Kerosine'] 

elif aromatics_removal_technique == "solvent_extraction":
    OutputKerosene = solvent_extraction_outputs['BPCD']['Raffinate'] * 365
    MJ_per_yr_OutputKerosene = solvent_extraction_outputs['MJ/yr']['Raffinate']
    ton_per_yr_OutputKerosene = solvent_extraction_outputs['lb/hr']['Raffinate'] * 8760 * conversion_parameters['lb_to_ton']

    OutputPropane = 0
    OutputSulfur = 0
    OutputBTX = solvent_extraction_outputs['lb/hr']['Aromatics Product'] * 8760 * conversion_parameters['lb_to_kg'] / fixed_parameters['density']['aromatics'] * conversion_parameters['L_to_gal']

    OutputSRKerosene = distillation_outputs['BPCD']['Kerosine'] * 365

    solvent_extraction_raffinate_AF = solvent_extraction_outputs['MJ/yr']['Raffinate'] / (solvent_extraction_outputs['MJ/yr']['Raffinate'] + solvent_extraction_outputs['MJ/yr']['Aromatics Product'])
    distillation_raffinate_kerosene_AF = distillation_kerosene_AF * solvent_extraction_outputs['MJ/yr']['Raffinate'] / distillation_outputs['MJ/yr']['Kerosine'] 
    crude_oil_raffinate_kerosene_AF = crude_oil_kerosene_AF * solvent_extraction_outputs['MJ/yr']['Raffinate'] / distillation_outputs['MJ/yr']['Kerosine'] 

else:
    OutputSRKerosene = distillation_outputs['BPCD']['Kerosine'] * 365
    OutputKerosene = distillation_outputs['BPCD']['Kerosine'] * 365
    OutputPropane = 0
    OutputSulfur = 0
    OutputBTX = 0

    MJ_per_yr_OutputKerosene = distillation_outputs['MJ/yr']['Kerosine']
    ton_per_yr_OutputKerosene = distillation_outputs['lb/hr']['Kerosine'] * 8760 * conversion_parameters['lb_to_ton']

Cost_NGStart = input_price_data['variable_input_prices']['NG'] # Price of NG $/MMBtu
Cost_PowerStart = input_price_data['variable_input_prices'][electricity_choice] # Price of electricity $/kWh
Cost_CoolingWaterStart = input_price_data['variable_input_prices']['cooling_water']# Price of cooling water $/gallon
Cost_FeedWater = input_price_data['variable_input_prices']['feed_water']
Cost_BoilerFeedWater = input_price_data['variable_input_prices']['boiler_feed_water']
Cost_ByproductsStart = 0
CostLightNaphtha = input_price_data['refinery_petroleum_product_prices']['naphtha'] 
CostHeavyNaphtha = input_price_data['refinery_petroleum_product_prices']['naphtha']
CostLightGasOil = input_price_data['refinery_petroleum_product_prices']['light_gas_oil'] 
CostHeavyGasOil = input_price_data['refinery_petroleum_product_prices']['heavy_gas_oil']
CostAtmResidue = input_price_data['refinery_petroleum_product_prices']['residues'] 
CostLightVacuumGasOil = input_price_data['refinery_petroleum_product_prices']['vacuum_gas_oil']
CostHeavyVacuumGasOil = input_price_data['refinery_petroleum_product_prices']['vacuum_gas_oil']
CostVacuumResidues = input_price_data['refinery_petroleum_product_prices']['residues']
CostPropane = input_price_data['refinery_petroleum_product_spot_prices']['propane'] * conversion_parameters['bbl_to_gallon']
CostSulfur = input_price_data['refinery_petroleum_product_spot_prices']['sulfur'] 
# Assume BTX compostion of 19:49:32 https://onlinelibrary.wiley.com/doi/abs/10.1002/0471238961.02202419230505.a01
price_BTX = input_price_data['refinery_petroleum_product_spot_prices']['benzene'] * 0.19 + input_price_data['refinery_petroleum_product_spot_prices']['toluene'] * 0.49 + input_price_data['refinery_petroleum_product_spot_prices']['xylene'] * 0.32
print(f'Price of BTX: ${price_BTX:,.2f}')
CostBTX = [0, price_BTX]


utility_cost_map = {
    'fuel (MMBtu)': Cost_NGStart,
    'power (kWh)': Cost_PowerStart,
    'H2 (scf)': Cost_HydrogenStart,
    'cooling_water (gal)': Cost_CoolingWaterStart,
    'feed_water (lb)': Cost_FeedWater,
    'boiler_feed_water (gal)': Cost_BoilerFeedWater,
    'feed_gas (MMBtu)': Cost_NGStart,

    }

raw_material_cost_map = {
    'catalyst_replacement (kg)': input_price_data['raw_material_prices']['NiMo_Al2_O3']
}

key_map = {
    'fuel (MMBtu)': 'fuel',
    'power (kWh)': 'power',
    'H2 (scf)': 'H2',
    'feed_water (lb)': 'feed_water',
    'cooling_water_crclt (gal)': 'cooling_water_crclt',
    'catalyst_replacement (kg)': 'catalyst_replacement',
    'waste_heat_steam (lb)': 'waste_heat_steam',
    'boiler_feed_water (gal)': 'boiler_feed_water',
    'feed_gas (MMBtu)': 'feed_gas',
}


print(f"\nUtilities Cost")
total_energy_cost_min = 0
total_energy_cost_max = 0
total_energy_cost = 0
treated_kero_utility_data['Total Cost'] = {}
SR_kero_utility_data['Total Cost'] = {}
has_range = False  # Track whether we encounter any range usage

for util, cost in utility_cost_map.items():
    if util not in key_map:
        continue 

    mapped_key = key_map[util]
    if mapped_key not in treated_kero_utility_data['Total']:
        continue  # Prevents KeyError

    usage = treated_kero_utility_data['Total'][key_map[util]]

    if isinstance(usage, list):  # Range
        has_range = True
        usage_min, usage_max = usage
        total_cost_min = usage_min * cost
        total_cost_max = usage_max * cost
        total_energy_cost_min += total_cost_min
        total_energy_cost_max += total_cost_max
        print(f"{util} Cost Range: [{total_cost_min:,.2f}, {total_cost_max:,.2f}]")
        treated_kero_utility_data['Total Cost'][util] = [total_cost_min, total_cost_max]

    else:  # Scalar
        total_cost = usage * cost
        print(f"{util} Cost: {total_cost:,.2f}")

        if has_range:
            # If we've already seen a range, treat scalar as equal contribution to min and max
            total_energy_cost_min += total_cost
            total_energy_cost_max += total_cost
        else:
            # For now, accumulate scalar-only total
            total_energy_cost += total_cost
        treated_kero_utility_data['Total Cost'][util] = total_cost

# After loop, decide which totals to keep
if has_range:
    print(f"\nTotal Utility Cost Range: [{total_energy_cost_min:,.2f}, {total_energy_cost_max:,.2f}]")
    total_energy_cost = [total_energy_cost_min, total_energy_cost_max]
else:
    print(f"\nTotal Utility Cost: {total_energy_cost:,.2f}")

total_distillation_energy_cost_min = 0
total_distillation_energy_cost_max = 0
total_distillation_energy_cost = 0
SR_kero_utility_data['Total Cost'] = {}
has_range = False  # Track whether we encounter any range usage

for util, cost in utility_cost_map.items():
    if util not in key_map:
        continue 

    mapped_key = key_map[util]

    if mapped_key not in SR_kero_utility_data['Total']:
        continue  # Prevents KeyError

    usage_distillation = SR_kero_utility_data['Total'][key_map[util]]

    if isinstance(usage_distillation, list):  # Range
        has_range = True
        usage_distillation_min, usage_distillation_max = usage_distillation

        total_cost_min = usage_distillation_min * cost
        total_cost_max = usage_distillation_max * cost

        total_distillation_energy_cost_min += total_cost_min
        total_distillation_energy_cost_max += total_cost_max
        SR_kero_utility_data['Total Cost'][util] = [total_cost_min, total_cost_max]
        
    else:  # Scalar
        total_distillation_cost = usage_distillation * cost

        if has_range:
            # If we've already seen a range, treat scalar as equal contribution to min and max
            total_distillation_energy_cost_min += total_distillation_cost
            total_distillation_energy_cost_max += total_distillation_cost
        else:
            # For now, accumulate scalar-only total
            total_distillation_energy_cost += total_distillation_cost
        SR_kero_utility_data['Total Cost'][util] = total_distillation_cost

# After loop, decide which totals to keep
if has_range:
    print(f"\nTotal Distillation Utility Cost Range: [{total_distillation_energy_cost_min:,.2f}, {total_distillation_energy_cost_max:,.2f}]")
    total_distillation_energy_cost = [total_distillation_energy_cost_min, total_distillation_energy_cost_max]
else:
    print(f"\nTotal Distillation Utility Cost: {total_distillation_energy_cost:,.2f}")


print(f"\nRaw Material Cost")
total_raw_material_cost_min = 0
total_raw_material_cost_max = 0
total_raw_material_cost = 0
  
raw_material_total_column = []
for raw_mat, cost in raw_material_cost_map.items():
    if raw_mat not in key_map:
        continue 

    mapped_key = key_map[raw_mat]
    if mapped_key not in treated_kero_utility_data['Total']:
        continue  # Prevents KeyError

    usage = treated_kero_utility_data['Total'][key_map[raw_mat]]

    if isinstance(usage, list):  # Range
        has_range = True
        usage_min, usage_max = usage
        total_cost_min = usage_min * cost
        total_cost_max = usage_max * cost
        print(f"{raw_mat}: Usage = [{usage_min:.2f}, {usage_max:.2f}], "
              f"Unit Cost = {cost:.4f}, Total Cost = [{total_cost_min:,.2f}, {total_cost_max:,.2f}]")
        total_raw_material_cost_min += total_cost_min
        total_raw_material_cost_max += total_cost_max
        treated_kero_utility_data['Total Cost'][raw_mat] = [total_cost_min, total_cost_max]

    else:  # Scalar
        total_cost = usage * cost
        print(f"{raw_mat}: Usage = {usage:.2f}, Unit Cost = {cost:.4f}, Total Cost = {total_cost:,.2f}")
        if has_range:
            # If we already have ranges, add scalar contribution to both min and max
            total_raw_material_cost_min += total_cost
            total_raw_material_cost_max += total_cost

        else:
            total_raw_material_cost += total_cost
        treated_kero_utility_data['Total Cost'][util] = total_cost

# Decide which totals to output
if has_range:
    print(f"\nTotal Raw Material Cost Range: [{total_raw_material_cost_min:,.2f}, {total_raw_material_cost_max:,.2f}]")
    total_raw_material_cost = [total_raw_material_cost_min, total_raw_material_cost_max]
else:
    print(f"\nTotal Raw Material Cost: {total_raw_material_cost:,.2f}")


total_distillation_raw_material_cost_min = 0
total_distillation_raw_material_cost_max = 0
total_distillation_raw_material_cost = 0
  
raw_material_distillation_total_column = []
for raw_mat, cost in raw_material_cost_map.items():
    if raw_mat not in key_map:
        continue 

    mapped_key = key_map[raw_mat]
    if mapped_key not in SR_kero_utility_data['Total']:
        continue  # Prevents KeyError

    usage_distillation = SR_kero_utility_data['Total'][mapped_key]

    if isinstance(usage, list):  # Range
        has_range = True
        usage_min, usage_max = usage_distillation
        total_distillation_cost_min = usage_min * cost
        total_distillation_cost_max = usage_max * cost

        total_distillation_raw_material_cost_min += total_distillation_cost_min
        total_distillation_raw_material_cost_max += total_distillation_cost_max
        SR_kero_utility_data['Total Cost'][raw_mat] = [total_distillation_cost_min, total_distillation_cost_max]

    else:  # Scalar
        total_distillation_cost = usage_distillation * cost
        if has_range:
            # If we already have ranges, add scalar contribution to both min and max
            total_distillation_raw_material_cost_min += total_distillation_cost
            total_distillation_raw_material_cost_max += total_distillation_cost

        else:
            total_distillation_raw_material_cost += total_distillation_cost
        SR_kero_utility_data['Total Cost'][util] = total_distillation_cost

# Decide which totals to output
if has_range:
    print(f"\nTotal Distillation Raw Material Cost Range: [{total_distillation_raw_material_cost_min:,.2f}, {total_distillation_raw_material_cost_max:,.2f}]")
    total_distillation_raw_material_cost = [total_distillation_raw_material_cost_min, total_distillation_raw_material_cost_max]
else:
    print(f"\nTotal Distillation Raw Material Cost: {total_distillation_raw_material_cost:,.2f}")

formatted_df_utilities['Total Cost'] = formatted_df_utilities['Utility'].map(
    treated_kero_utility_data['Total Cost']
)

print(formatted_df_utilities.to_string(index=False))

if aromatics_removal_technique == 'solvent_extraction':
    residence_time = solvent_data['LLE_residence_time'] * 3

    if isinstance(solvent_recycle_lb_hr, (list, tuple)):
        # Initial solvent input and investment as a range
        initial_solvent_input_lb = tuple(lb * residence_time for lb in solvent_recycle_lb_hr)
        initial_solvent_investment = tuple(
            lb * conversion_parameters['lb_to_ton'] * input_price_data['raw_material_prices']['sulfolane']
            for lb in initial_solvent_input_lb
        )
        print(f'Initial solvent investment: ${initial_solvent_investment[0]:,.2f} - ${initial_solvent_investment[1]:,.2f}')
    else:
        # Single value
        initial_solvent_input_lb = solvent_recycle_lb_hr * residence_time
        initial_solvent_investment = initial_solvent_input_lb * conversion_parameters['lb_to_ton'] * input_price_data['raw_material_prices']['sulfolane']
        print(f'Initial solvent investment: ${initial_solvent_investment:,.2f}')
else:
    initial_solvent_investment = 0


plant_capacity_kg_day = distillation_inputs['lb/hr']['Crude oil'] * fixed_parameters['conversion_parameters']['lb_to_kg'] * 24

def extract(value, index):
    """Return min (index=0) or max (index=1) if value is a 2-element range, else return scalar.""" 
    if isinstance(value, (list, tuple)) and len(value) == 2: 
        return value[index] 
    return value 

# Initialize empty lists
additional_capital_costs = []
working_capital_cost = []
cost_of_land = []
cost_of_buildings = []
total_capital_investment = []
fixed_capital_investment = []

additional_capital_costs_SR = []
working_capital_cost_SR = []
cost_of_land_SR = []
cost_of_buildings_SR = []
total_capital_investment_SR = []
fixed_capital_investment_SR = []

print(total_purchase_equipment_cost)

print(total_purchase_equipment_cost_SR)
# Collect min/max values
for i in [0, 1]:
    additional_capital_costs_i, working_capital_cost_i, cost_of_land_i, cost_of_buildings_i, \
    total_capital_investment_i, fixed_capital_investment_i = calculate_additional_capital_costs(
        extract(total_purchase_equipment_cost, i), 
        financial_data, 
        extract(initial_solvent_investment, i) 
    )

    additional_capital_costs.append(additional_capital_costs_i)
    working_capital_cost.append(working_capital_cost_i)
    cost_of_land.append(cost_of_land_i)
    cost_of_buildings.append(cost_of_buildings_i)
    total_capital_investment.append(total_capital_investment_i)
    fixed_capital_investment.append(fixed_capital_investment_i)

    additional_capital_costs_SR_i, working_capital_cost_SR_i, cost_of_land_SR_i, cost_of_buildings_SR_i, \
    total_capital_investment_SR_i, fixed_capital_investment_SR_i = calculate_additional_capital_costs(
        extract(total_purchase_equipment_cost_SR, i), 
        financial_data, 0.0)

    additional_capital_costs_SR.append(additional_capital_costs_SR_i)
    working_capital_cost_SR.append(working_capital_cost_SR_i)
    cost_of_land_SR.append(cost_of_land_SR_i)
    cost_of_buildings_SR.append(cost_of_buildings_SR_i)
    total_capital_investment_SR.append(total_capital_investment_SR_i)
    fixed_capital_investment_SR.append(fixed_capital_investment_SR_i)

# --- Calculate operating costs once (return ranges if inputs are ranges) ---
total_product_costs = []
non_feedstock_raw_mat_cost = []
utilities_cost = []
total_fixed_oper_cost = []
total_variable_oper_cost = []
total_operating_costs = []
direct_operating_costs = []
variable_operating_costs = []

total_product_costs_SR = []
non_feedstock_raw_mat_cost_SR = []
utilities_cost_SR = []
total_fixed_oper_cost_SR = []
total_variable_oper_cost_SR = []
total_operating_costs_SR = []
direct_operating_costs_SR = []
variable_operating_costs_SR = []

for i in [0, 1]:

   # non_feedstock_raw_mat_cost_i, utilities_cost_i, total_fixed_oper_cost_i, total_variable_oper_cost_i, \
   #     total_operating_costs_i, direct_operating_costs_i, variable_operating_costs_i = calculate_additional_operating_costs_gary_handwerk(
   #     extract(fixed_capital_investment, i), 
   #     financial_data, 
   #     extract(total_raw_material_cost, i),
   #     extract(total_energy_cost, i),
   #     extract(annual_crude_input_cost, i), 
   #     plant_capacity_kg_day, 
   #     processing_steps)
    
    total_product_cost_i, non_feedstock_raw_mat_cost_i, utilities_cost_i, total_fixed_oper_cost_i, \
    total_variable_oper_cost_i, total_operating_costs_i, direct_operating_costs_i, variable_operating_costs_i = calculate_additional_operating_costs(
        extract(total_raw_material_cost, i),
        extract(total_energy_cost, i),
        financial_data,
        extract(fixed_capital_investment, i),
        extract(annual_crude_input_cost, i),
        plant_capacity_kg_day,
        processing_steps
    )

    total_product_cost_SR_i, non_feedstock_raw_mat_cost_SR_i, utilities_cost_SR_i, total_fixed_oper_cost_SR_i, \
    total_variable_oper_cost_SR_i, total_operating_costs_SR_i, direct_operating_costs_SR_i, variable_operating_costs_SR_i = calculate_additional_operating_costs(
        extract(total_distillation_raw_material_cost, i),
        extract(total_distillation_energy_cost, i),
        financial_data,
        extract(fixed_capital_investment_SR, i),
        extract(annual_crude_input_cost, i),
        plant_capacity_kg_day,
        processing_steps
    )

    def clean_dict(d):
        return {k: float(v) if isinstance(v, (np.floating, np.float64)) else v for k, v in d.items()}

    direct_operating_costs_i = clean_dict(direct_operating_costs_i)
    variable_operating_costs_i = clean_dict(variable_operating_costs_i)

    direct_operating_costs_SR_i = clean_dict(direct_operating_costs_SR_i)
    variable_operating_costs_SR_i = clean_dict(variable_operating_costs_SR_i)

    non_feedstock_raw_mat_cost.append(non_feedstock_raw_mat_cost_i)
    utilities_cost.append(utilities_cost_i)
    total_fixed_oper_cost.append(total_fixed_oper_cost_i)
    total_variable_oper_cost.append(total_variable_oper_cost_i)
    total_operating_costs.append(total_operating_costs_i)
    direct_operating_costs.append(direct_operating_costs_i)
    variable_operating_costs.append(variable_operating_costs_i)

    non_feedstock_raw_mat_cost_SR.append(non_feedstock_raw_mat_cost_SR_i)
    utilities_cost_SR.append(utilities_cost_SR_i)
    total_fixed_oper_cost_SR.append(total_fixed_oper_cost_SR_i)
    total_variable_oper_cost_SR.append(total_variable_oper_cost_SR_i)
    total_operating_costs_SR.append(total_operating_costs_SR_i)
    direct_operating_costs_SR.append(direct_operating_costs_SR_i)
    variable_operating_costs_SR.append(variable_operating_costs_SR_i)

def collapse_range(values):
    """Flatten nested [min,max] pairs and return overall [min,max]."""
    flat = np.array(values).flatten()
    return [float(np.min(flat)), float(np.max(flat))]

def collapse_cost_dicts(dict_list):
    """
    Collapse a list of cost dictionaries, each possibly containing [min, max] pairs,
    into a single dictionary with overall [min, max] for each category.
    """
    collapsed = {}
    for d in dict_list:
        for k, v in d.items():
            # Normalize value to [min, max]
            if isinstance(v, (list, tuple)) and len(v) == 2:
                vmin, vmax = float(v[0]), float(v[1])
            else:
                vmin = vmax = float(v)

            # Initialize or update range
            if k not in collapsed:
                collapsed[k] = [vmin, vmax]
            else:
                collapsed[k][0] = min(collapsed[k][0], vmin)
                collapsed[k][1] = max(collapsed[k][1], vmax)
    return collapsed

non_feedstock_raw_mat_cost = collapse_range(non_feedstock_raw_mat_cost)
utilities_cost = collapse_range(utilities_cost)
total_fixed_oper_cost = collapse_range(total_fixed_oper_cost)
total_variable_oper_cost = collapse_range(total_variable_oper_cost)
total_operating_costs = collapse_range(total_operating_costs)
direct_operating_costs = collapse_cost_dicts(direct_operating_costs)
variable_operating_costs = collapse_cost_dicts(variable_operating_costs)

non_feedstock_raw_mat_cost_SR = collapse_range(non_feedstock_raw_mat_cost_SR)
utilities_cost_SR = collapse_range(utilities_cost_SR)
total_fixed_oper_cost_SR = collapse_range(total_fixed_oper_cost_SR)
total_variable_oper_cost_SR = collapse_range(total_variable_oper_cost_SR)
total_operating_costs_SR = collapse_range(total_operating_costs_SR)
direct_operating_costs_SR = collapse_cost_dicts(direct_operating_costs_SR)
variable_operating_costs_SR = collapse_cost_dicts(variable_operating_costs_SR)


# --- Working capital as a range ---
WC = (
    extract(working_capital_cost, 0) / extract(fixed_capital_investment, 0),
    extract(working_capital_cost, 1) / extract(fixed_capital_investment, 1)
)

WC_SR = (
    extract(working_capital_cost_SR, 0) / extract(fixed_capital_investment_SR, 0),
    extract(working_capital_cost_SR, 1) / extract(fixed_capital_investment_SR, 1)
)

# --- Total variable operating cost WITHOUT utilities & feed as a range ---
total_variable_oper_cost_without_utilities_and_feed = (
    extract(total_variable_oper_cost, 0) - extract(annual_crude_input_cost, 0) - extract(utilities_cost, 0),
    extract(total_variable_oper_cost, 1) - extract(annual_crude_input_cost, 1) - extract(utilities_cost, 1)
)

total_variable_oper_cost_without_utilities_and_feed_SR = (
    extract(total_variable_oper_cost_SR, 0) - extract(annual_crude_input_cost, 0) - extract(utilities_cost_SR, 0),
    extract(total_variable_oper_cost_SR, 1) - extract(annual_crude_input_cost, 1) - extract(utilities_cost_SR, 1)
)

# --- Run discounted cash flow for min & max if ranges exist ---
results = {}
results_SR = {}
input_records = []
input_records_SR = []

for i, label in enumerate(["Min MSP", "Max MSP"]):

    # For minimum price for treated kerosene, set aromatics price high
    cost_btx_i = CostBTX[1 - i] if isinstance(CostBTX, (list, tuple)) else CostBTX

    if isinstance(OutputPropane, np.ndarray):
        OutputPropane = OutputPropane.tolist()  # numpy array -> python list
        OutputPropane = sorted(OutputPropane)

    # For minimum price for treated kerosene, set propane output high
    propane_i = OutputPropane[1 - i] if isinstance(OutputPropane, (list, tuple)) else OutputPropane

    # Collect inputs for this run
    inputs_i = {
        "fixed_capital_investment": extract(fixed_capital_investment, i),
        "total_fixed_oper_cost": extract(total_fixed_oper_cost, i),
        "total_variable_oper_cost_without_utilities_and_feed": extract(total_variable_oper_cost_without_utilities_and_feed, i),
        "annual_crude_input_cost": extract(annual_crude_input_cost, i),
        "fuel": extract(treated_kero_utility_data['Total'].get('fuel', 0), i),
        "NG for steam production": extract(treated_kero_utility_data['Total'].get('NG for steam production', 0), i),
        "SMR feed gas": extract(treated_kero_utility_data['Total'].get('feed_gas', 0), i),
        "power": extract(treated_kero_utility_data['Total']['power'], i),
        "H2": extract(treated_kero_utility_data['Total'].get('H2', 0), i),
        "cooling_water": extract(treated_kero_utility_data['Total'].get('cooling_water_crclt', 0), i),
        "output kerosene": OutputKerosene,  "output light naphtha": OutputLightNaphtha,  "output heavy naphtha": OutputHeavyNaphtha,
        "output light gas oil": OutputLightGasOil,  "output heavy gas oil": OutputHeavyGasOil,  "output atmospheric residues": OutputAtmResidue,
        "output light vacuum gas oil": OutputLightVacuumGasOil,  "output heavy vacuum gas oil": OutputHeavyVacuumGasOil,  "output vacuum residues": OutputVacuumResidues,
        "output propane": propane_i, "output sulfur": OutputSulfur, "output BTX": OutputBTX,
        "price light naphtha": CostLightNaphtha, "price heavy naphtha": CostHeavyNaphtha, "price light gas oil": CostLightGasOil,
        "price heavy gas oil": CostHeavyGasOil, "price atmospheric residues": CostAtmResidue, "price light vacuum gas oil": CostLightVacuumGasOil,
        "price heavy vacuum gas oil": CostHeavyVacuumGasOil, "price vacuum residues": CostVacuumResidues, "price propane": CostPropane,
        "price sulfur": CostSulfur, "price BTX": cost_btx_i, "price NG start": Cost_NGStart, "price power": Cost_PowerStart,
        "price H2": Cost_HydrogenStart, "price cooling water": Cost_CoolingWaterStart,
        "revenue of byproducts": Cost_ByproductsStart, "equity": financial_data['financial_assumptions']['equity'], "loan interest": financial_data['financial_assumptions']['loan_interest'],
        "loan term": financial_data['financial_assumptions']['loan_term'], "WC": extract(WC, i), "FCI Det": financial_data['financial_assumptions']['FCIDet'],
        "price NG growth": financial_data['financial_assumptions']['CostNGGrowth'], "WC": extract(WC, i), "price NG STD": financial_data['financial_assumptions']['CostNGSTD'],
        "WACC": financial_data['financial_assumptions']['WACC'], "inflation":  financial_data['financial_assumptions']['inflation'], "ITR": financial_data['financial_assumptions']['ITR']
    }

    Cost_MD, Product_Names_MD = discounted_cash_flow(
        extract(fixed_capital_investment, i),
        extract(total_fixed_oper_cost, i),
        extract(total_variable_oper_cost_without_utilities_and_feed, i),
        extract(annual_crude_input_cost, i),
        extract(treated_kero_utility_data['Total'].get('fuel', 0), i), extract(treated_kero_utility_data['Total'].get('NG for steam production', 0), i),
        extract(treated_kero_utility_data['Total'].get('feed_gas', 0), i), extract(treated_kero_utility_data['Total']['power'], i), 
        extract(treated_kero_utility_data['Total'].get('H2', 0), i), extract(treated_kero_utility_data['Total'].get('cooling_water', 0), i),
        OutputKerosene, OutputLightNaphtha, OutputHeavyNaphtha,
        OutputLightGasOil, OutputHeavyGasOil, OutputAtmResidue,
        OutputLightVacuumGasOil, OutputHeavyVacuumGasOil, OutputVacuumResidues,
        propane_i, OutputSulfur, OutputBTX,
        CostLightNaphtha, CostHeavyNaphtha, CostLightGasOil, CostHeavyGasOil,
        CostAtmResidue, CostLightVacuumGasOil, CostHeavyVacuumGasOil,
        CostVacuumResidues, CostPropane, CostSulfur, cost_btx_i,
        Cost_NGStart, Cost_PowerStart, Cost_HydrogenStart,
        Cost_CoolingWaterStart, Cost_ByproductsStart,
        financial_data['financial_assumptions']['equity'],
        financial_data['financial_assumptions']['loan_interest'],
        financial_data['financial_assumptions']['loan_term'],
        extract(WC, i),
        financial_data['financial_assumptions']['FCIDet'],
        financial_data['financial_assumptions']['CostNGGrowth'],
        financial_data['financial_assumptions']['CostNGSTD'],
        financial_data['financial_assumptions']['WACC'],
        financial_data['financial_assumptions']['inflation'],
        financial_data['financial_assumptions']['ITR']
    )
    results[label] = (Cost_MD, Product_Names_MD)

    inputs_i.update({name: cost for name, cost in zip(Product_Names_MD, Cost_MD)})
    input_records.append({"scenario": label, **inputs_i})

    # Collect inputs for this run
    inputs_SR_i = {
        "fixed_capital_investment": extract(fixed_capital_investment_SR, i),
        "total_fixed_oper_cost": extract(total_fixed_oper_cost_SR, i),
        "total_variable_oper_cost_without_utilities_and_feed": extract(total_variable_oper_cost_without_utilities_and_feed_SR, i),
        "annual_crude_input_cost": extract(annual_crude_input_cost, i),
        "fuel": extract(SR_kero_utility_data['Total'].get('fuel', 0), i),
        "NG for steam production": extract(SR_kero_utility_data['Total'].get('NG for steam production', 0), i),
        "SMR feed gas": extract(SR_kero_utility_data['Total'].get('feed_gas', 0), i),
        "power": extract(SR_kero_utility_data['Total']['power'], i),
        "H2": extract(SR_kero_utility_data['Total'].get('H2', 0), i),
        "cooling_water": extract(SR_kero_utility_data['Total'].get('cooling_water_crclt', 0), i),
        "output kerosene": OutputSRKerosene,  "output light naphtha": OutputLightNaphtha,  "output heavy naphtha": OutputHeavyNaphtha,
        "output light gas oil": OutputLightGasOil,  "output heavy gas oil": OutputHeavyGasOil,  "output atmospheric residues": OutputAtmResidue,
        "output light vacuum gas oil": OutputLightVacuumGasOil,  "output heavy vacuum gas oil": OutputHeavyVacuumGasOil,  "output vacuum residues": OutputVacuumResidues,
        "output propane": 0, "output sulfur": 0, "output BTX": 0,
        "price light naphtha": CostLightNaphtha, "price heavy naphtha": CostHeavyNaphtha, "price light gas oil": CostLightGasOil,
        "price heavy gas oil": CostHeavyGasOil, "price atmospheric residues": CostAtmResidue, "price light vacuum gas oil": CostLightVacuumGasOil,
        "price heavy vacuum gas oil": CostHeavyVacuumGasOil, "price vacuum residues": CostVacuumResidues, "price propane": CostPropane,
        "price sulfur": CostSulfur, "price BTX": cost_btx_i, "price NG start": Cost_NGStart, "price power": Cost_PowerStart,
        "price H2": Cost_HydrogenStart, "price cooling water": Cost_CoolingWaterStart,
        "revenue of byproducts": Cost_ByproductsStart, "equity": financial_data['financial_assumptions']['equity'], "loan interest": financial_data['financial_assumptions']['loan_interest'],
        "loan term": financial_data['financial_assumptions']['loan_term'], "WC": extract(WC, i), "FCI Det": financial_data['financial_assumptions']['FCIDet'],
        "price NG growth": financial_data['financial_assumptions']['CostNGGrowth'], "WC": extract(WC_SR, i), "price NG STD": financial_data['financial_assumptions']['CostNGSTD'],
        "WACC": financial_data['financial_assumptions']['WACC'], "inflation":  financial_data['financial_assumptions']['inflation'], "ITR": financial_data['financial_assumptions']['ITR']
    }

    Cost_SR_kerosene, Product_Names_SR_kerosene = discounted_cash_flow(
        extract(fixed_capital_investment_SR, i),
        extract(total_fixed_oper_cost_SR, i),
        extract(total_variable_oper_cost_without_utilities_and_feed_SR, i),
        extract(annual_crude_input_cost, i),
        extract(SR_kero_utility_data['Total'].get('fuel', 0), i), extract(SR_kero_utility_data['Total'].get('NG for steam production', 0), i),
        extract(SR_kero_utility_data['Total'].get('feed_gas', 0), i), extract(SR_kero_utility_data['Total']['power'], i), 
        extract(SR_kero_utility_data['Total'].get('H2', 0), i), extract(SR_kero_utility_data['Total'].get('cooling_water', 0), i),
        OutputSRKerosene, OutputLightNaphtha, OutputHeavyNaphtha,
        OutputLightGasOil, OutputHeavyGasOil, OutputAtmResidue,
        OutputLightVacuumGasOil, OutputHeavyVacuumGasOil, OutputVacuumResidues,
        0, 0, 0, # Propane, Sulfur, BTX
        CostLightNaphtha, CostHeavyNaphtha, CostLightGasOil, CostHeavyGasOil,
        CostAtmResidue, CostLightVacuumGasOil, CostHeavyVacuumGasOil,
        CostVacuumResidues, CostPropane, CostSulfur, cost_btx_i,
        Cost_NGStart, Cost_PowerStart, Cost_HydrogenStart,
        Cost_CoolingWaterStart, Cost_ByproductsStart,
        financial_data['financial_assumptions']['equity'],
        financial_data['financial_assumptions']['loan_interest'],
        financial_data['financial_assumptions']['loan_term'],
        extract(WC_SR, i),
        financial_data['financial_assumptions']['FCIDet'],
        financial_data['financial_assumptions']['CostNGGrowth'],
        financial_data['financial_assumptions']['CostNGSTD'],
        financial_data['financial_assumptions']['WACC'],
        financial_data['financial_assumptions']['inflation'],
        financial_data['financial_assumptions']['ITR']
    )
    results_SR[label] = (Cost_SR_kerosene, Product_Names_SR_kerosene)

    inputs_SR_i.update({name: cost for name, cost in zip(Product_Names_SR_kerosene, Cost_SR_kerosene)})
    input_records_SR.append({"scenario": label, **inputs_SR_i})

all_keys = [k for k in input_records[0].keys() if k != "scenario"]
dcfror_table = {}

for key in all_keys:
    dcfror_table[key] = {
        "Min MSP (Treated)": input_records[0][key],
        "Max MSP (Treated)": input_records[1][key],
        "Min MSP (SR)": input_records_SR[0][key],
        "Max MSP (SR)": input_records_SR[1][key],
    }

df_dcfror_pivot = pd.DataFrame(dcfror_table).T  # Transpose so inputs are rows
df_dcfror_pivot = df_dcfror_pivot[["Min MSP (Treated)", "Max MSP (Treated)", "Min MSP (SR)", "Max MSP (SR)"]]  # Ensure column order
df_dcfror_pivot = df_dcfror_pivot.reset_index()
df_dcfror_pivot = df_dcfror_pivot.rename(columns={"index": "Input"}) 
print(df_dcfror_pivot)

if "Min MSP" in results and "Max MSP" in results_SR:
    Cost_SR_min, Product_Names_SR_kerosene = results_SR["Min MSP"]
    Cost_SR_max, _ = results_SR["Max MSP"]
else:
    # Fallback: only single value case
    Cost_SR_min = Cost_MD_SR_max = results_SR["Min MSP"][0]
    Product_Names_SR_kerosene = results_SR["Min MSP"][1]

print("\n---SR Kerosene Refinery product prices per gallon ---")
if "Min MSP" in results_SR and "Max MSP" in results_SR:
    # Range mode
    for name, cost_min, cost_max in zip(Product_Names_SR_kerosene, Cost_SR_min, Cost_SR_max):
        cost_min_gal = cost_min / 42
        cost_max_gal = cost_max / 42
        if cost_min_gal == cost_max_gal:
            print(f"{name}: ${cost_min_gal:.2f} per gal")
        else:
            print(f"{name}: ${cost_min_gal:.2f} - ${cost_max_gal:.2f} per gal")
else:
    # Single value mode
    for name, cost in zip(Product_Names_SR_kerosene, Cost_SR_kerosene):
        cost_gal = cost / 42
        print(f"{name}: ${cost_gal:.2f} per gal")

if "Min MSP" in results and "Max MSP" in results:
    Cost_MD_min, Product_Names_MD = results["Min MSP"]
    Cost_MD_max, _ = results["Max MSP"]
else:
    # Fallback: only single value case
    Cost_MD_min = Cost_MD_max = results["Min MSP"][0]
    Product_Names_MD = results["Min MSP"][1]

if aromatics_removal_technique != 'none':
    print("\n--- Treated Kerosene Refinery product prices per bbl ---")
    for name, cost_min, cost_max in zip(Product_Names_MD, Cost_MD_min, Cost_MD_max):
        if cost_min == cost_max:
            print(f"{name}: ${cost_min:.2f} per bbl")
        else:
            print(f"{name}: ${cost_min:.2f} - ${cost_max:.2f} per bbl")

    print("\n--- Treated Kerosene Refinery product prices per gallon ---")
    if "Min MSP" in results and "Max MSP" in results:
        # Range mode
        for name, cost_min, cost_max in zip(Product_Names_MD, Cost_MD_min, Cost_MD_max):
            cost_min_gal = cost_min / 42
            cost_max_gal = cost_max / 42
            if cost_min_gal == cost_max_gal:
                print(f"{name}: ${cost_min_gal:.2f} per gal")
            else:
                print(f"{name}: ${cost_min_gal:.2f} - ${cost_max_gal:.2f} per gal")
    else:
        # Single value mode
        for name, cost in zip(Product_Names_MD, Cost_MD):
            cost_gal = cost / 42
            print(f"{name}: ${cost_gal:.2f} per gal")

    # --- Compute cost premium for kerosene ---
    print("\n--- Kerosene Cost Premium ---")
    try:
        # Find kerosene index
        idx_kerosene = next(
            i for i, n in enumerate(Product_Names_MD) if "kerosene" in n.lower()
        )

        # Get untreated kerosene price(s)
        treated_min = Cost_MD_min[idx_kerosene] / 42
        treated_max = Cost_MD_max[idx_kerosene] / 42

        # Get SR kerosene price(s)
        sr_idx = next(
            i for i, n in enumerate(Product_Names_SR_kerosene) if "kerosene" in n.lower()
        )
        untreated_min = Cost_SR_min[sr_idx] / 42
        untreated_max = Cost_SR_max[sr_idx] / 42

        # Compute premium
        premium_min = treated_min - untreated_min
        premium_max = treated_max - untreated_max

        print(f"Kerosene Price Premium: ${premium_min:.2f} – ${premium_max:.2f} per gal")

    except StopIteration:
        print("Kerosene not found in product list.")
    except Exception as e:
        print(f"Error computing cost premium: {e}")

# --- Additional Capital Costs ---
min_dict, max_dict = additional_capital_costs
additional_rows = []
for item in min_dict.keys():
    additional_rows.append({
        'Cost Category': item.replace('_', ' ').title(),
        'Min (USD)': round(min_dict[item], 2),
        'Max (USD)': round(max_dict[item], 2)
    })

# --- Main Capital Costs ---
def to_min_max(value):
    """Convert a scalar or 2-element list/tuple into min/max."""
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return round(value[0], 2), round(value[1], 2)
    else:
        v = round(float(value), 2)
        return v, v

main_rows = []
for label, val in [
    ('Total Purchase Equipment Cost', total_purchase_equipment_cost),
    ('Fixed Capital Investment', fixed_capital_investment),
    ('Working Capital', working_capital_cost),
    ('Total Capital Investment', total_capital_investment)
]:
    min_val, max_val = to_min_max(val)
    main_rows.append({'Cost Category': label, 'Min (USD)': min_val, 'Max (USD)': max_val})

# Insert additional_rows after Total Purchase Equipment Cost
final_rows = [main_rows[0]] + additional_rows + main_rows[1:]

# Create DataFrame
capital_costs_df = pd.DataFrame(final_rows)

# Display
print("\n--- Capital Costs ---")
print(capital_costs_df)

# --- Direct Operating Costs ---
direct_rows = []
for k, v in direct_operating_costs.items():
    vmin, vmax = v  # unpack the min/max from the list
    direct_rows.append({
        'Cost Category': k.replace('_', ' ').title(),
        'Min (USD)': round(vmin, 2),
        'Max (USD)': round(vmax, 2)
    })

# --- Variable Operating Costs ---
variable_rows = []
for k, v in variable_operating_costs.items():
    vmin, vmax = v  # unpack the min/max from the list
    variable_rows.append({
        'Cost Category': k.replace('_', ' ').title(),
        'Min (USD)': round(vmin, 2),
        'Max (USD)': round(vmax, 2)
    })


# --- Total Operating Costs ---
total_rows = [
    {
        'Cost Category': 'Fixed Operating Cost',
        'Min (USD)': round(total_fixed_oper_cost[0], 2),
        'Max (USD)': round(total_fixed_oper_cost[1], 2)
    },
    {
        'Cost Category': 'Total Variable Operating Cost',
        'Min (USD)': round(total_variable_oper_cost[0], 2),
        'Max (USD)': round(total_variable_oper_cost[1], 2)
    },
    {
        'Cost Category': 'Total Operating Cost',
        'Min (USD)': round(total_operating_costs[0], 2),
        'Max (USD)': round(total_operating_costs[1], 2)
    }
]

# Combine all rows
final_operating_rows = direct_rows + variable_rows + total_rows

# Create DataFrame
operating_costs_df = pd.DataFrame(final_operating_rows)

# Display
print("\n--- Operating Costs ---")
print(operating_costs_df)

# DCFROR product values per barrel
if "min" in results and "max" in results:
    # Range mode
    dcfror_df = pd.DataFrame({
        'Product': Product_Names_MD,
        'Cost ($/bbl)': [
            f"${cost_min:.2f} - ${cost_max:.2f}"
            for cost_min, cost_max in zip(Cost_MD_min, Cost_MD_max)
        ]
    })
else:
    # Single value mode
    dcfror_df = pd.DataFrame({
        'Product': Product_Names_MD,
        'Cost ($/bbl)': [f"${cost:.2f}" for cost in Cost_MD]
    })



# LIFE-CYCLE ANALYSIS

lca_data['raw_material_emissions']['NiMo_Al2_O3'] = ( lca_data['raw_material_emissions']['NiMo_Al2_O3_production'] + lca_data['raw_material_emissions']['NiMo_Al2_O3_spent'] ) * conversion_parameters['lb_to_ton']

utility_LCA_map = {
    'fuel (MMBtu)': lca_data['variable_input_emissions']['NG'],
    'power (kWh)': lca_data['variable_input_emissions'][electricity_choice],
    'H2 (scf)': 0, # FIXXX!!!!!!!!!!!!!!
    'cooling_water (gal)': lca_data['variable_input_emissions']['cooling_water'],
    'feed_water (lb)': lca_data['variable_input_emissions']['feed_water'],
    'boiler_feed_water (gal)': lca_data['variable_input_emissions']['boiler_feed_water'],
    'catalyst_replacement (kg)': lca_data['raw_material_emissions']['NiMo_Al2_O3'],

    
    }


raw_material_LCA_map = {
    'catalyst_replacement (kg)': lca_data['raw_material_emissions']['NiMo_Al2_O3']
}

utility_indices = {u: i for i, u in enumerate(treated_kero_utility_data['Utility'])}

if aromatics_removal_technique == 'hydrotreatment':
    allocation_factors = {
    'Upstream Crude Oil': crude_oil_HT_kerosene_AF,
    'Atmospheric Distillation': distillation_HT_kerosene_AF,
    'Vacuum Distillation': distillation_HT_kerosene_AF,
    'Distillation': distillation_HT_kerosene_AF,
    'Hydrotreatment': hydrotreatment_HT_kerosene_AF,
    'Amine Gas Treating': hydrotreatment_HT_kerosene_AF,
    'Claus Process': hydrotreatment_HT_kerosene_AF,
    'SCOT Process': hydrotreatment_HT_kerosene_AF,
    'SMR': hydrotreatment_HT_kerosene_AF,
    }

    material_ref_feed = {
    'Distillation': 150000,
    'Hydrotreatment': 20000
    }

elif aromatics_removal_technique == 'solvent_extraction':
    allocation_factors = {
    'Upstream Crude Oil': crude_oil_raffinate_kerosene_AF,
    'Atmospheric Distillation': distillation_raffinate_kerosene_AF,
    'Vacuum Distillation': distillation_raffinate_kerosene_AF,
    'Solvent extraction': solvent_extraction_raffinate_AF
    }

else:
    allocation_factors = {
    'Upstream Crude Oil': crude_oil_kerosene_AF,
    'Atmospheric Distillation': distillation_kerosene_AF,
    'Vacuum Distillation': distillation_kerosene_AF
    }


lca_rows_data = []  # store detailed results

upstream_lca_per_MJ = lca_data['emissions']['upstream_crude_oil_prod_trans'] * allocation_factors['Upstream Crude Oil']

lca_rows_data.append({
        "Unit Operation": 'Crude prod. and trans.',
        "Item": 'Upstream crude oil',
        "AF": allocation_factors['Upstream Crude Oil'],
        "Impact (gCO2e/MJ kerosene)": f"{upstream_lca_per_MJ:.2f}"
        })


#def calculate_lca(material_dict):
#    construction_lca_results = {}
#    for mat, qty in material_dict.items():
#        # Multiply by LCA factor if available
#        factor = lca_data['materials'].get(mat, 0)
#        construction_lca_results[mat] = feed_ratio * qty * factor * af / (MJ_per_yr_OutputKerosene * user_inputs['Refinery']['lifetime'])  # gCO2e/MJ
#    return construction_lca_results

#print("\nPlant Construction Life-Cycle Analysis (LCA)\n")
#total_embodied_lca_per_MJ = 0
#if aromatics_removal_technique == 'hydrotreatment':
#    plant_sections = ["Distillation", "Hydrotreatment"]

#    for section in plant_sections:
#        if section == 'Distillation':
#            feed = distillation_inputs['BPCD']['Crude oil'] 
#        if section == 'Hydrotreatment':
#            feed = hydrotreatment_inputs['BPCD']['Kerosine'] 

#        af = allocation_factors.get(section, 1.0)
#        feed_ratio = feed / material_ref_feed.get(section, 1.0)
#
#        print(f"--- {section} ---")
#        materials = material_data.get(section, {})
#        construction_lca_results = calculate_lca(materials)
#        total_emissions = sum(construction_lca_results.values())
    
#        for mat, em in construction_lca_results.items():
#            print(f"{mat:15s}: {em:,.4f} gCO2e/MJ")
#        print(f"Total emissions: {total_emissions:,.4f} gCO2e/MJ \n")

#        total_embodied_lca_per_MJ += total_emissions  # accumulate total

#        lca_rows_data.append({
#                "Unit Operation": section,
#                "Item": 'Embodied emissions',
#                "AF": af,
#                "Impact (gCO2e/MJ kerosene)": f"{total_emissions:.2f}"
#            })


print(f"\nUtility Life-Cycle Analysis (LCA) by Unit Operation with AF")
total_energy_lca_per_MJ_min = 0
total_energy_lca_per_MJ_max = 0
has_range_lca = False

for unit_op, usage_list in treated_kero_utility_data.items():
    if unit_op in ['Utility', 'Total', 'Total Cost']:
        continue

    for util, impact in utility_LCA_map.items():
        if util not in utility_indices:
            continue
        idx = utility_indices[util]
        unit_usage = usage_list[idx]

        # Allocation Factor
        af = allocation_factors.get(unit_op, 1.0)

        # Normalize unit_usage: 0-d arrays → scalar
        if isinstance(unit_usage, np.ndarray) and unit_usage.ndim == 0:
            unit_usage = float(unit_usage) 

        if isinstance(unit_usage, (list, np.ndarray)) and len(unit_usage) == 2:
            has_range_lca = True
            usage_min, usage_max = unit_usage[0], unit_usage[1]
            lca_min = usage_min * impact * af
            lca_max = usage_max * impact * af

            lca_min_per_MJ_kerosene = 1000 * lca_min / MJ_per_yr_OutputKerosene
            lca_max_per_MJ_kerosene = 1000 * lca_max / MJ_per_yr_OutputKerosene

            print(f"{util}: Annual Usage = [{usage_min:.2f}, {usage_max:.2f}], "
              f"Unit Impact = {impact:.4f}, Annual Impact = [{lca_min:,.2f}, {lca_max:,.2f}] with AF = {af:2f}, Impact gCO2e per MJ Kerosene = [{lca_min_per_MJ_kerosene:,.4f}, {lca_max_per_MJ_kerosene:,.4f}]")
            
            total_energy_lca_per_MJ_min += lca_min_per_MJ_kerosene
            total_energy_lca_per_MJ_max += lca_max_per_MJ_kerosene

            lca_rows_data.append({
                "Unit Operation": unit_op,
                "Item": util,
                "AF": af,
                "Impact (gCO2e/MJ kerosene)": f"[{lca_min_per_MJ_kerosene:.2f}, {lca_max_per_MJ_kerosene:.2f}]"
            })

        else:

            # Scalar usage
            lca_scalar = unit_usage * impact * af 
            lca_per_MJ_kerosene = 1000 * lca_scalar / MJ_per_yr_OutputKerosene
            print(f"{util}: Annual Usage = {unit_usage:.2f}, Unit Impact = {impact:.4f}, Annual Impact = {lca_scalar:,.2f} with AF = {af:2f}, Impact gCO2e per MJ Kerosene = {lca_per_MJ_kerosene:,.4f} ")

            total_energy_lca_per_MJ_min += lca_per_MJ_kerosene
            total_energy_lca_per_MJ_max += lca_per_MJ_kerosene

            lca_rows_data.append({
                "Unit Operation": unit_op,
                "Item": util,
                "AF": af,
                "Impact (gCO2e/MJ kerosene)": f"{lca_per_MJ_kerosene:.2f}"
            })

# Decide which total to display
if total_energy_lca_per_MJ_min == total_energy_lca_per_MJ_max:
    total_energy_lca_per_MJ = total_energy_lca_per_MJ_min
    print(f"Energy LCA Impact (gCO2e/MJ), allocated to kerosene): "
          f"{total_energy_lca_per_MJ:,.2f}")
    
else:
    total_energy_lca_per_MJ = [total_energy_lca_per_MJ_min, total_energy_lca_per_MJ_max]
    print(f"Energy LCA Impact Range (gCO2e/MJ), allocated to kerosene): "
          f"[{total_energy_lca_per_MJ_min:,.2f}, {total_energy_lca_per_MJ_max:,.2f}]")



total_raw_mat_lca_per_MJ_min = 0
total_raw_mat_lca_per_MJ_max = 0
has_range_lca = False  # Separate flag for LCA

print("\nRaw Material Life-Cycle Analysis (LCA)\n")

for unit_op, usage_list in treated_kero_utility_data.items():
    if unit_op in ['Utility', 'Raw Materials', 'Total', 'Total Cost']:
        continue

    for material, impact in raw_material_LCA_map.items():
        if material not in utility_indices:
            continue
        idx = utility_indices[material]
        unit_usage = usage_list[idx]

        # Allocation Factor
        af = allocation_factors.get(unit_op, 1.0)

        # Normalize 0-d arrays
        if isinstance(unit_usage, np.ndarray) and unit_usage.ndim == 0:
            unit_usage = float(unit_usage)

        if isinstance(unit_usage, (list, np.ndarray)) and len(unit_usage) == 2:
            has_range_lca = True
            usage_min, usage_max = unit_usage[0], unit_usage[1]
            lca_min = usage_min * impact * af
            lca_max = usage_max * impact * af

            lca_min_per_MJ_kerosene = 1000 * lca_min / MJ_per_yr_OutputKerosene
            lca_max_per_MJ_kerosene = 1000 * lca_max / MJ_per_yr_OutputKerosene

            print(f"{material}: Annual Usage = [{usage_min:.2f}, {usage_max:.2f}], "
                  f"Unit Impact = {impact:.4f}, Annual Impact = [{lca_min:,.2f}, {lca_max:,.2f}] "
                  f"with AF = {af:.2f}, Impact gCO2e per MJ Kerosene = "
                  f"[{lca_min_per_MJ_kerosene:,.4f}, {lca_max_per_MJ_kerosene:,.4f}]")

            total_raw_mat_lca_per_MJ_min += lca_min_per_MJ_kerosene
            total_raw_mat_lca_per_MJ_max += lca_max_per_MJ_kerosene

            lca_rows_data.append({
                "Unit Operation": unit_op,
                "Item": material,
                "AF": af,
                "Impact (gCO2e/MJ kerosene)": f"[{lca_min_per_MJ_kerosene:.2f}, {lca_max_per_MJ_kerosene:.2f}]"
            })

        else:

            # Scalar usage
            lca_scalar = unit_usage * impact * af 
            lca_per_MJ_kerosene = 1000 * lca_scalar / MJ_per_yr_OutputKerosene

            print(f"{material}: Annual Usage = {unit_usage:.2f}, Unit Impact = {impact:.4f}, "
                  f"Annual Impact = {lca_scalar:,.2f} with AF = {af:.2f}, "
                  f"Impact gCO2e per MJ Kerosene = {lca_per_MJ_kerosene:,.4f}")

            total_raw_mat_lca_per_MJ_min += lca_per_MJ_kerosene
            total_raw_mat_lca_per_MJ_max += lca_per_MJ_kerosene

            lca_rows_data.append({
                "Unit Operation": unit_op,
                "Item": material,
                "AF": af,
                "Impact (gCO2e/MJ kerosene)": f"{lca_per_MJ_kerosene:.2f}"
            })

# Display totals
if total_raw_mat_lca_per_MJ_min == total_raw_mat_lca_per_MJ_max:
    total_raw_mat_lca_per_MJ = total_raw_mat_lca_per_MJ_min
    print(f"\nRaw Material LCA Impact (gCO2e/MJ), allocated to kerosene: "
          f"{total_raw_mat_lca_per_MJ:,.2f}")

else:
    total_raw_mat_lca_per_MJ = [total_raw_mat_lca_per_MJ_min, total_raw_mat_lca_per_MJ_max]
    print(f"\nRaw Material LCA Impact Range (gCO2e/MJ), allocated to kerosene: "
          f"[{total_raw_mat_lca_per_MJ_min:,.2f}, {total_raw_mat_lca_per_MJ_max:,.2f}]")


combustion_lca_per_MJ = lca_data['emissions']['jet_fuel_combustion']

lca_rows_data.append({
        "Unit Operation": 'Combustion',
        "Item": 'Combustion',
        "AF": 1.0,
        "Impact (gCO2e/MJ kerosene)": f"{combustion_lca_per_MJ:.2f}"
        })

# Combine total energy and total raw material LCA
if isinstance(total_energy_lca_per_MJ, list) or isinstance(total_raw_mat_lca_per_MJ, list):
    # At least one has a range
    # Convert scalars to ranges if needed
    if not isinstance(total_energy_lca_per_MJ, list):
        total_energy_lca_per_MJ = [total_energy_lca_per_MJ, total_energy_lca_per_MJ]
    if not isinstance(total_raw_mat_lca_per_MJ, list):
        total_raw_mat_lca_per_MJ = [total_raw_mat_lca_per_MJ, total_raw_mat_lca_per_MJ]

    total_lca_min_per_MJ = (upstream_lca_per_MJ + total_energy_lca_per_MJ[0] + total_raw_mat_lca_per_MJ[0] + combustion_lca_per_MJ) 
    total_lca_max_per_MJ = (upstream_lca_per_MJ + total_energy_lca_per_MJ[1] + total_raw_mat_lca_per_MJ[1] + combustion_lca_per_MJ)
    total_lca_per_MJ = [total_lca_min_per_MJ, total_lca_max_per_MJ]
    print(f"\nTotal LCA Impact Range gCO2e per MJ of Treated Kerosene: [{total_lca_min_per_MJ:,.2f}, {total_lca_max_per_MJ:,.2f}]")

    total_lca_min_per_bbl = total_lca_min_per_MJ * MJ_per_yr_OutputKerosene / OutputKerosene
    total_lca_max_per_bbl = total_lca_max_per_MJ * MJ_per_yr_OutputKerosene / OutputKerosene

    total_lca_min_per_ton = total_lca_min_per_MJ * MJ_per_yr_OutputKerosene / (ton_per_yr_OutputKerosene * 1000)
    total_lca_max_per_ton = total_lca_max_per_MJ * MJ_per_yr_OutputKerosene / (ton_per_yr_OutputKerosene * 1000)

    print(f"Total LCA Impact Range gCO2e per bbl of Treated Kerosene: [{total_lca_min_per_bbl:,.2f}, {total_lca_max_per_bbl:,.2f}]")
    print(f"Total LCA Impact Range kgCO2e per tonne of Treated Kerosene: [{total_lca_min_per_ton:,.2f}, {total_lca_max_per_ton:,.2f}]")

else:
    # Both are scalars    
    total_lca_per_MJ = (upstream_lca_per_MJ + total_energy_lca_per_MJ + total_raw_mat_lca_per_MJ + combustion_lca_per_MJ) 
    print(f"\nTotal LCA Impact gCO2e per MJ of Treated Kerosene: {total_lca_per_MJ:,.2f}")

    total_lca_per_bbl = total_lca_per_MJ * MJ_per_yr_OutputKerosene / OutputKerosene

    total_lca_per_ton = total_lca_per_MJ * MJ_per_yr_OutputKerosene / (ton_per_yr_OutputKerosene * 1000)

    print(f"\nTotal LCA Impact gCO2e per bbl of Treated Kerosene: {total_lca_per_bbl:,.2f}")
    print(f"Total LCA Impact kgCO2e per tonne of Treated Kerosene: {total_lca_per_ton:,.2f}")

lca_details_df = pd.DataFrame(lca_rows_data)

lca_summary_df = pd.DataFrame({
    "Category": [
        "Energy LCA Impact",
        "Raw Material LCA Impact",
        "Total LCA Impact"
    ],
    "Value (gCO2e/MJ kerosene)": [
        total_energy_lca_per_MJ if not isinstance(total_energy_lca_per_MJ, list) else f"[{total_energy_lca_per_MJ[0]:.2f}, {total_energy_lca_per_MJ[1]:.2f}]",
        total_raw_mat_lca_per_MJ if not isinstance(total_raw_mat_lca_per_MJ, list) else f"[{total_raw_mat_lca_per_MJ[0]:.2f}, {total_raw_mat_lca_per_MJ[1]:.2f}]",
        total_lca_per_MJ if not isinstance(total_lca_per_MJ, list) else f"[{total_lca_per_MJ[0]:.2f}, {total_lca_per_MJ[1]:.2f}]"
    ]
})

# Convert lca_details_df to numeric min/max columns for plotting
lca_details_df['Impact_min'] = lca_details_df['Impact (gCO2e/MJ kerosene)'].apply(
    lambda x: float(x.strip("[]").split(",")[0]) if "[" in str(x) else float(x)
)
lca_details_df['Impact_max'] = lca_details_df['Impact (gCO2e/MJ kerosene)'].apply(
    lambda x: float(x.strip("[]").split(",")[1]) if "[" in str(x) else float(x)
)

# --- Group some unit operations before analysis ---
uo_group_map = {
    'Atmospheric Distillation': 'Distillation',
    'Vacuum Distillation': 'Distillation',
    'Amine Gas Treating': 'Sulfur Recovery',
    'Claus Process': 'Sulfur Recovery',
    'SCOT Process': 'Sulfur Recovery'
}

# Replace with group names where applicable
lca_details_df['Unit Operation Grouped'] = lca_details_df['Unit Operation'].replace(uo_group_map)

# --- Recompute unique groups ---
unit_ops = lca_details_df['Unit Operation Grouped'].unique()
utilities = lca_details_df['Item'].unique()
colors = plt.cm.tab20.colors  # color palette

# --- Compute per-utility impacts per grouped unit operation (avg, min, max) ---
avg_matrix = []
err_lower_matrix = []
err_upper_matrix = []

for uo in unit_ops:
    avg_row = []
    lower_row = []
    upper_row = []
    for util in utilities:
        subset = lca_details_df[
            (lca_details_df['Unit Operation Grouped'] == uo) &
            (lca_details_df['Item'] == util)
        ]
        if len(subset) == 0:
            avg_row.append(0)
            lower_row.append(0)
            upper_row.append(0)
            continue
        
        val_min = subset['Impact_min'].sum()
        val_max = subset['Impact_max'].sum()
        avg_val = (val_min + val_max) / 2

        avg_row.append(avg_val)
        lower_row.append(avg_val - val_min)
        upper_row.append(val_max - avg_val)
    avg_matrix.append(avg_row)
    err_lower_matrix.append(lower_row)
    err_upper_matrix.append(upper_row)

avg_matrix = np.array(avg_matrix)
err_lower_matrix = np.array(err_lower_matrix)
err_upper_matrix = np.array(err_upper_matrix)

# --- Cumulative Waterfall with error bars centered at top ---
cumulative_bottoms = np.zeros(len(unit_ops)+1)
fig, ax = plt.subplots(figsize=(12,6))

for i, uo in enumerate(unit_ops):
    bottom = cumulative_bottoms[i]
    for j, util in enumerate(utilities):
        avg_val = avg_matrix[i, j]
        ax.bar(uo, avg_val, bottom=bottom, color=colors[j % len(colors)], width=0.6,
               label=util if i == 0 else "")
        
        # Error bar centered at top
        y_top = bottom + avg_val
        err = err_lower_matrix[i, j] + err_upper_matrix[i, j]
        ax.errorbar(x=i, y=y_top, yerr=err/2, fmt='none', ecolor='black', capsize=5, lw=1)
        bottom += avg_val
    cumulative_bottoms[i+1] = bottom

# --- Customize plot ---
ax.set_ylabel('Impact (gCO2e/MJ kerosene)')
ax.set_title('LCA Allocated to Treated Kerosene')
ax.set_xticks(range(len(unit_ops)))
ax.set_xticklabels(unit_ops, rotation=45, ha='right')

# One legend entry per utility
handles, labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.05,1), loc='upper left')

plt.tight_layout()







with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
    # Existing sheets
    round_dataframe(df_distillation).to_excel(writer, sheet_name="Distillation", index=False)
    
    if 'df_hydrotreatment' in globals():
        round_dataframe(df_hydrotreatment).to_excel(writer, sheet_name="Hydrotreatment", index=False)
        round_dataframe(df_amine_gas_treating).to_excel(writer, sheet_name="Amine gas treating", index=False)

    if 'df_solvent_extraction' in globals():
        round_dataframe(df_solvent_extraction).to_excel(writer, sheet_name="Solvent extraction", index=False)

    # Capital costs sheet
    capital_costs_df.to_excel(writer, sheet_name="Capital Costs", index=False)

    # Operating costs sheet
    operating_costs_df.to_excel(writer, sheet_name="Operating Costs", index=False)

    # DCFROR Inputs sheet
    df_dcfror_pivot.to_excel(writer, sheet_name="DCFROR Inputs", index=False)

    # DCFROR Results sheet
    dcfror_df.to_excel(writer, sheet_name="DCFROR Results", index=False)

    # Equipment costs sheet
    equipment_costs.to_excel(writer, sheet_name="Equipment Costs", index=False)

    # Utilities sheet
    formatted_df_utilities.to_excel(writer, sheet_name="Utilities", index=False)

    # --- write both to same sheet ---
    lca_details_df.to_excel(writer, sheet_name="LCA Results", index=False, startrow=0)
    lca_summary_df.to_excel(writer, sheet_name="LCA Results", index=False, startrow=len(lca_details_df) + 3)

print(f"DataFrames successfully saved to Excel file: {output_file}")
