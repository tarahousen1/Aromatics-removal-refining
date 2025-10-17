import matplotlib.pyplot as plt
import numpy as np
import toml
import pandas as pd
from functions.hydrocracking_functions import find_lbhr_conversion
from functions.hydrocracking_functions import calculate_molecular_fractions
from functions.hydrocracking_functions import power_law_kinetic_model
from functions.hydrocracking_functions import langmuir_hinshelwood_model
from functions.hydrocracking_functions import multi_parameter_kinetic_model
from functions.hydrocracking_functions import compute_hydrogen_consumption
import os
from scipy.interpolate import interp1d
import math

from functions.perc_sulfur_content_in_SR_cuts_US_low_sulfur import interpolate_sulfur_in_product
from functions.approx_H2_hydrocracking_API_vs_C5_to_180 import interpolate_naphtha_vol
from functions.yields_C5_180_and_180_400_hydrocrackates import interpolate_C5_180
from functions.Kw_hydrocracker_products import interpolate_Kw_product
from functions.H2_content_of_hydrocarbons_hydrocracking import interpolate_H2_content
from functions.nitrogen_distribution import interpolate_N_content
from functions.aromatics_content_vs_h2_content import estimate_aromatics_content
from functions.aromatics_hydrogenation_10Mpa import interpolate_perc_aromatics_hydrogenation
from functions.aromatics_saturation_efficiency import interpolate_aromatics_saturation_efficiency
from functions.liquid_product_yield import interpolate_liquid_yield

# CHECKKKK
density_monoaromatics = 0.875
density_polyaromatics = 1.05

density_paraffins = 0.74
density_napthenes = 0.86


# Create output directory if it doesn't exist
output_dir = "/Users/tarahousen/Documents/Aromatics_removal_desulfurization_refining/outputs"
os.makedirs(output_dir, exist_ok=True)

# DISTILLATION -------------------------------------------------------------------------------------------------------------------

def calculate_api_gravity(specific_gravity):
    " Calculate API gravity given specific gravity  "
    return 141.5 / specific_gravity - 131.5

def calculate_specific_gravity(API):
    " Calculate specific gravity at 60F given API gravity  "
    return 141.5 / (API + 131.5)

def calculate_characterization_factor(Tb, SG):
    " Calculate chacterization factor (Kw) given mean avg boiling point in degrees Rankine and specific gravity at 60F  "
    Kw = (Tb**(1/3)) / SG
    return Kw

def calculate_specific_gravity_given_Kw(Tb, Kw):
    " Calculate specific gravity at 60F given mean avg boiling point in degrees Rankine and Kw  "

    SG = (Tb**(1/3)) / Kw
    return SG


nan = np.nan

# Load data from TOML files
user_inputs = toml.load('user_inputs.toml')
refinery_data = user_inputs['Refinery']
crude_data = user_inputs['North_Slope_Alaska_crude_data_cuts']['cut']
crude_oil_properties = user_inputs['North_Slope_Alaska_crude_oil_properties']
hydrotreatment_operating_parameters = user_inputs['hydrotreatment_operating_parameters']

refinery_type = refinery_data['refinery_type']
aromatics_removal_technique = refinery_data['aromatics_removal_technique']

refinery_utility_inputs = toml.load('refinery_utility_inputs.toml')
atmospheric_distillation_utility_inputs = refinery_utility_inputs['atmospheric_distillation_utility_data']
vacuum_distillation_utility_inputs = refinery_utility_inputs['vacuum_distillation_utility_data']
desalter_utility_inputs = refinery_utility_inputs['desalter_utility_data']
hydrocracking_utility_data = refinery_utility_inputs['hydrocracking_utility_data']
hydrotreatment_utility_data = refinery_utility_inputs['hydrotreatment_utility_data']

fixed_parameters = toml.load('fixed_parameters.toml')
conversion_parameters = fixed_parameters['conversion_parameters']


filtered_cuts = [cut for cut in crude_data if cut['high_temp'] != "None"]
volume_percent = np.array([cut['vol_percent'] for cut in filtered_cuts], dtype=np.float64)

upper_temperature = np.array([cut['high_temp'] for cut in filtered_cuts], dtype=np.float64)
lower_temperature = np.array([cut['low_temp'] for cut in filtered_cuts], dtype=np.float64)

mid_pt_temp = (lower_temperature + upper_temperature) / 2
sp_gravity = np.array([cut['sp_gr'] for cut in filtered_cuts], dtype=np.float64)

api_gravity = np.array([calculate_api_gravity(sg) for sg in sp_gravity])
interp_api_gravity_at_upper_temp = np.interp(upper_temperature, mid_pt_temp, api_gravity)

# Load density conversion table
file_path = "/Users/tarahousen/Documents/Aromatics_removal_desulfurization_refining/excel_files/density_conv_table.xlsx"
density_conv_table = pd.read_excel(file_path, sheet_name='Sheet1')

# Load data
file_path = "/Users/tarahousen/Documents/Aromatics_removal_desulfurization_refining/excel_files/tbp_data_with_api.xlsx"

df = pd.read_excel(file_path)
df['Cumulative Vol%'] = df['Vol%'].cumsum()
df['Mid Vol%'] = df['Vol%'].cumsum() - df['Vol%'] / 2

print(df)
# Filter out rows where 'TBP Low (°F)' is NaN
df = df[df['TBP Low (°F)'].notna()]

# Calculate midpoints and cumulative volume
df['TBP Mid (°F)'] = (df['TBP Low (°F)'] + df['TBP High (°F)']) / 2

# Plot TBP and Gravity Midpercent Curves for Crude Oil
fig, ax1 = plt.subplots(figsize=(8, 7))
color = 'tab:blue'
ax1.set_xlabel('Cumulative Volume Percent (%)')
ax1.set_ylabel('Boiling Point Temperature (°F)', color=color)
ax1.plot(df['Cumulative Vol%'], df['TBP High (°F)'], color=color, linestyle='--', label='TBP Temperature')
ax1.tick_params(axis='y', labelcolor=color)
ax1.grid(True)
ax1.set_ylim(0, 1200)
ax1.set_xlim(0, 100)
ax2 = ax1.twinx()
color = 'tab:green'
ax2.set_ylabel('API Gravity', color=color)
ax2.plot(df['Mid Vol%'], df['°API'], color=color, linestyle='-.', label='API Gravity')
ax2.tick_params(axis='y', labelcolor=color)
ax2.set_ylim(0, 120)


# Define cut points for crude oil fractions
atmospheric_distillation_cuts = {
    "Butanes and lighter": (0, 90),
    "LSR gasoline (90-190°F)": (90, 190),
    "HSR gasoline (190-330°F)": (190, 330),
    "Kerosine (330-480°F)": (330, 480),
    "Light gas oil (480-610°F)": (480, 610),
    "610°F +": (610, 1300)
}

vacuum_distillation_cuts = {
    "Heavy gas oil (610-800°F)": (610, 800),
    "Vacuum gas oil (800-1050°F)": (800, 1050),
    "1050°F +": (1050, 1300),
}

cut_colors = {
    "Butanes and lighter": 'lightgray',
    "LSR gasoline (90-190°F)": 'lightblue',
    "HSR gasoline (190-330°F)": 'lightgreen',
    "Kerosine (330-480°F)": 'khaki',
    "Light gas oil (480-610°F)": 'lightcoral',
    "Heavy gas oil (610-800°F)": 'plum',
    "Vacuum gas oil (800-1050°F)": 'powderblue'
}


# Atmospheric distillation
print("Atmospheric distillation unit:")
atmospheric_distillation_inputs = {
    'Volume %': {},
    'API': {},
    'Specific Gravity': {},
    'Characterization Factor': {},
    'BPCD': {},
    'lb/hr from bbl/day': {},
    'lb/hr': {},
    'wt% S': {},
    'lb/hr S': {},
    'wt %': {},
    'wppm N': {},
    'lb/hr N': {},

}

crude_oil_properties['specific_gravity'] = calculate_specific_gravity(crude_oil_properties['API'])
feed_lbhr_conv = find_lbhr_conversion(crude_oil_properties['specific_gravity'], density_conv_table)
sulfur_percent = float(crude_oil_properties['crude_sulfur_perc'])
crude_lbhr = feed_lbhr_conv * refinery_data['input_bpcd']
feed_sulfur_lbhr = sulfur_percent/100 * crude_lbhr

atmospheric_distillation_inputs['API']['Crude oil'] = crude_oil_properties['API']
atmospheric_distillation_inputs['Specific Gravity']['Crude oil'] = crude_oil_properties['specific_gravity']
atmospheric_distillation_inputs['BPCD']['Crude oil'] = refinery_data['input_bpcd']
atmospheric_distillation_inputs['lb/hr from bbl/day']['Crude oil'] = feed_lbhr_conv if not np.isnan(feed_lbhr_conv) else 'N/A'
atmospheric_distillation_inputs['lb/hr']['Crude oil'] =  crude_lbhr if not np.isnan(crude_lbhr) else 'N/A'
atmospheric_distillation_inputs['wt% S']['Crude oil'] = sulfur_percent if not np.isnan(sulfur_percent) else 'N/A'
atmospheric_distillation_inputs['lb/hr S']['Crude oil'] = feed_sulfur_lbhr if not np.isnan(feed_sulfur_lbhr) else 'N/A'
atmospheric_distillation_inputs['wppm N']['Crude oil'] = crude_oil_properties['wppm_N_crude']
atmospheric_distillation_inputs['lb/hr N']['Crude oil'] = (crude_oil_properties['wppm_N_crude'] / 1000000) * atmospheric_distillation_inputs['lb/hr']['Crude oil']

atmospheric_distillation_outputs = {
    'Volume %': {},
    'API': {},
    'Specific Gravity': {},
    'Characterization Factor': {},
    'BPCD': {},
    'lb/hr from bbl/day': {},
    'lb/hr': {},
    'wt% S': {},
    'lb/hr S': {},
    'wt %': {},
    'wt% H2': {},
    'lb/hr H2': {},
    'wppm N': {},
    'lb/hr N': {},
    'molecular weight': {},
    'Refractory index at 20C': {},
    'Refractory intercept': {},
    'vol% paraffins': {},
    'BPCD paraffins': {},
    'lb/hr paraffins': {},
    'wt% paraffins': {},
    'vol% napthenes': {},
    'BPCD napthenes': {},
    'lb/hr napthenes': {},
    'wt% napthenes': {},
    'vol% aromatics': {},
    'vol% monoaromatics': {},
    'BPCD monoaromatics': {},
    'lb/hr monoaromatics': {},
    'wt% monoaromatics': {},
    'vol% polyaromatics': {},
    'BPCD polyaromatics': {},
    'lb/hr polyaromatics': {},
    'wt% polyaromatics': {},
}

def caclulate_MeABP_given_VABP(VABP, T_90, T_10):
    # Convert temp to kelvin
    T_90 = (T_90 - 32) * 5/9 + 273.15
    T_10 = (T_10 - 32) * 5/9 + 273.15

    # Calculate slope
    SL = (T_90 - T_10) / 80

    # Convert temp to kelvin
    VABP = (VABP - 32) * 5/9 + 273.15

    exponent = -1.53181 - 0.0128 * (VABP - 273.15)**0.6667 + 3.646064 * SL**0.333
    delta_T = np.exp(exponent)

    MeABP = VABP - delta_T

    # Convert temp to F
    MeABP = (MeABP - 273.15) * 9/5 + 32

    return MeABP

def interpolate_volume(temp_query):
    " Linearly interpolate volume_percent for a given temperature "
    return np.interp(temp_query, df['TBP High (°F)'], df['Cumulative Vol%'])


def interpolate_temp(volume):
    " Linearly interpolate temperature for a given volume_percent "
    return np.interp(volume, df['Cumulative Vol%'], df['TBP High (°F)'])


raw_h2_data = user_inputs['North_Slope_Alaska_crude_data_H2_content']['hydrogen_by_temp_f']
unique_h2_data = sorted({entry['temp_f']: entry['hydrogen_wt_percent'] for entry in raw_h2_data}.items())
h2_temps, h2_wt_percents = zip(*unique_h2_data)
interp_h2 = interp1d(h2_temps, h2_wt_percents, fill_value="extrapolate")       

for cut_name, (start_temp, end_temp) in atmospheric_distillation_cuts.items():
    if cut_name == '610°F +':
        continue  # Skip this cut

    if cut_name == "Butanes and lighter":
        vol_start = 0
    else:
        vol_start = interpolate_volume(start_temp)                      # Interpolate volume at lower temp bound of cut
    vol_end = interpolate_volume(end_temp)                              # Interpolate volume at upper temp bound of cut

    cut_volume = vol_end - vol_start   

    # Calculate API for cut                                        
    vol_avg_boiling_pt = (start_temp + end_temp) / 2                                           # Calculate mean avg boiling pt

    SG = np.interp(vol_avg_boiling_pt, mid_pt_temp, sp_gravity)         # Interpolate SG at mean avg boiling pt

    api = calculate_api_gravity(SG)

    lbhr_conv = find_lbhr_conversion(SG, density_conv_table)             # Obtain BPCD to lb/hr conversion from density conversion table
    bpcd = cut_volume/100 * atmospheric_distillation_inputs['BPCD']['Crude oil']                # Calculate BCPD
    lbhr_value = lbhr_conv * bpcd                                                               # Calculate lb/hr

    T_90 = interpolate_temp(90)
    T_10 = interpolate_temp(10)

    mean_avg_boiling_pt = caclulate_MeABP_given_VABP(vol_avg_boiling_pt, T_90, T_10)    # MeABP in F
    print(cut_name)
    print(mean_avg_boiling_pt)

    sulfur_wt_perc = interpolate_sulfur_in_product(crude_oil_properties['crude_sulfur_perc'], vol_avg_boiling_pt)      # Determine percent weight Sulfur content from plot in 'perc_sulfur_content_in_SR_cuts_US_low_sulfur'
    sulfur_lb_hr = sulfur_wt_perc/100 * lbhr_value              # Determine lb/hr Sulfur
    print(sulfur_wt_perc)

    mean_avg_boiling_pt_Rankine = mean_avg_boiling_pt + 459.67
    k_w = calculate_characterization_factor(mean_avg_boiling_pt_Rankine, SG)  
    
    atmospheric_distillation_outputs['Volume %'][cut_name] = cut_volume
    atmospheric_distillation_outputs['API'][cut_name] = api
    atmospheric_distillation_outputs['Specific Gravity'][cut_name] = SG
    atmospheric_distillation_outputs['Characterization Factor'][cut_name] = k_w
    atmospheric_distillation_outputs['BPCD'][cut_name] = bpcd
    atmospheric_distillation_outputs['lb/hr from bbl/day'][cut_name] = lbhr_conv if not np.isnan(lbhr_conv) else 'N/A'
    atmospheric_distillation_outputs['lb/hr'][cut_name] = lbhr_value if not np.isnan(lbhr_conv) else 'N/A'
    atmospheric_distillation_outputs['wt% S'][cut_name] = sulfur_wt_perc if not np.isnan(sulfur_wt_perc) else 'N/A'
    atmospheric_distillation_outputs['lb/hr S'][cut_name] = sulfur_lb_hr if not np.isnan(sulfur_lb_hr) else 'N/A'

    ax1.axvspan(vol_start, vol_end, color=cut_colors.get(cut_name, 'grey'), alpha=0.3)
    mid_vol = (vol_start + vol_end) / 2
    ax1.text(mid_vol, mean_avg_boiling_pt + 20, cut_name, rotation=90, ha='center', va='bottom', fontsize=8, color='black')

    # Interpolate H2 wt% for the cut
    h2_wt_percent = float(interp_h2(end_temp))  # wt% H2
    h2_lbhr = h2_wt_percent / 100 * lbhr_value  # lb/hr H2
    atmospheric_distillation_outputs['wt% H2'][cut_name] = h2_wt_percent
    atmospheric_distillation_outputs['lb/hr H2'][cut_name] = h2_lbhr

    perc_of_crude_N_content = interpolate_N_content(start_temp, end_temp)

    atmospheric_distillation_outputs['lb/hr N'][cut_name] = perc_of_crude_N_content * atmospheric_distillation_inputs['lb/hr N']['Crude oil'] / 100
    atmospheric_distillation_outputs['wppm N'][cut_name] = 1000000 * atmospheric_distillation_outputs['lb/hr N'][cut_name] / atmospheric_distillation_outputs['lb/hr'][cut_name] 

    Tb_K = mean_avg_boiling_pt_Rankine * 5/9

    M, n, R_i, x_P, x_N, x_MA, x_PA, x_A = calculate_molecular_fractions(atmospheric_distillation_outputs['Specific Gravity'][cut_name], Tb_K, atmospheric_distillation_outputs['Characterization Factor'][cut_name], atmospheric_distillation_outputs['API'][cut_name], atmospheric_distillation_outputs['wt% H2'][cut_name])
    atmospheric_distillation_outputs['molecular weight'][cut_name] = M
    atmospheric_distillation_outputs['Refractory index at 20C'][cut_name] = n
    atmospheric_distillation_outputs['Refractory intercept'][cut_name] = R_i

    if x_P is not None and x_N is not None and x_MA is not None and x_PA is not None:
        atmospheric_distillation_outputs['vol% paraffins'][cut_name] = x_P * 100
        atmospheric_distillation_outputs['BPCD paraffins'][cut_name] = x_P * atmospheric_distillation_outputs['BPCD'][cut_name]
        atmospheric_distillation_outputs['lb/hr paraffins'][cut_name] = atmospheric_distillation_outputs['lb/hr from bbl/day'][cut_name] * atmospheric_distillation_outputs['BPCD paraffins'][cut_name] 
        atmospheric_distillation_outputs['wt% paraffins'][cut_name] = 100 * atmospheric_distillation_outputs['lb/hr paraffins'][cut_name] / atmospheric_distillation_outputs['lb/hr'][cut_name]

        atmospheric_distillation_outputs['vol% napthenes'][cut_name] = x_N * 100
        atmospheric_distillation_outputs['BPCD napthenes'][cut_name] = x_N * atmospheric_distillation_outputs['BPCD'][cut_name]
        atmospheric_distillation_outputs['lb/hr napthenes'][cut_name] = atmospheric_distillation_outputs['lb/hr from bbl/day'][cut_name] * atmospheric_distillation_outputs['BPCD napthenes'][cut_name] 
        atmospheric_distillation_outputs['wt% napthenes'][cut_name] = 100 * atmospheric_distillation_outputs['lb/hr napthenes'][cut_name] / atmospheric_distillation_outputs['lb/hr'][cut_name]

        atmospheric_distillation_outputs['vol% aromatics'][cut_name] = x_A * 100

        atmospheric_distillation_outputs['vol% monoaromatics'][cut_name] = x_MA * 100
        atmospheric_distillation_outputs['BPCD monoaromatics'][cut_name] = x_MA * atmospheric_distillation_outputs['BPCD'][cut_name]
        atmospheric_distillation_outputs['lb/hr monoaromatics'][cut_name] = atmospheric_distillation_outputs['lb/hr from bbl/day'][cut_name] * atmospheric_distillation_outputs['BPCD monoaromatics'][cut_name] 
        atmospheric_distillation_outputs['wt% monoaromatics'][cut_name] = 100 * atmospheric_distillation_outputs['lb/hr monoaromatics'][cut_name] / atmospheric_distillation_outputs['lb/hr'][cut_name]

        atmospheric_distillation_outputs['vol% polyaromatics'][cut_name] = x_PA * 100
        atmospheric_distillation_outputs['BPCD polyaromatics'][cut_name] = x_PA * atmospheric_distillation_outputs['BPCD'][cut_name]
        atmospheric_distillation_outputs['lb/hr polyaromatics'][cut_name] = atmospheric_distillation_outputs['lb/hr from bbl/day'][cut_name] * atmospheric_distillation_outputs['BPCD polyaromatics'][cut_name] 
        atmospheric_distillation_outputs['wt% polyaromatics'][cut_name] = 100 * atmospheric_distillation_outputs['lb/hr polyaromatics'][cut_name] / atmospheric_distillation_outputs['lb/hr'][cut_name]
    
    else:
        print(f"Skipping cut {cut_name} due to invalid molecular fraction calculation")

    #atmospheric_distillation_outputs['wt% paraffins'][cut_name] = x_P * density_paraffins / (x_P * density_paraffins + x_N * density_napthenes + x_MA * density_monoaromatics + x_PA * density_polyaromatics) * 100
    #atmospheric_distillation_outputs['wt% napthenes'][cut_name] = x_N * density_napthenes/ (x_P * density_paraffins + x_N * density_napthenes + x_MA * density_monoaromatics + x_PA * density_polyaromatics) * 100

    #atmospheric_distillation_outputs['vol% monoaromatics'][cut_name] = x_MA
    #atmospheric_distillation_outputs['wt% monoaromatics'][cut_name] = x_MA * density_monoaromatics/ (x_P * density_paraffins + x_N * density_napthenes + x_MA * density_monoaromatics + x_PA * density_polyaromatics) * 100

    #atmospheric_distillation_outputs['vol% polyaromatics'][cut_name] = x_PA
    #atmospheric_distillation_outputs['wt% polyaromatics'][cut_name] = x_PA * density_polyaromatics/ (x_P * density_paraffins + x_N * density_napthenes + x_MA * density_monoaromatics + x_PA * density_polyaromatics) * 100


total_atm_outputs_lbhr = sum(atmospheric_distillation_outputs['lb/hr'].values())
residue_lbhr = crude_lbhr - total_atm_outputs_lbhr
atmospheric_distillation_outputs['lb/hr']['610°F +'] = residue_lbhr

atmospheric_distillation_outputs_610F_N_content = interpolate_N_content(610, 1300)
atmospheric_distillation_outputs['lb/hr N']['610°F +'] = atmospheric_distillation_outputs_610F_N_content * atmospheric_distillation_inputs['lb/hr N']['Crude oil'] / 100
atmospheric_distillation_outputs['wppm N']['610°F +'] = 1000000 * atmospheric_distillation_outputs['lb/hr N']['610°F +'] / atmospheric_distillation_outputs['lb/hr']['610°F +'] 

total_atm_outputs_S_lbhr = sum(                                                                         
    value for value in atmospheric_distillation_outputs['lb/hr S'].values() if value != 'N/A'
)
residue_S_lbhr = feed_sulfur_lbhr - total_atm_outputs_S_lbhr                                        # Calculate lb/hr Sulfur in 610F + stream
atmospheric_distillation_outputs['lb/hr S']['610°F +'] = residue_S_lbhr
atmospheric_distillation_outputs['wt% S']['610°F +'] = 100 * residue_S_lbhr / residue_lbhr          # Calculate wt% Sulfur in 610F + stream

# Get average specific gravity of cuts above 610°F
specific_gravity_above_610 = sp_gravity[mid_pt_temp >= 610]
if len(specific_gravity_above_610) > 0:
    atmospheric_distillation_outputs['Specific Gravity']['610°F +'] = sum(specific_gravity_above_610) / len(specific_gravity_above_610)
else:
    atmospheric_distillation_outputs['Specific Gravity']['610°F +'] = 'N/A'

# Calculate API of 610F + stream
atmospheric_distillation_outputs['API']['610°F +'] = calculate_api_gravity(atmospheric_distillation_outputs['Specific Gravity']['610°F +'])

atmospheric_distillation_outputs['lb/hr from bbl/day']['610°F +'] = find_lbhr_conversion(atmospheric_distillation_outputs['Specific Gravity']['610°F +'], density_conv_table)
atmospheric_distillation_outputs['BPCD']['610°F +'] = atmospheric_distillation_outputs['lb/hr']['610°F +'] / atmospheric_distillation_outputs['lb/hr from bbl/day']['610°F +']
atmospheric_distillation_outputs['Volume %']['610°F +'] = 100 * atmospheric_distillation_outputs['BPCD']['610°F +'] / atmospheric_distillation_inputs['BPCD']['Crude oil']

for cut_name, (start_temp, end_temp) in atmospheric_distillation_cuts.items():
    atmospheric_distillation_outputs['wt %'][cut_name] = atmospheric_distillation_outputs['lb/hr'][cut_name] / total_atm_outputs_lbhr

df_atmospheric_distillation_inputs = pd.DataFrame(atmospheric_distillation_inputs).reset_index().rename(columns={'index': 'Cut'})
df_atmospheric_distillation_outputs = pd.DataFrame(atmospheric_distillation_outputs).reset_index().rename(columns={'index': 'Cut'})

inputs_label = pd.DataFrame([['Inputs'] + [''] * (df_atmospheric_distillation_inputs.shape[1] - 1)],
                            columns=df_atmospheric_distillation_inputs.columns)
outputs_label = pd.DataFrame([['Outputs'] + [''] * (df_atmospheric_distillation_inputs.shape[1] - 1)],
                             columns=df_atmospheric_distillation_inputs.columns)

# Calculate total inputs and outputs for BPCD, lb/hr, and sulfur lb/hr
total_atm_inputs_bpcd = df_atmospheric_distillation_inputs['BPCD'].apply(pd.to_numeric, errors='coerce').sum()
total_atm_outputs_bpcd = df_atmospheric_distillation_outputs['BPCD'].apply(pd.to_numeric, errors='coerce').sum()

total_atm_inputs_lbhr = df_atmospheric_distillation_inputs['lb/hr'].replace(',', '', regex=True).apply(pd.to_numeric, errors='coerce').sum()
total_atm_outputs_lbhr = df_atmospheric_distillation_outputs['lb/hr'].replace(',', '', regex=True).apply(pd.to_numeric, errors='coerce').sum()

total_atm_inputs_sulfur = df_atmospheric_distillation_inputs['lb/hr S'].apply(pd.to_numeric, errors='coerce').sum()
total_atm_outputs_sulfur = df_atmospheric_distillation_outputs['lb/hr S'].apply(pd.to_numeric, errors='coerce').sum()

total_atm_inputs_row = {
    'Cut': 'Total',
    'BPCD': total_atm_inputs_bpcd,
    'lb/hr': total_atm_inputs_lbhr,
    'lb/hr S': total_atm_inputs_sulfur
}
total_atm_outputs_row = {
    'Cut': 'Total',
    'BPCD': total_atm_outputs_bpcd,
    'lb/hr': total_atm_outputs_lbhr,
    'lb/hr S': total_atm_outputs_sulfur
}

# Convert to DataFrames
df_total_atm_inputs_row = pd.DataFrame([total_atm_inputs_row], columns=df_atmospheric_distillation_inputs.columns)
df_total_atm_outputs_row = pd.DataFrame([total_atm_outputs_row], columns=df_atmospheric_distillation_outputs.columns)

# Insert totals after inputs and outputs
df_atmospheric_distillation = pd.concat([
    inputs_label,
    df_atmospheric_distillation_inputs,
    df_total_atm_inputs_row,
    outputs_label,
    df_atmospheric_distillation_outputs,
    df_total_atm_outputs_row
], ignore_index=True)

df_atmospheric_distillation.fillna('', inplace=True)
print(df_atmospheric_distillation)


print("Vacuum distillation unit:")
vacuum_distillation_inputs = {
    'Volume %': {},
    'API': {},
    'Specific Gravity': {},
    'Characterization Factor': {},
    'BPCD': {},
    'lb/hr from bbl/day': {},
    'lb/hr': {},
    'wt% S': {},
    'lb/hr S': {},
    'wt %': {},
    'wppm N': {},
    'lb/hr N': {},
}

vacuum_cut_data = df_atmospheric_distillation_outputs[df_atmospheric_distillation_outputs['Cut'] == '610°F +'].iloc[0]

vacuum_distillation_inputs = {
    'Volume %': {'610°F +': vacuum_cut_data['Volume %']},
    'API': {'610°F +': vacuum_cut_data['API']},
    'Specific Gravity': {'610°F +': vacuum_cut_data['Specific Gravity']},
    'Characterization Factor': {'610°F +': vacuum_cut_data['Characterization Factor']},
    'BPCD': {'610°F +': vacuum_cut_data['BPCD']},
    'lb/hr from bbl/day': {'610°F +': vacuum_cut_data['lb/hr from bbl/day']},
    'lb/hr': {'610°F +': vacuum_cut_data['lb/hr']},
    'wt% S': {'610°F +': vacuum_cut_data['wt% S']},
    'lb/hr S': {'610°F +': vacuum_cut_data['lb/hr S']},
    'wppm N': {'610°F +': vacuum_cut_data['wppm N']},
    'lb/hr N':{'610°F +': vacuum_cut_data['lb/hr N']},
}

vacuum_distillation_outputs = {
    'Volume %': {},
    'API': {},
    'Specific Gravity': {},
    'Characterization Factor': {},
    'BPCD': {},
    'lb/hr from bbl/day': {},
    'lb/hr': {},
    'wt% S': {},
    'lb/hr S': {},
    'wt %': {},
    'wt% H2': {},
    'lb/hr H2': {},
    'wppm N': {},
    'lb/hr N': {},
}


for cut_name, (start_temp, end_temp) in vacuum_distillation_cuts.items():
    if cut_name == '1050°F +':
        continue  # Skip this cut
    vol_start = interpolate_volume(start_temp)                          # Interpolate volume at lower temp bound of cut
    vol_end = interpolate_volume(end_temp)                              # Interpolate volume at upper temp bound of cut

    cut_volume = vol_end - vol_start                                                        # Cut volume percentage

    # Calculate API for cut                                        
    vol_avg_boiling_pt = (start_temp + end_temp) / 2                                           # Calculate mean avg boiling pt

    SG = np.interp(vol_avg_boiling_pt, mid_pt_temp, sp_gravity)         # Interpolate SG at mean avg boiling pt

    api = calculate_api_gravity(SG)

    lbhr_conv = find_lbhr_conversion(SG, density_conv_table)             # Obtain BPCD to lb/hr conversion from density conversion table
    bpcd = cut_volume/100 * atmospheric_distillation_inputs['BPCD']['Crude oil']                # Calculate BCPD
    lbhr_value = lbhr_conv * bpcd                                                               # Calculate lb/hr

    T_90 = interpolate_temp(90)
    T_10 = interpolate_temp(10)

    mean_avg_boiling_pt = caclulate_MeABP_given_VABP(vol_avg_boiling_pt, T_90, T_10)

    sulfur_wt_perc = interpolate_sulfur_in_product(crude_oil_properties['crude_sulfur_perc'], mean_avg_boiling_pt)      # Determine percent weight Sulfur content from plot in 'perc_sulfur_content_in_SR_cuts_US_low_sulfur'
    sulfur_lb_hr = sulfur_wt_perc/100 * lbhr_value              # Determine lb/hr Sulfur

    mean_avg_boiling_pt_Rankine = mean_avg_boiling_pt + 459.67
    k_w = calculate_characterization_factor(mean_avg_boiling_pt_Rankine, SG)  

    vacuum_distillation_outputs['Volume %'][cut_name] = cut_volume
    vacuum_distillation_outputs['API'][cut_name] = api
    vacuum_distillation_outputs['Specific Gravity'][cut_name] = SG
    vacuum_distillation_outputs['Characterization Factor'][cut_name] = k_w
    vacuum_distillation_outputs['BPCD'][cut_name] = bpcd
    vacuum_distillation_outputs['lb/hr from bbl/day'][cut_name] = lbhr_conv if not np.isnan(lbhr_conv) else 'N/A'
    vacuum_distillation_outputs['lb/hr'][cut_name] = lbhr_value if not np.isnan(lbhr_conv) else 'N/A'
    vacuum_distillation_outputs['wt% S'][cut_name] = sulfur_wt_perc if not np.isnan(sulfur_wt_perc) else 'N/A'
    vacuum_distillation_outputs['lb/hr S'][cut_name] = sulfur_lb_hr if not np.isnan(sulfur_lb_hr) else 'N/A'

    # Interpolate H2 wt% for the cut
    h2_wt_percent = float(interp_h2(end_temp))  # wt% H2
    h2_lbhr = h2_wt_percent / 100 * lbhr_value             # lb/hr H2

    vacuum_distillation_outputs['wt% H2'][cut_name] = h2_wt_percent
    vacuum_distillation_outputs['lb/hr H2'][cut_name] = h2_lbhr

    perc_of_crude_N_content = interpolate_N_content(start_temp, end_temp)
    vacuum_distillation_outputs['lb/hr N'][cut_name] = perc_of_crude_N_content * atmospheric_distillation_inputs['lb/hr N']['Crude oil'] / 100
    vacuum_distillation_outputs['wppm N'][cut_name] = 1000000 * vacuum_distillation_outputs['lb/hr N'][cut_name] / vacuum_distillation_outputs['lb/hr'][cut_name] 


    ax1.axvspan(vol_start, vol_end, color=cut_colors.get(cut_name, 'grey'), alpha=0.3)
    if cut_name == "Vacuum gas oil (800-1050°F)":
        label = "Vacuum gas oil\n(800–1050°F)"
    else:
        label = cut_name
    mid_vol = (vol_start + vol_end) / 2
    ax1.text(mid_vol, mean_avg_boiling_pt + 20, label, rotation=90, ha='center', va='bottom', fontsize=8, color='black')

total_vac_outputs_lbhr = sum(vacuum_distillation_outputs['lb/hr'].values())
vac_residue_lbhr = residue_lbhr - total_vac_outputs_lbhr
vacuum_distillation_outputs['lb/hr']['1050°F +'] = vac_residue_lbhr

vacuum_distillation_outputs_1050F_N_content = interpolate_N_content(1050, 1300)
vacuum_distillation_outputs['lb/hr N']['1050°F +'] = vacuum_distillation_outputs_1050F_N_content * atmospheric_distillation_inputs['lb/hr N']['Crude oil'] / 100
vacuum_distillation_outputs['wppm N']['1050°F +'] = 1000000 * vacuum_distillation_outputs['lb/hr N']['1050°F +'] / vacuum_distillation_outputs['lb/hr']['1050°F +'] 


total_vac_outputs_S_lbhr = sum(                                                                         
    value for value in vacuum_distillation_outputs['lb/hr S'].values() if value != 'N/A'
)
vac_residue_S_lbhr = residue_S_lbhr - total_vac_outputs_S_lbhr                                        # Calculate lb/hr Sulfur in 610F + stream
vacuum_distillation_outputs['lb/hr S']['1050°F +'] = vac_residue_S_lbhr
vacuum_distillation_outputs['wt% S']['1050°F +'] = 100 * vac_residue_S_lbhr / vac_residue_lbhr    # Calculate wt% Sulfur in 610F + stream

# Get average specific gravity of cuts above 1050°F
specific_gravity_above_1050 = sp_gravity[mid_pt_temp >= 1050]
if len(specific_gravity_above_610) > 0:
    vacuum_distillation_outputs['Specific Gravity']['1050°F +'] = sum(specific_gravity_above_1050) / len(specific_gravity_above_1050)
elif len(specific_gravity_above_610) == 1:
    vacuum_distillation_outputs['Specific Gravity']['1050°F +'] = specific_gravity_above_1050
else:
    vacuum_distillation_outputs['Specific Gravity']['1050°F +'] = 'N/A'

# Calculate API of 610F + stream
vacuum_distillation_outputs['API']['1050°F +'] = calculate_api_gravity(vacuum_distillation_outputs['Specific Gravity']['1050°F +'])

vacuum_distillation_outputs['lb/hr from bbl/day']['1050°F +'] = find_lbhr_conversion(vacuum_distillation_outputs['Specific Gravity']['1050°F +'], density_conv_table)
vacuum_distillation_outputs['BPCD']['1050°F +'] = vacuum_distillation_outputs['lb/hr']['1050°F +'] / vacuum_distillation_outputs['lb/hr from bbl/day']['1050°F +']
vacuum_distillation_outputs['Volume %']['1050°F +'] = 100 * vacuum_distillation_outputs['BPCD']['1050°F +'] / vacuum_distillation_inputs['BPCD']['610°F +']

for cut_name, (start_temp, end_temp) in vacuum_distillation_cuts.items():
    vacuum_distillation_outputs['wt %'][cut_name] = vacuum_distillation_outputs['lb/hr'][cut_name] / residue_lbhr

df_vacuum_distillation_inputs = pd.DataFrame(vacuum_distillation_inputs).reset_index().rename(columns={'index': 'Cut'})
df_vacuum_distillation_outputs = pd.DataFrame(vacuum_distillation_outputs).reset_index().rename(columns={'index': 'Cut'})

inputs_label = pd.DataFrame([['Inputs'] + [''] * (df_vacuum_distillation_inputs.shape[1] - 1)],
                            columns=df_vacuum_distillation_inputs.columns)
outputs_label = pd.DataFrame([['Outputs'] + [''] * (df_vacuum_distillation_inputs.shape[1] - 1)],
                             columns=df_vacuum_distillation_inputs.columns)

total_vac_inputs_bpcd = df_vacuum_distillation_inputs['BPCD'].apply(pd.to_numeric, errors='coerce').sum()
total_vac_outputs_bpcd = df_vacuum_distillation_outputs['BPCD'].apply(pd.to_numeric, errors='coerce').sum()

total_vac_inputs_lbhr = df_vacuum_distillation_inputs['lb/hr'].replace(',', '', regex=True).apply(pd.to_numeric, errors='coerce').sum()
total_vac_outputs_lbhr = df_vacuum_distillation_outputs['lb/hr'].replace(',', '', regex=True).apply(pd.to_numeric, errors='coerce').sum()

total_vac_inputs_sulfur = df_vacuum_distillation_inputs['lb/hr S'].apply(pd.to_numeric, errors='coerce').sum()
total_vac_outputs_sulfur = df_vacuum_distillation_outputs['lb/hr S'].apply(pd.to_numeric, errors='coerce').sum()

total_inputs_row = {
    'Cut': 'Total',
    'BPCD': total_vac_inputs_bpcd,
    'lb/hr': total_vac_inputs_lbhr,
    'lb/hr S': total_vac_inputs_sulfur
}
total_outputs_row = {
    'Cut': 'Total',
    'BPCD': total_vac_outputs_bpcd,
    'lb/hr': total_vac_outputs_lbhr,
    'lb/hr S': total_vac_outputs_sulfur
}

df_total_inputs_row = pd.DataFrame([total_inputs_row], columns=df_vacuum_distillation_inputs.columns)
df_total_outputs_row = pd.DataFrame([total_outputs_row], columns=df_vacuum_distillation_outputs.columns)

# --- Insert totals after inputs and outputs ---
df_vacuum_distillation = pd.concat([
    inputs_label,
    df_vacuum_distillation_inputs,
    df_total_inputs_row,
    outputs_label,
    df_vacuum_distillation_outputs,
    df_total_outputs_row
], ignore_index=True)

df_vacuum_distillation.fillna('', inplace=True)
print(df_vacuum_distillation)

plt.title('TBP and Gravity Midpercent Curves - North Slope Alaska Crude')
fig.tight_layout()
fig.savefig('outputs/tbp_gravity_midpercent_curves.png', dpi=300)

# Determine utilities for atmospheric distillation
desalter_utilities = {
    key: (
        utility_data["amount"] * total_atm_inputs_bpcd,
        utility_data["unit"]
    )
    for key, utility_data in desalter_utility_inputs.items()
}

# Determine utilities for atmospheric distillation
daily_atmospheric_dist_utilities = {
    key: (
        utility_data["amount"] * total_atm_inputs_bpcd,
        utility_data["unit"]
    )
    for key, utility_data in atmospheric_distillation_utility_inputs.items()
}

# Determine utilities for vaccum distillation
daily_vacuum_dist_utilities = {
    key: (
        utility_data["amount"] * total_vac_inputs_bpcd,
        utility_data["unit"]
    )
    for key, utility_data in vacuum_distillation_utility_inputs.items()
}


# HYDROCRACKING -------------------------------------------------------------------------------------------------------------------

if aromatics_removal_technique == 'hydrocracking':
    # Initialize hydrocracking inputs and outputs
    hydro_cracking_inputs = {
    'Volume %': {},
    'API': {},
    'Specific Gravity': {},
    'Characterization Factor': {},
    'BPCD': {},
    'lb/hr from bbl/day': {},
    'lb/hr': {},
    'wt% S': {},
    'lb/hr S': {},
    'wt% H2': {},
    'lb/hr H2' : {},
    'lb/hr N': {},
    'wppm N': {},
    'molecular weight': {},
    'Refractory index at 20C': {},
    'Refractory intercept': {},
    'vol% paraffins': {},
    'BPCD paraffins': {},
    'lb/hr paraffins': {},
    'wt% paraffins': {},
    'vol% napthenes': {},
    'BPCD napthenes': {},
    'lb/hr napthenes': {},
    'wt% napthenes': {},
    'vol% aromatics': {},
    'vol% monoaromatics': {},
    'BPCD monoaromatics': {},
    'lb/hr monoaromatics': {},
    'wt% monoaromatics': {},
    'vol% polyaromatics': {},
    'BPCD polyaromatics': {},
    'lb/hr polyaromatics': {},
    'wt% polyaromatics': {},
    }

    hydro_cracking_outputs = {
    'Volume %': {},
    'API': {},
    'Specific Gravity': {},
    'Characterization Factor': {},
    'BPCD': {},
    'lb/hr from bbl/day': {},
    'wt %' : {},
    'lb/hr': {},
    'wt% S': {},
    'lb/hr S': {},
    'wt% H2': {},
    'lb/hr H2' : {},
    'lb/hr N': {},
    'wppm N': {},
    'molecular weight': {},
    'Refractory index at 20C': {},
    'Refractory intercept': {},
    'vol% paraffins': {},
    'BPCD paraffins': {},
    'lb/hr paraffins': {},
    'wt% paraffins': {},
    'vol% napthenes': {},
    'BPCD napthenes': {},
    'lb/hr napthenes': {},
    'wt% napthenes': {},
    'vol% aromatics': {},
    'vol% monoaromatics': {},
    'BPCD monoaromatics': {},
    'lb/hr monoaromatics': {},
    'wt% monoaromatics': {},
    'vol% polyaromatics': {},
    'BPCD polyaromatics': {},
    'lb/hr polyaromatics': {},
    'wt% polyaromatics': {},
    }

    # Feed for hydrotreatment is kerosine, light gas oil, heavy gas oil
    hydro_cracking_cuts = {
        #'Light gas oil (480-610°F)': atmospheric_distillation_outputs,
        'Heavy gas oil (610-800°F)': vacuum_distillation_outputs,
        'Vacuum gas oil (800-1050°F)': vacuum_distillation_outputs,
    }

    properties = ['BPCD', 'API', 'Specific Gravity', 'Characterization Factor', 'lb/hr from bbl/day', 'lb/hr', 'wt% S', 'lb/hr S', 'wt% H2', 'lb/hr H2', 'lb/hr N', 'wppm N', 'molecular weight', 'Refractory index at 20C', 'Refractory intercept', 'vol% paraffins', 'BPCD paraffins', 'lb/hr paraffins', 'wt% paraffins', 'vol% napthenes', 'BPCD napthenes', 'lb/hr napthenes', 'wt% napthenes', 'vol% aromatics', 'vol% monoaromatics', 'BPCD monoaromatics', 'lb/hr monoaromatics', 'wt% monoaromatics', 'vol% polyaromatics', 'BPCD polyaromatics', 'lb/hr polyaromatics', 'wt% polyaromatics']

    for prop in properties:
        for cut, source_df in hydro_cracking_cuts.items():
            hydro_cracking_inputs[prop][cut] = source_df[prop][cut]

    total_hydro_cracking_inputs_bpcd = sum(hydro_cracking_inputs['BPCD'].values())

    # Hydrocracking H2 input
    h2_input_cracking = 2000 # scf/bbl

    #hydro_cracking_inputs['Volume %']['Light gas oil (480-610°F)'] = 100 * hydro_cracking_inputs['BPCD']['Light gas oil (480-610°F)'] / total_hydro_cracking_inputs_bpcd
    hydro_cracking_inputs['Volume %']['Heavy gas oil (610-800°F)'] = 100 * hydro_cracking_inputs['BPCD']['Heavy gas oil (610-800°F)'] / total_hydro_cracking_inputs_bpcd
    hydro_cracking_inputs['Volume %']['Vacuum gas oil (800-1050°F)'] = 100 * hydro_cracking_inputs['BPCD']['Vacuum gas oil (800-1050°F)'] / total_hydro_cracking_inputs_bpcd

    #hydrocracking_feed_specific_gravity = (hydro_cracking_inputs['Volume %']['Light gas oil (480-610°F)'] * hydro_cracking_inputs['Specific Gravity']['Light gas oil (480-610°F)'] + hydro_cracking_inputs['Volume %']['Heavy gas oil (610-800°F)'] * hydro_cracking_inputs['Specific Gravity']['Heavy gas oil (610-800°F)'] + hydro_cracking_inputs['Volume %']['Vacuum gas oil (800-1050°F)'] * hydro_cracking_inputs['Specific Gravity']['Vacuum gas oil (800-1050°F)']) / 100
    hydrocracking_feed_specific_gravity = (hydro_cracking_inputs['Volume %']['Heavy gas oil (610-800°F)'] * hydro_cracking_inputs['Specific Gravity']['Heavy gas oil (610-800°F)'] + hydro_cracking_inputs['Volume %']['Vacuum gas oil (800-1050°F)'] * hydro_cracking_inputs['Specific Gravity']['Vacuum gas oil (800-1050°F)']) / 100
    hydrocracking_feed_avg_temp_Rankine = (480 + 1050) / 2 + 491.67

    hydro_cracking_feed_Kw = calculate_characterization_factor(hydrocracking_feed_avg_temp_Rankine, hydrocracking_feed_specific_gravity)
    print(f'Kw of hydrocracking feed: {hydro_cracking_feed_Kw}')

    hydrocracking_feed_api_gravity = calculate_api_gravity(hydrocracking_feed_specific_gravity)
    print(f'API of hydrocracking feed: {hydrocracking_feed_api_gravity}')

    hydro_cracking_outputs['Volume %']['C5 to 180°F'] = interpolate_naphtha_vol(hydrocracking_feed_api_gravity, h2_input_cracking, hydro_cracking_feed_Kw)
    hydro_cracking_outputs['BPCD']['C5 to 180°F'] = (hydro_cracking_outputs['Volume %']['C5 to 180°F'] / 100) * total_hydro_cracking_inputs_bpcd

    hydro_cracking_outputs['Volume %']['180-400°F'] = interpolate_C5_180(hydro_cracking_outputs['Volume %']['C5 to 180°F'], hydro_cracking_feed_Kw)
    hydro_cracking_outputs['BPCD']['180-400°F'] = (hydro_cracking_outputs['Volume %']['180-400°F'] / 100) * total_hydro_cracking_inputs_bpcd

    def calculate_vol_perc_butanes_wt_perc_propane_hydrocracking(LV_perc_C5_180):
        LV_perc_iC4 = 0.377 * LV_perc_C5_180
        LV_perc_nC4 = 0.186 * LV_perc_C5_180
        wt_perc_C3_and_lighter = 1.0 + 0.09 * LV_perc_C5_180
        return LV_perc_iC4, LV_perc_nC4, wt_perc_C3_and_lighter

    hydro_cracking_outputs['Volume %']['iC4'], hydro_cracking_outputs['Volume %']['nC4'], hydro_cracking_outputs['wt %']['C3 and lighter'] = calculate_vol_perc_butanes_wt_perc_propane_hydrocracking(hydro_cracking_outputs['Volume %']['C5 to 180°F'])

    hydro_cracking_outputs['BPCD']['iC4'] = (hydro_cracking_outputs['Volume %']['iC4'] / 100) * total_hydro_cracking_inputs_bpcd
    hydro_cracking_outputs['BPCD']['nC4'] = (hydro_cracking_outputs['Volume %']['nC4'] / 100) * total_hydro_cracking_inputs_bpcd

    avg_mid_boiling_point_temp_180_400_F = 281
    avg_mid_boiling_point_temp_C5_180_F = 131
    avg_mid_boiling_point_temp_180_400_Rankine = 281 + 491.67
    avg_mid_boiling_point_temp_C5_180_Rankine = 131 + 491.67

    hydro_cracking_outputs['Characterization Factor']['180-400°F'] = interpolate_Kw_product(avg_mid_boiling_point_temp_180_400_F, hydro_cracking_feed_Kw)
    hydro_cracking_outputs['Characterization Factor']['C5 to 180°F'] = interpolate_Kw_product(avg_mid_boiling_point_temp_C5_180_F, hydro_cracking_feed_Kw)

    hydro_cracking_outputs['Specific Gravity']['180-400°F'] = calculate_specific_gravity_given_Kw(avg_mid_boiling_point_temp_180_400_Rankine, hydro_cracking_outputs['Characterization Factor']['180-400°F'])
    hydro_cracking_outputs['Specific Gravity']['C5 to 180°F'] = calculate_specific_gravity_given_Kw(avg_mid_boiling_point_temp_C5_180_Rankine, hydro_cracking_outputs['Characterization Factor']['C5 to 180°F'])

    hydro_cracking_outputs['lb/hr from bbl/day']['180-400°F'] = find_lbhr_conversion(hydro_cracking_outputs['Specific Gravity']['180-400°F'], density_conv_table)
    hydro_cracking_outputs['lb/hr from bbl/day']['C5 to 180°F'] = find_lbhr_conversion(hydro_cracking_outputs['Specific Gravity']['C5 to 180°F'], density_conv_table)

    hydro_cracking_outputs['lb/hr from bbl/day']['iC4'] = 8.22
    hydro_cracking_outputs['lb/hr from bbl/day']['nC4'] = 8.51

    hydro_cracking_outputs['lb/hr']['180-400°F'] = hydro_cracking_outputs['lb/hr from bbl/day']['180-400°F'] * hydro_cracking_outputs['BPCD']['180-400°F']
    hydro_cracking_outputs['lb/hr']['C5 to 180°F'] = hydro_cracking_outputs['lb/hr from bbl/day']['C5 to 180°F'] * hydro_cracking_outputs['BPCD']['C5 to 180°F']
    hydro_cracking_outputs['lb/hr']['iC4'] = hydro_cracking_outputs['lb/hr from bbl/day']['iC4'] * hydro_cracking_outputs['BPCD']['iC4']
    hydro_cracking_outputs['lb/hr']['nC4'] = hydro_cracking_outputs['lb/hr from bbl/day']['nC4'] * hydro_cracking_outputs['BPCD']['nC4']

    hydro_cracking_outputs['lb/hr S']['H2S'] = sum(hydro_cracking_inputs['lb/hr S'].values())

    MW_S = 32.06 # g/mol
    MW_H2S = 34.08 # g/mol
    MW_H2 = 2.02 # g/mol
    hydro_cracking_outputs['lb/hr']['H2S'] = hydro_cracking_outputs['lb/hr S']['H2S'] * MW_H2S / MW_S
    hydro_cracking_outputs['lb/hr H2']['H2S'] = hydro_cracking_outputs['lb/hr']['H2S'] * MW_H2 / MW_H2S

    hydro_cracking_inputs['lb/hr']['Hydrogen'] = total_hydro_cracking_inputs_bpcd * h2_input_cracking * 0.00521 / 24 + hydro_cracking_outputs['lb/hr H2']['H2S']
    hydro_cracking_inputs['lb/hr H2']['Hydrogen'] = hydro_cracking_inputs['lb/hr']['Hydrogen']

    total_hydro_cracking_inputs_lb_hr = sum(hydro_cracking_inputs['lb/hr'].values())

    hydro_cracking_outputs['lb/hr']['C3 and lighter'] = (hydro_cracking_outputs['wt %']['C3 and lighter'] /100) * total_hydro_cracking_inputs_lb_hr
    total_hydro_cracking_outputs_lb_hr = sum(hydro_cracking_outputs['lb/hr'].values())

    avg_mid_boiling_point_temp_400_520_F = 460
    avg_mid_boiling_point_temp_400_520_Rankine = 460 + 491.67

    hydro_cracking_outputs['lb/hr']['400-520°F'] = total_hydro_cracking_inputs_lb_hr - total_hydro_cracking_outputs_lb_hr
    hydro_cracking_outputs['Characterization Factor']['400-520°F'] = interpolate_Kw_product(avg_mid_boiling_point_temp_400_520_F, hydro_cracking_feed_Kw)
    hydro_cracking_outputs['Specific Gravity']['400-520°F'] = calculate_specific_gravity_given_Kw(avg_mid_boiling_point_temp_400_520_Rankine, hydro_cracking_outputs['Characterization Factor']['400-520°F'])
    hydro_cracking_outputs['lb/hr from bbl/day']['400-520°F'] = find_lbhr_conversion(hydro_cracking_outputs['Specific Gravity']['400-520°F'], density_conv_table)
    hydro_cracking_outputs['BPCD']['400-520°F'] = hydro_cracking_outputs['lb/hr']['400-520°F'] / hydro_cracking_outputs['lb/hr from bbl/day']['400-520°F']
    total_bpcd_hydro_cracking_output = sum(
    value for value in hydro_cracking_outputs['BPCD'].values()
    if isinstance(value, (int, float))
    )
    hydro_cracking_outputs['Volume %']['400-520°F'] = 100 * hydro_cracking_outputs['BPCD']['400-520°F'] / total_bpcd_hydro_cracking_output

    # Calculate H2 wt% in outputs (PRELIM data)
    hydro_cracking_outputs['wt% H2']['C3 and lighter'] = 22.2
    hydro_cracking_outputs['wt% H2']['iC4'] = 17.2/2
    hydro_cracking_outputs['wt% H2']['nC4'] = 17.2/2

    hydro_cracking_outputs['wt% H2']['C5 to 180°F'] = interpolate_H2_content(hydro_cracking_outputs['Characterization Factor']['C5 to 180°F'], avg_mid_boiling_point_temp_C5_180_F) # 16.15     
    hydro_cracking_outputs['wt% H2']['180-400°F'] = 13.93
    hydro_cracking_outputs['wt% H2']['400-520°F'] = interpolate_H2_content(hydro_cracking_outputs['Characterization Factor']['400-520°F'], avg_mid_boiling_point_temp_400_520_F) # 13.66

    for cut in hydro_cracking_outputs['wt% H2']:
        wt_h2 = hydro_cracking_outputs['wt% H2'][cut]
        mass_flow = hydro_cracking_outputs['lb/hr'].get(cut, 0)
        hydro_cracking_outputs['lb/hr H2'][cut] = wt_h2 * mass_flow / 100

    df_hydrocracking_inputs = pd.DataFrame(hydro_cracking_inputs).reset_index().rename(columns={'index': 'Cut'})
    df_hydrocracking_outputs = pd.DataFrame(hydro_cracking_outputs).reset_index().rename(columns={'index': 'Cut'})

    desired_order = ['H2S', 'C3 and lighter', 'iC4', 'nC4', 'C5 to 180°F', '180-400°F', '400-520°F']

    # Sort the dataframe by the 'Cut' column using categorical ordering
    df_hydrocracking_outputs['Cut'] = pd.Categorical(df_hydrocracking_outputs['Cut'], categories=desired_order, ordered=True)
    df_hydrocracking_outputs = df_hydrocracking_outputs.sort_values('Cut').reset_index(drop=True)

    inputs_label = pd.DataFrame([['Inputs'] + [''] * (df_hydrocracking_inputs.shape[1] - 1)],
                            columns=df_hydrocracking_inputs.columns)
    outputs_label = pd.DataFrame([['Outputs'] + [''] * (df_hydrocracking_inputs.shape[1] - 1)],
                             columns=df_hydrocracking_inputs.columns)

    total_hydrocracking_inputs_bpcd = df_hydrocracking_inputs['BPCD'].apply(pd.to_numeric, errors='coerce').sum()
    total_hydrocracking_outputs_bpcd = df_hydrocracking_outputs['BPCD'].apply(pd.to_numeric, errors='coerce').sum()

    total_hydrocracking_inputs_lbhr = df_hydrocracking_inputs['lb/hr'].replace(',', '', regex=True).apply(pd.to_numeric, errors='coerce').sum()
    total_hydrocracking_outputs_lbhr = df_hydrocracking_outputs['lb/hr'].replace(',', '', regex=True).apply(pd.to_numeric, errors='coerce').sum()

    total_hydrocracking_inputs_sulfur = df_hydrocracking_inputs['lb/hr S'].apply(pd.to_numeric, errors='coerce').sum()
    total_hydrocracking_outputs_sulfur = df_hydrocracking_outputs['lb/hr S'].apply(pd.to_numeric, errors='coerce').sum()

    total_hydrocracking_inputs_nitrogen = df_hydrocracking_inputs['lb/hr N'].apply(pd.to_numeric, errors='coerce').sum()
    total_hydrocracking_inputs_wppm_nitrogen = total_hydrocracking_inputs_nitrogen/total_hydrocracking_inputs_lbhr * 1000000

    total_inputs_row = {
    'Cut': 'Total',
    'BPCD': total_hydrocracking_inputs_bpcd,
    'lb/hr': total_hydrocracking_inputs_lbhr,
    'lb/hr S': total_hydrocracking_inputs_sulfur,
    'lb/hr N': total_hydrocracking_inputs_nitrogen,
    'wppm N': total_hydrocracking_inputs_wppm_nitrogen,

    }
    total_outputs_row = {
    'Cut': 'Total',
    'BPCD': total_hydrocracking_outputs_bpcd,
    'lb/hr': total_hydrocracking_outputs_lbhr,
    'lb/hr S': total_hydrocracking_outputs_sulfur
    }

    df_total_inputs_row = pd.DataFrame([total_inputs_row], columns=df_hydrocracking_inputs.columns)
    df_total_outputs_row = pd.DataFrame([total_outputs_row], columns=df_hydrocracking_inputs.columns)

    df_hydrocracking = pd.concat([
    inputs_label,
    df_hydrocracking_inputs,
    df_total_inputs_row,
    outputs_label,
    df_hydrocracking_outputs,
    df_total_outputs_row
    ], ignore_index=True)

    df_hydrocracking.fillna('', inplace=True)

    print(df_hydrocracking)

    # Hydrogen balance for hydrocracking
    hydrogen_balance = {}
    hydrogen_balance['lb/hr'] = {}
    hydrogen_balance['scf/bbl'] = {}
    hydrogen_balance['lb/hr']['Sulfur removal'] = hydro_cracking_outputs['lb/hr H2']['H2S']
    hydrogen_balance['scf/bbl']['Sulfur removal'] = hydrogen_balance['lb/hr']['Sulfur removal'] * 24 / conversion_parameters['H2_scf_to_lb'] / total_hydro_cracking_inputs_bpcd

    hydrogen_balance['lb/hr']['Saturation after desulfurization'] = hydro_cracking_outputs['lb/hr H2']['H2S']
    hydrogen_balance['scf/bbl']['Saturation after desulfurization'] = hydrogen_balance['lb/hr']['Saturation after desulfurization'] * 24 / conversion_parameters['H2_scf_to_lb'] / total_hydro_cracking_inputs_bpcd

    scf_H2_per_perc_N_reduction = 320   # scf/bbl/wt% change (57 Nm3/m3/wt% change)
    change_in_N_perc = total_hydrocracking_inputs_wppm_nitrogen /10000
    hydrogen_balance['scf/bbl']['Denitrogenation'] = scf_H2_per_perc_N_reduction * change_in_N_perc
    hydrogen_balance['lb/hr']['Denitrogenation'] = hydrogen_balance['scf/bbl']['Denitrogenation'] * total_hydrocracking_inputs_bpcd / 24 * conversion_parameters['H2_scf_to_lb']

    hydrogen_balance['lb/hr']['H2 in liquid products'] = hydro_cracking_outputs['lb/hr H2']['iC4'] + hydro_cracking_outputs['lb/hr H2']['nC4'] + hydro_cracking_outputs['lb/hr H2']['C5 to 180°F'] + hydro_cracking_outputs['lb/hr H2']['180-400°F'] + hydro_cracking_outputs['lb/hr H2']['400-520°F']
    hydrogen_balance['scf/bbl']['H2 in liquid products'] = hydrogen_balance['lb/hr']['H2 in liquid products'] * 24 / conversion_parameters['H2_scf_to_lb'] / total_hydro_cracking_inputs_bpcd

    hydrogen_balance['lb/hr']['H2 in HC gas'] = hydro_cracking_outputs['lb/hr H2']['C3 and lighter']
    hydrogen_balance['scf/bbl']['H2 in HC gas'] = hydrogen_balance['lb/hr']['H2 in HC gas'] * 24 / conversion_parameters['H2_scf_to_lb'] / total_hydro_cracking_inputs_bpcd

    #hydrogen_balance['lb/hr']['hydrogen in hydrocracker feed'] = hydro_cracking_inputs['lb/hr H2']['Light gas oil (480-610°F)'] + hydro_cracking_inputs['lb/hr H2']['Heavy gas oil (610-800°F)'] + hydro_cracking_inputs['lb/hr H2']['Vacuum gas oil (800-1050°F)']
    hydrogen_balance['lb/hr']['H2 in hydrocracker feed'] =  hydro_cracking_inputs['lb/hr H2']['Heavy gas oil (610-800°F)'] + hydro_cracking_inputs['lb/hr H2']['Vacuum gas oil (800-1050°F)']
    hydrogen_balance['scf/bbl']['H2 in hydrocracker feed'] = hydrogen_balance['lb/hr']['H2 in hydrocracker feed'] * 24 / conversion_parameters['H2_scf_to_lb'] / total_hydro_cracking_inputs_bpcd

    hydrogen_balance['lb/hr']['H2 in solution losses'] = total_hydro_cracking_inputs_bpcd /24 # 1 lb/bbl
    hydrogen_balance['scf/bbl']['H2 in solution losses'] = hydrogen_balance['lb/hr']['H2 in solution losses'] * 24 / conversion_parameters['H2_scf_to_lb'] / total_hydro_cracking_inputs_bpcd

    hydrogen_balance['lb/hr']['H2 consumption'] = hydrogen_balance['lb/hr']['Sulfur removal'] + hydrogen_balance['lb/hr']['Saturation after desulfurization'] + hydrogen_balance['lb/hr']['Denitrogenation'] + hydrogen_balance['lb/hr']['H2 in liquid products'] + hydrogen_balance['lb/hr']['H2 in HC gas'] - hydrogen_balance['lb/hr']['H2 in hydrocracker feed'] + hydrogen_balance['lb/hr']['H2 in solution losses'] 

    hydrogen_balance['scf/bbl']['H2 consumption'] = hydrogen_balance['lb/hr']['H2 consumption'] * 24 / conversion_parameters['H2_scf_to_lb'] / total_hydro_cracking_inputs_bpcd

    hydrogen_balance_df = pd.DataFrame.from_dict(hydrogen_balance, orient='index').T
    hydrogen_balance_df.index.name = 'Hydrogen Balance Item'
    hydrogen_balance_df = hydrogen_balance_df.reset_index()

    hydrogen_balance_df = hydrogen_balance_df[['Hydrogen Balance Item', 'lb/hr', 'scf/bbl']]

    print(hydrogen_balance_df)

    hydrocracking_feed_H2_content = 100 * (df_hydrocracking_inputs['lb/hr H2'].apply(pd.to_numeric, errors='coerce').sum() - hydro_cracking_inputs['lb/hr H2']['Hydrogen'])/(total_hydrocracking_inputs_lbhr - hydro_cracking_inputs['lb/hr']['Hydrogen'])
    hydrocracking_feed_aromatics_content = estimate_aromatics_content(hydrocracking_feed_H2_content)

    hydrocracking_products_H2_content = 100 * (df_hydrocracking_outputs['lb/hr H2'].apply(pd.to_numeric, errors='coerce').sum() - hydro_cracking_outputs['lb/hr H2']['H2S'])/(total_hydrocracking_outputs_lbhr - hydro_cracking_outputs['lb/hr']['H2S'])
    hydrocracking_products_aromatics_content = estimate_aromatics_content(hydrocracking_products_H2_content)

    aromatics_content_reduction = 100 * (hydrocracking_feed_aromatics_content - hydrocracking_products_aromatics_content)/hydrocracking_feed_aromatics_content

    print(f'Aromatics content of hydrocracking feed  = {hydrocracking_feed_aromatics_content:.2f} %')
    print(f'Aromatics content hydrocracking products = {hydrocracking_products_aromatics_content:.2f} %')
    print(f'Aromatics content reduction = {aromatics_content_reduction:.2f} %')

    # Determine utilities for hydrocracking
    daily_hydrocracking_utilities = {
    key: (
        utility_data["amount"] * total_hydrocracking_inputs_bpcd,
        utility_data["unit"]
    )
    for key, utility_data in hydrocracking_utility_data.items()
    }

# HYDROTREATMENT -------------------------------------------------------------------------------------------------------------------
if aromatics_removal_technique == 'hydrotreatment':

    # Initialize hydrocracking inputs and outputs
    hydrotreatment_inputs = {
    'Volume %': {},
    'API': {},
    'Specific Gravity': {},
    'Characterization Factor': {},
    'BPCD': {},
    'lb/hr from bbl/day': {},
    'lb/hr': {},
    'wt% S': {},
    'lb/hr S': {},
    'wt% H2': {},
    'lb/hr H2' : {},
    'lb/hr N': {},
    'wppm N': {},
    'molecular weight': {},
    'Refractory index at 20C': {},
    'Refractory intercept': {},
    'vol% paraffins': {},
    'BPCD paraffins': {},
    'lb/hr paraffins': {},
    'wt% paraffins': {},
    'vol% napthenes': {},
    'BPCD napthenes': {},
    'lb/hr napthenes': {},
    'wt% napthenes': {},
    'vol% aromatics': {},
    'vol% monoaromatics': {},
    'BPCD monoaromatics': {},
    'lb/hr monoaromatics': {},
    'wt% monoaromatics': {},
    'vol% polyaromatics': {},
    'BPCD polyaromatics': {},
    'lb/hr polyaromatics': {},
    'wt% polyaromatics': {},
    }

    hydrotreatment_outputs = {
    'Volume %': {},
    'API': {},
    'Specific Gravity': {},
    'Characterization Factor': {},
    'BPCD': {},
    'lb/hr from bbl/day': {},
    'lb/hr': {},
    'wt% S': {},
    'lb/hr S': {},
    'wt% H2': {},
    'lb/hr H2' : {},
    'lb/hr N': {},
    'wppm N': {},
    'molecular weight': {},
    'Refractory index at 20C': {},
    'Refractory intercept': {},
    'vol% paraffins': {},
    'BPCD paraffins': {},
    'lb/hr paraffins': {},
    'wt% paraffins': {},
    'vol% napthenes': {},
    'BPCD napthenes': {},
    'lb/hr napthenes': {},
    'wt% napthenes': {},
    'vol% aromatics': {},
    'vol% monoaromatics': {},
    'BPCD monoaromatics': {},
    'lb/hr monoaromatics': {},
    'wt% monoaromatics': {},
    'vol% polyaromatics': {},
    'BPCD polyaromatics': {},
    'lb/hr polyaromatics': {},
    'wt% polyaromatics': {},
    }

    # Feed for hydrotreatment is kerosine
    hydrotreatment_cuts = {
        'Kerosine (330-480°F)': atmospheric_distillation_outputs
    }

    properties = ['BPCD', 'API', 'Specific Gravity', 'Characterization Factor', 'lb/hr from bbl/day', 'lb/hr', 'wt% S', 'lb/hr S', 'wt% H2', 'lb/hr H2', 'lb/hr N', 'wppm N', 'molecular weight', 'Refractory index at 20C', 'Refractory intercept', 'vol% paraffins', 'BPCD paraffins', 'lb/hr paraffins', 'wt% paraffins', 'vol% napthenes', 'BPCD napthenes', 'lb/hr napthenes', 'wt% napthenes', 'vol% aromatics', 'vol% monoaromatics', 'BPCD monoaromatics', 'lb/hr monoaromatics', 'wt% monoaromatics', 'vol% polyaromatics', 'BPCD polyaromatics', 'lb/hr polyaromatics', 'wt% polyaromatics']

    for prop in properties:
        for cut, source_df in hydrotreatment_cuts.items():
            hydrotreatment_inputs[prop][cut] = source_df[prop][cut]

    total_hydrotreatment_inputs_bpcd = sum(hydrotreatment_inputs['BPCD'].values())
    total_hydrotreatment_inputs_lb_hr_sulfur = pd.to_numeric(pd.Series(hydrotreatment_inputs['lb/hr S'].values()), errors='coerce').fillna(0).sum()
    total_hydrotreatment_inputs_wppm_nitrogen = sum(hydrotreatment_inputs['wppm N'].values())

    # Determine liquid hydrocarbon product yield from hydrotreatment
    liquid_product_yield = interpolate_liquid_yield(hydrotreatment_operating_parameters['reactor_temp'], hydrotreatment_operating_parameters['LHSV'], hydrotreatment_operating_parameters['reactor_pressure'])
    hydrotreatment_outputs['BPCD']['HT Kerosine (330-480°F)'] = liquid_product_yield / 100 * hydrotreatment_inputs['BPCD']['Kerosine (330-480°F)']
    
    mean_avg_boiling_pt = (330 + 480) / 2
    mean_avg_boiling_pt_Rankine = mean_avg_boiling_pt + 459.67

    # Convert BPCD hydrocarbon product to lb/hr
    # Assume API increase of 1
    hydrotreatment_outputs['API']['HT Kerosine (330-480°F)'] = hydrotreatment_inputs['API']['Kerosine (330-480°F)'] + 1
    hydrotreatment_outputs['Specific Gravity']['HT Kerosine (330-480°F)'] = calculate_specific_gravity(hydrotreatment_outputs['API']['HT Kerosine (330-480°F)'])
    hydrotreatment_outputs['Characterization Factor']['HT Kerosine (330-480°F)'] = calculate_characterization_factor(mean_avg_boiling_pt_Rankine, hydrotreatment_outputs['Specific Gravity']['HT Kerosine (330-480°F)'])
    hydrotreatment_outputs['lb/hr from bbl/day']['HT Kerosine (330-480°F)'] = find_lbhr_conversion(hydrotreatment_outputs['Specific Gravity']['HT Kerosine (330-480°F)'], density_conv_table)
    hydrotreatment_outputs['lb/hr']['HT Kerosine (330-480°F)'] = hydrotreatment_outputs['BPCD']['HT Kerosine (330-480°F)'] * hydrotreatment_outputs['lb/hr from bbl/day']['HT Kerosine (330-480°F)']

    # Determine aromatic saturation efficiency of hydrotreatment
    saturation_efficiency = interpolate_aromatics_saturation_efficiency(hydrotreatment_operating_parameters['reactor_temp'], hydrotreatment_operating_parameters['reactor_pressure'])
    hydrotreatment_outputs['wt% monoaromatics']['HT Kerosine (330-480°F)'] =  hydrotreatment_inputs['wt% monoaromatics']['Kerosine (330-480°F)'] * (1 - saturation_efficiency/100)
    
    hydrotreatment_outputs['wt% polyaromatics']['HT Kerosine (330-480°F)'] = hydrotreatment_inputs['wt% polyaromatics']['Kerosine (330-480°F)'] * (1 - saturation_efficiency/100)
    
    # WRITE INTERPOLATE FOR sulfur_content = 
    hydrotreatment_outputs['wt% S']['HT Kerosine (330-480°F)'] = 0
    hydrotreatment_outputs['lb/hr S']['HT Kerosine (330-480°F)'] = hydrotreatment_outputs['wt% S']['HT Kerosine (330-480°F)'] / 100 * hydrotreatment_outputs['lb/hr']['HT Kerosine (330-480°F)']

    hydrotreatment_outputs['wppm N']['HT Kerosine (330-480°F)'] = 0

    # Calculate output of H2S to remove sulfur
    hydrotreatment_outputs['lb/hr S']['H2S'] = total_hydrotreatment_inputs_lb_hr_sulfur - hydrotreatment_outputs['lb/hr S']['HT Kerosine (330-480°F)']
    hydrotreatment_outputs['lb/hr']['H2S'] = hydrotreatment_outputs['lb/hr S']['H2S'] * fixed_parameters['molecular_weight']['H2S'] / fixed_parameters['molecular_weight']['S']
    hydrotreatment_outputs['lb/hr H2']['H2S'] = hydrotreatment_outputs['lb/hr']['H2S'] * fixed_parameters['molecular_weight']['H2'] / fixed_parameters['molecular_weight']['H2S']

    hydrogen_partial_pressure = hydrotreatment_operating_parameters['reactor_pressure']/10        # Assume hydrogen partial pressure is reactor pressure convert bar to MPa (divide by 10)
    
    if pd.isna(hydrotreatment_inputs['wppm N']['Kerosine (330-480°F)']):
        hydrotreatment_inputs['wppm N']['Kerosine (330-480°F)'] = 0

    # Calculate chemical hydrogen consumption and dissolved hydrogen
    H_HDS, H_HDN, H_HDA, H_chem_total_Nm3_per_m3, H_chem_lb_per_hr, H_chem_scf_per_bbl, H_diss_lb_per_hr, H_diss_scf_per_bbl = compute_hydrogen_consumption(
    hydrotreatment_inputs['wt% S']['Kerosine (330-480°F)'], hydrotreatment_outputs['wt% S']['HT Kerosine (330-480°F)'],
    hydrotreatment_inputs['wppm N']['Kerosine (330-480°F)'], hydrotreatment_outputs['wppm N']['HT Kerosine (330-480°F)'],
    hydrotreatment_inputs['wt% polyaromatics']['Kerosine (330-480°F)'], hydrotreatment_outputs['wt% polyaromatics']['HT Kerosine (330-480°F)'],
    hydrotreatment_inputs['wt% monoaromatics']['Kerosine (330-480°F)'], hydrotreatment_outputs['wt% monoaromatics']['HT Kerosine (330-480°F)'],
    hydrotreatment_inputs['Specific Gravity']['Kerosine (330-480°F)'], hydrotreatment_outputs['Specific Gravity']['HT Kerosine (330-480°F)'],
    liquid_product_yield, total_hydrotreatment_inputs_bpcd,
    hydrotreatment_operating_parameters['H_purity'], hydrogen_partial_pressure, hydrotreatment_operating_parameters['gas_to_oil'])   
    print(f"HDS: {H_HDS:.3f} Nm³ H₂/m³ oil")
    print(f"HDN: {H_HDN:.3f} Nm³ H₂/m³ oil")
    print(f"HDA: {H_HDA:.3f} Nm³ H₂/m³ oil")
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
        print(f'Hydrogen consumtpion exceeds 800 scf/bbl --> indicative of hydrocracking (adjust operating conditions!)')

    hydrotreatment_operaring_hydrogen_input_Nm3_per_hr = hydrotreatment_operating_parameters['gas_to_oil'] * hydrotreatment_inputs['BPCD']['Kerosine (330-480°F)'] * conversion_parameters['BPCD_to_m3_per_hr']  

    hydrotreatment_operaring_hydrogen_input_lb_per_hr = hydrotreatment_operaring_hydrogen_input_Nm3_per_hr * conversion_parameters['H2_Nm3_to_lb']
    hydrotreatment_operaring_hydrogen_input_scf_per_bbl = hydrotreatment_operaring_hydrogen_input_lb_per_hr * conversion_parameters['H2_lb_to_scf'] / (hydrotreatment_inputs['BPCD']['Kerosine (330-480°F)'] / 24)

    print(f"Operating H₂ input (makeup + recycle): {hydrotreatment_operaring_hydrogen_input_lb_per_hr:.3f} lb/hr")
    print(f"Operating H₂ input (makeup + recycle): {hydrotreatment_operaring_hydrogen_input_scf_per_bbl:.3f} scf/bbl")

    hydrotreatment_inputs['lb/hr']['Hydrogen'] = H_chem_lb_per_hr + H_diss_lb_per_hr
    hydrotreatment_inputs['lb/hr H2']['Hydrogen'] = hydrotreatment_inputs['lb/hr']['Hydrogen']

    # Determine HC gas output from mass balance
    hydrotreatment_outputs['lb/hr']['HC gas'] = hydrotreatment_inputs['lb/hr']['Kerosine (330-480°F)'] + hydrotreatment_inputs['lb/hr']['Hydrogen'] - hydrotreatment_outputs['lb/hr']['HT Kerosine (330-480°F)'] - hydrotreatment_outputs['lb/hr']['H2S']

    # Molecular breakdown of HC gas
    hydrotreatment_outputs_wt_perc_CH4 = 0.0588 
    hydrotreatment_outputs_wt_perc_C2H6 = 0.0588
    hydrotreatment_outputs_wt_perc_C3H8 = 0.2124
    hydrotreatment_outputs_wt_perc_iC4 = 0.4425 
    hydrotreatment_outputs_wt_perc_nC4 = 0.2274

    # Weight percent of H2 in HC gas components
    wt_perc_H2_CH4 = 4 * fixed_parameters['molecular_weight']['H'] / fixed_parameters['molecular_weight']['CH4']
    wt_perc_H2_C2H6 = 6 * fixed_parameters['molecular_weight']['H'] / fixed_parameters['molecular_weight']['C2H6']
    wt_perc_H2_C3H8 = 8 * fixed_parameters['molecular_weight']['H'] / fixed_parameters['molecular_weight']['C3H8']
    wt_perc_H2_iC4 = 17.38/100
    wt_perc_H2_nC4 = 17.38/100

    hydrotreatment_outputs['lb/hr']['C3 and lighter'] = (hydrotreatment_outputs_wt_perc_CH4 + hydrotreatment_outputs_wt_perc_C2H6 + hydrotreatment_outputs_wt_perc_C3H8) * hydrotreatment_outputs['lb/hr']['HC gas']
    hydrotreatment_outputs['lb/hr']['iC4'] = hydrotreatment_outputs_wt_perc_iC4 * hydrotreatment_outputs['lb/hr']['HC gas']
    hydrotreatment_outputs['lb/hr']['nC4'] = hydrotreatment_outputs_wt_perc_nC4 * hydrotreatment_outputs['lb/hr']['HC gas']

    hydrotreatment_outputs['wt% H2']['C3 and lighter'] = 100* (wt_perc_H2_CH4 * hydrotreatment_outputs_wt_perc_CH4 + wt_perc_H2_C2H6 * hydrotreatment_outputs_wt_perc_C2H6  + wt_perc_H2_C3H8 * hydrotreatment_outputs_wt_perc_C3H8) / (hydrotreatment_outputs_wt_perc_CH4 + hydrotreatment_outputs_wt_perc_C2H6 + hydrotreatment_outputs_wt_perc_C3H8)
    hydrotreatment_outputs['lb/hr H2']['C3 and lighter'] = hydrotreatment_outputs['wt% H2']['C3 and lighter']/100 * hydrotreatment_outputs['lb/hr']['C3 and lighter']
    hydrotreatment_outputs['wt% H2']['iC4'] = wt_perc_H2_iC4 *100
    hydrotreatment_outputs['lb/hr H2']['iC4'] = hydrotreatment_outputs['wt% H2']['iC4'] * hydrotreatment_outputs['lb/hr']['iC4'] /100
    hydrotreatment_outputs['wt% H2']['nC4'] = wt_perc_H2_nC4 * 100
    hydrotreatment_outputs['lb/hr H2']['nC4'] = hydrotreatment_outputs['wt% H2']['nC4'] * hydrotreatment_outputs['lb/hr']['nC4'] /100

    hydrotreatment_outputs['wt% H2']['HC gas'] = (hydrotreatment_outputs_wt_perc_CH4 * wt_perc_H2_CH4 + hydrotreatment_outputs_wt_perc_C2H6 * wt_perc_H2_C2H6 + hydrotreatment_outputs_wt_perc_C3H8 * wt_perc_H2_C3H8 + hydrotreatment_outputs_wt_perc_iC4 * wt_perc_H2_iC4 + hydrotreatment_outputs_wt_perc_nC4 * wt_perc_H2_nC4) * 100
    
    #hydrotreatment_outputs['lb/hr H2']['HC gas'] = hydrotreatment_inputs['lb/hr H2']['Kerosine (330-480°F)'] + hydrotreatment_inputs['lb/hr H2']['Hydrogen'] - hydrotreatment_outputs['lb/hr H2']['HT Kerosine (330-480°F)'] - hydrotreatment_outputs['lb/hr H2']['H2S']
    #hydrotreatment_outputs['wt% H2']['HC gas'] = hydrotreatment_outputs['lb/hr H2']['HC gas'] / hydrotreatment_outputs['lb/hr']['HC gas']
    hydrotreatment_outputs['lb/hr H2']['HC gas'] = hydrotreatment_outputs['lb/hr']['HC gas'] * hydrotreatment_outputs['wt% H2']['HC gas'] / 100

    hydrogen_added_to_liquid_molecules = H_chem_lb_per_hr - hydrotreatment_outputs['lb/hr H2']['H2S'] - hydrotreatment_outputs['lb/hr H2']['HC gas']

    hydrotreatment_outputs['lb/hr H2']['HT Kerosine (330-480°F)'] = hydrogen_added_to_liquid_molecules + H_diss_lb_per_hr + hydrotreatment_inputs['lb/hr H2']['Kerosine (330-480°F)']
    hydrotreatment_outputs['wt% H2']['HT Kerosine (330-480°F)'] = hydrotreatment_outputs['lb/hr H2']['HT Kerosine (330-480°F)'] / hydrotreatment_outputs['lb/hr']['HT Kerosine (330-480°F)'] * 100

    hydrotreatment_outputs['lb/hr from bbl/day']['C3 and lighter'] = 7.42
    hydrotreatment_outputs['lb/hr from bbl/day']['iC4'] = 8.22
    hydrotreatment_outputs['lb/hr from bbl/day']['nC4'] = 8.51

    hydrotreatment_outputs['BPCD']['C3 and lighter'] = hydrotreatment_outputs['lb/hr']['C3 and lighter']/hydrotreatment_outputs['lb/hr from bbl/day']['C3 and lighter'] 
    hydrotreatment_outputs['BPCD']['iC4'] = hydrotreatment_outputs['lb/hr']['iC4']/hydrotreatment_outputs['lb/hr from bbl/day']['iC4'] 
    hydrotreatment_outputs['BPCD']['nC4'] = hydrotreatment_outputs['lb/hr']['nC4']/hydrotreatment_outputs['lb/hr from bbl/day']['nC4'] 

    df_hydrotreatment_inputs = pd.DataFrame(hydrotreatment_inputs).reset_index().rename(columns={'index': 'Cut'})
    df_hydrotreatment_outputs = pd.DataFrame(hydrotreatment_outputs).reset_index().rename(columns={'index': 'Cut'})
    df_hydrotreatment_outputs = df_hydrotreatment_outputs[df_hydrotreatment_outputs['Cut'] != 'total gas flow']
    df_hydrotreatment_outputs = df_hydrotreatment_outputs[df_hydrotreatment_outputs['Cut'] != 'HC gas']

    desired_order = ['H2S', 'C3 and lighter', 'iC4', 'nC4', 'HT Kerosine (330-480°F)']

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

    total_hydrotreatment_inputs_lbhr_H2 = df_hydrotreatment_inputs['lb/hr H2'].replace(',', '', regex=True).apply(pd.to_numeric, errors='coerce').sum()
    total_hydrotreatment_outputs_lbhr_H2 = df_hydrotreatment_outputs['lb/hr H2'].replace(',', '', regex=True).apply(pd.to_numeric, errors='coerce').sum()

    total_hydrotreatment_inputs_sulfur = df_hydrotreatment_inputs['lb/hr S'].apply(pd.to_numeric, errors='coerce').sum()
    total_hydrotreatment_outputs_sulfur = df_hydrotreatment_outputs['lb/hr S'].apply(pd.to_numeric, errors='coerce').sum()

    total_hydrotreatment_inputs_nitrogen = df_hydrotreatment_inputs['lb/hr N'].apply(pd.to_numeric, errors='coerce').sum()
    total_hydrocracking_inputs_wppm_nitrogen = total_hydrotreatment_inputs_nitrogen/total_hydrotreatment_inputs_lbhr * 1000000

    total_inputs_row = {
    'Cut': 'Total',
    'BPCD': total_hydrotreatmentinputs_bpcd,
    'lb/hr': total_hydrotreatment_inputs_lbhr,
    'lb/hr H2': total_hydrotreatment_inputs_lbhr_H2,
    'lb/hr S': total_hydrotreatment_inputs_sulfur,
    'lb/hr N': total_hydrotreatment_inputs_nitrogen,
    'wppm N': total_hydrotreatment_inputs_wppm_nitrogen,
    }

    total_outputs_row = {
    'Cut': 'Total',
    'BPCD': total_hydrotreatment_outputs_bpcd,
    'lb/hr': total_hydrotreatment_outputs_lbhr,
    'lb/hr H2': total_hydrotreatment_outputs_lbhr_H2,
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
    daily_hydrotreatment_utilities = {
    key: (
        utility_data["amount"] * total_hydrotreatment_inputs_bpcd,
        utility_data["unit"]
    )
    for key, utility_data in hydrotreatment_utility_data.items()
    }

    # Amine gas treating unit

    # Initialize hydrocracking inputs and outputs
    amine_gas_treating_inputs = {
    'Volume %': {},
    'API': {},
    'Specific Gravity': {},
    'Characterization Factor': {},
    'BPCD': {},
    'lb/hr from bbl/day': {},
    'lb/hr': {},
    'wt% S': {},
    'lb/hr S': {},
    'wt% H2': {},
    'lb/hr H2' : {}
    }

    # Feed for hydrotreatment is kerosine
    amine_gas_input_cuts = {
        'H2S': hydrotreatment_outputs,
        'C3 and lighter': hydrotreatment_outputs,
        'iC4': hydrotreatment_outputs,
        'nC4': hydrotreatment_outputs,
    }

    properties = ['BPCD', 'API', 'Specific Gravity', 'Characterization Factor', 'lb/hr from bbl/day', 'lb/hr', 'wt% S', 'lb/hr S', 'wt% H2', 'lb/hr H2']

    amine_gas_inputs = {}

    for prop in properties:
        for cut, source_df in amine_gas_input_cuts.items():
            print(source_df.columns)  # if it's a DataFrame
            amine_gas_treating_inputs[prop][cut] = source_df[prop][cut]

    df_amine_gas_input_cuts = pd.DataFrame(amine_gas_inputs).reset_index().rename(columns={'index': 'Cut'})
    print(df_amine_gas_input_cuts)

# Combine all unique keys from all utility sources
all_keys = (
    set(desalter_utilities.keys())
    | set(daily_atmospheric_dist_utilities.keys())
    | set(daily_vacuum_dist_utilities.keys())
    | (set(daily_hydrotreatment_utilities.keys()) if aromatics_removal_technique == "hydrotreatment" else set(daily_hydrocracking_utilities.keys()))
)

def get_utility_description(key):
    if aromatics_removal_technique == "hydrotreatment":
        return (
            desalter_utilities.get(key, ('', ''))[1] or
            daily_atmospheric_dist_utilities.get(key, ('', ''))[1] or
            daily_vacuum_dist_utilities.get(key, ('', ''))[1] or
            daily_hydrotreatment_utilities.get(key, ('', ''))[1] or

            ''
        )
    else:
        return (
            desalter_utilities.get(key, ('', ''))[1] or
            daily_atmospheric_dist_utilities.get(key, ('', ''))[1] or
            daily_vacuum_dist_utilities.get(key, ('', ''))[1] or
            daily_hydrocracking_utilities.get(key, ('', ''))[1] or
            ''
        )

# Start building utility_data with common columns
utility_data = {
    'Utility': [
        f"{key} ({get_utility_description(key)})" for key in all_keys
    ],
    'Desalter': [
        desalter_utilities.get(key, (0, ''))[0] for key in all_keys
    ],
    'Atmospheric Distillation': [
        daily_atmospheric_dist_utilities.get(key, (0, ''))[0] for key in all_keys
    ],
    'Vacuum Distillation': [
        daily_vacuum_dist_utilities.get(key, (0, ''))[0] for key in all_keys
    ],
}

# Conditionally add hydrotreatment or hydrocracking
if aromatics_removal_technique == "hydrotreatment":
    utility_data['Hydrotreatment'] = [
        daily_hydrotreatment_utilities.get(key, (0, ''))[0] for key in all_keys
    ]
if aromatics_removal_technique == "hydrocracking":
    utility_data['Hydrocracking'] = [
        daily_hydrocracking_utilities.get(key, (0, ''))[0] for key in all_keys
    ]

# Add Total column dynamically
utility_data['Total'] = []
for key in all_keys:
    total = (
        desalter_utilities.get(key, (0, ''))[0] +
        daily_atmospheric_dist_utilities.get(key, (0, ''))[0] +
        daily_vacuum_dist_utilities.get(key, (0, ''))[0]
    )
    if aromatics_removal_technique == "hydrotreatment":
        total += daily_hydrotreatment_utilities.get(key, (0, ''))[0]
    if aromatics_removal_technique == "hydrocracking":
        total += daily_hydrocracking_utilities.get(key, (0, ''))[0]
    utility_data['Total'].append(total)

# Create and print the DataFrame
df_utilities = pd.DataFrame(utility_data)
print(df_utilities.to_string(index=False))

# Save to Excel
output_file = os.path.join(output_dir, f"refinery_outputs_{aromatics_removal_technique}.xlsx")

# Helper function to round a DataFrame with custom precision
def round_dataframe(df):
    df_rounded = df.copy()
    for col in df_rounded.select_dtypes(include='number').columns:
        if col == 'Specific Gravity':
            df_rounded[col] = df_rounded[col].round(4)
        else:
            df_rounded[col] = df_rounded[col].round(2)
    return df_rounded


output_file = os.path.join(output_dir, f"refinery_outputs_{aromatics_removal_technique}.xlsx")

# Write rounded DataFrames to Excel
with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
    round_dataframe(df_atmospheric_distillation).to_excel(writer, sheet_name="Atmospheric Distillation", index=False)
    round_dataframe(df_vacuum_distillation).to_excel(writer, sheet_name="Vacuum Distillation", index=False)
    
    df_utilities = pd.DataFrame(utility_data)
    round_dataframe(df_utilities).to_excel(writer, sheet_name="Utilities", index=False)
    
    if 'df_hydrotreatment' in globals():
        round_dataframe(df_hydrotreatment).to_excel(writer, sheet_name="Hydrotreatment", index=False)
        #round_dataframe(hydrogen_balance_df).to_excel(writer, sheet_name="Hydrotreatment H2 balance", index=False)

    if 'df_hydrocracking' in globals():
        round_dataframe(df_hydrocracking).to_excel(writer, sheet_name="Hydrocracking", index=False)
        
        #round_dataframe(hydrogen_balance_df).to_excel(writer, sheet_name="Hydrocracking H2 balance", index=False)


