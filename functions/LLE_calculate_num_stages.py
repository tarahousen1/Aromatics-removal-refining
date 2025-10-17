import pandas as pd
from scipy.interpolate import interp1d

# Read Excel file
file_path = "/Users/tarahousen/Documents/Aromatics_removal_desulfurization_refining/excel_files/solvent_extraction_num_stages_vs_SF.xlsx"
df = pd.read_excel(file_path, header=2)
df.columns = df.columns.str.strip()  # remove extra spaces in header names

num_stages_data = df['X'].values
S_F_data = df['Y'].values

# Build inverse interpolator: given y → return x
inverse_interpolator = interp1d(S_F_data, num_stages_data, kind="linear", fill_value="extrapolate")

forward_interpolator = interp1d(num_stages_data, S_F_data, kind="linear", fill_value="extrapolate")


def interpolate_num_stages(S_F_ratio):
    """
    Interpolates solvent-to-feed ratio given number of stages.
    """
    return float(inverse_interpolator(S_F_ratio))

def interpolate_SF_ratio(num_stages):
    """
    Interpolates solvent-to-feed ratio given number of stages.
    """
    return float(forward_interpolator(num_stages))


