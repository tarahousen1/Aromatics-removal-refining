

import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

labor_hours_vs_plant_capacity_file_path = "/Users/tarahousen/Documents/Aromatics_removal_desulfurization_refining/excel_files/labor_hours_vs_plant_capacity.xlsx"

labor_hours_vs_plant_capacity_df = pd.read_excel(labor_hours_vs_plant_capacity_file_path, header=2)

# Fix column headers (strip spaces)
labor_hours_vs_plant_capacity_df.columns = labor_hours_vs_plant_capacity_df.columns.str.strip()

# Extract data
plant_capacity = labor_hours_vs_plant_capacity_df['X'].values
labor_hours = labor_hours_vs_plant_capacity_df['Y'].values

# Interpolation
labor_hours_interp = interp1d(plant_capacity, labor_hours, kind='linear', fill_value='extrapolate')

