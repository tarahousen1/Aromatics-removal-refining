import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.interpolate import interp1d
import itertools

# Load Excel data
file_path = "/Users/tarahousen/Documents/Aromatics_removal_desulfurization_refining/excel_files/experimental data aromatics kerosene saturation.xlsx"
df = pd.read_excel(file_path, header=1)

# Drop rows with missing data
df = df.dropna(subset=["Pressure (bar)", "LHSV (1/hr)", "H2/HC (mL/mL)", "Initial aromatics content", "Temperature (°C)"])

# Filter to LHSV = 1
filtered_df = df[df["LHSV (1/hr)"] == 1]

# Unique sorted Pressure and H2/HC values
unique_pressure = sorted(filtered_df["Pressure (bar)"].unique())
unique_h2hc = sorted(filtered_df["H2/HC (mL/mL)"].unique())

# Define color palette for Pressure (use a built-in colormap)
cmap = plt.get_cmap("viridis")  # can choose another like "plasma", "tab10", etc.
pressure_color_map = {p: cmap(i / (len(unique_pressure)-1)) for i, p in enumerate(unique_pressure)}

# Define markers for H2/HC
markers = ['o', 's', 'D', '^', 'v', 'X', '*', 'P', 'h']
h2hc_marker_map = {val: markers[i % len(markers)] for i, val in enumerate(unique_h2hc)}

plt.figure(figsize=(14, 8))

for (source, pressure, h2hc), group_df in filtered_df.groupby(["Source", "Pressure (bar)", "H2/HC (mL/mL)"]):
    color = pressure_color_map[pressure]
    marker = h2hc_marker_map[h2hc]
    linestyle = '-' if source == "[1]" else ':'
    
    plt.plot(
        group_df["Temperature (°C)"],
        group_df["Aromatics saturation efficiency"],
        label=f"Src {source}, {pressure} bar, H₂/HC={h2hc}",
        linestyle=linestyle,
        marker=marker,
        color=color
    )

plt.xlabel("Temperature (°C)", fontsize=12)
plt.ylabel("Aromatics Saturation Efficiency (%)", fontsize=12)
plt.title("Aromatics Saturation vs Temperature (LHSV = 1)", fontsize=14)
plt.grid(True)
plt.tight_layout()

# Legend for Pressure colors
color_legend = [Line2D([0], [0], color=pressure_color_map[p], lw=4) for p in unique_pressure]
color_labels = [f"Pressure = {p} bar" for p in unique_pressure]

# Legend for H2/HC markers
marker_legend = [Line2D([0], [0], marker=h2hc_marker_map[val], color='k', linestyle='', markersize=8) for val in unique_h2hc]
marker_labels = [f"H₂/HC = {val}" for val in unique_h2hc]

# Legend for line styles
line_legend = [
    Line2D([0], [0], color='k', lw=2, linestyle='-'),
    Line2D([0], [0], color='k', lw=2, linestyle=':')
]
line_labels = ['Source [1]', 'Source [2]']

# Add legends
first_legend = plt.legend(color_legend, color_labels, title="Pressure", loc='upper left', fontsize=10, title_fontsize=11)
plt.gca().add_artist(first_legend)
second_legend = plt.legend(marker_legend, marker_labels, title="H₂/HC Ratio", loc='upper right', fontsize=10, title_fontsize=11)
plt.gca().add_artist(second_legend)
plt.legend(line_legend, line_labels, title="Source", loc='lower right', fontsize=10, title_fontsize=11)


# Determine monoaromatics and polyaromatics content from source #1
df["Source"] = df["Source"].astype(str).str.strip("[]").astype(int)

mask_source1 = df["Source"] == 1

df_interp = df.copy()

# Select only rows with Source == 1
df_source1 = df_interp[mask_source1]

# Group by conditions that must match exactly for interpolation
group_cols = ["Pressure (bar)", "LHSV (1/hr)", "H2/HC (mL/mL)", "H2 purity"]

for group_vals, group_df in df_interp.loc[mask_source1].groupby(group_cols):
    known_points = group_df.dropna(subset=["Final monoaromatics content"])
    if len(known_points) < 2:
        continue
    f_interp = interp1d(
        known_points["Temperature (°C)"],
        known_points["Final monoaromatics content"],
        kind='linear',
        bounds_error=False,
        fill_value="extrapolate"
    )
    missing_rows = group_df[group_df["Final monoaromatics content"].isna()]
    for idx in missing_rows.index:
        temp_val = df_interp.at[idx, "Temperature (°C)"]
        interpolated_val = f_interp(temp_val)
        df_interp.at[idx, "Final monoaromatics content"] = float(interpolated_val)

    mask_over = df_interp["Final monoaromatics content"] > df_interp["Final aromatics content"]
    df_interp.loc[mask_over, "Final monoaromatics content"] = df_interp.loc[mask_over, "Final aromatics content"]


    df_interp["Final polyaromatics content"] = df_interp["Final aromatics content"] - df_interp["Final monoaromatics content"]

    # Monoaromatics saturation efficiency
    df_interp["Monoaromatics saturation efficiency"] = (
    (df_interp["Initial monoaromatics content"] - (df_interp["Final monoaromatics content"]))
    / df_interp["Initial monoaromatics content"] * 100
    )

    # Polyaromatics saturation efficiency
    df_interp["Polyaromatics saturation efficiency"] = (
    (df_interp["Initial polyaromatics content"] - (df_interp["Final polyaromatics content"]))
    / df_interp["Initial polyaromatics content"] * 100
    )


# Find rows with missing Sulfur removal %
missing_sulfur_mask = df_interp["Sulfur removal %"].isna()

# Loop over each group with same operating conditions
for group_vals, group_df in df_interp[mask_source1].groupby(group_cols):
    known_sulfur = group_df.dropna(subset=["Sulfur removal %"])
    if len(known_sulfur) < 2:
        # Need at least 2 points to interpolate
        continue
    f_sulfur_interp = interp1d(
        known_sulfur["Temperature (°C)"],
        known_sulfur["Sulfur removal %"],
        kind="linear",
        bounds_error=False,
        fill_value="extrapolate"
    )
    # Find rows in this group needing interpolation
    missing_rows_sulfur = group_df[group_df["Sulfur removal %"].isna()]
    for idx in missing_rows_sulfur.index:
        temp_val = df_interp.at[idx, "Temperature (°C)"]
        interpolated_val = f_sulfur_interp(temp_val)
        df_interp.at[idx, "Sulfur removal %"] = float(interpolated_val)

    
# List the columns you want to check for NaNs
cols_to_check = [
    "Temperature (°C)",
    "Pressure (bar)",
    "H2/HC (mL/mL)",
    "Initial aromatics content",
    "Initial monoaromatics content",
    "Initial polyaromatics content",
    "Final aromatics content",
    "Final monoaromatics content",
    "Final polyaromatics content",
    "Monoaromatics saturation efficiency",
    "Polyaromatics saturation efficiency",
    "Sulfur removal %"
]

filtered = df_interp.loc[mask_source1, cols_to_check].dropna()

filtered_sorted = filtered.sort_values(by="Temperature (°C)")

print(filtered_sorted)

cols_to_plot = [
    "Final aromatics content",
    "Final monoaromatics content",
    "Final polyaromatics content"
]

unique_pressures = filtered_sorted["Pressure (bar)"].unique()

# Assign a unique color per pressure
color_cycle = plt.cm.tab10.colors  # Up to 10 distinct colors
pressure_colors = {p: color_cycle[i % len(color_cycle)] for i, p in enumerate(unique_pressures)}

# Different line styles per column
linestyles = {
    "Final aromatics content": "solid",
    "Final monoaromatics content": "dashed",
    "Final polyaromatics content": "dotted",
}

# Create the plot
plt.figure(figsize=(10, 7))

for pressure in unique_pressures:
    df_p = filtered_sorted[filtered_sorted["Pressure (bar)"] == pressure]
    
    for col in cols_to_plot:
        plt.plot(
            df_p["Temperature (°C)"],
            df_p[col],
            label=f"{col} @ {pressure} bar",
            color=pressure_colors[pressure],
            linestyle=linestyles[col],
            marker='o'
        )

plt.xlabel("Temperature (°C)")
plt.ylabel("Content")
plt.title("Aromatics Content vs Temperature by Pressure")
plt.legend()
plt.grid(True)
plt.tight_layout()


cols_to_plot = [
    "Monoaromatics saturation efficiency",
    "Polyaromatics saturation efficiency",
]

unique_pressures = filtered_sorted["Pressure (bar)"].unique()

# Assign a unique color per pressure
color_cycle = plt.cm.tab10.colors  # Up to 10 distinct colors
pressure_colors = {p: color_cycle[i % len(color_cycle)] for i, p in enumerate(unique_pressures)}

# Different line styles per column
linestyles = {
    "Monoaromatics saturation efficiency": "solid",
    "Polyaromatics saturation efficiency": "dashed"
}

# Create the plot
plt.figure(figsize=(10, 7))

for pressure in unique_pressures:
    df_p = filtered_sorted[filtered_sorted["Pressure (bar)"] == pressure]
    
    for col in cols_to_plot:
        plt.plot(
            df_p["Temperature (°C)"],
            df_p[col],
            label=f"{col} @ {pressure} bar",
            color=pressure_colors[pressure],
            linestyle=linestyles[col],
            marker='o'
        )

plt.xlabel("Temperature (°C)")
plt.ylabel("%")
plt.title("Aromatics Saturation Efficiciency vs Temperature by Pressure")
plt.legend()
plt.grid(True)
plt.tight_layout()


cols_to_plot = [
    "Sulfur removal %"
]

unique_pressures = filtered_sorted["Pressure (bar)"].unique()

# Assign a unique color per pressure
color_cycle = plt.cm.tab10.colors  # Up to 10 distinct colors
pressure_colors = {p: color_cycle[i % len(color_cycle)] for i, p in enumerate(unique_pressures)}


# Create the plot
plt.figure(figsize=(10, 7))

for pressure in unique_pressures:
    df_p = filtered_sorted[filtered_sorted["Pressure (bar)"] == pressure]
    
    for col in cols_to_plot:
        plt.plot(
            df_p["Temperature (°C)"],
            df_p[col],
            label=f"{col} @ {pressure} bar",
            color=pressure_colors[pressure],
            marker='o'
        )

plt.xlabel("Temperature (°C)")
plt.ylabel("%")
plt.title("Sulfur removal % vs Temperature by Pressure")
plt.legend()
plt.grid(True)
plt.tight_layout()


def interpolate_aromatics_saturation_efficiency(temp, pressure):
    """
    Interpolate monoaromatics and polyaromatics saturation efficiencies
    at a given temperature and pressure.
    """
    # Select rows matching the given pressure and no NaNs in relevant columns
    df_p = df_interp[
        (df_interp["Pressure (bar)"] == pressure)
        & df_interp["Monoaromatics saturation efficiency"].notna()
        & df_interp["Polyaromatics saturation efficiency"].notna()
    ]
    
    if df_p.empty:
        raise ValueError(f"No data available for pressure {pressure} bar.")
    
    if len(df_p) < 2:
        raise ValueError(f"Need at least 2 data points to interpolate for pressure {pressure} bar.")
    
    # Sort by temperature (important for interpolation)
    df_p_sorted = df_p.sort_values(by="Temperature (°C)")
    
    # Interpolation functions
    mono_interp = interp1d(
        df_p_sorted["Temperature (°C)"],
        df_p_sorted["Monoaromatics saturation efficiency"],
        kind="linear",
        bounds_error=False,
        fill_value="extrapolate"
    )
    
    poly_interp = interp1d(
        df_p_sorted["Temperature (°C)"],
        df_p_sorted["Polyaromatics saturation efficiency"],
        kind="linear",
        bounds_error=False,
        fill_value="extrapolate"
    )
    
    # Compute interpolated values
    mono_eff = float(mono_interp(temp))
    poly_eff = float(poly_interp(temp))
    
    return mono_eff, poly_eff




def interpolate_sulfur_removal_perc(temp, pressure):
    """
    Interpolate monoaromatics and polyaromatics saturation efficiencies
    at a given temperature and pressure.
    """
    # Select rows matching the given pressure and no NaNs in relevant columns
    df_p = df_interp[
        (df_interp["Pressure (bar)"] == pressure)
        & df_interp["Sulfur removal %"].notna()
    ]
    
    if df_p.empty:
        raise ValueError(f"No data available for pressure {pressure} bar.")
    
    if len(df_p) < 2:
        raise ValueError(f"Need at least 2 data points to interpolate for pressure {pressure} bar.")
    
    # Sort by temperature (important for interpolation)
    df_p_sorted = df_p.sort_values(by="Temperature (°C)")
    
    # Interpolation functions
    sulfur_removal_interp = interp1d(
        df_p_sorted["Temperature (°C)"],
        df_p_sorted["Sulfur removal %"],
        kind="linear",
        bounds_error=False,
        fill_value="extrapolate"
    )
    
    
    # Compute interpolated values
    sulfur_removal = float(sulfur_removal_interp(temp))
    
    return sulfur_removal
