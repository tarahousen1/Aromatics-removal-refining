import matplotlib.pyplot as plt
import numpy as np



# Refinery capacities (BPSD)
refinery_capacities = [25000, 100000, 250000]

# Cost premiums min/max ($/gal)
# Order: Basrah 25k, Basrah 100k, Basrah 250k, Oman 25k, Oman 100k, Oman 250k, Murban 25k, Murban 100k, Murban 250k, 

# Hydrotreatment (HT)
#HT_min_usd_gal = [1.58, 0.88, 0.68, 1.42, 0.75, 0.56, 1.08, 0.70, 0.57]
#HT_max_usd_gal = [2.08, 1.19, 0.90, 1.93, 1.06, 0.79, 1.50, 0.96, 0.76]


# Assuming H2 consumption range
HT_min_usd_gal = [1.64, 0.93, 0.73, 1.49, 0.81, 0.62, 1.19, 0.81, 0.67]
HT_max_usd_gal = [2.15, 1.24, 0.95, 2.01, 1.12, 0.85, 1.62, 1.07, 0.86]

# Extractive Distillation (ED)
ED_min_usd_gal = [1.38, 0.64, 0.42, 0.99, 0.46, 0.30, 1.12, 0.48, 0.29]
ED_max_usd_gal = [1.64, 0.83, 0.59, 1.22, 0.63, 0.44, 1.39, 0.70, 0.48]

# Conversion: USD/gal → cent/L
def usd_per_gallon_to_cents_per_liter(usd):
    return np.array(usd) * 100 / 3.78541

# Convert min/max values
HT_min = usd_per_gallon_to_cents_per_liter(HT_min_usd_gal)
HT_max = usd_per_gallon_to_cents_per_liter(HT_max_usd_gal)
ED_min = usd_per_gallon_to_cents_per_liter(ED_min_usd_gal)
ED_max = usd_per_gallon_to_cents_per_liter(ED_max_usd_gal)

# Separate by location
mars_HT_min, mars_HT_max = HT_min[:3], HT_max[:3]
oman_HT_min, oman_HT_max = HT_min[3:6], HT_max[3:6]
murban_HT_min, murban_HT_max = HT_min[6:], HT_max[6:]

mars_ED_min, mars_ED_max = ED_min[:3], ED_max[:3] 
oman_ED_min, oman_ED_max = ED_min[3:6], ED_max[3:6]
murban_ED_min, murban_ED_max = ED_min[6:], ED_max[6:]

# Compute midpoints and errors
def compute_mid_err(min_vals, max_vals):
    mid = (min_vals + max_vals)/2
    err = max_vals - mid
    return mid, err

oman_HT_mid, oman_HT_err = compute_mid_err(oman_HT_min, oman_HT_max)
mars_HT_mid, mars_HT_err = compute_mid_err(mars_HT_min, mars_HT_max)
murban_HT_mid, murban_HT_err = compute_mid_err(murban_HT_min, murban_HT_max)

oman_ED_mid, oman_ED_err = compute_mid_err(oman_ED_min, oman_ED_max)
mars_ED_mid, mars_ED_err = compute_mid_err(mars_ED_min, mars_ED_max)
murban_ED_mid, murban_ED_err = compute_mid_err(murban_ED_min, murban_ED_max)

# Plot setup
x = np.arange(len(refinery_capacities))
width = 0.12  # narrower to fit all 6 bars

fig, ax = plt.subplots(figsize=(12, 6))

# Bars with error bars
bars1 = ax.bar(x - 2.5*width, mars_HT_mid, width, yerr=mars_HT_err,
               capsize=4, label='Mars HT', color='lightblue')
bars2 = ax.bar(x - 1.5*width, oman_HT_mid, width, yerr=oman_HT_err,
               capsize=4, label='Oman HT', color='yellowgreen')
bars3 = ax.bar(x - 0.5*width, murban_HT_mid, width, yerr=murban_HT_err,
               capsize=4, label='Murban HT', color='salmon')

bars4 = ax.bar(x + 0.5*width, mars_ED_mid, width, yerr=mars_ED_err,
               capsize=4, label='Mars ED', color='lightblue', hatch='//', edgecolor='cadetblue')
bars5 = ax.bar(x + 1.5*width, oman_ED_mid, width, yerr=oman_ED_err,
               capsize=4, label='Oman ED', color='yellowgreen', hatch='//', edgecolor='olivedrab')
bars6 = ax.bar(x + 2.5*width, murban_ED_mid, width, yerr=murban_ED_err,
               capsize=4, label='Murban ED', color='salmon', hatch='//', edgecolor='darkred')

# Labels
ax.set_xlabel('Refinery Capacity (barrels per day)')
ax.set_ylabel('Cost Premium (cent per L)')
ax.set_title('Cost Premium Range by Refinery Capacity, Location, and Process')
ax.set_xticks(x)
ax.set_xticklabels([f'{cap:,}' for cap in refinery_capacities])
ax.legend(ncol=2)

# Add mid-value labels
for bars in [bars1, bars2, bars3, bars4, bars5, bars6]:
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + 0.05, f'{yval:.2f}', 
                ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.show()


# Drew's numbers
refinery_capacities = [25000, 100000, 250000]

# Cost premiums min/max ($/gal)
# Order: Basrah 25k, Basrah 100k, Basrah 250k, Oman 25k, Oman 100k, Oman 250k, Murban 25k, Murban 100k, Murban 250k, 

# Hydrotreatment (HT)
#HT_min_usd_gal = [1.58, 0.88, 0.68, 1.42, 0.75, 0.56, 1.08, 0.70, 0.57]
#HT_max_usd_gal = [2.08, 1.19, 0.90, 1.93, 1.06, 0.79, 1.50, 0.96, 0.76]

# Hydrotreatment (HT)
HT_min_usd_gal = [0.55, 0.30, 0.20, 0.50, 0.27, 0.17, 0.39, 0.23, 0.16]
HT_max_usd_gal = [0.74, 0.42, 0.28, 0.69, 0.39, 0.26, 0.55, 0.33, 0.23]

# Extractive Distillation (ED)
ED_min_usd_gal = [0.05, 0.03, 0.03, 0.04, 0.03, 0.02, 0.04, 0.02, 0.02]
ED_max_usd_gal = [0.01, 0.07, 0.06, 0.10, 0.07, 0.05, 0.09, 0.06, 0.05]

# Refinery capacities (barrels per day)

# ---------------------------
# Conversion and utilities
# ---------------------------

def usd_per_gallon_to_cents_per_liter(usd):
    return np.array(usd) * 100 / 3.78541

def compute_mid_err(min_vals, max_vals):
    min_vals = np.array(min_vals)
    max_vals = np.array(max_vals)
    mid = (min_vals + max_vals) / 2
    err = max_vals - mid
    return mid, err

# ---------------------------
# Convert units
# ---------------------------

HT_min = usd_per_gallon_to_cents_per_liter(HT_min_usd_gal)
HT_max = usd_per_gallon_to_cents_per_liter(HT_max_usd_gal)
ED_min = usd_per_gallon_to_cents_per_liter(ED_min_usd_gal)
ED_max = usd_per_gallon_to_cents_per_liter(ED_max_usd_gal)

# ---------------------------
# Split by location
# ---------------------------

mars_HT_min, mars_HT_max = HT_min[:3], HT_max[:3]
oman_HT_min, oman_HT_max = HT_min[3:6], HT_max[3:6]
murban_HT_min, murban_HT_max = HT_min[6:], HT_max[6:]

mars_ED_min, mars_ED_max = ED_min[:3], ED_max[:3]
oman_ED_min, oman_ED_max = ED_min[3:6], ED_max[3:6]
murban_ED_min, murban_ED_max = ED_min[6:], ED_max[6:]

# ---------------------------
# Midpoints and errors
# ---------------------------

mars_HT_mid, mars_HT_err = compute_mid_err(mars_HT_min, mars_HT_max)
oman_HT_mid, oman_HT_err = compute_mid_err(oman_HT_min, oman_HT_max)
murban_HT_mid, murban_HT_err = compute_mid_err(murban_HT_min, murban_HT_max)

mars_ED_mid, mars_ED_err = compute_mid_err(mars_ED_min, mars_ED_max)
oman_ED_mid, oman_ED_err = compute_mid_err(oman_ED_min, oman_ED_max)
murban_ED_mid, murban_ED_err = compute_mid_err(murban_ED_min, murban_ED_max)

# ---------------------------
# Cost composition percentages
# ---------------------------

HT_components = {
    'FCI': 0.18,
    'DOC': 0.41,
    'NG': 0.03,
    'Power': 0.03,
    'Loans': 0.32,
    'Taxes': 0.32
}

ED_components = {
    'FCI': 0.19,
    'DOC': 0.43,
    'NG': 0.0,
    'Power': 0.03,
    'Loans': 0.33,
    'Taxes': 0.03
}

def compute_component_breakdown(mid_values, components):
    return {comp: mid_values * frac for comp, frac in components.items()}

mars_HT_breakdown = compute_component_breakdown(mars_HT_mid, HT_components)
oman_HT_breakdown = compute_component_breakdown(oman_HT_mid, HT_components)
murban_HT_breakdown = compute_component_breakdown(murban_HT_mid, HT_components)

mars_ED_breakdown = compute_component_breakdown(mars_ED_mid, ED_components)
oman_ED_breakdown = compute_component_breakdown(oman_ED_mid, ED_components)
murban_ED_breakdown = compute_component_breakdown(murban_ED_mid, ED_components)

# ---------------------------
# Plot setup
# ---------------------------

x = np.arange(len(refinery_capacities))
width = 0.12
fig, ax = plt.subplots(figsize=(8, 6))

# Component colors (consistent between processes)
component_colors = {
    'FCI': '#ff7f0e',   # bright orange
    'DOC': '#1f77b4',   # bright blue
    'NG': '#9467bd', # bright purple
    'Power': '#2ca02c',    # bright green 
    'Loans': '#e377c2', # bright pink
    'Taxes': '#d62728'  # bright red
}

# ---------------------------
# Plot stacked HT
# ---------------------------

def plot_stacked(ax, x_pos, breakdown, err, label_prefix, process, hatch=None):
    """Stacked bar plot with optional hatch and error bars"""
    bottom = np.zeros_like(x, dtype=float)
    for comp, vals in breakdown.items():
        ax.bar(
            x_pos, vals, width, bottom=bottom,
            color=component_colors[comp],
            hatch=hatch,
            edgecolor='black',
            alpha=0.9 if hatch is None else 0.7,
            label=f"{label_prefix} {process} {comp}" if x_pos[0] == x[0] else ""
        )
        bottom += vals
    # Safe total error bar
    ax.errorbar(x_pos, bottom, yerr=np.abs(err), fmt='none', ecolor='black', capsize=4)



# Hydrotreatment
plot_stacked(ax, x - 2.5*width, mars_HT_breakdown, mars_HT_err, 'Mars', 'HT')
plot_stacked(ax, x - 1.5*width, oman_HT_breakdown, oman_HT_err, 'Oman', 'HT')
plot_stacked(ax, x - 0.5*width, murban_HT_breakdown, murban_HT_err, 'Murban', 'HT')

# Extractive Distillation
plot_stacked(ax, x + 0.5*width, mars_ED_breakdown, mars_ED_err, 'Mars', 'ED', hatch='//')
plot_stacked(ax, x + 1.5*width, oman_ED_breakdown, oman_ED_err, 'Oman', 'ED', hatch='//')
plot_stacked(ax, x + 2.5*width, murban_ED_breakdown, murban_ED_err, 'Murban', 'ED', hatch='//')


# ---------------------------
# Labels and aesthetics
# ---------------------------

ax.set_xlabel('Refinery Capacity (barrels per day)', fontsize=12)
ax.set_ylabel('Cost Premium (cent per L)', fontsize=12)
ax.set_title('Cost Premium Composition by Refinery Capacity, Location, and Process', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels([f'{cap:,}' for cap in refinery_capacities])
ax.axhline(0, color='black', linewidth=1)
ax.legend(ncol=3, fontsize=8)
ax.grid(axis='y', linestyle='--', alpha=0.4)


# --- Component legend ---
from matplotlib.patches import Patch

component_patches = [
    Patch(facecolor=color, edgecolor='black', label=comp)
    for comp, color in component_colors.items()
]

# --- Process legend (solid vs. hatched) ---
process_patches = [
    Patch(facecolor='white', edgecolor='black', label='Hydrotreatment (HT)', hatch=None),
    Patch(facecolor='lightgray', edgecolor='black', label='Liquid-Liquid Extraction (LLE)', hatch='//')
]

# Combine legends
first_legend = ax.legend(handles=component_patches, title="Cost Components",
                         ncol=3, fontsize=9, title_fontsize=10,
                         loc='upper left', bbox_to_anchor=(1.02, 1))
ax.add_artist(first_legend)

ax.legend(handles=process_patches, title="Process Type",
          fontsize=9, title_fontsize=10,
          loc='upper left', bbox_to_anchor=(1.02, 0.75))

plt.tight_layout()
plt.show()






# Drew's numbers old operating cost system!!!!
refinery_capacities = [25000, 100000, 250000]

# Cost premiums min/max ($/gal)
# Order: Basrah 25k, Basrah 100k, Basrah 250k, Oman 25k, Oman 100k, Oman 250k, Murban 25k, Murban 100k, Murban 250k, 

# Hydrotreatment (HT)
#HT_min_usd_gal = [1.58, 0.88, 0.68, 1.42, 0.75, 0.56, 1.08, 0.70, 0.57]
#HT_max_usd_gal = [2.08, 1.19, 0.90, 1.93, 1.06, 0.79, 1.50, 0.96, 0.76]

# Hydrotreatment (HT)
HT_min_usd_gal = [0.68, 0.36, 0.23, 0.62, 0.34, 0.22, 0.48, 0.27, 0.18]
HT_max_usd_gal = [1.03, 0.57, 0.38, 0.97, 0.55, 0.37, 0.77, 0.46, 0.31]

# Extractive Distillation (ED)
ED_min_usd_gal = [0.09, 0.05, 0.04, 0.07, 0.04, 0.04, 0.07, 0.04, 0.03]
ED_max_usd_gal = [0.19, 0.12, 0.09, 0.18, 0.12, 0.09, 0.17, 0.11, 0.08]

# Refinery capacities (barrels per day)

# ---------------------------
# Conversion and utilities
# ---------------------------

def usd_per_gallon_to_cents_per_liter(usd):
    return np.array(usd) * 100 / 3.78541

def compute_mid_err(min_vals, max_vals):
    min_vals = np.array(min_vals)
    max_vals = np.array(max_vals)
    mid = (min_vals + max_vals) / 2
    err = max_vals - mid
    return mid, err

# ---------------------------
# Convert units
# ---------------------------

HT_min = usd_per_gallon_to_cents_per_liter(HT_min_usd_gal)
HT_max = usd_per_gallon_to_cents_per_liter(HT_max_usd_gal)
ED_min = usd_per_gallon_to_cents_per_liter(ED_min_usd_gal)
ED_max = usd_per_gallon_to_cents_per_liter(ED_max_usd_gal)

# ---------------------------
# Split by location
# ---------------------------

mars_HT_min, mars_HT_max = HT_min[:3], HT_max[:3]
oman_HT_min, oman_HT_max = HT_min[3:6], HT_max[3:6]
murban_HT_min, murban_HT_max = HT_min[6:], HT_max[6:]

mars_ED_min, mars_ED_max = ED_min[:3], ED_max[:3]
oman_ED_min, oman_ED_max = ED_min[3:6], ED_max[3:6]
murban_ED_min, murban_ED_max = ED_min[6:], ED_max[6:]

# ---------------------------
# Midpoints and errors
# ---------------------------

mars_HT_mid, mars_HT_err = compute_mid_err(mars_HT_min, mars_HT_max)
oman_HT_mid, oman_HT_err = compute_mid_err(oman_HT_min, oman_HT_max)
murban_HT_mid, murban_HT_err = compute_mid_err(murban_HT_min, murban_HT_max)

mars_ED_mid, mars_ED_err = compute_mid_err(mars_ED_min, mars_ED_max)
oman_ED_mid, oman_ED_err = compute_mid_err(oman_ED_min, oman_ED_max)
murban_ED_mid, murban_ED_err = compute_mid_err(murban_ED_min, murban_ED_max)

# ---------------------------
# Cost composition percentages
# ---------------------------

HT_components = {
    'FCI': 0.06,
    'DOC': 0.76,
    'NG': 0.02,
    'Power': 0.03,
    'Loans': 0.11,
    'Taxes': 0.01
}

ED_components = {
    'FCI': 0.10,
    'DOC': 0.62,
    'NG': 0.01,
    'Power': 0.01,
    'Loans': 0.23,
    'Taxes': 0.02
}

def compute_component_breakdown(mid_values, components):
    return {comp: mid_values * frac for comp, frac in components.items()}

mars_HT_breakdown = compute_component_breakdown(mars_HT_mid, HT_components)
oman_HT_breakdown = compute_component_breakdown(oman_HT_mid, HT_components)
murban_HT_breakdown = compute_component_breakdown(murban_HT_mid, HT_components)

mars_ED_breakdown = compute_component_breakdown(mars_ED_mid, ED_components)
oman_ED_breakdown = compute_component_breakdown(oman_ED_mid, ED_components)
murban_ED_breakdown = compute_component_breakdown(murban_ED_mid, ED_components)

# ---------------------------
# Plot setup
# ---------------------------

x = np.arange(len(refinery_capacities))
width = 0.12
fig, ax = plt.subplots(figsize=(8, 6))

# Component colors (consistent between processes)
component_colors = {
    'FCI': '#ff7f0e',   # bright orange
    'DOC': '#1f77b4',   # bright blue
    'NG': '#9467bd', # bright purple
    'Power': '#2ca02c',    # bright green 
    'Loans': '#e377c2', # bright pink
    'Taxes': '#d62728'  # bright red
}

# ---------------------------
# Plot stacked HT
# ---------------------------

def plot_stacked(ax, x_pos, breakdown, err, label_prefix, process, hatch=None):
    """Stacked bar plot with optional hatch and error bars"""
    bottom = np.zeros_like(x, dtype=float)
    for comp, vals in breakdown.items():
        ax.bar(
            x_pos, vals, width, bottom=bottom,
            color=component_colors[comp],
            hatch=hatch,
            edgecolor='black',
            alpha=0.9 if hatch is None else 0.7,
            label=f"{label_prefix} {process} {comp}" if x_pos[0] == x[0] else ""
        )
        bottom += vals
    # Safe total error bar
    ax.errorbar(x_pos, bottom, yerr=np.abs(err), fmt='none', ecolor='black', capsize=4)



# Hydrotreatment
plot_stacked(ax, x - 2.5*width, mars_HT_breakdown, mars_HT_err, 'Mars', 'HT')
plot_stacked(ax, x - 1.5*width, oman_HT_breakdown, oman_HT_err, 'Oman', 'HT')
plot_stacked(ax, x - 0.5*width, murban_HT_breakdown, murban_HT_err, 'Murban', 'HT')

# Extractive Distillation
plot_stacked(ax, x + 0.5*width, mars_ED_breakdown, mars_ED_err, 'Mars', 'ED', hatch='//')
plot_stacked(ax, x + 1.5*width, oman_ED_breakdown, oman_ED_err, 'Oman', 'ED', hatch='//')
plot_stacked(ax, x + 2.5*width, murban_ED_breakdown, murban_ED_err, 'Murban', 'ED', hatch='//')


# ---------------------------
# Labels and aesthetics
# ---------------------------

ax.set_xlabel('Refinery Capacity (barrels per day)', fontsize=12)
ax.set_ylabel('Cost Premium (cent per L)', fontsize=12)
ax.set_title('Cost Premium Composition by Refinery Capacity, Location, and Process', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels([f'{cap:,}' for cap in refinery_capacities])
ax.axhline(0, color='black', linewidth=1)
ax.legend(ncol=3, fontsize=8)
ax.grid(axis='y', linestyle='--', alpha=0.4)


# --- Component legend ---
from matplotlib.patches import Patch

component_patches = [
    Patch(facecolor=color, edgecolor='black', label=comp)
    for comp, color in component_colors.items()
]

# --- Process legend (solid vs. hatched) ---
process_patches = [
    Patch(facecolor='white', edgecolor='black', label='Hydrotreatment (HT)', hatch=None),
    Patch(facecolor='lightgray', edgecolor='black', label='Liquid-Liquid Extraction (LLE)', hatch='//')
]

# Combine legends
first_legend = ax.legend(handles=component_patches, title="Cost Components",
                         ncol=3, fontsize=9, title_fontsize=10,
                         loc='upper left', bbox_to_anchor=(1.02, 1))
ax.add_artist(first_legend)

ax.legend(handles=process_patches, title="Process Type",
          fontsize=9, title_fontsize=10,
          loc='upper left', bbox_to_anchor=(1.02, 0.75))

plt.tight_layout()
plt.show()



