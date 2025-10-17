import matplotlib.pyplot as plt
import numpy as np
from adjustText import adjust_text

# Data https://www.researchgate.net/figure/Data-of-API-gravity-and-sulfur-content-in-the-universal-crude-oil-Data-Referenced-by_tbl1_280610435
crude_names = [
    "Iraq - Basrah Heavy","Alaska - Prudhoe Bay", "Abu Dhabi - Murban", "US - Heavy Louisiana Sweet", "Kuwait", "US - West Texas Intermediate",
    "North Sea - Brent", "Mexico - Maya", "Saudi Arabia Heavy", "Saudi Arabia Light",
    "Iran Heavy", "Iran Light", "UAE Dubai", "Oman", "Nigeria - Bonny light",
    "Algeria - Sahara Blend", "Ecuador - Oriente", "Malaysia - Tapis", "US - Mars",
    "Western Canada Select"
]

api_gravity = [23.7, 27.8, 40.0, 34.8, 34.5, 39.6, 38.3, 21.8, 27.7, 32.8, 30.2, 33.1, 30.4, 33.2, 33.4, 45, 24.1, 45.2, 30.3, 20.3]
sulfur_content = [4.12, 0.94, 0.03, 0.48, 2.44, 0.24, 0.37, 3.33, 2.87, 1.97, 1.77, 1.5, 2.13, 1.29, 0.16, 0.09, 1.51, 0.03, 1.91, 3.43]
bpd_values = [
    1200000, # Iraq - Basrah Heavy
    496906,  # Alaska - Prudhoe Bay
    1700000, # "Abu Dhabi - Murban"
    3160000, # US - Heavy Louisiana Sweet
    2900000, # Kuwait
    5600000, # US - West Texas Intermediate
    625000,  # North Sea - Brent
    436000,  # Mexico - Maya
    3325000, # Saudi Arabia Heavy
    3230000, # Saudi Arabia Light
    3257000, # Iran Heavy
    3800000, # Iran Light
    3100000, # UAE Dubai
    1000000, # Oman
    1300000, # Nigeria - Bonny light
    445000,  # Algeria - Sahara Blend
    482000,  # Ecuador - Oriente
    200000,  # Malaysia - Tapis
    600000,  # US - Mars
    890000,  # Western Canada Select
]

# Normalize BPD values for marker sizing (adjust scaling as needed)
bpd_scaled = np.array(bpd_values)
bpd_scaled = 700 * (bpd_scaled / bpd_scaled.max())  # Increased from 100 to 300

# Create scatter plot
plt.figure(figsize=(12, 7))
colors = plt.cm.tab20(np.linspace(0, 1, len(crude_names)))
sc = plt.scatter(api_gravity, sulfur_content, color=colors, s=bpd_scaled, edgecolor='black', alpha=0.7)


# Annotate points using adjustText to avoid overlapping
# Annotate points using adjustText to avoid overlapping circles and text
texts = []
for i, name in enumerate(crude_names):
    # Offset label slightly to the right and upward
    x = api_gravity[i] + 0.3
    y = sulfur_content[i] + 0.05
    texts.append(plt.text(x, y, name, fontsize=9))

adjust_text(texts, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))

# Labels and title
plt.xlabel("API Gravity (°API)")
plt.ylabel("Sulfur Content (%)")
plt.title("API Gravity and Sulfur Content of Crude Oils")

# Grid and layout
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('outputs/api_sulfur_content_crude.png', dpi=300)
