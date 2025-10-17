import numpy as np
import matplotlib.pyplot as plt

# https://asmedigitalcollection.asme.org/GT/proceedings/GT1990/79061/V003T06A033/238033
# Define the range of hydrogen content values
h_values = np.linspace(12.5, 14.5, 100)

# Calculate TAR for each hydrogen content
tar_values = 298.0 - 20.03 * h_values

# Plotting with switched axes
plt.figure(figsize=(8, 5))
plt.plot(tar_values, h_values, label='H = (298.0 - TAR) / 20.03', color='green')
plt.xlim(0, 55)  
plt.xlabel('Total Aromatics Content (vol%)')
plt.ylabel('Hydrogen Content (wt%)')
plt.title('Hydrogen Content vs Total Aromatics')
plt.grid(True)
plt.legend()
plt.tight_layout()

def estimate_aromatics_content(hydrogen_content):
    """
    Estimate total aromatics content (vol%) from hydrogen content (wt%).

    Parameters:
    hydrogen_content (float or array-like): Hydrogen content in wt%

    Returns:
    float or array-like: Estimated aromatics content in vol%
    """
    return 298.0 - 20.03 * np.asarray(hydrogen_content)