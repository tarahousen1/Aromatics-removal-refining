import numpy as np
import toml
import math
from scipy.special import lambertw

fixed_parameters = toml.load('input_text_files/fixed_parameters.toml')
conversion_parameters = fixed_parameters['conversion_parameters']

# Function to find lb/hr conversion based on specific gravity
def find_lbhr_conversion(specific_gravity, density_conv_table):
    """
    This function finds the lb/hr conversion for a given specific gravity from the density conversion table.

    """
    lbhr_conversion = np.nan
    if not np.isnan(specific_gravity):
        valid_rows = density_conv_table[['Specific Gravity', 'lb/hr* from bbl/day']].dropna()
        sg_col = valid_rows['Specific Gravity'].astype(float)
        lbhr_col = valid_rows['lb/hr* from bbl/day'].astype(float)
        
        # Find the index of the row with the closest specific gravity
        idx = (np.abs(sg_col - specific_gravity)).values.argmin()
        
        # Get the corresponding lb/hr value
        lbhr_conversion = lbhr_col.iloc[idx]
    
    return lbhr_conversion

def calculate_molecular_fractions(SG, Tb, Kw, API, H_wt_percent):   
    # Characterization and Properties of Petroleum Fractions. Ed. Riazi, MR. 100 Barr Harbor Drive, PO Box C700, West Conshohocken, PA 19428-2959: ASTM International, 2007

    CH = (100 - H_wt_percent) / H_wt_percent

    # Characterization and Properties of Petroleum Fractions Eqn 2.50 
    M = 1.6607e-4 * ( (Tb **2.1962) * (SG**-1.0164) )   # (Tb in K)

    # Characterization and Properties of Petroleum Fractions Eqn 2.111
    d_20 = SG - 0.0045 * (2.34 - 1.9*SG)

    # Characterization and Properties of Petroleum Fractions Eqn 2.182
    kinematic_viscosity_38C = 4.39371 - 1.94733 * Kw + 0.12769 * Kw**2 + 0.00032629 * API **2 - 0.0118246 * Kw * API + ( 0.171617 * Kw**2 + 10.9943 * API + 0.0950663 * API **2 - 0.8260218 * Kw * API) / ( API + 50.3642 - 4.78231 * Kw )
    
    "   Valid for M = 70-300    "
    if 70 < M < 300:
    
        I = 0.3773* Tb**-0.02269 * SG**0.9182       # Eqn 2.115
        n = ((1 + 2*I)/(1- I))** 0.5                # Eqn 2.114
        R_i = n - d_20/2                            # Eqn 2.14
        # Calculate factor m:
        m = M * (n - 1.475)

        if M < 200:

            VGF = -1.816 + 3.484 * SG - 0.1156 * np.log(kinematic_viscosity_38C)    #  Eqn 3.68

            # Determine molecular fraction of paraffins and napthenes
            x_P = -13.359 + 14.4591 * R_i - 1.41344 * VGF                           # Eqn 3.70
            x_N = 23.9825 - 23.3304 * R_i + 0.81517 * VGF                           # Eqn 3.71
        
            # Determine molecular fraction of paraffins and napthenes
            #x_P = 3.7387 - 4.0829 * SG + 0.014772 * m
            #x_N = -1.5027 + 2.10152 * SG - 0.02388 * m

        elif 600 > M > 200:

            # Determine molecular fraction of paraffins and napthenes
            #x_P = 2.5737 + 1.0133 * R_i - 3.573 * VGC                               # Eqn 3.73
            #x_N = 2.464 - 3.6701 * R_i + 1.96312 * VGC                              # Eqn 3.74
            x_P = 1.9842 - 0.27722 * R_i - 0.15643 * CH                              # Eqn 3.79
            x_N = 0.5977 - 0.761745 * R_i + 0.068048 * CH                            # Eqn 3.80

        x_A = 1 - x_P - x_N
        if x_A < 0:
            x_A = 0
            x_P = x_P / (x_P + x_N)
            x_N = 1 - x_P

        # Determine molecular fraction of monoaromatics and polyaromatics
        "   Valid for aromatic content = 0.05–0.96 and M = 80–250    "
        if 0.05 < x_A < 0.96:
            if 80 < M < 250:
                x_MA = -62.8245 + 59.90816 * R_i - 0.0248335 * m
                x_PA = 11.88175 - 11.2213 * R_i + 0.023745 * m
                if x_MA + x_PA != x_A:
                    x_MA = x_A * x_MA / (x_MA + x_PA)
                    x_PA = x_A * x_PA / (x_MA + x_PA)
            else:
                print('Molecular weight out of range for calculation of monoaromatics and polyaromatics content')
                return M, n, R_i, x_P, x_N, None, None, None, x_A
        else:
            print('Aromatics content out of range for calculation of monoaromatics and polyaromatics content')
            return M, n, R_i, x_P, x_N, None, None, None, x_A

    else:
        print('Molecular weight out of range for calculation of parameter I')
        return M, None, None, None, None, None, None, None

    return M, n, R_i, x_P, x_N, x_MA, x_PA, x_A

def compute_hydrogen_consumption1(
    S_f, S_p, N_f, N_p, PNA_f, PNA_p, MA_f, MA_p,
    sgf, sgp, MW_feed, MW_prod, Yp, total_hydrotreatment_inputs_bpcd,
    H_purity, P_H, GO
    ):
    """
    Computes hydrogen consumption for HDS, HDN, HDA, and hydrogen dissolved.

    GO can be:
      - A single value (int or float)
      - A 2-element list/tuple [min, max] representing a range
    """

    def _calc(GO_value):
        # Hydrogen consumption for desulfurization
        H_HDS = 3.6/0.032 * 1000 * sgf * ( S_f/(1_000_000) -  ( (S_p/(1_000_000) * sgp / sgf) * Yp/100 ) ) * 0.02241

        # Hydrogen consumption for denitrogenation
        H_HDN = 5/0.014 * 1000 * sgf * ( N_f/(1_000_000) -  ( (N_p/(1_000_000) * sgp / sgf) * Yp/100 ) ) * 0.02241

        # Hydrogen consumption for saturation of aromatics
        H_HDA_poly = 2 * 1000 * sgf * (((PNA_f/100) / (MW_feed/1000)) - ((PNA_p/100) * (Yp/100) * (sgp/ sgf) / (MW_prod/1000))) * 0.02241
        H_HDA_mono = 3 * 1000 * sgf * (((MA_f/100) / (MW_feed/1000)) - ((MA_p/100) * (Yp/100) * (sgp/ sgf) / (MW_prod/1000))) * 0.02241
        H_HDA = H_HDA_poly + H_HDA_mono

        # Total chemical hydrogen
        H_chem_total_Nm3_per_m3 = H_HDS + H_HDN + H_HDA
        H_chem_scf_per_bbl = H_chem_total_Nm3_per_m3 / conversion_parameters['H2_scf_to_Nm3'] * conversion_parameters['bbl_to_m3']
        H_chem_lb_per_hr = H_chem_scf_per_bbl * total_hydrotreatment_inputs_bpcd / 24 * conversion_parameters['H2_scf_to_lb']

        # Dissolved hydrogen
        H_diss_scf_per_bbl = (-23.95817 + 0.67529 * H_purity - 3.56483 * P_H
                              + 2.15964e-3 * GO_value - 3.94802e-3 * H_purity**2
                              + 0.23328 * P_H**2 + 0.11314 * H_purity * P_H
                              - 1.69706e-4 * H_purity * GO_value)

        H_diss_lb_per_hr = H_diss_scf_per_bbl * total_hydrotreatment_inputs_bpcd / 24 * conversion_parameters['H2_scf_to_lb']

        return {
            "GO": GO_value,
            "H_HDS": H_HDS, "H_HDN": H_HDN, "H_HDA": H_HDA,
            "H_chem_total_Nm3_per_m3": H_chem_total_Nm3_per_m3,
            "H_chem_lb_per_hr": H_chem_lb_per_hr, "H_chem_scf_per_bbl": H_chem_scf_per_bbl,
            "H_diss_lb_per_hr": H_diss_lb_per_hr, "H_diss_scf_per_bbl": H_diss_scf_per_bbl
        }

    # Handle range
    if isinstance(GO, (list, tuple)) and len(GO) == 2:
        result_min = _calc(GO[0])
        result_max = _calc(GO[1])
        return {"min": result_min, "max": result_max}

    # Handle single value
    return _calc(GO)


def compute_hydrogen_consumption(
    S_f, S_p, N_f, N_p, PNA_f, PNA_p, MA_f, MA_p,
    sgf, sgp, MW_feed, MW_prod, Yp, total_hydrotreatment_inputs_bpcd,
    H_purity, P_H, GO
    ):

    """
    S_f: wppm sulfur in feed
    S_p: wppm sulfur in product
    N_f: wppm nitrogen in feed
    N_p: wppm nitrogen in product
    PNA_f: wt% of polyaromatics in feed (0-100%)
    PNA_p: wt% of polyaromatics in product (0-100%)
    MA_f: wt% of monoaromatics in feed (0-100%)
    MA_p: wt% of monoaromatics in product (0-100%)
    Yp: % volumetric conversion of feed to product (0-100%)
    sgf: specific gravity of feed
    sgp: specific gravity of product
    MW_feed: molecular weight of feed (g/mol)
    MW_prod: molecular weight of product (g/mol)

    """

    # Hydrogen consumption for desulfurization
    H_HDS = 3.6/0.032 * 1000 * sgf * ( S_f/(1000000) -  ( (S_p/(1000000) * sgp / sgf) * Yp/100 ) ) * 0.02241

    # Hydrogen consumption for denitrogenation
    H_HDN = 5/0.014 * 1000 * sgf * ( N_f/(1000000) -  ( (N_p/(1000000) * sgp / sgf) * Yp/100 ) ) * 0.02241

    H_HDA_poly = 2 * 1000 * sgf * (((PNA_f/100) / (MW_feed/1000)) - ((PNA_p/100) * (Yp/100) * (sgp/ sgf) / (MW_prod/1000))) * 0.02241
    H_HDA_mono = 3 * 1000 * sgf * (((MA_f/100) / (MW_feed/1000)) - ((MA_p/100) * (Yp/100) * (sgp/ sgf) / (MW_prod/1000))) * 0.02241
   
    H_HDA = H_HDA_poly + H_HDA_mono

    H_chem_total_Nm3_per_m3 = H_HDS + H_HDN + H_HDA  # Nm3 H2 / m3 oil
    H_chem_scf_per_bbl = H_chem_total_Nm3_per_m3 / conversion_parameters['H2_scf_to_Nm3'] * conversion_parameters['bbl_to_m3']
    H_chem_lb_per_hr = H_chem_scf_per_bbl * total_hydrotreatment_inputs_bpcd / 24 * conversion_parameters['H2_scf_to_lb']

    # Calculate non-reacted hydrogen dissolved in liquid phase
    H_diss_scf_per_bbl = -23.95817 + 0.67529 * H_purity - 3.56483 * P_H + 2.15964 * 10**(-3) * GO - 3.94802 * 10**(-3) * H_purity**2 + 0.23328 * P_H **2 + 0.11314 * H_purity * P_H - 1.69706 * 10**(-4) * H_purity * GO
    H_diss_lb_per_hr = H_diss_scf_per_bbl * total_hydrotreatment_inputs_bpcd / 24 * conversion_parameters['H2_scf_to_lb']

    return H_HDS, H_HDN, H_HDA, H_chem_total_Nm3_per_m3, H_chem_lb_per_hr, H_chem_scf_per_bbl, H_diss_lb_per_hr, H_diss_scf_per_bbl


def product_height_diameter(height_m, diameter_m):
    """
    Compute the range (min, max) of (height * diameter^1.5)
    given height and diameter as scalars or ranges (tuples/lists).

    Parameters
    ----------
    height_m : float or tuple
        Column height (m) or range (min, max)
    diameter_m : float or tuple
        Column diameter (m) or range (min, max)

    Returns
    -------
    tuple
        (min_value, max_value) of height * diameter^1.5
    """
    # Normalize to tuples
    h = height_m if isinstance(height_m, (tuple, list)) else (height_m, height_m)
    d = diameter_m if isinstance(diameter_m, (tuple, list)) else (diameter_m, diameter_m)

    # Compute all combinations
    values = [h_val * (d_val)**1.5 for h_val in h for d_val in d]

    return (min(values), max(values))



def tray_stack_height(column_height, tray_spacing, top_clearance=2.5, bottom_clearance=2.5):
    """
    Calculate tray stack height (m) for a given column height or range.

    Parameters
    ----------
    column_height : float or tuple
        Total shell height of the column (m) or range (min, max)
    tray_spacing : float
        Vertical distance between trays (m)
    top_clearance : float, optional
        Top disengagement/head space (m)
    bottom_clearance : float, optional
        Bottom liquid/reboiler/head space (m)

    Returns
    -------
    float or tuple
        Tray stack height (m) or range (min, max)
    """
    # Normalize to tuple
    heights = column_height if isinstance(column_height, (tuple, list)) else (column_height, column_height)

    results = []
    for h in heights:
        available_height = h - (top_clearance + bottom_clearance)
        n_trays = math.floor(available_height / tray_spacing)
        stack_height = n_trays * tray_spacing
        results.append(stack_height)

    # If a range was provided, return (min, max); else return scalar
    return (min(results), max(results)) if len(results) > 1 else results[0]
