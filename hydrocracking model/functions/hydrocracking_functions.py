import numpy as np
import toml
import math
from scipy.special import lambertw

fixed_parameters = toml.load('fixed_parameters.toml')
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

# Kinetic model of hydrotreatment
def power_law_kinetic_model(C_f, k_0, E, n, LHSV, T):
    """
    Calculate the outlet concentration (C_p) of a reactant in a flow reactor 
    using a power-law kinetic model under steady-state conditions.

    Parameters:
        C_f (float): Feed concentration of the reactant (e.g., wt% or mol/L).
        k_0 (float): Arrhenius pre-exponential factor (same units as rate constant).
        E (float): Activation energy in kJ/mol.
        n (float): Reaction order.
        LHSV (float): Liquid Hourly Space Velocity (hr⁻¹).
        T (float): Reaction temperature in Kelvin (K).

        Assumptions:
            - Ideal plug flow reactor (PFR) or CSTR with power-law kinetics.
            - Single reactant conversion; side reactions and product inhibition not considered.
            - Constant physical properties.

    Returns:
        C_p (float): Outlet concentration of the reactant after reaction (same unit as C_f).
    """
    R = 0.008314 
    T = T + 273.15 
    k = k_0 * math.exp(-E / (R * T))

    if n == 0:
        C_p = C_f - k/LHSV
    elif n == 1:
        C_p = C_f * math.exp(-k / LHSV)
    else:
        #denom = (n - 1) * k / LHSV + 1 / (C_f ** (n - 1))
        #C_p = (1 / denom) ** (1 / (n - 1))
        C_p = (C_f**(1 - n) - (1 - n) * (k / LHSV))**(1 / (1 - n))
    return k, C_p

def langmuir_hinshelwood_model(C_f, k, Ki, KH2, KH2S, PH2, PH2S, LHSV):
    """
    Calculate product concentration Cp using simplified Langmuir-Hinshelwood kinetic model.
    
    Parameters:
    C_f (float): Feed concentration (wt%)
    k (float): Apparent rate constant (ki)
    Ki (float): Adsorption equilibrium constant for reactant i
    KH2 (float): Adsorption constant for H2
    KH2S (float): Adsorption constant for H2S
    PH2 (float): Partial pressure of H2
    PH2S (float): Partial pressure of H2S
    LHSV (float): Liquid hourly space velocity
    
    Returns:
    Cp (float): Product concentration (wt%)
    """
    denominator = 1 + Ki * C_f + KH2 * PH2 + KH2S * PH2S
    A = Ki / denominator
    B = (k * PH2) / denominator
    
    # Lambert W function argument for solution (simplified form)
    arg = A * C_f * np.exp(A * C_f - B * LHSV)
    W = lambertw(arg)
    
    C_p = (W / A).real  # Take real part in case of small imaginary residual
    
    return k, C_p

def multi_parameter_kinetic_model(C_f, k_0, s, n, LHSV, T, P_H, m, G_O, q):
    """
    Calculate product concentration Cp using extended kinetic model.

    Parameters:
        C_f (float): Feed concentration
        k_0 (float): Pre-exponential factor
        s (float): Exponential temperature factor (e.g., activation energy / R)
        n (float): Reaction order
        LHSV (float): Liquid hourly space velocity (1/hr)
        T (float): Temperature in Kelvin
        P_H (float): Partial pressure of H₂
        m (float): Exponent on P_H
        G_O (float): Gas/Oil ratio
        q (float): Exponent on G/O

    Returns:
        Cp (float): Product concentration
    """
    T = T + 273.15 

    rate_term = k_0 * math.exp(-s / T) * (P_H ** m) * (G_O ** q)

    if n == 1:
        #Cp = C_f / math.exp(RHS)
        C_p = C_f * np.exp(-rate_term / LHSV)
    else:
        #denom = (1 / (C_f ** (n - 1))) + (n - 1) * RHS
        #Cp_power = 1 / denom
        #Cp = Cp_power ** (1 / (n - 1))
        C_p = (C_f**(1 - n) - (1 - n) * (rate_term / LHSV))**(1 / (1 - n))
    return rate_term, C_p


def hydrogen_consumption(S_f, S_p, N_f, N_p, PNA_f, PNA_p, MA_f, MA_p, sgf, sgp, ):
    """
    Calculate total hydrogen consumption based on stoichiometric equations
    Inputs:
        S_i, S_f: Initial and final sulfur content (wt%)
        N_i, N_f: Initial and final nitrogen content (wppm)
        PNA: Polyaromatics content (wt%)
        MA: Monoaromatics content (wt%)
        sgf: Specific gravity of feed
        sgp: Specific gravity of product
        Yp: Liquid product yield (fraction), default = 1.0
    Returns:
        H_total: Total hydrogen consumption in Nm3 H2/m3 oil
    """
    # HDS hydrogen consumption
    H_HDS = 0.0252 * sgf * ( S_f - (S_p *sgp / sgf) * Yp )

    # HDN hydrogen consumption
    H_HDN = 0.08 * sgf * ( (N_f/ (10**4)) - ((N_p/ (10**4)) *sgp / sgf) * Yp )

    # HDA hydrogen consumption
    H_HDA = 3.3 * sgf * ( PNA_f - ((PNA_p *sgp / sgf) * Yp)  + 3 * ( MA_f - (MA_p *sgp / sgf) * Yp))

    # Total hydrogen consumption
    H_total = H_HDS + H_HDN + H_HDA

    hydrotreatment_inputs['lb/hr']['Hydrogen'] = H_total

    # But we dont know Yp
    Yp = hydrotreatment_outputs['Volume %']['HT Kerosine (330-480°F)']
    hydrotreatment_outputs['BPCD']['HT Kerosine (330-480°F)'] = hydrotreatment_outputs['lb/hr']['HT Kerosine (330-480°F)']/ hydrotreatment_outputs['lb/hr from bbl/day']['HT Kerosine (330-480°F)']

    hydrotreatment_outputs['lb/hr']['HT Kerosine (330-480°F)'] = (hydrotreatment_inputs['lb/hr']['Kerosine (330-480°F)'] + hydrotreatment_inputs['lb/hr']['Hydrogen']) - ( hydrotreatment_outputs['lb/hr']['C3 and lighter'] + hydrotreatment_outputs['lb/hr']['iC4'] + hydrotreatment_outputs['lb/hr']['nC4'] )
    hydrotreatment_outputs['Volume %']['HT Kerosine (330-480°F)'] = hydrotreatment_outputs['BPCD']['HT Kerosine (330-480°F)']/  ( hydrotreatment_outputs['BPCD']['C3 and lighter'] + hydrotreatment_outputs['BPCD']['iC4'] + hydrotreatment_outputs['BPCD']['nC4'] + hydrotreatment_outputs['BPCD']['HT Kerosine (330-480°F)'] )

    return H_HDS, H_HDN, H_HDA, H_total
