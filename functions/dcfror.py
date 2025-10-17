
# DISCOUNTED CASH FLOW RATE OF RETURN FUNCTION

import toml
import numpy as np
from functions.depreciation import depgendb
from openpyxl import Workbook


def discounted_cash_flow(FCI, DOC, VOC, CrudeInputCost, InputNG, InputNG_SteamProduction, InputSMRFeedGas, InputPower, InputHydrogen, InputCoolingWater, OutputKerosene, OutputLightNaphtha, OutputHeavyNaphtha, OutputLightGasOil, OutputHeavyGasOil, OutputAtmResidue, OutputLightVacuumGasOil, OutputHeavyVacuumGasOil, OutputVacuumResidues, OutputPropane, OutputSulfur, OutputBTX, CostLightNaphtha, CostHeavyNaphtha, CostLightGasOil, CostHeavyGasOil, CostAtmResidue, CostLightVacuumGasOil, CostHeavyVacuumGasOil, CostVacuumResidues, CostPropane, CostSulfur, CostBTX, Cost_NGStart, Cost_PowerStart, Cost_HydrogenStart, Cost_CoolingWaterStart, Cost_ByproductsStart, equity, loan_interest, loan_term, WC, FCIDet, CostNGGrowth, CostNGSTD, WACC, inflation, ITR):
    
    
    # Load input parameters from the TOML files
    user_inputs = toml.load('/Users/tarahousen/Documents/Aromatics_removal_desulfurization_refining/input_text_files/user_inputs.toml')
    financial_assumptions_data = toml.load('/Users/tarahousen/Documents/Aromatics_removal_desulfurization_refining/input_text_files/financial_assumptions.toml')
    fixed_parameters =  toml.load('/Users/tarahousen/Documents/Aromatics_removal_desulfurization_refining/input_text_files/fixed_parameters.toml')



    CPY = financial_assumptions_data['financial_assumptions']['CPY'] # Construction period year
    CY_1 = financial_assumptions_data['financial_assumptions']['CY_1'] # Proportion of FCI spent during construction year 1
    CY_2 = financial_assumptions_data['financial_assumptions']['CY_2'] # Proportion of FCI spent during construction year 2
    CY_3 = financial_assumptions_data['financial_assumptions']['CY_3'] # Proportion of FCI spent during construction year 3
    DPY = financial_assumptions_data['financial_assumptions']['DPY'] # Depreciation period year


    HPY = financial_assumptions_data['financial_assumptions']['HPY'] # Operating hours per year
    VPY = financial_assumptions_data['financial_assumptions']['VPY'] # Valuation period years [Brynolf 2017]

    OPD = financial_assumptions_data['financial_assumptions']['OPD'] # operating days
    prod_capacity1 = financial_assumptions_data['financial_assumptions']['prod_capacity1'] # Production capacity in year 1 (50% in first 6 months, 100% onward) [Pearlson 2013]
    prod_capacity2 = financial_assumptions_data['financial_assumptions']['prod_capacity2'] # Production capacity in year 2 (100%)


    epsilon = 1e-4
    n = 0
    NPV_1 = 0
    NPV_2 = np.inf
    slope = 1.3e8
    Cost_MDi = 0

    while abs(NPV_2) > epsilon:
        n += 1
        NPV_1 = NPV_2

        # Flow Initialization
        Year = np.zeros(CPY + VPY)
        Flow_FCI = np.zeros(CPY + VPY)
        Flow_WC = np.zeros(CPY + VPY)
        Flow_LoanPrincipal = np.zeros(CPY + VPY)
        Flow_LoanInterest = np.zeros(CPY + VPY)
        Flow_Discount_Factor = np.zeros(CPY + VPY)
        Flow_TCI_Interest = np.zeros(CPY + VPY)
        Flow_TotalSales = np.zeros(CPY + VPY)
        Flow_DOC = np.zeros(CPY + VPY)
        Flow_NG = np.zeros(CPY + VPY)
        Flow_Power = np.zeros(CPY + VPY)
        Flow_H2 = np.zeros(CPY + VPY)
        Flow_CoolingWater = np.zeros(CPY + VPY)
        Flow_VOC = np.zeros(CPY + VPY)
        Flow_CrudeInputCost = np.zeros(CPY + VPY)
        Flow_EBITDA = np.zeros(CPY + VPY)
        Flow_VDB = np.zeros(CPY + VPY)
        Flow_EBIT = np.zeros(CPY + VPY)
        Flow_LoanPayment = np.zeros(CPY + VPY)
        Flow_TaxableIncome = np.zeros(CPY + VPY)
        Flow_Losses = np.zeros(CPY + VPY)
        Flow_IncomeTax = np.zeros(CPY + VPY)
        Flow_IncomeAfterTax = np.zeros(CPY + VPY)
        Flow_NetAnnualIncome = np.zeros(CPY + VPY)
        Flow_IRR = np.zeros(CPY + VPY)
        Flow_APV = np.zeros(CPY + VPY)
        Flow_LoanPaymentcomp = np.zeros(CPY + VPY)
        Flow_DOCcomp = np.zeros(CPY + VPY)
        Flow_VOCcomp = np.zeros(CPY + VPY)
        Flow_CrudeInputCostcomp = np.zeros(CPY + VPY)
        Flow_Taxcomp = np.zeros(CPY + VPY)
        Flow_KEROcomp = np.zeros(CPY + VPY)
        Flow_GAScomp = np.zeros(CPY + VPY)
        Flow_COPROcomp = np.zeros(CPY + VPY)
        Flow_RemainingValue = np.zeros(CPY + VPY)
        Flow_VBD = np.zeros(CPY + VPY)

        CostNG = Cost_NGStart * np.ones(CPY + VPY)
        CostPower = Cost_PowerStart * np.ones(CPY + VPY)
        CostHydrogen = Cost_HydrogenStart * np.ones(CPY + VPY)
        CostCoolingWater = Cost_CoolingWaterStart * np.ones(CPY + VPY)

        Cost_Byproducts = Cost_ByproductsStart * np.ones(CPY + VPY)

        CostLightNaphtha = CostLightNaphtha * np.ones(CPY + VPY)
        CostHeavyNaphtha = CostHeavyNaphtha * np.ones(CPY + VPY)
        CostLightGasOil = CostLightGasOil * np.ones(CPY + VPY)
        CostHeavyGasOil = CostHeavyGasOil * np.ones(CPY + VPY)
        CostAtmResidue = CostAtmResidue * np.ones(CPY + VPY)
        CostLightVacuumGasOil = CostLightVacuumGasOil * np.ones(CPY + VPY)
        CostHeavyVacuumGasOil = CostHeavyVacuumGasOil * np.ones(CPY + VPY)
        CostVacuumResidues = CostVacuumResidues * np.ones(CPY + VPY)
        CostPropane = CostPropane * np.ones(CPY + VPY)
        CostSulfur = CostSulfur * np.ones(CPY + VPY)
        CostBTX = CostBTX * np.ones(CPY + VPY)

        CY = [CY_1, CY_2, CY_3]

        for i in range(1, CPY + VPY + 1):
            Year[i-1] = i - CPY

        # Years -2 to 0
        Flow_WC[CPY-1] = -FCI*WC
        for i in range(1, CPY + 1):
            Flow_FCI[i-1] = -FCI * equity * CY[i-1]
            if i == 1:
                Flow_LoanPrincipal[i-1] = FCI * (1 - equity) * CY[i-1]
            else:
                Flow_LoanPrincipal[i-1] = (FCI * (1 - equity) * CY[i-1]) + Flow_LoanPrincipal[i - 2]

            Flow_LoanInterest[i-1] = -Flow_LoanPrincipal[i-1] * loan_interest
            Flow_Discount_Factor[i-1] = 1 / ((1 + WACC) ** Year[i-1])
            Flow_TCI_Interest[i-1] = (Flow_FCI[i-1] + Flow_WC[i-1] + Flow_LoanInterest[i-1]) * Flow_Discount_Factor[i-1]


        # Years 1 to 20 Initialization
        # OutputLightNaphtha, OutputHeavyNaphtha, OutputLightGasOil, OutputHeavyGasOil, OutputLightVacuumGasOil, OutputHeavyVacuumGasOil, CostNaphthaStart, CostLightGasOil, CostHeavyGasOil, CostVacuumGasOil
        
        Flow_TotalSales[CPY] = ((OutputLightNaphtha * CostLightNaphtha[CPY]) +
                                (OutputHeavyNaphtha * CostHeavyNaphtha[CPY]) +
                                    (OutputLightGasOil * CostLightGasOil[CPY]) +
                                    (OutputHeavyGasOil * CostHeavyGasOil[CPY]) + 
                                    (OutputAtmResidue  * CostAtmResidue[CPY]) +
                                    (OutputLightVacuumGasOil  * CostLightVacuumGasOil[CPY]) +
                                    (OutputHeavyVacuumGasOil * CostHeavyVacuumGasOil[CPY])  +
                                    (OutputVacuumResidues * CostVacuumResidues[CPY])  +
                                    (OutputKerosene * Cost_MDi) +
                                    (OutputPropane * CostPropane[CPY])  +
                                    (OutputSulfur * CostSulfur[CPY])  +
                                    (OutputBTX * CostBTX[CPY])  +
                                    Cost_Byproducts[CPY] )  * prod_capacity1
        Flow_DOC[CPY] = (-(FCIDet * DOC))
        Flow_NG[CPY] = ((InputNG  + InputNG_SteamProduction + InputSMRFeedGas)* CostNG[CPY]) * prod_capacity1 * -1
        Flow_Power[CPY] = (InputPower * CostPower[CPY]) * prod_capacity1 * -1
        Flow_H2[CPY] = (InputHydrogen * CostHydrogen[CPY]) * prod_capacity1 * -1
        Flow_CoolingWater[CPY] = (InputCoolingWater * CostCoolingWater[CPY]) * prod_capacity1 * -1
        Flow_VOC[CPY] = (VOC + ((InputNG + InputNG_SteamProduction + InputSMRFeedGas) * CostNG[CPY]) + (InputPower * CostPower[CPY]) + (InputHydrogen * CostHydrogen[CPY]) + (InputCoolingWater * CostCoolingWater[CPY])) * prod_capacity1 * -1
        Flow_CrudeInputCost[CPY] = CrudeInputCost * prod_capacity1 * -1
        Flow_EBITDA[CPY] = Flow_TotalSales[CPY] + Flow_DOC[CPY] + Flow_VOC[CPY] + Flow_CrudeInputCost[CPY]


        # Depreciation calculation function
        def calculate_depreciation(FCI, DPY):
            # Initialize variables
            DDB = np.transpose(depgendb(FCI, 0, DPY, 2))  # Assuming depgendb is a function defined elsewhere
            RemainingValue = FCI
            SDB = RemainingValue / DPY
            VDB = np.zeros(DPY)

            # Calculate depreciation using DDB until SDB is greater
            for j in range(DPY):
                if SDB < DDB[j]:
                    VDB[j] = DDB[j]
                    RemainingValue -= VDB[j]
                    SDB = RemainingValue / (DPY - j)
                else:
                    break

            # Complete depreciation using SDB for the remaining periods
            for k in range(j, DPY):
                VDB[k] = SDB

            Flow_VDB = np.zeros(CPY+VPY+1)
            Flow_RemainingValue = np.zeros(CPY+VPY+1)
            Flow_RemainingValue[CPY-1] = FCI

            for i in range(CPY + 1, CPY + DPY + 1):
                Flow_VDB[i - 1] = VDB[i - (CPY + 1)]  # Adjust index to start from CPY + 1

            return Flow_VDB

        Flow_VDB = calculate_depreciation(FCI, DPY)

        Flow_EBIT[CPY] = Flow_EBITDA[CPY] - Flow_VDB[CPY]

        # Annual Loan Payment
        def payper(loan_interest, loan_term, FCI):
            return FCI * (1 - equity) * loan_interest / (1 - (1 + loan_interest) ** (-loan_term))
        
        for i in range(CPY + 1, loan_term + CPY + 1):
            Flow_LoanPayment[i-1] = -payper(loan_interest, loan_term, FCI)

        # Loan payment and taxes
        Flow_LoanInterest[CPY] = -FCI*(1-equity)*loan_interest
        Flow_LoanPrincipal[CPY] = FCI*(1-equity) + Flow_LoanPayment[CPY]-Flow_LoanInterest[CPY]
        Flow_TaxableIncome[CPY] = Flow_EBIT[CPY] + Flow_Losses[CPY] + Flow_LoanInterest[CPY]

        if Flow_TaxableIncome[CPY] > 0:
            Flow_IncomeTax[CPY] = -ITR*Flow_TaxableIncome[CPY]

        Flow_IncomeAfterTax[CPY] = Flow_TaxableIncome[CPY] + Flow_IncomeTax[CPY]

        Flow_NetAnnualIncome[CPY] = Flow_EBITDA[CPY] + Flow_LoanPayment[CPY] + Flow_IncomeTax[CPY]

        Flow_IRR[CPY] = Flow_NetAnnualIncome[CPY]/(FCI*(1+WC))
        Flow_Discount_Factor[CPY] = 1/((1+WACC)**1)
        Flow_APV[CPY] = Flow_NetAnnualIncome[CPY]*Flow_Discount_Factor[CPY]

        Flow_LoanPaymentcomp[CPY] = Flow_LoanPayment[CPY]*Flow_Discount_Factor[CPY]
        Flow_DOCcomp[CPY] = Flow_DOC[CPY]*Flow_Discount_Factor[CPY]
        Flow_VOCcomp[CPY] = Flow_VOC[CPY]*Flow_Discount_Factor[CPY]
        Flow_CrudeInputCostcomp[CPY] = Flow_CrudeInputCost[CPY]*Flow_Discount_Factor[CPY]
        Flow_Taxcomp[CPY] = Flow_IncomeTax[CPY]*Flow_Discount_Factor[CPY]
        Flow_KEROcomp[CPY] = OutputKerosene*Cost_MDi *prod_capacity1*Flow_Discount_Factor[CPY]
        Flow_COPROcomp[CPY] = (OutputSulfur * CostSulfur[CPY] + OutputBTX * CostBTX[CPY] +OutputPropane * CostPropane[CPY] + OutputLightNaphtha*CostLightNaphtha[CPY] + OutputHeavyNaphtha*CostHeavyNaphtha[CPY] + OutputLightGasOil*CostLightGasOil[CPY] + OutputHeavyGasOil*CostHeavyGasOil[CPY] + OutputAtmResidue*CostAtmResidue[CPY] + OutputLightVacuumGasOil*CostLightVacuumGasOil[CPY] + OutputHeavyVacuumGasOil*CostHeavyVacuumGasOil[CPY] + OutputVacuumResidues*CostVacuumResidues[CPY])*prod_capacity1*Flow_Discount_Factor[CPY]

        # Years 2 to 20 Loop
        for i in range(CPY + 2, CPY + VPY + 1):
            Year[i-1] = i - CPY
            #CostPropane[i-1] = max(((0.3762 * CostGasoline[i-1]) + 0.1803), 0)
            #CostLPG[i-1] = max(((0.3762 * CostGasoline[i-1]) + 0.1803), 0)
            CostNG[i-1] = max(((CostNG[i-2] * np.exp(CostNGGrowth))), 0)
            
            if CostNG[i-1] > 13.5:
                CostNG[i-1] = 13.5
            elif CostNG[i-1] < 1.5:
                CostNG[i-1] = 1.5

            CostPower[i-1] = CostPower[i-2] # Fixed cost assumption
            Flow_TotalSales[i-1] = ((OutputLightNaphtha * CostLightNaphtha[i-1]) +
                                    (OutputHeavyNaphtha  * CostHeavyNaphtha[i-1]) +
                                    (OutputLightGasOil * CostLightGasOil[i-1]) +
                                    (OutputHeavyGasOil * CostHeavyGasOil[i-1]) + 
                                    (OutputAtmResidue  * CostAtmResidue[i-1]) +
                                    (OutputLightVacuumGasOil * CostLightVacuumGasOil[i-1] ) +
                                    (OutputHeavyVacuumGasOil * CostHeavyVacuumGasOil[i-1] ) +
                                    (OutputVacuumResidues * CostVacuumResidues[i-1])  +
                                    (OutputKerosene * Cost_MDi) +
                                    (OutputPropane * CostPropane[i-1]) + 
                                    (OutputSulfur * CostSulfur[i-1])  +
                                    (OutputBTX * CostBTX[i-1])  +
                                    Cost_Byproducts[i-1] ) * prod_capacity2 * ((1 + inflation) ** (Year[i-1] - 1))
            Flow_DOC[i-1] = -DOC * ((1 + inflation) ** (Year[i-1] - 1))
            Flow_NG[i-1] = ((InputNG + InputNG_SteamProduction + InputSMRFeedGas)* CostNG[i-1]) * prod_capacity2 * -1 * ((1 + inflation) ** (Year[i-1] - 1))
            Flow_Power[i-1] = (InputPower * CostPower[i-1]) * prod_capacity2 * -1 * ((1 + inflation) ** (Year[i-1] - 1))
            Flow_H2[i-1] = (InputHydrogen * CostHydrogen[i-1]) * prod_capacity2 * -1 * ((1 + inflation) ** (Year[i-1] - 1))
            Flow_CoolingWater[i-1] = (InputCoolingWater * CostCoolingWater[i-1]) * prod_capacity2 * -1 * ((1 + inflation) ** (Year[i-1] - 1))
            Flow_VOC[i-1] = (VOC + ((InputNG + InputNG_SteamProduction + InputSMRFeedGas)* CostNG[i-1]) + (InputPower * CostPower[i-1]) + (InputHydrogen * CostHydrogen[i-1])+ (InputCoolingWater * CostCoolingWater[i-1])) * prod_capacity2 * -1 * ((1 + inflation) ** (Year[i-1] - 1))
            Flow_CrudeInputCost[i-1] = CrudeInputCost * prod_capacity2 * -1 * ((1 + inflation) ** (Year[i-1] - 1))
            Flow_EBITDA[i-1] = Flow_TotalSales[i-1] + Flow_DOC[i-1] + Flow_VOC[i-1] + Flow_CrudeInputCost[i-1]
            Flow_RemainingValue[i-1] = FCI - Flow_VDB[i-1]
            Flow_EBIT[i-1] = Flow_EBITDA[i-1] - Flow_VDB[i-1]
            Flow_LoanInterest[i-1] = -Flow_LoanPrincipal[i-2] * loan_interest
            Flow_LoanPrincipal[i-1] = Flow_LoanPrincipal[i-2] + Flow_LoanPayment[i-1] - Flow_LoanInterest[i-1]
            Flow_TaxableIncome[i-1] = Flow_EBIT[i-1] + Flow_Losses[i-1] + Flow_LoanInterest[i-1]


            if Flow_TaxableIncome[i-2] < 0:
                Flow_Losses[i-1] = Flow_TaxableIncome[i-2]
                Flow_TaxableIncome[i-1] = Flow_EBIT[i-1] + Flow_Losses[i-1] + Flow_LoanInterest[i-1]
            if Flow_TaxableIncome[i-1] > 0:
                Flow_IncomeTax[i-1] = -ITR * Flow_TaxableIncome[i-1]

            Flow_IncomeAfterTax[i-1] = Flow_TaxableIncome[i-1] + Flow_IncomeTax[i-1]

            Flow_NetAnnualIncome[i-1] = Flow_EBITDA[i-1] + Flow_LoanPayment[i-1] + Flow_IncomeTax[i-1]
            Flow_IRR[i-1] = Flow_NetAnnualIncome[i-1]/(FCI + (1+WC))
            Flow_Discount_Factor[i-1] = 1/((1+WACC)**Year[i-1])
            Flow_APV[i-1] = Flow_NetAnnualIncome[i-1]*Flow_Discount_Factor[i-1]

            Flow_LoanPaymentcomp[i-1] = Flow_LoanPayment[i-1]*Flow_Discount_Factor[i-1]
            Flow_DOCcomp[i-1] = Flow_DOC[i-1]*Flow_Discount_Factor[i-1]
            Flow_VOCcomp[i-1] = Flow_VOC[i-1]*Flow_Discount_Factor[i-1]
            Flow_CrudeInputCostcomp[i-1] = Flow_CrudeInputCost[i-1]*Flow_Discount_Factor[i-1]
            Flow_Taxcomp[i-1] = Flow_IncomeTax[i-1]*Flow_Discount_Factor[i-1]
            Flow_KEROcomp[i-1] = ((OutputKerosene * Cost_MDi)) * prod_capacity2* ((1 + inflation)**(Year[i-1]-1)) * Flow_Discount_Factor[i-1]
            Flow_COPROcomp[i-1] = ( OutputBTX * CostBTX[i-1] + OutputSulfur * CostSulfur[i-1] + OutputPropane*CostPropane[i-1] + OutputLightNaphtha*CostLightNaphtha[i-1] + OutputHeavyNaphtha*CostHeavyNaphtha[i-1] + OutputLightGasOil*CostLightGasOil[i-1] + OutputHeavyGasOil*CostHeavyGasOil[i-1] + OutputAtmResidue  * CostAtmResidue[i-1] + OutputLightVacuumGasOil*CostLightVacuumGasOil[i-1] + OutputHeavyVacuumGasOil*CostHeavyVacuumGasOil[i-1] + OutputVacuumResidues*CostVacuumResidues[i-1])* prod_capacity2* ((1 + inflation)**(Year[i-1]-1)) * Flow_Discount_Factor[i-1]

        Flow_WC[i-1] = FCI * WC
        Flow_TCI_Interest[i-1] = (Flow_FCI[i-1] + Flow_WC[i-1]+ Flow_LoanInterest[i-1]) * Flow_Discount_Factor[i-1]

        
        NPV_2 = np.sum(Flow_APV) + np.sum(Flow_TCI_Interest)

        if n>1:
            slope = (NPV_2-NPV_1)/(Cost_MDi-Cost_MDi_old)


        Cost_MDi_old = Cost_MDi
        Cost_MDi = Cost_MDi_old - NPV_2/slope

        # Output the desired table (modify as needed)
        if n > 10:
            print("NPV did not converge after 10 iterations.")
            print("NPV:", NPV_2)
            #print("Cost_MD:", Cost_MDi_2)
            break

        # Create a workbook and select the active worksheet
        wb = Workbook()
        ws = wb.active

        # Define headers
        headers = ["Year", "FCI", "WC", "Loan Principal", "Loan interest", 
                "Discount factor", "TCI Interest", " ", "Total Sales", "DOC", "VOC", "NG Cost", "Power Cost", "H2 Cost", "Cooling Water Cost",
                "Crude Input Cost", "EBITDA", " ", "VDB", "EBIT", " ", "Loan Payment", "Taxable Income", "Income Tax",
                "Income After Tax"," ",  "Net Annual Income", "IRR", "APV", "", "Cost of NG", "Cost of Power", " ",  "Cost of Light Naphtha", "Cost of Heavy Naphtha",
                "Cost of Light Gas Oil",  "Cost of Heavy Gas Oil", "Cost of Light Vacuum Gas Oil",  "Cost of Heavy Vacuum Gas Oil", "Cost of Vacuum Residues", "Cost of Propane", "Cost of Sulfur", "Cost Kerosene",
                "FlowCOPROcomp"]

        # Create a list of lists to store transposed data
        transposed_data = []

        # Add headers as the first row of transposed data
        transposed_data.append(headers)

        for i in range(len(Year)):
            row = [Year[i], Flow_FCI[i], Flow_WC[i], Flow_LoanPrincipal[i], Flow_LoanInterest[i], 
                Flow_Discount_Factor[i], Flow_TCI_Interest[i], "", Flow_TotalSales[i], Flow_DOC[i], Flow_VOC[i], Flow_NG[i], Flow_Power[i], Flow_H2[i], Flow_CoolingWater[i],
                Flow_CrudeInputCost[i], Flow_EBITDA[i], "",Flow_VDB[i], Flow_EBIT[i], "", Flow_LoanPayment[i], Flow_TaxableIncome[i], Flow_IncomeTax[i],
                Flow_IncomeAfterTax[i], "", Flow_NetAnnualIncome[i], Flow_IRR[i], Flow_APV[i], "", CostNG[i], CostPower[i], " ",
                CostLightNaphtha[i], CostHeavyNaphtha[i],
                CostLightGasOil[i], CostHeavyGasOil[i], CostLightVacuumGasOil[i], CostHeavyVacuumGasOil[i], CostVacuumResidues[i], CostPropane[i], CostSulfur[i],
                Cost_MDi, Flow_COPROcomp[i]]
            transposed_data.append(row)

        # Transpose the data (convert rows to columns)
        transposed_data = list(zip(*transposed_data))

        # Add transposed data to the worksheet
        for row in transposed_data:
            ws.append(row)

        # Save the workbook
        # Define the output directory
        output_directory = '/Users/tarahousen/Documents/Aromatics_removal_desulfurization_refining/outputs'
        excel_filename = f'{output_directory}/RefineryDCFROR.xlsx'

        wb.save(excel_filename)

    Cost_Kero = Cost_MDi
    Cost_LightNaphtha = CostLightNaphtha[1]
    Cost_HeavyNaphtha = CostHeavyNaphtha[1]
    Cost_LightGasOil = CostLightGasOil[1]
    Cost_HeavyGasOil = CostHeavyGasOil[1]
    Cost_AtmResidue = CostAtmResidue[1]
    Cost_LightVacuumGasOil = CostLightVacuumGasOil[1]
    Cost_HeavyVacuumGasOil = CostHeavyVacuumGasOil[1]
    Cost_VacuumResidues = CostVacuumResidues[1]
    Cost_Propane = CostPropane[1]

    # List of tuples: (output, cost, name)
    products = [
    (OutputKerosene, Cost_Kero, "Kerosene"),
    (OutputLightNaphtha, Cost_LightNaphtha, "Light Naphtha"),
    (OutputHeavyNaphtha, Cost_HeavyNaphtha, "Heavy Naphtha"),
    (OutputLightGasOil, Cost_LightGasOil, "Light Gas Oil"),
    (OutputHeavyGasOil, Cost_HeavyGasOil, "Heavy Gas Oil"),
    (OutputAtmResidue, Cost_AtmResidue, "Atmospheric Residue"),
    (OutputLightVacuumGasOil, Cost_LightVacuumGasOil, "Light Vacuum Gas Oil"),
    (OutputHeavyVacuumGasOil, Cost_HeavyVacuumGasOil, "Heavy Vacuum Gas Oil"),
    (OutputVacuumResidues, Cost_VacuumResidues, "Vacuum Residue"),
    (OutputPropane, Cost_Propane, "Propane")
    ]

    # Build filtered lists
    Cost_MD = []
    Product_Names_MD = []

    for output, cost, name in products:
        Cost_MD.append(cost)
        Product_Names_MD.append(name)


    # --- MSP COST BREAKDOWN SECTION ---
    # Identify index for first operating year (end of construction)
    start_idx = CPY
    end_idx = CPY + VPY

    # Annual product output (convert to barrels or L/year as appropriate)
    annual_output = OutputKerosene * prod_capacity2 * HPY / OPD  # adjust scaling if your units differ

    # Average values (over operating period, undiscounted)
    avg_DOC = np.mean(-Flow_DOC[start_idx:end_idx])
    avg_NG_Cost = np.mean(-Flow_NG[start_idx:end_idx])
    avg_Power_Cost = np.mean(-Flow_Power[start_idx:end_idx])
    avg_H2_Cost = np.mean(-Flow_H2[start_idx:end_idx])
    avg_CoolingWater_Cost = np.mean(-Flow_CoolingWater[start_idx:end_idx])

    avg_CrudeInputCost = np.mean(-Flow_CrudeInputCost[start_idx:end_idx])
    avg_LoanPayment = np.mean(-Flow_LoanPayment[start_idx:end_idx])
    avg_Tax = np.mean(-Flow_IncomeTax[start_idx:end_idx])

    # --- Breakdown energy vs feedstock inside VOC ---
    # The VOC line includes both process utilities (NG, Power, H2, etc.) and non-energy variable costs.
    # We can reconstruct the energy portion from your cost multipliers:
    avg_energy_cost = np.mean((InputNG + InputNG_SteamProduction + InputSMRFeedGas) * Cost_NGStart + 
                              InputPower * Cost_PowerStart + 
                              InputHydrogen * Cost_HydrogenStart + 
                              InputCoolingWater * Cost_CoolingWaterStart)
    avg_feedstock_cost = avg_CrudeInputCost - avg_energy_cost

    # Capital recovery (equity-financed portion)
    CRF = (loan_interest * (1 + loan_interest) ** loan_term) / ((1 + loan_interest) ** loan_term - 1)
    annual_capital_recovery = FCI * equity * CRF

    # Normalize each component to $/bbl of kerosene
    MSP_components = {
        "DOC (Fixed Opex)": avg_DOC / annual_output,
        "NG Cost": avg_NG_Cost / annual_output,
        "Power Cost": avg_Power_Cost / annual_output,
        "H2 Cost": avg_H2_Cost / annual_output,
        "Cooling Water Cost": avg_CoolingWater_Cost / annual_output,
        "Loan Repayment": avg_LoanPayment / annual_output,
        "Capital Recovery (Equity)": annual_capital_recovery / annual_output,
        "Taxes": avg_Tax / annual_output,
    }

    # Compute total and percentage contributions
    MSP_total = sum(MSP_components.values())
    MSP_breakdown = {k: {"USD/bbl": v, "Share (%)": 100 * v / MSP_total} for k, v in MSP_components.items()}

    # Convert to readable DataFrame (optional)
    import pandas as pd
    MSP_breakdown_df = pd.DataFrame(MSP_breakdown).T
    print("\n--- MSP Cost Breakdown (USD/bbl of Kerosene) ---")
    print(MSP_breakdown_df.round(3))

    # Save optional breakdown output
    output_directory = '/Users/tarahousen/Documents/Aromatics_removal_desulfurization_refining/outputs'
    MSP_breakdown_df.to_excel(f"{output_directory}/MSP_breakdown.xlsx")

    return Cost_MD, Product_Names_MD



