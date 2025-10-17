import numpy as np
from functions.labor_hours import labor_hours_interp


# Extract and calculate additional capital costs based on cost factors
def calculate_additional_capital_costs(total_equipment_cost, financial_assumptions, solvent_investment_cost):
    additional_costs = {}
    working_capital_cost = 0
    total_direct_cost = 0
    remaning_cost_factors = 0

    for cost_factor, factor in financial_assumptions['direct_cost_factors'].items():
        additional_cost = total_equipment_cost * factor
        additional_costs[cost_factor] = additional_cost
    
        total_direct_cost += additional_cost
    total_direct_cost = total_direct_cost + total_equipment_cost + solvent_investment_cost

    additional_costs['engineering_and_supervision'] = total_direct_cost * financial_assumptions['indirect_cost_factors']['engineering_and_supervision']

    for cost_factor, factor in financial_assumptions['indirect_cost_factors'].items():
        if cost_factor != 'engineering_and_supervision':
            remaning_cost_factors += factor

    FCI = (total_direct_cost + additional_costs['engineering_and_supervision'])/ (1 - remaning_cost_factors)

    for cost_factor, factor in financial_assumptions['indirect_cost_factors'].items():
        if cost_factor != 'engineering_and_supervision':
            additional_cost = FCI * factor
            additional_costs[cost_factor] = additional_cost

    cost_of_land = additional_costs.get('land', 0)
    cost_of_buildings = additional_costs.get('buildings', 0)

    fixed_capital_investment = FCI
    total_capital_investment = fixed_capital_investment / (1 - financial_assumptions['working_capital']['working_capital'])
    working_capital_cost = total_capital_investment * financial_assumptions['working_capital']['working_capital']

    return additional_costs, working_capital_cost, cost_of_land, cost_of_buildings, total_capital_investment, fixed_capital_investment

def calculate_additional_operating_costs_gary_handwerk(fixed_capital_investment, financial_assumptions_data, non_feedstock_raw_mat_cost, total_energy_cost, annual_crude_input_cost, plant_capacity_kg_day, num_process_steps):
    
    plant_investment = fixed_capital_investment * (1 + financial_assumptions_data['financial_assumptions']['inflation'])**3
    print(f'plant_investment: {plant_investment}')

    direct_operating_costs = {}
    variable_operating_costs = {}

    utilities_cost = total_energy_cost 

    variable_operating_costs['feedstock'] = annual_crude_input_cost
    variable_operating_costs['raw materials'] = non_feedstock_raw_mat_cost
    variable_operating_costs['utilities'] = total_energy_cost

    total_variable_oper_cost = sum(variable_operating_costs.values()) 


    # Labor cost calculation
    labor_price = financial_assumptions_data['financial_assumptions']['cost_labor']
    labor_hours = labor_hours_interp(plant_capacity_kg_day) * num_process_steps

    direct_operating_cost_factors = financial_assumptions_data['direct_operating_cost_factors']

    direct_operating_costs = {
        'operating_labor': labor_price * labor_hours
    }

    direct_operating_costs['maintanence_costs'] = 0.055 * plant_investment 
    direct_operating_costs['taxes_costs'] = 0.01 * plant_investment
    direct_operating_costs['insurance_costs'] = 0.005 * plant_investment
    direct_operating_costs['misc_supplies'] = 0.0015 * plant_investment

    total_fixed_oper_cost = sum(direct_operating_costs.values())
    total_operating_costs = total_fixed_oper_cost + total_variable_oper_cost

    return non_feedstock_raw_mat_cost, utilities_cost, total_fixed_oper_cost, total_variable_oper_cost, total_operating_costs, direct_operating_costs, variable_operating_costs


def calculate_additional_operating_costs(
    non_feedstock_raw_mat_cost,
    total_energy_cost,
    financial_assumptions_data,
    fixed_capital_investment,
    annual_crude_input_cost,
    plant_capacity_kg_day,
    num_process_steps
):
    import numpy as np  # needed for simple min/max handling

    def get_range(val):
        """Ensure all values are returned as [min, max]."""
        if isinstance(val, (list, tuple, np.ndarray)):
            return [float(val[0]), float(val[-1])]
        else:
            return [float(val), float(val)]

    direct_operating_costs = {}
    variable_operating_costs = {}

    utilities_cost = total_energy_cost
    variable_operating_costs['feedstock'] = annual_crude_input_cost
    variable_operating_costs['raw materials'] = non_feedstock_raw_mat_cost
    variable_operating_costs['utilities'] = total_energy_cost

    total_variable_oper_cost = sum(variable_operating_costs.values())

    # Labor cost calculation
    labor_price = financial_assumptions_data['financial_assumptions']['cost_labor']
    labor_hours = labor_hours_interp(plant_capacity_kg_day) * num_process_steps
    direct_operating_cost_factors = financial_assumptions_data['direct_operating_cost_factors']

    # All fixed cost factors as [min, max]
    insurance = get_range(direct_operating_cost_factors['insurance'])
    maintenance = get_range(direct_operating_cost_factors['maintenance_and_repairs'])
    operating_supplies = get_range(direct_operating_cost_factors['operating_supplies'])
    local_taxes = get_range(direct_operating_cost_factors['local_taxes'])
    plant_overhead = get_range(direct_operating_cost_factors['plant_overhead'])
    admin_costs = get_range(direct_operating_cost_factors['admin_costs'])
    research = get_range(direct_operating_cost_factors['research_and_development'])

    operating_labor = [labor_price * labor_hours] * 2  # same min/max

    # Calculate fixed O&M ranges
    direct_operating_costs['operating_labor'] = operating_labor
    direct_operating_costs['insurance'] = [v * fixed_capital_investment for v in insurance]
    direct_operating_costs['maintenance_and_repairs'] = [v * fixed_capital_investment for v in maintenance]
    direct_operating_costs['operating_supplies'] = [v * fixed_capital_investment for v in operating_supplies]
    direct_operating_costs['local_taxes'] = [v * fixed_capital_investment for v in local_taxes]

    # Overhead and admin depend on other components
    base_for_overhead = [
        operating_labor[0] + direct_operating_costs['maintenance_and_repairs'][0],
        operating_labor[1] + direct_operating_costs['maintenance_and_repairs'][1]
    ]
    direct_operating_costs['plant_overhead'] = [plant_overhead[i] * base_for_overhead[i] for i in (0, 1)]
    direct_operating_costs['admin_costs'] = [admin_costs[i] * base_for_overhead[i] for i in (0, 1)]

    # Total fixed O&M (min, max)
    total_fixed_oper_cost = [
        sum(costs[0] for costs in direct_operating_costs.values()),
        sum(costs[1] for costs in direct_operating_costs.values())
    ]

    # Compute total product cost range
    total_product_cost = [
        (total_fixed_oper_cost[0] + total_variable_oper_cost) / (1 - research[0]),
        (total_fixed_oper_cost[1] + total_variable_oper_cost) / (1 - research[1])
    ]

    # Research & development cost (min, max)
    direct_operating_costs['research_and_developement'] = [
        research[i] * total_product_cost[i] for i in (0, 1)
    ]

    # Update totals including R&D
    total_fixed_oper_cost = [
        sum(costs[0] for costs in direct_operating_costs.values()),
        sum(costs[1] for costs in direct_operating_costs.values())
    ]

    total_operating_costs = [
        total_fixed_oper_cost[0] + total_variable_oper_cost,
        total_fixed_oper_cost[1] + total_variable_oper_cost
    ]

    return (
        total_product_cost,
        non_feedstock_raw_mat_cost,
        utilities_cost,
        total_fixed_oper_cost,
        total_variable_oper_cost,
        total_operating_costs,
        direct_operating_costs,
        variable_operating_costs
    )
