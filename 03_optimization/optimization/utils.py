# file with utility functions for optimization tasks
import pandas as pd


def map_costs_to_timestamps(costs: dict) -> pd.DataFrame:
    """ Maps costs from config to timestamp-costs tuples. Assume TOU tariffs for now, i.e., all days follow the same
    cost pattern (e.g. 00-08: A€, 08-12: B€, 12-16: C€, 16-00: X€). 

    Args:
        costs (dict): A dictionary containing the costs for buying and selling energy. The structure should be:
            {
                "c_buy": {
                    "default": float,  # Default cost for buying energy
                    "extra": {  # Extra costs for specific hours
                        "hour X": float,  # Cost for buying energy at hour X
                    }
                },
                "c_sell": {
                    "default": float,  # Default cost for selling energy
                    "extra": {  # Extra costs for specific hours
                        "hour Y": float,  # Cost for selling energy at hour Y
                    }
                }
            }

    """

    # TODO: Need to implement cost mapping for sub-hourly timestamps. Do I?

    # create a df with 24 entries. The index is the hour of the day, the columns are the costs (c_buy, c_sell)
    df = pd.DataFrame(index=range(24), columns=costs.keys())
    df.index.name = 'hour_of_day'


    # fill the df with the default costs
    for cost_type, cost_values in costs.items():
        if 'default' in cost_values:
            df[cost_type] = cost_values['default']
        
    # fill the df with the extra costs
    for cost_type, cost_values in costs.items():
        if 'extra' in cost_values:
            for hour, value in cost_values['extra'].items():
                hour_index = int(hour.split(' ')[-1])  # Extract the hour from "hour X"
                df.at[hour_index, cost_type] = value

    
    return df

