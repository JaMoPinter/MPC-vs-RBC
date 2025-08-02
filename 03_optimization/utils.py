# file with utility functions 
import pandas as pd
import os
import pyomo.environ as pyo
from math import pi
import numpy as np


def cdf_formula(name):
    ''' Returns the CDF formula for the specified distribution. Currently, the following distributions are supported:
    - normal: normal distribution
    - sum2gaussian: sum of two gaussian distributions '''

    def cdf_normal(x, mu, sig):
        ''' Gaussian CDF computation via Abramowitz-Stegun approximation without if-statements (Pyomo cannot use If-Statements). '''

        z = (x - mu) / sig  # standardize the normal distribution

        epsilon = 1e-6  # to avoid division by zero => Needed for sign function approximation
        sign_z = z / (pyo.sqrt(z**2 + epsilon))  # sign function approximation

        z_abs = z * sign_z  # absolute value of z

        d1 = 0.0498673470  # Coefficients for the Abramowitz-Stegun approximation
        d2 = 0.0211410061
        d3 = 0.0032776263
        d4 = 0.0000380036
        d5 = 0.0000488906
        d6 = 0.0000053830 

        t = 1 + d1 * z_abs + d2 * z_abs**2 + d3 * z_abs**3 + d4 * z_abs**4 + d5 * z_abs**5 + d6 * z_abs**6 

        return 0.5 + 0.5 * sign_z * (1 - t**(-16))

     
    
    if name == 'normal':
        return cdf_normal 

    elif name == 'sum-2-logistic-distributions':
        return lambda x, w1, w2, w3, w4, w5, w6: w1 / (1 + pyo.exp(-w2 *(x - w3))) + w4 / (1 + pyo.exp(-w5 *(x - w6)))

    elif name == 'sum2gaussian':
        return lambda x, w1, mu1, sig1, w2, mu2, sig2: w1 * cdf_normal(x, mu1, sig1) + w2 * cdf_normal(x, mu2, sig2)
    
    else:
        raise ValueError(f'CDF formula {name} not recognized')
    



def pdf_formula(name):
    ''' Returns the PDF formula for the specified distribution. '''
    if name == 'normal':
        return lambda x, mu, sig: 1 / (sig * pyo.sqrt(2 * pi)) * pyo.exp(-0.5 * ((x - mu) / sig)**2)
    
    elif name == 'sum-2-logistic-distributions':
        return lambda x, w1, w2, w3, w4, w5, w6: w1 * w2 * pyo.exp(-w2 * (x - w3)) / (1 + pyo.exp(-w2 * (x - w3)))**2 + w4 * w5 * pyo.exp(-w5 * (x - w6)) / (1 + pyo.exp(-w5 * (x - w6)))**2
    
    elif name == 'sum2gaussian':
        return lambda x, w1, mu1, sig1, w2, mu2, sig2: w1 / (sig1 * pyo.sqrt(2 * pi)) * pyo.exp(-0.5 * ((x - mu1) / sig1)**2) + w2 / (sig2 * pyo.sqrt(2 * pi)) * pyo.exp(-0.5 * ((x - mu2) / sig2)**2) 

    else:
        raise ValueError(f'PDF formula {name} not recognized')






def simpsons_rule(
    lb: float,
    ub: float,
    n: int,
    pdf: callable,
    weights: list,
    offset: float = 0.0
) -> float:
    """
    Numerically integrates x * pdf(x + offset, *weights) from lb to ub using Simpson's rule.

    Args:
        lb (float): lower bound of the integration
        ub (float): upper bound of the integration
        n (int): number of intervals (must be even)
        pdf (function): pdf(x, *weights)
        weights (list): weights for the pdf
        offset (float): shift the x-axis for the pdf

    Returns:
        float: approximated integral value
    """

    assert n % 2 == 0, "n must be even for Simpson's rule"

    h = (ub - lb) / n  # step size
    x = [lb + i * h for i in range(n+1)]
    integrand = lambda x, w: x * pdf((x + offset), *w)
    y = [integrand(xi, weights) for xi in x]

    approximated_integral = h / 3 * (y[0] + y[-1] + 4 * sum(y[1:-1:2]) + 2 * sum(y[2:-1:2]))
    return approximated_integral





def load_chunks(path, t_start, t_end, filter_col,parse_dates=None, chunksize=100_000, usecols=None):
    """ Load chunks of a CSV file and filter them by a date range. 
    
    Args:
        path (str): Path to the CSV file.
        t_start (pd.Timestamp): Start of the time range - time_fc_creation.
        t_end (pd.Timestamp): End of the time range - time_fc_creation.
        filter_col (str): Column to filter the data according to t_start and t_end.
        parse_dates (list): List of columns to parse as dates.
        chunksize (int): Number of rows per chunk to read from the CSV file.
        usecols (list, optional): List of columns to read from the CSV file. If None, all columns are read."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    
    chunks = pd.read_csv(path, chunksize=chunksize, usecols=usecols, parse_dates=parse_dates)
    dfs = []
    for chunk in chunks:
        mask = (chunk[filter_col] >= t_start) & (chunk[filter_col] <= t_end)
        dfs.append(chunk[mask])
    df = pd.concat(dfs, ignore_index=True)
    df.set_index(parse_dates, inplace=True)
    return df


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

