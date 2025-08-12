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
    

def cdf_formula_numpy(name):
    ''' Returns the CDF formula for the specified distribution. 

    This is needed becasue during optimization, pyomo cannot utilize other libraries (like numpy). However, for the 
    plotting, pyomo cannot be used. Therefore, this function allows a computation of the CDF using numpy. '''

    def cdf_normal(x, mu, sig):
        ''' Abramowitz-Stegun approximation of the normal CDF '''
        z = (x - mu) / sig

        def phi(z):
            if z < 0: 
                return 1 - phi(-z)

            d1 = 0.0498673470
            d2 = 0.0211410061
            d3 = 0.0032776263
            d4 = 0.0000380036
            d5 = 0.0000488906
            d6 = 0.0000053830 

            t = 1 + d1 * z + d2 * z**2 + d3 * z**3 + d4 * z**4 + d5 * z**5 + d6 * z**6

            return 1 - 0.5 * (t ** -16)
            
        return phi(z)  

    
    if name == 'normal':  # Abramowitz-Stegun approximation of the normal CDF
        return cdf_normal
    
    elif name == 'sum2gaussian':
        return lambda x, w1, mu1, sig1, w2, mu2, sig2: w1 * cdf_normal(x, mu1, sig1) + w2 * cdf_normal(x, mu2, sig2)
    
    elif name == 'sum-2-logistic-distributions':
        return lambda x, w1, w2, w3, w4, w5, w6: w1 / (1 + np.exp(-w2 *(x - w3))) + w4 / (1 + np.exp(-w5 *(x - w6)))

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



def pdf_formula_numpy(name):
    ''' Returns the PDF formula for the specified distribution. '''
    if name == 'normal':
        return lambda x, mu, sig: 1 / (sig * np.sqrt(2 * pi)) * np.exp(-0.5 * ((x - mu) / sig)**2)
    
    elif name == 'sum-2-logistic-distributions':
        return lambda x, w1, w2, w3, w4, w5, w6: w1 * w2 * np.exp(-w2 * (x - w3)) / (1 + np.exp(-w2 * (x - w3)))**2 + w4 * w5 * np.exp(-w5 * (x - w6)) / (1 + np.exp(-w5 * (x - w6)))**2
    
    elif name == 'sum2gaussian':
        return lambda x, w1, mu1, sig1, w2, mu2, sig2: w1 / (sig1 * np.sqrt(2 * pi)) * np.exp(-0.5 * ((x - mu1) / sig1)**2) + w2 / (sig2 * np.sqrt(2 * pi)) * np.exp(-0.5 * ((x - mu2) / sig2)**2) 

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



def dynamic_bounds(bounds, pdf_formula, weights):
    """
    Compute dynamic bounds to minimize errors when integrating the pdfs. Each time step has a different pdf shape
    and therefore different bounds are used for the integration bounds.

    Args:
    bounds (tuple): lower and upper integration bounds (static, should be conservative)
    pdf_formula (callable): the PDF function to be integrated
    weights (pd.DataFrame): weights for the PDF formula at each time step

    Returns:
        (low_bounds, high_bounds) as pd.Series aligned to weights.index
    '''
    """

    x = np.linspace(bounds[0], bounds[1], 10000)
    idx = weights.index

    low = pd.Series(np.nan, index=idx, dtype=float)
    high = pd.Series(np.nan, index=idx, dtype=float)

    for t, row in weights.iterrows():
        params = row.values
        # Evaluate PDF on the full grid in a numerically tolerant way
        with np.errstate(over='ignore', under='ignore', invalid='ignore'):
            y = pdf_formula(x, *params)

        # Replace non-finite with 0 (so they don't count as above threshold)
        y = np.where(np.isfinite(y), y, 0.0)

        mask = y > 1e-8
        if not np.any(mask):
            # Fallback: keep the original static bounds
            low.loc[t] = bounds[0]
            high.loc[t] = bounds[1]
            continue

        # First and last indices above threshold
        i0 = np.argmax(mask)  # leftmost True
        i1 = len(mask) - 1 - np.argmax(mask[::-1])  # rightmost True

        # Take the values before surpassing the threshold 1e-8
        lb_idx = max(i0 - 1, 0)
        ub_idx = min(i1 + 1, len(x) - 1)

        low.loc[t] = x[lb_idx]
        high.loc[t] = x[ub_idx]

    return low, high
