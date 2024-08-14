import pandas as pd
from PartialCointegration import *
from PCISimulation import *
import xarray as xr
from datetime import datetime
import matplotlib.pyplot as plt
import os

# Get todays date
today_date = datetime.today().strftime('%Y-%m-%d')

def f_generate_data(params, var_values, sample_sizes, nb_sims):
    """
    For each simulation and value of a varying parameter, generate a pair of partially cointegrated time-series of length = max(sample_sizes).

    Args:
        params (list): List of parameters for the data generation. The list should contain:
                       [alpha, beta, rho, sigma_m, sigma_r, sigma_x]
                       If one of the parameters is None, it indicates that this parameter will vary.
        var_values (list): List of values for the varying parameter.
        sample_sizes (list): List of sample sizes for which to generate data.
        nb_sims (int): Number of simulations to run for each value of the varying parameter.

    Returns:
        xarray.Dataset: A dataset containing the (nb_sims x var_values) pairs (X1, X2) of simulated time-series of length = max(sample_sizes).
    """

    max_sample_size = max(sample_sizes)
    alpha, beta, rho, sigma_m, sigma_r, sigma_x = params

    # Set lists for X1 & X2 data
    X1_data = []
    X2_data = []

    for var_value in var_values:
        # Set lists for simulations
        X1_simulations = []
        X2_simulations = []

        for _ in range(nb_sims):

            current_params = [alpha, beta, rho, sigma_m, sigma_r, sigma_x]
            current_params[current_params.index(None)] = var_value # Set varying parameter to current value

            data_generator = pci_generator(max_sample_size, *current_params) # Define generator object with max_sample_size length
            X1, X2, _, _, _ = data_generator.f_generate_pci() # Generate pair PCI time-series: X1, X2

            X1_simulations.append(X1)
            X2_simulations.append(X2)

        X1_data.append(X1_simulations)
        X2_data.append(X2_simulations)

    # Create the dataset containing the time-series data
    dataset = xr.Dataset({
        'X1': (['var_param', 'simulation', 'time'], X1_data),
        'X2': (['var_param', 'simulation', 'time'], X2_data)
    }, coords={'var_param': var_values, 
               'simulation': np.arange(nb_sims), 
               'time': np.arange(max_sample_size)})

    return dataset

def f_compute_estimates(dataset, sample_sizes):
    """
    Compute estimates of PCI model parameters for all (var_param x simulation x sample_sizes) pairs of time-series data X1, X2.

    Args:
        dataset (xarray.Dataset): Dataset containing the simulated data for X1 and X2.
        sample_sizes (list): List of sample sizes for which to compute estimates.

    Returns:
        xarray.Dataset: A dataset containing the estimated parameters for all (var_param x simulation x sample_sizes) pairs.
    """
    estimates_data = []

    for sample_size in sample_sizes:
        params_list = []

        for var_value in dataset['var_param'].values:
            var_list = []

            for sim in dataset['simulation'].values:
                X1 = dataset['X1'].sel(var_param=var_value, simulation=sim).values[:sample_size] # Slice the dataset X1
                X2 = dataset['X2'].sel(var_param=var_value, simulation=sim).values[:sample_size] # Slice the dataset X2
                pci_model = partial_cointegration(X1, X2) # Create PCI object
                _, beta, rho, sigma_M, sigma_R, _, _, lrt = pci_model.fit_pci(LRT_mode=True) # Estimate parameters of PCI model
                var_list.append([beta, rho, sigma_M, sigma_R, lrt])

            params_list.append(var_list)

        params_array = np.array(params_list)

        # Create dataset containing estimated parameters for sample_size
        estimates_data.append(xr.DataArray(data=params_array,
                                           dims=['var_param', 'simulation', 'estimates'],
                                           coords={'var_param': dataset['var_param'].values,
                                                   'simulation': dataset['simulation'].values,
                                                   'estimates': ['beta', 'rho', 'sigma_m', 'sigma_r', 'lrt']},
                                           name=f'estimates_sample_size_{sample_size}'))

    # Create dataset containing estimated parameters fore all sample_sizes
    final_dataset = xr.merge(estimates_data)

    return final_dataset

def f_compute_means(dataset):
    """
    Compute the mean of the dataset across all simulations.

    Args:
        dataset (xarray.Dataset): A dataset containing the mean for all simulations.

    Returns:
        xarray.Dataset: A dataset containing the mean values across all simulations.
    """
    return dataset.mean(dim="simulation")

def f_compute_mse(dataset, params:list):
    """
    Compute the mean squared error (MSE) between the estimated parameters and the actual parameters across all simulations.

    Args:
        dataset (xarray.Dataset): A dataset containing the estimated parameters for all (var_param x simulation x sameple_sizes) pairs.
        params (list): A list of actual parameter values [beta, rho, sigma_m, sigma_r],
                       with None indicating the varying parameter.

    Returns:
        xarray.Dataset: A dataset containing the mean squared error for each parameter.
    """
    mse_list = []

    # Flags to indicate which parameter is varying
    sigma_m_i = False
    rho_i = False

    # Determine which parameter is varying based on the position of None in the params list
    if params[3] is None:
        sigma_m_i = True
    elif params[2] is None:
        rho_i = True

    # Iterate over each value of the varying parameter
    for var in dataset["var_param"]:
        # Update the varying parameter in the actual parameters list
        if sigma_m_i:
            params[3] = float(var)
        elif rho_i:
            params[2] = float(var)

        # Compute the squared error between the estimated parameters and the actual parameters
        se = (params[1:] - dataset.sel(var_param=var)) ** 2

        # Compute the mean of the squared error across all simulations
        mse = se.mean(dim=["simulation"])

        # Append the MSE to the list
        mse_list.append(mse)

    # Concatenate the MSEs along the var_param dimension
    ds = xr.concat(mse_list, dim="var_param")

    # Drop the 'lrt' estimate if present
    ds = ds.drop("lrt", dim="estimates")

    return ds

def f_compute_critical_values_lrt(quantiles, sample_sizes, nb_sims):
    """
    Generate data that is purely a random walk, compute estimates for likelihood ratio test, 
    and calculate critical values based on sample quantiles. This is known as bootstrapping in statistics.

    Args:
        quantiles (list): List of quantiles to compute the critical values for.
        sample_sizes (list): List of sample sizes to generate the data for.
        nb_sims (int): Number of simulations to run.

    Returns:
        xarray.Dataset: Dataset containing the critical values for each quantile and sample size.
    """
    # Parameters for generating the data - with None in place for the varying parameter
    params_cr = [0, 1, 0, None, 1, 0.0236]
    var_values_cr = [0]  # Set sigma_m to

    # Generate data with the specified parameters, sample sizes, and number of simulations
    dataset_rw = f_generate_data(params_cr, var_values_cr, sample_sizes, nb_sims)

    # Compute estimates based on the generated data
    estimates_rw = f_compute_estimates(dataset_rw, sample_sizes).sel(estimates = "lrt")

    # Calculate critical values based on the provided quantiles for each sample size
    critical_values = estimates_rw.quantile(quantiles, dim="simulation")

    return critical_values

def f_compute_powers_lrt(estimate_dataset, sample_sizes, quantiles):
    """
    Compute the power of the likelihood ratio test for various quantiles and sample sizes. 
    *Def.: Power: percentage of the time that the test correctly rejects the null hypothesis when the null hypothesis is indeed false.
    The null hypothesis being: the series is a pure random walk
    
    Args:
        estimate_dataset (xarray.Dataset): Dataset containing the estimates from Monte Carlo simulations.
        sample_sizes (list): List of sample sizes used in the simulations.
        quantiles (list): List of quantiles for which to calculate the power.

    Returns:
        xarray.Dataset: Dataset containing the power values for each quantile and sample size.
    """
    # Find the critical values under the null hypothesis for the given quantiles and sample sizes
    critical_values = f_compute_critical_values_lrt(quantiles, sample_sizes, nb_sims=1000)

    # Determine the number of estimates that are below the critical values for each sample size
    # This indicates the number of times the null hypothesis would be rejected
    estimates_below_critical = (estimate_dataset.sel(estimates = "lrt") < critical_values.sel(var_param=0)).sum(dim='simulation')

    total_simulations = len(estimate_dataset['simulation']) # number of simulations

    
    powers = estimates_below_critical / total_simulations # power (% of values below critical value)

    # Return the dataset containing the power values for each quantile and sample size
    return powers


def prepare_output_path(filename):
    """
    Prepares the full path for a file in the Outputs directory, creating the directory if necessary.

    Args:
        filename (str): The name of the file to save in the Outputs directory.

    Returns:
        str: The full path to the file within the Outputs directory.
    """
    # Get the directory of the current script
    current_script_dir = os.path.dirname(os.path.abspath(__file__))

    # Define the path to the Outputs directory relative to the current script
    outputs_dir = os.path.join(current_script_dir, '..', 'Outputs')

    # Create the Outputs directory if it does not exist
    if not os.path.exists(outputs_dir):
        os.makedirs(outputs_dir)

    # Specify the path to the Excel file within the Outputs directory
    return os.path.join(outputs_dir, filename)

def f_create_estimates_table(mean_ds, mse_ds):
    """
    Creates a table with mean estimates and mean squared errors (MSE) and saves it to an Excel file.

    Args:
        mean_ds (xarray.Dataset): Dataset containing the mean estimates for each parameter.
        mse_ds (xarray.Dataset): Dataset containing the mean squared errors for each parameter.

    Returns:
        pandas.DataFrame: DataFrame containing the combined mean estimates and MSEs.
    """

    # Drop lrt estimate
    mean_ds = mean_ds.drop("lrt", dim="estimates")

    # Convert each dataset to a pandas DataFrame and reset the index
    df1 = mean_ds.to_dataframe().reset_index()
    df2 = mse_ds.to_dataframe().reset_index()

    # Rename columns to include 'mean_' and 'mse_' prefixes
    df1 = df1.rename(columns={col: 'mean_' + col.split('_')[-1] for col in df1.columns if 'estimates_sample_size' in col})
    df2 = df2.rename(columns={col: 'mse_' + col.split('_')[-1] for col in df2.columns if 'estimates_sample_size' in col})

    # Concatenate the DataFrames horizontally (side by side)
    estimates = pd.concat([df1, df2], axis=1)

    # Todays date
    global today_date

    # Excel output file path
    excel_file_path = prepare_output_path(f'estimates_table_{today_date}.xlsx')

    # Save the estimates DataFrame to excel
    estimates.to_excel(excel_file_path)

    # Return the estimates DataFrame
    return estimates

def f_create_power_table(powers):
    """
    Creates a table with power values for each quantile and sample size and saves it to an Excel file.

    Args:
        powers (xarray.Dataset): Dataset containing the power values for each quantile and sample size.

    Returns:
        pandas.DataFrame: DataFrame containing the power values.
    """

    # Convert the power dataset to a pandas DataFrame and reset the index
    powers = powers.to_dataframe().reset_index()

    # Todays date
    global today_date

    # Excel output file path
    excel_file_path = prepare_output_path(f'powers_table_{today_date}.xlsx')

    # Save the power DataFrame to excel
    powers.to_excel(excel_file_path)

    # Return the power DataFrame
    return powers


def f_create_critical_values_table(critical_values):
    """
    Creates a table with critical values for the likelihood ratio test (LRT) and saves it to an Excel file.

    Args:
        critical_values (xarray.Dataset): Dataset containing the critical values for the LRT.

    Returns:
        pandas.DataFrame: DataFrame containing the critical values.
    """
    # Select only the 'lrt' estimates from the dataset and convert it to a pandas DataFrame
    critical_values = critical_values.sel(estimates="lrt").to_dataframe().reset_index()

    # Todays date
    global today_date

    # Excel output file path
    excel_file_path = prepare_output_path(f'critical_values_table_{today_date}.xlsx')

    # Save the critival values DataFrame to excel
    critical_values.to_excel(excel_file_path)

    # Return the critical values DataFrame
    return critical_values

if __name__ == "__main__":
    #        [alpha, beta, rho, sigma_m, sigma_r, sigma_x]
    params = [0, 1, 0.9, None, 1, 0.0236] 
    var_param = [1]
    sample_sizes = [1000] # Devrait etre 10,000
    nb_sims = 10000 # Devrait etre 10,000

    data = f_generate_data(params, var_param, sample_sizes, nb_sims)
    estimates = f_compute_estimates(data, sample_sizes=sample_sizes)

    # ok 
    
















