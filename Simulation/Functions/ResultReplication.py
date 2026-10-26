import pandas as pd
from PartialCointegration import partial_cointegration
from partial_cointegration import FitError
from partial_cointegration.io import estimator_fingerprint
import numpy as np
import json
from PCISimulation import pci_generator
import xarray as xr
from datetime import datetime
import matplotlib.pyplot as plt
import os

# Get todays date
today_date = datetime.today().strftime("%Y-%m-%d")


def f_generate_data(params, var_values, sample_sizes, nb_sims, *, seed=0):
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

    if len(params) != 6 or sum(value is None for value in params) != 1:
        raise ValueError("Specify exactly one varying parameter in six named positions")
    if nb_sims < 1 or not sample_sizes or min(sample_sizes) < 4:
        raise ValueError("Invalid simulation sizes")
    max_sample_size = max(sample_sizes)
    alpha, beta, rho, sigma_m, sigma_r, sigma_x = params

    # Set lists for X1 & X2 data
    X1_data = []
    X2_data = []

    for cell, var_value in enumerate(var_values):
        # Set lists for simulations
        X1_simulations = []
        X2_simulations = []

        for rep in range(nb_sims):

            current_params = [alpha, beta, rho, sigma_m, sigma_r, sigma_x]
            current_params[current_params.index(None)] = (
                var_value  # Set varying parameter to current value
            )

            data_generator = pci_generator(
                max_sample_size,
                *current_params,
                seed=int(
                    np.random.SeedSequence([seed, cell, rep]).generate_state(1)[0]
                ),
            )  # Define generator object with max_sample_size length
            X1, X2, _, _, _ = (
                data_generator.f_generate_pci()
            )  # Generate pair PCI time-series: X1, X2

            X1_simulations.append(X1)
            X2_simulations.append(X2)

        X1_data.append(X1_simulations)
        X2_data.append(X2_simulations)

    # Create the dataset containing the time-series data
    dataset = xr.Dataset(
        {
            "X1": (["var_param", "simulation", "time"], X1_data),
            "X2": (["var_param", "simulation", "time"], X2_data),
        },
        coords={
            "var_param": var_values,
            "simulation": np.arange(nb_sims),
            "time": np.arange(max_sample_size),
        },
    )

    dataset.attrs.update(
        {"design": "geometric", "initialization": "known", "sigma_x": params[-1]}
    )
    return dataset


def f_compute_estimates(dataset, sample_sizes, *, seed=0, mode="exact"):
    """
    Compute estimates of PCI model parameters for all (var_param x simulation x sample_sizes) pairs of time-series data X1, X2.

    Args:
        dataset (xarray.Dataset): Dataset containing the simulated data for X1 and X2.
        sample_sizes (list): List of sample sizes for which to compute estimates.

    Returns:
        xarray.Dataset: A dataset containing the estimated parameters for all (var_param x simulation x sample_sizes) pairs.
    """
    if (
        not sample_sizes
        or min(sample_sizes) < 4
        or max(sample_sizes) > dataset.sizes["time"]
    ):
        raise ValueError("Requested sample exceeds generated data")
    estimates_data = []
    failures = []

    for sample_size in sample_sizes:
        params_list = []

        for var_value in dataset["var_param"].values:
            var_list = []

            for sim in dataset["simulation"].values:
                X1 = (
                    dataset["X1"]
                    .sel(var_param=var_value, simulation=sim)
                    .values[:sample_size]
                )  # Slice the dataset X1
                X2 = (
                    dataset["X2"]
                    .sel(var_param=var_value, simulation=sim)
                    .values[:sample_size]
                )  # Slice the dataset X2
                pci_model = partial_cointegration(
                    X1,
                    X2,
                    mode=mode,
                    seed=int(
                        np.random.SeedSequence(
                            [seed, int(sample_size), int(sim)]
                        ).generate_state(1)[0]
                    ),
                )  # Create PCI object
                try:
                    _, beta, rho, sigma_M, sigma_R, _, _, lrt = pci_model.fit_pci(
                        LRT_mode=True
                    )
                    var_list.append([beta, rho, sigma_M, sigma_R, lrt])
                except (FitError, ValueError) as exc:
                    var_list.append([np.nan] * 5)
                    failures.append(
                        {
                            "n": int(sample_size),
                            "cell": float(var_value),
                            "simulation": int(sim),
                            "error": str(exc),
                        }
                    )

            params_list.append(var_list)

        params_array = np.array(params_list)

        # Create dataset containing estimated parameters for sample_size
        estimates_data.append(
            xr.DataArray(
                data=params_array,
                dims=["var_param", "simulation", "estimates"],
                coords={
                    "var_param": dataset["var_param"].values,
                    "simulation": dataset["simulation"].values,
                    "estimates": ["beta", "rho", "sigma_m", "sigma_r", "lrt"],
                },
                name=f"estimates_sample_size_{sample_size}",
            )
        )

    # Create dataset containing estimated parameters fore all sample_sizes
    final_dataset = xr.merge(estimates_data)

    final_dataset.attrs["failures"] = json.dumps(failures)
    final_dataset.attrs["mode"] = mode
    final_dataset.attrs["seed"] = seed
    final_dataset.attrs["estimator_fingerprint"] = estimator_fingerprint()
    for key in ["design", "initialization", "sigma_x"]:
        if key in dataset.attrs:
            final_dataset.attrs[key] = dataset.attrs[key]
    return final_dataset


def f_compute_means(dataset):
    """
    Compute the mean of the dataset across all simulations.

    Args:
        dataset (xarray.Dataset): A dataset containing the mean for all simulations.

    Returns:
        xarray.Dataset: A dataset containing the mean values across all simulations.
    """
    result = dataset.mean(dim="simulation", skipna=False)
    result.attrs["failures"] = dataset.attrs.get("failures", "[]")
    return result


def f_compute_mse(dataset, params: list):
    """
    Compute the mean squared error (MSE) between the estimated parameters and the actual parameters across all simulations.

    Args:
        dataset (xarray.Dataset): A dataset containing the estimated parameters for all (var_param x simulation x sameple_sizes) pairs.
        params (list): Six truth values [alpha, beta, rho, sigma_m, sigma_r, sigma_x],
                       with None indicating the varying parameter.

    Returns:
        xarray.Dataset: A dataset containing the mean squared error for each parameter.
    """
    if len(params) != 6 or sum(value is None for value in params) != 1:
        raise ValueError("Expected six truth parameters with exactly one varying value")
    names = ["alpha", "beta", "rho", "sigma_m", "sigma_r", "sigma_x"]
    selected = ["beta", "rho", "sigma_m", "sigma_r"]
    results = []
    for value in dataset["var_param"].values:
        actual = dict(zip(names, params))
        actual[names[params.index(None)]] = float(value)
        truth = xr.DataArray(
            [actual[name] for name in selected],
            dims=["estimates"],
            coords={"estimates": selected},
        )
        squared = (dataset.sel(var_param=value, estimates=selected) - truth) ** 2
        results.append(
            squared.mean(dim="simulation", skipna=False).expand_dims(var_param=[value])
        )
    result = xr.concat(results, dim="var_param")
    result.attrs["failures"] = dataset.attrs.get("failures", "[]")
    return result


def f_compute_critical_values_lrt(quantiles, sample_sizes, nb_sims=10000, *, seed=1729):
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
    dataset_rw = f_generate_data(
        params_cr, var_values_cr, sample_sizes, nb_sims, seed=seed
    )

    # Compute estimates based on the generated data
    estimates_rw = f_compute_estimates(dataset_rw, sample_sizes).sel(estimates="lrt")

    # Calculate critical values based on the provided quantiles for each sample size
    if any(not np.isfinite(v.values).all() for v in estimates_rw.data_vars.values()):
        raise FitError("Calibration has failed fits; do not drop them silently")
    critical_values = estimates_rw.quantile(quantiles, dim="simulation")
    critical_values.attrs.update(
        {
            "mode": "exact",
            "seed": seed,
            "replications": nb_sims,
            "estimator_fingerprint": estimator_fingerprint(),
            "design": "geometric",
            "initialization": "known",
            "sigma_x": 0.0236,
        }
    )

    return critical_values


def f_compute_powers_lrt(
    estimate_dataset, sample_sizes, quantiles, *, critical_values=None
):
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
    if critical_values is None:
        raise ValueError(
            "Supply independently generated critical_values; evaluation never recalibrates implicitly"
        )
    if estimate_dataset.attrs.get("mode", "exact") != critical_values.attrs.get(
        "mode", "exact"
    ):
        raise ValueError("Calibration mode mismatch")
    for key in ["estimator_fingerprint", "design", "initialization", "sigma_x"]:
        if (
            key not in estimate_dataset.attrs
            or key not in critical_values.attrs
            or estimate_dataset.attrs[key] != critical_values.attrs[key]
        ):
            raise ValueError(f"Missing/mismatched calibration metadata: {key}")
    if estimate_dataset.attrs["estimator_fingerprint"] != estimator_fingerprint():
        raise ValueError(
            "Calibration/results were produced by another estimator version"
        )
    scores = estimate_dataset.sel(estimates="lrt")
    thresholds = critical_values.sel(var_param=0, quantile=quantiles)
    valid = np.isfinite(scores)
    counts = valid.sum(dim="simulation")
    rejected = ((scores < thresholds) & valid).sum(dim="simulation")
    power = rejected / counts.where(counts > 0)
    total = estimate_dataset.sizes["simulation"]
    power.attrs["failed_counts"] = json.dumps((total - counts).to_dict(data=True))
    power.attrs["unconditional_lower"] = json.dumps(
        (rejected / total).to_dict(data=True)
    )
    power.attrs["unconditional_upper"] = json.dumps(
        ((rejected + total - counts) / total).to_dict(data=True)
    )
    return power


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
    outputs_dir = os.path.join(current_script_dir, "..", "Outputs")

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
    mean_ds = mean_ds.drop_sel(estimates="lrt")

    # Convert each dataset to a pandas DataFrame and reset the index
    df1 = mean_ds.to_dataframe().reset_index()
    df2 = mse_ds.to_dataframe().reset_index()

    # Rename columns to include 'mean_' and 'mse_' prefixes
    df1 = df1.rename(
        columns={
            col: "mean_" + col.split("_")[-1]
            for col in df1.columns
            if "estimates_sample_size" in col
        }
    )
    df2 = df2.rename(
        columns={
            col: "mse_" + col.split("_")[-1]
            for col in df2.columns
            if "estimates_sample_size" in col
        }
    )

    # Concatenate the DataFrames horizontally (side by side)
    estimates = pd.concat([df1, df2], axis=1)

    # Todays date
    global today_date

    # Excel output file path
    excel_file_path = prepare_output_path(f"estimates_table_{today_date}.xlsx")

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
    excel_file_path = prepare_output_path(f"powers_table_{today_date}.xlsx")

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
    critical_values = critical_values.to_dataframe().reset_index()

    # Todays date
    global today_date

    # Excel output file path
    excel_file_path = prepare_output_path(f"critical_values_table_{today_date}.xlsx")

    # Save the critival values DataFrame to excel
    critical_values.to_excel(excel_file_path)

    # Return the critical values DataFrame
    return critical_values


if __name__ == "__main__":
    raise SystemExit("Use experiments/validate.py with an explicit study configuration")
