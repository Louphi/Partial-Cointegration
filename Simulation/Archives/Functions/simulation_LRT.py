from strategyV1.strategyV1 import *
from Simulation.Archives.Functions.generate_data import *
import pandas as pd


# commit and push
# Add function to visualise data
# Separate result replication function into two functions: one for parameters, one for LRT
# Add function to do hypothesis test
#   Simulate 10,000 random walk series and compute the LRT
#   Find the 0.05 quantile.
#   Simulate 10,000 PAR series, with varying levels of sigma_M, compute the lrt each time.
#   Find the % of simulations where the lrt < 5%_cv --> this is the power of our test.
#   Power of hypothesis test: probability of rejecting the null (lrt < 5%_cv) when the alternative is true.

def f_get_cv(x, quantiles: list) -> list:
    x_clean = [i for i in x if str(i) != "nan"]
    cv = np.quantile(x_clean, quantiles)
    return cv

def f_result_replication(fixed_params: list, var_param="sigma_M",
                         start=0, end=2, by=0.25, nb_sims=10000, mode="Parameters") -> list:
    """
    Args:
        fixed_params: A list of the values of the parameters that remain fixed for generating data
                      Either [N, alpha, beta, rho, sigma_R, sigma_X] or [N, alpha, beta, sigma_M, sigma_R, sigma_X].
        var_param: The parameter that we vary for generating data
                   Default is "sigma_M", other option is "rho".
        start: Starting value for the parameter that we vary
        end: Ending value for the parameter that we vary
        by: Steps between the values of the varying parameter
        nb_sims: Number of Monte Carlo simulations to run
        mode: What estimated parameters to return to the user
              If "Parameter", return the 5 estimated parameters of the PAR model.
              If "LRT", return the values of the max log-likelihood under the Random Walk Null Hypothesis and the PAR
              alternative, and the value of the associated likelihood ratio test.

    Returns: A list of n dataframes containing the values of the estimated parameters, or the values associated with LRT,
             for each simulation. Where n is the number of steps taken between start & end.

    """
    if var_param == "sigma_M":
        N, alpha, beta, rho, sigma_R, sigma_X  = fixed_params
        sigma_M_values = np.arange(start, end + by, by)
    if var_param == "rho":
        N, alpha, beta, sigma_M, sigma_R, sigma_X = fixed_params
        rho_values = np.arange(start, end + by, by)

    df_list = []

    if mode == "Parameters":
        if var_param == "sigma_M":
            for sigma_M in sigma_M_values:
                params = [N, alpha, beta, rho, sigma_M, sigma_R, sigma_X]
                df = _f_monte_carlo_params(nb_sims, params)
                df_list.append(df)

        elif var_param == "rho":
            for rho in rho_values:
                params = [N, alpha, beta, rho, sigma_M, sigma_R, sigma_X]
                df = _f_monte_carlo_params(nb_sims, params)
                df_list.append(df)

    elif mode == "LRT":
        if var_param == "sigma_M":
            for sigma_M in sigma_M_values:
                params = [N, alpha, beta, rho, sigma_M, sigma_R, sigma_X]
                df = _f_monte_carlo_LRT(nb_sims, params)
                df_list.append(df)

        elif var_param == "rho":
            for rho in rho_values:
                params = [N, alpha, beta, rho, sigma_M, sigma_R, sigma_X]
                df = _f_monte_carlo_LRT(nb_sims, params)
                df_list.append(df)

    return df_list

def _f_monte_carlo_params(nb_sims:int, params:list):
    print(params)
    N_s, alpha_s, beta_s, rho_s, sigma_M_s, sigma_R_s, sigma_X_s = params

    df = pd.DataFrame(data=0, columns=["alpha", "beta", "rho", "sigma_M", "sigma_R"], index=np.arange(nb_sims))

    for i in range(0, nb_sims):

        pci_pair = pci_generator(N_s, alpha_s, beta_s, rho_s, sigma_M_s, sigma_R_s, sigma_X_s)
        X1, X2, _, _, _ = pci_pair.f_generate_pci()
        pci_model = partial_cointegration(X1, X2)
        alpha, beta, rho, sigma_M, sigma_R, ll_model, ll_randomWalk, W = pci_model.fit_pci()

        df["alpha"][i] = alpha
        df["beta"][i] = beta
        df["rho"][i] = rho
        df["sigma_M"][i] = sigma_M
        df["sigma_R"][i] = sigma_R

    return df

def _f_monte_carlo_LRT(nb_sims:int, params:list):

    N_s, alpha_s, beta_s, rho_s, sigma_M_s, sigma_R_s, sigma_X_s = params

    df = pd.DataFrame(data=0, columns=["ll_rw", "ll_par", "lrt"], index=np.arange(nb_sims))

    for i in range(0, nb_sims):

        pci_pair = pci_generator(N_s, alpha_s, beta_s, rho_s, sigma_M_s, sigma_R_s, sigma_X_s)
        X1, X2, _, _, _ = pci_pair.f_generate_pci()
        pci_model = partial_cointegration(X1, X2)
        alpha, beta, rho, sigma_M, sigma_R, ll_par, ll_rw, W = pci_model.fit_pci()

        lrt = ll_rw - ll_par

        df["ll_rw"][i] = ll_rw
        df["ll_par"][i] = ll_par
        df["lrt"][i] = lrt

    return df

def f_df_to_xlsx(dataframes, excel_file="output.xlsx"):
    """
    Saves a list of dataframes to an Excel file, each dataframe in a separate sheet.

    Parameters:
        dataframes (list): List of pandas DataFrames.
        excel_file (str): File path for the Excel file to be saved.
    """
    with pd.ExcelWriter(excel_file) as writer:
        for i, df in enumerate(dataframes, start=1):
            sheet_name = f"Sheet_{i}"
            df.to_excel(writer, sheet_name=sheet_name, index=False)


params = [100, 0 , 1, 0.9, 1, 0.0236] #N, alpha, beta, rho, sigma_R, sigma_X

l = f_result_replication(params, "sigma_M", start=0, end=2, by=0.25, nb_sims=10000, mode="LRT")
f_df_to_xlsx(l)



# quantiles    = [0.01, 0.05, 0.10]
# sigma_m = 0.25:[0.966, 0.92, 0.846]
# sigma_m = 0.50:[0.9566, 0.8715, 0.7562]
# sigma_m = 0.75: [0.8489, 0.6655, 0.4939]
# sigma_m = 1: [0.6845, 0.4221, 0.2542]
# sigma_m = 1.25: [0.4224, 0.1845, 0.09]
# sigma_m = 1.50: [0.2223, 0.0689, 0.0261]
# sigma_m = 1.75: [0.1002, 0.0244, 0.0094]
# sigma_m = 2.00: [0.0452, 0.0072, 0.0018]
