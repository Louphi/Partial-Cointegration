##### Function to add : Trading rules #####

import pandas as pd
from partial_cointegration import *

def calculate_Rsq(rho, sigma_M, sigma_R):
    '''
    Calculate R^2 - proportion of variance attributable to mean reversion
    '''
    return (2*sigma_M**2) / (2*sigma_M**2 + (1+rho) * sigma_R**2)

def trading_parameters(X1, X2):
    time_series =  pd.DataFrame({"X1": X1, "X2": X2})
    parameters = pd.DataFrame(columns=["Alpha", "Beta", "rho", "sigma_M", "sigma_R", "Rsq", "sigmas_Ratio", "M_last", "M_last/sigma_M"])

    for window in time_series.rolling(window=390):
        if len(window) < 390:
            continue
        else:
            pci = partial_cointegration(window["X1"].to_numpy(), window["X2"].to_numpy())
            pci.fit_pci()
            Rsq = calculate_Rsq(pci.get_rho(), pci.get_sigma_M(), pci.get_sigma_R())
            pci.kalman_estimate(pci.get_W(),pci.get_rho(), pci.get_sigma_M(), pci.get_sigma_R())
            parameters.loc[len(parameters)] = [
                pci.get_alpha(),
                pci.get_beta(),
                pci.get_rho(),
                pci.get_sigma_M(),
                pci.get_sigma_R(),
                Rsq,
                pci.get_sigma_M() / pci.get_sigma_R(),
                pci.get_M_last(),
                pci.get_M_last() / pci.get_sigma_M()
                ]

    return parameters

N_3, alpha_3, beta_3, rho_3, sigma_M_3, sigma_R_3, sigma_X_3 = 600, 5, 0.3, 0.5, 0.9, 0.1, 0.003
params_3 = N_3, alpha_3, beta_3, rho_3, sigma_M_3, sigma_R_3, sigma_X_3
generator_3 = time_series_generator(*params_3)
X_3, Y_3, W_3, M_3, R_3 = generator_3.generate_pci()

params = trading_parameters(X_3, Y_3)

print(params.to_string())

plt.plot(params["M_last"])
plt.plot(2*params["sigma_M"])
plt.plot(-2*params["sigma_M"])
plt.show()
