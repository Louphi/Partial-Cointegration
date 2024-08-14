
import numpy as np
import pandas as pd
import PCISimulation as simul
import PartialCointegration as pci
from datetime import datetime
import warnings
# Suppress all warnings
warnings.filterwarnings('ignore')


class MonteCarloEstimation:

    def f_monte_carlo(n_simulation :int, params: list, csv_option: bool = True, LRT_mode: bool = True):
       
        n_periods, alpha_s, beta_s, rho_s, sigma_M_s, sigma_R_s, sigma_X_s = params
        # Simuler les paths : 10 000 x trajectory = (n_periods x n_simulation)
        simulPaths = simul.pci_generator(n_periods, alpha_s, beta_s, rho_s, sigma_M_s, sigma_R_s,
                                         sigma_X_s).f_generate_trajectories(trajectory=n_simulation)

        if LRT_mode:
            col_interest = ["alpha", "beta", "rho", "sigma_M", "sigma_R", "ll_par", "ll_rw", "lrt"]
        else:
            col_interest = ["alpha", "beta", "rho", "sigma_M", "sigma_R", "ll_par"]

        df = pd.DataFrame(data=0, columns=col_interest, index=np.arange(n_simulation))

        for i in range(0, n_simulation):
            
            X1 = simulPaths[:, i, 0]
            X2 = simulPaths[:, i, 1]
            pci_model = pci.partial_cointegration(X1, X2)
            df.loc[i, col_interest] = pci_model.fit_pci(LRT_mode=LRT_mode)

        if csv_option:
            if LRT_mode:
                path = "Outputs_LRT"
            else:
                path = "Outputs_parameters"
            try:
                now = datetime.now()
                # Format the date and time as a string with allowed characters only
                # For example, replacing colons with dashes and removing microseconds
                datetime_str = now.strftime("%Y-%m-%d_%H-%M-%S")
                csv_file = f"output{datetime_str}.csv"
                df.to_csv(f"{path}/{csv_file}", index=False)
            except Exception as e:
                print(f"error saving dataframe to csv : {e}")

        return df