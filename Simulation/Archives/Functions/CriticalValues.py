import pandas as pd
import numpy as np
import pickle

class CriticalValues:
    def __init__(self, quantile) -> None:
        
        
        with open(r"Outputs_LRT/critical_values_dict.pickle", "rb") as input_file:
            self.critical_values_dict = pickle.load(input_file)
            
        self.quantile = quantile
    
          
    def create_tableQuantile(self):
        n = list(self.critical_values_dict.keys())
        
        df = pd.DataFrame(data = 0, index = n, columns=[f"p={self.quantile[i]}" for i in range(len(self.quantile)) ])
        for n_sample in n:
            df.loc[n_sample] = np.quantile(np.array(self.critical_values_dict[n_sample]["lrt"]), self.quantile)
            
        return df 
    
    
    ## Generation of the data 
    
    """n_periods = 2500
alpha = 0
beta = 1
rho = 0
sigma_R = 1
sigma_M = 0
sigma_X = 0.0236
params = [n_periods, alpha, beta, rho, sigma_M, sigma_R, sigma_X]
n_simulation = 1000

df = monte_carlo.MonteCarloEstimation.f_monte_carlo(n_simulation = n_simulation, params= params, csv_option = True, LRT_mode = True) 
critical_values_dict ={"50": df_50, "100": df_100, "250": df_250,
                       "500": df_500, "1000": df_1000, "2500": df_2500}
with open(r"Outputs_LRT/critical_values_dict.pickle", "wb") as output_file:
        pickle.dump(critical_values_dict, output_file) """