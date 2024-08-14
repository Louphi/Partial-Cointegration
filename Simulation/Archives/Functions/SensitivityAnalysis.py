from Simulation.Archives.Functions.MonteCarloEstimation import *
import numpy as np
import pandas as pd
import concurrent
import warnings
# Suppress all warnings
warnings.filterwarnings('ignore')

def monte_carlo_wrapper(params):
    n_sample, alpha, beta, rho, sigma_M, sigma_R, sigma_X, n_simulation = params
    print(f'n_sample: {n_sample} and sigmaM : {sigma_M}')
    result = MonteCarloEstimation.f_monte_carlo(n_simulation=n_simulation, params=[n_sample, alpha, beta, rho, sigma_M, sigma_R, sigma_X], csv_option=False, LRT_mode=False)
    print(f'n_sample: {n_sample} and sigmaM : {sigma_M} : DONE')
    return f'{sigma_M}{n_sample}', result

class SensitivityAnalysis:
    
    def f_sensitivity_analysis(fixed_params: list, var_param : str ="sigma_M", n_simulation : int = 10000, n : list = [100, 1000, 10000]) -> dict:
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
    
        Mydict = {}
        lst_truevalue = []
        if var_param == "sigma_M":
            increment = (0, 2, 0.5)
            start, end, by = increment
            alpha, beta, rho, sigma_R, sigma_X  = fixed_params
            sigma_M_values = np.arange(start, end + by, by)
            lst_truevalue = [[beta, rho, sigma_M, sigma_R] for sigma_M in sigma_M_values]

            tasks = [(n_sample, alpha, beta, rho, sigma_M, sigma_R, sigma_X, n_simulation) for n_sample in n for sigma_M
                     in sigma_M_values]

            with concurrent.futures.ThreadPoolExecutor() as executor:
                results = list(executor.map(monte_carlo_wrapper, tasks))

            for key, result in results:
                Mydict[key] = result
                
            return {"Parameter": var_param, "Result" : Mydict, "Increment" : sigma_M_values, 'TrueValue': lst_truevalue, "n": n}
            
        elif var_param == "rho":
            increment = (1, 0.6, -0.1)
            start, end, by = increment 
            alpha, beta, sigma_M, sigma_R, sigma_X = fixed_params
            rho_values = np.arange(start, end + by, by)
            lst_truevalue = [[beta, rho, sigma_M, sigma_R] for rho in rho_values]

            tasks = [(n_sample, alpha, beta, rho, sigma_M, sigma_R, sigma_X, n_simulation) for n_sample in n for rho
                     in rho_values]

            with concurrent.futures.ThreadPoolExecutor() as executor:
                results = list(executor.map(monte_carlo_wrapper, tasks))

            for key, result in results:
                Mydict[key] = result
            
            return {"Parameter": var_param, "Result" : Mydict, "Increment" : rho_values, 'TrueValue': lst_truevalue, "n": n}





class create_table:
    def __init__(self, result_dict : dict):
        self.parameter = result_dict["Parameter"]
        self.res_data = result_dict["Result"]
        self.increment = result_dict["Increment"]
        self.true_value = result_dict["TrueValue"]
        self.true_value_data = [element for sous_liste in self.true_value for element in sous_liste]
        self.n_time = len(self.increment)
        self.n = result_dict["n"]

                          
                          
    def calculate_average_dict(self, res_data, increment, n_time, n):
        tmp_dict = {}
        
        for i in range(len(n)):
                n_sample = n[i]
                tmp = [res_data[f"{increment[j]}{n_sample}"].loc[:, ["beta", "rho", "sigma_M", "sigma_R"]].mean().values for j in range(n_time)]
                tmp_dict[n_sample] = [element for sous_liste in tmp for element in sous_liste]
        
        return tmp_dict

    def calculate_mse(self, df, true_value):
        
        return np.mean((df - true_value)**2)
    
    def calculate_mse_dict(self, res_data, increment, true_value, n_time, n):
        tmp_dict2 = {}
        for i in range(len(n)):
            print(n)
            n_sample = n[i]
            tmp_dict2[n_sample] = [self.calculate_mse(res_data[f"{increment[j]}{n_sample}"].loc[:, ["beta", "rho", "sigma_M", "sigma_R"]].iloc[:, x], true_value[j][x]) for j in range(n_time) for x in range(4)]
        return tmp_dict2
    
    def Rsquare(self, sigmaM, sigmaR, rho):
        return 2 * (sigmaM ** 2) / (2 * (sigmaM ** 2) + (1+rho) * sigmaR ** 2)
    
    
    def create_col_rsquare(self, true_value_data, parameter, increment, n_time):
        rho = true_value_data[1]
        sigma_R = true_value_data[3]
        
        if parameter == "sigma_M":
            rsq = [self.Rsquare(sigma_M, sigma_R, rho) for sigma_M in increment]
        else:
            sigma_M = true_value_data[2]
            rsq = [self.Rsquare(sigma_M, sigma_R, rho) for rho in increment]
            
        return np.repeat(np.round(rsq, 2), (n_time-1))
    
    def construct_table(self):
        tmp = np.zeros(20)
        tmp_dict = self.calculate_average_dict(self.res_data, self.increment, self.n_time, self.n)
        tmp_dict2 = self.calculate_mse_dict(self.res_data, self.increment, self.true_value, self.n_time, self.n)
        print(len(np.repeat(self.increment, (self.n_time-1))), len(["beta", "rho", "sigma_M", "sigma_R"] * self.n_time), len(self.true_value_data))
        """df = pd.DataFrame({self.parameter: np.repeat(self.increment, (self.n_time-1)),
                     "R_square": self.create_col_rsquare(self.true_value_data, self.parameter, self.increment, self.n_time),
                     "Parameter": ["beta", "rho", "sigma_M", "sigma_R"] * self.n_time, 
                     f"MSE{self.n[0]}": tmp_dict2[self.n[0]],
                     f"MSE{self.n[1]}": tmp_dict2[self.n[1]],
                     f"MSE{self.n[2]}": tmp_dict2[self.n[2]],
                     f"AVG{self.n[0]}":  tmp_dict[self.n[0]],
                     f"AVG{self.n[1]}":  tmp_dict[self.n[1]],
                     f"AVG{self.n[2]}": tmp_dict[self.n[2]],
                     "TrueParameter": self.true_value_data})"""
                     
        dynamic_columns = {}
        colname = ["MSE", "AVG"]
        dicts = [tmp_dict2, tmp_dict]
        for x in range(len(colname)):
            for n_value in self.n:
                dynamic_columns[f"{colname[x]}{n_value}"] = dicts[x][n_value]  # tmp_dict2 should contain MSE values for each n_value
               
        dynamic_columns["TrueParameter"] = self.true_value_data   
        

        # Static columns
        static_columns = {
            self.parameter: np.repeat(self.increment, (self.n_time-1)),
            "R_square": self.create_col_rsquare(self.true_value_data, self.parameter, self.increment, self.n_time),
            "Parameter": ["beta", "rho", "sigma_M", "sigma_R"] * self.n_time
        }

        # Combine static and dynamic columns
        data_frame_columns = {**static_columns, **dynamic_columns}

        # Create DataFrame
        df = pd.DataFrame(data_frame_columns)
        
        return df
