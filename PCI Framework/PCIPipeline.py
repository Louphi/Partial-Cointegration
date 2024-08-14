import pandas as pd
from itertools import permutations
from tqdm import tqdm
import numpy as np
import scipy.optimize as opt
import statsmodels.api as sm
import scipy.stats as stats
import PCIData

class DataTable:
    def __init__(self, pairName:list, constituents:list):
        self.pairName = pairName
        self.constituents = constituents
        self.dict_data={}

    def getData(self):
        if self.dict_data == {}:
            self.setData()
        else:
            pass
        return self.dict_data
    
    def setData(self):
        self.dict_data = dict(zip(self.pairName, self.constituents))
        return self.dict_data
class Definition:
    def f_load_adjPrice(filepath) -> pd.DataFrame:
        """Load pandas dataframe of the price series
        Args:
          filepath (str): Filepath to the adjPrice_nasdaq100.csv"""
        try:
            adjPrice = pd.read_pickle(filepath)
            adjPrice.index = adjPrice["date"]
        except Exception as e:
            print("Error : {e}")
            adjPrice = None
        return adjPrice
    
    def f_selectTicker(tickerList, adjPrice)-> pd.DataFrame:
        """ 
        Select the columns that are in tickerList

        Args:
            tickerList (list): List of all tickers to take into considerations
            adjPrice (pd.DataFrame): Adjusted price of all tickers
        """
        
        return adjPrice.loc[:, adjPrice.columns.isin(tickerList)]

    def f_insample_outsample(data, year):
        data.index = pd.to_datetime(data.index)
        
        start_of_year = pd.to_datetime(str(year))
        
        end_of_period = start_of_year + pd.DateOffset(months=6)  # This is "year + 6 months"
        start_of_period = start_of_year - pd.DateOffset(years=4) 
        
        # Adjusted filter condition using the corrected datetime offsets
        all_sample = data[(data.index < end_of_period) & (data.index >= start_of_period)]
        
        # Cleaning operations
        tmp = all_sample.dropna(how="all")
        all_sample_cleaned = tmp.dropna(axis=1, how='any')
        
        
        # Select in-sample data: data from the four years prior to the specified year.
        # It checks if the year in the data is less than the specified year
        # and greater than or equal to four years before the specified year.
        
        in_sample = all_sample_cleaned[(all_sample_cleaned.index < start_of_year) & (all_sample_cleaned.index >= start_of_period)]
        
        # Select out-sample data: data from the specified year but only for the first six months.
        # It checks if the year in the data is exactly the specified year and
        # if the month is less than or equal to 6 (January to June).
        out_sample = all_sample_cleaned[(all_sample_cleaned.index >= start_of_year) & (all_sample_cleaned.index <= end_of_period)]

        # Return both the in-sample and out-sample datasets.
        return in_sample, out_sample


    def pairDatabase(consituentList : list) -> pd.DataFrame:
    
        combinations = [(a, b) for a, b in permutations(consituentList, 2)]

        # Create a DataFrame
        pairDatabase = pd.DataFrame(combinations, columns=["Stock 1", "Stock 2"])
        pairDatabase["pair"] = pairDatabase["Stock 1"] + "/" + pairDatabase["Stock 2"]
        
        return pairDatabase

class Computation:
    def __init__(self, X1, X2) -> None:
        self.X1 = X1
        self.X2 = X2

    def Rsquare(sigmaM, sigmaR, rho):
        return 2 * (sigmaM ** 2) / (2 * (sigmaM ** 2) + (1+rho) * sigmaR ** 2)
   
    def fit_mle(self, W, tol=0.001):
        '''
        Maximum likelihood estimation of the associated Kalman filter

        Parameters:
        W (numpy.ndarray): the time-series which we want to model has both permanent and transient components

        Returns:
        rho (float): estimated value of rho
        sigma_M (float): estimated value of sigma_M
        sigma_R (float): estimated value of sigma_R
        '''

        # Set empty list for estimate and ll values
        estimates = []
        lls = []

        # Set distributions for random guesses
        rnd_rho = stats.uniform(loc=-1, scale=2)
        std = np.std(np.diff(W)) # Compute standard deviation of "pci process" (or residuals)
        rnd_sigma = stats.norm(loc=std, scale=std / 2) # Why scaled to std / 2? Can we find better? ####################

        # Define function for generating random initial guesses from above distributions
        def gen_x_i():
            return rnd_rho.rvs(), rnd_sigma.rvs(), rnd_sigma.rvs()

        # Minimize the negative log-likelihood function
        bounds = ((-1, 1), (0, np.inf), (0, np.inf)) # Set bounds for opt.
        x_i = self.lagvar_estimate_par(W) # Get initial guesses using lagged variance equations
        res = opt.minimize(self.f_to_min_par , x0=(x_i), args=(W), bounds=bounds, tol=tol) # Minimize -ll function

        if res.success: # If optimization is a success
            estimates.append(res.x) # Save estimates in list
            lls.append(-res.fun)  # Save log-likelihood in list
        # return res.x
    
        # Repeat optimization with different random initial values
    
        n_att = 0
        while len(lls) < 10 and n_att < 100:
            n_att += 1
            x_i = gen_x_i()
            res = opt.minimize(self.f_to_min_par, x0=(x_i), args=(W), bounds=bounds, tol=tol)

            if res.success:  # If optimization is a success
                estimates.append(res.x)  # Save estimates
                lls.append(-res.fun)  # Save log-likelihood

        try:
            argmax = np.argmax(lls)  # Find index of max likelihood
            return estimates[argmax] # Return estimates linked to max likelihood
        except:
            # print('Estimation failed!')
            return len(x0) * [np.nan]  # Returns nanns
        

    def lagvar_estimate_par(self, W):
        '''
        Estimate parameters of partial AR model using lagged variances. Used for first estimation of parameters

        Parameters
        W (numpy.ndarray): A partial autoregressive time series

        Returns:
        rho_lv (float): estimated value for rho
        sigma_M_lv (float): estimated value for sigma_M
        sigma_R_lv (float): estimated value for sigma_R
        '''

        # See Matthew Clegg: "Modeling Time Series with both Permanent and Transient
        # Component using the Partially Autoregressive Model". See equations on page 5.

        # Calculate variance of the lagged differences. Left hand side of equation (3)
        v1 = np.var(W[1:] - W[:-1])
        v2 = np.var(W[2:] - W[:-2])
        v3 = np.var(W[3:] - W[:-3])

        # Calculate rho from v1, v2, v3. Equations (4), page 5
        rho_lv = -(v1 - 2 * v2 + v3) / (2 * v1 - v2)

        # Calculate sigma_M. Equations (4), page 5
        if (rho_lv + 1) / (rho_lv - 1) * (v2 - 2 * v1) > 0:
            sigma_M_lv = np.sqrt(1 / 2 * (rho_lv + 1) / (rho_lv - 1) * (v2 - 2 * v1))
        else:
            sigma_M_lv = 0

        # Calculate sigma_M. Equations (4), page 5
        if v2 > 2 * sigma_M_lv ** 2:
            sigma_R_lv = np.sqrt(1 / 2 * (v2 - 2 * sigma_M_lv ** 2))
        else:
            sigma_R_lv = 0

        return rho_lv, sigma_M_lv, sigma_R_lv

    def fit_ols_on_diff(self):
        '''
        Fits an OLS model on the first differences of time series X1 and X2

        Parameters:
        X1 (numpy.ndarray): price time-series
        X2 (numpy.ndarray): price time-series

        Returns:
        results.params[0]: returns the Beta value of our OLS fit
        '''
        ret_X1 = np.diff(self.X1)
        ret_X2 = np.diff(self.X2)

        results = sm.OLS(ret_X2, ret_X1).fit()

        return results.params[0]

    def kalman_estimate(self, W, rho, sigma_M, sigma_R):
        '''
        Calculate estimates of mean-reverting series (M_t) and random walk series (R_t).

        Parameters:
        X (numpy.ndarray): A partial autoregressive time-series
        rho (float): AR(1) coefficient / mean reversion coefficient.
        sigma_M (float): standard deviation of the innovations of the mean-reverting component
        sigma_R (float): standard deviation of the innovations of the random walk component

        Returns:
        M (numpy.ndarray): An estimate of the mean reverting component of our time series
        R (numpy.ndarray): An estimate of the random walk component of our time series
        eps (numpy.ndarray): Prediction errors for each time step
        '''

        # See Matthew Clegg: "Modeling Time Series with both Permanent and Transient
        # Component using the Partially Autoregressive Model". See algo on page 9.

        # Create arrays for storing both components and prediction errors
        M = np.zeros(len(W))
        R = np.zeros(len(W))
        eps = np.zeros(len(W))

        # Set state at t=0
        if sigma_R == 0: # If series is purely mean-reverting
            M[0] = W[0]
            R[0] = 0
        else:
            M[0] = 0
            R[0] = W[0]

        # Calculate Kalman gains
        if sigma_M == 0: # If series is purely a random walk
            K_M = 0
            K_R = 1
        elif sigma_R == 0: # If series is purely mean-reverting
            K_M = 1
            K_R = 0
        else:
            # See equations (11), page 8
            sqr = np.sqrt((1 + rho) ** 2 * sigma_R ** 2 + 4 * sigma_M ** 2)
            K_M = 2 * sigma_M ** 2 / (sigma_R * (sqr + rho * sigma_R + sigma_R) + 2 * sigma_M ** 2) # Compute gain for M
            K_R = 2 * sigma_R / (sqr - rho * sigma_R + sigma_R) # Compute gain for R

        # Calculate estimates recursively
        for i in range(1, len(W)):
            w_hat = rho * M[i - 1] + R[i - 1] # Predicted value of W[i]
            eps[i] = W[i] - w_hat # Prediction error
            M[i] = rho * M[i - 1] + eps[i] * K_M
            R[i] = R[i - 1] + eps[i] * K_R

        return M, R, eps

    def calc_log_like(self, W, rho, sigma_M, sigma_R):
        '''
        Compute negative log likelihood function

        Parameters:
        X (numpy.ndarray): A partial autoregressive time series
        rho (float): AR(1) coefficient / mean reversion coefficient.
        sigma_M (float): standard deviation of the innovations of the mean-reverting component
        sigma_R (float): standard deviation of the innovations of the random walk component

        Returns:
        ll (float): Value of the log likelihood, a measure of goodness of fit for our model
        '''

        N = len(W)
        _, _, eps = self.kalman_estimate(W, rho, sigma_M, sigma_R) # Compute the prediction errors for each time step

        # Compute the value of the log-likelihood function
        ll = -(N - 1) / 2 * np.log(2 * np.pi * (sigma_M ** 2 + sigma_R ** 2)) - 1 / (
                2 * (sigma_M ** 2 + sigma_R ** 2)) * np.sum(eps[1:] ** 2)

        return ll

    def f_to_min_par(self, x_i, W):
        rho, sigma_M, sigma_R = x_i
        '''
        Define the function to minimize for PAR model
        '''
        return -self.calc_log_like(W, rho, sigma_M, sigma_R)

    def f_to_min_pci(self, x_i, X1, X2):
        '''
        Define function to minimize
        '''
        if len(x_i) == 5 :
            alpha, beta, rho, sigma_M, sigma_R = x_i
            
        elif len(x_i) == 3:
            alpha, beta, sigma_R = x_i
            rho = 0
            sigma_M = 0
        
        W = X2 - beta * X1 - alpha
        return -self.calc_log_like(W, rho, sigma_M, sigma_R)

    def fit_pci(self, tol=0.001, LRT_mode: bool = True):
        '''
        Fit partial cointegrated model to time series X1 and X2 such that:
            - X_2,t = alpha + beta * X_1,t + W_t
            - W_t = M_t + R_t
            - M_t = rho * M_t-1 + eps(M_t)
            - R_t = R_t-1 + eps(R_t)
            - eps(M_t) ∼ NID(0, sigma_M)
            - eps(R_t) ∼ NID(0, sigma_R)

        Parameters:
        X1 (numpy.ndarray): time series
        X2 (numpy.ndarray): time series, supposedly partially cointegrated with X1

        Returns:
        alpha (float): estimated value for alpha (linear regression parameter)
        beta (float): estimated value for beta (linear regression parameter)
        rho (float): estimated AR(1) coefficient / mean reversion coefficient.
        sigma_M (float): standard deviation of the innovations of the mean-reverting component
        sigma_R (float): standard deviation of the innovations of the random walk component
        '''

        # Calculate initial guess for beta with linear regression
        results = self.fit_ols_on_diff()
        beta_i = results

        # Calculate initial guess for alpha
        alpha_i = self.X2[0] - beta_i * self.X1[0]

        # Calculate residuals W and initial guesses for rho, sigma_M, sigma_R (params_i)
        W = self.X2 - alpha_i - beta_i * self.X1
        params_i = self.fit_mle(W)

        rho, sigma_M, sigma_R = params_i

        # perform optimization depending on the mode (Complete Model)
        x_i = (alpha_i, beta_i, rho, sigma_M, sigma_R)  # initial guess
        bounds = [(None, None), (None, None), (-1, 1), (0.0001, None), (0.0001, None)]
        res = opt.minimize(self.f_to_min_pci, x_i, args=(self.X1, self.X2), tol=tol, bounds=bounds)
        alpha, beta, rho, sigma_M, sigma_R = res.x
        ll_model = -res.fun

        # W = self.X2 - alpha - beta * self.X1

        if LRT_mode:
            # perform optimization for the random Walk H0)
            # perform optimization depending on the mode (Complete Model)
            x_i = (alpha_i, beta_i, sigma_R)  # initial guess
            bounds = [(None, None), (None, None), (0.0001, None)]
            res = opt.minimize(self.f_to_min_pci, x_i, args=(self.X1, self.X2), tol=tol, bounds=bounds)
            ll_randomWalk = -res.fun
            lrt = ll_randomWalk - ll_model
            return alpha, beta, rho, sigma_M, sigma_R, ll_model, ll_randomWalk, lrt

        return alpha, beta, rho, sigma_M, sigma_R, ll_model
    
    def f_get_elligibility(Stock1, Stock2, inSample):
        
        X = inSample[[Stock1, Stock2]].dropna()

        X1 = X[Stock1]
        X2 = X[Stock2]

        coint = Computation(X1, X2)
        alpha_hat, beta_hat, rho_hat, sigma_M_hat, sigma_R_hat, ll_model, ll_randomWalk, ll_ratio = coint.fit_pci(LRT_mode = True)
        # M_hat, R_hat, _ = coint.kalman_estimate(W_hat, rho_hat, sigma_M_hat, sigma_R_hat)

        Rsquare_res = Computation.Rsquare(sigma_M_hat, sigma_R_hat, rho_hat)

        # print([ll_randomWalk, ll_model])
        
        # for pair in pairName:

        return {"elligibility":[rho_hat, Rsquare_res, ll_ratio],
            "estimators" :[alpha_hat, beta_hat, rho_hat, sigma_M_hat, sigma_R_hat]}
    
    def f_compute_estimate(pairName: pd.DataFrame, inSample: pd.DataFrame, pairData: pd.DataFrame) -> tuple :
        N = len(pairName)
        elligibilityData = np.zeros((N, 3))
        estimationData = np.zeros((N, 5))
        
        

        for pair in tqdm(range(N), desc="Processing Pairs"):
            try:
                pairName_i = pairName[pair]
                # print(f"Doing : {pairName_i}")
                Stock1 = pairData["Stock 1"][pair]
                Stock2 = pairData["Stock 2"][pair]
                
                dict_res = Computation.f_get_elligibility(Stock1, Stock2, inSample)  # Call f_get_elligibility using Computation class
                elligibilityData[pair] = dict_res["elligibility"]
                # latentVariablesData[pairName[pair]] = dict_res["latentVariables"]
                estimationData[pair] = dict_res["estimators"]
                #print(f"{pairName_i}: Done")
            except Exception as e: 
                print(f"Error with {pairName_i} and {e}")
        return elligibilityData, estimationData


    def f_create_dataframe(elligibilityData: pd.DataFrame, estimationData: pd.DataFrame, pairName: pd.DataFrame)-> tuple:
    
        elligibilityData = pd.DataFrame(elligibilityData, index = pairName, columns=["rho", "Rsquare", "ll_ratio"])
        elligibilityData["ll_ratio"].fillna(0)
        elligibility_sorted_data = elligibilityData[
                                                    (elligibilityData["rho"] > 0.5) & 
                                                     (elligibilityData["Rsquare"] > 0.5)
                                                ].sort_values(by="ll_ratio", ascending=True)
        # Dataframe des paramètres estimés
        estimatorData = pd.DataFrame(estimationData, index = pairName, columns=["alpha_hat", "beta_hat", "rho_hat", "sigma_M_hat", "sigma_R_hat"])
        
        return elligibility_sorted_data, estimatorData
    
    def f_pairSelected(elligibility_sorted_data, n_keep :int):
        # Nombre de pair elligible
        n_pair = len(elligibility_sorted_data)
        if n_pair == 0:
                print("Elligibility_sorted_data is empty")
                return None
        
        n_elligible = len(elligibility_sorted_data.index)
        
        n = max(1, min(n_pair, n_keep))
        
        pairListSelected = elligibility_sorted_data.index[:n].to_list()

        return pairListSelected
    
    def get_stock_list(pairData, col):
        selected_pairs = pairData.loc[pairData["pair"].isin(col)]
        tmp = selected_pairs[["Stock 1", "Stock 2"]].values.flatten().tolist()
        stock_list = list(set(tmp))
        return stock_list
    
    def get_stock1_stock2(pair, pairData):
        S1, S2 = pairData.loc[pairData["pair"] == pair, ["Stock 1", "Stock 2"]].iloc[0].values
        return S1, S2
    
    def get_stock_price(df_price, stockList):
        return df_price[stockList].iloc[:, 0], df_price[stockList].iloc[:, 1]
    
    
    def f_create_Mt(pairListSelected, estimatorData, pairData, stock_price):
        
        estimator_Data_selected = estimatorData.loc[estimatorData.index.isin(pairListSelected)]
        
        n = len(pairListSelected)
        df_mt = np.zeros((len(stock_price), n))
        df_hedge = np.zeros((len(stock_price), n))

        for i in range(n):
            pair_tmp = pairListSelected[i]
            
            S1, S2 = Computation.get_stock1_stock2(pair_tmp, pairData)

            X1, X2 = Computation.get_stock_price(stock_price, [S1, S2])

            # Retrieve estimater
            alpha, beta, rho, sigmaM, sigmaR = estimator_Data_selected.loc[pair_tmp].to_list()

            W = X2 - alpha - beta * X1
            coint_tmp = Computation(X1, X2)
            _, Mt, _ = coint_tmp.kalman_estimate(W, rho, sigmaM, sigmaR)
            df_mt[:, i] = Mt
            # Number of share invested in S1 for 1$ invested in X2
            df_hedge[:, i] = beta / X2 # The amount of share
            
        return pd.DataFrame(data = df_mt, columns=pairListSelected), pd.DataFrame(data = df_hedge, columns = pairListSelected)
      
    
    def rolling_z_score(series: pd.Series, window: int) -> pd.Series:
        """
        Calculate the rolling Z-score of a given time series.
        
        Parameters:
        series (pd.Series): The time series data.
        window (int): The size of the rolling window.
        
        Returns:
        pd.Series: A series of rolling Z-scores.
        """
        # Convert numpy array to pandas Series if necessary
        # Convert numpy array to pandas Series if necessary
    # Initialize an array to hold the rolling Z-scores
        z_scores = np.full(series.shape, np.nan)

        # Loop through the series to calculate the rolling mean and standard deviation
        for i in range(window - 1, len(series)):
            window_values = series[i - window + 1 : i+1]
            #print([window_values[-1], series[i]])
            window_mean = np.mean(window_values)
            window_std = np.std(window_values, ddof=1)

            # Avoid division by zero
            if window_std != 0:
                z_scores[i] = (series.iloc[i] - window_mean) / window_std

        return z_scores
    
    def percentile_zscore(value, data):
        # Calculate the percentile-based z-score for a single value
        
        percentile = np.sum(data < value) / (len(data) - 1)  # Calculate percentile
        zscore = percentile * 2 - 1   # Transform to [-1, 1] range
        return zscore
    
    
    def calculate_zscores_historic(historical_data, initial_in_sample_size = 2):
        # Traiter nan values
        z = historical_data.copy()
        mask = historical_data.isna()
        non_nan_values_sample = historical_data[~mask]
        
        # Ensure there's enough historical data for the initial in-sample calculation
        if len(non_nan_values_sample) < initial_in_sample_size :
            raise ValueError("Not enough historical data for the specified initial in-sample size.")
        
        # Calculate in-sample z-scores for the initial baseline
        
        sample = list(non_nan_values_sample[:initial_in_sample_size])
        out_sample_data = list(non_nan_values_sample[initial_in_sample_size:])

        zscores = [Computation.percentile_zscore(x, np.array(sample)) for x in sample]
        
        # Calculate out-of-sample z-scores for any additional data
        for value in out_sample_data:
            # Update in-sample data with each new data point for out-of-sample calculation
            sample.append(value)
            # Calculate z-score for the new data point
            zscores.append(Computation.percentile_zscore(value, np.array(sample)))
        
        z[~mask] = zscores
        
        return z
    