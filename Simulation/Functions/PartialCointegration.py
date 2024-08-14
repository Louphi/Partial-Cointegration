

# Import library
import numpy as np
import statsmodels.api as sm
from scipy import stats
import scipy.optimize as opt


class partial_cointegration:

    def __init__(self, X1, X2):
        # Our two time-series to fit the PCI model to
        self.X1 = X1
        self.X2 = X2

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

        rho, sigma_M, sigma_R = params_i[0]

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

    '''    


     # perform optimization depending on the mode (Complete Model)
        x_i = (alpha_i, beta_i, rho, sigma_M, sigma_R)  # initial guess
        bounds = ((-np.inf, np.inf), (-np.inf, np.inf), (-1, 1), (0, np.inf), (0, np.inf))  # Set bounds for opt.
        res = opt.minimize(self.f_to_min_pci, x_i, args=(self.X1, self.X2), bounds=bounds, tol=tol)
        alpha, beta, rho, sigma_M, sigma_R = res.x
        ll_model = -res.fun
        rho, sigma_M, sigma_R = params_i[1]

        # perform optimization for the random Walk H0
        x_i = (alpha_i, beta_i, rho, sigma_M, sigma_R)  # initial guess
        bounds = ((-np.inf, np.inf), (-np.inf, np.inf), (0, 0), (0, 0), (0, np.inf))  # Set bounds for opt.
        res = opt.minimize(self.f_to_min_pci, x_i, args=(self.X1, self.X2), bounds=bounds, tol=tol)
        alpha, beta, rho, sigma_M, sigma_R = res.x
        ll_randomWalk = -res.fun
        W = self.X2 - alpha - beta * self.X1
    '''

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
        estimates_mr = []
        lls_mr = []

        estimates_rw = []
        lls_rw = []

        # Set distributions for random guesses
        rnd_rho = stats.uniform(loc=-1, scale=2)
        std = np.std(np.diff(W))  # Compute standard deviation of "pci process" (or residuals)
        rnd_sigma = stats.norm(loc=std,
                               scale=std / 2)  # Why scaled to std / 2? Can we find better? ####################

        # Define function for generating random initial guesses from above distributions
        def gen_x_i():
            return rnd_rho.rvs(), rnd_sigma.rvs(), rnd_sigma.rvs()

        # Minimize the negative log-likelihood function (regular model)
        bounds_mr = ((-1, 1), (0, np.inf), (0, np.inf))  # Set bounds for opt.
        x_i = self.lagvar_estimate_par(W)  # Get initial guesses using lagged variance equations
        res_mr = opt.minimize(self.f_to_min_par, x0=(x_i), args=(W), bounds=bounds_mr, tol=tol)  # Minimize -ll function

        if res_mr.success:  # If optimization is a success
            estimates_mr.append(res_mr.x)  # Save estimates in list
            lls_mr.append(-res_mr.fun)  # Save log-likelihood in list

        # Minimize the negative log-likelihood function (RW model H0)
        bounds_rw = ((0, 0), (0, 0), (0, np.inf))  # Set bounds for opt.
        res_rw = opt.minimize(self.f_to_min_par, x0=(x_i), args=(W), bounds=bounds_rw, tol=tol)  # Minimize -ll function

        if res_rw.success:  # If optimization is a success
            estimates_rw.append(res_rw.x)  # Save estimates in list
            lls_rw.append(-res_rw.fun)  # Save log-likelihood in list

        # Repeat optimization with different random initial values
        n_att = 0
        while len(lls_mr) < 10 and n_att < 100:
            n_att += 1
            x_i = gen_x_i()
            res_mr = opt.minimize(self.f_to_min_par, x0=(x_i), args=(W), bounds=bounds_mr, tol=tol)
            res_rw = opt.minimize(self.f_to_min_par, x0=(x_i), args=(W), bounds=bounds_rw, tol=tol)

            if res_mr.success:  # If MR optimization is a success
                estimates_mr.append(res_mr.x)  # Save estimates
                lls_mr.append(-res_mr.fun)  # Save log-likelihood

            if res_rw.success:  # If RW optimization is a success
                estimates_rw.append(res_rw.x)  # Save estimates
                lls_rw.append(-res_rw.fun)  # Save log-likelihood

        try:
            argmax_mr = np.argmax(lls_mr)  # Find index of min -likelihood mr WHY NOT ARGMIN?
            argmax_rw = np.argmax(lls_rw)  # Find index of min -likelihood rw
            return [estimates_mr[argmax_mr], estimates_rw[argmax_rw]]  # Return estimates linked to max likelihood

        except:
            # print('Estimation failed!')
            return len(x_i) * [np.nan]  # Returns nans

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
        results.params[0]: returns the Beta value of our OLS fits
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
        if sigma_R == 0:  # If series is purely mean-reverting
            M[0] = W[0]
            R[0] = 0
        else:
            M[0] = 0
            R[0] = W[0]

        # Calculate Kalman gains
        if sigma_M == 0:  # If series is purely a random walk
            K_M = 0
            K_R = 1
        elif sigma_R == 0:  # If series is purely mean-reverting
            K_M = 1
            K_R = 0
        else:
            # See equations (11), page 8
            sqr = np.sqrt((1 + rho) ** 2 * sigma_R ** 2 + 4 * sigma_M ** 2)
            K_M = 2 * sigma_M ** 2 / (
                        sigma_R * (sqr + rho * sigma_R + sigma_R) + 2 * sigma_M ** 2)  # Compute gain for M
            K_R = 2 * sigma_R / (sqr - rho * sigma_R + sigma_R)  # Compute gain for R

        # Calculate estimates recursively
        for i in range(1, len(W)):
            w_hat = rho * M[i - 1] + R[i - 1]  # Predicted value of W[i]
            eps[i] = W[i] - w_hat  # Prediction error
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
        _, _, eps = self.kalman_estimate(W, rho, sigma_M, sigma_R)  # Compute the prediction errors for each time step

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
        if len(x_i) == 5:
            alpha, beta, rho, sigma_M, sigma_R = x_i

        elif len(x_i) == 3:
            alpha, beta, sigma_R = x_i
            rho = 0
            sigma_M = 0

        W = X2 - beta * X1 - alpha

        return -self.calc_log_like(W, rho, sigma_M, sigma_R)
