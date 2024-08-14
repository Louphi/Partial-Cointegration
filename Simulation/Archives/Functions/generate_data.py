import numpy as np
from statsmodels.tsa.arima_process import arma_generate_sample


class pci_generator:
    def __init__(self, N, alpha, beta, rho, sigma_M, sigma_R, sigma_X):
        self.N = N
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.sigma_M = sigma_M
        self.sigma_R = sigma_R
        self.sigma_X = sigma_X

    def f_generate_pci(self):
        '''
        Generate two partially cointegrated time series of length N with given parameters, such that:
            - X_2,t = alpha + beta * X_1,t + W_t
            - W_t = M_t + R_t
            - M_t = rho * M_t-1 + eps(M_t)
            - R_t = R_t-1 + eps(R_t)
            - eps(M_t) ∼ NID(0, sigma_M)
            - eps(R_t) ∼ NID(0, sigma_R)

        Parameters:
        N (int): Length of the time series to generate.
        alpha (float): intercept
        beta (float): cointegration coefficient.
        rho (float): AR(1) coefficient / mean reversion coefficient.
        sigma_M (float): white noise variance of mean reverting component .
        sigma_R (float): white noise variance of random walk component.
        sigma_X (float): white noise variance of X1.

        Returns:
        tuple: A tuple containing two cointegrated time series, X1 and X2, which are numpy.ndarray.
        '''

        ret_X1 = self.sigma_X * np.random.randn(self.N)
        X1 = 100 * np.exp(np.cumsum(ret_X1))  # generate X1
        W, M, R = self.f_generate_par()  # generate PAR residual
        X2 = self.alpha + self.beta * X1 + W  # compute X2

        return X1, X2, W, M, R

    def f_generate_par(self):
        '''
        Generate PAR sample of length N with parameters rho, sigma_M, sigma_R

        Parameters:
        N (int): Length of the time series to generate.
        rho (float): AR(1) coefficient / mean reversion coefficient.
        sigma_M (float): white noise variance of mean reverting component .
        sigma_R (float): white noise variance of random walk component.

        Returns:
        numpy.ndarray: A partial autoregressive time series
        '''

        ar_M = [1, -self.rho]
        ar_R = [1, -1]
        ma = [1]

        M = arma_generate_sample(ar_M, ma, self.N, scale=self.sigma_M)
        R = arma_generate_sample(ar_R, ma, self.N, scale=self.sigma_R)
        S = M + R
        return S, M, R
