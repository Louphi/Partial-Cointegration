import pandas as pd
import MonteCarloEstimation as monte_carlo
import PartialCointegration as pci
import sys
sys.path.append('/Users/sebastiencaron/Desktop/PCI-Project/Simulation')

from importlib import reload
import PCISimulation as simul
import matplotlib.pyplot as plt
import SensitivityAnalysis as SA
import pickle


reload(pci)
reload(simul)

reload(SA)
reload(monte_carlo)

from datetime import datetime
import numpy as np
import warnings
# Suppress all warnings
warnings.filterwarnings('ignore')


n_sample = [100, 1000, 10000]
n_sim = 10000
alpha = 0
beta = 1
rho = 0.9
sigma_R = 1
sigma_M = 1
sigma_X = 0.0236
params = [alpha, beta, rho, sigma_M, sigma_R, sigma_X] #N, alpha, beta, rho, sigma_R, sigma_X
# On enleve sigma_M
fixed_params_SigmaM = [alpha, beta, rho, sigma_R, sigma_X]
fixed_params_rho = [alpha, beta, sigma_M, sigma_R, sigma_X]

result_dict = SA.SensitivityAnalysis.f_sensitivity_analysis(fixed_params = fixed_params_rho, var_param ="rho", n_simulation = n_sim, n = n_sample)

with open(r"Outputs_parameters/resultRho.pickle", "wb") as output_file:
    pickle.dump(result_dict, output_file)