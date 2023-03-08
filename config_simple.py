# Standard library imports
from dataclasses import dataclass

# Third party imports
import numpy as np

@dataclass
class Params:
    # model type
    data_type = "DPM"
    # model specific parameters
    # DPM
    alpha_DPM = 1
    # MFM
    alpha_MFM = 1
    beta_MFM  = 5
    # GM
    num_clusters = 5
    # Load
    filename = "data.npy"
    # data dimension
    K = 2
    # base distribution
    mu_G    = np.zeros(K)
    sigma_G = 5*np.eye(K)
    # mixture distribution 
    mu_U    = np.zeros(K)
    # NOTE: sigma_G must be a scaled version of sigma in our conjugate model
    # where sigma is either sigma_U or sigma_U+sigma_V is noise is included
    sigma_U = 1*np.eye(K)
    # measurement noise
    mu_V    = np.zeros(K)
    sigma_V = np.eye(K)
    include_noise = False
    # number of data points
    N = 50
    # plot data if true
    plot_data = True
    
    # covariance matrix and inverse for CAVI
    sigma = sigma_U+sigma_V if include_noise else sigma_U
    sigma_inv = np.linalg.inv(sigma)
    
    # hyperparameters for VI for DPM 
    lamda = np.empty(K+1)
    lamda1_temp = np.matmul(np.linalg.inv(sigma), sigma_G)
    lamda[-1] = 1/lamda1_temp[0,0]
    lamda[:-1] = lamda[-1]*mu_G
    alpha = 1
    
    # parameters for the algorithm
    init_type = 1
    # true_assignment should be added during runtime
    num_permutations = 30 # only for random permuated initialization
    max_iterations = 100
    T = 20 # truncation