# Standard library imports
from dataclasses import dataclass

# Third party imports
import numpy as np

@dataclass
class Params:
    # Use noisy DPM data y in this scenario
    data_type = "DPM"
    
    # Concentration parameter
    # alpha_DPM = 1 - set by simulation script
    
    # Data dimension
    K = 2 
    
    # Base distribution
    mu_G    = np.zeros(K)
    sigma_G = 5*np.eye(K)
    
    # Parameter noise
    mu_U    = np.zeros(K)
    # NOTE: sigma_G must be a scaled version of sigma in our conjugate model
    # where sigma is either sigma_U or sigma_U+sigma_V is noise is included
    sigma_U = np.eye(K)
    
    # Measurement noise
    mu_V    = np.zeros(K)
    sigma_V = np.eye(K)
    include_noise = True
    
    # Plot data if true
    plot_data = False
    
    # Covariance matrix and inverse of observations used for CAVI
    sigma = sigma_U+sigma_V if include_noise else sigma_U
    sigma_inv = np.linalg.inv(sigma)
    
    # Number of data points N set by simulation script
    
    # Hyperparameters for CAVI for DPM 
    # Assumed to be known
    lamda = np.empty(K+1)
    lamda1_temp = np.matmul(np.linalg.inv(sigma), sigma_G)
    lamda[-1] = 1/lamda1_temp[0,0]
    lamda[:-1] = lamda[-1]*mu_G
    # alpha = alpha_DPM - set by simulation script
    # If true, parameters for base distribution will be derived from data,
    # i.e., lamdas will be overwritten
    # data_driven_base_dist = True 
    
    # Truncation parameter
    T = 20
    
    # Relative change of ELBO for convergence
    eps = 1e-3
    
    # Max iterations performed if convergence criterion is not met
    max_iterations = 100
    
    # Initiallization type:
    # Uniform   - assign all datapoints equally likely to clusters
    # True      - use true hard assignments
    # Permute   - use random hard assignments
    # Unique    - assign each datapoint to its own cluster from 1 to T
    # AllInOne  - Put all datapoints in one cluster
    # Kmeans    - Use hard assignments of kmeans
    # DBSCAN    - Use hard assignments of dbscan
    init_type = 'Unique'
    # Number of initial permuations used when init_type = 'permute'
    num_permutations = 30