# Standard library imports
from dataclasses import dataclass

# Third party imports
import numpy as np

################################
# TEMPLATE FOR THE CONFIG FILE #
# SHOWS ALL POSSIBLE OPTIONS   #
################################

@dataclass
class Params:
    # Model type of observed data:
    # DPM   - Gaussian Dirchlet Process Mixture
    # MFM   - Gaussian Mixture of finite Mixtures
    # GM    - Gaussian Mixture with fixed number of clusters
    # load  - Load data set from file
    data_type = "DPM"
    
    # Model specific parameters - applied according to selected data type -----
    # Used to generate the data input for the CAVI algorithm
    
    # DPM
    alpha_DPM = 1
    
    # MFM
    alpha_MFM = 5
    beta_MFM  = 1
    
    # GM
    weights_GM       = np.ones(4) / 4
    cluster_means_GM = np.array([])
    # if empty mu_G and sigma_G is used to generate means
    if cluster_means_GM.size != 0:
        num_clusters_GM = cluster_means_GM.shape[0]
    else:
        num_clusters_GM = weights_GM.size
   
    # Load
    filename = "data.npy"
    
    # Data dimension
    K = 2
    
    # Base distribution (prior for the parameters for the means)
    mu_G    = np.zeros(K)
    sigma_G = 5*np.eye(K)
    
    # Parameter noise 
    mu_U    = np.zeros(K)
    # NOTE: sigma_G must be a scaled version of sigma in our conjugate model
    # where sigma is either sigma_U or sigma_U+sigma_V if noise is included
    sigma_U = np.eye(K)
    
    # Measurement noise
    mu_V    = np.zeros(K)
    sigma_V = np.eye(K)
    
    # Covariance matrix and inverse of observations used for CAVI
    include_noise = False
    sigma = sigma_U+sigma_V if include_noise else sigma_U
    sigma_inv = np.linalg.inv(sigma)
    # End model specific parameters -------------------------------------------
    
    # Number of data points
    N = 50
    
    # Plot observed data if true (only for K = 2)
    plot_data = True
    
    # Hyperparameters for CAVI for DPM 
    # Here assumed to be know, i.e, calculated from the model statistics,
    # but can also be set arbitrarily
    lamda = np.empty(K+1)
    lamda1_temp = np.matmul(np.linalg.inv(sigma), sigma_G)
    lamda[-1] = 1/lamda1_temp[0,0]
    lamda[:-1] = lamda[-1]*mu_G
    alpha = 1
    
    # Truncation parameter
    T = 20
    
    # Relative change of ELBO for convergence
    eps = 1e-2
    
    # Max iterations performed if convergence criterion is not met
    max_iterations = 100
    
    # Initiallization type:
    # Uniform   - assign all datapoints equally likely to clusters
    # True      - use true hard assignments (Not possible with load_data)
    # Permute   - use random hard assignments
    # Unique    - assign each datapoint to its own cluster from 1 to T
    # AllInOne  - Put all datapoints in one cluster
    # Kmeans    - Use hard assignments of kmeans
    # DBSCAN    - Use hard assignments of dbscan
    init_type = 'Uniform'
    # Number of initial permuations used when init_type = 'permute'
    num_permutations = 30