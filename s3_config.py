# Standard library imports
from dataclasses import dataclass

# Third party imports
import numpy as np

@dataclass
class Params:
    # Model type:
    # DPM   - Gaussian Dirchlet Process Mixture
    # MFM   - Gaussian Mixture of finite Mixtures
    # GM    - Gaussian Mixture with fixed number of clusters
    # load  - Load data set from file
    data_type = "GM"
    # model specific parameters
    # DPM
    alpha_DPM = 1
    # MFM
    alpha_MFM = 2
    beta_MFM  = 5
    # GM
    cluster_means_GM = np.array([[-6,-2.5],[-6,2.5],[6,2.5],[6,-2.5],\
                              [-2,-2.5],[-2,2.5],[2,2.5],[2,-2.5]])
    weights_GM = np.ones(cluster_means_GM.shape[0]) / cluster_means_GM.shape[0]
    if cluster_means_GM.size != 0:
        num_clusters_GM = cluster_means_GM.shape[0]
    else:
        num_clusters_GM = weights_GM.size
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
    N = 400
    # plot data if true
    plot_data =False
    
    # covariance matrix and inverse for CAVI
    sigma = sigma_U+sigma_V if include_noise else sigma_U
    sigma_inv = np.linalg.inv(sigma)
    
    # hyperparameters for VI for DPM 
    lamda = np.empty(K+1)
    lamda1_temp = np.matmul(np.linalg.inv(sigma), sigma_G)
    lamda[-1] = 1/lamda1_temp[0,0]
    lamda[:-1] = lamda[-1]*mu_G
    alpha = 1.4
    
    # parameters for the algorithm
    # Init type:
    # Uniform   - assign all datapoints equally likely to clusters
    # True      - use true hard assignments
    # Permute   - use random hard assignments
    # Unique    - assign each datapoint to its own cluster from 1 to T
    # AllInOne  - Put all datapoints in one cluster
    # Kmeans    - Use hard assignments of kmeans
    # DBSCAN    - Use hard assignments of dbscan
    init_type = 'permute'
    # true_assignment should be added during runtime
    num_permutations = 10 # only for random permuated initialization
    max_iterations = 100
    T = 20 # truncation