# Standard library imports
import timeit

# Third party imports
import numpy as np
import matplotlib.pyplot as plt

# Local application imports
from data.generation import generate_data
from vi import variational_updates as vu
from vi.elbo import compute_elbo

# TODO: THIS IS WIP

# random seed for testing purposes
# np.random.seed(255) 

# model parameters
K = 2
# hyperparameters of the base distribution
mu_G    = np.zeros(K)
sigma_G = 5*np.eye(K)
# mixture distribution 
mu_U    = np.zeros(K)
# NOTE: sigma_G must be a scaled version of sigma_Y in our conjugate model (see ExpFam_Chap2_page39)
sigma_U = 1*np.eye(K)
# measurement noise
mu_V    = np.zeros(K)
sigma_V = np.eye(K)

# number of mc runs
mcruns = 10
mse_x = np.zeros(mcruns)

# covariance matrix and inverse for CAVI
sigma = sigma_U + sigma_V
sigma_inv = np.linalg.inv(sigma)
# sample size
N = 50

# hyperparameters
lamda = np.empty(K+1)
lamda1_temp = np.matmul(np.linalg.inv(sigma), sigma_G)
lamda[-1] = 1/lamda1_temp[0,0]
lamda[:-1] = lamda[-1]*mu_G
alpha = 1 # concentration parameter - higher alpha more clusters

# max sample size
max_sample_size = 50

# mse array -> max_sample_size x mcruns
mse_x = np.zeros((max_sample_size, mcruns))

# elbo for each sample size rund
elbo_final = np.zeros(max_sample_size)
elbo_final[:] = -np.inf 

# start timer
t_0 = timeit.default_timer()

for n in range(max_sample_size):
    # store current sample size
    N = n+1 
   
    for l in range(mcruns):
        indicator_array, cluster_assignements, cluster_means, x, y = generate_data(N, alpha, mu_G, sigma_G, mu_U, sigma_U, mu_V, sigma_V, False)
        
        # input data for cavi
        data = y
        # initialization
        # NOTE: T has to be higher than the true number of clusters, even when it's known (figure out why) 
        
        # TODO: CALL CAVI HERE AND SAVE FINAL ELBO
               
        # postprocessing
        
        # MAP estimate of the cluster assignements
        estimated_indicator_array = np.argmax(phi,axis=1)
        
        # delete empty clusters
        cluster_indicators = np.unique(estimated_indicator_array)
        T_estimated = cluster_indicators.size
        
        # estimate of the cluster weights
        V = np.divide(gamma[:,0],np.sum(gamma,axis=1))
        pi = np.zeros(np.shape(V))
        pi[0] = V[0]
        for i in range (1,T):
            temp = 1
            for j in range (i):
                    temp = temp*(1-V[j])
            pi[i] = V[i]*temp
        
        # MMSE estimate of the cluster means
        estimated_cluster_means = np.zeros((T,K))   
        estimated_cluster_means = tau[:,:-1]/np.repeat(tau[:,-1,np.newaxis], 1, axis=1)
        estimated_cluster_means_reduced = estimated_cluster_means[cluster_indicators, :]
        
        # Sample mean of the clusters
        # cluster_average = np.zeros((T, K))
        # counts = np.zeros(T)
        # for i in range(N):
        #     cluster_average[estimated_indicator_array[i],:] += data[i]
        #     counts[estimated_indicator_array[i]] += 1
        # cluster_average = cluster_average[cluster_indicators, :]
        # counts = counts[cluster_indicators]
        # counts = np.repeat(counts[:,np.newaxis], K, axis=1)
        # cluster_average = np.divide(cluster_average, counts)
        
        # mmse estimator for objects x
        if not np.array_equiv(sigma,sigma_U):
            data_mean_temp = estimated_cluster_means[estimated_indicator_array,:]
            mmse_weight = np.matmul(sigma_U,sigma_inv)
            weighted_centered_data = np.einsum('ij,kj->ki',mmse_weight,data - data_mean_temp)
            estimated_x = data_mean_temp + weighted_centered_data
            # calculate mse
            mse_x[n,l] = 1/N*np.sum(np.linalg.norm(x - estimated_x,axis=1)**2)
# end timer
t_1 = timeit.default_timer()
# compute elapsed time
runtime = t_1 - t_0

# Plot
# TODO: contour plot of posterior
# plot_cluster_indicators = np.arange(T_estimated)
# mapper = lambda x: np.where(cluster_indicators == x)[0]
# plot_estimated_indicator_array = list(map(mapper, estimated_indicator_array))
# plt.figure()
# plt.title("Clustering - MMSE Mean")
# if T_estimated > 10:
#     print('More clusters than colors')
# colormap = plt.cm.get_cmap('tab20', 10)
# plt.scatter(estimated_cluster_means_reduced[:,0], estimated_cluster_means_reduced[:,1], c = colormap(plot_cluster_indicators), marker = "o")
# plt.scatter(data[:,0], data[:,1], c = colormap(plot_estimated_indicator_array), marker = '.')
# plt.figure()
# plt.title("Clustering - Cluster Sample Mean")   
# plt.scatter(cluster_average[:,0], cluster_average[:,1], c = colormap(plot_cluster_indicators), marker = "o")
# plt.scatter(data[:,0], data[:,1], c = colormap(plot_estimated_indicator_array), marker = '.')
