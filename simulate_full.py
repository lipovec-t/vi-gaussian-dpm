# Standard library imports
import timeit

# Third party imports
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st

# Local application imports
from data.generation import generate_data
from vi.cavi import coordinates_ascent

# random seed for testing purposes
np.random.seed(255) 

# model parameters
K = 2
# hyperparameters of the base distribution
mu_G    = np.zeros(K)
sigma_G = 5*np.eye(K)
# mixture distribution 
mu_U    = np.zeros(K)
# NOTE: sigma_G must be a scaled version of sigma_Y in our conjugate model
sigma_U = 1*np.eye(K)
# measurement noise
mu_V    = np.zeros(K)
sigma_V = np.eye(K)
# data points to simulate
N_array = np.arange(1,51)

# covariance matrix and inverse for CAVI
sigma = sigma_U + sigma_V
sigma_inv = np.linalg.inv(sigma)

# hyperparameters
lamda = np.empty(K+1)
lamda1_temp = np.matmul(np.linalg.inv(sigma), sigma_G)
lamda[-1] = 1/lamda1_temp[0,0]
lamda[:-1] = lamda[-1]*mu_G
alpha = 1 # concentration parameter - higher alpha more clusters

# parameters for the algorithm
phi_init_version = 1
max_iteration = 100
T = 20 # truncation

# number of MC runs
MC_runs = 500

# MSE array -> max_N  x MC_runs
MSE_x = np.zeros((N_array.size, MC_runs))

# elbo for each sample size run
elbo_final = np.zeros(N_array.size)
elbo_final[:] = -np.inf 

# start timer
t_0 = timeit.default_timer()

for i,N in enumerate(N_array):
    for j in range(MC_runs):
        # generate data
        indicator_array, cluster_assignements, cluster_means, x, data = \
            generate_data(N, alpha, mu_G, sigma_G, mu_U, sigma_U, mu_V, sigma_V, False)
        
        # CAVI
        elbo, tau, gamma, phi = \
            coordinates_ascent(data, max_iteration, phi_init_version, alpha,\
                               sigma, sigma_inv, mu_G, sigma_G, lamda, T)
        if elbo > elbo_final[i]:
            elbo_final[i] = elbo
               
        # postprocessing
        # MAP estimate of the cluster assignements
        indicator_array_est = np.argmax(phi,axis=1)
        
        # delete empty clusters
        cluster_indicators_est = np.unique(indicator_array_est)
        T_est = cluster_indicators_est.size
        
        # estimate of the cluster weights
        V = np.divide(gamma[:,0],np.sum(gamma,axis=1))
        pi = np.zeros(np.shape(V))
        pi[0] = V[0]
        for k in range (1,T):
            temp = 1
            for l in range (k):
                    temp = temp*(1-V[l])
            pi[k] = V[l]*temp
        
        # MMSE estimate of the cluster means
        cluster_means_est = np.zeros((T,K))   
        cluster_means_est = tau[:,:-1]/np.repeat(tau[:,-1,np.newaxis], 1, axis=1)
        cluster_means_est_reduced = cluster_means_est[cluster_indicators_est, :]
        
        # mmse estimator for objects x
        data_mean_temp = cluster_means_est[indicator_array_est,:]
        mmse_weight = np.matmul(sigma_U, sigma_inv)
        weighted_centered_data = np.einsum('ij,kj->ki', mmse_weight, data - data_mean_temp)
        x_est = data_mean_temp + weighted_centered_data
        # calculate mse
        MSE_x[i,j] = 1/N * np.sum(np.linalg.norm(x - x_est, axis=1)**2)
        
# end timer
t_1 = timeit.default_timer()

# compute elapsed time
runtime = t_1 - t_0

# mean MSE
MSE_x_avg = np.mean(MSE_x, axis=1)
# 95% confidence interval
(ci_min, ci_max) = st.t.interval(alpha=0.95, df=MSE_x.shape[1]-1, loc=MSE_x_avg, scale=st.sem(MSE_x, axis=1)) 
# Plot
params = {"text.usetex" : True,
          "font.family" : "serif",
          "font.size"   : "16"}
plt.rcParams.update(params)
fig, ax = plt.subplots()
ax.plot(N_array, MSE_x_avg, color='b', label=r'MSE VI $\alpha = {}$'.format(alpha))
ax.fill_between(N_array, ci_min, ci_max, color='b', alpha=.1, label=r'$95\%$ CI')
plt.xlabel('Number of objects')
plt.ylabel('Average MSE')
plt.legend()
plt.grid()
