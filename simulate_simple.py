# Standard library imports
import timeit

# Third party imports
import numpy as np
import matplotlib.pyplot as plt

# Local application imports
from data.generation import generate_data_rp, generate_data_gm
from data import restaurant_process
from vi.cavi import coordinates_ascent, initParams

# random seed for testing purposes
np.random.seed(255) 

# model parameters
# data dimension
K = 2
# base distribution
mu_G    = np.zeros(K)
sigma_G = 5*np.eye(K)
# mixture distribution 
mu_U    = np.zeros(K)
# NOTE: sigma_G must be a scaled version of sigma_Y in our conjugate model
sigma_U = 1*np.eye(K)
# measurement noise
mu_V    = np.zeros(K)
sigma_V = np.eye(K)
# number of data points
N = 50

# covariance matrix and inverse for CAVI
sigma = sigma_U 
sigma_inv = np.linalg.inv(sigma)

# generate data
data_type = "DPM"
plot_data = True
if data_type == "DPM":
    alpha_DPM = 1 # concentration parameter - higher alpha more clusters
    indicator_array, cluster_assignments, cluster_means, data, _ = \
        generate_data_rp(N, alpha_DPM, mu_G, sigma_G, mu_U, sigma_U, mu_V, sigma_V,
                      restaurant_process.rp_dpm, plot_data)
elif data_type == "MFM":
    alpha_MFM = 5 # kind of concentration parameter - higher alpha more clusters
    indicator_array, cluster_assignments, cluster_means, data, _ = \
        generate_data_rp(N, alpha_MFM, mu_G, sigma_G, mu_U, sigma_U, mu_V, sigma_V,
                      restaurant_process.rp_mfm, plot_data) 
elif data_type == "GM":
    num_clusters = 5
    indicator_array, cluster_assignments, cluster_means, data, _ = \
        generate_data_gm(N, num_clusters, mu_G, sigma_G, mu_U, sigma_U, mu_V,
                         sigma_V, plot_data)
elif data_type == "load":
    filename = "data.npy"
    data = np.load(filename)
    
# hyperparameters - assumed to be known
lamda = np.empty(K+1)
lamda1_temp = np.matmul(np.linalg.inv(sigma), sigma_G)
lamda[-1] = 1/lamda1_temp[0,0]
lamda[:-1] = lamda[-1]*mu_G
alpha = 1 # concentration parameter - higher alpha more clusters
    
# parameters for the algorithm
init_version = 1
num_permutations = 30 # for init version 3
init_params = initParams(init_version, cluster_assignments, num_permutations)
max_iteration = 100
T = 20 # truncation

# start timer
t_0 = timeit.default_timer()
# CAVI
elbo_final, tau, gamma, phi = \
    coordinates_ascent(data, max_iteration, init_params, alpha,\
                       sigma, sigma_inv, mu_G, sigma_G, lamda, T)
# end timer
t_1 = timeit.default_timer()

# postprocessing
# compute elapsed time
runtime = t_1 - t_0

# MAP estimate of the cluster assignements
indicator_array_est = np.argmax(phi,axis=1)

# delete empty clusters
cluster_indicators_est = np.unique(indicator_array_est)
T_est = cluster_indicators_est.size

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
cluster_means_est = np.zeros((T,K))   
cluster_means_est = tau[:,:-1]/np.repeat(tau[:,-1,np.newaxis], 1, axis=1)
cluster_means_est = cluster_means_est[cluster_indicators_est, :]

# Sample mean of the clusters
cluster_average = np.zeros((T, K))
counts = np.zeros(T)
for i in range(N):
    cluster_average[indicator_array_est[i],:] += data[i]
    counts[indicator_array_est[i]] += 1
cluster_average = cluster_average[cluster_indicators_est, :]
counts = counts[cluster_indicators_est]
counts = np.repeat(counts[:,np.newaxis], K, axis=1)
cluster_average = np.divide(cluster_average, counts)

# Plot
# TODO: contour plot of posterior
# map estimated cluster indicators to a range from 0 to T_est-1 for plotting
plot_cluster_indicators = np.arange(T_est)
mapper = lambda i: np.where(cluster_indicators_est == i)[0]
plot_indicator_array_est = list(map(mapper, indicator_array_est))
# Scatter plot with MMSE mean of clusters
plt.figure()
plt.title("Clustering DPM - MMSE Mean")
if T_est > 10:
    print('More clusters than colors')
colormap = plt.cm.get_cmap('tab20', 10)
cx, cy = cluster_means_est[:,0], cluster_means_est[:,1]
plt.scatter(cx, cy, c=colormap(plot_cluster_indicators), marker="o")
da, dy = data[:,0], data[:,1]
plt.scatter(da, dy, c=colormap(plot_indicator_array_est), marker='.')
# Scatter plot with sample mean of clusters
plt.figure()
plt.title("Clustering DPM - Cluster Sample Mean") 
cx, cy = cluster_average[:,0], cluster_average[:,1]   
plt.scatter(cx, cy, c = colormap(plot_cluster_indicators), marker="o")
da, dy = data[:,0], data[:,1]
plt.scatter(da, dy, c=colormap(plot_indicator_array_est), marker='.')
