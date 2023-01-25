import numpy as np
import timeit
import matplotlib.pyplot as plt
#import scipy as sp
#from scipy.special import digamma
from data_generation import generate_data
import variational_updates as vu
from elbo import compute_elbo

# random seed for testing purposes
np.random.seed(255) 

# model parameters
K = 2
# hyperparameters of the base distribution
mu_G    = np.zeros(K)
sigma_G = 5*np.eye(K)
# mixture distribution 
mu_U    = np.zeros(K)
# TODO: sigma_G must be a scaled version of sigma_Y in our conjugate model -> clarify this (see ExpFam_Chap2_page39)
sigma_U = 2*np.eye(K)
# measurement noise
mu_V    = np.zeros(K)
sigma_V = np.eye(K)

# covariance matrix and inverse for CAVI
sigma = sigma_U
sigma_inv = np.linalg.inv(sigma)
# sample size
N = 50

# hyperparameters
lamda = np.empty(K+1)
lamda1_temp = np.matmul(np.linalg.inv(sigma), sigma_G)
lamda[-1] = 1/lamda1_temp[0,0]
lamda[:-1] = lamda[-1]*mu_G
alpha = 1 # concentration parameter - higher alpha more clusters

# generate data
indicator_array, cluster_means, x, y = generate_data(N, alpha, mu_G, sigma_G, mu_U, sigma_U, mu_V, sigma_V, True)
T_true = cluster_means.shape[0]
true_assignment = np.zeros((N,T_true))
for i in range(N):
    true_assignment[i,indicator_array[i]] = 1

# input data for cavi
data = x
# initialization
# NOTE: T has to be higher than the true number of clusters, even when it's known (figure out why) 
T = 30
iterations = 500
elbo_final = -np.inf
phi_init_version = 1
if phi_init_version == 1:
    phi_init = 1/T * np.ones((N,T))
    num_permutations = 1
elif phi_init_version == 2:
    phi_init = np.zeros((N,T))
    phi_init[:,:T_true] = true_assignment
    num_permutations = 1
elif phi_init_version == 3:
    np.random.seed(1337)
    num_permutations = 20
    rand_indicators = [np.random.randint(0,T,N) for i in range(num_permutations)]
    phi_init = np.zeros((N,T))
elif phi_init_version == 4:
    T = N
    phi_init = np.eye(N)
    num_permutations = 1
elif phi_init_version == 5:
    num_permutations = T
    rand_indicators = [i*np.ones(T) for i in range(num_permutations)]
    phi_init = np.zeros((N,T))

    
# start timer
t_0 = timeit.default_timer()

# variational updates
elbo = np.zeros(iterations)
for j in range(num_permutations):
    if phi_init_version == 3:
        for k in range(N):
            phi_init[k,rand_indicators[j][k]] = 1  
    gamma_temp = vu.update_gamma(phi_init,alpha)
    tau_temp = vu.update_tau(data,lamda,phi_init)
    for i in range(iterations):
        # TODO: save variational parameters and investigate convergence of parameters (instead of ELBO)
        # TODO: compute all necessary expectations with functions
        # compute variational updates
        phi_temp = vu.update_phi(data, gamma_temp, tau_temp, lamda, sigma, sigma_inv)
        gamma_temp = vu.update_gamma(phi_temp, alpha)
        tau_temp = vu.update_tau(data, lamda, phi_temp)
        # compute elbo
        elbo[i] = compute_elbo(alpha, lamda, data, gamma_temp, phi_temp, tau_temp, sigma, mu_G, sigma_G, sigma_inv)
        if i>0 and np.abs(elbo[i]-elbo[i-1]) < 0.1:
            break
        
    if elbo[i] > elbo_final:
        elbo_final = elbo[i]
        tau = tau_temp
        gamma = gamma_temp
        phi = phi_temp

# end timer
t_1 = timeit.default_timer()

# postprocessing
# compute elapsed time
runtime = t_1 - t_0

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
estimated_cluster_means = estimated_cluster_means[cluster_indicators, :]

# Sample mean of the clusters
cluster_average = np.zeros((T, K))
counts = np.zeros(T)
for i in range(N):
    cluster_average[estimated_indicator_array[i],:] += data[i]
    counts[estimated_indicator_array[i]] += 1
cluster_average = cluster_average[cluster_indicators, :]
counts = counts[cluster_indicators]
counts = np.repeat(counts[:,np.newaxis], K, axis=1)
cluster_average = np.divide(cluster_average, counts)

# Plot
# TODO: contour plot of posterior
plot_cluster_indicators = np.arange(T_estimated)
mapper = lambda x: np.where(cluster_indicators == x)[0]
plot_estimated_indicator_array = list(map(mapper, estimated_indicator_array))
plt.figure()
plt.title("Clustering - MMSE Mean")
if T_estimated > 10:
    print('More clusters than colors')
colormap = plt.cm.get_cmap('tab20', 10)
plt.scatter(estimated_cluster_means[:,0], estimated_cluster_means[:,1], c = colormap(plot_cluster_indicators), marker = "o")
plt.scatter(data[:,0], data[:,1], c = colormap(plot_estimated_indicator_array), marker = '.')
plt.figure()
plt.title("Clustering - Cluster Sample Mean")   
plt.scatter(cluster_average[:,0], cluster_average[:,1], c = colormap(plot_cluster_indicators), marker = "o")
plt.scatter(data[:,0], data[:,1], c = colormap(plot_estimated_indicator_array), marker = '.')
