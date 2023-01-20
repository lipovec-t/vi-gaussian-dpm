import numpy as np
import matplotlib.pyplot as plt
#import scipy as sp
#from scipy.special import digamma
from data_generation import generate_data
import variational_updates as vu
from elbo import compute_elbo

# random seed for testing purposes
np.random.seed(1)

# model parameters
K = 2
# hyperparameters of the base distribution
mu_G    = np.zeros(K)
sigma_G = 5*np.eye(K)
# mixture distribution 
mu_U    = np.zeros(K)
# TODO: sigma_G must be a scaled version of sigma_U in our conjugate model -> clarify this (see ExpFam_Chap2_page39)
sigma_U = 2*np.eye(K)
# measurement noise
mu_V    = np.zeros(K)
sigma_V = np.eye(K)

# sample size
N = 40

# hyperparameters
lamda = np.empty(K+1)
lamda1_temp = np.matmul(np.linalg.inv(sigma_U), sigma_G)
lamda[-1] = 1/lamda1_temp[0,0]
lamda[:-1] = lamda[-1]*mu_G
alpha = 0.8 # concentration parameter - higher alpha more clusters

# generate data
indicator_array, cluster_means, x, y = generate_data(N, alpha, mu_G, sigma_G, mu_U, sigma_U, mu_V, sigma_V, True)
T_true = cluster_means.shape[0]
true_assignment = np.zeros((N,T_true))
for i in range(N):
    true_assignment[i,indicator_array[i]] = 1

# initialization
# NOTE: T has to be higher than the true number of clusters, even when it's known (figure out why) 
T = 30
iterations = 500
phi_init_version = 4
if phi_init_version == 1:
    phi = 1/T * np.ones((N,T))
    num_permutations = 1
elif phi_init_version == 2:
    phi = np.zeros((N,T))
    phi[:,:T_true] = true_assignment
    num_permutations = 1
elif phi_init_version == 3:
    np.random.seed(1337)
    num_permutations = 10
    rand_indicators = [np.random.randint(0,T,N) for i in range(num_permutations)]
    phi = np.zeros((N,T))
    for i in range(N):
        phi[i,rand_indicators[1][i]] = 1
elif phi_init_version == 4:
    T = N
    phi = np.eye(N)

gamma = vu.update_gamma(phi,alpha)
tau = vu.update_tau(x,lamda,phi)

# variational updates
elbo = np.zeros(iterations)
for i in range(iterations):
    # TODO: save variational parameters and investigate convergence of parameters (instead of ELBO)
    # TODO: compute all necessary expectations with functions
    # compute variational updates
    phi = vu.update_phi(x, gamma, tau, lamda, sigma_U)
    gamma = vu.update_gamma(phi,alpha)
    tau = vu.update_tau(x,lamda,phi)
    # compute elbo
    elbo[i] = compute_elbo(alpha, lamda, x, gamma, phi, tau, sigma_U, mu_G, sigma_G)
    if i>0 and np.abs(elbo[i]-elbo[i-1]) < 0.001:
        break

# postprocessing
# TODO: finish this, include RMSE of cluster means, adapt plots

# MAP estimate of the cluster assignements
estimated_indicator_array = np.argmax(phi,axis=1)

# delete empty clusters
cluster_indicators = np.unique(estimated_indicator_array)

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
for t in range(T):
    estimated_cluster_means[t,:] = tau[t,:-1]/tau[t,-1]

# Sample mean of the clusters
cluster_average = np.zeros((T, K))
counts = np.zeros(T)
for i in range(N):
    cluster_average[estimated_indicator_array[i],:] += x[i]
    counts[estimated_indicator_array[i]] += 1
counts = np.repeat(counts[:,np.newaxis], K, axis=1)
cluster_average = np.divide(cluster_average, counts)

# Plot
plt.figure()
plt.title("Clustering - MMSE Mean")   
colormap = plt.cm.get_cmap('tab20', T)
plt.scatter(estimated_cluster_means[:,0], estimated_cluster_means[:,1], c = colormap(range(T)), marker = "o")
plt.scatter(x[:,0], x[:,1], c = colormap(estimated_indicator_array), marker = '.')
plt.figure()
plt.title("Clustering - Cluster Sample Mean")   
colormap = plt.cm.get_cmap('tab20', T)
plt.scatter(cluster_average[:,0], cluster_average[:,1], c = colormap(range(T)), marker = "o")
plt.scatter(x[:,0], x[:,1], c = colormap(estimated_indicator_array), marker = '.')
# plt.xlim([-3, 5])
# plt.ylim([-2, 4])
