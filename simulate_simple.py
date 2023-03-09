# Standard library imports
import timeit

# Third party imports
import numpy as np
import matplotlib.pyplot as plt

# Local application imports
from data.generation import generate_data
from vi.cavi import coordinates_ascent
from config_simple import Params

# random seed for testing purposes
np.random.seed(255)

# load parameters
params = Params()

# generate data
indicator_array, cluster_assignments, cluster_means, x, y = \
    generate_data(params)
params.true_assignment = cluster_assignments
if params.include_noise == False:
    data = x
else:
    data = y

# start timer
t_0 = timeit.default_timer()

# CAVI
elbo_final, tau, gamma, phi = coordinates_ascent(data, params)
# end timer

t_1 = timeit.default_timer()
# compute elapsed time
runtime = t_1 - t_0

# postprocessing

# MAP estimate of the cluster assignements
indicator_array_est = np.argmax(phi,axis=1)

# delete empty clusters
cluster_indicators_est = np.unique(indicator_array_est)
T_est = cluster_indicators_est.size

# load parameters
T = params.T
K = params.K
N = params.N
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
