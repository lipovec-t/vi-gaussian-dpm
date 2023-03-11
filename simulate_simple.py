# Standard library imports
import timeit

# Third party imports
import numpy as np
import matplotlib.pyplot as plt

# Local application imports
from data.generation import generate_data
from vi.cavi import coordinates_ascent
from vi import postprocessing as pp
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

# end timer and compute elapsed time
t_1 = timeit.default_timer()
runtime = t_1 - t_0

# postprocessing

# number of mixture componentes in the fitted model
T = params.T

# MAP estimate of the cluster assignements
cluster_indicator_est = pp.est_clustering_map(phi)

# MMSE estimate of the cluster weights
cluster_weights_est = pp.est_cluster_weights_mmse(gamma)

# MMSE estimate of the cluster means 
cluster_means_est = pp.est_cluster_means_mmse(tau)

# Sample mean of the clusters
cluster_sample_mean, cluster_sample_weight, _ = \
    pp.cluster_sample_mean(data, cluster_indicator_est, T)

# Reduce results
results = (cluster_indicator_est, cluster_weights_est, cluster_means_est,\
           cluster_sample_mean)
results_reduced = pp.reduce_results(results)

# Plot
T_est = results_reduced[1].size
plot_cluster_indicators = np.arange(T_est)
# Scatter plot with MMSE mean of clusters
plt.figure()
plt.title("Clustering DPM - MMSE Mean")
if T_est > 10:
    print('More clusters than colors')
colormap = plt.cm.get_cmap('tab20', 10)
cx, cy = results_reduced[2][:,0], results_reduced[2][:,1]
plt.scatter(cx, cy, c=colormap(plot_cluster_indicators), marker="o")
da, dy = data[:,0], data[:,1]
plt.scatter(da, dy, c=colormap(results_reduced[0]), marker='.')
# Scatter plot with sample mean of clusters
plt.figure()
plt.title("Clustering DPM - Cluster Sample Mean") 
cx, cy = results_reduced[3][:,0], results_reduced[3][:,1]   
plt.scatter(cx, cy, c = colormap(plot_cluster_indicators), marker="o")
da, dy = data[:,0], data[:,1]
plt.scatter(da, dy, c=colormap(results_reduced[0]), marker='.')
