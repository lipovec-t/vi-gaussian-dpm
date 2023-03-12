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
results, results_reduced = pp.full_postprocessing(data, phi, gamma, tau, False)


# Plot
T_est = results_reduced["Estimated Cluster Weights"].size
plot_cluster_indicators = np.arange(T_est)
cluster_indicatores_est = results_reduced["Estimated Cluster Indicators"]
# Scatter plot with MMSE mean of clusters
plt.figure()
plt.title("Clustering DPM - MMSE Mean")
if T_est > 10:
    print('More clusters than colors')
colormap = plt.cm.get_cmap('tab20', 10)
cx = results_reduced["Estimated Cluster Means"][:,0]
cy = results_reduced["Estimated Cluster Means"][:,1]
plt.scatter(cx, cy, c=colormap(plot_cluster_indicators), marker="o")
da, dy = data[:,0], data[:,1]
plt.scatter(da, dy, c=colormap(cluster_indicatores_est), marker='.')
# Scatter plot with sample mean of clusters
plt.figure()
plt.title("Clustering DPM - Cluster Sample Mean") 
cx = results_reduced["Sample Mean of Clusters"][:,0]
cy = results_reduced["Sample Mean of Clusters"][:,1]  
plt.scatter(cx, cy, c = colormap(plot_cluster_indicators), marker="o")
da, dy = data[:,0], data[:,1]
plt.scatter(da, dy, c=colormap(cluster_indicatores_est), marker='.')
