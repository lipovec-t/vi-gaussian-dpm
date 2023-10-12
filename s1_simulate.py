# Standard library imports
import os

# Third party imports
import numpy as np
import matplotlib.pyplot as plt

# Local application imports
from data.generation import generate_data
from vi.cavi import coordinates_ascent
from vi import postprocessing as pp
from s1_config import Params

# Create folder where results are saved
os.makedirs('results', exist_ok=True)

# load parameters
params = Params()

# plot settings
os.environ["PATH"] += os.pathsep + '/Library/TeX/texbin'
plot_params = {"text.usetex"         : True,
               "font.family"         : "serif",
               "font.size"           : "11",
               'text.latex.preamble' : r"\usepackage{bm}"}
plt.rcParams.update(plot_params)

#%% CAVI 
np.random.seed(101)
alpha = 0.5
params.alpha_DPM = alpha
params.alpha     = alpha
data_dict = generate_data(params)
data = data_dict["Noisy Datapoints"]
_, _, tau, gamma, phi = coordinates_ascent(data_dict, params)

# Postprocessing
results, results_reduced =\
    pp.full_postprocessing(data_dict, phi, gamma, tau, relabel=True)
    
# Plot true clustering
title = "True Clustering"
indicatorArray = data_dict["True Cluster Indicators"]
meanArray = data_dict["True Cluster Means"]
pp.plot_clustering(data, title, indicatorArray, meanArray)
plt.savefig(f"results/true_clusters_alpha{alpha}.pdf".replace(".", "", 1),\
            format="pdf", bbox_inches="tight")

# Plot estimated clustering
title = "Estimated Clustering"
indicatorArray = results_reduced["MMSE Estimated Cluster Indicators"]
meanArray = results_reduced["MMSE Estimated Cluster Means"]
meanIndicators = results_reduced["MMSE Mean Indicators"]
pp.plot_clustering(data, title, indicatorArray, meanArray,\
                   meanIndicators=meanIndicators)
plt.savefig(f"results/est_clusters_alpha{alpha}.pdf".replace(".", "", 1),\
            format="pdf", bbox_inches="tight")

#%% CAVI - Compare ELBO
np.random.seed(2322259932) 
alpha = 5
params.alpha_DPM = alpha
params.alpha     = alpha
data_dict = generate_data(params)
data = data_dict["Noisy Datapoints"]

elbo_converged_it = np.zeros(7, dtype='int')

fig1, ax1 = plt.subplots()
ax1.set_title(r"Convergence for $\alpha = 5$")
ax1.set_xlabel("ELBO")
ax1.set_ylabel("Number of iterations")

params.init_type = "uniform"
elbo, elbo_converged_it[0], tau, gamma, phi = coordinates_ascent(data_dict, params)
ax1.plot(np.trim_zeros(elbo, 'b'), color='b', label="Uniform")
# ax1.plot(elbo_converged_it[0], elbo[elbo_converged_it[0]], marker='x')
plt.axvline(x=elbo_converged_it[0], color='b', linestyle='--')

params.init_type = "true"
elbo, elbo_converged_it[1], tau, gamma, phi = coordinates_ascent(data_dict, params)
ax1.plot(np.trim_zeros(elbo, 'b'), color='g', label="True")
plt.axvline(x=elbo_converged_it[1], color='g', linestyle='--')

params.init_type = "permute"
elbo, elbo_converged_it[2], tau, gamma, phi = coordinates_ascent(data_dict, params)
ax1.plot(np.trim_zeros(elbo, 'b'), color='r', label="Random")
plt.axvline(x=elbo_converged_it[2], color='r', linestyle='--')

params.init_type = "unique"
elbo, elbo_converged_it[3], tau, gamma, phi = coordinates_ascent(data_dict, params)
ax1.plot(np.trim_zeros(elbo, 'b'), color='c', label="Unique")
plt.axvline(x=elbo_converged_it[3], color='c', linestyle='--')

params.init_type = "AllInOne"
elbo, elbo_converged_it[4], tau, gamma, phi = coordinates_ascent(data_dict, params)
ax1.plot(np.trim_zeros(elbo, 'b'), color='m', label="One Cluster")
plt.axvline(x=elbo_converged_it[4], color='m', linestyle='--')

params.init_type = "Kmeans"
elbo, elbo_converged_it[5], tau, gamma, phi = coordinates_ascent(data_dict, params)
ax1.plot(np.trim_zeros(elbo, 'b'), color='y', label="KMeans")
plt.axvline(x=elbo_converged_it[5], color='y', linestyle='--')

params.init_type = "DBSCAN"
elbo, elbo_converged_it[6], tau, gamma, phi = coordinates_ascent(data_dict, params)
ax1.plot(np.trim_zeros(elbo, 'b'), color='k', label="DBSCAN")
plt.axvline(x=elbo_converged_it[6], color='k', linestyle='--')

ax1.set_xlim((0,30))
ax1.grid()
ax1.legend()
fig1.tight_layout()