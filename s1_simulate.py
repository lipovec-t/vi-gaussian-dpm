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
_, tau, gamma, phi = coordinates_ascent(data_dict, params)

# Postprocessing
results, results_reduced =\
    pp.full_postprocessing(data_dict, phi, gamma, tau, relabel=True)
    
# Plot true clustering
title = "True Clustering"
data = data_dict["Noisy Datapoints"]
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
np.random.seed(255)
data_dict = generate_data(params)
data = data_dict["Noisy Datapoints"]

plt.figure()
plt.title(r"Convergence for $\alpha = 5$")
plt.xlabel("Number of iterations")
plt.ylabel("ELBO")

params.init_type = "uniform"
elbo, tau, gamma, phi = coordinates_ascent(data_dict, params)
plt.plot(np.trim_zeros(elbo, 'b'), color='b', label="Uniform")

params.init_type = "true"
elbo, tau, gamma, phi = coordinates_ascent(data_dict, params)
plt.plot(np.trim_zeros(elbo, 'b'), color='g', label="True")

params.init_type = "permute"
elbo, tau, gamma, phi = coordinates_ascent(data_dict, params)
plt.plot(np.trim_zeros(elbo, 'b'), color='r', label="Random")

params.init_type = "unique"
elbo, tau, gamma, phi = coordinates_ascent(data_dict, params)
plt.plot(np.trim_zeros(elbo, 'b'), color='c', label="Unique")

params.init_type = "AllInOne"
elbo, tau, gamma, phi = coordinates_ascent(data_dict, params)
plt.plot(np.trim_zeros(elbo, 'b'), color='m', label="One Cluster")

params.init_type = "Kmeans"
elbo, tau, gamma, phi = coordinates_ascent(data_dict, params)
plt.plot(np.trim_zeros(elbo, 'b'), color='y', label="KMeans")

params.init_type = "DBSCAN"
elbo, tau, gamma, phi = coordinates_ascent(data_dict, params)
plt.plot(np.trim_zeros(elbo, 'b'), color='k', label="DBSCAN")

plt.xlim((0,30))
plt.grid()
plt.legend()
plt.tight_layout()