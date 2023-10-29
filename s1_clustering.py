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

#%% CAVI - Clustering
alphaArray = [0.5,1,5]
seedArray = [101,236,22]
params.N = 50

for (alpha, seed) in zip(alphaArray, seedArray):
    np.random.seed(seed)
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
    plt.savefig(f"results/true_clusters_alpha{alpha}".replace(".", "")+".pdf",\
                format="pdf", bbox_inches="tight")
    
    # Plot estimated clustering
    title = "Estimated Clustering"
    indicatorArray = results_reduced["MMSE Estimated Cluster Indicators"]
    meanArray = results_reduced["MMSE Estimated Cluster Means"]
    meanIndicators = results_reduced["MMSE Mean Indicators"]
    pp.plot_clustering(data, title, indicatorArray, meanArray,\
                        meanIndicators=meanIndicators)
    plt.savefig(f"results/est_clusters_alpha{alpha}".replace(".", "")+".pdf",\
                format="pdf", bbox_inches="tight")

