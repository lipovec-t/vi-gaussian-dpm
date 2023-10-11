# Standard library imports
import timeit

# Third party imports
import numpy as np
import matplotlib.pyplot as plt

# Local application imports
from data.generation import generate_data
from vi.cavi import coordinates_ascent
from vi import postprocessing as pp
from s3_config import Params

# random seed for testing purposes
np.random.seed(234)

# load parameters
params = Params()

# generate data
data_dict = generate_data(params)
if params.include_noise == False:
    data = data_dict["Datapoints"]
else:
    data = data_dict["Noisy Datapoints"]

# start timer
t_0 = timeit.default_timer()

# CAVI
elbo_final, elbo_converged_it, predictive, predictive_converged_it,\
    tau, gamma, phi = coordinates_ascent(data_dict, params)
# end timer

# end timer and compute elapsed time
t_1 = timeit.default_timer()
runtime = t_1 - t_0

# postprocessing
results, results_reduced =\
    pp.full_postprocessing(data_dict, phi, gamma, tau, False)
    
# plots
title = "Clustering"
indicatorArray = results_reduced["Estimated Cluster Indicators"]
meanArray = results_reduced["Estimated Cluster Means"]
pp.plot_clustering(data, title, indicatorArray, meanArray)

plt.figure()
plt.title("DPM - ELBO Convergence")
plt.xlabel("Iteration")
plt.ylabel("ELBO")
plt.xlim(1,50)
plt.grid()
plt.plot(elbo_final)
plt.axvline(x=elbo_converged_it[0], color='black', linestyle='--')
plt.axvline(x=elbo_converged_it[1], color='black', linestyle='--')
plt.axvline(x=elbo_converged_it[2], color='black', linestyle='--')
plt.tight_layout()

plt.figure()
plt.title("DPM - Predictive Convergence")
plt.xlabel("Iteration")
plt.ylabel("Average log predictive")
plt.xlim(1,50)
plt.grid()
plt.plot(predictive)
plt.axvline(x=predictive_converged_it[0], color='black', linestyle='--')
plt.axvline(x=predictive_converged_it[1], color='black', linestyle='--')
plt.axvline(x=predictive_converged_it[2], color='black', linestyle='--')
plt.tight_layout()
