# Standard library imports
import os

# Third party imports
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

# Local application imports
from data.generation import generate_data
from vi.cavi import coordinates_ascent
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

#%% CAVI - Convergence
np.random.seed(2322259932) 
alpha = 5
params.alpha_DPM = alpha
params.alpha     = alpha
params.N = 75
data_dict = generate_data(params)
data = data_dict["Noisy Datapoints"]

elbo_converged_it = np.zeros(8, dtype='int')
predictive_converged_it = np.zeros(8, dtype='int')

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.2, 3.5), sharey=True)
fig.suptitle(r'Convergence of the Predictive Distribution for $\alpha = 5$', y=0.94)
ax1.set_ylabel("Predictive")
ax1.set_xlabel("Number of iterations")
ax2.set_xlabel("Number of iterations")

params.init_type = "uniform"
_, _, predictive, predictive_converged_it[0], tau, gamma, phi = coordinates_ascent(data_dict, params)
ax1.plot(np.trim_zeros(predictive, 'b'), color='b', label="Uniform")
ax1.axvline(x=predictive_converged_it[0], color='b', linestyle='--')

params.init_type = "true"
_, _, predictive, predictive_converged_it[1], tau, gamma, phi = coordinates_ascent(data_dict, params)
ax1.plot(np.trim_zeros(predictive, 'b'), color='g', label="True")
ax1.axvline(x=predictive_converged_it[1], color='g', linestyle='--')

params.init_type = "permute"
_, _, predictive, predictive_converged_it[2], tau, gamma, phi = coordinates_ascent(data_dict, params)
ax1.plot(np.trim_zeros(predictive, 'b'), color='r', label="Random")
ax1.axvline(x=predictive_converged_it[2], color='r', linestyle='--')

params.init_type = "unique"
_, _, predictive, predictive_converged_it[3], tau, gamma, phi = coordinates_ascent(data_dict, params)
ax1.plot(np.trim_zeros(predictive, 'b'), color='c', label="Unique")
ax1.axvline(x=predictive_converged_it[3], color='c', linestyle='--')

params.init_type = "AllInOne"
_, _, predictive, predictive_converged_it[4], tau, gamma, phi = coordinates_ascent(data_dict, params)
ax2.plot(np.trim_zeros(predictive, 'b'), color='m', label="One Cluster")
ax2.axvline(x=predictive_converged_it[4], color='m', linestyle='--')

params.init_type = "Kmeans"
_, _, predictive, predictive_converged_it[5], tau, gamma, phi = coordinates_ascent(data_dict, params)
ax2.plot(np.trim_zeros(predictive, 'b'), color='y', label="KMeans")
ax2.axvline(x=predictive_converged_it[5], color='y', linestyle='--')

params.init_type = "DBSCAN"
_, _, predictive, predictive_converged_it[6], tau, gamma, phi = coordinates_ascent(data_dict, params)
ax2.plot(np.trim_zeros(predictive, 'b'), color='k', label="DBSCAN")
ax2.axvline(x=predictive_converged_it[6], color='k', linestyle='--')

params.init_type = "global"
_, _, predictive, predictive_converged_it[7], tau, gamma, phi = coordinates_ascent(data_dict, params)
ax2.plot(np.trim_zeros(predictive, 'b'), color='brown', label="Global")
ax2.axvline(x=predictive_converged_it[7], color='brown', linestyle='--')

# plot settings
xlim = 40
ax1.set_xlim((0,xlim))
ax1.xaxis.set_minor_locator(MultipleLocator(1))
ax1.grid()
ax1.legend()
ax2.set_xlim((0,xlim))
ax2.xaxis.set_minor_locator(MultipleLocator(1))
ax2.grid()
ax2.legend()
fig.tight_layout()

plt.savefig("results/convergence_predictive.pdf", format="pdf", bbox_inches="tight")