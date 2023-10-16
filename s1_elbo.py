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
params.N = 50
data_dict = generate_data(params)
data = data_dict["Noisy Datapoints"]

elbo_converged_it = np.zeros(8, dtype='int')

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.2, 3.5), sharey=True)
fig.suptitle(r'Convergence of the ELBO for $\alpha = 5$', y=0.94)
ax1.set_ylabel("ELBO")
ax1.set_xlabel("Number of iterations")
ax2.set_xlabel("Number of iterations")

params.init_type = "uniform"
elbo, elbo_converged_it[0], _, _, tau, gamma, phi = coordinates_ascent(data_dict, params)
ax2.plot(np.trim_zeros(elbo, 'b'), color='b', label="Uniform")
ax2.axvline(x=elbo_converged_it[0], color='b', linestyle='--')

params.init_type = "true"
elbo, elbo_converged_it[1], _, _, tau, gamma, phi = coordinates_ascent(data_dict, params)
ax1.plot(np.trim_zeros(elbo, 'b'), color='g', label="True")
ax1.axvline(x=elbo_converged_it[1], color='g', linestyle='--')

params.init_type = "permute"
elbo, elbo_converged_it[2], _, _, tau, gamma, phi = coordinates_ascent(data_dict, params)
ax1.plot(np.trim_zeros(elbo, 'b'), color='r', label="Random")
ax1.axvline(x=elbo_converged_it[2], color='r', linestyle='--')

params.init_type = "unique"
elbo, elbo_converged_it[3], _, _, tau, gamma, phi = coordinates_ascent(data_dict, params)
ax1.plot(np.trim_zeros(elbo, 'b'), color='c', label="Unique")
ax1.axvline(x=elbo_converged_it[3], color='c', linestyle='--')

params.init_type = "AllInOne"
elbo, elbo_converged_it[4], _, _, tau, gamma, phi = coordinates_ascent(data_dict, params)
ax1.plot(np.trim_zeros(elbo, 'b'), color='m', label="One Cluster")
ax1.axvline(x=elbo_converged_it[4], color='m', linestyle='--')

params.init_type = "Kmeans"
elbo, elbo_converged_it[5], _, _, tau, gamma, phi = coordinates_ascent(data_dict, params)
ax2.plot(np.trim_zeros(elbo, 'b'), color='y', label="KMeans")
ax2.axvline(x=elbo_converged_it[5], color='y', linestyle='--')

params.init_type = "DBSCAN"
elbo, elbo_converged_it[6], _, _, tau, gamma, phi = coordinates_ascent(data_dict, params)
ax2.plot(np.trim_zeros(elbo, 'b'), color='k', label="DBSCAN")
ax2.axvline(x=elbo_converged_it[6], color='k', linestyle='--')

params.init_type = "global"
elbo, elbo_converged_it[7], _, _, tau, gamma, phi = coordinates_ascent(data_dict, params)
ax2.plot(np.trim_zeros(elbo, 'b'), color='brown', label="Global")
ax2.axvline(x=elbo_converged_it[7], color='brown', linestyle='--')

# plot settings
xlim = 40
ax1.set_xlim((0,xlim))
ax1.xaxis.set_minor_locator(MultipleLocator(1))
ax1.grid()
ax1.legend(loc = 'center right')
ax2.set_xlim((0,xlim))
ax2.xaxis.set_minor_locator(MultipleLocator(1))
ax2.grid()
ax2.legend(loc = 'center right')
fig.tight_layout()

plt.savefig("results/convergence_elbo.pdf", format="pdf", bbox_inches="tight")