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
data_dict = generate_data(params)
data = data_dict["Noisy Datapoints"]

elbo_converged_it = np.zeros(8, dtype='int')
predictive_converged_it = np.zeros(8, dtype='int')

fig1, (ax11, ax12) = plt.subplots(1, 2, figsize=(6.2, 3.5), sharey=True)
fig1.suptitle(r'Convergence of the ELBO for $\alpha = 5$', y=0.94)
ax11.set_ylabel("ELBO")
ax11.set_xlabel("Number of iterations")
ax12.set_xlabel("Number of iterations")

fig2, (ax21, ax22) = plt.subplots(1, 2, figsize=(6.2, 3.5), sharey=True)
fig2.suptitle(r'Convergence of the Predictive Dist. for $\alpha = 5$')
ax21.set_xlabel("Predictive")
ax21.set_ylabel("Number of iterations")
ax22.set_ylabel("Number of iterations")

params.init_type = "uniform"
elbo, elbo_converged_it[0], predictive, predictive_converged_it[0], tau, gamma, phi = coordinates_ascent(data_dict, params)
ax12.plot(np.trim_zeros(elbo, 'b'), color='b', label="Uniform")
ax12.axvline(x=elbo_converged_it[0], color='b', linestyle='--')
ax21.plot(np.trim_zeros(predictive, 'b'), color='b', label="Uniform")
ax21.axvline(x=predictive_converged_it[0], color='b', linestyle='--')

params.init_type = "true"
elbo, elbo_converged_it[1], predictive, predictive_converged_it[1], tau, gamma, phi = coordinates_ascent(data_dict, params)
ax11.plot(np.trim_zeros(elbo, 'b'), color='g', label="True")
ax11.axvline(x=elbo_converged_it[1], color='g', linestyle='--')
ax21.plot(np.trim_zeros(predictive, 'b'), color='g', label="True")
ax21.axvline(x=predictive_converged_it[1], color='g', linestyle='--')

params.init_type = "permute"
elbo, elbo_converged_it[2], predictive, predictive_converged_it[2], tau, gamma, phi = coordinates_ascent(data_dict, params)
ax11.plot(np.trim_zeros(elbo, 'b'), color='r', label="Random")
ax11.axvline(x=elbo_converged_it[2], color='r', linestyle='--')
ax21.plot(np.trim_zeros(predictive, 'b'), color='r', label="Random")
ax21.axvline(x=predictive_converged_it[2], color='r', linestyle='--')

params.init_type = "unique"
elbo, elbo_converged_it[3], predictive, predictive_converged_it[3], tau, gamma, phi = coordinates_ascent(data_dict, params)
ax11.plot(np.trim_zeros(elbo, 'b'), color='c', label="Unique")
ax11.axvline(x=elbo_converged_it[3], color='c', linestyle='--')
ax21.plot(np.trim_zeros(predictive, 'b'), color='c', label="Unique")
ax21.axvline(x=predictive_converged_it[3], color='c', linestyle='--')

params.init_type = "AllInOne"
elbo, elbo_converged_it[4], predictive, predictive_converged_it[4], tau, gamma, phi = coordinates_ascent(data_dict, params)
ax11.plot(np.trim_zeros(elbo, 'b'), color='m', label="One Cluster")
ax11.axvline(x=elbo_converged_it[4], color='m', linestyle='--')
ax22.plot(np.trim_zeros(predictive, 'b'), color='m', label="One Cluster")
ax22.axvline(x=predictive_converged_it[4], color='m', linestyle='--')

params.init_type = "Kmeans"
elbo, elbo_converged_it[5], predictive, predictive_converged_it[5], tau, gamma, phi = coordinates_ascent(data_dict, params)
ax12.plot(np.trim_zeros(elbo, 'b'), color='y', label="KMeans")
ax12.axvline(x=elbo_converged_it[5], color='y', linestyle='--')
ax22.plot(np.trim_zeros(predictive, 'b'), color='y', label="KMeans")
ax22.axvline(x=predictive_converged_it[5], color='y', linestyle='--')

params.init_type = "DBSCAN"
elbo, elbo_converged_it[6], predictive, predictive_converged_it[6], tau, gamma, phi = coordinates_ascent(data_dict, params)
ax12.plot(np.trim_zeros(elbo, 'b'), color='k', label="DBSCAN")
ax12.axvline(x=elbo_converged_it[6], color='k', linestyle='--')
ax22.plot(np.trim_zeros(predictive, 'b'), color='k', label="DBSCAN")
ax22.axvline(x=predictive_converged_it[6], color='k', linestyle='--')

params.init_type = "global"
elbo, elbo_converged_it[7], predictive, predictive_converged_it[7], tau, gamma, phi = coordinates_ascent(data_dict, params)
ax12.plot(np.trim_zeros(elbo, 'b'), color='brown', label="Global")
ax12.axvline(x=elbo_converged_it[7], color='brown', linestyle='--')
ax22.plot(np.trim_zeros(predictive, 'b'), color='brown', label="Global")
ax22.axvline(x=predictive_converged_it[7], color='brown', linestyle='--')

xlim = 40

ax11.set_xlim((0,xlim))
ax11.xaxis.set_minor_locator(MultipleLocator(1))
ax11.grid()
ax11.legend(loc = 'center right')
ax12.set_xlim((0,xlim))
ax12.xaxis.set_minor_locator(MultipleLocator(1))
ax12.grid()
ax12.legend(loc = 'center right')
fig1.tight_layout()

ax21.set_xlim((0,xlim))
ax21.grid()
ax21.legend()
ax22.set_xlim((0,xlim))
ax22.grid()
ax22.legend()
fig2.tight_layout()

plt.close(fig2)
plt.savefig("results/convergence_elbo.pdf", format="pdf", bbox_inches="tight")