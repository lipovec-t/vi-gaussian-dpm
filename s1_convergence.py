# Standard library imports
import os

# Third party imports
import numpy as np
import matplotlib.pyplot as plt

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

fig1, ax1 = plt.subplots()
ax1.set_title(r"Convergence for $\alpha = 5$")
ax1.set_xlabel("ELBO")
ax1.set_ylabel("Number of iterations")

params.init_type = "uniform"
elbo, elbo_converged_it[0], tau, gamma, phi = coordinates_ascent(data_dict, params)
ax1.plot(np.trim_zeros(elbo, 'b'), color='b', label="Uniform")
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

params.init_type = "global"
elbo, elbo_converged_it[7], tau, gamma, phi = coordinates_ascent(data_dict, params)
ax1.plot(np.trim_zeros(elbo, 'b'), color='brown', label="Global")
plt.axvline(x=elbo_converged_it[7], color='brown', linestyle='--')

ax1.set_xlim((0,40))
ax1.grid()
ax1.legend()
fig1.tight_layout()