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
# np.random.seed(2322259932)
T = 30
alpha = 5
params.alpha_DPM = alpha
params.alpha     = alpha
data_dict = generate_data(params)

elbo_converged_it = np.zeros(8, dtype='int')
predictive_converged_it = np.zeros(8, dtype='int')

fig1, (ax11, ax12) = plt.subplots(1, 2, figsize=(6.2, 3.5), sharey=True)
fig1.suptitle(r'Convergence of the ELBO for $\alpha = 5$', y=0.94)
ax11.set_ylabel("ELBO")
ax11.set_xlabel("Number of iterations")
ax12.set_xlabel("Number of iterations")

params.init_type = "uniform"
params.T = T
elbo, elbo_converged_it[0], _, _, _ = coordinates_ascent(data_dict, params)
ax12.plot(np.trim_zeros(elbo, 'b'), color='b', label="Uniform")
ax12.axvline(x=elbo_converged_it[0], color='b', linestyle='--')

params.init_type = "true"
elbo, elbo_converged_it[1], _, _, _ = coordinates_ascent(data_dict, params)
ax11.plot(np.trim_zeros(elbo, 'b'), color='g', label="True")
ax11.axvline(x=elbo_converged_it[1], color='g', linestyle='--')

params.init_type = "permute"
params.T = T
elbo, elbo_converged_it[2], _, _, _ = coordinates_ascent(data_dict, params)
ax11.plot(np.trim_zeros(elbo, 'b'), color='r', label="Random")
ax11.axvline(x=elbo_converged_it[2], color='r', linestyle='--')

params.init_type = "unique"
elbo, elbo_converged_it[3], _, _, _ = coordinates_ascent(data_dict, params)
ax11.plot(np.trim_zeros(elbo, 'b'), color='c', label="Unique")
ax11.axvline(x=elbo_converged_it[3], color='c', linestyle='--')

params.init_type = "AllInOne"
params.T = T
elbo, elbo_converged_it[4], _, _, _ = coordinates_ascent(data_dict, params)
ax11.plot(np.trim_zeros(elbo, 'b'), color='m', label="One Cluster")
ax11.axvline(x=elbo_converged_it[4], color='m', linestyle='--')

params.init_type = "Kmeans"
params.T = T
elbo, elbo_converged_it[5], _, _, _ = coordinates_ascent(data_dict, params)
ax12.plot(np.trim_zeros(elbo, 'b'), color='y', label="KMeans")
ax12.axvline(x=elbo_converged_it[5], color='y', linestyle='--')

params.init_type = "DBSCAN"
params.T = T
elbo, elbo_converged_it[6], _, _, _ = coordinates_ascent(data_dict, params)
ax12.plot(np.trim_zeros(elbo, 'b'), color='k', label="DBSCAN")
ax12.axvline(x=elbo_converged_it[6], color='k', linestyle='--')

params.init_type = "global"
params.T = T
elbo, elbo_converged_it[7], _, _, _ = coordinates_ascent(data_dict, params)
ax12.plot(np.trim_zeros(elbo, 'b'), color='brown', label="Global")
ax12.axvline(x=elbo_converged_it[7], color='brown', linestyle='--')

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

