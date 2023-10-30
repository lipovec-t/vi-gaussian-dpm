# Standard library imports
import os

# Third party imports
import numpy as np
import matplotlib.pyplot as plt

# Local application imports
from data.generation import generate_data
from vi.cavi import coordinates_ascent
from vi import postprocessing as pp
from s2_config import Params

# plot settings
os.environ["PATH"] += os.pathsep + '/Library/TeX/texbin'
plot_params = {"text.usetex"         : True,
               "font.family"         : "serif",
               "font.size"           : "11",
               'text.latex.preamble' : r"\usepackage{bm}"}
plt.rcParams.update(plot_params)

# Random seed for testing purposes
np.random.seed(2214414)

# Create folder where results are saved
os.makedirs('results', exist_ok=True)

# Load parameters
params = Params()
K = params.K

# Perform Monte Carlo simulation for three different values of alpha
alpha = 1
params.alpha_DPM = alpha
params.alpha     = alpha

# Use known hyerparams for base dist
params.data_driven_base_dist = False

# set number of objects
N = 30
params.N = N

# Generate Data
data_dict = generate_data(params)

# CAVI
_, tau, gamma, phi = coordinates_ascent(data_dict, params)

# Postprocessing
results, results_reduced = \
    pp.full_postprocessing(data_dict, phi, gamma, tau, False)

# MMSE estimator for x
y = data_dict["Noisy Datapoints"]
x_est = pp.est_object_positions(y, results, params)

# Plot data-estimation association
plt.figure(figsize=(4, 3))
# mean_x = results_reduced["Estimated Cluster Means"][:,0]
# mean_y = results_reduced["Estimated Cluster Means"][:,1]
# plt.scatter(mean_x, mean_y, color="orange")
x1 = data_dict["Datapoints"][:,0]
x2 = y[:,0]
X_coords= np.array([x1, x2]) 
y1 = x_est[:,1]
y2 = data_dict["Datapoints"][:,1]
Y_coords=np.array([y1, y2])
plt.plot(X_coords, Y_coords, color='gray')
plt.scatter(x1, y1, color='red', label=r'Estimate $\hat{\bm{x}}_n$')
plt.scatter(x2, y2, color='black', label=r'True value $\bm{x}_n$')
plt.xlabel(r'$x_{n,1}$, $\hat{x}_{n,1}$')
plt.ylabel(r'$x_{n,2}$, $\hat{x}_{n,2}$')
plt.legend()
plt.tight_layout()