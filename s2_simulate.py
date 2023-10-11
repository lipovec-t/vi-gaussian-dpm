# Standard library imports
import timeit

# Third party imports
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
from tqdm import tqdm

# Local application imports
from data.generation import generate_data
from vi.cavi import coordinates_ascent
from vi import postprocessing as pp
from s2_config import Params

# Random seed for testing purposes
np.random.seed(255)

# Load parameters
params = Params()
K = params.K

# Perform Monte Carlo simulation for three different values of alpha
alpha = [0.5, 1, 5]
num_sim = len(alpha)

# set object count and number of MC runs
N_array = np.arange(1,51)
MC_runs = 500

# Store MSE for each simulation run
MSE_x = np.zeros((N_array.size, MC_runs, num_sim))

# Store max elbo for each monte carlo run
elbo_final = np.zeros((N_array.size, num_sim))
elbo_final[:] = -np.inf 

# Simulation
with tqdm(total=num_sim*MC_runs, position=0, desc='Simulation runs') as pbar:
    for k in range(num_sim):
        params.alpha = alpha[k]
        
        for i,N in enumerate(N_array):
            for j in range(MC_runs):
                # Generate data
                params.N = N
                data_dict = generate_data(params)
                
                # CAVI
                elbo, tau, gamma, phi = coordinates_ascent(data_dict, params)
                if elbo > elbo_final[i, k]:
                    elbo_final[i] = elbo
                       
                # Postprocessing
                results, _ = \
                    pp.full_postprocessing(data_dict, phi, gamma, tau, False)
                
                # MMSE estimator for x
                y = data_dict["Noisy Datapoints"]
                x_est = pp.est_object_positions(y, results, params)
        
                # Calculate MSE
                x = data_dict["Datapoints"]
                MSE_x[i, j, k] = pp.mse(x, x_est, K*N)
                
            # Update progress bar
            pbar.update(1)

# Save MSE results and note for simulation parameters
# TODO

# Postprocessing and plots
fig, ax = plt.subplots()
color = ['g', 'b', 'r']

# Compute and plot theoretical performance bounds (independent of alpha)
sigma_u = params.sigma_U[0,0]
sigma_v = params.sigma_V[0,0]
sigma_g = params.sigma_G[0,0]
MSE_1 = (sigma_v * (sigma_u + sigma_g)) / (sigma_g + sigma_u + sigma_v)
label = r'$\mathrm{MSE}_{\mathrm{min}}^{(1)}$'
plt.axhline(y=MSE_1, color='gray', linestyle='-', label=label)
MSE_2 = (sigma_v * sigma_u) / (sigma_u + sigma_v)
label = r'$\mathrm{MSE}_{\mathrm{min}}^{(2)}$'
plt.axhline(y=MSE_2, color='black', linestyle='-', label=label)

# Plot result for each alpha run
for k in range(num_sim):
    # mean MSE
    MSE_x_avg = np.mean(MSE_x[:, :, k], axis=1)
    
    # 95% confidence interval
    (ci_min, ci_max) = st.t.interval(confidence = 0.95, df = MSE_x.shape[1]-1,\
                                    loc = MSE_x_avg,\
                                    scale = st.sem(MSE_x[:, :, k], axis=1))     
    
    # Plot average MSE
    label = r'MSE VI $\alpha = {}$'.format(alpha[k])
    ax.plot(N_array, MSE_x_avg, color=color[k], label=label)
    
    # Plot confidence interval
    label = r'$95\%$ CI'
    ax.fill_between(N_array, ci_min, ci_max,\
                    color=color[k], alpha=.1, label=label)

# Plot settings
plt.xlabel('Number of objects')
plt.ylabel('Average MSE')
plt.legend()
plt.grid()
