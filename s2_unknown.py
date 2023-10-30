# Standard library imports
import os
import time

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

# Create folder where results are saved
os.makedirs('results', exist_ok=True)

# Load parameters
params = Params()
K = params.K

# Set concentration parameter of DPM model
params.alpha_DPM = 0.5 # has to be a list with at least one value

# Perform Monte Carlo simulation for different values of CAVI hyperparams
# alpha = [params.alpha_DPM/100, params.alpha_DPM*100]
alpha = [0.5]
num_alpha = len(alpha)

# Use data driven base dist
params.data_driven_base_dist = True 

# set object count and number of MC runs
N_array = np.arange(1,51)
num_N = len(N_array)
MC_runs = 1000

# Store MSE for each CAVI run
MSE_x = np.zeros((N_array.size, MC_runs, num_alpha))

# Store max elbo for each monte carlo run
elbo_final = np.zeros((N_array.size, num_alpha))
elbo_final[:] = -np.inf 

# Store runtime for each CAVI run
runtime = np.zeros((N_array.size, MC_runs, num_alpha))

#%% Simulation
with tqdm(total=num_alpha*num_N, position=0, desc='Simulation runs') as pbar:
    for k in range(num_alpha):
        # Set concentration parameter
        params.alpha     = alpha[k]
        
        for i,N in enumerate(N_array):
            for j in range(MC_runs):
                # Generate data
                params.N = N
                data_dict = generate_data(params)
                
                # CAVI
                elbo, tau, gamma, phi = coordinates_ascent(data_dict, params)
                
                # Check if elbo of this MC run is bigger than previous
                if elbo > elbo_final[i, k]:
                    elbo_final[i, k] = elbo
                       
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
    
        # Save MSE results for current alpha value
        np.save(f"results/mse{k+1}_alpha{alpha[k]}".replace(".", ""),\
                MSE_x[:, :, k])

#%% Save simulation notes 
time_str = time.strftime("%Y%m%d-%H%M%S")
alpha_values = ", ".join([str(alpha[k]) for k in range(num_alpha)])
permutations_str = f"Number of permutations: {params.num_permutations}\n"
truncation_str = f"Truncation parameter:   {params.T}\n"
notes = f"""\
Simulation with misspecified hyerparameters
Parameters of base distribution chosen with a data-driven approach
Simulation time:        {time_str} 
Trua Alpha value:       {params.alpha_DPM}
Alpha values:           {alpha_values} 
Initialization type:    {params.init_type}
{permutations_str if params.init_type.lower() == "permute" else ""}\
{truncation_str if params.init_type.lower() != "unique" else ""}\
Convergence threshold:  {params.eps:.2e}
Max iterations:         {params.max_iterations}
"""
with open("results/notes.txt","w+") as f:
    f.writelines(notes)

#%% Analyze and plot MSE 
fig, ax = plt.subplots()
color = ['g', 'b', 'r'] # adapt this if num_alpha > 3

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
for k in range(num_alpha):
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

