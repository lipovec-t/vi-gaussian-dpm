# Standard library imports
import glob

# Third party imports
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st
import scipy.io as io

#%% Get file names
N_array = np.arange(1,51)
npy_files = glob.glob('*.npy')
mat_files = glob.glob('comp_data/*.mat')
all_files = npy_files + mat_files
all_files.sort()
alpha_list = (0.5, 1, 5, 0.5, 1, 5) # simulated alpha values
color_list = ('g','b','r','g','b','r') # colors for each plot

#%% Plot simulation results and comparison data
fig, ax = plt.subplots()

for i,file in enumerate(all_files):
    alpha = alpha_list[i]
    color = color_list[i]
    
    # load data
    if file.endswith('.npy'):
        MSE_x = np.load(file)
        label = r'VI, $\alpha = {}$'.format(alpha)
        linestyle = 'solid'
    elif file.endswith('.mat'):
        MSE_x = io.loadmat(file)['mse_vec']
        label = r'Gibbs, $\alpha = {}$'.format(alpha)
        linestyle = 'dashed'
    else:
        raise TypeError('Works with .mat and .npy files only')
        
    # mean MSE
    MSE_x_avg = np.mean(MSE_x, axis=1)
    
    # 95% confidence interval
    (ci_min, ci_max) = st.t.interval(confidence=0.95, df=MSE_x.shape[1]-1,\
                                     loc=MSE_x_avg, scale=st.sem(MSE_x,axis=1)) 
    ax.plot(N_array, MSE_x_avg, color=color, linestyle=linestyle, label=label)
    # label = r'$95\%$ CI'
    ax.fill_between(N_array, ci_min, ci_max,\
                    color=color, linestyle=linestyle, alpha=.1)

#%% Compute and plot theoretical performance bounds

# Model parameters used for simulation
sigma_u = 1
sigma_v = 1
sigma_g = 5

MSE_1 = (sigma_v * (sigma_u + sigma_g)) / (sigma_g + sigma_u + sigma_v)
label = r'$\mathrm{MSE}_{\mathrm{min}}^{(1)}$'
plt.axhline(y=MSE_1, color='gray', linestyle='-', label=label)

MSE_2 = (sigma_v * sigma_u) / (sigma_u + sigma_v)
label = r'$\mathrm{MSE}_{\mathrm{min}}^{(2)}$'
plt.axhline(y=MSE_2, color='black', linestyle='-', label=label)

#%% Plot settings
plt.xlabel('Number of objects')
plt.ylabel('Average MSE')
plt.legend(fontsize = 'xx-small')
# Shrink current axis by 20%
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
# Put a legend to the right of the current axis
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.grid()