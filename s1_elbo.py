# Standard library imports
import os

# Third party imports
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from tqdm import tqdm

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

#%% CAVI - Convergence Analyzation
MC_runs = 500
T = 30
alpha = 5
params.alpha_DPM = alpha
params.alpha     = alpha
params.N = 50

it_counter = np.array(range(1,params.max_iterations+1))
elbo = np.zeros((MC_runs, params.max_iterations), dtype='float')
elbo_converged_it = np.zeros(MC_runs, dtype='int')

elbo_avg_array = np.zeros((8,params.max_iterations))
elbo_converged_it_avg_array = np.zeros(8)
ci_elbo_array = np.zeros((8,2,params.max_iterations))
ci_it_array = np.zeros((8,2))

#%% Simulate uniform initialization 
params.init_type = "uniform"
params.T = T
for i in tqdm(range(MC_runs), desc='(1/8) Uniform Init '):
    data_dict = generate_data(params)
    elbo[i, :], elbo_converged_it[i], _, _, _ = coordinates_ascent(data_dict, params)
    
# average results
elbo_avg_array[0,:] = np.mean(elbo, axis=0)
elbo_converged_it_avg_array[0] = np.mean(elbo_converged_it)
# 95% confidence interval
(ci_elbo_array[0,0,:], ci_elbo_array[0,1,:]) = st.t.interval(confidence = 0.95, df = elbo.shape[0]-1,\
                                loc = elbo_avg_array[0,:],\
                                scale = st.sem(elbo, axis=0))
(ci_it_array[0,0], ci_it_array[0,1]) = st.t.interval(confidence = 0.95, df = elbo_converged_it.shape[0]-1,\
                                loc = elbo_converged_it_avg_array[0],\
                                scale = st.sem(elbo_converged_it, axis=0))

#%% Simulate true initialization 
params.init_type = "true"
params.T = T
for i in tqdm(range(MC_runs), desc='(2/8) True Init    '):
    data_dict = generate_data(params)
    elbo[i, :], elbo_converged_it[i], _, _, _ = coordinates_ascent(data_dict, params)

# average results
elbo_avg_array[1,:] = np.mean(elbo, axis=0)
elbo_converged_it_avg_array[1] = np.mean(elbo_converged_it)
# 95% confidence interval
(ci_elbo_array[1,0,:], ci_elbo_array[1,1,:]) = st.t.interval(confidence = 0.95, df = elbo.shape[0]-1,\
                                loc = elbo_avg_array[1,:],\
                                scale = st.sem(elbo, axis=0))
(ci_it_array[1,0], ci_it_array[1,1]) = st.t.interval(confidence = 0.95, df = elbo_converged_it.shape[0]-1,\
                                loc = elbo_converged_it_avg_array[1],\
                                scale = st.sem(elbo_converged_it, axis=0))

#%% Simulate permute initialization 
params.init_type = "permute"
params.num_permutations = 10
params.T = T
for i in tqdm(range(MC_runs), desc='(3/8) Permute Init '):
    data_dict = generate_data(params)
    elbo[i, :], elbo_converged_it[i], _, _, _ = coordinates_ascent(data_dict, params)

# average results
elbo_avg_array[2,:] = np.mean(elbo, axis=0)
elbo_converged_it_avg_array[2] = np.mean(elbo_converged_it)
# 95% confidence interval
# 95% confidence interval
(ci_elbo_array[2,0,:], ci_elbo_array[2,1,:]) = st.t.interval(confidence = 0.95, df = elbo.shape[0]-1,\
                                loc = elbo_avg_array[2,:],\
                                scale = st.sem(elbo, axis=0))
(ci_it_array[2,0], ci_it_array[2,1]) = st.t.interval(confidence = 0.95, df = elbo_converged_it.shape[0]-1,\
                                loc = elbo_converged_it_avg_array[2],\
                                scale = st.sem(elbo_converged_it, axis=0))

#%% Simulate unique initialization 
params.init_type = "unique"
for i in tqdm(range(MC_runs), desc='(4/8) Unique Init  '):
    data_dict = generate_data(params)
    elbo[i, :], elbo_converged_it[i], _, _, _ = coordinates_ascent(data_dict, params)

# average results
elbo_avg_array[3,:] = np.mean(elbo, axis=0)
elbo_converged_it_avg_array[3] = np.mean(elbo_converged_it)
# 95% confidence interval
(ci_elbo_array[3,0,:], ci_elbo_array[3,1,:]) = st.t.interval(confidence = 0.95, df = elbo.shape[0]-1,\
                                loc = elbo_avg_array[3,:],\
                                scale = st.sem(elbo, axis=0))
(ci_it_array[3,0], ci_it_array[3,1]) = st.t.interval(confidence = 0.95, df = elbo_converged_it.shape[0]-1,\
                                loc = elbo_converged_it_avg_array[3],\
                                scale = st.sem(elbo_converged_it, axis=0))

#%% Simulate allinone initialization
params.init_type = "AllInOne"
params.T = T
for i in tqdm(range(MC_runs), desc='(5/8) AllInOne Init'):
    data_dict = generate_data(params)
    elbo[i, :], elbo_converged_it[i], _, _, _ = coordinates_ascent(data_dict, params)

# average results
elbo_avg_array[4,:] = np.mean(elbo, axis=0)
elbo_converged_it_avg_array[4] = np.mean(elbo_converged_it)
# 95% confidence interval
(ci_elbo_array[4,0,:], ci_elbo_array[4,1,:]) = st.t.interval(confidence = 0.95, df = elbo.shape[0]-1,\
                                loc = elbo_avg_array[4,:],\
                                scale = st.sem(elbo, axis=0))
(ci_it_array[4,0], ci_it_array[4,1]) = st.t.interval(confidence = 0.95, df = elbo_converged_it.shape[0]-1,\
                                loc = elbo_converged_it_avg_array[4],\
                                scale = st.sem(elbo_converged_it, axis=0))

#%% Simulate kmeans initialization 
params.init_type = "Kmeans"
params.T = T
for i in tqdm(range(MC_runs), desc='(6/8) KMeans Init  '):
    data_dict = generate_data(params)
    elbo[i, :], elbo_converged_it[i], _, _, _ = coordinates_ascent(data_dict, params)

# average results
elbo_avg_array[5,:] = np.mean(elbo, axis=0)
elbo_converged_it_avg_array[5] = np.mean(elbo_converged_it)
# 95% confidence interval
(ci_elbo_array[5,0,:], ci_elbo_array[5,1,:]) = st.t.interval(confidence = 0.95, df = elbo.shape[0]-1,\
                                loc = elbo_avg_array[5,:],\
                                scale = st.sem(elbo, axis=0))
(ci_it_array[5,0], ci_it_array[5,1]) = st.t.interval(confidence = 0.95, df = elbo_converged_it.shape[0]-1,\
                                loc = elbo_converged_it_avg_array[5],\
                                scale = st.sem(elbo_converged_it, axis=0))

#%% Simulate dbscan initialization
params.init_type = "DBSCAN"
params.T = T
for i in tqdm(range(MC_runs), desc='(7/8) DBSCAN Init  '):
    data_dict = generate_data(params)
    elbo[i, :], elbo_converged_it[i], _, _, _ = coordinates_ascent(data_dict, params)

# average results
elbo_avg_array[6,:] = np.mean(elbo, axis=0)
elbo_converged_it_avg_array[6] = np.mean(elbo_converged_it)
# 95% confidence interval
(ci_elbo_array[6,0,:], ci_elbo_array[6,1,:]) = st.t.interval(confidence = 0.95, df = elbo.shape[0]-1,\
                                loc = elbo_avg_array[6,:],\
                                scale = st.sem(elbo, axis=0))
(ci_it_array[6,0], ci_it_array[6,1]) = st.t.interval(confidence = 0.95, df = elbo_converged_it.shape[0]-1,\
                                loc = elbo_converged_it_avg_array[6],\
                                scale = st.sem(elbo_converged_it, axis=0))

#%% Simulate global initialization 
params.init_type = "global"
params.T = T
for i in tqdm(range(MC_runs), desc='(8/8) Global Init  '):
    data_dict = generate_data(params)
    elbo[i, :], elbo_converged_it[i], _, _, _ = coordinates_ascent(data_dict, params)
    
# average results
elbo_avg_array[7,:] = np.mean(elbo, axis=0)
elbo_converged_it_avg_array[7] = np.mean(elbo_converged_it)
# 95% confidence interval
(ci_elbo_array[7,0,:], ci_elbo_array[7,1,:]) = st.t.interval(confidence = 0.95, df = elbo.shape[0]-1,\
                                loc = elbo_avg_array[7,:],\
                                scale = st.sem(elbo, axis=0))
(ci_it_array[7,0], ci_it_array[7,1]) = st.t.interval(confidence = 0.95, df = elbo_converged_it.shape[0]-1,\
                                loc = elbo_converged_it_avg_array[7],\
                                scale = st.sem(elbo_converged_it, axis=0))

#%% plot results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3.5), sharey=True)
fig.suptitle(r'Convergence of the ELBO for $\alpha = 5$', y=0.94)
ax1.set_ylabel("Average ELBO")
ax1.set_xlabel("Number of iterations")
ax2.set_xlabel("Number of iterations")

label_array = ['Uniform', 'True', 'Random', 'Unique',\
               'One Cluster', 'KMeans', 'DBSCAN', 'Global'] 

color_array = ['purple', 'green', 'red', 'blue']
for i,j in enumerate([4,1,6,3]):
    ax1.plot(it_counter, elbo_avg_array[j,:], color=color_array[i], label=label_array[j])
    ax1.fill_between(it_counter, ci_elbo_array[j,0,:], ci_elbo_array[j,1,:], color=color_array[i], alpha=.1)
    ax1.axvline(x=elbo_converged_it_avg_array[j], color=color_array[i], linestyle='--')
    ax1.axvspan(ci_it_array[j,0], ci_it_array[j,1], alpha=0.1, color=color_array[i])

color_array = ['orange', 'olive', 'lightseagreen', 'brown'] 
for i,j in enumerate([0,5,2,7]):
    ax2.plot(it_counter, elbo_avg_array[j,:], color=color_array[i], label=label_array[j])
    ax2.fill_between(it_counter, ci_elbo_array[j,0,:], ci_elbo_array[j,1,:], color=color_array[i], alpha=.1)
    ax2.axvline(x=elbo_converged_it_avg_array[j], color=color_array[i], linestyle='--')
    ax2.axvspan(ci_it_array[j,0], ci_it_array[j,1], alpha=0.1, color=color_array[i])

xlim = 30
ax1.set_xlim((1,xlim))
ax1.xaxis.set_minor_locator(MultipleLocator(1))
ax1.grid()
ax1.legend(loc = 'center right', framealpha=1)
ax2.set_xlim((1,xlim))
ax2.xaxis.set_minor_locator(MultipleLocator(1))
ax2.grid()
ax2.legend(loc = 'center right', framealpha=1)
fig.tight_layout()

plt.savefig("results/convergence_elbo.pdf", format="pdf", bbox_inches="tight")