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

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.2, 3.5), sharey=True)
fig.suptitle(r'Convergence of the ELBO for $\alpha = 5$', y=0.94)
ax1.set_ylabel("ELBO")
ax1.set_xlabel("Number of iterations")
ax2.set_xlabel("Number of iterations")

#%% Simulate uniform initialization 
params.init_type = "uniform"
params.T = T
for i in tqdm(range(MC_runs), desc='(1/8) Uniform Init '):
    data_dict = generate_data(params)
    elbo[i, :], elbo_converged_it[i], _, _, _ = coordinates_ascent(data_dict, params)
    
# average results
elbo_avg = np.mean(elbo, axis=0)
elbo_converged_it_avg = np.mean(elbo_converged_it)
# 95% confidence interval
(ci_min1, ci_max1) = st.t.interval(confidence = 0.95, df = elbo.shape[0]-1,\
                                loc = elbo_avg,\
                                scale = st.sem(elbo, axis=0))
(ci_min2, ci_max2) = st.t.interval(confidence = 0.95, df = elbo_converged_it.shape[0]-1,\
                                loc = elbo_converged_it_avg,\
                                scale = st.sem(elbo_converged_it, axis=0))
# plot results
ax2.plot(it_counter, elbo_avg, color='b', label="Uniform")
ax2.fill_between(it_counter, ci_min1, ci_max1, color='b', alpha=.1)
ax2.axvline(x=elbo_converged_it_avg, color='b', linestyle='--')
ax2.axvspan(ci_min2, ci_max2, alpha=0.1, color='b')

#%% Simulate true initialization 
params.init_type = "true"
params.T = T
for i in tqdm(range(MC_runs), desc='(2/8) True Init    '):
    data_dict = generate_data(params)
    elbo[i, :], elbo_converged_it[i], _, _, _ = coordinates_ascent(data_dict, params)

# average results
elbo_avg = np.mean(elbo, axis=0)
elbo_converged_it_avg = np.mean(elbo_converged_it)
# 95% confidence interval
(ci_min1, ci_max1) = st.t.interval(confidence = 0.95, df = elbo.shape[0]-1,\
                                loc = elbo_avg,\
                                scale = st.sem(elbo, axis=0))
(ci_min2, ci_max2) = st.t.interval(confidence = 0.95, df = elbo_converged_it.shape[0]-1,\
                                loc = elbo_converged_it_avg,\
                                scale = st.sem(elbo_converged_it, axis=0))
# plot results
ax1.plot(it_counter, elbo_avg, color='g', label="True")
ax1.fill_between(it_counter, ci_min1, ci_max1, color='b', alpha=.1)
ax1.axvline(x=elbo_converged_it_avg, color='g', linestyle='--')
ax1.axvspan(ci_min2, ci_max2, alpha=0.1, color='b')

#%% Simulate permute initialization 
params.init_type = "permute"
params.num_permutations = 10
params.T = T
for i in tqdm(range(MC_runs), desc='(3/8) Permute Init '):
    data_dict = generate_data(params)
    elbo[i, :], elbo_converged_it[i], _, _, _ = coordinates_ascent(data_dict, params)

# average results
elbo_avg = np.mean(elbo, axis=0)
elbo_converged_it_avg = np.mean(elbo_converged_it)
# 95% confidence interval
(ci_min1, ci_max1) = st.t.interval(confidence = 0.95, df = elbo.shape[0]-1,\
                                loc = elbo_avg,\
                                scale = st.sem(elbo, axis=0))
(ci_min2, ci_max2) = st.t.interval(confidence = 0.95, df = elbo_converged_it.shape[0]-1,\
                                loc = elbo_converged_it_avg,\
                                scale = st.sem(elbo_converged_it, axis=0))

# plot results    
ax1.plot(it_counter, elbo_avg, color='r', label="Random")
ax1.fill_between(it_counter, ci_min1, ci_max1, color='r', alpha=.1)
ax1.axvline(x=elbo_converged_it_avg, color='r', linestyle='--')
ax1.axvspan(ci_min2, ci_max2, alpha=0.1, color='r')

#%% Simulate unique initialization 
params.init_type = "unique"
for i in tqdm(range(MC_runs), desc='(4/8) Unique Init  '):
    data_dict = generate_data(params)
    elbo[i, :], elbo_converged_it[i], _, _, _ = coordinates_ascent(data_dict, params)

# average results
elbo_avg = np.mean(elbo, axis=0)
elbo_converged_it_avg = np.mean(elbo_converged_it)
# 95% confidence interval
(ci_min1, ci_max1) = st.t.interval(confidence = 0.95, df = elbo.shape[0]-1,\
                                loc = elbo_avg,\
                                scale = st.sem(elbo, axis=0))
(ci_min2, ci_max2) = st.t.interval(confidence = 0.95, df = elbo_converged_it.shape[0]-1,\
                                loc = elbo_converged_it_avg,\
                                scale = st.sem(elbo_converged_it, axis=0))

# plot results    
ax1.plot(it_counter, elbo_avg, color='c', label="Unique")
ax1.fill_between(it_counter, ci_min1, ci_max1, color='c', alpha=.1)
ax1.axvline(x=elbo_converged_it_avg, color='c', linestyle='--')
ax1.axvspan(ci_min2, ci_max2, alpha=0.1, color='c')

#%% Simulate allinone initialization
params.init_type = "AllInOne"
params.T = T
for i in tqdm(range(MC_runs), desc='(5/8) AllInOne Init'):
    data_dict = generate_data(params)
    elbo[i, :], elbo_converged_it[i], _, _, _ = coordinates_ascent(data_dict, params)

# average results
elbo_avg = np.mean(elbo, axis=0)
elbo_converged_it_avg = np.mean(elbo_converged_it)
# 95% confidence interval
(ci_min1, ci_max1) = st.t.interval(confidence = 0.95, df = elbo.shape[0]-1,\
                                loc = elbo_avg,\
                                scale = st.sem(elbo, axis=0))
(ci_min2, ci_max2) = st.t.interval(confidence = 0.95, df = elbo_converged_it.shape[0]-1,\
                                loc = elbo_converged_it_avg,\
                                scale = st.sem(elbo_converged_it, axis=0))

# plot results    
ax1.plot(it_counter, elbo_avg, color='m', label="One Cluster")
ax1.fill_between(it_counter, ci_min1, ci_max1, color='m', alpha=.1)
ax1.axvline(x=elbo_converged_it_avg, color='m', linestyle='--')
ax1.axvspan(ci_min2, ci_max2, alpha=0.1, color='m')

#%% Simulate kmeans initialization 
params.init_type = "Kmeans"
params.T = T
for i in tqdm(range(MC_runs), desc='(6/8) KMeans Init  '):
    data_dict = generate_data(params)
    elbo[i, :], elbo_converged_it[i], _, _, _ = coordinates_ascent(data_dict, params)

# average results
elbo_avg = np.mean(elbo, axis=0)
elbo_converged_it_avg = np.mean(elbo_converged_it)
# 95% confidence interval
(ci_min1, ci_max1) = st.t.interval(confidence = 0.95, df = elbo.shape[0]-1,\
                                loc = elbo_avg,\
                                scale = st.sem(elbo, axis=0))
(ci_min2, ci_max2) = st.t.interval(confidence = 0.95, df = elbo_converged_it.shape[0]-1,\
                                loc = elbo_converged_it_avg,\
                                scale = st.sem(elbo_converged_it, axis=0))

# plot results   
ax2.plot(it_counter, elbo_avg, color='y', label="KMeans")
ax2.fill_between(it_counter, ci_min1, ci_max1, color='y', alpha=.1)
ax2.axvline(x=elbo_converged_it_avg, color='y', linestyle='--')
ax2.axvspan(ci_min2, ci_max2, alpha=0.1, color='y')

#%% Simulate dbscan initialization
params.init_type = "DBSCAN"
params.T = T
for i in tqdm(range(MC_runs), desc='(7/8) DBSCAN Init  '):
    data_dict = generate_data(params)
    elbo[i, :], elbo_converged_it[i], _, _, _ = coordinates_ascent(data_dict, params)

# average results
elbo_avg = np.mean(elbo, axis=0)
elbo_converged_it_avg = np.mean(elbo_converged_it)
# 95% confidence interval
(ci_min1, ci_max1) = st.t.interval(confidence = 0.95, df = elbo.shape[0]-1,\
                                loc = elbo_avg,\
                                scale = st.sem(elbo, axis=0))
(ci_min2, ci_max2) = st.t.interval(confidence = 0.95, df = elbo_converged_it.shape[0]-1,\
                                loc = elbo_converged_it_avg,\
                                scale = st.sem(elbo_converged_it, axis=0))

# plot results 
ax2.plot(it_counter, elbo_avg, color='k', label="DBSCAN")
ax2.fill_between(it_counter, ci_min1, ci_max1, color='k', alpha=.1)
ax2.axvline(x=elbo_converged_it_avg, color='k', linestyle='--')
ax2.axvspan(ci_min2, ci_max2, alpha=0.1, color='k')

#%% Simulate global initialization 
params.init_type = "global"
params.T = T
for i in tqdm(range(MC_runs), desc='(8/8) Global Init  '):
    data_dict = generate_data(params)
    elbo[i, :], elbo_converged_it[i], _, _, _ = coordinates_ascent(data_dict, params)
    
# average results
elbo_avg = np.mean(elbo, axis=0)
elbo_converged_it_avg = np.mean(elbo_converged_it)
# 95% confidence interval
(ci_min1, ci_max1) = st.t.interval(confidence = 0.95, df = elbo.shape[0]-1,\
                                loc = elbo_avg,\
                                scale = st.sem(elbo, axis=0))
(ci_min2, ci_max2) = st.t.interval(confidence = 0.95, df = elbo_converged_it.shape[0]-1,\
                                loc = elbo_converged_it_avg,\
                                scale = st.sem(elbo_converged_it, axis=0))
    
# plot results   
ax2.plot(it_counter, elbo_avg, color='brown', label="Global")
ax2.fill_between(it_counter, ci_min1, ci_max1, color='brown', alpha=.1)
ax2.axvline(x=elbo_converged_it_avg, color='brown', linestyle='--')
ax2.axvspan(ci_min2, ci_max2, alpha=0.1, color='brown')

#%% Set plot properties and save fig as pdf
xlim = 30
ax1.set_xlim((1,xlim))
ax1.xaxis.set_minor_locator(MultipleLocator(1))
ax1.grid()
ax1.legend(loc = 'center right')
ax2.set_xlim((1,xlim))
ax2.xaxis.set_minor_locator(MultipleLocator(1))
ax2.grid()
ax2.legend(loc = 'center right')
fig.tight_layout()

plt.savefig("results/convergence_elbo.pdf", format="pdf", bbox_inches="tight")