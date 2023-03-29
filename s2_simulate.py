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

# random seed for testing purposes
np.random.seed(255)

# load parameters
params = Params()
K = params.K
alpha = params.alpha

# set object count and number of MC runs
N_array = np.arange(1,51)
MC_runs = 50

# Store MSE for each simulation run
MSE_x = np.zeros((N_array.size, MC_runs))

# Store max elbo for each monte carlo run
elbo_final = np.zeros(N_array.size)
elbo_final[:] = -np.inf 

# start timer
t_0 = timeit.default_timer()

for i,N in enumerate(tqdm(N_array)):
    for j in range(MC_runs):
        # generate data
        params.N = N
        data_dict = generate_data(params)
        
        # CAVI
        elbo, tau, gamma, phi = coordinates_ascent(data_dict, params)
        if elbo > elbo_final[i]:
            elbo_final[i] = elbo
               
        # postprocessing
        results, results_reduced = \
            pp.full_postprocessing(data_dict, phi, gamma, tau, False)
        
        # mmse estimator for x
        y = data_dict["Noisy Datapoints"]
        x_est = pp.est_object_positions(y, results, params)

        # calculate mse
        x = data_dict["Datapoints"]
        MSE_x[i,j] = pp.mse(x, x_est, K*N)
        
# end timer and compute elapsed time
t_1 = timeit.default_timer()

runtime = t_1 - t_0

# mean MSE
MSE_x_avg = np.mean(MSE_x, axis=1)

# 95% confidence interval
(ci_min, ci_max) = st.t.interval(confidence = 0.95, df = MSE_x.shape[1]-1,\
                                loc = MSE_x_avg, scale = st.sem(MSE_x, axis=1)) 
    
# Plot
fig, ax = plt.subplots()
label = r'MSE VI $\alpha = {}$'.format(alpha)
ax.plot(N_array, MSE_x_avg, color = 'b', label=label)
label = r'$95\%$ CI'
ax.fill_between(N_array, ci_min, ci_max, color = 'b', alpha = .1, label=label)
plt.xlabel('Number of objects')
plt.ylabel('Average MSE')
plt.legend()
plt.grid()
