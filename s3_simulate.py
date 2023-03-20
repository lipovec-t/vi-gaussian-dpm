# Third party imports
import numpy as np
from tqdm import tqdm
import time
import pickle

# Local application imports
from data.generation import generate_data
from vi.cavi import coordinates_ascent
from vi import postprocessing as pp
from s3_config import Params

# NOTE: THIS IS SIMULATION SCENARIO IS WIP

# random seed for testing purposes
np.random.seed(255)

# load parameters
params = Params()

# number of MC runs
MC_runs = 100

save_results = True

# results = nested list which stores results for each simulation run such that:
# res[iMCrun][0] -> results_reduced
res = [[] for j in range(MC_runs)]

for i in tqdm(range(MC_runs)):
    # generate data
    indicator_array, cluster_assignments, cluster_means, data, _ = \
        generate_data(params)
    params.true_assignment = cluster_assignments
    # CAVI
    elbo_final, tau, gamma, phi = coordinates_ascent(data, params)
    res[i][:] = elbo_final, tau, gamma, phi
    # postprocessing
    results, results_reduced =\
        pp.full_postprocessing(data, phi, gamma, tau, False)
    res[i] = results_reduced
    
# save results
if save_results:
    timestr = time.strftime("%Y%m%d-%H%M%S")
    name = 'res-'
    filename = name + timestr + '.pkl'
    fp = open(filename, 'x')
    fp.close()
    with open(filename, 'wb') as f:
        pickle.dump(res, f)
        
# produce desired results
mean_cluster_number = 0
counter = 0
mean_distances = [0 for j in range(MC_runs)]
for i in range(MC_runs):
    T_est = res[i]["Estimated Number of Clusters"]
    if T_est == 8:
        counter += 1
    cluster_mean_est = res[i]["Estimated Cluster Means"]
    mean_cluster_number += T_est
    metric = np.zeros((T_est, 8))
    for j in range(T_est):
        distance = np.linalg.norm(cluster_means-cluster_mean_est[j,:], axis=1)
        metric[j,:] = distance
    min_distance = np.min(metric, axis=0)
    mean_distances[i] = np.mean(min_distance)
    
mean_cluster_number = mean_cluster_number / MC_runs
accuracy_clusters = counter / MC_runs * 100
accuracy_means = np.mean(mean_distances)


print("\n")
print("DPM SIMULATION")
print(f"Mean Estimated Clusters = {mean_cluster_number}")
print(f"Accuracy Estimated Cluster numbers = {accuracy_clusters}%")
print(f"Average distance to real cluster means = {accuracy_means}")