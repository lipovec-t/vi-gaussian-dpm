# Third party imports
import numpy as np
from tqdm import tqdm

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
MC_runs = 10

# save results?
save_results = True

# sim_data = list which stores results and ground truth for each simulation run such that:
# sim_data[iMCrun] -> results_reduced + data_dict
sim_data = [[] for j in range(MC_runs)]

for i in tqdm(range(MC_runs)):
    # generate data
    data_dict = generate_data(params)
    # CAVI
    elbo_final, tau, gamma, phi = coordinates_ascent(data_dict, params)
    # postprocessing
    data = data_dict["Datapoints"] # extract data from data_dict
    _, results_reduced =\
        pp.full_postprocessing(data_dict, phi, gamma, tau, False)
    sim_data[i] = dict(data_dict, **results_reduced)
    
# save results
if save_results:
    pp.save_results(sim_data, 'res-')
        
# produce desired results
mean_cluster_number = 0
counter = 0
mean_distances = [0 for j in range(MC_runs)]
for i in range(MC_runs):
    cluster_means = sim_data[i]["True Cluster Means"]
    T_est = sim_data[i]["Estimated Number of Clusters"]
    if T_est == 8:
        counter += 1
    cluster_mean_est = sim_data[i]["Estimated Cluster Means"]
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