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
est_cluster_number = np.zeros(MC_runs)
counter = 0
cluster_mean_distances = np.zeros(MC_runs)
accuracy_score = np.zeros(MC_runs)
OSPA = np.zeros(MC_runs)

for i in range(MC_runs):
    est_cluster_number[i] = sim_data[i]["Estimated Number of Clusters"]
    true_cluster_means = sim_data[i]["True Cluster Means"]
    est_cluster_means = sim_data[i]["Reordered Estimated Cluster Means"]
    true_indicators = sim_data[i]["True Cluster Indicators"]
    est_indicators = sim_data[i]["Relabelled Estimated Cluster Indicators"]
    mean_indicators = sim_data[i]["Mean Indicators"]
    
    accuracy_score[i] = pp.accuracy_score(true_indicators, est_indicators)
    
    OSPA[i], cluster_mean_distances[i] = pp.OSPA(true_cluster_means,\
                                                 est_cluster_means,\
                                                 mean_indicators)
    
mean_cluster_number = np.mean(est_cluster_number)
accuracy_clusters = np.mean(est_cluster_number == 8) * 100
accuracy_means = np.mean(cluster_mean_distances)
mean_accuracy_score = np.mean(accuracy_score)
mean_OSPA = np.mean(OSPA)


print("\n")
print("MFM SIMULATION")
print(f"Mean Estimated Clusters = {mean_cluster_number}")
print(f"Accuracy Estimated Cluster numbers = {accuracy_clusters}%")
print(f"Average distance to real cluster means = {accuracy_means}")
print(f"Average accuracy score = {mean_accuracy_score}")
print(f"Average OSPA = {mean_OSPA}")