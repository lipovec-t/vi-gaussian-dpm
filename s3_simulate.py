# Third party imports
import numpy as np
from tqdm import tqdm

# Local application imports
from data.generation import generate_data
from vi.cavi import coordinates_ascent
from vi import postprocessing as pp
from s3_config import Params

# random seed for testing purposes
np.random.seed(255)

# load parameters
params = Params()
# number of MC runs
MC_runs = 1000

# save results?
save_results = True

# sim_data = list which stores results and ground truth for each simulation run
# such that:
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
        pp.full_postprocessing(data_dict, phi, gamma, tau, True)
    sim_data[i] = dict(data_dict, **results_reduced)
    
# save results
if save_results:
    pp.save_results(sim_data, 'res-')
    
# plot results of one of the MC runs
MC_run_index = 0
data = sim_data[MC_run_index]["Datapoints"]

title = "Clustering DPM - MMSE Mean"
indicatorArray = sim_data[MC_run_index]["MMSE Estimated Cluster Indicators"]
meanArray = sim_data[MC_run_index]["MMSE Estimated Cluster Means"]
meanIndicators = sim_data[MC_run_index]["MMSE Mean Indicators"]
pp.plot_clustering(data, title,\
                   indicatorArray, meanArray, meanIndicators=meanIndicators)

title = "Clustering DPM - Cluster Sample Mean"
indicatorArray = sim_data[MC_run_index]["Sample Estimated Cluster Indicators"]
meanArray = sim_data[MC_run_index]["Sample Mean of Clusters"]
meanIndicators = sim_data[MC_run_index]["Sample Mean Indicators"]
pp.plot_clustering(data, title,\
                   indicatorArray, meanArray, meanIndicators=meanIndicators)
        
# produce desired metrics with simulation results
est_cluster_number = np.zeros(MC_runs)
true_cluster_number = np.zeros(MC_runs)
counter = 0
cluster_mean_distances = np.zeros(MC_runs)
accuracy_score = np.zeros(MC_runs)
c = 0.3 # cut-off distance
p = 2 # order of the OSPA metric
OSPA = np.zeros(MC_runs)
rmse = np.zeros(MC_runs)

for i in range(MC_runs):
    est_cluster_number[i] = sim_data[i]["Estimated Number of Clusters"]
    true_cluster_number[i] = sim_data[i]["True Number of Clusters"]
    true_cluster_means = sim_data[i]["True Cluster Means"]
    est_cluster_means = sim_data[i]["MMSE Estimated Cluster Means"]
    true_indicators = sim_data[i]["True Cluster Indicators"]
    est_indicators = sim_data[i]["MMSE Estimated Cluster Indicators"]
    mean_indicators = sim_data[i]["MMSE Mean Indicators"]
    
    accuracy_score[i] = pp.accuracy_score(true_indicators, est_indicators)
    
    OSPA[i], cluster_mean_distances[i], rmse[i] = pp.OSPA(true_cluster_means,\
                                                 est_cluster_means,\
                                                 mean_indicators, c, p)
    
mean_cluster_number = np.mean(est_cluster_number)
accuracy_clusters = np.mean(est_cluster_number == true_cluster_number) * 100
accuracy_means = np.mean(cluster_mean_distances)
mean_accuracy_score = np.mean(accuracy_score)
mean_OSPA = np.mean(OSPA)
mean_RMSE = np.mean(rmse)

print("\n")
print("DPM SIMULATION")
print(f"Mean Estimated Clusters = {mean_cluster_number}")
print(f"Accuracy Estimated Cluster Numbers = {accuracy_clusters}%")
print(f"Average Distance to Real Cluster Means = {accuracy_means}")
print(f"Average Accuracy Score = {mean_accuracy_score}")
print(f"OSPA Parameters: Cut-off distance = {c}, Order = {p}")
print(f"Average OSPA = {mean_OSPA}")
print(f"Average RMSE (including cutoff) = {mean_RMSE}")