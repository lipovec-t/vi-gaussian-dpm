# Third party imports
import numpy as np

# Local application imports
from vi import postprocessing as pp

filename = 'file.pkl'
sim_data = pp.load_results(filename)

# plot results of one of the MC runs
MC_run_index = 1
data = sim_data[MC_run_index]["Datapoints"]

title = "Data"
indicatorArray = sim_data[MC_run_index]["True Cluster Indicators"]
meanArray = sim_data[MC_run_index]["True Cluster Means"]
pp.plot_clustering(data, title, indicatorArray, meanArray)

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
MC_runs = len(sim_data)
est_cluster_number = np.zeros(MC_runs)
true_cluster_number = np.zeros(MC_runs)
cluster_mean_distances = np.zeros(MC_runs)
accuracy_score = np.zeros(MC_runs)
OSPA = np.zeros(MC_runs)
c = 0.3 # cut-off distance
p = 2 # order of the OSPA metric
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

