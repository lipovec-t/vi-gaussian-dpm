# Third party imports
import numpy as np
from tqdm import tqdm

# Local application imports
from data.generation import generate_data
from vi.cavi import coordinates_ascent
from vi import postprocessing as pp
from s3_config import Params

# same simulation as in s3_simulate but for different values of N

# random seed for testing purposes
np.random.seed(255)

# Set number of datapoints and MC runs
N_array = np.arange(50,3050,50)
MC_runs = 500

# Store metrics for each simulation run 
est_cluster_number = np.zeros((N_array.size,MC_runs))
cluster_mean_distances = np.zeros((N_array.size,MC_runs))
accuracy_score = np.zeros((N_array.size,MC_runs))
OSPA = np.zeros((N_array.size,MC_runs))
rmse = np.zeros((N_array.size,MC_runs))
true_cluster_number = np.zeros(MC_runs)

# Function for displaying progress of N-loop and MC-run loop
def myprogress(curr, N, width=8, bars = u'▉▊▋▌▍▎▏ '[::-1],full='█', empty=' '): 
    p = curr / N 
    nfull = int(p * width)
    return "{:>3.0%} |{}{}{}| {:>2}/{}"\
        .format(p, full * nfull,
                bars[int(len(bars) * ((p * width) % 1))],
                empty * (width - nfull - 1),
                curr, N)

# Parameters for OSPA
c = 1
p = 2

# sim_data:
# list of dictionaries which stores results and ground truth for each sim run
# such that:
# sim_data[iN][iMCrun] -> results_reduced + data_dict
sim_data = [dict() for i in range(N_array.size)]
for j in range(N_array.size):
    sim_data[j] = [[] for i in range(MC_runs)]


# load parameters
params = Params()

# save results?
save_results = True

pbar = tqdm(N_array,
                 bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')

for j,N in enumerate(pbar):  
    
    for i in range(MC_runs):
        pbar.set_postfix_str(myprogress(i, MC_runs))
        params.N = N
        # generate data
        data_dict = generate_data(params)
        # CAVI
        elbo_final, tau, gamma, phi = coordinates_ascent(data_dict, params)
        # postprocessing
        data = data_dict["Datapoints"] # extract data from data_dict
        _, results_reduced =\
            pp.full_postprocessing(data_dict, phi, gamma, tau, True)
        sim_data[j][i] = dict(data_dict, **results_reduced)
        
    for i in range(MC_runs):
        est_cluster_number[j,i] = sim_data[j][i]["Estimated Number of Clusters"]
        true_cluster_number[i] = sim_data[j][i]["True Number of Clusters"]
        true_cluster_means = sim_data[j][i]["True Cluster Means"]
        est_cluster_means = sim_data[j][i]["MMSE Estimated Cluster Means"]
        true_indicators = sim_data[j][i]["True Cluster Indicators"]
        est_indicators = sim_data[j][i]["MMSE Estimated Cluster Indicators"]
        mean_indicators = sim_data[j][i]["MMSE Mean Indicators"]
        
        accuracy_score[j,i] = pp.accuracy_score(true_indicators, est_indicators)
        
        OSPA[j,i], cluster_mean_distances[j,i], rmse[j,i] \
            = pp.OSPA(true_cluster_means, est_cluster_means, mean_indicators,\
                      c, p)
    

mean_cluster_number = np.mean(est_cluster_number,axis=1)
accuracy_clusters = np.mean(est_cluster_number == true_cluster_number,axis = 1) * 100
accuracy_means = np.mean(cluster_mean_distances,axis=1)
mean_accuracy_score = np.mean(accuracy_score,axis = 1)
mean_OSPA = np.mean(OSPA, axis = 1)
mean_RMSE = np.mean(rmse,axis = 1)

