import numpy as np
import matplotlib.pyplot as plt
import time
import pickle
from scipy.spatial import distance
from sklearn.metrics import accuracy_score as acc_score

def full_postprocessing(data_dict, phi, gamma, tau, relabel):
    """
    Performs full postprocessing given the simulation results

    Parameters
    ----------
    data : ndarray
        NxK data array.
    phi : ndarray
        NxT variational parameter phi representing soft assignments
    gamma : TYPE
        Tx2 variational parameter gamma representing determining mixture weight
        distribution.
    tau : TYPE
        Tx(K+1) variational parameter tau describing the exp. fam. mixture 
        distribution.
    relabel : boolean
        Determines if results shall be relabelled.

    Returns
    -------
    results : dict
        Dictonary holding the results for the T mixture components of the 
        fitted model.
    results_reduced : dict
        Dictonary holding the results for the non empty clusters out of the T
        mixture components.

    """
    # Extract datapoints form data_dict
    data = data_dict["Datapoints"]
    
    # number of mixture componentes in the fitted model
    T = gamma.shape[0]
    
    # MAP estimate of the cluster assignements
    cluster_indicator_est = est_clustering_map(phi)
    
    # MMSE estimate of the cluster weights
    cluster_weights_est = est_cluster_weights_mmse(gamma)
    
    # MMSE estimate of the cluster means 
    cluster_means_est = est_cluster_means_mmse(tau)
    
    # Sample mean of the clusters
    cluster_sample_mean, cluster_sample_weight, _ = \
        est_cluster_sample_mean(data, cluster_indicator_est, T)
        
    # Put results in dictionary
    results = {
        "Estimated Cluster Indicators"  : cluster_indicator_est,
        "Estimated Cluster Weights"     : cluster_weights_est,
        "Estimated Cluster Means"       : cluster_means_est,
        "Estimated Number of Clusters"  : T,
        "Sample Mean of Clusters"       : cluster_sample_mean,
        "Sample Weight of Clusters"     : cluster_sample_weight
        }
    
    # Get reduced results
    results_reduced = reduce_results(results)
    
    # Reorder reduced results
    if relabel == True:
        results_reduced = reorder_results(results_reduced, data_dict)
    
    return results, results_reduced

def est_clustering_map(phi):
    """
    Approx. MAP estimates of cluster assignments given soft assignments.

    Parameters
    ----------
    phi : ndarray
        NxT array of soft assignments.

    Returns
    -------
    Estimated cluster indicator for each datapoint.

    """
    return np.argmax(phi,axis=1)

def est_cluster_weights_mmse(gamma):
    """
    Approx. MMSE est. of the cluster weights given variational parameter gamma.

    Parameters
    ----------
    gamma : ndarray
        Tx2 variational parameter gamma describing T beta distributions.

    Returns
    -------
    Estimated cluster weight for each of the T clusters.

    """
    T = gamma.shape[0]
    V = np.divide(gamma[:,0],np.sum(gamma,axis=1))
    pi = np.zeros(np.shape(V))
    pi[0] = V[0]
    for i in range (1,T):
        temp = 1
        for j in range (i):
                temp = temp*(1-V[j])
        pi[i] = V[i]*temp
    return pi
    
# MMSE estimate for cluster means
def est_cluster_means_mmse(tau):
    """
    Approx. MMSE estimate of the cluster means given variational parameter tau.

    Parameters
    ----------
    tau : ndarray
        Tx(K+1) variational parameter describing T exp. fam. distributions.

    Returns
    -------
    Estimated cluster mean for each of the T clusters.

    """
    cluster_means_est = tau[:,:-1]/np.repeat(tau[:,-1,np.newaxis], 1, axis=1)
    return cluster_means_est

def est_cluster_sample_mean(data, cluster_indicators, num_clusters):
    """
    Sample means of the clusters. Sample mean of empty clusters is set to NaN.
    
    Parameters
    ----------
    data : ndarray
        NxK data array.
    cluster_indicators : ndarray
        Nx1 cluster indicators.

    Returns
    -------
    None.

    """
    N = data.shape[0]
    K = data.shape[1]
    cluster_sample_mean = np.zeros((num_clusters, K))
    counts = np.zeros(num_clusters)
    for i in range(N):
        cluster_sample_mean[cluster_indicators[i],:] += data[i]
        counts[cluster_indicators[i]] += 1
    cluster_sample_weights = np.divide(counts, N)
    counts = np.repeat(counts[:,np.newaxis], K, axis=1)
    cluster_sample_mean = np.divide(cluster_sample_mean, counts,\
                                    where = counts > 0)
    cluster_sample_mean[counts == 0] = np.nan
    return cluster_sample_mean, cluster_sample_weights, counts

def reduce_results(results):
    """
    Deletes results for empty clusters.

    Parameters
    ----------
    results : tuple
        Tuple containing different results for T clusters.

    Returns
    -------
    Tuple containing reduced results with clusters numbered from 0 to the 
    number of non-empty clusters.

    """
    eci = results["Estimated Cluster Indicators"]
    indicators_unique = np.unique(eci)
    # map estimated cluster indicators to a range from 0 to T_est-1
    mapper = lambda i: np.where(indicators_unique == i)[0]
    cluster_indicator_est = np.array(list(map(mapper, eci)))[:,0]
    results_reduced = dict()
    for key, value in results.items():
        if key == "Estimated Cluster Indicators":
            results_reduced[key] = cluster_indicator_est
        elif key == "Estimated Number of Clusters":
            results_reduced[key] = np.size(indicators_unique)
        else:
            results_reduced[key] = results[key][indicators_unique]
    return results_reduced

def reorder_results(results_reduced, data_dict):
    """
    Reorder results such that the estimated clusters have minimal distance to
    the true cluster locations. The reordering is performed for both the MMSE
    estimation results and the sample mean results.

    Parameters
    ----------
    results_reduced : dict
        Contains results for non-empty clusters
    data_dict : dict
        Contains ground truth.

    Returns
    -------
    result_reduced with relabelled/reordered indicators/means according to data

    """
    N = len(results_reduced["Estimated Cluster Indicators"])
    T_est = results_reduced["Estimated Number of Clusters"]
    T = data_dict["True Number of Clusters"]
    cluster_means = data_dict["True Cluster Means"]
    
    labels_est = results_reduced["Estimated Cluster Indicators"]
    
    # MMSE estimation results
    mmse_means = results_reduced["Estimated Cluster Means"]
    mmse_weights = results_reduced["Estimated Cluster Weights"]
    
    # Sample mean estimation results
    sample_means = results_reduced["Sample Mean of Clusters"]
    sample_weights = results_reduced["Sample Weight of Clusters"]
    
    # Compute distances between each pair of estimated and true cluster mean
    metric_mmse   = distance.cdist(mmse_means, cluster_means)
    metric_sample = distance.cdist(sample_means, cluster_means)
    
    # Relabel/Reorder the clusters for both the mmse and sample estimator case
    for i, metric in enumerate([metric_mmse, metric_sample]):
        label_mapping = np.argmin(metric, axis=1)
        
        # False in skip array means that it is an additional cluster
        skip_array = np.full((T_est,), True)
        indices, counts = np.unique(label_mapping, return_counts=True)
        duplicates = np.where(counts>1)[0]
        for j in duplicates:
            duplicates_indices = np.asarray(label_mapping==indices[j]).nonzero()[0]
            skip_array[duplicates_indices] = False
            duplicates_metrics = metric[duplicates_indices,indices[j]]
            duplicate_min_index = duplicates_indices[np.argmin(duplicates_metrics)]
            skip_array[duplicate_min_index] = True
        
        # indices in label_mapping of additional clusters
        add_cluster = np.arange(T_est)[~skip_array]
        T_add = add_cluster.size
    
        relabelled_est_ind = np.zeros(N).astype(int)
        if T_est >= T:          
            label_mapping[add_cluster] = np.arange(T, T+T_add)
        elif T_est < T:
            label_mapping[add_cluster] = np.arange(T, T+T_add)
            
        for j in range(T_est):
            temp = np.where(labels_est == j)
            relabelled_est_ind[temp] = label_mapping[j]
            
        if i == 0:   # MMSE CASE
            results_reduced["MMSE Estimated Cluster Indicators"] =\
                relabelled_est_ind
            # save association in mean indicators
            results_reduced["MMSE Estimated Cluster Means"] = mmse_means
            results_reduced["MMSE Estimated Cluster Weights"] = mmse_weights
            results_reduced["MMSE Mean Indicators"] = label_mapping
            
        elif i == 1: # SAMPLE CASE
            results_reduced["Sample Estimated Cluster Indicators"] =\
                relabelled_est_ind
            # save association in mean indicators
            results_reduced["Sample Mean of Clusters"] = sample_means 
            results_reduced["Sample Estimated Cluster Weights"] = sample_weights
            results_reduced["Sample Mean Indicators"] = label_mapping
    
    # delete unnecessary dictionary entries
    del results_reduced["Estimated Cluster Indicators"]
    del results_reduced["Estimated Cluster Means"]
    del results_reduced["Estimated Cluster Weights"]
    del results_reduced["Sample Weight of Clusters"]
        
    return results_reduced

def est_object_positions(y, results, params):
    """
    Computes the approx. MMSE estimator of the object features x given noisy y.
    The true MMSE estimator assumes the local parameters theta_n to be known.
    Model: y_n = x_n + v_n where v is AWGN. 

    Parameters
    ----------
    results_reduced : dict
        Estimated statistics of x.
    params : Params
        Model statistics (Covariance of x and v).

    Returns
    -------
    Estimated object positions x_est.

    """
    indicator_array_est = results["Estimated Cluster Indicators"]
    cluster_means_est = results["Estimated Cluster Means" ]
    mean_x = cluster_means_est[indicator_array_est,:]
    mmse_weight = np.matmul(params.sigma_U, params.sigma_inv)
    weighted_centered_x = np.einsum('ij,kj->ki', mmse_weight, y - mean_x)
    x_est = mean_x + weighted_centered_x
    return x_est

def mse(actual, predicted, normalizer):
    """
    Compute Mean Square Error (MSE) between actual and predicted value.

    Parameters
    ----------
    actual : ndarray
        True data values.
    predicted : ndarray
        Actual data values.
    normalizer : float
        Normalizing constant.

    Returns
    -------
    MSE.

    """
    MSE = 1/normalizer * np.sum(np.linalg.norm(actual - predicted, axis=1)**2)
    return MSE

def accuracy_score(ground_truth, predicted):
    """
    Computes accuracy score with sklearn library.
    Number of correctly labelled data points divided by total # data points.

    Parameters
    ----------
    ground_truth : ndarray of shape (N,)
        Ground truth cluster labels.
    predicted : ndarray of shape (N,)
        Predicted cluster labels.

    Returns
    -------
    float
        Accuaracy score.

    """
    return acc_score(ground_truth, predicted)

def OSPA(x, y, indicators, c, p):
    """
    Computes the traditional OSPA metric according to:
    https://de.mathworks.com/help/fusion/ref/trackospametric-system-object.html

    Parameters
    ----------
    x : ndarray of shape (N,K)
        Ground truth positions.
    y : ndarray of shape (M,K)
        Estimated positions.
    indicators : ndarray of shape (M,)
        Association of y to x.
    c : float
        cut-off distance
    p : int
        order of the OSPA metric

    Returns
    -------
    OSPA : float
        Traditional OSPA metric.

    """
    # number of objects
    x_len = x.shape[0]
    y_len = y.shape[0]
    
    if x_len < y_len:
        m, n = x_len, y_len
    else:
        m, n = y_len, x_len
        
    # save association between objects in x_temp and y_temp
    temp = np.isin(indicators, np.arange(x_len))
    x_temp = x[indicators[temp]]
    y_temp = y[temp]
    
    # distances between given objects
    distanceMatrix = distance.cdist(x_temp, y_temp)
    # normalized error squared
    distanceVector = np.diag(distanceMatrix)
    # cut-off based distance
    d_c = np.minimum(distanceVector, c)
    # localization error component
    d_loc = 1/n * np.sum(d_c**p)
    # cardinality error component
    d_card = (n-m)/n * c**p
    # traditional OSPA metric
    OSPA = (d_loc + d_card)**(1/p)
    # average eucledian distances between objects
    dist = np.mean(distanceVector)
    # RMSE
    rmse = np.sqrt(d_loc)
     
    return OSPA, dist, rmse
   
def plot_clustering(data, title, indicatorArray, meanArray, **kwargs):
    """
    Plot data with cluster indicators and cluster means.

    Parameters
    ----------
    data : ndarray
        NxK data array.
    title : string
        Title of plot.
    indicatorArray : ndarray
        Nx1 cluster indicators for data points.
    meanArray : ndarray
        Means of clusters.

    Returns
    -------
    None.

    """
    if "meanIndicators" in kwargs.keys():
        meanIndicators = kwargs["meanIndicators"]
    else:
        meanIndicators = np.unique(indicatorArray)
        
    plt.figure(figsize=(2.5,2))
    plt.title(title)
    T = meanArray.shape[0]
    if T > 20:
        print('More clusters than colors')
    colormap = plt.cm.get_cmap('tab20', 20)
    cx = meanArray[:,0]
    cy = meanArray[:,1]
    plt.scatter(cx, cy, c=colormap(meanIndicators),\
                marker="o", s=28, edgecolors="black")
    da, dy = data[:,0], data[:,1]
    plt.scatter(da, dy, c=colormap(indicatorArray), marker='.', s=20)
    plt.xlabel(r"$y_{n,1}$", labelpad=-6)
    plt.ylabel(r"$y_{n,2}$", labelpad=-6)

    

def plot_posterior(means, covariances, weights):
    #TODO for the case K = 2
    return

def save_results(result, name):
    """
    save variable result as pickle file

    Parameters
    ----------
    result : dict
        Dictonary with simulation results.
    name : string
        Additionally to the current time included in filename.

    Returns
    -------
    None.

    """
    timestr = time.strftime("%Y%m%d-%H%M%S")
    filename = name + timestr + '.pkl'
    fp = open(filename, 'x')
    fp.close()
    with open(filename, 'wb') as f:
        pickle.dump(result, f)
        
def load_results(filename):
    with open(filename, 'rb') as f: 
           return pickle.load(f)





