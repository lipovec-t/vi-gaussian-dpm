# TODO - each of the following shall be represented by its own function
# full postprocessing that includes everything below, can be called in sim file
import numpy as np
 
def est_clustering_map(phi):
    """
    MAP estimates of cluster assignments given soft assignments.

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
    MMSE estimate of the cluster weights given variational parameter gamma.

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
    
# MMSE estimate for cluster means
def est_cluster_means_mmse(tau):
    """
    MMSE stimate of the cluster means given variational parameter tau.

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

def cluster_sample_mean(data, cluster_indicators, num_clusters):
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
    counts = np.repeat(counts[:,np.newaxis], K, axis=1)
    cluster_sample_mean = np.divide(cluster_sample_mean, counts,\
                                    where = counts > 0)
    cluster_sample_mean[counts == 0] = np.nan
    return cluster_sample_mean

# MMSE estimator for cluster assigments?

# Sample mean of clustered data
# MAP estimator for cluster means?

# MAP estimator for cluster weights?
# plot posterior (contour plot, only for K=2, maybe also include K=1)
# plot estimated cluster assignments, cluster means and cluster weights



