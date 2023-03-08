import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

def generate_data_rp(N, alpha, mu_G, sigma_G, mu_U, sigma_U, mu_V, sigma_V, rp, plot):
    """
    Generation of data according to DPM or MFM
    """
    # base distribution
    G0 = multivariate_normal(mean = mu_G, cov = sigma_G)
    
    # Restaurant process
    indicator_array = rp(N, alpha)
    num_clusters = max(indicator_array)+1
    cluster_means = G0.rvs(num_clusters)
    num_clusters = cluster_means.shape[0]
    cluster_assignements = np.zeros((N,num_clusters))
    # TODO: vectorize this
    for i in range(N):
        cluster_assignements [i,indicator_array[i]] = 1
    
    if num_clusters == 1:
        cluster_means = np.repeat(cluster_means[np.newaxis,:], 1, axis = 0)
    
    # Mixture
    K = len(mu_G)
    x = np.empty((N,K))
    for i in range(N):
        mean = cluster_means[indicator_array[i]] + mu_U
        x[i,:] = multivariate_normal.rvs(mean = mean, cov = sigma_U, size = 1)
        
    # Measurement noise
    v = multivariate_normal.rvs(mean = mu_V, cov = sigma_V, size = N)
    y = x + v
    
    if plot:
        plt.figure()
        colormap = plt.cm.get_cmap('tab20', num_clusters)
        plt.title("Data")
        plt.scatter(cluster_means[:,0], cluster_means[:,1], c = colormap(range(num_clusters)), marker = "o")
        plt.scatter(x[:,0], x[:,1], c = colormap(indicator_array), marker = '.')
    
    return indicator_array, cluster_assignements, cluster_means, x, y

def generate_data_gm(N, num_clusters, mu_G, sigma_G, mu_U, sigma_U, mu_V, sigma_V, plot):
    """
    Generation of data according to a Gaussian mixture
    """
    indicator_array = np.zeros(N, int)
    
    points_per_cluster = N//num_clusters

    # Indicator array
    j = 0
    for i in range(N):
        indicator_array[i] = j
        if (i+1) % points_per_cluster == 0:
            j = j+1
    # distribute additional data points over clusters for the case where N/num_cluster != int
    for i in range(N%num_clusters):
        indicator_array[N-(i+1)] = i
        
    cluster_assignements = np.zeros((N,num_clusters))
    # TODO: vectorize this
    for i in range(N):
        cluster_assignements [i,indicator_array[i]] = 1

    # Draw cluster means 
    G0 = multivariate_normal(mean = mu_G, cov = sigma_G)
    cluster_means = G0.rvs(num_clusters)

    # Draw datapoints
    K = len(mu_G)
    x = np.empty((N,K))
    j = 0
    for i in range(N):
        x[i,:] = multivariate_normal.rvs(mean = cluster_means[indicator_array[i]], cov = sigma_U, size = 1)
            
    # Measurement noise
    v = multivariate_normal.rvs(mean = mu_V, cov = sigma_V, size = N)
    y = x + v

    if plot:       
        plt.figure()
        colormap = plt.cm.get_cmap('tab20', num_clusters)
        plt.title("Data")
        plt.scatter(cluster_means[:,0], cluster_means[:,1], c = colormap(range(num_clusters)), marker = "o")
        j = 0
        plt.scatter(x[:,0], x[:,1], c = colormap(indicator_array), marker = ".")
        
    return indicator_array, cluster_assignements, cluster_means, x, y
