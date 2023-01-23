import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from crp_generation import crp

def generate_data(N, alpha, mu_G, sigma_G, mu_U, sigma_U, mu_V, sigma_V, plot):
    # base distribution
    G0 = sp.stats.multivariate_normal(mean = mu_G, cov = sigma_G)
    
    # Dirichlet process
    indicator_array = crp(N, alpha)
    num_clusters = max(indicator_array)+1
    cluster_means = G0.rvs(num_clusters)
    
    if num_clusters == 1:
        cluster_means = np.repeat(cluster_means[np.newaxis,:], 1, axis = 0)
    
    # Dirichlet process mixture
    x = np.empty((N,2))
    for i in range(N):
        mean = cluster_means[indicator_array[i]] + mu_U
        x[i,:] = sp.stats.multivariate_normal.rvs(mean = mean, cov = sigma_U, size = 1)
        
    # Measurement noise
    v = sp.stats.multivariate_normal.rvs(mean = mu_V, cov = sigma_V, size = N)
    y = x + v
    
    if plot:
        plt.figure()
        colormap = plt.cm.get_cmap('tab20', num_clusters)
        plt.title("Data")
        plt.scatter(cluster_means[:,0], cluster_means[:,1], c = colormap(range(num_clusters)), marker = "o")
        plt.scatter(x[:,0], x[:,1], c = colormap(indicator_array), marker = '.')
    
    return indicator_array, cluster_means, x, y
