import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal, multinomial

def generate_data(N, alpha, mu_G, sigma_G, mu_U, sigma_U, mu_V, sigma_V, plot):
    """
    Generation of data
    """
    # base distribution
    G0 = multivariate_normal(mean = mu_G, cov = sigma_G)
    
    # Dirichlet process
    indicator_array = crp(N, alpha)
    num_clusters = max(indicator_array)+1
    cluster_means = G0.rvs(num_clusters)
    num_clusters = cluster_means.shape[0]
    cluster_assignements = np.zeros((N,num_clusters))
    # TODO: vectorize this
    for i in range(N):
        cluster_assignements [i,indicator_array[i]] = 1
    
    if num_clusters == 1:
        cluster_means = np.repeat(cluster_means[np.newaxis,:], 1, axis = 0)
    
    # Dirichlet process mixture
    x = np.empty((N,2))
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

def crp(N, alpha):
    """
    Chinese Restaurant Process
    """
    counts = []
    assignmentArray = np.empty(N, int)
    n = 0
    while n < N:
        # Compute the (unnormalized) probabilities of assigning the new object
        # to each of the existing groups, as well as a new group
        assign_probs = [None] * (len(counts) + 1)
        
        for i in range(len(counts)):
            assign_probs[i] = counts[i] / (n + alpha)
            
        assign_probs[-1] = alpha / (n + alpha)
        
        # Draw the new object's assignment from the discrete distribution
        multinomialDist = multinomial(1, assign_probs)
        assignment = multinomialDist.rvs(1)
        assignment = np.where(assignment[0] == 1)
        assignment = int(assignment[0])
        assignmentArray[n] = assignment
        
        # Update the counts for next time, adding a new count if a new group was
        # created
        if assignment == len(counts):
            counts.append(0)
          
        counts[assignment] += 1
        n += 1
    
    return assignmentArray
