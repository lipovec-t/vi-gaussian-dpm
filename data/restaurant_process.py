# Third party imports
import numpy as np
import matplotlib.pyplot as plt

# Local application imports
from scipy.stats import multinomial, poisson
from scipy.special import loggamma

def rp_dpm(N, alpha):
    """
    Chinese restaurant process for the dirichlet process.

    Parameters
    ----------
    N : int
        Number of indicators to be generated.
    alpha : float
        Concentration parameter.

    Returns
    -------
    indicator_array : ndarray
        1dim array which indicates which data point belongs to which cluster.

    """
    counts = []
    indicator_array = np.empty(N, int)
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
        indicator = multinomialDist.rvs(1)
        indicator = np.where(indicator[0] == 1)
        indicator = int(indicator[0])
        indicator_array[n] = indicator
        
        # Update the counts for next time,
        # adding a new count if a new group was created
        if indicator == len(counts):
            counts.append(0)
          
        counts[indicator] += 1
        n += 1
    
    return indicator_array

def rp_mfm(N, alpha, beta):
    """
    Restaurant type process for Mixture of finite Mixtures.

    Parameters
    ----------
    N : int
        Number of indicators to be generated.
    alpha : float
        Concentration like parameter for the poisson prior on the components.
    beta : float
        Parameter for the Dirichlet prior on the weights.

    Returns
    -------
    indicator_array : ndarray
        1dim array which indicates which data point belongs to which cluster.

    """
    counts = []
    indicator_array = np.zeros(N, int)
    n = 0
    
    while n < N:
        # Compute the (unnormalized) probabilities of assigning the new object
        # to each of the existing groups, as well as a new group
        assign_probs = [None] * (len(counts) + 1)
        
        for i in range(len(counts)):
            assign_probs[i] = counts[i] + beta
        
        t = len(counts)
        assign_probs[-1] =\
            _V_nt(n+1, t+1, beta, alpha)/_V_nt(n+1, t, beta, alpha) * beta
        
        # Draw the new object's assignment from the discrete distribution
        assign_probs = assign_probs / sum(assign_probs)
        multinomialDist = multinomial(1, assign_probs)
        indicator = multinomialDist.rvs(1)
        indicator = np.where(indicator[0] == 1)
        indicator = int(indicator[0])
        indicator_array[n] = indicator
        
        # Update the counts for next time, adding a new count if a new group 
        # was created
        if indicator == len(counts):
            counts.append(0)
          
        counts[indicator] += 1
        n += 1
    
    return indicator_array

def _V_nt(n, t, beta, alpha):
    """
    Coefficient V_n(t) of the exchangable distribution function of a MFM model.
    The code for this function is based on the miller paper.

    Parameters
    ----------
    n : int
        Partition [n]:={1,...,n}.
    t : int
        Number of parts/blocks in the partition.
    beta : float
        Parameter for the Dirichlet prior on the weights.
    alpha : float
        Concentration like parameter for the poisson prior on the components.

    Returns
    -------
    v_nt : float
        Log of the Coefficient V_n(t).

    """
    tolerance = 1e-12
    p_k = poisson(alpha)
    a,c,k,p = 0, -np.inf, 1, 0
    # Note: The first condition is false when a = c = -Inf
    while np.abs(a-c) > tolerance or p < 1.0-tolerance:  
        if k >= t:
            a = c
            b = loggamma(k+1) - loggamma(k-t+1)\
                - loggamma(k*beta+n) + loggamma(k*beta) + np.log(p_k.pmf(k-1))
            c = logsumexp(a,b)
        p += np.exp(np.log(p_k.pmf(k-1)))
        k = k+1
    log_v = c
  
    return log_v

def logsumexp(a,b):
    m = np.maximum(a,b)
    if m == -np.inf:
        return -np.inf
    else:
        return m + np.log(np.exp(a-m)+np.exp(b-m))

if __name__ == "__main__":
    # random seed for testing purposes
    np.random.seed(3274)
    # example for chinese restaurant process
    N = 200
    alpha = 10
    sample_indices = np.arange(1, N+1)
    
    # create cluster indicators according to chinese restaurant process
    indicator_array = rp_dpm(N, alpha)+1
    
    # indices of new restaurant tables
    _, indices = np.unique(indicator_array, return_index=True)

    # plot indicators and average number of tables
    plt.figure(figsize=(3.2,3))
    plt.scatter(sample_indices, indicator_array,\
                marker='.', color='k', sizes = 5*np.ones(N))
    plt.scatter(indices+1, indicator_array[indices],\
                marker='.', color='r', sizes = 12*np.ones(N))
    plt.plot(sample_indices, alpha*np.log(1+sample_indices/alpha))
    plt.xlabel(r'$n$')
    plt.ylabel(r'$z_n$')
    plt.tight_layout()