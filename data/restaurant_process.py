from math import prod
import numpy as np
from scipy.stats import multinomial, poisson

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
    v_nt : TYPE
        DESCRIPTION.

    """
    # prior for K
    p_k = poisson(alpha)
    # TODO: vectorize
    f1 = lambda x,m: prod(x+i for i in range(m)) # function for x^(m)
    f2 = lambda x,m: prod(x-i for i in range(m)) # function for x_(m)
    v_nt = 0
    k=1
    term = np.inf
    while k < 50:
        term = f2(k,t) / f1(beta*k,n) * p_k.pmf(k-1)
        v_nt += term
        k += 1
    return v_nt

