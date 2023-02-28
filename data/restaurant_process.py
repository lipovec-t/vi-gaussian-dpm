from math import prod
import numpy as np
from scipy.stats import multinomial, poisson

def rp_dpm(N, alpha):
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

def rp_mfm(N, beta):
    """
    Restaurant Process for MFM
    """
    counts = []
    assignmentArray = np.zeros(N, int)
    n = 0
    # TODO: make lamda a function parameter and rename it
    lamda = 5
    
    while n < N:
        # Compute the (unnormalized) probabilities of assigning the new object
        # to each of the existing groups, as well as a new group
        assign_probs = [None] * (len(counts) + 1)
        
        for i in range(len(counts)):
            assign_probs[i] = counts[i] + beta
        
        t = len(counts)
        assign_probs[-1] = _V_nt(n+1, t+1, beta, lamda)/_V_nt(n+1, t, beta, lamda) * beta
        
        # Draw the new object's assignment from the discrete distribution
        assign_probs = assign_probs / sum(assign_probs)
        multinomialDist = multinomial(1, assign_probs)
        assignment = multinomialDist.rvs(1)
        assignment = np.where(assignment[0] == 1)
        assignment = int(assignment[0])
        assignmentArray[n] = assignment
        
        # Update the counts for next time, adding a new count if a new group 
        # was created
        if assignment == len(counts):
            counts.append(0)
          
        counts[assignment] += 1
        n += 1
    
    return assignmentArray

def _V_nt(n, t, beta, lamda):
    # prior for K
    p_k = poisson(lamda)
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

