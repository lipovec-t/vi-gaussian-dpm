from scipy.stats import multinomial
import numpy as np

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
      
test = crp(50, 10)
# test = multinomial(1, [0.3, 0.5, 0.2])
# x = test.rvs(100000)
# counts = sum(x, 2) / 100000
