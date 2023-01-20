import numpy as np
from scipy.special import digamma as digamma_func

def log_1minusV(gamma):
    expec = digamma_func(gamma[:,1]) - digamma_func(gamma[:,0] + gamma[:, 1])
    return expec

def log_V(gamma):
    expec = digamma_func(gamma[:,0]) - digamma_func(gamma[:,0] + gamma[:, 1])
    return expec

def eta_and_logparteta(tau, sigma_U):
    # As the two expectations are always used together we have one function for both
    # NOTE: The formular for expec_2 is only valid if the off-diagonal 
    # elements of sigma_U are zero
    T = tau.shape[0]
    K = tau.shape[1] - 1
    expec_1 = np.empty((T,K))
    expec_2 = np.empty(T)
    for t in range(T):
        # TODO: something could be wrong here
        expec_1[t,:] = np.matmul(np.linalg.inv(sigma_U),tau[t,:-1]) / tau[t,-1]
        sigma_tau = np.linalg.inv(sigma_U) / tau[t,-1]
        temp = sigma_tau + np.diag(expec_1[t,:]**2)
        expec_2[t] = np.trace(np.multiply(temp, sigma_U)) * 0.5
    return expec_1, expec_2
