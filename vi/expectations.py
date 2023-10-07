import numpy as np
from scipy.special import digamma as digamma_func

def log_1minusV(gamma):
    expec = digamma_func(gamma[:,1]) - digamma_func(gamma[:,0] + gamma[:, 1])
    return expec

def log_V(gamma):
    expec = digamma_func(gamma[:,0]) - digamma_func(gamma[:,0] + gamma[:, 1])
    return expec

def eta_and_logparteta(tau, sigma, sigma_inv):
    # As the two expectations are always used together we have one function for both
    K = tau.shape[1] - 1
    expec_eta = np.einsum('ij,kj->ki',sigma_inv,tau[:,:-1]) / tau[:,-1,np.newaxis]
    expec_logparteta = 1/(2*tau[:,-1]**2) * (np.sum(tau[:,:-1]*tau[:,:-1], axis=1) + K*tau[:,-1])
    return expec_eta, expec_logparteta

    