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
    # computation of posterior expected natural parameter eta
    expec_eta = np.einsum('ij,kj->ki',sigma_inv,tau[:,:-1]) / tau[:,-1,np.newaxis]
    # computation of posterior expected log partition
    # temp = tau_1^T * sigma_inv * tau_1 vectorized over t
    temp = np.sum(tau[:,:-1] * np.dot(tau[:,:-1], sigma_inv), axis=1) 
    expec_logparteta = 1/(2*tau[:,-1]**2) * (temp + K*tau[:,-1])
    return expec_eta, expec_logparteta