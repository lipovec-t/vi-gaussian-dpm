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
    # NOTE: The formular for expec_2 is only valid if the off-diagonal 
    # elements of sigma are zero
    T = tau.shape[0]
    K = tau.shape[1] - 1
    # expec_eta1 = np.empty((T,K))
    # expec_logparteta1 = np.empty(T)
    # for t in range(T):
    #     expec_eta1[t,:] = np.matmul(sigma_inv,tau[t,:-1]) / tau[t,-1]
    #     sigma_tau = sigma_inv / tau[t,-1]
    #     temp = sigma_tau + np.diag(expec_eta1[t,:]**2)
    #     expec_logparteta1[t] = np.trace(np.multiply(temp, sigma)) * 0.5
    # without loop
    expec_eta = np.einsum('ij,kj->ki',sigma_inv,tau[:,:-1]) / tau[:,-1,np.newaxis]
    tau_temp = np.ones((K,K,T)) * tau[:,-1]
    sigma_tau = np.repeat(sigma_inv[:, :, np.newaxis], T, axis=2) / tau_temp
    temp = sigma_tau + np.moveaxis(np.multiply(np.eye(K),expec_eta[:,np.newaxis]**2),0,2)
    expec_logparteta = np.trace(np.multiply(temp,  np.repeat(sigma[:, :, np.newaxis], T, axis=2))) * 0.5
    return expec_eta, expec_logparteta