import numpy as np
from scipy.special import digamma as digamma_func

def log_1minusV(gamma):
    expec = digamma_func(gamma[:,1]) - digamma_func(gamma[:,0] + gamma[:, 1])
    return expec

def log_V(gamma):
    expec = digamma_func(gamma[:,0]) - digamma_func(gamma[:,0] + gamma[:, 1])
    return expec

def eta_and_logparteta(tau, sigma_U, inv_sigma_U):
    # As the two expectations are always used together we have one function for both
    # NOTE: The formular for expec_2 is only valid if the off-diagonal 
    # elements of sigma_U are zero
    T = tau.shape[0]
    K = tau.shape[1] - 1
    expec_eta1 = np.empty((T,K))
    expec_logparteta1 = np.empty(T)
    for t in range(T):
        expec_eta1[t,:] = np.matmul(inv_sigma_U,tau[t,:-1]) / tau[t,-1]
        sigma_tau = inv_sigma_U / tau[t,-1]
        temp = sigma_tau + np.diag(expec_eta1[t,:]**2)
        expec_logparteta1[t] = np.trace(np.multiply(temp, sigma_U)) * 0.5
    # without loop
    expec_eta = np.einsum('ij,kj->ki',inv_sigma_U,tau[:,:-1]) / tau[:,-1,np.newaxis]
    tau_temp = np.ones((K,K,T)) * tau[:,-1]
    sigma_tau = np.repeat(inv_sigma_U[:, :, np.newaxis], T, axis=2) / tau_temp
    temp = sigma_tau + np.moveaxis(np.multiply(np.eye(2),expec_eta[:,np.newaxis]**2),0,2)
    expec_logparteta = np.trace(np.multiply(temp,  np.repeat(sigma_U[:, :, np.newaxis], T, axis=2))) * 0.5
    return expec_eta1, expec_logparteta1
