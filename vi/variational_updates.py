import numpy as np
from scipy.special import digamma
from . import expectations as expec

# Dimensions
# K -> dimension of data points x_n
# gamma -> T x 2, [gamma_t1, gamma_t2]
# tau   -> T x (K+1), [tau_t11 ... tau_t1K, tau_t2]
# lamda -> (K+1) x 1, [lamda_t1, lamda_t2]  

def update_gamma(phi, alpha):
    T = phi.shape[1]
    N = phi.shape[0]
    gamma = np.empty((T,2))
    # gamma1 = np.zeros((T,2))
    gamma[:,0] = np.ones(T) + np.sum(phi, axis = 0)
    
    # temp = np.zeros((N, T-1))
    # for i in range(1,T):
    #     indices_to_sum = np.array(range(T)) >= i
    #     #TODO: use np.cumsum instead if possible
    #     temp[:,i-1] = np.sum(phi, axis = 1, where = indices_to_sum)
    # gamma1[:-1,1] = alpha*np.ones(T-1) + np.sum(temp, axis=0)
    # gamma1[-1,:] = np.array([1, 0.001])
    # Without loop
    phi_temp = np.flip(np.cumsum(np.flip(phi, axis=1), axis=1), axis=1)
    phi_temp = phi_temp[:,1:]
    gamma[:-1,1] = alpha*np.ones(T-1) + np.sum(phi_temp, axis=0)
    gamma[-1,:] = np.array([1, 0.001])
    return gamma

def update_tau(data, lamda, phi):
    T = phi.shape[1]
    K = data.shape[1]
    tau = np.empty((T,K+1))
    # tau_1 = np.empty((T,K+1))
    # for t in range(T):
    #     phi_t = phi[:,t]
    #     weighted_data = np.multiply(x,phi_t[:,np.newaxis])
    #     tau_1[t,:-1] = lamda[:-1]+np.sum(weighted_data,axis=0)
    #     tau_1[t,-1]=lamda[-1]+np.sum(phi_t)
    # version without loop
    phi_temp = np.repeat(phi, 2, axis=1)
    data_temp = np.tile(data,T)
    weighted_data = phi_temp*data_temp
    lamda_temp = np.tile(lamda[:-1],T)
    tau[:,:-1] = np.reshape(lamda_temp + np.sum(weighted_data, axis=0),(-1,2))
    tau[:,-1] = lamda[-1] + np.sum(phi, axis=0)
    return tau

def update_phi(data, gamma, tau, lamda, sigma, sigma_inv):
    N = data.shape[0]
    T = gamma.shape[0]
    phi = np.empty((N,T))
    A = expec.log_V(gamma)
    A_extended = np.repeat(A.T[np.newaxis,:], N, axis=0)
    B_temp = expec.log_1minusV(gamma)
    # the first element in B has to be zero due to stickbreaking
    B = np.zeros(T)
    B[1:] = np.cumsum(B_temp)[:-1]
    B_extended = np.repeat(B.T[np.newaxis,:], N, axis=0)
    C_temp, D = expec.eta_and_logparteta(tau, sigma, sigma_inv)
    C = np.dot(C_temp, data.T).T
    D_extended = np.repeat(D.T[np.newaxis,:], N, axis=0)
    S = A_extended + B_extended + C - D_extended
    phi = np.exp(S)
    normalizer = np.sum(phi, axis=1)
    phi = np.divide(phi, normalizer[:,np.newaxis])
    return phi
