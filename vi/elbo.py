import numpy as np
from scipy.special import gamma as gamma_func
from . import expectations as expec


# ELBO calculation
def compute_elbo(alpha, lamda, data, gamma, phi, tau, sigma, mu_G, sigma_G, sigma_inv):
    # number of clusters
    T = phi.shape[1]
    # data dimension
    K = data.shape[1]
    # number of data points
    N = data.shape[0]
    
    # T-1 gamma
    gamma_temp = gamma[:-1, :]
    
    # compute necessary expectations
    expec_eta, expec_logpart_eta    = expec.eta_and_logparteta(tau, sigma, sigma_inv)
    expec_log_1minusV               = expec.log_1minusV(gamma_temp)
    expec_log_V                     = expec.log_V(gamma_temp)
    
    # Term A 
    # NOTE: this term is zero for alpha = 1
    A = (alpha - 1) *  np.sum(expec_log_1minusV)\
        - (T-1) * (np.log(gamma_func(alpha)) - np.log(gamma_func(1+alpha)))
        
    # Term B 
    # with closed exp. family form  for the conjugate prior, i.e., without integration
    h = 1 / ((2*np.pi/lamda[-1])**(K/2)*np.linalg.det(sigma_inv)**0.5)
    a = np.dot(lamda[:-1], np.matmul(sigma_inv, lamda[:-1])) / (2*lamda[-1])
    B = T*(np.log(h) - a) + np.dot(lamda[:-1], np.sum(expec_eta, axis=0)) - lamda[-1]*np.sum(expec_logpart_eta)   

    
    # Term C
    C_1 = np.matmul(phi, expec.log_V(gamma))
    # temp = np.zeros((N, T))
    # for i in range(1,T): 
    #     indices_to_sum = np.array(range(T)) >= i
    #     #TODO: use np.cumsum instead if possible
    #     temp[:,i-1] = np.sum(phi, axis = 1, where = indices_to_sum)
    # C_21 = np.matmul(temp, expec.log_1minusV(gamma))
    
    #Without Loop
    phi_temp = np.flip(np.cumsum(np.flip(phi, axis=1), axis=1), axis=1)
    phi_temp = phi_temp[:,1:]
    phi_temp = np.column_stack((phi_temp,np.zeros(N)))
    C_2 = np.matmul(phi_temp, expec.log_1minusV(gamma))
    C = np.sum(C_1 + C_2)
    
    # Term D
    const = N/2*np.log(2*np.pi) + 0.5*np.log(np.linalg.det(sigma))
    D_1 = -0.5*np.sum(data*np.dot(sigma_inv, data.T).T, axis=1) - np.repeat(const, N, axis=0) 
    D_1 = np.repeat(D_1[:,np.newaxis], T, axis=1)
    D_2 = np.dot(expec_eta, data.T).T
    D_3 = np.repeat(expec_logpart_eta.T[np.newaxis,:], N, axis=0)
    D = np.sum(np.multiply(phi, D_1 + D_2 + D_3))
    
    # Term E
    E_1 = (gamma_temp[:,0] - np.ones(T-1)) * expec_log_V
    E_2 = (gamma_temp[:,1] - np.ones(T-1)) * expec_log_1minusV
    E_3 = np.log(gamma_func(gamma_temp[:,0])) + np.log(gamma_func(gamma_temp[:,1])) \
          - np.log(gamma_func(gamma_temp[:,0] + gamma_temp[:,1]))
    E = np.sum(E_1 + E_2 - E_3)
    
    # Term F
    h = 1 / ((2*np.pi/tau[:,-1])**(K/2)*np.linalg.det(sigma_inv)**0.5)
    a = np.sum(tau[:,:-1]*np.dot(sigma_inv, tau[:,:-1].T).T, axis=1) / (2*tau[:,-1])
    F = np.sum(np.log(h) + np.sum(tau[:,:-1]*expec_eta, axis=1) - tau[:,-1]*expec_logpart_eta - a)
    
    # Term G
    G = np.sum(phi*np.log(phi))
    
    # ELBO
    elbo = A + B + C + D - (E + F + G)
    
    return elbo
    
    
