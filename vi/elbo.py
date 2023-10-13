import numpy as np
from scipy.special import loggamma
from . import expectations as expec
from . import postprocessing as pp


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
        - (T-1) * (loggamma(alpha) - loggamma(1+alpha))
        
    # Term B 
    # with closed exp. family form  for the conjugate prior, i.e., without integration
    h = 1 / ((2*np.pi/lamda[-1])**(K/2)*np.linalg.det(sigma_inv)**0.5)
    a = np.dot(lamda[:-1], np.matmul(sigma_inv, lamda[:-1])) / (2*lamda[-1])
    B = T*(np.log(h) - a) + np.dot(lamda[:-1], np.sum(expec_eta, axis=0)) - lamda[-1]*np.sum(expec_logpart_eta)   

    
    # Term C
    C_1 = np.matmul(phi, expec.log_V(gamma))
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
    E_3 = loggamma(gamma_temp[:,0]) + loggamma(gamma_temp[:,1]) \
          - loggamma(gamma_temp[:,0] + gamma_temp[:,1])
    E = np.sum(E_1 + E_2 - E_3)
    
    # Term F
    h = 1 / ((2*np.pi/tau[:,-1])**(K/2)*np.linalg.det(sigma_inv)**0.5)
    a = np.sum(tau[:,:-1]*np.dot(sigma_inv, tau[:,:-1].T).T, axis=1) / (2*tau[:,-1])
    F = np.sum(np.log(h) + np.sum(tau[:,:-1]*expec_eta, axis=1) - tau[:,-1]*expec_logpart_eta - a)
    
    # Term G
    # change 0's to smallest number
    phi[np.where(phi==0)] = np.nextafter(0, 1)
    G = np.sum(phi*np.log(phi))
    
    # ELBO
    elbo = A + B + C + D - (E + F + G)
    
    return elbo


def compute_predictive(data, gamma, tau, sigma):
    """
    Compute value of the  predictive distribution for some given data.

    Parameters
    ----------

    Returns
    -------

    """
    T = gamma.shape[0]
    
    # compute estimate of the cluster weights
    pi_est = pp.est_cluster_weights_mmse(gamma)
    
    # compute estimate of the cluster means
    means_est = pp.est_cluster_means_mmse(tau)
    
    # compute predictive pdf for all data points
    covs = np.repeat(sigma[np.newaxis, :, :], T, axis=0)
    # here temp has shape (N, T)
    temp = np.exp(multiple_logpdfs(data, means_est, covs))
    temp = temp * pi_est[:,np.newaxis]
    temp = np.sum(temp, axis=0)
    # compute average log predictive
    predictive = np.mean(np.log(temp))
         
    return predictive

def multiple_logpdfs(xs, means, covs):
    """ 
    Vecotrize computation of multivariate normal pdf.
    Source: https://gregorygundersen.com/blog/2020/12/12/group-multivariate-normal-pdf/
    Assuming:
        - T parameters
        - K is the dimension of a single sample
        - N is the number of samples
        - `xs` has shape (N, K).
        - `means` has shape (T, K).
        - `covs` has shape (T, K, K).
    """
    # NumPy broadcasts `eigh`.
    vals, vecs = np.linalg.eigh(covs)

    # Compute the log determinants across the second axis.
    logdets = np.sum(np.log(vals), axis=1)

    # Invert the eigenvalues.
    valsinvs = 1./vals
    
    # Add a dimension to `valsinvs` so that NumPy broadcasts appropriately.
    Us   = vecs * np.sqrt(valsinvs)[:, None]
    devs = xs[:, None, :] - means[None, :, :]

    # Use `einsum` for matrix-vector multiplications across the first dimension.
    devUs = np.einsum('jnk,nki->jni', devs, Us)

    # Compute the Mahalanobis distance by squaring each term and summing.
    mahas = np.sum(np.square(devUs), axis=2)
    
    # Compute and broadcast scalar normalizers.
    dim    = xs.shape[1]
    log2pi = np.log(2 * np.pi)

    out = -0.5 * (dim * log2pi + mahas + logdets[None, :])
    return out.T