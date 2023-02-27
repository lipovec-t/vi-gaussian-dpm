# Third party imports
import numpy as np

# Local application imports
from . import expectations as expec
from vi.elbo import compute_elbo

# Dimensions
# K -> dimension of data points x_n
# gamma -> T x 2, [gamma_t1, gamma_t2]
# tau   -> T x (K+1), [tau_t11 ... tau_t1K, tau_t2]
# lamda -> (K+1) x 1, [lamda_t1, lamda_t2]  

def coordinates_ascent(data, max_iterations, initialization, alpha, sigma, sigma_inv, mu_G, sigma_G, lamda, truncation):
    # TODO: maybe use a dic for the parameters alpha, sigma, mu_G, sigma_G, truncation
    T = truncation
    N = data.shape[0]
    # TODO: fix init version 2 and 3
    phi_init, num_permutations = _init(initialization, T, N)
    elbo = np.zeros(max_iterations)
    elbo_final = -np.inf
    for j in range(num_permutations):             
        gamma_temp = update_gamma(phi_init,alpha)
        tau_temp = update_tau(data, lamda, phi_init)
        
        for i in range(max_iterations):
            # TODO: save variational parameters and investigate convergence of parameters (instead of ELBO)
            # compute variational updates
            phi_temp = update_phi(data, gamma_temp, tau_temp, lamda, sigma, sigma_inv)
            gamma_temp = update_gamma(phi_temp, alpha)
            tau_temp = update_tau(data, lamda, phi_temp)
            
            # compute elbo and check convergence
            elbo[i] = compute_elbo(alpha, lamda, data, gamma_temp, phi_temp, tau_temp, sigma, mu_G, sigma_G, sigma_inv)
            if i>0 and np.abs(elbo[i]-elbo[i-1]) < 0.1:
                break
            
        if elbo[i] > elbo_final:
            elbo_final = elbo[i]
            tau = tau_temp
            gamma = gamma_temp
            phi = phi_temp
            
    return elbo_final, tau, gamma, phi

def _init(version, T, N, *kwargs):
    # initialization
    # NOTE: T has to be higher than the true number of clusters
    # TODO: SAVE PHI INIT AS 3D ARRAY with num_permutation as 3rd dim.
    phi_init_version = 1
    if phi_init_version == 1:
        phi_init = 1/T * np.ones((N,T))
        num_permutations = 1
    elif phi_init_version == 2:
        phi_init = np.zeros((N,T))
        phi_init[:,:T_true] = true_assignment
        num_permutations = 1
    elif phi_init_version == 3:
        np.random.seed(1337)
        num_permutations = 30
        rand_indicators = [np.random.randint(0,T,N) for i in range(num_permutations)]
        phi_init = np.zeros((N,T))
        # TODO: do this for all permutations j
        for k in range(N):
            phi_init[k,rand_indicators[j][k]] = 1
    elif phi_init_version == 4:
        T = N
        phi_init = np.eye(N)
        num_permutations = 1
    elif phi_init_version == 5:
        num_permutations = T
        rand_indicators = [i*np.ones(T) for i in range(num_permutations)]
        phi_init = np.zeros((N,T))
        # TODO: do this for all permutations j
        for k in range(N):
            phi_init[k,rand_indicators[j][k]] = 1
    return phi_init, num_permutations

def update_gamma(phi, alpha):
    T = phi.shape[1]
    N = phi.shape[0]
    gamma = np.empty((T,2))
    gamma[:,0] = np.ones(T) + np.sum(phi, axis = 0)
    phi_temp = np.flip(np.cumsum(np.flip(phi, axis=1), axis=1), axis=1)
    phi_temp = phi_temp[:,1:]
    gamma[:-1,1] = alpha*np.ones(T-1) + np.sum(phi_temp, axis=0)
    gamma[-1,:] = np.array([1, 0.001])
    return gamma

def update_tau(data, lamda, phi):
    T = phi.shape[1]
    K = data.shape[1]
    tau = np.empty((T,K+1))
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
