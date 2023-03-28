# Third party imports
import numpy as np
from scipy.cluster.vq import kmeans2
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler


# Local application imports
from . import expectations as expec
from vi.elbo import compute_elbo

# Dimensions
# K -> dimension of data points x_n
# gamma -> T x 2, [gamma_t1, gamma_t2]
# tau   -> T x (K+1), [tau_t11 ... tau_t1K, tau_t2]
# lamda -> (K+1) x 1, [lamda_t1, lamda_t2]  

def coordinates_ascent(data_dict, params):
    # load params
    phi_init        = _init(data_dict, params)
    max_iterations  = params.max_iterations
    alpha           = params.alpha
    sigma           = params.sigma
    sigma_inv       = params.sigma_inv
    mu_G            = params.mu_G
    sigma_G         = params.sigma_G
    lamda           = params.lamda 
    
    # extract data from data_dict
    if params.include_noise:
        data = data_dict["Noisy Datapoints"]
    else:
        data = data_dict["Datapoints"]
    
    # multiple phi initializations are saved in the 3rd dim of phi_init
    num_permutations = phi_init.shape[2]
    elbo = np.zeros(max_iterations)
    elbo_final = -np.inf
    
    for j in range(num_permutations):
        phi_temp = phi_init[:,:,j]          
        gamma_temp = update_gamma(phi_temp,alpha)
        tau_temp = update_tau(data, lamda, phi_temp)
        
        for i in range(max_iterations):
            # compute variational updates
            phi_temp = update_phi(data, gamma_temp, tau_temp, \
                                  lamda, sigma, sigma_inv)
            gamma_temp = update_gamma(phi_temp, alpha)
            tau_temp = update_tau(data, lamda, phi_temp)
            
            # compute elbo and check convergence
            elbo[i] = compute_elbo(alpha, lamda, data, gamma_temp, phi_temp, \
                                   tau_temp, sigma, mu_G, sigma_G, sigma_inv)
            if i>0 and np.abs(elbo[i]-elbo[i-1])/np.abs(elbo[i-1]) * 100 < 1e-2:
                break
            
        if elbo[i] > elbo_final:
            elbo_final = elbo[i]
            tau = tau_temp
            gamma = gamma_temp
            phi = phi_temp
            
    return elbo_final, tau, gamma, phi

def _init(data_dict, params):
    # truncation parameter and number of data points
    T = params.T
    N = params.N
    
    # extract data from data_dict
    if params.include_noise:
        data = data_dict["Noisy Datapoints"]
    else:
        data = data_dict["Datapoints"]
        
    # initialization
    # NOTE: T has to be higher than the true number of clusters
    if params.init_type.lower() == 'uniform':
        phi_init = 1/T * np.ones((N,T,1))
    elif params.init_type.lower() == 'true':
        phi_init = np.zeros((N,T,1))
        true_assignment = data_dict["True Cluster Assignments"]
        T_true = true_assignment.shape[1]
        phi_init[:,:T_true,0] = true_assignment
    elif params.init_type.lower() == 'permute':
        num_perm = params.num_permutations
        rand_indicators = [np.random.randint(0,T,N) for i in range(num_perm)]
        phi_init = np.zeros((N,T,num_perm))
        for j in range(num_perm):
            for k in range(N):
               phi_init[k,rand_indicators[j][k],j] = 1
    elif params.init_type.lower() == 'unique':
        params.T = N
        phi_init = np.eye(N)
        phi_init = np.expand_dims(phi_init, axis=2)
    elif params.init_type.lower() == 'allinone':
        num_perm = T
        rand_indicators = [i*np.ones(T,int) for i in range(num_perm)]
        phi_init = np.zeros((N,T,num_perm))
        for j in range(num_perm):
            for k in range(N):
                phi_init[k,rand_indicators[j],j] = 1
    elif params.init_type.lower() == 'kmeans':
        # round mean of poisson prior to nearest int
        cluster_init = round(params.alpha * np.log((params.alpha+N)/params.alpha))
        # have at least one cluster
        if cluster_init == 0:
            cluster_init = 1
        _, label = kmeans2(data, cluster_init, minit='points')
        phi_init = np.zeros((N,T,1))
        for k in range(N):
           phi_init[k,label[k]] = 1
    elif params.init_type.lower() == 'dbscan':
        data_transformed = StandardScaler().fit_transform(data)
        db = DBSCAN(eps=0.3, min_samples=10).fit(data_transformed)
        label = db.labels_
        n_noise = list(label).count(-1)
        n_clusters = len(set(label)) - (1 if -1 in label else 0)
        if n_clusters > T:
            error_msg = "Number of clusters found by DBSCAN is higher than"\
                         +"the truncation parameter"
            raise ValueError(error_msg)
        # assign noisy labels to random cluster
        label[label==-1] = np.random.randint(0, max(label)+1, n_noise)
        phi_init = np.zeros((N,T,1))
        for k in range(N):
           phi_init[k,label[k]] = 1
    return phi_init

def update_gamma(phi, alpha):
    T = phi.shape[1]
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
    phi_temp = np.repeat(phi, K, axis=1)
    data_temp = np.tile(data,T)
    weighted_data = phi_temp*data_temp
    lamda_temp = np.tile(lamda[:-1],T)
    tau[:,:-1] = np.reshape(lamda_temp + np.sum(weighted_data, axis=0),(-1,K))
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
