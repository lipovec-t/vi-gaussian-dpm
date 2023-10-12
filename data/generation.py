# Third party imports
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal, multinomial

# Local application imports
if __name__ == "__main__":
    from restaurant_process import rp_dpm
else:
    from .restaurant_process import rp_dpm, rp_mfm

def generate_data(params):
    """
    Generates data according to the given parameters.

    Parameters
    ----------
    params : dataclass
        Includes all the model parameters used to generate the data.

    Returns
    -------
    indicator_array : ndarray
        Tx1 array which indicates which data points belongs to which cluster.
    cluster_assignements : ndarray
        TxN array of hard cluster assignments.
    cluster_means : ndarray
        Cluster means of the Gaussian mixture.
    x : ndarray
        NxK array of the generated data.
    y : ndarray
        NxK array of the generated data including noise.

    """
    # load config
    data_type   = params.data_type
    N           = params.N
    mu_G        = params.mu_G
    sigma_G     = params.sigma_G
    mu_U        = params.mu_U
    sigma_U     = params.sigma_U
    mu_V        = params.mu_V
    sigma_V     = params.sigma_V
    plot_data   = params.plot_data
    
    # generate data according to config
    if data_type.lower() == "dpm":
        # concentration parameter - higher alpha more clusters
        alpha_DPM = params.alpha_DPM
        indicator_array, cluster_assignments, cluster_means, x, y = \
            generate_data_rp(N, mu_G, sigma_G, mu_U, sigma_U, mu_V, sigma_V,\
                             plot_data, rp_dpm, alpha_DPM)
                
    elif data_type.lower() == "mfm":
        # kind of concentration parameter - higher alpha more clusters
        alpha_MFM = params.alpha_MFM 
        beta_MFM  = params.beta_MFM
        indicator_array, cluster_assignments, cluster_means, x, y = \
            generate_data_rp(N, mu_G, sigma_G, mu_U, sigma_U, mu_V, sigma_V,
                             plot_data, rp_mfm, alpha_MFM, beta_MFM)
                
    elif data_type.lower() == "gm":
        num_clusters = params.num_clusters_GM
        weights = params.weights_GM
        cluster_means = params.cluster_means_GM
        indicator_array, cluster_assignments, cluster_means, x, y = \
            generate_data_gm(N, num_clusters, weights, cluster_means,\
                             mu_G, sigma_G, mu_U, sigma_U, mu_V, sigma_V,\
                             plot_data)
                
    elif data_type.lower() == "load":
        filename = params.filename
        x = np.load(filename)
        params.N = x.shape[0]
        params.K = x.shape[1]
        # there is no ground truth w.r.t to the following vars in this case
        y = x
        indicator_array, cluster_assignments, cluster_means = [], [], []
        
    data = {
        "True Cluster Indicators"       : indicator_array,
        "True Cluster Assignments"      : cluster_assignments,
        "True Cluster Means"            : cluster_means,
        "True Number of Clusters"       : cluster_means.shape[0],
        "Datapoints"                    : x,
        "Noisy Datapoints"              : y
        }
    return data 

def generate_data_rp(N, mu_G, sigma_G, mu_U, sigma_U, mu_V, sigma_V, plot,\
                     rp, *args):
    """
    Generates data
    
    Parameters
    ----------
    N : int
        Number of datapoints to be generated.
    mu_G : ndarray
        Mean of the base distribution used to generate cluster means.
    sigma_G : ndarray
        Covariance of the base distribution used to generate the cluster means.
    mu_U : ndarray
        Used to determine the cluster means.
    sigma_U : ndarray
        Used to determine the cluster variances.
    mu_V : ndarray
        Mean of the AWGN.
    sigma_V : ndarray
        Covariance of AWGN.
    plot : boolean
        Determines if data shall be plotted.
    rp : function
        Restaurant process that is used to generate the indicator array.
    *args : tuple
        Tuple holding the parameters for the restaurant process.

    Returns
    -------
    indicator_array : ndarray
        Tx1 array which indicates which data points belongs to which cluster.
    cluster_assignements : ndarray
        TxN array of hard cluster assignments.
    cluster_means : ndarray
        Cluster means of the Gaussian mixture.
    x : ndarray
        NxK array of the generated data.
    y : ndarray
        NxK array of the generated data including noise.

    """
    # define base distribution
    G0 = multivariate_normal(mean = mu_G, cov = sigma_G)
    
    # Restaurant process
    indicator_array = rp(N, *args)
    num_clusters = max(indicator_array)+1
    
    # Draw cluster means from base distribution
    cluster_means = G0.rvs(num_clusters)
    
    # Compute cluster assignments
    cluster_assignements = np.zeros((N,num_clusters))
    # TODO: vectorize this
    for i in range(N):
        cluster_assignements [i,indicator_array[i]] = 1
    
    # Add dim if there is only one cluster such that cluster_means -> (1,K)
    if num_clusters == 1:
        cluster_means = np.repeat(cluster_means[np.newaxis,:], 1, axis = 0)
    
    # Mixture
    K = len(mu_G) # dimension of one data point
    x = np.empty((N,K)) # data matrix
    for i in range(N):
        mean = cluster_means[indicator_array[i]] + mu_U
        x[i,:] = multivariate_normal.rvs(mean = mean, cov = sigma_U, size = 1)
        
    # Measurement noise
    v = multivariate_normal.rvs(mean = mu_V, cov = sigma_V, size = N)
    y = x + v
    
    if plot and K == 2:           
        plt.figure()
        colormap = plt.cm.get_cmap('tab20', 20)
        plt.title("Data")
        cx, cy = cluster_means[:,0], cluster_means[:,1]
        plt.scatter(cx, cy, c = colormap(range(num_clusters)), marker = "o")
        dx, dy = y[:,0], y[:,1]
        plt.scatter(dx, dy, c = colormap(indicator_array), marker = '.')
    
    return indicator_array, cluster_assignements, cluster_means, x, y

def generate_data_gm(N, num_clusters, weights, cluster_means,\
                     mu_G, sigma_G, mu_U, sigma_U, mu_V, sigma_V, plot):
    """
    Generate data according to a Gaussian mixture with the given parameters.

    Parameters
    ----------
    N : int
        Number of datapoints to be generated.
    num_clusters : ndarray
        Number of clusters to be generated.
    weights : ndarray
        Cluster weights.
    mu_G : ndarray
        Mean of the base distribution used to generate cluster means.
    sigma_G : ndarray
        Covariance of the base distribution used to generate the cluster means.
    mu_U : ndarray
        Used to determine the cluster means.
    sigma_U : ndarray
        Used to determine the cluster variances.
    mu_V : ndarray
        Mean of the AWGN.
    sigma_V : ndarray
        Covariance of AWGN.
    plot : boolean
        Determines if data shall be plotted.
    **kwargs : dict
        If this dict contains cluster means then they are used to generate the
        data. Else the mu_G and sigma_G parameters are used to generate cluster
        means.

    Returns
    -------
    indicator_array : ndarray
        Tx1 array which indicates which data points belongs to which cluster.
    cluster_assignements : ndarray
        TxN array of hard cluster assignments.
    cluster_means : ndarray
        Cluster means of the Gaussian mixture.
    x : ndarray
        NxK array of the generated data.
    y : ndarray
        NxK array of the generated data including noise.

    """
    indicator_array = np.zeros(N, int)
    
    # generate cluster assignments and indicator array
    multinomialDist = multinomial(1, weights)
    cluster_assignements = multinomialDist.rvs(N)
    indicator_array  = np.where(cluster_assignements == 1)[1]
    

    # Draw cluster means 
    if not np.size(cluster_means):
        G0 = multivariate_normal(mean = mu_G, cov = sigma_G)
        cluster_means = G0.rvs(num_clusters)

    # Draw datapoints
    K = len(mu_G)
    x = np.empty((N,K))
    for i in range(N):
        mean = cluster_means[indicator_array[i]] + mu_U
        x[i,:] = multivariate_normal.rvs(mean = mean, cov = sigma_U, size = 1)
            
    # Measurement noise
    v = multivariate_normal.rvs(mean = mu_V, cov = sigma_V, size = N)
    y = x + v

    if plot:       
        plt.figure()
        colormap = plt.cm.get_cmap('tab20', 20)
        plt.title("Data")
        cx, cy = cluster_means[:,0], cluster_means[:,1]
        plt.scatter(cx, cy, c = colormap(range(num_clusters)), marker = "o")
        dx, dy = x[:,0], x[:,1]
        plt.scatter(dx, dy, c = colormap(indicator_array), marker = ".")
        
    return indicator_array, cluster_assignements, cluster_means, x, y


if __name__ == "__main__":
    # random seed for testing purposes
    np.random.seed(255)
    
    # set parameters
    K = 2
    N = 25
    # concentration parameter
    alpha = 2
    # parameter noise
    mu_U    = np.zeros(K)
    sigma_U = 1*np.eye(K)
    # base distribution
    mu_G    = np.zeros(K)
    sigma_G = 5*np.eye(K)
    # measurement noise
    mu_V    = np.zeros(K)
    sigma_V = np.eye(K)
    
    # generate data
    _, _, _, x, y = generate_data_rp(N, mu_G, sigma_G, mu_U, sigma_U, mu_V,\
                                     sigma_V, False, rp_dpm, alpha)
    
    # NOTE: need latex in system path
    #       os.environ["PATH"] += os.pathsep + '/Library/TeX/texbin'
    fig = plt.figure(figsize=(4.5,3.5))
    ax = fig.add_subplot(111)
    x1, x2 = x[:,0], x[:,1]
    y1, y2 = y[:,0], y[:,1]
    # connect x and y
    X_coords= np.array([x1, y1])
    Y_coords=np.array([x2, y2])
    # plt.plot(X_coords, Y_coords, color='0.6')
    # illustration of arrows is bigger in pdf plot
    ax.quiver(x1, x2, (y1-x1), (y2-x2), angles='xy', scale_units='xy', scale=1, width=1.5, units='dots')
    # plot DPM data without noise
    ax.scatter(x1, x2, marker = 'o', color='None', edgecolors='k', label=r'$\bm{x}_1,\ldots,\bm{x}_N$')
    # plot DPM data with noise
    ax.scatter(y1, y2, marker = 'o', color='0.4', edgecolors='k', label=r'$\bm{y}_1,\ldots,\bm{y}_N$')
    ax.set_xlabel(r'$x_{n,1}, y_{n,1}$')
    ax.set_ylabel(r'$x_{n,2}, y_{n,2}$')
    ax.legend()
    plt.tight_layout()
