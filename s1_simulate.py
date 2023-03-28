# Standard library imports
import timeit

# Third party imports
import numpy as np

# Local application imports
from data.generation import generate_data
from vi.cavi import coordinates_ascent
from vi import postprocessing as pp
from s1_config import Params

# random seed for testing purposes
np.random.seed(255)

# load parameters
params = Params()

# generate data
data_dict = generate_data(params)
if params.include_noise == False:
    data = data_dict["Datapoints"]
else:
    data = data_dict["Noisy Datapoints"]

# start timer
t_0 = timeit.default_timer()

# CAVI
elbo_final, tau, gamma, phi = coordinates_ascent(data_dict, params)
# end timer

# end timer and compute elapsed time
t_1 = timeit.default_timer()
runtime = t_1 - t_0

# postprocessing
results, results_reduced = pp.full_postprocessing(data_dict, phi, gamma, tau, True)


