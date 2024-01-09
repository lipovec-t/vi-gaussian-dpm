# Coordinate Ascent Variational Inference for Dirichlet Process Mixtures of Gaussians

## Overview
This Github repository contains the implementation of Coordinate Ascent Variational Inference (CAVI) for the Gaussian estimation problem described in <a id="1">[1]</a> and <a id="1">[2]</a>, which is summarized below. The implementation was used to generate the results presented in <a id="1">[1]</a> and is based on <a id="1">[3]</a>.

## Features
* Coordinate Ascent Variational Inference: The implementation focuses on the CAVI algorithm, a variational inference method known for its efficiency in approximating posterior distributions.
* Dirichlet Process Mixtures of Gaussians: The code supports the modeling of complex data structures through the use of Dirichlet Process Mixtures, allowing for automatic determination of the number of clusters. The mixture distribution is assumed to be Gaussian.
* Scalable and Extendable: The code is designed to handle large datasets efficiently. Customization and experimentation with different priors, likelihoods, and hyperparameters is possible through the modification of the corresponding equations in the vi module.

## Model Summary for the Estimation Problem
For details see <a id="1">[1]</a> and <a id="1">[2]</a>.

### Object features
We consider a Gaussian model for objects that are indexed by $n=1,\ldots,N$, where $N$ is the total number of objects.
Each object is described by a random feature vector $x_n \in \mathbb{R}^M$, which depends on a random local parameter vector $\theta_n \in \mathbb{R}^M$ through the equation
$$x_n = \theta_n + u_n, \quad n=1,\ldots,N.$$

Using the assumption $u_n \sim \mathcal{N}\\!\left(u_n;0, \Sigma_u\right)$ if follows that

$$
f(x_n|\theta_n) = \mathcal{N}\\!\left(x_n|\theta_n, \Sigma_u\right).
$$

### Dirichlet Process
The local parameters $\theta_n$ are assumed to be distributed according to a Dirichlet process (DP):

$$
\begin{align}
&G \sim \text{DP}(G_0, \alpha),\\
&\theta_n | G \sim G(\theta_n|\pi,\theta^\ast_{1:\infty}),
\end{align}
$$

with base distribution $G_0(\theta_k^\ast) = \mathcal{N}\\!\left(\mu_{\theta^\*}, \Sigma_{\theta^\ast}\right)$.
The realization $G$ of the DP is given by

$$
G(\theta_n|\pi,\theta^\ast_{1:\infty}) = \sum_{k=1}^{\infty} \pi_k \delta(\theta_n-\theta_k^\ast). 
$$

### Measurements
The $n$-th measurement (observation) $y_n \in \mathbb{R}^M$ is an altered version of the object feature $x_n$ corrupted by additive noise:
$$y_n = x_n + v_n, \quad n=1,\ldots,N,$$
where $v_n \sim \mathcal{N}\\!\left(v_n;0, \Sigma_v\right)$ and thus

$$
f(y_n|x_n) = \mathcal{N}\\!\left(y_n|x_n, \Sigma_v\right).
$$

Moreover, $y_n = \theta_n + u_n + v_n$, which entails that

$$
f(x_n|\theta_n) = \mathcal{N}\\!\left(y_n|\theta_n, \Sigma_u + \Sigma_v\right).
$$

### Dirichlet Process Mixture
The above model assumptions yield a mixture distribution for the object features and measurements in the form of

$$
f(x_n|G) = \sum_{k=1}^\infty \pi_k \\; \mathcal{N}\\!\left(x_n;\theta_k^\ast, \Sigma_u\right)
$$

and 

$$
f(y_n|G) = \sum_{k=1}^\infty \pi_k \\; \mathcal{N}\\!\left(x_n;\theta_k^\ast, \Sigma_u + \Sigma_v\right).
$$

Here, the mixture weights $\pi_k$ and the component means $\theta_k^\ast$ are determined by the DP.
The base distribution $G_0$ of the DP, and the mixands of the mixture, form a conjugate model and can be formulated in the exponential family framework.

Assignments of objects to mixture components are denoted as $z_n$ and can be modeled i.i.d. with a Categorical distribution (Multinomial with a single draw) where the probability of each object category (class) is given by the mixing weights $\pi_k$.

## CAVI for DPM models
Truncated mean field approximation of the posterior with variational parameters $\gamma_t$, $\tau_t$ and $\phi_n$:

$$q(v,\eta^*,z) = \prod_{t=1}^{T-1}q_{\gamma_t}(v_t) \prod_{t=1}^{T}q_{\tau_t}(\eta_t^\*) \prod_{n=1}^{N}q_{\phi_n}(z_n)$$

Following parameters have to be choosen for initialization:
- Concentration Parameter $\alpha$
- Mean $\mu_G$ and variance $\Sigma_G$ of the base distribution
- Cluster variance $\Sigma_u$
- Truncation parameter $T$
- Assignment probabilities $\phi_{nt}$

We use synthetic data $x$, that is produced using the data package of the project, to learn the posterior distribution $q(v,\eta^\*,z)$ of the model explained above. Given the posterior one can then estimate cluster means $\theta_l^*$, cluster assignments $z_n$ and cluster weights $\pi_l$ using well known estimators like the MMSE/MAP estimator. Note that the marginal distributions to do so are already obtained as output from the algorithm.

## Usage
To run the code follow these steps:
1. Clone the repository
```
git clone https://github.com/lipovec-t/vi-gaussian-dpm.git
```
2. Install dependencies
```
pip install -r requirements.txt
```
3. Run one of the simulation scripts s*_simulate.py in the IDE of your choice.

Customize the simulation scripts to your specific use case and adapt config files as needed.


Feel free to use, modify, and extend this implementation for your research or applications. If you encounter any issues or have suggestions, please let us know through the GitHub issues page.

## References
<a id="1">[1]</a> 
T. Lipovec,
“Variational Inference for Dirichlet Process Mixtures and Application to Gaussian Estimation,”
Master’s thesis, TU Wien, 2023.

<a id="1">[2]</a> 
E. Šauša,
“Advanced Bayesian Estimation in Hierarchical Gaussian Models: Dirichlet Process Mixtures and Clustering Gain,”
Master’s thesis, TU Wien, 2024.

<a id="1">[3]</a> 
D. M. Blei and M. I. Jordan, 
“Variational Inference for Dirichlet Process Mixtures,” 
Bayesian Analysis, vol. 1, no. 1, pp. 121-143, 2006.

<a id="1">[4]</a>
D. M. Blei, A. Kucukelbir, and J. D. McAuliffe,
“Variational Inference: A Review for Statisticians,”
Journal of the American Statistical Association, vol. 112, no. 518, pp. 859-877, 2018.
