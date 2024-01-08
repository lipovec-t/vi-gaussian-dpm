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
$$x_n = \theta_n + u_n, \quad n=1,\ldots,N$$

with $G_0 = \mathcal{N}\\!\left(\mu_G, \Sigma_G\right)$ and

$$
\begin{align}
u_n &\sim \mathcal{N}\\!\left(0, \Sigma_u\right)\\
G &\sim \text{DP}(G_0, \alpha)\\
\theta_n | G &\sim G\\
\end{align}
$$

This yields a mixture distribution in the form of

$$
f(x_n|G) = \sum_{l=1}^\infty \pi_l \\; \mathcal{N}\\!\left(x_n;\theta_l^*, \Sigma_u\right)
$$

where the mixture weights $\pi_l$ and the Gaussian cluster means $\theta_l^*$ are determined by the Dirichlet Process (DP).
The base distribution $G_0$ and mixands $f(x_n|\theta_l^\*) = \mathcal{N}\\!\left(x_n;\theta_l^\*, \Sigma_u\right)$ form a conjugate model and can be formulated in the exponential family framework as

$$
\begin{align}
f(x_n|\eta_l^\*) &= h(x_n) \cdot \exp\left(\eta_l^{*T} x_n - a(\eta_l^\*)\right)\\
f(\eta_l^\*;\lambda_1, \lambda_2) &= b \exp\left(\lambda_1^T \eta_l^\*- \lambda_2 a(\eta_l^\*)\right)
\end{align}
$$

with natural parameter $\eta_l^* = \Sigma_u^{-1} \theta_l^*$  and hyperparameters $\lambda_1$ and $\lambda_2$ satisfying following relation

$$
\begin{align}
\mu_{\eta_l^\*} &= \Sigma_u^{-1} \mu_G = \frac{1}{\lambda_2} \Sigma_u^{-1} \lambda_1\\
\Sigma_{\eta_l^\*} &= \Sigma_u^{-1} \Sigma_G \Sigma_u^{-1} = \frac{1}{\lambda_2} \Sigma_u^{-1},
\end{align}
$$

i.e., given $\mu_G$, $\Sigma_G$ and $\Sigma_u$ the hyperparameters are automatically determined. Note that the natural parameter $\eta_l^\* = \Sigma_u^{-1} \theta_l^\*$ is again Gaussian with $\mu_{\eta_l^\*}$ and $\Sigma_{\eta_l^\*}$ because it is in linear relation with $\theta_l^\*$.

Random cluster assignments are denoted as $z_n$ and are  i.i.d. distributed with a Categorical distribution (Multinomial with a single draw) where the probability of each category is given by the mixing weights $\pi_l$. The assignment variable $z_n$ indicates with which mixture component the data point $x_n$ is associated. Using the stick-breaking view the data can be described as arising from the following process:

$$
\begin{align}
  v_i &\sim \text{Beta}(1,\alpha)\\
  \eta_l^\* &\sim  G_0\\
  \pi_l &= v_i \prod_{j=1}^{l-1}(1-v_j)\\
  z_n | \pi &\sim \text{Mult}(1,\pi)\\
  x_n | z_n,\eta^\* &\sim f(x_n|\eta_{z_n}^\*)
\end{align}
$$

## CAVI for DPM models
Mean field approximation of the posterior with variational parameters $\gamma_t$, $\tau_t$ and $\phi_n$:

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
