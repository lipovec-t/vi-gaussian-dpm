# Coordinate Ascent Variational Inference for Dirichlet Process Mixtures of Gaussians

## Overview
This Github repository contains the implementation of Coordinate Ascent Variational Inference (CAVI) for the Gaussian estimation problem described in <a id="1">[1]</a> and <a id="1">[2]</a>, which is briefly summarized below. The implementation was used to generate the results presented in <a id="1">[1]</a> and is based on <a id="1">[3]</a>.

## Features
* Coordinate Ascent Variational Inference Algorithm: The implementation focuses on the CAVI algorithm, a variational inference method known for its efficiency in approximating posterior distributions.
* Dirichlet Process Mixtures of Gaussians: The code supports the modeling of complex data structures through the use of Dirichlet Process Mixtures, allowing for automatic determination of the number of clusters in the observations. The mixture distribution is assumed to be Gaussian.
* Scalable and Extendable: The code is designed to handle large datasets efficiently. Customization and experimentation with different priors, likelihoods, and hyperparameters is possible through the modification of the corresponding equations in the vi module.

## Model Summary of the Estimation Problem
For details see <a id="1">[1]</a> and <a id="1">[2]</a>.

### Object features
We consider a Gaussian model for objects that are indexed by $n=1,\ldots,N$, where $N$ is the total number of objects.
Each object is described by a random feature vector $x_n \in \mathbb{R}^M$, which depends on a random local parameter vector $\theta_n \in \mathbb{R}^M$ through the equation
$$x_n = \theta_n + u_n, \quad n=1,\ldots,N.$$

Using the assumption $u_n \sim \mathcal{N}\\!\left(u_n;0, \Sigma_u\right)$, it follows that

$$
f(x_n|\theta_n) = \mathcal{N}\\!\left(x_n|\theta_n, \Sigma_u\right),
$$

which is the conditional pdf of $x_n$ given $\theta_n$.

### Dirichlet Process
The local parameters $\theta_n$ are assumed to be distributed according to a discrete pdf $G$ that is realized from a Dirichlet process (DP):

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

The so called global parameters $\theta_k^\ast$ are i.i.d. according to the base distribution $G_0$ and the weights $\pi_k$ are obtained from a stick-breaking process.

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
The above model assumptions yield a Dirichlet process mixture (DPM) distribution for the object features $x_n$ and measurements $y_n$ in the form of

$$
f(x_n|G) = \sum_{k=1}^\infty \pi_k \\; \mathcal{N}\\!\left(x_n;\theta_k^\ast, \Sigma_u\right)
$$

and 

$$
f(y_n|G) = \sum_{k=1}^\infty \pi_k \\; \mathcal{N}\\!\left(x_n;\theta_k^\ast, \Sigma_u + \Sigma_v\right).
$$

The mixture weights $\pi_k$ and the component means $\theta_k^\ast$ follow from the DP.
The base distribution $G_0$ of the DP, and the mixands of the mixture, form a conjugate model and can be formulated in the exponential family framework.
Assignments of objects to mixture components are denoted as $z_n$ and can be modeled i.i.d. with a Categorical distribution (Multinomial with a single draw) where the probability of each object category (class) is given by the mixing weights $\pi_k$.

Goal is to estimate the assignments $z_n$, the global parameters $\theta_k^\ast$ and the weights $\pi_k$ either from directly observed object features $x_n$, or from noisy measurements $y_n$.
If noise is assumed, it is also possible to subsequently estimate the features $x_n$ from the measurements $y_n$ given estimates for the assignments $z_n$ and global parameters $\theta_k^\ast$.
The whole estimation problem is explained in detail in Chapter 5 of <a id="1">[1]</a>.

## CAVI and Approximate Inference
The implemented CAVI algorithm uses a mean field variational family to approximate the true posterior of the DPM.
This approximate posterior involves a truncated stick-breaking representation of the above model (truncated at level $T$) and is referred to as the variational pdf. It is parameterized by the variational parameters $\gamma_t$, $\tau_t$ and $\phi_n$:

$$
q(v,\eta^*,z) = \prod_{t=1}^{T-1} q_{t}(v_t;\gamma_t) \prod_{t=1}^{T} q_{t}(\eta_t^\ast;\tau_t) \prod_{n=1}^{N} q_{n}(z_n;\phi_n),
$$

where

$$
q_{t}(v_t;\gamma_t) = \mathcal{B}(v_t;\gamma_{t,1},\gamma_{t,2})
$$

is a Beta distribution,

$$
q_{t}(\eta_t^\ast;\tau_t) \propto \exp \big(\tau_{t,1}^{\mathrm{T}}  \eta_t^\ast - \tau_{t,2} a(\eta_t^\ast)\big)
$$

is a Gaussian distribution in exponential family form and

$$
q_{n}(z_n;\phi_n) = \mathcal{C}(z_n;\phi_n)
$$

is a Categorical distribution.
Here, $v_t$ denotes auxiliary variables of the stick-breaking construction of the DP and $\eta_t^\ast$ is a linearly transformed version of $\theta_t^\ast$.

The variational parameters are given by

$$
\begin{align}
&\gamma_{t,1} = 1 + \sum_{n=1}^{N} \phi_{n,t},\\
&\gamma_{t,2} = \alpha + \sum_{n=1}^{N} \sum_{j=t+1}^{T} \phi_{n,j},\\
&\tau_{t,1} = \lambda_1 + \sum_{n=1}^{N} \phi_{n,t} y_n,\\
&\tau_{t,2} = \lambda_2  + \sum_{n=1}^{N} \phi_{n,t},\\
&\phi_{n,t} = \frac{\exp(S_{n,t})}{\sum\limits_{j=1}^T \exp(S_{n,j})},\\
\end{align}
$$

with 

$$
S_{n,t} = \frac{1}{\tau_{t,2}} \tau_{t,1}^{\mathrm{T}} (\Sigma_{u} + \Sigma_{v})^{-1} y_n - \frac{1}{2 \tau_{t,2}^2} \left(M \tau_{t,2} + \tau_{t,1}^{\mathrm{T}} (\Sigma_{u} + \Sigma_{v})^{-1} \tau_{t,1}\right) + \Psi(\gamma_{t,1}) - \Psi(\gamma_{t,1} + \gamma_{t,2}) + \sum_{j=1}^{t-1} \Psi(\gamma_{j,2}) - \Psi(\gamma_{j,1} + \gamma_{j,2}).
$$

The CAVI algorithm approximates the true posterior pdf by calculating the variational parameters of the variational pdf in an interative manner. Convergence is declared when the relative change of the evidence lower bound falls below a predefined threshold.

Following parameters have to be choosen for initialization:
- Concentration parameter $\alpha$ of the DP
- Mean $\mu_{\theta^\*}$ and variance $\Sigma_{\theta^\ast}$ of the base distribution $G_0$ of the DP
- Variance $\Sigma_u$ 
- Truncation parameter $T$
- Assignment probabilities $\phi_{nt}$

The measurements $y_n$ (or $x_n$ if no noise is assumed), $n=1,\ldots,N$, are provided by the data module of the project. They can be generated or given by a file.

Given the approximate posterior, the means $\theta_k^*$ of the mixands, cluster assignments $z_n$ and cluster weights $\pi_k$ are determined by using approximate MMSE/MAP estimators in a postprocessing step (see postprocessing part of the vi module). Note that the marginal distributions to do so are already obtained as output from the CAVI algorithm. The final cluster means and cluster assignments are used to calculate MMSE estimates of the object features $x_n$ from the noisy measurements $y_n$.

For details see Chapter 4 and Chapter 5 of <a id="1">[1]</a>.

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

## License
This project is licensed under the MIT License - see the LICENSE file for details.

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
