# Variational Inference with Gaussian Dirichlet Process Mixtures

## Model
$$x_n = \theta_n + u_n$$

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

## CAVI
Mean field approximation of the posterior with variational parameters $\gamma_t$, $\tau_t$ and $\phi_n$:

$$q(v,\eta^*,z) = \prod_{t=1}^{T-1}q_{\gamma_t}(v_t) \prod_{t=1}^{T}q_{\tau_t}(\eta_t^\*) \prod_{n=1}^{N}q_{\phi_n}(z_n)$$

Following parameters have to be choosen for initialization:
- Concentration Parameter $\alpha$
- Mean $\mu_G$ and variance $\Sigma_G$ of the base distribution
- Cluster variance $\Sigma_u$
- Truncation parameter $T$
- Assignment probabilities $\phi_{nt}$

We use synthetic data $x$, that is produced using the data package of the project, to learn the posterior distribution $q(v,\eta^\*,z)$ of the model explained above. Given the posterior one can then estimate cluster means $\theta_l^*$, cluster assignments $z_n$ and cluster weights $\pi_l$ using well known estimators like the MMSE/MAP estimator. Note that the marginal distributions to do so are already obtained as output from the algorithm.
