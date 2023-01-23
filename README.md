# Variational Inference with Gaussian Dirichlet Process Mixtures

## Model
$$x_n = \theta_n + u_n$$

with

$$u_n \sim \mathcal{N}\!\left(0, \Sigma_u\right)\\[0.2cm]
\theta \sim f_\text{DP}(G, \alpha)\\[0.2cm]
\theta_l^* \sim \mathcal{N}\!\left(\mu_G, \Sigma_G\right)$$

This yields a mixture distribution in the form of
$$f(x_n|f_\text{DP}) = \sum_{l=1}^\infty \pi_l \; \mathcal{N}\!\left(x_n;\theta_l^*, \Sigma_u\right)$$
where the mixture weights $\pi_l$ and the Gaussian cluster means $\theta_l^*$ are determined by the Dirichlet Process (DP).
The base distribution $G$ and mixture distribution $f(x_n|\theta_l^*) = \mathcal{N}\!\left(x_n;\theta_l^*, \Sigma_u\right)$ form a conjugate model and can be formulated in the exponential family framework as
$$f(x_n|\eta_l^*) = h(x_n) * \exp\left(\eta_l^{*T} x_n - a(\eta_l^*)\right)\\[0.2cm]
f(\eta_l^*;\lambda_1, \lambda_2) = b \exp\left(\lambda_1^T \eta_l^*- \lambda_2 a(\eta_l^*)\right)$$
with natural parameter $\eta_l^* = \Sigma_u^{-1} \theta_l^*$  and hyperparameters $\lambda_1$ and $\lambda_2$ satisfying following relation
$$\mu_{\eta_l^*} = \Sigma_u^{-1} \mu_G = \frac{1}{\lambda_2} \Sigma_u^{-1} \lambda_1\\[0.2cm]
\Sigma_{\eta_l^*} = \Sigma_u^{-1} \Sigma_G \Sigma_u^{-1} = \frac{1}{\lambda_2} \Sigma_u^{-1},$$
i.e., given $\mu_G$, $\Sigma_G$ and $\Sigma_u$ the hyperparameters are automatically determined. Note that the natural parameter $\eta_l^* = \Sigma_u^{-1} \theta_l^*$ is again Gaussian with $\mu_{\eta_l^*}$ and $\Sigma_{\eta_l^*}$ because it is in linear relation with $\theta_l^*$.

Random cluster assignments are denoted as $z_n$ and are  i.i.d. distributed with a Categorical distribution (Multinomial with a single draw) where the probability of each category is given by the mixing weights $\pi_l$. The assignment variable $z_n$ indicates with which mixture component the data point $x_n$ is associated. Therefore the data can be described as arising from the following process:
1. First step
2. Second step
3. Third step

## CAVI
Mean field VI
$$q(v,\eta^*,z) = $$
Following parameters have to be choosen for initialization:
- Concentration Parameter $\alpha$
- Mean $\mu_G$ and variance $\Sigma_G$ of the base distribution
- Cluster variance $\Sigma_u$
- Truncation parameter $T$
- Assignment probabilities $\phi_{nt}$

We use synthetic data that comes from a Chinese Restaurant Process and perform VI on that. In general any arbitrary data set could be used to learn the posterior distribution $q(v,\eta^*,z)$ according to the model explained above. Given the posterior one can then estimate cluster means, cluster assignements and cluster weights according to well known estimators like the MMSE/MAP estimator. Note that the marginal distributions to do so are already obtained as output from the algorithm.
