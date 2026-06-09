---
title: Non-Parametric Bayesian Methods
type: docs
weight: 3
---

Non-parametric Bayesian methods provide a powerful framework for handling models where the number of parameters can grow with the data. Unlike traditional parametric approaches that fix the model complexity in advance, non-parametric methods allow the model to adapt its complexity based on the observed data. This is particularly useful for clustering problems where we do not know the number of clusters beforehand.

## Motivation for Non-Parametric Methods

Consider fitting a Gaussian Mixture Model (GMM) to data when we do not know the true number of clusters. Traditional Maximum Likelihood Estimation (MLE) fails catastrophically in this setting. If we allow infinitely many components, MLE will overfit by placing one component at each data point, leading to zero training error but terrible generalization.

The solution is to introduce regularization through a prior distribution. This naturally leads us to a Bayesian approach where we place priors on the model parameters. In the non-parametric setting, we consider GMMs with infinitely many components but use priors that encourage most mixing coefficients to be near zero. This allows the data to determine how many components are actually needed.

## Bayesian Inference for Multivariate Gaussians

Before tackling the full non-parametric case, we need to understand Bayesian inference for multivariate Gaussians. This builds on the conjugate prior framework we saw for univariate Gaussians.

### The Normal-Inverse-Wishart Prior

For a multivariate Gaussian with unknown mean $\mu$ and covariance $\Sigma$, the conjugate prior is the **Normal-Inverse-Wishart (NIW)** distribution. This joint distribution over $(\mu, \Sigma)$ is defined by four hyperparameters: $m_0$ (prior mean), $k_0$ (prior precision scale), $S_0$ (prior scatter matrix), and $\nu_0$ (prior degrees of freedom).

The NIW prior has a special structure that captures our prior beliefs:

$$
p(\mu, \Sigma) = p(\mu \mid \Sigma) p(\Sigma)
$$

where $\Sigma \sim \text{IW}(S_0, \nu_0)$ follows an Inverse-Wishart distribution and $\mu \mid \Sigma \sim \mathcal{N}(m_0, \Sigma/k_0)$ is a Gaussian conditioned on $\Sigma$.

Given data $X = \{x_1, \ldots, x_n\}$ where $x_i \sim \mathcal{N}(\mu, \Sigma)$, the posterior remains in the NIW family:

$$
p(\mu, \Sigma \mid X) = \text{NIW}(\mu, \Sigma \mid m_p, k_p, S_p, \nu_p)
$$

The posterior parameters update as:

$$
\begin{align}
m_p &= \frac{k_0}{k_0 + n} m_0 + \frac{n}{k_0 + n} \bar{x} \\
k_p &= k_0 + n \\
\nu_p &= \nu_0 + n \\
S_p &= S_0 + S_{\bar{x}} + k_0 m_0 m_0^T - k_p m_p m_p^T
\end{align}
$$

where $\bar{x} = \frac{1}{n} \sum_{i=1}^n x_i$ is the sample mean and $S_{\bar{x}} = \sum_{i=1}^n (x_i - \bar{x})(x_i - \bar{x})^T$ is the sample scatter matrix.

An important observation is that computing the posterior only requires the sample mean and sample scatter matrix. This means we do not need to store the entire dataset once we have computed these sufficient statistics.

### Semi-Conjugate Priors and Gibbs Sampling

When the prior is not fully conjugate, we cannot compute the posterior in closed form. For instance, if we use independent priors $\mu \sim \mathcal{N}(m_0, V_0)$ and $\Sigma \sim \text{IW}(S_0, \nu_0)$, the posterior is not tractable. However, the conditional posteriors remain tractable:

$$
\begin{align}
p(\mu \mid \Sigma, X) &= \mathcal{N}(m_p, V_p) \\
p(\Sigma \mid \mu, X) &= \text{IW}(S_p, \nu_p)
\end{align}
$$

This structure enables us to use **Gibbs sampling** for approximate inference.

## Gibbs Sampling

Gibbs sampling is a Markov Chain Monte Carlo (MCMC) method for generating samples from a joint distribution $p(\theta)$ when direct sampling is difficult but sampling from conditional distributions is tractable. The key idea is to iteratively sample each component of $\theta$ conditioned on the current values of all other components.

For a random vector $\Theta = (\Theta_1, \Theta_2, \ldots, \Theta_\ell)$ with intractable joint distribution $p(\theta)$, suppose we can sample from each conditional distribution $p(\theta_i \mid \theta_1, \ldots, \theta_{i-1}, \theta_{i+1}, \ldots, \theta_\ell)$. The Gibbs sampling algorithm proceeds as follows:

1. Initialize $\Theta^{(0)}$ arbitrarily
2. For each iteration $t = 1, 2, \ldots$:
   - Pick a random permutation $\pi$ of $\{1, 2, \ldots, \ell\}$
   - For $i = 1$ to $\ell$:
     - Sample $\Theta_{\pi(i)}^{(t)} \sim p(\cdot \mid \Theta_{\pi(1)}^{(t)}, \ldots, \Theta_{\pi(i-1)}^{(t)}, \Theta_{\pi(i+1)}^{(t-1)}, \ldots, \Theta_{\pi(\ell)}^{(t-1)})$

Under mild regularity conditions, the distribution of $\Theta^{(t)}$ converges to $p(\theta)$ as $t \to \infty$. In practice, we discard the first $M$ samples (the burn-in period) and use subsequent samples as approximate draws from $p(\theta)$.

The random permutation in step 2 is not strictly necessary for convergence, but it can improve mixing. In practice, a fixed scanning order often works well.

## Bayesian Inference for Gaussian Mixture Models

Now we consider the full GMM setting with a known finite number of clusters $K$. The model has several components:

- Mixing coefficients $\pi = (\pi_1, \ldots, \pi_K)$ with $\sum_{k=1}^K \pi_k = 1$
- Cluster parameters $(\mu_k, \Sigma_k)$ for $k = 1, \ldots, K$
- Latent cluster assignments $z_i \in \{1, \ldots, K\}$ for each data point $x_i$

The semi-conjugate priors are:

- $(\mu_k, \Sigma_k) \sim \text{NIW}(m_0, k_0, S_0, \nu_0)$ for each cluster $k$
- $\pi \sim \text{Dir}(\alpha_0, \ldots, \alpha_0)$ (Dirichlet distribution)

The Dirichlet distribution is a multivariate generalization of the Beta distribution. It is the conjugate prior for the categorical distribution, just as the Beta is conjugate to the Bernoulli.

### Understanding the Model with Bayesian Networks

We can represent the GMM as a Bayesian network to understand the conditional independence structure. The hyperparameters $(m_0, k_0, S_0, \nu_0)$ generate the cluster parameters $(\mu_k, \Sigma_k)$. The mixing coefficients $\pi$ (drawn from the Dirichlet prior with parameter $\alpha_0$) determine the cluster assignments $z_i$. Finally, each data point $x_i$ is generated from $\mathcal{N}(\mu_{z_i}, \Sigma_{z_i})$.

A key insight from this network structure is **d-separation**. Two variables are conditionally independent given observed variables if every path between them is blocked by an observed variable. A path is blocked if it contains:

- A chain $A \to B \to C$ where $B$ is observed
- A fork $A \leftarrow B \to C$ where $B$ is observed
- A collider $A \to B \leftarrow C$ where neither $B$ nor any descendant of $B$ is observed

For our GMM, the cluster assignments $z$ d-separate the cluster parameters from the mixing coefficients and from the hyperparameters. This leads to important simplifications:

$$
\begin{align}
p(\mu_k, \Sigma_k \mid z, \pi, X) &= p(\mu_k, \Sigma_k \mid z, X) \\
p(\pi \mid z, \mu_1, \Sigma_1, \ldots, \mu_K, \Sigma_K, X) &= p(\pi \mid z)
\end{align}
$$

Another useful observation is that the conditional distribution of $X_{-i}$ (all data except $x_i$) given $z$ is the same as the conditional distribution given $z_{-i}$ (all assignments except $z_i$):

$$
p(X_{-i} \mid z) = p(X_{-i} \mid z_{-i})
$$

This is because once we know which cluster each point belongs to, the actual label values (as opposed to the partition structure) do not matter.

### Collapsed Gibbs Sampling

Standard Gibbs sampling for GMMs would alternate between sampling $(z, \pi, \{\mu_k, \Sigma_k\})$. However, we can improve convergence by **marginalizing out** some variables. This technique is called **collapsed Gibbs sampling**.

The key idea is to integrate out the cluster parameters and mixing coefficients, sampling only the cluster assignments $z$. After obtaining samples of $z$, we can then sample the parameters conditioned on $z$ if needed.

For sampling $z_i$ given all other assignments $z_{-i}$ and data $X$, we use Bayes' rule:

$$
p(z_i = k \mid z_{-i}, X) \propto p(z_i = k \mid z_{-i}) \cdot p(X \mid z_i = k, z_{-i})
$$

We can simplify the likelihood term using d-separation. Since the data points are conditionally independent given the cluster assignments, and since $X_{-i}$ is conditionally independent of $z_i$ given $z_{-i}$:

$$
p(X \mid z_i = k, z_{-i}) = p(x_i \mid X_{-i}, z_i = k, z_{-i}) \cdot p(X_{-i} \mid z_i = k, z_{-i})
$$

The second factor does not depend on $k$, so we can treat it as a constant. The first factor is:

$$
p(x_i \mid X_{-i}, z_i = k, z_{-i}) = p(x_i \mid \{x_j : j \neq i, z_j = k\})
$$

This is the **posterior predictive distribution** for a new point from cluster $k$, given all the data currently assigned to that cluster (excluding $x_i$).

For the prior term $p(z_i = k \mid z_{-i})$, we use the Dirichlet-Multinomial conjugacy. With a symmetric Dirichlet prior $\text{Dir}(\alpha_0, \ldots, \alpha_0)$, the posterior for the mixing coefficients given $z_{-i}$ is $\text{Dir}(\alpha_0 + N_{1,-i}, \ldots, \alpha_0 + N_{K,-i})$ where $N_{k,-i}$ is the number of points (excluding $i$) assigned to cluster $k$. Integrating out $\pi$:

$$
p(z_i = k \mid z_{-i}) = \frac{\alpha_0 + N_{k,-i}}{\sum_{k'} (\alpha_0 + N_{k',-i})} = \frac{\alpha_0 + N_{k,-i}}{K \alpha_0 + n - 1}
$$

Wait, this is for a finite number of clusters. Let me reconsider this more carefully for the infinite case later.

### Rao-Blackwellization

Collapsed Gibbs sampling is not just a computational trick. It is theoretically justified by the **Rao-Blackwell theorem**, which states that conditioning on additional information reduces variance.

Suppose we want to estimate $\mathbb{E}[f(\Theta)]$ where $\Theta = (\Theta', Z)$. We could sample $(\Theta', Z)$ jointly and estimate using $\frac{1}{T} \sum_{t=1}^T f(\Theta'^{(t)}, Z^{(t)})$. However, the Rao-Blackwell theorem tells us that the estimator

$$
\frac{1}{T} \sum_{t=1}^T \mathbb{E}_{\Theta'}[f(\Theta', Z^{(t)}) \mid Z^{(t)}]
$$

has lower variance. This is exactly what collapsed Gibbs achieves: by marginalizing over $\Theta'$ and sampling only $Z$, we obtain better estimates.

More precisely, for any function $f$:

$$
\text{Var}_Z[\mathbb{E}_{\Theta'}[f(\Theta', Z) \mid Z]] \leq \text{Var}_{\Theta',Z}[f(\Theta', Z)]
$$

To prove this inequality, we use the law of total variance:

$$
\text{Var}_{\Theta',Z}[f(\Theta', Z)] = \mathbb{E}_Z[\text{Var}_{\Theta'}[f(\Theta', Z) \mid Z]] + \text{Var}_Z[\mathbb{E}_{\Theta'}[f(\Theta', Z) \mid Z]]
$$

Since variance is non-negative, the first term on the right is non-negative, which gives us the desired inequality.

This result shows that collapsed Gibbs sampling, which effectively uses the conditional expectation $\mathbb{E}_{\Theta'}[f(\Theta', Z) \mid Z]$, has lower variance than the joint sampler.

## Non-Parametric Gaussian Mixture Models

Now we address the central question: how do we perform clustering when the number of clusters is unknown? The non-parametric Bayesian approach is to consider a GMM with **infinitely many components** but use a prior that encourages most components to be unused.

### The Chinese Restaurant Process

The **Chinese Restaurant Process (CRP)** provides an elegant way to define a prior over cluster assignments for infinitely many potential clusters. Imagine a restaurant with infinitely many tables. Customers enter one at a time and choose where to sit according to the following rule:

- The first customer sits at the first table
- Customer $n+1$ sits:
  - At an occupied table $k$ with probability proportional to $N_k$ (the number of customers already at table $k$)
  - At a new unoccupied table with probability proportional to $\alpha$ (the concentration parameter)

More precisely, customer $n+1$ sits at table $k$ with probability:

$$
p(z_{n+1} = k \mid z_1, \ldots, z_n) = \begin{cases}
\frac{N_k}{\alpha + n} & \text{if } k \text{ is occupied} \\
\frac{\alpha}{\alpha + n} & \text{if } k \text{ is a new table}
\end{cases}
$$

The parameter $\alpha > 0$ is the **concentration parameter**. Larger $\alpha$ encourages more clusters to be created. After $n$ customers, the expected number of occupied tables is approximately $\alpha \log n$.

{{< callout type="example" title="Chinese Restaurant Process Simulation" >}}
Consider $\alpha = 1$ and watch how 10 customers are seated:

- Customer 1: Sits at table 1 (the first table)
- Customer 2: Table 1 with prob 1/2, new table with prob 1/2. Suppose sits at table 1.
- Customer 3: Table 1 with prob 2/3, new table with prob 1/3. Suppose sits at new table 2.
- Customer 4: Table 1 with prob 2/4, table 2 with prob 1/4, new table with prob 1/4.

The process continues with probabilities always proportional to current table sizes or $\alpha$ for new tables.
{{< /callout >}}

The CRP has a remarkable property: the probability of a particular seating arrangement depends only on the partition structure, not the order in which customers arrived. This is the property of **exchangeability**.

### Exchangeability

A sequence of random variables $X_1, X_2, \ldots, X_n$ is **exchangeable** if the joint distribution is invariant to permutations of the indices. Formally:

$$
p(X_1, \ldots, X_n) = p(X_{\pi(1)}, \ldots, X_{\pi(n)})
$$

for any permutation $\pi$ of $\{1, \ldots, n\}$.

An infinite sequence $X_1, X_2, \ldots$ is exchangeable if every finite subsequence is exchangeable.

Exchangeability is weaker than independence. Independent and identically distributed (i.i.d.) random variables are always exchangeable, but exchangeable variables need not be independent. The CRP generates exchangeable cluster assignments even though they are clearly not independent (later assignments depend on earlier ones through the table counts).

To verify exchangeability of the CRP, consider any permutation of customers. The probability of the resulting seating arrangement depends only on which tables have which sizes, not on the order customers arrived. For instance, having 3 customers at table 1 and 2 at table 2 has the same probability regardless of the arrival order.

More formally, if we have $K$ occupied tables with $N_1, N_2, \ldots, N_K$ customers respectively (with $\sum_k N_k = n$), the probability is:

$$
p(z_1, \ldots, z_n) = \frac{\alpha^K \prod_{k=1}^K (N_k - 1)!}{\alpha(\alpha+1)\cdots(\alpha+n-1)}
$$

This expression depends only on the partition structure $(N_1, \ldots, N_K)$, confirming exchangeability.

### De Finetti's Theorem

The connection between exchangeability and mixture models is formalized by **De Finetti's theorem**. In its simplest form for Bernoulli variables, the theorem states:

If $X_1, X_2, \ldots$ is an infinite exchangeable sequence of Bernoulli random variables, then there exists a probability measure $p(\theta)$ on $[0,1]$ such that:

$$
\Pr(X_1 = x_1, \ldots, X_n = x_n) = \int_0^1 \prod_{i=1}^n \theta^{x_i} (1-\theta)^{1-x_i} p(\theta) d\theta
$$

In other words, conditioned on some latent parameter $\theta$, the $X_i$ are independent Bernoulli$(\theta)$ random variables. The exchangeable sequence arises by marginalizing over $\theta$ with respect to some prior $p(\theta)$.

The general **De Finetti-Hewitt-Savage theorem** extends this to arbitrary random variables: any infinite exchangeable sequence can be represented as a mixture model. There exists a probability measure $\mu$ on the space of probability measures such that:

$$
\Pr(X_1 \in A_1, \ldots, X_n \in A_n) = \int \prod_{i=1}^n Q(A_i) \mu(Q) dQ
$$

This theorem provides the theoretical foundation for Bayesian non-parametric methods. The CRP cluster assignments are exchangeable, so by De Finetti's theorem, they arise from some underlying random probability measure $Q$. This measure is drawn from a distribution over probability measures, which brings us to the **Dirichlet Process**.

## The Dirichlet Process

The Dirichlet Process (DP) is a distribution over probability measures. It generalizes the Dirichlet distribution (which is a distribution over probability vectors) to the infinite-dimensional setting.

### Definition

Let $\Omega$ be a sample space, $\alpha > 0$ be a concentration parameter, and $H$ be a base distribution over $\Omega$. A random probability measure $G: \Omega \to \mathbb{R}^+$ follows a Dirichlet Process $DP(H, \alpha)$ if for any finite partition $(T_1, \ldots, T_m)$ of $\Omega$:

$$
(G(T_1), \ldots, G(T_m)) \sim \text{Dir}(\alpha H(T_1), \ldots, \alpha H(T_m))
$$

In other words, the probabilities that $G$ assigns to any finite partition follow a Dirichlet distribution with parameters determined by the base measure $H$ and concentration $\alpha$.

The base distribution $H$ specifies where the clusters are likely to be located. The expected value of $G$ is $H$, so $H$ can be thought of as the "prior mean" of the random measure $G$. The concentration parameter $\alpha$ controls how closely $G$ follows $H$. Larger $\alpha$ means $G$ is closer to $H$ in expectation.

### Connection to the Chinese Restaurant Process

The Dirichlet Process and Chinese Restaurant Process are intimately connected. If we draw $G \sim DP(H, \alpha)$ and then sample $\theta_i \sim G$ independently for $i = 1, 2, \ldots$, the resulting sequence is exchangeable. Moreover, the predictive distribution is:

$$
\theta_{n+1} \mid \theta_1, \ldots, \theta_n \sim \frac{1}{\alpha + n} \left( \sum_{k=1}^K N_k \delta_{\theta_k^*} + \alpha H \right)
$$

where $\theta_1^*, \ldots, \theta_K^*$ are the unique values among $\{\theta_1, \ldots, \theta_n\}$, $N_k$ is the number of times $\theta_k^*$ appears, and $\delta_{\theta_k^*}$ is a point mass at $\theta_k^*$.

This shows that with probability $\frac{N_k}{\alpha + n}$, the new sample equals an existing value $\theta_k^*$, and with probability $\frac{\alpha}{\alpha + n}$, it is drawn fresh from the base distribution $H$. This is exactly the Chinese Restaurant Process, where:

- Each unique value $\theta_k^*$ corresponds to a table
- The number of data points with that value ($N_k$) is the table size
- Choosing an existing value is like sitting at an occupied table
- Drawing from $H$ is like sitting at a new table

The CRP is thus the **marginal distribution** of cluster assignments when we integrate out the cluster parameters drawn from a Dirichlet Process.

## Collapsed Gibbs Sampling for Non-Parametric GMMs

Putting everything together, we can now perform Bayesian inference for Gaussian mixture models with an unknown number of clusters. The hierarchical model is:

- $G \sim DP(H, \alpha)$ where $H = \text{NIW}(m_0, k_0, S_0, \nu_0)$
- $(\mu_k, \Sigma_k) \sim G$ for each potential cluster
- $z_i$ indicates which cluster parameter value $x_i$ belongs to
- $x_i \sim \mathcal{N}(\mu_{z_i}, \Sigma_{z_i})$

The collapsed Gibbs sampler marginalizes out both the random measure $G$ and the cluster parameters $\{(\mu_k, \Sigma_k)\}$. We iteratively resample each $z_i$ given all other assignments and the data.

The sampling distribution is:

$$
p(z_i = k \mid z_{-i}, X) \propto p(z_i = k \mid z_{-i}) \cdot p(x_i \mid \{x_j : j \neq i, z_j = k\})
$$

For an existing cluster $k$ (one with $N_{k,-i} > 0$):

$$
p(z_i = k \mid z_{-i}) = \frac{N_{k,-i}}{\alpha + n - 1}
$$

where $N_{k,-i}$ is the number of data points (excluding $x_i$) currently assigned to cluster $k$.

The likelihood term $p(x_i \mid \{x_j : j \neq i, z_j = k\})$ is the posterior predictive distribution from the NIW prior updated with the data currently in cluster $k$. This is a multivariate Student-t distribution with parameters determined by the NIW posterior.

For a new cluster (the option to start a new table):

$$
p(z_i = * \mid z_{-i}) = \frac{\alpha}{\alpha + n - 1}
$$

The likelihood term $p(x_i \mid *)$ is the prior predictive distribution from the base NIW distribution $H$. This is also a multivariate Student-t distribution, but with parameters from the prior rather than updated with any data.

The complete algorithm is:

**For each iteration:**
1. For each data point $i$ (in random order):
   - Remove $x_i$ from its current cluster
   - For each existing cluster $k$ with $N_{k,-i} > 0$:
     - Compute $p_k(x_i) = p(x_i \mid \{x_j : j \neq i, z_j = k\})$ using NIW posterior predictive
     - Compute $p(z_i = k \mid z_{-i}) = \frac{N_{k,-i}}{\alpha + n - 1}$
   - For the new cluster option:
     - Compute $p_*(x_i) = p(x_i \mid H)$ using NIW prior predictive
     - Compute $p(z_i = * \mid z_{-i}) = \frac{\alpha}{\alpha + n - 1}$
   - Sample $z_i$ from the categorical distribution with probabilities proportional to:
     - $p_k(x_i) \cdot p(z_i = k \mid z_{-i})$ for each existing cluster $k$
     - $p_*(x_i) \cdot p(z_i = * \mid z_{-i})$ for a new cluster
   - Add $x_i$ to its new cluster (updating sufficient statistics)
   - If any cluster becomes empty, remove it and relabel

This algorithm automatically determines the number of clusters from the data. Clusters can be created or destroyed as the sampler runs. The concentration parameter $\alpha$ controls the trade-off: larger $\alpha$ encourages more clusters.

The beauty of this approach is that we never explicitly represent infinitely many clusters. At any point in the algorithm, we only track the currently occupied clusters (analogous to occupied tables in the CRP). The infinite capacity exists implicitly through the ability to always create new clusters.

## Summary

Non-parametric Bayesian methods provide a principled framework for models with unbounded complexity. The key insights are:

The Chinese Restaurant Process defines an exchangeable prior over cluster assignments without fixing the number of clusters in advance. The concentration parameter controls the expected number of clusters, which grows logarithmically with the data size. This combinatorial construction gives an intuitive understanding of the clustering process.

The Dirichlet Process is a distribution over probability measures that underlies the CRP. It generalizes the Dirichlet distribution to infinite dimensions and provides the rigorous measure-theoretic foundation for non-parametric mixture models. The DP is characterized by its base distribution (the expected measure) and concentration parameter (controlling variability).

Exchangeability and De Finetti's theorem connect the combinatorial CRP construction to the measure-theoretic DP construction. Any exchangeable sequence can be represented as a mixture model with respect to some random measure. This theorem justifies the Bayesian modeling approach and shows that exchangeable cluster assignments naturally arise from Dirichlet Process priors.

Collapsed Gibbs sampling marginalizes out continuous parameters to improve convergence. By integrating over the cluster parameters and the random measure $G$, we sample only the discrete cluster assignments. This is justified by the Rao-Blackwell theorem, which proves that marginalization reduces estimator variance compared to joint sampling.

Understanding d-separation in Bayesian networks reveals conditional independence structure that simplifies inference. In the GMM, the cluster assignments d-separate the cluster parameters from the mixing coefficients and hyperparameters. This allows us to factor the sampling distribution and work with simpler conditional distributions.

Together, these ideas enable practical Bayesian inference for clustering with unknown numbers of clusters. The approach is widely used in applications from topic modeling (where documents are clustered into topics) to genomics (where genes are clustered into functional groups) to computer vision (where image regions are clustered into objects).
