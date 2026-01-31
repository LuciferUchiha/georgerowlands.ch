---
title: Anomaly Detection
type: docs
weight: 20
---

Anomaly detection is the problem of identifying data points that deviate significantly from normal patterns. This task arises in many domains: banks detect fraudulent transactions, doctors identify abnormal cells in medical images, and network security systems flag suspicious behavior. The challenge is to learn what constitutes normal behavior from data and then recognize when new observations fall outside this learned pattern.

## Problem Formalization

Given a dataset $X = \{x_1, \ldots, x_n\} \subseteq \mathbb{R}^D$ drawn from a normal class $N \subseteq \mathbb{R}^D$, we want to compute a function $\varphi: \mathbb{R}^D \to \{0, 1\}$ such that $\varphi(x) = 1$ if and only if $x \notin N$.

The **normal class** $N$ represents the set of all data points that we consider to be normal or expected behavior. For example, in fraud detection, the normal class consists of all legitimate transactions. In manufacturing quality control, it consists of all acceptable products without defects.

The key difficulty is that we typically only have examples of normal data during training. We must learn to recognize anomalies without seeing them, which is why this is also called **one-class classification** or **novelty detection**.

## Probabilistic Anomaly Detection

A natural approach is to view anomalies as **unlikely events**. Instead of trying to explicitly define what makes a point anomalous, we model what normal data looks like probabilistically. The intuition is simple: if we train a model on normal data, then normal points should have high probability under this model, while anomalies (which the model has never seen) should have low probability.

We fit a probabilistic model $p_\theta(x)$ to the training data $X = \{x_1, \ldots, x_n\}$, where $\theta$ represents the parameters of the model. The probability density $p_\theta(x)$ tells us how likely a point $x$ is under the learned model of normal behavior. For example, if we train on images of cats, then $p_\theta(\text{cat image})$ should be high, while $p_\theta(\text{dog image})$ should be low. We define the **anomaly score** of a point $x$ as:

$$
\text{score}(x) = -\log p_\theta(x)
$$

Why use the negative log-probability instead of the probability itself? There are several practical and theoretical reasons. Probabilities can become extremely small for high-dimensional data, leading to numerical underflow. The logarithm maps these tiny values to manageable negative numbers. It also converts products of probabilities into sums, simplifying computation and differentiation. The negative sign ensures that high anomaly scores correspond to unlikely points.

From an information-theoretic perspective, $-\log p(x)$ measures the **surprise** of observing $x$ under the model (see [information theory](/garden/ml/bayesian/variationalInference/#information-theory-surprise-and-entropy)). Since anomalies are surprising under the model of normal behavior, using surprise as the anomaly score is natural.

To classify points, we threshold the anomaly score: $\varphi(x) = 1$ if $\text{score}(x) > \tau$ for some threshold $\tau$. The choice of threshold depends on the application and the desired trade-off between false positives and false negatives. A **lower threshold** (more sensitive) catches more anomalies but also flags more normal points as anomalous (high false positive rate). This might be acceptable for fraud detection where we would rather investigate a few extra transactions than miss actual fraud. A **higher threshold** (more specific) reduces false alarms but may miss true anomalies (high false negative rate). This might be preferred in manufacturing quality control where false alarms lead to unnecessary production stoppages and associated costs. In practice, the threshold is often chosen by examining the **ROC curve** (Receiver Operating Characteristic) on validation data and selecting a point that balances the two error types according to business requirements.

## Gaussian Mixture Models

For anomaly detection, we need a flexible family of probability distributions that can model complex patterns while remaining tractable. A single [multivariate Gaussian](/garden/maths/probabilityStatistics/multiVariateGaussian) is too restrictive as it can only model a single ellipsoidal cluster of points. Real data often has multiple modes or clusters. For example, in network traffic monitoring, normal behavior might include both "daytime office traffic" and "nighttime backup traffic" as distinct patterns.

A popular choice is the **Gaussian Mixture Model (GMM)**, which can approximate a wide variety of distributions by combining multiple Gaussian components. GMMs are powerful for several reasons. First, they are **flexible** and can capture multimodal behavior: each component can model a different cluster or mode in the data. Second, they remain **tractable** despite this flexibility. We can compute probabilities, derivatives, and perform inference efficiently, unlike many complex non-parametric models. Third, they have a **universal approximation** property: any continuous probability density on a compact domain can be approximated arbitrarily well by a GMM with sufficiently many components, similar to how neural networks can approximate any function.

A Gaussian Mixture Model with $K$ components is a probability distribution whose density is a weighted sum of $K$ Gaussian densities:

$$
p(x) = \sum_{k=1}^K \pi_k \mathcal{N}(x \mid \mu_k, \Sigma_k)
$$

where:
- $\pi_k \in \mathbb{R}$ are **mixing weights** satisfying $\pi_k > 0$ and $\sum_{k=1}^K \pi_k = 1$
- $\mu_k \in \mathbb{R}^d$ is the **mean** of the $k$-th Gaussian component
- $\Sigma_k \in \mathbb{R}^{d \times d}$ is the **covariance matrix** of the $k$-th component (positive definite)
- $\mathcal{N}(x \mid \mu_k, \Sigma_k)$ is the [multivariate Gaussian](/garden/maths/probabilityStatistics/multiVariateGaussian) density

Think of a GMM as modeling a distribution with **multiple clusters** or "galaxies" of points. Each Gaussian component $\mathcal{N}(x \mid \mu_k, \Sigma_k)$ represents one cluster, centered at $\mu_k$ with spread determined by $\Sigma_k$. The mixing weight $\pi_k$ determines how much probability mass is assigned to that cluster. If $\pi_1 = 0.7$ and $\pi_2 = 0.3$, then 70% of the data points belong to the first cluster and 30% to the second.

This interpretation leads naturally to viewing a GMM as a **latent variable model**. A latent variable is an unobserved quantity that helps explain the structure of the data. In a GMM, the latent variable is the cluster assignment: which of the $K$ components generated each data point. We do not observe this assignment directly, but if we knew it, the model would be much simpler. Imagine that each data point is generated by a two-step process:

1. First, we randomly choose one of the $K$ components. Component $k$ is chosen with probability $\pi_k$.
2. Then, we sample a point from the chosen Gaussian component.

Let $z \in \{1, \ldots, K\}$ denote the latent variable indicating which component generated the point. The joint distribution is:

$$
p(x, z) = p(z) p(x \mid z) = \pi_z \mathcal{N}(x \mid \mu_z, \Sigma_z)
$$

Marginalizing over $z$ gives the GMM density:

$$
p(x) = \sum_{z=1}^K p(x, z) = \sum_{k=1}^K \pi_k \mathcal{N}(x \mid \mu_k, \Sigma_k)
$$

This latent variable interpretation is crucial for the EM algorithm that we will derive later.

## Fitting GMMs with Maximum Likelihood

Given a dataset $X = \{x_1, \ldots, x_n\}$, how do we find the parameters $\theta$ of a GMM? The standard approach is **Maximum Likelihood Estimation (MLE)**: choose $\theta$ to maximize the probability of observing the data.

The **likelihood** of the data is:

$$
p_\theta(X) = \prod_{i=1}^n p_\theta(x_i) = \prod_{i=1}^n \sum_{k=1}^K \pi_k \mathcal{N}(x_i \mid \mu_k, \Sigma_k)
$$

In practice, we work with the **log-likelihood** because it is numerically more stable and easier to optimize:

$$
\log p_\theta(X) = \sum_{i=1}^n \log p_\theta(x_i) = \sum_{i=1}^n \log \left( \sum_{k=1}^K \pi_k \mathcal{N}(x_i \mid \mu_k, \Sigma_k) \right)
$$

The challenge is that this objective function is difficult to optimize directly. The logarithm is outside the sum over components, which means we cannot decompose the problem into independent optimization of each component. This is a typical dealbreaker in optimization.

### The Hidden Variable Trick

However, we can exploit the latent variable interpretation of GMMs to simplify the optimization. Imagine we knew which component generated each data point, the optimization would be much easier. Let $Z = \{z_1, \ldots, z_n\}$ where $z_i \in \{1, \ldots, K\}$ indicates which component generated $x_i$.

If we knew $z_i$ for each data point, then we would know exactly which Gaussian generated that point. The probability of observing both $x_i$ and $z_i$ together is simply the probability of choosing component $z_i$ times the probability that component generates $x_i$:

$$
p_\theta(x_i, z_i) = p(z_i) \cdot p(x_i \mid z_i) = \pi_{z_i} \cdot \mathcal{N}(x_i \mid \mu_{z_i}, \Sigma_{z_i})
$$

Notice that the sum over all $K$ components from the observed-data likelihood has **disappeared**. We no longer need to sum because $z_i$ tells us exactly which component to use. This is the key simplification.

The **complete-data log-likelihood** (including the latent variables) is therefore:

$$
\log p_\theta(X, Z) = \sum_{i=1}^n \log p_\theta(x_i, z_i) = \sum_{i=1}^n \left( \log \pi_{z_i} + \log \mathcal{N}(x_i \mid \mu_{z_i}, \Sigma_{z_i}) \right)
$$

This is much simpler than the observed-data log-likelihood because the logarithm is no longer applied to a sum over components. Each term depends on only one component (the one specified by $z_i$), so we could optimize each component's parameters independently if we knew $Z$.

However, we do not observe $Z$. The **Expectation-Maximization (EM) algorithm** provides a clever way to work around this by iteratively estimating $Z$ and optimizing $\theta$.

## The EM Algorithm

The EM algorithm is an iterative method for finding [maximum likelihood](/garden/ml/bayesian/bayesianLearning/#mle-and-map) estimates in models with latent variables. In our GMM setting, the latent variables $Z = \{z_1, \ldots, z_n\}$ are the unknown cluster assignments. Each $z_i$ indicates which of the $K$ Gaussian components generated data point $x_i$, but we never observe these assignments directly.

The key insight behind EM comes from the [Evidence Lower Bound (ELBO)](/garden/ml/bayesian/variationalInference/#deriving-the-elbo) decomposition. For any distribution $q(z)$ over the latent variables, the log-likelihood can be decomposed as:

$$
\log p_\theta(x) = \underbrace{\mathbb{E}_{z \sim q} \left[ \log \frac{p_\theta(x, z)}{q(z)} \right]}_{\text{ELBO: } \mathcal{L}(q, \theta)} + \underbrace{\text{KL}(q \| p_\theta(\cdot \mid x))}_{\geq 0}
$$

where the ELBO provides a lower bound on the log-likelihood. Since the KL divergence is always non-negative, maximizing the ELBO is equivalent to maximizing the log-likelihood when the KL term is zero, which occurs when $q(z) = p_\theta(z \mid x)$. For a detailed derivation and intuition, see the [variational inference notes](/garden/ml/bayesian/variationalInference/#deriving-the-elbo).

The EM algorithm alternates between two steps:

1. **E-step (Expectation)**: Fix the parameters $\theta$ and compute the posterior distribution over latent variables $q(z) = p_\theta(z \mid x)$. This distribution represents our current best guess about which component generated each data point. In the GMM case, this means computing the **responsibilities** $\gamma_{ik} = p(z_i = k \mid x_i, \theta)$, which is the probability that component $k$ generated point $x_i$.

2. **M-step (Maximization)**: Fix the distribution $q$ from the E-step and find new parameters $\theta$ that maximize the expected complete-data log-likelihood under $q$. In the GMM case, this means updating the means $\mu_k$, covariances $\Sigma_k$, and mixing weights $\pi_k$ based on the responsibilities.

The algorithm is initialized with some $\theta^{(0)}$ and then iterates:

**For $t = 1, 2, \ldots$:**

**E-step**: Compute $q^{(t)}(z) = p_{\theta^{(t-1)}}(z \mid x)$.

**M-step**: Set $\theta^{(t)} = \arg\max_\theta \mathcal{L}(q^{(t)}, \theta)$.

### Why EM Works

Each iteration of EM increases the log-likelihood (or keeps it the same). After the E-step at iteration $t$, we set $q^{(t)} = p_{\theta^{(t-1)}}(\cdot \mid x)$, which makes the KL divergence term in the ELBO decomposition zero. This means $\log p_{\theta^{(t-1)}}(x) = \mathcal{L}(q^{(t)}, \theta^{(t-1)})$.

After the M-step, we maximize the ELBO with respect to $\theta$:

$$
\mathcal{L}(q^{(t)}, \theta^{(t)}) \geq \mathcal{L}(q^{(t)}, \theta^{(t-1)}) = \log p_{\theta^{(t-1)}}(x)
$$

Since the ELBO is a lower bound on the log-likelihood, we have:

$$
\log p_{\theta^{(t)}}(x) \geq \mathcal{L}(q^{(t)}, \theta^{(t)}) \geq \log p_{\theta^{(t-1)}}(x)
$$

Thus, the log-likelihood increases monotonically and the algorithm converges to a local maximum.

### EM for Gaussian Mixture Models

For GMMs, the E-step computes the **responsibility** $\gamma_{ik}$, which is the posterior probability that component $k$ generated point $x_i$:

$$
\gamma_{ik} = p_{\theta^{(t-1)}}(z_i = k \mid x_i) = \frac{\pi_k^{(t-1)} \mathcal{N}(x_i \mid \mu_k^{(t-1)}, \Sigma_k^{(t-1)})}{\sum_{j=1}^K \pi_j^{(t-1)} \mathcal{N}(x_i \mid \mu_j^{(t-1)}, \Sigma_j^{(t-1)})}
$$

The M-step then updates the parameters using weighted maximum likelihood, where each point $x_i$ is weighted by $\gamma_{ik}$:

$$
\begin{align}
\pi_k^{(t)} &= \frac{1}{n} \sum_{i=1}^n \gamma_{ik} \\
\mu_k^{(t)} &= \frac{\sum_{i=1}^n \gamma_{ik} x_i}{\sum_{i=1}^n \gamma_{ik}} \\
\Sigma_k^{(t)} &= \frac{\sum_{i=1}^n \gamma_{ik} (x_i - \mu_k^{(t)})(x_i - \mu_k^{(t)})^T}{\sum_{i=1}^n \gamma_{ik}}
\end{align}
$$

These updates have intuitive interpretations:
- $\pi_k^{(t)}$ is the average responsibility of component $k$ across all points (the proportion of data assigned to component $k$).
- $\mu_k^{(t)}$ is the weighted mean of all points, weighted by their responsibility to component $k$.
- $\Sigma_k^{(t)}$ is the weighted covariance.

{{< callout type="example" title="EM Iteration" >}}
Suppose we have $n = 100$ data points and a GMM with $K = 3$ components. After initialization:

**E-step**: For each of the 100 points and each of the 3 components, we compute $\gamma_{ik}$ using the current parameters. This gives us a $100 \times 3$ matrix of responsibilities. For example, $\gamma_{17,2} = 0.85$ might mean that component 2 is highly responsible for point 17.

**M-step**: We update each component's parameters. For component 2, we compute the new mean $\mu_2^{(t)}$ as a weighted average of all 100 points, where point 17 has weight 0.85, point 5 might have weight 0.02, and so on. Points with high responsibility contribute more to the new parameters.

The algorithm repeats until convergence (when the log-likelihood stops increasing significantly).
{{< /callout >}}

## Anomaly Detection with PCA and GMM

Now we can put together all the pieces to build a complete anomaly detection system. The approach combines dimensionality reduction with density estimation in a two-stage pipeline.

Why not fit a GMM directly to the high-dimensional data? While this is possible in principle, it faces several challenges. First, the **curse of dimensionality** makes density estimation unreliable in high dimensions. As dimensionality increases, data becomes increasingly sparse and the volume of space grows exponentially. A GMM would need exponentially many components to cover this space, and we rarely have enough data to reliably estimate the parameters of so many high-dimensional Gaussians. Second, the **computational cost** of fitting a GMM scales poorly with dimension. Each Gaussian component has a covariance matrix with $D(D+1)/2$ parameters, so for $D = 1000$ dimensions and $K = 10$ components, we would need to estimate over 5 million covariance parameters. Third, high-dimensional data often lies near a **low-dimensional manifold**. Most of the variation in the data can be captured by a small number of directions. For example, images of faces vary primarily in lighting, pose, and identity, not independently in each pixel.

[PCA](/garden/maths/linearAlgebra/pca) addresses these issues by finding the directions of maximum variance in the training data. These directions typically capture the intrinsic structure of normal behavior. By projecting to a lower dimension $d \ll D$, we concentrate the data and make density estimation tractable. The pipeline has three stages:

1. **Dimensionality Reduction**: Apply PCA to the training data $X \subseteq \mathbb{R}^D$, obtaining a projection $\pi: \mathbb{R}^D \to \mathbb{R}^d$ where $d \ll D$. Choose $d$ to retain most of the variance (such as 95% or 99%).

2. **Density Estimation**: Fit a GMM with $K$ components to the projected data $\{\pi(x_1), \ldots, \pi(x_n)\}$ using the EM algorithm. This gives us a density model $p_\theta(\cdot)$ in the low-dimensional space $\mathbb{R}^d$.

3. **Anomaly Scoring**: For a new point $x$, compute its projection $z = \pi(x)$ and evaluate the anomaly score $-\log p_\theta(z)$. Using a threshold $\tau$, classify $x$ as an anomaly if the score exceeds $\tau$.
