---
title: Variational Inference
type: docs
weight: 3
math: true
---

In [Bayesian Linear Regression](/garden/ml/bayesian/bayesianlearning/), we saw how probabilistic inference allows us to compute the full posterior distribution over model parameters, naturally quantifying our uncertainty. The key insight was that when we combine a Gaussian prior with a Gaussian likelihood, the posterior is also Gaussian giving us a closed-form result thanks to **conjugacy**. This allowed us to derive exact formulas for the posterior mean and covariance, and from there make predictions that account for model uncertainty.

However, this tractability is the exception rather than the rule. In most practical settings, the posterior distribution is **intractable**, meaning we cannot easily compute it or find a closed form. This chapter explores **variational inference (VI)**, a powerful framework for approximating intractable posteriors by turning the inference problem into an optimization problem.

## The Challenge of Intractable Posteriors

Recall that Bayesian inference centers on computing the posterior distribution using Bayes' rule:

$$
p(\theta \mid \mathcal{D}) = \frac{p(\mathcal{D} \mid \theta) p(\theta)}{p(\mathcal{D})} = \frac{p(\mathcal{D}, \theta)}{p(\mathcal{D})}
$$

where $\theta$ represents our model parameters (e.g., weights in a regression model) and $\mathcal{D} = \{(x_i, y_i)\}_{i=1}^n$ is our observed data. The denominator $p(\mathcal{D})$ is called the **marginal likelihood** or **evidence** of the data, and it acts as a normalizing constant that ensures the posterior integrates to 1:

$$
p(\mathcal{D}) = \int p(\mathcal{D} \mid \theta) p(\theta) \, d\theta
$$

This integral is often called the **partition function** in statistical physics contexts. The fundamental challenge is that **computing this integral is typically intractable**. There are several reasons for this intractability:

1. **High dimensionality**: In modern machine learning, $\theta$ can have millions of dimensions (think neural network weights). Numerical integration in such high-dimensional spaces is computationally infeasible.

2. **Complex integrands**: Even in lower dimensions, the product of likelihood and prior often results in complicated distributions that do not have closed-form integrals.

The same problem arises when we want to make predictions. Given a new input $x_*$, the predictive distribution requires marginalizing over all possible parameter values:

$$
p(y_* \mid x_*, \mathcal{D}) = \int p(y_* \mid x_*, \theta) p(\theta \mid \mathcal{D}) \, d\theta
$$

Even if we could somehow compute the posterior $p(\theta \mid \mathcal{D})$, this integral would also be intractable in general.

However, in some special cases, exact inference is possible. In [Bayesian Linear Regression](/garden/ml/bayesian/bayesianlearning/), we exploited the following structure:

- **Prior**: $p(w) = \mathcal{N}(w \mid m_0, S_0)$ — Gaussian over weights
- **Likelihood**: $p(y \mid X, w) = \mathcal{N}(y \mid Xw, \sigma_n^2 I)$ — Gaussian noise model

Because Gaussians are closed under conditioning and marginalization (see [Multivariate Gaussian](/garden/maths/probabilitystatistics/multivariategaussian/)), the posterior is also Gaussian:

$$
p(w \mid X, y) = \mathcal{N}(w \mid \bar{w}, A^{-1})
$$

with closed-form expressions for $\bar{w}$ and $A$. This tractability extends to [Gaussian Processes](/garden/ml/bayesian/gaussianprocesses/), which can be viewed as Bayesian linear regression in an infinite-dimensional feature space.

However, many important models break this tractability. The moment we use a non-Gaussian likelihood (e.g., for classification) or a non-Gaussian prior, exact inference typically becomes impossible. In regression, we model continuous outputs with Gaussian noise, but **classification requires discrete outputs** (class labels). The likelihood must map continuous predictions to probabilities in $[0, 1]$, which requires non-linear functions like the sigmoid and products of sigmoids don't yield Gaussians.

### Bayesian Logistic Regression

To make the discussion concrete, we'll use **Bayesian logistic regression** as our running example throughout this chapter. Logistic regression extends linear regression to binary classification by using the **logistic (sigmoid) function** to model class probabilities:

$$
\sigma(z) = \frac{1}{1 + \exp(-z)} \in (0, 1)
$$

{{< figure 
    src="/images/ml/bayesLogisticRegression.jpg"
    alt="Logistic regression uses the sigmoid function to map linear predictions to probabilities."
    caption="Logistic regression uses the sigmoid function to map linear predictions to class probabilities."
>}}

The key idea is that the linear function $w^\top x$ can take any real value, but we need a probability in $[0, 1]$. The sigmoid function $\sigma$ "squashes" this real value into the valid probability range. Given an input $x$ and weights $w$, the probability of the positive class is:

$$
p(y = 1 \mid x, w) = \sigma(w^\top x)
$$

We interpret $\sigma(w^\top x)$ as the probability that input $x$ belongs to class $y = 1$. To make a prediction, we can:
- Output the probability directly (soft prediction)
- Threshold at 0.5: predict $y = 1$ if $\sigma(w^\top x) > 0.5$, else predict $y = -1$ (hard prediction)

Note that $\sigma(z) > 0.5 \iff z > 0$, so the decision boundary is the hyperplane $w^\top x = 0$. Points on one side are classified as positive, points on the other as negative.

and by symmetry, $p(y = -1 \mid x, w) = \sigma(-w^\top x) = 1 - \sigma(w^\top x)$. For a label $y \in \{-1, +1\}$, we can write both cases compactly as:

$$
p(y \mid x, w) = \sigma(y \cdot w^\top x)
$$

If we place a Gaussian prior on the weights:

$$
w \sim \mathcal{N}(0, \sigma_p^2 I) \quad \implies \quad p(w) = \mathcal{N}(w \mid 0, \sigma_p^2 I)
$$

then the posterior becomes:

$$
p(w \mid x_{1:n}, y_{1:n}) = \frac{p(y_{1:n} \mid x_{1:n}, w) p(w)}{p(y_{1:n} \mid x_{1:n})} = \frac{1}{Z} \prod_{i=1}^n \sigma(y_i w^\top x_i) \cdot \mathcal{N}(w \mid 0, \sigma_p^2 I)
$$

The problem is immediately apparent: the product of sigmoid functions combined with a Gaussian does not yield another Gaussian. So there is **no closed-form expression** for this posterior, and the normalizing constant involves an intractable integral over all possible weight vectors:

$$
Z = p(y_{1:n} \mid x_{1:n}) = \int \left[ \prod_{i=1}^n \sigma(y_i w^\top x_i) \right] \cdot \mathcal{N}(w \mid 0, \sigma_p^2 I) \, dw
$$

## Approximate Inference

Given that exact inference is often impossible, we need approximation strategies. Two main approaches have emerged:

1. **Markov Chain Monte Carlo (MCMC)**: Construct a Markov chain whose stationary distribution is the target posterior, then draw samples from this chain. MCMC methods are asymptotically exact but can be slow to converge and difficult to diagnose. This is a topic for future study.

2. **Variational Inference (VI)**: Approximate the intractable posterior with a simpler distribution from a tractable family, chosen to be as close as possible to the true posterior. VI transforms inference into optimization.

In this chapter, we focus on variational inference. Given an intractable posterior $p(\theta \mid \mathcal{D})$, we seek a simple, tractable distribution $q_\lambda(\theta)$ from a chosen **variational family** $\mathcal{Q}$ that is as close as possible to the true posterior:

$$
p(\theta \mid \mathcal{D}) \approx q_\lambda(\theta)
$$

where $\lambda$ are the **variational parameters** that we optimize to make $q$ a good approximation.

{{< figure 
    src="/images/ml/variationalApproximation.png"
    alt="Variational inference approximates an intractable posterior with a simpler distribution."
    caption="Variational inference approximates an intractable posterior with a simpler tractable distribution."
>}}

By "close," we typically mean minimizing some measure of divergence between distributions, most commonly the **Kullback-Leibler (KL) divergence**. This transforms the inference problem of computing high-dimensional integrals into an optimization problem of finding the best parameters $\lambda$. Optimization is a well-understood problem with efficient algorithms (like stochastic gradient descent) that perform well in practice.

## Laplace Approximation

Before introducing the full variational inference framework, we discuss a simpler method of approximate inference known as the **Laplace approximation**. This method was proposed by Pierre-Simon Laplace as early as 1774 as a technique for approximating integrals. The idea is elegantly simple: use a **Gaussian approximation** of the posterior distribution, constructed from a second-order Taylor expansion around the posterior mode.

{{< figure 
    src="/images/ml/laplaceApproximation.png"
    alt="Laplace approximation fits a Gaussian around the posterior mode, i.e the maximum a posteriori (MAP) estimate."
    caption="Laplace approximation fits a Gaussian around the posterior mode, i.e the maximum a posteriori (MAP) estimate."
>}}

### Taylor Expansion

Let $\psi(\theta)$ denote the log-posterior:

$$
\psi(\theta) := \log p(\theta \mid x_{1:n}, y_{1:n})
$$

We seek to approximate this function near its maximum—the **mode** of the posterior, denoted $\hat{\theta}$. This mode is precisely the **Maximum A Posteriori (MAP) estimate**:

$$
\hat{\theta} = \arg\max_\theta p(\theta \mid x_{1:n}, y_{1:n}) = \arg\max_\theta \psi(\theta)
$$

Using a second-order Taylor expansion around $\hat{\theta}$, we approximate $\psi(\theta)$ for $\theta$ near $\hat{\theta}$:

$$
\psi(\theta) \approx \hat{\psi}(\theta) = \psi(\hat{\theta}) + (\theta - \hat{\theta})^\top \nabla \psi(\hat{\theta}) + \frac{1}{2}(\theta - \hat{\theta})^\top H_\psi(\hat{\theta})(\theta - \hat{\theta})
$$

where $H_\psi(\hat{\theta})$ is the Hessian matrix of $\psi$ evaluated at $\hat{\theta}$. Since $\hat{\theta}$ is a maximum of $\psi$, the gradient vanishes at this point:

$$
\nabla \psi(\hat{\theta}) = 0
$$

This simplifies our approximation to:

$$
\hat{\psi}(\theta) = \psi(\hat{\theta}) + \frac{1}{2}(\theta - \hat{\theta})^\top H_\psi(\hat{\theta})(\theta - \hat{\theta})
$$

Now, let's compare this to the log-density of a Gaussian distribution. For $\mathcal{N}(\theta; \hat{\theta}, \Lambda^{-1})$ where $\Lambda$ is the precision matrix:
The notation $\mathcal{N}(\theta; \mu, \Sigma)$ means we evaluate the Gaussian PDF at the point $\theta$, where $\mu$ is the mean and $\Sigma$ is the covariance. The semicolon separates the variable we're evaluating from the distribution parameters. This is equivalent to saying $\theta \sim \mathcal{N}(\mu, \Sigma)$ and then computing the density at $\theta$.

$$
\log \mathcal{N}(\theta; \hat{\theta}, \Lambda^{-1}) = -\frac{1}{2}(\theta - \hat{\theta})^\top \Lambda (\theta - \hat{\theta}) + \text{const.}
$$

Matching the quadratic forms, we see that our Taylor approximation has exactly this structure if we identify:

$$
\Lambda = -H_\psi(\hat{\theta})
$$

For this approximation to be well-defined, the precision matrix $\Lambda$ (and hence the covariance $\Lambda^{-1}$) must be symmetric and positive semi-definite. Fortunately, this is guaranteed by the properties of the Hessian at a maximum:

1. **Symmetry**: The Hessian is symmetric for twice continuously differentiable functions (by Schwarz's theorem on equality of mixed partials).

2. **Positive semi-definiteness**: At a local maximum, the Hessian is negative semi-definite (second-order optimality condition). Therefore, $-H_\psi(\hat{\theta})$ is positive semi-definite.

This gives us the **Laplace approximation**:

$$
q(\theta) := \mathcal{N}(\theta; \hat{\theta}, \Lambda^{-1}) \propto \exp(\hat{\psi}(\theta)) = \frac{1}{Z} \exp\left( \hat{\psi}(\theta) \right)
$$

where $Z$ is the normalizing constant. Thus, the Laplace approximation approximates the posterior as a Gaussian centered at the MAP estimate $\hat{\theta}$ with precision given by the negative Hessian of the log-posterior at that point:

$$
\Lambda = -H_\psi(\hat{\theta}) = -H_\theta \log p(\theta \mid x_{1:n}, y_{1:n}) \Big|_{\theta = \hat{\theta}}
$$

{{< callout type="example" title="Laplace Approximation of a Gaussian" >}}
To verify our method, consider approximating a Gaussian $p(\theta) = \mathcal{N}(\theta; \mu, \Sigma)$ using Laplace approximation. 

The mode of $p$ is at $\mu$, which we verify by setting the gradient to zero:

$$
\nabla_\theta \log p(\theta) = -\Sigma^{-1}(\theta - \mu) = 0 \iff \theta = \mu
$$

The Hessian is:

$$
H_\theta \log p(\theta) = -\Sigma^{-1}
$$

Therefore, the Laplace approximation gives precision $\Lambda = -(-\Sigma^{-1}) = \Sigma^{-1}$, so covariance $\Lambda^{-1} = \Sigma$. The Laplace approximation recovers the original Gaussian exactly!
{{< /callout >}}

### Laplace Approximation for Bayesian Logistic Regression

Let's apply the Laplace approximation to our running example. First, we need to find the MAP estimate $\hat{w}$, then compute the Hessian at that point. From Bayes' rule:

$$
\begin{align*}
\hat{w} &= \arg\max_w p(w \mid x_{1:n}, y_{1:n}) \\
&= \arg\max_w \left[ \log p(y_{1:n} \mid x_{1:n}, w) + \log p(w) \right]
\end{align*}
$$

Substituting our likelihood and prior:

$$
\hat{w} = \arg\max_w \left[ \sum_{i=1}^n \log \sigma(y_i w^\top x_i) - \frac{1}{2\sigma_p^2} \|w\|^2 \right]
$$

Since $\sigma(z) = 1/(1 + e^{-z})$, we have $\log \sigma(z) = -\log(1 + e^{-z})$. Equivalently, minimizing the negative:

$$
\hat{w} = \arg\min_w \left[ \sum_{i=1}^n \underbrace{\log(1 + \exp(-y_i w^\top x_i))}_{\text{logistic loss } L_{\log}} + \frac{1}{2\sigma_p^2} \|w\|^2 \right]
$$

This is exactly **L2-regularized logistic regression**! The regularization parameter is $\gamma = 1/(2\sigma_p^2)$. This objective is convex and can be minimized using gradient-based methods like stochastic gradient descent.

**Deriving the Gradient:**

The logistic loss is $L_{\log}(w^\top x; y) = \log(1 + \exp(-y w^\top x))$. To find the gradient with respect to $w$, we apply the chain rule. Let $z = -y w^\top x$, so the loss is $\log(1 + e^z)$:

$$
\frac{\partial L_{\log}}{\partial w} = \frac{\partial \log(1 + e^z)}{\partial z} \cdot \frac{\partial z}{\partial w} = \frac{e^z}{1 + e^z} \cdot (-yx) = \sigma(z) \cdot (-yx) = -yx \cdot \sigma(-yw^\top x)
$$

where we used that $\frac{e^z}{1+e^z} = \sigma(z)$. So the gradient is:

$$
\nabla_w L_{\log}(w^\top x; y) = -yx \cdot \sigma(-yw^\top x)
$$

This has an intuitive interpretation. Notice that $\sigma(-yw^\top x) = 1 - \sigma(yw^\top x)$ is the probability that the model assigns the *wrong* class. Let's break this down:

- **Case 1: Correct and confident** ($y = 1$ and $w^\top x \gg 0$): Then $\sigma(w^\top x) \approx 1$, so $\sigma(-w^\top x) \approx 0$. The gradient is nearly zero—no update needed.
  
- **Case 2: Correct but uncertain** ($y = 1$ and $w^\top x \approx 0$): Then $\sigma(-w^\top x) \approx 0.5$. Moderate gradient pushes weights to be more confident.
  
- **Case 3: Incorrect prediction** ($y = 1$ but $w^\top x < 0$): Then $\sigma(-w^\top x) > 0.5$. Large gradient—strong signal to correct the mistake.

The gradient magnitude reflects how "surprised" the model is by the label.

This leads to the SGD update rule (with batch size 1):

$$
w \leftarrow w(1 - 2\lambda\eta_t) + \eta_t y x \sigma(-yw^\top x)
$$

where $\eta_t$ is the learning rate at step $t$. Giving us the final MAP estimate $\hat{w}$. The last step is to compute the covariance of the Laplace approximation by evaluating the Hessian of the log-posterior at $\hat{w}$ because remember:

$$
\Lambda = -H_w \log p(w \mid x_{1:n}, y_{1:n}) \Big|_{w=\hat{w}}
$$

for readability, we define the probability of the positive class under the MAP estimate:

$$
\pi_i := p(y_i = 1 \mid x_i, \hat{w}) = \sigma(\hat{w}^\top x_i)
$$

For the log-posterior, we need the second derivative of $L_{\log}$. From the gradient $\nabla_w L_{\log} = -yx \cdot \sigma(-yw^\top x)$, we differentiate again. Using the fact that $\frac{d\sigma(z)}{dz} = \sigma(z)(1 - \sigma(z))$:

$$
H_w L_{\log}(w^\top x; y) = \frac{\partial}{\partial w}\left[-yx \cdot \sigma(-yw^\top x)\right] = -yx \cdot \sigma(-yw^\top x)(1 - \sigma(-yw^\top x)) \cdot (-yx)^\top
$$

$$
= xx^\top \cdot \sigma(-yw^\top x) \cdot \sigma(yw^\top x) = xx^\top \cdot \sigma(w^\top x)(1 - \sigma(w^\top x))
$$

where the last step uses $\sigma(-z) = 1 - \sigma(z)$ and $\sigma(z)(1-\sigma(z)) = \sigma(-z)\sigma(z)$. At the MAP estimate, this becomes:

$$
H_w L_{\log}(w^\top x_i; y_i)\big|_{w=\hat{w}} = x_i x_i^\top \cdot \pi_i(1 - \pi_i)
$$

The term $\pi_i(1 - \pi_i)$ is the **variance of a Bernoulli random variable** with probability $\pi_i$. It's maximized at $\pi_i = 0.5$ (value = 0.25) and zero at the extremes $\pi_i \in \{0, 1\}$. This means:
- Points where the model is **uncertain** ($\pi_i \approx 0.5$) contribute most to the Hessian
- Points where the model is **confident** ($\pi_i \approx 0$ or $1$) contribute little

The precision matrix for the Laplace approximation is therefore:

$$
\Lambda = -H_w \log p(w \mid x_{1:n}, y_{1:n})\Big|_{w=\hat{w}} = \sigma_p^{-2}I + \sum_{i=1}^n x_i x_i^\top \pi_i(1-\pi_i)
$$

To then express this more compactly, we can use matrix notation. Let $X$ be the $n \times d$ design matrix with rows $x_i^\top$. Define $W = \text{diag}(\pi_1(1-\pi_1), \ldots, \pi_n(1-\pi_n))$ as an $n \times n$ diagonal matrix. Then:

$$
\sum_{i=1}^n x_i x_i^\top \pi_i(1-\pi_i) = \sum_{i=1}^n \pi_i(1-\pi_i) \cdot x_i x_i^\top = X^\top W X
$$

This is because $(X^\top W X)_{jk} = \sum_{i=1}^n W_{ii} X_{ij} X_{ik} = \sum_{i=1}^n \pi_i(1-\pi_i) x_{ij} x_{ik}$, which matches the $(j,k)$ entry of $\sum_i \pi_i(1-\pi_i) x_i x_i^\top$. So:

$$
\Lambda = \sigma_p^{-2}I + X^\top \text{diag}(\pi_i(1-\pi_i)) X
$$

Recall that precision $\Lambda$ is the *inverse* of covariance: $\Sigma = \Lambda^{-1}$. Higher precision means *lower* variance (more certainty in the posterior). So:

- Points with $\pi_i \approx 0.5$ (uncertain predictions) contribute *more* to precision $\Lambda$
- More precision means *smaller* posterior variance $\Sigma = \Lambda^{-1}$
- This makes sense: uncertain points are **informative**—they tell us a lot about where the decision boundary should be, so they reduce our uncertainty about the weights

Conversely, points that are already confidently classified (far from the decision boundary) don't help us pin down the weights—they would be classified correctly for a wide range of weight values.

Importantly, computing $\Lambda$ does **not** require the intractable normalizing constant $Z$!

### Making Predictions

Now that we have the Laplace approximation $q(w) = \mathcal{N}(w; \hat{w}, \Lambda^{-1})$ to the posterior, we can use it to make predictions. Recall that in Bayesian logistic regression, the predictive distribution for a new input $x_*$ is:

$$
p(y_* \mid x_*, x_{1:n}, y_{1:n}) = \int p(y_* \mid x_*, w) p(w \mid x_{1:n}, y_{1:n}) \, dw
$$

This is still a $d$-dimensional integral over the weight space, which appears intractable. However, we can exploit a key insight: the prediction $y_*$ depends on $w$ only through the **latent value** $f_* = w^\top x_*$. We can therefore rewrite:

$$
p(y_* \mid x_*, x_{1:n}, y_{1:n}) \approx \int p(y_* \mid f_*) p(f_* \mid x_*, x_{1:n}, y_{1:n}) \, df_*
$$

Since our Laplace approximation is Gaussian, $w \sim \mathcal{N}(\hat{w}, \Lambda^{-1})$, and $f_* = w^\top x_*$ is a linear function of $w$, the distribution of $f_*$ is also Gaussian (from the properties of [Multivariate Gaussians](/garden/maths/probabilitystatistics/multivariategaussian/)):

$$
f_* \mid x_* \sim \mathcal{N}(\hat{w}^\top x_*, x_*^\top \Lambda^{-1} x_*)
$$

The prediction integral has been reduced from $d$ dimensions to just **one dimension**:

$$
p(y_* = 1 \mid x_*, x_{1:n}, y_{1:n}) \approx \int \sigma(f_*) \mathcal{N}(f_*; \hat{w}^\top x_*, x_*^\top \Lambda^{-1} x_*) \, df_*
$$

While this one-dimensional integral is still not available in closed form, it can be efficiently approximated using numerical methods like **Gauss-Hermite quadrature** or simple Monte Carlo sampling.

### Limitations

The Laplace approximation has some appealing properties: it's relatively simple to apply, especially as a "post-hoc" procedure after computing the MAP estimate. It preserves the MAP as the posterior mean and adds uncertainty quantification around it.

However, the Laplace approximation has significant limitations:

1. **Greedy Mode-seeking**: The approximation is centered on the MAP, matching only the local curvature at that point. If the posterior is asymmetric, multi-modal, or has heavy tails, the Gaussian approximation can be arbitrarily poor.

2. **Overconfidence**: By fitting a Gaussian around the mode, the approximation often underestimates the true posterior variance, leading to overconfident predictions. This is especially problematic for small datasets or complex posteriors.

3. **Precision Matrix Computation**: Computing the Hessian and its inverse can be computationally expensive in high dimensions, limiting scalability. It is also essential that the precision matrix is of how quality as small changes can change the way the approximation looks as shown below.
   
{{< figure 
    src="/images/ml/laplaceCovariance.png"
    alt="Quality of the covariance matrix in Laplace approximation can significantly affect the approximation."
    caption="Quality of the covariance matrix in Laplace approximation can significantly affect the approximation."
>}}

These limitations motivate the development of more flexible approximate inference methods—variational inference—which we develop in the following sections.


## Information Theory: Surprise and Entropy

Information theory provides the mathematical framework for quantifying uncertainty and the "information content" of observations. Originally developed by Claude Shannon in 1948 for communication systems, information theory has become fundamental to machine learning and probabilistic inference. The key insight is that uncertainty can be measured not just by probability, but by **surprise**—a logarithmic transformation of probability that has remarkably useful properties.

### Surprise (Self-Information)

At the core of information theory is the concept of **surprise** (also called **self-information**, **information content**, or **Shannon information**), which quantifies how unexpected an observation is. The **surprise** about an event with probability $u$ is defined as:

$$
S[u] := -\log u
$$

When using $\log_2$ (base 2), the unit of surprise is called a **bit**. For example, a fair coin flip ($u = 0.5$) has surprise $S[0.5] = -\log_2(0.5) = 1$ bit—we need exactly 1 bit of information to encode which outcome occurred. When using the natural logarithm $\ln$, the unit is called a **nat** (natural unit).

{{< figure 
    src="/images/ml/informationSuprise.png"
    alt="The surprise function showing how surprise increases as probability decreases."
    caption="The surprise function S[u] = -log(u). Events with low probability have high surprise, while certain events (u → 1) have zero surprise."
>}}

The key properties of surprise are immediately visible from the plot:
- **Certain events have zero surprise**: $S[1] = 0$, so if something always happens, we learn nothing when it occurs.
- **Impossible events have infinite surprise**: $\lim_{u \to 0} S[u] = \infty$, so observing something "impossible" would be infinitely surprising.
- **Rare events are more surprising**: Since $-\log$ is strictly decreasing, lower probability means higher surprise.

But why is $-\log u$ a reasonable measure of surprise? The following axiomatic characterization shows that surprise is essentially uniquely defined by three intuitive properties:

1. **Anti-monotonicity**: $S[u] > S[v] \iff u < v$ — we are more surprised by unlikely events.
2. **Continuity**: $S$ is continuous — no jumps in surprise for infinitesimal changes in probability.
3. **Additivity for independent events**: $S[uv] = S[u] + S[v]$ — the surprise of independent events is additive.

### Entropy

The **entropy** of a distribution $p$ is the **expected surprise** about samples from $p$. In this way, entropy quantifies the inherent uncertainty associated with a distribution: if the entropy is large, we are more uncertain about the outcome $x \sim p$ than if the entropy were low. Formally:

$$
H[p] := \mathbb{E}_{x \sim p}[S[p(x)]] = \mathbb{E}_{x \sim p}[-\log p(x)]
$$

When $X \sim p$ is a random variable distributed according to $p$, we write $H[X] := H[p]$. For discrete and continuous distributions:

$$
H[p] = -\sum_x p(x) \log p(x) \quad \text{(discrete)}
$$

$$
H[p] = -\int p(x) \log p(x) \, dx \quad \text{(continuous)}
$$

Entropy has a beautiful interpretation: it measures the **average number of bits** (when using $\log_2$) required to encode outcomes from the distribution. Equivalently, it quantifies how "spread out" or "uncertain" a distribution is.

{{< figure 
    src="/images/ml/informationEntropy.png"
    alt="Visualization of entropy for different distributions."
    caption="Entropy measures the spread of a distribution. A peaky distribution (low entropy) is more predictable than a spread-out distribution (high entropy)."
>}}

From the image and definition, we can see several important properties of entropy:
- **Peaky distributions have low entropy**: If most of the probability mass is concentrated on a few outcomes, we can predict the outcome well, so there is little uncertainty.
- **Spread-out distributions have high entropy**: If the probability is spread evenly across many outcomes, we are very uncertain about what we will observe.
- **The uniform distribution maximizes entropy**: Among all distributions on a finite set, the uniform distribution has the highest entropy as it represents maximum uncertainty.

{{< callout type="warning" title="Entropy vs Variance" >}}
While entropy and variance both measure "spread," they are fundamentally different quantities. Variance measures the expected squared deviation from the mean and depends on the scale of the values. Entropy measures uncertainty in the probabilistic sense and is invariant to relabeling of outcomes. Two distributions can have the same variance but very different entropies, and vice versa.
{{< /callout >}}

{{< callout type="example" title="Coins" >}}
For a fair coin modeled as a Bernoulli distribution with parameter $p=0.5$:

$$
H[\text{Bern}(0.5)] = -2(0.5 \log_2 0.5) = -2(0.5 \cdot -1) = 1 \text{ bit}
$$

This means on average, each flip of a fair coin provides 1 bit of information. This makes sense because a fair coin has two equally likely outcomes (heads or tails), and we need exactly 1 bit to distinguish between them.

On the other hand, an unfair coin that lands heads 10% of the time has lower entropy:

$$
H[\text{Bern}(0.1)] = -0.1 \log_2 0.1 - 0.9 \log_2 0.9 \approx 0.469 \text{ bits}
$$

This reflects the fact that we are less uncertain about the outcome of a biased coin flip, we can "almost always" predict tails, so each flip provides less information on average.
{{< /callout >}}
{{< callout type="example" title="Uniform Distribution" >}}
For a discrete uniform distribution over $n$ outcomes we have:

$$
H[\text{Unif}(\{1, \ldots, n\})] = -\sum_{i=1}^n \frac{1}{n} \log_2 \frac{1}{n} = \log_2 n
$$

The uniform distribution has maximum entropy among all discrete distributions on $\{1, \ldots, n\}$. Notably, $\log_2 n$ is exactly the number of bits required to encode one of $n$ equally likely outcomes! For instance, with $n = 8$ outcomes, we need $\log_2 8 = 3$ bits.
{{< /callout >}}
{{< callout type="example" title="Gaussian Distribution" >}}
For a univariate Gaussian $\mathcal{N}(x; \mu, \sigma^2)$, we can derive the entropy explicitly. Recall the PDF:

$$
\mathcal{N}(x; \mu, \sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)
$$

Using the definition of entropy:

$$
\begin{align*}
H[\mathcal{N}(\mu, \sigma^2)] &= -\int \mathcal{N}(x; \mu, \sigma^2) \log \mathcal{N}(x; \mu, \sigma^2) \, dx \\
&= -\int \mathcal{N}(x; \mu, \sigma^2) \left[ -\log\sqrt{2\pi\sigma^2} - \frac{(x-\mu)^2}{2\sigma^2} \right] dx \\
&= \log\sqrt{2\pi\sigma^2} + \frac{1}{2\sigma^2} \mathbb{E}[(x-\mu)^2] \\
&= \log(\sigma\sqrt{2\pi}) + \frac{1}{2} \\
&= \log(\sigma\sqrt{2\pi e})
\end{align*}
$$

The entropy depends on $\sigma$ (the standard deviation) but not on $\mu$ (the mean). This makes sense: shifting a distribution doesn't change its uncertainty, but spreading it out (increasing $\sigma$) does. Specifically, entropy increases logarithmically with $\sigma$, so doubling the standard deviation adds $\log 2$ nats of entropy.
{{< /callout >}}
{{< callout type="example" title="Multivariate Gaussian Distribution" >}}
For a multivariate Gaussian $\mathcal{N}(\mu, \Sigma)$ in $d$ dimensions:

$$
H[\mathcal{N}(\mu, \Sigma)] = \frac{1}{2} \log \det(2\pi e \cdot \Sigma) = \frac{1}{2} \log\left((2\pi e)^d \det \Sigma\right)
$$

Note that entropy depends on $\Sigma$ only through its determinant. The determinant $\det \Sigma$ is also called the **generalized variance** and measures the "volume" of the distribution's credible sets, the hyperellipsoids containing most of the probability mass. A larger determinant means a larger "spread" in parameter space and hence higher uncertainty.
{{< /callout >}}

### Cross-Entropy

What is our average surprise when we assume data follows distribution $q$ but it actually follows $p$? This is captured by the concept of **cross-entropy**. The **cross-entropy** of a distribution $q$ relative to the distribution $p$ is:

$$
H[p \| q] := \mathbb{E}_{x \sim p}[S[q(x)]] = \mathbb{E}_{x \sim p}[-\log q(x)]
$$

Cross-entropy measures our average surprise when using the "wrong" distribution $q$ to model data that actually comes from the "true" distribution $p$. Intuitively, the closer $q$ is to $p$, the smaller the cross-entropy. This is why cross-entropy is commonly used as a loss function in machine learning: **minimizing cross-entropy encourages the model distribution $q$ to match the true data distribution $p$**.

The fundamental relationship between cross-entropy, entropy, and KL divergence is:

$$
H[p \| q] = H[p] + \text{KL}(p \| q)
$$

This decomposition has a beautiful interpretation:
- $H[p]$ is the **inherent uncertainty** in $p$—the minimum average surprise we would experience even if we knew the true distribution perfectly.
- $\text{KL}(p \| q)$ is the **additional surprise** due to using the wrong distribution $q$ instead of the true $p$.

Since $\text{KL}(p \| q) \geq 0$ (as we'll prove shortly), it follows that $H[p \| q] \geq H[p]$: we can never be *less* surprised on average by using a different distribution than the true one.

### Kullback-Leibler Divergence

The **Kullback-Leibler divergence** (or **KL divergence**, also called **relative entropy**) is the fundamental measure of "distance" between probability distributions in variational inference. It directly quantifies the "cost" of using an approximation $q$ instead of the true distribution $p$.

Given two distributions $p$ and $q$, the **KL divergence** of $q$ with respect to $p$ is:

$$
\text{KL}(p \| q) := H[p \| q] - H[p] = \mathbb{E}_{\theta \sim p}\left[\log \frac{p(\theta)}{q(\theta)}\right] = \int p(\theta) \log \frac{p(\theta)}{q(\theta)} \, d\theta
$$

In words, $\text{KL}(p \| q)$ measures the *additional* expected surprise when observing samples from $p$ that is due to assuming the wrong distribution $q$, beyond the inherent surprise in $p$ itself.

The KL divergence has several important properties:

1. **Non-negativity (Gibbs' inequality)**: $\text{KL}(p \| q) \geq 0$ for any distributions $p$ and $q$ because using the wrong distribution can never reduce surprise.
2. **Identity of indiscernibles**: $\text{KL}(p \| q) = 0$ if and only if $p = q$ almost surely, meaning the two distributions are identical.
3. **Asymmetry**: In general, $\text{KL}(p \| q) \neq \text{KL}(q \| p)$ so the KL divergence is **not** a true distance metric. 

The non-negativity follows from Jensen's inequality applied to the convex function $-\log$:

$$
\text{KL}(p \| q) = \mathbb{E}_{p}\left[-\log \frac{q(\theta)}{p(\theta)}\right] \geq -\log \mathbb{E}_{p}\left[\frac{q(\theta)}{p(\theta)}\right] = -\log \int q(\theta) d\theta = -\log(1) = 0
$$

The inequality is tight (equality holds) only when $q(\theta)/p(\theta)$ is constant, which requires $p = q$.

**Interpretation via likelihood ratios:** There is another illuminating interpretation of KL divergence. Suppose we are presented with a sequence $\theta_1, \ldots, \theta_n$ of independent samples from either distribution $p$ or distribution $q$, and we want to determine which one generated the data. A natural approach is to compare the likelihoods and choose $p$ if:

$$
\sum_{i=1}^n \log \frac{p(\theta_i)}{q(\theta_i)} > 0
$$

By the law of large numbers, if the true distribution is $p$, then:

$$
\frac{1}{n}\sum_{i=1}^n \log \frac{p(\theta_i)}{q(\theta_i)} \xrightarrow{\text{a.s.}} \mathbb{E}_{\theta \sim p}\left[\log\frac{p(\theta)}{q(\theta)}\right] = \text{KL}(p \| q)
$$

So $\text{KL}(p \| q)$ measures how quickly we can distinguish between $p$ and $q$: the larger the KL divergence, the easier it is to tell them apart from samples.

The asymmetry of KL divergence follows from the fact that the expectation is taken under $p$ in $\text{KL}(p \| q)$ but under $q$ in $\text{KL}(q \| p)$. This leads to very different behaviors when minimizing KL divergence in either direction:

- **Forward (inclusive) KL $\text{KL}(p \| q)$**: Expectation taken under the *true* distribution $p$. The approximation $q$ must have probability mass wherever $p$ does (to avoid $\log(q) \to -\infty$ and hence infinite KL). This is **mode-covering** or **mean-seeking**: $q$ tends to "spread out" to cover all modes of $p$, often overestimating variance.

- **Reverse (exclusive) KL $\text{KL}(q \| p)$**: Expectation taken under the *approximate* distribution $q$. Since the expectation is over $q$, regions where $q(\theta) = 0$ contribute nothing to the integral, even if $p(\theta) > 0$ there. This is **mode-seeking**: $q$ tends to concentrate on a single mode of $p$, often underestimating variance.

{{< figure 
    src="/images/ml/informationForwardReverseKL.jpg"
    alt="Comparison of forward and reverse KL divergence"
    caption="Forward KL tends to cover all modes but may over-estimate variance. Reverse KL tends to focus on a single mode but may under-estimate variance."
>}}

In variational inference, we use **reverse KL** for a crucial practical reason: computing $\text{KL}(p \| q)$ requires sampling from the intractable posterior $p$, which is exactly what we're trying to avoid! With reverse KL, we sample from $q$, which is tractable by construction. 

However, this choice has consequences: our approximations will be mode-seeking and may underestimate uncertainty, giving **overconfident** predictions. By focusing on a single mode and underestimating variance, the variational posterior may be too "peaked" compared to the true posterior. That said, reverse KL is not as greedy as Laplace approximation—it still optimizes over the entire distribution rather than just matching local curvature at the mode.

Crucially, if the true posterior $p(\cdot \mid \mathcal{D})$ is in the variational family $\mathcal{Q}$, then minimizing reverse KL recovers the true posterior exactly:

$$
\arg\min_{q \in \mathcal{Q}} \text{KL}(q \| p(\cdot \mid \mathcal{D})) = p(\cdot \mid \mathcal{D})
$$

To see why, suppose $p(\cdot \mid \mathcal{D}) \in \mathcal{Q}$. Then we can choose $q = p(\cdot \mid \mathcal{D})$, which gives:

$$
\text{KL}(p(\cdot \mid \mathcal{D}) \| p(\cdot \mid \mathcal{D})) = \mathbb{E}_{\theta \sim p(\cdot \mid \mathcal{D})}\left[\log \frac{p(\theta \mid \mathcal{D})}{p(\theta \mid \mathcal{D})}\right] = \mathbb{E}_{\theta \sim p(\cdot \mid \mathcal{D})}[\log 1] = 0
$$

Since KL divergence is non-negative and achieves its minimum of zero only when the two distributions are identical, no other choice of $q \in \mathcal{Q}$ can achieve a lower KL divergence. Therefore, the minimizer is exactly the true posterior.

{{< callout type="example" title="KL Divergence Between Gaussians" >}}
For two general Gaussians $p = \mathcal{N}(\mu_p, \Sigma_p)$ and $q = \mathcal{N}(\mu_q, \Sigma_q)$ in $d$ dimensions the KL divergence has a closed-form expression:

$$
\text{KL}(p \| q) = \frac{1}{2}\left( \text{tr}(\Sigma_q^{-1}\Sigma_p) + (\mu_p - \mu_q)^\top \Sigma_q^{-1}(\mu_p - \mu_q) - d + \log\frac{\det\Sigma_q}{\det\Sigma_p} \right)
$$

For unit variance Gaussians ($\Sigma_p = \Sigma_q = I$) this simplifies to:

$$
\text{KL}(p \| q) = \frac{1}{2} \|\mu_q - \mu_p\|_2^2
$$

For a diagonal Gaussian $p = \mathcal{N}(\mu, \text{diag}\{\sigma_i^2\})$ approximated by a standard normal $q = \mathcal{N}(0, I)$ we have:

$$
\text{KL}(p \| q) = \frac{1}{2} \sum_{i=1}^d \left( \sigma_i^2 + \mu_i^2 - 1 - \log \sigma_i^2 \right)
$$

This expression has intuitive terms: $\mu_i^2$ penalizes large means, $\sigma_i^2$ penalizes large variance, and $-\log\sigma_i^2$ penalizes small variance.
{{< /callout >}}

## Variational Inference

Having established the Laplace approximation as a first approach to approximate inference, we now develop a more principled framework. **Variational inference** provides a systematic way to approximate intractable posteriors by viewing inference as an optimization problem. The fundamental idea is to define a **variational family** of tractable distributions and find the member of this family that is "closest" to the true posterior. A **variational family** $\mathcal{Q} \subset \mathcal{P}$ is a class of probability distributions such that each distribution $q \in \mathcal{Q}$ is characterized by unique **variational parameters** $\lambda$. For example, the family of independent (diagonal covariance) Gaussians:

$$
\mathcal{Q} = \left\{ q(\theta) = \mathcal{N}\left(\theta; \mu, \text{diag}_{i \in [d]}\{\sigma_i^2\}\right) \right\}
$$

is parameterized by $\lambda := [\mu_{1:d}, \sigma^2_{1:d}]$ so we only need to optimize $2d$ parameters instead of the full $d \times d$ covariance matrix.

To measure "closeness" between distributions, we use the **Kullback-Leibler (KL) divergence**, leading to the optimization problem:

$$
q^* = \arg\min_{q \in \mathcal{Q}} \text{KL}(q \| p(\cdot \mid \mathcal{D})) = \arg\min_{\lambda} \text{KL}(q_\lambda \| p(\cdot \mid \mathcal{D}))
$$

However, directly minimizing the KL divergence is challenging because it requires knowledge of the true posterior $p(\theta \mid \mathcal{D})$, which is intractable. To overcome this, we derive an alternative objective function known as the **Evidence Lower Bound (ELBO)**, which we can maximize instead. 

### Deriving the ELBO

We start from the definition of the reverse KL divergence and manipulate it to isolate a tractable lower bound on the log model evidence $\log p(\mathcal{D})$. Starting from the reverse KL divergence between our variational distribution $q$ and the true posterior:

$$
\begin{align*}
\text{KL}(q \| p(\cdot \mid \mathcal{D})) &= \mathbb{E}_{\theta \sim q}\left[\log \frac{q(\theta)}{p(\theta \mid \mathcal{D})}\right] \\
&= \mathbb{E}_{\theta \sim q}\left[\log \frac{q(\theta) \cdot p(\mathcal{D})}{p(\theta, \mathcal{D})}\right] \quad \text{(using Bayes' rule: } p(\theta \mid \mathcal{D}) = \frac{p(\theta, \mathcal{D})}{p(\mathcal{D})}\text{)} \\
&= \log p(\mathcal{D}) + \mathbb{E}_{\theta \sim q}\left[\log \frac{q(\theta)}{p(\theta, \mathcal{D})}\right]
\end{align*}
$$

Note that $\log p(\mathcal{D})$ comes out of the expectation because it does not depend on $\theta$. Recalling that the entropy of $q$ is defined as $H[q] = -\mathbb{E}_{\theta \sim q}[\log q(\theta)]$, we can rewrite:

$$
\text{KL}(q \| p(\cdot \mid \mathcal{D})) = \log p(\mathcal{D}) - \mathbb{E}_{\theta \sim q}[\log p(\theta, \mathcal{D})] - H[q]
$$

After rearranging, we obtain the key decomposition:

$$
\log p(\mathcal{D}) = \text{KL}(q \| p(\cdot \mid \mathcal{D})) + \underbrace{\mathbb{E}_{\theta \sim q}[\log p(\theta, \mathcal{D})] + H[q]}_{\mathcal{L}(q; \mathcal{D})}
$$

where the term $\mathcal{L}(q; \mathcal{D})$ is the **Evidence Lower Bound (ELBO)**. Since $\text{KL}(q \| p) \geq 0$, we have:

$$
\log p(\mathcal{D}) \geq \mathcal{L}(q; \mathcal{D}) = \mathbb{E}_{\theta \sim q}[\log p(\theta, \mathcal{D})] + H[q]
$$

This inequality is the reason for the name "evidence **lower** bound", the ELBO is always less than or equal to the log evidence of the data $\log p(\mathcal{D})$. The reason that maximizing the ELBO minimizes reverse KL is that the log evidence $\log p(\mathcal{D})$ is constant with respect to $q$ as it depends only on the data and model, not on our choice of variational distribution. From our decomposition:

$$
\log p(\mathcal{D}) = \text{KL}(q \| p(\cdot \mid \mathcal{D})) + \mathcal{L}(q; \mathcal{D})
$$

Since the left-hand side is constant, increasing $\mathcal{L}(q; \mathcal{D})$ must decrease $\text{KL}(q \| p(\cdot \mid \mathcal{D}))$ by exactly the same amount. Therefore:

$$
\arg\max_{q} \mathcal{L}(q; \mathcal{D}) = \arg\min_{q} \text{KL}(q \| p(\cdot \mid \mathcal{D}))
$$

The ELBO can also be written in an alternative form by expanding the joint distribution $p(\theta, \mathcal{D}) = p(\mathcal{D} \mid \theta) p(\theta)$:

$$
\mathcal{L}(q; \mathcal{D}) = \mathbb{E}_{\theta \sim q}[\log p(\mathcal{D} \mid \theta)] - \text{KL}(q \| p(\theta))
$$

This form is particularly illuminating as it reveals that maximizing the ELBO seeks a variational distribution $q$ that:
1. **Explains the data well**: high expected log-likelihood $\mathbb{E}_{\theta \sim q}[\log p(\mathcal{D} \mid \theta)]$
2. **Stays close to the prior**: low KL divergence $\text{KL}(q \| p(\theta))$ to the prior

This is the Bayesian principle of balancing data fit with prior beliefs, expressed as an optimization objective! It also provides a natural regularization: the prior prevents $q$ from overfitting to the data by penalizing distributions that deviate too far from our prior beliefs.

### ELBO via Jensen's Inequality

The ELBO can also be derived directly using Jensen's inequality, which provides additional insight. Starting from the log evidence:

$$
\begin{align*}
\log p(\mathcal{D}) &= \log \int p(\mathcal{D}, \theta) \, d\theta \\
&= \log \int \frac{p(\mathcal{D}, \theta)}{q(\theta)} q(\theta) \, d\theta \\
&= \log \mathbb{E}_{\theta \sim q}\left[\frac{p(\mathcal{D}, \theta)}{q(\theta)}\right] \\
&\geq \mathbb{E}_{\theta \sim q}\left[\log \frac{p(\mathcal{D}, \theta)}{q(\theta)}\right] \quad \text{(Jensen's inequality, since $\log$ is concave)} \\
&= \mathcal{L}(q; \mathcal{D})
\end{align*}
$$

This derivation shows more directly why we call it the Evidence Lower Bound as it finds the lower bound starting with the log evidence obtained by applying Jensen's inequality. The bound is tight (equality holds) when the ratio $p(\mathcal{D}, \theta)/q(\theta)$ is constant in $\theta$. This occurs when $q(\theta) \propto p(\mathcal{D}, \theta) = p(\theta \mid \mathcal{D}) \cdot p(\mathcal{D})$, i.e., when $q$ equals the true posterior.

### Variational Inference vs. Laplace Approximation

Both Laplace approximation and Gaussian variational inference produce Gaussian approximations to the posterior, but they construct these approximations in fundamentally different ways. Understanding this difference illuminates why variational inference is more robust.

**Laplace approximation** is a *local* method: it fits the Gaussian by matching the posterior's behavior exactly at the MAP estimate $\hat{\theta}$. The mean is set to the mode, and the covariance is determined by the local curvature:

$$
\mu = \hat{\theta}, \quad \Sigma^{-1} = -\nabla_\theta^2 \log p(\theta, \mathcal{D})\big|_{\theta=\hat{\theta}}
$$

The conditions being matched are:
- The gradient of the log-posterior is zero at $\hat{\theta}$ (first-order condition for the mode)
- The precision equals the negative Hessian at $\hat{\theta}$ (second-order curvature matching)

**Gaussian variational inference** is a *global* method: it finds the Gaussian that best approximates the posterior across its entire support by maximizing the ELBO. It can be shown that the optimal Gaussian variational approximation satisfies:

$$
\mathbb{E}_{\theta \sim q}[\nabla_\theta \log p(\theta, \mathcal{D})] = 0, \quad \Sigma^{-1} = -\mathbb{E}_{\theta \sim q}[\nabla_\theta^2 \log p(\theta, \mathcal{D})]
$$

The key difference is the **averaging**: instead of matching conditions at a single point, VI matches them *in expectation* over the variational distribution $q$. This has the consequence that if the posterior is asymmetric, the VI mean will be pulled toward regions of higher probability mass, not just the mode. 

This averaging property makes variational inference less prone to the overconfidence that plagues Laplace approximation, particularly when the posterior is skewed or has heavier tails than a Gaussian.

{{< figure 
    src="/images/ml/variationalVsLaplace.png"
    alt="Comparison of variational inference and Laplace approximation"
    caption="Variational inference (blue) optimizes globally to cover the posterior, while Laplace approximation (red) fits locally around the mode."
>}}

### Gradient of the ELBO

To optimize the ELBO using gradient-based methods, we need to compute:

$$
\nabla_\lambda \mathcal{L}(q_\lambda; \mathcal{D}) = \nabla_\lambda \mathbb{E}_{\theta \sim q_\lambda}[\log p(\mathcal{D} \mid \theta)] - \nabla_\lambda \text{KL}(q_\lambda \| p(\theta))
$$

where $\lambda$ are the variational parameters of $q_\lambda$. The KL term often has closed-form gradients for standard distributions (e.g., Gaussians). The challenge is the first term: **we cannot simply move the gradient inside the expectation** because the distribution $q_\lambda$ over which we're taking the expectation depends on $\lambda$.

Two main approaches address this:

1. **Score function estimator** (also called REINFORCE or log-derivative trick): Rewrite the gradient using the identity $\nabla_\lambda q_\lambda(\theta) = q_\lambda(\theta) \nabla_\lambda \log q_\lambda(\theta)$ and estimate via Monte Carlo. This approach is general but often has high variance.

2. **Reparameterization trick**: Express $\theta$ as a deterministic function of $\lambda$ and a parameter-free random variable. This approach has lower variance and is the standard in modern variational inference.

We focus on the reparameterization trick. The key insight is to express samples from $q_\lambda(\theta)$ as a deterministic transformation of samples from a fixed, parameter-free distribution. Let $\varepsilon \sim \phi$ be a random variable independent of $\lambda$, and let $g: \mathbb{R}^d \to \mathbb{R}^d$ be a differentiable, invertible function. We then define:

$$
\theta := g(\varepsilon; \lambda)
$$

which induces a distribution over $\theta$ parameterized by $\lambda$. Using the change of variables formula, the density of $\theta$ is:

$$
q_\lambda(\theta) = \phi(\varepsilon) \cdot \left|\det\left(\frac{\partial g}{\partial \varepsilon}\right)\right|^{-1}
$$

The Jacobian determinant $|\det(\partial g / \partial \varepsilon)|$ accounts for how the transformation stretches or compresses probability mass. Crucially, we can rewrite expectations over $q_\lambda$ in terms of expectations over the fixed distribution $\phi$:

$$
\mathbb{E}_{\theta \sim q_\lambda}[f(\theta)] = \mathbb{E}_{\varepsilon \sim \phi}[f(g(\varepsilon; \lambda))]
$$

Now we can differentiate through the expectation:

$$
\nabla_\lambda \mathbb{E}_{\theta \sim q_\lambda}[f(\theta)] = \nabla_\lambda \mathbb{E}_{\varepsilon \sim \phi}[f(g(\varepsilon; \lambda))] = \mathbb{E}_{\varepsilon \sim \phi}[\nabla_\lambda f(g(\varepsilon; \lambda))]
$$

The gradient moves inside because the distribution $\phi$ does not depend on $\lambda$! This is the essence of the reparameterization trick.

{{< callout type="example" title="Reparameterization for Gaussians" >}}
For a Gaussian variational distribution $q_\lambda(\theta) = \mathcal{N}(\theta; \mu, \Sigma)$, the reparameterization is elegant. From the properties of Gaussians (see [Multivariate Gaussian](/garden/maths/probabilitystatistics/multivariategaussian/)), we can write:

$$
\theta = g(\varepsilon; \mu, C) := \mu + C\varepsilon, \quad \text{where } \varepsilon \sim \mathcal{N}(0, I)
$$

Here $C = \Sigma^{1/2}$ is the matrix square root (or Cholesky factor) of the covariance, so $\Sigma = CC^\top$.  The variational parameters are $\lambda = (\mu, C)$. This transformation maps standard normal samples $\varepsilon$ to samples from the desired Gaussian $q_\lambda(\theta)$. For **diagonal gaussians** with $\Sigma = \text{diag}\{\sigma_i^2\}$, this simplifies even further to:

$$
\theta_i = \mu_i + \sigma_i \varepsilon_i, \quad \varepsilon_i \sim \mathcal{N}(0, 1)
$$
{{< /callout >}}

### Reparameterized Gradient of the ELBO

Putting it all together, if we use Gaussian variational distributions with the reparameterization $\theta = C\varepsilon + \mu$, we can express the gradient of the expected log-likelihood term as:

$$
\begin{align*}
\nabla_\lambda \mathbb{E}_{\theta \sim q_\lambda}[\log p(\mathcal{D} \mid \theta)] &= \nabla_{C,\mu} \mathbb{E}_{\varepsilon \sim \mathcal{N}(0,I)}[\log p(\mathcal{D} \mid \theta)]_{\theta = C\varepsilon + \mu} \\
&= \mathbb{E}_{\varepsilon \sim \mathcal{N}(0,I)}[\nabla_{C,\mu} \log p(\mathcal{D} \mid \theta)]_{\theta = C\varepsilon + \mu}
\end{align*}
$$

This expectation can be approximated via Monte Carlo sampling:

$$
\approx \frac{1}{m} \sum_{j=1}^m \nabla_{C,\mu} \log p(\mathcal{D} \mid \theta)\Big|_{\theta = C\varepsilon_j + \mu}, \quad \varepsilon_j \stackrel{\text{iid}}{\sim} \mathcal{N}(0, I)
$$

Combined with the closed-form KL gradient, this gives us an unbiased estimator of $\nabla_\lambda \mathcal{L}$ that we can use with stochastic gradient descent. This procedure of approximating the true posterior using a variational posterior by maximizing the ELBO with stochastic optimization is called **black-box stochastic variational inference**.

### ELBO for Bayesian Logistic Regression

Let's apply our framework to our running example of Bayesian logistic regression. Recall the model setup:
- **Prior**: $w \sim \mathcal{N}(0, I)$ (using $\sigma_p = 1$ for simplicity)
- **Likelihood**: $p(y_{1:n} \mid x_{1:n}, w) = \prod_{i=1}^n \sigma(y_i w^\top x_i)$
- **Variational family**: Diagonal Gaussians $q_\lambda(w) = \mathcal{N}(w; \mu, \text{diag}\{\sigma_i^2\})$

The variational parameters are $\lambda = (\mu_1, \ldots, \mu_d, \sigma_1, \ldots, \sigma_d)$—a total of $2d$ parameters.


First, we compute the KL divergence between the variational distribution and the prior. Both are Gaussians, so we can use the closed-form expression from earlier. For a diagonal Gaussian $q = \mathcal{N}(\mu, \text{diag}\{\sigma_i^2\})$ and a standard normal prior $p = \mathcal{N}(0, I)$:

$$
\begin{align*}
\text{KL}(q_\lambda \| p(w)) &= \frac{1}{2}\left( \text{tr}(I^{-1} \cdot \text{diag}\{\sigma_i^2\}) + \mu^\top I^{-1} \mu - d + \log\frac{\det I}{\det(\text{diag}\{\sigma_i^2\})} \right) \\
&= \frac{1}{2}\left( \sum_{i=1}^d \sigma_i^2 + \sum_{i=1}^d \mu_i^2 - d + \log\frac{1}{\prod_{i=1}^d \sigma_i^2} \right) \\
&= \frac{1}{2}\left( \sum_{i=1}^d \sigma_i^2 + \sum_{i=1}^d \mu_i^2 - d - \sum_{i=1}^d \log \sigma_i^2 \right) \\
&= \frac{1}{2} \sum_{i=1}^d \left( \sigma_i^2 + \mu_i^2 - 1 - \log \sigma_i^2 \right)
\end{align*}
$$

This closed-form expression is easy to differentiate with respect to the variational parameters $\mu_i$ and $\sigma_i$. The expected log-likelihood under the variational distribution is:

$$
\begin{align*}
\mathbb{E}_{w \sim q_\lambda}[\log p(y_{1:n} \mid x_{1:n}, w)] &= \mathbb{E}_{w \sim q_\lambda}\left[\sum_{i=1}^n \log \sigma(y_i w^\top x_i)\right] \\
&= \mathbb{E}_{w \sim q_\lambda}\left[-\sum_{i=1}^n \log(1 + \exp(-y_i w^\top x_i))\right] \\
&= \mathbb{E}_{w \sim q_\lambda}\left[-\sum_{i=1}^n L_{\log}(w^\top x_i; y_i)\right]
\end{align*}
$$

where $L_{\log}(z; y) = \log(1 + \exp(-yz))$ is the logistic loss we encountered earlier. Unlike the KL term, this expectation does **not** have a closed form because of the non-linear logistic loss function. We must approximate it using Monte Carlo sampling.
Recall the general ELBO form:

$$
\mathcal{L}(q; \mathcal{D}) = \mathbb{E}_{\theta \sim q}[\log p(\mathcal{D} \mid \theta)] - \text{KL}(q \| p(\theta))
$$

Substituting our expressions for Bayesian logistic regression:

$$
\mathcal{L}(q_\lambda; \mathcal{D}) = \mathbb{E}_{w \sim q_\lambda}\left[-\sum_{i=1}^n L_{\log}(w^\top x_i; y_i)\right] - \frac{1}{2} \sum_{j=1}^d \left( \sigma_j^2 + \mu_j^2 - 1 - \log \sigma_j^2 \right)
$$

To optimize this, we use the reparameterization trick. With $w = \mu + \text{diag}(\sigma_1, \ldots, \sigma_d) \cdot \varepsilon$ where $\varepsilon \sim \mathcal{N}(0, I)$:

$$
\mathcal{L}(q_\lambda; \mathcal{D}) = \mathbb{E}_{\varepsilon \sim \mathcal{N}(0,I)}\left[-\sum_{i=1}^n L_{\log}((\mu + \text{diag}(\sigma) \cdot \varepsilon)^\top x_i; y_i)\right] - \frac{1}{2} \sum_{j=1}^d \left( \sigma_j^2 + \mu_j^2 - 1 - \log \sigma_j^2 \right)
$$

This can be optimized using stochastic gradient descent with:
- **Mini-batches over data points**: Instead of summing over all $n$ data points, sample a random subset.
- **Monte Carlo samples**: Sample $\varepsilon_1, \ldots, \varepsilon_m \stackrel{\text{iid}}{\sim} \mathcal{N}(0, I)$ to estimate the expectation.

The resulting algorithm is simple: at each iteration, sample a mini-batch of data and a few noise samples, compute the gradient of the ELBO estimate, and update the variational parameters.
