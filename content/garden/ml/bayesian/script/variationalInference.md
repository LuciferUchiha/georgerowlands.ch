
# Variational Inference

We have seen how to perform (efficient) probabilistic inference with Gaussians, exploiting their closed-form formulas for marginal and conditional distributions. But what if we work with other distributions? In this and the following chapter, we will discuss two methods of approximate inference. We begin by discussing variational (probabilistic) inference, which aims to find a good approximation of the posterior distribution from which it is easy to sample. In Chapter 6, we discuss Markov chain Monte Carlo methods, which approximate the sampling from the posterior distribution directly.

The fundamental idea behind variational inference is to approximate the true posterior distribution using a “simpler” posterior that is as close as possible to the true posterior:

[
p(\theta \mid x_{1:n}, y_{1:n}) = \frac{1}{Z} p(\theta, y_{1:n} \mid x_{1:n}) ;;\Longleftrightarrow;; q(\theta \mid \varrho)
]

[
= q_{\varrho}(\theta)
\tag{5.1}
]

where (\varrho) represents the parameters of the variational posterior (q_{\varrho}), also called variational parameters. In doing so, variational inference reduces probabilistic inference — where the fundamental difficulty lies in solving high-dimensional integrals — to an optimization problem. Optimizing (stochastic) objectives is a well-understood problem with efficient algorithms that perform well in practice.¹

¹ We provide an overview of first-order methods such as stochastic gradient descent in Appendix A.4.

---

## 5.1 Laplace Approximation

Before introducing a general framework of variational inference, we discuss a simpler method of approximate inference known as Laplace’s method. This method was proposed as a method of approximating integrals as early as 1774 by Pierre-Simon Laplace. The idea is to use a Gaussian approximation (that is, a second-order Taylor approximation) of the posterior distribution around its mode. Let

[
\psi(\theta) ;:=; \log p(\theta \mid x_{1:n}, y_{1:n})
\tag{5.2}
]

denote the log-posterior. Then, using a second-order Taylor approximation (A.53) around the mode (\hat{\theta}) of (\psi) (i.e., the MAP estimate), we obtain the approximation (\hat{\psi}) which is accurate for (\theta \approx \hat{\theta}):

[
\psi(\theta) ;\approx; \hat{\psi}(\theta)
]

[
= \psi(\hat{\theta}) + (\theta - \hat{\theta})^\top \nabla \psi(\hat{\theta})

* \frac{1}{2} (\theta - \hat{\theta})^\top H_\psi(\hat{\theta})(\theta - \hat{\theta})
  ]

[
= \psi(\hat{\theta}) + \frac{1}{2} (\theta - \hat{\theta})^\top H_\psi(\hat{\theta})(\theta - \hat{\theta}),
\quad \text{using } \nabla \psi(\hat{\theta}) = 0
\tag{5.3}
]

Compare this expression to the log-PDF of a Gaussian:

[
\log \mathcal{N}(\theta; \hat{\theta}, \Lambda^{-1})
= -\frac{1}{2} (\theta - \hat{\theta})^\top \Lambda (\theta - \hat{\theta}) + \text{const.}
\tag{5.4}
]

Since (\psi(\hat{\theta})) is constant with respect to (\theta),

[
\hat{\psi}(\theta)
= \log \mathcal{N}(\theta; \hat{\theta}, (-H_\psi(\hat{\theta}))^{-1}) + \text{const.}
\tag{5.5}
]

The Laplace approximation (q) of (p) is

[
q(\theta) := \mathcal{N}(\theta; \hat{\theta}, \Lambda^{-1}) \propto \exp(\hat{\psi}(\theta)),
\tag{5.6a}
]

where

[
\Lambda := -H_\psi(\hat{\theta})
= -H_\theta \log p(\theta \mid x_{1:n}, y_{1:n}) \Big|_{\theta=\hat{\theta}}.
\tag{5.6b}
]

Recall that for this approximation to be well-defined, the covariance matrix (\Lambda^{-1}) (or equivalently the precision matrix (\Lambda)) needs to be symmetric and positive semi-definite. Let us verify that this is indeed the case for sufficiently smooth (\psi).²

² (\psi) being twice continuously differentiable around (\hat{\theta}) is sufficient.

In this case, the Hessian Λ is sym- 2 ψ being twice continuously differen￾tiable around θˆ metric since the order of differentiation does not matter. Moreover, by is sufficient. the second-order optimality condition, $H_\psi(\hat{\theta})$ is negative semi-definite since (\hat{\theta}) is a local maximum of (\psi). Thus, (-H_\psi(\hat{\theta})) is positive semi-definite.

---

### Example 5.1: Laplace approximation of a Gaussian

Consider approximating the Gaussian density (p(\theta) = \mathcal{N}(\theta; \mu, \Sigma)) using a Laplace approximation.

We know that the mode of (p) is (\mu), which we can verify by computing the gradient,

[
\nabla_\theta \log p(\theta)
= \frac{1}{2} (2\Sigma^{-1}\theta - 2\Sigma^{-1}\mu)
= 0 ;\Longleftrightarrow; \theta = \mu.
\tag{5.7}
]

For the Hessian of (\log p(\theta)), we get

[
H_\theta \log p(\theta)
= D_\theta (\Sigma^{-1}\mu - \Sigma^{-1}\theta)^\top
= -(\Sigma^{-1})^\top
= -\Sigma^{-1},
\tag{5.8}
]

using ((A^{-1})^\top = (A^\top)^{-1}) and symmetry of (\Sigma).

We see that the Laplace approximation of a Gaussian (p(\theta)) is exact, which should not come as a surprise since the second-order Taylor approximation of (\log p(\theta)) is exact for Gaussians.

![Figure 5.1: The Laplace approximation](variationalInference.pdf#page=3)

The Laplace approximation matches the shape of the true posterior around its mode but may not represent it accurately elsewhere — often leading to extremely overconfident predictions.

An example is given in Figure 5.1. Nevertheless, the Laplace approximation has some de￾sirable properties such as being relatively easy to apply in a post-hoc manner, that is, after having already computed the MAP estimate. It preserves the MAP point estimate as its mean and just “adds” a lit￾tle uncertainty around it. However, the fact that it can be arbitrarily different from the true posterior makes it unsuitable for approximate probabilistic inference.

---

## 5.1.1 Example: Bayesian Logistic Regression

Logistic regression learns a classifier that decides for a given input whether it belongs to one of two classes. A sigmoid function, typically the logistic function,

[
\sigma(z) := \frac{1}{1 + \exp(-z)} \in (0,1),
\quad z = w^\top x,
\tag{5.9}
]

is used to obtain the class probabilities.

Bayesian logistic regression corresponds to Bayesian linear regression with a Bernoulli likelihood,

[
y \mid x, w \sim \text{Bern}(\sigma(w^\top x)),
\tag{5.10}
]

where (y \in {-1, 1}) is the binary class label.³ Observe that given a data point (x, y), the probability of a correct classification is

³ The same approach extends to Gaus￾sian processes where it is known as
Gaussian process classification, see Prob￾lem 5.2 and Hensman et al. (2015).

Below is the **direct continuation**, still **1:1 verbatim in Markdown**, with **no rewording, no omissions, no normalization**, continuing exactly where the previous message stopped.
Source remains the same PDF. 

---

[
p(y \mid x, w) =
\begin{cases}
\sigma(w^\top x) & \text{if } y = 1 \
1 - \sigma(w^\top x) & \text{if } y = -1
\end{cases}
= \sigma(y w^\top x),
\tag{5.11}
]

as the logistic function (\sigma) is symmetric around 0. Also, recall that Bayesian linear regression used the prior

[
p(w) = \mathcal{N}(w; 0, \sigma_p^2 I)
\propto \exp!\left(-\frac{1}{2\sigma_p^2}|w|^2\right).
]

Let us first find the posterior mode, that is, the MAP estimate of the weights:

[
\hat{w} = \arg\max_w p(w \mid x_{1:n}, y_{1:n})
]

[
= \arg\max_w p(w)p(y_{1:n} \mid x_{1:n}, w)
\quad \text{using Bayes’ rule (1.45)}
]

[
= \arg\max_w \log p(w) + \log p(y_{1:n} \mid x_{1:n}, w)
\quad \text{taking the logarithm}
]

[
= \arg\max_w
-\frac{1}{2\sigma_p^2}|w|^2

* \sum_{i=1}^n \log \sigma(y_i w^\top x_i)
  ]

[
= \arg\min_w
\frac{1}{2\sigma_p^2}|w|^2

* \sum_{i=1}^n \log(1 + \exp(-y_i w^\top x_i)).
  \tag{5.12}
  ]

Note that for (\lambda = \tfrac{1}{2\sigma_p^2}), the above optimization is equivalent to standard (regularized) logistic regression where

[
\varepsilon_{\log}(w^\top x; y)
:= \log(1 + \exp(-y w^\top x))
\tag{5.13}
]

is called logistic loss. The gradient of the logistic loss is given by (Problem 5.1 (1))

[
\nabla_w \varepsilon_{\log}(w^\top x; y)
= -y x \cdot \sigma(-y w^\top x).
\tag{5.14}
]

Recall that due to the symmetry of (\sigma) around 0, (\sigma(-y w^\top x)) is the probability that (x) was not classified as (y). Intuitively, if the model is “surprised” by the label, the gradient is large.

We can therefore use SGD with the (regularized) gradient step and with batch size 1,

[
w \leftarrow w(1 - 2\lambda \eta_t) + \eta_t y x \sigma(-y w^\top x),
\tag{5.15}
]

for the data point ((x, y)) picked uniformly at random from the training data. Here, (2\lambda \eta_t) is due to the gradient of the regularization term, in effect, performing weight decay.

---

### Example 5.2: Laplace approx. of Bayesian logistic regression

We have already found the mode of the posterior distribution, (\hat{w}). Let us denote by

[
\pi_i := P(y_i = 1 \mid x_i, \hat{w}) = \sigma(\hat{w}^\top x_i)
\tag{5.16}
]

the probability of (x_i) belonging to the positive class under the model given by the MAP estimate of the weights. For the precision matrix, we then have

[
\Lambda
= - H_w \log p(w \mid x_{1:n}, y_{1:n}) \big|_{w=\hat{w}}
]

[
= - H_w \log p(y_{1:n} \mid x_{1:n}, w)\big|_{w=\hat{w}}

* H_w \log p(w)\big|_{w=\hat{w}}
  ]

[
= \sum_{i=1}^n H_w \varepsilon_{\log}(w^\top x_i; y_i)\big|_{w=\hat{w}}

* \sigma_p^{-2} I
  ]

[
= \sum_{i=1}^n x_i x_i^\top \pi_i(1-\pi_i)

* \sigma_p^{-2} I.
  \tag{5.17}
  ]

Observe that (\pi_i(1-\pi_i) \approx 0) if (\pi_i \approx 1) or (\pi_i \approx 0). That is, if a training example is “well-explained” by (\hat{w}), then its contribution to the precision matrix is small. In contrast, we have (\pi_i(1-\pi_i)=0.25) for (\pi_i=0.5). Importantly, (\Lambda) does not depend on the normalization constant of the posterior distribution which is hard to compute.

In summary, we have that (\mathcal{N}(\hat{w}, \Lambda^{-1})) is the Laplace approximation of (p(w \mid x_{1:n}, y_{1:n})).

---

## 5.2 Predictions with a Variational Posterior

How can we make predictions using our variational approximation? We simply approximate the (intractable) true posterior with our variational posterior:

[
p(y_\star \mid x_\star, x_{1:n}, y_{1:n})
= \int p(y_\star \mid x_\star, \theta)p(\theta \mid x_{1:n}, y_{1:n}), d\theta
]

[
\approx \int p(y_\star \mid x_\star, \theta) q_{\varrho}(\theta), d\theta.
\tag{5.18}
]

A straightforward approach is to observe that Equation (5.18) can be viewed as an expectation over the variational posterior (q_{\varrho}) and approximated via Monte Carlo sampling:

[
= \mathbb{E}*{\theta \sim q*{\varrho}}[p(y_\star \mid x_\star, \theta)]
\tag{5.19}
]

[
\approx \frac{1}{m}\sum_{j=1}^m p(y_\star \mid x_\star, \theta_j),
\quad \theta_j \stackrel{iid}{\sim} q_{\varrho}.
\tag{5.20}
]

---

### Example 5.3: Predictions in Bayesian logistic regression

In the case of Bayesian logistic regression with a Gaussian approximation of the posterior, we can obtain more accurate predictions. Observe that the final prediction (y_\star) is conditionally independent of the model parameters (w) given the “latent value” (f_\star = w^\top x_\star):

[
p(y_\star \mid x_\star, x_{1:n}, y_{1:n})
\approx \int p(y_\star \mid x_\star, w) q_{\varrho}(w), dw
]

[
= \int \int p(y_\star \mid f_\star)
p(f_\star \mid x_\star, w)
q_{\varrho}(w), dw, df_\star
\tag{5.21}
]

[
= \int p(y_\star \mid f_\star) \int p(f_\star \mid x_\star, w)
q_{\varrho}(w), dw, df_\star
]

Continuing **verbatim, 1:1 in Markdown**, with **no changes, no omissions**, exactly following the PDF text.
Source unchanged. 

---

The outer integral can be readily approximated since it is only one-dimensional! The challenging part is the inner integral, which is a high-dimensional integral over the model weights (w). Since the posterior over weights (q_{\varrho}(w) = \mathcal{N}(w; \hat{w}, \Lambda^{-1})) is a Gaussian, we have due to the closedness properties of Gaussians (1.78) that

[
\int p(f_\star \mid x_\star, w) q_{\varrho}(w), dw
= \mathcal{N}(\hat{w}^\top x_\star,; x_\star^\top \Lambda^{-1} x_\star).
\tag{5.22}
]

Crucially, this is a one-dimensional Gaussian in function-space as opposed to the (d)-dimensional Gaussian (q_{\varrho}) in weight-space!

As we have seen in Equation (5.11), for Bayesian logistic regression, the prediction (y_\star) depends deterministically on the predicted latent value (f_\star):
(p(y_\star \mid f_\star) = \sigma(y_\star f_\star)). Combining Equations (5.21) and (5.22), we obtain

[
p(y_\star \mid x_\star, x_{1:n}, y_{1:n})
\approx
\int \sigma(y_\star f_\star),
\mathcal{N}(f_\star;; \hat{w}^\top x_\star,; x_\star^\top \Lambda^{-1} x_\star), df_\star.
\tag{5.23}
]

We have replaced the high-dimensional integral over the model parameters (\theta) by the one-dimensional integral over the prediction of our variational posterior (f_\star). While this integral is generally still intractable, it can be approximated efficiently using numerical quadrature methods such as the Gauss–Legendre quadrature or alternatively with Monte Carlo sampling.

---

## 5.3 Blueprint of Variational Inference

General probabilistic inference poses the challenge of approximating the posterior distribution with limited memory and computation, resource constraints also present in humans and other intelligent systems. These resource constraints require information to be compressed, and as we will see, such a compression poses a fundamental tradeoff between model accuracy (on the observed data) and model complexity (to avoid overfitting).

Laplace approximation approximates the true (intractable) posterior with a simpler one, by greedily matching mode and curvature around it. Can we find “less greedy” approaches? We can view variational probabilistic inference more generally as a family of approaches aiming to approximate the true posterior distribution by one that is closest (according to some criterion) among a “simpler” class of distributions.

To this end, we need to fix a class of distributions and define suitable criteria, which we can then optimize numerically. The key benefit is that we can reduce the (generally intractable) problem of high-dimensional integration to the (often much more tractable) problem of optimization.

---

### Definition 5.4 (Variational family)

Let (\mathcal{P}) be the class of all probability distributions. A variational family
(\mathcal{Q} \subset \mathcal{P}) is a class of distributions such that each distribution (q \in \mathcal{Q}) is characterized by unique variational parameters (\varrho \in \Lambda).

---

### Example 5.5: Family of independent Gaussians

A straightforward example for a variational family is the family of independent Gaussians,

[
\mathcal{Q}
===========

\left{
q(\theta) = \mathcal{N}(\theta;; \mu,; \operatorname{diag}_{i \in [d]}{\sigma_i^2})
\right},
\tag{5.24}
]

which is parameterized by

[
\varrho := [\mu_{1:d}, \sigma^2_{1:d}].
]

Such a multivariate distribution where all variables are independent is called a **mean-field distribution**. Importantly, this family of distributions is characterized by only (2d) parameters!

Note that Figure 5.4 is a generalization of the canonical distinction between estimation error and approximation error from Figure 1.7, only that here, we operate in the space of distributions over functions as opposed to the space of functions.

A common notion of distance between two distributions (q) and (p) is the **Kullback–Leibler divergence** (KL(q | p)) which we will define in the next section. Using this notion of distance, we need to solve the following optimization problem:

[
q^\star
:= \arg\min_{q \in \mathcal{Q}} KL(q | p)
= \arg\min_{\varrho \in \Lambda} KL(q_{\varrho} | p).
\tag{5.25}
]

In Section 5.4, we introduce information theory and the Kullback–Leibler divergence. Then, in Section 5.5, we discuss how the optimization problem of Equation (5.25) can be solved efficiently.

---

## 5.4 Information Theoretic Aspects of Uncertainty

One of our main objectives throughout this manuscript is to capture the “uncertainty” about events (A) in an appropriate probability space. One very natural measure of uncertainty is their probability, (P(A)). In this section, we will introduce an alternative measure of uncertainty, namely the so-called **“surprise”** about the event (A).

Continuing **verbatim, 1:1 in Markdown**, with **no changes, no omissions**, exactly following the PDF text.
Source unchanged. 

---

### 5.4.1 Surprise

The surprise about an event with probability (u) is defined as

[
S[u] := - \log u.
\tag{5.26}
]

Observe that the surprise is a function from (\mathbb{R}_{>0}) to (\mathbb{R}), where we let (S[0] = \infty). Moreover, for a discrete random variable (X), we have that (p(x) \le 1), and hence, (S[p(x)] \ge 0). But why is it reasonable to measure surprise by (-\log u)?

**Theorem 5.6 (Axiomatic characterization of surprise).**
The axioms

1. (S[u] > S[v] \iff u < v) (anti-monotonicity): we are more surprised by unlikely events
2. (S) is continuous: no jumps in surprise for infinitesimal changes of probability
3. (S[uv] = S[u] + S[v]) for independent events: the surprise of independent events is additive

characterize (S) up to a positive constant factor.

**Proof.** Observe that the third condition looks similar to the product rule of logarithms:
(\log(uv) = \log u + \log v). We can formalize this intuition by remembering Cauchy’s functional equation, (f(x+y) = f(x) + f(y)), which has the unique family of solutions ({f : x \mapsto cx : c \in \mathbb{R}}) if (f) is required to be continuous. Such a solution is called an *additive function*. Consider the function (g(x) := f(e^x)). Then, (g) is additive if and only if

[
f(e^{x+y}) = f(e^x e^y) = g(x+y) = g(x) + g(y) = f(e^x) + f(e^y).
]

This is precisely the third axiom of surprise for (f = S) and (e^x = u)! Hence, the second and third axioms of surprise imply that (g) must be additive and that (g(x) = S[e^x] = cx) for any (c \in \mathbb{R}). If we replace (e^x) by (u), we obtain (S[u] = c \log u). The first axiom of surprise implies that (c < 0), and thus, (S[u] = -c' \log u) for any (c' > 0). ∎

Importantly, surprise offers a different perspective on uncertainty as opposed to probability: the uncertainty about an event can either be interpreted in terms of its probability or in terms of its surprise, and the two “spaces of uncertainty” are related by a log-transform. Information theory is the study of uncertainty in terms of surprise.

---

### 5.4.2 Entropy

The entropy of a distribution (p) is the average surprise about samples from (p). In this way, entropy is a notion of uncertainty associated with the distribution (p): if the entropy of (p) is large, we are more uncertain about (x \sim p) than if the entropy of (p) were low. Formally,

[
H[p] := \mathbb{E}*{x \sim p}[S[p(x)]] = \mathbb{E}*{x \sim p}[-\log p(x)].
\tag{5.27}
]

When (X \sim p) is a random vector distributed according to (p), we write (H[X] := H[p]).

For discrete distributions it is common to use the logarithm with base 2 rather than the natural logarithm:

[
H[p] = - \sum_x p(x) \log_2 p(x)
\quad \text{(if (p) is discrete)},
\tag{5.28a}
]

[
H[p] = - \int p(x) \log p(x), dx
\quad \text{(if (p) is continuous)}.
\tag{5.28b}
]

---

**Fact 5.7 (Jensen’s inequality).**
Given a random variable (X) and a convex function (g : \mathbb{R} \to \mathbb{R}), we have

[
g(\mathbb{E}[X]) \le \mathbb{E}[g(X)].
\tag{5.29}
]

---

### Example 5.8: Examples of entropy

* **Fair coin:**
  [
  H[\text{Bern}(0.5)] = -2(0.5 \log_2 0.5) = 1.
  ]

* **Unfair coin:**
  [
  H[\text{Bern}(0.1)] = -0.1\log_2 0.1 - 0.9\log_2 0.9 \approx 0.469.
  ]

* **Uniform distribution:**
  [
  H[\text{Unif}({1,\dots,n})]
  = -\sum_{i=1}^n \frac{1}{n}\log_2 \frac{1}{n}
  = \log_2 n.
  ]

The uniform distribution has the maximum entropy among all discrete distributions supported on ({1,\dots,n}). Note that a fair coin corresponds to a uniform distribution with n = 2. Also observe that log2 n corresponds to the number of bits required to encode the outcome of the experiment.
In general, the entropy H[p] of a discrete distribution p can be interpreted as the average number of bits required to encode a sample x ↑ p, or in other words, the average “information” car￾ried by a sample x.

---

### Example 5.9: Entropy of a Gaussian

Let us derive the entropy of a univariate Gaussian. Recall the PDF

[
\mathcal{N}(x;\mu,\sigma^2)
= \frac{1}{Z}\exp!\left(-\frac{(x-\mu)^2}{2\sigma^2}\right),
\quad Z = \sqrt{2\pi\sigma^2}.
]

Using the definition of entropy (5.28b), we obtain

[
H[\mathcal{N}(\mu,\sigma^2)]
= -\int \frac{1}{Z} e^{-\frac{(x-\mu)^2}{2\sigma^2}}
\log!\left(\frac{1}{Z} e^{-\frac{(x-\mu)^2}{2\sigma^2}}\right) dx
]

[
= \log Z + \frac{1}{2\sigma^2}\mathbb{E}[(x-\mu)^2]
= \log(\sigma\sqrt{2\pi}) + \frac{1}{2}
= \log(\sigma\sqrt{2\pi e}).
\tag{5.30}
]

In general, the entropy of a Gaussian is

[
H[\mathcal{N}(\mu,\Sigma)]
= \frac{1}{2}\log\det(2\pi e,\Sigma)
= \frac{1}{2}\log!\big((2\pi e)^d \det \Sigma\big).
\tag{5.31}
]

Note that the entropy is a function of the determinant of the co￾variance matrix Σ. In general, there are various ways of “scalariz￾ing” the notion of uncertainty for a multivariate distribution. The
determinant of Σ measures the volume of the credible sets around
the mean µ, and is also called the generalized variance. Next to
entropy and generalized variance (which are closely related for
Gaussians), a common scalarization is the trace of Σ, which is also
called the total variance.

Continuing **verbatim, 1:1 in Markdown**, with **no changes, no omissions**, exactly following the PDF text.
Source unchanged. 

---

### 5.4.3 Cross-Entropy

How can we use entropy to measure our average surprise when assuming the data follows some distribution (q) but in reality the data follows a different distribution (p)?

**Definition 5.10 (Cross-entropy).**
The cross-entropy of a distribution (q) relative to the distribution (p) is

[
H[p | q] := \mathbb{E}*{x \sim p}[S[q(x)]]
= \mathbb{E}*{x \sim p}[-\log q(x)].
\tag{5.32}
]

Cross-entropy can also be expressed in terms of the KL-divergence (cf. Section 5.4.4) (KL(p | q)) which measures how “different” the distribution (q) is from a reference distribution (p),

[
H[p | q] = H[p] + KL(p | q) \ge H[p].
\tag{5.33}
]

Quite intuitively, the average surprise in samples from (p) with respect to the distribution (q) is given by the inherent uncertainty in (p) and the additional surprise that is due to us assuming the wrong data distribution (q). The “closer” (q) is to the true data distribution (p), the smaller is the additional average surprise.

---

### 5.4.4 Kullback–Leibler Divergence

As mentioned, the Kullback–Leibler divergence is a (non-metric) measure of distance between distributions. It is defined as follows.

**Definition 5.11 (Kullback–Leibler divergence, KL-divergence).**
Given two distributions (p) and (q), the Kullback–Leibler divergence (or relative entropy) of (q) with respect to (p),

[
KL(p | q) := H[p | q] - H[p]
\tag{5.34}
]

[
= \mathbb{E}_{\theta \sim p}[S[q(\theta)] - S[p(\theta)]]
\tag{5.35}
]

[
= \mathbb{E}_{\theta \sim p}!\left[\log\frac{p(\theta)}{q(\theta)}\right],
\tag{5.36}
]

measures how different (q) is from a reference distribution (p).

In words, (KL(p | q)) measures the additional expected surprise when observing samples from (p) that is due to assuming the (wrong) distribution (q) and which is not inherent in the distribution (p) already.⁷

⁷ The KL-divergence only captures the additional expected surprise since the surprise inherent in (p) (as measured by (H[p])) is subtracted.

The KL-divergence has the following properties:

* (KL(p | q) \ge 0) for any distributions (p) and (q) (Problem 5.5 (1))
* (KL(p | q) = 0) if and only if (p = q) almost surely (Problem 5.5 (2))
* There exist distributions (p) and (q) such that (KL(p | q) \ne KL(q | p))

The KL-divergence can simply be understood as a shifted version of cross-entropy, which is zero if we consider the divergence between two identical distributions.

---

We will briefly look at another interpretation for how KL-divergence measures “distance” between distributions. Suppose we are presented with a sequence (\theta_1,\dots,\theta_n) of independent samples from either a distribution (p) or a distribution (q), both of which are known. Which of (p) or (q) was used to generate the data is, however, unknown to us, and we would like to find out. A natural approach is to choose the distribution whose data likelihood is larger. That is, we choose (p) if

[
p(\theta_{1:n}) > q(\theta_{1:n})
]

and vice versa. Assuming that the samples are independent and rewriting the inequality slightly, we choose (p) if

[
\prod_{i=1}^n \frac{p(\theta_i)}{q(\theta_i)} > 1,
]

or equivalently if

[
\sum_{i=1}^n \log \frac{p(\theta_i)}{q(\theta_i)} > 0.
\tag{5.37}
]

Assume without loss of generality that (\theta_i \sim p). Then, using the law of large numbers (A.36),

[
\frac{1}{n}\sum_{i=1}^n \log \frac{p(\theta_i)}{q(\theta_i)}
;\xrightarrow{\text{a.s.}};
\mathbb{E}_{\theta \sim p}!\left[\log\frac{p(\theta)}{q(\theta)}\right]
= KL(p | q),
\tag{5.38}
]

as (n \to \infty). Plugging this into our decision criterion from Equation (5.37), we find that

[
\mathbb{E}!\left[\sum_{i=1}^n \log \frac{p(\theta_i)}{q(\theta_i)}\right]
= n, KL(p | q).
\tag{5.39}
]

In this way, (KL(p | q)) measures the observed “distance” between (p) and (q). Recall that assuming p ↔= q we have that KL(p≃q) > 0 with
probability 1, and therefore we correctly choose p with probability 1
as n ↘ ∞. Moreover, Hoeffding’s inequality (A.41) can be used to de￾termine “how quickly” samples converge to this limit, that is, how
quickly we can distinguish between p and q.

---

### Example 5.12: KL-divergence of Bernoulli random variables

Suppose we are given two Bernoulli distributions (\text{Bern}(p)) and (\text{Bern}(q)). Then, their KL-divergence is

[
KL(\text{Bern}(p) | \text{Bern}(q))
= \sum_{x \in {0,1}} \text{Bern}(x;p)\log\frac{\text{Bern}(x;p)}{\text{Bern}(x;q)}
]

[
= p\log\frac{p}{q} + (1-p)\log\frac{1-p}{1-q}.
\tag{5.40}
]

Observe that (KL(\text{Bern}(p)|\text{Bern}(q)) = 0) if and only if (p=q).

---

### Example 5.13: KL-divergence of Gaussians

Suppose we are given two Gaussian distributions
(p = \mathcal{N}(\mu_p,\Sigma_p)) and
(q = \mathcal{N}(\mu_q,\Sigma_q)) with dimension (d).
The KL-divergence of (p) and (q) is given by

[
KL(p | q)
= \frac{1}{2}\Big(
\operatorname{tr}(\Sigma_q^{-1}\Sigma_p)

* (\mu_p-\mu_q)^\top \Sigma_q^{-1}(\mu_p-\mu_q)

- d

* \log\frac{\det\Sigma_q}{\det\Sigma_p}
  \Big).
  \tag{5.41}
  ]

For independent Gaussians with unit variance, (\Sigma_p=\Sigma_q=I), the expression simplifies to the squared Euclidean distance,

[
KL(p | q) = \frac{1}{2}|\mu_q-\mu_p|_2^2.
\tag{5.42}
]

If we approximate independent Gaussians with variances (\sigma_i^2),

[
p = \mathcal{N}(\mu,\operatorname{diag}{\sigma_1^2,\dots,\sigma_d^2}),
]

by a standard normal distribution (q=\mathcal{N}(0,I)), the expression simplifies to

[
KL(p | q)
= \frac{1}{2}\sum_{i=1}^d (\sigma_i^2 + \mu_i^2 - 1 - \log\sigma_i^2).
\tag{5.43}
]

Here, the term µ2
i penalizes a large mean of p, the term σ2
i penal￾izes a large variance of p, and the term → log σ2
i penalizes a small
variance of p. As expected, KL(p≃q) is proportional to the amount
of information we lose by approximating p with the simpler dis￾tribution q

---

### 5.4.5 Forward and Reverse KL-divergence

(KL(p | q)) is also called the **forward (or inclusive) KL-divergence**. In contrast, (KL(q | p)) is called the **reverse (or exclusive) KL-divergence**. Figure 5.9 shows the approximations of a general Gaussian obtained when (\mathcal{Q}) is the family of diagonal (independent) Gaussians. Thereby,

[
q^\star_1 := \arg\min_{q \in \mathcal{Q}} KL(p | q)
\quad\text{and}\quad
q^\star_2 := \arg\min_{q \in \mathcal{Q}} KL(q | p).
]

(q^\star_1) is the result when using the forward KL-divergence and (q^\star_2) is the result when using reverse KL-divergence. It can be seen that the reverse KL-divergence tends to greedily select the mode and underestimate the variance which, in this case, leads to an overconfident prediction. The forward KL-divergence, in contrast, is more conservative and yields what one could consider the “desired” approximation.

Recall that in the blueprint of variational inference (5.25) we used the reverse KL-divergence. This is for computational reasons. Observe that to approximate the KL-divergence (KL(p | q)) using Monte Carlo sampling, we would need to obtain samples from (p), yet (p) is the intractable posterior distribution which we were trying to approximate in the first place. Crucially, observe that if the true posterior (p(\cdot \mid x_{1:n}, y_{1:n})) is in the variational family (\mathcal{Q}), then

[
\arg\min_{q \in \mathcal{Q}} KL(q | p(\cdot \mid x_{1:n}, y_{1:n}))
= p(\cdot \mid x_{1:n}, y_{1:n})
\quad \text{a.s.},
\tag{5.44}
]

as (\min_{q \in \mathcal{Q}} KL(q | p(\cdot \mid x_{1:n}, y_{1:n})) = 0) almost surely.

---

**Remark 5.14 (Greediness of reverse-KL).**
As in the previous example, consider the independent Gaussians

[
p = \mathcal{N}!\left(\mu,; \operatorname{diag}_{i \in [d]}{\sigma_i^2}\right),
]

which we seek to approximate by a standard normal distribution

[
q = \mathcal{N}(0,I).
]

Using (5.41), we obtain for the reverse KL-divergence,

[
KL(q | p)
= \frac{1}{2}
\sum_{i=1}^d
\left(
\sigma_i^{-2} + \frac{\mu_i^2}{\sigma_i^2}

* 1 + \log\sigma_i^2
  \right).
  \tag{5.45}
  ]

Here, (\sigma_i^{-2}) penalizes small variance, (\mu_i^2/\sigma_i^2) penalizes a large mean, and (\log\sigma_i^2) penalizes large variance. Compare this to the expression for the forward KL-divergence (KL(p | q)) in Equation (5.43). In particular, observe that reverse-KL penalizes large variance less strongly than forward-KL.

Note, however, that reverse-KL is not greedy in the same sense as Laplace approximation, as it does still take the variance into account and does not purely match the mode of (p).

---

### 5.4.6 Interlude: Minimizing Forward KL-Divergence

Before completing the blueprint of variational inference in Section 5.5 by showing how reverse-KL can be efficiently minimized, we will digress briefly and relate minimizing forward-KL to two other well-known inference algorithms. This discussion will deepen our understanding of the KL-divergence and its role in probabilistic inference, but feel free to skip ahead to Section 5.5 if you are eager to complete the blueprint.

---

#### Minimizing forward-KL as maximum likelihood estimation

First, we observe that minimizing the forward KL-divergence is equivalent to maximum likelihood estimation on an infinitely large sample size. The classical application of this result is in the setting where (p(x)) is a generative model, and we aim to estimate its density with the parameterized model (q_{\varrho}).

**Lemma 5.15 (Forward KL-divergence as MLE).**
Given some generative model (p(x)) and a likelihood (q_{\varrho}(x)=q(x \mid \varrho)) (that we use to approximate the true data distribution), we have

[
\arg\min_{\varrho \in \Lambda} KL(p | q_{\varrho})
= \arg\max_{\varrho \in \Lambda}
\lim_{n\to\infty}\frac{1}{n}\sum_{i=1}^n \log q(x_i \mid \varrho),
\tag{5.46}
]

where (x_i \stackrel{iid}{\sim} p) are independent samples from the true data distribution.

**Proof.**

[
KL(p | q_{\varrho})
= H[p | q_{\varrho}] - H[p]
\quad\text{using the definition of KL-divergence (5.34)}
]

[
= \mathbb{E}_{x \sim p}[-\log q(x \mid \varrho)] + \text{const.}
\quad\text{dropping } H[p]
]

[
\overset{a.s.}{=}

* \lim_{n\to\infty}\frac{1}{n}\sum_{i=1}^n \log q(x_i \mid \varrho) + \text{const.}
  ]

using Monte Carlo sampling, i.e., the law of large numbers (A.36). ∎

This tells us that any maximum likelihood estimate (q_{\varrho}) minimizes the forward KL-divergence to the empirical data distribution. Note that here, we aim to learn model parameters (\varrho) for estimating the probability of (x), whereas in the setting of variational probabilistic inference, we want to learn parameters (\varrho) of a distribution over (\theta), and (\theta) parameterizes a distribution over (x). This interpretation is therefore not immediately useful for probabilistic inference as a maximum likelihood estimate requires i.i.d. samples from (p), which we cannot easily obtain in this case.⁸

⁸ It is possible to obtain “approximate” samples using Markov chain Monte Carlo (MCMC) methods which we discuss in Chapter 6.

---

### Example 5.16: Minimizing cross-entropy

Minimizing the KL-divergence between (p) and (q_{\varrho}) is equivalent to minimizing cross-entropy since
(KL(p | q_{\varrho}) = H[p | q_{\varrho}] - H[p]) and (H[p]) is constant with respect to (p).

Let us consider an example in a binary classification problem with the label (y \in {0,1}) and predicted class probability (\hat{y} \in [0,1]) for some fixed input. It is natural to use cross-entropy as a measure of dissimilarity between (y) and (\hat{y}),

[
\varepsilon_{\text{bce}}(\hat{y}; y)
:= H[\text{Bern}(y) | \text{Bern}(\hat{y})]
]

[
= - \sum_{x \in {0,1}} \text{Bern}(x; y)\log \text{Bern}(x; \hat{y})
]

[
= - y \log \hat{y} - (1-y)\log(1-\hat{y}).
\tag{5.47}
]

This loss function is also known as the **binary cross-entropy loss** and we will discuss it in more detail in Section 7.1.3 in the context of neural networks.

---

### Minimizing forward-KL as moment matching

Now to a second interpretation of minimizing forward-KL. **Moment matching** (also known as the method of moments) is a technique for approximating an unknown distribution (p) with a parameterized distribution (q_{\varrho}) where (\varrho) is chosen such that (q_{\varrho}) matches the (estimated) moments of (p).

For example, given the estimates (a) and (B) of the first and second moment of (p),⁹

⁹ These estimates are computed using the samples from (p). For example, using a sample mean and a sample variance to compute the estimates of the first and second moment.

and if (q_{\varrho}) is a Gaussian with parameters (\varrho = {\mu, \Sigma}), then moment matching chooses (\varrho) as the solution to

[
\mathbb{E}*p[\theta] = a
\quad\Longleftrightarrow\quad
\mu = \mathbb{E}*{q_{\varrho}}[\theta]
]

[
\mathbb{E}*p[\theta\theta^\top] = B
\quad\Longleftrightarrow\quad
\Sigma + \mu\mu^\top = \mathbb{E}*{q_{\varrho}}[\theta\theta^\top],
]

using the definition of variance (1.35).

In general the number of moments to be matched (i.e., the number of equations) is adjusted such that it is equal to the number of parameters to be estimated. We will see now that the “matching” of moments is also ensured when (q_{\varrho}) is obtained by minimizing the forward KL-divergence within the family of Gaussians.

---

The Gaussian PDF can be expressed as

[
\mathcal{N}(\theta; \mu, \Sigma)
= \frac{1}{Z(\varrho)} \exp(\varrho^\top s(\theta)),
\tag{5.48}
]

where

[
\varrho :=
\begin{bmatrix}
\Sigma^{-1}\mu \
\operatorname{vec}[\Sigma^{-1}]
\end{bmatrix},
\tag{5.49}
]

[
s(\theta) :=
\begin{bmatrix}
\theta \
\operatorname{vec}!\left[-\tfrac{1}{2}\theta\theta^\top\right]
\end{bmatrix},
\tag{5.50}
]

and

[
Z(\varrho) := \int \exp(\varrho^\top s(\theta)), d\theta.
]

The family of distributions with densities of the form (5.48) — with an additional scaling constant (h(\theta)) which is often 1 — is called the **exponential family of distributions**. Here, (s(\theta)) are the sufficient statistics, (\varrho) are called the natural parameters, and (Z(\varrho)) is the normalizing constant. In this context, (Z(\varrho)) is often called the **partition function**.

---

To see that the Gaussian is indeed part of the exponential family as promised in Equations (5.49) and (5.50), consider

[
\mathcal{N}(\theta; \mu, \Sigma)
\propto \exp!\left(-\frac{1}{2}(\theta-\mu)^\top \Sigma^{-1}(\theta-\mu)\right)
]

[
\propto \exp!\left(
-\frac{1}{2}\theta^\top\Sigma^{-1}\theta

* \theta^\top\Sigma^{-1}\mu
  \right)
  ]

[
= \exp!\left(
\operatorname{tr}!\left(-\frac{1}{2}\theta\theta^\top\Sigma^{-1}\right)

* \theta^\top\Sigma^{-1}\mu
  \right)
  ]

[
= \exp!\left(
\operatorname{vec}!\left[-\frac{1}{2}\theta\theta^\top\right]^\top
\operatorname{vec}[\Sigma^{-1}]

* \theta^\top\Sigma^{-1}\mu
  \right),
  ]

using (\operatorname{tr}(AB)=\operatorname{vec}(A)^\top\operatorname{vec}(B)).

---

This allows us to express the forward KL-divergence as

[
KL(p | q_{\varrho})
= \int p(\theta)\log\frac{p(\theta)}{q_{\varrho}(\theta)}, d\theta
]

[
= - \int p(\theta),\varrho^\top s(\theta), d\theta

* \log Z(\varrho) + \text{const.}
  ]

Differentiating with respect to the natural parameters (\varrho) gives

[
\nabla_{\varrho} KL(p | q_{\varrho})
= - \mathbb{E}_{\theta \sim p}[s(\theta)]

* \mathbb{E}*{\theta \sim q*{\varrho}}[s(\theta)].
  ]

Hence, for any minimizer of (KL(p | q_{\varrho})), we have that the sufficient statistics under (p) and (q_{\varrho}) match:

[
\mathbb{E}*p[s(\theta)] = \mathbb{E}*{q_{\varrho}}[s(\theta)].
\tag{5.51}
]

Therefore, in the Gaussian case,

[
\mathbb{E}*p[\theta] = \mathbb{E}*{q_{\varrho}}[\theta]
\quad\text{and}\quad
\operatorname{Var}_p[\theta] = \Sigma,
\tag{5.52}
]

where (\mu) and (\Sigma) are the mean and variance of the approximation (q_{\varrho}), respectively. Implying that:

[
\mathbb{E}*p[\theta] = \mu \quad\text{and}\quad \text{Var}_p[\theta] = \mathbb{E}*p[\theta\theta^\top] - \mu\mu^\top = \Sigma.
]

where µ and Σ are the mean and variance of the approximation qϱ, re￾spectively. That is, a Gaussian qϱ minimizing KL(p≃qϱ) has the same
first and second moment as p. Combining this insight with our obser￾vation from Equation (5.46) that minimizing forward-KL is equivalent
to maximum likelihood estimation, we see that if we use MLE to fit a
Gaussian to given data, this Gaussian will eventually match the first
and second moments of the data distribution.

---

## 5.5 Evidence Lower Bound

Let us return to the blueprint of variational inference from Section 5.3. To complete this blueprint, it remains to show that the reverse KL-divergence can be minimized efficiently. We have

[
KL(q | p(\cdot \mid x_{1:n}, y_{1:n}))
= \mathbb{E}*{\theta \sim q}
\left[
\log \frac{q(\theta)}{p(\theta \mid x*{1:n}, y_{1:n})}
\right]
]

[
= \mathbb{E}*{\theta \sim q}
\left[
\log \frac{p(y*{1:n} \mid x_{1:n}) q(\theta)}
{p(y_{1:n}, \theta \mid x_{1:n})}
\right]
]

[
= \log p(y_{1:n} \mid x_{1:n})

* \mathbb{E}*{\theta \sim q}[\log p(y*{1:n}, \theta \mid x_{1:n})]
* H[q],
  ]

using the definition of conditional probability (1.8) and linearity of expectation (1.20). Rearranging terms yields

[
\mathcal{L}(q, p; \mathcal{D}*n)
= \log p(y*{1:n} \mid x_{1:n})

* KL(q | p(\cdot \mid x_{1:n}, y_{1:n})).
  \tag{5.53}
  ]

Here, (\mathcal{L}(q, p; \mathcal{D}_n)) is called the **evidence lower bound (ELBO)** given the data
(\mathcal{D}*n = {(x_i, y_i)}*{i=1}^n).

Thus, maximizing the ELBO coincides with minimizing reverse-KL. Maximizing the ELBO,

[
\mathcal{L}(q, p; \mathcal{D}*n)
:= \mathbb{E}*{\theta \sim q}[\log p(y_{1:n}, \theta \mid x_{1:n})]

* H[q],
  \tag{5.54}
  ]

selects (q) that has large joint likelihood (p(y_{1:n}, \theta \mid x_{1:n})) and large entropy (H[q]).

The ELBO can also be expressed in various other forms:

[
\mathcal{L}(q, p; \mathcal{D}*n)
= \mathbb{E}*{\theta \sim q}[\log p(y_{1:n}, \theta \mid x_{1:n}) - \log q(\theta)]
\tag{5.55a}
]

[
= \mathbb{E}*{\theta \sim q}
[\log p(y*{1:n} \mid x_{1:n}, \theta)

* \log p(\theta)

- \log q(\theta)]
  \tag{5.55b}
  ]

[
= \mathbb{E}*{\theta \sim q}[\log p(y*{1:n} \mid x_{1:n}, \theta)]

* KL(q | p(\cdot)).
  \tag{5.55c}
  ]

Here, (p(\cdot)) denotes the prior distribution.

Equation (5.55c) highlights the connection to probabilistic inference, namely that maximizing the ELBO selects a variational distribution (q) that is close to the prior distribution (p(\cdot)) while also maximizing the average likelihood of the data (p(y_{1:n} \mid x_{1:n}, \theta)) for (\theta \sim q). This is in contrast to maximum a posteriori estimation, which picks a single model (\theta) that maximizes the likelihood and proximity to the prior.

As an example, let us look at the case where the prior is noninformative, i.e., (p(\cdot) \propto 1). In this case, the ELBO simplifies to

[
\mathbb{E}*{\theta \sim q}[\log p(y*{1:n} \mid x_{1:n}, \theta)] + H[q] + \text{const.}
]

That is, maximizing the ELBO maximizes the average likelihood of the data under the variational distribution (q) while regularizing (q) to have high entropy.

Why is it reasonable to maximize the entropy of (q)? Consider two distributions (q_1) and (q_2) under which the data is “equally” likely and which are “equally” close to the prior. Maximizing the entropy selects the distribution that exhibits the most uncertainty which is in accordance with the maximum entropy principle.¹¹

¹¹ In Section 1.2.1, we discussed the maximum entropy principle and the related principle of indifference at length.

---

Recalling that KL-divergence is non-negative, it follows from Equation (5.53) that the evidence lower bound is a (uniform¹²) lower bound to the evidence (\log p(y_{1:n} \mid x_{1:n})):

[
\log p(y_{1:n} \mid x_{1:n})
\ge \mathcal{L}(q, p; \mathcal{D}_n).
\tag{5.56}
]

¹² That is, the bound holds for any variational distribution (q) (with full support).

This indicates that maximizing the evidence lower bound is an adequate method of model selection which can be used instead of maximizing the evidence (marginal likelihood) directly (as was discussed in Section 4.4.2). Note that this inequality lower bounds the logarithm of an integral by an expectation of a logarithm over some variational distribution (q). Hence, the ELBO is a family of lower bounds — one for each variational distribution. Such inequalities are called **variational inequalities**.

---

### Example 5.17: Gaussian VI vs Laplace approximation

Consider maximizing the ELBO within the variational family of Gaussians

[
\mathcal{Q} = { q(\theta) = \mathcal{N}(\theta; \mu, \Sigma) }.
]

How does this relate to Laplace approximation which also fits a Gaussian approximation to the posterior
(p(\theta \mid \mathcal{D}) \propto p(y_{1:n}, \theta \mid x_{1:n}))?

It turns out that both approximations are closely related. Indeed, it can be shown that while the Laplace approximation is fitted locally at the MAP estimate (\hat{\theta}), satisfying

[
0 = \nabla_\theta \log p(y_{1:n}, \theta \mid x_{1:n}),
\quad
\Sigma^{-1} = - H_\theta \log p(y_{1:n}, \theta \mid x_{1:n}) \big|_{\theta=\hat{\theta}},
\tag{5.57}
]

Gaussian variational inference satisfies the conditions of the Laplace approximation **on average** with respect to the approximation (q):

[
0 = \mathbb{E}*{\theta \sim q}[\nabla*\theta \log p(y_{1:n}, \theta \mid x_{1:n})],
]

[
\Sigma^{-1}
= - \mathbb{E}*{\theta \sim q}[H*\theta \log p(y_{1:n}, \theta \mid x_{1:n})].
\tag{5.58}
]

For this reason, the Gaussian variational approximation does not suffer from the same overconfidence as the Laplace approximation.¹³

¹³ see Figure 5.1

---

### Example 5.18: ELBO for Bayesian logistic regression

Recall that Bayesian logistic regression uses the prior distribution

[
w \sim \mathcal{N}(0, I).
]

¹⁴ We omit the scaling factor (\sigma_p^2) here for simplicity.

Suppose we use the variational family (\mathcal{Q}) of all diagonal Gaussians from Example 5.5. We have already seen in Equation (5.43) that for a prior (p \sim \mathcal{N}(0, I)) and a variational distribution

[
q_{\varrho} \sim \mathcal{N}!\left(\mu,; \operatorname{diag}_{i \in [d]}{\sigma_i^2}\right),
]

we have

[
KL(q_{\varrho} | p(\cdot))
= \frac{1}{2}\sum_{i=1}^d
(\sigma_i^2 + \mu_i^2 - 1 - \log \sigma_i^2).
]

It remains to find the expected likelihood under models from our approximate posterior:

[
\mathbb{E}*{w \sim q*{\varrho}}[\log p(y_{1:n} \mid x_{1:n}, w)]
= \mathbb{E}*{w \sim q*{\varrho}}
\left[
\sum_{i=1}^n \log p(y_i \mid x_i, w)
\right]
]

[
= \mathbb{E}*{w \sim q*{\varrho}}
\left[

* \sum_{i=1}^n \varepsilon_{\log}(w^\top x_i; y_i)
  \right].
  \tag{5.59}
  ]

---

## 5.5.1 Gradient of Evidence Lower Bound

We have yet to discuss how the optimization problem of maximizing the ELBO can be solved efficiently. A suitable tool is stochastic gradient descent (SGD); however, SGD requires unbiased gradient estimates of the loss

[
\varepsilon(\varrho; \mathcal{D}*n) := -\mathcal{L}(q*{\varrho}, p; \mathcal{D}_n).
]

That is, we need to obtain gradient estimates of

[
\nabla_{\varrho} \mathcal{L}(q_{\varrho}, p; \mathcal{D}*n)
= \nabla*{\varrho} \mathbb{E}*{\theta \sim q*{\varrho}}
[\log p(y_{1:n} \mid x_{1:n}, \theta)]

* \nabla_{\varrho} KL(q_{\varrho} | p(\cdot)).
  \tag{5.60}
  ]

Typically, the KL-divergence (and its gradient) can be computed exactly for commonly used variational families. For example, we have already seen a closed-form expression of the KL-divergence for Gaussians in Equation (5.41).

Obtaining the gradient of the expected log-likelihood is more difficult. This is because the expectation integrates over the measure (q_{\varrho}), which depends on the variational parameters (\varrho). Thus, we cannot move the gradient operator inside the expectation as commonly done (cf. Appendix A.1.5). There are two main techniques which are used to rewrite the gradient in such a way that Monte Carlo sampling becomes possible.

One approach is to use score gradients via the **score function trick**:

[
\nabla_{\varrho} \mathbb{E}*{\theta \sim q*{\varrho}}
[\log p(y_{1:n} \mid x_{1:n}, \theta)]
======================================

\mathbb{E}*{\theta \sim q*{\varrho}}
\big[
\log p(y_{1:n} \mid x_{1:n}, \theta),
\nabla_{\varrho} \log q_{\varrho}(\theta)
\big],
\tag{5.61}
]

which we introduce in Section 12.3.2 in the context of reinforcement learning. More common in the context of variational inference is the so-called **reparameterization trick**.

---

### Theorem 5.19 (Reparameterization trick)

Given a random variable (\varepsilon \sim \phi) (which is independent of (\varrho)) and given a differentiable and invertible function
(g : \mathbb{R}^d \to \mathbb{R}^d). We let

[
\theta := g(\varepsilon; \varrho).
]

Then,

[
q_{\varrho}(\theta)
= \phi(\varepsilon),
\left|\det(D_{\varepsilon} g(\varepsilon; \varrho))\right|^{-1},
\tag{5.62}
]

[
\mathbb{E}*{\theta \sim q*{\varrho}}[f(\theta)]
= \mathbb{E}_{\varepsilon \sim \phi}[f(g(\varepsilon; \varrho))],
\tag{5.63}
]

for a “nice” function (f : \mathbb{R}^d \to \mathbb{R}).

**Proof.** By the change of variables formula (1.43) and using
(\varepsilon = g^{-1}(\theta; \varrho)),

[
q_{\varrho}(\theta)
= \phi(\varepsilon),
\left|\det D_{\theta} g^{-1}(\theta; \varrho)\right|
]

[
= \phi(\varepsilon),
\left|\det(D_{\varepsilon} g(\varepsilon; \varrho))^{-1}\right|
]

[
= \phi(\varepsilon),
\left|\det(D_{\varepsilon} g(\varepsilon; \varrho))\right|^{-1}.
]

Equation (5.63) is a direct consequence of the law of the unconscious statistician (1.22). ∎

In other words, the reparameterization trick allows a change of “densities” by finding a function (g(\cdot;\varrho)) and a reference density (\phi) such that (q_{\varrho} = g(\cdot;\varrho)\sharp\phi) is the pushforward of (\phi) under perturbation (g). Applying the reparameterization trick, we can swap the order of gradient and expectation,

[
\nabla_{\varrho} \mathbb{E}*{\theta \sim q*{\varrho}}[f(\theta)]
= \mathbb{E}*{\varepsilon \sim \phi}
[\nabla*{\varrho} f(g(\varepsilon; \varrho))].
\tag{5.64}
]

We call a distribution (q_{\varrho}) **reparameterizable** if it admits reparameterization, i.e., if we can find (g) and a suitable reference density (\phi) which is independent of (\varrho).

---

### Example 5.20: Reparameterization trick for Gaussians

Suppose we use a Gaussian variational approximation,

[
q_{\varrho}(\theta) := \mathcal{N}(\theta; \mu, \Sigma),
]

where we assume (\Sigma) to have full rank (i.e., be invertible). We have seen in Equation (1.78) that a Gaussian random vector
(\varepsilon \sim \mathcal{N}(0, I)) can be transformed to follow the Gaussian distribution (q_{\varrho}) by using the linear transformation,

[
\theta = g(\varepsilon; \varrho) := \Sigma^{1/2}\varepsilon + \mu.
\tag{5.65}
]

In particular, we have

[
\varepsilon = g^{-1}(\theta; \varrho)
= \Sigma^{-1/2}(\theta - \mu),
\tag{5.66}
]

and

[
\phi(\varepsilon)
= q_{\varrho}(\theta), |\det \Sigma^{1/2}|.
\tag{5.67}
]

---

In the following, we write (C := \Sigma^{1/2}). Let us now derive the gradient estimate for the evidence lower bound assuming the Gaussian variational approximation from Example 5.20. This approach extends to any reparameterizable distribution.

[
\nabla_{\varrho} \mathbb{E}*{\theta \sim q*{\varrho}}
[\log p(y_{1:n} \mid x_{1:n}, \theta)]
= \nabla_{C,\mu}
\mathbb{E}*{\varepsilon \sim \mathcal{N}(0,I)}
\big[
\log p(y*{1:n} \mid x_{1:n}, \theta)\big]_{\theta=C\varepsilon+\mu}
\tag{5.68}
]

using the reparameterization trick (5.63).

[
= n \cdot \nabla_{C,\mu}
\mathbb{E}*{\varepsilon \sim \mathcal{N}(0,I)}
\left[
\frac{1}{n}\sum*{i=1}^n
\log p(y_i \mid x_i, \theta)
\right]_{\theta=C\varepsilon+\mu}
]

using independence of the data and extending with (n/n).

[
= n \cdot \nabla_{C,\mu}
\mathbb{E}*{\varepsilon \sim \mathcal{N}(0,I)}
\mathbb{E}*{i \sim \text{Unif}([n])}
\big[
\log p(y_i \mid x_i, \theta)
\big]_{\theta=C\varepsilon+\mu}
]

interpreting the sum as an expectation.

[
= n \cdot
\mathbb{E}*{\varepsilon \sim \mathcal{N}(0,I)}
\mathbb{E}*{i \sim \text{Unif}([n])}
\big[
\nabla_{C,\mu}
\log p(y_i \mid x_i, \theta)
\big]_{\theta=C\varepsilon+\mu}
\tag{5.69}
]

using Equation (A.5).

[
\approx n \cdot \frac{1}{m}
\sum_{j=1}^m
\nabla_{C,\mu}
\log p(y_{i_j} \mid x_{i_j}, \theta)
\Big|_{\theta=C\varepsilon_j+\mu},
\tag{5.70}
]

using Monte Carlo sampling, where
(\varepsilon_j \stackrel{iid}{\sim} \mathcal{N}(0,I)) and
(i_j \stackrel{iid}{\sim} \text{Unif}([n])).

This yields an unbiased gradient estimate, which we can use with stochastic gradient descent to maximize the evidence lower bound. We have successfully recast the difficult problems of learning and inference as an optimization problem!

The procedure of approximating the true posterior using a variational posterior by maximizing the evidence lower bound using stochastic optimization is also called **black box stochastic variational inference** (Ranganath et al., 2014; Titsias and Lázaro-Gredilla, 2014; Duvenaud and Adams, 2015). The only requirement is that we can obtain unbiased gradient estimates from the evidence lower bound (and the likelihood). We have just discussed one of many approaches to obtain such gradient estimates (Mohamed et al., 2020). If we use the variational family of diagonal Gaussians, we only require twice as many parameters as other inference techniques like MAP estimation. The performance can be improved by using natural gradients and variance reduction techniques for the gradient estimates such as control variates.

---

## 5.5.2 Minimizing Surprise via Exploration and Exploitation

Now that we have established a way to optimize the ELBO, let us dwell a bit more on its interpretation. Observe that the evidence can also be interpreted as the negative surprise about the observations under the prior distribution (p(\theta)), and by negating Equation (5.56), we obtain the variational upper bound

[
S[p(y_{1:n} \mid x_{1:n})]
\le
\mathbb{E}*{\theta \sim q}[S[p(y*{1:n}, \theta \mid x_{1:n})]]

* H[q]
  = -\mathcal{L}(q, p; \mathcal{D}_n).
  \tag{5.71}
  ]

Here, (-\mathcal{L}(q, p; \mathcal{D}_n)) is commonly called the (variational) **free energy** with respect to (q). Free energy can also be characterized as

[
-\mathcal{L}(q, p; \mathcal{D}_n)
=================================

\mathbb{E}*{\theta \sim q}[S[p(y*{1:n} \mid x_{1:n}, \theta)]]

* KL(q | p(\cdot)),
  \tag{5.72}
  ]

analogously to Equation (5.55c), and therefore its minimization is minimizing the average surprise about the data under the variational distribution (q) while maximizing proximity to the prior (p(\cdot)).

Systems that minimize the surprise in their observations are widely studied in many areas of science.¹⁵

¹⁵ Refer to the free energy principle which was originally introduced by the neuroscientist Karl Friston (Friston, 2010).

To minimize surprise, the free energy makes apparent a natural tradeoff between two extremes: on the one hand, models (q(\theta)) that “overfit” to observations (e.g., by using a point estimate of (\theta)), and hence result in a large surprise when new observations deviate from this specific belief. On the other hand, models (q(\theta)) that “underfit” to observations (e.g., by expecting outcomes with equal probability), and hence any observation results in a non-negligible surprise. Either of these extremes is undesirable.

As we have alluded to previously when introducing the ELBO, the two terms constituting free energy map neatly onto this tradeoff. Therein, the entropy (which is maximized) encourages (q) to have uncertainty, in other words, to “explore” beyond the finite data. In contrast, the energy (which is minimized) encourages (q) to fit the observed data closely, in other words, to “exploit” the finite data. This tradeoff is ubiquitous in approximations to probabilistic inference that deal with limited computational resources and limited time, and we will encounter it many times more.

Since this tradeoff is so fundamental and appears in many branches of science under different names, it is difficult to give it an appropriate unifying name. The essence of this tradeoff can be captured as a **principle of curiosity and conformity**, which suggests that reasoning under uncertainty requires curiosity to entertain and pursue alternative explanations of the data and conformity to make consistent predictions.

---

## Discussion

We have explored variational inference, where we approximate the intractable posterior distribution of probabilistic inference with a simpler distribution. We operationalized this idea by turning the inference problem, which requires computing high-dimensional integrals, into a tractable optimization problem. Gaussians are frequently used as variational distributions due to their versatility and compact representation.

Nevertheless, recall from Figure 5.4 that while the estimation error in variational inference can be small, choosing a variational family that is too simple can lead to a large approximation error. We have seen that for posteriors that are multimodal or have heavy tails, Gaussians may not provide a good approximation. In the next chapter, we will explore alternative techniques for approximate inference that can handle more complex posteriors.

---

## Problems

### 5.1 Logistic loss

1. Derive the gradient of (\varepsilon_{\log}) as given in Equation (5.14).
2. Show that
   [
   H_w \varepsilon_{\log}(w^\top x; y)
   = x x^\top \cdot \sigma(w^\top x)\cdot(1-\sigma(w^\top x)).
   \tag{5.73}
   ]
   **Hint.** Begin by deriving the first derivative of the logistic function, and use the chain rule of multivariate calculus,
   [
   D_x(f \circ g)
   = (Dg(x), f \circ g)\cdot D_x g,
   \tag{5.74}
   ]
   where (g : \mathbb{R}^n \to \mathbb{R}^k) and (f : \mathbb{R}^k \to \mathbb{R}^m).
3. Is the logistic loss (\varepsilon_{\log}) convex in (w)?

---

Continuing **verbatim, 1:1 in Markdown**, with **no changes, no omissions**, exactly following the PDF text.
Source unchanged. 

---

In the following, we write (C := \Sigma^{1/2}). Let us now derive the gradient estimate for the evidence lower bound assuming the Gaussian variational approximation from Example 5.20. This approach extends to any reparameterizable distribution.

[
\nabla_{\varrho} \mathbb{E}*{\theta \sim q*{\varrho}}
[\log p(y_{1:n} \mid x_{1:n}, \theta)]
= \nabla_{C,\mu}
\mathbb{E}*{\varepsilon \sim \mathcal{N}(0,I)}
\big[
\log p(y*{1:n} \mid x_{1:n}, \theta)\big]_{\theta=C\varepsilon+\mu}
\tag{5.68}
]

using the reparameterization trick (5.63).

[
= n \cdot \nabla_{C,\mu}
\mathbb{E}*{\varepsilon \sim \mathcal{N}(0,I)}
\left[
\frac{1}{n}\sum*{i=1}^n
\log p(y_i \mid x_i, \theta)
\right]_{\theta=C\varepsilon+\mu}
]

using independence of the data and extending with (n/n).

[
= n \cdot \nabla_{C,\mu}
\mathbb{E}*{\varepsilon \sim \mathcal{N}(0,I)}
\mathbb{E}*{i \sim \text{Unif}([n])}
\big[
\log p(y_i \mid x_i, \theta)
\big]_{\theta=C\varepsilon+\mu}
]

interpreting the sum as an expectation.

[
= n \cdot
\mathbb{E}*{\varepsilon \sim \mathcal{N}(0,I)}
\mathbb{E}*{i \sim \text{Unif}([n])}
\big[
\nabla_{C,\mu}
\log p(y_i \mid x_i, \theta)
\big]_{\theta=C\varepsilon+\mu}
\tag{5.69}
]

using Equation (A.5).

[
\approx n \cdot \frac{1}{m}
\sum_{j=1}^m
\nabla_{C,\mu}
\log p(y_{i_j} \mid x_{i_j}, \theta)
\Big|_{\theta=C\varepsilon_j+\mu},
\tag{5.70}
]

using Monte Carlo sampling, where
(\varepsilon_j \stackrel{iid}{\sim} \mathcal{N}(0,I)) and
(i_j \stackrel{iid}{\sim} \text{Unif}([n])).

This yields an unbiased gradient estimate, which we can use with stochastic gradient descent to maximize the evidence lower bound. We have successfully recast the difficult problems of learning and inference as an optimization problem!

The procedure of approximating the true posterior using a variational posterior by maximizing the evidence lower bound using stochastic optimization is also called **black box stochastic variational inference** (Ranganath et al., 2014; Titsias and Lázaro-Gredilla, 2014; Duvenaud and Adams, 2015). The only requirement is that we can obtain unbiased gradient estimates from the evidence lower bound (and the likelihood). We have just discussed one of many approaches to obtain such gradient estimates (Mohamed et al., 2020). If we use the variational family of diagonal Gaussians, we only require twice as many parameters as other inference techniques like MAP estimation. The performance can be improved by using natural gradients and variance reduction techniques for the gradient estimates such as control variates.

---

## 5.5.2 Minimizing Surprise via Exploration and Exploitation

Now that we have established a way to optimize the ELBO, let us dwell a bit more on its interpretation. Observe that the evidence can also be interpreted as the negative surprise about the observations under the prior distribution (p(\theta)), and by negating Equation (5.56), we obtain the variational upper bound

[
S[p(y_{1:n} \mid x_{1:n})]
\le
\mathbb{E}*{\theta \sim q}[S[p(y*{1:n}, \theta \mid x_{1:n})]]

* H[q]
  = -\mathcal{L}(q, p; \mathcal{D}_n).
  \tag{5.71}
  ]

Here, (-\mathcal{L}(q, p; \mathcal{D}_n)) is commonly called the (variational) **free energy** with respect to (q). Free energy can also be characterized as

[
-\mathcal{L}(q, p; \mathcal{D}_n)
=================================

\mathbb{E}*{\theta \sim q}[S[p(y*{1:n} \mid x_{1:n}, \theta)]]

* KL(q | p(\cdot)),
  \tag{5.72}
  ]

analogously to Equation (5.55c), and therefore its minimization is minimizing the average surprise about the data under the variational distribution (q) while maximizing proximity to the prior (p(\cdot)).

Systems that minimize the surprise in their observations are widely studied in many areas of science.¹⁵

¹⁵ Refer to the free energy principle which was originally introduced by the neuroscientist Karl Friston (Friston, 2010).

To minimize surprise, the free energy makes apparent a natural tradeoff between two extremes: on the one hand, models (q(\theta)) that “overfit” to observations (e.g., by using a point estimate of (\theta)), and hence result in a large surprise when new observations deviate from this specific belief. On the other hand, models (q(\theta)) that “underfit” to observations (e.g., by expecting outcomes with equal probability), and hence any observation results in a non-negligible surprise. Either of these extremes is undesirable.

As we have alluded to previously when introducing the ELBO, the two terms constituting free energy map neatly onto this tradeoff. Therein, the entropy (which is maximized) encourages (q) to have uncertainty, in other words, to “explore” beyond the finite data. In contrast, the energy (which is minimized) encourages (q) to fit the observed data closely, in other words, to “exploit” the finite data. This tradeoff is ubiquitous in approximations to probabilistic inference that deal with limited computational resources and limited time, and we will encounter it many times more.

Since this tradeoff is so fundamental and appears in many branches of science under different names, it is difficult to give it an appropriate unifying name. The essence of this tradeoff can be captured as a **principle of curiosity and conformity**, which suggests that reasoning under uncertainty requires curiosity to entertain and pursue alternative explanations of the data and conformity to make consistent predictions.

---

## Discussion

We have explored variational inference, where we approximate the intractable posterior distribution of probabilistic inference with a simpler distribution. We operationalized this idea by turning the inference problem, which requires computing high-dimensional integrals, into a tractable optimization problem. Gaussians are frequently used as variational distributions due to their versatility and compact representation.

Nevertheless, recall from Figure 5.4 that while the estimation error in variational inference can be small, choosing a variational family that is too simple can lead to a large approximation error. We have seen that for posteriors that are multimodal or have heavy tails, Gaussians may not provide a good approximation. In the next chapter, we will explore alternative techniques for approximate inference that can handle more complex posteriors.

---

## Problems

### 5.1 Logistic loss

1. Derive the gradient of (\varepsilon_{\log}) as given in Equation (5.14).
2. Show that
   [
   H_w \varepsilon_{\log}(w^\top x; y)
   = x x^\top \cdot \sigma(w^\top x)\cdot(1-\sigma(w^\top x)).
   \tag{5.73}
   ]
   **Hint.** Begin by deriving the first derivative of the logistic function, and use the chain rule of multivariate calculus,
   [
   D_x(f \circ g)
   = (Dg(x), f \circ g)\cdot D_x g,
   \tag{5.74}
   ]
   where (g : \mathbb{R}^n \to \mathbb{R}^k) and (f : \mathbb{R}^k \to \mathbb{R}^m).
3. Is the logistic loss (\varepsilon_{\log}) convex in (w)?

---

### 5.2 Gaussian process classification

In this exercise, we will study the use of Gaussian processes for classification tasks, commonly called Gaussian process classification (GPC). Linear logistic regression is extended to GPC by replacing the Gaussian prior over weights with a GP prior on (f),

[
f \sim \mathcal{GP}(0, k), \quad
y \mid x, f \sim \text{Bern}(\sigma(f(x))).
\tag{5.75}
]

Note that Bayesian logistic regression is the special case where (k) is the linear kernel and (\sigma) is the logistic function. This is analogous to the relationship of Bayesian linear regression and Gaussian process regression.

In the GP regression setting of Chapter 4, (y_i) was assumed to be a noisy observation of (f(x_i)). In the classification setting, we now have that (y_i \in {-1, +1}) is a binary class label and (f(x_i) \in \mathbb{R}) is a latent value. We study the setting where

[
\sigma(z) = \Phi(z; 0, \sigma_n^2)
]

is the CDF of a univariate Gaussian with mean 0 and variance (\sigma_n^2), also called a **probit likelihood**.

To make probabilistic predictions for a query point (x_\star), we first compute the distribution of the latent variable (f_\star),

[
p(f_\star \mid x_{1:n}, y_{1:n}, x_\star)
=========================================

\int
p(f_\star \mid x_{1:n}, x_\star, f),
p(f \mid x_{1:n}, y_{1:n}), df
\tag{5.76}
]

using the sum rule (1.7) and product rule (1.11), and (f_\star \perp y_{1:n} \mid f).

1. Assuming that we can efficiently compute (p(f_\star \mid x_{1:n}, y_{1:n}, x_\star)) (approximately), describe how we can find the predictive posterior
   [
   p(y_\star = +1 \mid x_{1:n}, y_{1:n}, x_\star).
   ]

2. The posterior over the latent variables is not a Gaussian as we used a non-Gaussian likelihood, and hence, the integral of the latent predictive posterior (5.76) is analytically intractable. A common technique is to approximate the latent posterior (p(f \mid x_{1:n}, y_{1:n})) with a Gaussian using a Laplace approximation
   [
   q = \mathcal{N}(\hat{f}, \Lambda^{-1}).
   ]
   It is generally not possible to obtain an analytical representation of the mode of the Laplace approximation (\hat{f}). Instead, (\hat{f}) is commonly found using a second-order optimization scheme such as Newton’s method.

   (a) Find the precision matrix (\Lambda) of the Laplace approximation.
   **Hint.** Observe that for a label (y_i \in {-1, +1}), the probability of a correct classification given the latent value (f_i) is
   [
   p(y_i \mid f_i) = \sigma(y_i f_i),
   ]
   where we use the symmetry of the probit likelihood around 0.

   (b) Assume that (k(x, x') = x^\top x') is the linear kernel ((\sigma_p = 1)) and that (\sigma) is the logistic function (5.9). Show for this setting that the matrix (\Lambda) derived in (a) is equivalent to the precision matrix (\Lambda') of the Laplace approximation of Bayesian logistic regression (5.17).¹⁶
   You may assume that (\hat{f}_i = \hat{w}^\top x_i).
   **Hint.** First derive under which condition (\Lambda) and (\Lambda') are “equivalent”.

   (c) Observe that the (approximate) latent predictive posterior
   [
   q(f_\star \mid x_{1:n}, y_{1:n}, x_\star)
   :=
   \int
   p(f_\star \mid x_{1:n}, x_\star, f),
   q(f \mid x_{1:n}, y_{1:n}), df
   ]
   which uses the Laplace-approximated latent posterior is Gaussian.¹⁷
   Determine its mean and variance.
   **Hint.** Condition on the latent variables (f) using the laws of total expectation and variance.

   (d) Compare the prediction
   [
   p(f_\star \mid x_{1:n}, y_{1:n}, x_\star)
   ]
   you obtained in (1) (but now using the Laplace-approximated latent predictive posterior) to the prediction
   [
   \sigma(\mathbb{E}*{f*\star \sim q}[f_\star]).
   ]
   Are they identical? If not, describe how they are different.

3. The use of the probit likelihood may seem arbitrary. Consider the following model which may be more natural,
   [
   f \sim \mathcal{GP}(0, k), \quad
   y = \mathbf{1}{f(x) + \varepsilon \ge 0},
   \quad
   \varepsilon \sim \mathcal{N}(0, \sigma_n^2).
   \tag{5.77}
   ]
   Show that the model from Equation (5.75) using a noise-free latent process with probit likelihood (\Phi(z; 0, \sigma_n^2)) is equivalent (in expectation over (\varepsilon)) to the model from Equation (5.77).

---

### 5.3 Jensen’s inequality

1. Prove the finite form of Jensen’s inequality.

**Theorem 5.21 (Jensen’s inequality, finite form).**
Let (f : \mathbb{R}^n \to \mathbb{R}) be a convex function. Suppose that
(x_1,\dots,x_k \in \mathbb{R}^n) and (\varrho_1,\dots,\varrho_k \ge 0) with
(\varrho_1 + \cdots + \varrho_k = 1). Then,

[
f(\varrho_1 x_1 + \cdots + \varrho_k x_k)
\le
\varrho_1 f(x_1) + \cdots + \varrho_k f(x_k).
\tag{5.78}
]

Observe that if (X) is a random variable with finite support, the above two versions of Jensen’s inequality are equivalent.

2. Show that for any discrete distribution (p) supported on a finite domain of size (n),
   [
   H[p] \le \log_2 n.
   ]
   This implies that the uniform distribution has maximum entropy.

---

### 5.4 Binary cross-entropy loss

Show that the logistic loss (5.13) is equivalent to the binary cross-entropy loss with (\hat{y} = \sigma(\hat{f})). That is,

[
\varepsilon_{\log}(\hat{f}; y)
= \varepsilon_{\text{bce}}(\hat{y}; y).
\tag{5.79}
]

---

### 5.5 Gibbs’ inequality

1. Prove (KL(p | q) \ge 0) which is also known as **Gibbs’ inequality**.

2. Let (p) and (q) be discrete distributions with finite identical support (\mathcal{A}). Show that
   (KL(p | q) = 0) if and only if (p = q).
   **Hint.** Use that if a function (f : \mathbb{R}^n \to \mathbb{R}) is strictly convex and
   (x_1,\dots,x_k \in \mathbb{R}^n), (\varrho_1,\dots,\varrho_k \ge 0),
   (\varrho_1 + \cdots + \varrho_k = 1), we have

   [
   f(\varrho_1 x_1 + \cdots + \varrho_k x_k)
   =========================================

   \varrho_1 f(x_1) + \cdots + \varrho_k f(x_k)
   \tag{5.80}
   ]

   if and only if (x_1 = \cdots = x_k).

---

### 5.6 Maximum entropy principle

In this exercise we will prove that the normal distribution is the distribution with maximal entropy among all (univariate) distributions supported on (\mathbb{R}) with fixed mean (\mu) and variance (\sigma^2). Let

[
g(x) := \mathcal{N}(x; \mu, \sigma^2),
]

and let (f(x)) be any distribution on (\mathbb{R}) with mean (\mu) and variance (\sigma^2).

1. Prove that
   [
   KL(f | g) = H[g] - H[f].
   ]
   **Hint.** Equivalently, show that (H[f | g] = H[g]).

2. Conclude that
   [
   H[g] \ge H[f].
   ]

---

### 5.7 Probabilistic inference as a consequence of the maximum entropy principle

Consider the family of generative models of the random vectors (X \in \mathcal{X}) and (Y \in \mathcal{Y}),

[
\Delta_{\mathcal{X} \times \mathcal{Y}}
=======================================

\left{
q : \mathcal{X} \times \mathcal{Y} \to \mathbb{R}*{\ge 0}
;\middle|;
\int*{\mathcal{X} \times \mathcal{Y}} q(x,y), dx, dy = 1
\right}.
]

Suppose that we observe (Y) to be (y'), and are looking for a (new) generative model that is consistent with this information, that is,

[
q(y) = \int_{\mathcal{X}} q(x,y), dx = \delta_{y'}(y),
]

where (\delta_{y'}) denotes the point density at (y').

1. Show that the optimization problem

[
\arg\min_{q \in \Delta_{\mathcal{X} \times \mathcal{Y}}}
KL(q_{X,Y} | p_{X,Y})
\quad\text{subject to}\quad
q(y) = \delta_{y'}(y)
]

is solved by
[
q(x,y) = \delta_{y'}(y), p(x \mid y).
]

2. Conclude that
   [
   q(x) = p(x \mid y').
   ]

---

### 5.8 KL-divergence of Gaussians

Derive Equation (5.41).

**Hint.** If (X \sim \mathcal{N}(\mu, \Sigma)) in (d) dimensions, then for any (m \in \mathbb{R}^d) and (A \in \mathbb{R}^{d \times d}),

[
\mathbb{E}[(X-m)^\top A (X-m)]
= (\mu-m)^\top A(\mu-m) + \operatorname{tr}(A\Sigma).
\tag{5.81}
]

---

### 5.9 Forward vs reverse KL

1. Consider a factored approximation (q(x,y) = q(x)q(y)) to a joint distribution (p(x,y)). Show that to minimize the forward KL (KL(p | q)) we should set (q(x) = p(x)) and (q(y) = p(y)).

2. Consider the following joint distribution (p), where the rows represent (y) and the columns (x):

|   | 1   | 2   | 3   | 4   |
| - | --- | --- | --- | --- |
| 1 | 1/8 | 1/8 | 0   | 0   |
| 2 | 1/8 | 1/8 | 0   | 0   |
| 3 | 0   | 0   | 1/4 | 0   |
| 4 | 0   | 0   | 0   | 1/4 |

Show that the reverse KL (KL(q | p)) for this (p) has three distinct minima. Identify those minima and evaluate (KL(q | p)) at each of them.

3. What is the value of (KL(q | p)) if we use the approximation (q(x,y) = p(x)p(y))?

---

### 5.10 Gaussian VI vs Laplace approximation

In this exercise, we compare the Laplace approximation from Section 5.1 to variational inference with the variational family of Gaussians,

[
\mathcal{Q} = {q(\theta) = \mathcal{N}(\theta; \mu, \Sigma)}.
]

1. Let (p) be any distribution on (\mathbb{R}), and let
   [
   q^\star = \arg\min_{q \in \mathcal{Q}} KL(p | q).
   ]
   Show that (q^\star) differs from the Laplace approximation of (p).

2. Show that
   [
   \tilde{q} = \arg\min_{q \in \mathcal{Q}} KL(q | p(\cdot \mid \mathcal{D}_n))
   ]
   satisfies Equation (5.58):

   [
   0 = \mathbb{E}*{\theta \sim \tilde{q}}[\nabla*\theta \log p(y_{1:n}, \theta \mid x_{1:n})],
   ]

   [
   \Sigma^{-1}
   = -\mathbb{E}*{\theta \sim \tilde{q}}[H*\theta \log p(y_{1:n}, \theta \mid x_{1:n})].
   ]

   **Hint 1.** For any positive definite and symmetric matrix (A),
   (\nabla_A \log\det(A) = A^{-1}).

   **Hint 2.** For any function (f) and Gaussian (p = \mathcal{N}(\mu, \Sigma)),

   [
   \nabla_\mu \mathbb{E}*{x \sim p}[f(x)]
   = \mathbb{E}*{x \sim p}[\nabla_x f(x)],
   ]

   [
   \nabla_\Sigma \mathbb{E}*{x \sim p}[f(x)]
   = \frac{1}{2}\mathbb{E}*{x \sim p}[H_x f(x)].
   \tag{5.82}
   ]

---

### 5.11 Gradient of reverse-KL

Suppose (p = \mathcal{N}(0, \sigma_p^2 I)) and

[
q_{\varrho} = \mathcal{N}(\mu, \operatorname{diag}{\sigma_1^2,\dots,\sigma_d^2}),
]

where (\mu = [\mu_1,\dots,\mu_d]) and
(\varrho = [\mu_1,\dots,\mu_d,\sigma_1,\dots,\sigma_d]). Show that the gradient of (KL(q_{\varrho} | p(\cdot))) with respect to (\varrho) is given by

[
\nabla_\mu KL(q_{\varrho} | p(\cdot))
= \sigma_p^{-2} \mu,
\tag{5.83a}
]

[
\nabla_{[\sigma_1 \dots \sigma_d]} KL(q_{\varrho} | p(\cdot))
=============================================================

\begin{bmatrix}
\frac{\sigma_1}{\sigma_p^2} - \frac{1}{\sigma_1} &
\dots &
\frac{\sigma_d}{\sigma_p^2} - \frac{1}{\sigma_d}
\end{bmatrix}.
\tag{5.83b}
]

---

### 5.12 Reparameterizable distributions

1. Let (X \sim \text{Unif}([a,b])) for any (a < b). That is,

[
p_X(x) =
\begin{cases}
\frac{1}{b-a} & x \in [a,b] \
0 & \text{otherwise}.
\end{cases}
\tag{5.84}
]

Show that (X) can be reparameterized in terms of (\text{Unif}([0,1])).

2. Let (Z \sim \mathcal{N}(\mu, \sigma^2)) and (X := e^Z). That is, (X) is log-normally distributed with parameters (\mu) and (\sigma^2). Show that (X) can be reparameterized in terms of (\mathcal{N}(0,1)).

3. Show that (\text{Cauchy}(0,1)) can be reparameterized in terms of (\text{Unif}([0,1])).

Finally, let us apply the reparameterization trick to compute the gradient of an expectation.

4. Let (\text{ReLU}(z) := \max{0,z}) and (w > 0). Show that

[
\frac{d}{d\mu}
\mathbb{E}_{x \sim \mathcal{N}(\mu,1)}[\text{ReLU}(wx)]
= w \Phi(\mu),
]

where (\Phi) denotes the CDF of the standard normal distribution.

---
