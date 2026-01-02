# Gaussian Processes

Let us remember our first attempt from Chapter 2 at scaling up Bayesian linear regression to nonlinear functions. We saw that we can model nonlinear functions by transforming the input space to a suitable higher-dimensional feature space, but found that this approach scales poorly if we require a large number of features. We then found something remarkable: by simply changing our perspective from a weight-space view to a function-space view, we could implement Bayesian linear regression without ever needing to compute the features explicitly. Under the function-space view, the key object describing the class of functions we can model is not the features (\phi(x)), but instead the kernel function which only implicitly defines a feature space. Our key observation in this chapter is that we can therefore stop reasoning about feature spaces, and instead directly work with kernel functions that describe “reasonable” classes of functions.

We are still concerned with the problem of estimating the value of a function
(f : \mathcal{X} \to \mathbb{R}) at arbitrary points (x_* \in \mathcal{X}) given training data ({x_i, y_i}_{i=1}^n), where the labels are assumed to be corrupted by homoscedastic Gaussian noise with variance (\sigma_n^2),

[
y_i = f(x_i) + \varepsilon_i, \quad \varepsilon_i \sim \mathcal{N}(0, \sigma_n^2).
]

As in Chapter 2 on Bayesian linear regression, we denote by (X) the design matrix (collection of training inputs) and by (y) the vector of training labels. We will represent the unknown function value at a point (x \in \mathcal{X}) by the random variable

[
f_x := f(x).
]

The collection of these random variables is then called a Gaussian process if any finite subset of them is jointly Gaussian:

**Definition 4.1 (Gaussian process, GP).**
A Gaussian process is an infinite set of random variables such that any finite number of them are jointly Gaussian and such that they are consistent under marginalization.¹

---

¹ That is, if you take a joint distribution for (n) variables and marginalize out one of them, you should recover the joint distribution for the remaining (n - 1) variables.

The fact that with a Gaussian process, any finite subset of the random variables is jointly Gaussian is the key property allowing us to perform exact probabilistic inference. Intuitively, a Gaussian process can be interpreted as a normal distribution over functions — and is therefore often called an “infinite-dimensional Gaussian”.

```
[Figure 4.1]
A Gaussian process can be interpreted as an infinite-dimensional Gaussian over functions. At any location x in the domain, this yields a distribution over values f(x) shown in red. The blue line corresponds to the MAP estimate (i.e., mean function of the Gaussian process), the dark gray region corresponds to the epistemic uncertainty and the light gray region denotes the additional aleatoric uncertainty.
```

A Gaussian process is characterized by a mean function
(\mu : \mathcal{X} \to \mathbb{R}) and a covariance function (or kernel function)
(k : \mathcal{X} \times \mathcal{X} \to \mathbb{R}) such that for any set of points
(A := {x_1, \dots, x_m} \subseteq \mathcal{X}), we have

[
f_A := [f_{x_1} ; \cdots ; f_{x_m}]^\top \sim \mathcal{N}(\mu_A, K_{AA}) \tag{4.1}
]

where

[
\mu_A :=
\begin{bmatrix}
\mu(x_1) \
\vdots \
\mu(x_m)
\end{bmatrix},
\quad
K_{AA} :=
\begin{bmatrix}
k(x_1, x_1) & \cdots & k(x_1, x_m) \
\vdots & \ddots & \vdots \
k(x_m, x_1) & \cdots & k(x_m, x_m)
\end{bmatrix}.
\tag{4.2}
]

We write (f \sim \mathrm{GP}(\mu, k)). In particular, given a mean function, covariance function, and using the homoscedastic noise assumption,

[
y_* \mid x_* \sim \mathcal{N}(\mu(x_*), k(x_*, x_*) + \sigma_n^2), \tag{4.3}
]

with (\sigma^2(x_*) = k(x_*, x_*)) as the epistemic uncertainty and (\sigma_n^2) as the aleatoric uncertainty. Commonly, for notational simplicity, the mean function is taken to be zero. Note that for a fixed mean this is not a restriction, as we can simply apply the zero-mean Gaussian process to the difference between the mean and the observations.²

---

² For alternative ways of representing a mean function, refer to Section 2.7 of *Gaussian Processes for Machine Learning* (Williams and Rasmussen, 2006).

---

## 4.1 Learning and Inference

First, let us look at learning and inference in the context of Gaussian processes. With slight abuse of our previous notation, let us denote the set of observed points by
(A := {x_1, \dots, x_n}). Given a prior (f \sim \mathrm{GP}(\mu, k)) and the noisy observations (y_i = f(x_i) + \varepsilon_i) with (\varepsilon_i \sim \mathcal{N}(0, \sigma_n^2)), we can then write the joint distribution of the observations (y_{1:n}) and the noise-free prediction (f_*) at a test point (x_*) as

[
\begin{bmatrix}
y \
f_*
\end{bmatrix}
\mid x_*, X
\sim \mathcal{N}(\tilde{\mu}, \tilde{K}), \tag{4.4}
]

where

[
\tilde{\mu} :=
\begin{bmatrix}
\mu_A \
\mu(x_*)
\end{bmatrix},
\quad
\tilde{K} :=
\begin{bmatrix}
K_{AA} + \sigma_n^2 I & k_{x_*,A} \
k_{A,x_*} & k(x_*, x_*)
\end{bmatrix},
\tag{4.5}
]

and

[
k_{x,A} :=
\begin{bmatrix}
k(x, x_1) \
\vdots \
k(x, x_n)
\end{bmatrix}.
]

Deriving the conditional distribution using Equation (1.53), we obtain that the Gaussian process posterior is given by

[
f \mid x_{1:n}, y_{1:n} \sim \mathrm{GP}(\mu', k'), \tag{4.6}
]

where

[
\mu'(x) := \mu(x) + k_{x,A}^\top (K_{AA} + \sigma_n^2 I)^{-1}(y_A - \mu_A), \tag{4.7}
]

[
k'(x, x') := k(x, x') - k_{x,A}^\top (K_{AA} + \sigma_n^2 I)^{-1} k_{x',A}. \tag{4.8}
]

Observe that analogously to Bayesian linear regression, the posterior covariance can only decrease when conditioning on additional data, and is independent of the observations (y_i).

We already studied inference in the function-space view of Bayesian linear regression, but did not make the predictive posterior explicit. Using Equation (4.6), the predictive posterior at (x_*) is simply

[
f_* \mid x_*, x_{1:n}, y_{1:n} \sim \mathcal{N}(\mu'(x_*), k'(x_*, x_*)). \tag{4.9}
]

---

## 4.2 Sampling

Often, we are not interested in the full predictive posterior distribution, but merely want to obtain samples of our Gaussian process model. We will briefly examine two approaches.

1. For the first approach, consider a discretized subset of points

[
f := [f_1, \dots, f_n]
]

that we want to sample.³ Note that (f \sim \mathcal{N}(\mu, K)). We have already seen in Equation (1.54) that

[
f = K^{1/2} \varepsilon + \mu \tag{4.10}
]

where (K^{1/2}) is the square root of (K) and (\varepsilon \sim \mathcal{N}(0, I)) is standard Gaussian noise.⁴ However, computing the square root of (K) takes (O(n^3)) time.

2. For the second approach, recall the product rule (1.11),

[
p(f_1, \dots, f_n) =
\prod_{i=1}^n p(f_i \mid f_{1:i-1}).
]

That is, the joint distribution factorizes neatly into a product where each factor only depends on the “outcomes” of preceding factors. We can therefore obtain samples one-by-one, each time conditioning on one more observation:

[
\begin{aligned}
f_1 &\sim p(f_1) \
f_2 &\sim p(f_2 \mid f_1) \
f_3 &\sim p(f_3 \mid f_1, f_2) \
&;;\vdots
\end{aligned}
\tag{4.11}
]

This general approach is known as forward sampling. Due to the matrix inverse in the formula of the GP posterior (4.6), this approach also takes (O(n^3)) time.

We will discuss more efficient approximate sampling methods in Section 4.5.

---

## 4.3 Kernel Functions

We have seen that kernel functions are the key object describing the class of functions a Gaussian process can model. Depending on the kernel function, the “shape” of functions that are realized from a Gaussian process varies greatly. Let us recap briefly from Section 2.4 what a kernel function is:

**Definition 4.2 (Kernel function).**
A kernel function (k : \mathcal{X} \times \mathcal{X} \to \mathbb{R}) satisfies

* (k(x, x') = k(x', x)) for any (x, x' \in \mathcal{X}) (symmetry), and
* (K_{AA}) is positive semi-definite for any (A \subseteq \mathcal{X}).

The two defining conditions ensure that for any (A \subseteq \mathcal{X}), (K_{AA}) is a valid covariance matrix. We say that a kernel function is positive definite if (K_{AA}) is positive definite for any (A \subseteq \mathcal{X}).

Intuitively, the kernel function evaluated at locations (x) and (x') describes how (f(x)) and (f(x')) are related, which we can express formally as

[
k(x, x') = \mathrm{Cov}\big[f(x), f(x')\big]. \tag{4.12}
]

If (x) and (x') are “close”, then (f(x)) and (f(x')) are usually taken to be positively correlated, encoding a “smooth” function.

In the following, we will discuss some of the most common kernel functions, how they can be combined to create “new” kernels, and how we can characterize the class of functions they can model.

## 4.3.1 Common Kernels  *(Page 4, continued)*

```
[Figure 4.2]
Functions sampled according to a Gaussian process with a linear kernel and φ = id.
```

```
[Figure 4.3]
Functions sampled according to a Gaussian process with a linear kernel and φ(x) = [1, x, x²] (left) and φ(x) = sin(x) (right).
```

First, we look into some of the most commonly used kernels. Often an additional factor σ² (output scale) is added, which we assume here to be 1 for simplicity.

1. **The linear kernel** is defined as

[
k(x, x'; \phi) := \phi(x)^\top \phi(x') \tag{4.13}
]

where (\phi) is a nonlinear transformation as introduced in Section 2.3 or the identity.

**Remark 4.3: GPs with linear kernel and BLR**
A Gaussian process with a linear kernel is equivalent to Bayesian linear regression. This follows directly from the function-space view of Bayesian linear regression (see Section 2.4) and comparing the derived kernel function (2.20) with the definition of the linear kernel (4.13).

2. **The Gaussian kernel** (also known as squared exponential kernel or radial basis function (RBF) kernel) is defined as

[
k(x, x'; h) := \exp!\left(-\frac{|x - x'|^2}{2h^2}\right) \tag{4.14}
]

where (h) is its length scale. The larger the length scale (h), the smoother the resulting functions.⁵ Furthermore, it turns out that the feature space corresponding to the Gaussian kernel is “infinitely dimensional”, as you will show in Problem 4.1. The Gaussian kernel already encodes a function class that we were not able to model under the weight-space view of Bayesian linear regression.

```
[Figure 4.4]
Functions sampled according to a Gaussian process with a Gaussian kernel and length scales h = 5 (left) and h = 1 (right).
```

```
[Figure 4.5]
Gaussian kernel with length scales h = 1, h = 0.5, and h = 0.2.
```

3. **The Laplace kernel** (also known as exponential kernel) is defined as

[
k(x, x'; h) := \exp!\left(-\frac{|x - x'|}{h}\right) \tag{4.15}
]

As can be seen in Figure 4.7, samples from a GP with Laplace kernel are non-smooth as opposed to the samples from a GP with Gaussian kernel.

```
[Figure 4.6]
Laplace kernel with length scales h = 1, h = 0.5, and h = 0.2.
```

```
[Figure 4.7]
Functions sampled according to a Gaussian process with a Laplace kernel and length scales h = 10 000 (left) and h = 10 (right).
```

4. **The Matérn kernel** trades the smoothness of the Gaussian and the Laplace kernels. As such, it is frequently used in practice to model “real world” functions that are relatively smooth. It is defined as

[
k(x, x'; \nu, h) :=
\frac{2^{1-\nu}}{\Gamma(\nu)}
\left(\frac{\sqrt{2\nu},|x - x'|}{h}\right)^{\nu}
K_\nu!\left(\frac{\sqrt{2\nu},|x - x'|}{h}\right)
\tag{4.16}
]

where (\Gamma) is the Gamma function, (K_\nu) the modified Bessel function of the second kind, and (h) a length scale parameter. For (\nu = 1/2), the Matérn kernel is equivalent to the Laplace kernel. For (\nu \to \infty), the Matérn kernel is equivalent to the Gaussian kernel. The resulting functions are (\lceil \nu \rceil - 1) times mean square differentiable.⁶ In particular, GPs with a Gaussian kernel are infinitely many times mean square differentiable whereas GPs with a Laplace kernel are mean square continuous but not mean square differentiable.

---

## 4.3.2 Composing Kernels

Given two kernels (k_1 : \mathcal{X} \times \mathcal{X} \to \mathbb{R}) and
(k_2 : \mathcal{X} \times \mathcal{X} \to \mathbb{R}), they can be composed to obtain a new kernel (k : \mathcal{X} \times \mathcal{X} \to \mathbb{R}) in the following ways:

* (k(x, x') := k_1(x, x') + k_2(x, x')),
* (k(x, x') := k_1(x, x') \cdot k_2(x, x')),
* (k(x, x') := c \cdot k_1(x, x')) for any (c > 0),
* (k(x, x') := f(k_1(x, x'))) for any polynomial (f) with positive coefficients or (f = \exp).

For example, the additive structure of a function
(f(x) := f_1(x) + f_2(x)) can be easily encoded in GP models. Suppose that
(f_1 \sim \mathrm{GP}(\mu_1, k_1)) and (f_2 \sim \mathrm{GP}(\mu_2, k_2)), then the distribution of the sum of those two functions
(f = f_1 + f_2 \sim \mathrm{GP}(\mu_1 + \mu_2, k_1 + k_2)) is another GP.⁷

Whereas the addition of two kernels (k_1) and (k_2) can be thought of as an OR operation (i.e., the kernel has high value if either (k_1) or (k_2) have high value), the multiplication of (k_1) and (k_2) can be thought of as an AND operation (i.e., the kernel has high value if both (k_1) and (k_2) have high value). For example, the product of two linear kernels results in functions which are quadratic.

As mentioned previously, the constant (c) of a scaled kernel function
(k'(x, x') := c \cdot k(x, x')) is generally called the output scale of a kernel, and it scales the variance
(\mathrm{Var}[f(x)] = c \cdot k(x, x)) of the predictions (f(x)) from (\mathrm{GP}(\mu, k')).

## 4.3.3 Stationarity and Isotropy

Kernel functions are commonly classified according to two properties:

**Definition 4.4 (Stationarity and isotropy).**
A kernel (k : \mathbb{R}^d \times \mathbb{R}^d \to \mathbb{R}) is called

* **stationary** (or shift-invariant) if there exists a function (\tilde{k}) such that
  [
  \tilde{k}(x - x') = k(x, x'),
  ]
* **isotropic** if there exists a function (\tilde{k}) such that
  [
  \tilde{k}(|x - x'|) = k(x, x')
  ]
  with (|\cdot|) any norm.

Note that stationarity is a necessary condition for isotropy. In other words, isotropy implies stationarity.

**Example 4.5: Stationarity and isotropy of kernels**

| kernel                                                                            | stationary | isotropic |
| --------------------------------------------------------------------------------- | ---------- | --------- |
| linear kernel                                                                     | no         | no        |
| Gaussian kernel                                                                   | yes        | yes       |
| (k(x, x') := \exp!\left(-|x - x'|_M^2\right)) where (M) is positive semi-definite | yes        | no        |

Here, (|\cdot|_M) denotes the Mahalanobis norm induced by matrix (M).

For (x' = x), stationarity implies that the kernel must only depend on 0. In other words, a stationary kernel must depend on relative locations only. This is clearly not the case for the linear kernel, which depends on the absolute locations of (x) and (x'). Therefore, the linear kernel cannot be isotropic either.

For the Gaussian kernel, isotropy follows immediately from its definition.

The last kernel is clearly stationary by definition, but not isotropic for general matrices (M). Note that for (M = I) it is indeed isotropic.

Stationarity encodes the idea that relative location matters more than absolute location: the process “looks the same” no matter where we shift it in the input space. This is often appropriate when we believe the same statistical behavior holds across the entire domain (e.g., no region is special). Isotropy goes one step further by requiring that the kernel depends only on the distance between points, so that all directions in the space are treated equally. In other words, there is no preferred orientation or axis. This is especially useful in settings where we expect uniform behavior in every direction (as with the Gaussian kernel). Such kernels are simpler to specify and interpret since we only need a single “scale” (like a length scale) rather than multiple parameters or directions.

## 4.3.4 Reproducing Kernel Hilbert Spaces

We can characterize the precise class of functions that can be modeled by a Gaussian process with a given kernel function. This corresponding function space is called a reproducing kernel Hilbert space (RKHS), and we will discuss it briefly in this section.

Recall that Gaussian processes keep track of a posterior distribution
(f \mid x_{1:n}, y_{1:n}) over functions. We will in fact show later that the corresponding MAP estimate (\hat{f}) corresponds to the solution to a regularized optimization problem in the RKHS space of functions. This duality is similar to the duality between the MAP estimate of Bayesian linear regression and ridge regression we observed in Chapter 2. So what is the reproducing kernel Hilbert space of a kernel function (k)?

**Definition 4.6 (Reproducing kernel Hilbert space, RKHS).**
Given a kernel (k : \mathcal{X} \times \mathcal{X} \to \mathbb{R}), its corresponding reproducing kernel Hilbert space is the space of functions (f) defined as

[
\mathcal{H}*k(\mathcal{X}) :=
\left{
f(\cdot) =
\sum*{i=1}^n \alpha_i k(x_i, \cdot)
:
n \in \mathbb{N},\ x_i \in \mathcal{X},\ \alpha_i \in \mathbb{R}
\right}.
\tag{4.17}
]

The inner product of the RKHS is defined as

[
\langle f, g \rangle_k :=
\sum_{i=1}^n
\sum_{j=1}^{n'}
\alpha_i \alpha'_j k(x_i, x'_j),
\tag{4.18}
]

where
(g(\cdot) = \sum_{j=1}^{n'} \alpha'_j k(x'_j, \cdot)),
and induces the norm (|f|_k = \sqrt{\langle f, f \rangle_k}).

You can think of the norm as measuring the “smoothness” or “complexity” of (f).
*Problem 4.4 (2)*

It is straightforward to check that for all (x \in \mathcal{X}), (k(x, \cdot) \in \mathcal{H}_k(\mathcal{X})). Moreover, the RKHS inner product (\langle \cdot, \cdot \rangle_k) satisfies for all (x \in \mathcal{X}) and (f \in \mathcal{H}_k(\mathcal{X})) that

[
f(x) = \langle f(\cdot), k(x, \cdot) \rangle_k,
]

which is also known as the reproducing property.
*Problem 4.4 (1)*

That is, evaluations of RKHS functions (f) are inner products in (\mathcal{H}_k(\mathcal{X})) parameterized by the “feature map” (k(x, \cdot)).

The representer theorem (Schölkopf et al., 2001) characterizes the solution to regularized optimization problems in RKHSs:

---

**Theorem 4.7 (Representer theorem).**
Let (k) be a kernel and let (\lambda > 0).
*Problem 4.5*

For (f \in \mathcal{H}*k(\mathcal{X})) and training data ({(x_i, f(x_i))}*{i=1}^n), let
(L(f(x_1), \dots, f(x_n)) \in \mathbb{R} \cup {\infty}) denote any loss function which depends on (f) only through its evaluation at the training points. Then, any minimizer

[
\hat{f} \in
\arg\min_{f \in \mathcal{H}_k(\mathcal{X})}
L(f(x_1), \dots, f(x_n)) + \lambda |f|_k^2
\tag{4.19}
]

admits a representation of the form

[
\hat{f}(x) = \hat{\varepsilon}^\top k_{x,{x_i}_{i=1}^n}
=======================================================

\sum_{i=1}^n \hat{\alpha}_i k(x, x_i)
\quad \text{for some } \hat{\varepsilon} \in \mathbb{R}^n.
\tag{4.20}
]

This statement is remarkable: the solutions to general regularized optimization problems over the generally infinite-dimensional space of functions (\mathcal{H}_k(\mathcal{X})) can be represented as a linear combination of the kernel functions evaluated at the training points. The representer theorem can be used to show that the MAP estimate of a Gaussian process corresponds to the solution of a regularized linear regression problem in the RKHS of the kernel function, namely,

[
\hat{f} :=
\arg\min_{f \in \mathcal{H}_k(\mathcal{X})}

* \log p(y_{1:n} \mid x_{1:n}, f) + \frac{1}{2}|f|_k^2.
  \tag{4.21}
  ]

Here, the first term corresponds to the likelihood, measuring the “quality of fit”. The regularization term limits the “complexity” of (\hat{f}). Regularization is necessary to prevent overfitting since in expressive RKHSs, there may be many functions that interpolate the training data perfectly.

This shows the close link between Gaussian process regression and Bayesian linear regression, with the kernel function (k) generalizing the inner product of feature maps to feature spaces of possibly “infinite dimensionality”. Because solutions can be represented as linear combinations of kernel evaluations at the training points, Gaussian processes remain computationally tractable even though they can model functions over “infinite-dimensional” feature spaces.

## 4.4 Model Selection

We have not yet discussed how to pick the hyperparameters (\theta) (e.g., parameters of kernels). A common technique in supervised learning is to select hyperparameters (\theta), such that the resulting function estimate (\hat{f}*\theta) leads to the most accurate predictions on hold-out validation data. After reviewing this approach, we contrast it with a probabilistic approach to model selection, which avoids using point estimates of (\hat{f}*\theta) and rather utilizes the full posterior.

### 4.4.1 Optimizing Validation Set Performance

A common approach to model selection is to split our data (D) into separate training set
(D_{\text{train}} := {(x_i^{\text{train}}, y_i^{\text{train}})}*{i=1}^n) and validation sets
(D*{\text{val}} := {(x_i^{\text{val}}, y_i^{\text{val}})}_{i=1}^m). We then optimize the model for a parameter candidate (\theta_j) using the training set. This is usually done by picking a point estimate (like the MAP estimate),

[
\hat{f}*j :=
\arg\max_f p(f \mid x^{\text{train}}*{1:n}, y^{\text{train}}_{1:n}). \tag{4.22}
]

Then, we score (\theta_j) according to the performance of (\hat{f}_j) on the validation set,

[
\hat{\theta} :=
\arg\max_{\theta_j}
p(y^{\text{val}}*{1:m} \mid x^{\text{val}}*{1:m}, \hat{f}_j). \tag{4.23}
]

This ensures that (\hat{f}*j) does not depend on (D*{\text{val}}).

**Remark 4.8: Approximating population risk**
Why is it useful to separate the data into a training and a validation set? Recall from Appendix A.3.5 that minimizing the empirical risk without separating training and validation data may lead to overfitting as both the loss and (\hat{f}_j) depend on the same data (D). In contrast, using independent training and validation sets, (\hat{f}*j) does not depend on (D*{\text{val}}), and we have that

[
\frac{1}{m}
\sum_{i=1}^m
\varepsilon(y_i^{\text{val}} \mid x_i^{\text{val}}, \hat{f}*j)
;;\Longleftrightarrow;;
\mathbb{E}*{(x,y)\sim P}
\big[
\varepsilon(y \mid x, \hat{f}_j)
\big],
\tag{4.24}
]

using Monte Carlo sampling.⁸

In words, for reasonably large (m), minimizing the empirical risk as we do in Equation (4.23) approximates minimizing the population risk.

While this approach often is quite effective at preventing overfitting as compared to using the same data for training and picking (\hat{\theta}), it still collapses the uncertainty in (f) into a point estimate. Can we do better?

### 4.4.2 Maximizing the Marginal Likelihood

We have already seen for Bayesian linear regression, that picking a point estimate loses a lot of information. Instead of optimizing the effects of (\theta) for a specific point estimate (\hat{f}) of the model (f), maximizing the marginal likelihood optimizes the effects of (\theta) across all realizations of (f). In this approach, we obtain our hyperparameter estimate via

[
\hat{\theta}*{\text{MLE}} :=
\arg\max*\theta
p(y_{1:n} \mid x_{1:n}, \theta)
\tag{4.25}
]

using the definition of marginal likelihood in Bayes’ rule (1.45)

[
= \arg\max_\theta
\int p(y_{1:n}, f \mid x_{1:n}, \theta), df
]

by conditioning on (f) using the sum rule (1.7)

[
= \arg\max_\theta
\int p(y_{1:n} \mid x_{1:n}, f, \theta), p(f \mid \theta), df
\tag{4.26}
]

using the product rule (1.11).

Remarkably, this approach typically avoids overfitting even though we do not use a separate training and validation set. The following table provides an intuitive argument for why maximizing the marginal likelihood is a good strategy.

```
[Table 4.1]
likelihood | prior
---------------------------------
“underfit” model (too simple θ)
small for “almost all” f | large
“overfit” model (too complex θ)
large for “few” f | small for “most” f
“just right”
moderate for “many” f | moderate
```

```
[Figure 4.8]
A schematic illustration of the marginal likelihood of a simple, intermediate, and complex model across all possible data sets.
```

For an “underfit” model, the likelihood is mostly small as the data cannot be well described, while the prior is large as there are “fewer” functions to choose from. For an “overfit” model, the likelihood is large for “some” functions (which would be picked if we were only minimizing the training error and not doing cross validation) but small for “most” functions. The prior is small, as the probability mass has to be distributed among “more” functions. Thus, in both cases, one term in the product will be small. Hence, maximizing the marginal likelihood naturally encourages trading between a large likelihood and a large prior.

In the context of Gaussian process regression, recall from Equation (4.3) that

[
y_{1:n} \mid x_{1:n}, \theta \sim \mathcal{N}(0, K_{f,\theta} + \sigma_n^2 I).
\tag{4.27}
]

We write (K_{y,\theta} := K_{f,\theta} + \sigma_n^2 I).

Continuing from Equation (4.25), we obtain

[
\hat{\theta}*{\text{MLE}}
= \arg\max*\theta \mathcal{N}(y; 0, K_{y,\theta})
]

[
= \arg\min_\theta
\frac{1}{2} y^\top K_{y,\theta}^{-1} y

* \frac{1}{2} \log \det K_{y,\theta}
* \frac{n}{2} \log 2\pi
  \tag{4.28}
  ]

taking the negative logarithm

[
= \arg\min_\theta
\frac{1}{2} y^\top K_{y,\theta}^{-1} y

* \frac{1}{2} \log \det K_{y,\theta}.
  \tag{4.29}
  ]

The last term is independent of (\theta).

The first term of the optimization objective describes the “goodness of fit” (i.e., the “alignment” of (y) with (K_{y,\theta})). The second term characterizes the “volume” of the model class. Thus, this optimization naturally trades the aforementioned objectives.

Marginal likelihood maximization is an empirical Bayes method. Often it is simply referred to as empirical Bayes. It also has the nice property that the gradient of its objective (the MLL loss) can be expressed in closed form, *Problem 4.7*,

[
\frac{\partial}{\partial \varrho_j}
\log p(y_{1:n} \mid x_{1:n}, \theta)
====================================

\frac{1}{2}
\operatorname{tr}!\left(
(\varepsilon \varepsilon^\top - K_{y,\theta}^{-1})
\frac{\partial K_{y,\theta}}{\partial \varrho_j}
\right),
\tag{4.30}
]

where (\varepsilon := K_{y,\theta}^{-1} y) and (\operatorname{tr}(M)) is the trace of a matrix (M). This optimization problem is, in general, non-convex.

```
[Figure 4.9]
An example of model selection by maximizing the log likelihood (without hyperpriors) using a linear, quadratic, Laplace, Matérn (ν = 3/2), and Gaussian kernel, respectively. They are used to learn the function x ↦ sin(x)/x + ε, ε ∼ N(0, 0.01), using SGD with learning rate 0.1.
```

Taking a step back, observe that taking a probabilistic perspective on model selection naturally led us to consider all realizations of our model (f) instead of using point estimates. However, we are still using point estimates for our model parameters (\theta). Continuing on our probabilistic adventure, we could place a prior (p(\theta)) on them too.⁹ We could use it to obtain the MAP estimate (still a point estimate!) which adds an additional regularization term

[
\hat{\theta}*{\text{MAP}} :=
\arg\max*\theta p(\theta \mid x_{1:n}, y_{1:n})
\tag{4.31}
]

[
= \arg\min_\theta

* \log p(\theta) - \log p(y_{1:n} \mid x_{1:n}, \theta),
  \tag{4.32}
  ]

using Bayes’ rule (1.45) and then taking the negative logarithm.

An alternative approach is to consider the full posterior distribution over parameters (\theta). The resulting predictive distribution is, however, intractable,

[
p(y_* \mid x_*, x_{1:n}, y_{1:n})
=================================

\int!!\int
p(y_* \mid x_*, f),
p(f \mid x_{1:n}, y_{1:n}, \theta),
p(\theta), df, d\theta.
\tag{4.33}
]

Recall that as the mode of Gaussians coincides with their mean, the MAP estimate corresponds to the mean of the predictive posterior.

As a final note, observe that in principle, there is nothing stopping us from descending deeper in the probabilistic hierarchy. The prior on the model parameters (\theta) is likely to have parameters too. Ultimately, we need to break out of this hierarchy of dependencies and choose a prior.

## 4.5 Approximations

```
[Figure 4.10]
The top plot shows contour lines of an empirical Bayes with two local optima. The bottom two plots show the Gaussian processes corresponding to the two optimal models. The left model with smaller lengthscale is chosen within a more flexible class of models, while the right model explains more observations through noise. Adapted from figure 5.5 of “Gaussian Processes for Machine Learning” (Williams and Rasmussen, 2006).
```

To learn a Gaussian process, we need to invert (n \times n) matrices, hence the computational cost is (O(n^3)). Compare this to Bayesian linear regression which allows us to learn a regression model in (O(nd^2)) time (even online) where (d) is the feature dimension. It is therefore natural to look for ways of approximating a Gaussian process.

### 4.5.1 Local Methods

Recall that during forward sampling, we had to condition on a larger and larger number of previous samples. When sampling at a location (x), a very simple approximation is to only condition on those samples (x') that are “close” (where (|k(x, x')| \ge \tau) for some (\tau > 0)). Essentially, this method “cuts off the tails” of the kernel function (k). However, (\tau) has to be chosen carefully as if (\tau) is chosen too large, samples become essentially independent.

This is one example of a sparse approximation of a Gaussian process. We will discuss more advanced sparse approximations known as “inducing point methods” in Section 4.5.3.

---

### 4.5.2 Kernel Function Approximation

Another method is to approximate the kernel function directly. The idea is to construct a “low-dimensional” feature map
(\phi : \mathbb{R}^d \to \mathbb{R}^m) that approximates the kernel,

[
k(x, x') \approx \phi(x)^\top \phi(x').
\tag{4.34}
]

Then, we can apply Bayesian linear regression, resulting in a time complexity of (O(nm^2 + m^3)).

One example of this approach are random Fourier features, which we will discuss in the following.

```
[Figure 4.11]
Illustration of Euler’s formula. It can be seen that e^{i\varphi} corresponds to a (counter-clockwise) rotation on the unit circle as \varphi varies from 0 to 2\pi.
```

**Remark 4.9: Fourier transform**
First, let us remind ourselves of Fourier transformations. The Fourier transform is a method of decomposing frequencies into their individual components.

Recall Euler’s formula which states that for any (x \in \mathbb{R}),

[
e^{ix} = \cos x + i \sin x
\tag{4.35}
]

where (i) is the imaginary unit of complex numbers. The formula is illustrated in Figure 4.11. Note that (e^{-i2\pi x}) corresponds to rotating clockwise around the unit circle in (\mathbb{R}^2) — completing a rotation whenever (x \in \mathbb{R}) reaches the next natural number.

We can scale (x) by a frequency (\xi): (e^{-i2\pi \xi x}). If (x \in \mathbb{R}^d), we can also scale each component (j) of (x) by a different frequency (\xi^{(j)}). Multiplying a function (f : \mathbb{R}^d \to \mathbb{R}) with the rotation around the unit circle with given frequencies (\xi), yields a quantity that describes the amplitude of the frequencies (\xi),

[
\hat{f}(\xi) :=
\int_{\mathbb{R}^d}
f(x), e^{-i2\pi \xi^\top x}, dx.
\tag{4.36}
]

(\hat{f}) is called the Fourier transform of (f). (f) is called the inverse Fourier transform of (\hat{f}), and can be computed using

[
f(x) =
\int_{\mathbb{R}^d}
\hat{f}(\xi), e^{i2\pi \xi^\top x}, d\xi.
\tag{4.37}
]

It is common to write (\omega := 2\pi \xi). See Figure 4.12 for an example. Refer to “But what is the Fourier Transform? A visual introduction” (Sanderson, 2018) for a visual introduction.

```
[Figure 4.12]
The Fourier transform of a rectangular pulse.
```

Because a stationary kernel (k : \mathbb{R}^d \times \mathbb{R}^d \to \mathbb{R}) can be interpreted as a function in one variable, it has an associated Fourier transform which we denote by (p(\omega)). That is,

[
k(x - x') =
\int_{\mathbb{R}^d}
p(\omega), e^{i\omega^\top (x - x')}, d\omega.
\tag{4.38}
]

**Fact 4.10 (Bochner’s theorem).**
A continuous stationary kernel on (\mathbb{R}^d) is positive definite if and only if its Fourier transform (p(\omega)) is non-negative.

Bochner’s theorem implies that when a continuous and stationary kernel is positive definite and scaled appropriately, its Fourier transform (p(\omega)) is a proper probability distribution. In this case, (p(\omega)) is called the spectral density of the kernel (k).

**Remark 4.11: Eigenvalue spectrum of stationary kernels**
When a kernel (k) is stationary (i.e., a univariate function of (x - x')), its eigenfunctions (with respect to the usual Lebesgue measure) turn out to be the complex exponentials (\exp(i\omega^\top (x - x'))). In simpler terms, you can think of these exponentials as “building blocks” at different frequencies (\omega). The spectral density (p(\omega)) associated with the kernel tells you how strongly each frequency contributes, i.e., how large the corresponding eigenvalue is.

A key insight of this analysis is that the rate at which these magnitudes (p(\omega)) decay with increasing frequency (\omega) reveals the smoothness of the processes governed by the kernel. If a kernel allocates more “power” to high frequencies (meaning the spectral density decays slowly), the resulting processes will appear “rougher”. Conversely, if high-frequency components are suppressed, the process will appear “smoother”.

For an in-depth introduction to the eigenfunction analysis of kernels, refer to section 4.3 of *Gaussian Processes for Machine Learning* (Williams and Rasmussen, 2006).

**Example 4.12: Spectral density of the Gaussian kernel**
The Gaussian kernel with length scale (h) has the spectral density

[
p(\omega)
=========

\int_{\mathbb{R}^d}
k(x - x'; h), e^{-i\omega^\top (x - x')}, d(x - x')
]

using the definition of the Fourier transform (4.36)

# [

\int_{\mathbb{R}^d}
\exp!\left(-\frac{|x|^2}{2h^2} - i\omega^\top x\right), dx
]

using the definition of the Gaussian kernel (4.14)

# [

(2\pi h^2)^{d/2}
\exp!\left(-\frac{h^2 |\omega|^2}{2}\right).
\tag{4.39}
]

The key idea is now to interpret the kernel as an expectation,

[
k(x - x')
=========

\int_{\mathbb{R}^d}
p(\omega), e^{i\omega^\top (x - x')}, d\omega
]

from Equation (4.38)

# [

\mathbb{E}_{\omega \sim p}
\big[
e^{i\omega^\top (x - x')}
\big]
]

by the definition of expectation (1.19)

# [

\mathbb{E}_{\omega \sim p}
\big[
\cos(\omega^\top x - \omega^\top x')

* i \sin(\omega^\top x - \omega^\top x')
  \big]
  ]

using Euler’s formula (4.35).

Observe that as both (k) and (p) are real, convergence of the integral implies

[
\mathbb{E}_{\omega \sim p}
\big[
\sin(\omega^\top x - \omega^\top x')
\big]
= 0.
]

Hence,

# [

\mathbb{E}_{\omega \sim p}
\big[
\cos(\omega^\top x - \omega^\top x')
\big]
]

# [

\mathbb{E}*{\omega \sim p}
\mathbb{E}*{b \sim \mathrm{Unif}([0,2\pi])}
\big[
\cos((\omega^\top x + b) - (\omega^\top x' + b))
\big]
]

expanding with (b \to b)

# [

\mathbb{E}*{\omega \sim p}
\mathbb{E}*{b \sim \mathrm{Unif}([0,2\pi])}
\big[
\cos(\omega^\top x + b)\cos(\omega^\top x' + b)

* \sin(\omega^\top x + b)\sin(\omega^\top x' + b)
  \big]
  ]

using the angle subtraction identity,
[
\cos(\alpha - \beta) = \cos\alpha\cos\beta + \sin\alpha\sin\beta
]

# [

\mathbb{E}*{\omega \sim p}
\mathbb{E}*{b \sim \mathrm{Unif}([0,2\pi])}
\big[
2 \cos(\omega^\top x + b)\cos(\omega^\top x' + b)
\big]
]

using
[
\mathbb{E}_b[\cos(\alpha + b)\cos(\beta + b)]
=============================================

\mathbb{E}_b[\sin(\alpha + b)\sin(\beta + b)]
\quad \text{for } b \sim \mathrm{Unif}([0,2\pi])
]

# [

\mathbb{E}*{\omega,b}
\big[
z*{\omega,b}(x)\cdot z_{\omega,b}(x')
\big]
\tag{4.40}
]

where

[
z_{\omega,b}(x) := \sqrt{2}\cos(\omega^\top x + b),
]

[
\approx
\frac{1}{m}
\sum_{i=1}^m
z_{\omega^{(i)},b^{(i)}}(x)\cdot
z_{\omega^{(i)},b^{(i)}}(x')
\tag{4.41}
]

using Monte Carlo sampling to estimate the expectation, see Example A.6, for independent samples (\omega^{(i)} \sim p) and (b^{(i)} \sim \mathrm{Unif}([0,2\pi])),

# [

z(x)^\top z(x')
\tag{4.42}
]

where the (randomized) feature map of random Fourier features is

[
z(x) :=
\frac{1}{\sqrt{m}}
[z_{\omega^{(1)},b^{(1)}}(x), \dots, z_{\omega^{(m)},b^{(m)}}(x)]^\top.
\tag{4.43}
]

---

Intuitively, each component of the feature map (z(x)) projects (x) onto a random direction (\omega) drawn from the (inverse) Fourier transform (p(\omega)) of (k(x - x')), and wraps this line onto the unit circle in (\mathbb{R}^2). After transforming two points (x) and (x') in this way, their inner product is an unbiased estimator of (k(x - x')). The mapping (z_{\omega,b}(x) = \sqrt{2}\cos(\omega^\top x + b)) additionally rotates the circle by a random amount (b) and projects the points onto the interval ([-1,1]).

```
[Figure 4.13]
Example of random Fourier features with where the number of features m is 5 (top) and 10 (bottom), respectively. The noise-free true function is shown in black and the mean of the Gaussian process is shown in blue.
```

Rahimi et al. (2007) show that Bayesian linear regression with the feature map (z) approximates Gaussian processes with a stationary kernel:

**Theorem 4.13 (Uniform convergence of Fourier features).**
Suppose (M) is a compact subset of (\mathbb{R}^d) with diameter (\mathrm{diam}(M)). Then for a stationary kernel (k), the random Fourier features (z), and any (\varepsilon > 0) it holds that

[
\mathbb{P}
\left(
\sup_{x,x' \in M}
\big|
z(x)^\top z(x') - k(x - x')
\big|
\le \varepsilon
\right)
\ge
1 -
\left(
\frac{28\sigma_p \mathrm{diam}(M)}{\varepsilon}
\right)^2
\exp!\left(
-\frac{m\varepsilon^2}{8(d+2)}
\right),
\tag{4.44}
]

where (\sigma_p^2 := \mathbb{E}_{\omega \sim p}[\omega^\top \omega]) is the second moment of (p), (m) is the dimension of (z(x)), and (d) is the dimension of (x).
*Problem 4.8*

Note that the error probability decays exponentially fast in the dimension of the Fourier feature space.

### 4.5.3 Data Sampling

Another natural approach is to only consider a (random) subset of the training data during learning. The naive approach is to subsample uniformly at random. Not very surprisingly, we can do much better.

```
[Figure 4.14]
Inducing points u are shown as vertical dotted red lines. The noise-free true function is shown in black and the mean of the Gaussian process is shown in blue. Observe that the true function is approximated “well” around the inducing points.
```

One subsampling method is the inducing points method (Quiñonero-Candela and Rasmussen, 2005). The idea is to summarize the data around so-called inducing points.¹⁰ For now, let us consider an arbitrary set of inducing points,

[
U := {x_1, \dots, x_k}.
]

Then, the original Gaussian process can be recovered using marginalization,

[
p(f_*, f)
=========

\int_{\mathbb{R}^k}
p(f_*, f, u), du
================

\int_{\mathbb{R}^k}
p(f_* \mid u), p(f \mid u), p(u), du,
\tag{4.45}
]

using the sum rule (1.7) and product rule (1.11),

where (f := [f(x_1)\ \cdots\ f(x_n)]^\top) and (f_* := f(x_*)) at some evaluation point (x_* \in \mathcal{X}). We use (u := [f(x_1)\ \cdots\ f(x_k)]^\top \in \mathbb{R}^k) to denote the predictions of the model at the inducing points (U). Due to the marginalization property of Gaussian processes (4.1), we have that

[
u \sim \mathcal{N}(0, K_{UU}).
]

The key idea is to approximate the joint prior, assuming that (f_*) and (f) are conditionally independent given (u),

[
p(f_*, f) \approx
\int_{\mathbb{R}^k}
p(f_* \mid u), p(f \mid u), p(u), du.
\tag{4.46}
]

Here, (p(f \mid u)) and (p(f_* \mid u)) are commonly called the training conditional and the testing conditional, respectively. Still denoting the observations by
(A = {x_1, \dots, x_n}) and defining (* := {x_*}), we know, using the closed-form expression for conditional Gaussians (1.53),

[
p(f \mid u) \sim \mathcal{N}(K_{AU} K_{UU}^{-1} u,; K_{AA} - Q_{AA}),
\tag{4.47a}
]

[
p(f_* \mid u) \sim \mathcal{N}(K_{*U} K_{UU}^{-1} u,; K_{**} - Q_{**}),
\tag{4.47b}
]

where

[
Q_{ab} := K_{aU} K_{UU}^{-1} K_{Ub}.
]

Intuitively, (K_{AA}) represents the prior covariance and (Q_{AA}) represents the covariance “explained” by the inducing points.¹¹

Computing the full covariance matrix is expensive. In the following, we mention two approximations to the covariance of the training conditional (and testing conditional).

---

**Example 4.14: Subset of regressors**

The subset of regressors (SoR) approximation is defined as

[
q_{\mathrm{SoR}}(f \mid u)
:= \mathcal{N}(K_{AU} K_{UU}^{-1} u,; 0),
\tag{4.48a}
]

[
q_{\mathrm{SoR}}(f_* \mid u)
:= \mathcal{N}(K_{*U} K_{UU}^{-1} u,; 0).
\tag{4.48b}
]

Comparing to Equation (4.47), SoR simply forgets about all variance and covariance.

```
[Figure 4.15]
Comparison of SoR (top) and FITC (bottom). The inducing points u are shown as vertical dotted red lines. The noise-free true function is shown in black and the mean of the Gaussian process is shown in blue.
```

---

**Example 4.15: Fully independent training conditional**

The fully independent training conditional (FITC) approximation is defined as

[
q_{\mathrm{FITC}}(f \mid u)
:= \mathcal{N}(K_{AU} K_{UU}^{-1} u,; \mathrm{diag}{K_{AA} - Q_{AA}}),
\tag{4.49a}
]

[
q_{\mathrm{FITC}}(f_* \mid u)
:= \mathcal{N}(K_{*U} K_{UU}^{-1} u,; \mathrm{diag}{K_{**} - Q_{**}}).
\tag{4.49b}
]

In contrast to SoR, FITC keeps track of the variances but forgets about the covariance.

The computational cost for inducing point methods SoR and FITC is dominated by the cost of inverting (K_{UU}). Thus, the time complexity is cubic in the number of inducing points, but only linear in the number of data points.

---

## Discussion

This chapter introduced Gaussian processes which leverage the function-space view on linear regression to perform exact probabilistic inference with flexible, nonlinear models. A Gaussian process can be seen as a non-parametric model since it can represent an infinite-dimensional parameter space. Instead, as we saw with the representer theorem, such non-parametric (i.e., “function-space”) models are directly represented as functions of the data points. While this can make these models more flexible than a simple linear parametric model in input space, it also makes them computationally expensive as the number of data points grows. To this end, we discussed several ways of approximating Gaussian processes.

Nevertheless, for today’s internet-scale datasets, modern machine learning typically relies on large parametric models that learn features from data. These models can effectively amortize the cost of inference during training by encoding information into a fixed set of parameters. In the following chapters, we will start to explore approaches to approximate probabilistic inference that can be applied to such models.

---

## Problems

### 4.1 Feature space of Gaussian kernel

1. Show that the univariate Gaussian kernel with length scale (h = 1) implicitly defines a feature space with basis vectors

[
\phi(x) =
\begin{bmatrix}
\phi_0(x) \
\phi_1(x) \
\vdots
\end{bmatrix}
]

with

[
\phi_j(x) = \frac{1}{\sqrt{j!}} e^{-x^2/2} x^j.
]

*Hint:* Use the Taylor series expansion of the exponential function,
(e^x = \sum_{j=0}^{\infty} \frac{x^j}{j!}).

2. Note that the vector (\phi(x)) is infinite-dimensional. Thus, taking the function-space view allows us to perform regression in an infinite-dimensional feature space. What is the effective dimension when regressing (n) univariate data points with a Gaussian kernel?

---

### 4.2 Kernels on the circle

Consider a dataset ({(x_i, y_i)}_{i=1}^n) with labels (y_i \in \mathbb{R}) and inputs (x_i) which lie on the unit circle (S \subset \mathbb{R}^2). In particular, any element of (S) can be identified with points in (\mathbb{R}^2) of form ((\cos(\varrho), \sin(\varrho))) or with the respective angles (\varrho \in [0, 2\pi)).

You now want to use GP regression to learn an unknown mapping from (S) to (\mathbb{R}) using this dataset. Thus, you need a valid kernel
(k : S \times S \to \mathbb{R}).

First, we look at kernels (k) which can be understood as analogous to the Gaussian kernel.

1. You think of the “extrinsic” kernel (k_e : S \times S \to \mathbb{R}) defined by

[
k_e(\varrho, \varrho')
:=
\exp!\left(
-\frac{|x(\varrho) - x(\varrho')|^2}{2\kappa^2}
\right),
]

where (x(\varrho) := (\cos(\varrho), \sin(\varrho))).
Is (k_e) positive semi-definite for all values of (\kappa > 0)?

2. Then, you think of an “intrinsic” kernel (k_i : S \times S \to \mathbb{R}) defined by

[
k_i(\varrho, \varrho')
:=
\exp!\left(
-\frac{d(\varrho, \varrho')^2}{2\kappa^2}
\right),
]

where

[
d(\varrho, \varrho')
:=
\min(|\varrho - \varrho'|,; |\varrho - \varrho' - 2\pi|,; |\varrho - \varrho' + 2\pi|)
]

is the standard arc length distance on the circle (S).

You would now like to test whether this kernel is positive semi-definite. We pick (\kappa = 2) and compute the kernel matrix (K) for the points corresponding to the angles
({0, \pi/2, \pi, 3\pi/2}). This kernel matrix (K) has eigenvectors
((1, 1, 1, 1)) and ((-1, 1, -1, 1)).

Now compute the eigenvalue corresponding to the eigenvector
((-1, 1, -1, 1)).

3. Is (k_i) positive semi-definite for (\kappa = 2)?

4. A mathematician friend of yours suggests to you yet another kernel for points on the circle (S), called the heat kernel. The kernel itself has a complicated expression but can be accurately approximated by

[
k_h(\varrho, \varrho')
:=
\frac{1}{C_\kappa}
\left(
1 +
\sum_{l=1}^{L-1}
e^{-\kappa^2 l^2 / 2}
, 2\cos(l(\varrho - \varrho'))
\right),
]

where (L \in \mathbb{N}) controls the quality of approximation and (C_\kappa > 0) is a normalizing constant that depends only on (\kappa).

Is (k_h) positive semi-definite for all values of (\kappa > 0) and (L \in \mathbb{N})?

*Hint:* Recall that
(\cos(a - b) = \cos a \cos b + \sin a \sin b).

---

### 4.3 A Kalman filter as a Gaussian process

Next we will show that the Kalman filter from Example 3.4 can be seen as a Gaussian process. To this end, we define

[
f : \mathbb{N}_0 \to \mathbb{R}, \quad t \mapsto X_t.
\tag{4.50}
]

Assuming that (X_0 \sim \mathcal{N}(0, \sigma_0^2)) and

[
X_{t+1} := X_t + \varepsilon_t
]

with independent noise (\varepsilon_t \sim \mathcal{N}(0, \sigma_x^2)), show that

[
f \sim \mathrm{GP}(0, k_{\mathrm{KF}})
]

where

[
k_{\mathrm{KF}}(t, t')
:= \sigma_0^2 + \sigma_x^2 \min{t, t'}.
\tag{4.52}
]

This particular kernel (k(t, t') := \min{t, t'}) but over the continuous-time domain defines the Wiener process (also known as Brownian motion).

---

### 4.4 Reproducing property and RKHS norm

1. Derive the reproducing property.
   *Hint:* Use (k(x, x') = \langle k(x, \cdot), k(x', \cdot) \rangle_k).

2. Show that the RKHS norm (|\cdot|_k) is a measure of smoothness by proving that for any (f \in \mathcal{H}_k(\mathcal{X})) and (x, y \in \mathcal{X}) it holds that

[
|f(x) - f(y)|
\le
|f|_k , |k(x, \cdot) - k(y, \cdot)|_k.
]

---

### 4.5 Representer theorem

With this, we can now derive the representer theorem (4.20).

*Hint:* Recall

1. the reproducing property
   (f(x) = \langle f, k(x, \cdot) \rangle_k) with (k(x, \cdot) \in \mathcal{H}_k(\mathcal{X})) which holds for all (f \in \mathcal{H}_k(\mathcal{X})) and (x \in \mathcal{X}), and

2. that the norm after projection is smaller or equal the norm before projection.

Then decompose (f) into parallel and orthogonal components with respect to
(\mathrm{span}{k(x_1, \cdot), \dots, k(x_n, \cdot)}).

---

## 4.6 MAP estimate of Gaussian processes

Let us denote by (A = {x_1, \dots, x_n}) the set of training points. We will now show that the MAP estimate of GP regression corresponds to the solution of the regularized linear regression problem in the RKHS stated in Equation (4.21):

[
\hat{f}
=======

\arg\min_{f \in \mathcal{H}_k(\mathcal{X})}

* \log p(y_{1:n} \mid x_{1:n}, f)

- \frac{1}{2}|f|_k^2.
  ]

In the following, we abbreviate (K = K_{AA}). We will also assume that the GP has a zero mean function.

1. Show that Equation (4.21) is equivalent to

[
\hat{\varepsilon}
=================

\arg\min_{\varepsilon \in \mathbb{R}^n}
|y - K\varepsilon|_2^2 + \lambda |\varepsilon|_K^2
\tag{4.53}
]

for some (\lambda > 0) which is also known as kernel ridge regression. Determine (\lambda).

2. Show that Equation (4.53) with the (\lambda) determined in (1) is equivalent to the MAP estimate of GP regression.

*Hint:* Recall from Equation (4.6) that the MAP estimate at a point (x_*) is

[
\mathbb{E}[f_* \mid x_*, X, y]
==============================

k_{x_*,A}^\top (K + \sigma_n^2 I)^{-1} y.
]

---

## 4.7 Gradient of the marginal likelihood

In this exercise, we derive Equation (4.30).

Recall that we were considering a dataset ((X, y)) of noise-perturbed evaluations
(y_i = f(x_i) + \varepsilon_i) where (\varepsilon_i \sim \mathcal{N}(0, \sigma_n^2)) and (f) is an unknown function. We make the hypothesis
(f \sim \mathrm{GP}(0, k_\theta)) with a zero mean function and the covariance function (k_\theta). We are interested in finding the hyperparameters (\theta) that maximize the marginal likelihood (p(y \mid X, \theta)).

1. Derive Equation (4.30).

*Hint:* You can use the following identities:

(a) for any invertible matrix (M),

[
\frac{\partial}{\partial \varrho_j} M^{-1}
==========================================

* M^{-1} \frac{\partial M}{\partial \varrho_j} M^{-1}
  \tag{4.54}
  ]

(b) for any symmetric positive definite matrix (M),

[
\frac{\partial}{\partial \varrho_j} \log \det(M)
================================================

\operatorname{tr}
!\left(
M^{-1} \frac{\partial M}{\partial \varrho_j}
\right).
\tag{4.55}
]

2. Assume now that the covariance function for the noisy targets (i.e., including the noise contribution) can be expressed as

[
k_{y,\theta}(x, x')
===================

\varrho_0 \tilde{k}(x, x')
]

where (\tilde{k}) is a valid kernel independent of (\varrho_0).
Show that (\frac{\partial}{\partial \varrho_0} \log p(y \mid X, \theta) = 0) admits a closed-form solution for (\varrho_0) which we denote by (\varrho_0^\star).

3. How should the optimal parameter (\varrho_0^\star) be scaled if we scale the labels (y) by a scalar (s)?

---

## 4.8 Uniform convergence of Fourier features

In this exercise, we will prove Theorem 4.13.

Let
(s(x, x') := z(x)^\top z(x')) and
(f(x, x') := s(x, x') - k(x, x')).
Observe that both functions are shift invariant, and we will therefore denote them as univariate functions with argument (\Delta := x - x' \in M_\Delta). Notice that our goal is to bound the probability of the event

[
\sup_{\Delta \in M_\Delta} |f(\Delta)| \le \varepsilon.
]

1. Show that for all (\Delta \in M_\Delta),

[
\mathbb{P}(|f(\Delta)| \le \varepsilon)
\ge
1 - 2\exp!\left(-\frac{m\varepsilon^2}{4}\right).
]

What we have derived in (1) is known as a pointwise convergence guarantee. However, we are interested in bounding the uniform convergence over the compact set (M_\Delta).

Our approach will be to “cover” the compact set (M_\Delta) using (T) balls of radius (r) whose centers we denote by ({\Delta_i}_{i=1}^T). It can be shown that this is possible for some
(T \le (4,\mathrm{diam}(M)/r)^d). It can furthermore be shown that

[
\bigcap_{i=1}^T
\left{
|f(\Delta_i)| < \frac{\varepsilon}{2}
\right}
;\wedge;
|\nabla f(\Delta^\star)|*2 < \frac{\varepsilon}{2r}
;\Rightarrow;
\sup*{\Delta \in M_\Delta} |f(\Delta)| < \varepsilon
]

where (\Delta^\star = \arg\max_{\Delta \in M_\Delta} |\nabla f(\Delta)|_2).

2. Prove

[
\mathbb{P}
!\left(
|\nabla f(\Delta^\star)|_2
\le
\frac{\varepsilon}{2r}
\right)
\ge
1 -
\left(
\frac{2r\sigma_p}{\varepsilon}
\right)^2.
]

*Hint:* Recall that the random Fourier feature approximation is unbiased, i.e.,
(\mathbb{E}[s(\Delta)] = k(\Delta)).

3. Prove

[
\mathbb{P}
!\left(
\bigcap_{i=1}^T
\left{
|f(\Delta_i)| \le \frac{\varepsilon}{2}
\right}
\right)
\ge
1 -
2T\exp!\left(-\frac{m\varepsilon^2}{16}\right).
]

4. Combine the results from (2) and (3) to prove Theorem 4.13.

*Hint:* You may use that

(a) (\alpha r^{-d} + \beta r^2 = 2\beta^{\frac{d}{d+2}} \alpha^{\frac{2}{d+2}}) for
(r = (\alpha/\beta)^{1/(d+2)}), and

(b) (\sigma_p \mathrm{diam}(M)/\varepsilon \ge 1).

5. Show that for the Gaussian kernel (4.14),

[
\sigma_p^2 = \frac{d}{h^2}.
]

*Hint:* First show (\sigma_p^2 = -\operatorname{tr}(H_\Delta k(0))).

---

## 4.9 Subset of regressors

1. Using an SoR approximation, prove the following:

[
q_{\mathrm{SoR}}(f, f_*)
========================

\mathcal{N}
!\left(
\begin{bmatrix} f \ f_* \end{bmatrix};
0,
\begin{bmatrix}
Q_{AA} & Q_{A*} \
Q_{*A} & Q_{**}
\end{bmatrix}
\right)
\tag{4.56}
]

[
q_{\mathrm{SoR}}(f_* \mid y)
============================

\mathcal{N}
!\left(
f_*;
Q_{*A}\tilde{Q}*{AA}^{-1} y,;
Q*{**} - Q_{*A}\tilde{Q}*{AA}^{-1} Q*{A*}
\right)
\tag{4.57}
]

where (\tilde{Q}*{ab} := Q*{ab} + \sigma_n^2).

2. Derive that the resulting model is a degenerate Gaussian process with covariance function

[
k_{\mathrm{SoR}}(x, x')
=======================

k_{x,U}^\top K_{UU}^{-1} k_{x',U}.
\tag{4.58}
]

---
