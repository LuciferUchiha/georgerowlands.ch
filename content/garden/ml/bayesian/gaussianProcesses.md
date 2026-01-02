---
title: Gaussian Processes
type: docs
weight: 2
---

<!--
Rough Outline:
link to bayesian learning and non-linear regression 

somehow from the kernel trick in BLR motivate GPs as a distribution over functions if we generalize to infinite domains we get GP. Hence GPs can be seen as infinite dimension multivariate gaussians?


Gaussian process (GP) = normal distribution over functions
Finite marginals are multivariate Gaussians
Closed form formulae for Bayesian posterior update exist
Parameterized by covariance function kernel = Covariance between function values

(infinite) set of random variables, indexed by some set there exists function mu and covariance function k such that any finite subset of these random variables has a multivariate normal distribution with mean given by mu and covariance given by k evaluated at the corresponding indices

Marginals of GP? typically interested in these? why? p(f(x)) = N (f(x); μ(x); k(x, x))

function requirements for being a valid covariance function? symmetric and positive semi-definite
Kernel function k encodes assumptions about correlation!

linear kernel = bayesian linear regression
polynomial kernel = bayesian polynomial regression what does this function look like?
sin kernel = periodic functions
squared exponential / gaussian / rbf kernel = smooth functions we have bandwidth hyperparameter that controls smoothness lengthscale? 
exponential kernel = less smooth functions again lengthscale hyperparameter

matern kernel = generalization of squared exponential and exponential with additional smoothness hyperparameter

composite kernels = sum and product of kernels is also a kernel combine to get more complex kernels

effects of kernel hyperparameters? lengthscale controls how quickly functions vary. variance controls vertical variation if h is too small very wiggly functions if h is too large very smooth functions?
if sigma_p^2 is too small functions close then too simple of model and close to zero mean if sigma_p^2? Why is this?
if sigma_n^2 is too large then very thick noise around function
and opposiests of these

stationary and isotropic kernels?

How do we make predictions with GPs? prior is GP with mean function and covariance function. then condition on observed data to get posterior GP. show formulae for posterior mean and covariance function

Closed form formulas for prediction!
èosterior covariance k’ does not depend on y_A

why can we without loss of generality assume zero mean function? show

sample from GP? forward sampling?

how to choose hyperparameters of kernel?cross validaiton on predictive performance?  \hat{f}_i = argmax_f p(y_train | f, theta, x_train) p(f | theta_i) for candidate theta_i
This is just MAP esimate? GP Posterio mean? then optimal theta is argmax_theta p(y_val | \hat{f}_i, theta_i, x_val)

The Bayesian perspective provides an alternative
approach: Maximize the marginal likelihood of the data

some integral. why doesthis wor? break down likelihood and prior for when parameters underfit, overfit and are just right?

Optimizing marginal likelihood for GPs we get log p(y | X, theta) and split into "goodness of fit" and "volumne" of plausible functions, i.e likelihood and prior
usually differentiable wrt to hyperparameters so can use gradient based optimization

show calculations of gradients, but there can be local optima.

Maximizing marginal likelihood is an example of an
Empirical Bayes method – estimating a prior
distribution from data
Integrating (rather than optimizing) over the unknown
function helps guarding against overfitting
Other possibilities exist:
Can place hyperprior on parameters of the prior and obtain
MAP estimate (corresponds to a regularization term)
Can integrate out the hyperprior (but also has params ...)

prediction with GP requires solving linear system which is O(n^3) can be expensive for large n in contrast bayesian linear regression is O(nd^2) and we can even do online updates. Sparse GPs and approximations to help with this?

Kernel function approximation

construct explicit “low-dimensional” feature
map that approximates the true kernel function not go into a lot of detail but such as random fourier features and nystrom approximation Computational cost: O(nm^2 + m^3) instead of O(n^3) with m << n?

local methods not go into detail but give idea

Covariance functions that decay with distance of
points (e.g., RBF, Matern, kernels) lend themselves to
local computations
To make a prediction at
point x, only condition on
points x’ close to x

inducing points method “Summarize” data via function values of f at a set u of
m inducing points

training conditional and testing conditional something with Q matrix

subset of regressors method replaces training conditional. Can be shown that resulting model is a degenerate GP covariacne function:

$$
k_{SR}(x, x') = k(x, Z) k(Z, Z)^{-1} k(Z, x')
$$

where Z are the inducing points?

Fully independent training conditional (FITC) method replaces testing conditional. Can be shown that resulting model is a GP:

$$
p_{FITC}(f | u) = \prod_{i=1}^{n} p(f_i | u) = N(K_{f,u} K_{u,u}^{-1} u, diag(K_{f,f} - Q_{f,f}))
$$

The computational cost for inducing point methods
SoR and FITC is dominated by the cost of inverting <2,2

§ Thus, it is cubic in the number of inducing points, but
linear in the number of data points
-->

Gaussian Processes (GPs) extend the concept of Bayesian Linear Regression to infinite-dimensional feature spaces. Instead of defining a prior over weights $w$ (Weight-Space View), we define a prior directly over functions $f$ (Function-Space View).

## Definition

A **Gaussian Process** is a collection of random variables, any finite number of which have a joint Gaussian distribution.

$$
f(x) \sim \mathcal{GP}(m(x), k(x, x'))
$$

It is fully specified by:
1.  **Mean Function** $m(x) = \mathbb{E}[f(x)]$ (often assumed to be 0 for simplicity).
2.  **Covariance Function (Kernel)** $k(x, x') = \mathbb{E}[(f(x) - m(x))(f(x') - m(x'))]$.

Intuitively, a GP defines a distribution over functions. If we sample from a GP, we get a function.

## Marginals

For any finite set of input points $X = \{x_1, \dots, x_n\}$, the corresponding function values $f = [f(x_1), \dots, f(x_n)]^T$ follow a Multivariate Gaussian distribution:

$$
p(f \mid X) = \mathcal{N}(f \mid \mu, K)
$$

where $\mu_i = m(x_i)$ and $K_{ij} = k(x_i, x_j)$.

## Covariance Functions (Kernels)

The kernel function $k(x, x')$ encodes our assumptions about the function we are modeling, such as smoothness and periodicity. It must be **symmetric** and **positive semi-definite** (PSD).

### Common Kernels

1.  **Linear Kernel**: $k(x, x') = x^T x'$. Corresponds to Bayesian Linear Regression.
2.  **Squared Exponential (RBF) Kernel**:
    $$
    k(x, x') = \sigma_f^2 \exp\left( -\frac{\|x - x'\|^2}{2l^2} \right)
    $$
    Produces smooth, infinitely differentiable functions.
3.  **Matern Kernel**: Generalization of RBF. Controls smoothness via parameter $\nu$.
    -   $\nu = 1/2$: Exponential Kernel (very rough, like Ornstein-Uhlenbeck process).
    -   $\nu \to \infty$: RBF Kernel.
4.  **Periodic Kernel**: For repeating patterns.

### Hyperparameters
Kernels have hyperparameters that control the shape of the functions:
-   **Lengthscale ($l$)**: Controls how quickly the function varies. Small $l$ $\to$ wiggly function. Large $l$ $\to$ smooth function.
-   **Signal Variance ($\sigma_f^2$)**: Controls the vertical scale (amplitude) of the function.

## Posterior and Predictions

We are interested in predicting the function value $f_*$ at a test point $x_*$, given training data $(X, y)$ where $y = f(X) + \varepsilon$ and $\varepsilon \sim \mathcal{N}(0, \sigma_n^2 I)$.

The joint distribution of the observed data $y$ and the test output $f_*$ is:

$$
\begin{bmatrix} y \\ f_* \end{bmatrix} \sim \mathcal{N}\left( \begin{bmatrix} 0 \\ 0 \end{bmatrix}, \begin{bmatrix} K(X, X) + \sigma_n^2 I & K(X, x_*) \\ K(x_*, X) & k(x_*, x_*) \end{bmatrix} \right)
$$

Using the conditioning rule for Gaussians, the posterior $p(f_* \mid x_*, X, y)$ is Gaussian:

$$
f_* \mid x_*, X, y \sim \mathcal{N}(\bar{f}_*, \mathbb{V}[f_*])
$$

**Predictive Mean:**
$$
\bar{f}_* = K(x_*, X) [K(X, X) + \sigma_n^2 I]^{-1} y
$$
This can be seen as a linear combination of kernel functions centered at the training points (Representer Theorem).

**Predictive Variance:**
$$
\mathbb{V}[f_*] = k(x_*, x_*) - K(x_*, X) [K(X, X) + \sigma_n^2 I]^{-1} K(X, x_*)
$$
The variance is the prior variance minus the information gained from the data. Note that the predictive variance depends only on the inputs $X$, not the targets $y$.

## Hyperparameter Optimization

How do we choose the kernel hyperparameters $\theta = \{l, \sigma_f, \sigma_n\}$?
We maximize the **Log Marginal Likelihood**:

$$
\log p(y \mid X, \theta) = \underbrace{-\frac{1}{2} y^T K_y^{-1} y}_{\text{Data Fit}} \underbrace{-\frac{1}{2} \log |K_y|}_{\text{Complexity Penalty}} - \frac{n}{2} \log 2\pi
$$

where $K_y = K(X, X) + \sigma_n^2 I$.
-   **Data Fit**: Encourages the model to fit the data well.
-   **Complexity Penalty**: Penalizes complex models (determinant of covariance matrix measures volume of function space).

This objective is usually non-convex, so we use gradient-based optimization (e.g., L-BFGS, Adam).

## Computational Complexity

The main bottleneck in GP regression is the inversion of the covariance matrix $K_y$, which is $N \times N$.
-   **Training Cost**: $O(N^3)$ (Cholesky decomposition).
-   **Prediction Cost**: $O(N)$ for mean, $O(N^2)$ for variance per test point.

This makes standard GPs intractable for datasets larger than $N \approx 10,000$.

## Approximations

To scale GPs to large datasets, we use approximations.

### Inducing Points Methods
We summarize the dataset using a smaller set of $M$ **inducing points** ($M \ll N$).
-   **Subset of Regressors (SoR)**: Approximates the kernel.
-   **Fully Independent Training Conditional (FITC)**: More accurate approximation that preserves diagonal variance.
-   **Sparse Variational GP (SVGP)**: Variational inference framework to jointly optimize inducing points and hyperparameters. Reduces complexity to $O(NM^2)$.

### Kernel Approximations
-   **Random Fourier Features**: Maps inputs to a finite-dimensional feature space using random projections, approximating the RBF kernel. Allows using linear regression solvers ($O(N)$).

