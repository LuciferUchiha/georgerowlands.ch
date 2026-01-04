---
title: Gaussian Processes
type: docs
weight: 2
---

In the [Bayesian Linear Regression](/garden/ml/bayesian/bayesianLearning/) section, we developed a complete probabilistic framework for regression. We started with linear models $f(\mathbf{x}) = \mathbf{w}^T \mathbf{x}$, placed a Gaussian prior on the weights $\mathbf{w} \sim \mathcal{N}(\mathbf{0}, \sigma_p^2 \mathbf{I}_d)$ where $\mathbf{w} \in \mathbb{R}^d$, and derived closed-form expressions for the posterior distribution over weights. This gave us not just point predictions, but full predictive distributions that quantify our uncertainty.

We then faced a fundamental limitation: **linear functions are too restrictive for most real-world problems**. To model non-linear relationships while preserving the mathematical convenience of linear regression, we introduced the idea of first transforming inputs using a fixed feature map $\boldsymbol{\phi}(\mathbf{x})$, and then applying linear regression in this new feature space. One popular choice is to use [basis function expansions](/garden/ml/bayesian/bayesianLearning/#non-linear-regression-and-kernels):

$$
f(\mathbf{x}) = \mathbf{w}^T \boldsymbol{\phi}(\mathbf{x})
$$

where $\boldsymbol{\phi}(\mathbf{x}) = [\phi_1(\mathbf{x}), \phi_2(\mathbf{x}), \ldots, \phi_M(\mathbf{x})]^T$ maps the $d$-dimensional input to an $M$-dimensional feature space. For polynomial regression, we might use $\boldsymbol{\phi}(x) = [1, x, x^2, \ldots, x^P]^T$ for a polynomial of degree $P$. This transforms a non-linear problem in the original space into a linear problem in the feature space.

However, we quickly encountered a **scaling problem**: for a polynomial of degree $P$ in $d$ dimensions, the number of basis functions grows as $\binom{d+P}{P}$ which is $O(d^P)$. For instance, with $d=100$ input dimensions and degree $P=3$, the feature dimension $M$ exceeds 170,000. This leads to:

1. **Computational and Storage Issues**: Working with feature vectors of dimension $M \gg n$ is costly, and we can't even store the $n \times M$ design matrix $\boldsymbol{\Phi}$.
2. **Overfitting**: High-dimensional models with $M$ parameters (where $M \gg n$) tend to overfit.
3. **Numerical Instability**: Computing inverses of large matrices or dealing with high degree polynomials can lead to numerical issues.

## The Kernel Trick

The breakthrough came from observing that in the dual formulation of Ridge Regression (and many Bayesian methods), **the features only appear through their inner products** $\boldsymbol{\phi}(\mathbf{x})^T \boldsymbol{\phi}(\mathbf{x}')$. This led to the **kernel trick**, where we define a kernel function:

$$
k(\mathbf{x}, \mathbf{x}') = \boldsymbol{\phi}(\mathbf{x})^T \boldsymbol{\phi}(\mathbf{x}')
$$

that computes the inner product in feature space **directly**, without ever computing $\boldsymbol{\phi}(\mathbf{x})$ explicitly. For example, the polynomial kernel $k(\mathbf{x}, \mathbf{x}') = (\mathbf{x}^T \mathbf{x}' + c)^P$ implicitly computes the inner product of all polynomial features up to degree $P$.

Recall from the Bayesian Linear Regression notes that the kernelized predictive mean, so the mean of the posterior predictive distribution for a new input $\mathbf{x}_*$, for Kernel Ridge Regression is:

$$
\hat{f}(\mathbf{x}_*) = \mathbf{k}_*^T (\mathbf{K} + \sigma_n^2 \mathbf{I}_n)^{-1} \mathbf{y}
$$

where $\mathbf{K} \in \mathbb{R}^{n \times n}$ is the kernel matrix with $K_{ij} = k(\mathbf{x}_i, \mathbf{x}_j)$ and $\mathbf{k}_* = [k(\mathbf{x}_*, \mathbf{x}_1), \ldots, k(\mathbf{x}_*, \mathbf{x}_n)]^T \in \mathbb{R}^n$ is the vector of kernel evaluations between the test point and the $n$ training points.

## From Weight-Space to Function-Space View

Now comes the key conceptual leap that leads us to Gaussian Processes. In Bayesian Linear Regression, we placed a prior on the **weights** $\mathbf{w}$ and derived the induced distribution over **functions** $f(\mathbf{x}) = \mathbf{w}^T \boldsymbol{\phi}(\mathbf{x})$. Specifically, we start with the prior we had on weights $\mathbf{w} \sim \mathcal{N}(\mathbf{0}, \sigma_p^2 \mathbf{I}_M)$ where $\mathbf{w} \in \mathbb{R}^M$, then for any set of $n$ inputs $\mathbf{X} = \{\mathbf{x}_1, \ldots, \mathbf{x}_n\}$, the vector of function values $\mathbf{f} = [f(\mathbf{x}_1), \ldots, f(\mathbf{x}_n)]^T$ can be written as:

$$
\mathbf{f} = \boldsymbol{\Phi} \mathbf{w}
$$

where $\boldsymbol{\Phi} \in \mathbb{R}^{n \times M}$ is the design matrix with rows $\boldsymbol{\phi}(\mathbf{x}_i)^T$. Since $\mathbf{f}$ is a linear transformation of a Gaussian, it is also Gaussian:

- **Mean**: $\mathbb{E}[\mathbf{f}] = \boldsymbol{\Phi} \mathbb{E}[\mathbf{w}] = \mathbf{0}$
- **Covariance**: $\text{Cov}[\mathbf{f}] = \boldsymbol{\Phi} \text{Cov}[\mathbf{w}] \boldsymbol{\Phi}^T = \sigma_p^2 \boldsymbol{\Phi} \boldsymbol{\Phi}^T = \mathbf{K} \in \mathbb{R}^{n \times n}$

where $K_{ij} = \sigma_p^2 \boldsymbol{\phi}(\mathbf{x}_i)^T \boldsymbol{\phi}(\mathbf{x}_j) = k(\mathbf{x}_i, \mathbf{x}_j)$. Thus:

$$
\mathbf{f} \mid \mathbf{X} \sim \mathcal{N}(\mathbf{0}, \mathbf{K})
$$

This is remarkable: **the prior over weights induces a prior over functions**, and this prior is fully characterized by the kernel function $k(\mathbf{x}, \mathbf{x}')$. The kernel defines the covariance between function values at any two points:

$$
\text{Cov}[f(\mathbf{x}), f(\mathbf{x}')] = k(\mathbf{x}, \mathbf{x}')
$$

This observation suggests a fundamental shift in perspective. Instead of reasoning about feature maps $\boldsymbol{\phi}(\mathbf{x})$ (which may be infinite-dimensional) or weight vectors $\mathbf{w}$ (which live in feature space), we can reason **directly about functions** using only the kernel function. The kernel implicitly defines a (possibly infinite-dimensional) feature space, but we never need to work with it explicitly. This is the **function-space view**, and it leads us to Gaussian Processes.

## Gaussian Processes

A **Gaussian Process (GP)** is a collection of random variables, where any finite number of which have a joint Gaussian distribution. More formally a Gaussian process is an infinite set of random variables $\{f_{\mathbf{x}} : \mathbf{x} \in \mathcal{X}\}$ indexed by some input space $\mathcal{X} \subseteq \mathbb{R}^d$ such that:

1. Any finite subset $\{f_{\mathbf{x}_1}, \ldots, f_{\mathbf{x}_m}\}$ is jointly Gaussian so that for any $m$ and any choice of points $A \subseteq \mathcal{X}$ with $A = \{\mathbf{x}_1, \ldots, \mathbf{x}_m\}$, the random vector:

$$
\mathbf{f}_A = [f_{\mathbf{x}_1}, \ldots, f_{\mathbf{x}_m}]^T \sim \mathcal{N}(\boldsymbol{\mu}_A, \mathbf{K}_{AA})
$$

modelled by some mean vector $\boldsymbol{\mu}_A \in \mathbb{R}^m$ and covariance matrix $\mathbf{K}_{AA} \in \mathbb{R}^{m \times m}$, where:

$$
\boldsymbol{\mu}_A = \begin{bmatrix} \mu(\mathbf{x}_1) \\ \vdots \\ \mu(\mathbf{x}_m) \end{bmatrix}, \quad
\mathbf{K}_{AA} = \begin{bmatrix} 
k(\mathbf{x}_1, \mathbf{x}_1) & \cdots & k(\mathbf{x}_1, \mathbf{x}_m) \\ 
\vdots & \ddots & \vdots \\ 
k(\mathbf{x}_m, \mathbf{x}_1) & \cdots & k(\mathbf{x}_m, \mathbf{x}_m) 
\end{bmatrix}
$$

where $\mu : \mathcal{X} \to \mathbb{R}$ is the mean function and $k : \mathcal{X} \times \mathcal{X} \to \mathbb{R}$ is the kernel (covariance) function.
2. The collection is **consistent under marginalization**: if you take a joint distribution for $n$ variables and marginalize out one of them, you recover the joint distribution for the remaining $n-1$ variables:

$$
p(f_{\mathbf{x}_1}, \ldots, f_{\mathbf{x}_{n-1}}) = \int p(f_{\mathbf{x}_1}, \ldots, f_{\mathbf{x}_n}) \, df_{\mathbf{x}_n}
$$

The consistency property ensures that the GP is well-defined as we consider different finite subsets of the infinite collection. Importantly, this means we can define a GP by specifying only the finite-dimensional distributions for all possible finite subsets, as long as they satisfy the marginalization property. Therefore a GP is completely specified by its **mean function** $\mu : \mathcal{X} \to \mathbb{R}$ and **kernel function** (covariance) $k : \mathcal{X} \times \mathcal{X} \to \mathbb{R}$:

$$
\begin{aligned}
\mu(\mathbf{x}) &= \mathbb{E}[f(\mathbf{x})] \\
k(\mathbf{x}, \mathbf{x}') &= \text{Cov}[f(\mathbf{x}), f(\mathbf{x}')] = \mathbb{E}[(f(\mathbf{x}) - \mu(\mathbf{x}))(f(\mathbf{x}') - \mu(\mathbf{x}'))]
\end{aligned}
$$

We write:

$$
f \sim \mathcal{GP}(\mu, k)
$$

Just as a multivariate Gaussian $\mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})$ defines a distribution over vectors in $\mathbb{R}^n$, a Gaussian Process defines a distribution over **functions** $f : \mathcal{X} \to \mathbb{R}$. You can think of a GP as an "infinite-dimensional Gaussian" where:
- The mean vector $\boldsymbol{\mu} \in \mathbb{R}^n$ becomes the mean function $\mu(\mathbf{x})$
- The covariance matrix $\boldsymbol{\Sigma} \in \mathbb{R}^{n \times n}$ becomes the kernel function $k(\mathbf{x}, \mathbf{x}')$

At any location $\mathbf{x}$ in the domain, the GP gives us a distribution over the function value $f(\mathbf{x})$. The mean function tells us the expected value, and the kernel tells us how function values at different locations are correlated. This is the key property that makes GPs tractable: although we're dealing with an infinite-dimensional object (a distribution over functions), any finite-dimensional "slice" is just a multivariate Gaussian, which we know how to work with.

For notational simplicity, the mean function is commonly taken to be zero: $\mu(\mathbf{x}) = 0$. This might seem restrictive, but it is not. If we have a GP with non-zero mean $f \sim \mathcal{GP}(\mu, k)$, we can always write:

$$
f(\mathbf{x}) = \mu(\mathbf{x}) + g(\mathbf{x})
$$

where $g \sim \mathcal{GP}(0, k)$ is a zero-mean GP. So we can:
1. Model the mean function $\mu(\mathbf{x})$ separately (e.g., as a parametric function or another GP)
2. Apply the zero-mean GP machinery to the residuals $y - \mu(\mathbf{x})$

In practice, we often subtract the empirical mean from the data before applying GP regression, or we use a constant or linear mean function that we fit alongside the kernel hyperparameters.

## Kernel Functions

As already shown above the mean can be assumed to be zero without loss of generality. Thus, a Gaussian Process is fully specified by its kernel function $k(\mathbf{x}, \mathbf{x}')$ and is the heart of a Gaussian Process. It encodes our assumptions about the functions we expect to see: how smooth they are, whether they're periodic, how quickly they vary, and more. 

Intuitively, the kernel measures the **similarity** or **correlation** between function values at two points:

$$
k(\mathbf{x}, \mathbf{x}') = \text{Cov}[f(\mathbf{x}), f(\mathbf{x}')]
$$

If $k(\mathbf{x}, \mathbf{x}')$ is large and positive, then $f(\mathbf{x})$ and $f(\mathbf{x}')$ tend to take similar values. If $k(\mathbf{x}, \mathbf{x}')$ is small, the function values are nearly independent. Most commonly used kernels are designed so that nearby points have high correlation (the function is "smooth"), while distant points have low correlation. But in general, the choice of kernel encodes our prior beliefs about the function's behavior and choosing an appropriate kernel is one of the most important decisions in GP modeling. However, not every function $k : \mathcal{X} \times \mathcal{X} \to \mathbb{R}$ can serve as a kernel. For $k$ to define a valid covariance function, it must satisfy:

1. **Symmetry**: $k(\mathbf{x}, \mathbf{x}') = k(\mathbf{x}', \mathbf{x})$ for all $\mathbf{x}, \mathbf{x}' \in \mathcal{X}$
2. **Positive semi-definiteness**: For any set of points $\{\mathbf{x}_1, \ldots, \mathbf{x}_n\}$, the Gram matrix $\mathbf{K}$ with $K_{ij} = k(\mathbf{x}_i, \mathbf{x}_j)$ is positive semi-definite

These conditions ensure that $\mathbf{K}$ is a valid covariance matrix (symmetric and PSD). Recall that a matrix $\mathbf{K}$ is positive semi-definite if $\mathbf{a}^T \mathbf{K} \mathbf{a} \geq 0$ for all vectors $\mathbf{a} \in \mathbb{R}^n$ or equivalently if all its eigenvalues are non-negative. **Why these conditions?** Symmetry follows from the symmetry of covariance: $\text{Cov}[f(\mathbf{x}), f(\mathbf{x}')] = \text{Cov}[f(\mathbf{x}'), f(\mathbf{x})]$. Positive semi-definiteness ensures that variances are non-negative: for any linear combination $g = \sum_i a_i f(\mathbf{x}_i)$, we have $\text{Var}[g] = \mathbf{a}^T \mathbf{K} \mathbf{a} \geq 0$.

In addition, we often require kernels to be **positive definite** rather than just semi-definite, so $\mathbf{a}^T \mathbf{K} \mathbf{a} > 0$ for all non-zero vectors $\mathbf{a}$. This ensures the kernel matrix is invertible, which is important for GP inference.

### Stationarity and Isotropy

Before introducing specific kernels, let's define two important properties that characterize many common kernels:

A kernel $k : \mathbb{R}^d \times \mathbb{R}^d \to \mathbb{R}$ is called:

- **Stationary** (or shift-invariant) if there exists a function $\tilde{k}$ such that:

$$
k(\mathbf{x}, \mathbf{x}') = \tilde{k}(\mathbf{x} - \mathbf{x}')
$$

so the kernel depends only on the **relative positions** of points, not their absolute locations. The process "looks the same" everywhere in the input space. This is appropriate when we believe the same statistical behavior holds across the entire domain.

- **Isotropic** if there exists a function $\tilde{k}$ such that:
  
$$
k(\mathbf{x}, \mathbf{x}') = \tilde{k}(\|\mathbf{x} - \mathbf{x}'\|_2)
$$

Isotropy goes further, the kernel depends only on the **distance** between points, treating all directions equally. There is no preferred orientation or axis in the input space. Therefore it follows that an isotropic kernel is a special case of a stationary kernel where the dependence is only on the magnitude of the difference vector, not its direction. So:

$$
\text{Isotropic} \implies \text{Stationary}
$$

### Linear Kernel

Now let's explore some commonly used kernel functions. Starting with the simplest, the **linear kernel**:

$$
k(\mathbf{x}, \mathbf{x}') = \sigma_p^2 \, \mathbf{x}^T \mathbf{x}'
$$

The linear kernel corresponds to standard Bayesian Linear Regression. The implicit feature map is $\boldsymbol{\phi}(\mathbf{x}) = \mathbf{x}$ (the identity). Functions drawn from a GP with a linear kernel are linear functions (hyperplanes). Importantly, the linear kernel is **not stationary** since $k(\mathbf{x}, \mathbf{x}) = \sigma_p^2 \|\mathbf{x}\|_2^2$ depends on the absolute position of $\mathbf{x}$. Points at the origin have zero prior variance, while points far from the origin have large variance—the process does not "look the same" everywhere.

{{< figure 
    src="/images/ml/gpLinearKernel.webp"
    caption="Samples from a GP prior with a linear kernel. Notice how all sampled functions are linear (straight lines through the origin)."
    alt="Samples from a GP prior with a linear kernel. Notice how all sampled functions are linear (straight lines through the origin)."
    width="400"
>}}

### Polynomial Kernel

Naturally extending the linear kernel, we have the **polynomial kernel**:

$$
k(\mathbf{x}, \mathbf{x}') = (\sigma_p^2 \mathbf{x}^T \mathbf{x}' + c)^P
$$

This implicitly computes the inner product of all polynomial features up to degree $P$. With $P=2$ and $c=1$ (a constant offset ensuring the kernel includes lower-degree terms), the implicit feature map for $\mathbf{x} \in \mathbb{R}^2$ is $\boldsymbol{\phi}(\mathbf{x}) = [1, \sqrt{2}x_1, \sqrt{2}x_2, x_1^2, \sqrt{2}x_1 x_2, x_2^2]^T$. Since the linear kernel is a special case with $P=1$, the polynomial kernel is also **not stationary**.

{{< figure 
    src="/images/ml/gpPolynomialKernel.webp"
    caption="Samples from GP priors with polynomial kernels of different degrees. Higher degrees allow for more complex polynomial functions."
    alt="Samples from GP priors with polynomial kernels of different degrees. Higher degrees allow for more complex polynomial functions."
        width="400"
>}}

### Periodic Kernel

If we expect the underlying function to exhibit periodic behavior, such as seasonal or cyclical patterns in the data (e.g., daily temperature variations), we can use the **periodic kernel**:

$$
k(\mathbf{x}, \mathbf{x}') = \sigma_f^2 \exp\left(-\frac{2\sin^2(\pi|\mathbf{x} - \mathbf{x}'|/p)}{\ell^2}\right)
$$

Here, $p$ is the period of the function, $\ell$ is the lengthscale controlling how quickly correlations decay, and $\sigma_f^2$ is the signal variance. Intuitively, the lengthscale determines the smoothness of variations within each period: small $\ell$ allows rapid changes, while large $\ell$ produces smoother variations. The signal variance $\sigma_f^2$ controls the vertical scale, with larger values allowing greater deviations from the mean.

The periodic kernel is stationary (in one dimension) and produces functions that repeat every $p$ units along the input axis.

{{< figure 
    src="/images/ml/gpPeriodicKernel.png"
    caption="Samples from a GP prior with a periodic kernel. The functions exhibit repeating patterns with period $p$."
    alt="Samples from a GP prior with a periodic kernel. The functions exhibit repeating patterns with period p."
>}}

### Squared Exponential (Gaussian/RBF) Kernel

Next we have the **Squared Exponential (SE)** kernel, also known as the **Gaussian** or **Radial Basis Function (RBF)** kernel:

$$
k(\mathbf{x}, \mathbf{x}') = \sigma_f^2 \exp\left(-\frac{\|\mathbf{x} - \mathbf{x}'\|^2}{2\ell^2}\right)
$$

This is the most widely used kernel. It produces **infinitely differentiable** (very smooth) functions. The hyperparameters are:

- **Lengthscale** $\ell$: Controls how quickly the function varies. Large $\ell$ means the function changes slowly (smooth, broad variations). Small $\ell$ means the function can change rapidly (wiggly).
- **Signal variance** $\sigma_f^2$: Controls the vertical scale of variations. Large $\sigma_f^2$ means the function can take values far from zero.

**Feature space interpretation**: The RBF kernel corresponds to an **infinite-dimensional** feature space. To see why, consider extending polynomial regression to all degrees. A naive feature map would be:

$$
\tilde{\boldsymbol{\phi}}(\mathbf{x}) = [\mathbf{x}^{\boldsymbol{\alpha}}]_{\boldsymbol{\alpha} \in \mathbb{N}^d}
$$

where $\mathbf{x}^{\boldsymbol{\alpha}} = x_1^{\alpha_1} \cdots x_d^{\alpha_d}$ are all possible monomials. However, the inner product $\tilde{\boldsymbol{\phi}}(\mathbf{x})^T \tilde{\boldsymbol{\phi}}(\mathbf{x}')$ **diverges** for most $\mathbf{x}, \mathbf{x}'$ because the infinite sum of monomial products grows unboundedly.

The solution is to introduce a **normalized feature map** that ensures convergence:

$$
\boldsymbol{\phi}(\mathbf{x}) = \exp\left(-\frac{\|\mathbf{x}\|_2^2}{2\ell^2}\right) \left[\frac{(\mathbf{x}/\ell)^{\boldsymbol{\alpha}}}{\sqrt{\boldsymbol{\alpha}!}}\right]_{\boldsymbol{\alpha} \in \mathbb{N}^d}
$$

The exponential prefactor ensures each component decays as $\|\mathbf{x}\|$ grows, while the factorial normalization $\sqrt{\boldsymbol{\alpha}!}$ ensures the sum converges. Computing the inner product:

$$
\boldsymbol{\phi}(\mathbf{x})^T \boldsymbol{\phi}(\mathbf{x}') = \exp\left(-\frac{\|\mathbf{x}\|_2^2}{2\ell^2}\right) \exp\left(-\frac{\|\mathbf{x}'\|_2^2}{2\ell^2}\right) \sum_{\boldsymbol{\alpha} \in \mathbb{N}^d} \frac{(\mathbf{x}/\ell)^{\boldsymbol{\alpha}} (\mathbf{x}'/\ell)^{\boldsymbol{\alpha}}}{\boldsymbol{\alpha}!}
$$

Using the multinomial series expansion $\sum_{\boldsymbol{\alpha} \in \mathbb{N}^d} \frac{\mathbf{u}^{\boldsymbol{\alpha}} \mathbf{v}^{\boldsymbol{\alpha}}}{\boldsymbol{\alpha}!} = \exp(\mathbf{u}^T \mathbf{v})$, this simplifies to:

$$
\boldsymbol{\phi}(\mathbf{x})^T \boldsymbol{\phi}(\mathbf{x}') = \exp\left(-\frac{\|\mathbf{x}\|_2^2 + \|\mathbf{x}'\|_2^2 - 2\mathbf{x}^T\mathbf{x}'}{2\ell^2}\right) = \exp\left(-\frac{\|\mathbf{x} - \mathbf{x}'\|_2^2}{2\ell^2}\right)
$$

Thus, the RBF kernel naturally emerges as the inner product in an infinite-dimensional space of normalized polynomial features. The kernel trick allows us to work with this infinite-dimensional representation implicitly!

{{< figure 
    src="/images/ml/gpGaussianKernel.webp"
    caption="Samples from a GP prior with a squared exponential (RBF) kernel. The functions are very smooth due to infinite differentiability."
    alt="Samples from a GP prior with a squared exponential (RBF) kernel. The functions are very smooth due to infinite differentiability."
    width="400"
>}}

Just like the periodic kernel, the squared exponential kernel has a lengthscale hyperparameter $\ell$ that determines how quickly correlations decay with distance. A larger lengthscale means points farther apart remain strongly correlated, producing smoother functions. A smaller lengthscale allows correlations to drop off quickly, permitting more rapid changes. The signal variance $\sigma_f^2$ controls the overall vertical scale of the function values.

{{< figure 
    src="/images/ml/gpGaussianLengthscale.png"
    caption="Effect of lengthscale $\ell$ on functions sampled from a GP with RBF kernel. Smaller lengthscales (left) produce more rapid variations, while larger lengthscales (right) yield smoother functions."
    alt="Effect of lengthscale on functions sampled from a GP with RBF kernel. Smaller lengthscales (left) produce more rapid variations, while larger lengthscales (right) yield smoother functions."
    width="500"
>}}

The name "squared exponential" comes from the squared distance $\|\mathbf{x} - \mathbf{x}'\|^2$ in the exponent. Why is it also called "Gaussian"? If we fix one point $\mathbf{x}'$ and view $k(\mathbf{x}, \mathbf{x}')$ as a function of $\mathbf{x}$, it has the shape of a Gaussian (normal) distribution centered at $\mathbf{x}'$. The kernel value decreases exponentially as we move away from $\mathbf{x}'$, following the familiar bell curve.

{{< figure 
    src="/images/ml/gpKernels.png"
    caption="Comparison of different kernel functions. The shape of the kernel determines how function values at different points are correlated."
    alt="Comparison of different kernel functions. The shape of the kernel determines how function values at different points are correlated."
    width="550"
>}}

This also gives us intuition for why the RBF kernel produces smooth functions: points close together in input space have high covariance, so the function values must be similar. This strong correlation enforces smoothness. This is also why the squared exponential kernel is stationary and isotropic: it depends only on the squared distance $\|\mathbf{x} - \mathbf{x}'\|^2$ between points, treating all locations and directions equally.

### Laplace (Absolute Exponential) Kernel

While the RBF kernel produces very smooth functions, sometimes we need to model functions that are continuous but not differentiable—functions with a "rough" or "jagged" appearance. For this, we use the **Laplace (Exponential)** kernel:

$$
k(\mathbf{x}, \mathbf{x}') = \sigma_f^2 \exp\left(-\frac{\|\mathbf{x} - \mathbf{x}'\|_2}{\ell}\right)
$$

The key difference from the squared exponential kernel is in the exponent: we use the **L2 norm** $\|\mathbf{x} - \mathbf{x}'\|_2$ rather than the squared L2 norm $\|\mathbf{x} - \mathbf{x}'\|^2$. The hyperparameters retain the same interpretation. The Laplace kernel is both stationary and isotropic, but yields rougher functions than the RBF kernel.

**Why does the Laplace kernel produce rough functions?** The smoothness of GP sample paths is determined by how the kernel behaves near $r = 0$ (where $r = \|\mathbf{x} - \mathbf{x}'\|$). For the **RBF kernel**, we have $k(r) = \sigma_f^2 e^{-r^2/2\ell^2}$, which has $k'(0) = 0$ (the kernel is "flat" at the origin). This flatness means the covariance decreases very gently for nearby points, forcing function values to be nearly identical—resulting in smooth, differentiable functions.

For the **Laplace kernel**, we have $k(r) = \sigma_f^2 e^{-r/\ell}$, which has $k'(0) = -\sigma_f^2/\ell \neq 0$ (the kernel has a "cusp" or "kink" at the origin). This non-zero derivative means the covariance drops off linearly (not quadratically) as points separate, allowing more variation between nearby function values. Mathematically, a kernel must be at least twice differentiable at $r=0$ for the GP to have differentiable sample paths. The Laplace kernel is not differentiable at $r=0$, so its sample paths are continuous but nowhere differentiable—they have sharp corners everywhere.

{{< figure 
    src="/images/ml/gpLaplacianKernel.webp"
    caption="Samples from a GP prior with a Laplace kernel. Unlike the smooth RBF samples, these functions have a rough, jagged appearance."
    alt="Samples from a GP prior with a Laplace kernel. Unlike the smooth RBF samples, these functions have a rough, jagged appearance."
>}}

#### Matérn Kernel

The **Matérn kernel** is a versatile family of kernels that generalizes both the RBF and Laplace kernels. It introduces an additional hyperparameter $\nu$ that controls the smoothness of the resulting functions:

$$
k(\mathbf{x}, \mathbf{x}'; \nu, \ell) = \sigma_f^2 \frac{2^{1-\nu}}{\Gamma(\nu)} \left(\frac{\sqrt{2\nu}\|\mathbf{x} - \mathbf{x}'\|}{\ell}\right)^\nu K_\nu\left(\frac{\sqrt{2\nu}\|\mathbf{x} - \mathbf{x}'\|}{\ell}\right)
$$

where $\Gamma$ is the gamma function and $K_\nu$ is the modified Bessel function of the second kind.

The Matérn kernel family **interpolates between the Laplace and RBF kernels** through the smoothness parameter $\nu$:
- $\nu = 1/2$: Equivalent to the Laplace kernel (continuous but not differentiable)
- $\nu = 3/2$: Once differentiable
- $\nu = 5/2$: Twice differentiable
- $\nu \to \infty$: Equivalent to the RBF kernel (infinitely differentiable)

The Matérn kernel is popular in practice because:
1. It provides explicit control over smoothness
2. Real-world functions are often smooth but not infinitely so
3. The RBF kernel can be "too smooth" for some applications

the Matérn $\nu = 3/2$ and $\nu = 5/2$ are popular choices for learning functions that are not infinitely differentiable (as assumed by the RBF kernel)

$$
k_{3/2}(r) = \sigma_f^2 \left(1 + \frac{\sqrt{3}r}{\ell}\right) \exp\left(-\frac{\sqrt{3}r}{\ell}\right)
$$

$$
k_{5/2}(r) = \sigma_f^2 \left(1 + \frac{\sqrt{5}r}{\ell} + \frac{5r^2}{3\ell^2}\right) \exp\left(-\frac{\sqrt{5}r}{\ell}\right)
$$

where $r = \|\mathbf{x} - \mathbf{x}'\|_2$.

### Rational Quadratic Kernel

The **Rational Quadratic (RQ)** kernel can be seen as a scale mixture of RBF kernels with different lengthscales. It is defined as:

$$
k(\mathbf{x}, \mathbf{x}') = \sigma_f^2 \left(1 + \frac{\|\mathbf{x} - \mathbf{x}'\|_2^2}{2\alpha \ell^2}\right)^{-\alpha}
$$

where $\alpha > 0$ is a shape parameter controlling the relative weighting of large-scale and small-scale variations. As $\alpha \to \infty$, the RQ kernel approaches the RBF kernel. For smaller $\alpha$, the kernel allows for more variation at multiple lengthscales. So graphically, you might see notice smaller "bumps" superimposed on larger trends which are due to the mixture of lengthscales.

### Composing Kernels

One of the powerful properties of kernels is that they can be **combined to create new valid kernels**. This allows us to build complex models from simple building blocks.

Given valid kernels $k_1$ and $k_2$, the following are also valid kernels:

1. **Sum**: $k(\mathbf{x}, \mathbf{x}') = k_1(\mathbf{x}, \mathbf{x}') + k_2(\mathbf{x}, \mathbf{x}')$
2. **Product**: $k(\mathbf{x}, \mathbf{x}') = k_1(\mathbf{x}, \mathbf{x}') \cdot k_2(\mathbf{x}, \mathbf{x}')$
3. **Scaling**: $k(\mathbf{x}, \mathbf{x}') = c \cdot k_1(\mathbf{x}, \mathbf{x}')$ for any $c > 0$
4. **Function composition**: $k(\mathbf{x}, \mathbf{x}') = g(k_1(\mathbf{x}, \mathbf{x}'))$ for certain functions $g$ (e.g., polynomials with positive coefficients, or $g = \exp$)

**Why does addition work?** If $f_1 \sim \mathcal{GP}(0, k_1)$ and $f_2 \sim \mathcal{GP}(0, k_2)$ are independent, then their sum $f = f_1 + f_2 \sim \mathcal{GP}(0, k_1 + k_2)$. The covariance of a sum of independent variables is the sum of their covariances. The same can be shown for the other operations. This also makes sense if you think about the polynomial kernel being built from sums and products of linear kernels.

Intuitively, composing kernels allows us to model more complex behaviors:
- **Sum** acts like an OR: the kernel is large if *either* $k_1$ or $k_2$ is large. Use this to model a function as the sum of components (e.g., trend + periodicity + noise).
- **Product** acts like an AND: the kernel is large only if *both* $k_1$ and $k_2$ are large. Use this for modulated effects (e.g., locally periodic functions where the periodic pattern varies across the domain).

{{< figure 
    src="/images/ml/gpComposition.jpg"
    caption="Composing kernels allows modeling complex structure. Here, combining kernels captures different aspects of the underlying function."
    alt="Composing kernels allows modeling complex structure. Here, combining kernels captures different aspects of the underlying function."
>}}

{{< callout type="example" title="Example: Trend + Seasonality" >}}
To model data with both a long-term trend and seasonal variation, we might use a linear kernel plus a periodic kernel and a small RBF kernel for noise:

$$
k(\mathbf{x}, \mathbf{x}') = k_{\text{linear}}(\mathbf{x}, \mathbf{x}') + k_{\text{periodic}}(\mathbf{x}, \mathbf{x}') + k_{\text{RBF}}(\mathbf{x}, \mathbf{x}')
$$
{{< /callout >}}

### Reproducing Kernel Hilbert Spaces

We can characterize the precise class of functions that can be modeled by a Gaussian Process with a given kernel function. This corresponding function space is called a **Reproducing Kernel Hilbert Space (RKHS)**. Understanding RKHS reveals a deep connection between GPs and regularized optimization.

Recall that GPs maintain a posterior distribution $f \mid \mathbf{X}, \mathbf{y}$ over functions. It turns out that the MAP estimate $\hat{f}$ (the posterior mean) corresponds to the solution of a regularized optimization problem in the RKHS. This duality is analogous to the relationship between the MAP estimate of Bayesian Linear Regression and Ridge Regression.

**Definition (Reproducing Kernel Hilbert Space):** Given a kernel $k : \mathcal{X} \times \mathcal{X} \to \mathbb{R}$, its corresponding RKHS is the space of functions:

$$
\mathcal{H}_k(\mathcal{X}) := \left\{ f(\cdot) = \sum_{i=1}^n \alpha_i k(\mathbf{x}_i, \cdot) : n \in \mathbb{N}, \mathbf{x}_i \in \mathcal{X}, \alpha_i \in \mathbb{R} \right\}
$$

The inner product of the RKHS is defined as:

$$
\langle f, g \rangle_k := \sum_{i=1}^n \sum_{j=1}^{n'} \alpha_i \alpha'_j k(\mathbf{x}_i, \mathbf{x}'_j)
$$

where $g(\cdot) = \sum_{j=1}^{n'} \alpha'_j k(\mathbf{x}'_j, \cdot)$, which induces the norm $\|f\|_k = \sqrt{\langle f, f \rangle_k}$.

The norm $\|f\|_k$ can be interpreted as measuring the "smoothness" or "complexity" of $f$. Functions with large RKHS norm are more "wiggly" or complex, while functions with small norm are smoother.

A key property is the **reproducing property**: for all $\mathbf{x} \in \mathcal{X}$ and $f \in \mathcal{H}_k(\mathcal{X})$:

$$
f(\mathbf{x}) = \langle f(\cdot), k(\mathbf{x}, \cdot) \rangle_k
$$

This means that evaluating $f$ at $\mathbf{x}$ is equivalent to taking an inner product with the kernel function $k(\mathbf{x}, \cdot)$, which acts as a "feature map" in the RKHS.

The **Representer Theorem** characterizes solutions to regularized optimization problems in RKHSs. Let $k$ be a kernel and $\lambda > 0$. For any loss function $L(f(\mathbf{x}_1), \ldots, f(\mathbf{x}_n))$ that depends on $f$ only through its evaluations at the training points, any minimizer of:

$$
\hat{f} \in \arg\min_{f \in \mathcal{H}_k(\mathcal{X})} L(f(\mathbf{x}_1), \ldots, f(\mathbf{x}_n)) + \lambda \|f\|_k^2
$$

admits a representation of the form:

$$
\hat{f}(\mathbf{x}) = \sum_{i=1}^n \hat{\alpha}_i k(\mathbf{x}, \mathbf{x}_i) \quad \text{for some } \hat{\boldsymbol{\alpha}} \in \mathbb{R}^n
$$

This means that solutions to regularized optimization over the generally **infinite-dimensional** space $\mathcal{H}_k(\mathcal{X})$ can always be represented as a finite linear combination of $n$ kernel evaluations at the training points!

**Connection to GP Regression:** The MAP estimate of a Gaussian Process corresponds to the solution of:

$$
\hat{f} = \arg\min_{f \in \mathcal{H}_k(\mathcal{X})} -\log p(\mathbf{y} \mid \mathbf{X}, f) + \frac{1}{2}\|f\|_k^2
$$

The first term measures the quality of fit (likelihood), while the regularization term $\frac{1}{2}\|f\|_k^2$ limits the complexity of $\hat{f}$, preventing overfitting. This regularization is necessary because in expressive RKHSs, many functions can interpolate the training data perfectly.

This connection shows that the kernel function $k$ generalizes the inner product of feature maps to feature spaces of possibly "infinite dimensionality." Because solutions can be represented as linear combinations of kernel evaluations at training points, GPs remain computationally tractable even when modeling functions over infinite-dimensional feature spaces.

## Learning and Inference

Now we arrive at the core of Gaussian Process regression: given observed data, how do we learn from our data and make predictions about function values at new locations?

We start by observing $n$ training points with inputs $\mathbf{X} = \{\mathbf{x}_1, \ldots, \mathbf{x}_n\}$ where each $\mathbf{x}_i \in \mathbb{R}^d$, and noisy labels $\mathbf{y} = [y_1, \ldots, y_n]^T \in \mathbb{R}^n$. We assume the labels are corrupted by Gaussian noise:

$$
y_i = f(\mathbf{x}_i) + \varepsilon_i, \quad \varepsilon_i \sim \mathcal{N}(0, \sigma_n^2)
$$

We place a GP prior on the function: $f \sim \mathcal{GP}(\mu, k)$. For simplicity, we'll use a zero mean function $\mu(\mathbf{x}) = 0$. Our goal is to predict the function value $f_*$ at a new test point $\mathbf{x}_*$.

The key insight is that the prior tells us the joint distribution of **all** function values—both at training points and at the test point. Let's denote:
- $\mathbf{f} = [f(\mathbf{x}_1), \ldots, f(\mathbf{x}_n)]^T \in \mathbb{R}^n$: function values at the $n$ training points
- $f_* = f(\mathbf{x}_*)$: function value at the test point

Since $\mathbf{f}$ and $f_*$ are jointly drawn from the GP, they are jointly Gaussian. The training observations $\mathbf{y} = \mathbf{f} + \boldsymbol{\varepsilon}$ are also jointly Gaussian with $f_*$ (a linear transformation of Gaussians is Gaussian). Therefre the joint distribution is:

$$
\begin{bmatrix} \mathbf{y} \\ f_* \end{bmatrix} \mid \mathbf{X}, \mathbf{x}_* \sim \mathcal{N}\left( \begin{bmatrix} \mathbf{0} \\ 0 \end{bmatrix}, \begin{bmatrix} \mathbf{K} + \sigma_n^2 \mathbf{I}_n & \mathbf{k}_* \\ \mathbf{k}_*^T & k(\mathbf{x}_*, \mathbf{x}_*) \end{bmatrix} \right)
$$

where:
- $\mathbf{K} \in \mathbb{R}^{n \times n}$ is the kernel matrix of training points: $K_{ij} = k(\mathbf{x}_i, \mathbf{x}_j)$
- $\mathbf{k}_* = [k(\mathbf{x}_1, \mathbf{x}_*), \ldots, k(\mathbf{x}_n, \mathbf{x}_*)]^T \in \mathbb{R}^n$ is the vector of covariances between training points and the test point
- $\sigma_n^2 \mathbf{I}_n$ accounts for the observation noise on $\mathbf{y}$ (but not on $f_*$, which is noise-free)

To make predictions, we **condition** on the observed data $\mathbf{y}$. Using the standard formulas for conditioning a multivariate Gaussian (see [Properties of the Gaussian](/garden/ml/bayesian/bayesianLearning/#properties-of-the-gaussian)), we obtain:

$$
f_* \mid \mathbf{X}, \mathbf{y}, \mathbf{x}_* \sim \mathcal{N}(\mu_*, \sigma_*^2)
$$

### Predictive Mean

The **predictive mean** $\mu_*$ is given by:

$$
\mu_* = \mathbf{k}_*^T (\mathbf{K} + \sigma_n^2 \mathbf{I}_n)^{-1} \mathbf{y}
$$

just like in the kernelized predictive mean from Bayesian Linear Regression, we can rewrite the predictive mean as:

$$
\mu_* = \sum_{i=1}^n \alpha_i k(\mathbf{x}_i, \mathbf{x}_*)
$$

where $\boldsymbol{\alpha} = (\mathbf{K} + \sigma_n^2 \mathbf{I}_n)^{-1} \mathbf{y} \in \mathbb{R}^n$. This reveals several important properties:

1. **The prediction is a weighted combination of kernel evaluations**: Each training point "votes" on the prediction, weighted by $\alpha_i$ and by its similarity to $\mathbf{x}_*$ (measured by the kernel).
2. **Identical to Kernel Ridge Regression**: If we identify $\lambda = \sigma_n^2$, this is exactly the Kernel Ridge Regression predictor! GP regression and kernel ridge regression give the same point predictions—but GP regression also provides uncertainty quantification and allows us to extend to a full posterior over functions.
3. **Noiseless limit**: In the limiting case $\sigma_n^2 \to 0$, we get $\mu_* = \mathbf{k}_*^T \mathbf{K}^{-1} \mathbf{y}$, which is equivalent to the kernelized Ordinary Least Squares Estimator (OLSE). In this case, the GP posterior mean interpolates exactly through all training points.

### Predictive Variance

The predictive distribution also gives us uncertainty about our prediction through the **predictive variance**:

$$
\sigma_*^2 = k(\mathbf{x}_*, \mathbf{x}_*) - \mathbf{k}_*^T (\mathbf{K} + \sigma_n^2 \mathbf{I}_n)^{-1} \mathbf{k}_*
$$

The predictive variance has a beautiful interpretation:

$$
\sigma_*^2 = \underbrace{k(\mathbf{x}_*, \mathbf{x}_*)}_{\text{prior variance}} - \underbrace{\mathbf{k}_*^T (\mathbf{K} + \sigma_n^2 \mathbf{I}_n)^{-1} \mathbf{k}_*}_{\text{variance reduction from data}}
$$

From this we can also see several important properties:

1. **The posterior variance is always less than or equal to the prior variance**: The second term is always non-negative (it's a quadratic form with a PSD matrix), so $\sigma_*^2 \leq k(\mathbf{x}_*, \mathbf{x}_*)$. Observing data can only reduce uncertainty.
2. **The variance does NOT depend on $\mathbf{y}$**: Only on the input locations $\mathbf{X}$ and $\mathbf{x}_*$. This is a fundamental property of Gaussian distributions—the posterior precision depends on where we observe, not what we observe.
3. **Variance is low near training data**: When $\mathbf{x}_*$ is close to training points, $\mathbf{k}_*$ has large entries and the variance reduction is substantial. When $\mathbf{x}_*$ is far from all training points, $\mathbf{k}_*$ has small entries and the variance approaches the prior variance—reflecting high uncertainty in unexplored regions.

The predictive variance naturally decomposes into epistemic and aleatoric components. As we observe more data, the epistemic uncertainty (model uncertainty) decreases, while the aleatoric uncertainty (observation noise) remains constant—it represents irreducible randomness in the data.

{{< figure 
    src="/images/ml/gpUncertainty.gif"
    caption="As more data is observed, epistemic uncertainty (green shaded region) decreases while aleatoric uncertainty (observation noise) remains constant."
    alt="As more data is observed, epistemic uncertainty (green shaded region) decreases while aleatoric uncertainty (observation noise) remains constant."
>}}

### Posterior over Functions

It turns out that the entire posterior distribution over functions is also a Gaussian Process! Specifically, the posterior GP is:

$$
f \mid \mathbf{X}, \mathbf{y} \sim \mathcal{GP}(\mu', k')
$$

where the **posterior mean function** is:

$$
\mu'(\mathbf{x}) = \mathbf{k}_{\mathbf{x}}^T (\mathbf{K} + \sigma_n^2 \mathbf{I}_n)^{-1} \mathbf{y}
$$

and the **posterior covariance function** is:

$$
k'(\mathbf{x}, \mathbf{x}') = k(\mathbf{x}, \mathbf{x}') - \mathbf{k}_{\mathbf{x}}^T (\mathbf{K} + \sigma_n^2 \mathbf{I}_n)^{-1} \mathbf{k}_{\mathbf{x}'}
$$

where $\mathbf{k}_{\mathbf{x}} = [k(\mathbf{x}_1, \mathbf{x}), \ldots, k(\mathbf{x}_n, \mathbf{x})]^T \in \mathbb{R}^n$.

This means after observing data, we still have a GP, just with an updated mean function (no longer zero) and a reduced covariance function. So just like in bayesian linear regression, where we were able to update our prior over weights to a posterior over weights, in GP regression we update our prior over functions to a posterior over functions. Allowing us have an online distribution over functions that captures our updated beliefs after seeing data.

A key benefit is that we can sample functions from both the prior and posterior GP, allowing us to visualize uncertainty in our function estimates. Notice that posterior samples pass through (or near) the training points, and the posterior variance $k'(\mathbf{x}, \mathbf{x})$ is low near training points and high far away.

{{< figure 
    src="/images/ml/gpInference.png"
    caption="GP regression: the posterior mean (blue line) interpolates the training data, while the shaded region shows the predictive uncertainty. Samples from the posterior (thin lines) all pass near the observed points."
    alt="GP regression: the posterior mean (blue line) interpolates the training data, while the shaded region shows the predictive uncertainty. Samples from the posterior (thin lines) all pass near the observed points."
>}}

## Sampling from a Gaussian Process

Often we want to visualize the GP by drawing sample functions from the prior or posterior. Since we can only represent a function at a finite number of points, we choose a set of $m$ test points $\mathbf{X}_* = \{\mathbf{x}_1^*, \ldots, \mathbf{x}_m^*\}$ and sample the vector $\mathbf{f}_* = [f(\mathbf{x}_1^*), \ldots, f(\mathbf{x}_m^*)]^T \in \mathbb{R}^m$. Sampling from the prior shows what functions the GP considers plausible *before* seeing any data. Sampling from the posterior shows the uncertainty remaining *after* conditioning on observations. 

There are two common methods for sampling from the multivariate Gaussian distribution:

### Cholesky Decomposition

To sample $\mathbf{f} \sim \mathcal{N}(\boldsymbol{\mu}, \mathbf{K})$ where $\mathbf{f} \in \mathbb{R}^m$, we use the fact that if $\boldsymbol{\varepsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I}_m)$, then:

$$
\mathbf{f} = \boldsymbol{\mu} + \mathbf{L} \boldsymbol{\varepsilon}
$$

where $\mathbf{L}$ is the **Cholesky decomposition** of $\mathbf{K}$: $\mathbf{K} = \mathbf{L}\mathbf{L}^T$ with $\mathbf{L}$ lower triangular.

**Algorithm**:
1. Compute the $m \times m$ kernel matrix $\mathbf{K}$ at the test points
2. Compute Cholesky decomposition $\mathbf{K} = \mathbf{L}\mathbf{L}^T$
3. Sample $\boldsymbol{\varepsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I}_m)$
4. Return $\mathbf{f} = \boldsymbol{\mu} + \mathbf{L}\boldsymbol{\varepsilon}$

This method is efficient and numerically stable. The complexity is $O(m^3)$ for the Cholesky decomposition of the $m \times m$ matrix. This is the most common way to sample from a GP.

### Forward Sampling

Another method is **forward sampling** (also called sequential sampling). Here, we sample each function value one at a time, conditioning on previously sampled values using the chain rule of probability:

$$
p(f_1, \ldots, f_m) = p(f_1) \prod_{i=2}^m p(f_i \mid f_1, \ldots, f_{i-1})
$$

We can sample points **sequentially**, each time conditioning on previously sampled values. 

**Algorithm**:
1. Sample $f_1 \sim p(f_1) = \mathcal{N}(\mu(\mathbf{x}_1), k(\mathbf{x}_1, \mathbf{x}_1))$
2. Sample $f_2 \sim p(f_2 \mid f_1)$ (using GP conditioning formulas)
3. Sample $f_3 \sim p(f_3 \mid f_1, f_2)$
4. And so on...

This is also known as **ancestral sampling**. Each conditional is Gaussian, so sampling is straightforward. However, each step requires updating the conditional distribution, which involves solving a linear system. The overall complexity is therefore also $O(m^3)$.

{{< figure 
    src="/images/ml/gpSampling.png"
    caption="Samples from a GP prior (left) and posterior (right). The posterior samples are constrained to pass through the observed data points."
    alt="Samples from a GP prior (left) and posterior (right). The posterior samples are constrained to pass through the observed data points."
>}}

## Hyperparameter Optimization

The kernel function typically has hyperparameters (lengthscale, signal variance, etc.) that we collectively denote $\theta$. The noise variance $\sigma_n^2$ is also a hyperparameter. How do we choose these?

### Cross-Validation Approach

A standard frequentist approach to model selection is to split our data $\mathcal{D}$ into separate training and validation sets:
- Training set: $\mathcal{D}_{\text{train}} = \{(\mathbf{x}_i^{\text{train}}, y_i^{\text{train}})\}_{i=1}^n$
- Validation set: $\mathcal{D}_{\text{val}} = \{(\mathbf{x}_i^{\text{val}}, y_i^{\text{val}})\}_{i=1}^m$

For each candidate hyperparameter setting $\theta_j$, we optimize the model using the training set. This is typically done by computing the posterior mean (which is also the MAP estimate as shown in [bayesian linear regression](/garden/ml/bayesian/bayesianLearning/#map-estimate-and-ridge-regression)):

$$
\hat{f}_j = \mu'(\mathbf{x}) = \mathbf{k}_{\mathbf{x}}^{\text{train}} (\mathbf{K}^{\text{train}} + \sigma_n^2 \mathbf{I}_n)^{-1} \mathbf{y}^{\text{train}}
$$

We then score each $\theta_j$ according to how well $\hat{f}_j$ predicts the validation data:

$$
\hat{\theta} = \arg\max_{\theta_j} p(\mathbf{y}^{\text{val}} \mid \mathbf{X}^{\text{val}}, \hat{f}_j)
$$

This ensures that the model $\hat{f}_j$ does not depend on $\mathcal{D}_{\text{val}}$, preventing overfitting to the validation set.

**Why separate training and validation?** Minimizing the training error alone can lead to overfitting, since both the loss function and the model depend on the same data. By using independent sets, the validation error provides an unbiased estimate of the true generalization performance. More formally, for reasonably large $m$, minimizing the empirical risk on the validation set:

$$
\frac{1}{m} \sum_{i=1}^m \ell(y_i^{\text{val}} \mid \mathbf{x}_i^{\text{val}}, \hat{f}_j) \approx \mathbb{E}_{(\mathbf{x},y) \sim P}[\ell(y \mid \mathbf{x}, \hat{f}_j)]
$$

approximates minimizing the **population risk** via Monte Carlo sampling.

While this approach is effective at preventing overfitting compared to using the same data for training and evaluation, it still collapses the full posterior uncertainty over $f$ into a single point estimate.

### Maximizing the Marginal Likelihood

Instead of optimizing hyperparameters for a specific point estimate $\hat{f}$, the Bayesian approach optimizes across **all realizations** of $f$ by maximizing the **marginal likelihood** (also called the **evidence**):

$$
\hat{\theta}_{\text{MLE}} = \arg\max_\theta p(\mathbf{y} \mid \mathbf{X}, \theta)
$$

Notice the key difference from cross-validation: we use the marginal likelihood $p(\mathbf{y} \mid \mathbf{X}, \theta)$ which integrates over all possible functions, rather than evaluating a single point estimate on held-out data.

Using the definition of marginal likelihood via Bayes' rule and the product rule, we can expand:

$$
p(\mathbf{y} \mid \mathbf{X}, \theta) = \int p(\mathbf{y} \mid \mathbf{f}, \mathbf{X}) p(\mathbf{f} \mid \mathbf{X}, \theta) \, d\mathbf{f}
$$

This integral asks: "averaging over all functions $f$ weighted by the prior, how likely is the observed data $\mathbf{y}$?" 

For GPs, this integral has a **closed-form solution**! Since both the likelihood $p(\mathbf{y} \mid \mathbf{f}, \mathbf{X}) = \mathcal{N}(\mathbf{y} \mid \mathbf{f}, \sigma_n^2 \mathbf{I}_n)$ and prior $p(\mathbf{f} \mid \mathbf{X}, \theta) = \mathcal{N}(\mathbf{f} \mid \mathbf{0}, \mathbf{K}_\theta)$ are Gaussian, the marginal distribution is also Gaussian:

$$
\mathbf{y} \mid \mathbf{X}, \theta \sim \mathcal{N}(\mathbf{0}, \mathbf{K}_\theta + \sigma_n^2 \mathbf{I}_n)
$$

where $\mathbf{K}_\theta \in \mathbb{R}^{n \times n}$ emphasizes that the kernel matrix depends on the hyperparameters $\theta$. We denote $\mathbf{K}_y := \mathbf{K}_\theta + \sigma_n^2 \mathbf{I}_n$ for brevity.

### The Log Marginal Likelihood

Continuing from the optimization objective, we take the negative logarithm (which converts the $\arg\max$ to $\arg\min$):

$$
\hat{\theta}_{\text{MLE}} = \arg\max_\theta \mathcal{N}(\mathbf{y} \mid \mathbf{0}, \mathbf{K}_y) = \arg\min_\theta \left[ \frac{1}{2} \mathbf{y}^T \mathbf{K}_y^{-1} \mathbf{y} + \frac{1}{2} \log |\mathbf{K}_y| + \frac{n}{2} \log 2\pi \right]
$$

Since the last term is independent of $\theta$, we can drop it:

$$
\hat{\theta}_{\text{MLE}} = \arg\min_\theta \left[ \frac{1}{2} \mathbf{y}^T \mathbf{K}_y^{-1} \mathbf{y} + \frac{1}{2} \log |\mathbf{K}_y| \right]
$$

Equivalently, we can write the log marginal likelihood (to be maximized) as:

$$
\log p(\mathbf{y} \mid \mathbf{X}, \theta) = -\frac{1}{2} \mathbf{y}^T \mathbf{K}_y^{-1} \mathbf{y} - \frac{1}{2} \log |\mathbf{K}_y| - \frac{n}{2} \log 2\pi
$$

This objective has a beautiful interpretation:

$$
\log p(\mathbf{y} \mid \mathbf{X}, \theta) = \underbrace{-\frac{1}{2} \mathbf{y}^T \mathbf{K}_y^{-1} \mathbf{y}}_{\text{Data Fit}} \underbrace{- \frac{1}{2} \log |\mathbf{K}_y|}_{\text{Complexity Penalty}} - \frac{n}{2} \log 2\pi
$$

- **Data Fit Term**: Measures the "alignment" of $\mathbf{y}$ with $\mathbf{K}_y$—how well the model explains the observed data. Small values indicate a good fit.
- **Complexity Penalty Term**: Measures the "volume" of the model class. The determinant $|\mathbf{K}_y|$ grows with the total variance of the model. Models with large $\mathbf{K}_y$ (high variance, more flexible) have large determinants and are penalized.

This automatic trade-off embodies **Occam's Razor**: among models that fit the data well, prefer simpler ones. This is the same principle we saw in [Bayesian Linear Regression](/garden/ml/bayesian/bayesianLearning/#hyperparameter-optimization) when selecting the prior and noise variances.

{{< figure 
    src="/images/ml/gpMarginalLikelihoodData.png"
    caption="Schematic illustration of the marginal likelihood for simple, intermediate, and complex models across all possible datasets. The intermediate model assigns highest probability to the actually observed data."
    alt="Schematic illustration of the marginal likelihood for simple, intermediate, and complex models across all possible datasets. The intermediate model assigns highest probability to the actually observed data."
>}}

### Gradient-Based Optimization

The log marginal likelihood is typically differentiable with respect to the hyperparameters $\theta$ (for common kernels like RBF, Matérn, etc.). The gradient can be computed in closed form:

$$
\frac{\partial}{\partial \theta_j} \log p(\mathbf{y} \mid \mathbf{X}, \theta) = \frac{1}{2} \text{tr}\left( (\boldsymbol{\alpha} \boldsymbol{\alpha}^T - \mathbf{K}_y^{-1}) \frac{\partial \mathbf{K}_y}{\partial \theta_j} \right)
$$

where $\boldsymbol{\alpha} = \mathbf{K}_y^{-1} \mathbf{y}$. This allows us to use gradient-based optimization methods (gradient descent, L-BFGS, etc.) to find optimal hyperparameters.

{{< callout type="warning" title="Non-Convexity and Local Optima" >}}
The marginal likelihood objective is generally **non-convex** in the hyperparameters. There may be multiple local optima, especially when the model is expressive (many hyperparameters) or when data is limited.

For example, the marginal likelihood landscape might have two local optima corresponding to qualitatively different models:
- A model with **smaller lengthscale** that fits the data closely, belonging to a more flexible function class
- A model with **larger lengthscale** that explains more variation through noise $\sigma_n^2$, using a smoother function class

Both may achieve similar marginal likelihood values despite representing very different interpretations of the data.

In practice, it is common to:
- Use multiple random restarts and select the best solution
- Initialize hyperparameters based on data statistics (e.g., lengthscale from input range)
- Use stochastic optimization methods
{{< /callout >}}

The marginal likelihood naturally guards against overfitting, even without a separate validation set. Here's the intuition:

| Model Type | Likelihood for typical $f$ | Prior on $f$ | Marginal Likelihood |
|-----------|---------------------------|--------------|---------------------|
| **Underfit** (too simple) | Small for most $f$ | Large (fewer functions) | Small (can't fit data) |
| **Overfit** (too complex) | Large for few $f$ | Small for most $f$ | Small (prior spread thin) |
| **Just right** | Moderate for many $f$ | Moderate | **Large** (balanced) |

For an **underfit model** (too simple $\theta$), the likelihood is mostly small because the restricted function class cannot describe the data well, while the prior is large since probability mass is concentrated on fewer functions. For an **overfit model** (too complex $\theta$), the likelihood is large for some functions (those that would be selected by minimizing training error alone) but small for most functions. The prior is small because probability mass must be spread across many more functions. In both cases, one term in the product $p(\mathbf{y} \mid \mathbf{X}, \theta) = \int p(\mathbf{y} \mid f) p(f \mid \theta) df$ will be small.

Hence, maximizing the marginal likelihood naturally encourages trading off between a large likelihood and a large prior. This is the embodiment of **Occam's Razor**: among models that explain the data well, prefer simpler ones.

### Empirical Bayes and Hyperpriors

We can take the Bayesian perspective one step further by placing **priors on the hyperparameters** themselves (called hyperpriors). Using Bayes' rule and taking the negative log, we obtain the MAP estimate:

$$
\hat{\theta}_{\text{MAP}} = \arg\max_\theta p(\theta \mid \mathbf{X}, \mathbf{y}) = \arg\min_\theta \left[ -\log p(\mathbf{y} \mid \mathbf{X}, \theta) - \log p(\theta) \right]
$$

The hyperprior $p(\theta)$ acts as a **regularizer**, adding a penalty term that prevents hyperparameters from taking extreme values. Common choices include:
- **Log-normal priors** on lengthscales: $\log \ell \sim \mathcal{N}(\mu_\ell, \sigma_\ell^2)$, ensuring positivity while penalizing very small or very large values
- **Gamma priors** on noise variance: encouraging small but non-zero noise
- **Half-Cauchy priors** for robust regularization with heavy tails

Maximizing the marginal likelihood to select hyperparameters is known as **Empirical Bayes** or **Type II Maximum Likelihood**. In principle, we could go even further and integrate out the hyperparameters entirely to obtain the **fully Bayesian** predictive distribution:

$$
p(f_* \mid \mathbf{x}_*, \mathbf{X}, \mathbf{y}) = \int p(f_* \mid \mathbf{x}_*, \mathbf{X}, \mathbf{y}, \theta) p(\theta \mid \mathbf{X}, \mathbf{y}) \, d\theta
$$

However, this integral is generally **intractable**. Maximizing the marginal likelihood to select hyperparameters is known as **Empirical Bayes** or **Type II Maximum Likelihood**. 

## Computational Complexity

The main computational bottleneck in GP regression comes from two operations:

1. **Solving the linear system** $(\mathbf{K} + \sigma_n^2 \mathbf{I}_n)^{-1} \mathbf{y}$ which is needed for the predictive mean
2. **Computing the log-determinant** $\log |\mathbf{K}_y|$ — needed for marginal likelihood evaluation during hyperparameter optimization

**Complexity breakdown:**
- **Training (forming $\mathbf{K}_y$)**: Computing all $n^2$ pairwise kernel evaluations takes $O(n^2)$ time
- **Training (Cholesky decomposition)**: Decomposing the $n \times n$ matrix $\mathbf{K}_y = \mathbf{L}\mathbf{L}^T$ takes $O(n^3)$ time. This gives us both the inverse (via backsubstitution) and the log-determinant ($\log |\mathbf{K}_y| = 2 \sum_i \log L_{ii}$)
- **Storage**: Storing the $n \times n$ kernel matrix requires $O(n^2)$ memory
- **Prediction (mean)**: After training, computing $\mu_* = \mathbf{k}_*^T \boldsymbol{\alpha}$ where $\boldsymbol{\alpha} = \mathbf{K}_y^{-1} \mathbf{y} \in \mathbb{R}^n$ is precomputed, takes $O(n)$ per test point
- **Prediction (variance)**: Computing $\sigma_*^2 = k(\mathbf{x}_*, \mathbf{x}_*) - \mathbf{k}_*^T \mathbf{K}_y^{-1} \mathbf{k}_*$ requires solving another linear system, taking $O(n^2)$ per test point (or $O(n)$ if using Cholesky factors)

For comparison, Bayesian Linear Regression with $d$-dimensional features takes $O(nd^2 + d^3)$ time for training and supports efficient online updates via the Sherman-Morrison formula. GPs with $n$ data points are equivalent to BLR with $n$ "features" (one per data point), explaining the $O(n^3)$ scaling.

This cubic complexity makes standard GPs impractical for datasets with more than a few thousand points. We therefore turn to approximate methods to scale GPs to larger datasets. 


### Kernel Function Approximation

One key idea is to construct a **low-dimensional feature map** $\boldsymbol{\phi} : \mathbb{R}^d \to \mathbb{R}^M$ with $M \ll n$ that approximates the kernel:

$$
k(\mathbf{x}, \mathbf{x}') \approx \boldsymbol{\phi}(\mathbf{x})^T \boldsymbol{\phi}(\mathbf{x}')
$$

Once we have such an approximation, we can apply standard Bayesian Linear Regression in the $M$-dimensional feature space, taking advantage of the $O(nM^2 + M^3)$ complexity instead of $O(n^3)$.

#### Random Fourier Features

For **stationary kernels**, a powerful approximation method is based on Fourier analysis. 

#### Nyström Method

An alternative approach comes from the **Nyström method** for low-rank matrix approximation. The idea is to approximate the $n \times n$ kernel matrix $\mathbf{K}$ using a small subset of $M \ll n$ landmark points $\mathbf{Z} = \{\mathbf{z}_1, \ldots, \mathbf{z}_M\}$.

### Local Methods

For kernels that decay with distance (RBF, Matérn), only nearby points contribute significantly to the prediction at $\mathbf{x}_*$. A simple approximation is to:
- Only condition on the $k$ points closest to $\mathbf{x}_*$ (where $k \ll n$)
- "Cut off" the kernel at some threshold $\tau$, only considering points where $|k(\mathbf{x}, \mathbf{x}')| \geq \tau$

This reduces the effective training set size from $n$ to $k$ for each prediction. However, it loses the global consistency of the GP model. The threshold $\tau$ or neighborhood size $k$ must be chosen carefully: too restrictive and samples become essentially independent; too permissive and we gain little computational benefit.

### Inducing Point Methods

The most popular sparse GP methods use **inducing points**: a set of $m \ll n$ pseudo-inputs $\mathbf{Z} = \{\mathbf{z}_1, \ldots, \mathbf{z}_m\}$ that summarize the $n$ training data points. These can be:
- A subset of the training points
- Optimized locations (learned jointly with hyperparameters)
- Points on a regular grid

Let $\mathbf{u} = [f(\mathbf{z}_1), \ldots, f(\mathbf{z}_m)]^T \in \mathbb{R}^m$ denote the function values at the inducing points. Due to the marginalization property of GPs, we have $\mathbf{u} \sim \mathcal{N}(\mathbf{0}, \mathbf{K}_{uu})$ where $\mathbf{K}_{uu} \in \mathbb{R}^{m \times m}$.

The original GP can be recovered via marginalization:

$$
p(f_*, \mathbf{f}) = \int p(f_*, \mathbf{f}, \mathbf{u}) \, d\mathbf{u} = \int p(f_* \mid \mathbf{u}) p(\mathbf{f} \mid \mathbf{u}) p(\mathbf{u}) \, d\mathbf{u}
$$

The key approximation is to assume that $\mathbf{f}$ and $f_*$ are **conditionally independent given $\mathbf{u}$**:

$$
p(f_*, \mathbf{f}) \approx \int p(f_* \mid \mathbf{u}) p(\mathbf{f} \mid \mathbf{u}) p(\mathbf{u}) \, d\mathbf{u}
$$

This means that once we know the function values at the inducing points, knowing $\mathbf{f}$ provides no additional information about $f_*$. The inducing points act as a "sufficient summary" of the training data.

Using the GP conditioning formulas, the **exact conditionals** (also called training and testing conditionals) are:

$$
p(\mathbf{f} \mid \mathbf{u}) = \mathcal{N}(\mathbf{K}_{fu} \mathbf{K}_{uu}^{-1} \mathbf{u}, \; \mathbf{K}_{ff} - \mathbf{Q}_{ff})
$$

$$
p(f_* \mid \mathbf{u}) = \mathcal{N}(\mathbf{k}_{*u} \mathbf{K}_{uu}^{-1} \mathbf{u}, \; k_{**} - Q_{**})
$$

where we define $\mathbf{Q}_{ab} := \mathbf{K}_{au} \mathbf{K}_{uu}^{-1} \mathbf{K}_{ub}$ as the covariance "explained" by the inducing points. Intuitively, $\mathbf{K}_{ff}$ represents the prior covariance and $\mathbf{Q}_{ff}$ represents the covariance that can be captured through the inducing points.

Different sparse approximations make different assumptions about how to approximate these conditionals:

### Subset of Regressors (SoR)

The **Subset of Regressors (SoR)** approximation (also called Deterministic Training Conditional, DTC) makes a strong simplifying assumption: given the inducing variables $\mathbf{u}$, the training function values $\mathbf{f}$ have **zero variance**:

$$
q_{\text{SoR}}(\mathbf{f} \mid \mathbf{u}) = \mathcal{N}(\mathbf{K}_{fu} \mathbf{K}_{uu}^{-1} \mathbf{u}, \; \mathbf{0})
$$

$$
q_{\text{SoR}}(f_* \mid \mathbf{u}) = \mathcal{N}(\mathbf{k}_{*u} \mathbf{K}_{uu}^{-1} \mathbf{u}, \; \mathbf{0})
$$

Compare this to the exact conditional $p(\mathbf{f} \mid \mathbf{u}) = \mathcal{N}(\mathbf{K}_{fu} \mathbf{K}_{uu}^{-1} \mathbf{u}, \; \mathbf{K}_{ff} - \mathbf{Q}_{ff})$. We are essentially ignoring all uncertainty not captured by the inducing points.

**Why "Subset of Regressors"?** The name comes from the resulting predictive mean, which can be written as:

$$
\mu_*(\mathbf{x}) = \mathbf{k}_{\mathbf{x}u} \mathbf{K}_{uu}^{-1} \tilde{\mathbf{u}}
$$

where $\tilde{\mathbf{u}}$ is the posterior mean of $\mathbf{u}$. This is equivalent to performing regression using only the $m$ inducing points as "regressors" (basis functions), with the kernel evaluations $k(\mathbf{z}_i, \mathbf{x})$ serving as the features.

The effective covariance function becomes:

$$
k_{\text{SoR}}(\mathbf{x}, \mathbf{x}') = \mathbf{k}_{\mathbf{x}u} \mathbf{K}_{uu}^{-1} \mathbf{k}_{u\mathbf{x}'} = Q_{\mathbf{x}\mathbf{x}'}
$$

This is a **degenerate** covariance with rank at most $m$. The key limitation: the predictive variance is determined entirely by the inducing points and can be zero far from any training data (if that region is also far from inducing points).

**Complexity**: Training requires $O(nm^2)$ time to compute the $n \times m$ matrix $\mathbf{K}_{fu}$ and $O(m^3)$ to invert the $m \times m$ matrix $\mathbf{K}_{uu}$. Total: $O(nm^2 + m^3)$, which is linear in $n$ for fixed $m$.

### Fully Independent Training Conditional (FITC)

The **Fully Independent Training Conditional (FITC)** approximation addresses SoR's degeneracy by keeping the **diagonal variances** from the exact conditional:

$$
q_{\text{FITC}}(\mathbf{f} \mid \mathbf{u}) = \mathcal{N}\left(\mathbf{K}_{fu} \mathbf{K}_{uu}^{-1} \mathbf{u}, \; \text{diag}(\mathbf{K}_{ff} - \mathbf{Q}_{ff})\right)
$$

$$
q_{\text{FITC}}(f_* \mid \mathbf{u}) = \mathcal{N}\left(\mathbf{k}_{*u} \mathbf{K}_{uu}^{-1} \mathbf{u}, \; \text{diag}(k_{**} - Q_{**})\right)
$$

The notation $\text{diag}(\mathbf{A})$ means we keep only the diagonal elements of $\mathbf{A}$, setting off-diagonal elements to zero. This corresponds to assuming that the training points are **conditionally independent** given $\mathbf{u}$:

$$
q(\mathbf{f} \mid \mathbf{u}) = \prod_{i=1}^n p(f_i \mid \mathbf{u})
$$

FITC retains the individual variances $k(\mathbf{x}_i, \mathbf{x}_i) - Q_{ii}$ at each training point, capturing how much uncertainty is "left over" after projecting through the inducing points. However, it ignores the correlations between training points (the off-diagonal terms). This results in a **non-degenerate** GP that maintains reasonable predictive variances even far from inducing points. The predictive variance now correctly reflects uncertainty in regions where both training data and inducing points are sparse.

**Complexity**: Same as SoR: $O(nm^2 + m^3)$ for training, which is cubic in the number of inducing points but **linear** in the number of data points. 
