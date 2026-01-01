---
title: Multivariate Gaussian
type: docs
weight: 8
---

So far we have mostly looked at single random variables or the joint distribution of two random variables. However, in many applications, we are interested in the joint distribution of a large number of random variables. For example, in an image, each pixel can be seen as a random variable, and we are interested in the joint distribution of all pixels.

## The Curse of Dimensionality

If we try to model the joint distribution of $n$ random variables using a discrete distribution, we run into a problem. Let's say we have $n$ binary random variables $X_1, \dots, X_n$, where each $X_i \in \{0, 1\}$. To fully specify the joint distribution $P(X_1, \dots, X_n)$, we need to assign a probability to each of the $2^n$ possible outcomes. Since the probabilities must sum to 1, we need $2^n - 1$ parameters.

This number grows exponentially with $n$. For a small $10 \times 10$ binary image, we would need $2^{100} - 1$ parameters, which is impossible to store or estimate. This is known as the **curse of dimensionality**.

Furthermore, performing inference (e.g., computing marginals or conditionals) is computationally expensive. To compute a marginal distribution $P(X_1)$, we would need to sum over all $2^{n-1}$ possible values of the other variables:

$$
P(X_1) = \sum_{x_2} \dots \sum_{x_n} P(X_1, x_2, \dots, x_n)
$$

This summation also grows exponentially with $n$. To compute the conditional distribution $P(X_1 | X_2 = x_2, \dots, X_n = x_n)$, we would need to use Bayes' rule, which again involves summing over an exponential number of terms:

$$
P(X_1 | X_2 = x_2, \dots, X_n = x_n) = \frac{P(X_1, x_2, \dots, x_n)}{\sum_{x_1} P(X_1, x_2, \dots, x_n)} = \frac{1}{Z} P(X_1, x_2, \dots, x_n)
$$

here, $Z$ is the normalization constant. Computing $Z$ requires summing over all possible values of $X_1$, which is again exponential in $n$.

## The Multivariate Gaussian Distribution

To overcome these issues, we often use continuous distributions that are defined by a smaller number of parameters and have nice computational properties. The most common choice is the **Multivariate Gaussian Distribution** (also known as the Multivariate Normal Distribution).

A random vector $\mathbf{X} = (X_1, \dots, X_n)^T \in \mathbb{R}^n$ is said to be normally distributed with mean vector $\boldsymbol{\mu} \in \mathbb{R}^n$ and covariance matrix $\boldsymbol{\Sigma} \in \mathbb{R}^{n \times n}$, denoted as $\mathbf{X} \sim \mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})$, if its probability density function (PDF) is given by:

$$
\mathcal{N}(\mathbf{x}; \boldsymbol{\mu}, \boldsymbol{\Sigma}) = \frac{1}{\sqrt{(2\pi)^n \det(\boldsymbol{\Sigma})}} \exp\left( -\frac{1}{2} (\mathbf{x} - \boldsymbol{\mu})^T \boldsymbol{\Sigma}^{-1} (\mathbf{x} - \boldsymbol{\mu}) \right)
$$

The multivariate Gaussian is fully specified by two parameters: the **mean vector** $\boldsymbol{\mu}$ and the **covariance matrix** $\boldsymbol{\Sigma}$.

-   **Mean vector $\boldsymbol{\mu} \in \mathbb{R}^n$**: This vector contains the expected values of each variable, i.e., $\mu_i = \mathbb{E}[X_i]$. It determines the center of the distribution.
-   **Covariance matrix $\boldsymbol{\Sigma} \in \mathbb{R}^{n \times n}$**: This matrix captures the spread and orientation of the distribution. The entry $\Sigma_{ij}$ is the covariance between $X_i$ and $X_j$:
    $$ \Sigma_{ij} = \text{Cov}(X_i, X_j) = \mathbb{E}[(X_i - \mu_i)(X_j - \mu_j)] $$
    The diagonal entries $\Sigma_{ii}$ represent the variance of each individual variable $X_i$.

The number of parameters in $\boldsymbol{\mu}$ is $n$. The covariance matrix $\boldsymbol{\Sigma}$ has $n^2$ entries, but it is **symmetric** ($\Sigma_{ij} = \Sigma_{ji}$), meaning we only need to specify the upper (or lower) triangular part, resulting in $\frac{n(n+1)}{2}$ unique parameters. Thus, the total number of parameters grows quadratically with $n$ ($O(n^2)$), which is a significant improvement over the exponential growth of discrete distributions.

The covariance matrix has two important properties:
-   **Symmetric**: This follows directly from the definition of covariance: $\text{Cov}(X_i, X_j) = \text{Cov}(X_j, X_i)$.
-   **Positive Semi-Definite**: For any vector $\mathbf{v} \in \mathbb{R}^n$, the quadratic form $\mathbf{v}^T \boldsymbol{\Sigma} \mathbf{v}$ is non-negative. This corresponds to the variance of the linear combination $Y = \mathbf{v}^T \mathbf{X}$, which must be non-negative ($\text{Var}(Y) \geq 0$).

In the definition of the PDF above, we require $\boldsymbol{\Sigma}$ to be invertible. A symmetric matrix is invertible if and only if it is **positive definite (PD)** (i.e., $\mathbf{v}^T \boldsymbol{\Sigma} \mathbf{v} > 0$ for all $\mathbf{v} \neq 0$). This is equivalent to saying that all its [eigenvalues](/garden/maths/linearAlgebra/eigendecomposition) are strictly positive. If an eigenvalue were 0, the distribution would be "flat" in the direction of the corresponding eigenvector, effectively collapsing to a lower-dimensional subspace (a "degenerate" Gaussian), and the PDF would not be well-defined in $\mathbb{R}^n$.

{{< figure 
    src="/images/multivariateGaussian.png"
    caption="Contour plots of bivariate Gaussian distributions with different covariance matrices. The ellipses represent level sets of the PDF."
    alt="Contour plots of bivariate Gaussian distributions with different covariance matrices."
>}}

The inverse of the covariance matrix, $\boldsymbol{\Lambda} = \boldsymbol{\Sigma}^{-1}$, is called the **precision matrix**. It is also symmetric and positive definite. While $\boldsymbol{\Sigma}$ captures marginal correlations, $\boldsymbol{\Lambda}$ captures **conditional independence**: if $\Lambda_{ij} = 0$, then $X_i$ and $X_j$ are conditionally independent given all other variables.

{{< callout type="info" title="Isotropic Gaussian" >}}
A special case is the **isotropic Gaussian**, where the covariance matrix is a multiple of the identity matrix: $\boldsymbol{\Sigma} = \sigma^2 \mathbf{I}$. In this case, the variables are uncorrelated and have the same variance. The level sets of the PDF are perfect hyperspheres (circles in 2D).
{{< /callout >}}

## Properties of the Multivariate Gaussian

One of the main reasons the Gaussian distribution is so popular is that it is closed under many operations, making inference tractable.

### Independence and Uncorrelatedness

For general random variables, uncorrelatedness (covariance is 0) does not imply independence. However, for Gaussian random variables, **uncorrelatedness implies independence**. This is a crucial property.

If variables are uncorrelated, the covariance matrix $\boldsymbol{\Sigma}$ is diagonal, i.e., $\Sigma_{ij} = 0$ for $i \neq j$. Consequently, its inverse $\boldsymbol{\Sigma}^{-1}$ is also diagonal with entries $1/\sigma_i^2$. The exponent in the PDF then becomes a sum:

$$
(\mathbf{x} - \boldsymbol{\mu})^T \boldsymbol{\Sigma}^{-1} (\mathbf{x} - \boldsymbol{\mu}) = \sum_{i=1}^n \frac{(x_i - \mu_i)^2}{\sigma_i^2}
$$

Since $\exp(\sum a_i) = \prod \exp(a_i)$, the joint PDF factors into a product of univariate Gaussian PDFs:

$$
\mathcal{N}(\mathbf{x}; \boldsymbol{\mu}, \text{diag}(\sigma_1^2, \dots, \sigma_n^2)) = \prod_{i=1}^n \frac{1}{\sqrt{2\pi\sigma_i^2}} \exp\left( -\frac{(x_i - \mu_i)^2}{2\sigma_i^2} \right) = \prod_{i=1}^n \mathcal{N}(x_i; \mu_i, \sigma_i^2)
$$

Since the joint distribution is the product of the marginals, the variables $X_1, \dots, X_n$ are **mutually independent**.

### Marginals and Conditionals

A key property is that both the marginal and conditional distributions of a joint Gaussian are also Gaussian.

Let's partition the random vector $\mathbf{X}$ into two parts $\mathbf{X}_A$ and $\mathbf{X}_B$ (where $A$ and $B$ are sets of indices):

$$
\mathbf{X} = \begin{pmatrix} \mathbf{X}_A \\ \mathbf{X}_B \end{pmatrix}, \quad \boldsymbol{\mu} = \begin{pmatrix} \boldsymbol{\mu}_A \\ \boldsymbol{\mu}_B \end{pmatrix}, \quad \boldsymbol{\Sigma} = \begin{pmatrix} \boldsymbol{\Sigma}_{AA} & \boldsymbol{\Sigma}_{AB} \\ \boldsymbol{\Sigma}_{BA} & \boldsymbol{\Sigma}_{BB} \end{pmatrix}
$$

The **marginal distribution** of a subset of variables is simply given by dropping the irrelevant parts of the mean and covariance. For $\mathbf{X}_A$, the marginal is:

$$
\mathbf{X}_A \sim \mathcal{N}(\boldsymbol{\mu}_A, \boldsymbol{\Sigma}_{AA})
$$

This is incredibly simple: we just take the sub-vector of means and the sub-matrix of covariances.

{{< callout type="proof" title="Derivation of Marginal" >}}
To find the marginal distribution $p(\mathbf{x}_A)$, we integrate out $\mathbf{x}_B$ from the joint PDF:

$$
p(\mathbf{x}_A) = \int p(\mathbf{x}_A, \mathbf{x}_B) \, d\mathbf{x}_B
$$

While this integral can be solved by completing the square in the exponent, a more elegant approach uses the **characteristic function** (or moment generating function). The characteristic function of a multivariate Gaussian random vector $\mathbf{X} \in \mathbb{R}^n$ is given by:

$$
\phi_{\mathbf{X}}(\mathbf{t}) = \mathbb{E}[e^{i\mathbf{t}^T\mathbf{X}}] = \exp(i\mathbf{t}^T\boldsymbol{\mu} - \frac{1}{2}\mathbf{t}^T\boldsymbol{\Sigma}\mathbf{t})
$$

For the marginal of $\mathbf{X}_A$, we consider the vector $\mathbf{t} = (\mathbf{t}_A, \mathbf{0})$. Then:

$$
\phi_{\mathbf{X}_A}(\mathbf{t}_A) = \phi_{\mathbf{X}}(\mathbf{t}_A, \mathbf{0}) = \exp(i\mathbf{t}_A^T\boldsymbol{\mu}_A - \frac{1}{2}\mathbf{t}_A^T\boldsymbol{\Sigma}_{AA}\mathbf{t}_A)
$$

This is exactly the characteristic function of a Gaussian with mean $\boldsymbol{\mu}_A$ and covariance $\boldsymbol{\Sigma}_{AA}$. Thus, $\mathbf{X}_A \sim \mathcal{N}(\boldsymbol{\mu}_A, \boldsymbol{\Sigma}_{AA})$.
{{< /callout >}}

The **conditional distribution** of $\mathbf{X}_A$ given that we have observed $\mathbf{X}_B = \mathbf{x}_B$ is also Gaussian:

$$
\mathbf{X}_A | \mathbf{X}_B = \mathbf{x}_B \sim \mathcal{N}(\boldsymbol{\mu}_{A|B}, \boldsymbol{\Sigma}_{A|B})
$$

where the conditional mean and covariance are given by:

$$
\begin{align*}
\boldsymbol{\mu}_{A|B} &= \boldsymbol{\mu}_A + \boldsymbol{\Sigma}_{AB} \boldsymbol{\Sigma}_{BB}^{-1} (\mathbf{x}_B - \boldsymbol{\mu}_B) \\
\boldsymbol{\Sigma}_{A|B} &= \boldsymbol{\Sigma}_{AA} - \boldsymbol{\Sigma}_{AB} \boldsymbol{\Sigma}_{BB}^{-1} \boldsymbol{\Sigma}_{BA}
\end{align*}
$$

**Intuition:**
-   **Mean**: The expected value of $\mathbf{X}_A$ is adjusted based on the observation $\mathbf{x}_B$. The term $\boldsymbol{\Sigma}_{AB} \boldsymbol{\Sigma}_{BB}^{-1} (\mathbf{x}_B - \boldsymbol{\mu}_B)$ represents the information gain from observing $\mathbf{x}_B$. If $\mathbf{X}_A$ and $\mathbf{X}_B$ are uncorrelated ($\boldsymbol{\Sigma}_{AB} = 0$), the mean remains $\boldsymbol{\mu}_A$.
-   **Covariance**: The uncertainty about $\mathbf{X}_A$ decreases after observing $\mathbf{X}_B$. The term $\boldsymbol{\Sigma}_{AB} \boldsymbol{\Sigma}_{BB}^{-1} \boldsymbol{\Sigma}_{BA}$ is positive semi-definite, so we are subtracting uncertainty. Note that the conditional covariance $\boldsymbol{\Sigma}_{A|B}$ does **not** depend on the observed value $\mathbf{x}_B$, only on the covariance structure.

{{< callout type="proof" title="Derivation of Conditional" >}}
The conditional density is proportional to the joint density, treating $\mathbf{x}_B$ as fixed:
$$
p(\mathbf{x}_A | \mathbf{x}_B) = \frac{p(\mathbf{x}_A, \mathbf{x}_B)}{p(\mathbf{x}_B)} \propto \exp\left( -\frac{1}{2} (\mathbf{x} - \boldsymbol{\mu})^T \boldsymbol{\Sigma}^{-1} (\mathbf{x} - \boldsymbol{\mu}) \right)
$$

To see this, we use the block matrix inversion lemma (Schur complement) on the precision matrix $\boldsymbol{\Lambda} = \boldsymbol{\Sigma}^{-1}$. The exponent is quadratic in $\mathbf{x}_A$. By completing the square with respect to $\mathbf{x}_A$, we can rewrite the exponent in the form:

$$
-\frac{1}{2} (\mathbf{x}_A - \boldsymbol{\mu}_{A|B})^T \boldsymbol{\Sigma}_{A|B}^{-1} (\mathbf{x}_A - \boldsymbol{\mu}_{A|B}) + C
$$

Matching the terms yields the expressions for $\boldsymbol{\mu}_{A|B}$ and $\boldsymbol{\Sigma}_{A|B}$. The fact that the result is a quadratic form in the exponent proves that the conditional distribution is indeed Gaussian.
{{< /callout >}}

### Affine Transformations

An **affine transformation** is a linear transformation followed by a translation. It takes the form $\mathbf{y} = \mathbf{A}\mathbf{x} + \mathbf{b}$, where $\mathbf{A}$ is a matrix and $\mathbf{b}$ is a vector. This generalizes the univariate case $Y = aX + b$.
We have already seen that linear transformations of univariate Gaussians yield another Gaussian. This property extends to the multivariate case. Specifically any affine transformation of a Gaussian random vector is also a Gaussian random vector. So if $\mathbf{X} \sim \mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})$ and $\mathbf{Y} = \mathbf{A}\mathbf{X} + \mathbf{b}$, then:

$$
\mathbf{Y} \sim \mathcal{N}(\mathbf{A}\boldsymbol{\mu} + \mathbf{b}, \mathbf{A}\boldsymbol{\Sigma}\mathbf{A}^T)
$$

{{< callout type="proof" title="Derivation of Parameters" >}}
We can derive the mean and covariance of $\mathbf{Y}$ using the linearity of expectation:
1.  **Mean**:

$$
\mathbb{E}[\mathbf{Y}] = \mathbb{E}[\mathbf{A}\mathbf{X} + \mathbf{b}] = \mathbf{A}\mathbb{E}[\mathbf{X}] + \mathbf{b} = \mathbf{A}\boldsymbol{\mu} + \mathbf{b}
$$

2.  **Covariance**:

$$
\begin{align*}
\text{Cov}(\mathbf{Y}) &= \mathbb{E}[(\mathbf{Y} - \mathbb{E}[\mathbf{Y}])(\mathbf{Y} - \mathbb{E}[\mathbf{Y}])^T] \\
&= \mathbb{E}[(\mathbf{A}\mathbf{X} + \mathbf{b} - (\mathbf{A}\boldsymbol{\mu} + \mathbf{b}))(\mathbf{A}\mathbf{X} + \mathbf{b} - (\mathbf{A}\boldsymbol{\mu} + \mathbf{b}))^T] \\
&= \mathbb{E}[(\mathbf{A}(\mathbf{X} - \boldsymbol{\mu}))(\mathbf{A}(\mathbf{X} - \boldsymbol{\mu}))^T] \\
&= \mathbf{A} \mathbb{E}[(\mathbf{X} - \boldsymbol{\mu})(\mathbf{X} - \boldsymbol{\mu})^T] \mathbf{A}^T \\
&= \mathbf{A} \boldsymbol{\Sigma} \mathbf{A}^T
\end{align*}
$$

The fact that $\mathbf{Y}$ follows a Gaussian distribution can be shown using characteristic functions: $\phi_{\mathbf{Y}}(\mathbf{t}) = e^{i\mathbf{t}^T\mathbf{b}} \phi_{\mathbf{X}}(\mathbf{A}^T\mathbf{t})$, which results in the characteristic function of the Gaussian derived above.
{{< /callout >}}

This property is very useful. For example, it implies that:
-   **Scaling**: Multiples of Gaussians are Gaussian.
-   **Sum**: The sum of two independent Gaussian random vectors $\mathbf{X} \sim \mathcal{N}(\boldsymbol{\mu}_X, \boldsymbol{\Sigma}_X)$ and $\mathbf{Y} \sim \mathcal{N}(\boldsymbol{\mu}_Y, \boldsymbol{\Sigma}_Y)$ is Gaussian. We can stack them into a single vector and apply a transformation matrix $\mathbf{A} = [\mathbf{I}, \mathbf{I}]$:

$$
\mathbf{X} + \mathbf{Y} \sim \mathcal{N}(\boldsymbol{\mu}_X + \boldsymbol{\mu}_Y, \boldsymbol{\Sigma}_X + \boldsymbol{\Sigma}_Y)
$$

### Conditional Linear Gaussian

Combining the properties above, we can view the relationship between variables in a Gaussian model in a specific way. From the conditional distribution formula, we saw that the conditional mean $\boldsymbol{\mu}_{A|B}$ is a linear function of $\mathbf{x}_B$.

Conversely, we can construct a joint Gaussian distribution from a marginal and a conditional distribution. Suppose we have:
1.  A marginal distribution for $\mathbf{X}_B$: $\mathbf{X}_B \sim \mathcal{N}(\boldsymbol{\mu}_B, \boldsymbol{\Sigma}_{BB})$.
2.  A conditional distribution for $\mathbf{X}_A$ given $\mathbf{X}_B$ that is linear with Gaussian noise:

$$
\mathbf{X}_A = \mathbf{A} \mathbf{X}_B + \mathbf{b} + \boldsymbol{\epsilon}
$$

where $\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \boldsymbol{\Sigma}_{\epsilon})$ is independent noise.

This setup is called a **Conditional Linear Gaussian** model. It implies that $\mathbf{X}_A$ given $\mathbf{X}_B$ is Gaussian:

$$
\mathbf{X}_A | \mathbf{X}_B \sim \mathcal{N}(\mathbf{A}\mathbf{X}_B + \mathbf{b}, \boldsymbol{\Sigma}_{\epsilon})
$$

More importantly, the joint distribution of $(\mathbf{X}_A, \mathbf{X}_B)$ is also Gaussian. We can find its parameters using the affine transformation property. Let $\mathbf{Z} = \begin{pmatrix} \mathbf{X}_B \\ \boldsymbol{\epsilon} \end{pmatrix}$. Then $\mathbf{Z}$ is Gaussian (since $\mathbf{X}_B$ and $\boldsymbol{\epsilon}$ are independent). We can write the joint vector as a linear transformation of $\mathbf{Z}$:

$$
\begin{pmatrix} \mathbf{X}_A \\ \mathbf{X}_B \end{pmatrix} = \begin{pmatrix} \mathbf{A} & \mathbf{I} \\ \mathbf{I} & \mathbf{0} \end{pmatrix} \begin{pmatrix} \mathbf{X}_B \\ \boldsymbol{\epsilon} \end{pmatrix} + \begin{pmatrix} \mathbf{b} \\ \mathbf{0} \end{pmatrix}
$$

Using the affine transformation rules, we can derive the joint mean and covariance. This structure is fundamental to many probabilistic models, including **Linear Regression** (where $\mathbf{X}_B$ are inputs and $\mathbf{X}_A$ are targets) and **Kalman Filters** (where $\mathbf{X}_B$ is the state and $\mathbf{X}_A$ is the observation).
