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

TODO: Dont use headers for parameters and covariance matrix. First exaplain the mean vector and covariance matrix and the interpretations so that you can then discuss properties of symmetricness, positive semi-definiteness, etc. and lastly introduce the precision matrix in that context. Why does positive definiteness mean it is invertible? did also write something about that in my linear algebra notes? if so, link to that.

Here, we assume that the covariance matrix $\boldsymbol{\Sigma}$ is invertible (i.e., positive definite). The term $\boldsymbol{\Lambda} = \boldsymbol{\Sigma}^{-1}$ is often called the **precision matrix**.

### Parameters and Complexity

The multivariate Gaussian is fully specified by its mean and covariance:
-   **Mean vector $\boldsymbol{\mu}$**: Contains $n$ parameters.
-   **Covariance matrix $\boldsymbol{\Sigma}$**: Contains $n^2$ entries, but since it is symmetric, it has only $\frac{n(n+1)}{2}$ unique parameters.

Thus, the number of parameters grows quadratically with $n$ ($O(n^2)$), which is much better than the exponential growth ($O(2^n)$) of discrete distributions.

### The Covariance Matrix

The covariance matrix $\boldsymbol{\Sigma}$ encodes the relationships between the variables. The entry $\Sigma_{ij}$ is the covariance between $X_i$ and $X_j$:

$$
\Sigma_{ij} = \text{Cov}(X_i, X_j) = \mathbb{E}[(X_i - \mu_i)(X_j - \mu_j)]
$$

The diagonal entries $\Sigma_{ii}$ are the variances of the individual variables.

**Properties:**
1.  **Symmetric**: $\Sigma_{ij} = \Sigma_{ji}$.
2.  **Positive Semi-Definite (PSD)**: For any vector $\mathbf{v} \in \mathbb{R}^n$, $\mathbf{v}^T \boldsymbol{\Sigma} \mathbf{v} \geq 0$. This ensures that the variance of any linear combination of the variables is non-negative. If $\boldsymbol{\Sigma}$ is invertible, it is **Positive Definite (PD)**, meaning $\mathbf{v}^T \boldsymbol{\Sigma} \mathbf{v} > 0$ for all $\mathbf{v} \neq 0$.

{{< callout type="info" title="Isotropic Gaussian" >}}
A special case is the **isotropic Gaussian**, where the covariance matrix is a multiple of the identity matrix: $\boldsymbol{\Sigma} = \sigma^2 \mathbf{I}$. In this case, the variables are uncorrelated and have the same variance. The level sets of the PDF are perfect hyperspheres (circles in 2D).
{{< /callout >}}

### Independence and Uncorrelatedness

For general random variables, uncorrelatedness (covariance is 0) does not imply independence. However, for Gaussian random variables, **uncorrelatedness implies independence**. This is an important property that simplifies analysis.

If $\boldsymbol{\Sigma}$ is a diagonal matrix (i.e., all off-diagonal elements are 0), then the PDF factors into a product of univariate Gaussians:

TODO: this means that each variable is independent of the others and then why can this be written as a product of univariate Gaussians?

$$
\mathcal{N}(\mathbf{x}; \boldsymbol{\mu}, \text{diag}(\sigma_1^2, \dots, \sigma_n^2)) = \prod_{i=1}^n \mathcal{N}(x_i; \mu_i, \sigma_i^2)
$$

This means the variables $X_1, \dots, X_n$ are mutually independent.

## Properties of the Multivariate Gaussian

TODO: why is the above independence and correlatedness property not in this section? it also seems like a property?

One of the main reasons the Gaussian distribution is so popular is that it is closed under many operations, making inference tractable.

### Marginals and Conditionals

A key property is that both the marginal and conditional distributions of a joint Gaussian are also Gaussian.

Let's partition the random vector $\mathbf{X}$ into two parts $\mathbf{X}_A$ and $\mathbf{X}_B$ (where $A$ and $B$ are sets of indices):

$$
\mathbf{X} = \begin{pmatrix} \mathbf{X}_A \\ \mathbf{X}_B \end{pmatrix}, \quad \boldsymbol{\mu} = \begin{pmatrix} \boldsymbol{\mu}_A \\ \boldsymbol{\mu}_B \end{pmatrix}, \quad \boldsymbol{\Sigma} = \begin{pmatrix} \boldsymbol{\Sigma}_{AA} & \boldsymbol{\Sigma}_{AB} \\ \boldsymbol{\Sigma}_{BA} & \boldsymbol{\Sigma}_{BB} \end{pmatrix}
$$

**Marginal Distribution:**
The marginal distribution of a subset of variables is simply given by dropping the irrelevant parts of the mean and covariance. For $\mathbf{X}_A$, the marginal is:

$$
\mathbf{X}_A \sim \mathcal{N}(\boldsymbol{\mu}_A, \boldsymbol{\Sigma}_{AA})
$$

This is incredibly simple: we just take the sub-vector of means and the sub-matrix of covariances.

TODO: proof of marginal being gaussian and that these are the parameters?

**Conditional Distribution:**
The conditional distribution of $\mathbf{X}_A$ given that we have observed $\mathbf{X}_B = \mathbf{x}_B$ is also Gaussian:

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
  
TODO: proof of conditional being gaussian and that these are the parameters?

### Affine Transformations

TODO: what is the difference between linear and affine again? an affine transformation is a linear transformation plus a translation? Linear makes sense to me as this was shown for univariate gaussians already. Give a brief reminder of that here?

Any affine transformation of a Gaussian random vector is also a Gaussian random vector.

If $\mathbf{X} \sim \mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})$ and $\mathbf{Y} = \mathbf{A}\mathbf{X} + \mathbf{b}$ for some matrix $\mathbf{A}$ and vector $\mathbf{b}$, then:

$$
\mathbf{Y} \sim \mathcal{N}(\mathbf{A}\boldsymbol{\mu} + \mathbf{b}, \mathbf{A}\boldsymbol{\Sigma}\mathbf{A}^T)
$$

This property is very useful. For example, it implies that:
-   **Scaling**: Multiples of Gaussians are Gaussian.
-   **Sum**: The sum of two independent Gaussian random vectors $\mathbf{X} \sim \mathcal{N}(\boldsymbol{\mu}_X, \boldsymbol{\Sigma}_X)$ and $\mathbf{Y} \sim \mathcal{N}(\boldsymbol{\mu}_Y, \boldsymbol{\Sigma}_Y)$ is Gaussian:
    $$ \mathbf{X} + \mathbf{Y} \sim \mathcal{N}(\boldsymbol{\mu}_X + \boldsymbol{\mu}_Y, \boldsymbol{\Sigma}_X + \boldsymbol{\Sigma}_Y) $$

TODO: Where is the proof of these properties? for the univariate case this was nicely shown and proven. Also because matrices are linear transformations we should also be able to show that we can multiply by matrices and add vectors and still have a gaussian. Maybe link to linear algebra notes again for properties of linear transformations?

### Conditional Linear Gaussian

TODO: I dont understand this part very well yet. Need to explain here more and show where this comes from and their derivations.

Combining the properties above, we can view the relationship between variables in a Gaussian model in a specific way. From the conditional distribution formula, we saw that the conditional mean $\boldsymbol{\mu}_{A|B}$ is a linear function of $\mathbf{x}_B$.

Conversely, if we have a relationship of the form:

$$
\mathbf{X}_A = \mathbf{A} \mathbf{X}_B + \mathbf{b} + \boldsymbol{\epsilon}
$$

where $\mathbf{X}_B \sim \mathcal{N}(\boldsymbol{\mu}_B, \boldsymbol{\Sigma}_{BB})$ and $\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \boldsymbol{\Sigma}_{\epsilon})$ is independent Gaussian noise, then the joint distribution of $(\mathbf{X}_A, \mathbf{X}_B)$ is Gaussian.

This is often called a **Conditional Linear Gaussian** model. It tells us that if a variable is a linear function of another variable plus Gaussian noise, they are jointly Gaussian. This is the foundation for many models, including Linear Regression.
