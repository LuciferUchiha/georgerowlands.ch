---
title: Principal Component Analysis
type: docs
weight: 15
---

Principal Component Analysis (PCA) is a fundamental dimensionality reduction technique that identifies the directions of maximum variance in high-dimensional data. By projecting data onto a lower-dimensional subspace, PCA enables efficient storage, visualization, and analysis while preserving the most important patterns in the data.

## Motivation

In many machine learning applications, we work with high-dimensional data. For example, images can be represented as vectors where each pixel corresponds to a dimension. A small $100 \times 100$ grayscale image requires 10,000 dimensions. This high dimensionality creates several challenges.

First, storing and processing high-dimensional data is computationally expensive. Operations like matrix multiplication and distance calculations scale poorly with dimension. Second, many machine learning algorithms suffer from the curse of dimensionality, where the amount of data needed to learn patterns grows exponentially with the number of dimensions.

However, high-dimensional data often contains redundancy. Many dimensions may be correlated or contain little useful information. PCA exploits this structure by finding a low-dimensional representation that captures most of the variation in the data.

## The PCA Problem

Given a dataset $X = \{x_1, \ldots, x_n\} \subseteq \mathbb{R}^D$ of $n$ points in $D$ dimensions, we want to find a linear projection $\pi: \mathbb{R}^D \to \mathbb{R}^d$ with $d \ll D$ such that the projected data $\pi(X)$ retains as much information as possible.

The key insight is to measure information using **variance**. Variance quantifies how spread out the data is. If we project data onto a direction where all points are clustered together, we lose the structure that distinguishes different points. Conversely, directions with high variance preserve the diversity of the data.

Formally, we seek a projection $\pi$ that maximizes the variance of the projected data. For a linear projection defined by a matrix $U \in \mathbb{R}^{D \times d}$, the projection of a point $x$ is given by $\pi(x) = U^T x$. We want to find $U$ such that the variance of $\{U^T x_1, \ldots, U^T x_n\}$ is maximized.

{{< figure
    src="/images/maths/pcaVariance.webp"
    caption="Projecting 2D data onto different 1D lines. The projection onto the horizontal axis (left) loses the structure that distinguishes the two clusters. The projection onto the direction of maximum variance (right) preserves this separation."
    alt="Comparison of PCA projections showing why variance matters"
>}}

**Intuition in 2D**: Consider a dataset forming two distinct clusters in 2D space. If we project this data onto the horizontal axis (a direction with low variance), the two clusters collapse into overlapping regions on the line, and we lose the ability to distinguish them. However, if we project onto the direction where the data varies the most (the direction connecting the cluster centers), the projected points maintain their separation. The variance of a projection quantifies how spread out the projected points are. High variance means the projection preserves distances between points, which in turn preserves the structure and patterns in the data. This is why maximizing variance is a sensible objective for dimensionality reduction.

## Variance and the Covariance Matrix

When we project high-dimensional vectors onto a direction $u$, we get scalars $u^T x_i$. These are just numbers (the coordinates along that direction), and we can compute their variance in the usual way. So while our data points $x_i$ are vectors, the projected values are scalars, and it is the variance of these scalar projections that we want to maximize. The covariance matrix $S$ provides a compact way to compute this variance for any direction $u$ using the quadratic form $u^T S u$.

Before deriving PCA, we need to connect variance to linear algebra. The **sample mean** of our dataset is:

$$
\bar{x} = \frac{1}{n} \sum_{i=1}^n x_i
$$

The **sample covariance matrix** $S \in \mathbb{R}^{D \times D}$ is defined as:

$$
S = \frac{1}{n} \sum_{i=1}^n (x_i - \bar{x})(x_i - \bar{x})^T
$$

This matrix captures how each pair of dimensions varies together. The diagonal entry $S_{jj}$ is the variance of the $j$-th dimension, while the off-diagonal entry $S_{jk}$ is the covariance between dimensions $j$ and $k$.

The covariance matrix is [**symmetric**](/garden/maths/linearAlgebra/eigendecomposition#eigenvalues-and-eigenvectors-of-symmetric-matrices) (since $S_{jk} = S_{kj}$) and [**positive semi-definite**](/garden/maths/linearAlgebra/eigendecomposition#positive-definite-and-positive-semi-definite-matrices) (since $v^T S v = \frac{1}{n} \sum_{i=1}^n (v^T(x_i - \bar{x}))^2 \geq 0$ for any vector $v$).

Without loss of generality, we can assume the data is **centered**, meaning $\bar{x} = 0$. If not, we can replace each $x_i$ with $x_i - \bar{x}$. With centered data, the covariance matrix simplifies to:

$$
S = \frac{1}{n} \sum_{i=1}^n x_i x_i^T
$$

## The One-Dimensional Case

We restrict to unit vectors ($\|u_1\| = 1$) because scaling a direction by a constant factor $c$ scales the variance by $c^2$. Without the unit length constraint, we could make the variance arbitrarily large by choosing $u$ with large norm. The unit length constraint ensures we are comparing directions, not magnitudes.

Let us first consider projecting onto a single direction $u_1 \in \mathbb{R}^D$ with $\|u_1\| = 1$. The projection of a point $x$ onto this direction is the scalar $u_1^T x$. The variance of the projected data is:

$$
\text{Var}(\pi(X)) = \frac{1}{n} \sum_{i=1}^n (u_1^T x_i - u_1^T \bar{x})^2
$$

Assuming centered data ($\bar{x} = 0$), this becomes:

$$
\text{Var}(\pi(X)) = \frac{1}{n} \sum_{i=1}^n (u_1^T x_i)^2 = \frac{1}{n} \sum_{i=1}^n u_1^T x_i x_i^T u_1 = u_1^T \left( \frac{1}{n} \sum_{i=1}^n x_i x_i^T \right) u_1 = u_1^T S u_1
$$

Therefore, maximizing variance is equivalent to maximizing the quadratic form $u_1^T S u_1$ subject to the constraint $u_1^T u_1 = 1$.

### Lagrange Multiplier Solution

To solve this constrained optimization problem, we use the method of Lagrange multipliers. We want to maximize:

$$
\mathcal{L}(u_1, \lambda) = u_1^T S u_1 - \lambda (u_1^T u_1 - 1)
$$

Taking the derivative with respect to $u_1$ and setting it to zero:

$$
\frac{\partial \mathcal{L}}{\partial u_1} = 2 S u_1 - 2 \lambda u_1 = 0
$$

This gives us the eigenvalue equation:

$$
S u_1 = \lambda u_1
$$

This tells us that the optimal direction $u_1$ is an [**eigenvector**](/garden/maths/linearAlgebra/eigendecomposition) of the covariance matrix $S$, and the corresponding [eigenvalue](/garden/maths/linearAlgebra/eigendecomposition) $\lambda$ equals the variance along that direction (since $u_1^T S u_1 = u_1^T \lambda u_1 = \lambda$).

To maximize variance, we should choose $u_1$ to be the eigenvector corresponding to the **largest eigenvalue** $\lambda_1^*$ of $S$. This eigenvector is called the **first principal component**.

{{< callout type="proof" title="Largest Eigenvalue Maximizes Variance" >}}
We need to verify that among all eigenvectors of $S$, the one with the largest eigenvalue gives the maximum variance.

Suppose $S$ has eigenvalues $\lambda_1 \geq \lambda_2 \geq \cdots \geq \lambda_D$ with corresponding unit eigenvectors $u_1, u_2, \ldots, u_D$. The [spectral theorem](/garden/maths/linearAlgebra/eigendecomposition) tells us these eigenvectors form an orthonormal basis for $\mathbb{R}^D$.

For any unit vector $v$, we can express it in this eigenvector basis as $v = \sum_{i=1}^D \alpha_i u_i$ where the coefficients $\alpha_i = u_i^T v$. Since $v$ is a unit vector, we have $\|v\|^2 = \sum_{i=1}^D \alpha_i^2 = 1$.

Now we compute the variance along direction $v$:
$$
v^T S v = v^T \left( \sum_{i=1}^D \lambda_i \alpha_i u_i \right)
$$

The key insight is that $Sv = S(\sum_{i=1}^D \alpha_i u_i) = \sum_{i=1}^D \alpha_i S u_i = \sum_{i=1}^D \alpha_i \lambda_i u_i$ (using $Su_i = \lambda_i u_i$).

Taking the inner product with $v$:
$$
v^T S v = \sum_{i=1}^D \alpha_i \lambda_i (u_i^T v) = \sum_{i=1}^D \lambda_i \alpha_i^2
$$

Since $\lambda_1$ is the largest eigenvalue, we have $\lambda_i \leq \lambda_1$ for all $i$. Therefore:
$$
v^T S v = \sum_{i=1}^D \lambda_i \alpha_i^2 \leq \lambda_1 \sum_{i=1}^D \alpha_i^2 = \lambda_1
$$

Equality holds when all the weight is on the first eigenvector, that is, when $\alpha_1 = 1$ and $\alpha_i = 0$ for $i > 1$. This means $v = u_1$.
{{< /callout >}}

## The General d-Dimensional Case

When projecting to $d > 1$ dimensions, we want the projection axes to be orthogonal for two reasons. First, orthogonal axes provide an unambiguous coordinate system in the projected space (each point has unique coordinates). Second, and more importantly, if the axes were not orthogonal, they would partially redundant information about the same directions in the data. Orthogonality ensures that each principal component captures distinct variation in the data. Fortunately, the eigenvectors of a symmetric matrix like $S$ are automatically orthogonal, so this requirement is naturally satisfied.

Now we extend PCA to find $d$ orthogonal directions $u_1, \ldots, u_d$ that capture the most variance. The key idea is to proceed iteratively:

1. Find the first principal component $u_1^*$ as the eigenvector of $S$ with the largest eigenvalue $\lambda_1^*$.

2. To find the second principal component, we want a direction orthogonal to $u_1^*$ that has maximum variance. Define $X_1 = \{x - (u_1^{*T} x) u_1^* : x \in X\}$ as the dataset with projections onto $u_1^*$ removed. Compute the covariance matrix $S_1$ of $X_1$ and find its largest eigenvector $u_2^*$.

3. Continue this process: at step $k$, remove projections onto $u_1^*, \ldots, u_{k-1}^*$, compute the covariance matrix of the residual, and find its largest eigenvector.

However, there is a more elegant formulation. The principal components $u_1^*, \ldots, u_d^*$ are simply the eigenvectors corresponding to the $d$ largest eigenvalues of $S$. These eigenvectors are [automatically orthogonal](/garden/maths/linearAlgebra/eigendecomposition#eigenvalues-and-eigenvectors-of-symmetric-matrices) (since $S$ is symmetric).

Therefore we just need to find the eigenvectors $u_1^*, \ldots, u_d^*$ corresponding to the largest eigenvalues $\lambda_1^* \geq \lambda_2^* \geq \cdots \geq \lambda_d^*$ of $S$. These form the **first d principal components** of the data. The projection $\pi: \mathbb{R}^D \to \mathbb{R}^d$ is then given by:

$$
\pi(x) = \begin{pmatrix} u_1^{*T} x \\ \vdots \\ u_d^{*T} x \end{pmatrix} = U_d^T x
$$

where $U_d = [u_1^*, \ldots, u_d^*] \in \mathbb{R}^{D \times d}$ contains the first $d$ principal components as columns.

{{< callout type="info" title="Spectral Theorem" >}}
The [spectral theorem](/garden/maths/linearAlgebra/eigendecomposition) states that any real symmetric matrix $S$ can be decomposed as:

$$
S = U \Lambda U^T
$$

where $U = [u_1, \ldots, u_D]$ is an orthogonal matrix whose columns are the eigenvectors of $S$, and $\Lambda = \text{diag}(\lambda_1, \ldots, \lambda_D)$ is a diagonal matrix of eigenvalues ordered such that $\lambda_1 \geq \lambda_2 \geq \cdots \geq \lambda_D$. The eigenvectors $\{u_i\}$ form an orthonormal basis, meaning $u_i^T u_j = \delta_{ij}$ (where $\delta_{ij}$ is the Kronecker delta: 1 if $i = j$ and 0 otherwise).

This decomposition is fundamental to PCA because it tells us that we can simultaneously diagonalize the covariance matrix and find an orthonormal basis of eigenvectors.
{{< /callout >}}

{{< figure
    src="/images/maths/pca3d.png"
    caption="PCA finds orthogonal directions of maximum variance. The first principal component (longest arrow) points in the direction of greatest spread. The second principal component (medium arrow) captures the next largest variation orthogonal to the first."
    alt="3D visualization of PCA showing principal components"
>}}

### Limitations of Linear Projections

A fundamental limitation of PCA is that it only finds **linear structure**. PCA projects data using matrix multiplication, which is a linear operation. If the data lies on a curved manifold (a nonlinear subspace), PCA will not capture this structure well.

For example, consider data sampled from a Swiss roll (a 2D surface rolled up in 3D space). The intrinsic dimensionality is 2, but PCA cannot unroll the surface because doing so requires a nonlinear transformation. PCA would instead find the 2D plane that best approximates the Swiss roll in a least-squares sense, which loses the intrinsic curved structure.

## Choosing the Number of Components

An important question is: how many principal components $d$ should we keep? This depends on how much variance we want to preserve.

The **total variance** in the original data is:

$$
\text{Var}_{\text{total}} = \text{tr}(S) = \sum_{j=1}^D S_{jj} = \sum_{j=1}^D \lambda_j
$$

where $\text{tr}(S)$ is the trace of $S$ (the sum of diagonal elements). The trace of a matrix equals the sum of its eigenvalues, which follows from the [spectral theorem](/garden/maths/linearAlgebra/eigendecomposition). Intuitively, the trace $\sum_{j=1}^D S_{jj}$ sums the variance along each original coordinate axis. The spectral theorem tells us this same total variance can be decomposed into variance along the eigenvector directions, which sum to $\sum_{j=1}^D \lambda_j$.

The variance captured by the first $d$ principal components is:

$$
\text{Var}_{d} = \sum_{j=1}^d \lambda_j
$$

The **proportion of variance explained** by the first $d$ components is:

$$
\frac{\text{Var}_{d}}{\text{Var}_{\text{total}}} = \frac{\sum_{j=1}^d \lambda_j}{\sum_{j=1}^D \lambda_j}
$$

A common heuristic is to choose $d$ such that this proportion exceeds a threshold (such as 0.95 or 0.99).
