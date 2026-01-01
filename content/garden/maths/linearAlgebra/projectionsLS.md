---
title: Projections and Least Squares
type: docs
weight: 10
---

The projection of a vector $\mathbf{b} \in \mathbb{R}^m$ onto a subspace $S$ of $\mathbb{R}^m$ is the point $\mathbf{p} \in S$ that is closest to $\mathbf{b}$. So we would define the projection as follows:

$$
\mathbf{p} = \text{proj}_S(\mathbf{b}) = \text{argmin}_{\mathbf{v} \in S} ||\mathbf{b} - \mathbf{v}||_2
$$

Let's first look at where we have a vector in 2D space and we want to project it onto a line. So a subspace of dimension 1 which is spanned by a single basis vector $\mathbf{a}$. So we have the column space of a single vector $S = \{\lambda \mathbf{a} : \lambda \in \mathbb{R} \, \mathbf{a} \in \mathbb{R}^2\}$ = C($\mathbf{a}$). If we visualize this, we can see that the projection of $\mathbf{b}$ onto the line, i.e the point $\mathbf{p}$ that is closest to $\mathbf{b}$ is there where the error vector $\mathbf{e} = \mathbf{b} - \mathbf{p}$ is orthogonal to the line. 

{{< figure
  src="/images/maths/vectorProjection.png"
  alt="Visualizing the projection of a vector onto a line"
  caption="Visualizing the projection of a vector onto a line"
>}}

So how do we find the point $\mathbf{p}$ using this information. We can also observe that $\mathbf{p}$ is a scalar multiple of $\mathbf{a}$, i.e $\mathbf{p} = \lambda \mathbf{a}$. So we can write the error vector $\mathbf{e}$ as $\mathbf{e} = \mathbf{b} - \lambda \mathbf{a}$. So we are looking for some $\lambda$ that minimizes the error vector. In other words we want to minimize the following:

$$
\begin{align*}
||\mathbf{e}|| &= ||\mathbf{b} - \lambda \mathbf{a}|| \\
&= ||\mathbf{b} - \lambda \mathbf{a}||^2 \\
&= (\mathbf{b} - \lambda \mathbf{a})^T(\mathbf{b} - \lambda \mathbf{a}) \\
&= \mathbf{b}^T\mathbf{b} - 2\lambda\mathbf{b}^T\mathbf{a} + \lambda^2\mathbf{a}^T\mathbf{a} \\
&= ||\mathbf{b}||^2 - 2\lambda\mathbf{b}^T\mathbf{a} + \lambda^2||\mathbf{a}||^2 \\
&= g(\lambda)
\end{align*}
$$

So we get some function $g(\lambda)$ that we want to minimize. So to find the minimum of this function we can take the derivative with respect to $\lambda$ and set it to 0. Because the values of $||\mathbf{b}||^2$ and $||\mathbf{a}||^2$ and $\mathbf{b}^T\mathbf{a}$ are all constants we get the following:

$$
\begin{align*}
g(\lambda) &= ||\mathbf{b}||^2 - 2\lambda\mathbf{b}^T\mathbf{a} + \lambda^2||\mathbf{a}||^2 \\
g'(\lambda) &= -2\mathbf{b}^T\mathbf{a} + 2\lambda||\mathbf{a}||^2 = 0 \\
2\lambda||\mathbf{a}||^2 &= 2\mathbf{b}^T\mathbf{a} \\
\lambda &= \frac{\mathbf{b}^T\mathbf{a}}{||\mathbf{a}||^2} \\
\lambda &= \frac{\mathbf{a}^T\mathbf{b}}{\mathbf{a}^T\mathbf{a}}
\end{align*}
$$

So we have found the formula for $\lambda$ which is the scalar multiple of $\mathbf{a}$ that gives us the projection of $\mathbf{b}$ onto the line. Now we just need to put it back together to get the projection $\mathbf{p}$ of a 2D vector $\mathbf{b}$ onto the line spanned by $\mathbf{a}$:

$$
\text{proj}_S(\mathbf{b}) = \mathbf{a} \lambda = \mathbf{a} \frac{\mathbf{a}^T\mathbf{b}}{\mathbf{a}^T\mathbf{a}} = \frac{\mathbf{aa}^T}{\mathbf{a}^T\mathbf{a}}\mathbf{b}
$$

We can rewrite this to isolate $\mathbf{b}$, then the nominator becomes the outer product of $\mathbf{a}$ with itself and the denominator is the inner product of $\mathbf{a}$ with itself. This then gives us the final formula for the projection of $\mathbf{b}$ onto the line spanned by $\mathbf{a}$.

$$
\text{proj}_S(\mathbf{b}) = \frac{\mathbf{aa}^T}{\mathbf{a}^T\mathbf{a}}\mathbf{b}
$$

Let's now check to see if our initial observation that the error vector $\mathbf{e} = \mathbf{b} - \mathbf{p}$ is orthogonal to the line is correct:

$$
\begin{align*}
(\mathbf{b} - \mathbf{p}) \perp \mathbf{a} &\implies (\mathbf{b} - \text{proj}_S(\mathbf{b})) \perp \mathbf{a} \\
&\implies 0 = \mathbf{a}^T(\mathbf{b} - \text{proj}_S(\mathbf{b})) \\
&= \mathbf{a}^T(\mathbf{b} - \frac{\mathbf{aa}^T}{\mathbf{a}^T\mathbf{a}}\mathbf{b}) \\
&= \mathbf{a}^T\mathbf{b} - \frac{\mathbf{a}^T\mathbf{aa}^T}{\mathbf{a}^T\mathbf{a}}\mathbf{b} \\
&= \mathbf{a}^T\mathbf{b} - \frac{\mathbf{a}^T\mathbf{a}}{\mathbf{a}^T\mathbf{a}}\mathbf{a}^T\mathbf{b} \\
&= \mathbf{a}^T\mathbf{b} - \mathbf{a}^T\mathbf{b} = 0
\end{align*}
$$

{{< callout type="example" >}}
Suppose we have:

$$
\mathbf{a} = \begin{bmatrix} 2 \\ 1 \end{bmatrix}, \quad \mathbf{b} = \begin{bmatrix} 3 \\ 4 \end{bmatrix}
$$

Then we can calculate the projection of $\mathbf{b}$ onto the line spanned by $\mathbf{a}$:

$$
\text{proj}_S(\mathbf{b}) = \frac{\mathbf{aa}^T}{\mathbf{a}^T\mathbf{a}}\mathbf{b}
$$

For this we first calculate the inner product $\mathbf{a}^T\mathbf{a}$, i.e the length of $\mathbf{a}$ squared:

$$
\mathbf{a}^T\mathbf{a} = \begin{bmatrix} 2 & 1 \end{bmatrix} \begin{bmatrix} 2 \\ 1 \end{bmatrix} = 2^2 + 1^2 = 4 + 1 = 5
$$

Then we calculate the outer product $\mathbf{aa}^T$:

$$
\mathbf{aa}^T = \begin{bmatrix} 2 \\ 1 \end{bmatrix} \begin{bmatrix} 2 & 1 \end{bmatrix} = \begin{bmatrix} 4 & 2 \\ 2 & 1 \end{bmatrix}
$$

Putting it all together we get:

$$
\begin{align*}
\text{proj}_S(\mathbf{b}) &= \frac{1}{5} \begin{bmatrix} 4 & 2 \\ 2 & 1 \end{bmatrix} \begin{bmatrix} 3 \\ 4 \end{bmatrix} \\
&= \frac{1}{5} \begin{bmatrix} 4 \cdot 3 + 2 \cdot 4 \\ 2 \cdot 3 + 1 \cdot 4 \end{bmatrix} = \frac{1}{5} \begin{bmatrix} 12 + 8 \\ 6 + 4 \end{bmatrix} = \frac{1}{5} \begin{bmatrix} 20 \\ 10 \end{bmatrix} = \begin{bmatrix} 4 \\ 2 \end{bmatrix}
\end{align*}
$$
{{< /callout >}}

## Projections in Higher Dimensions

In the general case, we want to project a vector $\mathbf{b} \in \mathbb{R}^m$ onto a subspace $S \subseteq \mathbb{R}^m$ of dimension $n$, where $n \leq m$. This subspace $S$ is spanned by $n$ linearly independent vectors, which we organize as the columns of a matrix $\mathbf{A} \in \mathbb{R}^{m \times n}$. The column space of $\mathbf{A}$ is then the subspace $S=C(\mathbf{A})$. Our goal is to then find $\text{proj}_S(\mathbf{b}) = \mathbf{p}$, the projection of $\mathbf{b}$ onto $S$, such that $\mathbf{p}$ is the closest point in $S$ to $\mathbf{b}$.

Because our vector $\mathbf{p}$ is in the subspace $S$, it can be expressed as a linear combination of the columns of $\mathbf{A}$. If we store the coefficients of the linear combination that gives us $\mathbf{p}$ in a vector $\hat{\mathbf{x}} \in \mathbb{R}^n$, we can write the projection as:

$$
\mathbf{A}\hat{\mathbf{x}} = \mathbf{p}.
$$

We already know this equation very well. However, we want to find the coefficients $\hat{\mathbf{x}}$ such that the projection $\mathbf{p}$ is as close as possible to $\mathbf{b}$. So just like in the 1D case, we want to minimize the error vector $\mathbf{e} = \mathbf{b} - \mathbf{p}$. This means we want to find $\hat{\mathbf{x}}$ such that the error vector is minimized. Specifically, we want to minimize the squared norm of the error:

$$
||\mathbf{e}|| = || \mathbf{b} - \mathbf{p} || = || \mathbf{b} - \mathbf{A}\hat{\mathbf{x}} ||
$$

To solve this, it is best to square the norm to get rid of the square root. We are allowed to do this as the norm is always positive. So we can write:

$$
\begin{align*}
||\mathbf{e}|| &= || \mathbf{b} - \mathbf{A}\hat{\mathbf{x}} || \\
||\mathbf{e}||^2 &= || \mathbf{b} - \mathbf{A}\hat{\mathbf{x}} ||^2 \\
&= (\mathbf{b} - \mathbf{A}\hat{\mathbf{x}})^T(\mathbf{b} - \mathbf{A}\hat{\mathbf{x}}) \\
&= \mathbf{b}^T\mathbf{b} - 2\hat{\mathbf{x}}^T\mathbf{A}^T\mathbf{b} + \hat{\mathbf{x}}^T\mathbf{A}^T\mathbf{A}\hat{\mathbf{x}}.
\end{align*}
$$

This again gives us a function $g(\hat{\mathbf{x}})$ very similar to the 1D case that we want to minimize. To minimize this function with respect to $\hat{\mathbf{x}}$, we compute the gradient with respect to $\hat{\mathbf{x}}$ and set it to zero. Here it is the gradient rather then the derivative as we are dealing with a vector $\hat{\mathbf{x}}$ rather than a scalar $\lambda$. Just like in the 1D case, we can use the fact that the terms $\mathbf{b}^T\mathbf{b}$, $\mathbf{A}^T\mathbf{b}$, and $\mathbf{A}^T\mathbf{A}$ are constants with respect to $\hat{\mathbf{x}}$. The gradient of $g(\hat{\mathbf{x}})$ is then:

$$
-2\mathbf{A}^T\mathbf{b} + 2\mathbf{A}^T\mathbf{A}\hat{\mathbf{x}} = 0
$$

{{< callout type="proof" title="Gradient Calculation" >}}
To find the gradient of $g(\hat{\mathbf{x}}) = \mathbf{b}^T\mathbf{b} - 2\hat{\mathbf{x}}^T\mathbf{A}^T\mathbf{b} + \hat{\mathbf{x}}^T\mathbf{A}^T\mathbf{A}\hat{\mathbf{x}}$ with respect to $\hat{\mathbf{x}}$, we look at each term:

1.  $\mathbf{b}^T\mathbf{b}$ is a constant with respect to $\hat{\mathbf{x}}$, so its gradient is $\mathbf{0}$.
2.  $-2\hat{\mathbf{x}}^T\mathbf{A}^T\mathbf{b}$ is a linear term. Let $\mathbf{c} = \mathbf{A}^T\mathbf{b}$. Then we have $-2\hat{\mathbf{x}}^T\mathbf{c}$. The gradient of $\hat{\mathbf{x}}^T\mathbf{c}$ is $\mathbf{c}$, so the gradient is $-2\mathbf{A}^T\mathbf{b}$.
3.  $\hat{\mathbf{x}}^T\mathbf{A}^T\mathbf{A}\hat{\mathbf{x}}$ is a quadratic form. Let $\mathbf{M} = \mathbf{A}^T\mathbf{A}$. Since $\mathbf{M}$ is symmetric, the gradient of $\hat{\mathbf{x}}^T\mathbf{M}\hat{\mathbf{x}}$ is $2\mathbf{M}\hat{\mathbf{x}}$. Thus, the gradient is $2\mathbf{A}^T\mathbf{A}\hat{\mathbf{x}}$.

Summing these gives the result: $\nabla g(\hat{\mathbf{x}}) = -2\mathbf{A}^T\mathbf{b} + 2\mathbf{A}^T\mathbf{A}\hat{\mathbf{x}}$.
{{< /callout >}}

After further simplifying the above we get the so-called **normal equations**. The normal equations are a set of very important equations in linear algebra and computer science. We will be revisiting them in the context of solving the least squares problem further down the line.

$$
\begin{align*}
-2\mathbf{A}^T\mathbf{b} + 2\mathbf{A}^T\mathbf{A}\hat{\mathbf{x}} &= 0 \\
2\mathbf{A}^T\mathbf{A}\hat{\mathbf{x}} &= 2\mathbf{A}^T\mathbf{b} \\
\mathbf{A}^T\mathbf{A}\hat{\mathbf{x}} = \mathbf{A}^T\mathbf{b}
\end{align*}
$$

We can also get the normal equations for the 1D case by rewriting the projection formula for the 1D case:

$$
\begin{align*}
\text{proj}_S(\mathbf{b}) &= \mathbf{a} \lambda = \mathbf{a} \frac{\mathbf{a}^T\mathbf{b}}{\mathbf{a}^T\mathbf{a}}\\
&\implies \mathbf{a}^T\mathbf{a} \lambda \mathbf{a} = \mathbf{a}^T\mathbf{b} \mathbf{a} \\
&\implies \mathbf{a}^T\mathbf{a} \lambda = \mathbf{a}^T\mathbf{b}
\end{align*}
$$

We defined the matrix $\mathbf{A}\in \mathbb{R}^{m \times n}$ as the matrix whose columns are the basis vectors of the subspace $S$. So we know that the columns of $\mathbf{A}$ are linearly independent and have full row rank. We then know that if we take the transpose of $\mathbf{A}$ and multiply it with $\mathbf{A}$ we get a square matrix $\mathbf{A}^T\mathbf{A} \in \mathbb{R}^{n \times n}$. This matrix is square of dimensionality $n$ and because the rank of $\mathbf{A}$ is $n$, and we know that $\mathbf{A}^T\mathbf{A}$ and $\mathbf{A}$ have the same rank, so we know that $\mathbf{A}^T\mathbf{A}$ is of full rank and therefore invertible. We also know that $\mathbf{A}^T\mathbf{A}$ is symmetric. So we have made from the generally non-square and non-invertible matrix $\mathbf{A}$ a square and invertible matrix $\mathbf{A}^T\mathbf{A}$. 

We can now use the fact that $\mathbf{A}^T\mathbf{A}$ is invertible to solve the normal equations for $\hat{\mathbf{x}}$:

$$
\begin{align*}
\mathbf{A}^T\mathbf{A}\hat{\mathbf{x}} &= \mathbf{A}^T\mathbf{b} \\
\hat{\mathbf{x}} = (\mathbf{A}^T\mathbf{A})^{-1}\mathbf{A}^T\mathbf{b}
\end{align*}
$$

Substituting $\hat{\mathbf{x}}$ back into $\mathbf{p} = \mathbf{A}\hat{\mathbf{x}}$, we obtain the projection of $\mathbf{b}$ onto $S$:

$$
\begin{align*}
\text{proj}_S(\mathbf{b}) &= \mathbf{A}\hat{\mathbf{x}} \\
\text{proj}_S(\mathbf{b}) = \mathbf{A}(\mathbf{A}^T\mathbf{A})^{-1}\mathbf{A}^T\mathbf{b}
\end{align*}
$$

This is the general formula for the projection of a vector onto a subspace. If we set the matrix $\mathbf{P} = \mathbf{A}(\mathbf{A}^T\mathbf{A})^{-1}\mathbf{A}^T$ we get the so called **projection matrix** for the subspace $S$. It allows us to write the projection concisely as:

$$
\text{proj}_S(\mathbf{b}) = \mathbf{P}\mathbf{b}
$$

We know that linear transformations can be represented by matrices. Therefore, the projection matrix $\mathbf{P}$ defines a linear transformation that projects any vector onto the subspace $S$. This transformation can be expressed as:

$$
\begin{align*}
T_{\mathbf{P}}&: \mathbb{R}^m \rightarrow \mathbb{R}^n \\
T_{\mathbf{P}}&(\mathbf{b}) = \mathbf{P}\mathbf{b} = \text{proj}_S(\mathbf{b})
\end{align*}
$$

Where the vector $\mathbf{b}$ of dimension $m$ is projected onto the subspace $S$ of dimension $n$. 

{{< callout type="example" >}}
Suppose we have:

$$
\mathbf{A} = \begin{bmatrix} 1 & 0 \\ 0 & 1 \\ 0 & 0 \end{bmatrix}, \quad \mathbf{b} = \begin{bmatrix} 3 \\ 4 \\ 5 \end{bmatrix}
$$

Then we can calculate the projection of $\mathbf{b}$ onto the subspace spanned by the columns of $\mathbf{A}$. For this we first calculate $\mathbf{A}^T\mathbf{A}$:

$$
\begin{align*}
\mathbf{A}^T\mathbf{A} &= \begin{bmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \end{bmatrix} \begin{bmatrix} 1 & 0 \\ 0 & 1 \\ 0 & 0 \end{bmatrix} = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}
\end{align*}
$$

This is the two dimensional identity matrix and its inverse is itself. So we can now calculate the projection matrix $\mathbf{P}$:

$$
\mathbf{P} = \mathbf{A}(\mathbf{A}^T\mathbf{A})^{-1}\mathbf{A}^T = \mathbf{A}\mathbf{A}^T
$$

where 

$$
\mathbf{A}\mathbf{A}^T = \begin{bmatrix} 1 & 0 \\ 0 & 1 \\ 0 & 0 \end{bmatrix} \begin{bmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \end{bmatrix} = \begin{bmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 0 \end{bmatrix}
$$

so putting it all together we get:

$$
\begin{align*}
\text{proj}_S(\mathbf{b}) &= \mathbf{P}\mathbf{b} \\
&= \begin{bmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 0 \end{bmatrix} \begin{bmatrix} 3 \\ 4 \\ 5 \end{bmatrix} \\
&= \begin{bmatrix} 3 \\ 4 \\ 0 \end{bmatrix}
\end{align*}
$$
{{< /callout >}}

The projection transformation also has some important properties. The first one being that a projection of a projection shouldn't change the vector, the so-called the idempotent property. Intuitively this makes sense, as projecting a vector onto a subspace and then projecting the result again onto the same subspace should yield the same result as it has already been projected. Formally this can be expressed as:

$$
\begin{align*}
T_{\mathbf{P}}(T_{\mathbf{P}}(\mathbf{b})) &= T_{\mathbf{P}}(\mathbf{b}) \\
\text{proj}_S(\text{proj}_S(\mathbf{b})) = \text{proj}_S(\mathbf{b})
\end{align*}
$$

{{< callout type="proof" >}}
If we are applying the projection to the projection of $\mathbf{b}$ onto the subspace $S$, we can write this as:

$$
\mathbf{P}(\mathbf{P}\mathbf{b}) = \mathbf{P}\mathbf{P}\mathbf{b} = \mathbf{P}^2\mathbf{b}
$$

So to show that the projection is idempotent we need to show that $\mathbf{P}^2b = \mathbf{P}b$ which is the same as $\mathbf{P}^2 = \mathbf{P}$. We can see this holds by substituting the definition of $\mathbf{P}$ and simplifying using associative property of matrix multiplication:

$$
\begin{align*}
\mathbf{P}^2 &= (\mathbf{A}(\mathbf{A}^T\mathbf{A})^{-1}\mathbf{A}^T)(\mathbf{A}(\mathbf{A}^T\mathbf{A})^{-1}\mathbf{A}^T) \\
&= \mathbf{A}(\mathbf{A}^T\mathbf{A})^{-1}(\mathbf{A}^T\mathbf{A})(\mathbf{A}^T\mathbf{A})^{-1}\mathbf{A}^T \\
&= \mathbf{A}(\mathbf{A}^T\mathbf{A})^{-1}\mathbf{A}^T \\
&= \mathbf{P}
\end{align*}
$$
{{< /callout >}}

{{< callout type="example" >}}
Previously we saw that given:

$$
\mathbf{A} = \begin{bmatrix} 1 & 0 \\ 0 & 1 \\ 0 & 0 \end{bmatrix}, \quad \mathbf{b} = \begin{bmatrix} 3 \\ 4 \\ 5 \end{bmatrix}
$$

we got the projection matrix:

$$
\mathbf{P} = \begin{bmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 0 \end{bmatrix}
$$

and that Projecting $\mathbf{b}$ onto the subspace spanned by the columns of $\mathbf{A}$ gives us:

$$
\text{proj}_S(\mathbf{b}) = \begin{bmatrix} 3 \\ 4 \\ 0 \end{bmatrix}
$$

If we now apply the projection matrix $\mathbf{P}$ to the projection of $\mathbf{b}$, we should get the same result:

$$
\begin{align*}
\mathbf{P}(\text{proj}_S(\mathbf{b})) &= \mathbf{P}\begin{bmatrix} 3 \\ 4 \\ 0 \end{bmatrix} \\
&= \begin{bmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 0 \end{bmatrix} \begin{bmatrix} 3 \\ 4 \\ 0 \end{bmatrix} \\
&= \begin{bmatrix} 3 \\ 4 \\ 0 \end{bmatrix} \\
&= \text{proj}_S(\mathbf{b})
\end{align*}
$$
{{< /callout >}}

## Projecting onto the Orthogonal Complement

We have discussed projecting a vector $\mathbf{b}$ onto a subspace $S$. But what if we want to project onto the space orthogonal to $S$? This space, denoted $S^{\perp}$, is called the orthogonal complement of $S$. It contains all vectors orthogonal to every vector in $S$.

Consider again the simple 2D case from earlier. Suppose we have a subspace $S$ spanned by vector $\mathbf{a}$, and we project a vector $\mathbf{b}$ onto the line spanned by $\mathbf{a}$. We obtained the projection as:

$$
\text{proj}_S(\mathbf{b}) = \frac{\mathbf{aa}^T}{\mathbf{a}^T\mathbf{a}}\mathbf{b}
$$

The orthogonal complement in this case is simply the line perpendicular to $S$. We want to find the component of $\mathbf{b}$ that lies in $S^{\perp}$. Intuitively, this is simply the error vector $\mathbf{e}$:

$$
\text{proj}_{S^{\perp}}(\mathbf{b}) = \mathbf{e} = \mathbf{b} - \text{proj}_S(\mathbf{b}) = \mathbf{b} - \frac{\mathbf{aa}^T}{\mathbf{a}^T\mathbf{a}}\mathbf{b}
$$

Generalizing to higher dimensions, the projection onto the orthogonal complement of a subspace $S$ is therefore defined as:

$$
\text{proj}_{S^{\perp}}(\mathbf{b}) = \mathbf{b} - \text{proj}_S(\mathbf{b}) = \mathbf{b} - \mathbf{P}\mathbf{b} = (\mathbf{I} - \mathbf{P})\mathbf{b}
$$

where $\mathbf{P} = \mathbf{A}(\mathbf{A}^T\mathbf{A})^{-1}\mathbf{A}^T$ is the projection matrix onto the subspace $S$.

{{< callout type="example" >}}
Suppose we have:

$$
\mathbf{A} = \begin{bmatrix} 1 & 0 \\ 0 & 1 \\ 0 & 0 \end{bmatrix}, \quad \mathbf{b} = \begin{bmatrix} 3 \\ 4 \\ 5 \end{bmatrix}
$$

First we calculate the projection onto the subspace spanned by the columns of $\mathbf{A}$ as seen above:

$$
\text{proj}_S(\mathbf{b}) = \mathbf{A}(\mathbf{A}^T\mathbf{A})^{-1}\mathbf{A}^T\mathbf{b} = \begin{bmatrix} 3 \\ 4 \\ 0 \end{bmatrix}
$$

Then, the projection onto the orthogonal complement is:

$$
\text{proj}_{S^{\perp}}(\mathbf{b}) = \mathbf{b} - \text{proj}_S(\mathbf{b}) = \begin{bmatrix} 3 \\ 4 \\ 5 \end{bmatrix} - \begin{bmatrix} 3 \\ 4 \\ 0 \end{bmatrix} = \begin{bmatrix} 0 \\ 0 \\ 5 \end{bmatrix}
$$
{{< /callout >}}

## Projections with Orthogonal Matrices

Suppose we are given the orthogonal matrix $\mathbf{Q} \in \mathbb{R}^{m \times n}$ that has orthonormal columns. We now want to project a vector $\mathbf{b} \in \mathbb{R}^m$ onto subspace spanned by the columns of $\mathbf{Q}$, so $C(\mathbf{Q})$. We have already seen that the projection of $\mathbf{b}$ onto $C(\mathbf{Q})$ is given by:

$$
\text{proj}_{C(\mathbf{Q})}(\mathbf{b}) = \mathbf{Q} \hat{\mathbf{x}}
$$

Where $\hat{\mathbf{x}}$ is the vector of coefficients that gives us the linear combination of the columns of $\mathbf{Q}$ that best approximates $\mathbf{b}$. We get $\hat{\mathbf{x}}$ by solving the normal equations:

$$
\hat{\mathbf{x}} = (\mathbf{Q}^T\mathbf{Q})^{-1}\mathbf{Q}^T\mathbf{b}
$$

However, because $\mathbf{Q}$ is an orthogonal matrix, we know that $\mathbf{Q}^T\mathbf{Q} = \mathbf{I}$, where $\mathbf{I}$ is the identity matrix. This means that the inverse of $\mathbf{Q}^T\mathbf{Q}$ is simply the identity matrix itself:

$$
\begin{align*}
\hat{\mathbf{x}} &= (\mathbf{Q}^T\mathbf{Q})^{-1}\mathbf{Q}^T\mathbf{b} \\
&=(\mathbf{I})^{-1}\mathbf{Q}^T\mathbf{b} \\
&=\mathbf{Q}^T\mathbf{b}
\end{align*}
$$

So the projection of $\mathbf{b}$ onto the subspace spanned by the columns of $\mathbf{Q}$ can be expressed as:

$$
\text{proj}_{C(\mathbf{Q})}(\mathbf{b}) = \mathbf{Q}(\mathbf{Q}^T\mathbf{b}).
$$

and the projection matrix $\mathbf{P}$ that projects any vector onto the subspace $C(\mathbf{Q})$ can be expressed as:

$$
\mathbf{P} = \mathbf{Q}\mathbf{Q}^T.
$$

Importantly, note that in general only $\mathbf{Q}^T\mathbf{Q}$ is the identity matrix for orthogonal matrices, not $\mathbf{Q}\mathbf{Q}^T$. The latter is the identity matrix only if $\mathbf{Q}$ is square, i.e. $m = n$, which would mean that we were projecting onto the whole space $\mathbb{R}^m$. 

{{< callout type="info" >}}
This is indeed related to a **change of basis**. If $\mathbf{Q}$ is a square orthogonal matrix, its columns form an orthonormal basis for $\mathbb{R}^m$. The operation $\mathbf{Q}^T\mathbf{b}$ computes the coordinates of $\mathbf{b}$ in this new basis. Multiplying these coordinates by $\mathbf{Q}$ (i.e., $\mathbf{Q}(\mathbf{Q}^T\mathbf{b})$) reconstructs the vector $\mathbf{b}$ from these coordinates. When $\mathbf{Q}$ is not square ($m > n$), $\mathbf{Q}^T\mathbf{b}$ gives the coordinates of the projection of $\mathbf{b}$ in the basis of the subspace $C(\mathbf{Q})$.
{{< /callout >}}

{{< callout type="example" >}}
We are given the orthogonal matrix $\mathbf{Q}$ and want to project the vector $\mathbf{b}$ onto the subspace spanned by the columns of $\mathbf{Q}$.

$$
\mathbf{Q} = \begin{bmatrix} 1 & 0 \\ 0 & 1 \\ 0 & 0 \end{bmatrix}, \quad \mathbf{b} = \begin{bmatrix} 3 \\ 4 \\ 5 \end{bmatrix}
$$

For this we first calculate $\mathbf{Q}^T \mathbf{b}$:

$$
\mathbf{Q}^T = \begin{bmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \end{bmatrix}, \quad \mathbf{Q}^T \mathbf{b} = \begin{bmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \end{bmatrix} \begin{bmatrix} 3 \\ 4 \\ 5 \end{bmatrix} = \begin{bmatrix} 3 \\ 4 \end{bmatrix}.
$$

and then the projection $\text{proj}_{C(\mathbf{Q})}(\mathbf{b})$:

$$
\text{proj}_{C(\mathbf{Q})}(\mathbf{b}) = \mathbf{Q} (\mathbf{Q}^T \mathbf{b}) = \begin{bmatrix} 1 & 0 \\ 0 & 1 \\ 0 & 0 \end{bmatrix} \begin{bmatrix} 3 \\ 4 \end{bmatrix} = \begin{bmatrix} 3 \\ 4 \\ 0 \end{bmatrix}.
$$

Thus, the projection of $\mathbf{b}$ onto the $xy$-plane is $\begin{bmatrix} 3 \\ 4 \\ 0 \end{bmatrix}$.
{{< /callout >}}

## Projections of Sets and the Farkas Lemma

If we have the matrix $\mathbf{A} \in \mathbb{Q}^{m \times n}$ and the vector $\mathbf{b} \in \mathbb{Q}^m$, then their either exists a vector $\mathbf{x} \in \mathbb{R}^n$ such that $\mathbf{A}\mathbf{x} \leq \mathbf{b}$ or there exists a vector $\mathbf{y} \in \mathbb{R}^m$ such that $\mathbf{y} \geq 0$ and $\mathbf{y}^T\mathbf{A} = 0$ and $\mathbf{y}^T\mathbf{b} < 0$. 

Basically this means that either the set defined by the linear inequalities $\mathbf{A}\mathbf{x} \leq \mathbf{b}$ which is a polyhedron is non-empty or the set defined by the linear inequalities $\mathbf{y} \geq 0$, $\mathbf{y}^T\mathbf{A} = 0$ and $\mathbf{y}^T\mathbf{b} < 0$ is non-empty. Meaning we can find a certificate $\mathbf{y}$ that proves that the set defined by the linear inequalities $\mathbf{A}\mathbf{x} \leq \mathbf{b}$ is empty.

The idea here is that we project onto the polyhedron step by step taking a dimension away at a time i.e rewriting inequalities in a reduced space, preserving polyhedral structure. At the final step, if the resulting polyhedron is non-empty, the original polyhedron was non-empty. If it was empty, one applies Farkas Lemma to produce a feasibility certificate (the vector $\mathbf{y}$) that proves the original polyhedron is empty. So in other words to show that their exists a solution to the linear inequalities we see if we can find a certificate that proves that set polyhedron is empty, if we can find such a certificate then the set is empty, if we cannot find such a certificate then the set is non-empty.