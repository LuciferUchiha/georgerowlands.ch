---
title: Matrix Completion
type: docs
weight: 12
---

Imagine the table of star ratings that a streaming service holds. Rows are users, columns are movies, and a cell holds the rating a user gave a movie. The overwhelming majority of cells are empty, since no one watches more than a tiny fraction of the catalogue. If we could guess the missing entries we could recommend each user the movies they are predicted to love. Stated abstractly the task is **matrix completion**: given a matrix in which only a small fraction of entries are observed, fill in the rest.

$$
\mathbf{A} = \begin{bmatrix} 3 & ? & ? & 2 \\ 0 & 1 & 2 & ? \\ ? & ? & -2 & -4 \\ ? & 5 & ? & ? \end{bmatrix} \quad \xrightarrow{\text{completion}} \quad \hat{\mathbf{A}}
$$

The plan is the following. Completion is impossible unless the entries are coupled, so we first pin down the weakest coupling that makes the problem well-posed, which leads to the **low-rank model** $\mathbf{A} \approx \mathbf{U}\mathbf{V}^\top$. We then solve the easy warm-up where the matrix is **fully observed**, whose answer is the [singular value decomposition](/garden/maths/linearAlgebra/eigendecomposition#singular-value-decomposition), and use it as a building block for the genuinely hard **partially observed** case, where we need iterative algorithms. Finally we ask when exact recovery is even possible. Throughout, the low-rank factorization is the close cousin of the [linear autoencoder](/garden/ml/computerVision/autoencoders#linear-autoencoders), since both impose a rank constraint to force a compressed, shared representation, and both lead straight to eigenvectors and singular values.

## Recommender Systems

The application that drives all of this is the **recommender system**, which analyses patterns of interest in items such as products or movies and produces personalized recommendations. The Netflix prize data is the canonical example, a matrix of users against movies with one-to-five star ratings, where fewer than one percent of the entries are filled in. Predicting the missing stars is exactly matrix completion, and a high predicted rating becomes a recommendation.

The strategy we use is **collaborative filtering**, which exploits the collective data of many users and generalizes across users, and possibly across items, rather than treating each user in isolation. The word collaborative captures the idea that users help fill in each other's ratings. This is in contrast to **content-based** filtering, which recommends items similar to ones a user already liked using features of the items themselves, such as genre or director. Collaborative filtering uses no such features and relies only on the rating matrix. Both are instances of **algorithmic selection**, the broad family of systems that decide which content to put in front of a user, and the same matrix completion machinery powers recommendations at Amazon, Netflix, YouTube, and online advertising.

## When is Matrix Completion Possible?

Filling in a matrix from a few entries sounds impossible, and without any assumption it is. Suppose the entries were drawn independently of one another, $a_{ij} \overset{\text{iid}}{\sim} p$, so that the joint distribution factorizes completely,

$$
P(\mathbf{A}) = \prod_{ij} p(a_{ij})
$$

Then no entry carries any information about any other. Using [mutual information](/garden/ml/bayesian/activelearning#mutual-information) as the measure of shared information, independence means $I(a_{ij}, a_{kl}) = 0$ for every pair $(i,j) \neq (k,l)$. Observing some entries would tell us exactly nothing about the unobserved ones, so the problem would be ill-posed. To make progress we must assume the entries are coupled.

### The Minimal Dependence Assumption

What is the weakest coupling that still deserves to be called a matrix rather than a bag of numbers? The answer is the identity of rows and columns. A matrix is more than a list of values because each value sits at the intersection of a particular row and a particular column. The minimal modelling assumption is therefore that entries sharing a row or a column are not independent. Two users who both rate a movie highly should make us update our belief about the other movies they have in common.

We can phrase this precisely with [conditional independence](/garden/ml/bayesian/activelearning#information-theory). The assumption is that an entry $a_{ij}$ is independent of all entries that share neither its row nor its column, once we condition on the entries that do share its row or column,

$$
a_{ij} \perp \{a_{kl} : k \neq i,\ l \neq j\} \ \mid\ \underbrace{\{a_{il} : l \neq j\}}_{\text{same row } i} \ \cup\ \underbrace{\{a_{kj} : k \neq i\}}_{\text{same column } j}
$$

In words, the rest of user $i$'s ratings and the rest of movie $j$'s ratings are all we need. Given those, the far-away entries add nothing. This is a strong structural assumption, and it relies on row and column identity rather than on looking at the marginal distribution of similar users directly.

{{< callout type="info" title="Conditional Independence is not Marginal Independence" >}}
Conditional independence given the row and column does not mean the entries are marginally independent. Because the matrix is only partially observed, the conditioning sets themselves are mostly unknown, and the unobserved entries become coupled indirectly. User $i$ and user $k$ may share no observed movie, yet a chain of intermediate users and movies links them. The effect is an indirect coupling between essentially all entries, which is exactly what lets information propagate from the observed cells into the unobserved ones.
{{< /callout >}}

## Formalization

We collect the data into a **rating matrix**

$$
\mathbf{A} \in \mathbb{R}^{n \times m}, \qquad n = \text{number of users (rows)}, \quad m = \text{number of items (columns)}
$$

so that $a_{ij}$ is the rating user $i$ gave item $j$. Since most entries are missing, we record which ones we actually see in a separate **observation matrix**

$$
\mathbf{\Omega} \in \{0, 1\}^{n \times m}, \qquad \omega_{ij} = 1 \iff a_{ij} \text{ is observed}
$$

The total number of observed ratings is $\sum_{i,j} \omega_{ij}$, and on real data this is a tiny fraction of the $nm$ cells, often one percent or less.

### Preprocessing

Before fitting any model it is worth asking what invariances the data has. Is a rating meaningful in absolute terms, or only relative to a user's habits? Is the scale the same for every user and item? These questions motivate two standard preprocessing steps.

**Centering** subtracts a mean to remove an offset. We can subtract either the mean of a user (a row) or the mean of an item (a column), computed over observed entries only,

$$
\mu^{\text{row}}_i = \frac{\sum_j \omega_{ij} a_{ij}}{\max\{1, \sum_j \omega_{ij}\}}, \qquad \mu^{\text{col}}_j = \frac{\sum_i \omega_{ij} a_{ij}}{\max\{1, \sum_i \omega_{ij}\}}
$$

The $\max\{1, \cdot\}$ in the denominator guards against dividing by zero for a user or item with no ratings. The two choices remove two different biases.

- **Centering by user** subtracts the row mean, $a_{ij} \mapsto a_{ij} - \mu^{\text{row}}_i$. This removes **rating bias**. A generous user who rates everything four or five stars and a harsh user who rates everything one or two become comparable once we look at how far each rating sits above or below that user's own average.
- **Centering by item** subtracts the column mean, $a_{ij} \mapsto a_{ij} - \mu^{\text{col}}_j$. This removes **item popularity bias**. A universally loved movie has a high column average, and subtracting it leaves how much each user liked the movie relative to everyone else, which is the part that actually distinguishes users.

**Variance normalization** goes one step further and also rescales to unit variance. For a rating $X$ with mean $\mu$ and variance $\sigma^2$ the standardized **z-score** is

$$
Z = \frac{X - \mu}{\sigma}
$$

where again $\mu$ and $\sigma$ are taken over a chosen row or column. Centering fixes the offset, and dividing by $\sigma$ fixes the spread, so the two together turn raw ratings into z-scores. The interpretation is that users differ not only in their average rating but in how much of the scale they use. One user spreads ratings across the full one-to-five range while another only ever gives threes and fours. Dividing by that user's standard deviation makes a rating one standard deviation above the mean mean the same level of enthusiasm for everyone.

Whether to normalize by rows or by columns is a modelling choice. Early collaborative filtering systems in the 1990s normalized per user, reasoning that this corrects for users who are generally more positive or who use a different part of the rating scale. It was later a notable finding (Sarwar et al., 2001) that normalizing per item, the transposed view, tends to work better. In practice it is worth trying both.

## The Low-Rank Model

We now commit to a model. We approximate the rating matrix by a product of two thin matrices,

$$
\mathbf{A} \approx \mathbf{U}\mathbf{V}^\top, \qquad \mathbf{U} \in \mathbb{R}^{n \times k}, \quad \mathbf{V} \in \mathbb{R}^{m \times k}
$$

Here $k$ is small, far smaller than $n$ or $m$. Writing $\mathbf{u}_i \in \mathbb{R}^k$ for the $i$-th row of $\mathbf{U}$ and $\mathbf{v}_j \in \mathbb{R}^k$ for the $j$-th row of $\mathbf{V}$, each entry of the approximation is an inner product of two $k$-dimensional vectors,

$$
a_{ij} \approx \mathbf{u}_i^\top \mathbf{v}_j = \sum_{r=1}^k u_{ir}\, v_{jr}
$$

This is the heart of collaborative filtering. Each user $i$ is summarized by a short vector $\mathbf{u}_i$ of **latent factors**, and each item $j$ by a vector $\mathbf{v}_j$ of the same $k$ factors. Read the factors off coordinate by coordinate. If the first factor is "amount of action", then $u_{i1}$ measures how much user $i$ likes action and $v_{j1}$ measures how much action movie $j$ has. Their product $u_{i1} v_{j1}$ is large exactly when an action-loving user meets an action-packed movie, and summing such products over all $k$ factors gives the predicted rating. The prediction is high when the user's tastes line up with the item's attributes across the board. We never hand-label the factors, the model discovers whatever $k$ directions best explain the observed ratings.

{{< callout type="example" title="A Two-Factor Model" >}}
Take $k = 2$ with factors "action" and "romance". A user who loves action and dislikes romance has $\mathbf{u}_i = (2, -1)$, and an action blockbuster with little romance has $\mathbf{v}_j = (3, 0)$. The predicted rating is

$$
\mathbf{u}_i^\top \mathbf{v}_j = 2 \cdot 3 + (-1) \cdot 0 = 6
$$

a strong recommendation. A romantic drama $\mathbf{v}_{j'} = (0, 2)$ instead gives $\mathbf{u}_i^\top \mathbf{v}_{j'} = 2 \cdot 0 + (-1) \cdot 2 = -2$, correctly predicting that this user would dislike it.
{{< /callout >}}

The product $\mathbf{U}\mathbf{V}^\top$ has [rank](/garden/maths/linearAlgebra/matrixRanks) at most $k$, so the model says the rating matrix is approximately low rank. This is exactly the rank constraint of a [linear autoencoder](/garden/ml/computerVision/autoencoders#the-rank-constraint), and the bottleneck dimension $k$ plays the role of the autoencoder's latent dimension $d$. Both models force the data through $k$ shared factors. The difference is where the factors come from. A linear autoencoder is handed a data matrix and reconstructs each column as $\mathbf{V}\mathbf{W}\mathbf{x}$, so one of its two factors, the codes $\mathbf{W}\mathbf{x}$, is pinned to the input by the encoder. In matrix factorization there is no separate input. The matrix $\mathbf{A}$ is the only data, and both $\mathbf{U}$ and $\mathbf{V}$ are free parameters we learn from scratch. A low-rank matrix also satisfies the minimal dependence assumption from before, since once the $k$-dimensional factors of row $i$ and column $j$ are fixed, the entry $a_{ij} = \mathbf{u}_i^\top \mathbf{v}_j$ is determined by exactly the row and column information we conditioned on.

### The Objective

We fit the factors by minimizing squared error on the observed entries. Using the [Hadamard product](/garden/maths/linearAlgebra/hadamardProduct) $\odot$ for the elementwise product, define the **projection operator** that keeps observed entries and zeros the rest,

$$
\Pi_\Omega(\mathbf{R}) = \mathbf{R} \odot \mathbf{\Omega}, \qquad (\Pi_\Omega(\mathbf{R}))_{ij} = \omega_{ij}\, r_{ij}
$$

The objective is the squared [Frobenius norm](/garden/maths/linearAlgebra/matrices) of the residual, confined to the observed cells,

$$
\min_{\mathbf{U}, \mathbf{V}}\ \frac{1}{2}\big\| \Pi_\Omega(\mathbf{A} - \mathbf{U}\mathbf{V}^\top) \big\|_F^2, \qquad \|\mathbf{R}\|_F^2 = \sum_{ij} r_{ij}^2
$$

The projection operator is what makes this matrix completion rather than plain matrix approximation. It tells the loss to ignore the unobserved entries entirely, fitting the factors only to what we have seen. Once the factors are fit, we read off the full product $\mathbf{U}\mathbf{V}^\top$ to predict the missing entries.

{{< callout type="info" title="Non-Identifiability of the Factors" >}}
The factorization is not unique. For any invertible $\mathbf{B} \in \mathbb{R}^{k \times k}$ we can write

$$
\mathbf{U}\mathbf{V}^\top = \mathbf{U}\mathbf{B}\mathbf{B}^{-1}\mathbf{V}^\top = (\mathbf{U}\mathbf{B})(\mathbf{V}\mathbf{B}^{-\top})^\top
$$

so $\mathbf{U}\mathbf{B}$ and $\mathbf{V}\mathbf{B}^{-\top}$ give exactly the same product and the same predictions. The simplest case is rescaling a single rank-1 factor, $\mathbf{u} \mapsto \lambda \mathbf{u}$ and $\mathbf{v} \mapsto \tfrac{1}{\lambda}\mathbf{v}$. This is the same non-identifiability as the [autoencoder weight matrices](/garden/ml/computerVision/autoencoders#uniqueness-of-the-solution) $\mathbf{P} = \mathbf{V}\mathbf{W}$, and it means we should not read meaning into the individual factors. Only the product, or the subspace it spans, is determined.
{{< /callout >}}

## The Rank-1 Model

To understand the gradients and the geometry of the objective we start with the smallest interesting case, $k = 1$, and we drop the observation mask so the matrix is fully observed. The model is now an **outer product** of two vectors,

$$
\mathbf{A} \approx \mathbf{u}\mathbf{v}^\top, \qquad \mathbf{u} \in \mathbb{R}^n,\ \mathbf{v} \in \mathbb{R}^m, \qquad a_{ij} \approx u_i v_j
$$

This is a **bilinear model**, meaning it is linear in each vector separately when the other is held fixed. Fixing $\mathbf{v}$, the map $\mathbf{u} \mapsto \mathbf{u}\mathbf{v}^\top$ is linear in $\mathbf{u}$, and symmetrically for $\mathbf{v}$. It is not jointly linear, since the product $u_i v_j$ couples the two, and that coupling is what makes the objective non-convex, as we will see. It is the minimal model that couples a whole row and a whole column at once, because changing a single $u_i$ rescales the entire row $i$ and changing a single $v_j$ rescales the entire column $j$.

The matrix $\mathbf{u}\mathbf{v}^\top$ is the prototypical rank-1 matrix. In fact rank-1 matrices and outer products are the same thing. Any outer product $\mathbf{u}\mathbf{v}^\top$ has rank $1$, since every column is a multiple of $\mathbf{u}$, and conversely any rank-1 matrix can be written as such an outer product.

### The Scalar Problem

Strip the problem down even further to the case where both factors are single numbers. The objective is

$$
\ell(u, v) = \frac{1}{2}(a - uv)^2, \qquad u, v \in \mathbb{R}
$$

which just asks for two numbers whose product equals a given third number $a$. To find the critical points we compute the two partial derivatives and set them to zero. By the chain rule,

$$
\frac{\partial \ell}{\partial u} = (a - uv)\cdot(-v) = v(uv - a), \qquad \frac{\partial \ell}{\partial v} = (a - uv)\cdot(-u) = u(uv - a)
$$

Setting both to zero, $v(uv - a) = 0$ and $u(uv - a) = 0$, there are two ways to satisfy them.

- **The minima.** If $uv = a$ then both equations hold and the loss is zero. Every pair $(u, v)$ whose product is $a$ is a global minimum. This solution set is the curve $uv = a$, a hyperbola.
- **The origin.** If instead $u = v = 0$ then both equations also hold, but now $uv = 0 \neq a$ (assuming $a \neq 0$), so the loss is $\tfrac{1}{2}a^2 > 0$. The origin is a critical point that is not a minimum.

{{< figure
    src="/images/ml/matrixCompletionScalarField.png"
    alt="Gradient field of the scalar loss with the hyperbola of minima and a saddle at the origin."
    caption="The negative gradient flow of $\frac{1}{2}(a-uv)^2$ for $a=1$. Every point on the hyperbola $uv=a$ (orange, two branches) is a global minimum with zero loss, and the origin is a saddle point that gradient descent has to escape."
>}}

The hyperbola $uv = a$ falls into two separate pieces, called **branches**. For $a > 0$ one branch sits in the quadrant where $u$ and $v$ are both positive and the other where they are both negative, since in either case the product is positive. Gradient descent slides downhill onto one of these branches, and which one it reaches depends on where it starts. The origin sits wedged between the branches, and it is the interesting case, because a critical point that is not a minimum is a **saddle point**.

{{< callout type="proof" title="The Origin is a Saddle Point" >}}
We classify the origin using the Hessian, the matrix of second derivatives. Differentiating the gradients above,

$$
\nabla^2 \ell(u, v) = \begin{pmatrix} v^2 & 2uv - a \\ 2uv - a & u^2 \end{pmatrix}, \qquad \nabla^2 \ell(0, 0) = \begin{pmatrix} 0 & -a \\ -a & 0 \end{pmatrix}
$$

A critical point is a minimum when the Hessian is positive definite (all eigenvalues positive) and a saddle when it has both positive and negative eigenvalues. We find the eigenvalues of the Hessian at the origin from its characteristic polynomial,

$$
\det\big(\lambda \mathbf{I} - \nabla^2 \ell(0,0)\big) = \det\begin{pmatrix} \lambda & a \\ a & \lambda \end{pmatrix} = \lambda^2 - a^2 = (\lambda - a)(\lambda + a)
$$

so the eigenvalues are $\pm a$. With one positive and one negative eigenvalue the Hessian is **indefinite**, which is the signature of a saddle point. The loss curves up along one eigenvector and down along the other.
{{< /callout >}}

An indefinite Hessian means the objective is **non-convex**, except in the degenerate case $a = 0$. This matters because convexity is the dividing line in optimization between problems we can reliably solve and problems we cannot. It is worth recording the three equivalent ways to state convexity, in increasing order of derivative information.

{{< callout type="info" title="Convexity" >}}
A set is **convex** if the line segment joining any two of its points stays inside the set. For a function $f$ on a convex domain there are three equivalent conditions.

- **Zeroth order (definition).** For all $\mathbf{x}, \mathbf{y}$ and all $t \in [0, 1]$,

$$
f(t\mathbf{x} + (1 - t)\mathbf{y}) \le t f(\mathbf{x}) + (1 - t) f(\mathbf{y})
$$

the chord lies above the graph.

- **First order.** If $f$ is differentiable,

$$
f(\mathbf{x}) \ge f(\mathbf{y}) + \nabla f(\mathbf{y})^\top (\mathbf{x} - \mathbf{y})
$$

the graph lies above each of its tangent planes.

- **Second order.** If $f$ is twice differentiable, the [Hessian is positive semi-definite](/garden/maths/linearAlgebra/eigendecomposition#positive-definite-and-positive-semi-definite-matrices),

$$
\nabla^2 f(\mathbf{x}) \succeq 0
$$

everywhere on the interior, so the surface curves upward in every direction.

Our scalar loss fails the second-order test at the origin, where the Hessian has a negative eigenvalue, so it is non-convex.
{{< /callout >}}

Setting the gradient to zero told us where the critical points are, but not which one gradient descent reaches or how the saddle slows it down. To see the dynamics we study the **gradient flow**, the continuous-time limit of gradient descent. A gradient descent step is $u_{t+1} = u_t - \eta\, \partial_u \ell$, and as the step size $\eta \to 0$ the discrete iterates trace out a smooth curve $u(t)$ whose velocity is the negative gradient. This curve solves an **ordinary differential equation**, an equation that specifies the instantaneous rate of change of a quantity,

$$
\frac{du}{dt} = -\frac{\partial \ell}{\partial u} = -v(uv - a), \qquad \frac{dv}{dt} = -\frac{\partial \ell}{\partial v} = -u(uv - a)
$$

Solving the ODE gives the exact trajectory, which is easier to reason about than the discrete steps. To make it tractable, start from a **balanced initialization** $u_0 = v_0$. The two equations are symmetric in $u$ and $v$, so if they start equal they stay equal, $u(t) = v(t)$ for all time. Then the natural quantity to track is the product $x = uv = u^2$, which is what the model actually predicts. Differentiating with the chain rule and substituting $\tfrac{du}{dt} = -v(uv-a) = -u(x-a)$,

$$
\frac{dx}{dt} = \frac{d(u^2)}{dt} = 2u\frac{du}{dt} = 2u\big(-u(x - a)\big) = -2u^2(x - a) = -2x(x - a)
$$

This is now a single ODE in one variable, and it has the closed-form solution

$$
x(t) = a + \frac{ac - a^2}{c\, e^{2at} + a - c}, \qquad c = x(0)
$$

which one can verify by differentiation or obtain from a symbolic solver. Reading off its behaviour is the point. At $t = 0$ the expression equals $c$, the initial product, and as $t \to \infty$ for $a > 0$ the exponential $e^{2at}$ dominates the denominator and drives $x(t) \to a$. So the predicted value $uv$ rises from its starting point and saturates at the target $a$, tracing an S-shaped curve.

{{< figure
    src="/images/ml/matrixCompletionScalarODE.png"
    alt="The gradient-flow solution x(t) for three initializations, all saturating at the target."
    caption="The gradient flow $x(t)=uv$ for three initializations, with target $a=1$. A small initialization $x(0)=0.01$ starts near the saddle $x=0$ and lingers there for a long time before the dynamics escape and saturate at $a$."
>}}

The initialization $c$ controls how long the trajectory lingers near the saddle. A very small $c$ starts close to the saddle point $x = 0$, where the gradient is nearly zero, so the trajectory crawls for a long time before the exponential growth kicks in and it escapes toward $a$. This slow escape from a saddle is a phenomenon that carries over to the full matrix problem and to deep networks in general.

The reassuring news is that matrix completion, despite being non-convex, is a problem we can still analyse in depth, approximate well in practice, and even solve exactly in special cases. The rest of the note makes good on that promise.

### From the Scalar Case Back to the Rank-1 Matrix

Returning to vectors, the loss for the fully observed rank-1 model is

$$
\ell(\mathbf{u}, \mathbf{v}) = \frac{1}{2}\|\mathbf{A} - \mathbf{u}\mathbf{v}^\top\|_F^2 = \frac{1}{2}\sum_{i,j}(a_{ij} - u_i v_j)^2
$$

We can find the gradient component by component, exactly as in the scalar case. Differentiating the sum with respect to a single $u_i$, only the terms in row $i$ survive,

$$
\frac{\partial \ell}{\partial u_i} = \sum_j (a_{ij} - u_i v_j)(-v_j) = \sum_j (u_i v_j - a_{ij}) v_j
$$

Collecting these into a vector, and writing the **residual matrix** $\mathbf{R} = \mathbf{u}\mathbf{v}^\top - \mathbf{A}$ purely as shorthand for the difference between the model and the data, the sum $\sum_j (u_i v_j - a_{ij}) v_j$ is the $i$-th entry of $\mathbf{R}\mathbf{v}$. The same computation for $\mathbf{v}$ gives

$$
\nabla_{\mathbf{u}} \ell = \mathbf{R}\mathbf{v}, \qquad \nabla_{\mathbf{v}} \ell = \mathbf{R}^\top \mathbf{u}, \qquad \mathbf{R} = \mathbf{u}\mathbf{v}^\top - \mathbf{A}
$$

These are the direct generalization of the scalar derivatives $v(uv-a)$ and $u(uv-a)$, with the scalar gap $uv - a$ replaced by the residual matrix $\mathbf{R}$.

The Hessian generalizes too. Stack the variables into one long vector $(\mathbf{u}, \mathbf{v})$ of length $n + m$. The Hessian is the block matrix of second derivatives, and we obtain each block by differentiating the gradients above. For example $\nabla_{\mathbf{u}} \ell = \mathbf{R}\mathbf{v} = (\mathbf{v}^\top\mathbf{v})\mathbf{u} - \mathbf{A}\mathbf{v} = \|\mathbf{v}\|^2 \mathbf{u} - \mathbf{A}\mathbf{v}$, whose derivative with respect to $\mathbf{u}$ is $\|\mathbf{v}\|^2 \mathbf{I}_n$ and with respect to $\mathbf{v}$ is $2\mathbf{u}\mathbf{v}^\top - \mathbf{A}$. Assembling all four blocks,

$$
\nabla^2 \ell(\mathbf{u}, \mathbf{v}) = \begin{pmatrix} \|\mathbf{v}\|^2 \mathbf{I}_{n} & 2\mathbf{u}\mathbf{v}^\top - \mathbf{A} \\ (2\mathbf{u}\mathbf{v}^\top - \mathbf{A})^\top & \|\mathbf{u}\|^2 \mathbf{I}_{m} \end{pmatrix}, \qquad \nabla^2 \ell(\mathbf{0}, \mathbf{0}) = \begin{pmatrix} \mathbf{0}_{n} & -\mathbf{A} \\ -\mathbf{A}^\top & \mathbf{0}_{m} \end{pmatrix}
$$

At the origin the diagonal blocks vanish and only the $-\mathbf{A}$ blocks remain. We do not need to compute the eigenvalues explicitly to classify it. The Hessian is symmetric, so its eigenvalues are real, and they sum to the trace, which here is zero because the diagonal is all zeros. A set of real numbers that sums to zero, and is not all zeros, must contain both a positive and a negative value. So as long as $\mathbf{A} \neq \mathbf{0}$ the Hessian is indefinite, the origin is a saddle point, and the problem is non-convex for every $n, m \ge 1$. This is exactly the scalar conclusion, where the eigenvalues were $\pm a$ and summed to zero, lifted to the matrix case.

### The Optimal Rank-1 Approximation

Despite the non-convexity we can find the global optimum in closed form. The key is to rewrite the loss so that the dependence on $\mathbf{u}$ and $\mathbf{v}$ becomes transparent. We use the identity $\|\mathbf{R}\|_F^2 = \text{tr}(\mathbf{R}^\top \mathbf{R})$, which holds because the $j$-th diagonal entry of $\mathbf{R}^\top \mathbf{R}$ is $\sum_i R_{ij}^2$, the squared norm of column $j$, and summing these diagonal entries with the trace recovers the sum of all squared entries,

$$
\text{tr}(\mathbf{R}^\top \mathbf{R}) = \sum_j (\mathbf{R}^\top \mathbf{R})_{jj} = \sum_j \sum_i R_{ij}^2 = \sum_{ij} R_{ij}^2 = \|\mathbf{R}\|_F^2
$$

{{< callout type="proof" title="Rewriting the Rank-1 Objective" >}}
Expand the squared residual with the trace identity and multiply out, using that the trace is linear and cyclic, meaning $\text{tr}(\mathbf{X}\mathbf{Y}) = \text{tr}(\mathbf{Y}\mathbf{X})$,

$$
\begin{aligned}
\frac{1}{2}\|\mathbf{A} - \mathbf{u}\mathbf{v}^\top\|_F^2 &= \frac{1}{2}\text{tr}\big((\mathbf{A} - \mathbf{u}\mathbf{v}^\top)^\top(\mathbf{A} - \mathbf{u}\mathbf{v}^\top)\big) \\
&= \frac{1}{2}\text{tr}\big(\mathbf{A}^\top \mathbf{A} - \mathbf{A}^\top\mathbf{u}\mathbf{v}^\top - \mathbf{v}\mathbf{u}^\top \mathbf{A} + \mathbf{v}\mathbf{u}^\top \mathbf{u}\mathbf{v}^\top\big) \\
&= \frac{1}{2}\underbrace{\text{tr}(\mathbf{A}^\top \mathbf{A})}_{\|\mathbf{A}\|_F^2,\ \text{const}} - \underbrace{\text{tr}(\mathbf{A}^\top\mathbf{u}\mathbf{v}^\top)}_{\mathbf{u}^\top \mathbf{A}\mathbf{v}} + \frac{1}{2}\underbrace{\text{tr}(\mathbf{v}\mathbf{u}^\top \mathbf{u}\mathbf{v}^\top)}_{\|\mathbf{u}\|^2\|\mathbf{v}\|^2}
\end{aligned}
$$

The two cross terms are equal scalars, since each equals its own trace and $\text{tr}(\mathbf{A}^\top\mathbf{u}\mathbf{v}^\top) = \text{tr}(\mathbf{v}^\top\mathbf{A}^\top\mathbf{u}) = \mathbf{u}^\top\mathbf{A}\mathbf{v}$, which is why they combine into a single $-\mathbf{u}^\top\mathbf{A}\mathbf{v}$. The last term factors as $\text{tr}(\mathbf{u}^\top\mathbf{u}\,\mathbf{v}^\top\mathbf{v}) = \|\mathbf{u}\|^2\|\mathbf{v}\|^2$. Dropping the constant $\tfrac{1}{2}\|\mathbf{A}\|_F^2$,

$$
\ell(\mathbf{u}, \mathbf{v}) = \frac{1}{2}\|\mathbf{u}\|^2 \|\mathbf{v}\|^2 - \mathbf{u}^\top \mathbf{A}\mathbf{v} + \text{const}
$$
{{< /callout >}}

This rewriting cleanly separates magnitude from direction. The first term depends only on the lengths $\|\mathbf{u}\|$ and $\|\mathbf{v}\|$, while the **direction** of the factors enters only through the bilinear term $\mathbf{u}^\top \mathbf{A}\mathbf{v}$. So if we fix the lengths and choose only directions, minimizing the loss means maximizing $\mathbf{u}^\top \mathbf{A}\mathbf{v}$. Restricting to unit vectors, the directions solve

$$
\max_{\mathbf{u}, \mathbf{v}}\ \mathbf{u}^\top \mathbf{A}\mathbf{v} \quad \text{s.t.}\quad \|\mathbf{u}\| = \|\mathbf{v}\| = 1
$$

{{< callout type="proof" title="Optimal Directions are Principal Eigenvectors" >}}
We solve the constrained problem with [Lagrange multipliers](/garden/maths/calculus/convexOptimization#the-method-of-lagrange-multipliers). The Lagrangian is

$$
\mathcal{L} = \mathbf{u}^\top \mathbf{A}\mathbf{v} - \mu_1(\mathbf{u}^\top \mathbf{u} - 1) - \mu_2(\mathbf{v}^\top \mathbf{v} - 1)
$$

Setting the gradient with respect to $\mathbf{u}$ to zero and solving,

$$
\nabla_{\mathbf{u}} \mathcal{L} = \mathbf{A}\mathbf{v} - 2\mu_1 \mathbf{u} = \mathbf{0} \implies \mathbf{u} = \frac{\mathbf{A}\mathbf{v}}{\|\mathbf{A}\mathbf{v}\|}
$$

where the normalization comes from the unit-length constraint $\|\mathbf{u}\| = 1$. By symmetry the condition on $\mathbf{v}$ is $\mathbf{v} = \mathbf{A}^\top \mathbf{u} / \|\mathbf{A}^\top \mathbf{u}\|$. Now substitute one into the other. Up to positive scalars, $\mathbf{u}$ is proportional to $\mathbf{A}\mathbf{v}$, and $\mathbf{v}$ is proportional to $\mathbf{A}^\top \mathbf{u}$, so

$$
\mathbf{u} \propto \mathbf{A}\mathbf{v} \propto \mathbf{A}(\mathbf{A}^\top \mathbf{u}) = (\mathbf{A}\mathbf{A}^\top)\mathbf{u}, \qquad \mathbf{v} \propto (\mathbf{A}^\top \mathbf{A})\mathbf{v}
$$

The relation $(\mathbf{A}\mathbf{A}^\top)\mathbf{u} = \lambda \mathbf{u}$ says precisely that $\mathbf{u}$ is an eigenvector of $\mathbf{A}\mathbf{A}^\top$, and likewise $\mathbf{v}$ is an eigenvector of $\mathbf{A}^\top \mathbf{A}$. To maximize $\mathbf{u}^\top \mathbf{A}\mathbf{v}$ we pick the **principal** eigenvectors, those with the largest eigenvalue.
{{< /callout >}}

The optimal rank-1 approximation is therefore built from the leading eigenvectors of $\mathbf{A}\mathbf{A}^\top$ and $\mathbf{A}^\top \mathbf{A}$,

$$
\mathbf{A} \approx \sigma_1 \mathbf{u}_1 \mathbf{v}_1^\top
$$

where $\mathbf{u}_1$ and $\mathbf{v}_1$ are the principal unit eigenvectors and $\sigma_1 = \sqrt{\lambda_1}$ is the square root of the principal eigenvalue $\lambda_1$, which is shared by $\mathbf{A}\mathbf{A}^\top$ and $\mathbf{A}^\top \mathbf{A}$. These quantities are precisely the leading **singular value** and **singular vectors** of $\mathbf{A}$. We can compute them with the [power method](/garden/ml/computerVision/autoencoders#power-method), the same iteration used for the principal eigenvector in PCA.

This points the way to the general case. The rank-1 solution is the top singular component of $\mathbf{A}$, so the rank-$k$ solution should be the top $k$ singular components. Making that precise is the job of the singular value decomposition.

## The Singular Value Decomposition

For the fully observed problem at general rank $k$ the optimal factorization is given exactly by the [singular value decomposition](/garden/maths/linearAlgebra/eigendecomposition#singular-value-decomposition) (SVD), which we recall here and connect to matrix completion.

{{< callout type="info" title="SVD Theorem" >}}
Every matrix $\mathbf{A} \in \mathbb{R}^{n \times m}$ can be written as

$$
\mathbf{A} = \mathbf{U}\mathbf{\Sigma}\mathbf{V}^\top, \qquad \mathbf{\Sigma} = \text{diag}(\sigma_1, \ldots, \sigma_{\min\{n,m\}}), \quad \sigma_1 \ge \sigma_2 \ge \cdots \ge 0
$$

where $\mathbf{U} \in \mathbb{R}^{n \times n}$ and $\mathbf{V} \in \mathbb{R}^{m \times m}$ are orthogonal and $\mathbf{\Sigma} \in \mathbb{R}^{n \times m}$ is rectangular diagonal. The columns $\mathbf{u}_i$ of $\mathbf{U}$ are the **left singular vectors**, the columns $\mathbf{v}_i$ of $\mathbf{V}$ the **right singular vectors**, and the $\sigma_i$ the **singular values**. The set of singular values is unique. The singular vectors are unique up to sign for distinct singular values, and a repeated singular value determines a unique subspace but not a unique basis within it.
{{< /callout >}}

Geometrically the singular values measure how much $\mathbf{A}$ stretches space. The defining relations $\mathbf{A}\mathbf{v}_i = \sigma_i \mathbf{u}_i$ say that the right singular vector $\mathbf{v}_i$ is sent to the left singular vector $\mathbf{u}_i$, scaled by $\sigma_i$. The largest singular value $\sigma_1$ is the largest stretch the matrix applies to any unit vector, a statement we make precise below with the spectral norm.

### Relation to the Eigendecomposition

The SVD is tied to the [eigendecomposition](/garden/maths/linearAlgebra/eigendecomposition) of the two symmetric products of $\mathbf{A}$ with its transpose, which is the link we already met in the rank-1 case. Substituting the SVD and using orthogonality $\mathbf{U}^\top \mathbf{U} = \mathbf{I}$ and $\mathbf{V}^\top \mathbf{V} = \mathbf{I}$,

$$
\mathbf{A}\mathbf{A}^\top = \mathbf{U}\mathbf{\Sigma}\mathbf{\Sigma}^\top \mathbf{U}^\top = \mathbf{U}\,\text{diag}(\sigma_1^2, \ldots)\,\mathbf{U}^\top, \qquad \mathbf{A}^\top \mathbf{A} = \mathbf{V}\mathbf{\Sigma}^\top \mathbf{\Sigma}\mathbf{V}^\top = \mathbf{V}\,\text{diag}(\sigma_1^2, \ldots)\,\mathbf{V}^\top
$$

So the left singular vectors are the eigenvectors of $\mathbf{A}\mathbf{A}^\top$, the right singular vectors are the eigenvectors of $\mathbf{A}^\top \mathbf{A}$, and the singular values are the square roots of the shared eigenvalues. This is also why the two products have the same nonzero eigenvalues, and it is exactly what the rank-1 Lagrange derivation rediscovered. For a symmetric matrix the SVD and the eigendecomposition coincide up to signs, and for a [symmetric positive semi-definite](/garden/maths/linearAlgebra/eigendecomposition#positive-definite-and-positive-semi-definite-matrices) matrix they are identical. Applied to a centered data matrix, the SVD yields the principal eigenvectors of the covariance, which is exactly [principal component analysis](/garden/maths/linearAlgebra/pca). The SVD, the eigendecomposition, and PCA are three views of the same structure.

### Frobenius and Spectral Norms

Two matrix norms read directly off the singular values, and both reappear in matrix completion. The Frobenius norm collects them in the 2-norm and the spectral norm picks out the largest.

{{< callout type="proof" title="Frobenius Norm via the SVD" >}}
Using $\|\mathbf{A}\|_F^2 = \text{tr}(\mathbf{A}^\top \mathbf{A})$, the SVD, the cyclic property of the trace, and orthogonality,

$$
\|\mathbf{A}\|_F^2 = \text{tr}(\mathbf{A}^\top \mathbf{A}) = \text{tr}(\mathbf{V}\mathbf{\Sigma}^\top \mathbf{U}^\top \mathbf{U}\mathbf{\Sigma}\mathbf{V}^\top) = \text{tr}(\mathbf{\Sigma}^\top \mathbf{\Sigma}) = \sum_{i=1}^{\min\{n,m\}} \sigma_i^2
$$

So the Frobenius norm is the Euclidean norm of the vector of singular values, $\|\mathbf{A}\|_F = \|\boldsymbol{\sigma}(\mathbf{A})\|_2$.
{{< /callout >}}

{{< callout type="proof" title="Spectral Norm via the SVD" >}}
The **spectral norm** is the largest factor by which $\mathbf{A}$ stretches a unit vector,

$$
\|\mathbf{A}\|_2 := \sup\{\|\mathbf{A}\mathbf{x}\| : \|\mathbf{x}\| = 1\}
$$

Orthogonal matrices preserve the Euclidean norm, since $\|\mathbf{U}\mathbf{x}\|^2 = \mathbf{x}^\top \mathbf{U}^\top \mathbf{U}\mathbf{x} = \mathbf{x}^\top \mathbf{x} = \|\mathbf{x}\|^2$. Substituting the SVD and changing variables to $\mathbf{z} = \mathbf{V}^\top \mathbf{x}$, which is still a unit vector,

$$
\sup_{\|\mathbf{x}\|=1}\|\mathbf{U}\mathbf{\Sigma}\mathbf{V}^\top \mathbf{x}\| = \sup_{\|\mathbf{x}\|=1}\|\mathbf{\Sigma}\mathbf{V}^\top \mathbf{x}\| = \sup_{\|\mathbf{z}\|=1}\|\mathbf{\Sigma}\mathbf{z}\| = \sigma_1
$$

The last step uses that the 2-norm of a diagonal matrix is its largest absolute entry, achieved by putting all the weight on the largest singular value. So $\|\mathbf{A}\|_2 = \sigma_1$, the maximal stretch.
{{< /callout >}}

### Eckart-Young and the Fully Observed Solution

The central fact is that truncating the SVD gives the best low-rank approximation. This is the [Eckart-Young theorem](/garden/maths/linearAlgebra/eigendecomposition#low-rank-approximation), which we use here as the solution to the fully observed problem.

{{< callout type="info" title="Eckart-Young Theorem" >}}
Let $\mathbf{A} \in \mathbb{R}^{n \times m}$ have SVD $\mathbf{A} = \mathbf{U}\mathbf{\Sigma}\mathbf{V}^\top$. For any $1 \le k \le \min\{n, m\}$, the truncated SVD

$$
\mathbf{A}_k := \mathbf{U}\,\text{diag}(\sigma_1, \ldots, \sigma_k, 0, \ldots)\,\mathbf{V}^\top = \sum_{i=1}^k \sigma_i \mathbf{u}_i \mathbf{v}_i^\top
$$

is a best rank-$k$ approximation of $\mathbf{A}$ in the Frobenius norm,

$$
\mathbf{A}_k \in \arg\min\{\|\mathbf{A} - \mathbf{B}\|_F : \text{rank}(\mathbf{B}) \le k\}
$$

The same $\mathbf{A}_k$ is also optimal in the spectral norm. Approximations for every $k$ can be read off a single SVD by keeping more or fewer leading terms.
{{< /callout >}}

The approximation error is controlled by the singular values we discard. The error matrix $\mathbf{A} - \mathbf{A}_k = \mathbf{U}\,\text{diag}(0, \ldots, 0, \sigma_{k+1}, \ldots)\,\mathbf{V}^\top$ is itself in SVD form, with singular values $\sigma_{k+1}, \sigma_{k+2}, \ldots$ in decreasing order. Applying the two norm formulas to it,

$$
\|\mathbf{A} - \mathbf{A}_k\|_F^2 = \sum_{i=k+1}^{\text{rank}(\mathbf{A})} \sigma_i^2, \qquad \|\mathbf{A} - \mathbf{A}_k\|_2 = \sigma_{k+1}
$$

The Frobenius error sums the squares of all the discarded singular values, the total energy left in the tail of the spectrum. The spectral error is just the largest discarded singular value $\sigma_{k+1}$, because that is the top singular value of the error matrix. The more components we keep, the smaller both errors, and a sharp drop in the singular values marks a natural cutoff for $k$.

This settles the fully observed problem completely. The rank-$k$ model $\mathbf{A} \approx \mathbf{U}\mathbf{V}^\top$ with a fully observed matrix is solved by the top $k$ singular components,

$$
\mathbf{A} \approx \sum_{i=1}^k \sigma_i \mathbf{u}_i \mathbf{v}_i^\top
$$

an additive superposition of $k$ rank-1 outer products, each a leading left singular vector paired with its right singular vector and weighted by its singular value. Low-rank approximation is non-convex even here, since the set of rank-$k$ matrices is not convex, yet computing the SVD costs only $O(\min\{nm^2, mn^2\})$ operations. That cost is the work of the standard SVD algorithm on an $n \times m$ matrix, and it makes this a rare example of a non-convex problem we can solve globally and efficiently.

It is worth being clear about what we have and have not done. We have solved the **fully observed** case, where every entry of $\mathbf{A}$ is known. This is not yet matrix completion, where most entries are missing. But it is the engine we will keep reaching for, because the iterative completion algorithms in the next section repeatedly call the SVD as a subroutine to project onto low rank.

## Matrix Completion

Now we put the observation mask back. The real problem fits a rank-$k$ matrix to the observed entries only,

$$
\min_{\text{rank}(\mathbf{B}) \le k}\ \frac{1}{2}\big\| \Pi_\Omega(\mathbf{A} - \mathbf{B}) \big\|_F^2
$$

and this small change ruins the clean SVD solution. The SVD finds the best low-rank fit to a complete matrix, but here most of $\mathbf{A}$ is unknown, so there is nothing to feed it.

The obvious workaround is **SVD with imputation**: guess the missing entries first, for example by filling each one with its row or column mean, then run the SVD on the completed matrix. This is a bad idea. Suppose a niche movie has only a handful of ratings averaging $3.5$. Imputation sets all its other entries to $3.5$, and now the SVD sees a column that is almost constant at $3.5$ and faithfully fits that fabricated structure. The low-rank model ends up explaining our guesses rather than the data, and the imputed values dominate precisely because they are the majority of the matrix. With incomplete observations the SVD is not directly applicable.

Worse, there is no easy fix. To see what the masked objective really is, expand the Frobenius norm. Since $\omega_{ij} \in \{0,1\}$ we have $\omega_{ij}^2 = \omega_{ij}$, so

$$
\big\| \Pi_\Omega(\mathbf{A} - \mathbf{B}) \big\|_F^2 = \sum_{i,j} \big(\omega_{ij}(a_{ij} - b_{ij})\big)^2 = \sum_{i,j} \omega_{ij}(a_{ij} - b_{ij})^2
$$

which is a **weighted** least squares problem with $0/1$ weights $w_{ij} = \omega_{ij}$. Finding the best rank-$k$ matrix under such weights,

$$
\min_{\text{rank}(\mathbf{B}) = k}\ \sum_{i,j} w_{ij}(a_{ij} - b_{ij})^2
$$

is **NP-hard**, even for rank $k = 1$. The reason the fully observed case was easy is that all weights were equal, which let the orthogonal invariance of the Frobenius norm bring in the SVD. Unequal $0/1$ weights destroy that invariance, and choosing how to trade off errors across the observed pattern becomes a combinatorial problem with no closed form. The fully observed case was a lucky special case, not the rule. For the incomplete case we turn to approximation algorithms, which come in two families.

- **Parameterized methods** build the rank constraint into the model by writing $\mathbf{B} = \mathbf{U}\mathbf{V}^\top$ and optimizing the factors. Alternating Least Squares is the main example.
- **Projection methods** keep an unconstrained matrix $\mathbf{B}$ and repeatedly project it back toward low rank or low norm, using the SVD as a subroutine. Singular Value Projection and Singular Value Thresholding are the examples.

### Alternating Least Squares

The first approach goes back to the factored model $\mathbf{A} \approx \mathbf{U}\mathbf{V}^\top$ and adds a regularizer,

$$
\phi(\mathbf{U}, \mathbf{V}) = \frac{1}{2}\big\| \Pi_\Omega(\mathbf{A} - \mathbf{U}\mathbf{V}^\top) \big\|_F^2 + \frac{\lambda}{2}\big(\|\mathbf{U}\|_F^2 + \|\mathbf{V}\|_F^2\big), \qquad \lambda > 0
$$

We keep the transpose so that $\mathbf{u}_i$ and $\mathbf{v}_j$ are both $k$-vectors, the latent factors of user $i$ and item $j$. The regularizer is there for two linked reasons. Recall the non-identifiability $\mathbf{u} \mapsto \lambda\mathbf{u}, \mathbf{v} \mapsto \tfrac{1}{\lambda}\mathbf{v}$, which leaves the product fixed while sending $\|\mathbf{U}\|_F \to \infty$. Without a penalty the optimum is not even well defined, since the factors can blow up at no cost. Adding $\tfrac{\lambda}{2}(\|\mathbf{U}\|_F^2 + \|\mathbf{V}\|_F^2)$ pins down the scale, and statistically it is a ridge penalty, a Gaussian prior on the factors that prevents overfitting when an item or user has very few ratings. As we will see, it also makes every subproblem invertible.

This objective is non-convex in $(\mathbf{U}, \mathbf{V})$ jointly, which we should expect from the rank-1 analysis. But it has a structure worth uncovering. Look at a single squared error and expand it,

$$
\Big(a_{ij} - \sum_{r=1}^k u_{ir} v_{jr}\Big)^2 = a_{ij}^2 - 2 a_{ij}\sum_{r} u_{ir} v_{jr} + \sum_{r,s} u_{ir} v_{jr} u_{is} v_{js}
$$

The last term is a product of four parameters, so the objective is a **degree-4 polynomial** in the entries of $\mathbf{U}$ and $\mathbf{V}$. The crucial observation is which parameters appear together. Every degree-4 monomial $u_{ir} v_{jr} u_{is} v_{js}$ involves only row $i$ of $\mathbf{U}$ and row $j$ of $\mathbf{V}$. Two different users $i \neq i'$ never multiply each other, and neither do two different items. The parameter interactions form a **bipartite graph** between the rows of $\mathbf{U}$ and the rows of $\mathbf{V}$, with edges only across the two sides and none within a side.

This separability is the opening we need. Suppose we **fix $\mathbf{U}$** and minimize over $\mathbf{V}$. Because no two columns of $\mathbf{V}^\top$ interact, the objective splits into independent pieces, one per item factor $\mathbf{v}_j$. Collecting the terms that involve $\mathbf{v}_j$, which are the observed entries of column $j$ plus its share of the regularizer,

$$
\phi_{\mathbf{U}}(\mathbf{v}_j) = \frac{1}{2}\sum_{i=1}^n \omega_{ij}(a_{ij} - \mathbf{u}_i^\top \mathbf{v}_j)^2 + \frac{\lambda}{2}\|\mathbf{v}_j\|^2
$$

This is an ordinary [regularized least squares](/garden/maths/linearAlgebra/projectionsLS) problem in $\mathbf{v}_j \in \mathbb{R}^k$. It is the familiar regression of the observed ratings of item $j$ onto the user factors, treating each user $i$ who rated $j$ as a training example with input $\mathbf{u}_i$ and target $a_{ij}$. Expanding the square and setting the gradient to zero gives the normal equations, whose solution is

$$
\mathbf{v}_j^\star = \Big(\underbrace{\sum_{i=1}^n \omega_{ij}\, \mathbf{u}_i \mathbf{u}_i^\top + \lambda \mathbf{I}_k}_{k \times k}\Big)^{-1} \underbrace{\sum_{i=1}^n \omega_{ij}\, a_{ij}\, \mathbf{u}_i}_{k \times 1}
$$

The matrix we invert is only $k \times k$, with $k$ small, and the sum runs over the users $i$ who actually rated item $j$ since $\omega_{ij}$ zeros out the rest. Here the regularizer earns its keep a second time. If item $j$ has fewer than $k$ ratings the sum $\sum_i \omega_{ij}\mathbf{u}_i\mathbf{u}_i^\top$ is singular, but the added $\lambda \mathbf{I}_k$ makes it invertible. By the symmetric argument, fixing $\mathbf{V}$ and solving for each user factor $\mathbf{u}_i$ has the identical form with the roles of users and items swapped.

We cannot solve for $\mathbf{U}$ and $\mathbf{V}$ at once, since jointly the problem is non-convex, but we have just seen that fixing either one makes the other an exactly solvable least squares problem. This is what suggests alternating.

{{< callout type="info" title="Alternating Least Squares (ALS)" >}}
Alternate between fully optimizing one factor matrix with the other held fixed,

$$
\mathbf{V}_{t+1} \leftarrow \arg\min_{\mathbf{V}} \phi(\mathbf{U}_t, \mathbf{V}), \qquad \mathbf{U}_{t+1} \leftarrow \arg\min_{\mathbf{U}} \phi(\mathbf{U}, \mathbf{V}_{t+1})
$$

Each half-step splits into $m$ or $n$ independent $k$-dimensional least squares problems, one per column or row, all of the form derived above and all solvable in parallel.
{{< /callout >}}

{{< figure
    src="/images/ml/matrixCompletionALS.png"
    alt="Objective versus iteration for ALS and gradient descent on a small completion problem."
    caption="On a small synthetic rank-3 completion problem ALS lowers the objective to near its floor in a handful of iterations, each one an exact least squares solve, while gradient descent on the same objective needs many more steps to catch up."
>}}

ALS differs from gradient descent in a useful way. Gradient descent takes a small step in all parameters at once, whereas ALS optimizes fully over one half of the parameters while freezing the other. Because each half-step is an exact minimization it can only lower the objective, so the objective decreases monotonically and the iteration converges to a fixed point. That fixed point is first-order optimal, satisfying $\nabla \phi(\mathbf{U}^\star, \mathbf{V}^\star) = \mathbf{0}$, but as with the saddle points from the rank-1 analysis it need not be a global minimum. Both ALS and gradient descent can stall at a non-optimal critical point.

A practical advantage of the separable structure is how cheaply the model extends. When a new user $i^\star$ joins with a handful of ratings, we do not refit anything. We hold the item factors $\mathbf{V}$ fixed and solve the single least squares problem for that user's factor,

$$
\mathbf{u}_{i^\star}^\star = \Big(\sum_{j} \omega_{i^\star j}\, \mathbf{v}_j \mathbf{v}_j^\top + \lambda \mathbf{I}_k\Big)^{-1} \sum_{j} \omega_{i^\star j}\, a_{i^\star j}\, \mathbf{v}_j
$$

which costs one $k \times k$ solve rather than a full retraining over all $n$ users and $m$ items. A new item is handled the same way with the roles swapped.

### Projection Methods

The second family keeps an unconstrained matrix $\mathbf{B}$ and repeatedly projects it back toward the low-rank or low-norm matrices, using the SVD as the projection step. Where ALS bakes the rank constraint into the parameterization, projection methods enforce it from the outside.

{{< figure
    src="/images/ml/matrixCompletionProjection.png"
    alt="Schematic of projected gradient descent alternating a gradient step with an SVD projection onto the rank-k set."
    caption="Singular value projection alternates a gradient step on the observed entries, which leaves the non-convex set of rank-$k$ matrices, with an SVD projection back onto it."
>}}

#### Singular Value Projection (SVP)

The first method is projected gradient descent on the rank-$k$ set. The unconstrained loss $\ell(\mathbf{B}) = \tfrac{1}{2}\|\Pi_\Omega(\mathbf{A} - \mathbf{B})\|_F^2$ is convex in $\mathbf{B}$ with gradient $-\Pi_\Omega(\mathbf{A} - \mathbf{B})$, so we take a gradient step and then project onto rank $k$,

$$
\mathbf{A}^0 = \mathbf{0}, \qquad \mathbf{A}^{t+1} = \big[\, \underbrace{\mathbf{A}^t + \eta\, \Pi_\Omega(\mathbf{A} - \mathbf{A}^t)}_{\text{gradient step on observed entries}} \,\big]_k, \qquad \eta > 0
$$

where $[\,\cdot\,]_k$ is the best rank-$k$ approximation computed by SVD via Eckart-Young. The gradient step touches only the observed entries, and the SVD projection is the non-convex step that pulls the iterate back onto the rank-$k$ set. The whole difficulty lives in that projection, since projecting onto a non-convex set is not a benign operation, and each step pays for a full SVD.

#### Nuclear Norm Relaxation

Rather than projecting onto the non-convex rank-$k$ set, we can replace the rank by a convex surrogate. To motivate it, look at the rank through its singular values. The rank is the number of nonzero singular values,

$$
\text{rank}(\mathbf{A}) = \#\{i : \sigma_i > 0\} = \|\boldsymbol{\sigma}(\mathbf{A})\|_0
$$

which is the $\ell_0$ count of the singular-value vector. Minimizing a count is combinatorial, exactly the obstacle that makes the rank constraint hard. The standard fix, borrowed from sparse regression, is to relax the $\ell_0$ count to the $\ell_1$ sum. Because singular values are non-negative, their $\ell_1$ norm is simply their sum, which defines the **nuclear norm**,

$$
\|\mathbf{A}\|_* = \sum_{i=1}^{\text{rank}(\mathbf{A})} \sigma_i = \|\boldsymbol{\sigma}(\mathbf{A})\|_1
$$

Compare the three norms side by side, all built from the singular-value vector. The nuclear norm is its $\ell_1$ norm, the Frobenius norm its $\ell_2$ norm, and the rank its $\ell_0$ count,

$$
\|\mathbf{A}\|_* = \|\boldsymbol{\sigma}(\mathbf{A})\|_1, \qquad \|\mathbf{A}\|_F = \|\boldsymbol{\sigma}(\mathbf{A})\|_2, \qquad \text{rank}(\mathbf{A}) = \|\boldsymbol{\sigma}(\mathbf{A})\|_0
$$

Just as the [Lasso](/garden/ml/linearRegression#lasso-regression) uses the $\ell_1$ norm to drive vector entries to zero and produce sparse solutions, the nuclear norm drives singular values to zero and produces low-rank solutions. The justification for this particular relaxation is that the nuclear norm is the tightest convex function below the rank.

{{< callout type="info" title="The Nuclear Norm is the Convex Envelope of Rank" >}}
The **convex envelope** of a function $f$ is the largest convex function $g$ with $g \le f$, the tightest convex lower bound. A theorem of Fazel et al. (2001) states that on the spectral-norm ball $\{\mathbf{A} : \|\mathbf{A}\|_2 \le 1\}$, the convex envelope of $\text{rank}(\mathbf{A})$ is the nuclear norm $\|\mathbf{A}\|_*$. This is the formal justification for using the nuclear norm as a convex surrogate for the rank.
{{< /callout >}}

{{< figure
    src="/images/ml/matrixCompletionConvexEnvelope.png"
    alt="The rank-like step function and its convex envelope, the absolute value, on a single coordinate."
    caption="On a single coordinate the convex envelope of the rank-like step function (gray), which counts whether a singular value is nonzero, is the absolute value (blue). This is the scalar analogue of the nuclear norm being the convex envelope of the rank."
>}}

With this surrogate we can convert the intractable rank-constrained problem into a convex one. The chain of objectives, following the lecture, is

$$
\begin{aligned}
\min_{\mathbf{B}}\ \tfrac{1}{2}\|\Pi_\Omega(\mathbf{A} - \mathbf{B})\|_F^2 \quad &\text{s.t.}\quad \text{rank}(\mathbf{B}) \le k &&\text{(original, non-convex)} \\
\min_{\mathbf{B}}\ \tfrac{1}{2}\|\Pi_\Omega(\mathbf{A} - \mathbf{B})\|_F^2 + \mu\, \text{rank}(\mathbf{B}) & &&\text{(Lagrangian form)} \\
\min_{\mathbf{B}}\ \tfrac{1}{2}\|\Pi_\Omega(\mathbf{A} - \mathbf{B})\|_F^2 + \mu\, \|\mathbf{B}\|_* & &&\text{(convex surrogate)}
\end{aligned}
$$

The last line is convex and the one we actually solve. To solve it we need one computational primitive, the proximal operator of the nuclear norm, which has a clean closed form.

#### Singular Value Thresholding (SVT)

The engine for nuclear norm problems is the **shrinkage operator**, which answers the question: given a matrix $\mathbf{A}$, what matrix $\mathbf{B}$ stays close to it while having a small nuclear norm? Concretely it solves $\min_{\mathbf{B}} \tfrac{1}{2}\|\mathbf{A} - \mathbf{B}\|_F^2 + \tau\|\mathbf{B}\|_*$, trading fidelity to $\mathbf{A}$ against a small nuclear norm. The minimizer keeps the singular vectors of $\mathbf{A}$ and soft-thresholds its singular values, shrinking each toward zero by $\tau$ and clipping at zero.

{{< callout type="info" title="Singular Value Thresholding (SVT)" >}}
Let $\mathbf{A}$ have SVD $\mathbf{A} = \mathbf{U}\,\text{diag}(\sigma_i)\,\mathbf{V}^\top$. The shrinkage operator (Cai et al., 2010) is

$$
\text{shrink}_\tau(\mathbf{A}) := \mathbf{U}\,\text{diag}\big((\sigma_i - \tau)_+\big)\,\mathbf{V}^\top = \arg\min_{\mathbf{B}}\Big(\frac{1}{2}\|\mathbf{A} - \mathbf{B}\|_F^2 + \tau\|\mathbf{B}\|_*\Big)
$$

where $(x)_+ = \max\{x, 0\}$. It shrinks every singular value by $\tau$ and clips negatives to zero, so singular values below $\tau$ are zeroed out and the rank drops as $\tau$ grows.
{{< /callout >}}

This is the exact matrix analogue of the soft-thresholding that solves the Lasso, with singular values playing the role of coefficients. Shrinking removes the small singular values, which is what lowers the rank, while only mildly perturbing the large ones that carry the signal. The **SVT algorithm** runs this operator inside an iteration that keeps the residual sparse,

$$
\mathbf{A}^0 = \mathbf{0}, \qquad \mathbf{A}^{t+1} = \mathbf{A}^t + \eta_t\, \Pi_\Omega\big(\mathbf{A} - \text{shrink}_\tau(\mathbf{A}^t)\big)
$$

With a suitable step-size schedule the shrunken iterate converges to the solution of the convex problem, which agrees with $\mathbf{A}$ on the observed entries and otherwise has minimal nuclear norm. Because the update is confined to the observed entries it stays sparse, which keeps each SVD cheap.

The two projection methods make complementary trade-offs. SVT maintains a **sparse** iterate (only the observed entries are touched) while letting the rank float, whereas SVP maintains a **rank-$k$** iterate while letting it be dense. Both beat a naive dense full-rank representation.

#### Randomized SVD

Every projection step needs an SVD, which is the bottleneck for large matrices, so it is worth knowing it can be made scalable.

{{< callout type="info" title="Randomized SVD" >}}
Randomized methods (Halko et al., 2011) approximate the SVD by first compressing the matrix. Find a semi-orthogonal $\mathbf{Q} \in \mathbb{R}^{n \times 2k}$ with few columns such that $\mathbf{A} \approx \mathbf{Q}\mathbf{Q}^\top \mathbf{A}$, then take the SVD of the much smaller $\mathbf{B} = \mathbf{Q}^\top \mathbf{A} = \tilde{\mathbf{U}}\tilde{\mathbf{\Sigma}}\tilde{\mathbf{V}}^\top$ and lift it back, $\mathbf{A} \approx (\mathbf{Q}\tilde{\mathbf{U}})\tilde{\mathbf{\Sigma}}\tilde{\mathbf{V}}^\top$. A good $\mathbf{Q}$ comes from a random projection. Draw a Gaussian $\mathbf{R} \in \mathbb{R}^{m \times 2k}$, form $\mathbf{Y} = (\mathbf{A}\mathbf{A}^\top)^q \mathbf{A}\mathbf{R}$ for a small power such as $q = 2$, and orthonormalize its columns with Gram-Schmidt. The power $q$ sharpens the dominant subspace, and small errors are tolerable since the result feeds back into an iterative algorithm.
{{< /callout >}}

## Exact Reconstruction

We close with a more ambitious question. If the true matrix really is rank $k$, can we recover it exactly, not just approximately, from a few observed entries? The answer is yes under conditions, and understanding those conditions explains how few entries can possibly suffice.

### How Many Entries are Needed?

The first bound is a counting argument. A rank-$k$ matrix has far fewer free parameters than its $nm$ entries, and we cannot hope to pin it down with fewer observations than it has parameters. Count the **degrees of freedom** from the SVD, taking $\mathbf{A} \in \mathbb{R}^{n \times n}$ for simplicity. There are $k$ singular values. For the singular vectors, the first left singular vector has $n - 1$ free parameters after the unit-length constraint, the second has $n - 2$ after unit length and orthogonality to the first, and so on,

$$
(n-1) + (n-2) + \cdots + (n-k) = nk - \frac{k(k+1)}{2}
$$

free parameters on each side. Adding the $k$ singular values and both sets of singular vectors, the number of independent parameters is

$$
S \ge k + 2\Big(nk - \frac{k(k+1)}{2}\Big) = 2nk - k^2
$$

So a rank-$k$ matrix is information-theoretically unrecoverable from fewer than $2nk - k^2$ entries. The useful thing about this number is its size. For $k \ll n$ it is roughly $2nk$, which is tiny next to the $n^2$ entries of the full matrix. With a million users and items and $k = 20$ factors, $2nk \approx 4 \times 10^7$ against $n^2 = 10^{12}$, a fraction of order $10^{-5}$. This is why observing one percent, or even far less, can be enough in principle. It also shows the bound is only necessary, not sufficient, since it counts parameters but says nothing about whether random samples actually expose them.

### Coverage and the Coupon Collector

The first reason $2nk - k^2$ samples are not enough is coverage. If we sample entries at random, we need many more than $n$ of them before every row and column even contains a single observation, and a row we never observe cannot be reconstructed.

{{< callout type="proof" title="Coupon Collector" >}}
Suppose each sample reveals one of $n$ coupons uniformly at random, and let $t_i$ be the number of samples needed to obtain a new coupon once $i - 1$ distinct coupons are held. Each draw is new with probability $(n - i + 1)/n$, so $t_i$ is [geometric](/garden/maths/probabilityStatistics/randomVariables#geometric-distribution) with expectation

$$
\mathbb{E}[t_i] = \frac{n}{n - i + 1}
$$

The expected number of samples to collect all $n$ coupons is then the harmonic sum

$$
\mathbb{E}[T] = \sum_{i=1}^n \mathbb{E}[t_i] = n\Big(1 + \frac{1}{2} + \cdots + \frac{1}{n}\Big) = n H_n
$$

where $H_n = \sum_{i=1}^n \tfrac{1}{i}$ is the $n$-th **harmonic number**, which grows like $H_n = \log n + O(1)$. So we need on the order of $n \log n$ random samples just to touch every row, a logarithmic factor above the naive count of $n$.
{{< /callout >}}

### Incoherence

The second reason is more subtle. Recovery is impossible if the matrix hides its information in a single entry. Suppose the leading singular vectors are standard basis vectors, $\mathbf{u}_1 = \mathbf{e}_i$ and $\mathbf{v}_1 = \mathbf{e}_j$. Then the entire contribution of $\sigma_1$ is concentrated in the single cell $(i, j)$, and unless we happen to sample that exact entry we can never recover $\sigma_1$. Low-rank recovery from sparse samples is therefore impossible without a regularity condition that forces the information to be spread out rather than concentrated.

{{< callout type="info" title="Incoherence" >}}
The regularity condition is **incoherence** (Candes and Tao, 2010). Define the projections onto the column space and row space of a rank-$k$ matrix with SVD $\mathbf{A} = \mathbf{U}\mathbf{\Sigma}\mathbf{V}^\top$,

$$
\mathbf{P} = \sum_{i=1}^k \mathbf{u}_i \mathbf{u}_i^\top, \qquad \mathbf{Q} = \sum_{i=1}^k \mathbf{v}_i \mathbf{v}_i^\top, \qquad \mathbf{E} = \sum_{i=1}^k \mathbf{u}_i \mathbf{v}_i^\top
$$

The matrix is incoherent with parameter $\mu$ if the entries of these are all small and uniform, of order $\mu\sqrt{k/n}$,

$$
|p_{ij}|, |q_{ij}| \le \mu\sqrt{\tfrac{k}{n}}\ (i \neq j), \qquad \Big|p_{ii} - \tfrac{k}{n}\Big|, \Big|q_{ii} - \tfrac{k}{n}\Big| \le \mu\sqrt{\tfrac{k}{n}}, \qquad |e_{ij}| \le \mu\sqrt{\tfrac{k}{n}}
$$

These technical conditions, from the compressed sensing literature, say the singular vectors are not aligned with any coordinate axis, so no single entry is special.
{{< /callout >}}

### The Reconstruction Theorem

With coverage handled by sampling enough entries and concentration ruled out by incoherence, the payoff is a clean recovery guarantee.

{{< callout type="info" title="Reconstruction Theorem (Candes and Tao, 2010)" >}}
Let $\mathbf{A} \in \mathbb{R}^{n \times n}$ be a rank-$k$ matrix that is incoherent with $\mu \ge 1$, and suppose $S$ entries are observed at random. There is a universal constant $C$ such that if

$$
S \ge C \mu^2 n k (\log n)^6
$$

then with probability at least $1 - n^{-3}$, the matrix $\mathbf{A}$ is the unique solution of the convex nuclear norm minimization

$$
\mathbf{A} = \arg\min_{\mathbf{B}}\ \|\mathbf{B}\|_* \quad \text{subject to}\quad \Pi_\Omega(\mathbf{B}) = \Pi_\Omega(\mathbf{A})
$$
{{< /callout >}}

This ties the whole note together. The sample count $C\mu^2 nk(\log n)^6$ is the degrees of freedom $nk$ from the counting argument, inflated only by the incoherence parameter and logarithmic factors, one of which is the $\log n$ coverage factor from the coupon collector. The recovery is achieved not by any bespoke combinatorial search but by the same convex nuclear norm minimization that the SVT algorithm solves. Under the right conditions, exact matrix completion is both possible and tractable.

## Non-Negative Matrix Factorization

So far the low-rank model has allowed the factors to take any sign. For many data types this is unnatural. Word counts, pixel intensities, and spectral measurements are all **non-negative** by their nature, and a factorization that uses negative values has no physical meaning for such data. **Non-negative matrix factorization (NMF)** builds this prior knowledge directly into the model by constraining both factors to be non-negative. It is a standalone factorization in its own right, and it is also the linear-algebra face of the [topic models](/garden/ml/topicModels#the-matrix-factorization-view) built on pLSA.

The setup is the fully observed low-rank model with one extra constraint. Given a non-negative data matrix $\mathbf{A} \in \mathbb{R}_{\ge 0}^{n \times m}$, we seek non-negative factors $\mathbf{U} \in \mathbb{R}_{\ge 0}^{n \times k}$ and $\mathbf{V} \in \mathbb{R}_{\ge 0}^{m \times k}$ whose product reconstructs the data,

$$
\phi(\mathbf{U}, \mathbf{V}) = \frac{1}{2}\big\| \mathbf{A} - \mathbf{U}\mathbf{V}^\top \big\|_F^2, \qquad \mathbf{U}, \mathbf{V} \ge 0
$$

where the inequality is entrywise. The squared Frobenius loss is the default, but it is not the only choice. When the data is counts, a **generalized KL divergence** between $\mathbf{A}$ and $\mathbf{U}\mathbf{V}^\top$ is the more natural loss, and it is exactly the objective that the [pLSA topic model](/garden/ml/topicModels#the-matrix-factorization-view) maximizes, since maximizing a count log-likelihood is the same as minimizing a KL divergence to the predicted frequencies. With the KL loss and added normalization constraints, NMF and pLSA are the same model.

### Why Non-Negativity Changes Everything

The constraint looks minor but it fundamentally changes the kind of decomposition we get. Without sign constraints, as in the [SVD](#the-singular-value-decomposition) or [PCA](/garden/maths/linearAlgebra/pca), the factors are free to cancel: a reconstruction can build a feature by adding a large positive component and subtracting another, so the basis vectors tend to be **holistic**, each one a dense pattern spread across the whole object with both positive and negative entries. The classic illustration is faces. PCA on a set of face images produces eigenfaces, ghostly whole-face templates that are combined with positive and negative coefficients to cancel each other into a specific face.

Non-negativity forbids cancellation. Since neither the factors nor their coefficients can be negative, the only way to build a reconstruction is to **add** pieces together, and the "ink" laid down by one factor can never be erased by another. The factors are therefore pushed to be **parts**: localized, sparse, additive components. On the same face data, NMF discovers basis images that look like individual facial features such as a nose, an eyebrow, or a mouth, and a face is assembled by switching on the parts it contains. The result is a part-based, additive, and often more interpretable decomposition than the holistic one PCA gives.

### Alternating Least Squares for NMF

The objective is, like the ALS objective from before, non-convex in $(\mathbf{U}, \mathbf{V})$ jointly but well-behaved in each factor separately. Fixing one factor leaves a least squares problem in the other, now with a non-negativity constraint, which suggests the same alternating strategy as [ALS for completion](#alternating-least-squares).

{{< callout type="info" title="Projected Alternating Least Squares" >}}
Alternate between solving for one factor with the other fixed, then projecting back onto the non-negative orthant,

$$
\mathbf{U} \leftarrow \arg\min_{\mathbf{U}} \tfrac{1}{2}\|\mathbf{A} - \mathbf{U}\mathbf{V}^\top\|_F^2, \qquad \mathbf{V} \leftarrow \arg\min_{\mathbf{V}} \tfrac{1}{2}\|\mathbf{A} - \mathbf{U}\mathbf{V}^\top\|_F^2
$$

after each solve, clip any negative entries back to zero,

$$
u_{iz} \leftarrow \max\{0, u_{iz}\}, \qquad v_{jz} \leftarrow \max\{0, v_{jz}\}
$$

so that all iterates stay non-negative. A good starting point such as NNDSVD (a non-negative initialization built from the SVD) speeds up convergence.
{{< /callout >}}

The unconstrained solve in each half-step is the same separable $k \times k$ least squares we derived for completion, since the fully observed case is just the special case where every weight $\omega_{ij} = 1$. The only addition is the clipping projection that enforces non-negativity. This projected version is simple and fast but approximate, because clipping after an unconstrained solve is not the same as solving the constrained problem exactly. A popular alternative is the family of **multiplicative update** rules, which rescale the factors by non-negative ratios and so preserve non-negativity automatically while monotonically decreasing the loss. Either way, as with all the methods in this note, we converge to a first-order critical point rather than a guaranteed global optimum.

NMF therefore slots neatly beside the rest of the low-rank toolkit. It is the same factored model $\mathbf{A} \approx \mathbf{U}\mathbf{V}^\top$ and the same alternating optimization, with non-negativity as an inductive bias that trades the orthogonal, cancellation-friendly basis of the SVD for a part-based, additive one. When the loss is the KL divergence and the factors are normalized into probabilities, it is precisely a [topic model](/garden/ml/topicModels).
