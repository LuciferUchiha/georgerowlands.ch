---
title: Autoencoders
type: docs
weight: 2
---

Deep learning excels at learning meaningful representations of data. Neural networks can be viewed as compositions of encoders that extract features and estimators that make predictions from those features. **Autoencoders** provide a framework for learning these representations in an unsupervised manner, without requiring labeled data. The core idea is deceptively simple: train a network to reconstruct its input through a bottleneck that forces it to learn a compressed representation.

More formally, an autoencoder is a neural network architecture consisting of two main components. The **encoder** $g_\phi: \mathbb{R}^D \to \mathbb{R}^d$ maps high-dimensional input data $\mathbf{x} \in \mathbb{R}^D$ to a lower-dimensional latent representation $\mathbf{z} \in \mathbb{R}^d$ where typically $d \ll D$. The **decoder** $f_\theta: \mathbb{R}^d \to \mathbb{R}^D$ attempts to reconstruct the original input from this compressed representation.

$$
\mathbf{z} = g_\phi(\mathbf{x}), \quad \hat{\mathbf{x}} = f_\theta(\mathbf{z}) = f_\theta(g_\phi(\mathbf{x}))
$$

{{< figure
    src="/images/ml/autoencoder.png"
    alt="Traditional autoencoder architecture with encoder, latent representation, and decoder."
    caption="An autoencoder compresses input through an encoder to a latent representation, then reconstructs it through a decoder."
>}}

The network is trained to minimize the reconstruction error between the input $\mathbf{x}$ and its reconstruction $\hat{\mathbf{x}}$. This is typically the mean squared error:

$$
\mathcal{L}(\theta, \phi) = \frac{1}{n}\sum_{i=1}^n \|\mathbf{x}_i - f_\theta(g_\phi(\mathbf{x}_i))\|^2
$$

The bottleneck structure (where $d \ll D$) forces the encoder to learn a compressed representation that captures the most important features of the data. This is similar in spirit to [principal component analysis (PCA)](/garden/maths/linearAlgebra/pca) which reduces dimensionality using linear projections, but in addition autoencoders can learn non-linear representations through the use of non-linear activation functions.

## Linear Autoencoders

The general autoencoder above is something of a black box. We know it compresses data through a bottleneck, but with arbitrary non-linear maps $g_\phi$ and $f_\theta$ it is hard to say precisely what representation it learns. Instead to improve understanding we study the simplest possible case, where the encoder and decoder are linear maps. Linear autoencoders can be analysed completely, and the answer is both elegant and familiar: spoilers, the optimal linear autoencoder performs an orthogonal projection onto the subspace of maximal variance, which is exactly what [principal component analysis](/garden/maths/linearAlgebra/pca) does.

Throughout this section we keep the notation of the general autoencoder. A data point $\mathbf{x} \in \mathbb{R}^D$ lives in a high-dimensional ambient space, its latent code $\mathbf{z} \in \mathbb{R}^d$ has $d \leq D$, and we have a dataset of $n$ points $\mathbf{x}_1, \ldots, \mathbf{x}_n$.

### From Loss to Risk

To make "good reconstruction" precise we need to measure the discrepancy between an input $\mathbf{x}$ and its reconstruction $\hat{\mathbf{x}}$. This is the job of a **loss** (or **distortion measure**), a function

$$
\mathcal{L}: \mathbb{R}^D \times \mathbb{R}^D \to \mathbb{R}, \quad (\mathbf{x}, \hat{\mathbf{x}}) \mapsto \mathcal{L}(\mathbf{x}, \hat{\mathbf{x}})
$$

that scores a single reconstruction. As mentioned before in the absence of domain-specific knowledge the default choice is the **quadratic loss**

$$
\mathcal{L}(\mathbf{x}, \hat{\mathbf{x}}) = \frac{1}{2}\|\mathbf{x} - \hat{\mathbf{x}}\|^2
$$

It is important to keep in mind that this is a convenient choice rather than the only one. For other data types other distortion measures are more natural. For example, for binary data, one would probably use the log loss (cross entropy) instead, just as in [logistic regression](/garden/ml/linearClassificationlogisticRegression).

The loss scores a single example. What we actually care about is the average loss, called the **risk**. The natural question is: average over which distribution? We assume the data is drawn from some fixed but unknown distribution $p$, written $\mathbf{x} \sim p$, and our sample $\mathbf{x}_1, \ldots, \mathbf{x}_n$ is drawn independently from it.

In practice we only have the sample, so the quantity we can actually minimize is the **empirical risk**, the average loss over the data we have:

$$
\mathcal{R}_{\text{emp}} = \frac{1}{n}\sum_{i=1}^n \mathcal{L}\big(\mathbf{x}_i, (G \circ F)(\mathbf{x}_i)\big)
$$

What we would really like to minimize is the average loss over the entire distribution rather than just our finite sample. This is the **expected risk**, also called the population or true risk:

$$
\mathcal{R} = \mathbb{E}_{\mathbf{x} \sim p}\big[\mathcal{L}(\mathbf{x}, (G \circ F)(\mathbf{x}))\big]
$$

The empirical risk is a [Monte Carlo](/garden/maths/probabilityStatistics/simulating) estimate of the expected risk, the sample average standing in for the expectation we cannot evaluate. Here $G \circ F$ is the **reconstruction map**, the composition of encoder and decoder that maps a data point to its reconstruction. Ideally we would want $G \circ F = \text{id}$, the identity function, but this is unachievable through a bottleneck of dimension $d < D$ unless the data happens to lie exactly on a $d$-dimensional subspace.

### Linear Encoder and Decoder

We now specialize the encoder $g_\phi$ and decoder $f_\theta$ to linear maps. The **linear encoder** and **linear decoder** are

$$
F: \mathbb{R}^D \to \mathbb{R}^d \quad \mathbf{z} = \mathbf{W}\mathbf{x} \qquad G: \mathbb{R}^d \to \mathbb{R}^D \quad \hat{\mathbf{x}} = \mathbf{V}\mathbf{z}
$$

with weight matrices $\mathbf{W} \in \mathbb{R}^{d \times D}$ and $\mathbf{V} \in \mathbb{R}^{D \times d}$. The reconstruction map is their composition, which by associativity of matrix multiplication is again a single matrix $\mathbf{P}$ the **reconstruction matrix**:

$$
(G \circ F)(\mathbf{x}) = \mathbf{V}(\mathbf{W}\mathbf{x}) = (\mathbf{V}\mathbf{W})\mathbf{x} = \mathbf{P}\mathbf{x} \qquad \mathbf{P} = \mathbf{V}\mathbf{W} \in \mathbb{R}^{D \times D}
$$

With the quadratic loss, the linear autoencoder objective is the expected risk, here written as an expectation over the data distribution:

$$
\mathcal{R}(\mathbf{W}, \mathbf{V}) = \mathcal{R}(\mathbf{P}) = \mathbb{E}\left[\frac{1}{2}\|\mathbf{x} - \mathbf{P}\mathbf{x}\|^2\right]
$$

Working with the single matrix $\mathbf{P} = \mathbf{V}\mathbf{W}$ is convenient, and it also explains why there is no point in stacking several linear layers. A two-layer linear encoder is itself just a matrix $\mathbf{W}_2 \mathbf{W}_1$, and a two-layer linear decoder just a matrix $\mathbf{V}_1 \mathbf{V}_2$, so by associativity of matrix multiplication the whole reconstruction map collapses back to a single matrix,

$$
(\mathbf{V}_1 \mathbf{V}_2)(\mathbf{W}_2 \mathbf{W}_1) = \underbrace{\mathbf{V}_1 \mathbf{V}_2 \mathbf{W}_2 \mathbf{W}_1}_{\mathbf{P}}
$$

Depth therefore buys no extra expressivity in the linear case, and nothing is lost by analysing the single reconstruction matrix $\mathbf{P}$.

### Affine vs Linear Maps

Recall that our linear encoder and decoder have the form $F(\mathbf{x}) = \mathbf{W}\mathbf{x}$ and $G(\mathbf{z}) = \mathbf{V}\mathbf{z}$, with no constant term. A natural question is whether we are giving up power by using these purely linear maps rather than **affine** ones of the form $F(\mathbf{x}) = \mathbf{W}\mathbf{x} + \mathbf{a}$ and $G(\mathbf{z}) = \mathbf{V}\mathbf{z} + \mathbf{b}$ that also carry bias terms $\mathbf{a}$ and $\mathbf{b}$. It turns out that for centered data and the quadratic loss affine maps reduce to linear maps, so we are not losing anything by restricting to linear maps. We just need to center the data first, which can be done as a preprocessing step before training the autoencoder.

**Centering** the data simply means that we subtract the sample mean from each data point, so replacing each $\mathbf{x}$ by $\mathbf{x} - \mathbb{E}[\mathbf{x}]$ so that $\mathbb{E}[\mathbf{x}] = 0$, i.e. the data is centered around the origin.

{{< callout type="proof" title="Affine Reconstruction Maps Reduce to Linear" >}}
Let $F(\mathbf{x}) = \mathbf{W}\mathbf{x} + \mathbf{a}$ and $G(\mathbf{z}) = \mathbf{V}\mathbf{z} + \mathbf{b}$ be affine. Their composition is again affine,

$$
(G \circ F)(\mathbf{x}) = \mathbf{V}(\mathbf{W}\mathbf{x} + \mathbf{a}) + \mathbf{b} = \mathbf{V}\mathbf{W}\mathbf{x} + \mathbf{V}\mathbf{a} + \mathbf{b} = \underbrace{\mathbf{V}\mathbf{W}}_{\mathbf{P}}\mathbf{x} + \mathbf{c}, \qquad \mathbf{c} = \mathbf{b} + \mathbf{V}\mathbf{a}
$$

so it is a linear map $\mathbf{P}$ plus a constant offset $\mathbf{c}$. We compare the risk of the affine map against the purely linear map ($\mathbf{c} = 0$). Expanding the squared norm,

$$
\mathbb{E}\|\mathbf{x} - (\mathbf{P}\mathbf{x} + \mathbf{c})\|^2 = \mathbb{E}\|\mathbf{x} - \mathbf{P}\mathbf{x}\|^2 + \|\mathbf{c}\|^2 - 2\big\langle \mathbf{c},\ \mathbb{E}\mathbf{x} - \mathbf{P}\,\mathbb{E}\mathbf{x} \big\rangle
$$

For centered data $\mathbb{E}\mathbf{x} = 0$, so the cross term vanishes and we are left with

$$
\mathbb{E}\|\mathbf{x} - (\mathbf{P}\mathbf{x} + \mathbf{c})\|^2 = \mathbb{E}\|\mathbf{x} - \mathbf{P}\mathbf{x}\|^2 + \|\mathbf{c}\|^2 \ \geq\ \mathbb{E}\|\mathbf{x} - \mathbf{P}\mathbf{x}\|^2
$$

with equality exactly when $\mathbf{c} = 0$. Hence the optimal offset is zero and the affine map is no better than the linear map $\mathbf{P}$.
{{< /callout >}}

In other words, once we center the data, the bias terms are redundant. This is why we can absorb the affine question away and study the linear reconstruction matrix $\mathbf{P}$ on its own.

It is worth being careful about what exactly this says. The centering argument above is the real result: for centered data and the squared loss, the best affine reconstruction has zero offset, so it gives no advantage over a linear one. This is a statement about the optimum, not a claim that affine maps are the same thing as linear maps. An affine map in $D$ dimensions is in fact strictly more general than a linear map in $D$ dimensions, since it has the extra bias.

A related but different observation is that any affine map can be rewritten as a linear map by a trick called **homogeneous coordinates**. We augment each input with a constant extra coordinate equal to $1$, and append the bias as an extra column of the weight matrix:

$$
\tilde{\mathbf{x}} = \begin{bmatrix} \mathbf{x} \\ 1 \end{bmatrix} \qquad \tilde{\mathbf{W}} = \begin{bmatrix} \mathbf{W} & \mathbf{a} \end{bmatrix} \qquad \tilde{\mathbf{W}}\tilde{\mathbf{x}} = \mathbf{W}\mathbf{x} + \mathbf{a}
$$

So the affine map $\mathbf{W}\mathbf{x} + \mathbf{a}$ in $D$ dimensions becomes a linear map $\tilde{\mathbf{W}}\tilde{\mathbf{x}}$ in $D + 1$ dimensions. This is a transformation, not an equivalence of power. The extra dimension is exactly what carries the bias, so a linear map in $D + 1$ dimensions can do strictly more than a linear map in $D$ dimensions. The augmentation just relabels an affine map as a higher-dimensional linear one, which is the same trick used to fold biases into weight matrices in neural networks. For centered data we need none of this, since the optimal offset is already zero. For the rest of this section we assume the data is centered.

### Non-Identifiability of the Factorization

Our goal is to minimize $\mathcal{R}(\mathbf{P}) = \frac{1}{2}\mathbb{E}\|\mathbf{x} - \mathbf{P}\mathbf{x}\|^2$ over reconstruction matrices $\mathbf{P} = \mathbf{V}\mathbf{W}$. Two distinct questions about uniqueness arise:

1. Is the optimal reconstruction matrix $\mathbf{P}$ unique?
2. Is the factorization of $\mathbf{P}$ into weight matrices $\mathbf{W}$ and $\mathbf{V}$ unique?

The answer to the first question is yes, as we will see once we identify the optimal $\mathbf{P}$ with an orthogonal projection. The answer to the second question is no. For any invertible matrix $\mathbf{A} \in \mathbb{R}^{d \times d}$ we can insert $\mathbf{A} \mathbf{A}^{-1} = \mathbf{I}$ in the middle of the product,

$$
\mathbf{V}\mathbf{W} = \mathbf{V}(\mathbf{A} \mathbf{A}^{-1})\mathbf{W} = \underbrace{(\mathbf{V}\mathbf{A})}_{= \mathbf{V}'}\underbrace{(\mathbf{A}^{-1}\mathbf{W})}_{= \mathbf{W}'}
$$

and the new weight matrices $\mathbf{V}'$ and $\mathbf{W}'$ give exactly the same reconstruction matrix $\mathbf{P} = \mathbf{V}'\mathbf{W}' = \mathbf{V}\mathbf{W}$. So the same projection can be realized by infinitely many different pairs of weight matrices.

The practical consequence is that the weight matrices are **non-identifiable**, and one should not over-interpret the individual columns of $\mathbf{V}$ or rows of $\mathbf{W}$ that a trained autoencoder produces. They are just one arbitrary representative out of a whole family. What is well defined is the reconstruction matrix $\mathbf{P}$ and the subspace it projects onto.

### The Rank Constraint

The bottleneck of width $d$ shows up algebraically as a **rank constraint** on $\mathbf{P}$. Recall that the rank of a matrix is the dimension of its image, $\text{rank}(\mathbf{A}) = \dim(\text{im}(\mathbf{A}))$, and that row rank equals column rank, so $\text{rank}(\mathbf{A}) = \text{rank}(\mathbf{A}^T)$ (see [matrix ranks](/garden/maths/linearAlgebra/matrixRanks) and the [column space](/garden/maths/linearAlgebra/vectorSpaces#column-space)). Since $\mathbf{P} = \mathbf{V}\mathbf{W}$ with $\mathbf{W} \in \mathbb{R}^{d \times D}$ and $\mathbf{V} \in \mathbb{R}^{D \times d}$, each weight matrix has rank at most $d$ (its smaller dimension), and the rank of the product is limited by this inner dimension:

$$
\text{rank}(\mathbf{P}) = \min\{\text{rank}(\mathbf{V}), \text{rank}(\mathbf{W})\} \le \min\{D, d\} = d
$$

So the maximal rank of $\mathbf{V}$, $\mathbf{W}$, and $\mathbf{P}$ is $d$, and the bottleneck layer is exactly a rank constraint $\text{rank}(\mathbf{P}) \le d$. Note this constraint does not depend on the factored form $\mathbf{P} = \mathbf{V}\mathbf{W}$ at all. The factorization is just one way of encoding the rank constraint into a parameterization. We can therefore study the cleaner problem

$$
\min_{\mathbf{P}:\ \text{rank}(\mathbf{P}) \le d} \mathcal{R}(\mathbf{P}), \qquad \mathcal{R}(\mathbf{P}) = \frac{1}{2}\mathbb{E}\|\mathbf{x} - \mathbf{P}\mathbf{x}\|^2
$$

Because $\mathbf{P}$ is linear and has rank at most $d$, its image $U = \text{im}(\mathbf{P})$ is a linear subspace of $\mathbb{R}^D$ of dimension at most $d$. Every reconstruction $\hat{\mathbf{x}} = \mathbf{P}\mathbf{x}$ lands in $U$. This suggests splitting the problem into two parts:

1. Given a fixed subspace $U$, find the optimal map $\mathbf{P}$ whose image is $U$.
2. Find the optimal subspace $U$ to use.

We solve the first part now, and the second part leads us to PCA.

### The Optimal Reconstruction Map is an Orthogonal Projection

Fix a subspace $U \subseteq \mathbb{R}^D$ with $\dim(U) \le d$. Among all maps with image $U$, which one minimizes the reconstruction loss? Intuitively, the best reconstruction of $\mathbf{x}$ that is forced to live in $U$ is the point of $U$ closest to $\mathbf{x}$. That point is the **orthogonal projection** of $\mathbf{x}$ onto $U$,

$$
\Pi_U(\mathbf{x}) = \arg\min_{\mathbf{x}' \in U} \|\mathbf{x} - \mathbf{x}'\|
$$

This closest point exists and is unique for every $\mathbf{x}$ and every subspace $U$. To connect this with the algebra of $\mathbf{P}$, recall the definition of a projection (see [projections and least squares](/garden/maths/linearAlgebra/projectionsLS)).

{{< callout type="info" title="Projections, Orthogonal and Oblique" >}}
A linear map $\mathbf{P}: \mathbb{R}^D \to \mathbb{R}^D$ is a **projection** onto $U$ if for every $\mathbf{x}$:

1. **Projection property:** $\mathbf{P}\mathbf{x} \in U$, the output lies in $U$.
2. **Idempotency:** $\mathbf{P}(\mathbf{P}\mathbf{x}) = \mathbf{P}\mathbf{x}$, projecting again changes nothing.

A projection is **orthogonal** if in addition its kernel is orthogonal to its image, $\ker(\mathbf{P}) \perp \text{im}(\mathbf{P})$. Equivalently, $\mathbf{P}$ is **self-adjoint** (symmetric):

$$
\langle \mathbf{P}\mathbf{x}, \mathbf{y} \rangle = \langle \mathbf{x}, \mathbf{P}\mathbf{y} \rangle \quad \text{for all } \mathbf{x}, \mathbf{y} \iff \mathbf{P} = \mathbf{P}^T
$$

Self-adjointness means $\mathbf{P}$ behaves the same whether it acts on the left or the right argument of an inner product. Geometrically it says the error $\mathbf{x} - \mathbf{P}\mathbf{x}$ is perpendicular to the subspace $U$, which is precisely what "closest point" requires. A projection that is not orthogonal is called **oblique**, it still lands in $U$ and is idempotent, but it drops $\mathbf{x}$ onto $U$ along a slanted direction rather than perpendicularly.
{{< /callout >}}

The two ways of stating orthogonality, "kernel perpendicular to image" and "self-adjoint", really are the same condition.

{{< callout type="proof" title="Orthogonality Equals Self-Adjointness" >}}
Let $\mathbf{P}$ be a projection, so $\mathbf{P}^2 = \mathbf{P}$. Every vector splits as $\mathbf{x} = \mathbf{P}\mathbf{x} + (\mathbf{x} - \mathbf{P}\mathbf{x})$, where $\mathbf{P}\mathbf{x} \in \text{im}(\mathbf{P})$ and $\mathbf{x} - \mathbf{P}\mathbf{x} \in \ker(\mathbf{P})$ since $\mathbf{P}(\mathbf{x} - \mathbf{P}\mathbf{x}) = \mathbf{P}\mathbf{x} - \mathbf{P}^2 \mathbf{x} = 0$.

**Self-adjoint $\Rightarrow$ kernel orthogonal to image.** Take $\mathbf{u} = \mathbf{P}\mathbf{a} \in \text{im}(\mathbf{P})$ and $\mathbf{w} \in \ker(\mathbf{P})$, so $\mathbf{P}\mathbf{w} = 0$. Then $\langle \mathbf{u}, \mathbf{w}\rangle = \langle \mathbf{P}\mathbf{a}, \mathbf{w}\rangle = \langle \mathbf{a}, \mathbf{P}\mathbf{w}\rangle = \langle \mathbf{a}, 0\rangle = 0$.

**Kernel orthogonal to image $\Rightarrow$ self-adjoint.** For any $\mathbf{x}, \mathbf{y}$, split $\mathbf{y} = \mathbf{P}\mathbf{y} + (\mathbf{y} - \mathbf{P}\mathbf{y})$. Since $\mathbf{P}\mathbf{x} \in \text{im}(\mathbf{P})$ and $\mathbf{y} - \mathbf{P}\mathbf{y} \in \ker(\mathbf{P})$ are orthogonal,

$$
\langle \mathbf{P}\mathbf{x}, \mathbf{y}\rangle = \langle \mathbf{P}\mathbf{x}, \mathbf{P}\mathbf{y}\rangle + \underbrace{\langle \mathbf{P}\mathbf{x}, \mathbf{y} - \mathbf{P}\mathbf{y}\rangle}_{= 0} = \langle \mathbf{P}\mathbf{x}, \mathbf{P}\mathbf{y}\rangle
$$

By the same argument $\langle \mathbf{x}, \mathbf{P}\mathbf{y}\rangle = \langle \mathbf{P}\mathbf{x}, \mathbf{P}\mathbf{y}\rangle$, so $\langle \mathbf{P}\mathbf{x}, \mathbf{y}\rangle = \langle \mathbf{x}, \mathbf{P}\mathbf{y}\rangle$.
{{< /callout >}}

We now verify that the closest-point map $\Pi_U$ really is an orthogonal projection. It is enough to show it is idempotent, orthogonal, and linear.

{{< callout type="proof" title="Closest-Point Map is an Orthogonal Projection" >}}
**Idempotency.** If $\mathbf{u} \in U$ then the closest point of $U$ to $\mathbf{u}$ is $\mathbf{u}$ itself, so $\Pi_U(\mathbf{u}) = \mathbf{u}$. Applying this to $\mathbf{u} = \Pi_U(\mathbf{x}) \in U$ gives $\Pi_U(\Pi_U(\mathbf{x})) = \Pi_U(\mathbf{x})$.

**Orthogonality.** We claim the error $\Pi_U(\mathbf{x}) - \mathbf{x}$ is orthogonal to $U$. Decompose it as $\Pi_U(\mathbf{x}) - \mathbf{x} = \mathbf{u} + \mathbf{u}^\perp$ with $\mathbf{u} \in U$ and $\mathbf{u}^\perp \in U^\perp$. Suppose $\mathbf{u} \neq 0$. Then since the cross term $\langle \mathbf{u}, \mathbf{u}^\perp\rangle = 0$,

$$
\|\Pi_U(\mathbf{x}) - \mathbf{x}\|^2 = \|\mathbf{u}\|^2 + \|\mathbf{u}^\perp\|^2 > \|\mathbf{u}^\perp\|^2 = \|\big(\Pi_U(\mathbf{x}) - \mathbf{u}\big) - \mathbf{x}\|^2
$$

But $\Pi_U(\mathbf{x}) - \mathbf{u} \in U$, so this says some other point of $U$ is strictly closer to $\mathbf{x}$ than $\Pi_U(\mathbf{x})$, contradicting that $\Pi_U(\mathbf{x})$ is the closest point. Hence $\mathbf{u} = 0$, that is $\Pi_U(\mathbf{x}) - \mathbf{x} \in U^\perp$.

**Linearity.** The map is homogeneous because the norm is homogeneous,

$$
\Pi_U(\alpha \mathbf{x}) = \arg\min_{\mathbf{x}' \in U}\|\alpha \mathbf{x} - \mathbf{x}'\| = \alpha \arg\min_{\mathbf{x}'' \in U}\|\mathbf{x} - \mathbf{x}''\| = \alpha\, \Pi_U(\mathbf{x})
$$

and a similar argument using the orthogonality of the error gives additivity, $\Pi_U(\mathbf{x} + \mathbf{y}) = \Pi_U(\mathbf{x}) + \Pi_U(\mathbf{y})$.
{{< /callout >}}

So for a given subspace $U$, the optimal reconstruction map is the orthogonal projection onto $U$, represented by a matrix $\mathbf{P}$ with $\text{rank}(\mathbf{P}) = \dim(U)$. **The optimal linear autoencoder represents a projection.** As a matrix, $\mathbf{P}$ has the two defining properties we will use repeatedly:

$$
\mathbf{P}^2 = \mathbf{P} \quad (\text{idempotent}), \qquad \mathbf{P} = \mathbf{P}^T \quad (\text{symmetric})
$$

Idempotency alone already pins down the spectrum of any projection: its eigenvalues can only be $0$ or $1$.

{{< callout type="proof" title="Eigenvalues of a Projection are 0 or 1" >}}
Suppose $\mathbf{P}\mathbf{v} = \lambda \mathbf{v}$ with $\mathbf{v} \neq 0$. Applying $\mathbf{P}$ again and using $\mathbf{P}^2 = \mathbf{P}$,

$$
\lambda \mathbf{v} = \mathbf{P}\mathbf{v} = \mathbf{P}^2 \mathbf{v} = \mathbf{P}(\lambda \mathbf{v}) = \lambda \mathbf{P} \mathbf{v} = \lambda^2 \mathbf{v}
$$

so $\lambda^2 = \lambda$, which forces $\lambda \in \{0, 1\}$. The eigenvectors with eigenvalue $1$ are the directions $\mathbf{P}$ keeps, spanning $\text{im}(\mathbf{P}) = U$, and those with eigenvalue $0$ are the directions $\mathbf{P}$ discards, spanning $\ker(\mathbf{P})$.
{{< /callout >}}

### Representing the Projection as a Matrix

We know the optimal map is an orthogonal projection. How do we write it as an explicit matrix, and what do the learned weight matrices look like?

Start with the simplest case, projecting onto a line spanned by a unit vector $\mathbf{u}$ (so $\|\mathbf{u}\| = 1$). The projection matrix is the outer product

$$
\mathbf{P}_u = \mathbf{u} \mathbf{u}^T, \qquad \mathbf{P}_u \mathbf{x} = (\mathbf{u} \mathbf{u}^T) \mathbf{x} = \mathbf{u}(\mathbf{u}^T \mathbf{x}) = \langle \mathbf{u}, \mathbf{x}\rangle\, \mathbf{u}
$$

The inner product $\langle \mathbf{u}, \mathbf{x}\rangle$ is the signed length of $\mathbf{x}$ along $\mathbf{u}$, and $\|\mathbf{P}_u \mathbf{x}\| = |\langle \mathbf{u}, \mathbf{x}\rangle| \le \|\mathbf{x}\|$, so a projection never lengthens a vector.

For a general subspace $U$ of dimension $d$, pick an **orthonormal basis** $\{\mathbf{u}_1, \ldots, \mathbf{u}_d\}$ and stack the basis vectors as the columns of $\mathbf{U} = [\mathbf{u}_1 \ \cdots \ \mathbf{u}_d] \in \mathbb{R}^{D \times d}$. The projection matrix is

$$
\mathbf{P} = \mathbf{U} \mathbf{U}^T, \qquad \mathbf{P}\mathbf{x} = \sum_{i=1}^d \langle \mathbf{u}_i, \mathbf{x}\rangle\, \mathbf{u}_i
$$

which is a sum of $d$ rank-one projections, one onto each basis direction. The coefficients $\langle \mathbf{u}_i, \mathbf{x}\rangle$ are precisely the coordinates of the projection in the orthonormal basis $\{\mathbf{u}_1, \ldots, \mathbf{u}_d\}$, collected into the vector $\mathbf{U}^T \mathbf{x}$. This is how one changes coordinates into an orthonormal basis in general: given any $\mathbf{v} \in \mathbb{R}^D$ and an orthonormal basis with matrix $\mathbf{U}$, its coordinates in that basis are $\mathbf{U}^T \mathbf{v}$. This satisfies the two required properties, using that the orthonormal columns give $\mathbf{U}^T \mathbf{U} = \mathbf{I}_d$:

$$
\mathbf{P}^T = (\mathbf{U} \mathbf{U}^T)^T = \mathbf{U} \mathbf{U}^T = \mathbf{P}, \qquad \mathbf{P} \mathbf{P} = \mathbf{U} \underbrace{\mathbf{U}^T \mathbf{U}}_{= \mathbf{I}_d} \mathbf{U}^T = \mathbf{U} \mathbf{U}^T = \mathbf{P}
$$

{{< callout type="info" title="Tied versus Untied Weights" >}}
The orthonormal form $\mathbf{P} = \mathbf{U} \mathbf{U}^T$ corresponds to an autoencoder with **tied (shared) parameters**, where the decoder is the transpose of the encoder, $\mathbf{V} = \mathbf{W}^T = \mathbf{U}$. The resulting weight matrix is **semi-orthogonal**, meaning its columns are orthonormal ($\mathbf{U}^T \mathbf{U} = \mathbf{I}_d$) even though it is not square. This is an extra constraint we impose, it is not what an unconstrained autoencoder finds on its own.
{{< /callout >}}

What if we do not tie the weights? Then the learned basis of $U$ is some set of linearly independent but not necessarily orthonormal vectors $\{\mathbf{v}_1, \ldots, \mathbf{v}_d\}$, the columns of $\mathbf{V} = [\mathbf{v}_1 \ \cdots \ \mathbf{v}_d]$. We can still recover the orthogonal projection onto $U = \text{span}\{\mathbf{v}_1, \ldots, \mathbf{v}_d\}$ using the [left Moore-Penrose pseudoinverse](/garden/maths/linearAlgebra/moorePenrose#full-column-rank):

$$
\mathbf{P} = \mathbf{V} \mathbf{V}^+, \qquad \mathbf{V}^+ = (\mathbf{V}^T \mathbf{V})^{-1} \mathbf{V}^T
$$

This is the same projection matrix $\mathbf{V}(\mathbf{V}^T \mathbf{V})^{-1} \mathbf{V}^T$ derived in the [projections note](/garden/maths/linearAlgebra/projectionsLS#projections-in-higher-dimensions), just written with the pseudoinverse. It is an orthogonal projection onto $U$ for any choice of basis $\mathbf{V}$.

{{< callout type="proof" title="P = VV^+ is the Orthogonal Projection onto U" >}}
Write $\mathbf{V}^+ = (\mathbf{V}^T \mathbf{V})^{-1} \mathbf{V}^T$, which is well defined because the columns of $\mathbf{V}$ are linearly independent, so $\mathbf{V}^T \mathbf{V}$ is invertible.

**Idempotency.**
$$
\mathbf{P}^2 = \mathbf{V}(\mathbf{V}^T \mathbf{V})^{-1}\underbrace{(\mathbf{V}^T \mathbf{V})(\mathbf{V}^T \mathbf{V})^{-1}}_{= \mathbf{I}_d}\mathbf{V}^T = \mathbf{V}(\mathbf{V}^T \mathbf{V})^{-1}\mathbf{V}^T = \mathbf{P}
$$

**Projects onto $U$.** Applying $\mathbf{P}$ to the whole matrix $\mathbf{V}$ gives $\mathbf{P}\mathbf{V} = \mathbf{V}(\mathbf{V}^T \mathbf{V})^{-1}\mathbf{V}^T \mathbf{V} = \mathbf{V}$, so $\mathbf{P}$ fixes every column of $\mathbf{V}$ and hence, by linearity, every vector of $U = \text{span}(\mathbf{V})$. Together with $\text{rank}(\mathbf{P}) = \dim(U)$ this means $\mathbf{P}$ is the projection onto $U$.

**Symmetry.**
$$
\mathbf{P}^T = \big(\mathbf{V}(\mathbf{V}^T \mathbf{V})^{-1}\mathbf{V}^T\big)^T = \mathbf{V}\big((\mathbf{V}^T \mathbf{V})^{-1}\big)^T \mathbf{V}^T = \mathbf{V}(\mathbf{V}^T \mathbf{V})^{-1}\mathbf{V}^T = \mathbf{P}
$$
using that $\mathbf{V}^T \mathbf{V}$ is symmetric, so its inverse is symmetric.
{{< /callout >}}

A useful corollary follows immediately. Given any decoder $\mathbf{V}$, the encoder $\mathbf{W}$ that minimizes the risk is the left pseudoinverse of $\mathbf{V}$,

$$
\mathbf{V}^+ = \arg\min_W \mathcal{R}(\mathbf{W}, \mathbf{V}), \qquad \mathcal{R}(\mathbf{W}, \mathbf{V}) = \frac{1}{2}\mathbb{E}\|\mathbf{x} - \mathbf{V}\mathbf{W}\mathbf{x}\|^2
$$

In words, once we have chosen a basis (the columns of $\mathbf{V}$) for the subspace, the optimal encoder is determined: it reads off the coordinates in that basis via the pseudoinverse. So the only real freedom left is the choice of subspace.

We can see the effect of that choice on a real dataset. The following uses the `digits` dataset from scikit-learn ($8 \times 8$ images, so $D = 64$) with a bottleneck of $d = 20$, comparing three choices of the decoder basis $\mathbf{V}$ against the optimal encoder $\mathbf{W} = \mathbf{V}^+$.

```python
import numpy as np
from sklearn.datasets import load_digits

# Load and center the data, columns are samples: X is (D, n)
X = np.transpose(load_digits().data - load_digits().data.mean(axis=0))
D, n = X.shape
d = 20  # bottleneck dimension

def mse(V):
    W = np.linalg.pinv(V)          # optimal encoder given V is the pseudoinverse
    return np.mean(np.sum((X - V @ W @ X) ** 2, axis=0))

V_random = np.random.rand(D, d)    # choice 1: random subspace
V_columns = X[:, :d]               # choice 2: first d data points as basis
print("random projection      MSE =", mse(V_random))   # ~13.5
print("random data columns    MSE =", mse(V_columns))  # ~4.40
```

A random subspace reconstructs poorly, and using actual data points as the basis does better, but neither is optimal. The optimal subspace, which we identify next, brings the MSE down to about $1.98$.

### Which Subspace? Maximizing Variance

So far we have not referred to the data distribution at all beyond centering. We now ask which subspace $U$ gives the smallest reconstruction risk. The key step is to rewrite the risk using the two properties of an orthogonal projection, idempotency ($\mathbf{P}^2 = \mathbf{P}$) and symmetry ($\mathbf{P} = \mathbf{P}^T$, so $\langle \mathbf{x}, \mathbf{P}\mathbf{x}\rangle = \langle \mathbf{P}\mathbf{x}, \mathbf{P}\mathbf{x}\rangle$):

$$
\begin{align*}
\mathcal{R}(\mathbf{P}) = \frac{1}{2}\mathbb{E}\|\mathbf{x} - \mathbf{P}\mathbf{x}\|^2 &= \frac{1}{2}\mathbb{E}\|\mathbf{x}\|^2 + \frac{1}{2}\mathbb{E}\|\mathbf{P}\mathbf{x}\|^2 - \mathbb{E}\langle \mathbf{x}, \mathbf{P}\mathbf{x}\rangle \\
&= \frac{1}{2}\mathbb{E}\|\mathbf{x}\|^2 + \frac{1}{2}\mathbb{E}\|\mathbf{P}\mathbf{x}\|^2 - \mathbb{E}\langle \mathbf{P}\mathbf{x}, \mathbf{P}\mathbf{x}\rangle \\
&= \frac{1}{2}\mathbb{E}\|\mathbf{x}\|^2 - \frac{1}{2}\mathbb{E}\|\mathbf{P}\mathbf{x}\|^2
\end{align*}
$$

For centered data, $\mathbb{E}\|\mathbf{x}\|^2 = \text{Var}[\mathbf{x}]$ and $\mathbb{E}\|\mathbf{P}\mathbf{x}\|^2 = \text{Var}[\mathbf{P}\mathbf{x}]$, since $\mathbb{E}[\mathbf{P}\mathbf{x}] = \mathbf{P}\,\mathbb{E}[\mathbf{x}] = 0$. So the risk is

$$
\mathcal{R}(\mathbf{P}) = \frac{1}{2}\Big(\underbrace{\text{Var}[\mathbf{x}]}_{\text{const}} - \text{Var}[\mathbf{P}\mathbf{x}]\Big)
$$

The first term is a constant that does not depend on $\mathbf{P}$. Therefore **minimizing the reconstruction loss is exactly the same as maximizing the variance of the projected data** $\text{Var}[\mathbf{P}\mathbf{x}]$. This is the precise sense in which a linear autoencoder does PCA: PCA is defined as the projection that [maximizes the variance of the projection](/garden/maths/linearAlgebra/pca#the-pca-problem), and we have just shown that this is what minimizing reconstruction error amounts to. Intuitively, the projection that loses the least information is the one that keeps the data as spread out as possible.

{{< figure
    src="/images/ml/linearAutoencoderVariance.png"
    alt="Projecting 2D data onto different lines, the line of maximum variance gives the smallest reconstruction error."
    caption="Minimizing reconstruction error (red, the squared distances to the line) is equivalent to maximizing the variance of the projected points (their spread along the line)."
>}}

The same idea seen from the variance side appears in the [PCA notes](/garden/maths/linearAlgebra/pca#the-pca-problem).

{{< figure
    src="/images/maths/pcaVariance.webp"
    alt="Projecting 2D data onto different 1D lines, low variance loses cluster structure while maximum variance preserves it."
    caption="Projecting onto a low-variance direction (left) collapses the two clusters and loses structure, which is exactly a large reconstruction error. Projecting onto the direction of maximum variance (right) keeps the points spread out and reconstructs them best."
>}}

#### The Covariance Matrix is a Sufficient Statistic

We can say something stronger about exactly which features of the data matter. Define the (centered) **covariance matrix**

$$
\mathbf{S} = \mathbb{E}[\mathbf{x} \mathbf{x}^T] \in \mathbb{R}^{D \times D}
$$

This is the population version of the [sample covariance matrix](/garden/maths/linearAlgebra/pca#variance-and-the-covariance-matrix) from the PCA notes, estimated from data as $\hat{\mathbf{S}} = \frac{1}{n}\sum_{i=1}^n \mathbf{x}_i \mathbf{x}_i^T$ for centered data (or with the unbiased factor $\frac{1}{n-1}$). It is [symmetric and positive semi-definite](/garden/maths/linearAlgebra/eigendecomposition#positive-definite-and-positive-semi-definite-matrices), the latter because for any direction $\mathbf{v}$ we have $\mathbf{v}^T \mathbf{S} \mathbf{v} = \mathbb{E}[(\mathbf{v}^T \mathbf{x})^2] \ge 0$, a variance is never negative. This is exactly the property we needed for the [spectral theorem](/garden/maths/linearAlgebra/eigendecomposition#eigenvalues-and-eigenvectors-of-symmetric-matrices) to apply with non-negative eigenvalues. We claim $\mathbf{S}$ is a **sufficient statistic** for the risk, meaning it summarizes all the information about the data that we need in order to evaluate and optimize $\mathcal{R}(\mathbf{P})$. Concretely,

$$
\mathcal{R}(\mathbf{P}) = -\frac{1}{2}\text{tr}\big(\mathbf{P}\, \mathbf{S}\big) + \text{const}
$$

The proof rewrites the variance term using the [trace](/garden/maths/linearAlgebra/eigendecomposition#the-rayleigh-quotient). We need three properties of the trace (see the [trace properties](/garden/maths/linearAlgebra/eigendecomposition#properties-of-eigenvalues-and-eigenvectors)):

- A scalar equals its own trace, and an inner product is a trace: $\langle \mathbf{x}, \mathbf{y}\rangle = \mathbf{x}^T \mathbf{y} = \text{tr}(\mathbf{x}^T \mathbf{y}) = \text{tr}(\mathbf{x} \mathbf{y}^T)$.
- The trace is linear and commutes with expectation: $\text{tr}(\alpha \mathbf{A} + \beta \mathbf{B}) = \alpha\,\text{tr}(\mathbf{A}) + \beta\,\text{tr}(\mathbf{B})$ and $\mathbb{E}[\text{tr}(\mathbf{A})] = \text{tr}(\mathbb{E}[\mathbf{A}])$.
- The trace is invariant under cyclic permutations: $\text{tr}(\mathbf{A}\mathbf{B}\mathbf{C}) = \text{tr}(\mathbf{C}\mathbf{A}\mathbf{B}) = \text{tr}(\mathbf{B}\mathbf{C}\mathbf{A})$. Note this holds only for cyclic reorderings, not arbitrary ones, even though the intermediate products change shape.

{{< callout type="proof" title="Covariance Matrix is a Sufficient Statistic" >}}
Using symmetry of $\mathbf{P}$ to write $\text{Var}[\mathbf{P}\mathbf{x}] = \mathbb{E}\langle \mathbf{P}\mathbf{x}, \mathbf{P}\mathbf{x}\rangle = \mathbb{E}\langle \mathbf{x}, \mathbf{P}\mathbf{x}\rangle$, then turning the inner product into a trace and applying the cyclic property,

$$
\text{Var}[\mathbf{P}\mathbf{x}] = \mathbb{E}\,\text{tr}(\mathbf{x}^T \mathbf{P} \mathbf{x}) = \mathbb{E}\,\text{tr}(\mathbf{P} \mathbf{x} \mathbf{x}^T) = \text{tr}\big(\mathbb{E}[\mathbf{P} \mathbf{x} \mathbf{x}^T]\big) = \text{tr}\big(\mathbf{P}\, \underbrace{\mathbb{E}[\mathbf{x} \mathbf{x}^T]}_{= \mathbf{S}}\big)
$$

Combining with $\mathcal{R}(\mathbf{P}) = \frac{1}{2}(\text{Var}[\mathbf{x}] - \text{Var}[\mathbf{P}\mathbf{x}])$ gives $\mathcal{R}(\mathbf{P}) = \frac{1}{2}\text{Var}[\mathbf{x}] - \frac{1}{2}\text{tr}(\mathbf{P} \mathbf{S})$, where the first term is constant in $\mathbf{P}$.
{{< /callout >}}

The optimal projection is therefore fully determined by the covariance matrix $\mathbf{S} = \mathbb{E}[\mathbf{x} \mathbf{x}^T]$ (together with the mean $\mathbb{E}[\mathbf{x}]$ used for centering). Nothing else about the data distribution matters.

### The PCA Theorem

We can now state and prove the result that ties everything together.

{{< callout type="info" title="PCA Theorem" >}}
The rank-$d$ projection that minimizes the squared reconstruction loss on centered data is the orthogonal projection onto the span of the $d$ principal (largest-eigenvalue) eigenvectors of the covariance matrix $\mathbf{S} = \mathbb{E}[\mathbf{x} \mathbf{x}^T]$. That is, the optimal $\mathbf{P} = \mathbf{U} \mathbf{U}^T$ where the columns of $\mathbf{U}$ are $d$ orthonormal eigenvectors of $\mathbf{S}$ corresponding to its $d$ largest eigenvalues.
{{< /callout >}}

From the previous section, maximizing $\text{Var}[\mathbf{P}\mathbf{x}] = \text{tr}(\mathbf{P} \mathbf{S})$ over rank-$d$ orthogonal projections $\mathbf{P} = \mathbf{U} \mathbf{U}^T$ (with $\mathbf{U}^T \mathbf{U} = \mathbf{I}_d$) is the optimization we must solve. We build up to it in two steps.

#### The Diagonal Case

First suppose the covariance is already diagonal,

$$
\mathbf{S} = \mathbf{\Lambda} = \text{diag}(\lambda_1, \ldots, \lambda_D), \qquad \lambda_1 \ge \lambda_2 \ge \cdots \ge \lambda_D \ge 0
$$

A diagonal covariance means the coordinates are uncorrelated, and $\lambda_j$ is just the variance of the $j$-th coordinate. In this case the answer is intuitive: project onto the $d$ coordinate axes with the largest variances. The optimal projection is

$$
\mathbf{P} = \sum_{i=1}^d \mathbf{e}_i \mathbf{e}_i^T = \begin{bmatrix} \mathbf{I}_d & 0 \\ 0 & 0 \end{bmatrix}
$$

where $\mathbf{e}_1, \ldots, \mathbf{e}_d$ are the first $d$ standard basis vectors.

{{< callout type="proof" title="Diagonal Case" >}}
The standard basis vectors are eigenvectors of any diagonal matrix, $\mathbf{\Lambda} \mathbf{e}_j = \lambda_j \mathbf{e}_j$, they are orthonormal, and ordering the diagonal entries in decreasing order orders them by principality. It remains to show that picking the first $d$ of them maximizes the variance. We maximize over $\mathbf{U} \in \mathbb{R}^{D \times d}$ with $\mathbf{U}^T \mathbf{U} = \mathbf{I}_d$:

$$
\text{Var}[\mathbf{P}\mathbf{x}] = \text{tr}(\mathbf{U} \mathbf{U}^T \mathbf{\Lambda}) = \sum_{i=1}^D \lambda_i \underbrace{\sum_{j=1}^d u_{ij}^2}_{=\, \gamma_i}
$$

Two facts pin down the weights $\gamma_i$. First, $\gamma_i \le 1$ for every $i$, because $\gamma_i = \|\mathbf{P} \mathbf{e}_i\|^2$ and a projection is non-expansive ($\|\mathbf{P} \mathbf{e}_i\| \le \|\mathbf{e}_i\| = 1$). Second, the total is fixed,

$$
\sum_{i=1}^D \gamma_i = \sum_{i,j} u_{ij}^2 = \|\mathbf{U}\|_F^2 = \text{tr}(\mathbf{U}^T \mathbf{U}) = \text{tr}(\mathbf{I}_d) = d
$$

So we are spreading a total weight of $d$ across the $\gamma_i \in [0,1]$ to maximize $\sum_i \lambda_i \gamma_i$ with the $\lambda_i$ sorted in decreasing order. The maximum puts all the weight on the largest eigenvalues, $\gamma_1 = \cdots = \gamma_d = 1$ and the rest zero, which is achieved by $\mathbf{U} = [\mathbf{e}_1 \ \cdots \ \mathbf{e}_d]$, giving $\mathbf{P} = \begin{bmatrix} \mathbf{I}_d & 0 \\ 0 & 0\end{bmatrix}$.
{{< /callout >}}

The maximal variance attained is $\text{Var}[\mathbf{P}\mathbf{x}] = \sum_{i=1}^d \lambda_i$, the sum of the $d$ largest variances. For a projection onto a single line ($d = 1$) that minimizes the squared loss, this is simply the largest eigenvalue $\lambda_1$ of the covariance. If some eigenvalues are equal, say $\lambda_i = \cdots = \lambda_{i+k}$, then the choice of axes within that tied block is not unique. Any orthonormal basis of the corresponding eigenspace works equally well, which is our first hint that the solution determines a subspace rather than a specific basis.

#### The General Case via the Spectral Theorem

For a general covariance matrix we use the [spectral theorem](/garden/maths/linearAlgebra/eigendecomposition#eigenvalues-and-eigenvectors-of-symmetric-matrices) to reduce to the diagonal case. Since $\mathbf{S}$ is symmetric and positive semi-definite, it can be diagonalized by an orthogonal matrix,

$$
\mathbf{S} = \mathbf{Q} \mathbf{\Lambda} \mathbf{Q}^T, \qquad \mathbf{\Lambda} = \text{diag}(\lambda_1, \ldots, \lambda_D), \quad \lambda_1 \ge \cdots \ge \lambda_D \ge 0
$$

where the columns of the orthogonal matrix $\mathbf{Q}$ are the orthonormal eigenvectors of $\mathbf{S}$ ordered by decreasing eigenvalue. Conversely, $\mathbf{Q}\mathbf{\Lambda} \mathbf{Q}^T$ for any orthogonal $\mathbf{Q}$ is a symmetric matrix whose eigenvalues are the diagonal entries of $\mathbf{\Lambda}$, the simplest example being $\mathbf{\Lambda}$ itself with $\mathbf{Q} = \mathbf{I}$. In this eigenbasis the action of $\mathbf{S}$ is just a coordinate-wise scaling: the coordinates of $\mathbf{S}\mathbf{x}$ relative to $\{\mathbf{q}_1, \ldots, \mathbf{q}_D\}$ are $\mathbf{Q}^T \mathbf{S} \mathbf{x} = \mathbf{\Lambda} \mathbf{Q}^T \mathbf{x}$, so the $i$-th coordinate is $\lambda_i\,(\mathbf{q}_i^T \mathbf{x})$. The optimal projection is then

$$
\mathbf{P} = \mathbf{U} \mathbf{U}^T, \qquad \mathbf{U} = \mathbf{Q} \begin{bmatrix} \mathbf{I}_d \\ 0 \end{bmatrix}
$$

that is, $\mathbf{U}$ consists of the first $d$ columns of $\mathbf{Q}$, the $d$ principal eigenvectors of $\mathbf{S}$.

{{< callout type="proof" title="General Case" >}}
Substitute $\mathbf{S} = \mathbf{Q}\mathbf{\Lambda} \mathbf{Q}^T$ into the variance and use the cyclic property of the trace,

$$
\text{Var}[\mathbf{P}\mathbf{x}] = \text{tr}\big(\mathbf{U} \mathbf{U}^T\, \mathbf{Q}\mathbf{\Lambda} \mathbf{Q}^T\big) = \text{tr}\big((\mathbf{Q}^T \mathbf{U})(\mathbf{Q}^T \mathbf{U})^T \mathbf{\Lambda}\big)
$$

Setting $\tilde{\mathbf{U}} = \mathbf{Q}^T \mathbf{U}$, this is exactly the diagonal problem from before (note $\tilde{\mathbf{U}}^T \tilde{\mathbf{U}} = \mathbf{U}^T \mathbf{Q} \mathbf{Q}^T \mathbf{U} = \mathbf{U}^T \mathbf{U} = \mathbf{I}_d$ since $\mathbf{Q}$ is orthogonal). Its maximizer is $\tilde{\mathbf{U}} = \mathbf{Q}^T \mathbf{U} = \begin{bmatrix} \mathbf{I}_d \\ 0\end{bmatrix}$, and multiplying by $\mathbf{Q}$ gives $\mathbf{U} = \mathbf{Q}\begin{bmatrix} \mathbf{I}_d \\ 0\end{bmatrix}$, the first $d$ eigenvectors of $\mathbf{S}$.
{{< /callout >}}

#### Subspace, Not Basis

There is an important subtlety. The linear autoencoder recovers the principal **subspace** $\text{span}\big(\mathbf{Q}\begin{bmatrix} \mathbf{I}_d \\ 0\end{bmatrix}\big)$, but it does not necessarily recover the PCA **basis** (the individual principal eigenvectors). This is because the optimal $\mathbf{P} = \mathbf{V} \mathbf{V}^+$ can use any basis $\mathbf{V}$ of the subspace, and as we saw the factorization $\mathbf{P} = \mathbf{V}\mathbf{W}$ is non-identifiable. So a trained linear autoencoder gives you the right subspace but generally a rotated, non-orthonormal basis within it, not the ordered principal components themselves. To get the actual principal components one must additionally diagonalize, for example with an eigendecomposition of the covariance.

#### Why Eigenvectors?

It may seem mysterious that eigenvectors, defined by $\mathbf{S} \mathbf{u} = \lambda \mathbf{u}$, show up in a variance-maximization problem that never mentioned them. Two facts about symmetric matrices explain it. First, the principal eigenvector is exactly the direction of maximal stretch: for a symmetric $\mathbf{S}$ with principal unit eigenvector $\mathbf{u}_1$,

$$
\|\mathbf{S} \mathbf{u}_1\| = \max_{\|\mathbf{v}\| = 1}\|\mathbf{S} \mathbf{v}\|
$$

which is the content of the [Rayleigh quotient](/garden/maths/linearAlgebra/eigendecomposition#the-rayleigh-quotient) bound. Second, eigenvectors of a symmetric matrix belonging to distinct eigenvalues are orthogonal, so they automatically form the orthonormal basis we want.

{{< callout type="proof" title="Eigenvectors of a Symmetric Matrix are Orthogonal" >}}
Let $\mathbf{S} \mathbf{u} = \lambda \mathbf{u}$ and $\mathbf{S} \mathbf{u}' = \lambda' \mathbf{u}'$ with $\lambda \neq \lambda'$. Using symmetry $\langle \mathbf{S} \mathbf{u}, \mathbf{u}'\rangle = \langle \mathbf{u}, \mathbf{S} \mathbf{u}'\rangle$,

$$
\lambda \langle \mathbf{u}, \mathbf{u}'\rangle = \langle \mathbf{S} \mathbf{u}, \mathbf{u}'\rangle = \langle \mathbf{u}, \mathbf{S} \mathbf{u}'\rangle = \lambda' \langle \mathbf{u}, \mathbf{u}'\rangle
$$

Hence $(\lambda - \lambda')\langle \mathbf{u}, \mathbf{u}'\rangle = 0$, and since $\lambda \neq \lambda'$ we get $\langle \mathbf{u}, \mathbf{u}'\rangle = 0$.
{{< /callout >}}

### Learning Algorithms

The theory tells us the optimum is built from the principal eigenvectors of the covariance matrix. There are several ways to actually compute it, trading off accuracy, scalability, and how much of the spectrum we need. Here $D$ is the data dimension, $n$ the number of samples, and $d$ the bottleneck size.

#### Matrix Method: Eigendecomposition

The direct approach forms the covariance matrix and diagonalizes it. Since the covariance is symmetric, we use a specialized routine (`numpy.linalg.eigh`, backed by LAPACK) and keep the $d$ eigenvectors with the largest eigenvalues.

```python
import numpy as np

covar = np.cov(X)                       # (D, D) covariance, X is (D, n)
evals, evecs = np.linalg.eigh(covar)    # ascending eigenvalues
V = evecs[:, -d:]                       # top d eigenvectors (decoder)
W = np.transpose(V)                     # tied encoder, since V is orthonormal
# On the digits dataset with d = 20 this gives MSE = 1.984,
# compared to 4.40 for the non-optimal basis from before.
```

The cost is $O(n D^2)$ to estimate the covariance plus $O(D^3)$ for the eigendecomposition. The cubic term is the bottleneck and becomes prohibitive when $D$ is large. This is wasteful when we only want a small number $d \ll D$ of components, because we compute the entire spectrum and then throw most of it away.

#### Power Method

When $d \ll D$ we should not need the full covariance spectrum. The **power method** (power iteration) is a scalable algorithm for the principal eigenvector alone. Starting from a random vector $\mathbf{v}^{(0)} \sim \mathcal{N}(0, \mathbf{I})$, repeatedly multiply by $\mathbf{S}$ and renormalize:

$$
\mathbf{v}^{(t+1)} = \frac{\mathbf{S} \mathbf{v}^{(t)}}{\|\mathbf{S} \mathbf{v}^{(t)}\|}
$$

Each multiplication stretches the vector most along the principal eigenvector, so after enough iterations the direction converges to $\mathbf{u}_1$.

{{< callout type="proof" title="Convergence of the Power Method" >}}
Let $\mathbf{S}$ be diagonalizable with eigenvalues $\lambda_1 > \lambda_2 \ge \cdots \ge \lambda_D \ge 0$ and orthonormal eigenvectors $\mathbf{u}_i$. Expand the start vector in the eigenbasis, $\mathbf{v}^{(0)} = \sum_i \alpha_i \mathbf{u}_i$, and assume $\alpha_1 \neq 0$ (true with probability one for a random start). Since each step applies $\mathbf{S}$ and renormalizes, after $k$ steps

$$
\mathbf{v}^{(k)} \propto \mathbf{S}^k \mathbf{v}^{(0)} = \sum_i \alpha_i \lambda_i^k \mathbf{u}_i = \lambda_1^k\left(\alpha_1 \mathbf{u}_1 + \sum_{i > 1}\alpha_i \left(\frac{\lambda_i}{\lambda_1}\right)^k \mathbf{u}_i\right)
$$

Because $\lambda_i / \lambda_1 < 1$ for $i > 1$, every term in the sum decays to zero, so after renormalization $\mathbf{v}^{(k)} \to \mathbf{u}_1$.
{{< /callout >}}

Each iteration costs $O(D^2)$, so $T$ iterations cost $O(T D^2)$. To get the next eigenvectors we **deflate**: once we have the principal eigenvector $\mathbf{q}_1$ and its eigenvalue $\lambda_1$ (from $\mathbf{S} \mathbf{q}_1 = \lambda_1 \mathbf{q}_1$), we subtract its contribution and rerun the power method on

$$
\mathbf{S}_2 = \mathbf{S} - \lambda_1 \mathbf{q}_1 \mathbf{q}_1^T
$$

This zeros out the top eigenvalue while leaving the rest untouched, since in the eigenbasis $\mathbf{Q}^T \mathbf{S} \mathbf{Q} = \text{diag}(\lambda_1, \lambda_2, \ldots) \mapsto \text{diag}(0, \lambda_2, \ldots) = \mathbf{Q}^T \mathbf{S}_2 \mathbf{Q}$. Repeating gives the top $d$ eigenvectors at total cost $O(T d D^2)$ plus the $O(n D^2)$ for the covariance, which is much cheaper than $O(D^3)$ when $d \ll D$.

```python
import numpy as np

def power_method(S, num_iters=100):
    v = np.random.randn(S.shape[0])
    for _ in range(num_iters):
        v = S @ v
        v = v / np.linalg.norm(v)   # renormalize to unit length
    return v

def top_eigenvectors(S, d, num_iters=100):
    S = S.copy()
    vecs = []
    for _ in range(d):
        v = power_method(S, num_iters)
        lam = v @ S @ v             # Rayleigh quotient gives the eigenvalue
        vecs.append(v)
        S = S - lam * np.outer(v, v)  # deflate: remove this direction
    return np.column_stack(vecs)

V = top_eigenvectors(covar, d)      # decoder, the top d principal eigenvectors
```

#### Stochastic Gradient Descent

The most generic approach treats the autoencoder as a neural network and trains it with [gradient descent](/garden/ml/gradientDescent), the workhorse of deep learning. We need the gradient of the single-example loss $\mathcal{L}(\mathbf{x}; \mathbf{P}) = \frac{1}{2}\|\mathbf{x} - \mathbf{P}\mathbf{x}\|^2$ with respect to the matrices. Expanding with the trace,

$$
\mathcal{L}(\mathbf{x}; \mathbf{P}) = \frac{1}{2}\|\mathbf{x}\|^2 - \text{tr}(\mathbf{P} \mathbf{x} \mathbf{x}^T) + \frac{1}{2}\text{tr}(\mathbf{P}^T \mathbf{P} \mathbf{x} \mathbf{x}^T)
$$

and applying the matrix differentiation rules $\frac{\partial}{\partial \mathbf{A}}\text{tr}(\mathbf{A}\mathbf{B}) = \mathbf{B}^T$ and $\frac{\partial}{\partial \mathbf{A}}\text{tr}(\mathbf{A}^T \mathbf{A} \mathbf{B}) = \mathbf{A}\mathbf{B}^T + \mathbf{A}\mathbf{B}$ (see the [matrix cookbook](https://www.math.uwaterloo.ca/~hwolkowi/matrixcookbook.pdf)), the gradient with respect to $\mathbf{P}$ is

$$
\nabla_P\, \mathcal{L}(\mathbf{x}; \mathbf{P}) = (\mathbf{P} - \mathbf{I})\mathbf{x} \mathbf{x}^T
$$

The first rule is worth seeing explicitly, since it is the workhorse of these derivations.

{{< callout type="proof" title="Gradient of tr(AB)" >}}
Writing the trace as a double sum, $\text{tr}(\mathbf{A}\mathbf{B}) = \sum_{i}\sum_{k} a_{ik} b_{ki}$. Differentiating with respect to a single entry $a_{ij}$ picks out the one term where $i$ and $k = j$ match,

$$
\frac{\partial}{\partial a_{ij}}\text{tr}(\mathbf{A}\mathbf{B}) = b_{ji} = (\mathbf{B}^T)_{ij}
$$

and collecting these entries into a matrix gives $\frac{\partial}{\partial \mathbf{A}}\text{tr}(\mathbf{A}\mathbf{B}) = \mathbf{B}^T$.
{{< /callout >}}

But we update the weight matrices $\mathbf{V}$ and $\mathbf{W}$, not $\mathbf{P} = \mathbf{V}\mathbf{W}$ directly. By the chain rule for matrix derivatives,

$$
\nabla_W\, \mathcal{L}(\mathbf{x}; \mathbf{V}\mathbf{W}) = \mathbf{V}^T (\mathbf{P} - \mathbf{I}) \mathbf{x} \mathbf{x}^T, \qquad \nabla_V\, \mathcal{L}(\mathbf{x}; \mathbf{V}\mathbf{W}) = (\mathbf{P} - \mathbf{I}) \mathbf{x} \mathbf{x}^T \mathbf{W}^T
$$

A quick dimension check confirms each gradient has the same shape as the matrix it updates. **Stochastic gradient descent** then picks a data point at random and takes a step against the gradient with learning rate $\eta > 0$:

$$
\mathbf{W} \leftarrow \mathbf{W} - \eta\, \nabla_W\, \mathcal{L}(\mathbf{x}; \mathbf{V}\mathbf{W}), \qquad \mathbf{V} \leftarrow \mathbf{V} - \eta\, \nabla_V\, \mathcal{L}(\mathbf{x}; \mathbf{V}\mathbf{W})
$$

In batched matrix form this becomes a few lines, using the covariance to average over a batch:

```python
Delta = (V @ W - np.identity(D)) @ covar   # (P - I) S
gradV = Delta @ np.transpose(W)
gradW = np.transpose(V) @ Delta
V = V - eta * gradV
W = W - eta * gradW
```

For a batch of size $k$ the cost per iteration is dominated by the matrix products, giving $O\big(T(d + k)D^2\big)$ over $T$ iterations. This scales well and lets us reuse all the tooling of deep learning, at the price of the usual gradient-descent caveats around step sizes and convergence speed.

### Non-Linear Autoencoders

Everything above used linear maps. A **non-linear autoencoder** replaces $F$ and $G$ by non-linear functions $F: \mathbb{R}^D \to \mathbb{R}^d$ and $G: \mathbb{R}^d \to \mathbb{R}^D$, typically built from neural network layers with non-linear activations. Since the class of non-linear functions includes the linear ones as a special case, a non-linear autoencoder can always match and usually beat the reconstruction loss of the best linear autoencoder. It can bend the latent subspace into a curved manifold, capturing structure that no flat projection can.

There is a catch, however. Linear autoencoders are solved globally by the PCA theorem, but non-linear autoencoders are trained by gradient descent with no global convergence guarantees. In practice a poorly optimized non-linear autoencoder can end up worse than the linear optimum. On the digits dataset the numbers are illustrative:

```text
d = 20:  PCA / linear AE  MSE = 1.985    non-linear AE  MSE = 1.360
d =  2:  PCA / linear AE  MSE = 13.42    non-linear AE  MSE = 5.99
```

The linear autoencoder reaches essentially the PCA optimum, and a well-trained non-linear autoencoder improves on it, dramatically so at very small bottlenecks where linear projection is too rigid (for example unrolling a curved 2D manifold embedded in 3D). This gap between what is achievable and what is reliably trainable is exactly what motivates more structured approaches to representation learning, which we turn to next.

## What Makes a Good Representation?

Before diving deeper, we should understand what makes a representation "good". A good representation should satisfy three key properties.

- **Informative:** Given a representation, it should be possible to reconstruct the original input with high fidelity. If the representation discards too much information, it becomes useless for downstream tasks. This property ensures that the latent code captures the essential structure of the data.

- **Disentangled:** Each component in the representation should correspond to a distinct, interpretable feature of the data. For example, in face images, different dimensions of the latent space might independently control attributes like age, gender, or expression. Disentanglement makes the representation more interpretable and easier to manipulate for specific purposes.

- **Robust:** Small perturbations or noise in the input should not drastically change the representation, and conversely, small changes in the representation should lead to smooth, meaningful changes in the reconstructed output. This robustness property is crucial for generalization and prevents the model from memorizing noise in the training data.

Traditional autoencoders are designed primarily to be informative through the reconstruction objective. However, they struggle with the other two properties. The latent space learned by a standard autoencoder can be irregular and discontinuous, making it difficult to generate new samples or interpolate meaningfully. Different regions of the latent space may correspond to very different data distributions, and some regions may not correspond to any valid data at all.

## The Infomax Principle

One classical approach to learning informative representations is the **infomax principle**, introduced by Linsker in 1988. This principle suggests maximizing the mutual information $I(X; Z)$ between the input $X$ and the learned representation $Z$.

Mutual information measures how much knowing one variable tells us about another. It is defined as the reduction in uncertainty about $X$ when we observe $Z$. You can find more details in the [Active Learning section on information theory](/garden/ml/bayesian/activelearning/#information-theory), but briefly the mutual information is:

$$
I(X; Z) = H[X] - H[X \mid Z]
$$

where $H[X]$ is the entropy (uncertainty) of $X$ and $H[X \mid Z]$ is the conditional entropy (remaining uncertainty about $X$ after observing $Z$).

Let $\mathcal{H} = \{enc_\theta: \mathcal{X} \to \mathcal{Z} \mid \theta \in \Theta\}$ be a parametrized family of encoder functions. Given a random variable $X$ with range in $\mathcal{X}$ and a conditional probability density function $p(x \mid z)$, the infomax principle argues that the best encoder is given by:

$$
\arg\max_{\theta} I(X; enc_\theta(X))
$$

To understand this optimization, we first need to express mutual information in terms of probability densities. Recall that mutual information can be written as:

$$
I(X; Z) = \mathbb{E}_{p(x,z)} \left[ \log \frac{p(x, z)}{p(x)p(z)} \right] = \mathbb{E}_{p(x,z)} [\log p(x \mid z)] - \mathbb{E}_{p(x)} [\log p(x)]
$$

where we use the fact that $p(x,z) = p(x \mid z)p(z)$. This shows that mutual information measures how much the conditional distribution $p(x \mid z)$ differs from the marginal $p(x)$.

For a sample $x_1, \ldots, x_n$, we can approximate the mutual information by maximizing the expected log-likelihood under the encoder distribution. This uses the **variational characterization of mutual information**, which provides a lower bound: for any conditional distribution $q(x \mid z)$, we have

$$
I(X; Z) \geq \mathbb{E}_{p(x,z)}[\log q(x \mid z)] - \mathbb{E}_{p(x)} [\log p(x)]
$$

with equality when $q(x \mid z) = p(x \mid z)$. The supremum over all possible $q$ achieves this equality. When we maximize with respect to the encoder parameters $\theta$, we are maximizing a lower bound on this mutual information. The second term $\mathbb{E}_{p(x)} [\log p(x)]$ does not depend on the encoder, so maximizing the bound is equivalent to maximizing the first term. In practice, for a given encoder $enc_\theta$, this leads to:

$$
I(X; enc_\theta(X)) \approx \sum_{i \leq n} \mathbb{E}_{Z \mid x_i} [\log p(x_i \mid Z)]
$$

This objective encourages the encoder to preserve as much information about the input as possible in the latent representation.

However, the infomax principle alone is insufficient for learning good representations. While it guarantees informativeness by definition, it does not necessarily lead to disentangled or robust representations. If the input space $\mathcal{X}$ and latent space $\mathcal{Z}$ are both complex enough, one can trivially maximize mutual information by finding a bijection from $\mathcal{X}$ to $\mathcal{Z}$. This would simply copy the input to the latent space without learning any meaningful compression or structure.

## Variational Autoencoders

Variational autoencoders address the limitations of traditional autoencoders by using a Bayesian approach to representation learning. Instead of learning a deterministic mapping from inputs to latent codes, VAEs learn a probability distribution over latent codes for each input. This probabilistic view enables VAEs to generate new samples and ensures a more regular, continuous latent space.

The key distinction between traditional autoencoders and VAEs is this probabilistic framing. A traditional autoencoder learns mappings $z = g_\phi(x)$ and $\hat{x} = f_\theta(z)$. A VAE instead learns:
- An **encoder** (also called recognition network or inference network) that outputs parameters of a distribution over latent codes: $q_\phi(z \mid x)$
- A **decoder** (also called generative network) that outputs parameters of a distribution over reconstructions: $p_\theta(x \mid z)$

### The Generative Model

To understand VAEs, it helps to start with the generative model of how we imagine data is created. We assume that each data point $x$ is generated through a two-step process:

1. A latent variable $z$ is sampled from a prior distribution $p(z)$. This latent variable represents the underlying factors that generate the data. For instance, for face images, $z$ might encode age, expression, pose, and lighting. The prior $p(z)$ is typically chosen to be a simple distribution such as a standard Gaussian $\mathcal{N}(0, I)$. This choice encourages the latent space to have a regular structure that is easy to sample from.

2. The observed data $x$ is generated from the latent variable according to a likelihood distribution $p_\theta(x \mid z)$. This likelihood is parameterized by a neural network (the decoder $f_\theta$) that maps latent variables to distributions over the data space. For continuous data like images, this is often a Gaussian with mean given by the decoder output: $p_\theta(x \mid z) = \mathcal{N}(x \mid \mu_\theta(z), \sigma^2 I)$, where $\mu_\theta(z) = f_\theta(z)$ is the neural network decoder.

The joint distribution of data and latent variables is then:

$$
p_\theta(x, z) = p_\theta(x \mid z) p(z)
$$

This is the **generative model**, which connects directly to the decoder intuition. The prior $p(z)$ represents our initial belief about the latent space structure (typically a standard Gaussian), while the likelihood $p_\theta(x \mid z)$ represents the decoder's ability to generate data from latent variables. The decoder parameters $\theta$ control how latent variables map to observed data.

Our goal in training is to find parameters $\theta$ that make this generative model assign high probability to the observed data. To evaluate how well the model fits the data, we need the **marginal likelihood** (also called the **evidence**), which is obtained by integrating out the latent variables:

$$
p_\theta(x) = \int p_\theta(x \mid z) p(z) dz
$$

The evidence tells us how probable a data point $x$ is under our model, averaging over all possible latent variables $z$ that could have generated it. However, this integration is generally intractable because it requires integrating over all possible latent variables. For high-dimensional latent spaces with complex decoders (neural networks), this cannot be done analytically or efficiently approximated.

{{< figure
    src="/images/ml/autoencoderVariational.png"
    alt="Variational autoencoder with probabilistic encoder and decoder."
    caption="A VAE encoder outputs distribution parameters (mean and variance), we sample from this distribution, and the decoder maps samples back to the data space."
>}}

### The Inference Model

To make sense of this generative model, we also need to perform **inference**: given an observed data point $x$, what latent code $z$ is likely to have generated it? This is captured by the **posterior distribution** $p_\theta(z \mid x)$. Using Bayes' rule:

$$
p_\theta(z \mid x) = \frac{p_\theta(x \mid z) p(z)}{p_\theta(x)}
$$

However, this posterior is intractable because the denominator $p_\theta(x)$ requires computing the integral we just mentioned. This is where variational inference comes in. We introduce a **variational distribution** (the encoder) $q_\phi(z \mid x)$ parameterized by $\phi$ that approximates the true posterior. The encoder takes a data point $x$ and outputs parameters of a distribution over latent variables, typically the mean and variance of a Gaussian: $q_\phi(z \mid x) = \mathcal{N}(z \mid \mu_\phi(x), \sigma_\phi^2(x) I)$.

Here we connect to the autoencoder view. The encoder neural network $g_\phi$ takes input $x$ and outputs $[\mu_\phi(x), \sigma_\phi(x)]$. We then sample $z \sim q_\phi(z \mid x) = \mathcal{N}(\mu_\phi(x), \sigma_\phi^2(x) I)$. The decoder neural network $f_\theta$ takes this sampled $z$ and outputs the parameters of the reconstruction distribution, typically just the mean $\mu_\theta(z)$, which we interpret as the reconstruction $\hat{x}$.

The **true posterior** $p_\theta(z \mid x)$ represents the ideal inference distribution: given an observed data point $x$, which latent variables $z$ most likely generated it under our generative model? This is exactly what we want our encoder to learn. If we could compute the true posterior, we would know the best latent representation for each data point. However, computing it requires the intractable evidence $p_\theta(x)$ in the denominator. The variational distribution $q_\phi(z \mid x)$ is our learned approximation to this ideal posterior. By training the encoder to approximate the true posterior, we ensure that the latent variables we sample are meaningful representations of the input data.

The variational inference framework tells us how to train both the encoder and decoder. We want to find parameters $\phi$ and $\theta$ such that:
1. The encoder $q_\phi(z \mid x)$ closely approximates the true posterior $p_\theta(z \mid x)$
2. The decoder $p_\theta(x \mid z)$ can reconstruct data well from latent variables

### The ELBO Objective

The VAE training objective is derived using variational inference. Our goal is to train a generative model that assigns high probability to the observed data. In probabilistic modeling, we achieve this by maximizing the log-likelihood $\log p_\theta(x)$ of the training data under the model. This is the standard principle of **maximum likelihood estimation**: we adjust the model parameters $\theta$ to make the observed data as probable as possible. The log is used for mathematical convenience (products become sums) and numerical stability (avoiding underflow with small probabilities).

However, as we saw earlier, computing $p_\theta(x) = \int p_\theta(x \mid z) p(z) dz$ is intractable. Since we cannot directly optimize the intractable log-likelihood $\log p_\theta(x)$, we instead maximize a tractable lower bound called the **Evidence Lower Bound (ELBO)**. The full derivation is covered in detail in the [Variational Inference section](/garden/ml/bayesian/variationalinference/#deriving-the-elbo), but here we focus on how it applies specifically to VAEs.

The ELBO relates to the log-likelihood through the KL divergence (see [KL divergence](/garden/ml/bayesian/variationalinference/#kullback-leibler-divergence) for details):

$$
\log p_\theta(x) = \text{KL}(q_\phi(z \mid x) \parallel p_\theta(z \mid x)) + \mathcal{L}(\theta, \phi; x)
$$

where the ELBO is:

$$
\mathcal{L}(\theta, \phi; x) = \mathbb{E}_{z \sim q_\phi(z \mid x)} [\log p_\theta(x \mid z)] - \text{KL}(q_\phi(z \mid x) \parallel p(z))
$$

To see why this is indeed a lower bound, rearrange the first equation:

$$
\mathcal{L}(\theta, \phi; x) = \log p_\theta(x) - \text{KL}(q_\phi(z \mid x) \parallel p_\theta(z \mid x))
$$

Since the KL divergence is always non-negative (that is, $\text{KL}(q \parallel p) \geq 0$ for any distributions $q$ and $p$, with equality only when $q = p$), we can write:

$$
\log p_\theta(x) = \mathcal{L}(\theta, \phi; x) + \text{KL}(q_\phi(z \mid x) \parallel p_\theta(z \mid x)) \geq \mathcal{L}(\theta, \phi; x)
$$

This shows that the ELBO is indeed a lower bound on the log-likelihood. The bound is tight when the KL divergence is zero, which happens when $q_\phi(z \mid x) = p_\theta(z \mid x)$, meaning our variational distribution exactly matches the true posterior.

Maximizing the ELBO simultaneously:
1. Maximizes the (approximate) marginal likelihood $\log p_\theta(x)$
2. Minimizes the approximation error $\text{KL}(q_\phi(z \mid x) \parallel p_\theta(z \mid x))$

The ELBO decomposes into two interpretable terms. The first term $\mathbb{E}_{z \sim q_\phi(z \mid x)} [\log p_\theta(x \mid z)]$ is the **reconstruction term**. It measures how well the decoder can reconstruct the input $x$ from latent variables sampled from the encoder's distribution. This encourages the model to be informative by ensuring that the latent representation captures enough information to reconstruct the data.

In practice, for a Gaussian likelihood $p_\theta(x \mid z) = \mathcal{N}(x \mid \mu_\theta(z), \sigma^2 I)$, where $\mu_\theta(z)$ is the mean predicted by the decoder network $f_\theta(z)$, we can derive the reconstruction term explicitly. The log-probability of a Gaussian is:

$$
\log p_\theta(x \mid z) = \log \mathcal{N}(x \mid \mu_\theta(z), \sigma^2 I) = -\frac{D}{2}\log(2\pi\sigma^2) - \frac{1}{2\sigma^2}\|x - \mu_\theta(z)\|^2
$$

where $D$ is the data dimensionality. The first term is a constant that does not depend on the parameters $\theta$ or $\phi$, so when optimizing, we can ignore it. Taking the expectation:

$$
\mathbb{E}_{z \sim q_\phi(z \mid x)} [\log p_\theta(x \mid z)] = -\frac{D}{2}\log(2\pi\sigma^2) - \frac{1}{2\sigma^2}\mathbb{E}_{z \sim q_\phi(z \mid x)} [\|x - \mu_\theta(z)\|^2]
$$

This is proportional to the negative mean squared error (MSE) between the input $x$ and the decoder's predicted mean $\mu_\theta(z)$:

$$
\mathbb{E}_{z \sim q_\phi(z \mid x)} [\log p_\theta(x \mid z)] \propto -\mathbb{E}_{z \sim q_\phi(z \mid x)} [\|x - \mu_\theta(z)\|^2]
$$

Thus maximizing the reconstruction term is equivalent to minimizing the MSE between inputs and reconstructions.

The second term $\text{KL}(q_\phi(z \mid x) \parallel p(z))$ is the **regularization term**. It measures how much the approximate posterior (encoder distribution) deviates from the prior. This term has several beneficial effects. It prevents the encoder from simply learning an arbitrary encoding by forcing the latent variables to follow a known, regular distribution (the prior). It promotes disentanglement by encouraging the latent dimensions to be approximately independent, since the prior is typically a factorized Gaussian. Finally, it enables generation by ensuring we can sample novel latent variables from the prior distribution $p(z) = \mathcal{N}(0, I)$ and decode them to get realistic data.

The balance between these two terms is crucial. The reconstruction term alone would lead to overfitting and pathological solutions where the encoder learns to ignore the prior, mapping each input to a unique but arbitrary point in latent space. The KL term alone would force the posterior to match the prior exactly, losing all information about the input. The VAE objective elegantly balances these competing objectives, learning a latent space that is both informative (captures data structure) and regular (follows a known prior).

For the special case where both $q_\phi(z \mid x) = \mathcal{N}(\mu, \sigma^2 I)$ and $p(z) = \mathcal{N}(0, I)$ are diagonal Gaussians, the KL divergence has a closed-form expression (see [Variational Inference](/garden/ml/bayesian/variationalinference/#elbo-for-bayesian-logistic-regression) for the derivation):

$$
\text{KL}(q_\phi(z \mid x) \parallel p(z)) = \frac{1}{2} \sum_{j=1}^d \left( \mu_j^2 + \sigma_j^2 - \log \sigma_j^2 - 1 \right)
$$

where $d$ is the dimensionality of the latent space. This closed-form expression provides significant computational benefits. Without it, we would need to approximate the KL divergence using Monte Carlo sampling: drawing multiple samples from $q_\phi(z \mid x)$ and computing the average of $\log q_\phi(z \mid x) - \log p(z)$. This would be slow (requiring many samples for accurate estimates), noisy (introducing high variance in gradients), and wasteful (sampling when an exact formula exists). The closed-form expression gives us the exact KL divergence value instantly using only the encoder outputs $\mu_\phi(x)$ and $\sigma_\phi(x)$, with no sampling or approximation error. This is one reason why Gaussian distributions are commonly chosen for both the prior and the variational distribution in VAEs.

### The Reparameterization Trick

To train the VAE, we need to compute gradients of the ELBO with respect to both the encoder parameters $\phi$ and decoder parameters $\theta$. The reconstruction term involves an expectation over the distribution $q_\phi(z \mid x)$, which itself depends on $\phi$. Computing gradients through this expectation is challenging because we cannot directly differentiate through a stochastic sampling operation.

The naive approach would be to approximate the expectation using Monte Carlo sampling:

$$
\mathbb{E}_{z \sim q_\phi(z \mid x)} [\log p_\theta(x \mid z)] \approx \frac{1}{L} \sum_{l=1}^L \log p_\theta(x \mid z^{(l)})
$$

where $z^{(l)} \sim q_\phi(z \mid x)$. However, we cannot backpropagate gradients through the sampling operation $z^{(l)} \sim q_\phi(z \mid x)$ because sampling is not a differentiable operation with respect to $\phi$.

The key insight is to use the **reparameterization trick** to reparameterize the random variable $z$ as a deterministic function of $\phi$, $x$, and an auxiliary noise variable $\epsilon$ that does not depend on $\phi$. For a Gaussian variational distribution $q_\phi(z \mid x) = \mathcal{N}(z \mid \mu_\phi(x), \sigma_\phi^2(x) I)$, we can express a sample $z$ as:

$$
z = \mu_\phi(x) + \sigma_\phi(x) \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)
$$

where $\odot$ denotes element-wise multiplication. We use element-wise (Hadamard) multiplication because all quantities are vectors: $\mu_\phi(x) \in \mathbb{R}^d$, $\sigma_\phi(x) \in \mathbb{R}^d$, and $\epsilon \in \mathbb{R}^d$ where $d$ is the latent space dimensionality. Each dimension $j$ of the latent space has its own mean $\mu_j$ and standard deviation $\sigma_j$, and we sample independent noise $\epsilon_j$ for each dimension. The element-wise multiplication $\sigma_\phi(x) \odot \epsilon$ scales the noise in each dimension by the corresponding standard deviation: $z_j = \mu_j + \sigma_j \cdot \epsilon_j$ for $j = 1, \ldots, d$. This reflects the fact that our variational distribution is a diagonal Gaussian where each dimension is independent with its own variance.

The noise $\epsilon$ is sampled from a fixed distribution that does not depend on the parameters $\phi$. The randomness in $z$ comes from $\epsilon$, but the transformation from $\epsilon$ to $z$ is deterministic and differentiable with respect to $\phi$.

Using this reparameterization, we can rewrite the expectation as:

$$
\mathbb{E}_{z \sim q_\phi(z \mid x)} [\log p_\theta(x \mid z)] = \mathbb{E}_{\epsilon \sim \mathcal{N}(0, I)} [\log p_\theta(x \mid \mu_\phi(x) + \sigma_\phi(x) \odot \epsilon)]
$$

Now the expectation is over a fixed distribution $\mathcal{N}(0, I)$, and we can approximate it by sampling $\epsilon$ and computing gradients through the deterministic function $g_\phi(x, \epsilon) = \mu_\phi(x) + \sigma_\phi(x) \odot \epsilon$. The Monte Carlo estimate becomes:

$$
\mathbb{E}_{\epsilon \sim \mathcal{N}(0, I)} [\log p_\theta(x \mid g_\phi(x, \epsilon))] \approx \frac{1}{L} \sum_{l=1}^L \log p_\theta(x \mid g_\phi(x, \epsilon^{(l)}))
$$

where $\epsilon^{(l)} \sim \mathcal{N}(0, I)$. This expression is fully differentiable with respect to both $\theta$ and $\phi$, allowing us to use standard backpropagation to compute gradients. In practice, the original Kingma and Welling paper found that a single sample ($L=1$) per data point is sufficient when using mini-batch SGD with reasonably large batch sizes. While more samples ($L > 1$) would provide a better estimate of the expectation, the mini-batch itself provides enough variance reduction to make single-sample estimates adequate for optimization. This makes the algorithm computationally efficient.

{{< callout type="example" title="Reparameterization in Practice" >}}
Consider encoding an image of a face. The encoder outputs $\mu_\phi(x) = [0.5, -0.3, 1.2]$ and $\sigma_\phi(x) = [0.1, 0.2, 0.15]$ for a three-dimensional latent space.

To sample a latent code, we first sample noise $\epsilon \sim \mathcal{N}(0, I)$, say $\epsilon = [0.8, -1.2, 0.3]$.

Then we compute: $z = \mu_\phi(x) + \sigma_\phi(x) \odot \epsilon = [0.5, -0.3, 1.2] + [0.1, 0.2, 0.15] \odot [0.8, -1.2, 0.3] = [0.58, -0.54, 1.245]$

This $z$ is then passed through the decoder. During backpropagation, gradients flow through the multiplication and addition operations back to the encoder parameters that produced $\mu_\phi(x)$ and $\sigma_\phi(x)$.
{{< /callout >}}

### Training

Training a VAE involves jointly optimizing the encoder and decoder parameters to maximize the ELBO over the training dataset. For a dataset $\mathcal{D} = \{x_1, \ldots, x_n\}$, the objective is:

$$
\max_{\theta, \phi} \sum_{i=1}^n \mathcal{L}(\theta, \phi; x_i)
$$

Using the decomposition of the ELBO and the reparameterization trick, the objective for a single data point becomes:

$$
\mathcal{L}(\theta, \phi; x) = \mathbb{E}_{\epsilon \sim \mathcal{N}(0, I)} [\log p_\theta(x \mid g_\phi(x, \epsilon))] - \text{KL}(q_\phi(z \mid x) \parallel p(z))
$$

In practice, we use stochastic gradient ascent (rather than descent) because we are maximizing the ELBO rather than minimizing a loss. For each mini-batch of data points $\{x^{(1)}, \ldots, x^{(M)}\}$ sampled from the dataset, we perform the following steps. For each data point $x^{(i)}$, we compute the encoder outputs $\mu_\phi(x^{(i)})$ and $\sigma_\phi(x^{(i)})$. We then sample noise $\epsilon^{(i)} \sim \mathcal{N}(0, I)$ and compute the latent code $z^{(i)} = \mu_\phi(x^{(i)}) + \sigma_\phi(x^{(i)}) \odot \epsilon^{(i)}$. Next, we compute the reconstruction log-likelihood $\log p_\theta(x^{(i)} \mid z^{(i)})$ and the KL divergence $\text{KL}(q_\phi(z \mid x^{(i)}) \parallel p(z))$ using the closed-form expression. After processing all data points in the mini-batch, we compute the average ELBO and take a gradient step to maximize it.

For the reconstruction term, the specific form of $\log p_\theta(x \mid z)$ depends on the type of data. For continuous data like images with pixel values in $[0, 1]$, a common choice is a Gaussian likelihood with identity covariance:

$$
\log p_\theta(x \mid z) = -\frac{1}{2} \|x - \mu_\theta(z)\|_2^2 + \text{const}
$$

This corresponds to a mean squared error (MSE) reconstruction loss.

### Sampling and Generation

Once trained, a VAE can be used in several ways. For **encoding** an existing data point, we pass it through the encoder to get the distribution parameters $\mu_\phi(x)$ and $\sigma_\phi(x)$, then sample $z \sim \mathcal{N}(\mu_\phi(x), \sigma_\phi^2(x) I)$ to get a latent code. We could also just use the mean $\mu_\phi(x)$ as a deterministic encoding.

For **reconstruction**, we encode the input to get $z$, then decode it through $\mu_\theta(z)$ to get the reconstruction $\hat{x}$. The reconstruction will generally not be identical to the input due to the information bottleneck and the stochasticity in the encoder.

For **generation** of new samples, this is where VAEs truly shine compared to traditional autoencoders. We sample a latent code from the prior $z \sim \mathcal{N}(0, I)$ and pass it through the decoder to get $\hat{x} = \mu_\theta(z)$. Since the training process ensures that the latent space is regularized to match this prior (through the KL term), samples from the prior should decode to plausible data points. This is a key advantage: we can generate infinite new samples without ever seeing them in the training data.

For **interpolation** between two data points, we can encode them to get latent codes $z_1$ and $z_2$, then interpolate in latent space: $z_t = (1-t) z_1 + t z_2$ for $t \in [0, 1]$. Decoding these interpolated latent codes produces a smooth transition between the original data points. The continuous and structured nature of the latent space (enforced by the KL regularization) allows for smooth interpolation. Unlike traditional autoencoders where the latent space can have "holes" or discontinuities, VAE latent spaces are dense and well-behaved.

The ability to generate new samples by sampling from the prior is the fundamental capability that distinguishes VAEs from traditional autoencoders. This generative capability comes from the probabilistic formulation and the KL regularization term that ensures the latent space has a known, regular structure.

### Controlling Generation

While basic generation involves sampling $z \sim \mathcal{N}(0, I)$, we can exert some control over what is generated. One approach is **conditional generation**. If we train a **conditional VAE** where both the encoder $q_\phi(z \mid x, c)$ and decoder $p_\theta(x \mid z, c)$ are conditioned on some additional information $c$ (like a class label), we can generate samples from a specific class by sampling $z \sim \mathcal{N}(0, I)$ and decoding with $\mu_\theta(z, c)$ where $c$ is the desired class.

Another approach is **latent space manipulation**. If the latent dimensions are disentangled, we can modify specific dimensions to change specific attributes. For example, in a VAE trained on faces, we might find that dimension 3 controls "smiling" while dimension 7 controls "age". We can then encode a face, modify dimension 3, and decode to get a smiling version of the same face. However, standard VAEs do not guarantee disentanglement. This limitation led to the development of $\beta$-VAE.

### beta-VAE

While the standard VAE objective encourages some degree of disentanglement through the KL regularization, it does not guarantee that different latent dimensions will correspond to independent factors of variation. **$\beta$-VAE** addresses this by modifying the ELBO to place stronger emphasis on the KL term:

$$
\mathcal{L}_{\beta}(\theta, \phi; x) = \mathbb{E}_{z \sim q_\phi(z \mid x)} [\log p_\theta(x \mid z)] - \beta \cdot \text{KL}(q_\phi(z \mid x) \parallel p(z))
$$

where $\beta > 1$ applies stronger regularization. Higher $\beta$ values encourage the approximate posterior $q_\phi(z \mid x)$ to be closer to the prior $p(z) = \mathcal{N}(0, I)$, which is factorized across dimensions. This encourages the latent dimensions to be more independent, leading to better disentanglement.

The trade-off is that higher $\beta$ values reduce reconstruction quality, as the model is forced to compress information more aggressively. The choice of $\beta$ depends on the application. For generation quality, $\beta = 1$ (standard VAE) may be best. For learning interpretable representations where we want to manipulate individual factors, higher $\beta$ values (e.g., $\beta = 4$ or more) may be preferable despite the reconstruction quality loss.
