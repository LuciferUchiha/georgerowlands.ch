---
title: Autoencoders
type: docs
weight: 2
---

Deep learning excels at learning meaningful representations of data. Neural networks can be viewed as compositions of encoders that extract features and estimators that make predictions from those features. **Autoencoders** provide a framework for learning these representations in an unsupervised manner, without requiring labeled data. The core idea is deceptively simple: train a network to reconstruct its input through a bottleneck which forces it to learn a compressed representation.

More formally, an autoencoder is a neural network architecture consisting of two main components: An **encoder** $g_\phi: \mathbb{R}^D \to \mathbb{R}^d$ which maps high-dimensional input data $\mathbf{x} \in \mathbb{R}^D$ to a lower-dimensional latent representation $\mathbf{z} \in \mathbb{R}^d$ where $d \ll D$ and a **decoder** $f_\theta: \mathbb{R}^d \to \mathbb{R}^D$ which attempts to reconstruct the original input from the compressed representation.

$$
\mathbf{z} = g_\phi(\mathbf{x}), \quad \hat{\mathbf{x}} = f_\theta(\mathbf{z}) = f_\theta(g_\phi(\mathbf{x}))
$$

{{< figure
    src="/images/ml/autoencoder.png"
    alt="Traditional autoencoder architecture with encoder, latent representation, and decoder."
    caption="An autoencoder compresses input through an encoder to a latent representation, then reconstructs it through a decoder."
>}}

The network is trained to minimize the reconstruction error between the input $\mathbf{x}$ and its reconstruction $\hat{\mathbf{x}}$. This is typically the mean squared error but can be any appropriate loss function depending on the data type and task.

$$
\mathcal{L}(\theta, \phi) = \frac{1}{n}\sum_{i=1}^n \|\mathbf{x}_i - f_\theta(g_\phi(\mathbf{x}_i))\|^2
$$

The bottleneck structure (where $d \ll D$) forces the encoder to learn a compressed representation that captures the most important features of the data to then be able to reconstruct it as accurately as possible. This is similar to [principal component analysis (PCA)](/garden/maths/linearAlgebra/pca) which reduces dimensionality using linear projections, but in addition autoencoders can learn non-linear representations by adding non-linear activation functions.

## Linear Autoencoders

The general autoencoder above is something of a black box. We know it compresses data through a bottleneck, but with arbitrary non-linear maps $g_\phi$ and $f_\theta$ it is hard to understand precisely what representation it learns. Instead we first study the simplest possible case, where the encoder and decoder are linear maps. It turns out that Linear autoencoders have an elegant solution: spoilers, the optimal linear autoencoder performs an orthogonal projection onto the subspace of maximal variance, which is exactly what [principal component analysis](/garden/maths/linearAlgebra/pca) does.

To make a "good reconstruction" precise we need to measure the discrepancy between an input $\mathbf{x}$ and its reconstruction $\hat{\mathbf{x}}$. This is the job of the **distortion measure** (or simply **loss**), a function that scores a single reconstruction

$$
\mathcal{L}: \mathbb{R}^D \times \mathbb{R}^D \to \mathbb{R}, \quad (\mathbf{x}, \hat{\mathbf{x}}) \mapsto \mathcal{L}(\mathbf{x}, \hat{\mathbf{x}})
$$

As mentioned before in the absence of domain-specific knowledge the default choice is the **quadratic loss**

$$
\mathcal{L}(\mathbf{x}, \hat{\mathbf{x}}) = \frac{1}{2}\|\mathbf{x} - \hat{\mathbf{x}}\|^2
$$

It is important to keep in mind that this is a convenient choice rather than the only one. For other data types other distortion measures are more natural. For example, for binary data, one would probably use the log loss (cross entropy) instead, just like in [logistic regression](/garden/ml/linearClassificationlogisticRegression).

The loss only scores a single example. What we actually care about is the average loss, often also called the **risk**. To have an average we need to know the distribution over which to average. We commonly assume that the data is drawn from some fixed but unknown distribution $p$, written $\mathbf{x} \sim p$, and our sample $\mathbf{x}_1, \ldots, \mathbf{x}_n$ is drawn independently from it.

The reconstruction of a point is $\hat{\mathbf{x}} = f_\theta(g_\phi(\mathbf{x}))$, the encoder followed by the decoder. We call this composition $f_\theta \circ g_\phi$ the **reconstruction map**. Ideally we would want $f_\theta \circ g_\phi = \text{id}$, the identity, but this is unachievable through a bottleneck of dimension $d < D$ unless the data happens to lie exactly on a $d$-dimensional subspace.

What we would really like to minimize is the average loss over the entire data distribution. This is the **expected risk**, also called the population or true risk:

$$
\mathcal{R} = \mathbb{E}_{\mathbf{x} \sim p}\big[\mathcal{L}(\mathbf{x}, f_\theta(g_\phi(\mathbf{x})))\big]
$$
 
We cannot actually evaluate this expected risk because we do not know the true distribution $p$. In practice we only have the sample, so the quantity we can actually minimize is the **empirical risk**, the average loss over the data we have:

$$
\mathcal{R}_{\text{emp}} = \mathbb{E}_{\mathbf{x} \sim S}\big[\mathcal{L}(\mathbf{x}, f_\theta(g_\phi(\mathbf{x})))\big] = \frac{1}{n}\sum_{i=1}^n \mathcal{L}\big(\mathbf{x}_i, f_\theta(g_\phi(\mathbf{x}_i))\big)
$$

where $\mathbb{E}_{\mathbf{x} \sim S}$ denotes the average over the sample $S = \{\mathbf{x}_1, \ldots, \mathbf{x}_n\}$. The empirical risk is a [Monte Carlo](/garden/maths/probabilityStatistics/simulating) estimate of the expected risk, converging to it as the sample size $n$ grows.

### Linear Encoder and Decoder

With our measure of reconstruction quality in place, we can now define the linear autoencoder. We set the encoder and decoder to be linear maps,

$$
g_\phi: \mathbb{R}^D \to \mathbb{R}^d, \quad \mathbf{z} = \mathbf{W}\mathbf{x} \qquad f_\theta: \mathbb{R}^d \to \mathbb{R}^D, \quad \hat{\mathbf{x}} = \mathbf{V}\mathbf{z}
$$

with weight matrices $\mathbf{W} \in \mathbb{R}^{d \times D}$ and $\mathbf{V} \in \mathbb{R}^{D \times d}$ as the only parameters. The reconstruction map is their composition, which by associativity of matrix multiplication is again a single matrix $\mathbf{P}$, the **reconstruction matrix**,

$$
(f_\theta \circ g_\phi)(\mathbf{x}) = \mathbf{V}(\mathbf{W}\mathbf{x}) = (\mathbf{V}\mathbf{W})\mathbf{x} = \mathbf{P}\mathbf{x}, \qquad \mathbf{P} = \mathbf{V}\mathbf{W} \in \mathbb{R}^{D \times D}
$$

With the quadratic loss, the linear autoencoder objective is the expected risk over the data distribution:

$$
\mathcal{R}(\mathbf{W}, \mathbf{V}) = \mathcal{R}(\mathbf{P}) = \mathbb{E}_{\mathbf{x} \sim p}\left[\frac{1}{2}\|\mathbf{x} - \mathbf{P}\mathbf{x}\|^2\right]
$$

Working with the single matrix $\mathbf{P} = \mathbf{V}\mathbf{W}$ is convenient, and it also explains why there is no point in stacking several linear layers so we can just focus on the single linear layer. This is because a two-layer linear encoder is itself just a matrix $\mathbf{W}_2 \mathbf{W}_1$, and a two-layer linear decoder just a matrix $\mathbf{V}_1 \mathbf{V}_2$, so by associativity of matrix multiplication the whole reconstruction map collapses back to a single matrix,

$$
\underbrace{(\mathbf{V}_1 \mathbf{V}_2)}_{\text{decoder }\mathbf{V}} \underbrace{(\mathbf{W}_2 \mathbf{W}_1)}_{\text{encoder }\mathbf{W}} = \underbrace{\mathbf{V}_1 \mathbf{V}_2 \mathbf{W}_2 \mathbf{W}_1}_{\mathbf{P}}
$$

Depth therefore buys no extra expressivity in the linear case, and nothing is lost by using a single layer.

### Affine vs Linear Maps

Recall that our linear encoder and decoder have the form $\mathbf{z} = \mathbf{W}\mathbf{x}$ and $\hat{\mathbf{x}} = \mathbf{V}\mathbf{z}$, with no constant term. A natural question is whether we are giving up representational power by using these purely linear maps rather than **affine** ones, $\mathbf{z} = \mathbf{W}\mathbf{x} + \mathbf{a}$ and $\hat{\mathbf{x}} = \mathbf{V}\mathbf{z} + \mathbf{b}$, that also carry bias terms $\mathbf{a}$ and $\mathbf{b}$. It turns out that for centered data with the quadratic loss affine maps reduce to linear maps, so we are not losing anything by restricting ourselves to linear maps.

Centering the data simply means that we subtract the sample mean from each data point, so replacing each $\mathbf{x}$ by $\mathbf{x} - \mathbb{E}[\mathbf{x}]$ so that $\mathbb{E}[\mathbf{x}] = 0$, hence the data is "centered" around the origin.

{{< callout type="proof" title="Affine Reconstruction Maps Reduce to Linear" >}}
We can prove this directly. Let the affine encoder and decoder be $\mathbf{z} = \mathbf{W}\mathbf{x} + \mathbf{a}$ and $\hat{\mathbf{x}} = \mathbf{V}\mathbf{z} + \mathbf{b}$. Their composition is then again affine,

$$
\hat{\mathbf{x}} = \mathbf{V}(\mathbf{W}\mathbf{x} + \mathbf{a}) + \mathbf{b} = \mathbf{V}\mathbf{W}\mathbf{x} + \mathbf{V}\mathbf{a} + \mathbf{b} = \underbrace{\mathbf{V}\mathbf{W}}_{\mathbf{P}}\mathbf{x} + \mathbf{c}, \qquad \mathbf{c} = \mathbf{b} + \mathbf{V}\mathbf{a}
$$

so it is a linear map $\mathbf{P}$ plus a constant offset $\mathbf{c}$. To find the risk of this affine map we expand the squared error, grouping it as $(\mathbf{x} - \mathbf{P}\mathbf{x}) - \mathbf{c}$,

$$
\begin{aligned}
\|\mathbf{x} - (\mathbf{P}\mathbf{x} + \mathbf{c})\|^2 &= \|(\mathbf{x} - \mathbf{P}\mathbf{x}) - \mathbf{c}\|^2 \\
&= \|\mathbf{x} - \mathbf{P}\mathbf{x}\|^2 - 2\,\mathbf{c}^T(\mathbf{x} - \mathbf{P}\mathbf{x}) + \|\mathbf{c}\|^2
\end{aligned}
$$

Now we take the expectation. The offset $\mathbf{c}$ is a constant, and the projection $\mathbf{P}$ is a fixed matrix, so by linearity of expectation it moves outside, $\mathbb{E}[\mathbf{P}\mathbf{x}] = \mathbf{P}\,\mathbb{E}[\mathbf{x}]$. The cross term therefore becomes $-2\,\mathbf{c}^T(\mathbb{E}[\mathbf{x}] - \mathbf{P}\,\mathbb{E}[\mathbf{x}])$, giving

$$
\mathbb{E}\big[\|\mathbf{x} - (\mathbf{P}\mathbf{x} + \mathbf{c})\|^2\big] = \mathbb{E}\big[\|\mathbf{x} - \mathbf{P}\mathbf{x}\|^2\big] + \|\mathbf{c}\|^2 - 2\,\mathbf{c}^T\big(\mathbb{E}[\mathbf{x}] - \mathbf{P}\,\mathbb{E}[\mathbf{x}]\big)
$$

For centered data $\mathbb{E}[\mathbf{x}] = \mathbf{0}$, so the entire cross term vanishes and we are left with

$$
\mathbb{E}\big[\|\mathbf{x} - (\mathbf{P}\mathbf{x} + \mathbf{c})\|^2\big] = \mathbb{E}\big[\|\mathbf{x} - \mathbf{P}\mathbf{x}\|^2\big] + \|\mathbf{c}\|^2
$$

Since $\|\mathbf{c}\|^2 \ge 0$, the affine risk is at least the linear risk,

$$
\mathbb{E}\big[\|\mathbf{x} - (\mathbf{P}\mathbf{x} + \mathbf{c})\|^2\big] \ \geq\ \mathbb{E}\big[\|\mathbf{x} - \mathbf{P}\mathbf{x}\|^2\big]
$$

with equality if and only if $\mathbf{c} = \mathbf{0}$. So the optimal offset is zero and the affine map is no better than the linear map $\mathbf{P}$.
{{< /callout >}}

In other words, once we center the data, the bias terms are redundant. This is why we can focus on the linear reconstruction matrix $\mathbf{P}$ on its own.

It is worth being careful about what exactly this says. The centering argument above is the real result: for centered data and the squared loss, the best affine reconstruction has zero offset, so it gives no advantage over a linear one. This is a statement about the optimum, not a claim that affine maps are the same thing as linear maps!

An affine map in $D$ dimensions is in fact strictly more general than a linear map in $D$ dimensions, since it has the extra bias. This can also be seen by that any affine map can be rewritten as a linear map by a trick called **homogeneous coordinates**. For this we augment each input with a constant extra coordinate equal to $1$, and append the bias as an extra column of the weight matrix:

$$
\tilde{\mathbf{x}} = \begin{bmatrix} \mathbf{x} \\ 1 \end{bmatrix} \qquad \tilde{\mathbf{W}} = \begin{bmatrix} \mathbf{W} & \mathbf{a} \end{bmatrix} \qquad \tilde{\mathbf{W}}\tilde{\mathbf{x}} = \mathbf{W}\mathbf{x} + \mathbf{a}
$$

So the affine map $\mathbf{W}\mathbf{x} + \mathbf{a}$ in $D$ dimensions becomes a linear map $\tilde{\mathbf{W}}\tilde{\mathbf{x}}$ in $D + 1$ dimensions. This is a transformation, not an equivalence of power. The extra dimension is exactly what carries the bias, so a linear map in $D + 1$ dimensions can do strictly more than a linear map in $D$ dimensions. The augmentation just relabels an affine map as a higher-dimensional linear one. This is exactly the same trick used to fold biases into weight matrices when discussing neural networks.

### Uniqueness of the Solution

A further question we can ask ourselves is about the uniqueness of the solution. Our goal is to minimize $\mathcal{R}(\mathbf{P}) = \frac{1}{2}\mathbb{E}\big[\|\mathbf{x} - \mathbf{P}\mathbf{x}\|^2\big]$ over reconstruction matrices $\mathbf{P} = \mathbf{V}\mathbf{W}$. From this formulation we can ask two questions about uniqueness:

1. Is the optimal reconstruction matrix $\mathbf{P}$ unique?
2. Is the factorization of $\mathbf{P}$ into weight matrices $\mathbf{W}$ and $\mathbf{V}$ unique?

We cannot answer the first question yet, since it depends on what the optimal $\mathbf{P}$ actually is. We will return to it once the [PCA theorem](#the-pca-theorem) has identified the optimum, where we will see that the optimal $\mathbf{P}$ is unique whenever the $d$-th and $(d+1)$-th largest variances differ.

The answer to the second question is no, there are infinitely many different pairs of weight matrices that give the same reconstruction matrix $\mathbf{P}$. We can show this quickly by noting that for any invertible matrix $\mathbf{A} \in \mathbb{R}^{d \times d}$, we can insert $\mathbf{A}\mathbf{A}^{-1} = \mathbf{I}$ in the middle of the product $\mathbf{V}\mathbf{W}$ without changing it:

$$
\mathbf{V}\mathbf{W} = \mathbf{V}(\mathbf{A} \mathbf{A}^{-1})\mathbf{W} = \underbrace{(\mathbf{V}\mathbf{A})}_{= \mathbf{V}'}\underbrace{(\mathbf{A}^{-1}\mathbf{W})}_{= \mathbf{W}'}
$$

the new weight matrices $\mathbf{V}'$ and $\mathbf{W}'$ give exactly the same reconstruction matrix $\mathbf{P} = \mathbf{V}'\mathbf{W}' = \mathbf{V}\mathbf{W}$. 

The practical consequence is that the weight matrices are **not unique**, and one should therefore not over-interpret the individual columns of $\mathbf{V}$ or rows of $\mathbf{W}$ that a trained autoencoder produces. They are just one arbitrary representative out of a whole family.

### The Rank Constraint

We've already seen that linear and affine maps have the same expressivity for centered data, but what is the actual expressivity of a linear autoencoder? Specifically how much can a linear autoencoders hope to reconstruct?

To answer this we know that every reconstruction $\hat{\mathbf{x}} = \mathbf{P}\mathbf{x}$ lies in the image of $\mathbf{P}$, and the dimension of that image is the [**rank**](/garden/maths/linearAlgebra/matrixRanks) of $\mathbf{P}$. The rank therefore measures how large a slice of the input space the autoencoder can reproduce, so the width-$d$ bottleneck will limit it by capping this rank.

Recall that the rank is the dimension of the image, $\text{rank}(\mathbf{A}) = \dim(\text{im}(\mathbf{A}))$, and that row rank equals column rank, so $\text{rank}(\mathbf{A}) = \text{rank}(\mathbf{A}^T)$ (see the [column space](/garden/maths/linearAlgebra/vectorSpaces#column-space)). Since $\mathbf{P} = \mathbf{V}\mathbf{W}$ with $\mathbf{W} \in \mathbb{R}^{d \times D}$ and $\mathbf{V} \in \mathbb{R}^{D \times d}$, each weight matrix has rank at most $d$ (its smaller dimension), and the product cannot exceed this inner dimension:

$$
\text{rank}(\mathbf{P}) = \min\{\text{rank}(\mathbf{V}), \text{rank}(\mathbf{W})\} \le \min\{D, d\} = d
$$

So the bottleneck layer is exactly a rank constraint $\text{rank}(\mathbf{P}) \le d$. Therefore we can rewrite the optimization over weight matrices $\mathbf{W}$ and $\mathbf{V}$ as an optimization directly over reconstruction matrices $\mathbf{P}$, with the rank constraint explicitly written:

$$
\min_{\mathbf{P}:\ \text{rank}(\mathbf{P}) \le d} \mathcal{R}(\mathbf{P}), \qquad \mathcal{R}(\mathbf{P}) = \frac{1}{2}\mathbb{E}\big[\|\mathbf{x} - \mathbf{P}\mathbf{x}\|^2\big]
$$

An important consequence of $\mathbf{P}$ being linear and having rank at most $d$, is that its image $U = \text{im}(\mathbf{P})$ is a linear subspace of $\mathbb{R}^D$ of dimension at most $d$, and every reconstruction lands in $U$. This splits the optimization into two clearly defined subproblems:

1. **The best map for a fixed subspace:** given a subspace $U$, which map $\mathbf{P}$ with image $U$ reconstructs the data best?
2. **The best subspace:** which subspace $U$ should we project onto in the first place?

### The Best Map for a Fixed Subspace

We start with the first subproblem by fixing a subspace $U \subseteq \mathbb{R}^D$ with $\dim(U) \le d$ and ask which map with image $U$ reconstructs the data best. The risk averages the per-point squared errors $\|\mathbf{x} - \mathbf{P}\mathbf{x}\|^2$, and since the reconstruction $\mathbf{P}\mathbf{x}$ is free to be any point of $U$, the best we can do for each $\mathbf{x}$ is to map it to the point of $U$ closest to it, which we denote $\Pi_U(\mathbf{x})$. This closest point is exactly the **projection** of $\mathbf{x}$ onto $U$ in the least-squares sense, the construction developed in the [projections and least squares note](/garden/maths/linearAlgebra/projectionsLS):

$$
\Pi_U(\mathbf{x}) = \arg\min_{\mathbf{x}' \in U} \|\mathbf{x} - \mathbf{x}'\|
$$

Projecting onto a linear subspace is a classic, well-understood operation, so before optimizing anything let us recall what a projection is and collect the properties we will use.

{{< callout type="info" title="Projection" >}}
A linear map $\mathbf{P}: \mathbb{R}^D \to \mathbb{R}^D$ is a **projection** onto $U$ if for every $\mathbf{x}$:

1. **Projection property:** $\mathbf{P}\mathbf{x} \in U$, the output lands in $U$.
2. **Idempotency:** $\mathbf{P}(\mathbf{P}\mathbf{x}) = \mathbf{P}\mathbf{x}$, projecting an already-projected point does nothing.

It is an **orthogonal** projection if, in addition, the error $\mathbf{x} - \mathbf{P}\mathbf{x}$ is perpendicular to $U$ for every $\mathbf{x}$. Otherwise it is **oblique**.
{{< /callout >}}

Our closest-point map $\Pi_U$ already meets the first two requirements: its output lies in $U$ by construction, and applying it again changes nothing, since a point already in $U$ is its own closest point. So it is a projection. Two things remain to pin down:

- **Existence and uniqueness:** because $U$ is a linear subspace, the closest point is guaranteed to exist and to be unique, so $\Pi_U$ really is a well-defined function.

{{< callout type="proof" title="The closest point exists and is unique" >}}
We prove the two parts in turn.

- **Existence:** the subspace $U$ is a closed set, and the squared distance $\mathbf{x}' \mapsto \|\mathbf{x} - \mathbf{x}'\|^2$ is continuous and grows without bound as $\|\mathbf{x}'\| \to \infty$. A continuous function that grows without bound attains its minimum over a closed set, so a closest point exists.
- **Uniqueness:** suppose two points $\mathbf{x}_1', \mathbf{x}_2' \in U$ both achieve the minimum distance $r = \|\mathbf{x} - \mathbf{x}_1'\| = \|\mathbf{x} - \mathbf{x}_2'\|$. Their midpoint $\tfrac{1}{2}(\mathbf{x}_1' + \mathbf{x}_2')$ also lies in $U$, because $U$ is a subspace. Expanding the squared distance from $\mathbf{x}$ to this midpoint,

$$
\Big\|\mathbf{x} - \tfrac{\mathbf{x}_1' + \mathbf{x}_2'}{2}\Big\|^2 = r^2 - \tfrac{1}{4}\|\mathbf{x}_1' - \mathbf{x}_2'\|^2
$$

If $\mathbf{x}_1' \neq \mathbf{x}_2'$ the right-hand side is strictly less than $r^2$, so the midpoint would be a point of $U$ strictly closer to $\mathbf{x}$, contradicting that $r$ is the minimum. Hence $\mathbf{x}_1' = \mathbf{x}_2'$ and the closest point is unique.
{{< /callout >}}

- **Orthogonality:** what singles this projection out among all maps with image $U$ is how the error sits. Because $\Pi_U(\mathbf{x})$ is the closest point of $U$, the error $\mathbf{x} - \Pi_U(\mathbf{x})$ is perpendicular to $U$, so $\Pi_U$ is an orthogonal projection. This is the same fact the projections note derives for the least-squares problem.

{{< callout type="proof" title="The error is perpendicular to the subspace" >}}
We argue by contradiction. Every vector of $\mathbb{R}^D$ splits uniquely into a part in $U$ and a part in its orthogonal complement $U^\perp$, the [orthogonal decomposition](/garden/maths/linearAlgebra/projectionsLS#projecting-onto-the-orthogonal-complement) $\mathbb{R}^D = U \oplus U^\perp$. Apply this to the error $\mathbf{e} = \Pi_U(\mathbf{x}) - \mathbf{x}$,

$$
\mathbf{e} = \mathbf{u} + \mathbf{u}^\perp, \qquad \mathbf{u} \in U, \quad \mathbf{u}^\perp \in U^\perp
$$

and suppose its in-subspace part is nonzero, $\mathbf{u} \neq 0$, which is exactly the statement that the error is not perpendicular to $U$. Now look at the point $\Pi_U(\mathbf{x}) - \mathbf{u}$, which still lies in $U$. Its error against $\mathbf{x}$ is just $\mathbf{u}^\perp$, since

$$
(\Pi_U(\mathbf{x}) - \mathbf{u}) - \mathbf{x} = \mathbf{e} - \mathbf{u} = \mathbf{u}^\perp
$$

and by the Pythagorean theorem the original error is strictly longer,

$$
\|\mathbf{e}\|^2 = \|\mathbf{u}\|^2 + \|\mathbf{u}^\perp\|^2 > \|\mathbf{u}^\perp\|^2
$$

So $\Pi_U(\mathbf{x}) - \mathbf{u}$ is a point of $U$ strictly closer to $\mathbf{x}$ than $\Pi_U(\mathbf{x})$ itself, contradicting that $\Pi_U(\mathbf{x})$ is the closest point. Hence $\mathbf{u} = 0$ and the error $\mathbf{x} - \Pi_U(\mathbf{x})$ lies in $U^\perp$.
{{< /callout >}}

Another important property is **linearity**, since it is what lets us represent $\Pi_U$ by a matrix. To prove linearity, we need to show both **homogeneity** and **additivity**. Homogeneity is immediate, as scaling the input scales its closest point by the same factor,

$$
\Pi_U(\alpha \mathbf{x}) = \arg\min_{\mathbf{x}' \in U}\|\alpha \mathbf{x} - \mathbf{x}'\| = \alpha \arg\min_{\mathbf{x}'' \in U}\|\mathbf{x} - \mathbf{x}''\| = \alpha\, \Pi_U(\mathbf{x})
$$

Additivity is where orthogonality earns its keep. The candidate $\Pi_U(\mathbf{x}) + \Pi_U(\mathbf{y})$ lies in $U$, and its error against $\mathbf{x} + \mathbf{y}$,

$$
(\mathbf{x} + \mathbf{y}) - \big(\Pi_U(\mathbf{x}) + \Pi_U(\mathbf{y})\big) = \big(\mathbf{x} - \Pi_U(\mathbf{x})\big) + \big(\mathbf{y} - \Pi_U(\mathbf{y})\big)
$$

is a sum of two vectors perpendicular to $U$, hence itself perpendicular to $U$. A point of $U$ whose error is perpendicular to $U$ is the closest point, by the same Pythagorean argument as above, so $\Pi_U(\mathbf{x} + \mathbf{y}) = \Pi_U(\mathbf{x}) + \Pi_U(\mathbf{y})$.

Now that $\Pi_U$ is linear we can represent it by a matrix, so from here on we write $\Pi_U(\mathbf{x}) = \mathbf{P}\mathbf{x}$. We already know this matrix is idempotent and has image $U$. Its orthogonality also brings a key property: a projection is orthogonal if and only if its matrix is **symmetric**, $\mathbf{P} = \mathbf{P}^T$.

{{< callout type="proof" title="A projection is orthogonal exactly when it is symmetric" >}}
By definition, $\mathbf{P}$ is an orthogonal projection when the error $\mathbf{x} - \mathbf{P}\mathbf{x}$ is perpendicular to the image $U = \text{im}(\mathbf{P})$ for every $\mathbf{x}$. Since $U$ is precisely the image of $\mathbf{P}$, every vector of $U$ is of the form $\mathbf{P}\mathbf{y}$ for some $\mathbf{y}$ (indeed a point already in $U$ satisfies $\mathbf{P}\mathbf{y} = \mathbf{y}$ by idempotency). Perpendicularity of the error to all of $U$ is therefore the same as

$$
(\mathbf{x} - \mathbf{P}\mathbf{x})^T \mathbf{P}\mathbf{y} = 0 \quad \text{for all } \mathbf{x}, \mathbf{y}
$$

- **($\Rightarrow$) Orthogonal implies symmetric:** multiply out the left-hand side and pull $\mathbf{x}$ and $\mathbf{y}$ outside,

$$
0 = (\mathbf{x} - \mathbf{P}\mathbf{x})^T \mathbf{P}\mathbf{y} = \mathbf{x}^T\mathbf{P}\mathbf{y} - \mathbf{x}^T\mathbf{P}^T\mathbf{P}\mathbf{y} = \mathbf{x}^T\big(\mathbf{P} - \mathbf{P}^T\mathbf{P}\big)\mathbf{y}
$$

Since this vanishes for every $\mathbf{x}$ and $\mathbf{y}$, the matrix in the middle must be zero, so $\mathbf{P} = \mathbf{P}^T\mathbf{P}$. The right-hand side is symmetric, because $(\mathbf{P}^T\mathbf{P})^T = \mathbf{P}^T\mathbf{P}$, and therefore so is $\mathbf{P}$, giving $\mathbf{P} = \mathbf{P}^T$.

- **($\Leftarrow$) Symmetric implies orthogonal:** now assume $\mathbf{P}^2 = \mathbf{P}$ and $\mathbf{P} = \mathbf{P}^T$. For any $\mathbf{x}, \mathbf{y}$,

$$
(\mathbf{x} - \mathbf{P}\mathbf{x})^T \mathbf{P}\mathbf{y} = \mathbf{x}^T(\mathbf{I} - \mathbf{P})^T \mathbf{P}\mathbf{y} = \mathbf{x}^T(\mathbf{I} - \mathbf{P}) \mathbf{P}\mathbf{y} = \mathbf{x}^T(\mathbf{P} - \mathbf{P}^2)\mathbf{y} = 0
$$

using $\mathbf{P}^T = \mathbf{P}$ in the second step and $\mathbf{P}^2 = \mathbf{P}$ in the last. So the error is perpendicular to the image, and $\mathbf{P}$ is an orthogonal projection.
{{< /callout >}}

Collecting the properties, the best map for a fixed subspace $U$ is the orthogonal projection onto $U$ which is a matrix of rank $\dim(U)$ that is both

$$
\mathbf{P}^2 = \mathbf{P} \quad (\text{idempotent}), \qquad \mathbf{P} = \mathbf{P}^T \quad (\text{symmetric})
$$

So, with its subspace fixed, the optimal linear autoencoder is nothing more than this projection. All that is left is to find the matrix $\mathbf{P}$, which we build up from the simplest case.

- **A single line:** the simplest subspace is a line. From the [projections note](/garden/maths/linearAlgebra/projectionsLS), the orthogonal projection of $\mathbf{x}$ onto the line spanned by a nonzero vector $\mathbf{a}$ is

$$
\mathbf{P}\mathbf{x} = \frac{\mathbf{a}\mathbf{a}^T}{\mathbf{a}^T\mathbf{a}}\,\mathbf{x}, \qquad \mathbf{P} = \frac{\mathbf{a}\mathbf{a}^T}{\mathbf{a}^T\mathbf{a}}
$$

{{< figure
    src="/images/maths/vectorProjection.png"
    alt="Orthogonal projection of a vector onto a line, with the error vector meeting the line at a right angle."
    caption="Projecting $\mathbf{x}$ onto the line spanned by $\mathbf{a}$. The projection $\mathbf{P}\mathbf{x}$ is the closest point on the line and the error is perpendicular to it. The image is from the projections note, where $\mathbf{x}$ is written $\mathbf{b}$ and the projection $\mathbf{p}$."
>}}

There is no need to carry the normalization $\mathbf{a}^T\mathbf{a}$ around, so we usually rescale the direction to a unit vector $\mathbf{u}$. Then $\|\mathbf{u}\|^2 = \mathbf{u}^T\mathbf{u} = 1$, the denominator vanishes, and the projection matrix is the clean outer product

$$
\mathbf{P} = \mathbf{u}\mathbf{u}^T, \qquad \mathbf{P}\mathbf{x} = \mathbf{u}\mathbf{u}^T\mathbf{x} = (\mathbf{u}^T\mathbf{x})\, \mathbf{u}
$$

So the unit vector is simply the convenient special case. The coefficient $\mathbf{u}^T\mathbf{x}$ is the signed length of $\mathbf{x}$ along $\mathbf{u}$, and $\|\mathbf{P}\mathbf{x}\| = |\mathbf{u}^T\mathbf{x}| \le \|\mathbf{x}\|$, so a projection never lengthens a vector.

- **A general basis:** a $d$-dimensional subspace works the same way, only the normalization becomes a matrix. Collect a basis $\mathbf{v}_1, \ldots, \mathbf{v}_d$ of $U$ as the columns of the decoder $\mathbf{V}$, so that $U = \text{span}(\mathbf{V})$ and every point of $U$ is $\mathbf{V}\mathbf{c}$ for some coordinate vector $\mathbf{c} \in \mathbb{R}^d$. The projection of $\mathbf{x}$ is the point $\mathbf{V}\mathbf{c}$ closest to $\mathbf{x}$, so $\mathbf{c}$ minimizes $\|\mathbf{x} - \mathbf{V}\mathbf{c}\|^2$. Setting the gradient to zero gives the **normal equations**, the same step as in [projections note](/garden/maths/linearAlgebra/projectionsLS#projections-in-higher-dimensions):

$$
\begin{aligned}
\mathbf{V}^T(\mathbf{x} - \mathbf{V}\mathbf{c}) = 0
&\implies \mathbf{V}^T\mathbf{V}\mathbf{c} = \mathbf{V}^T\mathbf{x} \\
&\implies \mathbf{c} = (\mathbf{V}^T\mathbf{V})^{-1}\mathbf{V}^T\mathbf{x}
\end{aligned}
$$

Putting this together, the decoder maps these coordinates back to the data space, $\mathbf{P}\mathbf{x} = \mathbf{V}\mathbf{c}$, which gives the projection matrix

$$
\mathbf{P} = \mathbf{V}(\mathbf{V}^T\mathbf{V})^{-1}\mathbf{V}^T = \mathbf{V}\mathbf{V}^+, \qquad \mathbf{V}^+ = (\mathbf{V}^T\mathbf{V})^{-1}\mathbf{V}^T
$$

For an autoencoder this is just $\mathbf{P} = \mathbf{V}\mathbf{W}$ again, where the encoder $\mathbf{W} = \mathbf{V}^+$ reads off the coordinates $\mathbf{c} = \mathbf{W}\mathbf{x}$ and the decoder $\mathbf{V}$ reconstructs $\mathbf{V}\mathbf{c}$. The matrix $\mathbf{V}^+ = (\mathbf{V}^T\mathbf{V})^{-1}\mathbf{V}^T$ is the [left Moore-Penrose pseudoinverse](/garden/maths/linearAlgebra/moorePenrose#full-column-rank) of $\mathbf{V}$, and it generalizes the line's $\frac{\mathbf{a}^T}{\mathbf{a}^T\mathbf{a}}$ to a basis of more than one vector.

{{< callout type="proof" title="P = VV^+ is the orthogonal projection onto U" >}}
First we need to check that $\mathbf{V}^+$ is well defined: the columns of $\mathbf{V}$ are linearly independent, so $\text{rank}(\mathbf{V}^T\mathbf{V}) = \text{rank}(\mathbf{V}) = d$ and the $d \times d$ matrix $\mathbf{V}^T\mathbf{V}$ is invertible.

We then check the three defining properties of $\mathbf{P} = \mathbf{V}\mathbf{V}^+$ in the same order as the definition:

- **Projects onto $U$:** applying $\mathbf{P}$ to the whole matrix $\mathbf{V}$ fixes its columns,

$$
\mathbf{P}\mathbf{V} = \mathbf{V}(\mathbf{V}^T \mathbf{V})^{-1}\mathbf{V}^T \mathbf{V} = \mathbf{V}
$$

so by linearity $\mathbf{P}$ fixes every vector of $U = \text{span}(\mathbf{V})$.

- **Idempotency:** multiplying $\mathbf{P}$ by itself, the inner factors collapse,

$$
\mathbf{P}^2 = \mathbf{V}(\mathbf{V}^T \mathbf{V})^{-1}\underbrace{(\mathbf{V}^T \mathbf{V})(\mathbf{V}^T \mathbf{V})^{-1}}_{= \mathbf{I}_d}\mathbf{V}^T = \mathbf{V}(\mathbf{V}^T \mathbf{V})^{-1}\mathbf{V}^T = \mathbf{P}
$$

- **Symmetry (orthogonality):** transposing leaves $\mathbf{P}$ unchanged,

$$
\mathbf{P}^T = \big(\mathbf{V}(\mathbf{V}^T \mathbf{V})^{-1}\mathbf{V}^T\big)^T = \mathbf{V}\big((\mathbf{V}^T \mathbf{V})^{-1}\big)^T \mathbf{V}^T = \mathbf{V}(\mathbf{V}^T \mathbf{V})^{-1}\mathbf{V}^T = \mathbf{P}
$$

using that $\mathbf{V}^T \mathbf{V}$ is symmetric, so its inverse is symmetric.
{{< /callout >}}

- **An orthonormal basis:** just as the unit vector tidied up the line, an orthonormal basis tidies up this general formula. We write $\mathbf{U}$ for the matrix whose columns are an orthonormal basis $\mathbf{u}_1, \ldots, \mathbf{u}_d$ of $U$. Orthonormality then means $\mathbf{U}^T\mathbf{U} = \mathbf{I}_d$, so the inverse $(\mathbf{U}^T\mathbf{U})^{-1}$ in the general formula is just the identity and the whole thing simplifies to,

$$
\mathbf{P} = \mathbf{U}(\mathbf{U}^T\mathbf{U})^{-1}\mathbf{U}^T = \mathbf{U}\,\mathbf{I}_d^{-1}\,\mathbf{U}^T = \mathbf{U}\mathbf{U}^T
$$

The encoder is likewise just $\mathbf{W} = \mathbf{V}^+ = \mathbf{U}^T$, so the coordinates are $\mathbf{U}^T\mathbf{x}$ and $\mathbf{P}\mathbf{x} = \sum_{i=1}^d (\mathbf{u}_i^T\mathbf{x})\, \mathbf{u}_i$ is the sum of the line projections onto each basis direction.

It is worth pausing on the two products of $\mathbf{U}$. We have $\mathbf{U}^T\mathbf{U} = \mathbf{I}_d$ because the columns are orthonormal: the $(i,j)$ entry of $\mathbf{U}^T\mathbf{U}$ is $\mathbf{u}_i^T\mathbf{u}_j = \delta_{ij}$, which is $1$ on the diagonal and $0$ off it. The other product $\mathbf{U}\mathbf{U}^T$ is **not** the identity. Because $\mathbf{U}$ is $D \times d$ with $d \le D$, the matrix $\mathbf{U}\mathbf{U}^T$ has rank only $d$, and it is precisely our projection $\mathbf{P}$ onto the $d$-dimensional subspace $U$. The two coincide only when $d = D$, that is when $\mathbf{U}$ is square and the projection is onto all of $\mathbb{R}^D$.

{{< callout type="proof" title="P = UU^T is the orthogonal projection onto U" >}}
The three properties simplify in the same order, each using $\mathbf{U}^T\mathbf{U} = \mathbf{I}_d$:

- **Projects onto $U$:** it fixes the basis,

$$
\mathbf{P}\mathbf{U} = \mathbf{U}\mathbf{U}^T\mathbf{U} = \mathbf{U}
$$

so $\mathbf{P}$ fixes every vector of $U$.

- **Idempotency:** the inner $\mathbf{U}^T\mathbf{U}$ collapses to the identity,

$$
\mathbf{P}^2 = \mathbf{U}\underbrace{\mathbf{U}^T\mathbf{U}}_{= \mathbf{I}_d}\mathbf{U}^T = \mathbf{U}\mathbf{U}^T = \mathbf{P}
$$

- **Symmetry (orthogonality):** the outer product is its own transpose,

$$
\mathbf{P}^T = (\mathbf{U}\mathbf{U}^T)^T = \mathbf{U}\mathbf{U}^T = \mathbf{P}
$$
{{< /callout >}}

This is the orthonormal-basis projection matrix from the [projections note](/garden/maths/linearAlgebra/projectionsLS#projections-with-orthogonal-matrices).

#### Tied vs Untied Weights

The two matrix forms above, $\mathbf{V}\mathbf{V}^+$ for a general basis and $\mathbf{U}\mathbf{U}^T$ for an orthonormal one, are the two ways an autoencoder can carry the projection $\mathbf{P} = \mathbf{V}\mathbf{W}$. Which one it ends up with comes down to a single modelling choice: whether the encoder and decoder keep independent parameters or are forced to share them. Either way the projection $\mathbf{P}$ is the same, only its split into the weight matrices $\mathbf{V}$ and $\mathbf{W}$ changes. The distinction is worth drawing out, because tying the weights quietly builds in an extra constraint.

- **Untied weights:** with independent weights the decoder $\mathbf{V}$ is just some basis of $U$, and the encoder that minimizes the risk is forced to be its pseudoinverse,

$$
\mathbf{W} = \mathbf{V}^+ = \arg\min_{\mathbf{W}} \mathcal{R}(\mathbf{W}, \mathbf{V}), \qquad \mathcal{R}(\mathbf{W}, \mathbf{V}) = \frac{1}{2}\mathbb{E}\big[\|\mathbf{x} - \mathbf{V}\mathbf{W}\mathbf{x}\|^2\big]
$$

giving $\mathbf{P} = \mathbf{V}\mathbf{V}^+$. Once the basis (the columns of $\mathbf{V}$) is chosen the encoder is determined, so the only real freedom left is the choice of subspace $U$ and the choice of basis $\mathbf{V}$ within it. The optimal $\mathbf{P}$ is unique, but infinitely many bases $\mathbf{V}$ give the same projection.

- **Tied weights:** if we instead share parameters by forcing $\mathbf{V} = \mathbf{W}^T$, then for $\mathbf{P} = \mathbf{V}\mathbf{V}^T$ to be a projection the shared matrix must have orthonormal columns, $\mathbf{V}^T\mathbf{V} = \mathbf{I}_d$, which lands us back on the orthonormal form $\mathbf{P} = \mathbf{U}\mathbf{U}^T$. A matrix with orthonormal columns that is not square is called **semi-orthogonal**, so tying builds orthonormality in by hand.

### The Best Subspace

So far we have not referred to the data distribution at all beyond centering. We now ask which subspace $U$ gives the smallest reconstruction risk. So far we know that $\mathbf{P}$ is an *orthogonal* projection, so for every point the reconstruction $\mathbf{P}\mathbf{x}$ and the error $\mathbf{x} - \mathbf{P}\mathbf{x}$ are perpendicular.

Because we can rewrite $\mathbf{x} = \mathbf{P}\mathbf{x} + (\mathbf{x} - \mathbf{P}\mathbf{x})$ splitting $\mathbf{x}$ into two perpendicular pieces we can apply the Pythagorean theorem to get the squared length of $\mathbf{x}$ in terms of the squared lengths of its projection and error,

$$
\|\mathbf{x}\|^2 = \|\mathbf{P}\mathbf{x}\|^2 + \|\mathbf{x} - \mathbf{P}\mathbf{x}\|^2
$$

which we can rearrange to express the squared error in terms of the squared length of the projection,

$$
\|\mathbf{x} - \mathbf{P}\mathbf{x}\|^2 = \|\mathbf{x}\|^2 - \|\mathbf{P}\mathbf{x}\|^2
$$

The input length $\|\mathbf{x}\|^2$ is fixed, so reconstructing a point well means keeping the length of its projection $\|\mathbf{P}\mathbf{x}\|^2$ as large as possible. Averaging over the data and inserting the $\frac{1}{2}$ we get the risk in the form

$$
\mathcal{R}(\mathbf{P}) = \frac{1}{2}\mathbb{E}\big[\|\mathbf{x} - \mathbf{P}\mathbf{x}\|^2\big] = \frac{1}{2}\mathbb{E}\big[\|\mathbf{x}\|^2\big] - \frac{1}{2}\mathbb{E}\big[\|\mathbf{P}\mathbf{x}\|^2\big]
$$

Both of these expected squared lengths are **variances**. Recall that the [variance](/garden/maths/probabilityStatistics/expectationVarianceCovariance#variance-and-standard-deviation) of a random vector $\mathbf{z}$ is $\text{Var}[\mathbf{z}] = \mathbb{E}\big[\|\mathbf{z} - \mathbb{E}[\mathbf{z}]\|^2\big]$, the average squared distance from its mean, which is just the sum of the variances of the individual coordinates. Because our data is centered we have $\mathbb{E}[\mathbf{x}] = \mathbf{0}$ which means the first term is the variance of the data:

$$
\text{Var}[\mathbf{x}] = \mathbb{E}\big[\|\mathbf{x} - \mathbb{E}[\mathbf{x}]\|^2\big] = \mathbb{E}\big[\|\mathbf{x}\|^2\big]
$$

and the second term is the variance of the projected data because we can take the projection matrix $\mathbf{P}$ out of the expectation due to linearity of expectation:

$$
\text{Var}[\mathbf{P}\mathbf{x}] = \mathbb{E}\big[\|\mathbf{P}\mathbf{x} - \mathbb{E}[\mathbf{P}\mathbf{x}]\|^2\big] = \mathbb{E}\big[\|\mathbf{P}\mathbf{x} - \mathbf{P}\mathbb{E}[\mathbf{x}]\|^2\big] = \mathbb{E}\big[\|\mathbf{P}\mathbf{x}\|^2\big]
$$

The risk is therefore

$$
\mathcal{R}(\mathbf{P}) = \frac{1}{2}\Big(\underbrace{\text{Var}[\mathbf{x}]}_{\text{const}} - \text{Var}[\mathbf{P}\mathbf{x}]\Big)
$$

The first term is a constant that does not depend on $\mathbf{P}$. Therefore **minimizing the reconstruction loss is exactly the same as maximizing the variance of the projected data** $\text{Var}[\mathbf{P}\mathbf{x}]$. This is precisely why autoencoders do PCA. We define PCA as the projection that [maximizes the variance of the projection](/garden/maths/linearAlgebra/pca#the-pca-problem), and we have just shown that this is what minimizing reconstruction error amounts to. Intuitively, the projection that loses the least information is the one that keeps the data as spread out as possible.

{{< figure
    src="/images/maths/pcaVariance.webp"
    alt="Projecting 2D data onto different 1D lines, low variance loses cluster structure while maximum variance preserves it."
    caption="Projecting onto a low-variance direction (left) collapses the two clusters and loses structure, which is exactly a large reconstruction error. Projecting onto the direction of maximum variance (right) keeps the points spread out and reconstructs them best."
>}}

We have reduced the problem to maximizing the projected variance $\text{Var}[\mathbf{P}\mathbf{x}] = \mathbb{E}\big[\|\mathbf{P}\mathbf{x}\|^2\big]$. This still looks like it might depend on the data distribution in complicated ways, so let us find exactly which features of the data it actually uses. Writing the squared norm as a dot product, $\|\mathbf{P}\mathbf{x}\|^2 = (\mathbf{P}\mathbf{x})^T(\mathbf{P}\mathbf{x})$, and using that $\mathbf{P}$ is a symmetric idempotent projection so $\mathbf{P}^T\mathbf{P} = \mathbf{P}\mathbf{P} = \mathbf{P}$ we get:

$$
\text{Var}[\mathbf{P}\mathbf{x}] = \mathbb{E}\big[(\mathbf{P}\mathbf{x})^T(\mathbf{P}\mathbf{x})\big] = \mathbb{E}\big[\mathbf{x}^T \mathbf{P}^T\mathbf{P}\, \mathbf{x}\big] = \mathbb{E}\big[\mathbf{x}^T \mathbf{P}\, \mathbf{x}\big]
$$

The quantity $\mathbf{x}^T \mathbf{P}\, \mathbf{x}$ tangles the projection $\mathbf{P}$ and the data point $\mathbf{x}$ together. To separate them we use two small but very useful trick. Firstly a scalar equals its own [trace](/garden/maths/linearAlgebra/eigendecomposition#the-rayleigh-quotient), i.e. $s = \text{tr}(s)$ therefore the trace of a dot product is just the dot product itself.

Secondly the trace can be cycled, $\text{tr}(\mathbf{A}\mathbf{B}) = \text{tr}(\mathbf{B}\mathbf{A})$ (see the [trace properties](/garden/maths/linearAlgebra/eigendecomposition#properties-of-eigenvalues-and-eigenvectors)). Since $\mathbf{x}^T \mathbf{P}\, \mathbf{x}$ is a scalar, we may write it as a trace and cycle the data to one side,

$$
\mathbf{x}^T \mathbf{P}\, \mathbf{x} = \text{tr}(\mathbf{x}^T \mathbf{P}\, \mathbf{x}) = \text{tr}(\mathbf{P}\, \mathbf{x}\mathbf{x}^T)
$$

Now the data appears only inside the outer product $\mathbf{x}\mathbf{x}^T$. We can further rewrite this by using the fact that the trace is **linear**, since it merely sums diagonal entries, so it commutes with the expectation, $\mathbb{E}[\text{tr}(\mathbf{M})] = \text{tr}(\mathbb{E}[\mathbf{M}])$ and just like before because $\mathbf{P}$ is a fixed matrix, linearity of expectation lets it be moved out of the expectation so we get,

$$
\text{Var}[\mathbf{P}\mathbf{x}] = \mathbb{E}\big[\text{tr}(\mathbf{P}\, \mathbf{x}\mathbf{x}^T)\big] = \text{tr}\big(\mathbb{E}[\mathbf{P}\, \mathbf{x}\mathbf{x}^T]\big) = \text{tr}\big(\mathbf{P}\, \mathbb{E}[\mathbf{x}\mathbf{x}^T]\big)
$$

So the projected variance depends on the data through the single matrix $\mathbb{E}[\mathbf{x}\mathbf{x}^T]$, the averaged outer product. This is actually exactly the **covariance matrix** of the data, which we write $\mathbf{\Sigma}$!

The general definition of the [covariance matrix](/garden/maths/probabilityStatistics/expectationVarianceCovariance#covariance-and-correlation) is

$$
\mathbf{\Sigma} = \mathbb{E}\big[(\mathbf{x} - \mathbb{E}[\mathbf{x}])(\mathbf{x} - \mathbb{E}[\mathbf{x}])^T\big]
$$

whose $(j, k)$ entry is the covariance between coordinates $j$ and $k$, so $\mathbf{\Sigma}$ records how the data varies and co-varies along every pair of directions. For our centered data $\mathbb{E}[\mathbf{x}] = \mathbf{0}$, the subtractions vanish, and it reduces to exactly the averaged outer product above,

$$
\mathbf{\Sigma} = \mathbb{E}[\mathbf{x}\mathbf{x}^T] \in \mathbb{R}^{D \times D}
$$

In practice it is estimated from the sample as $\hat{\mathbf{\Sigma}} = \frac{1}{n}\sum_{i=1}^n \mathbf{x}_i \mathbf{x}_i^T$ for centered data (or with the unbiased factor $\frac{1}{n-1}$). It has two important properties: firstly it is symmetric, $\mathbf{\Sigma}^T = \mathbf{\Sigma}$, which is clear from the outer product, and secondly it is a **positive semi-definite** matrix.

{{< callout type="proof" title="The covariance matrix is positive semi-definite" >}}
For a matrix to be positive semi-definite we must show $\mathbf{v}^T \mathbf{\Sigma}\, \mathbf{v} \ge 0$ for every direction $\mathbf{v}$ (see [positive semi-definite matrices](/garden/maths/linearAlgebra/eigendecomposition#positive-definite-and-positive-semi-definite-matrices)). Substituting $\mathbf{\Sigma} = \mathbb{E}[\mathbf{x}\mathbf{x}^T]$ and pulling the fixed vector $\mathbf{v}$ inside the expectation,

$$
\mathbf{v}^T \mathbf{\Sigma}\, \mathbf{v} = \mathbf{v}^T\,\mathbb{E}[\mathbf{x}\mathbf{x}^T]\,\mathbf{v} = \mathbb{E}\big[\mathbf{v}^T\mathbf{x}\,\mathbf{x}^T\mathbf{v}\big] = \mathbb{E}\big[(\mathbf{v}^T\mathbf{x})^2\big] \ge 0
$$

since it is the expectation of a square, which is never negative. The quantity $\mathbf{v}^T\mathbf{\Sigma}\mathbf{v} = \text{Var}[\mathbf{v}^T\mathbf{x}]$ is exactly the variance of the data along $\mathbf{v}$, so this just says a variance cannot be negative.

A couple of other routes reach the same conclusion. Each rank-one term $\mathbf{x}\mathbf{x}^T$ is itself positive semi-definite, and $\mathbf{\Sigma}$ is an average of these, so it inherits the property since any non-negative combination of PSD matrices is PSD. Equivalently, the sample estimate $\hat{\mathbf{\Sigma}} = \frac{1}{n}\mathbf{X}^T\mathbf{X}$ (with the centered points stacked as the rows of $\mathbf{X}$) is a Gram matrix, and every Gram matrix $\mathbf{A}^T\mathbf{A}$ is positive semi-definite by the same one-line argument, $\mathbf{v}^T\mathbf{A}^T\mathbf{A}\mathbf{v} = \|\mathbf{A}\mathbf{v}\|^2 \ge 0$. Being symmetric and PSD means all eigenvalues of $\mathbf{\Sigma}$ are real and non-negative, which is what we rely on below when they turn out to be the retained variances.
{{< /callout >}}

Substituting $\text{Var}[\mathbf{P}\mathbf{x}] = \text{tr}(\mathbf{P}\mathbf{\Sigma})$ back into the risk gives a clean expression in $\mathbf{\Sigma}$,

$$
\mathcal{R}(\mathbf{P}) = \frac{1}{2}\text{Var}[\mathbf{x}] - \frac{1}{2}\text{tr}(\mathbf{P}\mathbf{\Sigma}) = -\frac{1}{2}\text{tr}(\mathbf{P}\mathbf{\Sigma}) + \text{const}
$$

So the optimal projection depends on the data only through $\mathbf{\Sigma}$ (together with the mean $\mathbb{E}[\mathbf{x}]$ we used to center). In the language of statistics, $\mathbf{\Sigma}$ is a **sufficient statistic** for the problem as it captures everything we need in order to evaluate and minimize the risk, and nothing else about the distribution matters.

### The PCA Theorem

The previous section reduced the entire problem to a single optimization: maximize the retained variance $\text{Var}[\mathbf{P}\mathbf{x}] = \text{tr}(\mathbf{P}\mathbf{\Sigma})$ over rank-$d$ orthogonal projections $\mathbf{P} = \mathbf{U}\mathbf{U}^T$, with the covariance $\mathbf{\Sigma}$ the only thing about the data that matters. It remains to actually solve this maximization. The answer is built from the eigenvectors of $\mathbf{\Sigma}$, so before stating it we look at the eigenstructure of the two matrices involved, the projection $\mathbf{P}$ and the covariance $\mathbf{\Sigma}$, which is what makes eigenvectors appear in the first place.

- **The projection side:** a rank-$d$ orthogonal projection $\mathbf{P} = \mathbf{U}\mathbf{U}^T$ leaves every direction of $U$ untouched, $\mathbf{P}\mathbf{u}_i = \mathbf{u}_i$, and annihilates everything orthogonal to it, $\mathbf{P}\mathbf{w} = \mathbf{0}$ for $\mathbf{w} \in U^\perp$. So its eigenvalues are just $1$ on the subspace $U$ it keeps and $0$ on the complement it discards. Choosing the projection is therefore the same as choosing which $d$-dimensional subspace receives the eigenvalue $1$.

{{< callout type="proof" title="Eigenvalues of a Projection are 0 or 1" >}}
Suppose $\mathbf{P}\mathbf{v} = \lambda \mathbf{v}$ with $\mathbf{v} \neq 0$. Applying $\mathbf{P}$ again and using $\mathbf{P}^2 = \mathbf{P}$,

$$
\lambda \mathbf{v} = \mathbf{P}\mathbf{v} = \mathbf{P}^2 \mathbf{v} = \mathbf{P}(\lambda \mathbf{v}) = \lambda \mathbf{P} \mathbf{v} = \lambda^2 \mathbf{v}
$$

so $\lambda^2 = \lambda$, which forces $\lambda \in \{0, 1\}$. The eigenvectors with eigenvalue $1$ span the image $\text{im}(\mathbf{P}) = U$ that $\mathbf{P}$ keeps, and those with eigenvalue $0$ span the kernel $\ker(\mathbf{P})$ that it discards.
{{< /callout >}}

- **The covariance side:** this leaves the real question: which subspace should receive the eigenvalue $1$? It may seem mysterious that the eigenvectors of $\mathbf{\Sigma}$, defined by $\mathbf{\Sigma}\mathbf{u} = \lambda\mathbf{u}$, are the answer, given that variance maximization never mentioned them. Two facts about symmetric matrices explain it. First, take the simplest case of projecting onto a single unit direction $\mathbf{v}$. The variance it retains is

$$
\text{Var}[\mathbf{P}\mathbf{x}] = \text{tr}(\mathbf{v}\mathbf{v}^T \mathbf{\Sigma}) = \mathbf{v}^T \mathbf{\Sigma}\, \mathbf{v}
$$

which is the [Rayleigh quotient](/garden/maths/linearAlgebra/eigendecomposition#the-rayleigh-quotient) of $\mathbf{\Sigma}$. Among all unit vectors this is maximized by the leading eigenvector $\mathbf{u}_1$, where it equals the largest eigenvalue $\lambda_1$. So the single best direction is exactly the top eigenvector of $\mathbf{\Sigma}$, which the [PCA notes](/garden/maths/linearAlgebra/pca#lagrange-multiplier-solution) derive directly with a Lagrange multiplier. Second, eigenvectors of a symmetric matrix belonging to distinct eigenvalues are orthogonal, so the leading eigenvectors automatically form the orthonormal basis we need for higher-dimensional projections.

{{< callout type="proof" title="Eigenvectors of a Symmetric Matrix are Orthogonal" >}}
Let $\mathbf{\Sigma} \mathbf{u} = \lambda \mathbf{u}$ and $\mathbf{\Sigma} \mathbf{u}' = \lambda' \mathbf{u}'$ with $\lambda \neq \lambda'$. Because $\mathbf{\Sigma}$ is symmetric, $(\mathbf{\Sigma}\mathbf{u})^T\mathbf{u}' = \mathbf{u}^T\mathbf{\Sigma}^T\mathbf{u}' = \mathbf{u}^T\mathbf{\Sigma}\mathbf{u}'$, so evaluating $\mathbf{u}^T\mathbf{\Sigma}\mathbf{u}'$ from both sides gives

$$
\lambda\, \mathbf{u}^T\mathbf{u}' = (\mathbf{\Sigma}\mathbf{u})^T\mathbf{u}' = \mathbf{u}^T\mathbf{\Sigma}\mathbf{u}' = \mathbf{u}^T(\lambda'\mathbf{u}') = \lambda'\, \mathbf{u}^T\mathbf{u}'
$$

Hence $(\lambda - \lambda')\,\mathbf{u}^T\mathbf{u}' = 0$, and since $\lambda \neq \lambda'$ we get $\mathbf{u}^T\mathbf{u}' = 0$.
{{< /callout >}}

Putting the two sides together, we must hand the projection's $d$ eigenvalue-$1$ directions to the subspace that retains the most variance, and the single-direction argument points straight at the top eigenvectors of $\mathbf{\Sigma}$. This is exactly the PCA theorem.

{{< callout type="info" title="PCA Theorem" >}}
The rank-$d$ projection that minimizes the squared reconstruction loss on centered data is the orthogonal projection onto the span of the $d$ principal (largest-eigenvalue) eigenvectors of the covariance matrix $\mathbf{\Sigma} = \mathbb{E}[\mathbf{x} \mathbf{x}^T]$. That is, the optimal $\mathbf{P} = \mathbf{U} \mathbf{U}^T$ where the columns of $\mathbf{U}$ are $d$ orthonormal eigenvectors of $\mathbf{\Sigma}$ corresponding to its $d$ largest eigenvalues.
{{< /callout >}}

The single-direction argument above is only a heuristic for the full problem, so we now prove it properly by maximizing $\text{Var}[\mathbf{P}\mathbf{x}] = \text{tr}(\mathbf{P} \mathbf{\Sigma})$ over rank-$d$ orthogonal projections $\mathbf{P} = \mathbf{U} \mathbf{U}^T$ (with $\mathbf{U}^T \mathbf{U} = \mathbf{I}_d$), building up in two steps.

#### The Diagonal Case

First suppose the covariance is already diagonal,

$$
\mathbf{\Sigma} = \mathbf{\Lambda} = \text{diag}(\lambda_1, \ldots, \lambda_D), \qquad \lambda_1 \ge \lambda_2 \ge \cdots \ge \lambda_D \ge 0
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

So we are spreading a total weight of $d$ across the $\gamma_i \in [0,1]$ to maximize $\sum_i \lambda_i \gamma_i$ with the $\lambda_i$ sorted in decreasing order. The maximum puts all the weight on the largest eigenvalues, $\gamma_1 = \cdots = \gamma_d = 1$ and the rest zero, which is achieved by taking the columns of $\mathbf{U}$ to be the first $d$ standard basis vectors $\mathbf{e}_1, \ldots, \mathbf{e}_d$, giving $\mathbf{P} = \begin{bmatrix} \mathbf{I}_d & 0 \\ 0 & 0\end{bmatrix}$.
{{< /callout >}}

The maximal variance attained is $\text{Var}[\mathbf{P}\mathbf{x}] = \sum_{i=1}^d \lambda_i$, the sum of the $d$ largest variances. For a projection onto a single line ($d = 1$) that minimizes the squared loss, this is simply the largest eigenvalue $\lambda_1$ of the covariance. If some eigenvalues are equal, say $\lambda_i = \cdots = \lambda_{i+k}$, then the choice of axes within that tied block is not unique. Any orthonormal basis of the corresponding eigenspace works equally well, which is our first hint that the solution determines a subspace rather than a specific basis.

#### The General Case via the Spectral Theorem

For a general covariance matrix we use the [spectral theorem](/garden/maths/linearAlgebra/eigendecomposition#eigenvalues-and-eigenvectors-of-symmetric-matrices) to reduce to the diagonal case. Since $\mathbf{\Sigma}$ is symmetric and positive semi-definite, it can be diagonalized by an orthogonal matrix,

$$
\mathbf{\Sigma} = \mathbf{Q} \mathbf{\Lambda} \mathbf{Q}^T, \qquad \mathbf{\Lambda} = \text{diag}(\lambda_1, \ldots, \lambda_D), \quad \lambda_1 \ge \cdots \ge \lambda_D \ge 0
$$

where the columns of the orthogonal matrix $\mathbf{Q}$ are the orthonormal eigenvectors of $\mathbf{\Sigma}$ ordered by decreasing eigenvalue. Conversely, $\mathbf{Q}\mathbf{\Lambda} \mathbf{Q}^T$ for any orthogonal $\mathbf{Q}$ is a symmetric matrix whose eigenvalues are the diagonal entries of $\mathbf{\Lambda}$, the simplest example being $\mathbf{\Lambda}$ itself with $\mathbf{Q} = \mathbf{I}$. In this eigenbasis the action of $\mathbf{\Sigma}$ is just a coordinate-wise scaling: the coordinates of $\mathbf{\Sigma}\mathbf{x}$ relative to $\{\mathbf{q}_1, \ldots, \mathbf{q}_D\}$ are $\mathbf{Q}^T \mathbf{\Sigma} \mathbf{x} = \mathbf{\Lambda} \mathbf{Q}^T \mathbf{x}$, so the $i$-th coordinate is $\lambda_i\,(\mathbf{q}_i^T \mathbf{x})$. The optimal projection is then

$$
\mathbf{P} = \mathbf{U} \mathbf{U}^T, \qquad \mathbf{U} = \mathbf{Q} \begin{bmatrix} \mathbf{I}_d \\ 0 \end{bmatrix}
$$

that is, $\mathbf{U}$ consists of the first $d$ columns of $\mathbf{Q}$, the $d$ principal eigenvectors of $\mathbf{\Sigma}$.

{{< callout type="proof" title="General Case" >}}
Substitute $\mathbf{\Sigma} = \mathbf{Q}\mathbf{\Lambda} \mathbf{Q}^T$ into the variance and use the cyclic property of the trace,

$$
\text{Var}[\mathbf{P}\mathbf{x}] = \text{tr}\big(\mathbf{U} \mathbf{U}^T\, \mathbf{Q}\mathbf{\Lambda} \mathbf{Q}^T\big) = \text{tr}\big((\mathbf{Q}^T \mathbf{U})(\mathbf{Q}^T \mathbf{U})^T \mathbf{\Lambda}\big)
$$

Setting $\tilde{\mathbf{U}} = \mathbf{Q}^T \mathbf{U}$, this is exactly the diagonal problem from before (note $\tilde{\mathbf{U}}^T \tilde{\mathbf{U}} = \mathbf{U}^T \mathbf{Q} \mathbf{Q}^T \mathbf{U} = \mathbf{U}^T \mathbf{U} = \mathbf{I}_d$ since $\mathbf{Q}$ is orthogonal). Its maximizer is $\tilde{\mathbf{U}} = \mathbf{Q}^T \mathbf{U} = \begin{bmatrix} \mathbf{I}_d \\ 0\end{bmatrix}$, and multiplying by $\mathbf{Q}$ gives $\mathbf{U} = \mathbf{Q}\begin{bmatrix} \mathbf{I}_d \\ 0\end{bmatrix}$, the first $d$ eigenvectors of $\mathbf{\Sigma}$.
{{< /callout >}}

#### Subspace, Not Basis

There is an important subtlety. The linear autoencoder recovers the principal **subspace** $\text{span}\big(\mathbf{Q}\begin{bmatrix} \mathbf{I}_d \\ 0\end{bmatrix}\big)$, but it does not necessarily recover the PCA **basis** (the individual principal eigenvectors). This is because the optimal $\mathbf{P} = \mathbf{V} \mathbf{V}^+$ can use any basis $\mathbf{V}$ of the subspace, and as we saw the factorization $\mathbf{P} = \mathbf{V}\mathbf{W}$ is non-identifiable. So a trained linear autoencoder gives you the right subspace but generally a rotated, non-orthonormal basis within it, not the ordered principal components themselves. To get the actual principal components one must additionally diagonalize, for example with an eigendecomposition of the covariance.

This also resolves the [earlier question](#uniqueness-of-the-solution) of whether the optimal $\mathbf{P}$ is unique. The projection, and with it the principal subspace, is unique exactly when there is a strict gap $\lambda_d > \lambda_{d+1}$ at the cutoff. If the $d$-th and $(d+1)$-th eigenvalues are equal, any $d$-dimensional subspace lying inside that tied block of eigenvectors retains the same variance, so the optimal $\mathbf{P}$ is no longer unique.

### Learning Algorithms

The theory tells us the optimum is built from the principal eigenvectors of the covariance matrix. There are several ways to actually compute it, trading off accuracy, scalability, and how much of the spectrum we need.

#### Eigendecomposition

The direct approach forms the covariance matrix and diagonalizes it. The cost is $O(n D^2)$ to estimate the covariance plus $O(D^3)$ for the eigendecomposition. The cubic term is the bottleneck and becomes prohibitive when $D$ is large. This is wasteful when we only want a small number $d \ll D$ of components, because we compute the entire spectrum and then throw most of it away.

#### Power Method

When $d \ll D$, computing the whole spectrum is wasteful. The **power method** (power iteration) computes just the principal eigenvector $\mathbf{u}_1$, the top principal component (the direction of largest variance). The idea rests on what multiplying by $\mathbf{\Sigma}$ does to a vector: written in the eigenbasis of $\mathbf{\Sigma}$, multiplication scales each eigen-component by its eigenvalue $\lambda_i$, so the components along large-variance directions grow fastest. Repeating the multiplication amplifies the dominant direction more and more relative to the rest, and renormalizing at each step keeps the vector at unit length so it cannot blow up or shrink to zero. Starting from a random unit vector $\mathbf{v}^{(0)} \sim \mathcal{N}(0, \mathbf{I})$ we iterate

$$
\mathbf{v}^{(t+1)} = \frac{\mathbf{\Sigma} \mathbf{v}^{(t)}}{\|\mathbf{\Sigma} \mathbf{v}^{(t)}\|}
$$

and after enough steps the direction settles on $\mathbf{u}_1$. This is not the [Gram-Schmidt](/garden/maths/linearAlgebra/projectionsLS) process, which orthogonalizes a set of vectors; here we are amplifying a single direction. Orthogonalization does enter, but only later, when we deflate to find the remaining components.

{{< callout type="proof" title="Convergence of the Power Method" >}}
Let $\mathbf{\Sigma}$ have eigenvalues $\lambda_1 > \lambda_2 \ge \cdots \ge \lambda_D \ge 0$ with orthonormal eigenvectors $\mathbf{u}_i$, assuming a strict gap $\lambda_1 > \lambda_2$ so there is a unique top direction. Expand the start vector in this eigenbasis,

$$
\mathbf{v}^{(0)} = \sum_i \alpha_i \mathbf{u}_i
$$

with $\alpha_1 \neq 0$ (true with probability one for a random start). Renormalization only rescales the vector, it does not change its direction, so we can track the direction by applying $\mathbf{\Sigma}^k$ alone. Each application scales the $i$-th component by $\lambda_i^k$,

$$
\mathbf{\Sigma}^k \mathbf{v}^{(0)} = \sum_i \alpha_i \lambda_i^k \mathbf{u}_i = \lambda_1^k\bigg(\alpha_1 \mathbf{u}_1 + \sum_{i > 1}\alpha_i \Big(\tfrac{\lambda_i}{\lambda_1}\Big)^k \mathbf{u}_i\bigg)
$$

where we have pulled out the largest factor $\lambda_1^k$. Because $\lambda_i / \lambda_1 < 1$ for every $i > 1$, each ratio $(\lambda_i/\lambda_1)^k \to 0$ as $k$ grows, so every term except the first dies away. The bracket converges to $\alpha_1\mathbf{u}_1$, and after renormalization $\mathbf{v}^{(k)} \to \mathbf{u}_1$. The closer $\lambda_2 / \lambda_1$ is to $1$, the slower this decay, so how well-separated the top eigenvalue is decides how many iterations we need.
{{< /callout >}}

Once the iteration has converged, the eigenvalue is cheap to read off: it is the [Rayleigh quotient](/garden/maths/linearAlgebra/eigendecomposition#the-rayleigh-quotient) $\lambda_1 = \mathbf{u}_1^T\mathbf{\Sigma}\mathbf{u}_1$ (equivalently the norm $\|\mathbf{\Sigma}\mathbf{v}^{(t)}\|$ just before the last renormalization), one extra matrix-vector product at $O(D^2)$, not a cubic eigendecomposition. To get the next eigenvectors we **deflate**: with the principal eigenvector $\mathbf{u}_1$ and its eigenvalue $\lambda_1$ in hand, we subtract its contribution and rerun the power method on

$$
\mathbf{\Sigma}_2 = \mathbf{\Sigma} - \lambda_1 \mathbf{u}_1 \mathbf{u}_1^T
$$

This zeros out the top eigenvalue while leaving the rest untouched, since in the eigenbasis $\mathbf{Q}^T \mathbf{\Sigma} \mathbf{Q} = \text{diag}(\lambda_1, \lambda_2, \ldots) \mapsto \text{diag}(0, \lambda_2, \ldots) = \mathbf{Q}^T \mathbf{\Sigma}_2 \mathbf{Q}$, so the new principal eigenvector is the old second one.

Each matrix-vector product costs $O(D^2)$, so $T$ iterations for one component cost $O(T D^2)$, and the top $d$ components cost $O(T d D^2)$ plus the $O(n D^2)$ to form the covariance. Compared with the $O(D^3)$ of a full eigendecomposition, this wins exactly when $T d \ll D$, that is when we need few components and the iteration converges quickly. The iteration count $T$ is not fixed in advance but set by a convergence tolerance, stopping once $\mathbf{v}^{(t)}$ barely changes between steps. A well-separated top eigenvalue needs only tens of iterations, whereas a small gap $\lambda_2/\lambda_1 \approx 1$ needs many more.

#### Gradient Descent

Probably the most generic approach treats the autoencoder as a neural network and trains it with [gradient descent](/garden/ml/gradientDescent), the workhorse of deep learning. We need the gradient of the single-example loss $\mathcal{L}(\mathbf{x}; \mathbf{P}) = \frac{1}{2}\|\mathbf{x} - \mathbf{P}\mathbf{x}\|^2$. Expanding the squared norm as a trace,

$$
\mathcal{L}(\mathbf{x}; \mathbf{P}) = \frac{1}{2}(\mathbf{x} - \mathbf{P}\mathbf{x})^T(\mathbf{x} - \mathbf{P}\mathbf{x}) = \frac{1}{2}\|\mathbf{x}\|^2 - \text{tr}(\mathbf{P} \mathbf{x} \mathbf{x}^T) + \frac{1}{2}\text{tr}(\mathbf{P}^T \mathbf{P} \mathbf{x} \mathbf{x}^T)
$$

We differentiate term by term with the matrix rules $\frac{\partial}{\partial \mathbf{A}}\text{tr}(\mathbf{A}\mathbf{B}) = \mathbf{B}^T$ and $\frac{\partial}{\partial \mathbf{A}}\text{tr}(\mathbf{A}^T \mathbf{A} \mathbf{B}) = \mathbf{A}(\mathbf{B} + \mathbf{B}^T)$ (see the [matrix cookbook](https://www.math.uwaterloo.ca/~hwolkowi/matrixcookbook.pdf)). The first term is constant in $\mathbf{P}$. The second gives $\frac{\partial}{\partial \mathbf{P}}\text{tr}(\mathbf{P}\mathbf{x}\mathbf{x}^T) = (\mathbf{x}\mathbf{x}^T)^T = \mathbf{x}\mathbf{x}^T$. For the third, with $\mathbf{B} = \mathbf{x}\mathbf{x}^T$ symmetric so $\mathbf{B} + \mathbf{B}^T = 2\mathbf{x}\mathbf{x}^T$, we get $\frac{1}{2}\,\mathbf{P}(2\mathbf{x}\mathbf{x}^T) = \mathbf{P}\mathbf{x}\mathbf{x}^T$. Putting the three together,

$$
\nabla_{\mathbf{P}}\, \mathcal{L}(\mathbf{x}; \mathbf{P}) = \mathbf{0} - \mathbf{x}\mathbf{x}^T + \mathbf{P}\mathbf{x}\mathbf{x}^T = (\mathbf{P} - \mathbf{I})\mathbf{x} \mathbf{x}^T
$$

But we update the weight matrices $\mathbf{W}$ and $\mathbf{V}$, not $\mathbf{P} = \mathbf{V}\mathbf{W}$ directly. The chain rule for matrix derivatives propagates $\nabla_{\mathbf{P}}\mathcal{L}$ through the product $\mathbf{P} = \mathbf{V}\mathbf{W}$: differentiating with respect to $\mathbf{W}$ multiplies on the left by $\mathbf{V}^T$, and with respect to $\mathbf{V}$ multiplies on the right by $\mathbf{W}^T$,

$$
\nabla_{\mathbf{W}}\, \mathcal{L} = \mathbf{V}^T \big(\nabla_{\mathbf{P}}\mathcal{L}\big) = \mathbf{V}^T (\mathbf{P} - \mathbf{I}) \mathbf{x} \mathbf{x}^T, \qquad \nabla_{\mathbf{V}}\, \mathcal{L} = \big(\nabla_{\mathbf{P}}\mathcal{L}\big) \mathbf{W}^T = (\mathbf{P} - \mathbf{I}) \mathbf{x} \mathbf{x}^T \mathbf{W}^T
$$

A quick dimension check confirms each gradient has the same shape as the matrix it updates: $\nabla_{\mathbf{W}}\mathcal{L}$ is $d \times D$ like $\mathbf{W}$, and $\nabla_{\mathbf{V}}\mathcal{L}$ is $D \times d$ like $\mathbf{V}$.

Full-batch **gradient descent** averages this gradient over all $n$ data points and steps against it with learning rate $\eta > 0$,

$$
\mathbf{W} \leftarrow \mathbf{W} - \eta\, \frac{1}{n}\sum_{i=1}^n \nabla_{\mathbf{W}}\, \mathcal{L}(\mathbf{x}_i), \qquad \mathbf{V} \leftarrow \mathbf{V} - \eta\, \frac{1}{n}\sum_{i=1}^n \nabla_{\mathbf{V}}\, \mathcal{L}(\mathbf{x}_i)
$$

Each step sweeps the whole dataset, so its cost grows with $n$, which is expensive for large datasets. **Stochastic gradient descent** (SGD) avoids the full sum by estimating the gradient from a single random point, or a small batch of size $k$, at each step,

$$
\mathbf{W} \leftarrow \mathbf{W} - \eta\, \nabla_{\mathbf{W}}\, \mathcal{L}(\mathbf{x}), \qquad \mathbf{V} \leftarrow \mathbf{V} - \eta\, \nabla_{\mathbf{V}}\, \mathcal{L}(\mathbf{x})
$$

The cost per step is dominated by the matrix products, $O\big((d + k)D^2\big)$, now independent of $n$. This scales well and lets us reuse all the tooling of deep learning, at the price of the usual gradient-descent caveats around step sizes and convergence speed.

### Non-Linear Autoencoders

Everything above used linear maps. A **non-linear autoencoder** lets the encoder $g_\phi: \mathbb{R}^D \to \mathbb{R}^d$ and decoder $f_\theta: \mathbb{R}^d \to \mathbb{R}^D$ be non-linear functions, typically built from neural network layers with non-linear activations. Since the class of non-linear functions includes the linear ones as a special case, a non-linear autoencoder can always match and usually beat the reconstruction loss of the best linear autoencoder. It can bend the latent subspace into a curved manifold, capturing structure that no flat projection can.

There is a catch, however. Linear autoencoders are solved globally by the PCA theorem, but non-linear autoencoders are trained by gradient descent with no global convergence guarantees. In practice a poorly optimized non-linear autoencoder can end up worse than the linear optimum.

{{< callout type="info" title="From Autoencoders to Matrix Factorization" >}}
The linear autoencoder learns a one-sided rank-$d$ map, reconstructing every data point through the projection $\mathbf{P} = \mathbf{V}\mathbf{W}$. A closely related model frees both sides and factors the whole data matrix as $\mathbf{A} \approx \mathbf{U}\mathbf{V}^\top$ with rank $k$. When the matrix is fully observed this is solved by the truncated SVD, the same eigenvector and singular value machinery in another guise. When the matrix is only partially observed it becomes [matrix completion](/garden/ml/matrixCompletion), the basis of recommender systems, where the rank constraint is what lets us fill in the missing entries.
{{< /callout >}}

## What Makes a Good Representation?

{{< callout type="todo">}}
Im not sure yet if this can be moved into the above section somewhere or if the VAE part should be a seperate page etc. For now we just ignore the below part.
{{< /callout >}}

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

Let $\mathcal{H} = \{enc_\theta: \mathcal{X} \to \mathcal{Z} \mid \theta \in \Theta\}$ be a parametrized family of encoder functions. Given a random variable $X$ with range in $\mathcal{X}$ and a conditional probability density function $p(\mathbf{x} \mid \mathbf{z})$, the infomax principle argues that the best encoder is given by:

$$
\arg\max_{\theta} I(X; enc_\theta(X))
$$

To understand this optimization, we first need to express mutual information in terms of probability densities. Recall that mutual information can be written as:

$$
I(X; Z) = \mathbb{E}_{p(\mathbf{x},\mathbf{z})} \left[ \log \frac{p(\mathbf{x}, \mathbf{z})}{p(\mathbf{x})p(\mathbf{z})} \right] = \mathbb{E}_{p(\mathbf{x},\mathbf{z})} [\log p(\mathbf{x} \mid \mathbf{z})] - \mathbb{E}_{p(\mathbf{x})} [\log p(\mathbf{x})]
$$

where we use the fact that $p(\mathbf{x},\mathbf{z}) = p(\mathbf{x} \mid \mathbf{z})p(\mathbf{z})$. This shows that mutual information measures how much the conditional distribution $p(\mathbf{x} \mid \mathbf{z})$ differs from the marginal $p(\mathbf{x})$.

For a sample $\mathbf{x}_1, \ldots, \mathbf{x}_n$, we can approximate the mutual information by maximizing the expected log-likelihood under the encoder distribution. This uses the **variational characterization of mutual information**, which provides a lower bound: for any conditional distribution $q(\mathbf{x} \mid \mathbf{z})$, we have

$$
I(X; Z) \geq \mathbb{E}_{p(\mathbf{x},\mathbf{z})}[\log q(\mathbf{x} \mid \mathbf{z})] - \mathbb{E}_{p(\mathbf{x})} [\log p(\mathbf{x})]
$$

with equality when $q(\mathbf{x} \mid \mathbf{z}) = p(\mathbf{x} \mid \mathbf{z})$. The supremum over all possible $q$ achieves this equality. When we maximize with respect to the encoder parameters $\theta$, we are maximizing a lower bound on this mutual information. The second term $\mathbb{E}_{p(\mathbf{x})} [\log p(\mathbf{x})]$ does not depend on the encoder, so maximizing the bound is equivalent to maximizing the first term. In practice, for a given encoder $enc_\theta$, this leads to:

$$
I(X; enc_\theta(X)) \approx \sum_{i \leq n} \mathbb{E}_{Z \mid \mathbf{x}_i} [\log p(\mathbf{x}_i \mid Z)]
$$

This objective encourages the encoder to preserve as much information about the input as possible in the latent representation.

However, the infomax principle alone is insufficient for learning good representations. While it guarantees informativeness by definition, it does not necessarily lead to disentangled or robust representations. If the input space $\mathcal{X}$ and latent space $\mathcal{Z}$ are both complex enough, one can trivially maximize mutual information by finding a bijection from $\mathcal{X}$ to $\mathcal{Z}$. This would simply copy the input to the latent space without learning any meaningful compression or structure.

## Variational Autoencoders

Variational autoencoders address the limitations of traditional autoencoders by using a Bayesian approach to representation learning. Instead of learning a deterministic mapping from inputs to latent codes, VAEs learn a probability distribution over latent codes for each input. This probabilistic view enables VAEs to generate new samples and ensures a more regular, continuous latent space.

The key distinction between traditional autoencoders and VAEs is this probabilistic framing. A traditional autoencoder learns mappings $\mathbf{z} = g_\phi(\mathbf{x})$ and $\hat{\mathbf{x}} = f_\theta(\mathbf{z})$. A VAE instead learns:
- An **encoder** (also called recognition network or inference network) that outputs parameters of a distribution over latent codes: $q_\phi(\mathbf{z} \mid \mathbf{x})$
- A **decoder** (also called generative network) that outputs parameters of a distribution over reconstructions: $p_\theta(\mathbf{x} \mid \mathbf{z})$

### The Generative Model

To understand VAEs, it helps to start with the generative model of how we imagine data is created. We assume that each data point $\mathbf{x}$ is generated through a two-step process:

1. A latent variable $\mathbf{z}$ is sampled from a prior distribution $p(\mathbf{z})$. This latent variable represents the underlying factors that generate the data. For instance, for face images, $\mathbf{z}$ might encode age, expression, pose, and lighting. The prior $p(\mathbf{z})$ is typically chosen to be a simple distribution such as a standard Gaussian $\mathcal{N}(0, \mathbf{I})$. This choice encourages the latent space to have a regular structure that is easy to sample from.

2. The observed data $\mathbf{x}$ is generated from the latent variable according to a likelihood distribution $p_\theta(\mathbf{x} \mid \mathbf{z})$. This likelihood is parameterized by a neural network (the decoder $f_\theta$) that maps latent variables to distributions over the data space. For continuous data like images, this is often a Gaussian with mean given by the decoder output: $p_\theta(\mathbf{x} \mid \mathbf{z}) = \mathcal{N}(\mathbf{x} \mid \boldsymbol{\mu}_\theta(\mathbf{z}), \sigma^2 \mathbf{I})$, where $\boldsymbol{\mu}_\theta(\mathbf{z}) = f_\theta(\mathbf{z})$ is the neural network decoder.

The joint distribution of data and latent variables is then:

$$
p_\theta(\mathbf{x}, \mathbf{z}) = p_\theta(\mathbf{x} \mid \mathbf{z}) p(\mathbf{z})
$$

This is the **generative model**, which connects directly to the decoder intuition. The prior $p(\mathbf{z})$ represents our initial belief about the latent space structure (typically a standard Gaussian), while the likelihood $p_\theta(\mathbf{x} \mid \mathbf{z})$ represents the decoder's ability to generate data from latent variables. The decoder parameters $\theta$ control how latent variables map to observed data.

Our goal in training is to find parameters $\theta$ that make this generative model assign high probability to the observed data. To evaluate how well the model fits the data, we need the **marginal likelihood** (also called the **evidence**), which is obtained by integrating out the latent variables:

$$
p_\theta(\mathbf{x}) = \int p_\theta(\mathbf{x} \mid \mathbf{z}) p(\mathbf{z}) d\mathbf{z}
$$

The evidence tells us how probable a data point $\mathbf{x}$ is under our model, averaging over all possible latent variables $\mathbf{z}$ that could have generated it. However, this integration is generally intractable because it requires integrating over all possible latent variables. For high-dimensional latent spaces with complex decoders (neural networks), this cannot be done analytically or efficiently approximated.

{{< figure
    src="/images/ml/autoencoderVariational.png"
    alt="Variational autoencoder with probabilistic encoder and decoder."
    caption="A VAE encoder outputs distribution parameters (mean and variance), we sample from this distribution, and the decoder maps samples back to the data space."
>}}

### The Inference Model

To make sense of this generative model, we also need to perform **inference**: given an observed data point $\mathbf{x}$, what latent code $\mathbf{z}$ is likely to have generated it? This is captured by the **posterior distribution** $p_\theta(\mathbf{z} \mid \mathbf{x})$. Using Bayes' rule:

$$
p_\theta(\mathbf{z} \mid \mathbf{x}) = \frac{p_\theta(\mathbf{x} \mid \mathbf{z}) p(\mathbf{z})}{p_\theta(\mathbf{x})}
$$

However, this posterior is intractable because the denominator $p_\theta(\mathbf{x})$ requires computing the integral we just mentioned. This is where variational inference comes in. We introduce a **variational distribution** (the encoder) $q_\phi(\mathbf{z} \mid \mathbf{x})$ parameterized by $\phi$ that approximates the true posterior. The encoder takes a data point $\mathbf{x}$ and outputs parameters of a distribution over latent variables, typically the mean and variance of a Gaussian: $q_\phi(\mathbf{z} \mid \mathbf{x}) = \mathcal{N}(\mathbf{z} \mid \boldsymbol{\mu}_\phi(\mathbf{x}), \boldsymbol{\sigma}_\phi^2(\mathbf{x}) \mathbf{I})$.

Here we connect to the autoencoder view. The encoder neural network $g_\phi$ takes input $\mathbf{x}$ and outputs $[\boldsymbol{\mu}_\phi(\mathbf{x}), \boldsymbol{\sigma}_\phi(\mathbf{x})]$. We then sample $\mathbf{z} \sim q_\phi(\mathbf{z} \mid \mathbf{x}) = \mathcal{N}(\boldsymbol{\mu}_\phi(\mathbf{x}), \boldsymbol{\sigma}_\phi^2(\mathbf{x}) \mathbf{I})$. The decoder neural network $f_\theta$ takes this sampled $\mathbf{z}$ and outputs the parameters of the reconstruction distribution, typically just the mean $\boldsymbol{\mu}_\theta(\mathbf{z})$, which we interpret as the reconstruction $\hat{\mathbf{x}}$.

The **true posterior** $p_\theta(\mathbf{z} \mid \mathbf{x})$ represents the ideal inference distribution: given an observed data point $\mathbf{x}$, which latent variables $\mathbf{z}$ most likely generated it under our generative model? This is exactly what we want our encoder to learn. If we could compute the true posterior, we would know the best latent representation for each data point. However, computing it requires the intractable evidence $p_\theta(\mathbf{x})$ in the denominator. The variational distribution $q_\phi(\mathbf{z} \mid \mathbf{x})$ is our learned approximation to this ideal posterior. By training the encoder to approximate the true posterior, we ensure that the latent variables we sample are meaningful representations of the input data.

The variational inference framework tells us how to train both the encoder and decoder. We want to find parameters $\phi$ and $\theta$ such that:
1. The encoder $q_\phi(\mathbf{z} \mid \mathbf{x})$ closely approximates the true posterior $p_\theta(\mathbf{z} \mid \mathbf{x})$
2. The decoder $p_\theta(\mathbf{x} \mid \mathbf{z})$ can reconstruct data well from latent variables

### The ELBO Objective

The VAE training objective is derived using variational inference. Our goal is to train a generative model that assigns high probability to the observed data. In probabilistic modeling, we achieve this by maximizing the log-likelihood $\log p_\theta(\mathbf{x})$ of the training data under the model. This is the standard principle of **maximum likelihood estimation**: we adjust the model parameters $\theta$ to make the observed data as probable as possible. The log is used for mathematical convenience (products become sums) and numerical stability (avoiding underflow with small probabilities).

However, as we saw earlier, computing $p_\theta(\mathbf{x}) = \int p_\theta(\mathbf{x} \mid \mathbf{z}) p(\mathbf{z}) d\mathbf{z}$ is intractable. Since we cannot directly optimize the intractable log-likelihood $\log p_\theta(\mathbf{x})$, we instead maximize a tractable lower bound called the **Evidence Lower Bound (ELBO)**. The same bound is derived in general in the [Variational Inference section](/garden/ml/bayesian/variationalinference/#deriving-the-elbo), here we give the short self-contained derivation for the VAE.

{{< callout type="proof" title="Deriving the ELBO from the log-likelihood" >}}
The trick is to introduce the encoder $q_\phi(\mathbf{z} \mid \mathbf{x})$ as an auxiliary distribution and rewrite the log-likelihood as an expectation under it. Since $\log p_\theta(\mathbf{x})$ does not depend on $\mathbf{z}$, taking the expectation over $q_\phi(\mathbf{z} \mid \mathbf{x})$ leaves it unchanged,

$$
\log p_\theta(\mathbf{x}) = \mathbb{E}_{q_\phi(\mathbf{z} \mid \mathbf{x})}\big[\log p_\theta(\mathbf{x})\big]
$$

Now use Bayes' rule $p_\theta(\mathbf{x}) = \tfrac{p_\theta(\mathbf{x}, \mathbf{z})}{p_\theta(\mathbf{z} \mid \mathbf{x})}$ and multiply inside the logarithm by $\tfrac{q_\phi(\mathbf{z} \mid \mathbf{x})}{q_\phi(\mathbf{z} \mid \mathbf{x})} = 1$,

$$
\log p_\theta(\mathbf{x}) = \mathbb{E}_{q_\phi}\left[\log \frac{p_\theta(\mathbf{x}, \mathbf{z})}{p_\theta(\mathbf{z} \mid \mathbf{x})}\right] = \mathbb{E}_{q_\phi}\left[\log \frac{p_\theta(\mathbf{x}, \mathbf{z})}{q_\phi(\mathbf{z} \mid \mathbf{x})} \cdot \frac{q_\phi(\mathbf{z} \mid \mathbf{x})}{p_\theta(\mathbf{z} \mid \mathbf{x})}\right]
$$

Splitting the logarithm separates a tractable term from an intractable one,

$$
\log p_\theta(\mathbf{x}) = \underbrace{\mathbb{E}_{q_\phi}\left[\log \frac{p_\theta(\mathbf{x}, \mathbf{z})}{q_\phi(\mathbf{z} \mid \mathbf{x})}\right]}_{\mathcal{L}(\theta, \phi; \mathbf{x})} + \underbrace{\mathbb{E}_{q_\phi}\left[\log \frac{q_\phi(\mathbf{z} \mid \mathbf{x})}{p_\theta(\mathbf{z} \mid \mathbf{x})}\right]}_{\text{KL}(q_\phi(\mathbf{z} \mid \mathbf{x}) \parallel p_\theta(\mathbf{z} \mid \mathbf{x})) \ge 0}
$$

The second term is the KL divergence from the encoder to the true posterior, which is non-negative and which we cannot compute because $p_\theta(\mathbf{z} \mid \mathbf{x})$ is intractable. Dropping it gives the lower bound,

$$
\log p_\theta(\mathbf{x}) \ge \mathbb{E}_{q_\phi}\big[\log p_\theta(\mathbf{x}, \mathbf{z}) - \log q_\phi(\mathbf{z} \mid \mathbf{x})\big] := \mathcal{L}(\theta, \phi; \mathbf{x})
$$
{{< /callout >}}

This is the same decomposition written in terms of the KL divergence to the true posterior (see [KL divergence](/garden/ml/bayesian/variationalinference/#kullback-leibler-divergence) for details):

$$
\log p_\theta(\mathbf{x}) = \text{KL}(q_\phi(\mathbf{z} \mid \mathbf{x}) \parallel p_\theta(\mathbf{z} \mid \mathbf{x})) + \mathcal{L}(\theta, \phi; \mathbf{x})
$$

where the ELBO is:

$$
\mathcal{L}(\theta, \phi; \mathbf{x}) = \mathbb{E}_{\mathbf{z} \sim q_\phi(\mathbf{z} \mid \mathbf{x})} [\log p_\theta(\mathbf{x} \mid \mathbf{z})] - \text{KL}(q_\phi(\mathbf{z} \mid \mathbf{x}) \parallel p(\mathbf{z}))
$$

This **reconstruction minus regularization** form follows from the joint-form ELBO of the derivation above by factorizing $p_\theta(\mathbf{x}, \mathbf{z}) = p_\theta(\mathbf{x} \mid \mathbf{z})\, p(\mathbf{z})$ and regrouping,

$$
\mathcal{L}(\theta, \phi; \mathbf{x}) = \mathbb{E}_{q_\phi}\big[\log p_\theta(\mathbf{x}, \mathbf{z}) - \log q_\phi(\mathbf{z} \mid \mathbf{x})\big] = \underbrace{\mathbb{E}_{q_\phi}[\log p_\theta(\mathbf{x} \mid \mathbf{z})]}_{\text{reconstruction}} - \underbrace{\mathbb{E}_{q_\phi}\Big[\log \tfrac{q_\phi(\mathbf{z} \mid \mathbf{x})}{p(\mathbf{z})}\Big]}_{\text{KL}(q_\phi(\mathbf{z} \mid \mathbf{x}) \parallel p(\mathbf{z}))}
$$

The two equivalent forms of the ELBO play different roles. The KL-to-the-true-posterior form below shows *why* it is a lower bound, while this reconstruction minus regularization form is the one we actually optimize. To see why it is a lower bound, rearrange the first equation:

$$
\mathcal{L}(\theta, \phi; \mathbf{x}) = \log p_\theta(\mathbf{x}) - \text{KL}(q_\phi(\mathbf{z} \mid \mathbf{x}) \parallel p_\theta(\mathbf{z} \mid \mathbf{x}))
$$

Since the KL divergence is always non-negative (that is, $\text{KL}(q \parallel p) \geq 0$ for any distributions $q$ and $p$, with equality only when $q = p$), we can write:

$$
\log p_\theta(\mathbf{x}) = \mathcal{L}(\theta, \phi; \mathbf{x}) + \text{KL}(q_\phi(\mathbf{z} \mid \mathbf{x}) \parallel p_\theta(\mathbf{z} \mid \mathbf{x})) \geq \mathcal{L}(\theta, \phi; \mathbf{x})
$$

This shows that the ELBO is indeed a lower bound on the log-likelihood. The bound is tight when the KL divergence is zero, which happens when $q_\phi(\mathbf{z} \mid \mathbf{x}) = p_\theta(\mathbf{z} \mid \mathbf{x})$, meaning our variational distribution exactly matches the true posterior.

Maximizing the ELBO simultaneously:
1. Maximizes the (approximate) marginal likelihood $\log p_\theta(\mathbf{x})$
2. Minimizes the approximation error $\text{KL}(q_\phi(\mathbf{z} \mid \mathbf{x}) \parallel p_\theta(\mathbf{z} \mid \mathbf{x}))$

The ELBO decomposes into two interpretable terms. The first term $\mathbb{E}_{\mathbf{z} \sim q_\phi(\mathbf{z} \mid \mathbf{x})} [\log p_\theta(\mathbf{x} \mid \mathbf{z})]$ is the **reconstruction term**. It measures how well the decoder can reconstruct the input $\mathbf{x}$ from latent variables sampled from the encoder's distribution. This encourages the model to be informative by ensuring that the latent representation captures enough information to reconstruct the data.

In practice, for a Gaussian likelihood $p_\theta(\mathbf{x} \mid \mathbf{z}) = \mathcal{N}(\mathbf{x} \mid \boldsymbol{\mu}_\theta(\mathbf{z}), \sigma^2 \mathbf{I})$, where $\boldsymbol{\mu}_\theta(\mathbf{z})$ is the mean predicted by the decoder network $f_\theta(\mathbf{z})$, we can derive the reconstruction term explicitly. The log-probability of a Gaussian is:

$$
\log p_\theta(\mathbf{x} \mid \mathbf{z}) = \log \mathcal{N}(\mathbf{x} \mid \boldsymbol{\mu}_\theta(\mathbf{z}), \sigma^2 \mathbf{I}) = -\frac{D}{2}\log(2\pi\sigma^2) - \frac{1}{2\sigma^2}\|\mathbf{x} - \boldsymbol{\mu}_\theta(\mathbf{z})\|^2
$$

where $D$ is the data dimensionality. The first term is a constant that does not depend on the parameters $\theta$ or $\phi$, so when optimizing, we can ignore it. Taking the expectation:

$$
\mathbb{E}_{\mathbf{z} \sim q_\phi(\mathbf{z} \mid \mathbf{x})} [\log p_\theta(\mathbf{x} \mid \mathbf{z})] = -\frac{D}{2}\log(2\pi\sigma^2) - \frac{1}{2\sigma^2}\mathbb{E}_{\mathbf{z} \sim q_\phi(\mathbf{z} \mid \mathbf{x})} [\|\mathbf{x} - \boldsymbol{\mu}_\theta(\mathbf{z})\|^2]
$$

This is proportional to the negative mean squared error (MSE) between the input $\mathbf{x}$ and the decoder's predicted mean $\boldsymbol{\mu}_\theta(\mathbf{z})$:

$$
\mathbb{E}_{\mathbf{z} \sim q_\phi(\mathbf{z} \mid \mathbf{x})} [\log p_\theta(\mathbf{x} \mid \mathbf{z})] \propto -\mathbb{E}_{\mathbf{z} \sim q_\phi(\mathbf{z} \mid \mathbf{x})} [\|\mathbf{x} - \boldsymbol{\mu}_\theta(\mathbf{z})\|^2]
$$

Thus maximizing the reconstruction term is equivalent to minimizing the MSE between inputs and reconstructions.

The second term $\text{KL}(q_\phi(\mathbf{z} \mid \mathbf{x}) \parallel p(\mathbf{z}))$ is the **regularization term**. It measures how much the approximate posterior (encoder distribution) deviates from the prior. This term has several beneficial effects. It prevents the encoder from simply learning an arbitrary encoding by forcing the latent variables to follow a known, regular distribution (the prior). It promotes disentanglement by encouraging the latent dimensions to be approximately independent, since the prior is typically a factorized Gaussian. Finally, it enables generation by ensuring we can sample novel latent variables from the prior distribution $p(\mathbf{z}) = \mathcal{N}(0, \mathbf{I})$ and decode them to get realistic data.

The balance between these two terms is crucial. The reconstruction term alone would lead to overfitting and pathological solutions where the encoder learns to ignore the prior, mapping each input to a unique but arbitrary point in latent space. The KL term alone would force the posterior to match the prior exactly, losing all information about the input. The VAE objective elegantly balances these competing objectives, learning a latent space that is both informative (captures data structure) and regular (follows a known prior).

For the special case where both $q_\phi(\mathbf{z} \mid \mathbf{x}) = \mathcal{N}(\boldsymbol{\mu}, \sigma^2 \mathbf{I})$ and $p(\mathbf{z}) = \mathcal{N}(0, \mathbf{I})$ are diagonal Gaussians, the KL divergence has a closed-form expression (see [Variational Inference](/garden/ml/bayesian/variationalinference/#elbo-for-bayesian-logistic-regression) for the derivation):

$$
\text{KL}(q_\phi(\mathbf{z} \mid \mathbf{x}) \parallel p(\mathbf{z})) = \frac{1}{2} \sum_{j=1}^d \left( \mu_j^2 + \sigma_j^2 - \log \sigma_j^2 - 1 \right)
$$

where $d$ is the dimensionality of the latent space. This closed-form expression provides significant computational benefits. Without it, we would need to approximate the KL divergence using Monte Carlo sampling: drawing multiple samples from $q_\phi(\mathbf{z} \mid \mathbf{x})$ and computing the average of $\log q_\phi(\mathbf{z} \mid \mathbf{x}) - \log p(\mathbf{z})$. This would be slow (requiring many samples for accurate estimates), noisy (introducing high variance in gradients), and wasteful (sampling when an exact formula exists). The closed-form expression gives us the exact KL divergence value instantly using only the encoder outputs $\boldsymbol{\mu}_\phi(\mathbf{x})$ and $\boldsymbol{\sigma}_\phi(\mathbf{x})$, with no sampling or approximation error. This is one reason why Gaussian distributions are commonly chosen for both the prior and the variational distribution in VAEs.

### The Reparameterization Trick

To train the VAE, we need to compute gradients of the ELBO with respect to both the decoder parameters $\theta$ and the encoder parameters $\phi$. These two gradients are not equally difficult, and seeing why isolates exactly where the problem is.

The **decoder gradient is easy**. The expectation in the ELBO is taken over $q_\phi(\mathbf{z} \mid \mathbf{x})$, which does not depend on $\theta$, so the gradient passes straight inside the expectation and a single Monte Carlo sample suffices,

$$
\nabla_\theta\, \mathcal{L}(\theta, \phi; \mathbf{x}) = \mathbb{E}_{q_\phi(\mathbf{z} \mid \mathbf{x})}\big[\nabla_\theta \log p_\theta(\mathbf{x} \mid \mathbf{z})\big] \approx \nabla_\theta \log p_\theta(\mathbf{x} \mid \mathbf{z}), \quad \mathbf{z} \sim q_\phi(\mathbf{z} \mid \mathbf{x})
$$

Here $\mathbf{z}$ is treated as a fixed sample, and only the term $\log p_\theta(\mathbf{x} \mid \mathbf{z})$ depends on $\theta$.

The **encoder gradient is the hard one**. The reconstruction term is an expectation over $q_\phi(\mathbf{z} \mid \mathbf{x})$, and now the distribution we are averaging over *itself* depends on the parameter $\phi$ we are differentiating. We therefore cannot move $\nabla_\phi$ inside the expectation, and we cannot backpropagate through the stochastic step $\mathbf{z} \sim q_\phi(\mathbf{z} \mid \mathbf{x})$ because drawing a sample is not a differentiable operation with respect to $\phi$. The naive approach would be to approximate the expectation using Monte Carlo sampling:

$$
\mathbb{E}_{\mathbf{z} \sim q_\phi(\mathbf{z} \mid \mathbf{x})} [\log p_\theta(\mathbf{x} \mid \mathbf{z})] \approx \frac{1}{L} \sum_{l=1}^L \log p_\theta(\mathbf{x} \mid \mathbf{z}^{(l)})
$$

where $\mathbf{z}^{(l)} \sim q_\phi(\mathbf{z} \mid \mathbf{x})$. As noted above, this estimator blocks the gradient, since the samples $\mathbf{z}^{(l)}$ are drawn from a distribution that depends on $\phi$ in a non-differentiable way.

The key insight is to use the **reparameterization trick** to reparameterize the random variable $\mathbf{z}$ as a deterministic function of $\phi$, $\mathbf{x}$, and an auxiliary noise variable $\boldsymbol{\epsilon}$ that does not depend on $\phi$. For a Gaussian variational distribution $q_\phi(\mathbf{z} \mid \mathbf{x}) = \mathcal{N}(\mathbf{z} \mid \boldsymbol{\mu}_\phi(\mathbf{x}), \boldsymbol{\sigma}_\phi^2(\mathbf{x}) \mathbf{I})$, we can express a sample $\mathbf{z}$ as:

$$
\mathbf{z} = \boldsymbol{\mu}_\phi(\mathbf{x}) + \boldsymbol{\sigma}_\phi(\mathbf{x}) \odot \boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(0, \mathbf{I})
$$

where $\odot$ denotes element-wise multiplication. We use element-wise (Hadamard) multiplication because all quantities are vectors: $\boldsymbol{\mu}_\phi(\mathbf{x}) \in \mathbb{R}^d$, $\boldsymbol{\sigma}_\phi(\mathbf{x}) \in \mathbb{R}^d$, and $\boldsymbol{\epsilon} \in \mathbb{R}^d$ where $d$ is the latent space dimensionality. Each dimension $j$ of the latent space has its own mean $\mu_j$ and standard deviation $\sigma_j$, and we sample independent noise $\epsilon_j$ for each dimension. The element-wise multiplication $\boldsymbol{\sigma}_\phi(\mathbf{x}) \odot \boldsymbol{\epsilon}$ scales the noise in each dimension by the corresponding standard deviation: $z_j = \mu_j + \sigma_j \cdot \epsilon_j$ for $j = 1, \ldots, d$. This reflects the fact that our variational distribution is a diagonal Gaussian where each dimension is independent with its own variance.

The noise $\boldsymbol{\epsilon}$ is sampled from a fixed distribution that does not depend on the parameters $\phi$. The randomness in $\mathbf{z}$ comes from $\boldsymbol{\epsilon}$, but the transformation from $\boldsymbol{\epsilon}$ to $\mathbf{z}$ is deterministic and differentiable with respect to $\phi$.

Using this reparameterization, we can rewrite the expectation as:

$$
\mathbb{E}_{\mathbf{z} \sim q_\phi(\mathbf{z} \mid \mathbf{x})} [\log p_\theta(\mathbf{x} \mid \mathbf{z})] = \mathbb{E}_{\boldsymbol{\epsilon} \sim \mathcal{N}(0, \mathbf{I})} [\log p_\theta(\mathbf{x} \mid \boldsymbol{\mu}_\phi(\mathbf{x}) + \boldsymbol{\sigma}_\phi(\mathbf{x}) \odot \boldsymbol{\epsilon})]
$$

Now the expectation is over a fixed distribution $\mathcal{N}(0, \mathbf{I})$, and we can approximate it by sampling $\boldsymbol{\epsilon}$ and computing gradients through the deterministic function $g_\phi(\mathbf{x}, \boldsymbol{\epsilon}) = \boldsymbol{\mu}_\phi(\mathbf{x}) + \boldsymbol{\sigma}_\phi(\mathbf{x}) \odot \boldsymbol{\epsilon}$. The Monte Carlo estimate becomes:

$$
\mathbb{E}_{\boldsymbol{\epsilon} \sim \mathcal{N}(0, \mathbf{I})} [\log p_\theta(\mathbf{x} \mid g_\phi(\mathbf{x}, \boldsymbol{\epsilon}))] \approx \frac{1}{L} \sum_{l=1}^L \log p_\theta(\mathbf{x} \mid g_\phi(\mathbf{x}, \boldsymbol{\epsilon}^{(l)}))
$$

where $\boldsymbol{\epsilon}^{(l)} \sim \mathcal{N}(0, \mathbf{I})$. This expression is fully differentiable with respect to both $\theta$ and $\phi$, allowing us to use standard backpropagation to compute gradients. In practice, the original Kingma and Welling paper found that a single sample ($L=1$) per data point is sufficient when using mini-batch SGD with reasonably large batch sizes. While more samples ($L > 1$) would provide a better estimate of the expectation, the mini-batch itself provides enough variance reduction to make single-sample estimates adequate for optimization. This makes the algorithm computationally efficient.

{{< callout type="example" title="Reparameterization in Practice" >}}
Consider encoding an image of a face. The encoder outputs $\boldsymbol{\mu}_\phi(\mathbf{x}) = [0.5, -0.3, 1.2]$ and $\boldsymbol{\sigma}_\phi(\mathbf{x}) = [0.1, 0.2, 0.15]$ for a three-dimensional latent space.

To sample a latent code, we first sample noise $\boldsymbol{\epsilon} \sim \mathcal{N}(0, \mathbf{I})$, say $\boldsymbol{\epsilon} = [0.8, -1.2, 0.3]$.

Then we compute: $\mathbf{z} = \boldsymbol{\mu}_\phi(\mathbf{x}) + \boldsymbol{\sigma}_\phi(\mathbf{x}) \odot \boldsymbol{\epsilon} = [0.5, -0.3, 1.2] + [0.1, 0.2, 0.15] \odot [0.8, -1.2, 0.3] = [0.58, -0.54, 1.245]$

This $\mathbf{z}$ is then passed through the decoder. During backpropagation, gradients flow through the multiplication and addition operations back to the encoder parameters that produced $\boldsymbol{\mu}_\phi(\mathbf{x})$ and $\boldsymbol{\sigma}_\phi(\mathbf{x})$.
{{< /callout >}}

### Training

Training a VAE involves jointly optimizing the encoder and decoder parameters to maximize the ELBO over the training dataset. For a dataset $\mathcal{D} = \{\mathbf{x}_1, \ldots, \mathbf{x}_n\}$, the objective is:

$$
\max_{\theta, \phi} \sum_{i=1}^n \mathcal{L}(\theta, \phi; \mathbf{x}_i)
$$

Using the decomposition of the ELBO and the reparameterization trick, the objective for a single data point becomes:

$$
\mathcal{L}(\theta, \phi; \mathbf{x}) = \mathbb{E}_{\boldsymbol{\epsilon} \sim \mathcal{N}(0, \mathbf{I})} [\log p_\theta(\mathbf{x} \mid g_\phi(\mathbf{x}, \boldsymbol{\epsilon}))] - \text{KL}(q_\phi(\mathbf{z} \mid \mathbf{x}) \parallel p(\mathbf{z}))
$$

In practice, we use stochastic gradient ascent (rather than descent) because we are maximizing the ELBO rather than minimizing a loss. For each mini-batch of data points $\{\mathbf{x}^{(1)}, \ldots, \mathbf{x}^{(M)}\}$ sampled from the dataset, we perform the following steps. For each data point $\mathbf{x}^{(i)}$, we compute the encoder outputs $\boldsymbol{\mu}_\phi(\mathbf{x}^{(i)})$ and $\boldsymbol{\sigma}_\phi(\mathbf{x}^{(i)})$. We then sample noise $\boldsymbol{\epsilon}^{(i)} \sim \mathcal{N}(0, \mathbf{I})$ and compute the latent code $\mathbf{z}^{(i)} = \boldsymbol{\mu}_\phi(\mathbf{x}^{(i)}) + \boldsymbol{\sigma}_\phi(\mathbf{x}^{(i)}) \odot \boldsymbol{\epsilon}^{(i)}$. Next, we compute the reconstruction log-likelihood $\log p_\theta(\mathbf{x}^{(i)} \mid \mathbf{z}^{(i)})$ and the KL divergence $\text{KL}(q_\phi(\mathbf{z} \mid \mathbf{x}^{(i)}) \parallel p(\mathbf{z}))$ using the closed-form expression. After processing all data points in the mini-batch, we compute the average ELBO and take a gradient step to maximize it.

For the reconstruction term, the specific form of $\log p_\theta(\mathbf{x} \mid \mathbf{z})$ depends on the type of data. For continuous data like images with pixel values in $[0, 1]$, a common choice is a Gaussian likelihood with identity covariance:

$$
\log p_\theta(\mathbf{x} \mid \mathbf{z}) = -\frac{1}{2} \|\mathbf{x} - \boldsymbol{\mu}_\theta(\mathbf{z})\|_2^2 + \text{const}
$$

This corresponds to a mean squared error (MSE) reconstruction loss.

### Sampling and Generation

Once trained, a VAE can be used in several ways. For **encoding** an existing data point, we pass it through the encoder to get the distribution parameters $\boldsymbol{\mu}_\phi(\mathbf{x})$ and $\boldsymbol{\sigma}_\phi(\mathbf{x})$, then sample $\mathbf{z} \sim \mathcal{N}(\boldsymbol{\mu}_\phi(\mathbf{x}), \boldsymbol{\sigma}_\phi^2(\mathbf{x}) \mathbf{I})$ to get a latent code. We could also just use the mean $\boldsymbol{\mu}_\phi(\mathbf{x})$ as a deterministic encoding.

For **reconstruction**, we encode the input to get $\mathbf{z}$, then decode it through $\boldsymbol{\mu}_\theta(\mathbf{z})$ to get the reconstruction $\hat{\mathbf{x}}$. The reconstruction will generally not be identical to the input due to the information bottleneck and the stochasticity in the encoder.

For **generation** of new samples, this is where VAEs truly shine compared to traditional autoencoders. We sample a latent code from the prior $\mathbf{z} \sim \mathcal{N}(0, \mathbf{I})$ and pass it through the decoder to get $\hat{\mathbf{x}} = \boldsymbol{\mu}_\theta(\mathbf{z})$. Since the training process ensures that the latent space is regularized to match this prior (through the KL term), samples from the prior should decode to plausible data points. This is a key advantage: we can generate infinite new samples without ever seeing them in the training data.

For **interpolation** between two data points, we can encode them to get latent codes $\mathbf{z}_1$ and $\mathbf{z}_2$, then interpolate in latent space: $\mathbf{z}_t = (1-t) \mathbf{z}_1 + t \mathbf{z}_2$ for $t \in [0, 1]$. Decoding these interpolated latent codes produces a smooth transition between the original data points. The continuous and structured nature of the latent space (enforced by the KL regularization) allows for smooth interpolation. Unlike traditional autoencoders where the latent space can have "holes" or discontinuities, VAE latent spaces are dense and well-behaved.

The ability to generate new samples by sampling from the prior is the fundamental capability that distinguishes VAEs from traditional autoencoders. This generative capability comes from the probabilistic formulation and the KL regularization term that ensures the latent space has a known, regular structure.

### Controlling Generation

While basic generation involves sampling $\mathbf{z} \sim \mathcal{N}(0, \mathbf{I})$, we can exert some control over what is generated. One approach is **conditional generation**. If we train a **conditional VAE** where both the encoder $q_\phi(\mathbf{z} \mid \mathbf{x}, c)$ and decoder $p_\theta(\mathbf{x} \mid \mathbf{z}, c)$ are conditioned on some additional information $c$ (like a class label), we can generate samples from a specific class by sampling $\mathbf{z} \sim \mathcal{N}(0, \mathbf{I})$ and decoding with $\boldsymbol{\mu}_\theta(\mathbf{z}, c)$ where $c$ is the desired class.

Another approach is **latent space manipulation**. If the latent dimensions are disentangled, we can modify specific dimensions to change specific attributes. For example, in a VAE trained on faces, we might find that dimension 3 controls "smiling" while dimension 7 controls "age". We can then encode a face, modify dimension 3, and decode to get a smiling version of the same face. Whether this works at all depends on what the latent code has actually learned, which is the subject of the next section.

### Factors of Variation and Disentanglement

Why should a single latent dimension correspond to something interpretable like "smiling" in the first place? The answer lies in the ELBO trade-off. The encoder compresses a high-dimensional image $\mathbf{x} \in \mathbb{R}^{H \times W \times 3}$ into a code $\mathbf{z} \in \mathbb{R}^d$ with $d \ll H W$, and the two ELBO terms pull on that code in opposite ways. The reconstruction term forces $\mathbf{z}$ to retain enough information to rebuild $\mathbf{x}$, while the KL term pulls $\mathbf{z}$ toward the smooth, factorized prior $\mathcal{N}(0, \mathbf{I})$. Caught between the two, $\mathbf{z}$ cannot afford to store pixel-level detail. It is pushed to capture the few underlying **factors of variation** that actually explain the data.

This connects to a broader hypothesis: real-world data is generated by a small number of independent underlying causes. For faces, a handful of factors (pose, expression, hair, lighting) explain most of the variation. The aspiration for a trained VAE is that each dimension of $\mathbf{z}$ captures one such factor, so that moving along a single axis changes one attribute and leaves the others fixed.

A representation with this property is called **disentangled**: different coordinates of $\mathbf{z}$ encode different, independent factors. In an **entangled** representation the axes are mixed, so a single coordinate changes several attributes at once (rotation *and* smile, say). There is a simple test for disentanglement:

1. Fix a point $\mathbf{z}$ in latent space, corresponding to a specific generated example such as one face.
2. Vary a single coordinate $z_i$ while holding all others fixed.
3. Decode and observe whether exactly one interpretable attribute changes.

Vanilla VAEs do **not** guarantee disentanglement. The factorized prior nudges the latent dimensions toward being uncorrelated, but nothing in the objective forces each axis onto a distinct semantic factor. Encouraging disentanglement explicitly is the motivation for $\beta$-VAE.

### beta-VAE

While the standard VAE objective encourages some degree of disentanglement through the KL regularization, it does not guarantee that different latent dimensions will correspond to independent factors of variation. **$\beta$-VAE** addresses this by modifying the ELBO to place stronger emphasis on the KL term:

$$
\mathcal{L}_{\beta}(\theta, \phi; \mathbf{x}) = \mathbb{E}_{\mathbf{z} \sim q_\phi(\mathbf{z} \mid \mathbf{x})} [\log p_\theta(\mathbf{x} \mid \mathbf{z})] - \beta \cdot \text{KL}(q_\phi(\mathbf{z} \mid \mathbf{x}) \parallel p(\mathbf{z}))
$$

where $\beta > 1$ applies stronger regularization. Higher $\beta$ values encourage the approximate posterior $q_\phi(\mathbf{z} \mid \mathbf{x})$ to be closer to the prior $p(\mathbf{z}) = \mathcal{N}(0, \mathbf{I})$, which is factorized across dimensions. This encourages the latent dimensions to be more independent, leading to better disentanglement.

The trade-off is that higher $\beta$ values reduce reconstruction quality, as the model is forced to compress information more aggressively. The choice of $\beta$ depends on the application. For generation quality, $\beta = 1$ (standard VAE) may be best. For learning interpretable representations where we want to manipulate individual factors, higher $\beta$ values (e.g., $\beta = 4$ or more) may be preferable despite the reconstruction quality loss.

{{< callout type="info" title="What VAEs give us, and what stays open" >}}

- **Strengths:** principled likelihood-based training through the ELBO, a structured latent space that supports smooth interpolation and latent arithmetic, and latent dimensions that can capture interpretable factors of variation.
- **Open issues:** samples are often blurry, because a Gaussian decoder averages over the modes consistent with a code rather than committing to one. Disentanglement is not guaranteed and needs variants such as $\beta$-VAE or stronger priors. And the ELBO is only a lower bound, so there is an unmeasured gap between it and the true log-likelihood.

These limitations motivate the other generative families in this garden. [Generative adversarial networks](/garden/ml/computerVision/gan) drop the likelihood entirely and optimize an adversarial objective, trading the blurry-sample problem for sharper outputs at the cost of harder training. [Diffusion models](/garden/ml/computerVision/diffusion) keep a likelihood-based view but replace the single bottleneck with a long chain of denoising steps, currently giving the best sample quality. Common extensions of the VAE itself include VQ-VAE (discrete latents), hierarchical VAEs, and normalizing-flow priors.
{{< /callout >}}
