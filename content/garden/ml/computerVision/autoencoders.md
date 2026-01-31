---
title: Autoencoders
type: docs
weight: 2
---

Deep learning excels at learning meaningful representations of data. Neural networks can be viewed as compositions of encoders that extract features and estimators that make predictions from those features. **Autoencoders** provide a framework for learning these representations in an unsupervised manner, without requiring labeled data. The core idea is deceptively simple: train a network to reconstruct its input through a bottleneck that forces it to learn a compressed representation.

## Traditional Autoencoders

An autoencoder is a neural network architecture consisting of two main components. The **encoder** $g_\phi: \mathbb{R}^D \to \mathbb{R}^d$ maps high-dimensional input data $x \in \mathbb{R}^D$ to a lower-dimensional latent representation $z \in \mathbb{R}^d$ where typically $d \ll D$. The **decoder** $f_\theta: \mathbb{R}^d \to \mathbb{R}^D$ attempts to reconstruct the original input from this compressed representation.

$$
z = g_\phi(x), \quad \hat{x} = f_\theta(z) = f_\theta(g_\phi(x))
$$

{{< figure
    src="/images/ml/autoencoder.png"
    alt="Traditional autoencoder architecture with encoder, latent representation, and decoder."
    caption="An autoencoder compresses input through an encoder to a latent representation, then reconstructs it through a decoder."
>}}

The network is trained to minimize the reconstruction error between the input $x$ and its reconstruction $\hat{x}$. For continuous data, this is typically the mean squared error:

$$
\mathcal{L}(\theta, \phi) = \frac{1}{n}\sum_{i=1}^n \|x_i - f_\theta(g_\phi(x_i))\|^2
$$

The bottleneck structure (where $d \ll D$) forces the encoder to learn a compressed representation that captures the most important features of the data. This is similar in spirit to principal component analysis (PCA) which reduces dimensionality using linear projections, but autoencoders can learn non-linear representations through the use of non-linear activation functions.

### What Makes a Good Representation?

Before diving deeper, we should understand what makes a representation "good". A good representation should satisfy three key properties.

- **Informative:** Given a representation, it should be possible to reconstruct the original input with high fidelity. If the representation discards too much information, it becomes useless for downstream tasks. This property ensures that the latent code captures the essential structure of the data.

- **Disentangled:** Each component in the representation should correspond to a distinct, interpretable feature of the data. For example, in face images, different dimensions of the latent space might independently control attributes like age, gender, or expression. Disentanglement makes the representation more interpretable and easier to manipulate for specific purposes.

- **Robust:** Small perturbations or noise in the input should not drastically change the representation, and conversely, small changes in the representation should lead to smooth, meaningful changes in the reconstructed output. This robustness property is crucial for generalization and prevents the model from memorizing noise in the training data.

Traditional autoencoders are designed primarily to be informative through the reconstruction objective. However, they struggle with the other two properties. The latent space learned by a standard autoencoder can be irregular and discontinuous, making it difficult to generate new samples or interpolate meaningfully. Different regions of the latent space may correspond to very different data distributions, and some regions may not correspond to any valid data at all.

### The Infomax Principle

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
