---
title: Diffusion Models
type: docs
weight: 4
---

A common goal in machine learning is to learn generative models that can produce new data samples that closely resemble a given dataset or in other words that can produce new and realistic samples matching the world we know. So given data samples ${x_1, x_2, ..., x_N}$ drawn from an unknown data distribution $q$ on $R^d$, the goal is to learn a model $p_\theta$ parametrized by $\theta$ that can generate new samples $\hat{x} \sim p_\theta$ such that the distribution of generated samples closely approximates the true data distribution, so $p_\theta \approx q$. 

For example in the case of images, our data samples could be all images of the internet and we want to learn a model that can generate new images that look like real images. So our underlying data distribution $q$ and the dimension $d$ is very high (e.g. if we consider $64 \times 64$ black and white images then $d = 64 \times 64 = 4,096$. Note that the pixel values are then usually scaled to then be in $[-1,1]$ not $[0,255]$). This $d$ dimensional space is also called the **pixel space** where each dimension corresponds to the intensity value of a pixel in the image, but in general our data distribution could be anything such as videos such as in [Video Diffusion Models](https://arxiv.org/abs/2204.03458) or proteins such as in [RF Diffusion](https://www.nature.com/articles/s41586-023-06415-8).

{{< figure 
    src="/images/ml/diffusionFluxExample.png"
    alt="Image generated using open source black-forest-labs/FLUX.1-dev model with some extra addons."
    caption="Image generated using open source black-forest-labs/FLUX.1-dev model with some extra addons."
    width="300"
>}}

Modeling such high dimensional distributions is very challenging due to the **curse of dimensionality** and the complex structure of real world data. The high dimensional space is also mostly empty with regards to our region of interest, also referred to as the **data manifold** (e.g. natural images) making it difficult to learn meaningful patterns. In the case of images, an easy way to think of this is that we are only interested in pictures of humans with say 2 eyes, 2 arms and 2 legs. But the pixel space also contains all sorts of other images that do not correspond to real humans such as images with 3 eyes or 5 arms etc. which do not exist in the real world. So the data distribution $q$ is concentrated on a very small area/manifold within the high dimensional pixel space.

This is also related to the **data manifold hypothesis** which states that real world high dimensional data such as images, audio or text actually lie on a low dimensional manifold embedded within the high dimensional space, the so called **ambient space**. This means that although the data lives in a high dimensional space, the intrinsic dimensionality of the data is much lower due to the underlying structure and correlations present in real world data. For example, natural images have strong spatial correlations and patterns that can be exploited to represent them more efficiently. This has important implications for machine learning as it suggests that we can learn more efficient representations and models by focusing on the low dimensional manifold rather than the entire high dimensional space.

{{< figure 
    src="/images/ml/diffusionDataManifold.png"
    alt="Illustration of the relative sparse nature of high dimensional spaces and its dense data manifold projected in 2D. Visualization is from the \"Swiss Roll\" problem."
    caption="Illustration of the relative sparse nature of high dimensional spaces and its dense data manifold projected in 2D. Visualization is from the \"Swiss Roll\" problem."
>}}

## Denoising Diffusion Probabilistic Models

As is common in computer science and machine learning, it is often easier to solve a complex problem by breaking it down into smaller subproblems. In the case of generative modeling, instead of learning to directly generate samples from the complex data distribution $q$, we can instead learn to gradually improve our samples in an iterative/recursive manner to reach the desired result. Think of it as painting a picture step by step, starting from a blank canvas and adding more and more details and layering colors of paint until the final masterpiece is complete, rather than trying to paint the entire picture in one go. 

The idea of diffusion models is to use a stochastic process to gradually transform simple known distributions such as Gaussian noise into our complex data distribution $q$. **The main idea behind diffusion models is to define a forward diffusion process that gradually adds noise to the data samples until they become pure noise, and then learn a neural network to reverse this process and denoise the noisy samples back to the original data distribution.** By iteratively applying this denoising process starting from a pure noise sample, we can generate new samples that closely resemble samples from the true data distribution.

{{< callout type="info" title="Used Resources" >}}

| Type | Author/Creator | Resource |
|------|----------------|----------|
| Paper | Ho et al. | [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) |
| Paper | Dhariwal & Nichol | [Diffusion Models Beat GANs on Image Synthesis](https://arxiv.org/abs/2105.05233) |
| Paper | Dhariwal & Nichol | [Improved Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2102.09672) |
| Paper | Song et al. | [Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502) |
| Paper | Rombach et al. | [High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752) |
| Repo | diff-usion | [Awesome Diffusion Models Resources](https://github.com/diff-usion/Awesome-Diffusion-Models) |
| Article | Karagiannakos & Adaloglou | [How diffusion models work: the math from scratch](https://theaisummer.com/diffusion-models/) |
| Article | Ayan Das | [An introduction to Diffusion Probabilistic Models](https://ayandas.me/blogs/2021-12-04-diffusion-prob-models.html) |
| Article | Lilian Weng | [What are Diffusion Models?](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/) |
| Article | Pramod Goyal | [Demystifying Diffusion Models](https://goyalpramod.github.io/blogs/demysitifying_diffusion_models/) |
| Article | Yuan & Permenter | [Diffusion models from scratch, from a new theoretical perspective](https://www.chenyang.co/diffusion.html) |
| Article | Rogge & Rasul | [Annotated Diffusion Models](https://huggingface.co/blog/annotated-diffusion) |
| Article | Sander Dieleman | [Generative Modeling in latent space](https://sander.ai/2025/04/15/latents.html) |
| Video | Outlier | [Diffusion Models - Paper Explanation - Math Explained](https://youtu.be/HoKDTa5jHvg?si=IplGAGuURk3X95CN) |
| Video | deepia | [Diffusion Models: DDPM - Generative AI Animated](https://youtu.be/EhndHhIvWWw?si=5YBshM9vLifM7TfJ) |
| Video | 3Blue1Brown | [But how do AI images and videos actually work?](https://youtu.be/iv-5mZ_9CPY?si=cRoStGlbHANAPTCb) |
| Video | AI Coffee Break with Letitia | [How does Stable Diffusion work? - Latent Diffusion Models EXPLAINED](https://youtu.be/J87hffSMB60?si=eqMDzsmW1i457lmw) |
| Video | Depth First | [More Than Image Generators: A Science of Problem-Solving using Probability | Diffusion Model](https://youtu.be/Fk2I6pa6UeA?si=W2cLiuQAx-f5PZKm) |

{{< /callout >}}

### Forward Diffusion Process

First we define the forward diffusion process which gradually "perturbs/destroys" the data samples and moves it to our simple known distribution. In our case we will use a gaussian distribution as our simple known distribution. 

We then define the forwards diffusion process as a **markov chain** that adds small amounts of gaussian noise to the data samples over $T$ time steps. So if we let $x_0$ be a data sample drawn from our true data distribution $x_0 \sim q(x)$, then the forward diffusion process results in a sequence of noisy samples $x_1, x_2, ..., x_T$ where each $x_t$ is obtained by adding gaussian noise to the previous sample $x_{t-1}$. 

{{< figure 
    src="/images/ml/diffusionForward.png"
    alt="Illustration of the forward diffusion process gradually adding noise to a data sample over T time steps."
    caption="Illustration of the forward diffusion process gradually adding noise to a data sample over T time steps."
>}}

{{< figure 
    src="/images/ml/diffusionForwardAnimation.gif"
    alt="Animation showing the forward diffusion process gradually adding noise to a toy distribution."
    caption="Animation showing the forward diffusion process gradually adding noise to a toy distribution."
>}}

The amount of noise added at each time step is controlled by the so called **variance or noise schedule** $\beta_1, \beta_2, ..., \beta_T$ where each $\beta_t$ is a small positive value as variance must be positive. The choice of the noise schedule and this particular construction of the forward process is an algorithmic design choice that has been found to work well in practice for diffusion models. However, it can also be linked to physics and the concept of Langevin dynamics which we will discuss later. Usually the noise schedule is chosen to be some monotonic increasing function in the range of $[0, 1]$ such that more noise is added at later time steps. A common choice is to have an isotropic gaussian, so the covariance matrix is a scaled identity matrix $\beta_t I$ where $I$ is the identity matrix. The schedule could then be a linear schedule where $\beta_t$ increases linearly from a small value (e.g. 0.0001) to a larger value (e.g. 0.02) over $T$ time steps or a cosine schedule as proposed in [Improved Denoising Diffusion Probabilistic Models by OpenAI](https://arxiv.org/abs/2102.09672).

{{< figure 
    src="/images/ml/diffusionNoiseSchedule.png"
    alt="Example of linear (top) and cosine (bottom) noise schedules used in diffusion models."
    caption="Example of linear (top) and cosine (bottom) noise schedules used in diffusion models."
>}}

The reasoning for why the cosine schedule may work better is because intuitively we can see that in the beginning we don't want the image to be destroyed too quickly as we want to retain most of the structure of the image in the early stages of the diffusion process. So we want to add noise slowly in the beginning and then more rapidly towards the end when the image is already mostly destroyed. 

So we can define the forward diffusion process of gradually adding noise as:

$$
x_t = \sqrt{1 - \beta_t} x_{t-1} + \epsilon_t \text{ where } \epsilon_t \sim N(0, \beta_t I)
$$

Note that we scale down the previous sample $x_{t-1}$ by $\sqrt{1 - \beta_t}$ to ensure that the overall variance of $x_t$ does not explode but instead has the desired effect of gradually adding noise to the data sample and forcing the original signal to decay over time. Because the variables are independent they can be added and we can show this explosion of variance more formally by calculating the variance of $x_t$:

$$
\text{Var}(x_t) = \text{Var}(x_{t-1}) + \text{Var}(\epsilon_t) = \text{Var}(x_{t-1}) + \beta_t I
$$

Because variance is positive, over time it accumulates and can become very large. By scaling down $x_{t-1}$ we ensure that the variance remains stable or in other preserved hence it is often referred to as **variance preserving**, especially if $\text{Var}(x_{t-1}) = I$ then it is clear that the variance stays constant:

$$
\text{Var}(x_t) = (1 - \beta_t) \text{Var}(x_{t-1}) + \beta_t I 
= (1 - \beta_t) I + \beta_t I = I
$$

Importantly because this is a markov chain, each sample $x_t$ only depends on the previous sample $x_{t-1}$ and not on any earlier samples, so we have:

$$
q(x_t | x_{t-1}, x_{t-2}, ..., x_0) = q(x_t | x_{t-1})
$$

which also means we can write describe the entire **forward diffusion process as a joint distribution** over the entire sequence of samples from time step 1 to T. We can use the product rule of probability and the markov property to write the joint distribution as:

$$
q(x_{1:T} | x_0) = \prod_{t=1}^{T} q(x_t | x_{t-1})
$$

where $x_{1:T}$ denotes the sequence of noisy samples from time step 1 to $T$ that we obtain by applying the forward diffusion process to the original data sample $x_0$. Using the recurrence relation defined above for the forward diffusion process, we can also write the **conditional distribution at each time step as a gaussian distribution**:

$$
q(x_t | x_{t-1}) = N(x_t; \sqrt{1 - \beta_t} x_{t-1}, \beta_t I)
$$

where we evaluate the gaussian at $x_t$ with mean $\sqrt{1 - \beta_t} x_{t-1}$ and covariance $\beta_t I$. The mean represents the scaled down previous sample while the covariance represents the amount of noise added at this time step. 

Because the forward diffusion process is a linear gaussian markov process(a markov process composed of linear transformations and gaussian noise), we can also **derive a closed form expression for the distribution of $x_t$ given the original data sample $x_0$ by unrolling the recurrence relation and using the reparameterization trick**. Before unrolling the relation, we first define some notation to make the equations cleaner:

- We define $\alpha_t = 1 - \beta_t$ which represents the scaling factor applied to the previous sample at time step t.
- We define $\bar{\alpha}_t = \prod_{s=1}^{t} \alpha_s$ as the cumulative product of the $\alpha_t$ values up to time step t.
- The gaussian noise added at each time step $\epsilon_t \sim N(0, \beta_t I)$ can be equivalently written as $\epsilon_t = \sqrt{\beta_t} z_t$ where $z_t \sim N(0, I)$ is an independent standard normal random variable.

Now we can unroll the recurrence relation for $x_t$:

$$
\begin{align*}
x_t &= \sqrt{1 - \beta_t} x_{t-1} + \epsilon_t \\
&= \sqrt{\alpha_t} x_{t-1} + \sqrt{\beta_t} z_t \\
&= \sqrt{\alpha_t}\left( \sqrt{\alpha_{t-1}} x_{t-2} + \sqrt{\beta_{t-1}} z_{t-1} \right) + \sqrt{\beta_t} z_t \\
&= \sqrt{\alpha_t \alpha_{t-1}} x_{t-2} + \sqrt{\alpha_t \beta_{t-1}} z_{t-1} + \sqrt{\beta_t} z_t \\
&= \ldots \\
&= \sqrt{\prod_{j=1}^{t} \alpha_j} x_0 + \sum_{s=1}^{t} \left( \sqrt{\beta_s} \prod_{j=s+1}^{t} \sqrt{\alpha_j} \right) z_s \\
\end{align*}
$$

We can simplify the product term in the summation further using the definition of $\bar{\alpha}_t$:

$$
\prod_{j=s+1}^{t} \alpha_j = \frac{\bar{\alpha}_t}{\bar{\alpha}_s},
$$

so the unrolled expression becomes:

$$
x_t = \sqrt{\bar{\alpha}_t} x_0 + \sum_{s=1}^{t} \left( \sqrt{\beta_s} \sqrt{\frac{\bar{\alpha}_t}{\bar{\alpha}_s}} \right) z_s
$$

Which can also be shown more compactly as:

$$
x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon \text{ where } \epsilon \sim N(0, I)
$$

To then construct the gaussian distribution $q(x_t | x_0)$, we need to compute the mean and variance of $x_t$ given $x_0$. Because $\mathbb{E}(z_s) = 0$ for all s, we can compute the expectation of $x_t$ given $x_0$ as:

$$
\mathbb{E}(x_t \mid x_0) = \sqrt{\bar{\alpha}_t} x_0
$$

To calculate the variance of $\text{Var}(x_t \mid x_0)$, we only need to consider the noise terms since the first term involving $x_0$ is deterministic given $x_0$ and thus has zero variance. Because the noise terms are independent, we can compute the variance of each term in the sum separately and then sum them up:

$$
\text{Var}\left(
\sqrt{\beta_s}
\sqrt{\frac{\bar{\alpha}_t}{\bar{\alpha}_s}} z_s
\right) =
\beta_s
\frac{\bar{\alpha}_t}{\bar{\alpha}_s} I
$$

So the total variance is:

$$
\text{Var}(x_t \mid x_0) =
\sum_{s=1}^{t} \beta_s
\frac{\bar{\alpha}_t}{\bar{\alpha}_s} I
$$

it can also be shown that the following identity holds:

$$
\sum_{s=1}^{t} \beta_s \frac{\bar{\alpha}_t}{\bar{\alpha}_s} = 1 - \bar{\alpha}_t
$$

which gives us the final expression for the variance:

$$
\text{Var}(x_t \mid x_0) = (1 - \bar{\alpha}_t) I
$$

putting this together we get the closed form expression for the distribution of $x_t$ given $x_0$ as:

$$
q(x_t | x_0) = N(x_t; \sqrt{\bar{\alpha}_t} x_0, (1 - \bar{\alpha}_t) I)
$$

where we have reparameterized from the sum of gaussians to a single gaussian depending on $x_0$ and the parameter $\bar{\alpha}_t$. This means that picking a time step $t$ and sampling from $q(x_t | x_0)$ is equivalent to scaling down the original data sample $x_0$ by $\sqrt{\bar{\alpha}_t}$ and adding gaussian noise with variance $(1 - \bar{\alpha}_t) I$. This is useful as it allows us to **directly sample noisy samples at any time step $t$ without having to iteratively apply the forward diffusion process** from time step 1 to $t$ which can be computationally expensive for large $T$.

Under some mild assumptions on the noise schedule $\beta_t$ (e.g. $ 0< \beta_t < 1$), it can be shown that this defines a **ergodic markov chain** meaning that as $T$ approaches infinity, the distribution of $x_T$ becomes independent of the initial data sample $x_0$ and converges to a stationary distribution where in our case the stationary distribution is a standard normal distribution:

$$
\lim_{T \to \infty} q(x_T | x_0) = N(0, I)
$$

This means that after enough time steps of adding noise, the data samples become indistinguishable from pure Gaussian noise. Intuitively this makes sense as we want $\sqrt{\bar{\alpha}_T} \to 0$ and $(1 - \bar{\alpha}_T) \to 1$ as $T$ approaches infinity, so we pick a noise schedule $\beta_t$ that ensures this. In other words, we are scaling down the original signal to zero while continuously adding random noise, so eventually the original signal is completely lost and we are left with just noise. This is key for diffusion models as it allows us to start from pure Gaussian noise and then reverse the diffusion process to generate new data samples. 

In bayesian terms we can also interpret our complex target distribution $q(x_0)$ as the posterior distribution and the known simple gaussian distribution as the prior distribution. 

{{< figure 
    src="/images/ml/diffusionNoiseConvergence.png"
    alt="Illustration of the convergence of the forward diffusion process to pure Gaussian noise over T time steps."
    caption="Illustration of the convergence of the forward diffusion process to pure Gaussian noise over T time steps."
    width="500"
>}}

### Reverse Denoising Process

Now that we have defined the forward diffusion process that gradually adds noise to the data samples, we can define the reverse denoising process that aims to reverse this process and recover the original data samples from the noisy samples. **The key idea here is that because the forward process results in pure Gaussian noise after enough time steps, we can start from pure noise and then iteratively denoise the samples to generate new data samples.** So in other words, we want to learn a markov chain that starts from a sample $x_T \sim N(0, I)$ and then iteratively applies denoising steps to obtain samples $x_{T-1}, x_{T-2}, ..., x_0$ where finally $x_0$ should closely resemble samples from the true data distribution. 

{{< figure 
    src="/images/ml/diffusionReverse.png"
    alt="Illustration of the reverse denoising process gradually removing noise from a noisy sample over T time steps."
    caption="Illustration of the reverse denoising process gradually removing noise from a noisy sample over T time steps."
>}}

{{< figure 
    src="/images/ml/diffusionReverseAnimation.gif"
    alt="Animation showing the reverse denoising process gradually removing noise from a toy distribution."
    caption="Animation showing the reverse denoising process gradually removing noise from a toy distribution."
>}}

So we want the distribution of the final sample $x_0$ obtained from reversing the noise addition process. We can write this is as a joint distribution over the entire sequence of samples from time step $T$ to 0. However, remember that the reverse process is also a markov chain, so each sample $x_{t-1}$ only depends on the current sample $x_t$ and not on any later samples. We can also use the product rule of probability to write the joint distribution as:

$$
q(x_0:T) = q(x_T) \prod_{t=1}^{T} q(x_{t-1} | x_t, x_{t+1}, ..., x_T) = q(x_T) \prod_{t=1}^{T} q(x_{t-1} | x_t)
$$

Notice that for this we need to know the reverse conditional distributions as we already know the initial distribution $q(x_T) = N(0, I)$. The reverse conditional distributions represent the probability of obtaining the previous slightly less noisy sample $x_{t-1}$ given the current noisy sample $x_t$. So we need to compute:

$$
q(x_{t-1} | x_t)
$$

if we rewrite this using bayes theorem we get:

$$
q(x_{t-1} | x_t) = \frac{q(x_t | x_{t-1}) q(x_{t-1})}{q(x_t)}
$$

Unfortunately we can see that this is intractable to compute directly as we do not have access to the true data distribution $q(x_{t-1})$ nor the marginal distribution $q(x_t)$. However, it was shown that the reverse conditional probability is tractable if we in addition condition on the original data sample $x_0$. So in other words, if we are given both the current noisy sample $x_t$ and the original data sample $x_0$ as a "hint", we stand a chance at computing the reverse conditional distribution. We have already seen this in the forward process where we derived the closed form expression for $q(x_t | x_0)$ to be a gaussian distribution. The same applies here: 

$$
q(x_{t-1} | x_t) = N(x_{t-1}; \mu(x_t, t), \Sigma(x_t, t))
$$

Additionally conditioning on $x_0$ makes the formulation of the problem and derivation of the solution possible as we will see later on. So if we restrict ourselves to just having an isotropic gaussian noise model just like in the forward diffusion process, we can write:

$$
q(x_{t-1} | x_t, x_0) = N(x_{t-1}; \tilde{\mu}(x_t, x_0), \tilde{\beta}_t(x_t, x_0) I)
$$

Or if we assume that the variance is only dependent on the time step t and not on the samples $x_t$ and $x_0$, we can simplify this further to:

$$
q(x_{t-1} | x_t, x_0) = N(x_{t-1}; \tilde{\mu}(x_t, x_0), \tilde{\beta}_t I)
$$

But we still only have access to $x_0$ at training time but not at sampling time as we only start from pure noise $x_T \sim N(0, I)$ and we don't know the parameters $\tilde{\mu}(x_t, x_0)$ and $\tilde{\beta}_t(x_t, x_0)$ of the reverse conditional distribution. So the idea is that we can instead learn a neural network model $p_\theta(x_{t-1} | x_t)$ parametrized by $\theta$ to approximate this reverse conditional distribution:

$$
p_\theta(x_{t-1} | x_t) \approx q(x_{t-1} | x_t, x_0)
$$

So in other words, we want these two distributions to match as closely as possible. This is the main learning objective of diffusion models and can be achieved by finding the optimal parameters $\theta$ that minimize the KL divergence between the two distributions which we will also see comes out of our derivations later on:

$$
p_\theta^* = \arg \min_\theta D_{KL}(q(x_{t-1} | x_t, x_0) \| p_\theta(x_{t-1} | x_t))
$$

this leads to us wanting to learn some mean and variance functions $\mu_\theta(x_t, t)$ and $\Sigma_\theta(x_t, t)$ such that the learned reverse conditional distribution $p_\theta(x_{t-1} | x_t)$ closely matches the true reverse conditional distribution $q(x_{t-1} | x_t, x_0)$. We can describe the entire reverse denoising process as a markov chain with the following trajectory distribution where we set the initial distribution $p(x_T) = N(0, I)$ to match the stationary distribution of the forward diffusion process:

$$
p_\theta(x_{0:T}) = p(x_T) \prod_{t=1}^{T} p_\theta(x_{t-1} | x_t)
$$

To achieve this, as is often the case in machine learning, we want to find parameters $\theta$ that maximize the likelihood of the observed data samples under the model $p(x_0 | \theta) = p_\theta(x_0)$. However, directly maximizing the likelihood is intractable because we would need to marginalize over all possible trajectories from $x_T$ to $x_0$, in other words over all possible ways noise could have been added:

$$
p_\theta(x_0) = \int p_\theta(x_{0:T}) dx_{1:T}
$$

This integral is intractable due to the high dimensionality of the space and the complex dependencies between the variables. Instead of maximizing the likelihood or the log likelihood directly, we can maximize a lower bound on the log likelihood $\log p_\theta(x_0)$ called the **evidence lower bound (ELBO)** using [variational inference](/garden/ml/bayesian/variationalinference/). We use this lower bound as a surrogate objective that is easier to optimize and still leads to good solutions. Because we prefer minimization problems in machine learning, we will derive our objective in terms of minimizing the negative ELBO, which is equivalent to maximizing the ELBO itself.

{{< figure 
    src="/images/ml/variationalInference.png"
    alt="Visualization of variational inference as minimizing the KL divergence between the true posterior and an approximate distribution."
    caption="Visualization of variational inference as minimizing the KL divergence between the true posterior and an approximate distribution."
    width="400"
>}}

{{< figure 
    src="/images/ml/elbo.png"
    alt="The evidence lower bound (ELBO) as a lower bound on the log likelihood."
    caption="The evidence lower bound (ELBO) as a lower bound on the log likelihood."
    width="400"
>}}

In diffusion models, compared to standard Bayesian inference where we infer the posterior given a prior and likelihood, we are instead trying to learn the reverse denoising process that best explains how to generate the observed data samples. The roles are somewhat different:

- **Prior**: The forward diffusion process $p_\theta(x_{1:T})$ which represents a specific trajectory of adding noise to the data samples.
- **Likelihood**: The reverse denoising process $p_\theta(x_0 | x_{1:T})$ which represents how to remove noise from the noisy samples to recover the data.
- **Posterior**: The forward trajectory distribution $q(x_{1:T} | x_0)$ which represents the true way noise was added to the data samples given we start from $x_0$. Using Bayes theorem, we can write:

$$
p_\theta(x_{1:T} | x_0) = \frac{p_\theta(x_0 | x_{1:T}) p_\theta(x_{1:T})}{p_\theta(x_0)} = \frac{1}{Z_\theta} p_\theta(x_0 | x_{1:T}) p_\theta(x_{1:T})
$$

where $Z_\theta = p_\theta(x_0)$ is the normalizing constant, which is the marginal data distribution computed as $Z_\theta = \int p_\theta(x_0 | x_{1:T}) p_\theta(x_{1:T}) dx_{1:T}$. Both this normalizing constant and the marginal data distribution $p_\theta(x_0)$ are intractable to compute directly because they require integrating over all possible noise trajectories in high-dimensional space. This is why we use the ELBO as a tractable surrogate objective that lower bounds the log likelihood.

There are two common derivations of the ELBO. The first derives it from the [cross entropy](/garden/ml/bayesian/variationalinference/#cross-entropy) between the data distribution and model distribution using Jensen's inequality, while the second derives it from the KL divergence decomposition. We will present both for completeness.

### ELBO Derivation via Jensen's Inequality

The [cross entropy](/garden/ml/bayesian/variationalinference/#cross-entropy) between the true data distribution $q(x_0)$ and the model distribution $p_\theta(x_0)$ is defined as:

$$
H(q, p_\theta) = - \mathbb{E}_{x_0 \sim q(x_0)} [\log p_\theta(x_0)]
$$

This measures how well the model distribution matches the true data distribution. Minimizing the cross entropy is equivalent to maximizing the expected log likelihood $\mathbb{E}_{x_0 \sim q(x_0)} [\log p_\theta(x_0)]$ of the observed data samples under the model. However, as mentioned earlier, directly computing the log likelihood $\log p_\theta(x_0)$ is intractable because it requires marginalizing over all possible trajectories.

To derive a tractable objective, we use [Jensen's inequality](/garden/maths/probabilitystatistics/expectationvariancecovariance/#jensens-inequality), which states that for a concave function $f$ and a random variable $X$:

$$
f(\mathbb{E}[X]) \geq \mathbb{E}[f(X)]
$$

Since the logarithm is a concave function, we can apply Jensen's inequality to obtain a lower bound on the log likelihood. We introduce the forward trajectory distribution $q(x_{1:T} | x_0)$ as an importance sampling distribution:

$$
\begin{align*}
\log p_\theta(x_0) &= \log \int p_\theta(x_{0:T}) dx_{1:T} \\
&= \log \int q(x_{1:T} | x_0) \frac{p_\theta(x_{0:T})}{q(x_{1:T} | x_0)} dx_{1:T} \\
&= \log \mathbb{E}_{q(x_{1:T} | x_0)} \left[ \frac{p_\theta(x_{0:T})}{q(x_{1:T} | x_0)} \right] \\
&\geq \mathbb{E}_{q(x_{1:T} | x_0)} \left[ \log \frac{p_\theta(x_{0:T})}{q(x_{1:T} | x_0)} \right] \quad \text{(by Jensen's inequality)}
\end{align*}
$$

This final expression is the ELBO:

$$
\text{ELBO} = \mathbb{E}_{q(x_{1:T} | x_0)} \left[ \log \frac{p_\theta(x_{0:T})}{q(x_{1:T} | x_0)} \right]
$$

We have shown that $\log p_\theta(x_0) \geq \text{ELBO}$, confirming that the ELBO is a lower bound on the log likelihood. Therefore, maximizing the ELBO with respect to $\theta$ will push the log likelihood upward, which is our objective.

### ELBO Derivation via KL Divergence

An alternative derivation comes from the [KL divergence](/garden/ml/bayesian/variationalinference/#kullback-leibler-divergence) between the true forward trajectory distribution $q(x_{1:T} | x_0)$ and the model trajectory distribution $p_\theta(x_{1:T} | x_0)$. The KL divergence is defined as:

$$
D_{KL}(q(x_{1:T} | x_0) \| p_\theta(x_{1:T} | x_0)) = \mathbb{E}_{q(x_{1:T} | x_0)} \left[ \log \frac{q(x_{1:T} | x_0)}{p_\theta(x_{1:T} | x_0)} \right]
$$

A fundamental property of KL divergence is that it is always non-negative, meaning $D_{KL}(q \| p) \geq 0$ for any distributions $q$ and $p$. We can use this property to derive the ELBO. Starting from the non-negativity:

$$
\begin{align*}
D_{KL}(q(x_{1:T} | x_0) \| p_\theta(x_{1:T} | x_0)) &\geq 0 \\
\mathbb{E}_{q(x_{1:T} | x_0)} \left[ \log \frac{q(x_{1:T} | x_0)}{p_\theta(x_{1:T} | x_0)} \right] &\geq 0
\end{align*}
$$

We can expand the denominator using Bayes theorem, recalling that $p_\theta(x_{1:T} | x_0) = \frac{p_\theta(x_0 | x_{1:T}) p_\theta(x_{1:T})}{p_\theta(x_0)} = \frac{p_\theta(x_{0:T})}{p_\theta(x_0)}$:

$$
\begin{align*}
\mathbb{E}_{q(x_{1:T} | x_0)} \left[ \log \frac{q(x_{1:T} | x_0)}{p_\theta(x_{1:T} | x_0)} \right] &\geq 0 \\
\mathbb{E}_{q(x_{1:T} | x_0)} \left[ \log \frac{q(x_{1:T} | x_0)}{\frac{p_\theta(x_{0:T})}{p_\theta(x_0)}} \right] &\geq 0 \\
\mathbb{E}_{q(x_{1:T} | x_0)} \left[ \log \frac{q(x_{1:T} | x_0)}{p_\theta(x_{0:T})} + \log p_\theta(x_0) \right] &\geq 0 \\
\mathbb{E}_{q(x_{1:T} | x_0)} \left[ \log \frac{q(x_{1:T} | x_0)}{p_\theta(x_{0:T})} \right] + \log p_\theta(x_0) &\geq 0
\end{align*}
$$

where in the last step we used the fact that $\log p_\theta(x_0)$ does not depend on $x_{1:T}$, so it comes out of the expectation. Rearranging recovers the same lower bound as the Jensen derivation:

$$
\log p_\theta(x_0) \geq \mathbb{E}_{q(x_{1:T} | x_0)} \left[ \log \frac{p_\theta(x_{0:T})}{q(x_{1:T} | x_0)} \right] = \text{ELBO}
$$

This derivation, however, only used the fact that $D_{KL} \geq 0$, replacing the KL with its lower bound of zero. That is why it produces an inequality rather than an equality. We can recover an exact relationship by instead starting directly from the definition of $D_{KL}$ and performing the same Bayes substitution:

$$
\begin{align*}
D_{KL}(q(x_{1:T} | x_0) \| p_\theta(x_{1:T} | x_0)) &= \mathbb{E}_{q(x_{1:T} | x_0)} \left[ \log \frac{q(x_{1:T} | x_0)}{p_\theta(x_{1:T} | x_0)} \right] \\
&= \mathbb{E}_{q(x_{1:T} | x_0)} \left[ \log \frac{q(x_{1:T} | x_0)}{\frac{p_\theta(x_{0:T})}{p_\theta(x_0)}} \right] \quad \text{(Bayes)} \\
&= \mathbb{E}_{q(x_{1:T} | x_0)} \left[ \log \frac{q(x_{1:T} | x_0)}{p_\theta(x_{0:T})} + \log p_\theta(x_0) \right] \\
&= \mathbb{E}_{q(x_{1:T} | x_0)} \left[ \log \frac{q(x_{1:T} | x_0)}{p_\theta(x_{0:T})} \right] + \log p_\theta(x_0) \\
&= -\text{ELBO} + \log p_\theta(x_0)
\end{align*}
$$

Every line here is an equality. Rearranging gives the exact decomposition:

$$
\log p_\theta(x_0) = D_{KL}(q(x_{1:T} | x_0) \| p_\theta(x_{1:T} | x_0)) + \text{ELBO}
$$

Since $D_{KL} \geq 0$, the ELBO can never exceed $\log p_\theta(x_0)$, recovering the lower bound. Crucially, the ELBO equals the log likelihood exactly when $D_{KL}(q(x_{1:T} | x_0) \| p_\theta(x_{1:T} | x_0)) = 0$, which occurs only when the learned reverse process $p_\theta(x_{1:T} | x_0)$ perfectly matches the true forward trajectory distribution $q(x_{1:T} | x_0)$.

### Expanding the ELBO for Optimization

Since $\log p_\theta(x_0)$ is fixed with respect to $\theta$, the decomposition above tells us that maximizing the ELBO is equivalent to minimising the KL divergence between the forward and reverse processes, which is exactly what we want: our learned reverse process should match the true forward process.

To make the ELBO tractable for optimization, we need to expand it into terms that we can actually compute and differentiate. The key insight is that during training we have access to the original data sample $x_0$, which allows us to condition on it. This conditioning dramatically reduces the variance of our estimates and makes the optimization stable. We can now rewrite the ELBO in a more convenient form:

$$
\begin{align*}
L_{\text{ELBO}} &= \mathbb{E}_{q(x_{1:T} | x_0)} \left[ \log \frac{q(x_{1:T} | x_0)}{p_\theta(x_{0:T})} \right] \\
&= \mathbb{E}_{q( \cdot | x_0)} \left[ \log \frac{\prod_{t=1}^{T} q(x_t | x_{t-1})}{p(x_T) \prod_{t=1}^{T} p_\theta(x_{t-1} | x_t)} \right] \text{ (by joint distributions) } \\
&= \mathbb{E}_{q(\cdot | x_0)} \left[ -\log p(x_T) + \log \frac{\prod_{t=1}^{T} q(x_t | x_{t-1})}{\prod_{t=1}^{T} p_\theta(x_{t-1} | x_t)} \right] \text{ (by log and division) } \\
&= \mathbb{E}_{q(\cdot | x_0)} \left[ -\log p(x_T) + \sum_{t=1}^{T} \log \frac{q(x_t | x_{t-1})}{p_\theta(x_{t-1} | x_t)} \right] \text{ (by log and product) } \\
&= \mathbb{E}_{q(\cdot | x_0)} \left[ -\log p(x_T) + \sum_{t=2}^{T} \log \frac{q(x_t | x_{t-1})}{p_\theta(x_{t-1} | x_t)} + \log \frac{q(x_1 | x_0)}{p_\theta(x_0 | x_1)} \right] \text{ (separating last term) } \\
&= \mathbb{E}_{q(\cdot | x_0)} \left[ -\log p(x_T) + \sum_{t=2}^{T} \log \frac{\frac{q(x_{t-1} | x_t) q(x_t)}{q(x_{t-1})}}{p_\theta(x_{t-1} | x_t)} + \log \frac{q(x_1 | x_0)}{p_\theta(x_0 | x_1)} \right] \text{ (by bayes theorem) } \\
&= \mathbb{E}_{q(\cdot | x_0)} \left[ -\log p(x_T) + \sum_{t=2}^{T} \log \frac{\frac{q(x_{t-1} | x_t, x_0) q(x_t | x_0)}{q(x_{t-1} | x_0)}}{p_\theta(x_{t-1} | x_t)} + \log \frac{q(x_1 | x_0)}{p_\theta(x_0 | x_1)} \right] \text{ (conditioning on } x_0 \text{) } \\
&= \mathbb{E}_{q(\cdot | x_0)} \left[ -\log p(x_T) + \sum_{t=2}^{T} \left( \log \frac{q(x_{t-1} | x_t, x_0)q(x_t | x_0)}{p_\theta(x_{t-1} | x_t) q(x_{t-1} | x_0)} \right) + \log \frac{q(x_1 | x_0)}{p_\theta(x_0 | x_1)} \right] \\
&= \mathbb{E}_{q(\cdot | x_0)} \left[ -\log p(x_T) + \sum_{t=2}^{T} \left( \log \frac{q(x_{t-1} | x_t, x_0)}{p_\theta(x_{t-1} | x_t)} \right) + \sum_{t=2}^{T} \left( \log \frac{q(x_t | x_0)}{q(x_{t-1} | x_0)} \right) + \log \frac{q(x_1 | x_0)}{p_\theta(x_0 | x_1)} \right] \\
&= \mathbb{E}_{q(\cdot | x_0)} \left[ -\log p(x_T) + \sum_{t=2}^{T} \left( \log \frac{q(x_{t-1} | x_t, x_0)}{p_\theta(x_{t-1} | x_t)} \right) + \log \frac{q(x_T | x_0)}{q(x_1 | x_0)} + \log \frac{q(x_1 | x_0)}{p_\theta(x_0 | x_1)} \right] \text{ (telescoping sum) } \\
&= \mathbb{E}_{q(\cdot | x_0)} \left[ -\log p(x_T) + \sum_{t=2}^{T} \left( \log \frac{q(x_{t-1} | x_t, x_0)}{p_\theta(x_{t-1} | x_t)} \right) + \log q(x_T | x_0) - \log p_\theta(x_0 | x_1) \right] \\
&= \mathbb{E}_{q(\cdot | x_0)} \left[ \log \frac{q(x_T | x_0)}{p(x_T)} + \sum_{t=2}^{T} \left( \log \frac{q(x_{t-1} | x_t, x_0)}{p_\theta(x_{t-1} | x_t)} \right) - \log p_\theta(x_0 | x_1) \right] \\
&= \mathbb{E}_{q(\cdot | x_0)} \left[\underbrace{D_{KL}(q(x_T | x_0) \| p(x_T))}_{L_T} + \sum_{t=2}^{T} \underbrace{D_{KL}(q(x_{t-1} | x_t, x_0) \| p_\theta(x_{t-1} | x_t))}_{L_{t-1}} \underbrace{- \log p_\theta(x_0 | x_1)}_{L_0} \right]
\end{align*}
$$

This is our final expression for the ELBO, which we will use as our training objective. Analyzing the different terms in this expression reveals three main components:

**Term 1: Prior Matching** $L_T = D_{KL}(q(x_T | x_0) \| p(x_T))$

This term measures how well the final noisy sample $x_T$ matches the prior distribution $p(x_T) = N(0, I)$. Importantly, this term can be ignored during training because it has no learnable parameters. If we choose a large enough number of diffusion steps $T$ and an appropriate noise schedule $\beta_t$, the ergodic property of the forward diffusion process ensures that $q(x_T | x_0) \approx N(0, I)$ regardless of the starting data sample $x_0$. Since we set $p(x_T) = N(0, I)$ to match this, the KL divergence is approximately zero and independent of $\theta$.

**Term 2: Denoising Steps** $L_{t-1} = D_{KL}(q(x_{t-1} | x_t, x_0) \| p_\theta(x_{t-1} | x_t))$ for $t = 2, ..., T$

These terms measure how well our learned reverse process $p_\theta(x_{t-1} | x_t)$ matches the true reverse conditional distribution $q(x_{t-1} | x_t, x_0)$ at each time step. This is the core of our training objective and directly corresponds to our goal of learning to denoise samples.

**Term 3: Reconstruction** $L_0 = - \log p_\theta(x_0 | x_1)$

This term represents the final reconstruction step from the slightly noisy sample $x_1$ to the clean data $x_0$. In practice, this term is often simplified or handled differently during sampling. For image data scaled to the range $[-1, 1]$, the discrete nature of pixel values requires special treatment, but this is typically handled as a post-processing step.

Given these simplifications, our main training objective becomes minimizing the sum of KL divergences between the true and learned reverse conditional distributions:

$$
\min_\theta \sum_{t=2}^{T} D_{KL}(q(x_{t-1} | x_t, x_0) \| p_\theta(x_{t-1} | x_t))
$$

This directly matches our initial goal of learning the reverse denoising process by minimizing the discrepancy between the true and approximate reverse transitions at each step.

### Parameterizing the Reverse Process

Recall that we previously established that conditioning on $x_0$ makes the reverse conditional distribution tractable, allowing us to express it as a Gaussian distribution:

$$
q(x_{t-1} | x_t, x_0) = N(x_{t-1}; \tilde{\mu}(x_t, x_0), \tilde{\beta}_t(x_t, x_0) I)
$$

Applying Bayes theorem, we can express this reverse conditional distribution in terms of quantities we already know:

$$
q(x_{t-1} | x_t, x_0) = \frac{q(x_t | x_{t-1}) q(x_{t-1} | x_0)}{q(x_t | x_0)}
$$

Each of these components is a Gaussian distribution. We have already derived closed-form expressions for the forward conditional distribution $q(x_t | x_{t-1}) = N(x_t; \sqrt{\alpha_t} x_{t-1}, \beta_t I)$ and the marginal distribution $q(x_t | x_0) = N(x_t; \sqrt{\bar{\alpha}_t} x_0, (1 - \bar{\alpha}_t) I)$. The conditional distribution $q(x_{t-1} | x_0)$ can be derived in the same way, giving $q(x_{t-1} | x_0) = N(x_{t-1}; \sqrt{\bar{\alpha}_{t-1}} x_0, (1 - \bar{\alpha}_{t-1}) I)$.

Since the product of two Gaussian distributions is proportional to another Gaussian, we can find the closed-form expression by computing the mean and variance of the resulting Gaussian as a function of $x_t$ and $x_0$:

$$
q(x_{t-1} | x_t, x_0) \propto q(x_t | x_{t-1}) q(x_{t-1} | x_0)
$$

{{< callout type="todo" >}}
Actually show the calculations
{{< /callout >}}

This results in the following **closed form expression for the variance of the reverse conditional distribution**:

$$
\tilde{\beta}_t = \frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t} \beta_t
$$

This variance still depends on the time step t but no longer on $x_t$ or $x_0$ which is good as we want to be able to sample from this distribution without knowing $x_0$. Remember we picked an isotropic gaussian $\Sigma(x_t, t) = \tilde{\beta}_t I$. In the [DDPM paper](https://arxiv.org/abs/2006.11239) the authors tried different choices for $\tilde{\beta}_t$ and found that setting $\tilde{\beta}_t = \beta_t$ works well in practice. We know $\beta_t$ from the noise schedule, so we have a closed form expression for the variance of the reverse conditional distribution which just depends on the time step t and no longer on $x_t$ or $x_0$.

{{< callout type="todo" >}}
Actually show the calculations
{{< /callout >}}

If we then also calculate the mean of this distribution we get:

$$
\tilde{\mu}(x_t, x_0) = \frac{\sqrt{\alpha_t}(1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t}x_t + \frac{\sqrt{\bar{\alpha}_{t-1}} \beta_t}{1 - \bar{\alpha}_t}x_0
$$

This mean function still depends on both $x_t$ and $x_0$. However, we do not have access to $x_0$ at sampling time as we only start from pure noise $x_T \sim N(0, I)$. However, remember that we could previously reparameterize $x_t$ in terms of $x_0$ and some standard gaussian noise $\epsilon$ as:

$$
x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon \text{ where } \epsilon \sim N(0, I)
$$

If we rearrange this expression we can express $x_0$ in terms of $x_t$ and $\epsilon$ as:

$$
x_0 = \frac{1}{\sqrt{\bar{\alpha}_t}} \left( x_t - \sqrt{1 - \bar{\alpha}_t} \epsilon \right)
$$

If we then substitute this expression for $x_0$ into the mean function $\tilde{\mu}(x_t, x_0)$ from above and simplify we can **express the mean function without requiring $x_0$ solely in terms of $x_t$ and some random noise $\epsilon$** showing that the mean of the reverse conditional distribution really is just a scaled version of the current noisy sample $x_t$ minus some noise term:

$$
\begin{align*}
\tilde{\mu}_t(x_t) &= \frac{\sqrt{\alpha_t}(1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t}x_t + \frac{\sqrt{\bar{\alpha}_{t-1}} \beta_t}{1 - \bar{\alpha}_t}x_0 \\
&= \frac{\sqrt{\alpha_t}(1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t}x_t + \frac{\sqrt{\bar{\alpha}_{t-1}} \beta_t}{1 - \bar{\alpha}_t} \left( \frac{1}{\sqrt{\bar{\alpha}_t}} \left( x_t - \sqrt{1 - \bar{\alpha}_t} \epsilon_t \right) \right) \\
&= \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \epsilon_t \right)
\end{align*}
$$

So in other words our training of $p_\theta(x_{t-1} | x_t)$ now reduces to learning the mean function as we can express the reverse conditional distribution as:

$$
p_\theta(x_{t-1} | x_t) = N(x_{t-1}; \mu_\theta(x_t, t), \Sigma_\theta(x_t, t))
$$

where we can set $\Sigma_\theta(x_t, t) = \sigma_t^2 I$ with $\sigma_t^2 = \beta_t$ as mentioned above. So we would like to train $\mu_\theta(x_t, t)$ to be an estimator of the true mean of the reverse conditional distribution:

$$
\mu_\theta(x_t, t) \approx \tilde{\mu}(x_t, t) = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \epsilon_t \right)
$$

Because at training time we have access to $x_t$ **we can reparameterize it to instead predict the noise $\epsilon_t$ from the input $x_t$ and time step t.** This has been found to work better in practice as predicting just the noise is easier than predicting the entire denoised sample. So we can rewrite the mean function as:

$$
\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \epsilon_\theta(x_t, t) \right)
$$

where $\epsilon_\theta(x_t, t)$ is a neural network that predicts the noise added at time step t given the noisy sample $x_t$. Therefore the slightly denoised sample can be computed as:

$$
\begin{align*}
x_{t-1} &= N\left( x_{t-1}; \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \epsilon_\theta(x_t, t) \right), \Sigma_\theta(x_t, t) \right)\\
&= \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \epsilon_\theta(x_t, t) \right) + \beta_t z_t \text{ where } z_t \sim N(0, I)
\end{align*}
$$

It is important that we add noise back at each denoising step to maintain stochasticity in the process and ensure diversity in the generated samples. Without this added noise, the model would only predict the mean of the reverse conditional distribution, which could lead to mode collapse where the model generates similar samples repeatedly rather than exploring the full diversity of the data distribution.

### Deriving the Training Loss

We have established that our training objective reduces to minimizing the KL divergence between the true reverse conditional distribution $q(x_{t-1} | x_t, x_0)$ and the learned reverse conditional distribution $p_\theta(x_{t-1} | x_t)$ at each time step. The general form of the KL divergence between two Gaussian distributions $N(\mu_1, \Sigma_1)$ and $N(\mu_2, \Sigma_2)$ in $\mathbb{R}^d$ is given by:

$$
D_{KL}(N(\mu_1, \Sigma_1) \| N(\mu_2, \Sigma_2)) = \frac{1}{2} \left( \log \frac{|\Sigma_2|}{|\Sigma_1|} - d + \text{tr}(\Sigma_2^{-1} \Sigma_1) + (\mu_2 - \mu_1)^T \Sigma_2^{-1} (\mu_2 - \mu_1) \right)
$$

{{< callout type="info" title="KL Divergence Between Gaussians" >}}
For two multivariate Gaussian distributions $p = N(\mu_1, \Sigma_1)$ and $q = N(\mu_2, \Sigma_2)$, the KL divergence can be derived from the definition:

$$
\begin{align*}
D_{KL}(p \| q) &= \mathbb{E}_{x \sim p}\left[\log \frac{p(x)}{q(x)}\right] \\
&= \mathbb{E}_{x \sim p}\left[\log p(x) - \log q(x)\right] \\
&= \mathbb{E}_{x \sim p}\left[\log p(x)\right] - \mathbb{E}_{x \sim p}\left[\log q(x)\right]
\end{align*}
$$

The first term is the negative entropy of $p$, which for a Gaussian is $-H[p] = \frac{1}{2}\log|\Sigma_1| + \frac{d}{2}\log(2\pi e)$. The second term is the cross-entropy. For the log density of $q$ evaluated at $x \sim p$:

$$
\log q(x) = -\frac{1}{2}\log|\Sigma_2| - \frac{d}{2}\log(2\pi) - \frac{1}{2}(x - \mu_2)^T\Sigma_2^{-1}(x - \mu_2)
$$

Taking the expectation over $x \sim p$ and using $\mathbb{E}_p[(x - \mu_2)^T\Sigma_2^{-1}(x - \mu_2)] = \text{tr}(\Sigma_2^{-1}\Sigma_1) + (\mu_1 - \mu_2)^T\Sigma_2^{-1}(\mu_1 - \mu_2)$, we obtain after simplification:

$$
D_{KL}(p \| q) = \frac{1}{2}\left(\log\frac{|\Sigma_2|}{|\Sigma_1|} - d + \text{tr}(\Sigma_2^{-1}\Sigma_1) + (\mu_1 - \mu_2)^T\Sigma_2^{-1}(\mu_1 - \mu_2)\right)
$$
{{< /callout >}}

For our specific case, both distributions are isotropic Gaussians with covariances $\Sigma_1 = \tilde{\beta}_t I$ and $\Sigma_2 = \sigma_t^2 I$. This simplifies the KL divergence considerably. The trace term becomes $\text{tr}(\sigma_t^{-2} I \cdot \tilde{\beta}_t I) = \frac{d \tilde{\beta}_t}{\sigma_t^2}$, and the determinant ratio gives $\log(\frac{\sigma_t^2}{\tilde{\beta}_t})^d = d \log(\frac{\sigma_t^2}{\tilde{\beta}_t})$. For isotropic covariances, the KL divergence simplifies to:

$$
D_{KL}(q(x_{t-1} | x_t, x_0) \| p_\theta(x_{t-1} | x_t)) = \frac{1}{2\sigma_t^2} \| \tilde{\mu}(x_t, x_0) - \mu_\theta(x_t, t) \|^2 + C
$$

where $C$ represents terms that do not depend on $\theta$ and can be ignored for optimization. Now we substitute our expressions for the means. Recall that:

$$
\tilde{\mu}(x_t, x_0) = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \epsilon \right) \quad \text{and} \quad \mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \epsilon_\theta(x_t, t) \right)
$$

Substituting these into the KL divergence and simplifying:

$$
\begin{align*}
L_t &= \mathbb{E}_{x_0, \epsilon \sim N(0, I)} \left[ \frac{1}{2\sigma_t^2} \left\| \tilde{\mu}(x_t, x_0) - \mu_\theta(x_t, t) \right\|^2 \right] \\
&= \mathbb{E}_{x_0, \epsilon} \left[ \frac{1}{2\sigma_t^2} \left\| \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \epsilon \right) - \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \epsilon_\theta(x_t, t) \right) \right\|^2 \right] \\
&= \mathbb{E}_{x_0, \epsilon} \left[ \frac{1}{2\sigma_t^2} \left\| \frac{1}{\sqrt{\alpha_t}} \cdot \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \left( \epsilon - \epsilon_\theta(x_t, t) \right) \right\|^2 \right] \\
&= \mathbb{E}_{x_0, \epsilon} \left[ \frac{\beta_t^2}{2\sigma_t^2 \alpha_t (1 - \bar{\alpha}_t)} \left\| \epsilon - \epsilon_\theta(x_t, t) \right\|^2 \right]
\end{align*}
$$

This shows that minimizing the KL divergence is equivalent to minimizing the mean squared error between the true noise $\epsilon$ and the predicted noise $\epsilon_\theta(x_t, t)$, weighted by the factor $\frac{\beta_t^2}{2\sigma_t^2 \alpha_t (1 - \bar{\alpha}_t)}$.

In the [DDPM paper](https://arxiv.org/abs/2006.11239), the authors showed empirically that a simplified version of this loss function outperforms the full variational bound. When we set the variance schedule $\sigma_t^2 = \beta_t$, the time-dependent weighting factor $\frac{\beta_t^2}{2\sigma_t^2 \alpha_t (1 - \bar{\alpha}_t)}$ can be dropped during optimization. This simplification leads to the final training objective for the denoising model:

$$
L_t^{\text{simple}} = \mathbb{E}_{x_0, \epsilon \sim N(0, I)} \left[ \| \epsilon - \epsilon_\theta (x_t, t) \|^2 \right] = \mathbb{E}_{x_0, \epsilon \sim N(0, I)} \left[ \| \epsilon - \epsilon_\theta (\sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon, t) \|^2 \right]
$$

This simplified objective can be scaled by a factor of one half to match the standard MSE loss convention:

$$
L_t^{\text{simple}} = \frac{1}{2} \mathbb{E}_{x_0, \epsilon \sim N(0, I)} \left[ \| \epsilon - \epsilon_\theta (x_t, t) \|^2 \right] = \frac{1}{2} \mathbb{E}_{x_0, \epsilon \sim N(0, I)} \left[ \| \epsilon - \epsilon_\theta (\sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon, t) \|^2 \right]
$$

This is the final training objective for diffusion models. It is remarkably simple: a mean squared error loss between the true noise added at each time step and the predicted noise from the neural network model. This objective can be optimized using standard stochastic gradient descent and backpropagation to learn the parameters $\theta$ of the denoising network.

### Training & Sampling

The above definitions and derivations can be summarized in the following algorithms for training and sampling from a DDPM. The training algorithm uses mini-batch gradient descent with batch size $B$, which means we sample $B$ data points from the dataset at each training step and compute the loss over the entire batch before updating the model parameters. This batching is more computationally efficient and leads to better convergence properties compared to processing single samples.

{{< figure 
    src="/images/ml/diffusionDDPMTraining.png" 
    alt="Pseudo-code for batched training of DDPMs."
    caption="Pseudo-code for batched training of DDPMs."
    width="600"
>}}

{{< figure 
    src="/images/ml/diffusionDDPMSampling.png" 
    alt="Pseudo-code for sampling from a trained DDPM."
    caption="Pseudo-code for sampling from a trained DDPM."
    width="600"
>}}

An important detail in the training algorithm is that we sample the time step $t$ uniformly at random from $\{1, 2, ..., T\}$ for each data point in the batch. This ensures that the model learns to denoise from all time steps equally well. Rather than calculating the loss over all $T$ time steps for each data point, which would require $T$ forward passes through the model, we sample just one time step per data point. Because we are using stochastic gradient descent, this single sampled time step provides an unbiased estimate of the full loss over all time steps. This is a standard technique in training deep generative models that significantly reduces computational cost while maintaining good convergence properties.

For the sampling algorithm, we start from pure Gaussian noise $x_T \sim N(0, I)$ and iteratively apply the denoising steps to obtain $x_{T-1}, x_{T-2}, ..., x_0$. The final step from $x_1$ to $x_0$ is treated specially. In practice, we may want to apply post-processing to ensure that the generated sample is in the valid range for the data, such as clipping pixel values to the range $[0, 1]$ for images. This is achieved by setting $z_1 = 0$ in the final step, effectively turning off the noise addition and using only the predicted mean.

## Noise-Conditional Score Networks

We have seen that diffusion models can be formulated as learning to reverse a noising process by predicting the noise $\epsilon$ added at each step. However, there is an alternative and mathematically elegant perspective that connects diffusion models to concepts from physics and statistical mechanics. This view, formalized in the paper [Score-Based Generative Modeling through Stochastic Differential Equations](https://arxiv.org/abs/2011.13456) by Yang Song et al., shows that diffusion models can be interpreted as learning the **score function** of the data distribution at different noise levels.

The score-based perspective offers several important insights. First, it reveals that predicting noise $\epsilon$ and predicting the score are equivalent tasks, just different parameterizations of the same underlying quantity. Second, it connects diffusion models to **Langevin dynamics**, a well-studied sampling method from statistical physics that uses gradients of the log density to generate samples. Third, it unifies discrete-time diffusion models (like DDPM) with continuous-time **stochastic differential equations (SDEs)**, providing a more general mathematical framework.

In this section, we will build up this alternative view step by step. We will start by introducing stochastic differential equations and Brownian motion, which provide the mathematical foundation for understanding diffusion as a continuous process. Then we will introduce the score function and Langevin dynamics, showing how gradients of the log density can be used for sampling. We will see how **score matching** allows us to learn the score from data without knowing the density, and how adding noise at multiple scales through **Noise Conditional Score Matching (NCSM)** makes this practical. Finally, we will show how the reverse SDE formulation enables generation, and how DDPM emerges as a special discretization of this continuous framework.

Please note that I am not a physicist nor a mathematician so my understanding of the following concepts is limited and I might get some details wrong. However, I will try my best to explain the main ideas behind this connection in an intuitive way. If you notice any mistakes or have improvement suggestions please let me know or submit a PR!

{{< figure
    src="/images/ml/diffusionSDEOverview.png"
    alt="Overview diagram of the SDE framework for diffusion models showing the forward noising process, the reverse denoising process, and the score function that enables time reversal."
    caption="SDE framework overview: the forward SDE gradually adds noise to data over time, while Anderson's theorem shows the reverse SDE (guided by the score function) can denoise and generate samples."
    width="800"
>}}

{{< callout type="info" title="Used Resources" >}}

| Type | Author/Creator | Resource |
|------|----------------|----------|
| Paper | Song et al. | [Score-Based Generative Modeling through Stochastic Differential Equations](https://arxiv.org/abs/2011.13456) |
| Paper | Aapo Hyvärinen | [Estimation of Non-Normalized Statistical Models by Score Matching](https://jmlr.org/papers/v6/hyvarinen05a.html) |
| Article | Yang Song | [Generative Modeling by Estimating Gradients of the Data Distribution](https://yang-song.net/blog/2021/score/) |
| Video | Outlier | [Diffusion Models From Scratch - Score-Based Generative Models Explained - Math Explained](https://youtu.be/B4oHJpEJBAA?si=umIgu9FOpPd1lMfV) |
| Video | Deepia | [Score-based Diffusion Models - Generative AI Animated](https://youtu.be/lUljxdkolK8?si=-JrhMthczTG1rXQy) |
| Video | Julia Turc | [The physics behind diffusion models](https://youtu.be/R0uMcXsfo2o?si=K4FC4W2s_BRKADcF) |

{{< /callout >}}

### Ito SDE and Brownian Motion

To understand the connection to physics, it is helpful to think of images as particles in high-dimensional space and the diffusion process as a stochastic process that moves these particles around. We start by describing the movement of a single particle in space over time. This leads us to an **ordinary differential equation (ODE)** where the change in position of the particle is determined by some function $f(x, t)$ called the **drift function**. This function takes the current position of the particle $x$ and the current time $t$ and outputs the change in position of the particle at that time step:

$$
dx = f(x, t)dt \equiv \frac{dx}{dt} = f(x,t)
$$

So importantly this is a **deterministic function that determines how the particle moves in space over time.** Think of the particle being acted on by some clear deterministic forces such as gravity or wind that push it in a certain direction. If our particle $x \in \mathbb{R}^d$ is in d-dimensional space then the drift function $f(x, t)$ outputs a d-dimensional vector that determines the direction and speed of movement of the particle at each time step and is usually also referred to as the **drift field**.

However, **in reality the movement of particles is often influenced by random forces such as collisions with other particles or thermal fluctuations**. To model this we need to introduce some randomness into the dynamics via a **stochastic differential equation (SDE)**:

$$
dX = f(X, t)dt + g(t)dW_t
$$

Because of the randomness we now use a capital X to denote that this is a random variable representing the position of the particle at time t. We again have our drift function f(X, t) which determines the deterministic part of the movement. But now we also have an additional term $g(t)dW_t$ which introduces randomness into the dynamics. Here $W_t$ is so called **Brownian motion** (or Wiener process) which is a mathematical model for random continuous movement. The function $g(t)$ called the **diffusion coefficient** controls the amount of noise added at each time step. So this term basically adds some random perturbations to the movement of the particle at each time step, simulating the effect of random forces acting on it.

{{< figure 
    src="/images/ml/diffusionSDE.png" 
    alt="1D illustration of how a stochastic differential equation (SDE) can change over time."
    caption="1D illustration of how a stochastic differential equation (SDE) can change over time."
    width="600"
>}}

{{< figure 
    src="/images/ml/diffusionBrownianMotion.gif" 
    alt="A particle undergoing Brownian motion in 2D space."
    caption="A particle undergoing Brownian motion in 2D space."
    width="400"
>}}

To understand what Brownian motion is let's first look at a the discrete ODE. We can think of this function as defining a sequence of positions $x(0), x(1), ..., x(T)$ over $T$ time steps where the position at each time step is determined by the previous position plus some deterministic change. So in other words this also defines a **process** or a **$T$-long deterministic walk** in space where we start from some initial position $x(0)$ and then iteratively apply the update rule defined by the drift function $f(x, t)$ to get the next position:

$$
x(t+1) = x(t) + f(x(t), t)
$$

More generally, to allow for variable step sizes, we can construct a partition of the interval $[t_0, T]$ with step sizes $\Delta t_i = t_{i+1} - t_i$:

$$
t_0 < t_1 < \dots < t_n = T
$$

A **$T$-long, $n$-step deterministic walk** is then defined by the recurrence:

$$
x(t_{i+1}) = x(t_i) + f(x(t_i), t_i)\Delta t_i
$$

After $n$ steps this gives us:

$$
x(T) = x(t_0) + \sum_{i=0}^{n-1} f(x(t_i), t_i),\Delta t_i
$$

This sum is exactly a [Riemann sum]() approximating the time integral of the drift along the trajectory from $t_0$ to $T$. As we refine the partition $n \to \infty$ and $\max_i \Delta t_i \to 0$, the Riemann sum converges to the integral:

$$
x(T) = x(t_0) + \int_{t_0}^{T} f(x(t), t)dt
$$

or in differential form:

$$
dx = f(x,t)dt \equiv \frac{dx}{dt} = f(x,t)
$$

Again we now want to introduce **randomness** to this idea. Instead of a deterministic walk, we can define a **$T$-long random walk** which again consists of a sequence of positions $(W(0), W(1), \dots, W(T))$ over $T$ time steps. But now, instead of the position at each time step being determined solely by the previous position plus some deterministic change, we add some random noise to each step. Specifically, we can define the increment is standard Gaussian noise:

$$
W(t) - W(t-1) \sim \mathcal{N}(0, I),
$$

So at each time step the position changes by a random Gaussian vector with mean 0 and identity covariance matrix. Importantly, we also require that the increments are **independent**: the change at each time step does not depend on the previous changes. This gives us a simple model of random movement in space. 

We can again generalize this to non-equidistant time steps. So where we had a partition of the interval $[t_0, T]$ with step sizes $\Delta t_i = t_{i} - t_{i-1}$:

$$
W(t_i) - W(t_{i-1}) \sim \mathcal{N}\big(0, \Delta t_i I\big),
$$

Notice that we now scale the covariance of the Gaussian increment by the step size $(t_i - t_{i-1})$. This ensures that larger time steps lead to larger expected displacements, which makes sense intuitively. 

We can again turn this into a **T$-long, $n$-step random walk** and evaluate it after $n$ steps:

$$
W(T) = W(t_0) + \sum_{i=1}^{n} \big( W(t_i) - W(t_{i-1}) \big),
$$

If we now refine the partition more and more (smaller and smaller ($\Delta t_i$)), this random walk converges to a continuous-time stochastic process called **Brownian motion** or a **Wiener process**.

Formally, a $d$-dimensional Brownian motion (Wiener process) $(W(t))_{t \in [t_0, T]}$ is a continuous-time stochastic process with the following properties:

- **Initial condition**: $W(t_0) = 0$ almost surely.
- **Independent increments**: For any $t_0 < s < t$, the increment $W(t) - W(s)$ is independent of the past values $(W(u))_{u \leq s}$.
- **Gaussian increments**: For any $t_0 \leq s < t$, the increment $W(t) - W(s)$ is normally distributed with mean 0 and covariance proportional to the time difference:

$$
W(t) - W(s) \sim \mathcal{N}(0, (t - s) I).
$$
- **Continuous paths**: With probability 1, so almost surely, the function $t \mapsto W(t)$ is continuous in $t$, meaning there are no jumps or discontinuities in the trajectory.

Intuitively Brownian motion is the **continuous-time limit of a Gaussian random walk**, a particle undergoing continuous, jittery motion, endlessly "wiggling" in random directions.

Given a partition $t_0 < t_1 < \dots < t_n = T$, we can also define the **Riemann–Ito sum**:

$$
\sum_{i=1}^{n} g(t_{i-1})\big( W(t_i) - W(t_{i-1}) \big).
$$

where again each increment is a random vector and is distributed as:

$$
W(t_i) - W(t_{i-1}) \sim \mathcal{N}(0, \Delta t_i, I),
$$

You can think of this sum as accumulating random Gaussian "kicks" at each time step, weighted by the function $g(t)$. As we refine the partition $n \to \infty$ and $\max_i \Delta t_i \to 0$, this Riemann–Ito sum converges to the **Ito integral**:

$$
\int_{t_0}^{T} g(t),dW(t).
$$

rewritten in differential form as:

$$
dW(t) \sim \mathcal{N}(0, dt, I),
$$

Informally, you can think of the infinitesimal increment of Brownian motion as a tiny Gaussian "kick" whose variance is proportional to the infinitesimal time step $dt$ and the Ito integral then accumulates all these kicks, weighted by the function $g(t)$, over the interval from $t_0$ to $T$.

We can now combine the deterministic drift and the stochastic Brownian part to get a **stochastic differential equation (SDE)** of Ito type as seen before but now understanding that the Brownian motion part is like adding random Gaussian noise at each time step:

$$
dX = f(X,t)dt + g(t)dW_t.
$$

More generally, we could also let $g(t)$ depend on both state and time $g(X, t)$ but for diffusion models we usually just let it depend on time.

### Euler-Maruyama Method

Now that we have an Ito–SDE of the form

$$
dX_t = f(X_t, t)dt + g(t)dW_t
$$

we need a way to **actually simulate** the stochastic process ${X_t}_{t \in [0,T]}$ on a computer. So for a given ODE or SDE and starting point $x_0$ at time $t=0$, we want to simulate the trajectory of the process over the time interval $[0,T]$, so how the particle's position changes over time according to the dynamics defined by the SDE.

Just like for ODEs, we cannot usually solve SDEs analytically, so we resort to **numerical discretization**. Here, **discretization** means approximating the continuous-time process by a discrete-time process evaluated at a finite set of time points. The standard workhorse here is the **Euler–Maruyama method**, which is the stochastic extension of the Euler method for ODEs.

Let’s first start with the Euler method for ODEs. Consider an ODE

$$
\frac{dx}{dt} = f(x,t) \qquad x(0) = x_0
$$

We want to approximate the solution $x(t)$ over a time interval $[0,T]$. For this we discretize the time interval $[0,T]$ into $n$ equal steps of size $\Delta t = \tfrac{T}{n}$:

$$
0 = t_0 < t_1 < \dots < t_n = T \qquad t_{i+1} - t_i = \Delta t
$$

We can rewrite the ODE in integral form. Integrating both sides from $t_i$ to $t_{i+1}$ gives us:

$$
x(t_{i+1}) - x(t_i) = \int_{t_i}^{t_{i+1}} f(x(t), t)dt
$$

The Euler method then approximates this integral by a **Riemann sum**. Over the small interval $[t_i, t_{i+1}]$ we approximate $f(x(t), t)$ by its value at the left endpoint $(x(t_i), t_i)$:

$$
\int_{t_i}^{t_{i+1}} f(x(t), t)dt \approx f(x(t_i), t_i)\Delta t
$$

Substituting this approximation into the integral equation yields

$$
x(t_{i+1}) \approx x(t_i) + f(x(t_i), t_i)\Delta t
$$

If we write $x_i := x(t_i)$, the discrete update is

$$
x_{i+1} = x_i + f(x_i, t_i)\Delta t
$$

So for each time step we just take the current position $x_i$ and add the deterministic change $f(x_i, t_i)$ scaled by the time step size $\Delta t$ to get the next position $x_{i+1}$. So if we start from the initial position $x_0$ at time $t_0 = 0$, we can iteratively apply this update rule to get the entire trajectory:

$$
x_n = x_0 + \sum_{i=0}^{n-1} f(x_i, t_i)\Delta t
$$

As we let $\Delta t \to 0$ and $n \to \infty$ (with $T = n\Delta t$ fixed), this discrete-time approximation converges to the true solution of the ODE.

We can now extend this idea to SDEs. For an Ito-SDE we can again rewrite it in integral form:

$$
X_T = X_0 + \int_{0}^{T} f(X_t, t)dt + \int_{0}^{T} g(t)dW_t
$$

where $X_t$ is now a random variable representing the state of the stochastic process at time $t$, and $X_T$ is the random state at the final time $T$. We discretize the time interval $[0,T]$ using the same partition as in the ODE case:

$$
0 = t_0 < t_1 < \dots < t_n = T \qquad \Delta t = t_{i+1} - t_i
$$

The first deterministic integral is again approximated by a Riemann sum like before in the Euler method for ODEs. The second part is different though: here the stochastic integral is approximated by a **Riemann–Ito sum** using the increments of Brownian motion. For this we recall the key property of Brownian motion that for $t_{i-1} < t_i$ we have independent Gaussian increments

$$
\Delta W_i := W_{t_i} - W_{t_{i-1}} \sim \mathcal{N}(0, \Delta t I)
$$

Importantly any Gaussian with mean $0$ and covariance $\Delta t I$ can also scaled down to a standard normal by dividing by its standard deviation in this case $\sqrt{\Delta t}$. So we can equivalently write the increment as

$$
\Delta W_i = \sqrt{\Delta t} \epsilon_i \qquad \epsilon_i \sim \mathcal{N}(0, I)
$$

Putting this together, the **Euler–Maruyama approximation** of the SDE becomes

$$
X_{i+1} = X_i + f(X_i, t_i)\Delta t + g(t_i)\sqrt{\Delta t}\epsilon_i \qquad \epsilon_i \sim \mathcal{N}(0, I)
$$

where $X_i$ is the approximation to $X_{t_i}$ obtained by the scheme. So Euler–Maruyama is literally just one **deterministic Euler step** via $f(X_i,t_i)\Delta t$ plus one **stochastic Gaussian kick** via $g(t_i)\sqrt{\Delta t}\epsilon_i$. Again, as we refine the time steps $\Delta t \to 0$, the discrete process ${X_i}_{i=0}^{n}$ converges to the continuous-time SDE solution.

Note that the Euler–Maruyama update only depends on the **current state** $X_i$ and a fresh Gaussian random variable $\epsilon_i$. This means the discretized process $X_0, X_1, \dots, X_n$ defined by the Euler–Maruyama method is a sequence of random variables where each next state only depends on the current state and some independent noise. In other words, the sequence ${X_i}$ forms a **Markov chain** with the Markov property

$$
\mathbb{P}(X_{i+1} \mid X_i, X_{i-1}, \dots, X_0) = \mathbb{P}(X_{i+1} \mid X_i)
$$

In particular, the conditional distribution given $X_i$ is Gaussian with $g(t_i)$ controlling the covariance:

$$
X_{i+1} \mid X_i \sim \mathcal{N}\Big(X_i + f(X_i, t_i)\Delta t, g(t_i)^2 \Delta t I\Big)
$$

This is exactly the kind of Gaussian noising step we have been using in the diffusion model.

### Langevin Dynamics

In the case of diffusion models where our images are particles in space we want to have an SDE that matches our idea of the forward diffusion process that gradually adds noise to the data until it becomes pure Gaussian noise. So our particles then end up randomly scattered in space according to a Gaussian distribution. This can be rather easily achieved as we will seen later on, where we will define the **variance exploding SDE (VE-SDE) and the variance preserving SDE (VP-SDE)** that matches our noising process in diffusion models.

However, the more interesting question is how to define a reverse SDE that **denoises** the data, so that we can start from pure Gaussian noise and then gradually remove noise until we get back samples from the data distribution. In other words we want to define an SDE that will move our particles such that they will converge to a desired target distribution $P^*(X)$ over time, in our case the data distribution $q(x_0)$.

This idea of defining an SDE that converges to a desired target distribution has been studied in statistical physics and is known as **Langevin dynamics**. The key insight is that we can choose the drift term $f(X, t)$ in the SDE such that the stochastic process converges to the desired distribution $P^*(X)$ as time goes to infinity. The so called **Langevin SDE** is defined as:

$$
dX_t = \frac{1}{2} \nabla_X \log P^*(X_t)dt + dW_t
$$

where $P^*(X_t)$ is probability density function (PDF) of the target distribution we want to converge to. More precisely $P^*(X_t)$ is the likelihood of the random variable $X_t$ with $t \in [0, T]$ taking on the value $x$ at time $t$ according to the target distribution. 

The word **equilibrium** is also often used here to refer to this target distribution. The idea is that if we run this SDE for a long time, the distribution of the random variable $X_t$ will converge to the target distribution $P^*(X)$ regardless of the initial distribution of $X_0$. In other words, as $t \to \infty$, the distribution of $X_t$ approaches $P^*(X)$ and the process reaches an equilibrium/steady state.

For our drift term we have $\frac{1}{2} \nabla_X \log P^*(X_t)$ which is half the gradient of the log density of the target distribution. Our diffusion term is just standard Brownian motion $dW_t$ which adds random Gaussian noise, preventing the particle from simply collapsing into the nearest local maximum (mode) and instead allowing it to explore the entire distribution. Remember that we can think of the SDE as defining a continuous-time stochastic process where at each time step the position of the particle is updated by a deterministic drift part and a stochastic noise part and because it is Brownian motion the noise part is just independent Gaussian noise added at each time step leading to us defining a Markov chain when we discretize it using Euler-Maruyama.

This gradient of the log density is a key quantity known as the **score function** of the distribution $P^*(X)$ and is denoted as:

$$
s^*(X) = \nabla_X \log P^*(X)
$$

The idea of the score function is that it points in the direction of increasing probability density. So if we evaluate the score function at a point $X$ in space, it gives us a vector field which tells us which direction we should move in order to increase the likelihood of sampling from the target distribution. Think of it as being similar to the vector field given by the gradient of a scalar function, which points in the direction of steepest ascent used in optimization with gradient descent. This is exactly what we want for denoising in our diffusion models: we want to move our particles towards regions of higher probability density according to the target distribution.

{{< figure 
    src="/images/ml/diffusionScoreFunction.gif" 
    alt="Illustration of the score function pointing towards regions of higher probability density. Here the PDF is a mixture of two gaussians."
    caption="Illustration of the score function pointing towards regions of higher probability density. Here the PDF is a mixture of two gaussians."
    width="400"
>}}

The Langevin SDE can be derived using the **Fokker-Planck equation** which describes how the probability density function of a stochastic process evolves over time. Specifically if we have a probability density function $q(x, t)$ describing the likelihood of the random variable $X(t)$ taking on the value x at time t. The evolution of this density over time is governed by the Fokker-Planck equation:

$$
\frac{\partial q(X, t)}{\partial t} = - \nabla_X \cdot \left( f(X, t)q(X, t) - \frac{1}{2}g(t)^2 \nabla_X q(X, t) \right) = -\nabla_X \cdot J(X, t)
$$

where $J(x, t)$ is the so called **probability current or probability flux**. This is called current because it describes the flow of probability mass in the vector field in the space of $\mathbb{R}^d$ over time. The intuition behind this equation is that the change in probability density at a point X over time is determined by the net flow of probability mass into or out of that point due to both the drift term $f(X, t)$ and the diffusion term $g(t)$. 

An important note is that the operator $\nabla_X \cdot$ is the **divergence operator** which is different from the gradient operator $\nabla_X$ used before:
- $\nabla_x f(x)$ for a scalar $f: \mathbb{R}^d \to \mathbb{R}$ gives a vector pointing in the direction of steepest ascent.
- $\nabla_X \cdot v(X)$ for a vector field $v: \mathbb{R}^d \to \mathbb{R}^d$ gives a scalar quantifying the net outflow of the vector field from an infinitesimal neighborhood around X. So it measure the "inflow" and "outflow" of probability mass at a given point and can be interpreted as the local rate of change of probability mass. Similar to the mass conservation equation in fluid dynamics where water can not just disappear or appear out of nowhere or in the max-flow/min-cut theorem in graph theory where the total flow into a node must equal the total flow out of the node unless it is a source or sink.

So the divergence of the vector field $J$ which produces a scalar is defined as:

$$
\nabla_X \cdot J(X, t) = \sum_{i=1}^{d} \frac{\partial v_i(X)}{\partial x_i}
$$

In physical terms it quantifies the outflow - inflow of probability mass at a given point X in R^d. local rate of change of probability mass. Similar to the mass conservation equation in fluid dynamics where water can not just disappear or appear out of nowhere.

We now want to pick the drift term $f(X, t)$ such that the stochastic process converges to a random variable with a our desired target distribution $P^*(X)$ at equilibrium so when time goes to infinity. In other words we want to apply the Fokker-Planck equation such that the distribution reaches a steady state. This means that the time derivative of the density function becomes zero:

$$
\frac{\partial q(X, t)}{\partial t} = 0 \text{ for all } X, t
$$

so probability mass is neither created nor destroyed at any point $X$ in space over time and has an equilibrium distribution equal to the target distribution $P^*(X)$. So as $t \to \infty$ we have:

$$
q(X, t) = P^*(X)
$$

Plugging this condition into the Fokker-Planck equation gives us:

$$
\begin{align*}
0 &= - \nabla_X \cdot J(X, t) \\
&= - \nabla_X \cdot \left( f(X, t)q(X, t) - \frac{1}{2}g(t)^2 \nabla_X q(X, t) \right) \\
&= - \nabla_X \cdot \left( f(X, t)P^*(X) - \frac{1}{2}g(t)^2 \nabla_X P^*(X) \right)
\end{align*}
$$

Now we need to solve this for the drift term $f(X, t)$. A sufficient condition for this equation to hold is if the flux vanishes pointwise:

$$
\begin{align*}
f(X, t)P^*(X) - \frac{1}{2}g(t)^2 \nabla_X P^*(X) &= 0 \\
f(X, t) &= \frac{1}{2}\frac{g(t)^2 \nabla_X P^*(X)}{P^*(X)} \\
f(X, t) &= \frac{1}{2}g(t)^2 \nabla_X \log P^*(X)
\end{align*}
$$

If we set $g(t) = 1$ for simplicity we get the final form of the Langevin SDE:

$$
dX_t = \frac{1}{2} \nabla_X \log P^*(X_t)dt + dW_t
$$

Now that we have an SDE that converges to our desired target distribution $P^*(X)$ we can use this for denoising in diffusion models. For this we need to discretize the SDE using the Euler-Maruyama method which results in the so called **Unadjusted Langevin Algorithm (ULA)**. If we discretize the Langevin SDE by partitioning the time interval $[0, T]$ into equal steps $t_0 < t_1 < \dots < t_n = T$ with step size $\Delta t = t_{i+1} - t_i$ we get the following update rule:

$$
\begin{align*}
X_{i+1} &= X_i + \frac{1}{2} \nabla_X \log P^*(X_i)\Delta t + \sqrt{\Delta t}\epsilon_i \\
&= X_i + \frac{\Delta t}{2} \nabla_X \log P^*(X_i) + \sqrt{\Delta t}\epsilon_i \qquad \epsilon_i \sim \mathcal{N}(0, I)
\end{align*}
$$

Remember that the Brownian motion part just adds independent Gaussian noise at each time step. So we can interpret this update rule as a Markov chain where at each time step we take the current position $X_i$ and move it in the direction of increasing probability density according to the score function $\nabla_X \log P^*(X_i)$ scaled by the step size $\frac{\Delta t}{2}$ and then add some Gaussian noise $\sqrt{\Delta t}\epsilon_i$ to allow exploration of the distribution:

$$
X_{i+1} \mid X_i \sim \mathcal{N}\Big(X_i + \frac{\Delta t}{2} \nabla_X \log P^*(X_i), \Delta t I\Big)
$$

It is common to reparameterize the step size as $\Delta t = 2\alpha$ for some $\alpha > 0$ which gives us the final ULA update rule:

$$
X_{i+1} = X_i + \alpha \nabla_X \log P^*(X_i) + \sqrt{2\alpha}\epsilon_i \qquad \epsilon_i \sim \mathcal{N}(0, I)
$$

{{< figure 
    src="/images/ml/diffusionScoreMatching.png" 
>}}

### Score Matching

Langevin dynamics shows that if we know the score function $\nabla_X \log P^*(X)$, we can sample from any target distribution $P^*(X)$ by following the score and adding noise. However, there are two fundamental problems with this approach:

1. **Unknown density**: In practice, we only have access to samples from the data distribution (the training data), not the actual density function $P^*(X)$ itself. Without the density, we cannot directly compute its gradient, the score function $\nabla_X \log P^*(X)$.

2. **Intractable normalization**: Even if we tried to model the density explicitly, computing the normalizing constant would be intractable for high-dimensional data like images. The density has the form $P^*(X) = \frac{1}{Z}e^{-E(X)}$ where $Z = \int e^{-E(X)}dX$ is the partition function.

We need a way to estimate the score function from data samples drawn from the target distribution $P^*$. This is where **score matching** comes in. Instead of trying to model the entire density function $P^*(X)$ directly, score matching focuses on learning an estimate of the score function $\nabla_X \log P^*(X)$ directly from data samples. We do this by training a neural network $s_\theta(X)$ to approximate the score function:

$$
s_\theta(X) \approx \nabla_X \log P^*(X)
$$

To train this network we can use the **score matching objective** introduced by Hyvärinen in 2005. The idea starts with using the mean squared error loss between the true score function and the predicted score function:

$$
L(\theta) = \frac{1}{2} \mathbb{E}_{X \sim P^*} [\|s_\theta(X) - \nabla_X \log P^*(X)\|^2]
$$

The problem here is that we still need the true score function $\nabla_X \log P^*(X)$ which we do not have access to. However, Hyvärinen showed that we can expand and simplify this loss function to get rid of the dependence on the true score function using good old integration by parts.

We start by expanding the squared norm:

$$
\begin{align*}
L(\theta) &= \frac{1}{2} \mathbb{E}_{X \sim P^*} [\|s_\theta(X) - \nabla_X \log P^*(X)\|^2] \\
&= \frac{1}{2} \mathbb{E}_{X \sim P^*} [\|s_\theta(X)\|^2] - \mathbb{E}_{X \sim P^*} [s_\theta(X) \cdot \nabla_X \log P^*(X)] + \frac{1}{2} \mathbb{E}_{X \sim P^*} [\|\nabla_X \log P^*(X)\|^2]
\end{align*}
$$

We can ignore the last term as it does not depend on the parameters $\theta$ of our score model. The key is to simplify the middle term. We write out the expectation as an integral and apply the chain rule to rewrite $\nabla_X \log P^*(X) = \frac{\nabla_X P^*(X)}{P^*(X)}$:

$$
\begin{align*}
\mathbb{E}_{X \sim P^*} [s_\theta(X) \cdot \nabla_X \log P^*(X)] &= \int_{\mathbb{R}^d} s_\theta(X) \cdot \nabla_X \log P^*(X) P^*(X) dX \\
&= \int_{\mathbb{R}^d} s_\theta(X) \cdot \frac{\nabla_X P^*(X)}{P^*(X)} P^*(X) dX \\
&= \int_{\mathbb{R}^d} s_\theta(X) \cdot \nabla_X P^*(X) dX
\end{align*}
$$

Now we apply integration by parts. For a vector field $v(X)$ and a scalar field $f(X)$, the divergence theorem states:

$$
\int_{\mathbb{R}^d} v(X) \cdot \nabla_X f(X) dX = -\int_{\mathbb{R}^d} f(X) \nabla_X \cdot v(X) dX
$$

assuming the boundary terms vanish, which holds for probability densities that decay to zero at infinity. Setting $f(X) = P^*(X)$ and $v(X) = s_\theta(X)$:

$$
\begin{align*}
\int_{\mathbb{R}^d} s_\theta(X) \cdot \nabla_X P^*(X) dX &= -\int_{\mathbb{R}^d} P^*(X) \nabla_X \cdot s_\theta(X) dX \\
&= -\mathbb{E}_{X \sim P^*} [\nabla_X \cdot s_\theta(X)]
\end{align*}
$$

Substituting this back into our original objective, we obtain the **score matching objective** or **Hyvärinen's objective**:

$$
L(\theta) = \mathbb{E}_{X \sim P^*} \left[ \frac{1}{2} \|s_\theta(X)\|^2 + \nabla_X \cdot s_\theta(X) \right]
$$

This is a remarkable result. The objective no longer contains the unknown true score function $\nabla_X \log P^*(X)$. Instead, it only requires computing the predicted score $s_\theta(X)$ and its divergence $\nabla_X \cdot s_\theta(X) = \sum_{i=1}^{d} \frac{\partial s_\theta^{(i)}(X)}{\partial X_i}$, both of which can be computed directly from the network using automatic differentiation.

The two terms in the objective have intuitive interpretations:

The first term $\|s_\theta(X)\|^2$ penalizes large score magnitudes. Near data points, which correspond to regions of high probability density, the gradient of the log density should be small because these points are near local maxima of the density. The score function should thus have small magnitude at data points.

The second term $\nabla_X \cdot s_\theta(X)$ is the divergence of the score field. The divergence measures the net outflow of the vector field from an infinitesimal neighborhood. This term ensures that the score field has the correct structure, encouraging the score vectors to point toward regions of higher probability density rather than away from them.

However, there are still two significant challenges with this approach that limit its practical applicability:

1. **Computational Cost of the Divergence**: The divergence term $\nabla_X \cdot s_\theta(X) = \text{Tr}\left(\frac{\partial s_\theta(X)}{\partial X}\right) = \sum_{i=1}^{d} \frac{\partial s_\theta^{(i)}(X)}{\partial X_i}$ requires computing the trace of the Jacobian matrix of the score model. For high-dimensional data such as images, where $d$ could be millions of dimensions, a naive implementation would require one backpropagation pass per input dimension to compute all the diagonal elements of the Jacobian. This makes training computationally infeasible for realistic applications.

2. **Score Estimation in Low-Density Regions**: The score function learned through score matching is only well-defined and accurate near regions of high data density where we have training samples. Far from the data manifold in low-density regions, the score function may not provide useful guidance. This is problematic for generative modeling because we need to start sampling from random noise (a very low-density region relative to the data) and gradually move toward the data distribution. If the score function is poorly estimated in low-density regions, Langevin dynamics may fail to generate good samples. Intuitively, a particle starting very far from any data points has no training signal to guide it toward high-density regions, so the learned score may point in an arbitrary direction.

{{< figure
    src="/images/ml/diffusionScorePitfall.png"
    alt="Illustration of the score matching pitfall in low-density regions where the score function is poorly defined away from the data manifold."
    caption="The score matching pitfall: score functions learned from data are only reliable near training samples (high-density regions) and may point in arbitrary directions in low-density regions far from the data manifold."
    width="700"
>}}

### Noise Conditional Score Matching (NCSM)

To address both challenges simultaneously, we use **Noise Conditional Score Matching (NCSM)**. The key idea is to add Gaussian noise to the data samples at multiple noise levels and train our score model to predict the score function of these noise-perturbed distributions:

$$
\tilde{X} = X + \epsilon \text{ where } \epsilon \sim N(0, \sigma^2 I)
$$

The added noise smooths out the data distribution, making the score function well-defined everywhere in $\mathbb{R}^d$ and improving generalization to low density regions.

{{< figure
    src="/images/ml/diffusionScorePertrubed.png"
    alt="Illustration showing how adding Gaussian noise at multiple scales makes the score function well-defined throughout the space, covering both high and low density regions."
    caption="Noise perturbation solves the low-density problem: by adding Gaussian noise at multiple scales, the perturbed distribution smoothly covers the entire space, making the score function well-defined everywhere."
    width="700"
>}}

This results in the modified objective:

$$
L(\theta) = \frac{1}{2} E_{\tilde{X} \sim P_\sigma} [||s_\theta(\tilde{X}, \sigma) - \nabla_{\tilde{X}} \log P_\sigma(\tilde{X})||^2]
$$

Note that the score model now also takes the noise level $\sigma$ as an additional input, allowing it to adapt its predictions based on the amount of noise added to the data. This is important because the score function of the perturbed distribution $P_\sigma$ will vary depending on the noise level. But how does this help us avoid computing the divergence term? If we split this objective again just like before we can throw away the last term which does not depend on $\theta$ and focus on the last term:

$$
L(\theta) = \frac{1}{2} E_{\tilde{X} \sim P_\sigma} [||s_\theta(\tilde{X}, \sigma)||^2] - E_{\tilde{X} \sim P_\sigma} [s_\theta(\tilde{X}, \sigma) \cdot \nabla_{\tilde{X}} \log P_\sigma(\tilde{X})]
$$

If we use the defintion of expectation under the marginal distribution we can rewrite the objective as:

$$
\begin{align*}
E_{\tilde{X} \sim P_\sigma} [s_\theta(\tilde{X}, \sigma) \cdot \nabla_{\tilde{X}} \log P_\sigma(\tilde{X})] &= \int s_\theta(\tilde{X}, \sigma) \cdot \nabla_{\tilde{X}} \log P_\sigma(\tilde{X}) P_\sigma(\tilde{X}) d\tilde{X} \\
&= \int s_\theta(\tilde{X}, \sigma) \cdot \nabla_{\tilde{X}} P_\sigma(\tilde{X}) d\tilde{X} \\
&= \int s_\theta(\tilde{X}, \sigma) \cdot \nabla_{\tilde{X}} \int P_\sigma(X) P_\sigma(\tilde{X}|X) dX d\tilde{X} \text{Marginalization} \\
&= \int s_\theta(\tilde{X}, \sigma) \cdot \int P_\sigma(X) \nabla_{\tilde{X}} P_\sigma(\tilde{X}|X) dX d\tilde{X} \text{Leibniz rule} \\
&= \int \int s_\theta(\tilde{X}, \sigma) \cdot \nabla_{\tilde{X}} P_\sigma(\tilde{X}|X) P_\sigma(X) dX d\tilde{X} \\
&= \int \int s_\theta(\tilde{X}, \sigma) \cdot \nabla_{\tilde{X}} \log P_\sigma(\tilde{X}|X) P_\sigma(\tilde{X}|X) P_\sigma(X) dX d\tilde{X} \text{Chain rule} \\
&= E_{X, \tilde{X}|X} [s_\theta(\tilde{X}, \sigma) \cdot \nabla_{\tilde{X}} \log P_\sigma(\tilde{X}|X)]
\end{align*}
$$

So we have rewritten the expectation over the marginal distribution $P_\sigma(\tilde{X})$ into an expectation over the joint distribution of the original data and the perturbed data $P_\sigma(X, \tilde{X}) = P_\sigma(X) P_\sigma(\tilde{X}|X)$. This is useful because the conditional distribution $P_\sigma(\tilde{X}|X)$ is often easier to work with since it is defined by our noise model which is just Gaussian noise addition due to $\tilde{X} = X + \epsilon$ with $\epsilon \sim N(0, \sigma^2 I)$. The PDF of this conditional distribution is known:

$$
P_\sigma(\tilde{X}|X) = \frac{1}{(2\pi \sigma^2)^{\frac{d}{2}}} \exp \left( -\frac{||\tilde{X} - X||^2}{2\sigma^2} \right)
$$

We can then obtain the score function of this conditional distribution by taking the gradient of the log PDF:

$$
\nabla_{\tilde{X}} \log P_\sigma(\tilde{X}|X) = -\frac{1}{\sigma^2} (\tilde{X} - X) = -\frac{1}{\sigma^2} \epsilon
$$

with $\epsilon \sim N(0, \sigma^2 I)$. This means we can now compute the score function of the conditional distribution exactly without needing to compute any divergence terms. Putting this all together we get the final Noise Conditional Score Matching (NCSM) objective:

$$
L_\text{NCSM}(\theta) = \frac{1}{2} E_{X, \tilde{X}|X} [||s_\theta(\tilde{X}, \sigma) + \frac{1}{\sigma^2} (\tilde{X} - X)||^2] = \frac{1}{2} E_{X, \epsilon} [||s_\theta(X + \epsilon, \sigma) + \frac{1}{\sigma^2} \epsilon||^2] \qquad \epsilon \sim N(0, \sigma^2 I)
$$

So now we can train our score model $s_\theta$ to approximate the score function of the perturbed data distribution by minimizing this objective using samples from the original data distribution and adding Gaussian noise. This allows us to effectively learn the score function needed for denoising in diffusion models without needing to compute intractable divergence terms. We can also interpret this objective as follows:

- The $\tilde{X} = X + \epsilon$ represents a noisy version of the original data point $X$ obtained by adding Gaussian noise $\epsilon$ with standard deviation $\sigma$.
- The term $\frac{1}{\sigma^2}\epsilon$ represents the score function and therefore the direction in which we should move $\tilde{X}$ to increase its likelihood under the perturbed data distribution which in other words tells us how to denoise $\tilde{X}$ back towards the original data point $X$.

### Annealed Langevin Dynamics

NCSM shows us how to learn the score function at a single noise level $\sigma$. However, a crucial question remains: what value of $\sigma$ should we use? If $\sigma$ is too small, we face the same problem as before where the score is only well-defined near the data manifold. If $\sigma$ is too large, the noisy distribution becomes too diffuse and far from the original data distribution, making the learned score less useful for generating actual data samples.

The solution is to train with multiple noise levels simultaneously. We borrow an idea from physics called **annealing**, where we use a sequence of noise levels ranging from large to small. The intuition is that at high noise levels, the perturbed distribution covers the entire space smoothly, allowing the score to be well-defined everywhere. At low noise levels, the distribution is close to the true data distribution, giving us accurate scores near the data. By using multiple scales, we get the best of both worlds.

We define a geometric sequence of noise levels:

$$
\sigma_1 > \sigma_2 > \cdots > \sigma_K > 0
$$

where typically $\sigma_1$ is large enough that the perturbed distribution $P_{\sigma_1}$ approximately fills the entire ambient space, and $\sigma_K$ is small enough that $P_{\sigma_K}$ is close to the original data distribution. A common choice is a geometric sequence where $\sigma_k = \sigma_1 \cdot r^{k-1}$ for some ratio $r < 1$.

{{< figure
    src="/images/ml/diffusionScoreVarianceScales.gif"
    alt="Animation showing the impact of different variance scales on the score function, demonstrating how larger variances smooth out the score field."
    caption="Impact of variance scales on score functions: different noise levels (variances) produce score functions with varying degrees of smoothness, with larger variances creating smoother score fields that cover the entire space."
    width="600"
>}}

The overall training objective becomes a weighted sum of the NCSM objectives at each noise level:

$$
L_{\text{multi}}(\theta) = \sum_{k=1}^K \lambda(\sigma_k) \mathbb{E}_{X, \epsilon} \left[ \|s_\theta(X + \sigma_k \epsilon, \sigma_k) + \frac{1}{\sigma_k^2} \epsilon\|^2 \right] = \sum_{k=1}^K \lambda(\sigma_k) L_{\text{NCSM}}(\theta; \sigma_k)
$$

where $\epsilon \sim \mathcal{N}(0, I)$ is standard Gaussian noise. The weighting function $\lambda(\sigma_k)$ balances the contributions from different noise levels. A common choice is $\lambda(\sigma_k) = \sigma_k^2$, which gives more weight to higher noise levels. This weighting ensures that losses at different scales contribute more equally to the gradient updates, since the score magnitude naturally scales as $\frac{1}{\sigma}$ and squaring the error gives a factor of $\frac{1}{\sigma^2}$.

**Sampling via Annealed Langevin Dynamics**

Once we have trained a score model $s_\theta(X, \sigma)$ that works across multiple noise levels, we can use **annealed Langevin dynamics** for sampling. The idea is to start with the highest noise level $\sigma_1$ (where the distribution is approximately uniform over the space) and gradually decrease the noise level while running Langevin dynamics at each level.

The algorithm proceeds as follows:

1. Initialize $X_0 \sim \mathcal{N}(0, \sigma_1^2 I)$ from the prior distribution at the highest noise level
2. For each noise level $k = 1, 2, \ldots, K$:
   - Run $M$ steps of Langevin dynamics using the score at noise level $\sigma_k$:
   $$
   X_{i+1} = X_i + \alpha_k s_\theta(X_i, \sigma_k) + \sqrt{2\alpha_k} z_i \quad \text{for } i = 0, 1, \ldots, M-1
   $$
   where $z_i \sim \mathcal{N}(0, I)$ and $\alpha_k$ is the step size
3. Use the final $X_M$ as the initialization for the next noise level $\sigma_{k+1}$

The step size $\alpha_k$ at each noise level is typically set proportional to $\sigma_k^2$, so $\alpha_k = \epsilon \cdot \frac{\sigma_k^2}{\sigma_K^2}$ for some small constant $\epsilon$. This scaling ensures that the steps remain appropriately sized relative to the current noise level.

As we anneal from high to low noise, the samples progressively refine from coarse approximations to fine-grained details, eventually producing high-quality samples from the data distribution. This multi-scale approach is the key insight that makes score-based generative models practical.

### Anderson's Reversal and the Reverse SDE

We have now established the key building blocks for score-based diffusion models:

- **Langevin Dynamics** showed that if we have the score function (gradient of log density) and add noise, we can sample from any target distribution. This proves that generative modeling is possible using score functions.
- **Score Matching** showed that we can learn the score function from data without knowing the density itself, by using integration by parts to eliminate the intractable normalizing constant.
- **NCSM** showed that we can learn score functions at multiple noise levels by adding Gaussian noise to the data, which makes the score well-defined everywhere and avoids computing expensive divergence terms.
- **Annealed Langevin Dynamics** showed that by using multiple noise scales and gradually decreasing the noise level, we can generate high-quality samples starting from pure noise.

Now comes the key mathematical result that unifies these ideas into the continuous framework of diffusion models. In 1982, Brian Anderson proved a remarkable theorem about the time-reversal of stochastic processes. **Anderson's theorem** states that if we have a forward diffusion process described by an SDE, then the time-reversed process is also described by an SDE with a specific form involving the score function.

Consider a general **forward SDE** that describes how we gradually add noise to data over time:

$$
dX_t = f(X_t, t)dt + g(t)dW_t
$$

where $f(X_t, t)$ is the **drift term** controlling the deterministic part of the evolution, $g(t)$ is the **diffusion coefficient** controlling the amount of noise, and $W_t$ is Brownian motion. This SDE defines a stochastic process that starts at time $t=0$ with samples from the data distribution and evolves forward in time to $t=T$, progressively adding noise.

The forward process induces a sequence of **marginal distributions** $p_t(X)$ describing the probability density of $X_t$ at each time $t$. At $t=0$, we have $p_0(X) = q_{\text{data}}(X)$ which is our data distribution. As time progresses, the distribution becomes increasingly noisy and spread out. At the final time $t=T$, we typically choose the process parameters so that $p_T(X) \approx \mathcal{N}(0, \sigma_T^2 I)$ is approximately a simple prior distribution like an isotropic Gaussian.

Anderson's theorem tells us that the time-reversed process, which evolves backwards from time $T$ to time $0$, is also governed by an SDE. If we define the reversed process as $\tilde{X}_t = X_{T-t}$, then it satisfies the **reverse-time SDE**:

$$
dX_t = \left[f(X_t, t) - g(t)^2 \nabla_X \log p_t(X_t)\right]dt + g(t)d\tilde{W}_t
$$

where $\tilde{W}_t$ is a **Brownian motion running backwards in time**, also called the reverse Brownian motion. Notice that we use $X_t$ for the reverse process rather than $\tilde{X}_t$ to emphasize that this is the generative sampling process we will use.

The remarkable feature of this reverse SDE is the appearance of the **score function** $\nabla_X \log p_t(X_t)$ in the drift term. The original drift $f(X_t, t)$ is modified by subtracting $g(t)^2 \nabla_X \log p_t(X_t)$. This additional term guides the reverse process towards regions of high probability density at each time step. Intuitively, the score tells the process which direction to move in order to denoise the data and recover samples from the data distribution.

The reverse SDE can be derived rigorously using the Fokker-Planck equation, which describes how the probability density $p_t(X)$ evolves over time under the forward SDE. By considering the time-reversed Fokker-Planck equation and matching it to an SDE form, one arrives at Anderson's formula. The derivation shows that the score function naturally emerges as the correction term needed to reverse the diffusion process.

The reverse SDE provides us with a recipe for **generating samples**. If we can estimate the score function $\nabla_X \log p_t(X)$ at each time $t$, then we can simulate the reverse SDE to generate samples. We start with a sample from the prior distribution at time $T$, $X_T \sim \mathcal{N}(0, \sigma_T^2 I)$, then simulate the reverse SDE backwards in time from $t=T$ to $t=0$ using the Euler-Maruyama discretization, and the final state $X_0$ will be a sample from the data distribution $p_0(X) = q_{\text{data}}(X)$.

{{< figure
    src="/images/ml/diffusionReverseSDE.gif"
    alt="Animation showing the reverse SDE process starting from noise and progressively denoising to generate a sample from the data distribution."
    caption="Reverse SDE in action: starting from random Gaussian noise, the reverse-time SDE guided by the learned score function progressively removes noise to generate data samples."
    width="600"
>}}

Using Euler-Maruyama with time steps $T = t_0 > t_1 > \cdots > t_N = 0$ and step size $\Delta t = t_i - t_{i+1}$, the discretized reverse update is:

$$
X_{i+1} = X_i + \left[f(X_i, t_i) - g(t_i)^2 s_\theta(X_i, t_i)\right]\Delta t + g(t_i)\sqrt{\Delta t}\epsilon_i
$$

where $\epsilon_i \sim \mathcal{N}(0, I)$ and $s_\theta(X, t)$ is our learned score model approximating $\nabla_X \log p_t(X)$.

The challenge is **learning the score function** $\nabla_X \log p_t(X)$ at every time step $t \in [0, T]$. Here we can apply NCSM again, but now the time $t$ plays the role of the noise level. We train a **time-conditional score model** $s_\theta(X, t)$ to estimate the score function of the marginal distribution at each time step:

$$
s_\theta(X, t) \approx \nabla_X \log p_t(X)
$$

The training objective for the time-conditional score model follows the same NCSM principle. At each time $t$, we have a conditional distribution $p_t(X_t | X_0)$ defined by the forward process, and we can compute its score exactly just as we did for noise-perturbed data in NCSM. The specific form of this score depends on how the forward SDE is defined.

### Variance Preserving SDE (VP-SDE) and DDPM

We now show how the discrete-time DDPM formulation emerges as a special case of the continuous SDE framework. The key is defining a forward SDE whose discretization matches the DDPM forward process.

In DDPM, the forward noising process at discrete time steps is:

$$
x_t = \sqrt{1 - \beta_t} x_{t-1} + \sqrt{\beta_t} \epsilon_t \qquad \epsilon_t \sim \mathcal{N}(0, I)
$$

where $\beta_t$ is the noise schedule. This can also be written using the cumulative product $\bar{\alpha}_t = \prod_{s=1}^t (1 - \beta_s)$ as:

$$
x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon \qquad \epsilon \sim \mathcal{N}(0, I)
$$

This formulation preserves variance over time. If $x_0$ has unit variance and is independent of $\epsilon$, then $x_t$ also has unit variance for all $t$. This motivates the name **Variance Preserving SDE**, abbreviated as **VP-SDE**.

To derive the continuous-time SDE corresponding to DDPM, we take the discrete process and consider the limit as the time step size goes to zero. Starting from the discrete forward step and applying a Taylor expansion, we can show that the continuous-time limit is the VP-SDE:

$$
dX_t = -\frac{1}{2} \beta(t) X_t dt + \sqrt{\beta(t)} dW_t
$$

where $\beta(t)$ is a continuous noise schedule function. The drift term $-\frac{1}{2}\beta(t) X_t$ scales the data towards zero, while the diffusion term $\sqrt{\beta(t)} dW_t$ adds Gaussian noise. The specific coefficients ensure that the variance is preserved over time when starting from unit variance data. The marginal distribution at time $t$ under this VP-SDE is Gaussian:

$$
p_t(X_t | X_0) = \mathcal{N}\left(X_t; \sqrt{\bar{\alpha}_t} X_0, (1 - \bar{\alpha}_t) I\right)
$$

where $\bar{\alpha}_t = \exp\left(-\int_0^t \beta(s) ds\right)$. This matches exactly the form we derived for DDPM.

A key insight is that predicting the noise $\epsilon$ in DDPM is equivalent to predicting the score function. Given the conditional distribution above, we can compute its score:

$$
\begin{align*}
\nabla_{X_t} \log p_t(X_t | X_0) &= \nabla_{X_t} \log \mathcal{N}\left(X_t; \sqrt{\bar{\alpha}_t} X_0, (1 - \bar{\alpha}_t) I\right) \\
&= -\frac{X_t - \sqrt{\bar{\alpha}_t} X_0}{1 - \bar{\alpha}_t} \\
&= -\frac{\epsilon}{\sqrt{1 - \bar{\alpha}_t}}
\end{align*}
$$

where we used $X_t = \sqrt{\bar{\alpha}_t} X_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon$. This shows that the score function is directly related to the noise $\epsilon$ through:

$$
\epsilon = -\sqrt{1 - \bar{\alpha}_t} \nabla_{X_t} \log p_t(X_t | X_0)
$$

Therefore, a model $\epsilon_\theta(X_t, t)$ that predicts noise is related to a score model $s_\theta(X_t, t)$ by:

$$
s_\theta(X_t, t) = -\frac{\epsilon_\theta(X_t, t)}{\sqrt{1 - \bar{\alpha}_t}}
$$

This connection shows that the DDPM training objective (minimizing noise prediction error) and the score matching objective are equivalent up to a weighting factor:

$$
\begin{align*}
L_{\text{DDPM}}(\theta) &= \mathbb{E}_{X_0, \epsilon, t} \left[ \| \epsilon - \epsilon_\theta (X_t, t) \|^2 \right] \\
L_{\text{score}}(\theta) &= \mathbb{E}_{X_0, \epsilon, t} \left[ (1 - \bar{\alpha}_t) \| s_\theta (X_t, t) - \nabla_{X_t} \log p_t(X_t | X_0) \|^2 \right]
\end{align*}
$$

These two objectives are equivalent, just parameterized differently. In practice, DDPM uses noise prediction $\epsilon_\theta$ while score-based models use score prediction $s_\theta$, but they are learning the same underlying quantity.

## Denoising Diffusion Implicit Models

So far we have seen that sampling from diffusion models requires simulating the reverse process using many time steps. In DDPM, this typically requires 1000 steps to generate high-quality samples, making generation slow. [Denoising Diffusion Implicit Models (DDIM)](https://arxiv.org/abs/2010.02502), introduced by Song, Meng, and Ermon in 2020, addresses this limitation by defining a **non-Markovian** generative process that enables much faster sampling while using the same trained model from DDPM.

The key insight behind DDIM is that the DDPM training objective only depends on the **marginal distributions** $q(x_t | x_0)$ at each time step, not on the full joint distribution $q(x_{1:T} | x_0)$ of the entire trajectory. This means that many different forward processes can share the same marginals and thus use the same trained model. DDIM exploits this freedom to define a more efficient reverse process.

Instead of the Markovian forward process used in DDPM where $q(x_t | x_{t-1}, x_0) = q(x_t | x_{t-1})$ depends only on the previous step, DDIM uses a **non-Markovian** forward process where the transition depends directly on both the current noisy state and the original clean data:

$$
q_\sigma(x_{t-1} | x_t, x_0) = \mathcal{N}\left(x_{t-1}; \sqrt{\bar{\alpha}_{t-1}} \frac{x_t - \sqrt{1-\bar{\alpha}_t} \epsilon_\theta(x_t, t)}{\sqrt{\bar{\alpha}_t}} + \sqrt{1 - \bar{\alpha}_{t-1} - \sigma_t^2} \epsilon_\theta(x_t, t), \sigma_t^2 I\right)
$$

where $\epsilon_\theta(x_t, t)$ is the trained noise prediction model and $\sigma_t$ is a parameter that controls the amount of stochasticity. This can be rewritten as the sampling update:

$$
x_{t-1} = \sqrt{\bar{\alpha}_{t-1}} \underbrace{\frac{x_t - \sqrt{1-\bar{\alpha}_t} \epsilon_\theta(x_t, t)}{\sqrt{\bar{\alpha}_t}}}_{\text{predicted } x_0} + \underbrace{\sqrt{1 - \bar{\alpha}_{t-1} - \sigma_t^2} \epsilon_\theta(x_t, t)}_{\text{direction pointing to } x_t} + \underbrace{\sigma_t \epsilon}_{\text{random noise}}
$$

where $\epsilon \sim \mathcal{N}(0, I)$. The formula shows three components: a prediction of the clean image $x_0$, a directional term pointing from $x_0$ towards $x_t$, and a random noise term.

The crucial feature is the $\sigma_t$ parameter, which is commonly set as $\sigma_t^2 = \eta \cdot \tilde{\beta}_t$ where $\eta$ controls the stochasticity and $\tilde{\beta}_t$ is related to the noise schedule. When $\eta = 1$ and $\sigma_t = \sqrt{\frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t}} \sqrt{1-\frac{\bar{\alpha}_t}{\bar{\alpha}_{t-1}}}$, this recovers the original DDPM. When $\eta = 0$ and thus $\sigma_t = 0$, the random noise term vanishes and we get a **deterministic** generative process:

$$
x_{t-1} = \sqrt{\bar{\alpha}_{t-1}} \frac{x_t - \sqrt{1-\bar{\alpha}_t} \epsilon_\theta(x_t, t)}{\sqrt{\bar{\alpha}_t}} + \sqrt{1 - \bar{\alpha}_{t-1}} \epsilon_\theta(x_t, t)
$$

This deterministic mapping means that starting from the same initial noise $x_T$, we will always generate the same sample $x_0$. This property is valuable for applications requiring consistency and reproducibility.

{{< figure
    src="/images/ml/diffusionDDIM.png"
    alt="Comparison of DDPM stochastic sampling vs DDIM deterministic sampling trajectories."
    caption="Comparison of DDPM stochastic sampling vs DDIM deterministic sampling trajectories."
>}}

The main advantage of DDIM is that it allows **accelerated sampling** by skipping time steps. Because the process is non-Markovian, we can define a subsequence of time steps $\tau = (\tau_1, \tau_2, \ldots, \tau_S)$ where $\tau_1 = T$ and $\tau_S = 0$, and only perform updates at these steps. Instead of using all 1000 time steps as in DDPM, we might use only 50 or 100 steps. DDIM can produce high-quality samples **10 times to 50 times faster** in terms of wall-clock time compared to DDPM by using this strided sampling schedule.

Interestingly, this deterministic process can be interpreted as discretizing a **probability flow ODE**, a concept from the score-based SDE framework. A probability flow ODE is an ordinary differential equation (without stochastic terms) that has the same marginal distributions as the reverse SDE. For the VP-SDE, this ODE takes the form:

$$
dX_t = \left[f(X_t, t) - \frac{1}{2}g(t)^2 \nabla_X \log p_t(X_t)\right]dt
$$

This ODE has no stochastic $dW_t$ term and generates samples by following deterministic trajectories along the probability distribution.

In practice, deterministic DDIM sampling often produces slightly sharper images because the deterministic trajectory more directly follows the high-probability path through the distribution. However, this can also mean slightly lower sample diversity compared to stochastic DDPM sampling, since the stochastic noise allows exploration of different modes of the distribution. The parameter $\eta$ allows interpolating between these two extremes, trading off between sampling speed, determinism, and diversity.

## Diffusion Backbones

Notice that so far we have not specified the architecture of the model to be used in the reverse denoising process. Our only requirement is that the the dimensionality of the input and output match the data dimensionality d. So in the case of images we need a model that takes in an image and outputs an image of the same size. 

### U-Net

The original choice for the denoising model is the U-Net architecture which was first proposed in the context of biomedical image segmentation in [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597). The U-Net is a type of convolutional neural network (CNN) that has an encoder-decoder structure with skip connections between corresponding layers in the encoder and decoder paths. The encoder path consists of a series of convolutional and MaxPooling layers that progressively downsample the input image such that the spatial information is reduced while feature information is increased. The decoder path then consists of a series of upsampling and convolutional layers that progressively reconstruct the image back to its original size. Importantly just like in the ResNet architecture, skip connections are used to directly connect feature maps from the encoder to the decoder at corresponding spatial resolutions. This allows the decoder to leverage both high-level semantic information from the encoder as well as low-level spatial details from earlier layers, resulting in more accurate reconstructions and avoiding gradient vanishing issues.

{{< figure 
    src="/images/ml/mlUNet.png" 
    alt="U-Net Architecture used in the original Biomedical Image Segmentation paper."
    caption="U-Net Architecture used in the original Biomedical Image Segmentation paper."
>}}

Also has attention somewhore

and to know at which time it is uses sinusoidaL embeddings?

Second paper by openai "Diffusion Models Beat GANs on Image Synthesis" with some changes and also introduces classifier guidance.

### Diffusion Transformers

skip for now

## Latent Diffusion Models

The diffusion models we have discussed so far operate directly in pixel space, meaning they denoise images at the full resolution where each pixel is treated as a separate dimension. For a $512 \times 512$ RGB image, this corresponds to a dimensionality of $d = 512 \times 512 \times 3 = 786'432$. Operating in such high-dimensional spaces presents several computational challenges. First, each denoising step requires processing the entire high-dimensional representation through the neural network backbone, leading to substantial computational costs that scale with image resolution. Second, storing intermediate activations and gradients for such large tensors requires significant memory, limiting the batch sizes that can be used during training and the resolution of images that can be generated. 

[Latent Diffusion Models (LDMs)](https://arxiv.org/abs/2112.10752), introduced by Rombach et al. in their 2022 CVPR paper "High-Resolution Image Synthesis with Latent Diffusion Models", address these computational limitations by applying diffusion in a compressed latent space rather than directly in pixel space. This is the architecture underlying Stable Diffusion and many modern text-to-image models. The key insight is that natural images contain significant redundancy and can be compressed into much lower-dimensional representations without losing perceptually important information. By performing diffusion in this compressed space, we can achieve dramatic computational savings while maintaining high-quality generation.

{{< figure
    src="/images/ml/diffusionLatent.png"
    alt="Latent diffusion models apply the diffusion process in the compressed latent space of a pretrained autoencoder."
    caption="Latent diffusion models compress images into a lower-dimensional latent space using a VAE encoder, perform diffusion in this latent space, then decode back to pixel space using the VAE decoder."
>}}

### Pretrained Autoencoder

The foundation of latent diffusion is a pretrained [autoencoder](/garden/ml/computerVision/autoencoders/) that learns to compress images into a lower-dimensional latent representation. Specifically, LDMs use a [Variational Autoencoder (VAE)](/garden/ml/computerVision/autoencoders/#variational-autoencoders) trained with a perceptual loss that preserves visually important details while discarding imperceptible information.

The VAE consists of an encoder $\mathcal{E}$ that maps an image $x \in \mathbb{R}^{H \times W \times 3}$ to a latent representation $z = \mathcal{E}(x) \in \mathbb{R}^{h \times w \times c}$, and a decoder $\mathcal{D}$ that reconstructs the image from the latent variable as $\tilde{x} = \mathcal{D}(z) \in \mathbb{R}^{H \times W \times 3}$. The spatial dimensions are downsampled by a factor $f = \frac{H}{h} = \frac{W}{w}$ called the downsampling factor. Common choices are $f = 4, 8, 16$, meaning a $512 \times 512$ image is compressed to $128 \times 128$, $64 \times 64$, or $32 \times 32$ latent representations respectively.

The compression ratio is determined by the downsampling factor and the number of latent channels. For a downsampling factor $f = 8$ with $c = 4$ latent channels, a $512 \times 512 \times 3 = 786'432$ dimensional image is compressed to $64 \times 64 \times 4 = 16'384$ dimensions, achieving a compression ratio of $\frac{786'432}{16'384} = 48\times$. This means the latent space has 48 times fewer dimensions than the pixel space, leading to corresponding reductions in computation and memory.

Recall from the [VAE notes](/garden/ml/computerVision/autoencoders/#the-elbo-objective) that a standard VAE maximizes the Evidence Lower Bound (ELBO):

$$
\mathcal{L}(\theta, \phi; x) = \mathbb{E}_{z \sim q_\phi(z \mid x)} [\log p_\theta(x \mid z)] - \text{KL}(q_\phi(z \mid x) \parallel p(z))
$$

where the first term is the reconstruction term and the second term is the KL regularization that encourages the encoder distribution $q_\phi(z \mid x)$ to match the prior $p(z) = \mathcal{N}(0, I)$. For a Gaussian likelihood, the reconstruction term becomes proportional to negative mean squared error.

For latent diffusion models, the VAE is trained with a modified objective that uses perceptual losses rather than simple pixel-wise reconstruction. The training objective is:

$$
\mathcal{L}_{\text{VAE}} = \mathcal{L}_{\text{rec}}(x, \mathcal{D}(\mathcal{E}(x))) + \mathcal{L}_{\text{reg}}(\mathcal{E}(x))
$$

The reconstruction loss combines multiple components to capture perceptual quality:

$$
\mathcal{L}_{\text{rec}} = \|x - \mathcal{D}(\mathcal{E}(x))\|_1 + \mathcal{L}_{\text{perceptual}} + \mathcal{L}_{\text{adv}}
$$

The first term is an $L_1$ pixel-wise loss that ensures basic reconstruction accuracy. The perceptual loss $\mathcal{L}_{\text{perceptual}}$ measures reconstruction quality in feature space rather than pixel space. It is computed by extracting features from intermediate layers of a pretrained network (such as a VGG network or a discriminator) for both the original image and the reconstruction, then computing the distance between these features:

$$
\mathcal{L}_{\text{perceptual}} = \sum_{l} \|f_l(x) - f_l(\mathcal{D}(\mathcal{E}(x)))\|_2^2
$$

where $f_l$ denotes features from layer $l$. This ensures that reconstructions are visually similar to the originals in terms of high-level semantic content rather than just pixel values. The adversarial loss $\mathcal{L}_{\text{adv}}$ uses a discriminator to encourage reconstructions to lie on the manifold of realistic images, further improving perceptual quality.

The regularization term $\mathcal{L}_{\text{reg}}$ is a KL divergence penalty, but LDMs use much weaker regularization than standard VAEs:

$$
\mathcal{L}_{\text{reg}} = \text{KL}(q_\phi(z \mid x) \parallel p(z))
$$

This is typically weighted by a small coefficient (or implemented as a KL-regularized variant) to prioritize reconstruction quality over strict adherence to the Gaussian prior. The goal is to learn a compressed representation that preserves perceptually important details while maintaining a reasonably regular latent space that the diffusion model can learn.

Once the VAE is trained, the encoder and decoder are frozen and used to map between pixel space and latent space during diffusion model training and sampling.

### Diffusion in Latent Space

With a pretrained VAE, latent diffusion operates entirely in the compressed latent space. Given a dataset of images $\{x_1, x_2, \ldots, x_N\}$, we first encode them to latent representations $\{z_1, z_2, \ldots, z_N\}$ where $z_i = \mathcal{E}(x_i)$. We then train a diffusion model $\epsilon_\theta$ to denoise these latent representations rather than the original images.

The forward diffusion process adds Gaussian noise to the latent representations following the same schedule as standard diffusion:

$$
q(z_t | z_0) = \mathcal{N}(z_t; \sqrt{\bar{\alpha}_t} z_0, (1 - \bar{\alpha}_t) I)
$$

where $z_0 = \mathcal{E}(x)$ is the clean latent variable and $z_t$ is the noisy latent at time step $t$. This is identical to the pixel-space formulation, except we are now working with latent variables $z \in \mathbb{R}^{h \times w \times c}$ instead of images $x \in \mathbb{R}^{H \times W \times 3}$.

The reverse denoising process learns to predict the noise $\epsilon$ added at each step, just as in the original DDPM formulation. The denoising network $\epsilon_\theta(z_t, t)$ takes as input the noisy latent $z_t$ and the time step $t$, and outputs an estimate of the noise. The training objective is the simplified denoising objective:

$$
\mathcal{L}_{\text{LDM}} = \mathbb{E}_{z \sim \mathcal{E}(x), \epsilon \sim \mathcal{N}(0,I), t} \left[ \| \epsilon - \epsilon_\theta(z_t, t) \|_2^2 \right]
$$

where $z_t = \sqrt{\bar{\alpha}_t} z + \sqrt{1 - \bar{\alpha}_t} \epsilon$ is the noisy latent at time step $t$. This is exactly the same training objective as DDPM, but operating on the compressed latent representations.

The denoising network architecture is typically a U-Net adapted to the latent space dimensions. Since the latent space has much smaller spatial dimensions but more channels than RGB images, the U-Net processes inputs of shape $h \times w \times c$ rather than $H \times W \times 3$. The time step $t$ is typically embedded using sinusoidal position encodings and injected into the network through adaptive group normalization layers, allowing the network to adapt its behavior based on the noise level.

### Sampling Process

Generating new images with a latent diffusion model involves three steps. First, we sample pure Gaussian noise in the latent space: $z_T \sim \mathcal{N}(0, I)$ where $z_T \in \mathbb{R}^{h \times w \times c}$. Second, we apply the learned reverse diffusion process to iteratively denoise the latent variable. Using the DDIM sampling scheme for efficiency, we update:

$$
z_{t-1} = \sqrt{\bar{\alpha}_{t-1}} \underbrace{\frac{z_t - \sqrt{1-\bar{\alpha}_t} \epsilon_\theta(z_t, t)}{\sqrt{\bar{\alpha}_t}}}_{\text{predicted } z_0} + \sqrt{1 - \bar{\alpha}_{t-1}} \epsilon_\theta(z_t, t)
$$

This deterministic update is applied for $t = T, T-1, \ldots, 1$ to obtain the clean latent variable $z_0$. Third, we decode the clean latent variable back to pixel space using the VAE decoder: $x = \mathcal{D}(z_0)$.

The decoder is deterministic and runs only once at the end of generation, so the computational cost is negligible compared to the iterative denoising process. All the expensive iterative denoising happens in the compressed latent space, providing substantial speedups.

## Conditional Diffusion

So far we have discussed unconditional diffusion models that learn to generate samples from the data distribution without any external control. However, many applications require conditional generation where we want to control specific aspects of the generated samples. For example, we might want to generate images from text descriptions (text-to-image), generate images conditioned on semantic layouts (layout-to-image), or generate specific classes of images (class-conditional generation).

There are two fundamentally different approaches to conditional generation in diffusion models: **classifier guidance** and **classifier-free guidance**. These differ not just in their sampling procedures but in their entire training paradigms. Classifier guidance trains an unconditional diffusion model and uses a separate classifier to guide generation at sampling time, while classifier-free guidance trains a single conditional model that learns both conditional and unconditional denoising simultaneously. We present these approaches in order, starting with classifier guidance to understand the mathematical foundations of guided diffusion.

### Classifier Guidance

Classifier guidance was introduced in "Diffusion Models Beat GANs on Image Synthesis" by Dhariwal and Nichol. The approach uses a pretrained classifier to guide an unconditional diffusion model toward desired conditions at sampling time.

During training, we build two separate models. First, we train an unconditional diffusion model $\epsilon_\theta(x_t, t)$ using the standard DDPM objective. Second, we train a classifier $p_\phi(c | x_t, t)$ that can classify noisy images at any noise level $t$. This classifier is trained on noisy samples from the forward diffusion process, learning to recognize classes even in heavily corrupted images.

At inference time, to generate a sample of class $c$, we start from pure noise $x_T \sim \mathcal{N}(0, I)$ and iteratively denoise using a modified score that incorporates the classifier gradient. The key mathematical insight comes from Bayes' rule applied to the score function:

$$
\nabla_{x_t} \log p(x_t | c) = \nabla_{x_t} \log p(x_t) + \nabla_{x_t} \log p(c | x_t)
$$

This shows that the conditional score (how to denoise to get class $c$) equals the unconditional score plus the gradient of the log probability that the current noisy image belongs to class $c$. The unconditional score is approximated by our trained diffusion model, and the classifier provides the second term.

Since the noise prediction $\epsilon_\theta(x_t, t)$ relates to the score as $\epsilon_\theta(x_t, t) \approx -\sqrt{1 - \bar{\alpha}_t} \nabla_{x_t} \log p(x_t)$, we can modify it to incorporate the classifier gradient. At each denoising step, we compute:

$$
\tilde{\epsilon}_\theta(x_t, t, c) = \epsilon_\theta(x_t, t) - w \sqrt{1 - \bar{\alpha}_t} \nabla_{x_t} \log p_\phi(c | x_t, t)
$$

where $w \geq 1$ is a guidance scale controlling conditioning strength. The term $\nabla_{x_t} \log p_\phi(c | x_t, t)$ is computed by evaluating the classifier on the current noisy image $x_t$ and backpropagating through it with respect to the input. This gradient points in the direction that would make the classifier more confident that $x_t$ belongs to class $c$.

To generate an image of class $c$:
1. Start with noise $x_T \sim \mathcal{N}(0, I)$
2. For each time step $t = T, T-1, \ldots, 1$:
   - Compute unconditional noise prediction: $\epsilon_\theta(x_t, t)$
   - Compute classifier gradient: $\nabla_{x_t} \log p_\phi(c | x_t, t)$
   - Compute guided noise prediction: $\tilde{\epsilon}_\theta(x_t, t, c) = \epsilon_\theta(x_t, t) - w \sqrt{1 - \bar{\alpha}_t} \nabla_{x_t} \log p_\phi(c | x_t, t)$
   - Update using this guided prediction: $x_{t-1} = \text{DDPM or DDIM update}(\tilde{\epsilon}_\theta)$
3. Return $x_0$

The guidance scale $w$ allows trading off between sample quality and conditioning strength. Higher values produce samples that more strongly match the desired class but may reduce diversity or realism.

The main limitation of classifier guidance is the need to train a separate noise-robust classifier. This classifier must work across all noise levels, which is non-trivial. Additionally, running both the diffusion model and classifier at each sampling step increases computational cost. These limitations motivated the development of classifier-free guidance.

### Conditional Training

Unlike classifier guidance which uses an unconditional model, conditional diffusion approaches train a model that directly incorporates the conditioning information. When we have paired training data (images with associated labels, captions, or other conditioning information), we modify the denoising network to take conditioning as an additional input.

The noise prediction network becomes $\epsilon_\theta(x_t, t, c)$ where $c$ is the conditioning signal. The training objective extends the unconditional DDPM objective:

$$
\mathcal{L}_{\text{conditional}} = \mathbb{E}_{x, c, \epsilon, t} \left[ \| \epsilon - \epsilon_\theta(x_t, t, c) \|_2^2 \right]
$$

where $x \sim q(x)$ is sampled from the data distribution, $c \sim p(c|x)$ is the conditioning information associated with $x$ (such as a caption describing the image), $\epsilon \sim \mathcal{N}(0, I)$ is random noise, and $t \sim \text{Uniform}(1, T)$ is a random time step. The key question is how to architecturally incorporate the conditioning signal $c$ into the network, which depends on the nature and structure of the conditioning information.

### Conditioning Mechanisms

There are several architectural approaches for incorporating conditioning information into the denoising network.

**Concatenation-based conditioning** is the simplest approach. The conditioning information is encoded into a tensor with spatial dimensions matching the noisy input $x_t$, and the two are concatenated along the channel dimension before being fed into the U-Net. For example, if $x_t \in \mathbb{R}^{H \times W \times 3}$ and $c$ is encoded as $c_{\text{enc}} \in \mathbb{R}^{H \times W \times d}$, we concatenate to form $[x_t, c_{\text{enc}}] \in \mathbb{R}^{H \times W \times (3+d)}$ and process this through the U-Net. This approach works well when the conditioning is spatially aligned with the output, such as conditioning on segmentation maps or depth maps where each spatial location in the condition corresponds to a location in the output image. The U-Net's convolutional layers can then locally combine information from the noisy image and the condition.

**Adaptive normalization** injects conditioning through the normalization layers of the network. Instead of using standard normalization, we use adaptive normalization layers where the normalization parameters (scale and shift) are predicted from the conditioning $c$. For a feature map $h$, instead of normalizing with fixed parameters, we compute:

$$
\text{AdaGN}(h, c) = \gamma(c) \frac{h - \mu(h)}{\sigma(h)} + \beta(c)
$$

where $\mu(h)$ and $\sigma(h)$ are the mean and standard deviation of the feature map, and $\gamma(c)$ and $\beta(c)$ are scale and shift parameters predicted from the conditioning via small neural networks. This approach allows the conditioning to modulate the entire network's behavior. It works particularly well for global conditioning information like class labels or text embeddings that do not have spatial structure.

**Cross-attention conditioning** is the most flexible and powerful approach, particularly for text-to-image generation. This method was introduced in the latent diffusion paper and has become the standard for modern text-to-image models. The conditioning information is first encoded into a sequence of embeddings, and then the denoising network attends to these embeddings using [cross-attention layers](/garden/ml/llms/attention/#cross-attention) from the [Transformer architecture](/garden/ml/llms/transformers/).

For text conditioning, we use a pretrained text encoder to convert the text prompt into embeddings. The [CLIP model](/garden/ml/computerVision/clip/) has become a popular choice because it was trained on large datasets of image-text pairs and learns a joint embedding space where semantically related images and text are close together. CLIP consists of separate image and text encoders trained with a contrastive objective, ensuring that the text representations capture visual concepts. For conditional diffusion, we use only the CLIP text encoder, which is a [Transformer](/garden/ml/llms/transformers/) that processes tokenized text through multiple self-attention layers.

Given a text prompt like "a photo of a cat sitting on a couch", we first tokenize it into individual tokens such as ["a", "photo", "of", "a", "cat", "sitting", "on", "a", "couch"]. We then pass this sequence through the text encoder to obtain a sequence of contextualized embeddings $c = \{c_1, c_2, \ldots, c_n\} \in \mathbb{R}^{n \times d_c}$ where $n$ is the number of tokens and $d_c$ is the embedding dimension. Each embedding $c_i$ is different and captures the semantic meaning and contextual role of its corresponding token. The embedding for "cat" encodes information about the animal concept, while the embedding for "sitting" encodes the action.

The Transformer text encoder processes the entire sequence using self-attention, allowing each token's embedding to be informed by the surrounding context. This contextualization means that the same word can have different embeddings depending on its context (for example, "bank" in "river bank" versus "money bank"). The text encoder is typically frozen during diffusion model training, using the representations learned during its pretraining. This allows the diffusion model to leverage the encoder's existing semantic understanding without the computational cost of fine-tuning a large language model. While CLIP is commonly used, other text encoders like T5 or BERT can also be employed.

To integrate cross-attention into the U-Net architecture, we insert cross-attention layers at various resolutions that allow the image features to attend to the text embeddings. Given intermediate feature maps $h \in \mathbb{R}^{h \times w \times d_h}$ from the U-Net, we flatten them to a sequence $h \in \mathbb{R}^{(hw) \times d_h}$ and compute cross-attention with the text embeddings $c \in \mathbb{R}^{n \times d_c}$.

Following the attention mechanism described in the [attention notes](/garden/ml/llms/attention/#cross-attention), we compute queries from the image features and keys and values from the text embeddings:

$$
Q = hW_Q \in \mathbb{R}^{(hw) \times d_k}, \quad K = cW_K \in \mathbb{R}^{n \times d_k}, \quad V = cW_V \in \mathbb{R}^{n \times d_v}
$$

where $W_Q \in \mathbb{R}^{d_h \times d_k}$, $W_K \in \mathbb{R}^{d_c \times d_k}$, and $W_V \in \mathbb{R}^{d_c \times d_v}$ are learnable projection matrices. The attention output is computed as:

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V \in \mathbb{R}^{(hw) \times d_v}
$$

The attention weight matrix has shape $(hw) \times n$, where entry $(i,j)$ represents how much spatial position $i$ in the image attends to text token $j$. This allows different regions of the generated image to focus on different parts of the text prompt. For example, when generating "a red ball next to a blue cube", the region generating the ball can attend strongly to the tokens "red" and "ball", while the region generating the cube attends to "blue" and "cube".

These cross-attention layers are placed within the U-Net's residual blocks at each resolution level throughout both the encoder (downsampling) and decoder (upsampling) paths. At each resolution, a typical block processes features through:

1. Residual convolutional layers with adaptive group normalization (incorporating the time embedding)
2. Spatial self-attention (allowing spatial locations to exchange information)
3. Cross-attention to text embeddings (incorporating the conditioning)

This pattern is repeated at multiple scales as the U-Net downsamples and then upsamples the spatial resolution. The architecture is inspired by the [Transformer decoder](/garden/ml/llms/transformers/#decoder-architecture) which similarly interleaves self-attention and cross-attention. By applying cross-attention at multiple resolutions, the conditioning influences both coarse structure (global composition and layout at low resolutions) and fine details (textures and local features at high resolutions).

### Classifier-Free Guidance

While conditional diffusion models can generate samples matching the conditioning, they often struggle with the trade-off between sample quality and conditioning strength. Samples that strictly follow the conditioning may lack diversity, while samples with high diversity may not faithfully follow the conditioning. [Classifier-free guidance](https://arxiv.org/abs/2207.12598), introduced by Ho and Salimans, provides a way to control this trade-off without requiring a separate classifier network.

The key insight is to train a single conditional diffusion model that can operate both conditionally and unconditionally. During training, we randomly drop the conditioning information with some probability $p_{\text{uncond}}$ (typically 10-20%). When the conditioning is dropped, the model learns the unconditional distribution $p(x)$, while with conditioning it learns $p(x|c)$. This is implemented by replacing the conditioning $c$ with a special null token or empty embedding during these training steps.

Formally, the training objective becomes:

$$
\mathcal{L}_{\text{CFG}} = \mathbb{E}_{x, c, \epsilon, t} \left[ \| \epsilon - \epsilon_\theta(x_t, t, c') \|_2^2 \right]
$$

where $c' = \emptyset$ with probability $p_{\text{uncond}}$ and $c' = c$ otherwise. The model thus learns to predict noise both with and without conditioning, sharing most parameters but adapting its behavior based on whether conditioning is provided.

At inference time, classifier-free guidance achieves stronger conditioning by combining both the conditional and unconditional predictions from the same model. To understand this intuitively, consider denoising a noisy image $x_t$ while adhering to condition $c$ (such as "a red car"). The unconditional prediction $\epsilon_\theta(x_t, t, \emptyset)$ tells us what noise to remove if we have no specific requirements, denoising toward any plausible object. The conditional prediction $\epsilon_\theta(x_t, t, c)$ tells us what noise to remove to get a sample matching $c$, denoising specifically toward a red car.

The difference between these predictions, $\epsilon_\theta(x_t, t, c) - \epsilon_\theta(x_t, t, \emptyset)$, represents the direction in noise space that moves from "any object" toward "red car specifically". By starting from the unconditional prediction and adding a scaled version of this difference, we control how strongly the result adheres to the conditioning.

At each denoising step during sampling, we compute both predictions and combine them as:

$$
\tilde{\epsilon}_\theta(x_t, t, c) = \epsilon_\theta(x_t, t, \emptyset) + w \cdot \left(\epsilon_\theta(x_t, t, c) - \epsilon_\theta(x_t, t, \emptyset)\right)
$$

where $w \geq 1$ is the guidance scale. This can be rewritten as a weighted combination:

$$
\tilde{\epsilon}_\theta(x_t, t, c) = (1 - w) \epsilon_\theta(x_t, t, \emptyset) + w \cdot \epsilon_\theta(x_t, t, c)
$$

The guidance scale $w$ controls how strongly we push toward the conditioning. When $w = 1$, we get exactly the conditional prediction $\epsilon_\theta(x_t, t, c)$. When $w > 1$, we amplify the difference from the unconditional baseline, making the result adhere even more strongly to the condition. The conditional prediction already tries to match $c$, but by amplifying its difference from the unconditional prediction with $w > 1$, we push even harder in the direction of the conditioning.

Higher guidance scales produce samples that more faithfully match the conditioning but may reduce diversity and sometimes lead to oversaturated or less realistic images. Lower guidance scales produce more diverse and often more realistic samples but with weaker adherence to the conditioning. In practice, guidance scales of $w = 7$ to $w = 15$ are commonly used for text-to-image generation, with the optimal value depending on the specific model and application.

The theoretical justification for classifier-free guidance comes from the relationship between the conditional and unconditional score functions. The score function $\nabla_x \log p(x|c)$ can be decomposed as:

$$
\nabla_x \log p(x|c) = \nabla_x \log p(x) + \nabla_x \log p(c|x)
$$

The unconditional noise prediction $\epsilon_\theta(x_t, t, \emptyset)$ approximates the unconditional score, while the difference $\epsilon_\theta(x_t, t, c) - \epsilon_\theta(x_t, t, \emptyset)$ approximates the classifier gradient $\nabla_x \log p(c|x)$. The guidance scale $w$ acts as a temperature parameter that sharpens the conditional distribution.

Classifier-free guidance has become the standard approach for conditional generation in modern diffusion models. It requires no additional classifier network, uses the same diffusion model for both guided and unguided generation, and provides a simple hyperparameter to control the conditioning strength. Stable Diffusion and most text-to-image models rely heavily on classifier-free guidance to achieve their impressive adherence to text prompts.

### LoRa

skip for now

### ControlNet

skip for now

### DreamBooth

skip for now

## Super Resolution Diffusion Models

skip for now
