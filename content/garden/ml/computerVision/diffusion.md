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

Used Resources:
- https://yang-song.net/blog/2021/score/
- https://theaisummer.com/diffusion-models/
- https://ayandas.me/blogs/2021-12-04-diffusion-prob-models.html
- https://lilianweng.github.io/posts/2021-07-11-diffusion-models/
- https://goyalpramod.github.io/blogs/demysitifying_diffusion_models/
- https://www.chenyang.co/diffusion.html
- https://huggingface.co/blog/annotated-diffusion
- https://github.com/diff-usion/Awesome-Diffusion-Models
- https://sander.ai/
- And all the corresponding papers linked throughout the text.
- https://www.youtube.com/watch?v=HoKDTa5jHvg&t=1487s
- https://www.youtube.com/watch?v=lUljxdkolK8&t=380s

## Denoising Diffusion Probabilistic Models

As is common in computer science and machine learning, it is often easier to solve a complex problem by breaking it down into smaller subproblems. In the case of generative modeling, instead of learning to directly generate samples from the complex data distribution $q$, we can instead learn to gradually improve our samples in an iterative/recursive manner to reach the desired result. Think of it as painting a picture step by step, starting from a blank canvas and adding more and more details and layering colors of paint until the final masterpiece is complete, rather than trying to paint the entire picture in one go. 

The idea of diffusion models is to use a stochastic process to gradually transform simple known distributions such as Gaussian noise into our complex data distribution $q$. **The main idea behind diffusion models is to define a forward diffusion process that gradually adds noise to the data samples until they become pure noise, and then learn a neural network to reverse this process and denoise the noisy samples back to the original data distribution.** By iteratively applying this denoising process starting from a pure noise sample, we can generate new samples that closely resemble samples from the true data distribution.

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
q(x_t | x_{t_1}, x_{t-2}, ..., x_0) = q(x_t | x_{t-1})
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

But we still onyl have access to $x_0$ at training time but not at sampling time as we only start from pure noise $x_T \sim N(0, I)$ and we don't know the parameters $\tilde{\mu}(x_t, x_0)$ and $\tilde{\beta}_t(x_t, x_0)$ of the reverse conditional distribution. So the idea is that we can instead learn a neural network model $p_\theta(x_{t-1} | x_t)$ parametrized by $\theta$ to approximate this reverse conditional distribution:

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

To achieve this as is often the case in machine learning, we want to find some parameters $\theta$ that maximize the likelihood of the observed data samples under the model $p(x_0 | \theta) = p_\theta(x_0)$. However, directly maximizing the likelihood is intractable because we would need to marginalize over all possible trajectories from $x_T$ to $x_0$. So in other words, over all possible ways noise could have been added:

$$
p_\theta(x_0) = \int p_\theta(x_{0:T}) dx_{1:T}
$$

This integral is intractable due to the high dimensionality of the space and the complex dependencies between the variables. So instead of maximizing the likelihood or the log likelihood directly, we can instead **maximize a lower bound on the log likelihood $\log p_\theta(x_0)$ called the evidence lower bound (ELBO) using variational inference**. We use this lower bound as a surrogate objective that is easier to optimize and still leads to good enough solutions, but not necessarily the optimal solution due to being a lower bound. Because we also prefer minimization problems in machine learning, we usually **minimize the negative log likelihood or equivalently minimize the negative ELBO**.

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

In this case compared to usually bayesian inference, we are not trying to find the posterior distribution given a prior and a likelihood function, but instead we are trying to find the likelihood function (the reverse denoising process) and the prior the trajectory distribution (the forward diffusion process) that best explains the observed data samples:

- **Prior (Unknown)**: The forward diffusion process $p_\theta(x_{1:T})$ which represents a specific way of adding noise to the data samples.
- **Likelihood (Unknown)**: The reverse denoising process $p_\theta(x_0 | x_{1:T})$ which represents a specific way of removing noise from the noisy samples.
- **Posterior (Known)**: The forward trajectory distribution $p_\theta(x_{1:T} | x_0)$ which represents the true way noise was added to the data samples. This can be rewritten as follows using bayes theorem:

$$
p_\theta(x_{1:T} | x_0) = \frac{1}{Z_\theta} p_\theta(x_0 | x_{1:T}) p_\theta(x_{1:T})
$$

where $Z_\theta$ is the normalizing constant $Z_\theta = \int p_\theta(x_0 | x_{1:T}) p_\theta(x_{1:T}) dx_{1:T}$. Both this normalizing constant and the data distribution $p_\theta(x_0)$ are intractable to compute directly as these both involve an integral over the entire space, which is why we resort to maximizing the ELBO instead and in turn maximizing the log likelihood.

There are two possible derivations of the ELBO, one can be derived from minimizing the cross entropy between the true data distribution and the model distribution, while the other can be derived from minimizing the KL divergence between the true trajectory distribution and the model trajectory distribution. 

First we start with **deriving the ELBO form the cross entropy**. The cross entropy between the true data distribution $q(x_0)$ and the model distribution $p_\theta(x_0)$ is defined as:

$$
L_{\text{CE}} = H(q, p_\theta) = - \mathbb{E}_{x_0 \sim q(x_0)} [\log p_\theta(x_0)]
$$

which measures how well the model distribution matches the true data distribution. Minimizing this cross entropy is equivalent to maximizing the log likelihood of the observed data samples under the model. However, as mentioned earlier, directly maximizing the log likelihood is intractable due to the need to marginalize over all possible trajectories, so instead we can use [Jensen's inequality](/garden/maths/probabilitystatistics/expectationvariancecovariance/#jensens-inequality) which states that for a concave function $f$ and a random variable $X$, we have:

$$
f(\mathbb{E}[X]) \geq \mathbb{E}[f(X)]
$$

Luckily the log function is concave, so we can apply Jensen's inequality to the log likelihood and derive the lower bound:

$$
\begin{align*}
\log p_\theta(x_0) &= \log \int p_\theta(x_{0:T}) dx_{1:T} \\
&= \log \int q(x_{1:T} | x_0) \frac{p_\theta(x_{0:T})}{q(x_{1:T} | x_0)} dx_{1:T} \\
&= \log \mathbb{E} \left[q(x_{1:T} | x_0) \frac{p_\theta(x_{0:T})}{q(x_{1:T} | x_0)} \right] &\text{ (Hint to: Cross Entropy) } \\
&= \log \mathbb{E}_{q(x_{1:T} | x_0)} \left[ \frac{p_\theta(x_{0:T})}{q(x_{1:T} | x_0)} \right] &\text{ (by definition of expectation) } \\
&\geq \mathbb{E}_{q(x_{1:T} | x_0)} \left[ \log \frac{p_\theta(x_{0:T})}{q(x_{1:T} | x_0)} \right] &\text{ (by Jensen's inequality) } \\
L_{\text{ELBO}} &= \mathbb{E}_{q(x_{1:T} | x_0)} \left[ \log \frac{q(x_{1:T} | x_0)}{p_\theta(x_{1:T})} \right] \leq \mathbb{E}_{x_0 \sim q(x_0)} [\log p_\theta(x_0)]
\end{align*}
$$

So to maximize the log likelihood we need to minimize the negative ELBO loss which is a lower bound on the negative log likelihood:

$$
\min -L_{\text{ELBO}} = - \mathbb{E}_{q(x_{1:T} | x_0)} \left[ \log \frac{p_\theta(x_{0:T})}{q(x_{1:T} | x_0)} \right] \geq - \mathbb{E}_{q(x_0)} [\log p_\theta(x_0)]
$$

The other way is to **derive the ELBO from the definition of the KL divergence** between the true trajectory distribution $q(x_{1:T} | x_0)$ and the model trajectory distribution $p_\theta(x_{1:T} | x_0)$. The KL divergence is defined as:

$$
D_{KL}(q(x_{1:T} | x_0) \| p_\theta(x_{1:T} | x_0)) = \mathbb{E}_{q(x_{1:T} | x_0)} \left[ \log \frac{q(x_{1:T} | x_0)}{p_\theta(x_{1:T} | x_0)} \right]
$$

Importantly the KL divergence is always non-negative, so we have:

$$
\begin{align*}
0 &\leq D_{KL}(q(x_{1:T} | x_0) \| p_\theta(x_{1:T} | x_0)) \\
-\log p_\theta(x_0) &\leq - \log p_\theta(x_0) + D_{KL}(q(x_{1:T} | x_0) \| p_\theta(x_{1:T} | x_0)) \\
&= -\log p_\theta(x_0) + \mathbb{E}_{q(x_{1:T} | x_0)} \left[ \log \frac{q(x_{1:T} | x_0)}{p_\theta(x_{1:T} | x_0)} \right] & \text{ (by definition of KL divergence) } \\
&= -\log p_\theta(x_0) + \mathbb{E}_{q(x_{1:T} | x_0)} \left[ \log \frac{q(x_{1:T} | x_0)}{\frac{p_\theta(x_0 | x_{1:T}) p_\theta(x_{1:T})}{p_\theta(x_0)}} \right] & \text{ (by bayes theorem) } \\
&= -\log p_\theta(x_0) + \mathbb{E}_{q(x_{1:T} | x_0)} \left[ \log \frac{q(x_{1:T} | x_0)}{\frac{p_\theta(x_{0:T})}{p_\theta(x_0)}} \right] & \text{ (by joint distribution) } \\
&= -\log p_\theta(x_0) + \mathbb{E}_{q(x_{1:T} | x_0)} \left[ \log \frac{q(x_{1:T} | x_0)}{p_\theta(x_{0:T})} + \log p_\theta(x_0) \right] & \text{ (by log and division) } \\
&= -\log p_\theta(x_0) + \log p_\theta(x_0) + \mathbb{E}_{q(x_{1:T} | x_0)} \left[ \log \frac{q(x_{1:T} | x_0)}{p_\theta(x_{0:T})} \right] & \text{ (by linearity of expectation) } \\
L_{\text{ELBO}} &= \mathbb{E}_{q(x_{1:T} | x_0)} \left[ \log \frac{q(x_{1:T} | x_0)}{p_\theta(x_{0:T})} \right]
\end{align*}
$$

Now we have successfully derived the ELBO from two different perspectives and gotten rid of the intractable $\log p_\theta(x_0)$ term. We can now rewrite the ELBO in a more convenient form for optimization where we will make use of having access to $x_0$ during training and conditioning on it as otherwise all the terms would have high variance and be difficult to estimate but giving it the original data sample as a hint makes the estimation easier and more stable:

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

This is now our final expression for the ELBO which we can then minimize the negative of. If we analyze the different terms in this expression we can see that it consists of three main parts:

- The first term $L_t = D_{KL}(q(x_T | x_0) \| p_\theta(x_T))$ measures how well the model can match the distribution of the final noisy sample $x_T$ to the prior distribution. This term can be ignored in practice as if we choose a large enough T and an appropriate noise schedule $\beta_t$, then $q(x_T | x_0)$ will be very close to $N(0, I)$ anyway due to the ergodic property of the forward diffusion process and we pick $p_\theta(x_T) = N(0, I)$ which has no trainable parameters and **is therefore just a constant**, hence this term does not contribute to the optimization and can be ignored during training.

- The last term $L_0 = - \log p_\theta(x_0 | x_1)$ is the last step where we want to reconstruct the original data sample from the slightly noisy sample $x_1$. Because we scaled our image data from being in $[0, 255]$ to $[-1, 1]$ the last step does some weird stuff but this can basically be ignored and **in practice this term is just omitted during training and then at sampling we handle the final step slightly differently**.

So we only need to focus on the main terms $L_t$ for $t = 1, 2, ..., T-1$ which measure how well the model can match the reverse conditional distributions at each time step exactly as we initially set out to do. Performing these simplification results in a **variance reduction**. So our final training objective reduces to minimizing the sum of KL divergences between the true reverse conditional distributions and the learned reverse conditional distributions at each time step. This matches our initial intuition of what we wanted to achieve:

$$
\min_\theta \sum_{t=2}^{T} D_{KL}(q(x_{t-1} | x_t, x_0) \| p_\theta(x_{t-1} | x_t))
$$

Remember that we previously said that if we additionally condition on $x_0$ the problem becomes tractable and we can rewrite the reverse conditional distribution as a gaussian:

$$
q(x_{t-1} | x_t, x_0) = N(x_{t-1}; \tilde{\mu}(x_t, x_0), \tilde{\beta}_t(x_t, x_0) I)
$$

If we now apply bayes theorem again we can derive closed form expressions we get:

$$
q(x_{t-1} | x_t, x_0) = \frac{q(x_t | x_{t-1}) q(x_{t-1} | x_0)}{q(x_t | x_0)}
$$

Each of these components are gaussians as we have already derived the forward conditional distribution $q(x_t | x_{t-1})$ and the marginal distribution $q(x_t | x_0)$ above. We can also derive the conditional distribution $q(x_{t-1} | x_0)$ in a similar way to how we derived $q(x_t | x_0)$ which is just different by one time step. So we can get its closed form by multiplying two gaussians together and reparameterizing to find the mean and variance of the resulting gaussian as a function of $x_t$ and $x_0$:

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

It is important that we add some noise back at each denoising step to maintain stochasticity in the process and ensure diversity in the generated samples. This **avoids collapsing to a single mode and helps explore the data distribution better as otherwise the model is only predicting the mean of the reverse conditional distribution**.

Above we derived that our training objective reduces to minimizing the KL divergence between the true reverse conditional distribution and the learned reverse conditional distribution at each time step. The general form of the KL divergence between two gaussians $N(\mu_1, \Sigma_1)$ and $N(\mu_2, \Sigma_2)$ in $\mathbb{R}^d$ is given by:

{{< callout type="todo" >}}
Actually show the calculations
{{< /callout >}}

$$
D_{KL}(N(\mu_1, \Sigma_1) \| N(\mu_2, \Sigma_2)) = \frac{1}{2} \left( \log \frac{|\Sigma_2|}{|\Sigma_1|} - d + \text{tr}(\Sigma_2^{-1} \Sigma_1) + (\mu_2 - \mu_1)^T \Sigma_2^{-1} (\mu_2 - \mu_1) \right)
$$

If we substitute in our expressions of $\tilde{\mu}(x_t, t)$, $\mu_\theta(x_t, t)$, $\tilde{\beta}_t$ and $\sigma_t^2$ into this expression and simplify **we get something similar to a mean squared error loss between the true noise $\epsilon_t$ and the predicted noise $\epsilon_\theta(x_t, t)$** scaled by some factor:

$$
\begin{align*}
L_t &= \mathbb{E}_{x_0, \epsilon \sim N(0, I)} \left[ \frac{1}{2 \| \Sigma_\theta (x_t, t) \|^2} \| \tilde{\mu}(x_t, t) - \mu_\theta (x_t, t) \|^2 \right] \\
&= \mathbb{E}_{x_0, \epsilon \sim N(0, I)} \left[ \frac{1}{2 \sigma_t^2} \left\| \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \epsilon_t \right) - \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \epsilon_\theta (x_t, t) \right) \right\|^2 \right] \\
&= \mathbb{E}_{x_0, \epsilon \sim N(0, I)} \left[ \frac{1}{2 \sigma_t^2} \left\| \frac{\beta_t}{\sqrt{\alpha_t} \sqrt{1 - \bar{\alpha}_t}} \left( \epsilon_t - \epsilon_\theta (x_t, t) \right) \right\|^2 \right] \\
&= \mathbb{E}_{x_0, \epsilon \sim N(0, I)} \left[ \frac{\beta_t^2}{2 \sigma_t^2 \alpha_t (1 - \bar{\alpha}_t)} \| \epsilon_t - \epsilon_\theta (x_t, t) \|^2 \right] 
\end{align*}
$$

In the [DDPM paper](https://arxiv.org/abs/2006.11239) the authors showed that empirically a simplified version of this loss function outperforms the full variational bound ignoring as when we set the noise schedule $\sigma_t^2 = \beta_t$ the scaling factor can be ignored during optimization. This results in our final simplified loss function for training the denoising model:

$$
L_t^{\text{simple}} = \mathbb{E}_{x_0, \epsilon \sim N(0, I)} \left[ \| \epsilon - \epsilon_\theta (x_t, t) \|^2 \right] = \mathbb{E}_{x_0, \epsilon \sim N(0, I)} \left[ \| \epsilon - \epsilon_\theta (\sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon, t) \|^2 \right]
$$

This can then of course be scaled by a half to match the usual MSE loss convention and make the gradients nicer:

$$
L_t^{\text{simple}} = \frac{1}{2} \mathbb{E}_{x_0, \epsilon \sim N(0, I)} \left[ \| \epsilon - \epsilon_\theta (x_t, t) \|^2 \right] = \frac{1}{2} \mathbb{E}_{x_0, \epsilon \sim N(0, I)} \left[ \| \epsilon - \epsilon_\theta (\sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon, t) \|^2 \right]
$$

This is now our final training objective for diffusion models which is just a simple mean squared error loss between the true noise added at each time step and the predicted noise from the neural network model. We can then optimize this loss using stochastic gradient descent and backpropagation to learn the parameters $\theta$ of the denoising model.

### Training & Sampling

The above definitions and derivations can be summarized in the following pseudo-code for training and sampling from a DDPM. In this case we are using mini-batch training with a batch size of $B$. This means that we sample $B$ data points from the dataset at each training step and compute the loss over the entire batch before updating the model parameters, this is more efficient and leads to better convergence properties.

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

Importantly note that during training **we sample the time step $t$ uniformly at random from $\{1, 2, ..., T\}$ for each data point in the batch.** This ensures that the model learns to denoise from all time steps equally well. You might be wondering why we do not calculate the loss over all time steps for each data point instead of just one random time step. It is clear that calculating the loss over all time steps would be a lot more computationally expensive as we would need to perform T forward passes through the model for each data point in the batch instead of just one. But because we are using stochastic gradient descent anyway we can just sample one time step per data point and this gives us an unbiased estimate of the full loss over all time steps. This is a common technique in training deep generative models to reduce computational cost while still providing good convergence properties.

Another important detail is that during sampling we start from pure Gaussian noise $x_T \sim N(0, I)$ and then iteratively apply the denoising steps to obtain $x_{T-1}, x_{T-2}, ..., x_0$. The **special case is the final step** where we obtain the generated data sample $x_0$. In practice we might want to apply some post-processing to ensure that the generated sample is in the valid range for the data (e.g. clipping pixel values to [0, 1] for images), this is achieved here by "turning off" the noise addition in the final step by setting $z_1 = 0$.

## Noise-Conditional Score Networks

It turns out that diffusion models can also be linked to the field of physics and thermodynamics. This connection was first made explicit in the paper [Score-Based Generative Modeling through Stochastic Differential Equations](https://arxiv.org/abs/2011.13456) by Yang Song et al. In this work the authors show that diffusion models can be interpreted as learning the so called score function of the data distribution at different noise levels.

Please note that I am not a physicist nor a mathematician so my understanding of the following concepts is limited and I might get some details wrong. However, I will try my best to explain the main ideas behind this connection in an intuitive way. If you notice any mistakes or have improvement suggestions please let me know or submit a PR!

{{< figure 
    src="/images/ml/diffusionScoreMatching.png" 
>}}

{{< figure 
    src="/images/ml/diffusionScorePitfall.png"
>}}

{{< figure
    src="/images/ml/diffusionScorePertrubed.png"
>}}

{{< figure 
    src="/images/ml/diffusionScoreVarianceScales.gif"
>}}

{{< figure 
    src="/images/ml/diffusionSDEOverview.png"
>}}
{{< figure 
    src="/images/ml/diffusionReverseSDE.gif"
>}}
{{< figure 
    src="/images/ml/diffusionScoreForwardSDE.gif"
>}}

### Ito SDE and Brownian Motion

Let's first take a step back. Personally to understand this link to physics I found it helpful to think of the images in space as particles and then the diffusion process as a stochastic process that moves these particles around in space. First let's start with describing the movement of such a particle in space over time. This leads us to an **ordered differential equation (ODE)** where the change in position of the particle is determined by some $f(x, t)$ function called the **drift function** which takes in the current position of the particle x and the current time t and outputs the change in position of the particle at that time step:

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

If we now refine the parition more and more (smaller and smaller ($\Delta t_i$)), this random walk converges to a continuous-time stochastic process called **Brownian motion** or a **Wiener process**.

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
dX_t = f(X_t, t),dt + g(t)dW_t
$$

we need a way to **actually simulate** the stochastic process ${X_t}_{t \in [0,T]}$ on a computer. So for a given ODE or SDE and starting point $x_0$ at time $t=0$, we want to simulate the trajectory of the process over the time interval $[0,T]$, so how the particles position changes over time according to the dynamics defined by the SDE.

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
\Delta W_i := W_{t_i} - W_{t_{i-1}} \sim \mathcal{N}(0, \Delta t,I)
$$

Importantly any Gaussian with mean $0$ and covariance $\Delta t,I$ can also scaled down to a standard normal by dividing by its standard deviation in this case $\sqrt{\Delta t}$. So we can equivalently write the increment as

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

For our drift term we have $\frac{1}{2} \nabla_X \log P^*(X_t)$ which is half the gradient of the log density of the target distribution. Our diffusion term is just standard Brownian motion $dW_t$ which adds random Gaussian noise, preventing the particle from simply collapsing into the nearest local maximum (mode) and instead allowing it to explore the entire distribution. Remember that we can think of the SDE as defining a continous-time stochastic process where at each time step the position of the particle is updated by a deterministic drift part and a stochastic noise part and because it is brownian motion the noise part is just independent Gaussian noise added at each time step leading to us defining a markov chain when we discretize it using euler-maruyama.

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

The Langevin SDE can be derived using the **Fokker-Planck equation** which describes how the probability density function of a stochastic process evolves over time. Specifically if we have a probability denstity function $q(x, t)$ describing the likelihood of the random variable $X(t)$ taking on the value x at time t. The evolution of this density over time is governed by the Fokker-Planck equation:

$$
\frac{\partial q(X, t)}{\partial t} = - \nabla_X \cdot \left( f(X, t)q(X, t) - \frac{1}{2}g(t)^2 \nabla_X q(X, t) \right) = -\nabla_X \cdot J(X, t)
$$

where $J(x, t)$ is the so called **probability current or probability flux**. This is called current because it describes the flow of probability mass in the vector field in the space of $\mathbb{R}^d$ over time. Theh intuition behind this equation is that the change in probability density at a point X over time is determined by the net flow of probability mass into or out of that point due to both the drift term $f(X, t)$ and the diffusion term $g(t)$. 

An important note is that the operator $\nabla_X \cdot$ is the **divergence operator** which is different from the gradient operator $\nabla_X$ used before:
- $\nabla_x f(x)$ for a scalar $f: \mathbb{R}^d \to \mathbb{R}$ gives a vector pointing in the direction of steepest ascent.
- $\nabla_X \cdot v(X)$ for a vector field $v: \mathbb{R}^d \to \mathbb{R}^d$ gives a scalar quantifying the net outflow of the vector field from an infinitesimal neighborhood around X. So it measure the "inflow" and "outflow" of probability mass at a given point and can be interpreted as the local rate of change of probability mass. Similar to the mass conservation equation in fluid dynamics where water can not just disappear or appear out of nowhere or in the max-flow/min-cut theorem in graph theory where the total flow into a node must equal the total flow out of the node unless it is a source or sink.

So the divergence of the vector field $J$ which produces a scalar is defined as:

$$
\nabla_X \cdot J(X, t) = \sum_{i=1}^{d} \frac{\partial v_i(X)}{\partial x_i}
$$

In pyhsical terms it quantifies the outflow - inflow of probability mass at a given point X in R^d. local rate of change of probability mass. Similar to the mass conservation equation in fluid dynamics where water can not just disappear or appear out of nowhere.

We now want to pick the drift term $f(X, t)$ such that the stochastic process converges to a random variable with a our desired target distribution $P^*(X)$ at equilibrium so when time goes to infinity. In other words we want to apply the Fokker-Planck equation such that the distribtion reaches a steady state. This means that the time derivative of the density function becomes zero:

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

Remember that the brownian motion part just adds independent Gaussian noise at each time step. So we can interpret this update rule as a markov chain where at each time step we take the current position $X_i$ and move it in the direction of increasing probability density according to the score function $\nabla_X \log P^*(X_i)$ scaled by the step size $\Delta t/2$ and then add some Gaussian noise $\sqrt{\Delta t}\epsilon_i$ to allow exploration of the distribution:

$$
X_{i+1} \mid X_i \sim \mathcal{N}\Big(X_i + \frac{\Delta t}{2} \nabla_X \log P^*(X_i), \Delta t I\Big)
$$

It is common to reparameterize the step size as $\Delta t = 2\alpha$ for some $\alpha > 0$ which gives us the final ULA update rule:

$$
X_{i+1} = X_i + \alpha \nabla_X \log P^*(X_i) + \sqrt{2\alpha}\epsilon_i \qquad \epsilon_i \sim \mathcal{N}(0, I)
$$

### Score Matching

With this we could theoretically have a process that moves our particles towards the target distribution $P^*(X)$ by following the score function. However, there are two problems with this approach:

1. We don't know the target distribution $P^*(X)$ in practice as we only have access to samples from this distribution (the training data) but not the actual density function and therefore also not the score function $\nabla_X \log P^*(X)$.
2. Even if we had access to the target distribution, computing the score function $\nabla_X \log P^*(X)$ exactly is often intractable for high-dimensional data such as images.

So we need a way to estimate the score function from data samples drawn from the target distribution $P^*$. This is where **score matching** comes in. Instead of trying to model the entire density function $P^*(X)$ directly, score matching focuses on estimating the score function $\nabla_X \log P^*(X)$ directly from data samples. We do this by training a neural network to approximate the score function:

$$
s_\theta \approx \nabla_X \log P^*(X)
$$

To train this network we can use the **score matching objective** introduced by Hyvärinen (2005). The idea starts with using the mean squared error (MSE) loss between the true score function and the predicted score function:

$$
L(\theta) = \frac{1}{2} E_{X \sim P^*} [||s_\theta(X) - \nabla_X \log P^*(X)||^2]
$$

the problem here is that we still need the true score function $\nabla_X \log P^*(X)$ which we do not have access to. However, Hyvärinen showed that we can expand and simplify this loss function to get rid of the dependence on the true score function by expanding and using integration by parts. 

$$
\begin{align*}
L(\theta) &= \frac{1}{2} E_{X \sim P^*} [||s_\theta(X) - \nabla_X \log P^*(X)||^2] \\
&= \frac{1}{2} E_{X \sim P^*} [||s_\theta(X)||^2] - E_{X \sim P^*} [s_\theta(X) \cdot \nabla_X \log P^*(X)] + \frac{1}{2} E_{X \sim P^*} [||\nabla_X \log P^*(X)||^2] \\
\end{align*}
$$

We can ignore the last term as it does not depend on the parameters $\theta$ of our score model. Let's focus on the middle term:

$$
\begin{align*}
E_{X \sim P^*} [s_\theta(X) \cdot \nabla_X \log P^*(X)] = \int_{R^d} s_\theta(X) \cdot \nabla_X \log P^*(X) P^*(X) dX \text{Chain rule} \\
&= \int_{R^d} s_\theta(X) \cdot \nabla_X P^*(X) dX \\
&= -\int_{R^d} P^*(X) \nabla_X \cdot s_\theta(X) dX \text{Integration by parts} \\
&= -E_{X \sim P^*} [\nabla_X \cdot s_\theta(X)]
\end{align*}
$$

This results in the so called **score matching objective** or **Hyvärinen's objective**:

$$
L(\theta) = E_{X \sim P^*} \left[ \frac{1}{2} ||s_\theta(X)||^2 + \nabla_X \cdot s_\theta(X) \right]
$$

This objective can be interpreted as penalizing two complementary aspects:
- The first term $||s_\theta(X)||^2$ encourages the score model to predict a small vector near data points. Since data points are high likelihood points of the target distribution, the score should be small there.
- The second term $\nabla_X \cdot s_\theta(X)$ (divergence) ensures that the data points behave like local optima of the log density by penalizing divergence and encouraging the score model to point towards regions of higher probability density.

However, there are still two challanges with this approach:

1. Firstly the divergence term $\nabla_X \cdot s_\theta(X)= \text{Tr}(\frac{\partial s_\theta(X)}{\partial X})$ requires computing the trace of the Jacobian matrix of the score model which can be computationally infeasable for high dimensional data as a naive implementation would require one backpropagation per input dimension.

2. Secondly, the score is only informative near regions of high data density where we have training samples and generalizes poorly to low density regions far away from the data manifold. This is problematic for generative modeling where we need to be able to sample from the entire data distribution including low density regions. You can think of this as if the particle is very far away from the data manifold, the score function may not provide useful guidance on how to move towards higher density regions and instead just point in a generic direction.

To adress both of these issues, we use **Noise Conditional Score Matching (NCSM)**, where we add Gaussian noise to the data samples and train our score model to predict the score function of the perturbed data distribution:

$$
\tilde{X} = X + \epsilon \text{ where } \epsilon \sim N(0, \sigma^2 I)
$$

The added noise smooths out the data distribution, making the score function well-defined everywhere in $\mathbb{R}^d$ and improving generalization to low density regions. This results in the modified objective:

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
P_\sigma(\tilde{X}|X) = \frac{1}{(2\pi \sigma^2)^{d/2}} \exp \left( -\frac{||\tilde{X} - X||^2}{2\sigma^2} \right)
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


How do we find good sigma? borrow another idea from physics called annealing where we start with a large value of sigma and gradually decrease it over time during training. This allows the score model to first learn the coarse structure of the data distribution at high noise levels and then refine its estimates at lower noise levels to converge to the true score function of the original data distribution. This leads to annealed langevin sampling.

So we need to train with multiple noise levels so we will have a sequence of noise levels $\sigma_1 > \sigma_2 > ... > \sigma_K > 0$ each pair $(\tilde{X}, \sigma_k)$ represents a perturbed data point at noise level $\sigma_k$. The overall training objective becomes:

$$
L_{multi}(\theta) = \sum_{k=1}^K \lambda(\sigma_k) E_{X, \epsilon} \left[ ||s_\theta(X + \sigma_k \epsilon, \sigma_k) + \frac{1}{\sigma_k^2} \epsilon||^2 \right] = \sum_{k=1}^K \lambda(\sigma_k) L_{NCSM}(\theta; \sigma_k)
$$

the lambda function is a weighting function that can be used to balance the contributions from different noise levels during training. the idea is to ensure that losses different scales contribute equally to the overall objective and to control the relative importance of each noise level. A common choice is to set $\lambda(\sigma_k) = \sigma_k^2$ which gives more weight to higher noise levels where the score estimates are less accurate.

the step size $\alpha_k$ is usually set proportional to $\sigma_k^2$ which results in total in annealed langevin sampling.


### Anderson's Reversal

So we have now achieved the following:

- Langevin Dynamics proved that if you have a gradient (score) and noise, you can converge to a distribution. This proves Generative Modeling is possible via gradients.
- NCSM proved that you can learn those gradients from data without knowing the density, but only if you smooth the data with noise. This proves Training is possible.


Now comes the key idea for diffusion models. 

Reverse SDE unified these ideas into a continuous process, showing that "adding noise slowly" and "removing noise slowly" are mathematically symmetrical processes

What actually is the forward SDE now so that we then calcualte the reverse SDE? How does this link with langevin dynamics and score matching?

we don't just add noise but also scale down the features of the data which results in the forwards defussion process.

Brian Anderson 1982 showed that the reverse of a diffusion process is also a diffusion process with modified drift term. Then the time-reversed SDE is given by:

$$
\tilde{X}(t) = X(T - t)
$$

of the forward SDE:

$$
dX = [f(X, t) - g(t)^2 \nabla_X \log P_t(X)]dt + g(t)d\tilde{W}_t
$$

use X not X tilde despite being reverse. Drift term has an additional term involving the score function of the marginal distribution at time t. can be derived using the Fokker-Planck equation. Brownian motion is replaced by a new Brownian motion $\tilde{W}_t$ running backwards in time. the idea of the additional term is that it guides back the backward process back towards high probability regions of the data distribution at each time step.

If we can estimate the score function $\nabla_X \log P_t(X)$ at each time step t then we can use the reverse SDE to generate new samples from the data distribution by simulating the backward process starting from pure noise at time T and iteratively applying the reverse SDE until reaching time 0 resulting in a sample from the target distribution. The samples can be generated using the Euler-Maruyama method as before but now applied to the reverse SDE.

The challange is learning the score function at every time step t_1, t_2 etc. We can do this again with NCSM, noise conditional score matching. We train a time-conditional score model $s_\theta(X, t)$ to estimate the score function of the marginal distribution at each time step:

$$
s_\theta(X, t) \approx \nabla_X \log P_t(X(t) | X(0))
$$

here t plays the role of the noise level sigma in NCSM. So the conditional score will depend on how the forward process is defined. Putting this all together with the Euler-Maruyama method for simulating the reverse SDE gives us the following update step for generating samples:

$$
X(t_{i-1}) = X(t_i) + [f(X(t_i), t_i) - g(t_i)^2 s_\theta(X(t_i), t_i)]\Delta t + g(t_i)\sqrt{\Delta t}\epsilon_i
$$

where $\epsilon_i \sim N(0, I)$ are independent standard normal random variables. 

We can define the SDE corresponding to the DDPM forward process (also called the **VP-SDE**, VP stands for variance preserving) as:

Like in the video from x_i we discretize and use taylor expansion to get the differential form:

$$
dX = -\frac{1}{2} \beta(t) X dt + \sqrt{\beta(t)} \sqrt{dt} \epsilon_t \text{ where } \epsilon_t \sim N(0, I)
$$

Remember that $\sqrt{dt} \epsilon_t$ is equivalent to $dW_t$ where $W_t$ is Brownian motion. So we can rewrite this as:

$$
dX = -\frac{1}{2} \beta(t) X dt + \sqrt{\beta(t)} dW_t
$$

what is explanation? something with derivates maybe?

$$
\begin{align*}
L_{\text{DDPM}} &= \mathbb{E}_{x_0, \epsilon \sim N(0, I), t} \left[ \frac{1}{2} \| \epsilon - \epsilon_\theta (x_t, t) \|^2 \right] \\
L_{\text{NCSM}} &= \mathbb{E}_{x_0, \epsilon \sim N(0, I), t} \left[ \frac{1}{2} \| s_\theta (x_t, t) + \frac{\epsilon}{\beta_t} \|^2 \right]
\end{align*}
$$

## Denoising Diffusion Implicit Models

Determinisitc variant such as DDIM remove the stochastic term by setting $Z_i = 0$ in the update step. This results in a deterministic mapping from pure noise to data samples. This improves generation speed but with slightly reduced sample diversity? Intuitevly it converges faster to high likelihood regions but may miss some modes of the distribution due to lack of stochastic exploration. Wasn^t this shown to be bad by 3blue1brown?

Basically it can be shown that there is an ODE that matches the planck SDE and this ODE can be solved deterministically. So instead of simulating the SDE with noise we can solve the corresponding ODE without noise to get a deterministic mapping from noise to data samples. has smaller scaling of our step size do not go towards the mean?

## Diffusion Backbones

Notice that so far we have not specified the architecture of the model to be used in the reverse denoising process. Our only requirement is that the the dimensionality of the input and output match the data dimensionality d. So in the case of images we need a model that takes in an image and outputs an image of the same size. 

### U-Net

The original choice for the denoising model is the U-Net architecture which was first proposed in the context of biomedical image segmentation in [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597). The U-Net is a type of convolutional neural network (CNN) that has an encoder-decoder structure with skip connections between corresponding layers in the encoder and decoder paths. The encoder path consists of a series of convolutional and MaxPooling layers that progressively downsample the input image such that the spatial information is reduced while feature information is increased. The decoder path then consists of a series of upsampling and convolutional layers that progressively reconstruct the image back to its original size. Importantly just like in the ResNet architecture, skip connections are used to directly connect feature maps from the encoder to the decoder at corresponding spatial resolutions. This allows the decoder to leverage both high-level semantic information from the encoder as well as low-level spatial details from earlier layers, resulting in more accurate reconstructions and avoiding gradient vanishing issues.

{{< figure 
    src="/garden/ml/diffusion/mlUNet.png" 
    alt="U-Net Architecture used in the original Biomedical Image Segmentation paper."
    caption="U-Net Architecture used in the original Biomedical Image Segmentation paper."
>}}

Also has attention somewhore

and to know at which time it is uses sinusoidaL embeddings?

Second paper by openai "Diffusion Models Beat GANs on Image Synthesis" with some changes and also introduces classifier guidance.

### Diffusion Transformers

## Latent Diffusion Models

uses Vae to compress image into latent space and then runs diffusion in latent space. much faster and less memory. also allows higher res images.

## Conditional Diffusion

one using just embedding concatentation another using cross attention?

the embeddings can come from text encoders such as CLIP or from other modalities such as segmentation maps etc.

### Classifier Guided

### Classifier-Free Guidance

### Dall-E

### Imagen

### LoRa

### ControlNet

### DreamBooth


## Super Resolution Diffusion Models

