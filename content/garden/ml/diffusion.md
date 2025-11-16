---
title: Diffusion Models
type: docs
weight: 16
---

A common goal in machine learning is to learn generative models that can produce new data samples that closely resemble a given dataset or in other words that can produce new and realistic samples matching the world we know. So given data samples ${x_1, x_2, ..., x_N}$ drawn from an unknown data distribution $q$ on $R^d$, the goal is to learn a model $p_\theta$ parametrized by $\theta$ that can generate new samples $\hat{x} \sim p_\theta$ such that the distribution of generated samples closely approximates the true data distribution, so $p_\theta \approx q$. 

For example in the case of images, our data samples could be all images of the internet and we want to learn a model that can generate new images that look like real images. So our underlying data distribution $q$ and the dimension $d$ is very high (e.g. if we consider $64 \times 64$ black and white images then $d = 64 \times 64 = 4,096$. Note that the pixel values are then usually scaled to then be in $[0,1]$ not $[0,255]$). This $d$ dimensional space is also called the pixel space where each dimension corresponds to the intensity value of a pixel in the image, but in general our data distribution could be anything such as videos such as in [Video Diffusion Models](https://arxiv.org/abs/2204.03458) or proteins such as in [RF Diffusion](https://www.nature.com/articles/s41586-023-06415-8).

Modeling such high dimensional distributions is very challenging due to the curse of dimensionality and the complex structure of real world data. The high dimensional space is also mostly empty with regards to our region of interest, also referred to as the data manifold (e.g. natural images) making it difficult to learn meaningful patterns. In the case of images, an easy way to think of this is that we are only interested in pictures of humans with say 2 eyes, 2 arms and 2 legs. But the pixel space also contains all sorts of other images that do not correspond to real humans such as images with 3 eyes or 5 arms etc. which do not exist in the real world. So the data distribution $q$ is concentrated on a very small area/manifold within the high dimensional pixel space.

Used Resources:
- https://yang-song.net/blog/2021/score/
- https://theaisummer.com/diffusion-models/
- https://ayandas.me/blogs/2021-12-04-diffusion-prob-models.html
- https://lilianweng.github.io/posts/2021-07-11-diffusion-models/
- https://goyalpramod.github.io/blogs/demysitifying_diffusion_models/
- https://huggingface.co/blog/annotated-diffusion
- https://github.com/diff-usion/Awesome-Diffusion-Models
- And all the corresponding papers linked throughout the text.

## Denoising Diffusion Probabilistic Models

As is common in computer science and machine learning, it is often easier to solve a complex problem by breaking it down into smaller subproblems. In the case of generative modeling, instead of learning to directly generate samples from the complex data distribution $q$, we can instead learn to gradually improve our samples in an iterative/recursive manner to reach the desired result. Think of it as painting a picture step by step, starting from a blank canvas and adding more and more details and layering colors of paint until the final masterpiece is complete, rather than trying to paint the entire picture in one go. 

The idea of diffusion models is to use a stochastic process to gradually transform simple known distributions such as Gaussian noise into our complex data distribution $q$. The main idea behind diffusion models is to define a forward diffusion process that gradually adds noise to the data samples until they become pure noise, and then learn a neural network to reverse this process and denoise the noisy samples back to the original data distribution. By iteratively applying this denoising process starting from a pure noise sample, we can generate new samples that closely resemble samples from the true data distribution.

### Forward Diffusion Process

First we define the forward diffusion process which gradually "perturbs/destroys" the data samples and moves it to our simple known distribution. In our case we will use a gaussian distribution as our simple known distribution. 

We then define the forwards diffusion process as a markov chain that adds small amounts of gaussian noise to the data samples over $T$ time steps. So if we let $x_0$ be a data sample drawn from our true data distribution $x_0 \sim q(x)$, then the forward diffusion process results in a sequence of noisy samples $x_1, x_2, ..., x_T$ where each $x_t$ is obtained by adding gaussian noise to the previous sample $x_{t-1}$. The amount of noise added at each time step is controlled by the so called **variance or noise schedule** $\beta_1, \beta_2, ..., \beta_T$ where each $\beta_t$ is a small positive value as variance must be positive. The choice of the noise schedule and this particular construction of the forward process is an algorithmic design choice that has been found to work well in practice for diffusion models. However, it can also be linked to physics and the concept of Langevin dynamics which we will discuss later. Usually the noise schedule is chosen to be some monotonic increasing function such that more noise is added at later time steps. A common choice is to use a linear schedule where $\beta_t$ increases linearly from a small value (e.g. 0.0001) to a larger value (e.g. 0.02) over $T$ time steps or a cosine schedule as proposed in [Improved Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2102.09672).

So we can define the forward diffusion process of gradually adding noise as:

$$
x_t = \sqrt{1 - \beta_t} x_{t-1} + \epsilon_t \text{ where } \epsilon_t \sim N(0, \beta_t I)
$$

Note that we scale down the previous sample $x_{t-1}$ by $\sqrt{1 - \beta_t}$ to ensure that the overall variance of $x_t$ does not explode but instead has the desired effect of gradually adding noise to the data sample and forcing the original signal to decay over time. Because the variables are independent they can be added and we can show this explosion of variance more formally by calculating the variance of $x_t$:

$$
\text{Var}(x_t) = \text{Var}(x_{t-1}) + \text{Var}(\epsilon_t) = \text{Var}(x_{t-1}) + \beta_t I
$$

Because variance is positive, over time it accumulates and can become very large. By scaling down $x_{t-1}$ we ensure that the variance remains stable, especially if $\text{Var}(x_{t-1}) = I$ then it is clear that the variance stays constant:

$$
\text{Var}(x_t) = (1 - \beta_t) \text{Var}(x_{t-1}) + \beta_t I 
= (1 - \beta_t) I + \beta_t I = I
$$

Importantly because this is a markov chain, each sample $x_t$ only depends on the previous sample $x_{t-1}$ and not on any earlier samples, so we have:

$$
q(x_t | x_{t_1}, x_{t-2}, ..., x_0) = q(x_t | x_{t-1})
$$

which also means we can write the joint distribution over the entire sequence as:

$$
q(x_{1:T} | x_0) = \prod_{t=1}^{T} q(x_t | x_{t-1})
$$

where $x_{1:T}$ denotes the sequence of noisy samples from time step 1 to $T$ that we obtain by applying the forward diffusion process to the original data sample $x_0$. Using the recurrence relation defined above for the forward diffusion process, we can also write the conditional distribution at each time step as a gaussian distribution:

$$
q(x_t | x_{t-1}) = N(x_t; \sqrt{1 - \beta_t} x_{t-1}, \beta_t I)
$$

where we evaluate the gaussian at $x_t$ with mean $\sqrt{1 - \beta_t} x_{t-1}$ and covariance $\beta_t I$. The mean represents the scaled down previous sample while the covariance represents the amount of noise added at this time step. 

Because the forward diffusion process is a linear gaussian markov process(a markov process composed of linear transformations and gaussian noise), we can also derive a closed form expression for the distribution of $x_t$ given the original data sample $x_0$ by unrolling the recurrence relation and using the reparameterization trick. Before unrolling the relation, we first define some notation to make the equations cleaner:

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

Because $\mathbb{E}(z_s) = 0$ for all s, we can compute the expectation of $x_t$ given $x_0$ as:

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

where we have reparameterized from the sum of gaussians to a single gaussian depending on $x_0$ and the parameter $\bar{\alpha}_t$. This means that picking a time step $t$ and sampling from $q(x_t | x_0)$ is equivalent to scaling down the original data sample $x_0$ by $\sqrt{\bar{\alpha}_t}$ and adding gaussian noise with variance $(1 - \bar{\alpha}_t) I$. This is useful as it allows us to directly sample noisy samples at any time step $t$ without having to iteratively apply the forward diffusion process from time step 1 to $t$ which can be computationally expensive for large $T$.

Under some mild assumptions on the noise schedule $\beta_t$ (e.g. $ 0< \beta_t < 1$), it can be shown that this defines as **ergodic** markov chain meaning that as $T$ approaches infinity, the distribution of $x_T$ becomes independent of the initial data sample $x_0$ and converges to a stationary distribution where in our case the stationary distribution is a standard normal distribution:

$$
\lim_{T \to \infty} q(x_T | x_0) = N(0, I)
$$

This means that after enough time steps of adding noise, the data samples become indistinguishable from pure Gaussian noise. Intuitively this makes sense as we want $\sqrt{\bar{\alpha}_T} \to 0$ and $(1 - \bar{\alpha}_T) \to 1$ as $T$ approaches infinity, so we pick a noise schedule $\beta_t$ that ensures this. In other words, we are scaling down the original signal to zero while continuously adding random noise, so eventually the original signal is completely lost and we are left with just noise. This is key for diffusion models as it allows us to start from pure Gaussian noise and then reverse the diffusion process to generate new data samples. 

### Reverse Denoising Process


we don't need to do all T steps per batch because it is an expectation and uniformly sample $t$ from 1 to T for each data point in the batch?

## Noise-Conditional Score Networks

We can also link the process of diffusion models to physics and the concept of Langevin dynamics.

inspired by physics, the dynamics of iterative transformatiosn such as ???? are described by ordinary differential equations (ODEs) where $f$ determines the change in the data point with respect to time:

$$
dx = f(x, t)dt \equiv \frac{dx}{dt} = f(x, t)
$$

However, this only describes deterministic transformations. To model stochastic processes we need to introduce some randomness into the dynamics via a stochastic differential equation (SDE):

$$
dX = f(X, t)dt + g(t)dW_t
$$

X here is a random variable representing the data point at time t,
where W_t is Brownian motion (Wiener process) introducing randomness into the dynamics, and g(t) is a function controlling the amount of noise added at each time step.

### Ito Calculus

### Langevin Dynamics

### Score Matching

### Euler-Maruyama Method

### Annealed Langevin Dynamics

### Anderson's Reversal

## Denoising Diffusion Implicit Models

Determinisitc variant such as DDIM remove the stochastic term by setting $Z_i = 0$ in the update step. This results in a deterministic mapping from pure noise to data samples. This improves generation speed but with slightly reduced sample diversity? Intuitevly it converges faster to high likelihood regions but may miss some modes of the distribution due to lack of stochastic exploration.

## Diffusion Backbones

Notice that so far we have not specified the architecture of the model to be used in the reverse denoising process. Our only requirement is that the the dimensionality of the input and output match the data dimensionality d. So in the case of images we need a model that takes in an image and outputs an image of the same size. 

### U-Net

The original choice for the denoising model is the U-Net architecture which was first proposed in the context of biomedical image segmentation in [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597). The U-Net is a type of convolutional neural network (CNN) that has an encoder-decoder structure with skip connections between corresponding layers in the encoder and decoder paths. The encoder path consists of a series of convolutional and MaxPooling layers that progressively downsample the input image such that the spatial information is reduced while feature information is increased. The decoder path then consists of a series of upsampling and convolutional layers that progressively reconstruct the image back to its original size. Importantly just like in the ResNet architecture, skip connections are used to directly connect feature maps from the encoder to the decoder at corresponding spatial resolutions. This allows the decoder to leverage both high-level semantic information from the encoder as well as low-level spatial details from earlier layers, resulting in more accurate reconstructions and avoiding gradient vanishing issues.

{{< figure 
    src="/garden/ml/diffusion/mlUNet.png" 
    alt="U-Net Architecture used in the original Biomedical Image Segmentation paper."
    caption="U-Net Architecture used in the original Biomedical Image Segmentation paper."
>}}

### Diffusion Transformers

## Latent Diffusion Models

## Super Resolution Diffusion Models

## Conditional Diffusion

### Classifier Guided

### Classifier-Free Guidance

### Dall-E

### Imagen

### LoRa

### ControlNet

### DreamBooth

true distribution is complex but we need some sort of stochastic process to model it. Idea is to learn a transformation from a simple known distribution (e.g. Gaussian noise) to a meaningful data point (e.g. an image). Noise to strucute -> essence of genrative modeling.

If not a stochastic process then would have a determinitic transfromation which suffer from mode collapse (GANs)? and produce only "average" samples?

## Ito Calculus

integrals to dynmaics. Not to be confused with graph random walks.
T-long deterministic walk, sequence of points in R^d, x(0), x(1), ..., x(T). which satisfies the following reccurence relation:

$$
x(t+1) = x(t) + f(x(t), t)
$$

where f(x(t), t) is the deterministic change in the data point at time step t. time dependent force field f. Force acting on that particle at position x(t) and time t.

more genreall to allow for variable step. So $t_0 < t_1 < ... < t_N = T$ be a partition of the interval [t_0, T] into n subintervals of variable length $\Delta t_i = t_{i+1} - t_i$. A t-long n-step walk is defined by the reccurence relation:

$$
x(t_{i+1}) = x(t_i) + f(x(t_i), t_i)(t_{i+1} - t_i) = x(t_i) + f(x(t_i), t_i)\Delta t_i
$$

Or equivalently after n steps:

$$
x(T) = x(t_0) + \sum_{i=0}^{n-1} f(x(t_i), t_i)\Delta t_i
$$

Notice this looks like a Riemann sum approximating the integral of f(x(t), t) from t_0 to T as n approaches infinity and the maximum step size approaches zero:

$$
x(T) = x(t_0) + \int_{t_0}^{T} f(x(t), t)dt
$$

or in differential form:

$$
dx = f(x, t)dt \equiv \frac{dx}{dt} = f(x, t)
$$

Now lets add some randomness. Instead of a deterministic walk, we want to define a t-long random walk $W(0), W(1), ..., W(T)$ where the increments are random variables drawn from a normal distribution:

$$
W(t) - W(t-1) \sim N(0, I)
$$

so each increment is an independent Gaussian noise vector with mean 0 and identity covariance matrix. Then for non equidistant we again define a partition $t_0 < t_1 < ... < t_N = T$ and the t-long n-step random walk is defined by the reccurence relation:

$$
W(t_i) - W(t_{i-1}) \sim N(0, (t_i - t_{i-1})I)
$$

the increments $W(t_i) - W(t_{i-1})$ are independent Gaussian noise vectors with mean 0 and covariance matrix proportional to the time step size.

### Riemann-Ito Sum

As seen before this looks like a Riemann sum but now with random increments. If we are given a weighting function g(t) controlling the amount of noise added at each time step, we can define the Riemann-Ito sum approximating the stochastic integral:

$$
\sum_{i=1}^{n} g(t_{i-1})(W(t_i) - W(t_{i-1}))
$$

as n approaches infinity this converges to a well -defined stochastic integral:

$$
\lim_{n \to \infty} \sum_{i=1}^{n} g(t_{i-1})(W(t_i) - W(t_{i-1})) = \int_{t_0}^{T} g(t)dW(t)
$$

also known as the Ito integral. because of the random variable $W$ the integral itself is a random variable. The random differential $dW(t)$ can be thought of as having a centered Gaussian distribution with differential variance $dW(t) \sim N(0, dtI)$. small kick in a random direction?

W(t) is Brownian motion (Wiener process) which is a continous time stochastic process $(W(t))_{t \in [t_0, T]}$ with the following properties:
- **Initial Condition**: $W(t_0) = 0$ almost surely, so with probability 1 the process starts at 0.
- **Independent Increments**: For every $t > t_0$, the future increment $W(t + \Delta t) - W(t)$ is independent of the past values $W(s)$ for all $s \leq t$.
- **Gaussian Increments**: The increments are normally distributed with mean 0 and variance proportional to the time step size: $W(t + \Delta t) - W(t) \sim N(0, \Delta tI)$.
- **Continuous Paths**: The function $t \mapsto W(t)$ is almost surely continuous, meaning there are no jumps or discontinuities in the process.

Can define a stochastic process(what is a stochastic process?):

$$
X(T) = X(0) + \int_{t_0}^{T} f(X(t), t)dt + \int_{t_0}^{T} g(t)dW(t)
$$

where the first term is the deterministic drift part (predictable smooth part of the motion) and the second term is the stochastic diffusion part (random wiggles). This generates a family of random variables $(X(t))_{t \in [t_0, T]}$ describing the evolution of the data point over time under the influence of both deterministic and stochastic forces in the same probability space. not just a single path. each sample is different. Can be written in differential form as:

$$
dX = f(X, t)dt + g(t)dW_t
$$

the so called stochastic differential equation (SDE). where $f(X, t)$ is called deterministic dynamics or drift coefficient and $g(t)$ is called stochastic dynamics or diffusion coefficient.

example with SDE visualization and interpreation:

$$
X(T) = \int_{0}^{T} 2 dt + \int_{0}^{T} t dW(t)
$$

## Langevin Dynamics

we can define a probability denstity function $q(x, t)$ describing the likelihood of the random variable $X(t)$ taking on the value x at time t. The evolution of this density over time is governed by the Fokker-Planck equation:

$$
\frac{\partial q(x, t)}{\partial t} = -\nabla_X \cdot J(X, t)
$$

where $J(x, t)$ is the probability current or probability flux defined as:

$$
J(X, t) = f(X, t)q(X, t) - \frac{1}{2}g(t)^2 \nabla_X q(X, t)
$$

this is called current because it describes the flow of probability mass in the space R^d over time? For any region in the space R^d, the integral of J over the boundary of that region gives the net rate at which probability mass is flowing in or out of that region???? Wtf meaning

The divergence of the vector field $J$ is defined as which produces a scalar not a vector like the gradient:

$$
\nabla_X \cdot J(X, t) = \sum_{i=1}^{d} \frac{\partial v_i(X)}{\partial x_i}
$$

In pyhsical terms it quantifies the outflow - inflow of probability mass at a given point X in R^d. local rate of change of probability mass. Similar to the mass conservation equation in fluid dynamics where water can not just disappear or appear out of nowhere.

Note that $\nabla_X f(X)$ and $\nabla_X \cdot v(X)$ are the same gradient operator but act on different functions. The former produces a vector pointed in the direction of steepest ascent of the scalar function $f: R^d \to R$ while the latter produces a scalar quantifying the net outflow of $v$ from an infinitesimal neighborhood around X and is equal to the trace of the Jacobian matrix of $v$.

How do you pick the drift term $f(X, t)$ so that the stochastic process goverend by the SDE 

$$
dX = f(X, t)dt + dW
$$

converges to a random variable with a desired target distribution $P^*$? For this we apply the Fokker-Planck equation such that the Distribtion of $X(t)$ reaches a steady state, so we want to stabilize the process over time. This means that the time derivative of the density function becomes zero:

$$
\frac{\partial q(X, t)}{\partial t} = 0 \text{ for all } X, t
$$

so probability mass is neither created nor destroyed at any point X in R^d over time. 

and has an equilibrium distribution equal to the target distribution $P^*$:

$$
q(X, t) = P^*(X)
$$

Plugging this condition into the Fokker-Planck equation gives:

$$
\nabla_X \cdot (f(X, t)P^*(X) - \frac{1}{2} \nabla_X P^*(X)) = 0
$$

In other words the following condition also suffices:

$$
f(X, t)P^*(X) - \frac{1}{2} \nabla_X P^*(X) = 0
$$

now we need to solve for the drift term $f(X, t)$ using the chain rule?:

$$
f(X, t) = \frac{1}{2} \frac{\nabla_X P^*(X)}{P^*(X)} = \frac{1}{2} \nabla_X \log P^*(X)
$$

Therefore the SDE governing the stochastic process that converges to the target distribution $P^*$ and is goverend by the langevin SDE und suitable regularity conditions???:

$$
dX = \frac{1}{2} \nabla_X \log P^*(X)dt + dW
$$

defines a contionous time markov process whose stationary distribution is exactly the target distribution $P^*$. The first term insures that the process is attracted towards regions of high probability under $P^*$ while the second term introduces random exploration via Brownian motion. 

We can also define the score function of the target distribution $P^*$ as the gradient of its log density:

$$
s^*(X) = \nabla_X \log P^*(X)
$$

Can we visualize the score function which points to the means for a Gaussian mixture model? Without brownian motion the dynamics would just follow the score function and converge to the nearest mode. But with brownian motion the dynamics can escape local modes and explore the entire distribution.

## Euler-Maruyama Method

From the langevin SDE we can simulate the stochastic process using the Euler-Maruyama method which is a numerical scheme for approximating solutions to SDEs. The challange with SDEs is that they can not be solved analytically in most cases due to the presence of the stochastic term and they contionously evolve over time. reference back to above reccurence relation for random walks? 
Euler method for discretitizing time in ODEs is by approximating the integral over a small time step $\Delta t$ using a Riemann sum, for a finitie $n$. As we increse $\Delta t$ the approximation becomes more accurate can show convergence to the true solution as $\Delta t$ approaches 0.

For SDEs we need to account for the stochastic term involving Brownian motion. For the first term we can use the euler method as before, but for the second term we need to approximate the stochastic integral using the properties of Brownian motion:

$$
\begin{align*}
X(T) &= X(0) + \int_{t_0}^{T} f(X(t), t)dt + \int_{t_0}^{T} g(t)dW(t) \\
&= X(0) + \lim_{n \to \infty} \sum_{i=1}^{n} f(X(t_i), t_i)(t_i - t_{i-1}) + \lim_{n \to \infty} \sum_{i=1}^{n} g(t_i)(W(t_i) - W(t_{i-1})) \\
&= X(0) + \lim_{n \to \infty} \sum_{i=1}^{n} f(X(t_i), t_i)\Delta t + \sum_{i=1}^{n} g(t_i)\Delta W
\end{align*}
$$

So for finite $n$ we can approximate the solution at by unrolling the following reccurence relation:

$$
X(t_{i+1}) = X(t_i) + f(X(t_i), t_i)\Delta t + g(t_i)\Delta W
$$

Importantly we recall two important properties of Brownian motion, i.e the independent gaussian increments:

$$
\Delta W \sim N(0, \Delta t I)
$$

So in other words $\Delta W = \sqrt{\Delta t}\epsilon_i$ where $\epsilon_i \sim N(0, I)$ is a standard normal random variable independent of the past. Where does this come from? standard gaussian scaled by its variance?

We can therefore define the Euler-Maruyama update step for simulating the SDE as:

$$
X(t_i) = X(t_{i-1}) + f(X(t_{i-1}), t_{i-1})\Delta t + g(t_{i-1})\sqrt{\Delta t}\epsilon_i
$$

where $\epsilon_i \sim N(0, I)$ are independent standard normal random variables. By iteratively applying this update step from the initial condition $X(0)$ we can simulate the stochastic process governed by the SDE over the desired time interval.

Can plot the the different sample paths generated by the Euler-Maruyama method over T, i.e the empirical distribution of X(T) after many simulations. and also the distribution at the final Term T using a histogram and compare it to the target distribution P*.

However, the global convergence is half order? meaning the local error per step is at least $O(\sqrt{\Delta t})$. Smaller step sizes are needed for higher accuracy but at the cost of increased computational time just like euler method for ODEs.

Applying Euler-Maruyama to the Langevin SDE gives the following update step:

$$
X(t_i) = X(t_{i-1}) + \frac{\Delta t}{2} \nabla_X \log P^*(X(t_{i-1})) + \sqrt{\Delta t}\epsilon_i
$$

So we get a markov chain where each step consists of a deterministic move in the direction of the score function plus a random gaussian perturbation. By iterating this update step many times starting from an initial point $X(0)$ we can simulate the langevin dynamics and generate samples from the target distribution $P^*$.

$$
\P(X(t_i) | X(t_0), ..., X(t_{i-1})) = \P(X(t_i) | X(t_{i-1}))
$$

Notebook example of simulating langevin dynamics using euler-maruyama method and visualizing the results. 

Unadjusted Langevin Algorithm (ULA) which is just the euler-maruyama discretization of the langevin SDE and the step size is defined as $\alpha$:

$$
X(i) = X(i-1) + \alpha \nabla_X \log P^*(X) + \sqrt{2\alpha}\epsilon_i
$$

where $X(0), \epsilon_i \sim N(0, I)$.

The biggest problem is taht we need to know the score function $\nabla_X \log P^*(X)$ of the target distribution $P^*$ which is usually unknown or computationally infeasable.

So instead we want to estimate the score function from data samples drawn from the target distribution $P^*$. This is where score matching comes in. So we use a parametric family of functions, i.e. model to approximate the score function parametrized by $\theta$:

$$
s_\theta: R^d \to R^d \text{ such that } s_\theta(X) \approx \nabla_X \log P^*(X)
$$

A simple loss function to train this model could be the mean squared error between the estimated score and the true score:

$$
L(\theta) = \frac{1}{2} E_{X \sim P^*} [||s_\theta(X) - \nabla_X \log P^*(X)||^2]
$$

why the 1/2 factor? just for convenience when taking derivatives. But we still have the problem that we do not know the true score function $\nabla_X \log P^*(X)$. For this we can use Hyvarinen's score matching technique which allows us to train the score model without knowing the true score function. If we expand the loss function we get

$$
L(\theta) = \frac{1}{2} E_P(||s_\theta(X)||^2) - E_P[s_\theta(X) \cdot \nabla_X \log P(X)] + \frac{1}{2} E_P(||\nabla_X \log P(X)||^2)
$$

why now just P? is this just the general form? To remove the dependence on the true score function we can apply integration by parts formula in multiple dimensions. So let $v: R^d \to R^d$ be a vector field and $p: R^d \to R$ be a scalar field. Then the integration by parts formula states that:

$$
\int_{R^d} v(X) \cdot \nabla_X p(X) dX = -\int_{R^d} p(X) \nabla_X \cdot v(X) dX
$$

by the chain rule we also have:

$$
\nabla_X \log P(X) \cdot P(X) = \nabla_X P(X)
$$

Putting these together we get:

$$
\begin{align*}
E_P[s_\theta(X) \cdot \nabla_X \log P(X)] &= \int_{R^d} s_\theta(X) \cdot \nabla_X \log P(X) P(X) dX \\
&= \int_{R^d} s_\theta(X) \cdot \nabla_X P(X) dX \\
&= -\int_{R^d} P(X) \nabla_X \cdot s_\theta(X) dX \\
&= -E_P[\nabla_X \cdot s_\theta(X)]
\end{align*}
$$

Putting this back into the loss function gives us the so called **score matching objective** or **Hyvarinen's objective**:

$$
L(\theta) = E_{X \sim P} \left[ \frac{1}{2} ||s_\theta(X)||^2 + \nabla_X \cdot s_\theta(X) \right]
$$

the objective can be interpreted as penalizing two complimentary aspects? Doesn't make sense to me:
- The first term $||s_\theta(X)||^2$ encourages the score model to predict a small vector near data points. Have to assume the data points are high likelihood data points of the target distribution. So once it is in the high likelihood region the score should be small.
- The second term $\nabla_X \cdot s_\theta(X)$ (divergence) ensures that the data points behave like a local optimum of the log density by penalizing divergence.

Hwoever, there are still two challanges with this approach. Firstly the divergence term $\nabla_X \cdot s_\theta(X)= \text{Tr}(\frac{\partial s_\theta(X)}{\partial X})$ requires computing the trace of the Jacobian matrix of the score model which can be computationally infeasable for high dimensional data as a naive implementation would require one backpropagation per input dimension? For example videos with millions of dimensions. 

Secondly, the score is only informative near regions of high data density where we have training samples and generalizes poorly to low density regions far away from the data manifold. This is problematic for generative modeling where we need to be able to sample from the entire data distribution including low density regions. Can be visaulized.

This results in Noise Conditional score matching (NCSM) where we add Gaussian noise to the data samples at different scales and train a score model to estimate the score function of the perturbed data distribution. By adding noise we smooth out the data distribution and make the score function well-defined everywhere in R^d.

So let $X \sim P(X)$ be a random variable followign the target data distribution. We define a perturbed version of X by adding Gaussian noise with standard deviation $\sigma$:

$$
\tilde{X} = X + \epsilon \text{ where } \epsilon \sim N(0, \sigma^2 I)
$$

Can visuallized the petrubed data distribution and the true and noisy score functions. The variance $\sigma^2$ controls the amount of smoothing applied to the data distribution. Larger values of $\sigma$ result in a smoother distribution with a well-defined score function everywhere. But destroys the fine strucute of the data. Smaller values of $\sigma$ preserve more details but the score function fails to cover low density regions. This results in the following objective:

$$
\min_\theta \frac{1}{2} E_{\tilde{X} \sim P_\sigma} [||s_\theta(\tilde{X}, \sigma) - \nabla_{\tilde{X}} \log P_\sigma(\tilde{X})||^2]
$$

where $P_\sigma$ is the perturbed data distribution obtained by convolving the original data distribution P with a Gaussian kernel of variance $\sigma^2$. Agai we can expand and rewrite the objective. The expectation of the cross term can be rewritten as the conditional distribution $P(\tilde{X}|X)$:

$$
E_{\tilde{X}}[s_\theta(\tilde{X}, \sigma) \cdot \nabla_{\tilde{X}} \log P_\sigma(\tilde{X})] = E_{X, \tilde{X}|X}[s_\theta(\tilde{X}, \sigma) \cdot \nabla_{\tilde{X}} \log P(\tilde{X}|X)]
$$

The above needs to be proven as all these calculations are very confussing. Uses the leibniz rule to interchange differentiation and integration.

This results in:

$$
\min_\theta \frac{1}{2} E_{\tilde{X}}[||s_\theta(\tilde{X}, \sigma)||^2] + E_{X, \tilde{X}|X}[s_\theta(\tilde{X}, \sigma) \cdot \nabla_{\tilde{X}} \log P(\tilde{X}|X)]
$$

By using marginliazion we can also get:

$$
E_{\tilde{X}}[||s_\theta(\tilde{X}, \sigma)||^2] = \int s_\theta(\tilde{X}, \sigma)^2 P_\sigma(\tilde{X}) d\tilde{X} = \int \int s_\theta(\tilde{X}, \sigma) P_\sigma(\tilde{X}|X) P(X) dX d\tilde{X} = E_{X, \tilde{X}|X}[||s_\theta(\tilde{X}, \sigma)||^2]
$$

Giving us

$$
E_{X, \tilde{X}|X} \left[ ||s_\theta(\tilde{X}, \sigma)||^2 - 2 E_{X, \tilde{X}|X}[||s_\theta(\tilde{X}, \sigma) \cdot \nabla_{\tilde{X}} \log P(\tilde{X}|X)||^2 \right] + E_{X, \tilde{X}|X} \left[ ||\nabla_{\tilde{X}} \log P(\tilde{X}|X)||^2 \right] - E_{X, \tilde{X}|X} \left[ ||\nabla_{\tilde{X}} \log P(\tilde{X}|X)||^2 \right]
$$

So we get:

$$
\min_\theta \frac{1}{2} E_{\tilde{X}}[||s_\theta(\tilde{X}, \sigma) - \nabla_{\tilde{X}} \log P(\tilde{X})||^2] = \min_\theta \frac{1}{2} E_{X, \tilde{X}|X} [||s_\theta(\tilde{X}, \sigma) - \nabla_{\tilde{X}} \log P(\tilde{X}|X)||^2]
$$

key here is the conditional distribution $P(\tilde{X}|X)$ is easier to work with since it is just a Gaussian distribution with known mean and variance:

$$
P(\tilde{X}|X) = \frac{1}{(2\pi \sigma^2)^{d/2}} \exp \left( -\frac{||\tilde{X} - X||^2}{2\sigma^2} \right)
$$

takign the log and the gradient we get:

$$
\nabla_{\tilde{X}} \log P(\tilde{X}|X) = -\frac{1}{\sigma^2} (\tilde{X} - X) = -\frac{1}{\sigma^2} \epsilon
$$

Therefore the noise conditional score matching objective simplifies to:

$$
L_{NCSM}(\theta) = \frac{1}{2} E_{X, \epsilon} \left[ ||s_\theta(X + \sigma \epsilon, \sigma) + \frac{1}{\sigma^2} \epsilon||^2 \right] \text{ where } \epsilon \sim N(0, I)
$$

now we can efficiently train the score model using samples from the data distribution and Gaussian noise. the model learns to undo the effect of the added noise and estimate the score function of the perturbed data distribution. using SGD of mini batches.

How do we find good sigma? borrow another idea from physics called annealing where we start with a large value of sigma and gradually decrease it over time during training. This allows the score model to first learn the coarse structure of the data distribution at high noise levels and then refine its estimates at lower noise levels to converge to the true score function of the original data distribution. This leads to annealed langevin sampling.

So we need to train with multiple noise levels so we will have a sequence of noise levels $\sigma_1 > \sigma_2 > ... > \sigma_K > 0$ each pair $(\tilde{X}, \sigma_k)$ represents a perturbed data point at noise level $\sigma_k$. The overall training objective becomes:

$$
L_{multi}(\theta) = \sum_{k=1}^K \lambda(\sigma_k) E_{X, \epsilon} \left[ ||s_\theta(X + \sigma_k \epsilon, \sigma_k) + \frac{1}{\sigma_k^2} \epsilon||^2 \right] = \sum_{k=1}^K \lambda(\sigma_k) L_{NCSM}(\theta; \sigma_k)
$$

the lambda function is a weighting function that can be used to balance the contributions from different noise levels during training. the idea is to ensure that losses different scales contribute equally to the overall objective and to control the relative importance of each noise level. A common choice is to set $\lambda(\sigma_k) = \sigma_k^2$ which gives more weight to higher noise levels where the score estimates are less accurate.

the step size $\alpha_k$ is usually set proportional to $\sigma_k^2$ which results in total in annealed langevin sampling.

Code of this process. the target distribution is just a mixture of gaussians.

### Forward SDE and Anderson's Reversal

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

## DDPM

recall that the general form the forward SDE is given by:

$$
dX = f(X, t)dt + g(t)dW_t
$$

where f(X, t) is the deterministic drift term and g(t) is the stochastic injection of gaussian noise. as t increases the distribution of X(t) becomes smoother and converges to a simple known distribution such as a standard Gaussian.

this pulls samples towards the origin as time increases representing gaussian noise. The process addes isotropic gaussian noise at each time step with variance controlled by g(t) whilst preserving the overall strucutres and global variance of the data distribution ensuring numerical stability and interpretability.

For the forward diffusion process in DDPMs we choose the drift and diffusion terms as:

$$
f(X, t) = -\frac{1}{2} \beta(t) X \text{ and } g(t) = \sqrt{\beta(t)}
$$

where $\beta(t)$ is a non-negative, monotonically increasing function called the noise schedule. This choice results in a forward SDE that gradually adds gaussian noise to the data points while scaling them down over time. This results in the variance-preserving(VP) SDE:

$$
dX = -\frac{1}{2} \beta(t) X dt + \sqrt{\beta(t)} dW_t
$$

we call it variance-preserving because if $X(0)$ has unit variance, then $Var(X(t)) \approx 1$ for all t. So doesn't explode or vanish over time. In practice we garantee this by normalizing the data in the training dataset to have zero mean and unit variance before training the score model.

To discretize the VP SDE using the Euler-Maruyama method we define a partition of the time interval [0, T] into N equidistant steps with step size $\Delta t = T/N = t_n - t_{n-1}$. So $0 = t_0 < t_1 < ... < t_N = 1$. So is T = 1? This gives us:

$$
X(t_i) = X(t_{i-1}) - \frac{1}{2} \beta(t_{i-1}) X(t_{i-1}) \Delta t + \sqrt{\beta(t_{i-1})} \sqrt{\Delta t} \epsilon_i
$$

and we define the discrete noise schedule as $\beta_i = \beta(t_{i-1}) \Delta t$ resulting in the reccurence relation:

$$
X_i = (1-\frac{1}{2} \beta_i) X_{i-1} + \sqrt{\beta_i} \epsilon_i
$$

we need to use a first order taylor approximation of beta around 0? Why? as long as beta is small enough this is a good approximation. So $1-\frac{1}{2} \beta_i \approx \sqrt{1 - \beta_i}$ giving us the final reccurence relation:

$$
X_i = \sqrt{1 - \beta_i} X_{i-1} + \sqrt{\beta_i} \epsilon_i
$$

This allows us to arbitraly go from any step i to any step j > i directly without simulating all the intermediate steps. $\alpha_i = 1 - \beta_i$ and $\bar{\alpha}_i = \prod_{k=1}^{i} \alpha_k$. Unrolling the reccurence relation gives us:

$$
X_i = \sqrt{\bar{\alpha}_i} X_0 + \sqrt{1 - \bar{\alpha}_i} \epsilon
$$

where $\epsilon \sim N(0, I)$. If we condition on $X_0$ we get the following closed form distribution:

$$
X_i | X_0 \sim N(\sqrt{\bar{\alpha}_i} X_0, (1 - \bar{\alpha}_i) I)
$$

this is essential for efficient training and for writing the losses as we will see later. If we then assume that $\alpha_i \in (0, 1)$ then gives us that as i goes to infinity $\\bar{\alpha}_i$ goes to 0 and $X_i$ converges to pure Gaussian noise:

$$
X_\infty | X_0 \sim N(0, I)
$$

show notebook that implements DDPM using MNIST dataset for forward process.

The reversal process is given by $P(X_{i-1} | X_i)$ by bayes rule and the markov property. We know that the forward process is:

$$
X_i | X_{i-1} \sim N(\sqrt{\alpha_i} X_{i-1}, (1 - \alpha_i) I) \text{ and } X_{i-1} | X_0 \sim N(\sqrt{\bar{\alpha}_{i-1}} X_0, (1 - \bar{\alpha}_{i-1}) I)
$$

so using bayes rule for gaussians we get:

$$
P(X_{i-1} | X_i, X_0) \propto P(X_i | X_{i-1}) P(X_{i-1} | X_0)
$$

Therefore the product is also a gaussian with mean and covariance given by:

$$
P(X_{i-1} | X_i, X_0) = N(\tilde{\mu_i}(X_i, X_0), \tilde{\beta}_i I)
$$

using the standard formula we can compute the mean and covariance:

$$
\tilde{\mu_i}(X_i, X_0) = \frac{\sqrt{\bar{\alpha_i}}}{1 - \bar{\alpha}_{i-1}} X_i + \frac{\sqrt{\alpha_{i-1}}\beta_i}{1- \bar{\alpha}_i}X_0
$$

and the covariance:

$$
\tilde{\beta}_i = \frac{(1 - \bar{\alpha}_{i-1})}{1 - \bar{\alpha}_i}\beta_i
$$

to elimante the dependence on $X_0$ we can use the reccurence relation to express $X_0$ in terms of $X_i$ and the noise $\epsilon$:

$$
X_i = \sqrt{\bar{\alpha}_i} X_0 + \sqrt{1 - \bar{\alpha}_i} \bar{\epsilon}_i \implies X_0 = \frac{1}{\sqrt{\bar{\alpha}_i}} (X_i - \sqrt{1 - \bar{\alpha}_i} \bar{\epsilon}_i)
$$

where $\bar{\epsilon}_i \sim N(0, I)$ is the known noise used to corrupt the data point $X_0$ to get $X_i$. We can then replace X_0 in these expressions which results in the conditional probabiity of the reverse diffusion process:

$$
X_{i-1} | X_i, X_0 \sim N(\bar{\mu}_i(X_i), \tilde{\beta}_i I)
$$

To be able to sample from the reverse process we need to use the reparameterization trick to express the sampling step in terms of standard gaussian noise???

$$
X_{i-1} = \frac{1}{\sqrt{\alpha_i}} \left( X_i - \frac{1-\alpha_i}{\sqrt{1 - \bar{\alpha}_i}} \bar{\epsilon}_i \right) + \sqrt{\tilde{\beta}_i} Z
$$

where $Z \sim N(0, I)$ is standard gaussian noise independent of everything else. This allows us to sample from the reverse process by first sampling $Z$ and then applying the above transformation. Each update is then adding gaussian noise with mean shifted in the direction of denoising, opposite of the forward noise process $\bar{\epsilon}_i$. We use a neural network to predict the noise $\bar{\epsilon}_i$ given the current noisy sample $X_i$ and the time step i. This results in the following update step for sampling from the reverse diffusion process:

$$
X_{i-1} = \frac{1}{\sqrt{\alpha_i}} \left( X_i - \frac{1-\alpha_i}{\sqrt{1 - \bar{\alpha}_i}} \epsilon_\theta(X_i, i) \right) + \sqrt{\tilde{\beta}_i} Z
$$

where $\epsilon_\theta(X_i, i)$ is the neural network's prediction of the noise at time step i given the noisy input $X_i$. The training objective minimizes the expected mean squared error between the true noise $\bar{\epsilon}_i$ and the predicted noise $\epsilon_\theta(X_i, i)$ over all time steps:

$$
L(\theta) = E_{i, X_0, \bar{\epsilon}_i} [\frac{1}{2} ||\bar{\epsilon}_i - \epsilon_\theta(X_i, i)||^2]
$$

by learning this noise the network implicitly learns the strucutre of the data distribution at different noise levels allowing it to effectively denoise samples during the reverse diffusion process and generate new data points from the target distribution. 

In each iteration the reverse process 3 key operations are performed: 

1. **Denoising Step**: The term $\frac{1-\alpha_i}{\sqrt{1 - \bar{\alpha}_i}} \epsilon_\theta(X_i, i)$ removes the estimated noise from the current sample $X_i$ based on the neural network's prediction. This shifts the sample closer to the data manifold, i.e towards higher data likelihood regions under the target distribution $P^*$.
2. **Rescaling Step**: Multiplying by $\frac{1}{\sqrt{\alpha_i}}$ inverts the signal contraction applied during the forward diffusion process. This ensures that the overall variance of the samples remains stable throughout the reverse process.
3. **Random Perturbation Step**: The term $\sqrt{\tilde{\beta}_i} Z$ adds a small amount of gaussian noise to the sample. This stochasticity helps the reverse process explore the data distribution and prevents it from getting stuck in local modes or collapsing to a single deterministic point.

T determines the trade-off between sample quality and computational cost. Larger T allows for more gradual and smoother denoising but requires more computation. 

the schedule $\beta_i$ controls the amount of noise added at each step in the forward process and consequently the amount of denoising required in the reverse process. It plays a key role in sample diversity and quality/sharpness. Often is chosen to be a linear or cosine schedule increasing from a small value to a larger value over the T steps. 
