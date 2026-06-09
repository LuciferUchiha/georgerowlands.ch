---
title: Optimizers
type: docs
weight: 5
---

Plain [gradient descent](/garden/ml/gradientDescent) takes a fixed-size step against the gradient. The gradients themselves come from [backpropagation](/garden/ml/backprop), and the convergence analysis shows that the speed of plain gradient descent is governed by the conditioning of the loss, so it slows to a crawl, gets stuck, or even diverges in long narrow valleys, near saddle points, and on flat plateaus. **Optimizers** are the practical algorithms that work around these failure modes. They keep the basic idea of stepping against the gradient but reshape the update so that ill-conditioning and small gradients hurt less.

Two families of ideas do the reshaping, and the strongest modern optimizers combine them. **Momentum** methods give the iterate a velocity that accumulates past gradients. **Adaptive** methods give each parameter its own step size derived from its own gradient history. Adam combines both, and Muon goes further by preconditioning whole weight matrices.

## Why Plain Gradient Descent Struggles

It is worth being concrete about how plain gradient descent fails, because each optimizer is a direct response to one of these failures. There are three recurring problems, and they all trace back to the same two limitations, a single global step size $\eta$ and a reliance on only the current gradient.

**Ill-conditioning.** As the [gradient descent](/garden/ml/gradientDescent#step-size-and-conditioning) note shows, on a quadratic the update decouples along the eigenvectors of the Hessian, and each direction contracts by a factor $1 - \eta\lambda_i$ per step, where $\lambda_i$ is the curvature in that direction. One step size has to serve every direction at once. To stop the steepest direction from diverging we need $\eta < 2/\lambda_{\max}$, but then the flattest direction contracts by $1 - \eta\lambda_{\min} \approx 1$ and barely moves. Imagine a long narrow valley, steep across and almost flat along its floor. Gradient descent bounces back and forth across the steep walls while creeping slowly towards the minimum along the floor, and the larger the condition number $\kappa = \lambda_{\max}/\lambda_{\min}$ the worse the zigzag.

**Small gradients at plateaus and saddles.** Because the step is proportional to the gradient, gradient descent slows to a halt wherever the gradient is small, even when the loss is still high. At a **saddle point** the gradient vanishes although the point is not a minimum, the surface curving up in some directions and down in others. In the high-dimensional non-convex landscape of a deep network these vastly outnumber genuine local minima, so this is the usual way training stalls. Plain gradient descent keeps no memory that it was moving, so once the gradient shrinks it simply stops.

{{< figure
  src="/images/ml/optimizerSaddle.png"
  alt="Gradient descent stalling at a saddle point while noisy SGD escapes."
  caption="At a saddle the gradient along the escape direction is tiny, so plain gradient descent stalls. The noise in stochastic gradient descent knocks the iterate off the unstable direction and lets it escape."
>}}

**One step size for every parameter.** The same scalar $\eta$ multiplies the gradient of every parameter, yet gradient magnitudes differ enormously across parameters and layers. A learning rate small enough to stay stable for the largest-gradient parameter is far too small for the rest, so most parameters learn too slowly, while a rate large enough for the rest makes the loss oscillate or diverge along the steep ones.

{{< figure
  src="/images/ml/optimizerLearningRate.png"
  alt="Gradient descent on a parabola with a too-small, a good, and a too-large learning rate."
  caption="The learning rate has to be chosen with care. Too small crawls, too large oscillates and diverges, and the right value depends on the curvature, which differs across parameters."
>}}

## Momentum

### Heavy Ball

Momentum gives the iterate inertia, like a heavy ball rolling across the loss surface rather than a point that restarts from rest at every step. A rolling ball coasts through small bumps and flat stretches instead of stopping, and in a narrow valley its side-to-side motion cancels out over successive steps while the steady downhill component keeps building. So it accelerates the consistent directions and damps the oscillating ones. The **heavy ball method** (Polyak, 1964) adds a fraction of the previous step to the current one,

$$
\boldsymbol{\theta}_{k+1} = \boldsymbol{\theta}_k - \eta\,\nabla \mathcal{L}(\boldsymbol{\theta}_k) + \beta\,\underbrace{(\boldsymbol{\theta}_k - \boldsymbol{\theta}_{k-1})}_{\text{extrapolation}}, \qquad \beta \in (0, 1)
$$

The new step is the gradient step plus a $\beta$-scaled copy of the previous step. Adding those two vectors bends the update away from the pure gradient direction and towards the direction the iterate was already travelling.

{{< figure
  src="/images/ml/optimizerHeavyBall.png"
  alt="The heavy ball update as the sum of the gradient step and the momentum extrapolation."
  caption="The heavy-ball step is the sum of two vectors, the gradient step $-\\eta\\nabla\\mathcal{L}(\\theta_k)$ and the momentum term $\\beta(\\theta_k - \\theta_{k-1})$ that carries over the previous step."
>}}

To see the effect of the momentum term, suppose the gradient is roughly constant $\nabla \mathcal{L}$ over a stretch of steps. The increments then build up geometrically,

$$
\begin{align*}
\boldsymbol{\theta}_1 - \boldsymbol{\theta}_0 &= -\eta\,\nabla \mathcal{L} \\
\boldsymbol{\theta}_2 - \boldsymbol{\theta}_1 &= -\eta(1 + \beta)\,\nabla \mathcal{L} \\
\boldsymbol{\theta}_3 - \boldsymbol{\theta}_2 &= -\eta(1 + \beta + \beta^2)\,\nabla \mathcal{L} \\
&\quad\vdots \\
\lim_{k \to \infty} (\boldsymbol{\theta}_k - \boldsymbol{\theta}_{k-1}) &= -\eta \sum_{i=0}^\infty \beta^i\,\nabla \mathcal{L} = -\frac{\eta}{1 - \beta}\,\nabla \mathcal{L}
\end{align*}
$$

So along a consistent direction momentum boosts the effective step size by a factor $\frac{1}{1-\beta}$. With $\beta = 0.9$ that is a tenfold acceleration. The flip side is that too large a $\beta$ creates oscillations and instability, so in practice $\beta$ is tuned in the range $[0.9, 0.95]$.

### The Velocity Form

The momentum method (1986) writes the same idea using an explicit velocity vector $\mathbf{v}$,

$$
\begin{align*}
\mathbf{v}_{k+1} &= \beta\,\mathbf{v}_k - \eta\,\nabla \mathcal{L}(\boldsymbol{\theta}_k) \\
\boldsymbol{\theta}_{k+1} &= \boldsymbol{\theta}_k + \mathbf{v}_{k+1}
\end{align*}
$$

This is not a different algorithm, it is the heavy ball written differently. Since $\mathbf{v}_{k+1} = \boldsymbol{\theta}_{k+1} - \boldsymbol{\theta}_k$ is just the step taken at iteration $k$, substituting $\mathbf{v}_k = \boldsymbol{\theta}_k - \boldsymbol{\theta}_{k-1}$ into the first line gives

$$
\boldsymbol{\theta}_{k+1} - \boldsymbol{\theta}_k = \beta(\boldsymbol{\theta}_k - \boldsymbol{\theta}_{k-1}) - \eta\,\nabla \mathcal{L}(\boldsymbol{\theta}_k)
$$

which is exactly the heavy-ball update. For constant $\eta$ and $\beta$ the two are identical, the velocity form simply names the running step $\mathbf{v}$ as a state we carry forward. The reason it is the more common form is that it makes the key quantity explicit. Unrolling the recursion gives $\mathbf{v}_{k+1} = -\eta\sum_{j \le k}\beta^{k-j}\nabla \mathcal{L}(\boldsymbol{\theta}_j)$, so $\mathbf{v}$ is an exponentially weighted running average of the past gradients, a smoothed gradient. This smoothed gradient is exactly the first moment that Adam reuses later.

{{< figure
  src="/images/ml/optimizerMomentum.png"
  alt="Gradient descent zigzagging while momentum follows a smoother, faster path on an ill-conditioned bowl."
  caption="On an ill-conditioned bowl the oscillating part of the gradient cancels in the velocity average while the consistent downhill part adds up, so momentum follows a smoother and faster path than the zigzagging gradient descent."
>}}

### Nesterov Acceleration

**Nesterov acceleration** (1983) makes a small but important change to where the gradient is measured. Heavy ball computes the gradient at the current point and only then adds the momentum jump, committing to that jump blindly. Nesterov takes the momentum jump first, to a look-ahead point $\boldsymbol{\theta}'_{k+1}$, and evaluates the gradient there,

$$
\begin{align*}
\boldsymbol{\theta}'_{k+1} &= \boldsymbol{\theta}_k + \beta(\boldsymbol{\theta}_k - \boldsymbol{\theta}_{k-1}) \\
\boldsymbol{\theta}_{k+1} &= \boldsymbol{\theta}'_{k+1} - \eta\,\nabla \mathcal{L}(\boldsymbol{\theta}'_{k+1})
\end{align*}
$$

The difference is leap-then-look versus look-then-leap. Because the gradient is read at the point momentum is carrying the iterate towards, it already reflects what is about to happen. If momentum is about to overshoot the minimum, the gradient at the look-ahead point points back and acts as a brake one step earlier than heavy ball would feel it, so the iterate corrects before overshooting rather than after.

{{< figure
  src="/images/ml/optimizerNesterovDecomp.png"
  alt="The Nesterov update, with the gradient evaluated at the momentum look-ahead point."
  caption="Nesterov first follows the momentum jump to the look-ahead point $\\theta'$, then takes the gradient step there. Reading the gradient after the jump lets it brake an overshoot before it happens, where heavy ball would only feel it a step later."
>}}

This earlier correction lets Nesterov stay stable at higher momentum. Beyond the intuition it has real theoretical advantages, it is optimal in the convex case with respect to the lower bound on first-order methods, and it is accelerated in the strongly-convex case.

## Adaptive Methods

Momentum still uses one global step size. Adaptive methods instead give every parameter its own step size, scaled by the history of gradients that parameter has seen.

### AdaGrad

**AdaGrad** (2011) gives each coordinate a step size that shrinks in proportion to how much that coordinate has already been updated. It keeps a running sum $\gamma_i$ of the squared partial derivatives and divides the base step size by its square root,

$$
\begin{align*}
\gamma_i^k &= \gamma_i^{k-1} + \big[\partial_i \mathcal{L}(\boldsymbol{\theta}^k)\big]^2 \\
\eta_i^k &= \frac{\eta}{\sqrt{\gamma_i^k} + \delta} \\
\theta_i^{k+1} &= \theta_i^k - \eta_i^k\,\partial_i \mathcal{L}(\boldsymbol{\theta}^k)
\end{align*}
$$

where $\partial_i \mathcal{L} := \partial \mathcal{L} / \partial \theta_i$, the base learning rate is $\eta$, and $\delta$ is a small constant for numerical stability. The idea is **normalisation**. The accumulator $\gamma_i^k$ measures the total size of the gradients coordinate $i$ has seen, and dividing by $\sqrt{\gamma_i^k}$ cancels that scale, so every coordinate makes progress at a comparable rate whether its gradients are naturally large or small. This is the cure for the one-step-size-for-everything problem, and a useful side effect is that rare features, whose weights are updated only occasionally and so accumulate little, keep a large step and are not drowned out. The weakness is that $\gamma_i$ only ever grows, so the step sizes shrink monotonically towards zero and the optimization eventually stalls, because $\gamma_i$ keeps remembering gradients from the distant past that are no longer relevant.

### RMSProp

**RMSProp** (2012) fixes AdaGrad's vanishing step sizes with a single change, it replaces the running sum with an exponential moving average so that old gradients are gradually forgotten.

{{< callout type="info" title="Exponential moving average" >}}
An **exponential moving average** (EMA) of a sequence $x_1, x_2, \dots$ is the running quantity

$$
m_k = \rho\, m_{k-1} + (1 - \rho)\, x_k, \qquad \rho \in [0, 1)
$$

Unrolling it gives $m_k = (1-\rho)\sum_{j \le k} \rho^{k-j} x_j$, a weighted average of the past values whose weights decay geometrically into the past, so it tracks the recent average over a window of roughly the last $\frac{1}{1-\rho}$ terms. A plain running sum is the extreme case with no forgetting, every past term kept at full weight, which is why it grows without bound. The EMA instead stays a bounded estimate of the current typical value.
{{< /callout >}}

Applying the EMA to the squared gradients gives the RMSProp update,

$$
\begin{align*}
\gamma_i^k &= \rho\,\gamma_i^{k-1} + (1 - \rho)\big[\partial_i \mathcal{L}(\boldsymbol{\theta}^k)\big]^2 \\
\eta_i^k &= \frac{\eta}{\sqrt{\gamma_i^k} + \delta} \\
\theta_i^{k+1} &= \theta_i^k - \eta_i^k\,\partial_i \mathcal{L}(\boldsymbol{\theta}^k)
\end{align*}
$$

with a typical decay rate $\rho = 0.9$. The only change from AdaGrad is the $\rho$ in front of $\gamma_i^{k-1}$, which turns the unbounded running sum into a forgetting average. Now $\gamma_i^k$ estimates the current typical squared gradient of coordinate $i$ rather than the all-time total, so the per-parameter step size tracks the present scale of the gradient and stays responsive throughout training instead of decaying to zero just because many steps have passed. Note that this is not weight decay. Weight decay shrinks the parameters towards zero, whereas RMSProp only rescales the step using the magnitude of recent gradients. The two are unrelated, a point we return to with AdamW below.

### Adam

**Adam** (2014), short for Adaptive Moment Estimation, combines momentum and RMSProp. It keeps an exponential moving average of the gradient (the first moment, like momentum) and of the squared gradient (the second moment, like RMSProp), then steps in the first-moment direction with a per-coordinate size set by the second,

$$
\begin{align*}
g_i^k &= \beta\,g_i^{k-1} + (1 - \beta)\,\partial_i \mathcal{L}(\boldsymbol{\theta}^k) \\
h_i^k &= \alpha\,h_i^{k-1} + (1 - \alpha)\,\big[\partial_i \mathcal{L}(\boldsymbol{\theta}^k)\big]^2 \\
\theta_i^{k+1} &= \theta_i^k - \frac{\eta}{\sqrt{h_i^k} + \delta}\,g_i^k
\end{align*}
$$

The first moment $g_i^k$ is a smoothed gradient and decides which direction to move, exactly the momentum idea. The second moment $h_i^k$ estimates the recent magnitude of the gradient in each coordinate and decides how far to step there, exactly the RMSProp idea. Dividing one by the square root of the other makes the update roughly scale-free, the effective step in every coordinate is of order $\eta$ no matter how large or small that coordinate's gradients are. The ratio also behaves like a confidence measure, when a coordinate's gradients consistently point the same way $g_i^k$ stays large relative to $\sqrt{h_i^k}$ and Adam takes a near-full step, whereas when they oscillate or are mostly noise $g_i^k$ averages towards zero while $\sqrt{h_i^k}$ does not, so the step shrinks. Typical hyperparameters are $\beta = 0.9$, $\alpha = 0.999$, $\delta = 10^{-8}$ and base rate $\eta = 0.001$, and the gradient is usually stochastic, computed on a minibatch as in SGD.

{{< callout type="info" title="Bias correction" >}}
The moving averages are initialised at zero, so in the early steps they are biased towards zero, having not yet accumulated enough history. Adam corrects for this with

$$
\hat g_i^k = \frac{g_i^k}{1 - \beta^k}, \qquad \hat h_i^k = \frac{h_i^k}{1 - \alpha^k}
$$

and uses $\hat g$ and $\hat h$ in the update. Since $\beta^k$ and $\alpha^k$ are close to one early on, the correction divides by a small number, which inflates the first few steps to compensate for the zero start. As $k$ grows the correction factors tend to one and it fades away.
{{< /callout >}}

{{< figure
  src="/images/ml/optimizerAdaptive.png"
  alt="Gradient descent versus RMSProp and Adam on an ill-conditioned bowl."
  caption="By rescaling each coordinate by its own recent gradient magnitude, RMSProp and Adam take comparable steps in the steep and flat directions and head almost straight to the minimum, where plain gradient descent zigzags."
>}}

### AdamW

A common refinement is **AdamW**, which decouples weight decay from the gradient update. The two are easy to confuse, so it helps to write out what each one is.

- **$L_2$ regularisation** adds a penalty $\tfrac{\lambda}{2}\|\boldsymbol{\theta}\|^2$ to the loss. Its gradient is $\lambda\boldsymbol{\theta}$, so it enters the optimizer as an extra term on the gradient, $\partial_i \mathcal{L} + \lambda\theta_i$, which then flows through the moment estimates and the $1/\sqrt{h_i}$ scaling like any other gradient.
- **Weight decay** shrinks each weight directly, $\theta_i \leftarrow \theta_i - \eta\lambda\theta_i$, applied separately from the gradient step.

For plain SGD these two are equivalent, but for Adam they are not. Folding the penalty into the gradient means the decay term $\lambda\theta_i$ gets divided by $\sqrt{h_i}$ along with everything else, so a weight that has seen large gradients is decayed less and a quiet weight is decayed more. The amount of regularisation a weight receives ends up depending on its gradient history, which is not what we want from a penalty meant to pull all weights towards zero evenly. AdamW keeps the adaptive step on the plain gradient and applies the decay directly to the weights,

$$
\theta_i^{k+1} = \theta_i^k - \eta\,\frac{\hat g_i^k}{\sqrt{\hat h_i^k} + \delta} - \eta\lambda\,\theta_i^k
$$

so every weight is now shrunk by the same fraction $\eta\lambda$ each step regardless of its gradient scale. This decoupling is the difference between Adam and AdamW, and it tends to generalise better in practice.

### Muon

Momentum and Adam treat every weight as an independent scalar. They never use the fact that the weights of a layer form a matrix and that the gradient of a weight matrix has structure of its own. In practice that gradient, and the momentum built from it, is highly anisotropic. A few singular directions carry almost all of the magnitude while many others are tiny. This is ill-conditioning at the level of a single matrix, and an elementwise method cannot see it, let alone fix it.

**Muon**, short for MomentUm Orthogonalized by Newton-schulz, attacks this directly for two-dimensional parameters. It keeps an ordinary momentum buffer of the gradient matrix, but before stepping it orthogonalizes that buffer. Writing the reduced SVD of the momentum matrix as $\mathbf{M} = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^T$, the orthogonalized update is

$$
\mathbf{O} = \mathbf{U}\mathbf{V}^T
$$

which keeps the singular directions of $\mathbf{M}$ but resets every singular value to one. This is the closest semi-orthogonal matrix to $\mathbf{M}$, the polar factor. Geometrically it spreads the update energy equally across all directions instead of letting the few dominant singular directions take over, the matrix analogue of the per-coordinate rescaling that adaptive methods perform on vectors.

Computing an SVD every step would be far too expensive, so Muon approximates $\mathbf{U}\mathbf{V}^T$ with a fixed, small number of Newton-Schulz iterations. Starting from $\mathbf{X}_0 = \mathbf{M}/\|\mathbf{M}\|$ and iterating a quintic polynomial in $\mathbf{X}$,

$$
\mathbf{X}_{j+1} = a\,\mathbf{X}_j + b\,(\mathbf{X}_j\mathbf{X}_j^T)\mathbf{X}_j + c\,(\mathbf{X}_j\mathbf{X}_j^T)^2\mathbf{X}_j
$$

with tuned coefficients such as $(a, b, c) = (3.4445, -4.7750, 2.0315)$, drives all the singular values towards one in about five steps using only matrix multiplications, which are cheap and highly parallel on a GPU. Putting the pieces together, the Muon update is

$$
\begin{align*}
\mathbf{M}_k &= \beta\,\mathbf{M}_{k-1} + \mathbf{G}_k \\
\mathbf{O}_k &= \text{NewtonSchulz}(\mathbf{M}_k) \\
\boldsymbol{\theta}_k &= \boldsymbol{\theta}_{k-1} - \eta\,\mathbf{O}_k
\end{align*}
$$

where $\mathbf{G}_k$ is the gradient matrix. Muon is applied only to the two-dimensional weight matrices of the hidden layers. One-dimensional parameters such as biases and normalisation gains, together with the embedding and output layers, are still trained with AdamW, since the orthogonalization only makes sense for a matrix.

Muon was introduced by Keller Jordan and collaborators in 2024 and first drew attention by setting records in the nanoGPT training speedrun. The Kimi team at Moonshot AI then showed it scales to very large language models (Moonlight and Kimi K2), once a weight decay term is added and the update is rescaled so that its typical magnitude matches what AdamW would take. For transformer training it has been reported to converge faster and reach lower losses than AdamW at the same compute, which is why it has quickly become a serious alternative to the Adam family.

## Choosing an Optimizer

Putting the methods on the same ill-conditioned bowl shows what each one adds. Gradient descent zigzags, momentum and Nesterov smooth and accelerate the path, and the adaptive methods rescale per coordinate and head almost straight to the minimum.

{{< figure
  src="/images/ml/optimizerComparison.png"
  alt="Gradient descent, momentum, Nesterov and Adam trajectories on the same ill-conditioned bowl."
  caption="The same starting point under four optimizers. Each ingredient, momentum and then adaptivity, makes the path to the minimum more direct."
>}}

There is no single best optimizer, the right choice depends on the task, the model and the data. In practice **Adam** or **AdamW** is a strong and robust default for most deep learning, and **Muon** is an emerging alternative that is winning ground on large transformers.

## Further Resources

- [Optimization for Deep Learning (Momentum, RMSProp, AdaGrad, Adam)](https://www.youtube.com/watch?v=NE88eqLngkg)
- [Who's Adam and What's He Optimizing? A Deep Dive into Optimizers](https://www.youtube.com/watch?v=MD2fYip6QsQ)
- [The Algorithm that Helps Machines Learn (AdamW)](https://www.youtube.com/watch?v=1_nujVNUsto)
- [This Simple Optimizer Is Revolutionizing How We Train AI (Muon)](https://www.youtube.com/watch?v=bO5nvE289ec)
