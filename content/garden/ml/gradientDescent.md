---
title: Gradient Descent
type: docs
weight: 3
---

Training a model means choosing the parameters that minimise a loss function. For all but the simplest models there is no analytical closed-form solution, so instead we use iterative optimisation methods that take a small step at each iteration to lower the loss. **Gradient descent** is the workhorse for this. At each step it moves in the direction that decreases the loss fastest, the negative gradient.

The **gradient** $\nabla \mathcal{L}(\boldsymbol{\theta})$ is the vector of partial derivatives of the loss with respect to each parameter, $\nabla \mathcal{L}(\boldsymbol{\theta}) = \big(\tfrac{\partial \mathcal{L}}{\partial \theta_1}, \dots, \tfrac{\partial \mathcal{L}}{\partial \theta_d}\big)$. It points in the direction in which the loss increases fastest, so its negative points in the direction of steepest decrease (see [differentiable functions](/garden/maths/calculus/differentiableFunctions) for the underlying derivative). Following the negative gradient is just walking downhill on the loss surface.

{{< figure
  src="/images/ml/gradientDescent.png"
  alt="A path descending a non-convex loss surface towards a minimum."
  caption="Gradient descent walks downhill on the loss surface. On a non-convex surface like this one, where it ends up depends on where it starts."
>}}

If the loss is **convex**, gradient descent is guaranteed to reach the global minimum, because a convex function has no suboptimal local minima to get trapped in. The difficulty is that the losses we care about for deep models are usually non-convex, a high-dimensional surface dotted with local minima, flat plateaus, and saddle points, and gradient descent can slow down or stall on any of them. Much of this note is about the conditions under which we can still say where gradient descent ends up and how fast it gets there. [Backpropagation](/garden/ml/backprop) is what computes the gradient cheaply, and the [optimizers](/garden/ml/optimizers) note builds the practical variants on top of the theory developed here.

## Gradient Descent

Let $\mathcal{L} : \mathbb{R}^d \to \mathbb{R}$ be the loss as a function of the parameters $\boldsymbol{\theta} \in \mathbb{R}^d$. Gradient descent starts from some $\boldsymbol{\theta}_0$ and repeatedly steps against the gradient with a **step size** (or **learning rate**) $\eta > 0$,

$$
\boldsymbol{\theta}_{k+1} = \boldsymbol{\theta}_k - \eta\,\nabla \mathcal{L}(\boldsymbol{\theta}_k)
$$

Why is the negative gradient the right direction? Suppose we are at $\boldsymbol{\theta}$ and consider stepping by a small vector $\mathbf{d}$ to $\boldsymbol{\theta} + \mathbf{d}$. A first-order [Taylor expansion](/garden/maths/calculus/taylor) approximates the loss near $\boldsymbol{\theta}$ by a flat tilted plane,

$$
\mathcal{L}(\boldsymbol{\theta} + \mathbf{d}) \approx \mathcal{L}(\boldsymbol{\theta}) + \nabla \mathcal{L}(\boldsymbol{\theta})^T \mathbf{d}
$$

To make the loss drop as much as possible we want to choose, among all directions $\mathbf{d}$ of some fixed small length, the one that makes the inner product $\nabla \mathcal{L}(\boldsymbol{\theta})^T \mathbf{d}$ as negative as possible. The [Cauchy-Schwarz inequality](/garden/maths/linearAlgebra/vectors) says $|\nabla \mathcal{L}(\boldsymbol{\theta})^T \mathbf{d}| \le \|\nabla \mathcal{L}(\boldsymbol{\theta})\|\,\|\mathbf{d}\|$, with equality only when $\mathbf{d}$ is parallel to the gradient. The inner product is therefore most negative when $\mathbf{d}$ points exactly opposite to the gradient, so $-\nabla \mathcal{L}(\boldsymbol{\theta})$ is the locally steepest-downhill direction. The learning rate $\eta$ sets how far we trust that linear picture before recomputing the gradient. Too large a step overshoots into territory where the approximation no longer holds, too small a step makes needlessly slow progress.

{{< figure
  src="/images/ml/gradientDescentTaylorExpansion.png"
  alt="A curve, its tangent line at a point, and the linear step taken along the negative gradient direction."
  caption="The first-order Taylor expansion approximates the loss by its tangent at the current point. Stepping along $\\Delta x = -\\nabla \\mathcal{L}$ moves down that tangent, and the step size controls how far we go before the approximation stops being trustworthy."
>}}

{{< callout type="example" title="Steepest descent in one dimension" >}}
Take $\mathcal{L}(\theta) = \tfrac{1}{2}\theta^2$, so $\mathcal{L}'(\theta) = \theta$. At $\theta_0 = 3$ the loss is $4.5$ and the derivative is $3$, a positive slope, so the tangent line $\mathcal{L}(3) + 3(\theta - 3)$ goes uphill to the right and downhill to the left. Stepping against the derivative, $\theta_1 = 3 - \eta\cdot 3$, moves left and lowers the loss. With $\eta = 0.1$ we get $\theta_1 = 2.7$ and $\mathcal{L}(\theta_1) = 3.645 < 4.5$. In one dimension "opposite to the gradient" is just "downhill", in higher dimensions the Cauchy-Schwarz argument is what picks out the single best direction.
{{< /callout >}}

### Stochastic Gradient Descent

In machine learning the quantity we truly want to minimise is the **risk**, the expected loss over the whole data distribution, $\mathbb{E}_{(\mathbf{x}, y)}[\ell_{\boldsymbol{\theta}}(\mathbf{x}, y)]$. We cannot compute that expectation because we do not have the distribution, only a finite training set, so we minimise the **empirical risk**, the average loss over the $n$ training examples,

$$
\mathcal{L}(\boldsymbol{\theta}) = \frac{1}{n}\sum_{i=1}^n \ell_i(\boldsymbol{\theta})
$$

where $\ell_i(\boldsymbol{\theta})$ is the loss on the $i$-th example. This is a [Monte-Carlo estimate](/garden/maths/probabilityStatistics/largeSamples) of the risk, the sample mean standing in for the expectation. Computing the full gradient still means summing over all $n$ examples at every step, which is expensive for large datasets. **Stochastic gradient descent (SGD)** goes one step further and estimates the gradient itself from a random **minibatch** $S_t \subseteq \{1, \dots, n\}$,

$$
\boldsymbol{\theta}_{k+1} = \boldsymbol{\theta}_k - \eta\,\frac{1}{|S_t|}\sum_{i \in S_t} \nabla \ell_i(\boldsymbol{\theta}_k)
$$

The minibatch gradient is an **unbiased estimate** of the full gradient. An estimate is [unbiased](/garden/maths/probabilityStatistics/estimators) when its expected value equals the quantity it estimates, so there is no systematic error, only noise that averages out. Here the randomness is the choice of minibatch.

{{< callout type="proof" title="The minibatch gradient is unbiased" >}}
Draw a minibatch of fixed size $m = |S_t|$ uniformly at random from the $n$ examples. By symmetry each index $i$ is equally likely to be included, and since the $m$ slots are filled from $n$ examples the probability that any given $i$ is in the batch is $\mathbb{P}(i \in S_t) = m/n$. Writing the minibatch gradient with an indicator and using linearity of expectation,

$$
\mathbb{E}_{S_t}\!\left[\frac{1}{m}\sum_{i \in S_t} \nabla \ell_i(\boldsymbol{\theta})\right]
= \frac{1}{m}\sum_{i=1}^n \mathbb{P}(i \in S_t)\,\nabla \ell_i(\boldsymbol{\theta})
= \frac{1}{m}\sum_{i=1}^n \frac{m}{n}\,\nabla \ell_i(\boldsymbol{\theta})
= \frac{1}{n}\sum_{i=1}^n \nabla \ell_i(\boldsymbol{\theta})
= \nabla \mathcal{L}(\boldsymbol{\theta})
$$

The batch size $m$ cancels, leaving exactly the full gradient.
{{< /callout >}}

So on average SGD steps in the same direction as full-batch gradient descent, only with random noise around it. That noise is cheap, since each step looks at a handful of examples rather than all $n$, and it is even helpful for escaping saddle points, where the gradient nearly vanishes but the point is not a minimum and the noise nudges the iterate off the unstable directions. The convergence analysis below is written for full-batch gradient descent, since it isolates the geometry of the loss from the sampling noise.

## The Local Quadratic Model

For a non-convex loss it is hard to say anything globally, but the picture near a minimum is simple, and that is what determines the final stretch of training. Any smooth loss looks like a quadratic bowl close to a minimum $\boldsymbol{\theta}^*$, for the same reason any smooth one-dimensional function looks like a parabola near its bottom. A second-order [Taylor expansion](/garden/maths/calculus/taylor) around $\boldsymbol{\theta}^*$ makes this precise,

$$
\mathcal{L}(\boldsymbol{\theta}) \approx \mathcal{L}(\boldsymbol{\theta}^*) + \nabla\mathcal{L}(\boldsymbol{\theta}^*)^T(\boldsymbol{\theta} - \boldsymbol{\theta}^*) + \tfrac{1}{2}(\boldsymbol{\theta} - \boldsymbol{\theta}^*)^T \mathbf{H}\,(\boldsymbol{\theta} - \boldsymbol{\theta}^*)
$$

The gradient is zero at a minimum, so the linear term vanishes and the leading behaviour is the quadratic term built from the **Hessian** $\mathbf{H} = \nabla^2 \mathcal{L}(\boldsymbol{\theta}^*)$, the matrix of second derivatives that measures curvature. At a minimum the Hessian is positive semidefinite. Understanding gradient descent on a quadratic therefore tells us how it behaves near any minimum, so we study the convex quadratic

$$
\mathcal{L}(\boldsymbol{\theta}) = \tfrac{1}{2}\boldsymbol{\theta}^T \mathbf{H}\,\boldsymbol{\theta} - \mathbf{b}^T\boldsymbol{\theta}, \qquad \mathbf{H} \succ 0
$$

Here $\mathbf{H}$ is the symmetric positive definite curvature matrix and $\mathbf{b}$ is a linear tilt that shifts where the minimum sits. The gradient is $\nabla\mathcal{L}(\boldsymbol{\theta}) = \mathbf{H}\boldsymbol{\theta} - \mathbf{b}$, which is zero at the minimum $\boldsymbol{\theta}^* = \mathbf{H}^{-1}\mathbf{b}$.

{{< callout type="example" title="A concrete 2D quadratic" >}}
Take $\mathbf{H} = \begin{bmatrix} 10 & 0 \\ 0 & 1 \end{bmatrix}$ and $\mathbf{b} = \mathbf{0}$, so $\mathcal{L}(\boldsymbol{\theta}) = 5\theta_1^2 + \tfrac{1}{2}\theta_2^2$. The contours are ellipses, ten times steeper along $\theta_1$ than along $\theta_2$, and the gradient is $\nabla\mathcal{L} = (10\theta_1, \theta_2)$. The two diagonal entries of $\mathbf{H}$ are the curvatures of the bowl along the two axes. A non-zero $\mathbf{b}$, say $\mathbf{b} = (10, 2)$, would just move the minimum to $\boldsymbol{\theta}^* = \mathbf{H}^{-1}\mathbf{b} = (1, 2)$ without changing the shape. The mismatch between the steep and flat directions is exactly what makes gradient descent struggle, as we see next.
{{< /callout >}}

## Coordinate Decoupling

The quadratic still mixes the coordinates through the off-diagonal entries of $\mathbf{H}$, which geometrically means the elliptical contours of the bowl are tilted with respect to the parameter axes. The fix is to rotate into the natural axes of the bowl, the eigenvectors of $\mathbf{H}$, which are its principal axes, with the eigenvalues giving the curvature along each. Since $\mathbf{H}$ is symmetric, the [spectral theorem](/garden/maths/linearAlgebra/eigendecomposition) diagonalises it as $\mathbf{H} = \mathbf{U}\boldsymbol{\Lambda}\mathbf{U}^T$ with $\mathbf{U}^T\mathbf{U} = \mathbf{I}$ and $\boldsymbol{\Lambda} = \text{diag}(\lambda_1, \dots, \lambda_d)$.

Now change to the rotated coordinates $\boldsymbol{\nu} = \mathbf{U}^T\boldsymbol{\theta}$, equivalently $\boldsymbol{\theta} = \mathbf{U}\boldsymbol{\nu}$, and write $\mathbf{q} = \mathbf{U}^T\mathbf{b}$. Substituting into the quadratic and using $\mathbf{U}^T\mathbf{H}\mathbf{U} = \boldsymbol{\Lambda}$,

$$
\mathcal{L}(\mathbf{U}\boldsymbol{\nu}) = \tfrac{1}{2}(\mathbf{U}\boldsymbol{\nu})^T \mathbf{H}\,(\mathbf{U}\boldsymbol{\nu}) - \mathbf{b}^T \mathbf{U}\boldsymbol{\nu}
= \tfrac{1}{2}\boldsymbol{\nu}^T \boldsymbol{\Lambda}\,\boldsymbol{\nu} - \mathbf{q}^T\boldsymbol{\nu}
= \sum_{i=1}^d \left(\frac{\lambda_i}{2}\nu_i^2 - q_i \nu_i\right)
$$

The cross terms are gone, because $\boldsymbol{\Lambda}$ is diagonal. The loss has separated into a sum of independent one-dimensional parabolas, one per eigenvalue,

$$
\mathcal{L}(\boldsymbol{\nu}) = \sum_{i=1}^d \mathcal{L}_i(\nu_i), \qquad \mathcal{L}_i(\nu_i) = \frac{\lambda_i}{2}\nu_i^2 - q_i \nu_i
$$

Because gradient descent is rotation-equivariant, in these coordinates each $\nu_i$ evolves entirely on its own, driven only by its own curvature $\lambda_i$. The behaviour of the whole algorithm is just the behaviour of these $d$ decoupled scalar problems, which is what makes the quadratic so tractable.

## Step Size and Conditioning

Now the analysis reduces to a single coordinate $\mathcal{L}_i(\nu_i) = \tfrac{\lambda_i}{2}\nu_i^2 - q_i\nu_i$. Its minimum is at $\nu_i^* = q_i / \lambda_i$. It is convenient to measure progress by the shifted loss, which is just how far the current value sits above the minimum value along this coordinate,

$$
\tilde{\mathcal{L}}_i(\nu_i) = \mathcal{L}_i(\nu_i) - \mathcal{L}_i(\nu_i^*) = \frac{1}{2\lambda_i}(\lambda_i \nu_i - q_i)^2 \ge 0
$$

A gradient step on this coordinate is $\nu_i^+ = \nu_i - \eta\,\mathcal{L}_i'(\nu_i) = \nu_i - \eta\,(\lambda_i \nu_i - q_i)$. Substituting and factoring shows that each step multiplies the shifted loss by a constant factor,

$$
\tilde{\mathcal{L}}_i(\nu_i^+) = \frac{1}{2\lambda_i}\big(\lambda_i \nu_i^+ - q_i\big)^2 = \frac{1}{2\lambda_i}\big((1 - \lambda_i\eta)(\lambda_i \nu_i - q_i)\big)^2 = (1 - \lambda_i\eta)^2\,\tilde{\mathcal{L}}_i(\nu_i)
$$

That factor $(1 - \lambda_i\eta)^2$ is the **contraction rate** of the coordinate. The coordinate makes progress only when the rate is below one, that is when $|1 - \lambda_i\eta| < 1$, which means

$$
0 < \eta < \frac{2}{\lambda_i}
$$

Here is the problem in one line. The step size has to satisfy this for every coordinate at once, but the coordinates have different curvatures $\lambda_i$. To keep the steepest direction from diverging we need $\eta < 2/\lambda_{\max}$, yet the flattest direction then contracts by only $1 - \eta\lambda_{\min} \approx 1$ and barely moves. One global step size cannot suit both. The best compromise balances the two worst coordinates, the largest and smallest eigenvalues. Minimising the worst contraction $\max_i (1 - \eta\lambda_i)^2$ over $\eta$ gives

$$
\eta^* = \arg\min_\eta \max_i (1 - \eta\lambda_i)^2 = \frac{2}{\lambda_{\max} + \lambda_{\min}}
$$

and at this step size the slowest coordinate contracts by the weakest rate

$$
\rho = \left(\frac{\lambda_{\max} - \lambda_{\min}}{\lambda_{\max} + \lambda_{\min}}\right)^2 = \left(\frac{\kappa - 1}{\kappa + 1}\right)^2, \qquad \kappa = \frac{\lambda_{\max}}{\lambda_{\min}}
$$

The quantity $\kappa$ is the **condition number** of $\mathbf{H}$, the ratio of its largest to smallest eigenvalue, equivalently the ratio of the steepest to the flattest curvature of the bowl. It is what we gain by all this analysis, a single number that controls the speed.

- If $\kappa = 1$ the bowl is perfectly round, $\rho = 0$, and gradient descent reaches the minimum in a single step, because one step size is exactly right for every direction.
- If $\kappa \gg 1$ the bowl is a long narrow valley, $\rho \to 1$, and convergence is slow. The iterate overshoots the steep direction and crawls along the flat one, so it zigzags across the valley instead of running down its floor.

{{< figure
  src="/images/ml/conditionNumberZigzag.png"
  alt="Gradient descent zigzagging across the steep walls of an elliptical valley towards the minimum."
  caption="On an ill-conditioned quadratic the step size is limited by the steep direction, so gradient descent zigzags across the valley while making slow progress along its flat floor."
>}}

Gradient descent converges fastest when $\mathbf{H}$ is well-conditioned. This is the same spectrum that appears in [PCA](/garden/maths/linearAlgebra/pca), and ill-conditioning is the central reason the practical [optimizers](/garden/ml/optimizers) exist, they reshape the effective geometry so that a large $\kappa$ hurts less.

## Smoothness

The quadratic analysis assumed an exact second-order model. For a general loss we need a condition that keeps the gradient informative over a whole neighbourhood, so that a step of a fixed size cannot overshoot into a region where the gradient it was based on is meaningless. That condition is built on Lipschitz continuity, which is worth recalling.

{{< callout type="info" title="Lipschitz continuity" >}}
A function $f$ is **$L$-Lipschitz continuous** if it never changes faster than a fixed rate $L$,

$$
\|f(\mathbf{x}) - f(\mathbf{y})\| \le L\,\|\mathbf{x} - \mathbf{y}\| \qquad (\forall\, \mathbf{x}, \mathbf{y})
$$

In one dimension this bounds the slope everywhere by $L$. Geometrically, from any point on the graph the function must stay inside a cone of slopes $\pm L$, so it cannot jump or kink arbitrarily sharply. The smaller $L$, the more slowly the function is allowed to vary.

{{< figure
  src="/images/ml/lipshitzContinous.png"
  alt="A Lipschitz-continuous function staying within a bounded-slope cone, next to a non-Lipschitz function that escapes it."
  caption="Left, an $L$-Lipschitz function stays within the cone of slopes $\\pm L$ at every point. Right, a function whose slope is unbounded is not Lipschitz."
>}}
{{< /callout >}}

A loss is **$L$-smooth** when its *gradient* is $L$-Lipschitz continuous,

$$
\|\nabla \mathcal{L}(\boldsymbol{\theta}) - \nabla \mathcal{L}(\boldsymbol{\theta}')\| \le L\,\|\boldsymbol{\theta} - \boldsymbol{\theta}'\| \qquad (\forall\, \boldsymbol{\theta}, \boldsymbol{\theta}')
$$

so the gradient is not allowed to change too fast. For a twice-differentiable loss this is the same as bounding the Hessian, $\nabla^2 \mathcal{L}(\boldsymbol{\theta}) \preceq L\,\mathbf{I}$, so $L$ caps the largest curvature anywhere on the surface. Because $L$ limits how fast the gradient can change, a step based on the current gradient stays trustworthy for a distance of order $1/L$, which is why a step size around $\eta = 1/L$ turns out to be safe. Geometrically, $L$-smoothness means the loss never curves upward faster than a parabola of curvature $L$, so from any point the graph stays below its own tangent plane with such a parabola added on top. Writing that bound out gives the **descent lemma**, an upper bound on the loss after a step,

$$
\mathcal{L}(\boldsymbol{\theta}') - \mathcal{L}(\boldsymbol{\theta}) \le \langle \nabla \mathcal{L}(\boldsymbol{\theta}), \boldsymbol{\theta}' - \boldsymbol{\theta}\rangle + \frac{L}{2}\|\boldsymbol{\theta}' - \boldsymbol{\theta}\|^2
$$

For the gradient descent step $\boldsymbol{\theta}' = \boldsymbol{\theta} - \eta\nabla\mathcal{L}(\boldsymbol{\theta})$ this becomes

$$
\mathcal{L}(\boldsymbol{\theta}') - \mathcal{L}(\boldsymbol{\theta}) \le -\eta\left(1 - \frac{L\eta}{2}\right)\|\nabla \mathcal{L}(\boldsymbol{\theta})\|^2
$$

The right-hand side is negative as long as $\eta < 2/L$, and choosing $\eta = 1/L$ maximises the guaranteed decrease, giving the clean bound

$$
\mathcal{L}(\boldsymbol{\theta}') - \mathcal{L}(\boldsymbol{\theta}) \le -\frac{1}{2L}\|\nabla \mathcal{L}(\boldsymbol{\theta})\|^2
$$

Each step decreases the loss by at least an amount proportional to the squared gradient norm, and since this holds at every point the loss decreases monotonically. A small gradient, or a large smoothness constant $L$ meaning a steep landscape, both force a small step. Both convergence results below are built on this single inequality, they differ only in the extra assumption they add to turn a guaranteed small step into a guarantee about where we end up.

### Finding Critical Points

What does repeatedly decreasing the loss buy us? On a non-convex landscape we cannot promise to reach the global minimum, but we can promise to reach a **critical point**, a place where the gradient is nearly zero. Call $\boldsymbol{\theta}$ an $\epsilon$-critical point if $\|\nabla \mathcal{L}(\boldsymbol{\theta})\| \le \epsilon$. The reason one must show up is a simple accounting argument. Each step removes an amount of loss proportional to its squared gradient, yet the total drop can never exceed the initial gap above the minimum, so the gradients cannot all stay large and at least one iterate must have a small one.

{{< callout type="proof" title="Gradient descent finds an epsilon-critical point in O(epsilon^-2) steps" >}}
Sum the descent-lemma bound over the first $k$ steps. With $C := \mathcal{L}(\boldsymbol{\theta}_0) - \mathcal{L}^* \ge 0$ and using that $\mathcal{L}(\boldsymbol{\theta}_k) \ge \mathcal{L}^*$,

$$
-C \le \mathcal{L}(\boldsymbol{\theta}_k) - \mathcal{L}(\boldsymbol{\theta}_0) \le -\frac{1}{2L}\sum_{r=0}^{k-1}\|\nabla \mathcal{L}(\boldsymbol{\theta}_r)\|^2
$$

Rearranging gives an average bound on the squared gradient norms,

$$
\frac{1}{k}\sum_{r=0}^{k-1}\|\nabla \mathcal{L}(\boldsymbol{\theta}_r)\|^2 \le \frac{2LC}{k}
$$

The smallest term is no larger than the average, so for at least one iterate $\boldsymbol{\theta}' \in \{\boldsymbol{\theta}_0, \dots, \boldsymbol{\theta}_{k-1}\}$,

$$
\|\nabla \mathcal{L}(\boldsymbol{\theta}')\|^2 \le \frac{2LC}{k} \overset{!}{\le} \epsilon^2 \quad\Longleftrightarrow\quad k \ge \frac{2LC}{\epsilon^2}
$$
{{< /callout >}}

So smoothness alone is enough to reach an $\epsilon$-critical point in $O(\epsilon^{-2})$ steps. The catch is that a critical point is only a place where the gradient is small. It need not be a minimum, it could be a saddle point, and the bound says nothing about how good the loss value there is. To promise convergence to the minimum value we need a condition that connects the gradient back to the suboptimality $\mathcal{L}(\boldsymbol{\theta}) - \mathcal{L}^*$.

## The Polyak-Lojasiewicz Condition

{{< callout type="info" title="Polyak-Lojasiewicz condition" >}}
A differentiable function $\mathcal{L}$ obeys the **Polyak-Lojasiewicz (PL) condition** with parameter $\mu > 0$ if

$$
\tfrac{1}{2}\|\nabla \mathcal{L}(\boldsymbol{\theta})\|^2 \ge \mu\,(\mathcal{L}(\boldsymbol{\theta}) - \mathcal{L}^*) \qquad (\forall\,\boldsymbol{\theta}), \qquad \mathcal{L}^* = \min_{\boldsymbol{\theta}} \mathcal{L}(\boldsymbol{\theta})
$$

In words, the gradient is not allowed to go flat while we are still far above the minimum. Wherever the loss is large the gradient must be large too, so a method that follows the gradient cannot stall above the minimum.
{{< /callout >}}

This is exactly the missing link from the previous section. Smoothness alone only drove the gradient small, but PL turns a small gradient into a small suboptimality, so a near-critical point is now guaranteed to be near-optimal too. That upgrades the $O(\epsilon^{-2})$ guarantee into geometric (linear) convergence to the actual minimum value.

{{< callout type="proof" title="L-smooth and mu-PL implies geometric convergence" >}}
Let $\mathcal{L}$ be differentiable, $L$-smooth and $\mu$-PL. Then gradient descent with step size $\eta = 1/L$ satisfies

$$
\mathcal{L}(\boldsymbol{\theta}_k) - \mathcal{L}^* \le \left(1 - \frac{\mu}{L}\right)^k (\mathcal{L}(\boldsymbol{\theta}_0) - \mathcal{L}^*)
$$

The proof combines the two ingredients we already have. First, the descent lemma with $\eta = 1/L$ gives

$$
\mathcal{L}(\boldsymbol{\theta}_{k+1}) - \mathcal{L}(\boldsymbol{\theta}_k) \le -\frac{1}{2L}\|\nabla \mathcal{L}(\boldsymbol{\theta}_k)\|^2
$$

The PL condition then rewrites the gradient norm in terms of the suboptimality,

$$
-\|\nabla \mathcal{L}(\boldsymbol{\theta}_k)\|^2 \le -2\mu\,(\mathcal{L}(\boldsymbol{\theta}_k) - \mathcal{L}^*)
$$

Combining the two and subtracting $\mathcal{L}^*$ from both sides yields a one-step contraction,

$$
\mathcal{L}(\boldsymbol{\theta}_{k+1}) - \mathcal{L}^* \le \left(1 - \frac{\mu}{L}\right)(\mathcal{L}(\boldsymbol{\theta}_k) - \mathcal{L}^*)
$$

and the claim follows by induction.
{{< /callout >}}

The contraction factor $1 - \mu/L$ lies in $[0, 1)$ since $\mu \le L$, so every step removes a fixed fraction of the remaining gap to the minimum and the suboptimality shrinks exponentially fast. As an illustration, $f(x) = 4x^2$ satisfies PL with $\mu = 8$, while $f(x) = x^4$ does not satisfy PL at $x = 0$, because its gradient flattens out faster than the suboptimality near the minimum.

{{< figure
  src="/images/ml/plCondition.png"
  alt="The curves 4x^2 and x^4, the first satisfying the PL condition and the second not."
  caption="$f(x) = 4x^2$ satisfies the PL condition, its gradient grows in step with the suboptimality. $f(x) = x^4$ does not, its gradient vanishes too quickly near the minimum, so progress stalls."
>}}

## Strong Convexity

A natural and well-known way to guarantee the PL condition is strong convexity.

{{< callout type="info" title="Strong convexity" >}}
A differentiable $\mathcal{L}$ is **$\mu$-strongly convex** for some $\mu > 0$ if

$$
\mathcal{L}(\boldsymbol{\theta}') \ge \mathcal{L}(\boldsymbol{\theta}) + \langle \nabla \mathcal{L}(\boldsymbol{\theta}), \boldsymbol{\theta}' - \boldsymbol{\theta}\rangle + \frac{\mu}{2}\|\boldsymbol{\theta}' - \boldsymbol{\theta}\|^2 \qquad (\forall\,\boldsymbol{\theta}', \boldsymbol{\theta})
$$

This is the first-order convexity condition with a quadratic cushion of curvature $\mu$ added underneath, so the function sits above not just its tangent but a parabola resting on that tangent, and is curved at least as much as that parabola everywhere. The case $\mu = 0$ recovers ordinary convexity. A positive definite quadratic is strongly convex with $\mu = \lambda_{\min}$, and for a twice-differentiable $\mathcal{L}$ strong convexity and smoothness together sandwich the Hessian,

$$
0 \prec \mu\,\mathbf{I} \preceq \nabla^2 \mathcal{L}(\boldsymbol{\theta}) \preceq L\,\mathbf{I} \qquad (\forall\,\boldsymbol{\theta})
$$

So strong convexity floors the curvature from below by $\mu$ exactly as smoothness caps it from above by $L$, and their ratio $L/\mu$ is the condition number $\kappa$ from before.
{{< /callout >}}

{{< figure
  src="/images/ml/strongConvexity.png"
  alt="A function with quadratic lower bounds tangent to it at two points, illustrating strong convexity."
  caption="Strong convexity means a quadratic of curvature $\\mu$ can be slid under the graph at every point, touching it tangentially. The function lies above each such parabola, not just above each tangent line."
>}}

Strong convexity is the exact mirror of smoothness. Smoothness slips a parabola of curvature $L$ over the graph as a ceiling, strong convexity slips one of curvature $\mu$ under it as a floor, and together they trap the loss between two parabolas at every point. Their ratio is the condition number $\kappa = L/\mu$, the very quantity that set the speed of gradient descent on the quadratic, now measuring how much room sits between the tightest floor and the loosest ceiling.

{{< callout type="proof" title="Strong convexity implies PL with the same mu" >}}
Minimise both sides of the strong convexity inequality over $\boldsymbol{\theta}'$. The left side minimised is $\mathcal{L}^* - \mathcal{L}(\boldsymbol{\theta})$. The right side is a quadratic in $\boldsymbol{\theta}'$, minimised at $\boldsymbol{\theta}' = \boldsymbol{\theta} - \tfrac{1}{\mu}\nabla \mathcal{L}(\boldsymbol{\theta})$,

$$
\mathcal{L}^* - \mathcal{L}(\boldsymbol{\theta}) \ge \min_{\boldsymbol{\theta}'}\left\{ \langle \nabla \mathcal{L}(\boldsymbol{\theta}), \boldsymbol{\theta}' - \boldsymbol{\theta}\rangle + \frac{\mu}{2}\|\boldsymbol{\theta}' - \boldsymbol{\theta}\|^2 \right\} = -\frac{1}{2\mu}\|\nabla \mathcal{L}(\boldsymbol{\theta})\|^2
$$

Rearranging gives $\tfrac{1}{2}\|\nabla \mathcal{L}(\boldsymbol{\theta})\|^2 \ge \mu\,(\mathcal{L}(\boldsymbol{\theta}) - \mathcal{L}^*)$, which is exactly the PL condition.
{{< /callout >}}

So every strongly convex function is PL, and gradient descent converges geometrically on it. The PL condition is strictly weaker, it does not even require convexity, which is what makes it the more useful tool for analysing deep networks.

## What This Means for Deep Networks

None of these conditions hold globally for a real deep network. Its loss is non-convex, and in high dimensions the critical points it is littered with are mostly saddle points and plateaus rather than bad local minima. Plain gradient descent crawls near a saddle, since the gradient there is small even though the point is not a minimum, but the noise in stochastic gradient descent nudges the iterate off the unstable directions and lets it escape.

What the theory does give is a local picture. Smoothness and the PL or strong convexity conditions typically hold over a region around a good minimum, which is enough to guarantee fast local convergence once gradient descent gets close, even though it makes no claim about how good that minimum is. The recurring obstacle throughout, on the quadratic and beyond, is the conditioning $\kappa = L/\mu$, the spread between the steepest and flattest curvature. The closer it is to one the faster gradient descent converges. The [optimizers](/garden/ml/optimizers) note takes this as its starting point and develops momentum and adaptive methods that cope with ill-conditioning, small gradients, and saddle points far better than plain gradient descent.
