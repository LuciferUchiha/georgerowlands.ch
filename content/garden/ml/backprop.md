---
title: Backpropagation
type: docs
weight: 4
---

A neural network is a composition of layers. Most commonly each layer applies a linear map followed by a non-linear activation, and stacking many of them gives a deep neural network (see the [neural networks](/garden/ml/neuralNetworks) note for the model itself). Training such a network means adjusting the weights of every layer so that the network minimises a loss function, and to do that with [gradient descent](/garden/ml/gradientDescent) we need the gradient of the loss with respect to every single weight.

**Backpropagation** is the algorithm that computes all of these gradients efficiently. The key idea is that a network is one big composite function, so the chain rule lets us compute the gradients layer by layer, reusing the work already done for the layers closer to the output. We first build intuition with a small worked example, then derive the general matrix form, and finally see how automatic differentiation turns the derivation into a [dynamic programming](/garden/cs/algd/dynamicProgramming/introduction) algorithm that is implemented in every deep learning framework.

## Forward Pass

The forward pass, sometimes also called forward propagation, is the process of calculating the output of a neural network given an input. We run the input through the network layer by layer, applying the activation function to the output of each layer, and the output of the last layer is the output of the network.

Each layer is defined by a weight matrix $\mathbf{W}$ and a bias vector $\mathbf{b}$. The vector $\mathbf{z}$ holds the **pre-activations** and the vector $\mathbf{h}$ the **activations**, the outputs of the layer after the activation function $\sigma$ is applied,

$$
\begin{align*}
\mathbf{z} &= \mathbf{W}\mathbf{x} + \mathbf{b} \\
\mathbf{h} &= \sigma(\mathbf{z})
\end{align*}
$$

{{< callout type="example" title="Forward pass through a 2-2-1 network" >}}
Take a small network with an input layer of size 2, a hidden layer of size 2, and an output layer of size 1, using the sigmoid activation for both layers. The weights $w_1, w_2, w_3, w_4$ connect the two inputs to the two hidden neurons, and $w_5, w_6$ connect the two hidden neurons to the output.

| Variable | Value |
| -------- | ----- |
| $x_1$    | 0.888 |
| $x_2$    | -0.49 |
| $w_1$    | 1.76  |
| $w_2$    | 0.4   |
| $w_3$    | 0.97  |
| $w_4$    | 2.24  |
| $w_5$    | 1.86  |
| $w_6$    | -0.97 |
| $b_1$    | 0     |
| $b_2$    | 0     |
| $b_3$    | 0     |

First the pre-activation and activation of the first hidden neuron,

$$
\begin{align*}
z_1 &= x_1w_1 + x_2w_2 + b_1 = 0.888 \cdot 1.76 + (-0.49) \cdot 0.4 + 0 = 1.367 \\
h_1 &= \sigma(z_1) = \frac{1}{1 + e^{-1.367}} = 0.797
\end{align*}
$$

The same for the other neuron gives $z_2 = 0.888 \cdot 0.97 + (-0.49) \cdot 2.24 + 0 = -0.236$ and $h_2 = \sigma(-0.236) = 0.441$. With the hidden activations in hand we compute the output,

$$
\begin{align*}
z_3 &= h_1w_5 + h_2w_6 + b_3 = 0.797 \cdot 1.86 + 0.441 \cdot (-0.97) + 0 = 1.055 \\
y &= \sigma(z_3) = \frac{1}{1 + e^{-1.055}} = 0.741
\end{align*}
$$

{{< figure
  src="/images/ml/backpropForwardPass.gif"
  alt="The forward pass of the simple neural network."
  caption="The forward pass of the simple 2-2-1 network."
>}}
{{< /callout >}}

We can also write these calculations in matrix form, which is more efficient and easier to generalise to larger networks. Collecting the weights of the hidden layer into $\mathbf{W}^{(1)}$ and the output weights into $\mathbf{W}^{(2)}$, the whole forward pass is just two matrix multiplications, each followed by an activation,

$$
\mathbf{W}^{(1)} = \begin{bmatrix} w_1 & w_2 \\ w_3 & w_4 \end{bmatrix} = \begin{bmatrix} 1.76 & 0.4 \\ 0.97 & 2.24 \end{bmatrix}, \qquad
\mathbf{W}^{(2)} = \begin{bmatrix} w_5 & w_6 \end{bmatrix} = \begin{bmatrix} 1.86 & -0.97 \end{bmatrix}
$$

$$
\begin{align*}
\mathbf{z}^{(1)} &= \mathbf{W}^{(1)}\mathbf{x} + \mathbf{b}^{(1)}, & \mathbf{h} &= \sigma(\mathbf{z}^{(1)}) \\
z^{(2)} &= \mathbf{W}^{(2)}\mathbf{h} + b^{(2)}, & y &= \sigma(z^{(2)})
\end{align*}
$$

The whole network is therefore the composite function $y = \sigma(\mathbf{W}^{(2)}\sigma(\mathbf{W}^{(1)}\mathbf{x} + \mathbf{b}^{(1)}) + b^{(2)})$. A deeper network simply stacks more of these linear-then-activation layers. This compositional structure is exactly what backpropagation exploits.

## The Chain Rule

The backpropagation algorithm is built on the chain rule from calculus, so let us start with a brief reminder (see [differentiable functions](/garden/maths/calculus/differentiableFunctions) for the full treatment).

{{< callout type="info" title="Chain Rule" >}}
If we have the differentiable functions $f(x)$ and $g(x)$ and the composite function $h(x) = f(g(x))$, where the function $f$ is applied to the output of $g$, then the derivative of $h$ with respect to $x$ is

$$
h'(x) = f'(g(x))g'(x) \qquad \text{or} \qquad \frac{dh}{dx} = \frac{df}{dg}\frac{dg}{dx}
$$

Notice that the denominator $dg$ is the same as the following numerator, which can be thought of as "the chain". The chain rule also makes sense intuitively if we think of $dg$ cancelling out in the numerator and denominator.

For example, if $h(x) = (x^2 + 1)^3$, then we can write $h(x) = f(g(x))$ with $f(x) = x^3$ and $g(x) = x^2 + 1$, and the derivative is

$$
h'(x) = f'(g(x))g'(x) = 3(x^2 + 1)^2 \cdot 2x = 6x(x^2 + 1)^2
$$

The key takeaway is that the derivative of a composite function can be calculated step by step, working from the outermost function inward. This is the key idea behind backpropagation, since a neural network is just one big composite function with many variables and many inner functions.
{{< /callout >}}

A neural network does not have a single input variable but a whole vector of activations feeding into each layer, so we need the multivariable version of the chain rule.

{{< callout type="info" title="Multivariate Chain Rule" >}}
Suppose a scalar $\ell$ depends on the intermediate variables $u_1, \dots, u_m$, and each $u_j$ in turn depends on a variable $x$. A change in $x$ reaches $\ell$ through every $u_j$, and the contributions add up,

$$
\frac{d\ell}{dx} = \sum_{j=1}^m \frac{\partial \ell}{\partial u_j}\frac{\partial u_j}{\partial x}
$$

If instead $\mathbf{u} = \mathbf{g}(\mathbf{x})$ is a vector-valued map and $\ell = f(\mathbf{u})$, stacking these sums for every input coordinate is exactly a matrix-vector product,

$$
\nabla_{\mathbf{x}} \ell = \mathbf{J}_{\mathbf{g}}^T \, \nabla_{\mathbf{u}} \ell, \qquad [\mathbf{J}_{\mathbf{g}}]_{ji} = \frac{\partial u_j}{\partial x_i}
$$

where $\mathbf{J}_{\mathbf{g}}$ is the Jacobian matrix of $\mathbf{g}$, defined in the next section. This "sum over all paths" is the rule that backpropagation applies at every layer.
{{< /callout >}}

## Backpropagation

We want the gradient of the loss with respect to the weights of every layer. Write the network as a composition of layer maps $\mathbf{H}_1, \dots, \mathbf{H}_L$, with $\mathbf{z}_k = \mathbf{H}_k(\mathbf{z}_{k-1})$ the activations of layer $k$, the input $\mathbf{z}_0 = \mathbf{x}$, and the loss $\ell$ computed from the output $\mathbf{z}_L$. A naive approach would differentiate the loss with respect to each weight separately, but that repeats almost all of the work, because an early weight influences the loss only through every layer above it. Backpropagation organises the computation so that each shared intermediate result is computed exactly once.

### The Error Signal

The central object of the algorithm is the **error signal**, also called the **delta**. For layer $k$ it is the gradient of the loss with respect to that layer's activations,

$$
\boldsymbol{\delta}_k := \frac{\partial \ell}{\partial \mathbf{z}_k}
$$

It measures how much the loss responds to a small change in the activations $\mathbf{z}_k$, or informally, how strongly the loss "wants" that layer's output to change.

{{< callout type="info" title="Is the error signal not just the gradient?" >}}
It is a gradient, but not the gradient we ultimately want. The error signal is the gradient of the loss with respect to an intermediate activation, an internal quantity inside the network. What we actually need for the update is the gradient with respect to the weights. The error signal is the bridge between the two. Once we know how sensitive the loss is to a layer's output, a single extra step of the chain rule turns that into the gradient with respect to that layer's weights, so we compute the error signals first and read the weight gradients off them.
{{< /callout >}}

To see that last step, zoom in on a single unit $h = \phi(\langle \mathbf{w}, \mathbf{z}\rangle)$ with weights $\mathbf{w}$, input $\mathbf{z}$ from the previous layer, and activation function $\phi$. It helps to fix the two directions.

- **Upstream** is the input side, toward $\mathbf{x}$. The vector $\mathbf{z}$ is the upstream activation flowing into the unit during the forward pass.
- **Downstream** is the output side, toward the loss. The error signal $\delta = \partial \ell / \partial h$ flows back from there during the backward pass.

By the chain rule the unit's weight gradient factors into one piece from each direction plus a local one,

$$
\nabla_{\mathbf{w}} \ell = \frac{\partial \ell}{\partial h}\,\nabla_{\mathbf{w}} h = \underbrace{\delta}_{\text{downstream}} \cdot \underbrace{\phi'(\langle \mathbf{w}, \mathbf{z}\rangle)}_{\text{local}} \cdot \underbrace{\mathbf{z}}_{\text{upstream}}
$$

The downstream error signal $\delta$ says how much the loss cares about this unit's output, the local sensitivity $\phi'$ says how much the output moves when the pre-activation moves, and the upstream input $\mathbf{z}$ says how much the pre-activation moves when each weight moves. Everything except $\delta$ is already known from the forward pass, so the whole problem reduces to computing the error signals efficiently. The recurrence below does exactly that, reusing each downstream error signal for every unit beneath it instead of recomputing it.

### The Recurrence

To move an error signal from one layer to the previous one we need the Jacobian of a layer map, so it is worth being clear on how a Jacobian relates to the other derivative objects.

{{< callout type="info" title="Derivative, gradient, Jacobian, Hessian" >}}
These are all derivatives. They differ only in the shapes of their input and output.

- A **derivative** $f'(x)$ is for a scalar function of a scalar $f : \mathbb{R} \to \mathbb{R}$. It is a single number, the slope.
- A **gradient** $\nabla f(\mathbf{x})$ is for a scalar function of a vector $f : \mathbb{R}^n \to \mathbb{R}$. It is the vector of the $n$ partial derivatives and points in the direction of steepest increase.
- A **Jacobian** $\mathbf{J}_{\mathbf{f}}$ is for a vector function of a vector $\mathbf{f} : \mathbb{R}^n \to \mathbb{R}^m$. It is the $m \times n$ matrix of every partial derivative $\partial f_i / \partial x_j$, so its $i$-th row is the gradient of the $i$-th output. A gradient is just the special case $m = 1$, a one-row Jacobian.
- A **Hessian** $\nabla^2 f(\mathbf{x})$ is the $n \times n$ matrix of second partial derivatives of a scalar function of a vector. It describes curvature and is used in the [gradient descent](/garden/ml/gradientDescent) note.
{{< /callout >}}

A layer map $\mathbf{H} : \mathbb{R}^n \to \mathbb{R}^m$ takes the activation vector $\mathbf{z}$ of one layer to the activation vector $\mathbf{h} = \mathbf{H}(\mathbf{z})$ of the next, so its Jacobian $[\mathbf{J}_{\mathbf{H}}]_{ij} = \partial h_i / \partial z_j$ says how output component $i$ responds to input component $j$. It is the best local linear approximation of the map, which comes straight from the first-order [Taylor expansion](/garden/maths/calculus/taylor) of a vector-valued function, $\mathbf{H}(\mathbf{z} + \Delta\mathbf{z}) \approx \mathbf{H}(\mathbf{z}) + \mathbf{J}_{\mathbf{H}}(\mathbf{z})\,\Delta\mathbf{z}$. Near $\mathbf{z}$ the map behaves like multiplication by the matrix $\mathbf{J}_{\mathbf{H}}$, which is exactly the object the multivariate chain rule multiplies together.

This is the heart of the algorithm. The loss reaches layer $k$ only through layer $k+1$, so the multivariate chain rule expresses the error signal at layer $k$ in terms of the one at layer $k+1$,

$$
\boldsymbol{\delta}_k = \frac{\partial \ell}{\partial \mathbf{z}_k} = \left[\frac{\partial \mathbf{z}_{k+1}}{\partial \mathbf{z}_k}\right]^T \frac{\partial \ell}{\partial \mathbf{z}_{k+1}} = \mathbf{J}_{k+1}^T \, \boldsymbol{\delta}_{k+1}
$$

where $\mathbf{J}_k = \mathbf{J}_{\mathbf{H}_k}$. Unrolling the recurrence from the output shows that the error signal at any layer is the output error pushed back through a product of transposed layer Jacobians,

$$
\boldsymbol{\delta}_k = \mathbf{J}_{k+1}^T \mathbf{J}_{k+2}^T \cdots \mathbf{J}_L^T \, \boldsymbol{\delta}_L
$$

The reason this is efficient is that we never form that product as a matrix. We start from the output error $\boldsymbol{\delta}_L$ and multiply by one transposed Jacobian at a time, moving backwards through the network. Each step is a **vector-Jacobian product**, the product of an upstream gradient (a vector) with a Jacobian, which costs about as much as one layer of the forward pass. Keeping a vector at every step, rather than building a full Jacobian, is what makes the backward pass as cheap as the forward pass. Each error signal $\boldsymbol{\delta}_k$, once computed, is then reused for every weight in layer $k$ and to produce $\boldsymbol{\delta}_{k-1}$, whereas differentiating each weight separately would recompute these shared products over and over.

{{< callout type="warning" title="The backward pass needs the forward pass" >}}
The error signals carry implicit dependencies on the input $\mathbf{x}$ and target $\mathbf{y}$, since $\boldsymbol{\delta}_L = \boldsymbol{\delta}_L(\mathbf{x}, \mathbf{y})$ and each Jacobian $\mathbf{J}_k = \mathbf{J}_k(\mathbf{z}_{k-1})$ depends on the activation $\mathbf{z}_{k-1}$ feeding into it. Evaluating the Jacobians therefore requires the activations from a forward pass. This is why training always runs a forward pass first and caches the activations, then a backward pass that consumes them.
{{< /callout >}}

For the usual ridge layer $\mathbf{H}(\mathbf{z}) = \phi(\mathbf{W}\mathbf{z})$ the Jacobian has a very simple form. Write $\mathbf{a} = \mathbf{W}\mathbf{z}$ for the pre-activation. The linear part contributes the matrix $\mathbf{W}$ through $\partial a_i / \partial z_j = w_{ij}$, and the activation $\phi$ acts on each component on its own, so $\partial h_i / \partial a_l = \phi'(a_i)$ only when $l = i$ and is zero otherwise. That diagonal of activation derivatives times $\mathbf{W}$ gives, by the chain rule,

$$
\begin{align*}
\text{linear } \phi = \text{id}: \quad & \mathbf{J}_{\mathbf{H}} = \mathbf{W} \\
\text{ReLU}: \quad & \mathbf{J}_{\mathbf{H}} = \text{diag}(\boldsymbol{\chi})\,\mathbf{W}, \quad \boldsymbol{\chi} = \mathbb{1}[\mathbf{W}\mathbf{z} > 0] \in \{0, 1\}^m \\
\text{sigmoid}: \quad & \mathbf{J}_{\mathbf{H}} = \text{diag}(\boldsymbol{\chi})\,\mathbf{W}, \quad \boldsymbol{\chi} = \sigma'(\mathbf{W}\mathbf{z}) \in (0, 1)^m
\end{align*}
$$

The diagonal factor $\boldsymbol{\chi}$ is just the activation derivative at the pre-activation, a gate that for ReLU passes a coordinate through or zeros it out, and for the sigmoid scales it by a number in $(0, 1)$. Plugging this Jacobian into the recurrence, propagating an error signal back through a layer means multiplying elementwise by $\phi'$ and then by $\mathbf{W}^T$. Writing the pre-activation as $\mathbf{a}_k = \mathbf{W}_k \mathbf{z}_{k-1} + \mathbf{b}_k$, the layer's gradients are

$$
\begin{align*}
\boldsymbol{\delta}_{\mathbf{a}_k} &= \boldsymbol{\delta}_k \odot \phi'(\mathbf{a}_k) & &\text{error at the pre-activation} \\
\frac{\partial \ell}{\partial \mathbf{W}_k} &= \boldsymbol{\delta}_{\mathbf{a}_k}\,\mathbf{z}_{k-1}^T & &\text{outer product of error and input} \\
\frac{\partial \ell}{\partial \mathbf{b}_k} &= \boldsymbol{\delta}_{\mathbf{a}_k} & &\text{bias sees the error directly} \\
\boldsymbol{\delta}_{k-1} &= \mathbf{W}_k^T \boldsymbol{\delta}_{\mathbf{a}_k} & &\text{error passed to the layer below}
\end{align*}
$$

where $\odot$ is the elementwise product. The weight gradient is an outer product of the layer's pre-activation error with its input, which is exactly the single-unit rule $\delta\,\phi'\,\mathbf{z}$ applied to every unit at once. To see where the outer product comes from at the component level, a single linear layer $\mathbf{y} = \mathbf{W}\mathbf{z}$ has $\partial y_i / \partial w_{jk} = z_k\,\mathbb{1}\{i = j\}$, since $w_{jk}$ appears only in output $y_j$, so the chain rule collapses to one term,

$$
\left[\frac{\partial \ell}{\partial \mathbf{W}}\right]_{jk} = \sum_{i} \frac{\partial \ell}{\partial y_i}\frac{\partial y_i}{\partial w_{jk}} = \frac{\partial \ell}{\partial y_j}\,z_k \quad\Longrightarrow\quad \frac{\partial \ell}{\partial \mathbf{W}} = \frac{\partial \ell}{\partial \mathbf{y}}\,\mathbf{z}^T
$$

### The Algorithm

The recurrence needs a starting value, the error signal $\boldsymbol{\delta}_L$ at the output, which depends only on the loss function,

$$
\begin{align*}
\text{squared loss } \ell = \tfrac{1}{2}\|\mathbf{y} - \boldsymbol{\Psi}(\mathbf{x})\|^2: \quad & \boldsymbol{\delta}_L = \boldsymbol{\Psi}(\mathbf{x}) - \mathbf{y} \\
\text{logistic loss } \ell = -\ln \sigma(y\Psi(\mathbf{x})), \quad y \in \{-1, 1\}: \quad & \delta_L = -y\,\sigma(-y\Psi(\mathbf{x}))
\end{align*}
$$

The squared-loss error signal is simply the residual, the gap between prediction and target. For the logistic loss the derivation uses the identity $\sigma'(z) = \sigma(z)\sigma(-z)$ from the [vanishing gradients](#vanishing-and-exploding-gradients) section below. With the starting value in hand, training a network on a minibatch results in the following loop

1. Perform a forward pass, computing and caching the activations $\mathbf{z}_1, \dots, \mathbf{z}_L$.
2. Calculate the loss-specific error signal $\boldsymbol{\delta}_L$ at the output.
3. Backpropagate via vector-Jacobian products to obtain $\boldsymbol{\delta}_{L-1}, \dots, \boldsymbol{\delta}_1$.
4. Read the weight gradients off the error signals and accumulate them over the minibatch.
5. Perform a minibatch [gradient descent](/garden/ml/gradientDescent) update with the accumulated gradients.

Applied to the 2-2-1 network from the example, with the squared loss and target $y^*$, the output error is $\delta_3 = (y - y^*)\,\sigma'(z_3)$. The gradient of the output weight $w_5$ is then $\delta_3\, h_1$, since $h_1$ is the input feeding that weight. To reach a hidden weight like $w_1$ we propagate the error one layer further, $\delta_1 = \delta_3\, w_5\, \sigma'(z_1)$, and the gradient is $\delta_1\, x_1$. Notice how the single factor $\delta_3$ is reused for every weight beneath it, which is precisely the saving backpropagation gives over differentiating each weight on its own.

## Vanishing and Exploding Gradients

The backpropagation recurrence $\boldsymbol{\delta}_k = \mathbf{J}_{k+1}^T \cdots \mathbf{J}_L^T \boldsymbol{\delta}_L$ multiplies many layer Jacobians together, and that product is the source of a notorious training problem. If the Jacobians consistently shrink the signal, the product of many of them shrinks geometrically and the gradient of an early layer becomes vanishingly small. This is the **vanishing gradient** problem. If the Jacobians consistently amplify the signal, the product blows up instead and we get the **exploding gradient** problem. Either way the early layers are hard to train, with gradients too small to make progress or too large to be stable.

We can see the vanishing case clearly from the sigmoid. Its derivative comes from the quotient rule on $\sigma(z) = (1 + e^{-z})^{-1}$,

$$
\sigma'(z) = \frac{e^{-z}}{(1 + e^{-z})^2} = \frac{1}{1+e^{-z}}\cdot\frac{e^{-z}}{1+e^{-z}} = \sigma(z)\,\big(1 - \sigma(z)\big) = \sigma(z)\,\sigma(-z)
$$

Since $\sigma(z) \in (0, 1)$, the product $\sigma(z)\sigma(-z)$ is at most $\tfrac{1}{4}$, reached at $z = 0$, and it is close to zero whenever the unit is saturated, that is whenever the pre-activation is large in magnitude. A sigmoid layer Jacobian is $\text{diag}(\boldsymbol{\chi})\mathbf{W}$, so the backpropagated signal scales roughly like the product of the activation-derivative gates and the weight-matrix sizes across layers. Two regimes follow.

- **Vanishing**: with sigmoid activations every gate is below $\tfrac{1}{4}$, so a $d$-layer network multiplies the signal by something on the order of $4^{-d}$, and the early-layer gradients all but disappear.
- **Exploding**: if the weight matrices have large singular values, say spectral norm $s > 1$, the signal grows roughly like $s^{d}$, so $s = 1.5$ over $20$ layers already inflates the gradient by a factor of more than $3000$.

There are several standard remedies.

- A different activation such as ReLU or Leaky ReLU, whose derivative is $1$ on the active half-space and so does not shrink the signal.
- Batch normalisation, which keeps the pre-activations in a well-behaved range and the gates away from saturation.
- Residual connections, also known as skip connections, which add an identity path so that a Jacobian close to the identity is always available.

This is one of the main reasons ReLU and its variants replaced the sigmoid as the default hidden activation.

## Automatic Differentiation

So far we have derived the gradients of the loss with respect to the weights by hand. That is fine for toy examples but impractical for large networks, and we would rather not re-derive gradients every time we change the architecture. **Automatic differentiation** (autodiff) solves this by computing the gradients for us, automatically and exactly.

Autodiff rests on the same chain rule and on the fact that every operation in a network is built from elementary operations, such as addition and multiplication, and elementary functions, such as the exponential, sine and cosine. Each elementary operation has a known local derivative, so if we record the sequence of operations we can apply the chain rule mechanically. That record is the **computational graph**, a directed acyclic graph whose nodes are operations and whose edges carry data from one operation to the next. The linear list of intermediate operations is sometimes called the evaluation trace or Wengert list. A forward pass evaluates the nodes in order, and a backward pass walks the graph in reverse, attaching to each node the gradient of the loss with respect to its output and multiplying by the node's local Jacobian. This reverse traversal is exactly backpropagation, generalised from whole layers to arbitrary elementary operations.

{{< callout type="info" title="Backpropagation is dynamic programming" >}}
Reverse-mode autodiff is a [dynamic programming](/garden/cs/algd/dynamicProgramming/introduction) algorithm. The error signals $\boldsymbol{\delta}_k$ are the subproblems, and they overlap heavily, since the error signal of a node is needed by every parameter at that node and is also the ingredient for the error signals of the nodes feeding into it, through the recurrence $\boldsymbol{\delta}_k = \mathbf{J}_{k+1}^T \boldsymbol{\delta}_{k+1}$. Backpropagation solves each subproblem once and stores it for reuse, which is the memoisation at the heart of dynamic programming. The forward-pass cache of activations does the same job for the forward quantities. Differentiating each weight independently would instead recompute these shared subproblems over and over.
{{< /callout >}}

{{< figure
  src="/images/ml/computationalGraph.png"
  alt="A computational graph showing the forward operations on the left and the backward gradient operations on the right."
  caption="Computational graph of $z = \ell(w)$ with $w = \log(x_1 \cdot x_2)\cdot\sin(x_2)$. The forward pass (left, blue) records each elementary operation, and the backward pass (right, green) computes a local gradient for each one. Where a value feeds into more than one operation, the gradients coming back along the different paths are added together. Figure adapted from the PyTorch blog."
>}}

The local rules at each node explain a question that often comes up, why an addition node simply copies the incoming gradient.

- Addition $c = a + b$ has $\partial c / \partial a = \partial c / \partial b = 1$, so its local Jacobian is the identity and it copies the upstream gradient unchanged to each input.
- Multiplication $c = a \cdot b$ has $\partial c / \partial a = b$ and $\partial c / \partial b = a$, so it routes the upstream gradient to each input scaled by the other input, effectively swapping the operands.
- A value that fans out, feeding several downstream operations, accumulates gradients. The gradients arriving along the different paths are summed, which is the multivariate chain rule once more.

Modern libraries implement all of this for us. Theano pioneered symbolic graph-based autodiff and heavily influenced TensorFlow, PyTorch and JAX, while [PyTorch](https://pytorch.org/docs/stable/notes/autograd.html) and [JAX](https://docs.jax.dev/en/latest/automatic-differentiation.html) use imperative, dynamic autodiff that builds the graph as the code runs. In all of them autodiff gives efficient and correct gradients, lets differentiable simulations be plugged into optimizers, and frees us from deriving gradients by hand.
