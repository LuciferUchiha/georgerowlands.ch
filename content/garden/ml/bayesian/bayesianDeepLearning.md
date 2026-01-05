---
title: Bayesian Deep Learning
type: docs
weight: 4
math: true
---

In [Bayesian Linear Regression](/garden/ml/bayesian/bayesianlearning/), we saw how placing a prior distribution over weights and computing the full posterior allows us to quantify uncertainty in our predictions. The key insight was that when we combine a Gaussian prior with a Gaussian likelihood, the posterior is also Gaussian, giving us closed-form expressions for both the posterior distribution and the predictive distribution. We then extended this framework in [Variational Inference](/garden/ml/bayesian/variationalinference/) to handle intractable posteriors, such as those arising in Bayesian logistic regression, by approximating the true posterior with a simpler distribution from a tractable family.

However, both Bayesian linear regression and Bayesian logistic regression are **linear models**:,the function relating inputs to outputs is a linear combination of the input features (possibly after a fixed nonlinear transformation via basis functions). In practice, many real-world relationships are highly nonlinear, and the success of modern machine learning is largely attributable to the use of **neural networks**, which can learn complex nonlinear representations directly from data. This chapter explores how to bring the Bayesian perspective to neural networks, yielding **Bayesian Deep Learning and Bayesian Neural Networks (BNNs)** that combine the representational power of deep learning with principled uncertainty quantification.

## From Linear Models to Neural Networks

Recall from [Bayesian Linear Regression](/garden/ml/bayesian/bayesianlearning/) that a linear model assumes the relationship between input $x \in \mathbb{R}^d$ and output $y \in \mathbb{R}$ takes the form $y = w^\top x + \varepsilon$, where $w$ are the weights and $\varepsilon$ is Gaussian noise. While we can extend this to nonlinear features using basis functions $\phi(x)$, the choice of these features requires domain expertise and limits the model's flexibility.

Neural networks overcome this limitation by **learning** the feature representation. A neural network is a parametric function $f: \mathbb{R}^d \to \mathbb{R}^k$ that composes multiple layers of linear transformations with nonlinear activation functions:

$$
f(x; \theta) = \phi(W_L \cdot \phi(W_{L-1} \cdots \phi(W_1 x)))
$$

where $\theta := [W_1, \ldots, W_L]$ collects all the weight matrices $W_\ell \in \mathbb{R}^{n_\ell \times n_{\ell-1}}$ across the $L$ layers, and $\phi: \mathbb{R} \to \mathbb{R}$ is a nonlinear **activation function** applied element-wise. This nested structure is why neural networks are often called "deep" models. The network can be visualized as a computation graph where columns represent layers: the leftmost column is the input layer, the rightmost is the output layer, and the intermediate columns are hidden layers. The activations of layer $\ell$ are computed as:

$$
\nu^{(\ell)} := \phi(W_\ell \nu^{(\ell-1)})
$$

where $\nu^{(0)} = x$ is the input. The outputs of the final layer (before any probability normalization) are often called **logits**.

The choice of activation function $\phi$ is crucial. Two particularly common choices are the **hyperbolic tangent (Tanh)** defined as $\text{Tanh}(z) := \frac{\exp(z) - \exp(-z)}{\exp(z) + \exp(-z)} \in (-1, 1)$, which is a scaled and shifted variant of the sigmoid function, and the **rectified linear unit (ReLU)** defined as $\text{ReLU}(z) := \max\{z, 0\} \in [0, \infty)$, which is particularly popular because its gradients do not vanish as $z \to \pm\infty$, leading to more stable training.

The activation function must be nonlinear; otherwise, any composition of layers would still represent a linear function (since the composition of linear functions is linear). With nonlinear activations, neural networks become **universal function approximators**: a network with just a single hidden layer (of sufficient width) and a non-polynomial activation can approximate any continuous function to arbitrary accuracy. This remarkable result, known as the **universal approximation theorem**, provides theoretical justification for the representational power of neural networks. More advanced architectures such as convolutional neural networks (CNNs) for images and transformers for sequences build upon these foundations but are beyond our scope here.

{{< figure 
    src="/images/ml/neuralNetwork.png"
    caption="A fully connected neural network with an input layer, two hidden layers with ReLU activations, and an output layer. Each node in a layer is connected to every node in the subsequent layer via learnable weights."
    alt="A fully connected neural network with an input layer, two hidden layers with ReLU activations, and an output layer. Each node in a layer is connected to every node in the subsequent layer via learnable weights."
>}}

## Bayesian Neural Networks

Why do we need uncertainty quantification in neural networks? Standard neural networks produce point predictions without any indication of confidence. This can be problematic in safety-critical applications. Consider an autonomous vehicle that must detect pedestrians: a standard object detector might output a bounding box with high confidence even when the image is ambiguous or contains an object it has never seen before. Without uncertainty quantification, the system cannot distinguish between "I am confident this is a pedestrian" and "I have no idea what this is, but I will guess pedestrian."

Similarly, in medical diagnosis, a neural network might predict a disease with 95% confidence, but this confidence is often poorly calibrated and does not account for the model's epistemic uncertainty, i.e. its uncertainty arising from limited training data. A Bayesian approach allows the model to express "I am uncertain because I have not seen many examples like this," which is crucial for flagging cases that require human review. Beyond safety, uncertainty estimates enable **active learning** (querying the most informative data points), **out-of-distribution detection** (identifying inputs that differ from training data), and **exploration in reinforcement learning** (balancing exploitation of known rewards with exploration of uncertain states).

How can we perform probabilistic inference with neural networks? The strategy mirrors what we did for linear models: we place a **prior distribution** over the parameters and use Bayes' rule to compute the **posterior distribution** after observing data. The key difference is that instead of placing a prior over a weight vector $w \in \mathbb{R}^d$, we now place a prior over all the network parameters $\theta = [W_1, \ldots, W_L]$, which can number in the millions for modern networks.

We adopt the same Gaussian prior that worked well for linear models:

$$
\theta \sim \mathcal{N}(0, \sigma_p^2 I)
$$

This prior encodes our belief that, before seeing any data, the weights should be small (centered around zero) with variance controlled by $\sigma_p^2$. So you can think of each weight being drawn independently from a Gaussian distribution with mean 0 and variance $\sigma_p^2$. The isotropic structure (identity covariance) treats all weights as independent and identically distributed a priori. For the likelihood, we parameterize the data distribution using the neural network. In regression with Gaussian noise:

$$
y \mid x, \theta \sim \mathcal{N}(f(x; \theta), \sigma_n^2)
$$

The neural network $f(x; \theta)$ provides the mean of the Gaussian, while $\sigma_n^2$ is the noise variance. This is directly analogous to Bayesian linear regression, where we had $y \mid x, w \sim \mathcal{N}(w^\top x, \sigma_n^2)$, but now the linear function $w^\top x$ is replaced by the nonlinear neural network $f(x; \theta)$.

The crucial conceptual shift is that **the weights become random variables** rather than fixed parameters to be estimated. Instead of finding a single "best" weight configuration, we maintain a distribution over all possible weight configurations. Each sample from this distribution corresponds to a different neural network, and our predictions integrate over all these possible networks, weighted by their posterior probability.

What does this mean for predictions? Given a new input $x_*$, the predictive distribution marginalizes over the posterior:

$$
p(y_* \mid x_*, \mathcal{D}) = \int p(y_* \mid x_*, \theta) p(\theta \mid \mathcal{D}) \, d\theta
$$

Rather than obtaining a single predicted value, we get a **distribution over outputs** that captures our uncertainty. This uncertainty has two sources: the inherent noise in the data (aleatoric uncertainty) and our uncertainty about which network is correct (epistemic uncertainty).

{{< figure 
    src="/images/ml/bayesNeuralNetwork.png"
    caption="A Bayesian neural network places probability distributions over the weights rather than point estimates. Each weight is represented by a distribution (e.g., Gaussian), and predictions integrate over all possible weight configurations."
    alt="A Bayesian neural network places probability distributions over the weights rather than point estimates. Each weight is represented by a distribution (e.g., Gaussian), and predictions integrate over all possible weight configurations."
>}}

### MAP Estimation

Before tackling the full Bayesian treatment, let us first consider **Maximum A Posteriori (MAP) estimation** for neural networks. This gives us a single point estimate of the weights that maximizes the posterior probability, serving as a stepping stone toward understanding the full posterior. Recall from [Bayesian Linear Regression](/garden/ml/bayesian/bayesianlearning/) that the MAP estimate maximizes the posterior, which is proportional to the likelihood times the prior. So if we take the logarithm, the MAP estimate is:

$$
\hat{\theta}_{\text{MAP}} = \arg\max_\theta \left[ \log p(\theta) + \sum_{i=1}^n \log p(y_i \mid x_i, \theta) \right]
$$

For our Gaussian likelihood and prior, let us derive the explicit form of this objective. The log-likelihood for a single data point under the Gaussian noise model is:

$$
\log p(y_i \mid x_i, \theta) = \log \mathcal{N}(y_i; f(x_i; \theta), \sigma_n^2) = -\frac{1}{2}\log(2\pi\sigma_n^2) - \frac{(y_i - f(x_i; \theta))^2}{2\sigma_n^2}
$$

The first term is constant with respect to $\theta$, so maximizing the log-likelihood is equivalent to minimizing the squared error $(y_i - f(x_i; \theta))^2$. This explains why minimizing mean squared error corresponds to maximum likelihood estimation under Gaussian noise. Similarly, for the isotropic Gaussian prior:

$$
\log p(\theta) = -\frac{1}{2}\log\det(2\pi\sigma_p^2 I) - \frac{\|\theta\|_2^2}{2\sigma_p^2} \propto -\frac{\|\theta\|_2^2}{2\sigma_p^2}
$$

Combining these and converting maximization to minimization of the negative:

$$
\hat{\theta}_{\text{MAP}} = \arg\min_\theta \left[ \frac{1}{2\sigma_p^2}\|\theta\|_2^2 + \frac{1}{2\sigma_n^2}\sum_{i=1}^n (y_i - f(x_i; \theta))^2 \right]
$$

This is precisely **L2-regularized** (or weight-decay regularized) neural network training! The squared error term encourages the network to fit the data, while the $\|\theta\|_2^2$ term penalizes large weights. The ratio $\lambda := \sigma_n^2/\sigma_p^2$ controls the regularization strength: larger noise variance $\sigma_n^2$ or smaller prior variance $\sigma_p^2$ leads to stronger regularization.

This connection provides a Bayesian interpretation of the common practice of weight decay: **training a neural network with L2 regularization is equivalent to computing the MAP estimate under a Gaussian prior**. The regularization coefficient $\lambda$ directly corresponds to the ratio of noise variance to prior variance. To optimize this objective, we use gradient-based methods. The gradient update rule takes the form:

$$
\theta \leftarrow \theta(1 - \lambda\eta_t) + \eta_t \sum_{i=1}^n \nabla_\theta \log p(y_i \mid x_i, \theta)
$$

where $\eta_t$ is the learning rate at step $t$ and $\lambda = 1/\sigma_p^2$. The factor $(1 - \lambda\eta_t)$ multiplying the current weights implements the "decay" toward zero, while the gradient term pushes the weights toward better data fit. The gradients of the likelihood can be computed efficiently using **backpropagation**, which exploits the compositional structure of neural networks to compute gradients in time linear in the network size.

### Heteroscedastic Noise

The model we have considered so far uses a fixed noise variance $\sigma_n^2$ that is the same for all inputs. This is called **homoscedastic noise** and it is the same assumption we made in Bayesian linear regression. However, in many real-world settings, the noise level varies across the input domain. For example, measurements of physical quantities often have noise that scales with the magnitude of the measurement: measuring large forces might involve larger absolute measurement errors than measuring small forces due to for example vibration. This input-dependent noise is called **heteroscedastic noise**.

{{< figure 
    src="/images/ml/scedasticy.gif"
    caption="Left: Homoscedastic noise where the variance is constant across all input values. Right: Heteroscedastic noise where the variance changes depending on the input, with some regions exhibiting higher noise than others."
    alt="Left: Homoscedastic noise where the variance is constant across all input values. Right: Heteroscedastic noise where the variance changes depending on the input, with some regions exhibiting higher noise than others."
>}}

Neural networks provide a natural way to model heteroscedastic noise: instead of outputting just a mean prediction, the network can output **both a mean and a variance** for each input. We use a neural network with two outputs $f_1$ and $f_2$ and define the likelihood as:

$$
y \mid x, \theta \sim \mathcal{N}(\mu(x; \theta), \sigma^2(x; \theta))
$$

where the mean and variance are parameterized as:

$$
\mu(x; \theta) := f_1(x; \theta), \quad \sigma^2(x; \theta) := \exp(f_2(x; \theta))
$$

The exponential transformation ensures that the variance is always positive, regardless of the value of $f_2$. This is a common trick: rather than constraining the network output directly, we let it output an unconstrained value and apply a suitable transformation (here, exponentiation) to enforce the constraint.

{{< figure 
    src="/images/ml/bayesNeuralNetworkHetero.webp"
    caption="A neural network architecture for heteroscedastic regression. The network has two output heads: one predicting the mean and another predicting the log-variance. The exponential transformation ensures the variance remains positive."
    alt="A neural network architecture for heteroscedastic regression. The network has two output heads: one predicting the mean and another predicting the log-variance. The exponential transformation ensures the variance remains positive."
>}}

How does this change the MAP objective? The log-likelihood for a single data point now includes terms that depend on the variance:

$$
\begin{align*}
\log p(y_i \mid x_i, \theta) &= \log \mathcal{N}(y_i; \mu(x_i; \theta), \sigma^2(x_i; \theta)) \\
&= -\frac{1}{2}\log(2\pi) - \frac{1}{2}\log\sigma^2(x_i; \theta) - \frac{(y_i - \mu(x_i; \theta))^2}{2\sigma^2(x_i; \theta)}
\end{align*}
$$

Notice that unlike the homoscedastic case, the normalizing constant $-\frac{1}{2}\log\sigma^2(x_i; \theta)$ now depends on the parameters through $\sigma^2(x_i; \theta)$ and cannot be ignored. Dropping the constant $-\frac{1}{2}\log(2\pi)$ and negating to obtain the loss, the MAP objective becomes:

$$
\hat{\theta}_{\text{MAP}} = \arg\min_\theta \left[ \frac{1}{2\sigma_p^2}\|\theta\|_2^2 + \frac{1}{2}\sum_{i=1}^n \left( \log\sigma^2(x_i; \theta) + \frac{(y_i - \mu(x_i; \theta))^2}{\sigma^2(x_i; \theta)} \right) \right]
$$

This loss function has an elegant interpretation. The network can explain a data point $(x_i, y_i)$ in two ways: either by making an accurate prediction $\mu(x_i; \theta) \approx y_i$, which makes the squared error term small, or by predicting high variance $\sigma^2(x_i; \theta)$, which reduces the contribution of the squared error by dividing by a large number. However, the $\log\sigma^2$ term **penalizes** large variances, preventing the network from simply attributing all errors to noise.

This mechanism allows the network to "learn its own aleatoric uncertainty." In regions where the data is inherently noisy and no single function can fit all points well, the network learns to predict high variance. In regions where the data follows a clear pattern, the network predicts low variance and focuses on making accurate mean predictions.

However, it is important to remember that MAP estimation still produces a **point estimate** of the weights. While we can now capture input-dependent aleatoric uncertainty through $\sigma^2(x; \theta)$, we do not capture **epistemic uncertainty** about the model itself. To quantify our uncertainty about which network is correct, we need the full posterior distribution over weights.

## Bayes by Backprop

The fundamental challenge of Bayesian neural networks is that **exact inference is intractable**. Unlike Bayesian linear regression where the Gaussian-Gaussian conjugacy gives us a closed-form posterior, the nonlinearity of neural networks breaks this conjugacy. Even with Gaussian priors and likelihoods, the posterior over weights is not Gaussian and cannot be computed analytically.

To understand why, recall that the posterior is proportional to the product of likelihood and prior:

$$
p(\theta \mid \mathcal{D}) \propto p(\mathcal{D} \mid \theta) p(\theta) = \left[\prod_{i=1}^n \mathcal{N}(y_i; f(x_i; \theta), \sigma_n^2)\right] \cdot \mathcal{N}(\theta; 0, \sigma_p^2 I)
$$

Because the neural network function $f(x; \theta)$ is a highly nonlinear function of $\theta$, the product of Gaussian likelihoods is not Gaussian in $\theta$. The resulting posterior can be multimodal (having multiple local maxima), asymmetric, and highly complex in the high-dimensional weight space. Computing the normalizing constant $p(\mathcal{D}) = \int p(\mathcal{D} \mid \theta) p(\theta) d\theta$ involves an intractable integral over potentially millions of dimensions.

This intractability motivates the use of **approximate inference** methods. We have already encountered two such methods in [Variational Inference](/garden/ml/bayesian/variationalinference/): the Laplace approximation, which fits a Gaussian around the MAP estimate using local curvature information, and variational inference, which finds the best approximation within a tractable family by maximizing the evidence lower bound (ELBO). Both methods are applicable to neural networks, and their gradients can be computed using backpropagation.

**Black-box stochastic variational inference**, when applied to neural networks, is often called **Bayes by Backprop**. The key idea is to approximate the intractable posterior $p(\theta \mid \mathcal{D})$ with a simpler variational distribution $q_\lambda(\theta)$ from a tractable family, where $\lambda$ are variational parameters that we optimize.

A natural choice for the variational family is the family of **independent (diagonal) Gaussians**:

$$
q_\lambda(\theta) = \prod_{j=1}^d \mathcal{N}(\theta_j; \mu_j, \sigma_j^2)
$$

where $d$ is the total number of weights in the network. This family is parameterized by $\lambda = (\mu_1, \ldots, \mu_d, \sigma_1, \ldots, \sigma_d)$, giving us $2d$ variational parameters. The diagonal structure means each weight is treated as independent in the variational approximation, which is a significant simplification but keeps the number of parameters manageable.

Recall from [Variational Inference](/garden/ml/bayesian/variationalinference/) that we maximize the **Evidence Lower Bound (ELBO)**:

$$
\mathcal{L}(q_\lambda; \mathcal{D}) = \mathbb{E}_{\theta \sim q_\lambda}[\log p(\mathcal{D} \mid \theta)] - \text{KL}(q_\lambda \| p(\theta))
$$

The first term encourages the variational distribution to place mass on weight configurations that explain the data well. The second term regularizes the variational distribution to stay close to the prior, preventing overfitting. For Gaussian distributions, the KL divergence has a closed form. When $q_\lambda = \mathcal{N}(\mu, \text{diag}\{\sigma_j^2\})$ and $p(\theta) = \mathcal{N}(0, \sigma_p^2 I)$:

$$
\text{KL}(q_\lambda \| p(\theta)) = \frac{1}{2}\sum_{j=1}^d \left( \frac{\sigma_j^2}{\sigma_p^2} + \frac{\mu_j^2}{\sigma_p^2} - 1 - \log\frac{\sigma_j^2}{\sigma_p^2} \right)
$$

The expected log-likelihood term is more challenging because it involves the nonlinear neural network. We cannot compute $\mathbb{E}_{\theta \sim q_\lambda}[\log p(\mathcal{D} \mid \theta)]$ in closed form, so we resort to Monte Carlo estimation. Moreover, to compute gradients with respect to $\lambda$, we use the **reparameterization trick**: instead of sampling $\theta \sim q_\lambda$ directly, we write:

$$
\theta = \mu + \text{diag}(\sigma_1, \ldots, \sigma_d) \cdot \varepsilon, \quad \varepsilon \sim \mathcal{N}(0, I)
$$

This expresses $\theta$ as a deterministic function of $\lambda = (\mu, \sigma)$ and a parameter-free random variable $\varepsilon$. The gradient can now pass through the sampling operation:

$$
\nabla_\lambda \mathbb{E}_{\theta \sim q_\lambda}[\log p(\mathcal{D} \mid \theta)] = \mathbb{E}_{\varepsilon \sim \mathcal{N}(0,I)}[\nabla_\lambda \log p(\mathcal{D} \mid \theta)]_{\theta = \mu + \text{diag}(\sigma) \cdot \varepsilon}
$$

In practice, we approximate this expectation with a single sample (or a small number of samples) per gradient step, combined with mini-batching over data points. The resulting algorithm is simple: at each iteration, sample noise $\varepsilon \sim \mathcal{N}(0, I)$, compute the sampled weights $\theta = \mu + \sigma \odot \varepsilon$ (where $\odot$ denotes element-wise multiplication), evaluate the loss on a mini-batch, and backpropagate to update $\mu$ and $\sigma$.

### Making Predictions with Bayesian Neural Networks

Once we have trained a variational approximation $q_\lambda(\theta)$ to the posterior, how do we make predictions? The Bayesian predictive distribution marginalizes over the posterior:

$$
p(y_* \mid x_*, \mathcal{D}) = \int p(y_* \mid x_*, \theta) p(\theta \mid \mathcal{D}) \, d\theta
$$

Since the true posterior is intractable, we approximate it with the variational posterior:

$$
p(y_* \mid x_*, \mathcal{D}) \approx \int p(y_* \mid x_*, \theta) q_\lambda(\theta) \, d\theta = \mathbb{E}_{\theta \sim q_\lambda}[p(y_* \mid x_*, \theta)]
$$

This expectation is still intractable due to the nonlinear neural network, but we can approximate it using Monte Carlo sampling. We draw $m$ weight samples from the variational posterior and average the predictions:

$$
p(y_* \mid x_*, \mathcal{D}) \approx \frac{1}{m}\sum_{j=1}^m p(y_* \mid x_*, \theta^{(j)}), \quad \theta^{(j)} \stackrel{\text{iid}}{\sim} q_\lambda
$$

For a Gaussian likelihood with heteroscedastic noise, each sample $\theta^{(j)}$ gives us a Gaussian predictive distribution $\mathcal{N}(y_*; \mu(x_*; \theta^{(j)}), \sigma^2(x_*; \theta^{(j)}))$. The approximate predictive distribution is therefore a **mixture of Gaussians**:

$$
p(y_* \mid x_*, \mathcal{D}) \approx \frac{1}{m}\sum_{j=1}^m \mathcal{N}(y_*; \mu(x_*; \theta^{(j)}), \sigma^2(x_*; \theta^{(j)}))
$$

This mixture captures both sources of uncertainty. Different weight samples $\theta^{(j)}$ may predict different means $\mu(x_*; \theta^{(j)})$, reflecting epistemic uncertainty about which function is correct. Each component also has its own variance $\sigma^2(x_*; \theta^{(j)})$, reflecting aleatoric uncertainty.

Intuitively, variational inference in Bayesian neural networks can be understood as **averaging the predictions of multiple neural networks** drawn from the variational posterior. Each sampled network represents one plausible explanation of the data, and the final prediction combines all these explanations weighted by their posterior probability.

{{< figure 
    src="/images/ml/gaussianMixtureModel.png"
    caption="The approximate predictive distribution as a mixture of Gaussians. Each sampled weight configuration produces a different Gaussian prediction (colored curves), and the mixture (black curve) captures uncertainty from both aleatoric noise (width of individual Gaussians) and epistemic uncertainty (spread of means)."
    alt="The approximate predictive distribution as a mixture of Gaussians. Each sampled weight configuration produces a different Gaussian prediction (colored curves), and the mixture (black curve) captures uncertainty from both aleatoric noise (width of individual Gaussians) and epistemic uncertainty (spread of means)."
>}}

### Uncertainty

The Monte Carlo samples allow us to estimate both the mean and variance of our predictions, and importantly, to **decompose the total uncertainty** into its aleatoric and epistemic components.

Using the Monte Carlo samples $\theta^{(j)}$, we can estimate the predictive mean:

$$
\mathbb{E}[y_* \mid x_*, \mathcal{D}] \approx \frac{1}{m}\sum_{j=1}^m \mu(x_*; \theta^{(j)}) =: \bar{\mu}(x_*)
$$

For the predictive variance, we apply the **law of total variance** (see [Bayesian Linear Regression](/garden/ml/bayesian/bayesianlearning/)):

$$
\text{Var}[y_* \mid x_*, \mathcal{D}] = \underbrace{\mathbb{E}_\theta[\text{Var}[y_* \mid x_*, \theta]]}_{\text{Aleatoric}} + \underbrace{\text{Var}_\theta[\mathbb{E}[y_* \mid x_*, \theta]]}_{\text{Epistemic}}
$$

The first term is the **expected variance** of the predictive distribution given the weights, representing the inherent noise in the data that cannot be reduced even with infinite data. The second term is the **variance of the expected prediction** across different weight configurations, representing our uncertainty about the model that decreases as we observe more data.

We can estimate these using our Monte Carlo samples:

$$
\text{Var}[y_* \mid x_*, \mathcal{D}] \approx \underbrace{\frac{1}{m}\sum_{j=1}^m \sigma^2(x_*; \theta^{(j)})}_{\text{Aleatoric}} + \underbrace{\frac{1}{m-1}\sum_{j=1}^m (\mu(x_*; \theta^{(j)}) - \bar{\mu}(x_*))^2}_{\text{Epistemic}}
$$

The first term uses the sample mean of the predicted variances. The second term uses the sample variance of the predicted means (with the Bessel correction $m-1$ for an unbiased estimate).

This decomposition is valuable for understanding model behavior. In regions where the model has seen little data, the epistemic uncertainty should be high. In regions with inherently noisy data, the aleatoric uncertainty should be high. A well-calibrated Bayesian neural network distinguishes between these two types of uncertainty, which is crucial for applications like active learning (where we want to query points with high epistemic uncertainty) and safety-critical systems (where we need to know when the model is uncertain about its predictions).

{{< figure 
    src="/images/ml/bayesNeuralNetworkUncertainty.png"
    caption="Uncertainty decomposition in a Bayesian neural network regression. The total predictive uncertainty (shaded region) combines aleatoric uncertainty (inherent noise in the data) and epistemic uncertainty (model uncertainty due to limited data). Epistemic uncertainty is typically higher in regions far from training data."
    alt="Uncertainty decomposition in a Bayesian neural network regression. The total predictive uncertainty (shaded region) combines aleatoric uncertainty (inherent noise in the data) and epistemic uncertainty (model uncertainty due to limited data). Epistemic uncertainty is typically higher in regions far from training data."
>}}

This uncertainty decomposition extends naturally to more complex tasks. In **Bayesian object detection** (e.g., Bayesian R-CNN), the network outputs not just bounding box coordinates but also uncertainty estimates for each coordinate. This allows the model to express "I am confident about the left edge of this object but uncertain about the right edge" — information that is valuable for downstream decision-making. For instance, an autonomous vehicle might slow down when object boundaries are uncertain, or a robotic manipulation system might request a different camera angle when grasp points are ambiguous.

## Dropout as Approximate Bayesian Inference

An elegant connection exists between **dropout regularization**, a widely used technique for preventing overfitting in neural networks, and approximate Bayesian inference. This connection,allows us to interpret standard dropout-trained networks as performing approximate variational inference. Dropout works by randomly "dropping out" (setting to zero) units in the network during training. At each forward pass, each unit is independently set to zero with probability $p$ (the dropout rate). This can be seen as training an ensemble of subnetworks that share weights, which improves generalization by preventing co-adaptation of features. A closely related technique is **dropconnect**, which randomly drops individual weights (edges in the computation graph) rather than entire units (vertices). We will focus on dropconnect, but the analysis extends to standard dropout as well.

{{< figure 
    src="/images/ml/dropout.webp"
    caption="Dropout regularization. Left: A standard fully connected network. Right: The same network with dropout applied, where randomly selected units (crossed out) are set to zero during training. This prevents co-adaptation and can be interpreted as training an ensemble of subnetworks."
    alt="Dropout regularization. Left: A standard fully connected network. Right: The same network with dropout applied, where randomly selected units (crossed out) are set to zero during training. This prevents co-adaptation and can be interpreted as training an ensemble of subnetworks."
>}}

The key insight is that dropconnect can be interpreted as variational inference with a specific variational family. Consider a variational distribution over the weights of the form:

$$
q(\theta \mid \lambda) = \prod_{j=1}^d q_j(\theta_j \mid \lambda_j)
$$

where each weight independently follows a mixture distribution:

$$
q_j(\theta_j \mid \lambda_j) = p \cdot \delta_0(\theta_j) + (1-p) \cdot \delta_{\lambda_j}(\theta_j)
$$

Here, $\delta_\alpha$ is the Dirac delta function placing all mass at $\alpha$. In words, each weight is either zero (with probability $p$) or takes value $\lambda_j$ (with probability $1-p$). The variational parameters $\lambda = (\lambda_1, \ldots, \lambda_d)$ correspond to the "original" weights of the network before dropout.

Sampling from this variational distribution is equivalent to applying a dropout mask: we sample a binary vector $z$ with entries $z_j \sim \text{Bernoulli}(1-p)$ and compute $\theta = z \odot \lambda$, where $\odot$ denotes element-wise multiplication. Each sample corresponds to one of $2^d$ possible subnetworks.

The KL divergence term in the ELBO is not tractable for the Dirac mixture, but can be approximated using a Gaussian mixture:

$$
q_j(\theta_j \mid \lambda_j) = p \cdot \mathcal{N}(\theta_j; 0, 1) + (1-p) \cdot \mathcal{N}(\theta_j; \lambda_j, 1)
$$

It can be shown that for large networks, this KL divergence is approximately:

$$
\text{KL}(q_\lambda \| p(\theta)) \approx \frac{p}{2}\|\lambda\|_2^2
$$

This means that maximizing the ELBO reduces to minimizing:

$$
-\frac{1}{m}\sum_{i=1}^m \log p(y_{1:n} \mid x_{1:n}, \theta^{(i)}) + \frac{p}{2}\|\lambda\|_2^2
$$

where $\theta^{(i)} = z^{(i)} \odot \lambda$ with random dropout masks $z^{(i)}$. This is exactly the standard L2-regularized loss function of a neural network trained with dropconnect!

The crucial implication is that for this interpretation to be valid, **we must also apply dropout during inference**, not just training. The approximate predictive distribution becomes:

$$
p(y_* \mid x_*, \mathcal{D}) \approx \frac{1}{m}\sum_{i=1}^m p(y_* \mid x_*, \theta^{(i)}), \quad \theta^{(i)} \sim q_\lambda
$$

where each sample corresponds to a different dropout mask. This is called **Monte Carlo dropout**: we make multiple forward passes with different random dropout masks and average the predictions. This provides uncertainty estimates from an ordinary dropout-trained network without any additional training, making it an attractive practical approach.

A potential issue with dropout is that for typical dropout probabilities, the masks $z^{(i)}$ overlap significantly, making the predictions $p(y_* \mid x_*, \theta^{(i)})$ highly correlated. This can lead to underestimation of epistemic uncertainty. Techniques like **masksembles** address this by using a fixed set of pre-defined masks with controlled overlap.

{{< figure 
    src="/images/ml/monteCarloDropout.png"
    caption="Monte Carlo dropout for uncertainty estimation. During inference, we perform multiple forward passes with different random dropout masks and average the predictions. The variance across predictions provides an estimate of epistemic uncertainty."
    alt="Monte Carlo dropout for uncertainty estimation. During inference, we perform multiple forward passes with different random dropout masks and average the predictions. The variance across predictions provides an estimate of epistemic uncertainty."
>}}

## Stochastic Weight Averaging

Another approach to approximate Bayesian inference leverages the trajectory of stochastic gradient descent (SGD) during training. The key observation is that SGD does not converge to a single point but rather oscillates around a region of low loss. We can use this trajectory to estimate a distribution over weights.

**Stochastic Weight Averaging (SWA)** simply takes the average of the weights visited during training:

$$
\mu := \frac{1}{T}\sum_{t=1}^T \theta^{(t)}
$$

where $\theta^{(t)}$ is the weight configuration at iteration $t$. This average often generalizes better than the final iterate because it corresponds to the center of the region explored by SGD.

**Stochastic Weight Averaging - Gaussian (SWAG)** extends this idea by maintaining not just the mean but also a Gaussian approximation to the distribution over weights:

$$
\theta \sim \mathcal{N}(\mu, \Sigma)
$$

where both the mean and covariance are estimated from the trajectory:

$$
\mu := \frac{1}{T}\sum_{t=1}^T \theta^{(t)}, \quad \Sigma := \frac{1}{T-1}\sum_{t=1}^T (\theta^{(t)} - \mu)(\theta^{(t)} - \mu)^\top
$$

In practice, we cannot store the full covariance matrix for large networks. SWAG addresses this by using a low-rank approximation or by maintaining only the diagonal. The estimates can be computed efficiently using **running averages**:

$$
\mu \leftarrow \frac{1}{T+1}(T\mu + \theta), \quad A \leftarrow \frac{1}{T+1}(TA + \theta\theta^\top)
$$

where upon observing a new weight sample $\theta$, we update the running estimates. The covariance can be recovered from these moments using $\Sigma = \frac{T}{T-1}(A - \mu\mu^\top)$.

To make predictions, we sample weights from the learned Gaussian and average the neural network predictions, exactly as in variational inference. The appeal of SWAG is that it requires minimal changes to standard training and leverages the exploration already performed by SGD.

Interestingly, this approach works even without injecting additional noise during training (as would be required by methods like Stochastic Gradient Langevin Dynamics). The stochasticity of mini-batch gradients provides enough exploration to estimate a meaningful distribution over weights.

{{< figure 
    src="/images/ml/swag.png"
    caption="SWAG estimates a Gaussian distribution over weights from the SGD trajectory. The x-axis shows weight values, and the y-axis shows the (approximate) posterior density. The trajectory samples (dots) cluster around the mode of the posterior, and SWAG fits a Gaussian to capture this distribution."
    alt="SWAG estimates a Gaussian distribution over weights from the SGD trajectory. The x-axis shows weight values, and the y-axis shows the (approximate) posterior density. The trajectory samples (dots) cluster around the mode of the posterior, and SWAG fits a Gaussian to capture this distribution."
>}}

The visualization shows how SWAG leverages the SGD trajectory to approximate the posterior. The key insight is that SGD naturally explores the region around the posterior mode, with the step size and mini-batch noise determining how broadly it explores. SWAG fits a Gaussian to these trajectory samples, placing its mean at the center of the explored region (approximately the posterior mode) and its covariance capturing the spread of the trajectory. When we sample from this Gaussian for prediction, we draw weights from the high-density region of the approximate posterior, where the model fits the data well while remaining plausibly close to different good solutions. This is precisely what Bayesian prediction requires: averaging over plausible weight configurations weighted by their posterior probability.

## Probabilistic Ensembles

A conceptually simple approach to approximate Bayesian inference is to train **multiple independent neural networks** and combine their predictions. This is known as **ensembling** or **probabilistic ensembles**.

The classical approach uses **bootstrap sampling**: we create $m$ training sets by sampling uniformly from the original data with replacement, then train one network on each bootstrap sample. The different training sets induce diversity among the networks, and the predictive distribution is approximated by averaging:

$$
p(y_* \mid x_*, \mathcal{D}) \approx \frac{1}{m}\sum_{i=1}^m p(y_* \mid x_*, \theta^{(i)})
$$

where $\theta^{(i)}$ is the MAP estimate obtained from the $i$-th training set. In practice, for large neural networks where finding the global optimum is infeasible, it is common to simply train $m$ networks on the full dataset with **different random initializations** and **different random shufflings** of the training data. The stochasticity of SGD combined with different starting points typically produces sufficient diversity.

Ensembles have connections to other approximate inference methods. They can be viewed as a form of "extreme" dropout where each model uses a completely non-overlapping subset of parameters (i.e., each ensemble member is a separate network). Unlike dropout, ensembles do not suffer from correlated predictions because the models are trained independently. The ensemble approach is **not equivalent to Monte Carlo sampling** from the true posterior, although the formulas look similar. The key difference is that ensembles sample from the empirical distribution of MAP estimates under different training conditions, not from the true posterior. Nevertheless, ensembles are remarkably effective in practice and remain one of the most reliable methods for uncertainty quantification in deep learning. Ensembling can also be combined with other approximate inference techniques. For example, one could train an ensemble of networks, each of which uses dropout or has its own Laplace approximation, yielding a **mixture of Gaussians** as the posterior approximation.

## Calibration

A crucial property of probabilistic models is **calibration**: the model's confidence should match its accuracy. A well-calibrated model that predicts an event with 80% probability should be correct about 80% of the time. This section discusses how to measure and improve calibration in Bayesian neural networks.

For classification, we group predictions into $M$ bins based on the predicted probability for the positive class. Within each bin, we compare the average predicted probability (**confidence**) to the actual frequency of positive outcomes (**accuracy**):

$$
\text{conf}(B_m) := \frac{1}{|B_m|}\sum_{i \in B_m} p(y_i = 1 \mid x_i), \quad \text{freq}(B_m) := \frac{1}{|B_m|}\sum_{i \in B_m} \mathbb{1}\{y_i = 1\}
$$

A perfectly calibrated model has $\text{conf}(B_m) = \text{freq}(B_m)$ for all bins, appearing as a diagonal line on the reliability diagram. Deviations from the diagonal indicate miscalibration: if the bars are above the diagonal, the model is **underconfident**; if below, it is **overconfident**.

Two common metrics summarize calibration quality. The **Expected Calibration Error (ECE)** is the weighted average deviation from perfect calibration:

$$
\varepsilon_{\text{ECE}} := \sum_{m=1}^M \frac{|B_m|}{n} \left| \text{freq}(B_m) - \text{conf}(B_m) \right|
$$

The **Maximum Calibration Error (MCE)** is the worst-case deviation across all bins:

$$
\varepsilon_{\text{MCE}} := \max_{m \in [M]} \left| \text{freq}(B_m) - \text{conf}(B_m) \right|
$$

{{< figure 
    src="/images/ml/reliabilityDiagram.png"
    caption="Reliability diagram for assessing model calibration. The x-axis shows predicted confidence (binned), and the y-axis shows the actual accuracy within each bin. A perfectly calibrated model lies on the diagonal (dashed line). Bars above the diagonal indicate underconfidence; bars below indicate overconfidence. The gap between each bar and the diagonal contributes to the calibration error."
    alt="Reliability diagram for assessing model calibration. The x-axis shows predicted confidence (binned), and the y-axis shows the actual accuracy within each bin. A perfectly calibrated model lies on the diagonal (dashed line). Bars above the diagonal indicate underconfidence; bars below indicate overconfidence. The gap between each bar and the diagonal contributes to the calibration error."
>}}