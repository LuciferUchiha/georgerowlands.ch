---
title: Bayesian Learning
type: docs
weight: 1
---

/garden/maths/probabilitystatistics/multivariategaussian/ might be useful and http://localhost:1313/garden/maths/probabilitystatistics/estimators/#maximum-likelihood-estimation-mle

---
title: Bayesian Learning
type: docs
weight: 1
---

<!--
Rough Outline:
probabilistic inference

bayesian inference with priors and posteriors. prior is prob of model?, likelihood is prob of data given model? posterior is prob of model given data?

{{< figure 
    src="/images/ml/bayesPriorLikelihoodPosterior.webp"
    caption="Bayes' Theorem relates the prior, likelihood, and posterior distributions."
    alt="Bayes' Theorem relates the prior, likelihood, and posterior distributions."
>}}

conjugate priors and 

Reminder of MLE and then introduce MAP as MLE with prior as regularization

bayesian linear regression

{{< figure 
    src="/images/ml/bayesLinearRegression.gif"
    caption="Bayesian Linear Regression with on the right the samples from the posterior distribution over functions as new data points are observed. On the left, the uncertainty in the weights is shown."
    alt="Bayesian Linear Regression with on the right the samples from the posterior distribution over functions as new data points are observed. On the left, the uncertainty in the weights is shown."
>}}

continos model assumptions?
seen linear regresion using OLS (not connected to probabilistic view yet)

bayesian graphical model introduction and then show the graph for bayesian linear regression

MLE results in linear regression?

OLSE gauss markov theorem?

Then use MAP with gaussian prior to get ridge regression show full derivation of posterior and predictive posterior distribution

Making predictions in BLR for test points using poseterior and f^*

Ridge regression can be viewed as approximating the full posterior by (placing all mass on) its mode???

ridge predicts using MAP estimate of weights, Bayesian linear regression predicts using full posterior predictive distribution i.e average over all possible weights weighted by their posterior probability? This part is key to

somewhere epistemic vs aleatoric uncertainty

aleatoric can't be reduced with more data (inherent noise in data) epistemic can be reduced with more data (uncertainty in model parameters) are additive in predictive uncertainty?

{{< figure 
    src="/images/ml/bayesAleatoricEpistemicUncertainty.png"
    caption="Predictive distribution in Bayesian Linear Regression showing uncertainty."
    alt="Predictive distribution in Bayesian Linear Regression showing uncertainty."
>}}

lasso regression as laplace prior???

How do we choose hyperparameters like the prior variance or noise variance? one way would be to use cross validation but in bayesian approach we can also do evidence maximization / type II MLE where we treat these hyperparameters as parameters to be estimated by maximizing the marginal likelihood (evidence) of the data integrating out the model parameters. Define evidence and show how to compute it for bayesian linear regression. then show how to optimize it w.r.t hyperparameters?

recursive bayesian learning(Online Bayesian Linear Regression) this is where we update our posterior as new data comes in rather than recomputing from scratch each time what performance benefits does it give us?

If d is large, computing the inverse every round is very expensive. Can you use the recursive structure
you found in the previous question to reduce the computational complexity of every round to O(d^2)? Shermann-Morrison formula or woodbury identity?

Bayesian decision theory, pick the action that minimizes the expected cost for sqaured loss this is the mean of the predictive posterior what about for asymmetric loss functions with over and underestimation costs?

Can apply linear method (like BLR) on nonlinearly
transformed data. However, computational cost increases
with dimensionality of the feature space!

kernel trick Express problem s.t. it only depends on inner products
§ Replace inner products by kernels

weight vs function space view?

prediction in weight space vs function space? Were given X, y and want to predict y^* for new x^*

infinite domains resulting in gaussian processes

so much to cover here!
-->

/garden/maths/probabilitystatistics/multivariategaussian/ might be useful and http://localhost:1313/garden/maths/probabilitystatistics/estimators/#maximum-likelihood-estimation-mle

## Probabilistic Inference

Probabilistic inference is the process of updating our beliefs about uncertain quantities (such as model parameters or future observations) in light of new evidence (data). Unlike logical inference, which deals with certainties ($A \implies B$), probabilistic inference deals with plausibilities.

The core rule of probabilistic inference is **Bayes' Rule**:

$$
p(\theta \mid \mathcal{D}) = \frac{p(\mathcal{D} \mid \theta) p(\theta)}{p(\mathcal{D})}
$$

where:
- $\theta$ represents the parameters or hypotheses we are interested in.
- $\mathcal{D}$ represents the observed data.
- **$p(\theta)$ is the Prior**: Our initial belief about $\theta$ before seeing any data.
- **$p(\mathcal{D} \mid \theta)$ is the Likelihood**: How probable the data $\mathcal{D}$ is, assuming $\theta$ is true.
- **$p(\theta \mid \mathcal{D})$ is the Posterior**: Our updated belief about $\theta$ after observing $\mathcal{D}$.
- **$p(\mathcal{D})$ is the Marginal Likelihood (or Evidence)**: The probability of the data under all possible hypotheses. It acts as a normalizing constant ensuring the posterior integrates to 1.

$$
p(\mathcal{D}) = \int p(\mathcal{D} \mid \theta) p(\theta) d\theta
$$

{{< figure 
    src="/images/ml/bayesPriorLikelihoodPosterior.webp"
    caption="Bayes' Theorem relates the prior, likelihood, and posterior distributions."
    alt="Bayes' Theorem relates the prior, likelihood, and posterior distributions."
>}}

### Conjugate Priors

If the posterior distribution $p(\theta \mid \mathcal{D})$ is in the same family as the prior probability distribution $p(\theta)$, the prior and likelihood are called **conjugate distributions**, and the prior is called a **conjugate prior** for the likelihood. This property is very useful as it allows for closed-form updates.

For example, the Gaussian distribution is self-conjugate with respect to a Gaussian likelihood. If the likelihood is Gaussian and the prior is Gaussian, the posterior is also Gaussian.

## MLE vs MAP

### Maximum Likelihood Estimation (MLE)
MLE seeks the parameter $\theta$ that makes the observed data most likely. It ignores the prior (or assumes a uniform prior).

$$
\hat{\theta}_{\text{MLE}} = \arg\max_{\theta} p(\mathcal{D} \mid \theta) = \arg\max_{\theta} \sum_{i} \log p(y_i \mid x_i, \theta)
$$

For a linear model with Gaussian noise, MLE is equivalent to minimizing the **Sum of Squared Errors (SSE)**, leading to the Ordinary Least Squares (OLS) solution.

### Maximum A Posteriori (MAP)
MAP incorporates the prior belief. It seeks the mode of the posterior distribution.

$$
\hat{\theta}_{\text{MAP}} = \arg\max_{\theta} p(\theta \mid \mathcal{D}) = \arg\max_{\theta} [ \log p(\mathcal{D} \mid \theta) + \log p(\theta) ]
$$

MAP can be seen as MLE with a regularization term coming from the prior.
- A **Gaussian Prior** $w \sim \mathcal{N}(0, \sigma_p^2 I)$ corresponds to **L2 Regularization (Ridge Regression)**.
- A **Laplace Prior** $w \sim \text{Laplace}(0, b)$ corresponds to **L1 Regularization (Lasso Regression)**.

## Bayesian Linear Regression (Weight-Space View)

In Bayesian Linear Regression (BLR), instead of finding a single "best" weight vector $w$ (as in MLE or MAP), we maintain a distribution over all possible weight vectors.

### Model Setup
We assume a linear model with additive Gaussian noise:
$$
y = w^T x + \varepsilon, \quad \varepsilon \sim \mathcal{N}(0, \sigma_n^2)
$$
Likelihood:
$$
p(y \mid X, w) = \mathcal{N}(y \mid Xw, \sigma_n^2 I)
$$
Prior on weights:
$$
p(w) = \mathcal{N}(w \mid 0, \Sigma_p)
$$
usually $\Sigma_p = \sigma_p^2 I$.

### Posterior Distribution
Using Bayes' rule and the properties of Gaussians (completing the square), the posterior $p(w \mid X, y)$ is also Gaussian:

$$
p(w \mid X, y) = \mathcal{N}(w \mid \bar{w}, A^{-1})
$$

where:
- $A = \sigma_n^{-2} X^T X + \Sigma_p^{-1}$ is the precision matrix of the posterior.
- $\bar{w} = \sigma_n^{-2} A^{-1} X^T y$ is the mean of the posterior (which is also the MAP estimate).

{{< figure 
    src="/images/ml/bayesLinearRegression.gif"
    caption="Bayesian Linear Regression with on the right the samples from the posterior distribution over functions as new data points are observed. On the left, the uncertainty in the weights is shown."
    alt="Bayesian Linear Regression with on the right the samples from the posterior distribution over functions as new data points are observed. On the left, the uncertainty in the weights is shown."
>}}

## Predictive Distribution

To make a prediction for a new input $x_*$, we don't just use the MAP weights. We marginalize over the posterior distribution of weights. This averages predictions from all possible models, weighted by their posterior probability.

$$
p(y_* \mid x_*, X, y) = \int p(y_* \mid x_*, w) p(w \mid X, y) dw
$$

Since everything is Gaussian, this integral can be solved in closed form:

$$
p(y_* \mid x_*, X, y) = \mathcal{N}(y_* \mid \bar{w}^T x_*, \sigma_n^2 + x_*^T A^{-1} x_*)
$$

The predictive variance has two components:
$$
\mathbb{V}[y_*] = \underbrace{\sigma_n^2}_{\text{Aleatoric}} + \underbrace{x_*^T A^{-1} x_*}_{\text{Epistemic}}
$$

## Epistemic vs Aleatoric Uncertainty

Bayesian methods allow us to distinguish between two types of uncertainty:

1.  **Aleatoric Uncertainty ($\sigma_n^2$)**: Uncertainty inherent in the data (noise). It cannot be reduced by gathering more data.
2.  **Epistemic Uncertainty ($x_*^T A^{-1} x_*$)**: Uncertainty in the model parameters. It arises from lack of data and **can be reduced** by observing more data. As $N \to \infty$, the posterior precision $A$ increases, and this term goes to 0.

{{< figure 
    src="/images/ml/bayesAleatoricEpistemicUncertainty.png"
    caption="Predictive distribution in Bayesian Linear Regression showing uncertainty."
    alt="Predictive distribution in Bayesian Linear Regression showing uncertainty."
>}}

## Model Selection and Evidence Maximization

How do we choose hyperparameters like the prior variance $\sigma_p^2$ or noise variance $\sigma_n^2$?
Instead of Cross-Validation, we can use **Evidence Maximization** (also called Type II MLE or Empirical Bayes).

We maximize the **Marginal Likelihood** (Evidence) with respect to the hyperparameters:

$$
p(y \mid X, \alpha, \beta) = \int p(y \mid X, w, \beta) p(w \mid \alpha) dw
$$

For BLR, the log marginal likelihood is:
$$
\log p(y \mid X) = -\frac{1}{2} y^T C^{-1} y - \frac{1}{2} \log |C| - \frac{n}{2} \log(2\pi)
$$
where $C = \sigma_p^2 X X^T + \sigma_n^2 I$.

Maximizing evidence automatically embodies **Occam's Razor**: it penalizes overly complex models (which spread their probability mass too thin) and overly simple models (which cannot fit the data).

## Recursive Bayesian Learning

One of the strengths of Bayesian learning is that it can be done **online**. The posterior after observing $N$ data points becomes the **prior** for the $(N+1)$-th data point.

$$
p(w \mid \mathcal{D}_{N+1}) \propto p(y_{N+1} \mid x_{N+1}, w) p(w \mid \mathcal{D}_N)
$$

This avoids recomputing the inverse of the full design matrix from scratch. For BLR, we can use the **Sherman-Morrison-Woodbury** formula to update the inverse covariance matrix in $O(d^2)$ time instead of $O(d^3)$.

## Decision Theory

Bayesian inference gives us a distribution. To make a decision (e.g., a point prediction), we need a **Loss Function** $L(y, \hat{y})$. We choose the action that minimizes the **Expected Loss**:

$$
\hat{y} = \arg\min_{a} \mathbb{E}_{p(y_* \mid \mathcal{D})} [L(y_*, a)]
$$

- **Squared Loss** $L(y, \hat{y}) = (y - \hat{y})^2 \implies$ Mean of predictive distribution.
- **Absolute Loss** $L(y, \hat{y}) = |y - \hat{y}| \implies$ Median of predictive distribution.
- **0-1 Loss** (Classification) $\implies$ Mode (MAP class).

## Kernel Trick and Basis Functions

Linear models are limited. We can extend them by mapping inputs to a high-dimensional feature space using basis functions $\phi(x)$:
$$
y = w^T \phi(x) + \varepsilon
$$
However, if $\phi(x)$ is very high-dimensional, computing with it is expensive.
We observe that in the dual representation (function-space view), the solution only depends on inner products $\phi(x)^T \phi(x')$.

We can define a **Kernel Function** $k(x, x') = \phi(x)^T \phi(x')$ to compute this inner product directly without explicitly computing $\phi(x)$. This allows us to work with infinite-dimensional feature spaces (like RBF kernel) efficiently. This leads us to **Gaussian Processes**.
