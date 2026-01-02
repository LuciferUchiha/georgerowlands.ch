---
title: Bayesian Linear Regression
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

## Probabilistic Inference

Probabilistic inference is the process of updating our beliefs about uncertain quantities (such as model parameters or future observations) in light of new evidence (data). Unlike logical inference, which deals with certainties (such as $A \implies B$), probabilistic inference deals with plausibilities.

As humans, we make plausible inferences constantly. For example, if we see wet ground, we might infer it rained. If we then check and see the sky is blue, we might update that belief to someone used sprinklers instead. 

The core rule of probabilistic inference is **Bayes' Rule** which why it is often called **Bayesian Inference**.

$$
\mathbb{P}(\theta \mid \mathcal{D}) = \frac{\mathbb{P}(\mathcal{D} \mid \theta) \mathbb{P}(\theta)}{\mathbb{P}(\mathcal{D})}
$$

This equation relates four key components:

1.  **The Prior $\mathbb{P}(\theta)$**: This represents our initial belief about the hypothesis $\theta$ *before* seeing any data. In our example, this is our prior probability that it rained today. Note that priors can be subjective or "wrong" in the context of the actual event, if we live in a desert, our prior for rain is low. If we blindly trust a strong prior (e.g., "it never rains"), we might ignore the evidence of wet ground and assume sprinklers, even if it actually did rain.
2.  **The Likelihood $\mathbb{P}(\mathcal{D} \mid \theta)$**: This measures how well the hypothesis explains the data. It asks: "If it had rained ($\theta$), how likely is it that the ground would be wet ($\mathcal{D}$)?" This is usually high. Conversely, "If it was sunny, how likely is the ground wet?" is low (unless sprinklers are involved).
3.  **The Posterior $\mathbb{P}(\theta \mid \mathcal{D})$**: This is what we want to compute, our updated belief after observing the data. "Given the ground is wet, what is the probability it rained?" It balances the prior belief with the likelihood of the evidence.
4.  **The Marginal Likelihood (Evidence) $\mathbb{P}(\mathcal{D})$**: This acts as a normalizing constant to ensure the posterior probabilities sum to 1. It represents the probability of the data under all possible hypotheses:
    $$ \mathbb{P}(\mathcal{D}) = \int \mathbb{P}(\mathcal{D} \mid \theta) \mathbb{P}(\theta) d\theta $$

{{< figure 
    src="/images/ml/bayesPriorLikelihoodPosterior.webp"
    caption="Bayes' Theorem relates the prior, likelihood, and posterior distributions."
    alt="Bayes' Theorem relates the prior, likelihood, and posterior distributions."
>}}

### Conjugate Priors

Computing the posterior often involves a difficult integration for the marginal likelihood $\mathbb{P}(\mathcal{D})$. This integral can be intractable for complex models or high-dimensional parameter spaces. However, if we choose the prior carefully, we can avoid this integration.

If the posterior distribution $\mathbb{P}(\theta \mid \mathcal{D})$ ends up being in the same probability distribution family as the prior $\mathbb{P}(\theta)$, we say the prior is **conjugate** to the likelihood. This allows for closed-form updates: we simply update the parameters of the distribution (e.g., mean and variance in the Gaussian case) rather than computing complex integrals. 

One common example is the **Gaussian-Gaussian** conjugacy, where a Gaussian prior combined with a Gaussian likelihood results in a Gaussian posterior.

## MLE and MAP

Before diving into full Bayesian Linear Regression, it is useful to understand two common point estimation techniques: **Maximum Likelihood Estimation (MLE)** and **Maximum A Posteriori (MAP)**. These methods give us a single "best" parameter value rather than a full distribution, but they are closely related to the Bayesian approach.

**Maximum Likelihood Estimation (MLE)** seeks the parameter $\theta$ that makes the observed data $\mathcal{D}$ most probable. It asks: "Under which parameter setting is the data we observed most likely to have occurred?"

For a more detailed introduction, see the [MLE section](/garden/maths/probabilityStatistics/estimators/#maximum-likelihood-estimation-mle).

$$
\hat{\theta}_{\text{MLE}} = \arg\max_{\theta} \mathbb{P}(\mathcal{D} \mid \theta)
$$

In practice, we often maximize the **log-likelihood** because it turns products into sums, making derivatives easier to compute, and it doesn't change the location of the maximum (since $\log$ is monotonic).

$$
\hat{\theta}_{\text{MLE}} = \arg\max_{\theta} \log \mathbb{P}(\mathcal{D} \mid \theta)
$$

It is important to note that MLE does not incorporate any prior beliefs about the parameters; it relies solely on the observed data. So if you were to toss a coin 10 times and get 10 heads, MLE concludes the probability of heads is 1.0.

Now let's see how MLE applies to Linear Regression. In Linear Regression, we assume the data is generated by a linear function with some Gaussian noise:

$$
y_i = w^T x_i + \varepsilon_i, \quad \varepsilon_i \sim \mathcal{N}(0, \sigma_n^2)
$$

We can then formulate the likelihood of observing $y_i$ given $x_i$ and weights $w$ as:

$$
\mathbb{P}(y_i \mid x_i, w) = \mathcal{N}(y_i \mid w^T x_i, \sigma_n^2)
$$

This follows from the assumption that the noise $\varepsilon_i$ is Gaussian. Since $y_i$ is the sum of a deterministic term $w^T x_i$ and a random variable $\varepsilon_i \sim \mathcal{N}(0, \sigma_n^2)$, $y_i$ itself is a random variable following a Gaussian distribution with mean $w^T x_i$ and variance $\sigma_n^2$. The likelihood function is simply the probability density of observing the data $y_i$ under this distribution.

If we then consider the entire dataset $\mathcal{D} = \{(x_i, y_i)\}_{i=1}^n$, the MLE objective becomes:

$$
\hat{w}_{\text{MLE}} = \arg\max_{w} \sum_{i=1}^n \log \mathbb{P}(y_i \mid x_i, w)
$$

Specifically, using the Gaussian likelihood and assuming the data points are independent and identically distributed (i.i.d.), we have:

$$
\mathbb{P}(y \mid X, w) = \prod_{i=1}^n \frac{1}{\sqrt{2\pi\sigma_n^2}} \exp\left( -\frac{(y_i - w^T x_i)^2}{2\sigma_n^2} \right)
$$

Taking the log-likelihood results in:

$$
\begin{align*}
\log \mathbb{P}(y \mid X, w) &= \sum_{i=1}^n \left[ -\frac{1}{2} \log(2\pi\sigma_n^2) - \frac{(y_i - w^T x_i)^2}{2\sigma_n^2} \right] \\
&= -\frac{n}{2} \log(2\pi\sigma_n^2) - \frac{1}{2\sigma_n^2} \sum_{i=1}^n (y_i - w^T x_i)^2
\end{align*}
$$

Maximizing this quantity with respect to $w$ is equivalent to minimizing the negative term (the sum of squared errors) as the first term does not depend on $w$, only on $\sigma_n^2$ and $n$.

$$
\hat{w}_{\text{MLE}} = \arg\min_{w} \sum_{i=1}^n (y_i - w^T x_i)^2 = \arg\min_{w} ||y - Xw||_2^2
$$

Notice that this is exactly the same as minimizing the **Residual Sum of Squares (RSS)** in Ordinary Least Squares regression. So we have shown that **MLE for Linear Regression with Gaussian noise is equivalent to Ordinary Least Squares**:

$$
\hat{w}_{\text{MLE}} = \hat{w}_{\text{OLS}}
$$

This also makes sense as if you remember in the [linear regression section](/garden/ml/basics/linearRegression/) we said that OLS is the best linear unbiased estimator (BLUE) under the Gauss-Markov assumptions and that by definition of the [MLE in statistics section](/garden/maths/probabilitystatistics/estimators/#maximum-likelihood-estimation-mle) it is also unbiased and efficient. Thus they must be the same estimator.

Another important thing to note is that the MLE estimate does not depend on the noise variance $\sigma_n^2$. This is because it only scales the log-likelihood by a constant factor, which does not affect the location of the maximum. Thus, MLE focuses solely on fitting the data, without considering the noise level of the observations.

Next , we will see how to incorporate prior beliefs about the parameters using **Maximum A Posteriori (MAP) estimation**. Instead of maximizing just the likelihood, it maximizes the **posterior** probability of the parameters given the data:

$$
\hat{\theta}_{\text{MAP}} = \arg\max_{\theta} \mathbb{P}(\theta \mid \mathcal{D})
$$

We get this posterior using Bayes' Rule:

$$
\mathbb{P}(\theta \mid \mathcal{D}) = \frac{\mathbb{P}(\mathcal{D} \mid \theta) \mathbb{P}(\theta)}{\mathbb{P}(\mathcal{D})}
$$

Another way to write this is:

$$
\mathbb{P}(\theta \mid \mathcal{D}) \propto \mathbb{P}(\mathcal{D} \mid \theta) \mathbb{P}(\theta)
$$

Meaning the posterior is proportional to the likelihood times the prior. This is because $\mathbb{P}(\mathcal{D})$ does not depend on $\theta$ and acts as a normalizing constant. If we then take the logarithm (to simplify calculations), the MAP objective becomes:

$$
\hat{\theta}_{\text{MAP}} = \arg\max_{\theta} [ \underbrace{\log \mathbb{P}(\mathcal{D} \mid \theta)}_{\text{Log-Likelihood}} + \underbrace{\log \mathbb{P}(\theta)}_{\text{Log-Prior}} ]
$$

So we can see that **MAP estimation adds a prior term to the MLE objective**. This prior term acts as a regularizer, influencing the estimate based on our prior beliefs about $\theta$. So if we have a strong prior that the coin is fair, seeing 10 heads might only shift our estimate slightly away from 0.5, rather than all the way to 1.0. 

We can also retrieve the MLE as a special case of MAP by using a uniform (non-informative) prior, which contributes no additional information. A uniform prior means that all parameter values are equally likely a priori, so the prior term $\log \mathbb{P}(\theta)$ is constant and does not affect the maximization. So in this case, MAP reduces to MLE:

$$
\hat{\theta}_{\text{MAP}} = \arg\max_{\theta} \log \mathbb{P}(\mathcal{D} \mid \theta) + \text{const} = \arg\max_{\theta} \log \mathbb{P}(\mathcal{D} \mid \theta) = \hat{\theta}_{\text{MLE}}
$$

Let us now apply some specific priors to Linear Regression to see how MAP estimation leads to regularized regression methods. So we want to pick some prior $\mathbb{P}(w)$ for the weights $w$. An initial assumption is that the weight are from a zero-mean gaussian distribution. This can be interpreted as believing that the weights should be small (close to zero) before seeing any data. So formally we have:

$$
w \sim \mathcal{N}(0, \sigma_p^2 I)
$$

and thus from the definition of the multivariate gaussian distribution we have:

$$
\begin{align*}
\mathbb{P}(w) &= \frac{1}{(2\pi)^{d/2} |\sigma_p^2 I|^{1/2}} \exp\left( -\frac{1}{2} w^T (\sigma_p^2 I)^{-1} w \right) \\
&= \frac{1}{(2\pi)^{d/2} (\sigma_p^2)^{d/2}} \exp\left( -\frac{1}{2\sigma_p^2} w^T w \right)
\end{align*}
$$

If we then take the logarithm we get:

$$
\log \mathbb{P}(w) = -\frac{d}{2} \log(2\pi) - \frac{d}{2} \log(\sigma_p^2) - \frac{1}{2\sigma_p^2} ||w||_2^2
$$

If we ignore the constant terms (which do not depend on $w$), we have:

$$
\log \mathbb{P}(w) \propto -\frac{1}{2\sigma_p^2} ||w||_2^2
$$


Substituting this into the MAP objective we get:

$$
\hat{w}_{\text{MAP}} = \arg\min_{w} \left[ \frac{1}{2\sigma_n^2} ||y - Xw||_2^2 + \frac{1}{2\sigma_p^2} ||w||_2^2 \right]
$$

Notice that this is equivalent to minimizing the sum of squared errors (the MLE objective) plus an L2 regularization term on the weights. This is exactly the objective for **Ridge Regression**! In particular we can also also multiply through by $2\sigma_n^2$ (which does not change the location of the minimum) to get: 

$$
\hat{w}_{\text{MAP}} = \arg\min_{w} \left[ ||y - Xw||_2^2 + \lambda ||w||_2^2 \right]
$$

where $\lambda = \frac{\sigma_n^2}{\sigma_p^2}$ is the regularization parameter that balances data fit and weight magnitude. Thus, we have shown that **MAP estimation with a Gaussian prior on the weights leads to Ridge Regression**.

Another common prior is the Laplace prior, which is like a very peaked distribution around zero. This prior encourages sparsity in the weights, meaning it pushes many weights to be exactly zero. This is useful when we believe that only a few features are relevant for predicting the target. Formally, we assume:

$$
\mathbb{P}(w) = \prod_{j=1}^d \frac{\lambda}{2} \exp(-\lambda |w_j|)
$$

Taking the logarithm gives:

$$
\log \mathbb{P}(w) = \sum_{j=1}^d \left[ \log\left(\frac{\lambda}{2}\right) - \lambda |w_j| \right] = d \log\left(\frac{\lambda}{2}\right) - \lambda ||w||_1
$$

Ignoring the constant term, we have:

$$
\log \mathbb{P}(w) \propto -\lambda ||w||_1
$$

Substituting this into the MAP objective yields:

$$
\hat{w}_{\text{MAP}} = \arg\min_{w} \left[ \frac{1}{2\sigma_n^2} ||y - Xw||_2^2 + \lambda ||w||_1 \right]
$$

This is exactly the objective for **Lasso Regression**! Thus, we have shown that **MAP estimation with a Laplace prior on the weights leads to Lasso Regression**.

## Bayesian Linear Regression

While MLE and MAP provide useful point estimates, they discard valuable information about the reliability of those estimates. **Bayesian Linear Regression (BLR)** addresses this by computing the **full posterior distribution** $\mathbb{P}(w \mid X, y)$ rather than a single "best" weight vector. This distribution captures our uncertainty, which is crucial when data is scarce or noisy. A wide posterior indicates low confidence, while a narrow posterior suggests high confidence.

We begin by specifying the probabilistic model with a Gaussian likelihood and a Gaussian prior. The likelihood $\mathbb{P}(y \mid X, w) = \mathcal{N}(y \mid Xw, \sigma_n^2 I)$ describes the data generation process, while the prior $\mathbb{P}(w) = \mathcal{N}(w \mid m_0, \Sigma_0)$ encodes our initial beliefs about the weights. Typically, we use a zero-mean isotropic prior ($m_0=0, \Sigma_0 = \sigma_p^2 I$).

To find the posterior distribution, we apply Bayes' rule:

$$
\mathbb{P}(w \mid X, y) = \frac{\mathbb{P}(y \mid X, w) \mathbb{P}(w)}{\mathbb{P}(y \mid X)}
$$

Since the denominator $\mathbb{P}(y \mid X)$ is constant with respect to $w$, we can focus on the numerator. Working with the log-density allows us to simplify the product of exponentials into a sum:

$$
\log \mathbb{P}(w \mid X, y) \propto -\frac{1}{2\sigma_n^2} (y - Xw)^T (y - Xw) - \frac{1}{2} (w - m_0)^T \Sigma_0^{-1} (w - m_0)
$$

By expanding the quadratic terms and grouping them by powers of $w$, we can identify the functional form of the posterior. The terms quadratic in $w$ determine the posterior precision (inverse covariance), while the linear terms determine the shifted mean. Matching these to the standard Gaussian log-density reveals that the posterior is indeed Gaussian $\mathbb{P}(w \mid X, y) = \mathcal{N}(w \mid \bar{w}, A^{-1})$ with:

1.  **Posterior Precision** $A = \sigma_n^{-2} X^T X + \Sigma_0^{-1}$. This is the sum of the data precision and the prior precision. As we observe more data, the data term dominates, increasing the total precision and thus shrinking the covariance (uncertainty).
2.  **Posterior Mean** $\bar{w} = A^{-1} (\sigma_n^{-2} X^T y + \Sigma_0^{-1} m_0)$. This represents a weighted average of the data estimate and the prior mean. Notably, if we use a zero-mean prior ($m_0=0$), this expression simplifies exactly to the **Ridge Regression** solution (and the MAP estimate).

This result provides a powerful interpretation: Ridge Regression is not just an ad-hoc regularization technique, but the mean of the Bayesian posterior under a Gaussian prior. However, BLR gives us more than just the mean; the posterior covariance $A^{-1}$ quantifies the correlation and uncertainty of the weights.

Since each weight vector $w$ defines a linear function $f(x) = w^T x$, the posterior distribution over weights implies a distribution over linear functions. We can visualize this uncertainty by sampling multiple weight vectors from the posterior and plotting the resulting lines. In regions with little data, the lines will fan out, visually demonstrating our lack of knowledge. Where data is abundant, the lines will bundle tightly together.

{{< figure 
    src="/images/ml/bayesLinearRegression.gif"
    caption="Bayesian Linear Regression: As more data points (right) are observed, the posterior distribution over functions narrows, and the uncertainty in weights (left) decreases."
    alt="Bayesian Linear Regression animation"
>}}

### Predictive Distribution

The ultimate goal is often to make predictions for a new input $x_*$. Instead of relying on a single "best" weight vector, Bayesian inference employs **marginalization**: we average the predictions of *all* possible weight vectors, weighted by their posterior probability.

We seek the distribution of the new target $y_*$, which is the function value $f_* = w^T x_*$ plus observation noise $\varepsilon_*$.

$$
\mathbb{P}(y_* \mid x_*, X, y) = \int \mathbb{P}(y_* \mid x_*, w) \mathbb{P}(w \mid X, y) dw
$$

Since $y_*$ is a linear combination of two independent Gaussian variables (the weights $w$ and the noise $\varepsilon_*$), the resulting distribution is also Gaussian. We can compute its moments directly:

- **Mean**: $\mathbb{E}[y_*] = \mathbb{E}[w^T x_* + \varepsilon_*] = \bar{w}^T x_*$. This confirms that the Bayesian predictive mean is identical to the prediction made by the MAP (or Ridge) point estimate.
- **Variance**: $\text{Var}[y_*] = \text{Var}[x_*^T w] + \text{Var}[\varepsilon_*] = x_*^T \text{Var}[w] x_* + \sigma_n^2 = x_*^T A^{-1} x_* + \sigma_n^2$.

Thus, the predictive distribution is given by:

$$
\mathbb{P}(y_* \mid x_*, X, y) = \mathcal{N}(y_* \mid \bar{w}^T x_*, \sigma_n^2 + x_*^T A^{-1} x_*)
$$

Crucially, the predictive variance is not constant. It consists of the intrinsic noise $\sigma_n^2$ plus a term $x_*^T A^{-1} x_*$ that depends on the input $x_*$. This means the model is "aware" of its own ignorance: it predicts higher uncertainty for inputs that are far from the training data, where the posterior uncertainty about weights has a larger impact.

### Aleatoric vs Epistemic Uncertainty

The variance of the predictive distribution captures our uncertainty about the prediction at $x_*$. This uncertainty has two components:

$$
\text{Var}[y_*] = \underbrace{\sigma_n^2}_{\text{Aleatoric}} + \underbrace{x_*^T A^{-1} x_*}_{\text{Epistemic}}
$$

We can interpret these two sources of uncertainty:

1.  **Aleatoric Uncertainty ($\sigma_n^2$)**: The inherent noise in the data. This is irreducible; no matter how much data we collect, the noise remains.
2.  **Epistemic Uncertainty ($x_*^T A^{-1} x_*$)**: The uncertainty in our model parameters $w$. This **can be reduced** by collecting more data (which increases the precision $A$) or by choosing better priors.

The epistemic uncertainty is highest in regions of the input space where we have little or no training data. As we gather more data in those regions, the posterior over weights becomes more concentrated, reducing this uncertainty. This is not easily shown mathematically, but can be visualized by plotting the predictive distribution as we add more data points. Intuitively you can think of it as the model saying "I don't know much about this region, so my predictions here are uncertain and I should be cautious."

{{< figure 
    src="/images/ml/bayesAleatoricEpistemicUncertainty.png"
    caption="Predictive uncertainty decomposes into noise (aleatoric) and model uncertainty (epistemic)."
    alt="Aleatoric vs Epistemic Uncertainty"
    width="400"
>}}

### Hyperparameters and Model Selection

We have so far assumed that the noise variance $\sigma_n^2$ and prior variance $\sigma_p^2$ are known. In practice, these are often unknown and need to be estimated from the data. So how do we choose these **hyperparameters**?

The standard frequentist approach is to use cross-validation, just like in regularized regression. However, the Bayesian framework offers a more elegant solution. Specifically, we can use the data to estimate hyperparameters by maximizing the **Marginal Likelihood** (also called the **Evidence**). This is also known as **Type II Maximum Likelihood**. The marginal likelihood is the probability of the observed data under the model, integrating out the weights:

$$
\mathbb{P}(y \mid X, \sigma_n^2, \sigma_p^2) = \int \mathbb{P}(y \mid X, w, \sigma_n^2) \mathbb{P}(w \mid \sigma_p^2) dw
$$

Maximizing this quantity with respect to the hyperparameters $\sigma_n^2, \sigma_p^2$ allows us to find values that best explain the observed data while accounting for model complexity. Because both the likelihood and prior are Gaussian, this integral can be computed in closed form, resulting in:

$$
\mathbb{P}(y \mid X, \sigma_n^2, \sigma_p^2) = \mathcal{N}(y \mid 0, \sigma_n^2 I + \sigma_p^2 X X^T)
$$

We can then optimize the log marginal likelihood:

$$
\log \mathbb{P}(y \mid X, \sigma_n^2, \sigma_p^2) = -\frac{n}{2} \log(2\pi) - \frac{1}{2} \log |\sigma_n^2 I + \sigma_p^2 X X^T| - \frac{1}{2} y^T (\sigma_n^2 I + \sigma_p^2 X X^T)^{-1} y
$$

This optimization can be performed using gradient-based methods. The resulting hyperparameters balance data fit and model complexity, embodying the principle of **Occam's Razor**, which favors simpler models that adequately explain the data:
- **Model Complexity**: More complex models (e.g., larger prior variance) can fit the training data better but risk overfitting. These models spread their probability mass thinly over many possible datasets, and thus assign lower probability to the actual observed data.
- **Data Fit**: Simpler models (e.g., smaller prior variance) may not fit the data well, leading to low likelihood.
- **Optimal Balance**: The hyperparameters that maximize the marginal likelihood strike the best balance, assigning the highest probability to the observed data.

### Recursive Bayesian Learning

Another powerful aspect of Bayesian learning is that it can be updated recursively as new data arrives. This means that Bayesian inference can be performed **online**. The posterior after observing $N$ data points becomes the **prior** for the $(N+1)$-th data point.

$$
\mathbb{P}(w \mid \mathcal{D}_{N+1}) \propto \mathbb{P}(y_{N+1} \mid x_{N+1}, w) \mathbb{P}(w \mid \mathcal{D}_N)
$$

This recursive property holds because the data points are assumed to be independent given the parameters. Mathematically, the likelihood of the full dataset factorizes into the likelihood of the new point times the likelihood of the previous points. Since the posterior $\mathbb{P}(w \mid \mathcal{D}_N)$ already encapsulates all information from the first $N$ points (proportional to prior times likelihood of $\mathcal{D}_N$), multiplying it by the likelihood of the new point $y_{N+1}$ correctly gives us the new posterior (up to a normalization constant). This allows us to update our model one data point at a time without storing or reprocessing the full dataset.

For Bayesian Linear Regression, this update is particularly efficient. Recall from the previous section that the posterior precision matrix after observing $N$ data points is defined as the sum of the data precision and the **initial prior precision** $\Sigma_0^{-1}$ (where $\Sigma_0$ is the covariance of the initial prior $\mathbb{P}(w)$):

$$
A_N = \sigma_n^{-2} X_N^T X_N + \Sigma_0^{-1}
$$

When we observe a new data point $(x_{N+1}, y_{N+1})$, we append it to our design matrix $X$. The new term $X_{N+1}^T X_{N+1}$ can be written as the sum of the old term and the outer product of the new input:

$$
X_{N+1}^T X_{N+1} = \sum_{i=1}^{N+1} x_i x_i^T = \left( \sum_{i=1}^{N} x_i x_i^T \right) + x_{N+1} x_{N+1}^T = X_N^T X_N + x_{N+1} x_{N+1}^T
$$

Substituting this back into the definition of the precision matrix, we get a simple additive update rule:

$$
A_{N+1} = \sigma_n^{-2} (X_N^T X_N + x_{N+1} x_{N+1}^T) + \Sigma_0^{-1} = \underbrace{(\sigma_n^{-2} X_N^T X_N + \Sigma_0^{-1})}_{A_N} + \sigma_n^{-2} x_{N+1} x_{N+1}^T
$$

$$
A_{N+1} = A_N + \sigma_n^{-2} x_{N+1} x_{N+1}^T
$$

Inverting this matrix to get the covariance $S_{N+1} = A_{N+1}^{-1}$ would normally cost $O(d^3)$. However, since the update is a rank-1 addition (adding the outer product of a vector), we can use the **Sherman-Morrison** formula to update the inverse directly in $O(d^2)$ time. The formula states that $(A + uv^T)^{-1} = A^{-1} - \frac{A^{-1}uv^TA^{-1}}{1 + v^TA^{-1}u}$.

Applying this with $u = \sigma_n^{-2} x_{N+1}$ and $v = x_{N+1}$, and letting $S_N = A_N^{-1}$, we derive the update for the covariance matrix:

$$
S_{N+1} = S_N - \frac{S_N (\sigma_n^{-2} x_{N+1}) x_{N+1}^T S_N}{1 + x_{N+1}^T S_N (\sigma_n^{-2} x_{N+1})} = S_N - \frac{S_N x_{N+1} x_{N+1}^T S_N}{\sigma_n^2 + x_{N+1}^T S_N x_{N+1}}
$$

Similarly, the mean vector can be updated using the new covariance and the prediction error on the new point:

$$
m_{N+1} = m_N + S_{N+1} \sigma_n^{-2} x_{N+1} (y_{N+1} - m_N^T x_{N+1})
$$

These equations constitute the **Recursive Least Squares (RLS)** algorithm, which is widely used in signal processing and control systems for real-time estimation.

## Kernel Trick and Basis Functions

Linear models are limited to linear relationships. We can extend them by mapping inputs to a high-dimensional feature space using basis functions $\phi(x)$:

$$
y = w^T \phi(x) + \varepsilon
$$

However, if $\phi(x)$ is very high-dimensional (or infinite), computing with it is expensive.
We observe that in the dual representation (function-space view), the solution only depends on inner products $\phi(x)^T \phi(x')$.

We can define a **Kernel Function** $k(x, x') = \phi(x)^T \phi(x')$ to compute this inner product directly. This allows us to work with infinite-dimensional feature spaces (like the RBF kernel) efficiently, leading to **Gaussian Processes**.
