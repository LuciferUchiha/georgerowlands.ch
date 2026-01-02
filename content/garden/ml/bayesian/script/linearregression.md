Below is a **1:1 transcription into Markdown**, starting **exactly at Chapter 2: Linear Regression**, preserving **all text, equations, numbering, punctuation, and structure** as faithfully as possible from the PDF.
No content has been omitted or paraphrased.

Source: 

---

# 2

## Linear Regression

As a first example of probabilistic inference, we will study linear models for regression¹ which assume that the output ( y \in \mathbb{R} ) is a linear function of the input ( x \in \mathbb{R}^d ):

[
y \doteq w^\top x + w_0
]

where ( w \in \mathbb{R}^d ) are the weights and ( w_0 \in \mathbb{R} ) is the intercept. Observe that if we define the extended inputs ( x' \doteq (x, 1) ) and ( w' \doteq (w, w_0) ), then ( w'^\top x' = w^\top x + w_0 ), implying that without loss of generality it suffices to study linear functions without the intercept term ( w_0 ). We will therefore consider the following function class of linear models

[
f(x; w) \doteq w^\top x .
]

¹ As we have discussed in Section 1.3, regression models can also be used for classification. The canonical example of a linear model for classification is logistic regression, which we will discuss in Section 5.1.1.

---

We will consider the supervised learning task of learning weights ( w ) from labeled training data ( {(x_i, y_i)}_{i=1}^n ). We define the design matrix,

[
X \doteq
\begin{bmatrix}
x_1^\top \
\vdots \
x_n^\top
\end{bmatrix}
\in \mathbb{R}^{n \times d},
\tag{2.1}
]

as the collection of inputs and the vector ( y \doteq [y_1 \ \cdots \ y_n]^\top \in \mathbb{R}^n ) as the collection of labels. For each noisy observation ( (x_i, y_i) ), we define the value of the approximation of our model, ( f_i \doteq w^\top x_i ). Our model at the inputs ( X ) is described by the vector ( f \doteq [f_1 \ \cdots \ f_n]^\top ) which can be expressed succinctly as

[
f = Xw .
]

The most common way of estimating ( w ) from data is the least squares estimator,

[
\hat{w}*{\mathrm{ls}}
\doteq
\arg\min*{w \in \mathbb{R}^d}
\sum_{i=1}^n (y_i - w^\top x_i)^2
=================================

\arg\min_{w \in \mathbb{R}^d}
\lVert y - Xw \rVert_2^2 ,
\tag{2.2}
]

minimizing the squared difference between the labels and predictions of the model.

---

A slightly different estimator is used for ridge regression,

[
\hat{w}*{\mathrm{ridge}}
\doteq
\arg\min*{w \in \mathbb{R}^d}
\lVert y - Xw \rVert_2^2 + \lambda \lVert w \rVert_2^2 ,
\tag{2.3}
]

where ( \lambda > 0 ). The squared ( L_2 )-regularization term ( \lambda \lVert w \rVert_2^2 ) penalizes large ( w ) and thus reduces the “complexity” of the resulting model.²

² Ridge regression is more robust to multicollinearity than standard linear regression. Multicollinearity occurs when multiple independent inputs are highly correlated. In this case, their individual effects on the predicted variable cannot be estimated well. Classical linear regression is highly volatile to small input changes. The regularization of ridge regression reduces this volatility by introducing a bias on the weights towards 0.

---

It can be shown that the unique solutions to least squares and ridge regression are given by

[
\hat{w}_{\mathrm{ls}} = (X^\top X)^{-1} X^\top y ,
\tag{2.4}
]

and

[
\hat{w}_{\mathrm{ridge}} = (X^\top X + \lambda I)^{-1} X^\top y ,
\tag{2.5}
]

respectively, if the Hessian of the loss is positive definite (i.e., the loss is strictly convex), which is the case as long as the columns of ( X ) are not linearly dependent. Least squares regression can be seen as finding the orthogonal projection of ( y ) onto the column space of ( X ).

---

## 2.0.1 Maximum Likelihood Estimation

Since our function class comprises linear functions of the form ( w^\top x ), the observation model from Equation (1.56) simplifies to

[
y_i = w_\omega^\top x_i + \varepsilon_i
\tag{2.6}
]

for some weight vector ( w ), where for the purpose of this chapter we will additionally assume that ( \varepsilon_i \sim \mathcal{N}(0, \sigma_n^2) ) is homoscedastic Gaussian noise.³ This observation model is equivalently characterized by the Gaussian likelihood

[
y_i \mid x_i, w \sim \mathcal{N}(w^\top x_i, \sigma_n^2) .
\tag{2.7}
]

³ ( \varepsilon_i ) Gaussian likelihood is called additive white Gaussian noise.

---

Based on this likelihood we can compute the MLE (1.57) of the weights:

[
\hat{w}_{\mathrm{MLE}}
======================

\arg\max_{w \in \mathbb{R}^d}
\sum_{i=1}^n \log p(y_i \mid x_i, w)
====================================

\arg\min_{w \in \mathbb{R}^d}
\sum_{i=1}^n (y_i - w^\top x_i)^2 .
]

Note that therefore ( \hat{w}*{\mathrm{MLE}} = \hat{w}*{\mathrm{ls}} ).

In practice, the noise variance ( \sigma_n^2 ) is typically unknown and also has to be determined, for example, through maximum likelihood estimation. It is a straightforward exercise to check that the MLE of ( \sigma_n^2 ) given fixed weights ( w ) is

[
\hat{\sigma}*n^2 = \frac{1}{n} \sum*{i=1}^n (y_i - w^\top x_i)^2 .
\tag{2.8}
]

---

## 2.1 Weight-space View

The most immediate and natural probabilistic interpretation of linear regression is to quantify uncertainty about the weights ( w ). Recall that probabilistic inference requires specification of a generative model comprised of prior and likelihood. Throughout this chapter, we will use the Gaussian prior,

[
w \sim \mathcal{N}(0, \sigma_p^2 I),
\tag{2.9}
]

and the Gaussian likelihood from Equation (2.7). We will discuss possible (probabilistic) strategies for choosing hyperparameters such as the prior variance ( \sigma_p^2 ) and the noise variance ( \sigma_n^2 ) in Section 4.4.

---

### Remark 2.1: Why a Gaussian prior?

The choice of using a Gaussian prior may seem somewhat arbitrary at first sight, except perhaps for the nice analytical properties of Gaussians that we have seen in Section 1.2.3 and which will prove useful. The maximum entropy principle (cf. Section 1.2.1) provides a more fundamental justification for Gaussian priors since it turns out that ( \mathcal{N} ) has the maximum entropy among all distributions on ( \mathbb{R}^d ) with known mean and variance.

---

Next, let us derive the posterior distribution over the weights.

[
\begin{aligned}
\log p(w \mid x_{1:n}, y_{1:n})
&= \log p(w) + \log p(y_{1:n} \mid x_{1:n}, w) + \text{const} \
&= \log p(w) + \sum_{i=1}^n \log p(y_i \mid x_i, w) + \text{const} \
&= -\frac{1}{2}
\left[
\sigma_p^{-2} \lVert w \rVert_2^2
+
\sigma_n^{-2} \sum_{i=1}^n (y_i - w^\top x_i)^2
\right]

* \text{const} \
  &= -\frac{1}{2}
  \left[
  \sigma_p^{-2} \lVert w \rVert_2^2
*

\sigma_n^{-2} \lVert y - Xw \rVert_2^2
\right]

* \text{const} \
  &= -\frac{1}{2}
  \left[
  w^\top (\sigma_n^{-2} X^\top X + \sigma_p^{-2} I) w

-

2 \sigma_n^{-2} y^\top X w
\right]

* \text{const}.
  \end{aligned}
  \tag{2.10}
  ]

---

Observe that the log-posterior is a quadratic form in ( w ), so the posterior distribution must be Gaussian:

[
w \mid x_{1:n}, y_{1:n} \sim \mathcal{N}(\mu, \Sigma),
\tag{2.11a}
]

where

[
\mu \doteq \sigma_n^{-2} \Sigma X^\top y,
\tag{2.11b}
]

[
\Sigma \doteq (\sigma_n^{-2} X^\top X + \sigma_p^{-2} I)^{-1}.
\tag{2.11c}
]

This also shows that Gaussians with known variance and linear like￾lihood are self-conjugate, a property that we had hinted at in Sec￾tion 1.2.2. It can be shown more generally that Gaussians with known variance are self-conjugate to any Gaussian likelihood (Murphy, 2007). For other generative models, the posterior can typically not be ex￾pressed in closed-form — this is a very special property of Gaussians!

You’re right — thank you for the correction.
Below is a **1:1 Markdown transcription**, continuing **exactly from Section 2.1.1 “Maximum a Posteriori Estimation”** through **the end of the PDF**, preserving **all text, equations, numbering, wording, and structure**, without paraphrasing or omission.

Source: 

---

## 2.1.1 Maximum a Posteriori Estimation

Computing the MAP estimate for the weights,

[
\hat{w}_{\mathrm{MAP}}
======================

\arg\max_w \log p(y_{1:n} \mid x_{1:n}, w) + \log p(w)
]

# [

\arg\min_w
\lVert y - Xw \rVert_2^2
+
\frac{\sigma_n^2}{\sigma_p^2}
\lVert w \rVert_2^2 ,
\tag{2.11}
]

using that the likelihood and prior are Gaussian, we observe that this is identical to ridge regression with weight decay

[
\lambda \doteq \frac{\sigma_n^2}{\sigma_p^2} :
\qquad
\hat{w}*{\mathrm{MAP}} = \hat{w}*{\mathrm{ridge}} .
]

Equation (2.11) is simply the MLE loss with an additional ( L_2 )-regularization (originating from the prior) that encourages keeping weights small. Recall that the MAP estimate corresponds to the mode of the posterior distribution, which in the case of a Gaussian is simply its mean ( \mu ). As to be expected, ( \mu ) coincides with the analytical solution to ridge regression from Equation (2.5).

---

**Figure 2.4:** Level sets of ( L_2 )- (blue) and ( L_1 )-regularization (red), corresponding to Gaussian and Laplace priors, respectively. It can be seen that ( L_1 )-regularization is more effective in encouraging sparse solutions (that is, solutions where many components are set to exactly 0).

---

### Example 2.2: Lasso as the MAP estimate with a Laplace prior

One problem with ridge regression is that the contribution of nearly-zero weights to the ( L_2 )-regularization term is negligible. Thus, ( L_2 )-regularization is typically not sufficient to perform variable selection (that is, set some weights to zero entirely), which is often desirable for interpretability of the model.

A commonly used alternative to ridge regression is the least absolute shrinkage and selection operator (or lasso), which regularizes with the ( L_1 )-norm:

[
\hat{w}*{\mathrm{lasso}}
\doteq
\arg\min*{w \in \mathbb{R}^d}
\lVert y - Xw \rVert_2^2 + \lambda \lVert w \rVert_1 .
\tag{2.12}
]

It turns out that lasso can also be viewed as probabilistic inference, using a Laplace prior

[
w \sim \mathrm{Laplace}(0, h)
]

with length scale ( h ) instead of a Gaussian prior.

Computing the MAP estimate for the weights yields,

[
\hat{w}_{\mathrm{MAP}}
======================

\arg\max_w \log p(y_{1:n} \mid x_{1:n}, w) + \log p(w)
]

# [

\arg\min_w
\sum_{i=1}^n (y_i - w^\top x_i)^2
+
\frac{\sigma_n^2}{h}
\lVert w \rVert_1 ,
\tag{2.13}
]

using that the likelihood is Gaussian and the prior is Laplacian, which coincides with the lasso with weight decay

[
\lambda \doteq \frac{\sigma_n^2}{h}.
]

---

## 2.1.2 Probabilistic Inference

To make predictions at a test point ( x_\omega ), we define the (model-)predicted point

[
f_\omega \doteq \hat{w}*{\mathrm{MAP}}^\top x*\omega
]

and obtain the label prediction

[
y_\omega \mid x_\omega, x_{1:n}, y_{1:n}
\sim
\mathcal{N}(f_\omega, \sigma_n^2).
\tag{2.14}
]

Here we observe that using point estimates such as the MAP estimate does not quantify uncertainty in the weights. The MAP estimate simply collapses all mass of the posterior around its mode. This can be harmful when we are highly unsure about the best model, e.g., because we have observed insufficient data.

Rather than selecting a single weight vector ( \hat{w} ) to make predictions, we can use the full posterior distribution. This is known as **Bayesian linear regression (BLR)**.

To make predictions at a test point ( x_\omega ), we let

[
f_\omega \doteq w^\top x_\omega
]

which has the distribution

[
f_\omega \mid x_\omega, x_{1:n}, y_{1:n}
\sim
\mathcal{N}(\mu^\top x_\omega, x_\omega^\top \Sigma x_\omega),
\tag{2.15}
]

using the closedness of Gaussians under linear transformations (1.78).

Note that this does not take into account the noise in the labels ( \sigma_n^2 ). For the label prediction ( y_\omega ), we obtain

[
y_\omega \mid x_\omega, x_{1:n}, y_{1:n}
\sim
\mathcal{N}(\mu^\top x_\omega, x_\omega^\top \Sigma x_\omega + \sigma_n^2),
\tag{2.16}
]

using additivity of Gaussians (1.79).

---

## 2.1.3 Recursive Probabilistic Inference

We have already discussed the recursive properties of probabilistic inference in Section 1.3.6. For Bayesian linear regression with a Gaussian prior and likelihood, this principle can be used to derive an efficient online algorithm since also the posterior is a Gaussian,

[
p^{(t)}(w) = \mathcal{N}(w; \mu^{(t)}, \Sigma^{(t)}),
\tag{2.17}
]

which can be stored efficiently using only ( \mathcal{O}(d^2) ) parameters. This leads to an efficient online algorithm for Bayesian linear regression with time-independent memory complexity ( \mathcal{O}(d^2) ) and round complexity ( \mathcal{O}(d^2) ).

The interpretation of Bayesian linear regression as an online algorithm also highlights similarities to other sequential models such as Kalman filters, which we discuss in Chapter 3. In Example 3.5, we will learn that online Bayesian linear regression is, in fact, an example of a Kalman filter.

---

## 2.2 Aleatoric and Epistemic Uncertainty

The predictive posterior distribution from Equation (2.16) highlights a decomposition of uncertainty wherein

* ( x_\omega^\top \Sigma x_\omega ) corresponds to the uncertainty about our model due to the lack of data (commonly referred to as **epistemic uncertainty**), and
* ( \sigma_n^2 ) corresponds to the uncertainty about the labels that cannot be explained by the inputs and any model from the model class (commonly referred to as **aleatoric uncertainty**, “irreducible noise”, or simply “(label) noise”).

A natural probabilistic approach is to represent epistemic uncertainty with a probability distribution over models. Intuitively, the variance of this distribution measures our uncertainty about the model and its mode corresponds to our current best (point) estimate.

It is a practical modeling choice how much inaccuracy to attribute to
epistemic or aleatoric uncertainty. Generally, when a poor model is
used to explain a process, more inaccuracy has to be attributed to irre￾ducible noise. For example, when a linear model is used to “explain”
a nonlinear process, most uncertainty is aleatoric as the model cannot
explain the data well. As we use more expressive models, a larger
portion of the uncertainty can be explained by the data.

Epistemic and aleatoric uncertainty can be formally defined in terms of the law of total variance (1.41),

[
\mathrm{Var}[y_\omega \mid x_\omega]
====================================

\mathbb{E}*\theta \big[ \mathrm{Var}*{y_\omega}[y_\omega \mid x_\omega, \theta] \big]
+
\mathrm{Var}*\theta \big[ \mathbb{E}*{y_\omega}[y_\omega \mid x_\omega, \theta] \big].
\tag{2.18}
]

Here, the mean variability of predictions y^* averaged across all mod￾els θ is the estimated aleatoric uncertainty. In contrast, the variability of
the mean prediction y^* under each model θ is the estimated epistemic
uncertainty. This decomposition of uncertainty will appear frequently
throughout this manuscript.

## 2.3 Non-linear Regression

We can use linear regression not only to learn linear functions. The trick is to apply a nonlinear transformation
(\phi : \mathbb{R}^d \to \mathbb{R}^e) to the features (x_i), where (d) is the dimension of the input space and (e) is the dimension of the designed feature space. We denote the design matrix comprised of transformed features by (\Phi \in \mathbb{R}^{n \times e}). Note that if the feature transformation (\phi) is the identity function then (\Phi = X).

*Figure 2.6: Applying linear regression with a feature space of polynomials of degree 10. The least squares estimate is shown in blue, ridge regression in red, and lasso in green.*

### Example 2.3: Polynomial regression

Let
[
\phi(x) = [x^2, x, 1] \quad \text{and} \quad w = [a, b, c].
]
Then the function that our model learns is given as
[
f = ax^2 + bx + c.
]

Thus, our model can exactly represent all polynomials up to degree 2.

However, to learn polynomials of degree (m) in (d) input dimensions, we need to apply the nonlinear transformation
[
\phi(x) = [1, x_1, \ldots, x_d, x_1^2, \ldots, x_d^2, x_1 x_2, \ldots, x_{d-1} x_d, \ldots, x_d^{m+1} \cdots x_d].
]

Note that the feature dimension
[
e = \sum_{i=0}^m \binom{d+i-1}{i} = \Theta(d^m).
]

Observe that the vector contains (\binom{d+i-1}{i}) monomials of degree (i) as this is the number of ways to choose (i) times from (d) items with replacement and without consideration of order. To see this, consider the following encoding: We take a sequence of (d+i-1) spots. Selecting any subset of (i) spots, we interpret the remaining (d-1) spots as “barriers” separating each of the (d) items. The selected spots correspond to the number of times each item has been selected. For example, if 2 items are to be selected out of a total of 4 items with replacement, one possible configuration is “● || ● |” where ● denotes a selected spot and | denotes a barrier. This configuration encodes that the first and third item have each been chosen once. The number of possible configurations — each encoding a unique outcome — is therefore (\binom{d+i-1}{i}).

Thus, the dimension of the feature space grows exponentially in the degree of polynomials and input dimensions. Even for relatively small (m) and (d), this becomes completely unmanageable.

The example of polynomials highlights that it may be inefficient to keep track of the weights (w \in \mathbb{R}^e) when (e) is large, and that it may be useful to instead consider a reparameterization which is of dimension (n) rather than of the feature dimension.

---

## 2.4 Function-space View

Let us now look at Bayesian linear regression through a different lens. Previously, we have been interpreting it as a distribution over the weights (w) of a linear function (f = \Phi w). The key idea is that for a finite set of inputs (ensuring that the design matrix is well-defined), we can equivalently consider a distribution directly over the estimated function values (f). We call this the function-space view of Bayesian linear regression.

Instead of considering a prior over the weights (w \sim \mathcal{N}(0, \sigma_p^2 I)) as we have done previously, we now impose a prior directly on the values of our model at the observations. Using that Gaussians are closed under linear maps (1.78), we obtain the equivalent prior
[
f \mid X \sim \mathcal{N}(\Phi \mathbb{E}[w], \Phi \operatorname{Var}[w] \Phi^\top)
= \mathcal{N}(0, \sigma_p^2 \Phi \Phi^\top)
\tag{2.19}
]
where (K \in \mathbb{R}^{n \times n}) is the so-called kernel matrix. Observe that the entries of the kernel matrix can be expressed as
[
K(i, j) = \sigma_p^2 , \phi(x_i)^\top \phi(x_j).
]

*Figure 2.7: An illustration of the function-space view. The model is described by the points ((x_i, f_i)).*

You may say that nothing has changed, and you would be right — that is precisely the point. Note, however, that the shape of the kernel matrix is (n \times n) rather than the (e \times e) covariance matrix over weights, which becomes unmanageable when (e) is large. The kernel matrix (K) has entries only for the finite set of observed inputs. However, in principle, we could have observed any input, and this motivates the definition of the kernel function
[
k(x, x') = \sigma_p^2 , \phi(x)^\top \phi(x')
\tag{2.20}
]
for arbitrary inputs (x) and (x').

A kernel matrix is simply a finite “view” of the kernel function,
[
K =
\begin{bmatrix}
k(x_1, x_1) & \cdots & k(x_1, x_n) \
\vdots & \ddots & \vdots \
k(x_n, x_1) & \cdots & k(x_n, x_n)
\end{bmatrix}.
\tag{2.21}
]

Observe that by definition of the kernel matrix in Equation (2.19), the kernel matrix is a covariance matrix and the kernel function measures the covariance of the function values (f(x)) and (f(x')) given inputs (x) and (x'):
[
k(x, x') = \operatorname{Cov}[f(x), f(x')].
\tag{2.22}
]

Moreover, note that we have reformulated the learning algorithm such that the feature space is now implicit in the choice of kernel, and the kernel is defined by inner products of (nonlinearly transformed) inputs. In other words, the choice of kernel implicitly determines the class of functions that (f) is sampled from (without expressing the functions explicitly in closed-form), which encodes our prior beliefs. This is known as the kernel trick.

---

### 2.4.1 Learning and Predictions

We have already kernelized the Bayesian linear regression prior. The posterior distribution (f \mid X, y) is again Gaussian due to the closedness properties of Gaussians, analogously to our derivation of the prior kernel matrix in Equation (2.19).

It remains to show that we can also rely on the kernel trick for predictions. Given the test point (x_\omega), we define
[
\tilde{\Phi} =
\begin{bmatrix}
\Phi \
\phi(x_\omega)^\top
\end{bmatrix}, \quad
\tilde{y} =
\begin{bmatrix}
y \
y_\omega
\end{bmatrix}, \quad
\tilde{f} =
\begin{bmatrix}
f \
f_\omega
\end{bmatrix}.
]

We immediately obtain (\tilde{f} = \tilde{\Phi} w). Analogously to our analysis of predictions from the weight-space view, we add the label noise to obtain the estimate (\tilde{y} = \tilde{f} + \tilde{\varepsilon}) where (\tilde{\varepsilon} = [\varepsilon_1 \cdots \varepsilon_n \ \varepsilon_\omega]^\top \sim \mathcal{N}(0, \sigma_n^2 I)) is the independent label noise. Applying the same reasoning as we did for the prior, we obtain
[
\tilde{f} \mid X, x_\omega \sim \mathcal{N}(0, \tilde{K})
\tag{2.23}
]
where (\tilde{K} = \sigma_p^2 \tilde{\Phi} \tilde{\Phi}^\top). Adding the label noise yields
[
\tilde{y} \mid X, x_\omega \sim \mathcal{N}(0, \tilde{K} + \sigma_n^2 I).
\tag{2.24}
]

Finally, we can conclude from the closedness of Gaussian random vectors under conditional distributions (1.53) that the predictive posterior (y_\omega \mid x_\omega, X, y) follows again a normal distribution. We will do a full derivation of the posterior and predictive posterior in Section 4.1.

---

### 2.4.2 Efficient Polynomial Regression

But how does the kernel trick address our concerns about efficiency raised in Section 2.3? After all, computing the kernel for a feature space of dimension (e) still requires computing sums of length (e), which is prohibitive when (e) is large. The kernel trick opens up a couple of new doors for us:

1. For certain feature transformations (\phi), we may be able to find an easier to compute expression equivalent to (\phi(x)^\top \phi(x')).
2. If this is not possible, we could approximate the inner product by an easier to compute expression.
3. Or, alternatively, we may decide not to care very much about the exact feature transformation and simply experiment with kernels that induce some feature space (which may even be infinitely dimensional).

We will explore the third approach when we revisit kernels in Section 4.3. A polynomial feature transformation can be computed efficiently in closed-form.

**Fact 2.4.** For the polynomial feature transformation (\phi) up to degree (m) from Example 2.3, it can be shown that up to constant factors,
[
\phi(x)^\top \phi(x') = (1 + x^\top x')^m.
\tag{2.25}
]

For example, for input dimension 2, the kernel ((1 + x^\top x')^2) corresponds to the feature vector
[
\phi(x) = [1, \sqrt{2}x_1, \sqrt{2}x_2, \sqrt{2}x_1 x_2, x_1^2, x_2^2]^\top.
]

---

## Discussion

We have explored a probabilistic perspective on linear models, and seen that classical approaches such as least squares and ridge regression can be interpreted as approximate probabilistic inference. We then saw that we can even perform exact probabilistic inference efficiently if we adopt a Gaussian prior and Gaussian noise assumption. These are already powerful tools, which are often applied also to nonlinear models if we treat the latent feature space — which was either human-designed or learned via deep learning — as fixed.

In the next chapter, we will digress briefly from the storyline on “learning” to see how we can adopt a similar probabilistic perspective to track latent states over time. Then, in Chapter 4, we will see how we can use the function-space view and kernel trick to learn flexible nonlinear models with exact probabilistic inference, without ever explicitly representing the feature space.

---

## Problems

### 2.1 Closed-form linear regression

1. Derive the unique solutions to least squares and ridge regression from Equations (2.4) and (2.5).
2. For an (n \times m) matrix (A) and vector (x \in \mathbb{R}^m), we call (\Pi_A x) the orthogonal projection of (x) onto (\operatorname{span}{A} = {Ax' \mid x' \in \mathbb{R}^m}). In particular, an orthogonal projection satisfies (x - \Pi_A x \perp Ax') for all (x' \in \mathbb{R}^m). Show that (\hat{w}*{\mathrm{ls}}) from Equation (2.4) is such that (X \hat{w}*{\mathrm{ls}}) is the unique closest point to (y) on (\operatorname{span}{X}), i.e., it satisfies (X \hat{w}_{\mathrm{ls}} = \Pi_X y).

### 2.2 MLE of noise variance

Show that the MLE of (\sigma_n^2) given fixed weights (w) is
[
\hat{\sigma}*n^2 = \frac{1}{n} \sum*{i=1}^n (y_i - w^\top x_i)^2.
\tag{2.26}
]

### 2.3 Variance of least squares around training data

Show that the variance of a prediction at the point ([1 \ x_\omega]^\top) is smallest when (x_\omega) is the mean of the training data. More formally, show that if inputs are of the form (x_i = [1 \ x_i]^\top) where (x_i \in \mathbb{R}) and (\hat{w}*{\mathrm{ls}}) is the least squares estimate, then
[
\operatorname{Var}[y*\omega \mid [1 \ x_\omega]^\top, \hat{w}*{\mathrm{ls}}]
]
is minimized for
[
x*\omega = \frac{1}{n} \sum_{i=1}^n x_i.
]

### 2.4 Bayesian linear regression

Suppose you are given the following observations
[
X =
\begin{bmatrix}
1 & 1 \
1 & 2 \
2 & 1 \
2 & 2
\end{bmatrix},
\quad
y =
\begin{bmatrix}
2.4 \
4.3 \
3.1 \
4.9
\end{bmatrix}
]
and assume the data follows a linear model with homoscedastic noise (\mathcal{N}(0, \sigma_n^2)) where (\sigma_n^2 = 0.1).

1. Find the maximum likelihood estimate (\hat{w}_{\mathrm{MLE}}) given the data.
2. Now assume that we have a prior (p(w) = \mathcal{N}(w; 0, \sigma_p^2 I)) with (\sigma_p^2 = 0.05). Find the MAP estimate (\hat{w}_{\mathrm{MAP}}) given the data and the prior.
3. Use the posterior (p(w \mid X, y)) to get a posterior prediction for the label (y_\omega) at (x_\omega = [3 \ 3]^\top). Report the mean and the variance of this prediction.
4. How would you have to change the prior (p(w)) such that (\hat{w}*{\mathrm{MAP}} \to \hat{w}*{\mathrm{MLE}})?

### 2.5 Online Bayesian linear regression

1. Can you design an algorithm that updates the posterior (as opposed to recalculating it from scratch using Equation (2.10)) in a smarter way? The requirement is that the memory should not grow as (O(t)).
2. If (d) is large, computing the inverse every round is very expensive. Can you use the recursive structure you found in the previous question to bring down the computational complexity of every round to (O(d^2))?

The resulting efficient online algorithm is known as online Bayesian linear regression.

### 2.6 Aleatoric and epistemic uncertainty of BLR

Prove for Bayesian linear regression that (x_\omega^\top \Sigma x_\omega) is the epistemic uncertainty and (\sigma_n^2) the aleatoric uncertainty in (y_\omega) under the decomposition of Equation (2.18).

### 2.7 Hyperpriors

We consider a dataset ({(x_i, y_i)}_{i=1}^n) of size (n), where (x_i \in \mathbb{R}^d) denotes the feature vector and (y_i \in \mathbb{R}) denotes the label of the (i)-th data point. Let (\varepsilon_i) be i.i.d. samples from the Gaussian distribution (\mathcal{N}(0, \lambda^{-1})) for a given (\lambda > 0). We collect the labels in a vector (y \in \mathbb{R}^n), the features in a matrix (X \in \mathbb{R}^{n \times d}), and the noise in a vector (\varepsilon \in \mathbb{R}^n). The labels are generated according to
[
y = Xw + \varepsilon.
]

To perform Bayesian Linear Regression, we consider the prior distribution over the parameter vector (w) to be (\mathcal{N}(\mu, \lambda^{-1} I_d)), where (I_d) denotes the (d)-dimensional identity matrix and (\mu \in \mathbb{R}^d) is a hyperparameter.

1. Given this Bayesian data model, what is the conditional covariance matrix
   [
   \Sigma_y = \operatorname{Var}[y \mid X, \mu, \lambda]?
   ]
2. Calculate the maximum likelihood estimate of the hyperparameter (\mu).
3. Since we are unsure about the hyperparameter (\mu), we decide to model our uncertainty about (\mu) by placing the “hyperprior” (\mu \sim \mathcal{N}(0, I_d)). Is the posterior distribution (p(\mu \mid X, y, \lambda)) a Gaussian distribution? If yes, what are its mean vector and covariance matrix?
4. What is the posterior distribution (p(\lambda \mid X, y, \mu))?

*Hint:* For any (a \in \mathbb{R}), (A \in \mathbb{R}^{n \times n}) it holds that (\det(aA) = a^n \det(A)).

---