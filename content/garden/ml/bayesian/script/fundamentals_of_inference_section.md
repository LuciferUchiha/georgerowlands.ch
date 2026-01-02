# Fundamentals of Inference

## 1.2 Probabilistic Inference


Recall the logical implication “If it is raining, the ground is wet.” from the beginning of this chapter. Suppose that we look outside a window and see that it is not raining: will the ground be dry? Logical reasoning does not permit drawing an inference of this kind, as there might be reasons other than rain for which the ground could be wet (e.g., sprinklers). However, intuitively, by observing that it is not raining, we have just excluded the possibility that the ground is wet because of rain, and therefore we would deem it “more likely” that the ground is dry than before. In other words, if we were to walk outside now and the ground was wet, we would be more surprised than we would have been if we had not looked outside the window before.

As humans, we are constantly making such “plausible” inferences of our beliefs: be it about the weather, the outcomes of our daily decisions, or the behavior of others. Probabilistic inference is the process of updating such a prior belief ( P[W] ) to a posterior belief ( P[W \mid R] ) upon observing ( R ), where—to reduce clutter—we write ( W ) for “The ground is wet” and ( R ) for “It is raining”.

The computational rule of probabilistic inference is **Bayes’ rule**.

---

### Theorem 1.18 (Bayes’ rule)

Given random vectors ( X \in \mathbb{R}^n ) and ( Y \in \mathbb{R}^m ), we have for any ( x \in \mathbb{R}^n ), ( y \in \mathbb{R}^m ) that

[
p(x \mid y) = \frac{p(y \mid x), p(x)}{p(y)}. \tag{1.45}
]

---

### Proof

Bayes’ rule is a direct consequence of the definition of conditional densities (1.10) and the product rule (1.11).

It is useful to consider the meaning of each term separately:

* the posterior ( p(x \mid y) ) is the updated belief about ( x ) after observing ( y ),
* the prior ( p(x) ) is the initial belief about ( x ),
* the (conditional) likelihood ( p(y \mid x) ) describes how likely the observations ( y ) are under a given value ( x ),
* the joint likelihood ( p(x, y) = p(y \mid x)p(x) ) combines prior and likelihood,
* the marginal likelihood ( p(y) ) describes how likely the observations ( y ) are across all values of ( x ).

The marginal likelihood can be computed using the sum rule (1.7) or the law of total probability (1.12),
[
p(y) = \int_{\Omega_X} p(y \mid x), p(x), dx. \tag{1.46}
]

Note, however, that the marginal likelihood is simply normalizing the conditional distribution to integrate to one, and therefore a constant with respect to ( x ). For this reason, ( p(y) ) is commonly called the **normalizing constant**.

---

### Example 1.19: Plausible inferences

Let us confirm our intuition from the above example. The logical implication “If it is raining, the ground is wet.” (denoted ( R \rightarrow W )) can be succinctly expressed as ( P(W \mid R) = 1 ). Since ( P(W) \le 1 ), we know that

[
P(R \mid W) = \frac{P(W \mid R), P(R)}{P(W)} = \frac{P(R)}{P(W)} \ge P(R).
]

That is, observing that the ground is wet makes it more likely to be raining.

From ( P(R \mid W) \ge P(R) ) we know ( P[R \mid W] \ge P[R] ), which leads us to conclude that

[
P[W \mid \neg R] = \frac{P[R \mid W], P(W)}{P[R]} \le P(W),
]

that is, having observed it not to be raining made the ground less likely to be wet.

Example 1.19 is called a **plausible inference** because the observation of ( R ) does not completely determine the truth value of ( W ), and hence, does not permit logical inference. In the case, however, that logical inference is permitted, it coincides with probabilistic inference.

---

### Example 1.20: Logical inferences

For example, if we were to observe that the ground is not wet, then logical inference implies that it must not be raining: ( \neg W \rightarrow \neg R ). This is called the **contrapositive** of ( R \rightarrow W ).

Indeed, by probabilistic inference, we obtain analogously
[
P[\neg R \mid \neg W] = \frac{P[\neg W \mid \neg R], P(\neg R)}{P[\neg W]} = 0,
]
as ( P(W \mid R) = 1 ).

Observe that a logical inference does not depend on the prior ( P(R) ): even if the prior was ( P(R)=1 ), after observing that the ground is not wet, we are forced to conclude that it is not raining to maintain logical consistency.

The examples highlight that while logical inference does not require the notion of a prior, plausible (probabilistic) inference does.

---

## 1.2.1 Where do priors come from?

The computational rule of probabilistic inference as shown in Equation (1.45) necessitates the specification of a prior ( p(x) ). Different priors can lead to the deduction of dramatically different posteriors, as one can easily see by considering the extreme cases of a prior that is a point density at ( x=x_0 ) and a prior that is “uniform” over ( \mathbb{R}^n ).

In the former case, the posterior will be a point density at ( x_0 ) regardless of the likelihood. In other words, no evidence can alter the “prior belief” the learner ascribed to ( x ). In the latter case, the learner has “no prior belief”, and therefore the posterior will be proportional to the likelihood.

Someone who follows the Bayesian interpretation of probability might argue that everything is conditional, meaning that the prior is simply a posterior of all former observations. While this might seem natural, this lacks an explanation for “the first day”.

Someone else who is more inclined towards the frequentist interpretation might object to the existence of a prior belief altogether, arguing that a prior is subjective and therefore not a valid or desirable input to a learning algorithm.

Put differently, a frequentist “has the belief not to have any belief”. This is perfectly compatible with probabilistic inference, as long as the prior is chosen to be **noninformative**:
[
p(x) \propto \text{const}. \tag{1.47}
]

---

### Principle of Indifference

Choosing a noninformative prior in the absence of any evidence is known as the **principle of indifference** or the **principle of insufficient reason**, which dates back to Pierre-Simon Laplace.

---

### Example 1.21: Why be indifferent?

Consider a criminal trial with three suspects, A, B, and C. The collected evidence shows that suspect C cannot have committed the crime, however it does not yield any information about suspects A and B. Clearly, any distribution respecting the data must assign zero probability to suspect C. However, any distribution interpolating between ( (1,0,0) ) and ( (0,1,0) ) respects the data.

The principle of indifference suggests that the desired distribution is ( (1/2, 1/2, 0) ), and indeed, any alternative distribution seems unreasonable.

---

### Remark 1.22: Noninformative and improper priors

It is not necessarily required that the prior ( p(x) ) is a valid distribution (i.e., integrates to 1). Consider for example the noninformative prior ( p(x) \propto \mathbb{1}{x \in I} ) where ( I \subset \mathbb{R}^n ) is an infinitely large interval. Such a prior is called an **improper prior**.

We can still derive meaning from the posterior of a given likelihood and improper prior as long as the posterior is a valid distribution.

Laplace’s principle of indifference can be generalized to cases where some evidence is available. The **maximum entropy principle**, originally proposed by Jaynes (1968), states that one should choose as prior from all possible distributions that are consistent with prior knowledge, the one that makes the least “additional assumptions”, i.e., is the least “informative”. In philosophy, this principle is known as **Occam’s razor** or the **principle of parsimony**. The “informativeness” of a distribution ( p ) is quantified by its entropy which is defined as

[
H[p] ; .= ; \mathbb{E}_{x \sim p}[-\log p(x)]. \tag{1.48}
]

The more concentrated ( p ) is, the less is its entropy; the more diffuse ( p ) is, the greater is its entropy.
We give a thorough introduction to entropy in Section 5.4.

In the absence of any prior knowledge, the uniform distribution has the highest entropy, and hence, the maximum entropy principle suggests a noninformative prior (as does Laplace’s principle of indifference). This only holds true when the set of possible outcomes of ( x ) is finite (or a bounded continuous interval), as in this case the noninformative prior is a proper distribution—the uniform distribution. In the “infinite case”, there is no uniform distribution and the noninformative prior can be attained from the maximum entropy principle as the limiting solution as the number of possible outcomes of ( x ) is increased.

In contrast, if the evidence perfectly determines the value of ( x ), then the only consistent explanation is the point density at ( x ). The maximum entropy principle characterizes a reasonable choice of prior for these two extreme cases and all cases in between.

The computational rule of probabilistic inference (1.45) can in fact be derived as a consequence of the maximum entropy principle in the sense that the posterior is the least “informative” distribution among all distributions that are consistent with the prior and the observations.
(Problem 5.7)

---

## 1.2.2 Conjugate Priors

If the prior ( p(x) ) and posterior ( p(x \mid y) ) are of the same family of distributions, the prior is called a **conjugate prior** to the likelihood ( p(y \mid x) ).

This is a very desirable property, as it allows us to recursively apply the same learning algorithm implementing probabilistic inference. We will see in Chapter 2 that under some conditions the Gaussian is self-conjugate. That is, if we have a Gaussian prior and a Gaussian likelihood then our posterior will also be Gaussian. This will provide us with the first efficient implementation of probabilistic inference.

---

### Example 1.23: Conjugacy of beta and binomial distribution

As an example for conjugacy, we will show that the beta distribution is a conjugate prior to a binomial likelihood. Recall the PMF of the binomial distribution

[
\mathrm{Bin}(k; n, \theta) = \binom{n}{k}\theta^k(1-\theta)^{n-k}, \tag{1.49}
]

and the PDF of the beta distribution,

[
\mathrm{Beta}(\theta; \alpha, \beta) \propto \theta^{\alpha-1}(1-\theta)^{\beta-1}. \tag{1.50}
]

We assume the prior ( \theta \sim \mathrm{Beta}(\alpha, \beta) ) and likelihood ( k \mid \theta \sim \mathrm{Bin}(n, \theta) ).
Let ( n_H = k ) be the number of heads and ( n_T = n-k ) the number of tails in the binomial trial.

Then,
[
\begin{aligned}
p(\theta \mid k)
&\propto p(k \mid \theta)p(\theta) \quad \text{using Bayes’ rule (1.45)} \
&\propto \theta^{n_H}(1-\theta)^{n_T}\theta^{\alpha-1}(1-\theta)^{\beta-1} \
&= \theta^{\alpha+n_H-1}(1-\theta)^{\beta+n_T-1}.
\end{aligned}
]

Thus,
[
\theta \mid k \sim \mathrm{Beta}(\alpha+n_H,; \beta+n_T).
]

This same conjugacy can be shown for the multivariate generalization of the beta distribution, the **Dirichlet distribution**, and the multivariate generalization of the binomial distribution, the **multinomial distribution**.

---

## 1.2.3 Tractable Inference with the Normal Distribution

Using arbitrary distributions for learning and inference is computationally very expensive when the number of dimensions is large—even in the discrete setting. For example, computing marginal distributions using the sum rule yields an exponentially long sum in the size of the random vector. Similarly, the normalizing constant of the conditional distribution is a sum of exponential length. Even to represent any discrete joint probability distribution requires space that is exponential in the number of dimensions (cf. Figure 1.5).

**Figure 1.5:** A table representing a joint distribution of ( n ) binary random variables. The table has ( 2^n ) rows. The number of parameters is ( 2^n-1 ) since the final probability is determined by all other probabilities as they must sum to one.

One strategy to get around this computational blowup is to restrict the class of distributions. Gaussians are a popular choice for this purpose since they have extremely useful properties: they have a compact representation and—as we will see in Chapter 2—they allow for closed-form probabilistic inference.

In Equation (1.5), we have already seen the PDF of the univariate Gaussian distribution. A random vector ( X \in \mathbb{R}^n ) is normally distributed, ( X \sim \mathcal{N}(\mu, \Sigma) ), if its PDF is

[
\mathcal{N}(x;\mu,\Sigma)
; .= ;
\frac{1}{\sqrt{\det(2\pi\Sigma)}}
\exp!\left(
-\frac{1}{2}(x-\mu)^\top \Sigma^{-1}(x-\mu)
\right). \tag{1.51}
]

Here, ( \mu \in \mathbb{R}^n ) is the mean vector and ( \Sigma \in \mathbb{R}^{n\times n} ) the covariance matrix. We call ( \Lambda = \Sigma^{-1} ) the **precision matrix**.
( X ) is also called a **Gaussian random vector (GRV)**.
( \mathcal{N}(0, I) ) is the multivariate standard normal distribution.

We call a Gaussian **isotropic** if its covariance matrix is of the form ( \Sigma = \sigma^2 I ) for some ( \sigma^2 \in \mathbb{R} ). In this case, the sublevel sets of the PDF are perfect spheres as can be seen in Figure 1.6.

**Figure 1.6:** PDFs of two-dimensional Gaussians with mean 0 and covariance matrices
[
\Sigma_1 = \begin{pmatrix}1&0\0&1\end{pmatrix},\qquad
\Sigma_2 = \begin{pmatrix}1&0.9\0.9&1\end{pmatrix}.
]

Note that a Gaussian can be represented using only ( O(n^2) ) parameters. In the case of a diagonal covariance matrix, which corresponds to ( n ) independent univariate Gaussians, we just need ( O(n) ) parameters.

---

### Properties of the Gaussian

In Equation (1.51), we assume that the covariance matrix ( \Sigma ) is invertible, i.e., does not have eigenvalue 0. This is not a restriction since it can be shown that a covariance matrix has a zero eigenvalue if and only if there exists a deterministic linear relationship between some variables in the joint distribution. As we have already seen that a covariance matrix does not have negative eigenvalues, this ensures that ( \Sigma ) and ( \Lambda ) are positive definite.

An important property of the normal distribution is that it is **closed under marginalization and conditioning**.

---

### Theorem 1.24 (Marginal and conditional distribution)

Consider the Gaussian random vector ( X ) and fix index sets ( A \subseteq [n] ) and ( B \subseteq [n] ). Then, for any such marginal distribution,

[
X_A \sim \mathcal{N}(\mu_A,\Sigma_{AA}), \tag{1.52}
]

and for any such conditional distribution,

[
X_A \mid X_B=x_B \sim \mathcal{N}(\mu_{A\mid B},\Sigma_{A\mid B}),
]

where

[
\mu_{A\mid B} = \mu_A + \Sigma_{AB}\Sigma_{BB}^{-1}(x_B-\mu_B), \tag{1.53b}
]
[
\Sigma_{A\mid B} = \Sigma_{AA} - \Sigma_{AB}\Sigma_{BB}^{-1}\Sigma_{BA}. \tag{1.53c}
]

Theorem 1.24 provides a closed-form characterization of probabilistic inference for the case that random variables are jointly Gaussian.

Observe that upon inference, the variance can only shrink. Moreover, how much the variance is reduced depends purely on where the observations are made (the choice of ( B )) but not on what the observations are. In contrast, the posterior mean depends affinely on ( x_B ).

It can be shown that Gaussians are additive and closed under affine transformations. The closedness under affine transformations implies that a Gaussian ( X \sim \mathcal{N}(\mu,\Sigma) ) is equivalently characterized as

[
X = \Sigma^{1/2}Y + \mu, \tag{1.54}
]

where ( Y \sim \mathcal{N}(0,I) ).

Importantly, this implies together with Theorem 1.24 and additivity that:

> **Any affine transformation of a Gaussian random vector is a Gaussian random vector.**

---

## 1.3 Supervised Learning and Point Estimates

Throughout the first part of this manuscript, we will focus mostly on the supervised learning problem where we want to learn a function
[
f^\star : \mathcal{X} \to \mathcal{Y}
]
from labeled training data.

That is, we are given a collection of labeled examples
[
\mathcal{D}*n := {(x_i,y_i)}*{i=1}^n,
]
where ( x_i \in \mathcal{X} ) are inputs and ( y_i \in \mathcal{Y} ) are outputs (labels), and we want to find a function ( \hat f ) that best approximates ( f^\star ).

It is common to choose ( \hat f ) from a parameterized function class ( \mathcal{F}(\Theta) ), where each function ( f_\theta ) is described by parameters ( \theta \in \Theta ).

**Figure 1.7:** Illustration of estimation error and approximation error.
( f^\star ) denotes the true function and ( \hat f ) is the best approximation from the function class ( \mathcal{F} ).

---

## Remark 1.25: What this manuscript is about and not about

As illustrated in fig. 1.7, the restriction to a function class leads to
two sources of error: the estimation error of having “incorrectly”
determined (\hat f) within the function class, and the approximation error
of the function class itself. Choosing a “good” function class
/ architecture with small approximation error is therefore critical
for any practical application of machine learning. We will discuss
various function classes, from linear models to deep neural net-
works, however, determining the “right” function class will not
be the focus of this manuscript. To keep the exposition simple,
we will assume in the following that (f^\star \in \mathcal F(\Theta)) with parameters
(\theta^\star \in \Theta).

Instead, we will focus on the problem of estimation/inference
within a given function class. We will see that inference in smaller
function classes is often more computationally efficient since the
search space is smaller or — in the case of Gaussians knows a
known tractable structure. On the other hand, larger function
classes are more expressive and therefore can typically better ap-
proximate the ground truth (f^\star).

We differentiate between the task of regression where
[
Y := \mathbb R^k,
]
and the task of classification where
[
Y := \mathcal C
]
and (\mathcal C) is an (m)-element set of classes. In other words, regression
is the task of predicting a continuous label, whereas classification
is the task of predicting a discrete class label. These two tasks are
intimately related: in fact, we can think of classification tasks as a
regression problem where we learn a probability distribution over
class labels. In this regression problem,
[
Y := \Delta^{\mathcal C}
]
where (\Delta^{\mathcal C}) denotes the set of all probability distributions over the set
of classes (\mathcal C) which is an ((m-1))-dimensional convex polytope in the
(m)-dimensional space of probabilities ([0,1]^m) (cf. Appendix A.1.2).

For now, let us stick to the regression setting. We will assume that
the observations are noisy, that is,
[
y_i \stackrel{\text{iid}}{\sim} p(\cdot \mid x_i, \theta^\star)
]
for some known conditional distribution (p(\cdot \mid x_i, \theta)) but unknown
parameter (\theta^\star). Our assumption can equivalently be formulated as
[
y_i = f_\theta(x_i) ;+; \varepsilon_i(x_i)
\tag{1.56}
]
where (f_\theta(x_i)) is the mean of (p(\cdot \mid x_i, \theta)) and
(\varepsilon_i(x_i) = y_i - f_\theta(x_i)) is some independent zero-mean noise,
for example (but not necessarily) Gaussian. When the noise distribution may
depend on (x_i), the noise is said to be **heteroscedastic** and otherwise the
noise is called **homoscedastic**.

---

## 1.3.1 Maximum Likelihood Estimation

A common approach to finding (\hat f) is to select the model (f \in \mathcal F(\Theta)) under
which the training data is most likely. This is called the **maximum likelihood
estimate (or MLE)**:
[
\hat\theta_{\text{MLE}}
:= \arg\max_{\theta \in \Theta} p(y_{1:n} \mid x_{1:n}, \theta)
\tag{1.57}
]
[
= \arg\max_{\theta \in \Theta} \prod_{i=1}^n p(y_i \mid x_i, \theta),
]
using the independence of the training data (1.56).

Such products of probabilities are often numerically unstable, which
is why one typically takes the logarithm:
[
= \arg\max_{\theta \in \Theta} \sum_{i=1}^n \log p(y_i \mid x_i, \theta)
\quad \text{(log-likelihood).}
\tag{1.58}
]

We will denote the negative log-likelihood by (\varepsilon_{\text{nll}}(\theta; \mathcal D_n)).

The MLE is often used in practice due to its desirable asymptotic
properties as the sample size (n) increases. We give a brief summary
here and provide additional background and definitions in Appendix A.3.

To give any guarantees on the convergence of the MLE, we necessarily
need to assume that (\theta^\star) is **identifiable**. If additionally,
(\varepsilon_{\text{nll}}) is “well-behaved” then standard results say that the
MLE is **consistent** and **asymptotically normal** (Van der Vaart, 2000):
[
\hat\theta_{\text{MLE}} \xrightarrow{P} \theta^\star
\quad\text{and}\quad
\hat\theta_{\text{MLE}} \xrightarrow{D} \mathcal N(\theta^\star, S_n)
\quad \text{as } n \to \infty.
\tag{1.59}
]

Here, we denote by (S_n) the asymptotic variance of the MLE which can
be understood as measuring the “quality” of the estimate. This implies
in some sense that the MLE is asymptotically unbiased. Moreover, the
MLE can be shown to be **asymptotically efficient**, which is to say that
there exists no other consistent estimator with a “smaller” asymptotic
variance.

The situation is quite different in the finite sample regime. Here, the
MLE need not be unbiased, and it is susceptible to overfitting to the
(finite) training data as we discuss in more detail in Appendix A.3.5.

---

## 1.3.2 Using Priors: Maximum a Posteriori Estimation

We can incorporate prior assumptions about the parameters (\theta^\star) into the
estimation procedure. One approach of this kind is to find the **mode of the
posterior distribution**, called the **maximum a posteriori estimate (or MAP
estimate)**:
[
\hat\theta_{\text{MAP}}
:= \arg\max_{\theta \in \Theta} p(\theta \mid x_{1:n}, y_{1:n})
\tag{1.60}
]
[
= \arg\max_{\theta \in \Theta} p(y_{1:n} \mid x_{1:n}, \theta), p(\theta)
\tag{1.61}
]
by Bayes’ rule (1.45)
[
= \arg\max_{\theta \in \Theta}
\log p(\theta) + \sum_{i=1}^n \log p(y_i \mid x_i, \theta)
\tag{1.62}
]
[
= \arg\min_{\theta \in \Theta}
\Big(

* \log p(\theta)

- \varepsilon_{\text{nll}}(\theta; \mathcal D_n)
  \Big).
  \tag{1.63}
  ]

Here, the log-prior (\log p(\theta)) acts as a **regularizer**. Common regularizers
are given, for example, by

* (p(\theta) = \mathcal N(\theta; 0, (2\lambda)^{-1} I)), which yields
  (-\log p(\theta) = \lambda \lVert \theta \rVert_2^2 + \text{const}),

* (p(\theta) = \text{Laplace}(\theta; 0, \lambda^{-1})), which yields
  (-\log p(\theta) = \lambda \lVert \theta \rVert_1 + \text{const}),

* a uniform prior (cf. Section 1.2.1) for which the MAP is equivalent
  to the MLE.

The Gaussian and Laplace regularizers act as **simplicity biases**, preferring
simpler models over more complex ones, which empirically tends to reduce
the risk of overfitting. However, one may also encode more nuanced
information about the (assumed) structure of (\theta^\star) into the prior.

An alternative way of encoding a prior is by restricting the function
class to some (\Theta' \subseteq \Theta), for example to rotation- and
translation-invariant models as often done when the inputs are images.
This effectively sets (p(\theta)=0) for all (\theta \in \Theta \setminus \Theta') but
is better suited for numerical optimization than to impose this constraint
directly on the prior.

Encoding prior assumptions into the function class or into the parameter
estimation can accelerate learning and improve generalization performance
dramatically, yet importantly, incorporating a prior can also inhibit
learning in case the prior is “wrong”.

---

## 1.3.3 When does the prior matter?

We have seen that the MLE has desirable asymptotic properties, and that
MAP estimation can be seen as a regularized MLE where the type of
regularization is encoded by the prior. Is it possible to derive similar
asymptotic results for the MAP estimate?

To answer this question, we will look at the asymptotic effect of the
prior on the posterior more generally. **Doob’s consistency theorem**
states that assuming parameters are identifiable, there exists
(\Theta' \subseteq \Theta) with (p(\Theta') = 1) such that the posterior is
consistent for any (\theta^\star \in \Theta'):
[
\theta \mid \mathcal D_n \xrightarrow{P} \theta^\star
\quad \text{as } n \to \infty.
\tag{1.64}
]

In words, Doob’s consistency theorem tells us that for any prior
distribution, the posterior is guaranteed to converge to a point density
in a (small) neighborhood of the true parameter as long as the prior
assigns non-zero mass to that neighborhood.

---

## Remark 1.27: Cromwell’s rule

In the case where (|\Theta|) is finite, Doob’s consistency theorem strongly
suggests that the prior should not assign probability 0 (or probability 1
for that matter) to any individual parameter (\theta \in \Theta), unless we
know with certainty that (\theta^\star = \theta). This is called **Cromwell’s
rule**, and a prior obeying this rule is always well-specified.

Under the same assumption that the prior is well-specified (and
regularity conditions), the **Bernstein–von Mises theorem** establishes the
asymptotic normality of the posterior distribution:
[
\theta \mid \mathcal D_n \xrightarrow{D} \mathcal N(\theta^\star, S_n)
\quad \text{as } n \to \infty.
\tag{1.65}
]

These results link probabilistic inference to maximum likelihood
estimation in the asymptotic limit of infinite data. Intuitively, in the
limit of infinite data, the prior is “overwhelmed” by the observations.

---

## 1.3.4 Estimation vs Inference

You can interpret a single parameter vector (\theta \in \Theta) as “one possible
explanation” of the data. Maximum likelihood and maximum a posteriori
estimation are examples of **estimation algorithms** which return one such
parameter vector — called a **point estimate**.

### Example 1.28: Point estimates and invalid logical inferences

To see why point estimates can be problematic, recall Example 1.20.
We have seen that the logical implication

> “If it is raining, the ground is wet.”

can be expressed as (P(W \mid R)=1). Observing that “the ground is wet”
does not permit logical inference, yet, the maximum likelihood estimate
of (R) is (\hat R_{\text{MLE}}=1). This is logically inconsistent since there
might be other explanations for the ground to be wet, such as a sprinkler.

In practice, we never observe an infinite amount of data. Example 1.28
demonstrates that on a finite sample, point estimates may perform
invalid logical inferences, and can therefore lure us into a false sense
of certainty.

---

## Remark 1.29: MLE and MAP are approximations of inference

The MLE and MAP estimate can be seen as a naive approximation of
probabilistic inference, represented by a point density which collapses
all probability mass at the mode of the posterior distribution.

In this manuscript, we will focus mainly on algorithms for probabilistic
inference which compute or approximate the distribution
(p(\theta \mid x_{1:n}, y_{1:n})) over parameters.

---

## 1.3.5 Probabilistic Inference and Prediction

The prior distribution (p(\theta)) can be interpreted as the degree of our
belief that the model parameterized by (\theta) “describes the (previously
seen) data best”. The likelihood captures how likely the training data is
under a particular model:
[
p(y_{1:n} \mid x_{1:n}, \theta)
= \prod_{i=1}^n p(y_i \mid x_i, \theta).
\tag{1.66}
]

The posterior then represents our belief about the best model after
seeing the training data:
[
p(\theta \mid x_{1:n}, y_{1:n})
= \frac{1}{Z} p(\theta)\prod_{i=1}^n p(y_i \mid x_i, \theta),
\tag{1.67a}
]
with
[
Z = \int_\Theta p(\theta)\prod_{i=1}^n p(y_i \mid x_i, \theta), d\theta.
\tag{1.67b}
]

We can then use our learned model for prediction at a new input (x^\star):
[
p(y^\star \mid x^\star, x_{1:n}, y_{1:n})
= \int_\Theta p(y^\star \mid x^\star, \theta), p(\theta \mid x_{1:n}, y_{1:n}), d\theta.
\tag{1.68}
]

The distribution (p(y^\star \mid x^\star, x_{1:n}, y_{1:n})) is called the
**predictive posterior**.

---

## 1.3.6 Recursive Probabilistic Inference and Memory

We have already alluded to the fact that probabilistic inference has a
recursive structure. Let
[
p^{(t)}(\theta) := p(\theta \mid x_{1:t}, y_{1:t}),
\tag{1.70}
]
with (p^{(0)}(\theta)=p(\theta)). Then
[
p^{(t+1)}(\theta)
\propto p^{(t)}(\theta), p(y_{t+1} \mid \theta).
\tag{1.71}
]

Intuitively, the posterior distribution at time (t) “absorbs” or
“summarizes” all seen data.

---

## 1.4 Outlook: Decision Theory

How can we use our predictions to make concrete decisions under
uncertainty? Making decisions using a probabilistic model (p(y \mid x)) is
commonly formalized by

* a set of possible actions (\mathcal A), and
* a reward function (r(y,a)\in\mathbb R).

Standard decision theory recommends picking the action with the largest
expected utility:
[
a^\star(x)
:= \arg\max_{a\in\mathcal A} \mathbb E_{y\mid x}[r(y,a)].
\tag{1.72}
]

Here, a^\star is called the optimal decision rule because, under the given
probabilistic model, no other rule can yield a higher expected utility

Let us consider some examples of reward functions and their corresponding optimal decisions:

## Example 1.30: Reward functions

Under the decision rule from Equation (1.72), different reward
functions (r) can lead to different decisions. Let us examine two
reward functions for the case where (Y = A = \mathbb R).

* Alternatively to considering (r) as a reward function, we can
  interpret (-r) as the loss of taking action (a) when the true output
  is (y). If our goal is for our actions (a) to “mimic” the output (y), a
  natural choice is the squared loss,
  [
  -r(y,a) = (y-a)^2.
  ]
  It turns out that under the squared loss, the optimal decision is
  simply the mean:
  [
  a^\star(x) = \mathbb E[y \mid x].
  ]

* To contrast this, we consider the asymmetric loss,
  [
  -r(y,a) =
  c_1 \max{y-a,0}
  \quad\text{(underestimation error)}
  +
  c_2 \max{a-y,0}
  \quad\text{(overestimation error)},
  ]
  which penalizes underestimation and overestimation differently.

  When (y \mid x \sim \mathcal N(\mu_x, \sigma_x^2)), then the optimal decision is
  [
  a^\star(x)
  = \mu_x + \sigma_x \cdot \Phi^{-1}!\left(\frac{c_1}{c_1 + c_2}\right),
  ]
  where (\Phi) is the CDF of the standard normal distribution.

  Note that if (c_1 = c_2), then the second term vanishes and the optimal
  decision is the same as under the squared loss. If (c_1 > c_2), the
  second term is positive (i.e., optimistic) to avoid underestimation,
  and if (c_1 < c_2), the second term is negative (i.e., pessimistic) to
  avoid overestimation. We will find these notions of optimism and
  pessimism to be useful in many decision-making scenarios.

While Equation (1.72) describes how to make optimal decisions given a
(posterior) probabilistic model, it does not tell us how to learn or
improve this model in the first place. That is, these decisions are only
optimal under the assumption that we cannot use their outcomes and
our resulting observations to update our model and inform future
decisions. When we start to consider the effect of our decisions on
future data and future posteriors, answering “how do I make optimal
decisions?” becomes more complex, and we will study this in Part II on
sequential decision-making.

---

## Discussion

In this chapter, we have learned about the fundamental concepts of
probabilistic inference. We have seen that probabilistic inference is the
natural extension of logical reasoning to domains with uncertainty. We
have also derived the central principle of probabilistic inference, Bayes’
rule, which is simple to state but often computationally challenging.

In the next part of this manuscript, we will explore settings where exact
inference is tractable, as well as modern approaches to approximate
probabilistic inference.

---

## Overview of Mathematical Background

We have included brief summaries of the fundamentals of parameter
estimation (mean estimation in particular) and optimization in
Appendices A.3 and A.4, respectively, which we will refer back to
throughout the manuscript. Appendix A.2 discusses the correspondence
of Gaussians and quadratic forms. Appendix A.5 comprises a list of useful
matrix identities and inequalities.

---

## Problems

### 1.1 Properties of probability

Let ((\Omega, \mathcal A, P)) be a probability space. Derive the following
properties of probability from the Kolmogorov axioms:

1. For any (A,B \in \mathcal A), if (A \subseteq B) then (P(A) \le P(B)).
2. For any (A \in \mathcal A),
   [
   P(A^c) = 1 - P(A).
   ]
3. For any countable set of events ({A_i \in \mathcal A}*i),
   [
   P!\left(\bigcup*{i=1}^{\infty} A_i\right)
   \le \sum_{i=1}^{\infty} P(A_i),
   \tag{1.73}
   ]
   which is called a union bound.

---

### 1.2 Random walks on graphs

Let (G) be a simple connected finite graph. We start at a vertex (u) of
(G). At every step, we move to one of the neighbors of the current
vertex uniformly at random, e.g., if the vertex has 3 neighbors, we move
to one of them, each with probability (1/3). What is the probability that
the walk visits a given vertex (v) eventually?

---

### 1.3 Law of total expectation

Show that if ({A_i}*{i=1}^k) are a partition of (\Omega) and (X) is a random
vector,
[
\mathbb E[X]
= \sum*{i=1}^k \mathbb E[X \mid A_i] \cdot P(A_i).
\tag{1.74}
]

---

### 1.4 Covariance matrices are positive semi-definite

Prove that a covariance matrix (\Sigma) is always positive semi-definite.
That is, all of its eigenvalues are greater or equal to zero, or
equivalently,
[
x^\top \Sigma x \ge 0
\quad\text{for any } x \in \mathbb R^n.
]

---

### 1.5 Probabilistic inference

As a result of a medical screening, one of the tests revealed a serious
disease in a person. The test has a high accuracy of 99% (the probability
of a positive response in the presence of a disease is 99% and the
probability of a negative response in the absence of a disease is also
99%). However, the disease is quite rare and occurs only in one person
per 10 000. Calculate the probability of the examined person having the
identified disease.

---

### 1.6 Zero eigenvalues of covariance matrices

We say that a random vector (X \in \mathbb R^n) is not linearly independent
if for some (\varepsilon \in \mathbb R^n \setminus {0}), (\varepsilon^\top X = 0).

1. Show that if (X) is not linearly independent, then (\operatorname{Var}[X])
   has a zero eigenvalue.
2. Show that if (\operatorname{Var}[X]) has a zero eigenvalue, then (X) is not
   linearly independent.

*Hint:* Consider the variance of (\rho^\top X) where (\rho) is the eigenvector
corresponding to the zero eigenvalue.

---

### 1.7 Product of Gaussian PDFs

Let (\mu_1,\mu_2 \in \mathbb R^n) be mean vectors and
(\Sigma_1,\Sigma_2 \in \mathbb R^{n\times n}) be covariance matrices. Prove that
[
\mathcal N(x;\mu,\Sigma)
\propto
\mathcal N(x;\mu_1,\Sigma_1)\cdot \mathcal N(x;\mu_2,\Sigma_2)
\tag{1.75}
]
for some mean vector (\mu \in \mathbb R^n) and covariance matrix
(\Sigma \in \mathbb R^{n\times n}).

---

### 1.8 Independence of Gaussians

Show that two jointly Gaussian random vectors, (X) and (Y), are independent
if and only if (X) and (Y) are uncorrelated.

---

### 1.9 Marginal / conditional distribution of a Gaussian

Prove Theorem 1.25. That is, show that

1. every marginal of a Gaussian is Gaussian; and
2. conditioning on a subset of variables of a joint Gaussian is Gaussian

by finding their corresponding PDFs.

*Hint:* You may use that for matrices (\Sigma) and (\Lambda) such that
(\Sigma^{-1} = \Lambda),

* if (\Sigma) and (\Lambda) are symmetric,
  [
  \begin{bmatrix}
  x_A \
  x_B
  \end{bmatrix}^{!\top}
  \begin{bmatrix}
  \Lambda_{AA} & \Lambda_{AB} \
  \Lambda_{BA} & \Lambda_{BB}
  \end{bmatrix}
  \begin{bmatrix}
  x_A \
  x_B
  \end{bmatrix}
  =============

  x_A^\top \Lambda_{AA} x_A

  * x_A^\top \Lambda_{AB} x_B
  * x_B^\top \Lambda_{BA} x_A
  * x_B^\top \Lambda_{BB} x_B
    ]
    [
    =
    x_A^\top(\Lambda_{AA} - \Lambda_{AB}\Lambda_{BB}^{-1}\Lambda_{BA})x_A
  *

  (x_B + \Lambda_{BB}^{-1}\Lambda_{BA}x_A)^\top
  \Lambda_{BB}
  (x_B + \Lambda_{BB}^{-1}\Lambda_{BA}x_A),
  ]

* (\Lambda_{BB}^{-1} = \Sigma_{BB} - \Sigma_{BA}\Sigma_{AA}^{-1}\Sigma_{AB}),

* (\Lambda_{BB}^{-1}\Lambda_{BA} = -\Sigma_{BA}\Sigma_{AA}^{-1}).

The final two equations follow from the general characterization of the inverse
of a block matrix (Petersen et al., 2008, section 9.1.3).

---

### 1.10 Closedness properties of Gaussians

Recall the notion of a moment-generating function (MGF) of a random vector
(X \in \mathbb R^n) which is defined as
[
\varphi_X(t)
:= \mathbb E \exp!\big[t^\top X\big],
\quad \text{for all } t \in \mathbb R^n.
\tag{1.76}
]

An MGF uniquely characterizes a distribution. The MGF of the multivariate
Gaussian (X \sim \mathcal N(\mu,\Sigma)) is
[
\varphi_X(t)
= \exp!\left(
t^\top \mu + \frac{1}{2} t^\top \Sigma t
\right).
\tag{1.77}
]

This generalizes the MGF of the univariate Gaussian from Equation (A.40).

Prove the following facts.

1. **Closedness under affine transformations:**
   Given an (n)-dimensional Gaussian (X \sim \mathcal N(\mu,\Sigma)), and
   (A \in \mathbb R^{m\times n}) and (b \in \mathbb R^m),
   [
   AX + b \sim \mathcal N(A\mu + b, A\Sigma A^\top).
   \tag{1.78}
   ]

2. **Additivity:**
   Given two independent Gaussian random vectors
   (X \sim \mathcal N(\mu,\Sigma)) and
   (X' \sim \mathcal N(\mu',\Sigma')) in (\mathbb R^n),
   [
   X + X' \sim \mathcal N(\mu + \mu', \Sigma + \Sigma').
   \tag{1.79}
   ]

These properties are unique to Gaussians and a reason for why they are
widely used for learning and inference.

---

### 1.11 Expectation and variance of Gaussians

Derive that (\mathbb E[X] = \mu) and (\operatorname{Var}[X] = \Sigma) when
(X \sim \mathcal N(\mu,\Sigma)).

*Hint:* First derive the expectation and variance of a univariate standard
normal random variable.

---

### 1.12 Non-affine transformations of Gaussians

Answer the following questions with yes or no.

1. Does there exist any non-affine transformation of a Gaussian random
   vector which is Gaussian? If yes, give an example.

2. Let (X,Y,Z) be independent standard normal random variables. Is
   [
   \frac{X + YZ}{1 + Z^2}
   ]
   Gaussian?

---

### 1.13 Decision theory

Derive the optimal decisions under the squared loss and the asymmetric loss
from Example 1.30.

---
