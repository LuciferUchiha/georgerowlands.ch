---
title: Active Learning
type: docs
weight: 5
---

In [Bayesian Linear Regression](/garden/ml/bayesian/bayesianlearning/) and [Gaussian Processes](/garden/ml/bayesian/gaussianprocesses/), we saw that probabilistic machine learning naturally provides uncertainty estimates alongside predictions. Specifically, we learned to distinguish between **aleatoric uncertainty** (irreducible noise in the data) and **epistemic uncertainty** (uncertainty about the model itself, reducible by gathering more data). A natural question arises: how can we leverage these uncertainty estimates to improve our models more efficiently?

This chapter explores **active learning**, a framework where the learner (model) strategically chooses which data points to observe next. Rather than passively receiving a fixed dataset, an active learner queries the most informative locations in the input space to maximize learning efficiency. This is particularly valuable when data collection is expensive, time-consuming, or potentially dangerous.

## Motivation

In classical supervised learning, we are given a fixed dataset $\mathcal{D} = \{(x_i, y_i)\}_{i=1}^n$ and our task is to learn a model that generalizes well. But in many real-world scenarios, we have some control over which data we observe. A robot learning to navigate can choose which actions to try. A scientist designing experiments can choose which conditions to test. An engineer optimizing a system can choose which configurations to evaluate.

Consider a pharmaceutical researcher studying the interaction between two drugs. Each experiment, testing a particular combination of dosages, is costly and time-consuming. The researcher cannot afford to exhaustively test all possible combinations. Instead, they want to strategically select the next experiment to maximally improve their understanding of the drug interaction landscape. Similarly, in image classification, after training a model we might notice that certain classes are frequently confused with each other, say "cat" and "dog" or "5" and "6" in digit recognition. Rather than collecting more data uniformly across all classes, we could focus our labelling budget on examples from these confused classes, where additional data would be most beneficial.


The key insight from Bayesian learning is that our models not only make predictions but also quantify how uncertain those predictions are. Intuitively, we should collect data where we are most uncertain, as those observations will be most informative. But how do we formalize "most informative"? And given the combinatorial explosion of possible observation sets, how do we efficiently find good solutions?

{{< figure 
    src="/images/ml/bayesActiveLearning.webp"
    alt="After selecting a new data point, the model's uncertainty is reduced in that region."
    caption="After selecting a new data point, the model's uncertainty is reduced in that region."
    width="600"
>}}

## Bayesian Learning with Gaussian Processes

To ground our discussion, let us briefly recall the Bayesian framework for regression using [Gaussian Processes](/garden/ml/bayesian/gaussianprocesses/). We model the unknown function $f : \mathcal{X} \to \mathbb{R}$ as drawn from a GP prior:

$$
f \sim \mathcal{GP}(0, k)
$$

where $k : \mathcal{X} \times \mathcal{X} \to \mathbb{R}$ is the kernel function encoding our assumptions about $f$'s smoothness and structure. Given noisy observations $y_i = f(x_i) + \varepsilon_i$ with $\varepsilon_i \sim \mathcal{N}(0, \sigma_n^2)$, the posterior distribution over $f$ remains a Gaussian Process with updated mean and covariance functions.

A crucial property for active learning is that the **posterior covariance depends only on the input locations $\mathbf{X}$, not on the observed outputs $\mathbf{y}$**. Recall from [GP inference](/garden/ml/bayesian/gaussianprocesses/#predictive-variance) that the predictive variance (which defines the uncertainty of the prediction) at a test point $x_*$ is:

$$
\sigma^2(x_*) = k(x_*, x_*) - \mathbf{k}_*^T (\mathbf{K} + \sigma_n^2 \mathbf{I})^{-1} \mathbf{k}_*
$$

This variance depends on where we have observed (through $\mathbf{K}$ and $\mathbf{k}_*$), but not on what values we observed. This means we can **plan where to observe before actually making observations**, which is essential for designing efficient data collection strategies.

The central question of active learning is: given a "ground set" $\mathcal{X}$ of potential observation locations, which subset $S \subseteq \mathcal{X}$ should we choose to observe? We want to select points whose observations provide the most useful information about the unknown function $f$ and then use those observations to improve our model.

## Information Theory

To quantify "useful information," we turn to information theory. The fundamental concept we need is **mutual information**, which measures how much observing one random variable tells us about another. Before defining mutual information, we briefly review the necessary information-theoretic concepts. You can find more in [Surprise and Entropy](/garden/ml/bayesian/variationalinference/#information-theory-surprise-and-entropy).

Recall that the **entropy** $H[X]$ of a random variable $X$ is the expected "surprise" of observing $X$. It quantifies our uncertainty about $X$ before observing it:

$$
H[X] = \mathbb{E}_{x \sim p}[-\log p(x)]
$$

So if $X$ is very uncertain (spread out distribution), the entropy is high, if $X$ is very certain (concentrated distribution), the entropy is low as there is less surprise in observing it. 

For a Gaussian $X \sim \mathcal{N}(\mu, \sigma^2)$, the entropy is $H[X] = \frac{1}{2}\log(2\pi e \sigma^2)$, which increases with variance. Higher entropy means more uncertainty.

### Conditional Entropy

A natural extension is the **conditional entropy** $H[X \mid Y]$, not to be confused with [cross entropy](/garden/ml/bayesian/variationalinference/#cross-entropy) $H(p \| q)$ which measures the difference between two distributions. The conditional entropy quantifies the uncertainty remaining about $X$ after observing another random variable $Y$:

$$
H[X \mid Y] = \mathbb{E}_{y \sim p(y)}[H[X \mid Y = y]] = \mathbb{E}_{(x,y) \sim p(x,y)}[-\log p(x \mid y)]
$$

The conditional entropy $H[X \mid Y]$ answers the question: "After I observe $Y$, how uncertain will I still be about $X$, on average?" It is important to distinguish this from $H[X \mid Y = y]$, which is the entropy of $X$ given a specific observed value $y$. The conditional entropy $H[X \mid Y]$ averages over all possible realizations of $Y$.

A fundamental property of entropy is **monotonicity**: conditioning on additional information can never increase entropy:

$$
H[X \mid Y] \leq H[X]
$$

This is sometimes called the "information never hurts" principle. Observing $Y$ can only reduce (or maintain) our uncertainty about $X$, never increase it.

### Mutual Information

We are now ready to define the central quantity for active learning. The **mutual information** (also called **expected information gain**) between random variables $X$ and $Y$ is defined as:

$$
I(X; Y) := H[X] - H[X \mid Y]
$$

This measures the expected reduction in uncertainty about $X$ when we observe $Y$. Equivalently, mutual information quantifies how much information $Y$ provides about $X$. Several key properties follow immediately from this definition:

1. **Non-negativity**: $I(X; Y) \geq 0$ since $H[X \mid Y] \leq H[X]$. Observing $Y$ cannot increase uncertainty about $X$.

2. **Symmetry**: $I(X; Y) = I(Y; X)$. The information $Y$ provides about $X$ equals the information $X$ provides about $Y$. This can be shown using the equivalent form:
$$
I(X; Y) = H[X] + H[Y] - H[X, Y]
$$

3. **Zero iff independent**: $I(X; Y) = 0$ if and only if $X$ and $Y$ are independent. Independent variables tell us nothing about each other.

The symmetry of mutual information follows from the chain rule for entropy. The **joint entropy** of $X$ and $Y$ is $H[X, Y] = \mathbb{E}_{(x,y)}[-\log p(x,y)]$, and it satisfies:

$$
H[X, Y] = H[X] + H[Y \mid X] = H[Y] + H[X \mid Y]
$$

Rearranging gives $H[X] - H[X \mid Y] = H[Y] - H[Y \mid X]$, establishing symmetry.

{{< figure 
    src="/images/ml/informationMutual.png"
    alt="Venn diagram showing the relationship between entropy, conditional entropy, joint entropy, and mutual information."
    caption="Venn diagram showing the relationship between entropy, conditional entropy, joint entropy, and mutual information."
    width="400"
>}}

The Venn diagram provides a visual intuition for how these quantities relate. Each circle represents the total uncertainty (entropy) in that variable. The **joint entropy** $H[X,Y]$ is the area of the union: the total uncertainty when considering both variables together. The **mutual information** $I(X;Y)$ is the intersection: the shared uncertainty, or equivalently, the information that each variable contains about the other. The **conditional entropies** $H[X \mid Y]$ and $H[Y \mid X]$ are the non-overlapping parts of each circle: the uncertainty that remains in one variable after observing the other.

From the diagram, we can read off several relationships. The joint entropy is $H[X,Y] = H[X] + H[Y] - I(X;Y)$, since adding the two circles double-counts the intersection. The conditional entropy is $H[X \mid Y] = H[X] - I(X;Y)$, the part of $H[X]$ not shared with $Y$. When $X$ and $Y$ are independent, the circles do not overlap and $I(X;Y) = 0$.

There is another useful way to view mutual information. Starting from the definition and expanding:

$$
\begin{aligned}
I(X; Y) &= H[X] - H[X \mid Y] \\
&= \mathbb{E}_{x \sim p(x)}[-\log p(x)] - \mathbb{E}_{(x,y) \sim p(x,y)}[-\log p(x \mid y)] \\
&= \mathbb{E}_{(x,y) \sim p(x,y)}\left[-\log p(x) + \log p(x \mid y)\right] \\
&= \mathbb{E}_{(x,y) \sim p(x,y)}\left[\log \frac{p(x \mid y)}{p(x)}\right] \\
&= \mathbb{E}_{(x,y) \sim p(x,y)}\left[\log \frac{p(x,y)}{p(x)p(y)}\right]
\end{aligned}
$$

where the last step uses Bayes' rule $p(x \mid y) = p(x,y)/p(y)$. This final form is exactly the **KL divergence** between the joint distribution and the product of marginals:

$$
I(X; Y) = \text{KL}(p(x,y) \| p(x)p(y))
$$

So mutual information measures how much the joint distribution differs from what it would be if $X$ and $Y$ were independent. It quantifies the "cost" (in nats) of incorrectly assuming independence. When $X$ and $Y$ are truly independent, $p(x,y) = p(x)p(y)$ and the KL divergence is zero. The more dependent they are, the larger the divergence, and the more information each provides about the other.

{{< callout type="example" title="Mutual Information for Gaussians" >}}
Let $X \sim \mathcal{N}(\mu, \Sigma)$ be a $d$-dimensional Gaussian random vector, and let $Y = X + \varepsilon$ where $\varepsilon \sim \mathcal{N}(0, \sigma_n^2 I)$ is independent noise. What is the information gain about $X$ when observing $Y$?

Using symmetry, we compute $I(X; Y) = I(Y; X) = H[Y] - H[Y \mid X]$. We know that $Y \sim \mathcal{N}(\mu, \Sigma + \sigma_n^2 I)$ since it is the sum of two independent Gaussians. Given $X$, we have $Y \mid X \sim \mathcal{N}(X, \sigma_n^2 I)$, so $H[Y \mid X] = H[\varepsilon]$. For a $d$-dimensional Gaussian with covariance $\sigma_n^2 I$, the entropy is $\frac{d}{2}\log(2\pi e \sigma_n^2)$. The factor of $d$ appears because the entropy of a multivariate Gaussian is $\frac{1}{2}\log\det(2\pi e \Sigma)$, and for $\Sigma = \sigma_n^2 I$, we have $\det(\sigma_n^2 I) = \sigma_n^{2d}$, giving us the $d/2$ factor.

For the marginal entropy, note that $Y \sim \mathcal{N}(\mu, \Sigma + \sigma_n^2 I)$, so:

$$
H[Y] = \frac{1}{2}\log\det(2\pi e (\Sigma + \sigma_n^2 I))
$$

Therefore:

$$
I(X; Y) = \frac{1}{2}\log\det(2\pi e (\Sigma + \sigma_n^2 I)) - \frac{d}{2}\log(2\pi e \sigma_n^2) = \frac{1}{2}\log\det(I + \sigma_n^{-2}\Sigma)
$$

This result is intuitive: the information gain increases with the signal variance $\Sigma$ and decreases with the noise variance $\sigma_n^2$. When noise dominates ($\sigma_n^2 \gg \|\Sigma\|$), the observation $Y$ tells us little about $X$. When signal dominates ($\|\Sigma\| \gg \sigma_n^2$), observing $Y$ is highly informative.
{{< /callout >}}

### Utility of Information Gain

Armed with mutual information, we can now formalize the active learning objective. Suppose we have a ground set $\mathcal{X}$ of candidate observation locations. For any location $x \in \mathcal{X}$, we can make a noisy observation:

$$
y_x = f(x) + \varepsilon_x, \quad \varepsilon_x \sim \mathcal{N}(0, \sigma_n^2)
$$

where $f_x := f(x)$ denotes the (random) function value at $x$. For a set of observations $S = \{x_1, \ldots, x_n\} \subseteq \mathcal{X}$, we write $\mathbf{y}_S = [y_{x_1}, \ldots, y_{x_n}]^T$ and $\mathbf{f}_S = [f_{x_1}, \ldots, f_{x_n}]^T$.

Our goal is to find a subset $S \subseteq \mathcal{X}$ of size $n$ (our observation budget) that maximizes the information gain about the function values:

$$
I(S) := I(\mathbf{f}_S; \mathbf{y}_S) = H[\mathbf{f}_S] - H[\mathbf{f}_S \mid \mathbf{y}_S]
$$

The first term $H[\mathbf{f}_S]$ is our prior uncertainty about the function values at $S$, and the second term $H[\mathbf{f}_S \mid \mathbf{y}_S]$ is our expected posterior uncertainty after observing $\mathbf{y}_S$. Maximizing $I(S)$ means choosing observation locations that maximally reduce our uncertainty about the function.

This formulation connects active learning to **optimal experimental design** in statistics. The objective $I(S)$ is sometimes called an **intrinsic reward or utility function**, as it measures the internal benefit (uncertainty reduction) rather than an external reward.

{{< figure 
    src="/images/ml/informationGain.png"
    alt="High vs low information gain point selection."
    caption="Information gain depends on where we observe. The middle panel shows points with high information gain (large uncertainty reduction), while the right panel shows points with low information gain (small uncertainty reduction)."
    width="600"
>}}

### The Combinatorial Challenge

Finding the optimal set $S^* = \arg\max_{S \subseteq \mathcal{X}, |S| = n} I(S)$ is a **combinatorial optimization problem**. With $|\mathcal{X}| = N$ candidate points and a budget of $n$ observations, there are $\binom{N}{n}$ possible subsets to consider. For $N = 1000$ and $n = 50$, this is approximately $10^{85}$ subsets, far too many to enumerate.

In fact, maximizing mutual information is **NP-hard** in general. This means that (assuming P $\neq$ NP) there is no polynomial-time algorithm that finds the exact optimum for all instances. Intuitively, the hardness arises because the value of observing a point depends on what other points we observe, creating complex dependencies that preclude simple independent optimization.

Despite this computational hardness, we will see that the structure of mutual information allows for efficient approximate solutions with provable guarantees.

### Submodularity

The key to efficient optimization is a property called **submodularity**, which captures the intuitive notion of "diminishing returns." Before defining it formally, let us introduce the concept of marginal gain.

Given a set function $F : \mathcal{P}(\mathcal{X}) \to \mathbb{R}$ (where $\mathcal{P}(\mathcal{X})$ denotes the power set of $\mathcal{X}$), the **marginal gain** of adding element $x$ to set $A$ is:

$$
\Delta F(x \mid A) := F(A \cup \{x\}) - F(A)
$$

This measures how much the function value increases when we add $x$ to the existing set $A$.

A set function $F$ is **submodular** if for any $x \in \mathcal{X}$ and any $A \subseteq B \subseteq \mathcal{X}$:

$$
\Delta F(x \mid A) \geq \Delta F(x \mid B)
$$

In words, adding $x$ to a smaller set $A$ provides at least as much marginal benefit as adding $x$ to a larger set $B \supseteq A$. This is the **diminishing returns** property: as we accumulate more elements, the marginal value of each additional element decreases (or stays the same). You can also think of it the other way round: the more you have, the less you gain from adding more (note that it can still be equal).

Submodularity can be viewed as a discrete analogue of concavity for continuous functions. Just as a concave function has decreasing marginal returns (the derivative decreases), a submodular function has decreasing marginal gains.

{{< figure 
    src="/images/ml/submodularity.webp"
    alt="Submodularity as the discrete analogue of concavity."
    caption="Submodularity is the discrete analogue of concavity. Just as a concave function has decreasing slopes, a submodular function has decreasing marginal gains."
    width="450"
>}}

A function $F$ is **monotone** if $F(A) \leq F(B)$ whenever $A \subseteq B$. Monotonicity means that adding elements never decreases the function value, so more observations are always at least as good as fewer. If $F$ is both monotone and submodular, we say it is **monotone submodular**. 

The distinction matters for optimization guarantees. For general submodular functions (which may be non-monotone), the greedy algorithm can perform arbitrarily poorly. However, when the function is also monotone, the greedy algorithm achieves the $(1 - 1/e)$ approximation guarantee. Intuitively, monotonicity ensures that we never "regret" adding an element, while submodularity ensures that greedy choices remain near-optimal even as we accumulate elements. Together, these properties make the optimization problem tractable.

The information gain function $I(S)$ is monotone submodular. This is the key structural property that enables efficient optimization. Just like concave functions can be efficiently maximized using gradient ascent, monotone submodular functions can be efficiently approximated using greedy algorithms.

{{< callout type="proof" title="Submodularity of Information Gain">}}
To show submodularity, we first derive the marginal gain. When we add a single point $x$ to an existing set $A$, the marginal information gain is:

$$
\Delta I(x \mid A) = I(A \cup \{x\}) - I(A) = I(f_x; y_x \mid \mathbf{y}_A)
$$

This is the **conditional mutual information** of $f_x$ and $y_x$ given we have already observed $\mathbf{y}_A$. To derive the closed form, we use the definition of mutual information and properties of Gaussians:

$$
I(f_x; y_x \mid \mathbf{y}_A) = H[y_x \mid \mathbf{y}_A] - H[y_x \mid f_x, \mathbf{y}_A]
$$

The second term simplifies because $y_x = f_x + \varepsilon_x$, and once we know $f_x$, the only uncertainty in $y_x$ comes from the noise: $H[y_x \mid f_x, \mathbf{y}_A] = H[\varepsilon_x] = \frac{1}{2}\log(2\pi e \sigma_n^2)$. For the first term, after observing $\mathbf{y}_A$, the posterior distribution of $y_x$ is Gaussian with variance $\sigma_A^2(x) + \sigma_n^2$ (posterior variance of $f_x$ plus noise variance), so $H[y_x \mid \mathbf{y}_A] = \frac{1}{2}\log(2\pi e (\sigma_A^2(x) + \sigma_n^2))$. Taking the difference:

$$
\Delta I(x \mid A) = \frac{1}{2}\log(2\pi e (\sigma_A^2(x) + \sigma_n^2)) - \frac{1}{2}\log(2\pi e \sigma_n^2) = \frac{1}{2}\log\left(1 + \frac{\sigma_A^2(x)}{\sigma_n^2}\right)
$$

where $\sigma_A^2(x)$ is the posterior variance at $x$ after observing $A$. This derivation uses the fact that $y_x = f_x + \varepsilon_x$ and that $H[y_x \mid f_x] = H[\varepsilon_x]$ since the noise is independent of the function value.

Now we can prove submodularity. For $A \subseteq B$:

$$
\Delta I(x \mid A) = H[y_x \mid \mathbf{y}_A] - H[\varepsilon_x] \geq H[y_x \mid \mathbf{y}_B] - H[\varepsilon_x] = \Delta I(x \mid B)
$$

The inequality follows from the "information never hurts" principle: conditioning on more observations (going from $\mathbf{y}_A$ to $\mathbf{y}_B$) can only reduce entropy. Since $A \subseteq B$, we have $H[y_x \mid \mathbf{y}_A] \geq H[y_x \mid \mathbf{y}_B]$, and the submodularity inequality follows.

For monotonicity, we again appeal to the "information never hurts" principle. When $A \subseteq B$, we can write $B = A \cup C$ for some $C \subseteq \mathcal{X} \setminus A$. Since mutual information is non-negative, adding more observation locations can only increase (or maintain) the total information gain:

$$
I(B) = I(A \cup C) = I(A) + I(C \mid A) \geq I(A)
$$

where $I(C \mid A) = I(\mathbf{f}_C; \mathbf{y}_C \mid \mathbf{y}_A) \geq 0$ is the additional information gained from observing $C$ after already observing $A$. This is non-negative by the non-negativity of mutual information. Thus $I(A) \leq I(B)$ whenever $A \subseteq B$, establishing monotonicity. Putting it all together, $I(S)$ is monotone submodular.
{{< /callout >}}

## Greedy Maximization of Information Gain

Now that we know $I(S)$ is monotone submodular, we can leverage this structure to efficiently find a good observation set. Intuitevely, submodularity means that the marginal benefit of adding a point decreases as we add more points. This suggests a simple greedy strategy: at each step, add the point that provides the largest marginal increase in information gain. So the idea of the **greedy algorithm** is to iteratively add the point that provides the largest marginal gain until we reach our budget. Formally starting with $S_0 = \emptyset$, at each step $t = 0, 1, \ldots, n-1$we in general want to find for some submodular function $F$:

$$
x_{t+1} = \arg\max_{x \in \mathcal{X}} F(S_t \cup \{x\})
$$

then set $S_{t+1} = S_t \cup \{x_{t+1}\}$. After $n$ iterations, we return $S_n$. For any non-negative monotone submodular function this is equivalent to maximizing the marginal gain:

$$
x_{t+1} = \arg\max_{x \in \mathcal{X}} \Delta F(x \mid S_t) = \arg\max_{x \in \mathcal{X}} F(S_t \cup \{x\}) - F(S_t)
$$

In our case, the function is the information gain $I(S)$, so we have:

$$
x_{t+1} = \arg\max_{x \in \mathcal{X}} \Delta I(x \mid S_t) = \arg\max_{x \in \mathcal{X}} I(f_x; y_x \mid \mathbf{y}_{S_t})
$$

The remarkable fact is that this simple greedy algorithm achieves a **constant-factor approximation** to the optimal solution. In particular, for any non-negative monotone submodular function $F$, the greedy algorithm satisfies:

$$
F(S_n) \geq \left(1 - \frac{1}{e}\right) \max_{|S| = n} F(S) \approx 0.632 \cdot F(S^*)
$$

This means the greedy solution is always at least 63.2% as good as the optimal solution, regardless of the problem instance. This is a strong guarantee, especially given that finding the true optimum is NP-hard.

## Acquisition Functions and Uncertainty Sampling

The greedy approach requires solving an optimization problem at each step: $x_{t+1} = \arg\max_{x \in \mathcal{X}} \Delta I(x \mid S_t)$. The function being maximized is called an **acquisition function**, as it determines which point we "acquire" next.

For Gaussian Processes with homoscedastic noise (constant noise variance $\sigma_n^2$ across all inputs), the marginal information gain simplifies beautifully. Starting from the definition of marginal gain and using our earlier derivation:

$$
\begin{aligned}
\Delta I(x \mid S_t) &= I(S_t \cup \{x\}) - I(S_t) \\
&= I(f_x; y_x \mid \mathbf{y}_{S_t}) \\
&= H[y_x \mid \mathbf{y}_{S_t}] - H[y_x \mid f_x, \mathbf{y}_{S_t}] \\
&= H[y_x \mid \mathbf{y}_{S_t}] - H[\varepsilon_x]
\end{aligned}
$$

The posterior distribution of $y_x$ given $\mathbf{y}_{S_t}$ is Gaussian with variance $\sigma_t^2(x) + \sigma_n^2$, where $\sigma_t^2(x)$ is the GP posterior variance at $x$. Since $H[\varepsilon_x] = \frac{1}{2}\log(2\pi e \sigma_n^2)$ and $H[y_x \mid \mathbf{y}_{S_t}] = \frac{1}{2}\log(2\pi e (\sigma_t^2(x) + \sigma_n^2))$:

$$
\Delta I(x \mid S_t) = \frac{1}{2}\log\left(\frac{\sigma_t^2(x) + \sigma_n^2}{\sigma_n^2}\right) = \frac{1}{2}\log\left(1 + \frac{\sigma_t^2(x)}{\sigma_n^2}\right)
$$

where $\sigma_t^2(x)$ is the posterior variance at $x$ after observing $S_t$. Since $\log$ is monotonically increasing:

$$
x_{t+1} = \arg\max_{x \in \mathcal{X}} \sigma_t^2(x)
$$

This is called **uncertainty sampling**: we simply select the point with the highest predictive variance, i.e., where the model is most uncertain.

{{< figure 
    src="/images/ml/bayesAcquisitionFunction.png"
    alt="Acquisition function for uncertainty sampling."
    caption="Uncertainty sampling selects the point with highest posterior variance (predictive uncertainty). The acquisition function peaks where the model is most uncertain."
>}}

Uncertainty sampling is remarkably intuitive. Points with high posterior variance are exactly those where our model predictions are least reliable. By observing at these locations, we gain the most information and reduce uncertainty most effectively. The submodularity theory tells us this greedy heuristic is near-optimal.

The posterior variance $\sigma_t^2(x)$ can be computed efficiently using the GP formulas:

$$
\sigma_t^2(x) = k(x, x) - \mathbf{k}_x^T (\mathbf{K}_{S_t} + \sigma_n^2 \mathbf{I})^{-1} \mathbf{k}_x
$$

where $\mathbf{k}_x = [k(x_1, x), \ldots, k(x_t, x)]^T$ for $S_t = \{x_1, \ldots, x_t\}$.

## Heteroscedastic Noise

Uncertainty sampling works well when the observation noise is constant across the input space. But what if different regions have different noise levels? This is called **heteroscedastic noise**, where $\varepsilon_x \sim \mathcal{N}(0, \sigma_n^2(x))$ depends on $x$.

In the heteroscedastic setting, uncertainty sampling can fail dramatically. Consider a region with very high aleatoric uncertainty (noisy observations) but moderate epistemic uncertainty (the model is fairly confident about the underlying function). The total predictive variance is high, so uncertainty sampling will repeatedly query this region. But since the noise is high, each observation provides little information about the underlying function. So we are wasting our budget on observations dominated by irreducible noise.

Recall the decomposition of predictive variance into aleatoric and epistemic components:

$$
\text{Var}[y_* \mid x_*] = \underbrace{\sigma_n^2(x_*)}_{\text{Aleatoric}} + \underbrace{\sigma_\text{epistemic}^2(x_*)}_{\text{Epistemic}}
$$

For effective learning, we want to reduce epistemic uncertainty, which captures our model's ignorance. Aleatoric uncertainty is irreducible and does not decrease with more observations at the same location. Looking back at our marginal information gain formula:

$$
\Delta I(x \mid S_t) = \frac{1}{2}\log\left(1 + \frac{\sigma_t^2(x)}{\sigma_n^2(x)}\right)
$$

With heteroscedastic noise, this naturally accounts for varying noise levels. If $\sigma_n^2(x)$ is large at some location, the information gain at that location is reduced, even if the epistemic uncertainty $\sigma_t^2(x)$ is also large. The optimal acquisition becomes:

$$
x_{t+1} = \arg\max_{x \in \mathcal{X}} \frac{\sigma_t^2(x)}{\sigma_n^2(x)}
$$

This **signal-to-noise ratio** criterion balances epistemic uncertainty (we want high $\sigma_t^2(x)$) against aleatoric uncertainty (we want low $\sigma_n^2(x)$). We prefer locations where the model is uncertain and observations are reliable.

## Active Learning for Classification

So far we have focused on regression, where observations are continuous and corrupted by Gaussian noise. Active learning extends naturally to **classification**, where observations are discrete class labels. This brings us back to the image classification example from the motivation: when certain classes are frequently confused, we want to strategically collect labels for examples that will best disambiguate them. The principles remain the same, but the acquisition functions take different forms.

In probabilistic classification, for each input $x$, the model produces a distribution over class labels $p(y \mid x)$. For binary classification with labels $y \in \{-1, +1\}$, this might be a Bernoulli distribution parameterized by a sigmoid:

$$
p(y = 1 \mid x, \theta) = \sigma(\theta^T x)
$$

A natural analogue of uncertainty sampling for classification is to select points that maximize the **entropy of the predicted label**:

$$
x_{t+1} = \arg\max_{x \in \mathcal{X}} H[y_x \mid x_{1:t}, y_{1:t}]
$$

For binary classification, the entropy is maximized when $p(y = 1 \mid x) = 0.5$, corresponding to maximum uncertainty about the label. This tends to select points near the **decision boundary**, where the classifier is most uncertain.

However, this approach has a similar problem to uncertainty sampling with heteroscedastic noise. High entropy near the decision boundary might reflect genuine model uncertainty (epistemic), but it might also reflect inherent label noise (aleatoric). If the decision boundary passes through a region with significant label noise, we might repeatedly query there without improving the model.

### Bayesian Active Learning by Disagreement (BALD)

To address this limitation, we can use **mutual information** directly, applying the same principle that worked for heteroscedastic regression. Instead of maximizing the entropy of the predicted label, we maximize the mutual information between the model parameters $\theta$ and the predicted label $y_x$:

$$
x_{t+1} = \arg\max_{x \in \mathcal{X}} I(\theta; y_x \mid x_{1:t}, y_{1:t})
$$

This objective, called **BALD** (Bayesian Active Learning by Disagreement), selects points that maximally reduce our uncertainty about the model parameters themselves.

Note that BALD is formulated for **classification** rather than regression. Why the distinction? In regression with Gaussian noise, we saw that uncertainty sampling already handles heteroscedastic noise correctly through the signal-to-noise ratio $\sigma_t^2(x)/\sigma_n^2(x)$. The GP framework gives us closed-form expressions for both epistemic and aleatoric uncertainty, and the mutual information naturally separates them.

Classification is fundamentally different. The observations $y_x$ are **discrete class labels**, not continuous values corrupted by additive noise. There is no simple "noise variance" to divide by. Instead, the aleatoric uncertainty manifests as inherent class overlap or label noise in certain regions. For discrete outputs, we cannot use the Gaussian entropy formulas. Instead, we work directly with the entropy of the categorical distribution $p(y \mid x, \theta)$. This is where BALD becomes essential: it provides a principled way to separate epistemic uncertainty (disagreement between models about which class is correct) from aleatoric uncertainty (inherent ambiguity that even a perfect model cannot resolve). To derive a tractable form, we use symmetry of mutual information:

$$
I(\theta; y_x \mid \mathcal{D}) = I(y_x; \theta \mid \mathcal{D}) = H[y_x \mid \mathcal{D}] - H[y_x \mid \theta, \mathcal{D}]
$$

where $\mathcal{D} = (x_{1:t}, y_{1:t})$ denotes our current observations. The symmetry is useful because while $H[\theta \mid y_x, \mathcal{D}]$ (the posterior entropy over parameters after observing $y_x$) is difficult to compute, $H[y_x \mid \theta, \mathcal{D}]$ is tractable: given the model parameters $\theta$, we can directly compute the predictive distribution $p(y_x \mid x, \theta)$ and its entropy.

The second term $H[y_x \mid \theta, \mathcal{D}]$ still involves the random variable $\theta$. By the law of iterated expectations, we can write this as an expectation over the posterior:

$$
H[y_x \mid \theta, \mathcal{D}] = \mathbb{E}_{\theta \sim p(\theta \mid \mathcal{D})}[H[y_x \mid \theta]]
$$

This gives us the final decomposition:

$$
I(\theta; y_x \mid \mathcal{D}) = \underbrace{H[y_x \mid \mathcal{D}]}_{\text{Total predictive uncertainty}} - \underbrace{\mathbb{E}_{\theta \mid \mathcal{D}}[H[y_x \mid \theta]]}_{\text{Expected aleatoric uncertainty}}
$$

We can interpret the two terms as follows:

1. **Total predictive uncertainty** $H[y_x \mid \mathcal{D}]$: The entropy of the average prediction (averaged over the posterior over $\theta$). This is large when the model's average prediction is uncertain.

2. **Expected aleatoric uncertainty** $\mathbb{E}_{\theta}[H[y_x \mid \theta]]$: The average entropy of individual predictions. This is large when individual models (sampled from the posterior) are uncertain about their own predictions.

BALD selects points where the first term is large and the second term is small. In words: we want points where **different models disagree**, even though each individual model is confident. If models sampled from the posterior make confident but different predictions at $x$, then observing $y_x$ will help us distinguish between these models, reducing our uncertainty about $\theta$. Conversely, if all models agree (low first term) or all models are individually uncertain (high second term), then $x$ is not informative. Agreement means the observation won't change our beliefs much. Individual uncertainty means the observation will be dominated by noise.

In practice, the posterior over $\theta$ is often approximated using Monte Carlo samples or variational inference. Given $M$ samples $\theta_1, \ldots, \theta_M$ from the posterior:

$$
H[y_x \mid \mathcal{D}] \approx H\left[\frac{1}{M}\sum_{m=1}^M p(y \mid x, \theta_m)\right]
$$

$$
\mathbb{E}_{\theta}[H[y_x \mid \theta]] \approx \frac{1}{M}\sum_{m=1}^M H[p(y \mid x, \theta_m)]
$$

The difference gives the BALD acquisition function. This can be computed efficiently for many model classes.

{{< figure 
    src="/images/ml/bayesActiveVsUniform.png"
    alt="MNIST classification accuracy using BALD active learning vs random sampling. BALD selects informative points that improve accuracy faster."
    caption="MNIST classification accuracy using BALD active learning vs random sampling. BALD selects informative points that improve accuracy faster."
    width="600"
>}}

## Transductive Active Learning

Throughout this chapter, we have focused on **inductive active learning**: selecting observations to build the best possible model of the entire function $f$ across its domain $\mathcal{X}$. But in many applications, we ultimately care about predictions at specific target locations.

Consider a student preparing for an exam. Rather than trying to master all possible topics equally (inductive learning), they might focus on topics most likely to appear on the test (transductive learning). Similarly, in active learning, if we know we will be evaluated at specific test points, we can target our data collection accordingly.

**Transductive active learning** adapts this idea. Suppose we know in advance that we need to predict $f(x_*)$ at a specific target $x_*$. Rather than maximizing information about $f$ globally, we maximize information specifically about $f_{x_*}$:

$$
x_{t+1} = \arg\max_{x \in \mathcal{X}'} I(f_{x_*}; y_x \mid x_{1:t}, y_{1:t})
$$

where $\mathcal{X}'$ is the set of candidate observation locations (which might not include $x_*$ itself, perhaps because direct observation at $x_*$ is impossible).

Equivalently, we can minimize the posterior entropy about the target:

$$
x_{t+1} = \arg\min_{x \in \mathcal{X}'} H[f_{x_*} \mid x_{1:t}, y_{1:t}, y_x]
$$

The transductive objective can be decomposed to reveal the trade-off it captures:

$$
I(f_{x_*}; y_x \mid \mathcal{D}) = \underbrace{I(f_{x_*}; y_x)}_{\text{Relevance}} - \underbrace{I(f_{x_*}; y_x; \mathbf{y}_\mathcal{D})}_{\text{Redundancy}}
$$

The first term measures how informative $y_x$ is about $f_{x_*}$ in isolation, which captures the **relevance** of observing at $x$. The second term is the interaction information, measuring **redundancy** between $y_x$ and our existing observations $\mathbf{y}_\mathcal{D}$ with respect to learning about $f_{x_*}$.

Transductive active learning balances relevance and diversity. We want to observe at locations relevant to the target (high correlation with $x_*$), but we also want to avoid redundancy with existing observations. This is a middle ground between pure relevance-based retrieval (which ignores diversity) and inductive active learning (which ignores the specific target).

### Generalization to Target Sets

The single-target formulation extends naturally to multiple targets. Suppose we have a target set $\mathcal{A} \subseteq \mathcal{X}$ where we want accurate predictions, and a sample space $\mathcal{S} \subseteq \mathcal{X}$ from which we can draw observations. Given previous observations $\mathcal{D}_{n-1} = \{(x_i, y_i)\}_{i < n}$, we select the next query point by maximizing information about $f$ restricted to the target set:

$$
x_n = \arg\max_{x \in \mathcal{S}} I(\{f_{x'}\}_{x' \in \mathcal{A}}; y_x \mid \mathcal{D}_{n-1})
$$

This is the **information-based transductive learning (ITL)** objective. Rather than learning $f$ everywhere or at a single point, we specifically target information relevant to predictions on $\mathcal{A}$.

When $f \sim \mathcal{GP}(\mu, k)$ is a Gaussian Process, this mutual information has a convenient closed form. Using properties of Gaussian distributions and the definition of mutual information, one can show:

$$
I(\{f_{x'}\}_{x' \in \mathcal{A}}; y_x \mid \mathcal{D}_{n-1}) = \frac{1}{2} \log \left( \frac{\text{Var}(y_x \mid \mathcal{D}_{n-1})}{\text{Var}(y_x \mid \{f_{x'}\}_{x' \in \mathcal{A}}, \mathcal{D}_{n-1})} \right)
$$

The numerator $\text{Var}(y_x \mid \mathcal{D}_{n-1})$ is our current predictive variance at $x$ given only previous observations. The denominator $\text{Var}(y_x \mid \{f_{x'}\}_{x' \in \mathcal{A}}, \mathcal{D}_{n-1})$ is the variance that would remain if we also knew the true function values on the target set $\mathcal{A}$.

The ratio inside the logarithm measures how much knowing the target values would reduce our uncertainty about $y_x$. If this ratio is large, then $y_x$ is highly correlated with the target set, meaning observing at $x$ will be informative about $\mathcal{A}$. If the ratio is close to 1, then knowing $\mathcal{A}$ does not help predict $y_x$, so observing at $x$ provides little target-relevant information.

This variance-ratio formula makes ITL computationally tractable. Both variances can be computed from the GP posterior covariance matrix. When $\mathcal{A} = \mathcal{X}$ (the entire domain), ITL reduces to standard inductive active learning. When $\mathcal{A} = \{x_*\}$ is a single point, it reduces to the single-target transductive objective discussed above.
