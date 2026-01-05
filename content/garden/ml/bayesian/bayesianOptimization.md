---
title: Bayesian Optimization
type: docs
weight: 6
---

In [Active Learning](/garden/ml/bayesian/activelearning/), we explored how to strategically select data points to maximally reduce uncertainty about an unknown function. The key insight was that by leveraging uncertainty quantification from [Gaussian Processes](/garden/ml/bayesian/gaussianprocesses/), we could choose observations that provide the most information, leading to more efficient learning compared to random sampling. However, the goal was purely epistemic: we wanted to understand the function as well as possible.

In many real-world scenarios, understanding is not enough. We want to use our models to achieve goals, specifically to find the optimum of an unknown function. Consider the following examples:

- A pharmaceutical researcher optimizing drug dosages to maximize efficacy while minimizing side effects. Each clinical trial is expensive, time-consuming, and involves ethical considerations.
- An engineer tuning the parameters of a manufacturing process to maximize yield. Each production run costs resources and may produce defective products.
- A machine learning practitioner selecting hyperparameters (learning rate, regularization strength, network architecture) to maximize validation accuracy. Each training run may take hours or days.
- A materials scientist searching for optimal alloy compositions. Synthesizing and testing each composition requires laboratory resources and expertise.

In all these cases, we have a function $f : \mathcal{X} \to \mathbb{R}$ that we want to maximize (or minimize), but we can only observe it through costly, noisy evaluations. We cannot compute gradients analytically, and we certainly cannot afford to evaluate the function everywhere. This is the domain of **Bayesian optimization**. Formally given some unknown function $f : \mathcal{X} \to \mathbb{R}$, we want to find its maximum:

$$
x^* = \arg\max_{x \in \mathcal{X}} f(x)
$$

The crucial constraint is that we can only access $f$ through noisy evaluations. At each step $t$, we choose an input $x_t \in \mathcal{X}$ and observe:

$$
y_t = f(x_t) + \varepsilon_t
$$

The relationship between Bayesian optimization and active learning is subtle but important. Both use acquisition functions to sequentially select query points, and both leverage Gaussian Processes for uncertainty quantification. The key difference lies in the objective: in active learning, we want to learn the function $f$ as accurately as possible everywhere in the domain (an epistemic goal), while in Bayesian optimization, we want to find the location of the maximum of $f$ (an optimization goal). The acquisition functions reflect this difference: active learning uses criteria like uncertainty sampling or information gain that reduce uncertainty globally, while Bayesian optimization uses criteria like UCB or Expected Improvement that balance uncertainty reduction with seeking high function values. You can think of active learning as "exploration for its own sake" and Bayesian optimization as "exploration in service of exploitation." Both frameworks share the same computational machinery (GPs, acquisition function optimization), but they answer different questions about the unknown function.

The observation noise $\varepsilon_t \sim \mathcal{N}(0, \sigma_n^2)$ captures measurement uncertainty. The function $f$ is a "black box" in the sense that we do not have access to its analytical form, gradients, or any other structural information beyond the evaluations we collect.

{{< figure 
    src="/images/ml/bayesOptimization.jpg"
    alt="Iterative process of finding the optimum of an unknown function using Bayesian optimization."
    caption="Iterative process of finding the optimum of an unknown function using Bayesian optimization."
    width="300"
>}}

For optimization to be possible at all without exhaustive evaluation, we need to assume some structure in $f$. The key assumption is that **similar inputs yield similar outputs**, meaning the function is "smooth" in some sense. This is precisely what we encode by placing a Gaussian Process prior on $f$:

$$
f \sim \mathcal{GP}(\mu_0, k)
$$

The kernel function $k$ encodes our beliefs about the smoothness and structure of $f$. This assumed correlation is what allows us to generalize from observed evaluations to unexplored regions, making optimization from a limited number of samples possible.

The Bayesian optimization algorithm proceeds as follows. We start with a GP prior $f \sim \mathcal{GP}(\mu_0, k_0)$. Then for each round $t = 1, 2, \ldots, T$:

1. Choose the next query point $x_t$ by maximizing an **acquisition function** $\mathcal{A}$:
$$
x_t = \arg\max_{x \in \mathcal{X}} \mathcal{A}(x; \mu_{t-1}, k_{t-1})
$$
1. Observe $y_t = f(x_t) + \varepsilon_t$
2. Update the GP posterior to obtain $\mu_t$ and $k_t$

The acquisition function $\mathcal{A}$ is the key ingredient that determines how we balance **exploration** (querying uncertain regions) with **exploitation** (querying promising regions). Different acquisition functions lead to different algorithms, and much of the theory and practice of Bayesian optimization revolves around designing and analyzing these functions.

## The Exploration-Exploitation Dilemma

The central challenge in Bayesian optimization is balancing two competing objectives. **Exploration** means choosing points where we are uncertain about the function value, points that are informative and help us learn more about $f$. These typically lie far from previous observations and have high posterior variance. In active learning, this was our sole objective, since we simply wanted to reduce uncertainty as efficiently as possible.

**Exploitation** means choosing points where we expect the function to have high values, points that are promising for finding the maximum. These typically lie near previously observed high values and have high posterior mean. If our only goal were to find a single good point, we might simply query where the posterior mean is highest.

The dilemma arises because these objectives often conflict. The point with the highest posterior mean is usually a region we have already explored well (hence the confident high prediction), while the most uncertain regions may or may not contain the global maximum. Too much exploration wastes queries on regions far from the optimum. Too much exploitation risks getting stuck in local optima, never discovering better regions.

{{< figure 
    src="/images/ml/explorationExploitation.jpg"
    alt="The exploration-exploitation trade-off."
    caption="The exploration-exploitation trade-off."
    width="250"
>}}

The exploration-exploitation dilemma is ubiquitous in sequential decision-making under uncertainty. It appears in [reinforcement learning](/garden/ml/rl/fundamentals/) (should an agent try new actions or stick with known good ones?), in clinical trials (should we test new treatments or use proven ones?), and in many other domains. Understanding how to balance these objectives is one of the central themes in decision theory and machine learning.

## Online Learning and Bandits

Bayesian optimization is closely connected to a family of problems in online learning, where we must make sequential decisions while learning about the environment (hence online). The most canonical example is the **multi-armed bandit** problem, which provides a clean mathematical formalization of the exploration-exploitation dilemma.

### Multi-Armed Bandits

The name "multi-armed bandit" comes from the image of a gambler facing a row of slot machines (one-armed bandits), each with an unknown payout distribution. At each time step, the gambler must choose which machine to play. The challenge is that playing a machine provides information about its payout distribution, but the gambler also wants to maximize total winnings.

Formally, we have $K$ possible actions (arms), each with an unknown reward distribution. At each round $t$, we select an action $a_t \in \{1, \ldots, K\}$ and observe a reward $r_t$ drawn from the distribution associated with arm $a_t$. Our goal is to maximize the cumulative reward $\sum_{t=1}^T r_t$ over a time horizon $T$.

{{< figure 
    src="/images/ml/bayesMultiArmedBandit.jpg"
    alt="The multi-armed bandit problem: choosing which arm to pull to maximize reward."
    caption="In the multi-armed bandit problem, we must balance exploring arms with uncertain rewards (to learn their distributions) with exploiting arms known to have high rewards."
    width="400"
>}}

The connection to Bayesian optimization becomes clear when we view it as a continuous bandit problem. Instead of a finite set of arms, we have a continuous action space $\mathcal{X}$. Instead of independent reward distributions, we have a correlated reward function $f : \mathcal{X} \to \mathbb{R}$ modeled by a Gaussian Process. The correlation structure, encoded by the kernel, is what allows us to generalize from observed evaluations to nearby points, making the infinite-armed case tractable.

To make this concrete, consider hyperparameter tuning for a neural network. Each "arm" corresponds to a specific hyperparameter configuration $x = (\text{learning rate}, \text{dropout rate}, \text{hidden units})$, and the "reward" $f(x)$ is the validation accuracy achieved with those hyperparameters. In a discrete bandit, we might have $K = 100$ pre-defined configurations to choose from, and pulling arm $k$ means training the network with configuration $k$ and observing the (noisy) validation accuracy. The rewards are independent across arms since there is no assumed relationship between configurations.

In the continuous Bayesian optimization formulation, we instead treat the hyperparameter space as continuous: $x \in [10^{-5}, 10^{-1}] \times [0, 0.5] \times [32, 512]$. The key insight is that the reward function $f$ is not arbitrary. We expect that if two configurations are similar (e.g., learning rates of 0.01 and 0.011), their validation accuracies should also be similar. This correlation structure is exactly what the GP kernel captures. When we observe that $f(0.01, 0.2, 128) = 0.92$, the GP updates its beliefs not just at that point, but also at nearby points, inferring that $f(0.011, 0.2, 128) \approx 0.92$ as well. This generalization is what makes optimization possible despite having infinitely many "arms."

Multi-armed bandits can also be viewed as a special case of [reinforcement learning](/garden/ml/rl/fundamentals/) where there is only a single state and multiple actions. In full reinforcement learning, our actions affect which state we transition to, creating complex dependencies between decisions. Bandits strip away this complexity, focusing solely on the exploration-exploitation trade-off in action selection.

### Regret

To evaluate how well an algorithm balances exploration and exploitation, we need a performance metric. The standard choice in online learning and bayesian optimization is **regret**, which measures the cumulative loss compared to the optimal strategy in hindsight. The **instantaneous regret** at round $t$ is the difference between the maximum possible reward and the reward we actually obtained:

$$
r_t = f(x^*) - f(x_t)
$$

where $x^* = \arg\max_{x \in \mathcal{X}} f(x)$ is the global optimum. This can be seen as "how far away" we are from the best possible action at round $t$. The **cumulative regret** over $T$ rounds is the sum of instantaneous regrets:

$$
R_T = \sum_{t=1}^T r_t = \sum_{t=1}^T \left[ f(x^*) - f(x_t) \right] = T \cdot f(x^*) - \sum_{t=1}^T f(x_t)
$$

this measures the total cost of not knowing the optimum in advance. An algorithm that quickly identifies and exploits the optimum will have low cumulative regret, while an algorithm that explores too much or gets stuck in suboptimal regions will have high cumulative regret. The goal is to find algorithms that achieve **sublinear regret**:

$$
\lim_{T \to \infty} \frac{R_T}{T} = 0
$$

In calculus terms, sublinear regret means $R_T$ grows slower than $T$ as $T \to \infty$. If $R_T = \Theta(T^\alpha)$ for some $\alpha < 1$, then $R_T/T = T^{\alpha - 1} \to 0$. For example, $R_T = O(\sqrt{T})$ gives $R_T/T = O(1/\sqrt{T}) \to 0$. In contrast, linear regret $R_T = \Theta(T)$ gives $R_T/T = \Theta(1)$, a constant that does not decay.

Why is this distinction so important? Linear regret means we accumulate a constant amount of regret per round on average, forever. We never truly learn the optimum; we keep making mistakes at a constant rate. Sublinear regret means we eventually stop making mistakes. The average per-round regret $R_T/T \to 0$ implies that most rounds have near-zero instantaneous regret, meaning $f(x_t) \approx f(x^*)$ for most $t$.

This condition says that the average regret per round goes to zero as we collect more data. Equivalently, sublinear regret implies that we are converging to the optimal action. To see why, note that if $R_T = o(T)$ (sublinear), then the average regret $R_T / T \to 0$. Since the instantaneous regret is non-negative, this means that eventually $r_t \to 0$ for most rounds, which implies $f(x_t) \to f(x^*)$. Why is sublinear regret the right goal? Consider two extreme strategies:

1. **Pure exploitation**: Always query the point with the highest posterior mean. If we happen to start near the global optimum, this works well. But if the initial observations are misleading, we might never discover better regions. In the worst case, we incur constant instantaneous regret forever if we get stuck in a local optimum, leading to linear cumulative regret $R_T = \Omega(T)$.

2. **Pure exploration**: Query points uniformly at random (or use uncertainty sampling from active learning). We will eventually explore everywhere, but we waste many queries on suboptimal regions. Suppose at each round we pick a random point $\tilde{x}$ uniformly from $\mathcal{X}$. The expected instantaneous regret is $\mathbb{E}[f(x^*) - f(\tilde{x})] = \Delta > 0$, some positive constant depending on the function shape. Since we do this every round, the expected cumulative regret is $\mathbb{E}[R_T] = T \cdot \Delta$, which is linear in $T$. Even if we only explore with probability $\varepsilon$ and exploit with probability $1 - \varepsilon$, we still incur expected regret at least $\varepsilon \cdot \Delta$ per round, giving $R_T \geq \varepsilon \cdot \Delta \cdot T$, which remains linear.

Achieving sublinear regret requires a careful balance: enough exploration to find the optimum, but decreasing exploration over time as we become more confident. 

## Acquisition Functions

Given a Gaussian Process posterior $f \mid \mathcal{D}_t \sim \mathcal{GP}(\mu_t, k_t)$ after $t$ observations, we need a strategy to select the next query point $x_{t+1}$. This is where **acquisition functions** come in. An acquisition function $\mathcal{A}(x; \mu_t, k_t)$ assigns a score to each candidate point, and we query the point with the highest score:

$$
x_{t+1} = \arg\max_{x \in \mathcal{X}} \mathcal{A}(x; \mu_t, k_t)
$$

A good acquisition function should balance exploitation (high predicted value) with exploration (high uncertainty). We have already encountered one acquisition function in active learning: [uncertainty sampling](/garden/ml/bayesian/activelearning/#uncertainty-sampling), which selects $x_{t+1} = \arg\max_x \sigma_t^2(x)$. However, uncertainty sampling focuses purely on exploration and ignores the optimization objective entirely. It would happily query low-value regions forever as long as they are uncertain.

For optimization, we need acquisition functions that incorporate both the posterior mean (where do we expect high values?) and the posterior variance (where are we uncertain?).

https://anhosu.com/post/acquisition-functions-r/

### Upper Confidence Bound (UCB)

The principle of optimism in the face of uncertainty suggests a simple strategy: at each round, act as if the unknown function takes the best plausible value at each point. To understand UCB, recall that for a GP posterior, the function value at any point $x$ follows a Gaussian distribution: $f(x) \mid \mathcal{D}_t \sim \mathcal{N}(\mu_t(x), \sigma_t^2(x))$. For a Gaussian random variable, about 95% of the probability mass lies within two standard deviations of the mean. So $f(x)$ lies in the interval $[\mu_t(x) - 2\sigma_t(x), \mu_t(x) + 2\sigma_t(x)]$ with high probability.

The upper confidence bound is the top of this interval. It represents the highest plausible value of $f(x)$ given our current knowledge. More generally, we define:

$$
\text{UCB}_t(x) = \mu_t(x) + \beta_{t+1} \sigma_t(x)
$$

where $\mu_t(x)$ is the posterior mean (our best guess for $f(x)$ given the data), $\sigma_t(x) = \sqrt{k_t(x, x)}$ is the posterior standard deviation (our uncertainty about $f(x)$), and $\beta_t > 0$ controls how many standard deviations above the mean we consider "plausible." Larger $\beta_t$ means wider confidence intervals and more exploration. The term "confidence bound" comes from this interpretation: $\mu_t(x) + \beta_t \sigma_t(x)$ is an upper bound on $f(x)$ that holds with high probability (determined by $\beta_t$).

The UCB acquisition function selects the point that maximizes this upper bound:

$$
x_{t+1} = \arg\max_{x \in \mathcal{X}} \left[ \mu_t(x) + \beta_{t+1} \sigma_t(x) \right]
$$

If the true function $f$ lies within the confidence bounds $[\mu_t(x) - \beta_t \sigma_t(x), \mu_t(x) + \beta_t \sigma_t(x)]$ with high probability, then the global maximum $f(x^*)$ must lie below the upper confidence bound somewhere. By repeatedly selecting $x_{t+1} = \arg\max_x \text{UCB}_t(x)$ and evaluating $f$ at that point, we either find a new high value (exploitation success) or we reduce uncertainty in a region we thought was promising (we learn it was not as good as hoped, which is also progress).

{{< figure 
    src="/images/ml/bayesUCB.png"
    alt="The Upper Confidence Bound acquisition function."
    caption="The Upper Confidence Bound acquisition function."
    width="600"
>}}

UCB naturally trades off exploration and exploitation through the parameter $\beta_t$:

- When $\beta_t = 0$, we have pure exploitation: UCB reduces to simply maximizing the posterior mean.
- As $\beta_t \to \infty$, we have pure exploration: UCB reduces to uncertainty sampling (maximizing posterior variance).

The power of UCB lies not just in its intuitive appeal, but in its provable performance guarantees. Under appropriate assumptions on $\beta_t$, the cumulative regret of GP-UCB satisfies:

$$
R_T = O^*\left(\sqrt{T \cdot \gamma_T} \right)
$$

or the average regret satisfies:

$$
\frac{R_T}{T} = O^*\left( \sqrt{\frac{\gamma_T}{T}} \right)
$$

where $\gamma_T$ is the **maximum information gain** after $T$ observations:

$$
\gamma_T = \max_{S \subseteq \mathcal{X}, |S| = T} I(\mathbf{f}_S; \mathbf{y}_S) = \max_{S \subseteq \mathcal{X}, |S| = T} \frac{1}{2} \log \det\left( \mathbf{I} + \sigma_n^{-2} \mathbf{K}_{SS} \right)
$$

Recall from [active learning](/garden/ml/bayesian/activelearning/#utility-of-information-gain) that $I(\mathbf{f}_S; \mathbf{y}_S)$ is the mutual information between the function values $\mathbf{f}_S$ at locations $S$ and the corresponding noisy observations $\mathbf{y}_S$. The maximum information gain $\gamma_T$ measures how much we could possibly learn about $f$ by optimally selecting $T$ observation locations.

The regret bound reveals a deep connection between optimization and information: **the rate at which we can optimize depends on the rate at which we can learn**. If we can learn a lot from few samples (small $\gamma_T$), then we can also optimize efficiently (small regret). Therfore, the maximum information gain $\gamma_T$ plays a central role in the regret bound. It captures the "complexity" of learning the function $f$ under the assumed kernel. Different kernels lead to different information gain rates, which in turn determine the achievable regret. 

Recall from active learning that the information gain function is submodular, meaning we get diminishing returns from additional observations. This property leads to bounds on $\gamma_T$ that depend on the kernel's smoothness properties:

| Kernel | Information Gain $\gamma_T$ |
|--------|---------------------------|
| Linear | $O(d \log T)$ |
| Squared Exponential (RBF) | $O((\log T)^{d+1})$ |
| Matérn ($\nu > 1/2$) | $O\left( T^{\frac{d}{2\nu + d}} (\log T)^{\frac{2\nu}{2\nu + d}} \right)$ |

- **Linear kernel**: The function is a hyperplane in $d$ dimensions, and we need only $O(d)$ samples to determine it up to noise. Additional samples provide diminishing information, leading to logarithmic growth.
- **RBF kernel**: The function is infinitely smooth, and nearby observations are highly correlated. Each observation "covers" a neighborhood, and we can efficiently learn about the entire domain with $O((\log T)^{d+1})$ information growth. This very slow (polylogarithmic) growth leads to excellent regret bounds.
- **Matérn kernel**: With smoothness parameter $\nu$, the function has $\lceil \nu \rceil - 1$ continuous derivatives. Rougher functions (smaller $\nu$) require more samples to characterize, leading to faster information gain growth. As $\nu \to \infty$, the Matérn kernel approaches the RBF kernel and therfore also becomes easier to learn.

Note that when $\gamma_T = O(T)$ (linear growth, corresponding to independent observations), the regret bound becomes $R_T = O(\sqrt{T \cdot T}) = O(T)$, which is linear and thus not sublinear. This happens when the kernel implies no correlation structure, so we cannot generalize between observations. For optimization to be efficient, we need the kernel to encode meaningful smoothness assumptions.

The dimension $d$ appears prominently in these bounds, reflecting the **curse of dimensionality**: in higher dimensions, it takes exponentially more samples to cover the space. This is a fundamental limitation of Bayesian optimization, which is most effective in low-to-moderate dimensional settings (typically $d < 20$).

{{< figure 
    src="/images/ml/informationGainKernels.png"
    alt="Information gain growth for different kernels."
    caption="Information gain growth for different kernels."
    width="600"
>}}

### Probability of Improvement (PI)

An alternative to UCB is to directly consider how likely each point is to improve upon our current best observation. Let $\hat{f}_t = \max_{i \leq t} y_i$ be the best observed value so far. The **probability of improvement (PI)** at point $x$ is:

$$
\text{PI}_t(x) = \mathbb{P}\left( f(x) > \hat{f}_t \mid \mathcal{D}_t \right)
$$

Under the GP posterior, $f(x) \mid \mathcal{D}_t \sim \mathcal{N}(\mu_t(x), \sigma_t^2(x))$. To compute the probability that $f(x) > \hat{f}_t$, we standardize: let $Z = (f(x) - \mu_t(x))/\sigma_t(x)$, so $Z \sim \mathcal{N}(0,1)$. Then:

$$
\mathbb{P}(f(x) > \hat{f}_t) = \mathbb{P}\left(Z > \frac{\hat{f}_t - \mu_t(x)}{\sigma_t(x)}\right) = 1 - \Phi\left(\frac{\hat{f}_t - \mu_t(x)}{\sigma_t(x)}\right) = \Phi\left(\frac{\mu_t(x) - \hat{f}_t}{\sigma_t(x)}\right)
$$

The last equality uses the symmetry of the standard normal: $1 - \Phi(-z) = \Phi(z)$. Defining $z_t(x) = (\mu_t(x) - \hat{f}_t)/\sigma_t(x)$, we get:

$$
\text{PI}_t(x) = \Phi(z_t(x))
$$

where $\Phi$ is the CDF of the standard normal distribution. The acquisition function selects $x_{t+1} = \arg\max_x \text{PI}_t(x)$.

PI is intuitive but has a significant drawback: it considers only whether an improvement occurs, not how large the improvement might be. As a result, PI tends to favor exploitation over exploration. It prefers points with high mean and low variance (where improvement is likely but small) over points with moderate mean and high variance (where improvement is less certain but potentially much larger).

{{< figure 
    src="/images/ml/bayesProbabilityImprovement.png"
    alt="The Probability of Improvement acquisition function."
    caption="Probability of Improvement acquisition function."
    width="600"
>}}

### Expected Improvement (EI)

A more balanced approach is to consider not just the probability of improvement, but the expected magnitude of improvement. The **improvement** at point $x$ is:

$$
I_t(x) = \max(0, f(x) - \hat{f}_t) = (f(x) - \hat{f}_t)^+
$$

The **expected improvement (EI)** is the expectation of this quantity under the GP posterior:

$$
\text{EI}_t(x) = \mathbb{E}\left[ (f(x) - \hat{f}_t)^+ \mid \mathcal{D}_t \right]
$$

This expectation can be computed in closed form because $f(x) \mid \mathcal{D}_t$ is Gaussian. The derivation (see the proof below) yields:

$$
\text{EI}_t(x) = (\mu_t(x) - \hat{f}_t) \Phi(z_t(x)) + \sigma_t(x) \phi(z_t(x))
$$

where $z_t(x) = (\mu_t(x) - \hat{f}_t) / \sigma_t(x)$, and $\phi$ and $\Phi$ are the PDF and CDF of the standard normal distribution.

To build intuition, consider extreme cases. If $\mu_t(x) \gg \hat{f}_t$ (we expect the point to be much better than our current best), then $z_t(x)$ is large and positive, $\Phi(z_t) \approx 1$, and EI $\approx \mu_t(x) - \hat{f}_t$: the expected improvement is approximately the predicted gain. If $\mu_t(x) \ll \hat{f}_t$ (the point is expected to be much worse), then $z_t(x)$ is large and negative, $\Phi(z_t) \approx 0$, $\phi(z_t) \approx 0$, and EI $\approx 0$: we do not expect improvement. The interesting case is when $\mu_t(x) \approx \hat{f}_t$, where the second term $\sigma_t(x) \phi(z_t)$ becomes important, it captures the value of uncertainty when we are near the boundary of improvement.

The EI formula reveals how it balances exploration and exploitation:

- The first term $(\mu_t(x) - \hat{f}_t) \Phi(z_t(x))$ is large when the predicted value exceeds the current best and when this excess is likely (high $\Phi(z_t)$). This is the **exploitation** component.
- The second term $\sigma_t(x) \phi(z_t(x))$ is large when uncertainty is high. This is the **exploration** component.

{{< figure 
    src="/images/ml/bayesExpectedImprovement.png"
    alt="The Expected Improvement acquisition function."
    caption="Expected Improvement acquisition function."
    width="600"
>}}

Expected Improvement is one of the most widely used acquisition functions in practice. It achieves regret bounds similar to UCB and is less sensitive to hyperparameter tuning (UCB requires choosing $\beta_t$, while EI is parameter-free). However, EI can become numerically flat in regions far from the optimum, making optimization of the acquisition function itself challenging. A practical remedy is to optimize $\log \text{EI}$ instead.

{{< callout type="proof" title="Derivation of Expected Improvement" >}}
Let $Z = (f(x) - \mu_t(x)) / \sigma_t(x)$ be the standardized posterior, so $Z \sim \mathcal{N}(0, 1)$ and $f(x) = \mu_t(x) + \sigma_t(x) Z$. The improvement is:

$$
I_t(x) = (\mu_t(x) + \sigma_t(x) Z - \hat{f}_t)^+ = (\sigma_t(x)(Z - (-z_t(x))))^+
$$

where $z_t(x) = (\mu_t(x) - \hat{f}_t) / \sigma_t(x)$. For this to be positive, we need $Z > -z_t(x)$. The expected improvement is:

$$
\text{EI}_t(x) = \sigma_t(x) \int_{-z_t(x)}^{\infty} (z + z_t(x)) \phi(z) dz
$$

Splitting the integral:

$$
= \sigma_t(x) \left[ z_t(x) \int_{-z_t(x)}^{\infty} \phi(z) dz + \int_{-z_t(x)}^{\infty} z \phi(z) dz \right]
$$

The first integral is $\Phi(z_t(x))$ (by symmetry of the normal distribution). For the second, note that $\phi'(z) = -z\phi(z)$, so:

$$
\int_{-z_t(x)}^{\infty} z \phi(z) dz = -\phi(z)\Big|_{-z_t(x)}^{\infty} = \phi(-z_t(x)) = \phi(z_t(x))
$$

Combining and substituting back:

$$
\text{EI}_t(x) = (\mu_t(x) - \hat{f}_t) \Phi(z_t(x)) + \sigma_t(x) \phi(z_t(x))
$$
{{< /callout >}}

### Thompson Sampling

The acquisition functions we have seen so far are deterministic, so given the current GP posterior, they always select the same next query point. An alternative approach introduces randomness directly into the selection process. What if we could ask "which point is most likely to be optimal?" Formally, we want to compute the **probability of maximality**:

$$
\pi(x \mid \mathcal{D}_t) = \mathbb{P}_{f \mid \mathcal{D}_t}\left( f(x) = \max_{x' \in \mathcal{X}} f(x') \right)
$$

This is the probability that $x$ is the global maximizer, where the probability is over the posterior distribution of $f$. A natural strategy would be to sample the next query point from this distribution:

$$
x_{t+1} \sim \pi(\cdot \mid \mathcal{D}_t)
$$

Unfortunately, computing $\pi$ is generally intractable because it requires integrating over the entire posterior distribution of functions. However, we can obtain samples from $\pi$ using a clever Monte Carlo approach. The key observation is that $\pi(x \mid \mathcal{D}_t)$ can be written as an expectation:

$$
\pi(x \mid \mathcal{D}_t) = \mathbb{E}_{f \mid \mathcal{D}_t}\left[ \mathbf{1}\left\{ x = \arg\max_{x'} f(x') \right\} \right]
$$

This says: sample a function $f$ from the posterior, find its maximizer $x^* = \arg\max_{x'} f(x')$, and $\pi(x)$ equals the probability that $x^* = x$. While we cannot compute this expectation analytically, we can sample from it. If we draw one function $\tilde{f}$ from the posterior and find its maximizer $\tilde{x}^* = \arg\max_x \tilde{f}(x)$, then $\tilde{x}^*$ is a sample from $\pi$. This is exactly the probability matching distribution we wanted.

This insight leads to **Thompson sampling**:

1. Draw a function sample $\tilde{f}_{t+1} \sim p(f \mid \mathcal{D}_t)$ from the GP posterior
2. Select $x_{t+1} = \arg\max_{x \in \mathcal{X}} \tilde{f}_{t+1}(x)$

The randomness in Thompson sampling comes from the function sample, not from the action selection (which is deterministic given the sample). This randomness naturally balances exploration and exploitation. Thompson sampling is also widely used in [reinforcement learning](/garden/ml/rl/tabularrl/), where it provides a principled approach to exploration in more complex sequential decision-making problems.

{{< figure 
    src="/images/ml/bayesThompsonSampling.webp"
    alt="Thompson sampling draws a posterior sample and here minimizes it."
    caption="Thompson sampling draws a posterior sample and here minimizes it."
    width="400"
>}}

### Summary of Acquisition Functions

The following table summarizes the acquisition functions we have discussed. Throughout, $z_t(x) = (\mu_t(x) - \hat{f}_t)/\sigma_t(x)$ denotes the standardized improvement, where $\hat{f}_t = \max_{i \leq t} y_i$ is the best observation so far.

| Acquisition Function | Formula | Exploration-Exploitation Balance |
|---------------------|---------|--------------------------------|
| Uncertainty Sampling | $\sigma_t^2(x)$ | Pure exploration |
| UCB | $\mu_t(x) + \beta_t \sigma_t(x)$ | Controlled by $\beta_t$ |
| PI | $\Phi(z_t(x))$ | Favors exploitation |
| EI | $(\mu_t(x) - \hat{f}_t)\Phi(z_t) + \sigma_t(x)\phi(z_t)$ | Balanced, adaptive |
| Thompson Sampling | $\arg\max_x \tilde{f}(x)$ where $\tilde{f} \sim p(f \mid \mathcal{D})$ | Balanced, stochastic |

{{< figure 
    src="/images/ml/bayesAcqusitionFunctionSummary.webp"
    alt="Different acquisition functions applied to the same GP posterior."
    caption="Different acquisition functions applied to the same GP posterior."
>}}
