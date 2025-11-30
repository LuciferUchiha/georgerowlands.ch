---
title: Markov Chains & Decision Processes
type: docs
weight: 12
---

Markov Chains are a key concept in probability theory and have many applications in computer science and machine learning. For example the PageRank algorithm used by Google Search to rank web pages is based on a Markov Chain model of web surfing behavior or Diffusion models also use Markov Chains to model the process of gradually transforming random noise into coherent images.

The same goes for Markov Decision Processes, which extend Markov Chains to include the possibility of making decisions at each state, allowing for the modeling of sequential decision-making problems and potential rewards. These decision processes form the basis of many reinforcement learning algorithms, where an agent learns to make optimal decisions by interacting with an environment.

## Markov Chains

Before defining Markov chains, it is helpful to think carefully about what we mean by a "process" or more specifically, a "process over time". We will begin with deterministic processes, which are much easier to describe, and then move toward stochastic processes where randomness plays a role. 

A deterministic process is one in which the state evolves with complete certainty according to some rule. We imagine that at each point in time, the system is in a well-defined state. If we start in a particular state at time $t = 0$, everything that happens afterwards is perfectly predictable. In the simplest setting, time is discrete. We write the state at time $n$ as $x_n$ and we assume there is some function $f$ that tells us exactly what comes next:

$$
x_{n+1} = f(x_n) \quad \text{with an initial state } x_0 \text{ given}
$$

Here the state space $S$ might be something like ${1,2,3}$ or $\mathbb{R}^d$ or anything else. If $S$ is discrete, $f$ could be a lookup table. If $S$ is continuous, $f$ could be a real function like $f(x) = 2x + 1$ or a formula like in the case of [sequences defined by recurrence relations].

A deterministic process can also evolve in continuous time. Instead of a sequence $(x_0, x_1, \dots)$, we have a function $x(t)$ describing the state at every real-valued time $t \ge 0$. Instead of difference equations we often have ordinary differential equations (ODEs) that describe how the state changes infinitesimally:

$$
\frac{dx(t)}{dt} = g(x(t)) \quad \text{with an initial state } x(0) = x_0 \text{ given}
$$

The idea of such a process is also briefly discussed when talking about the movement of particles in physics in the context of [score based diffusion models](). There, the movement of particles is modeled as a deterministic process based on differential equations that describe how the position and velocity of particles change over time. In that case, time is continuous and the states live in a continuous space (e.g. $\mathbb{R}^3$ for positions).

On the other hand, a stochastic process is a process that evolves in time but with randomness. We no longer have a deterministic rule $x_{n+1} = f(x_n)$ that tells us exactly what must happen. Instead, we have a family of random variables:

$$
\{X_t \mid t \in T\}
$$

where $T$ is the index set representing time (which can be discrete $T = {0,1,2,\dots}$ or continuous $T = [0,\infty)$ and each $X_t$ is a random variable taking values in some state space $S$. Formally, a stochastic process is defined on a probability space $(\Omega, \mathcal{F}, \mathbb{P})$, and each $X_t : \Omega \to S$ is a random variable. But intuitively, you can think of $X_t$ as “the state of the system at time $t$”, except the state is now random. Because of this stochasticity, we can assign probabilities to different possible future states depending on present or past states:

$$
\mathbb{P}(X_{t+1} = x \mid X_t, X_{t-1}, \dots, X_0)
$$

Note that we use capital letters to denote random variables. This means that the next state $X_{n+1}$ is not determined solely by the current state $X_n$, but rather has a distribution of possible outcomes based on some probability distribution $P$. In the context of diffusion models, this randomness can be thought of as random movements of particles due to thermal fluctuations or other sources of noise just like Brownian motion. 

When discussing Bernoulli random variables such as repeated coin flips, we implicitly discussed a simple stochastic process where each coin flip is independent of the previous ones. So in other words, each coin flip is a random variable $X_n$ that takes on the value 1 (heads) with probability $p$ and the value 0 (tails) with probability $1-p$. The sequence of coin flips can be thought of as a stochastic process where each flip is independent of the previous ones. Importantly, we remember that in each coin flip, the outcome only depends on the probability $p$ and not on the previous outcomes. So in other words, the probability of getting heads on the next flip is always $p$ regardless of how many heads or tails we got in previous flips. Meaning that the process has no memory of previous states. This property is known as the **Markov property**. More formally, the markov property states that the conditional probability distribution of future states of the process depends only on the present state, not on the sequence of events that preceded it:

$$
P(X_{n+1} | X_n, X_{n-1}, ..., X_0) = P(X_{n+1} | X_n)
$$

We call a stochastic process that satisfies the Markov property a **Markov Chain**. In a Markov Chain, the future state depends only on the current state and not on the sequence of events that preceded it. In general, this doesn't have to be the case for all stochastic processes. WE can easily come up with examples of stochastic processes that do not satisfy the Markov property. For example, consider we want to model the weather as a stochastic process. The weather on a given day may depend not only on the weather of the previous day but also on the weather of several days before that. In this case, the process does not satisfy the Markov property because the future state (weather) depends on more than just the current state.

In the rest of these notes, we will mostly work with **discrete-time, discrete-state** Markov chains, since these are the ones that connect directly to graphs, matrices, PageRank and other useful concepts. However, you should keep in the back of your mind that the same Markov idea applies far beyond the discrete setting.

{{< callout type="example" >}}
Let us look at a simple weather model which we will use as a running example. The state space is as follows:

$$
S = {\text{Sunny}, \text{Cloudy}, \text{Rainy}}
$$

so we have a discrete state space with three possible weather conditions. We will also assume discrete time steps $n = 0,1,2,\dots$ where $X_n$ is the weather on day $n$. We can now specify a markov chain by giving the probabilities of moving from one weather state to another. For example, we might say:

- If today is Rainy, then tomorrow is Rainy again with probability $0.5$, Cloudy with probability $0.3$, and Sunny with probability $0.2$.
- If today is Cloudy, then tomorrow is Rainy with probability $0.2$, Cloudy again with probability $0.4$, and Sunny with probability $0.4$.
- If today is Sunny, then tomorrow is Rainy with probability $0.1$, Cloudy with probability $0.1$, and Sunny again with probability $0.8$.

Notice that importantly, the probabilities of tomorrow's weather depend only on today's weather and not on any previous days. This is the Markov property in action. Also every possible current weather state has a full probability distribution over the next day's weather added up to 1. 

If we say that the following values correspond to the states 

$$
\text{Sunny} = 1, \quad \text{Cloudy} = 2, \quad \text{Rainy} = 3
$$

we can formalize the transition probabilities as follows:

$$
\mathbb{P}(X_{n+1} = j \mid X_n = i) = p_{ij}
$$

where $i,j \in {1,2,3}$ correspond to the weather states. For example, $p_{31} = 0.2$ is the probability of going from Rainy (state 3) to Sunny (state 1).
markovChainWeather.webp
{{< /callout >}}

### Homogeneous Markov Chains

We now focus on the setting that is most useful in computer science and machine learning, namely discrete-time Markov chains with a finite state space. So using the the markov property, we can then describe the probabilities of moving from one state to another using the following notation:

$$
p_{ij} := \mathbb{P}(X_{n+1} = j \mid X_n = i)
$$

we call these probabilities the **transition probabilities**. Specifically, $p_{ij}$ is the probability of moving from state $i$ to state $j$ in one time step. We notice that these probabilities do not depend on the time step $n$. This is an important simplification and is what people usually mean by a **time-homogeneous Markov chain**. In other words, the transition probabilities are the same at every time step. In general, we could have **time-inhomogeneous chains** where the transition probabilities depend on $n$, but then the analysis gets heavier and we lose the nice simplifications. An example of a time-inhomogeneous chain could be a weather model where the transition probabilities change with the seasons. For example, in summer, the probability of going from Sunny to Rainy might be lower than in winter.

An important property of the transition probabilities is that they must form valid probability distributions. So for each fixed state $i$, the probabilities to move from $i$ to any state $j$ must be non-negative and add up to 1:

$$
p_{ij} \ge 0
\quad \text{and} \quad
\sum_{j=1}^{m} p_{ij} = 1 \quad \text{for each } i \in {1,\dots,m}
$$

We can collect all these numbers into a matrix to then get a compact representation of the Markov chain. This matrix $P$ is called the **transition matrix** of the Markov chain, or sometimes also the **transition or markov kernel**. Because each row is a probability distribution over next states, we call such matrices **row-stochastic**.

$$
P = (p_{ij})_{i,j=1}^m =
\begin{bmatrix}
p_{11} & p_{12} & \dots & p_{1m} \\
p_{21} & p_{22} & \dots & p_{2m} \\
\vdots & \vdots & \ddots & \vdots \\
p_{m1} & p_{m2} & \dots & p_{mm}
\end{bmatrix}
$$

the computer scientist in you should recognize this looks like an adjacency matrix of a directed graph where each state is a node and there is a directed edge from node $i$ to node $j$ if $p_{ij} > 0$. The only difference is that in an adjacency matrix the entries are usually just 0s and 1s indicating the absence or presence of an edge, whereas here we have probabilities between 0 and 1 indicating how likely it is to move along that edge. A markov chain can therefore also be interpreted as a random walk on a directed, graph where the edge weights are given by the transition probabilities. This representation is extremely powerful, because it connects Markov chains to graph theory, linear algebra and other areas of computer science. 

{{< callout type="example" >}}
If we go back to our weather example, we can write down the transition matrix as follows:

$$
P = \begin{bmatrix}
0.8 & 0.1 & 0.1 \\
0.3 & 0.4 & 0.3 \\
0.2 & 0.3 & 0.5
\end{bmatrix}
$$

and we can visualize the Markov chain as a directed graph:

{{< figure 
    src="/images/maths/markovChainWeather.webp"
    alt="Graph representation of a Markov chain modeling weather transitions between Sunny, Cloudy, and Rainy states."
    caption="Graph representation of the weather Markov chain."
    width="400"
>}}
{{< /callout >}}

Because we can represent Markov chains using matrices, we can use tools from linear algebra to analyse them. In particular, matrix multiplication and eigenvalues/eigenvectors will play a key role in understanding the long-term behaviour of Markov chains. First, let's see what it would look like if we started in some initial state and then let the chain evolve over time. We can encode our initial state as a distribution over states:

$$
q^{(0)} = (q^{(0)}_1, \dots, q^{(0)}_m)
$$

where $m$ is the number of states and each entry represents the probability of starting in that state. So if we are guaranteed to start in state $i$, we have a one-hot vector with $q^{(0)}_i = 1$ and all other entries 0. If we start in state $i$ with probability $0.5$ and state $j$ with probability $0.5$, we have $q^{(0)}_i = 0.5$, $q^{(0)}_j = 0.5$ and all other entries 0. This can then of course be generalized to any time step $n$:

$$
q^{(n)} = (q^{(n)}_1, \dots, q^{(n)}_m) \text{ where } q^{(n)}_i = \mathbb{P}(X_n = i) \text{ and } \sum_{i=1}^m q^{(n)}_i = 1
$$

now let's see how we can compute the distribution at time step $n+1$ based on the distribution at time step $n$. From the law of total probability, we have:

$$
\mathbb{P}(X_{n+1} = j) = \sum_{i=1}^m \mathbb{P}(X_{n+1} = j \mid X_n = i) \mathbb{P}(X_n = i)
$$

Inserting our notation, this becomes:

$$
q^{(n+1)}_j = \sum_{i=1}^m p_{ij} q^{(n)}_i
$$

we quickly recognize that this is just matrix multiplication! If we write $q^{(n)}$ as a row vector, we can express the entire distribution update as:

$$
q^{(n+1)} = q^{(n)} P
$$

where $P$ is the transition matrix of the Markov chain. This is a very powerful relation because it allows us to compute the distribution at any time step by simply multiplying the initial distribution by powers of the transition matrix. If we unroll this recursion, we also get:

$$
q^{(n)} = q^{(0)} P^n
$$

So the entire time evolution of the Markov chain is captured by the powers of the transition matrix $P$. This can obviously be computed efficiently using standard matrix multiplication algorithms and iterative squaring for large powers.

### Regular Markov Chains

A central question in the study of Markov chains is what happens in the long run. Specifically, we want to understand the behaviour of the distribution $q^{(n)}$ as $n$ goes to infinity. Do we converge to some sort of stationary/equilibrium distribution, or does the chain keep changing forever or oscillate between different distributions? We define such a stationary distribution as a vector that remains unchanged when we apply the transition matrix. More formally:

$$
q^\ast = q^\ast P
$$

This means that if we start in the distribution $q^\ast$, we stay in that distribution forever. Such a distribution is also called an **invariant distribution** or **equilibrium distribution**. Finding stationary distributions is important because they often represent the long-term behaviour of the Markov chain. If we start in any initial distribution $q^{(0)}$ and let the chain evolve, we would like to know if $q^{(n)}$ converges to $q^\ast$ as $n$ goes to infinity. This stationary distribution can be interpreted as the long-term probabilities of being in each state after many transitions as the probabilities stabilize over time. 

It is obvious that if we start in the stationary distribution, we remain in it. So if $q^{(0)} = q^\ast$, then $q^{(n)} = q^\ast$ for all $n$. But what if we start somewhere else? Will we eventually converge to $q^\ast$? The answer depends on the structure of the Markov chain.

So if $X_0$ is distributed according to $\mu^\ast$, then $X_1$ is also distributed according to $\mu^\ast$, and the same holds for all $X_n$. In other words, if you start in equilibrium, you stay in equilibrium. But what if you start somewhere else? Will the distribution of $X_n$ converge to $\mu^\ast$ as $n \to \infty$? 

{{< callout type="todo" >}}
Regular markov chains, irreducible, aperiodic, etc. convergence to unique stationary distribution. Link to page rank and eigenvalues of transition matrix.
Detailed balance condition? 
This then also leads to the idea of markov chain monte carlo methods where we want to sample from some complicated distribution by designing a markov chain that has that distribution as its stationary distribution.
{{< /callout >}}

### Learning Markov Models

So far we have treated Markov chains as given: somebody hands us the transition matrix $P$ and an initial distribution $q^{(0)}$ and we analyse what happens. However, in practice in computer science and machine learning, we often have to go the other way around. We have data that we believe was generated by some underlying Markov process, but we don't know the transition probabilities. Our goal is to learn or estimate these probabilities from the data. 

{{< callout type="todo" >}}
Estimating transition probabilities from data. Maximum likelihood estimation. Counting transitions in observed sequences. Smoothing techniques for small data. Applications in natural language processing, bioinformatics, etc.
Bi-grams and n-grams as markov models for text.
{{< /callout >}}

### Hidden Markov Models

{{< callout type="todo" >}}
What are hidden markov models, forward algorithm, viterbi algorithm, baum welch algorithm? Is this related to partially observable markov decision processes?
{{< /callout >}}