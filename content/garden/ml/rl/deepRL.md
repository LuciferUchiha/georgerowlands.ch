---
title: Deep Reinforcement Learning
type: docs
weight: 6
---

In [Tabular Reinforcement Learning](/garden/ml/rl/tabularrl), we can represent the value functions and policies using tables, which is feasible for small state and action spaces. However, in many real-world applications, the state and action spaces are large or continuous, making tabular methods impractical. To address this challenge, we can use function approximation techniques, such as neural networks, to represent the value functions and policies. This leads us to the field of **deep reinforcement learning (deep RL)**, which combines reinforcement learning with deep learning.

Importantly we treat discrete and continuous action spaces differently here. If we for example approximate a policy with a neural network in a discrete action space, we can simply have the network output a probability distribution over the discrete actions. This then becomes a classification problem where the output layer uses a softmax activation to produce probabilities for each action. In contrast, in continuous action spaces, we do not just want it to output a single action like in regression, but rather a distribution over actions. To model this, we can have the neural network output the parameters of a probability distribution (e.g., mean and variance for a Gaussian distribution) from which we can then sample actions. This allows us to capture the uncertainty in action selection and also enables exploration in continuous action spaces.

{{< figure 
    src="/images/ml/rlDiscreteContinousActions.jpg"
    caption="Illustration of deep reinforcement learning in discrete and continuous action spaces."
    alt="Illustration of deep reinforcement learning in discrete and continuous action spaces."
>}}

## Model-Free Methods

To show that function approximation can work in RL, we can show that the TD and Q-learning updates can be interpreted as optimization problems, specifically making them a form of stochastic gradient descent (SGD). In tabular TD-learning, the update rule is:

$$
v_\pi(s_t) \leftarrow v_\pi(s_t) + \alpha \left[ r_{t+1} + \gamma v_\pi(s_{t+1}) - v_\pi(s_t) \right].
$$

where we are essentially trying to perform a stochastic approximation of the Bellman expectation equation $v_\pi(s) = \mathcal{B}_\pi v_\pi(s)$}. Each update step produces a estimate of the bellman target $r_{t+1} + \gamma v_\pi(s_{t+1})$. Therefore if we parameterize the value function with parameters $\theta$, we can consider a loss function that measures the squared difference between the current value estimate and the bellman target:

$$
\hat{L}_{TD}(s, r, s'; \theta) =  \frac{1}{2} \left( v_\pi(s) - v_\pi(s; \theta) \right)^2 = \frac{1}{2} \left( r + \gamma v_\pi(s') - v_\pi(s; \theta) \right)^2.
$$

where the target is $v_\pi(s) = r + \gamma v_\pi(s')$ is the expected return following the bellman equation. We can then find the negative gradient of this loss with respect to the parameters $\theta$:

$$
\nabla_\theta \hat{L}_{TD}(s, r, s'; \theta) = v_\pi(s; \theta) - \left( r + \gamma v_\pi(s') \right).
$$

This gradient points in the direction of reducing the difference between the current value estimate and the bellman target. However, we do not have access to the true value function $v_\pi(s')$, so instead we can use bootstrapping and replace it with our current estimate $v_\pi(s'; \theta)$. This results in the following loss function:

$$
L_{TD}(s, r, s'; \theta) =  \frac{1}{2} \left( r + \gamma v_\pi(s'; \theta^{old}) - v_\pi(s; \theta) \right)^2,
$$

where $\theta^{old}$ are the parameters from the previous iteration used to compute the target. Then taking the negative gradient of this loss gives us the TD update rule:

$$
\delta_{TD} = -\nabla_\theta L_{TD}(s, r, s'; \theta) = r + \gamma v_\pi(s'; \theta^{old}) - v_\pi(s; \theta).
$$

also known as the TD error. So if we then perform gradient descent on this loss function using stochastic gradient descent (SGD), we recover the TD update rule:

$$
\begin{align*}
v_\pi(s; \theta) &\leftarrow  v_\pi(s; \theta^{old}) - \alpha \nabla_\theta L_{TD}(s, r, s'; \theta) \\
&= v_\pi(s; \theta^{old}) + \alpha \left[ r + \gamma v_\pi(s'; \theta^{old}) - v_\pi(s; \theta) \right].
\end{align*}
$$

### Deep Q-Networks (DQN)

Above we showed that TD learning can be interpreted as optimizing a loss function based on the Bellman equation using stochastic gradient descent. Similarly, we can show that Q-learning can also be interpreted as optimizing a loss function based on the Bellman optimality equation. Specifically, in tabular Q-learning, the update rule is:

$$
q(s_t, a_t) \leftarrow q(s_t, a_t) + \alpha \left[ r_{t+1} + \gamma \max_{a'} q(s_{t+1}, a') - q(s_t, a_t) \right].
$$

If we then use a function approximator parameterized by $\theta$ to represent the action-value function, we can define a loss function that measures the squared difference between the current action-value estimate and the bellman optimality target:

$$
\hat{L}_{Q}(s, a, r, s'; \theta) =  \frac{1}{2} \left( q(s, a) - q(s, a; \theta) \right)^2 = \frac{1}{2} \left( r + \gamma \max_{a'} q(s', a') - q(s, a; \theta) \right)^2.
$$

This loss function is known as the **Squared Bellman Optimality Error (SBOE)**. Here again we use bootstrapping to replace the true action-value function $q(s', a')$ with our current estimate $q(s', a'; \theta_{old})$ when computing the target. This results in the following loss function:

$$
L_{Q}(s, a, r, s'; \theta) =  \frac{1}{2} \left( r + \gamma \max_{a'} q(s', a'; \theta^{old}) - q(s, a; \theta) \right)^2,
$$

where $\theta^{old}$ are the parameters from the previous iteration used to compute the target. Taking the negative gradient of this loss with respect to the parameters $\theta$ gives us:

$$
\delta_{Q} = -\nabla_\theta L_{Q}(s, a, r, s'; \theta) = r + \gamma \max_{a'} q(s', a'; \theta^{old}) - q(s, a; \theta).
$$

This gradient points in the direction of reducing the difference between the current action-value estimate and the bellman optimality target. If we then perform gradient descent on this loss function using stochastic gradient descent (SGD), we recover the Q-learning update rule:

$$
\begin{align*}
q(s, a; \theta) &\leftarrow  q(s, a; \theta^{old}) - \alpha \nabla_\theta L_{Q}(s, a, r, s'; \theta) \\
&= q(s, a; \theta^{old}) + \alpha \left[ r + \gamma \max_{a'} q(s', a'; \theta^{old}) - q(s, a; \theta) \right] \\
&= q(s, a; \theta^{old}) + \alpha \delta_{Q}.
\end{align*}
$$

Which is exactly the Q-learning update rule and thus shows that Q-learning can also be interpreted as optimizing a loss function based on the Bellman optimality equation using stochastic gradient descent. If we were to use the following linear function approximator:

$$
q(s, a; \theta) = \theta^T \phi(s,a)
$$

where $\phi(s,a)$ is a feature vector representing the state-action pair $(s,a)$ such as one-hot encoding or more complex features like radial basis functions (RBFs), and $\theta$ is the weight vector we are trying to learn, we can compute the update and its gradient exactly as:

$$
\begin{align*}
q(s, a; \theta) &\leftarrow q(s, a; \theta^{old}) + \alpha \nabla_\theta \theta^{old\,T}\phi(s,a) \\
&= q(s, a; \theta^{old}) + \alpha \phi(s,a).
\end{align*}
$$

Google DeepMind extended this idea to use deep neural networks as function approximators for the action-value function, leading to the development of the **neural fitted Q-iteration or deep Q-networks(DQN)** algorithm. DQN demonstrated that deep neural networks could effectively approximate the action-value function in high-dimensional state spaces, such as those encountered in Atari 2600 games, achieving human-level performance on many of them. Here the gradient updates are computed using backpropagation through the neural network and automatic differentiation.

One of the issues with DQN is that the bootstrapping estimate changes after each update, as the parameters $\theta$ are updated, which is expensive and can lead to instability and divergence during training. Especially since consecutive samples are often highly correlated (think of a robot moving in a room, in the next time step it will be in a very similar position as before), which violates the i.i.d. assumption made by many optimization algorithms like SGD.

To address this, Google DeepMind introduced the concept **replay buffers** and **target networks** in DQN. Specifically the "online network" with parameters $\theta$ is used to select actions and update the Q-values, while a separate "target network" with parameters $\theta^{old}$ is used to compute the target values in the loss function. The target network's parameters are updated less frequently (e.g. when the replay buffer $D$ is big enough) by copying the weights from the online network. We then update the online network by sampling mini-batches of experience tuples $(s, a, r, s')$ from the replay buffer and performing gradient descent on the following loss function:

$$
L_{DQN}(s, a, r, s'; \theta, \theta^{old}) =  \frac{1}{2} \left( r + \gamma \max_{a'} q(s', a'; \theta^{old}) - q(s, a; \theta) \right)^2,
$$

This technique is known as **experience replay**, and it helps break the correlations between consecutive samples by randomly sampling from a buffer of past experiences. This leads to more stable and efficient learning.

{{< figure 
    src="/images/ml/rlDQN.png"
    caption="Architecture of Deep Q-Network (DQN) with experience replay and target network."
    alt="Architecture of Deep Q-Network (DQN) with experience replay and target network."
>}}

{{< figure 
    src="/images/ml/rlDQNExperienceReplay.jpg"
    caption="Pseudo-code illustration of the Deep Q-Network (DQN) algorithm with experience replay and target network."
    alt="Pseudo-code illustration of the Deep Q-Network (DQN) algorithm with experience replay and target network."
>}}

Another issue with DQN is that it can overestimate action-values due to the max operator in the target or in other words it has a maximization bias, which can lead to suboptimal policies. To address this, an extension called **Double DQN (DDQN)** was proposed to decouple the action selection and action evaluation in the target computation. Specifically, in Double DQN, instead of picking the optimal action with respect to the old network, it picks the optimal action with respect to the new network:

$$
L_{DDQN}(s, a, r, s'; \theta, \theta^{old}) =  \frac{1}{2} \left( r + \gamma q(s', \arg\max_{a'} q(s', a'; \theta); \theta^{old}) - q(s, a; \theta) \right)^2,
$$

This helps reduce the overestimation bias and leads to more accurate action-value estimates. The key difference is that the action that maximizes the Q-value is selected using the online network (with parameters $\theta$), while the value of that action is evaluated using the target network (with parameters $\theta^{old}$). Making the estimation less biased and more stable.

{{< figure 
    src="/images/ml/rlDDQN.png"
    caption="Comparison of DQN and Double DQN (DDQN) target computation."
    alt="Comparison of DQN and Double DQN (DDQN) target computation."
>}}

### Policy Gradient Methods

In Reinforcement Learning, we often distinguish between **Evaluation** (evaluating a policy, i.e., finding its value function) and **Control** (finding the optimal policy). Value-based methods like DQN solve the Control problem indirectly: they predict the value of actions and then derive a policy by being greedy. 

The problem with these value-based methods is that they are only learning the policy indirectly through the value function. This can lead to suboptimal policies, especially in complex environments where the value function may not accurately capture the optimal policy. Additionally, they can struggle in high-dimensional or continuous action spaces, as finding the action that maximizes the Q-value can be challenging. 

Instead we want to directly learn the policy itself, which leads us to **Policy search/gradient** methods. These methods tackle the Control problem directly. They parameterize the policy itself and adjust its parameters to maximize the expected return, effectively "controlling" the agent's behavior without necessarily needing a value function as an intermediate step (though we often use one to reduce variance, as we'll see in Actor-Critic methods).

Whereas with Q-learning, exploration can be encouraged by using an epsilon-greedy policy, softmax exploration, or an optimistic initialization, policy gradient methods typically use stochastic policies that inherently encourage exploration by sampling actions according to a probability distribution.

So we want to directly learn a parameterized policy $\pi_\phi(a|s)$ (where $\phi$ are the parameters, e.g., weights of a neural network). This leads us to **Policy Gradient methods**. We define a performance objective $J(\phi)$ representing the expected return, and we optimize the policy parameters $\phi$ directly using stochastic gradient ascent:

$$
\phi \leftarrow \phi + \eta \nabla_\phi J(\phi).
$$

Our objective function is the expected return over trajectories generated by the policy:

$$
J(\phi) = \mathbb{E}_{\tau \sim \Pi_\phi} \left[ G_{0} \right] = \mathbb{E}_{\tau \sim \Pi_\phi} \left[ \sum_{t=0}^{\infty} \gamma^t R_t \right]
$$

We can also define a version using a finite horizon $T$ which for larger $T$ approaches the infinite horizon case:

$$
J(\phi) \approx J_T(\phi) =
\mathbb{E}_{\tau \sim \Pi_\phi} \left[ G_{0:T} \right] = \mathbb{E}_{\tau \sim \Pi_\phi} \left[ \sum_{t=0}^{T-1} \gamma^t R_t \right]
$$

$\Pi_\phi(\tau)$ is the probability distribution over trajectories $\tau = (s_0, a_0, r_0, s_1, ...)$ induced by the policy $\pi_\phi$. For an episodic task, the trajectory distribution can be expressed as:

$$
\Pi_\phi(\tau) = p(s_0) \prod_{t=0}^{T-1} \pi_\phi(a_t | s_t) P(s_{t+1} | s_t, a_t)
$$

To optimize the policy parameters $\phi$, we need to compute the gradient of the objective function $J(\phi)$ with respect to $\phi$. However, directly computing this gradient is challenging due to the dependence of the trajectory distribution $\Pi_\phi(\tau)$ on the environment dynamics $P(s'|s,a)$, which are unknown.

Instead, we can use the **policy gradient theorem**, which provides a way to compute the policy gradient without requiring knowledge of the environment dynamics. The idea is that we can find a surrogate objective whose gradient matches that of the expected return without needing to know the transition probabilities. The policy gradient theorem states that the gradient of the expected return can be expressed as:

$$
\nabla_\phi J_T(\phi)
= \mathbb{E}_{\tau \sim \Pi_\phi} [G_{0:T}]
= \mathbb{E}_{\tau \sim \Pi_\phi} \left[ G_{0:T} \sum_{t=0}^{T-1} \nabla_\phi \log \pi_\phi(a_t | s_t) \right]
$$

Here the function $\log \pi_\phi(a_t | s_t)$ is the log-probability of taking action $a_t$ in state $s_t$ under the policy $\pi_\phi$ is the so-called **score function**. 

{{< callout type="proof" >}}
We can derive this surrogate objective by using the log-derivative trick (also known as the score function trick) from statistics, which allows us to move the gradient operator inside the expectation:

$$
\nabla_\phi \Pi_\phi(\tau) = \Pi_\phi(\tau) \nabla_\phi \log \Pi_\phi(\tau)
$$

Then we can derive the policy gradient as follows:

$$
\begin{align*}
\nabla_\phi J_T(\phi) &= \nabla_\phi \mathbb{E}_{\tau \sim \Pi_\phi} \left[ G_{0:T} \right] \\
&= \nabla_\phi \int \Pi_\phi(\tau) G_{0:T} d\tau \\
&= \int G_{0:T} \nabla_\phi \Pi_\phi(\tau) d\tau \\
&= \int G_{0:T} \Pi_\phi(\tau) \nabla_\phi \log \Pi_\phi(\tau) d\tau \\
&= \mathbb{E}_{\tau \sim \Pi_\phi} \left[ G_{0:T} \nabla_\phi \log \Pi_\phi(\tau) \right]
\end{align*}
$$

{{< /callout >}}

So to calculate the policy gradient, we just need to calculate:

$$
\begin{align*}
\nabla_\phi \log \Pi_\phi(\tau) &= \nabla_\phi \left( \log p(s_0) + \sum_{t=0}^{T-1} \log \pi_\phi(a_t | s_t) + \log P(s_{t+1} | s_t, a_t) \right) \\
&= \nabla_\phi p(s_0) + \nabla_\phi \sum_{t=0}^{T-1} \log \pi_\phi(a_t | s_t) + \nabla_\phi \log P(s_{t+1} | s_t, a_t) \\
&= \sum_{t=0}^{T-1} \nabla_\phi \log \pi_\phi(a_t | s_t)
\end{align*}
$$

Crucially the terms involving the initial state distribution $p(s_0)$ and the environment dynamics $P(s'|s,a)$ vanish, as they do not depend on the policy parameters $\phi$. This is what allows us to compute the policy gradient without knowledge of the environment dynamics. So we can express the policy gradient and our unbiased score function estimator as:

$$
\nabla_\phi J_T(\phi)
= \mathbb{E}_{\tau \sim \Pi_\phi} \left[ G_{0:T} \sum_{t=0}^{T-1} \nabla_\phi \log \pi_\phi(a_t | s_t) \right]
$$

#### REINFORCE Algorithm

The simplest policy gradient algorithm is the **REINFORCE** algorithm, which uses Monte Carlo estimates of the return to compute the policy gradient. Specifically, we can estimate the policy gradient using sampled trajectories as follows:

$$
\hat{\nabla}_\phi J_T(\phi) = \frac{1}{N} \sum_{i=1}^{N} \left[ G_{0:T}^{(i)} \sum_{t=0}^{T-1} \nabla_\phi \log \pi_\phi(a_t^{(i)} | s_t^{(i)}) \right]
$$

Here, $N$ is the number of sampled trajectories, and $G_{0:T}^{(i)}$ is the return for the $i$-th trajectory. The REINFORCE algorithm updates the policy parameters via stochastic gradient ascent (ascent since we are maximizing the expected return):

$$
\phi \leftarrow \phi + \alpha \hat{\nabla}_\phi J_T(\phi).
$$

where $\alpha$ is the learning rate. Intuitively, the REINFORCE algorithm increases the probability of actions that lead to higher returns and decreases the probability of actions that lead to lower returns.

However, there is a major issue with REINFORCE, which is that despite being an unbiased estimator, the policy gradient estimate has very high variance due to the use of the full return $G_{0:T}$. This high variance can lead to slow convergence and instability during training. To address this, we can introduce **baselines** to reduce the variance of the policy gradient estimate. The key idea is that subtracting a baseline $b$ from the return does not change the expected value of the policy gradient, but it can significantly reduce its variance:

$$
\mathbb{E}_{\tau \sim \Pi_\phi} \left[ \left( G_{0:T} - b \right) \sum_{t=0}^{T-1} \nabla_\phi \log \pi_\phi(a_t | s_t) \right] = \mathbb{E}_{\tau \sim \Pi_\phi} \left[ G_{0:T} \sum_{t=0}^{T-1} \nabla_\phi \log \pi_\phi(a_t | s_t) \right]
$$

{{< callout type="proof" >}}
We can prove that subtracting a baseline does not change the expected value of the policy gradient by first using the linearity of expectation to separate the two terms:

$$
\begin{align*}
&\mathbb{E}_{\tau \sim \Pi_\phi} \left[ \left( G_{0:T} - b \right) \sum_{t=0}^{T-1} \nabla_\phi \log \pi_\phi(a_t | s_t) \right] \\
&= \mathbb{E}_{\tau \sim \Pi_\phi} \left[ G_{0:T} \sum_{t=0}^{T-1} \nabla_\phi \log \pi_\phi(a_t | s_t) \right] - \mathbb{E}_{\tau \sim \Pi_\phi} \left[ b \sum_{t=0}^{T-1} \nabla_\phi \log \pi_\phi(a_t | s_t) \right]
\end{align*}
$$

We then want to show that the second term is zero:

$$
\begin{align*}
\mathbb{E}_{\tau \sim \Pi_\phi} \left[ b \sum_{t=0}^{T-1} \nabla_\phi \log \pi_\phi(a_t | s_t) \right] &= b \mathbb{E}_{\tau \sim \Pi_\phi} \left[ \sum_{t=0}^{T-1} \nabla_\phi \log \pi_\phi(a_t | s_t) \right] \\
&= b \sum_{t=0}^{T-1} \mathbb{E}_{\tau \sim \Pi_\phi} \left[ \nabla_\phi \log \pi_\phi(a_t | s_t) \right] \\
&= b \sum_{t=0}^{T-1} \int \Pi_\phi(\tau) \nabla_\phi \log \pi_\phi(a_t | s_t) d\tau \\
&= b \sum_{t=0}^{T-1} \int \nabla_\phi \Pi_\phi(\tau) d\tau \\
&= b \sum_{t=0}^{T-1} \nabla_\phi \int \Pi_\phi(\tau) d\tau \\
&= b \sum_{t=0}^{T-1} \nabla_\phi 1 \\
&= b \sum_{t=0}^{T-1} 0 \\
&= 0
\end{align*}
$$
{{< /callout >}}

There are several choices for the baseline $b$. A popular choice is to use the reward-to-go rather than the full return, which considers only the rewards received after time step $t$:

$$
G_{t:T} = \sum_{k=t}^{T-1} \gamma^{k-t} R_k
$$

this way we only consider "downstream" rewards when updating the policy at time step $t$ and do not include rewards which cannot be influenced by the action taken at time step $t$. This can further reduce variance. The reward-to-go can be constructed by using the return minus the bounded discounted payoff from time $t$:

$$
G_{t:T} = G_{0:T} - \sum_{k=0}^{t-1} \gamma^k R_k
$$

so our baseline becomes a time and state-dependent baseline $b_t = G_{0:t}$. This leads to the following policy gradient estimate:

$$
\hat{\nabla}_\phi J_T(\phi) = \frac{1}{N} \sum_{i=1}^{N} \left[ \sum_{t=0}^{T-1} \left( G_{t:T}^{(i)} \nabla_\phi \log \pi_\phi(a_t^{(i)} | s_t^{(i)}) \right) \right]
$$

Another popular choice is to use independent baseline such as the average return across the sampled trajectories:

$$
b = \frac{1}{N} \sum_{i=1}^{N} G_{0:T}^{(i)}
$$

this way all the gradient estimates are centered around the average return, which can help stabilize learning and reduce variance. The gradient vectors that yield above-average returns will have a positive contribution, while those yielding below-average returns will have a negative contribution, encouraging the policy to favor actions that lead to higher returns and disfavor actions that lead to lower returns. By subtracting this baseline the magnitudes of the gradient estimates are also reduced, which can help prevent large updates that could destabilize learning. 

Rather than just subtracting the mean, we can also devide by the standard deviation to standardize the returns:

$$
\hat{\nabla}_\phi J_T(\phi) = \frac{1}{N} \sum_{i=1}^{N} \left[ \sum_{t=0}^{T-1} \left( \frac{G_{t:T}^{(i)} - \mu}{\sigma} \nabla_\phi \log \pi_\phi(a_t^{(i)} | s_t^{(i)}) \right) \right]
$$

where $\mu$ and $\sigma$ are the mean and standard deviation of the returns across the sampled trajectories. This standardization helps ensure that the gradient estimates have a consistent scale, which can improve the stability and convergence of the learning process.

However, even with baselines, the REINFORCE algorithm can still suffer from high variance and is not garantueed to converge to a local optimum due to getting stuck in poor local optima. 

### Actor–Critic

reinforce where we use advantage as a baseline? this means we need to learn a value function as well hence the critic?

Hybrid methods combining TD learning with policy gradients.

But also a version where we start with Q Learning rather then TD learning?

* Actor updates the policy
* Critic estimates value function
* Lower variance than pure policy gradients

online/q actor critic? We additionally learn an action-value function $q_\pi(s,a; w)$ parameterized by $w$ using TD learning such as SARSA or Q-learning. The critic provides an estimate of returns to the actor, which uses this information to update the policy parameters $\phi$. 

by using the bootstrapped value estimates from the critic, we can reduce the variance of the policy gradient estimate however this introduces bias since the value estimates may not be accurate.

#### Advantage Actor–Critic

what if also learned a state-value function 
 and use it as a baseline to compute advantage estimates. End up just learning the advantage function directly.

 Things with positive negative?

Introduce Advantage function to reduce variance.

Balance bias-variance trade-off of the policy gradient estimate results in GAE?

### Trust Region Policy Optimization (TRPO)

This is the clipped version? to increase sample efficiency and stability.

### Proximal Policy Optimization (PPO)
A stable and practical policy gradient method.

* Avoiding destructive large policy updates
* Clipped surrogate objective
* Why PPO works well in practice

special case of TRPO?

### Group Related Policy Optimization (GRPO)

### Maximum Entropy RL (MERL)

motivates exploration by augmenting the reward with an entropy term.

soft actor-critic (SAC)?

### Deep Deterministic Policy Gradient (DDPG)

T3, twin delayed DDPG (TD3)? Similar idea to DQN and double DQN to reduce overestimation bias.

### Stochastic Value Gradients (SVG)

### Soft Actor-Critic (SAC)

soft Q-learning?

soft actor-critic (SAC)?

## Model-Based

World models

### Model Predictive Control (MPC)

random shooting methods

for stochastic dynamics we use:

trajectory sampling methods like thompson sampling?

### Pilco?

### Pets?

### Hallucinated upper confidence reinforcement learning

### Constrained and Safe RL
