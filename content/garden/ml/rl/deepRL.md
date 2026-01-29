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

where $\alpha$ is the learning rate. Equivalently, we can express this as minimizing the following **policy loss**:

$$
L_{REINFORCE}(\phi) = -\frac{1}{N} \sum_{i=1}^{N} \left[ G_{0:T}^{(i)} \sum_{t=0}^{T-1} \log \pi_\phi(a_t^{(i)} | s_t^{(i)}) \right]
$$

The negative sign converts the gradient ascent problem into a gradient descent problem, which is the standard form used by most deep learning frameworks. Intuitively, the REINFORCE algorithm increases the probability of actions that lead to higher returns and decreases the probability of actions that lead to lower returns.

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

Intuitively, this means that at each time step $t$, we update the policy based on the rewards that are actually influenced by the action taken at that time step, removing any causality-violating information from future rewards. Hence, it is also often referred to as the causality trick.

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

```python title="REINFORCE Algorithm Pseudo-code"
Inputs: policy pi_phi(a|s), discount gamma, step size alpha
Option: standardize_returns = True/False

repeat (over training iterations):
    D = empty list                                        # batch of trajectories

    for episode = 1..E:
        tau = rollout(env, pi_phi)                        # tau has (s_t, a_t, r_t), t=0..T-1
        append tau to D
    end for

    # compute reward-to-go for all timesteps in the batch
    G_all = empty list                                    # will store all G[t] across all trajectories
    for each trajectory tau in D:
        for t = 0..T-1:
            G[t] = sum_{k=t..T-1} (gamma^(k-t) * r[k])
            append G[t] to G_all
        end for
        store G[t] with tau
    end for

    # optional standardization (across the whole batch)
    if standardize_returns:
        mu    = mean(G_all)
        sigma = std(G_all) + 1e-8
    end if

    # policy update
    grad = 0
    for each trajectory tau in D:
        for t = 0..T-1:
            if standardize_returns:
                G_hat = (G[t] - mu) / sigma
            else:
                G_hat = G[t]
            end if
            grad += G_hat * grad_phi log pi_phi(a[t] | s[t])
        end for
    end for
    phi = phi + alpha * grad                               # gradient ascent
until done
```

### Actor–Critic

We saw in the Gradient theorem that we need to compute the expected return $G_{0:T}$ to estimate the policy gradient. In the REINFORCE algorithm, we used Monte Carlo estimates of the return, which can have high variance due to multiplying a noisy reward signal over long trajectories with a high dimensional policy gradient. 

To reduce this variance, we introduce the idea of an **Actor-Critic** method. The key idea is that rather then using the full return $G_{0:T}$, we can use a learned value function to estimate the expected return, which can provide lower-variance estimates. This estimated value function is called the **critic**, while the policy being optimized is called the **actor**. The critic evaluates the actions taken by the actor and provides feedback to improve the policy, in a way the critic can also be thought of as bootstrapping the return estimates or providing a learned baseline for the policy gradient.

Specifically, we can use  **Temporal Difference (TD) learning** (such as SARSA or Q-learning) to estimate the action-value function $q_\pi(s,a)$ or state-value function $v_\pi(s)$, which can then be used to compute the policy gradient, making it a **model-free** hybrid method of value-based and policy-based methods. The most common approach is the **online actor-critic** method, also called **Q-Actor-Critic (Q-AC)**, which uses the action-value function to estimate the expected return. This works out well since we can rewrite the policy gradient using the action-value function:

$$
\nabla_\phi J_T(\phi) = \mathbb{E}_{\tau \sim \Pi_\phi} \left[ \sum_{t=0}^{T-1} q_\pi(s_t, a_t) \nabla_\phi \log \pi_\phi(a_t | s_t) \right]
$$

{{< callout type="proof" >}}

{{< /callout >}}

To actually implement this, we need to learn the action-value function $q_\pi(s,a)$ using TD learning. We can use the SARSA update rule to learn the action-value function:

$$
\delta_{TD} = r_{t+1} + \gamma q_\pi(s_{t+1}, a_{t+1}) - q_\pi(s_t, a_t)
$$

where $a_{t+1} \sim \pi_\phi(\cdot | s_{t+1})$ is the action taken in the next state according to the current policy. We can then use this TD error $\delta_{TD}$ to update the action-value function using stochastic gradient descent:

$$
q_\pi(s_t, a_t) \leftarrow q_\pi(s_t, a_t) + \alpha \delta_{TD}
$$

We can then use the learned action-value function to compute the policy gradient estimate and update the policy parameters via stochastic gradient ascent:

$$
\phi \leftarrow \phi + \alpha \hat{\nabla}_\phi J_T(\phi) = \phi + \alpha \delta_{TD} \nabla_\phi \log \pi_\phi(a_t | s_t)
$$

The corresponding **loss functions** for the Q-Actor-Critic method are for the critic:

$$
L_{Q}(\theta) = \frac{1}{2} \left( r + \gamma q_\pi(s', a'; \theta^{old}) - q_\pi(s, a; \theta) \right)^2
$$

just like in Q-learning, where we try to minimize the squared Bellman error for the action-value function, and for the actor:

$$
L_{\pi}(\phi) = -q_\pi(s, a; \theta) \log \pi_\phi(a | s)
$$

where we try to maximize the expected action-value under the current policy, which is equivalent to minimizing the negative expected action-value.Importantly the learning rates for the actor and critic can be different, i.e., we can use $\alpha_{actor}$ and $\alpha_{critic}$. This method effectively combines the benefits of value-based and policy-based methods, allowing for more efficient learning and better convergence properties by reducing the variance of the policy gradient estimates. However, remember that since we are using bootstrapping to estimate the action-value function, the policy gradient estimate can be biased, which can lead to suboptimal policies if not handled carefully, in comparison the REINFORCE algorithm is unbiased but has high variance. So we are again facing a bias-variance trade-off.

{{< figure 
    src="/images/ml/rlActorCritic.png"
    caption="Architecture of an Actor-Critic method with separate networks for the actor (policy) and critic (value function)."
    alt="Architecture of an Actor-Critic method with separate networks for the actor (policy) and critic (value function)."
    width="600"
>}}

#### Advantage Actor–Critic (A2C)

An issue with the above Q-AC method is not only did we introduce bias through bootstrapping, but the variance can still be high since the action-value function $Q$ can have high variance itself and can be hard to learn. Depending on the design of the reward function, the action-value function can vary significantly for different actions in the same state, leading to noisy estimates. Because we are multiplying the action-value function with the policy gradient depending on the reward function design this can lead to destructively large updates. In addition if for example the reward function is always positive, the action-value function will also always be positive, leading to consistently positive updates to the policy parameters, despite some actions being worse than others. To mitigate this, we can use the **Advantage function** instead of the action-value function in the policy gradient estimate:

$$
A_\pi(s,a) = q_\pi(s,a) - v_\pi(s)
$$

The advantage function measures how much better or worse an action is compared to the average action in that state, as given by the state-value function. By using the advantage function, we can center the action-value estimates around the state-value, which helps reduce variance and stabilize learning. If the action taken is better than average, the advantage will be positive, leading to an increase in the probability of taking that action. Conversely, if the action is worse than average, the advantage will be negative, leading to a decrease in the probability of taking that action. This centering effect helps ensure that the policy updates are more balanced and less sensitive to the absolute values of the action-value function. 

So the policy gradient estimate becomes:

$$
\hat{\nabla}_\phi J_T(\phi) = \mathbb{E}_{\tau \sim \Pi_\phi} \left[ \sum_{t=0}^{T-1} (q_\pi(s_t, a_t) - v_\pi(s_t)) \nabla_\phi \log \pi_\phi(a_t | s_t) \right] = \mathbb{E}_{\tau \sim \Pi_\phi} \left[ \sum_{t=0}^{T-1} A_\pi(s_t, a_t) \nabla_\phi \log \pi_\phi(a_t | s_t) \right]
$$

So if we learn both the action-value function $q_\pi(s,a)$ and the state-value function $v_\pi(s)$ using TD learning, we can compute the advantage function and use it in the policy gradient estimate. We can also see that if $q_\pi(s,a)$ represents the expected return following the bellman equation then the state-value function $v_\pi(s)$ can be used as a baseline to reduce variance in the policy gradient estimate.

A naive approach would be to learn both $q_\pi(s,a)$ and $v_\pi(s)$ separately using TD learning, but this can be inefficient and redundant since both functions are related. Specifically remember that the action-value function can be expressed in terms of the state-value function and the policy:

$$
\begin{align*}
q_\pi(s,a) &= \mathbb{E}_\pi \left[ G_t | S_t = s, A_t = a \right] \\
&= \mathbb{E}_\pi \left[ R_{t+1} + \gamma v_\pi(S_{t+1}) | S_t = s, A_t = a \right] \\
&\approx r + \gamma v_\pi(s')
\end{align*}
$$

So instead of learning both functions separately, we can just learn the state-value function $v_\pi(s)$ and use it to then calculate the approximated action-value function and thus the advantage function:

$$
A_\pi(s,a) = q_\pi(s,a) - v_\pi(s) \approx r + \gamma v_\pi(s') - v_\pi(s)
$$

Notice that the term on the right is exactly the TD error $\delta_{TD}$ we used to update the action-value function in the Q-AC method. So we can use the TD error as an unbiased estimate of the advantage function:

$$
\hat{A}_\pi(s,a) \approx r + \gamma v_\pi(s') - v_\pi(s) = \delta_{TD}
$$

This leads to the **Advantage Actor-Critic (A2C)** method, where we only learn the state-value function $v_\pi(s)$ using TD learning and use the TD error as an estimate of the advantage function in the policy gradient estimate. So we update the critic, i.e., the state-value function, using the TD error:

$$
v_\pi(s_t) \leftarrow v_\pi(s_t) + \alpha \delta_{TD}
$$

The loss function for the critic can be defined as the squared TD error:

$$
L_{V}(s, r, s'; \theta) =  \frac{1}{2} \left( r + \gamma v_\pi(s'; \theta^{old}) - v_\pi(s; \theta) \right)^2,
$$

And we update the actor, i.e., the policy, using the TD error as an estimate of the advantage function:

$$
\phi \leftarrow \phi + \alpha A_\pi(s_t, a_t) \nabla_\phi \log \pi_\phi(a_t | s_t) = \phi + \alpha \delta_{TD} \nabla_\phi \log \pi_\phi(a_t | s_t)
$$

The corresponding actor loss for A2C is:

$$
L_{\pi}(\phi) = -A_\pi(s, a) \log \pi_\phi(a | s) = -\delta_{TD} \log \pi_\phi(a | s)
$$

{{< figure 
    src="/images/ml/rlA2C.png"
    caption="Architecture of the Advantage Actor-Critic (A2C) method."
    alt="Architecture of the Advantage Actor-Critic (A2C) method."
    width="300"
>}}

```python title="Advantage Actor-Critic (A2C) Algorithm Pseudo-code"
Algorithm: A2C (one-step TD advantage)

Inputs: policy pi_phi(a|s), value V_theta(s),
        discount gamma, actor step alpha_pi, critic step alpha_v

repeat (over training iterations):
    s = env.reset()

    while not done:
        a ~ pi_phi(. | s)                            # sample action from policy
        s_next, r, done = env.step(a)

        # critic target and TD error
        v      = V_theta(s)
        v_next = 0 if done else V_theta(s_next)
        delta  = r + gamma * v_next - v              # TD error = advantage estimate

        # critic update: minimize (delta^2)
        theta = theta - alpha_v * grad_theta (delta^2)

        # actor update: maximize log pi * advantage  (or minimize negative)
        phi = phi + alpha_pi * ( delta * grad_phi log pi_phi(a | s) )

        s = s_next
    end while
until done
```

#### Generalized Advantage Estimation (GAE)

Using the one-step TD error as an estimate of the advantage function in the A2C method reduces variances compared to using the full return, but it instead introduces bias due to bootstrapping. To balance this bias-variance trade-off, we can use **n-step returns** to estimate the advantage function. The n-step return considers rewards over multiple time steps before bootstrapping, which can help reduce bias while still keeping variance manageable:

- **1-step TD Error**: $\delta_t^{(1)} = r_{t+1} + \gamma v_\pi(s_{t+1}) - v_\pi(s_t)$.
- **2-step TD Error**: $\delta_t^{(2)} = r_{t+1} + \gamma r_{t+2} + \gamma^2 v_\pi(s_{t+2}) - v_\pi(s_t)$.
- **n-step TD Error**: $\delta_t^{(n)} = \sum_{k=0}^{n-1} \gamma^k r_{t+k+1} + \gamma^n v_\pi(s_{t+n}) - v_\pi(s_t)$.

Using n-step returns, we can define the advantage estimate as:

$$
A_t^{GAE(\lambda)} = \sum_{k=0}^{\infty} (\gamma \lambda)^k \delta_{t+k} = \delta_t + \gamma \lambda \delta_{t+1} + (\gamma \lambda)^2 \delta_{t+2} + \cdots + (\gamma \lambda)^{n-1} \delta_{t+n-1}
$$

where $\delta_{t+k}$ is the TD error at time step $t+k$, and $\lambda \in [0,1]$ is a parameter that controls the bias-variance trade-off. When $\lambda = 0$, we recover the one-step TD error, which has low variance but high bias and results in the loss:

$$
L_{V}(s, r, s'; \theta) =  \frac{1}{2} \left( r + \gamma v_\pi(s'; \theta^{old}) - v_\pi(s; \theta) \right)^2,
$$

When $\lambda = 1$, we recover the full return loss from the Monte Carlo method, which is unbiased but has high variance:

$$
L_{V}(s, r, s'; \theta) =  \frac{1}{2} \left( G_{0:T} - v_\pi(s; \theta) \right)^2,
$$

By choosing an intermediate value for $\lambda$, we can balance bias and variance in the advantage estimates. In practice, values of $\lambda$ between 0.9 and 0.99 are often used to achieve a good balance. We can rewrite the advantage as follows:

$$
\begin{align*}
A_t^{GAE(\lambda)} &= (r_{t+1} + \gamma v_\pi(s_{t+1})) - v_\pi(s_t) + \gamma \lambda \left( (r_{t+2} + \gamma v_\pi(s_{t+2})) - v_\pi(s_{t+1}) \right) + (\gamma \lambda)^2 \left( (r_{t+3} + \gamma v_\pi(s_{t+3})) - v_\pi(s_{t+2}) \right) + \cdots \\
&= \hat{V}_t^{\lambda} - v_\pi(s_t)
\end{align*}
$$

where $\hat{V}_t^{\lambda}$ is the $\lambda$-return defined as:

$$
\hat{V}_t^{\lambda} = (1 - \lambda) \sum_{n=1}^{\infty} \lambda^{n-1} V_t^{(n)}
$$

and $V_t^{(n)}$ is the n-step return:

$$
V_t^{(n)} = \sum_{k=0}^{n-1} \gamma^k r_{t+k+1} + \gamma^n v_\pi(s_{t+n})
$$

This is kind of like a weighted telecoping sum of n-step returns, where the weights decrease exponentially with $n$ controlled by $\lambda$. Using this $\lambda$-return, we can define the critic loss as:

$$
L_{V}(s, r, s'; \theta) =  \frac{1}{2} \left( \hat{V}_t^{\lambda} - v_\pi(s; \theta) \right)^2,
$$

The actor loss remains the same as in A2C, using the GAE advantage estimates:

$$
L_{\pi}(\phi) = -A_t^{GAE(\lambda)} \log \pi_\phi(a | s)
$$

{{< figure 
    src="/images/ml/rlTDLambda.webp"
    caption="Impact of the lambda parameter in TD(λ) on bias and variance."
    alt="Impact of the lambda parameter in TD(λ) on bias and variance."
    width="600"
>}}

```python title="Advantage Actor-Critic with GAE (A2C-GAE) Algorithm Pseudo-code"
Algorithm: Actor-Critic with GAE(lambda) + K critic updates

Inputs: pi_phi(a|s), V_theta(s), gamma, lambda
Hyperparams: actor stepsize alpha_pi, critic stepsize alpha_v
Loop params: critic_updates K

repeat (over epochs):
    D = collect trajectories using pi_phi                 # store (s_t, a_t, r_t, done_t)

    # compute values and GAE advantages (and value targets)
    for each trajectory in D:
        A_next = 0
        for t = T-1 down to 0:
            v      = V_theta(s[t])
            v_next = 0 if done[t] else V_theta(s[t+1])
            delta  = r[t] + gamma * v_next - v
            A[t]   = delta + gamma * lambda * A_next
            R[t]   = A[t] + v                              # value target
            A_next = A[t]
        end for
    end for

    # critic update: do multiple gradient steps on the same batch targets R_t
    for k = 1..K:
        theta = theta - alpha_v * grad_theta mean( (V_theta(s_t) - R_t)^2 )
    end for

    # actor update (uses A_t computed above)
    phi = phi + alpha_pi * grad_phi mean( log pi_phi(a_t|s_t) * A_t )
until done
```

#### Asynchronous Advantage Actor-Critic (A3C)

Google deepmind's A3C algorithm. For now we will skip this.

### Trust Region Policy Optimization (TRPO)

A major limitation of the methods above is sample inefficiency. Because they are on-policy, we collect data, perform one gradient update, and then must discard the data. The idea of Trust Region Policy Optimization (TRPO) is to enable faster convergence by defining a trust region constraint on the policy updates, allowing multiple updates per batch of data, while ensuring that the policy does not change too much in a single update, which could lead to performance collapse. The key idea is to optimize a surrogate objective function subject to a constraint on the KL-divergence between the old and new policies to ensure that the new policy is not too different from the old policy:

$$
\phi \leftarrow \arg\max_\phi \hat{J(\phi)} \quad \text{subject to} \quad D_{KL}(\pi_{\phi^{old}} || \pi_\phi) \leq \delta
$$

where $\hat{J(\phi)}$ is the surrogate objective function, $D_{KL}(\pi_{\phi^{old}} || \pi_\phi)$ is the KL-divergence between the old policy $\pi_{\phi^{old}}$ and the new policy $\pi_\phi$, and $\delta$ is a small positive constant that defines the size of the trust region. 

{{< figure 
    src="/images/ml/rlTRPO.png"
    caption="Trust Region Policy Optimization (TRPO) constrains policy updates to stay within a trust region defined by the KL-divergence between the old and new policies."
    alt="Trust Region Policy Optimization (TRPO) constrains policy updates to stay within a trust region defined by the KL-divergence between the old and new policies."
    width="400"
>}}

The surrogate objective function will now not only receive data from the current policy but also from the old policy. To make sure we are optimizing the expected return of the new policy, we need to correct for the fact that the data was collected under the old policy. This is done using **importance sampling** with the importance sampling ratio:

$$
w_t(a_t, s_t) = \frac{\pi_\phi(a_t | s_t)}{\pi_{\phi^{old}}(a_t | s_t)}
$$

This ratio corrects for the difference in action probabilities between the old and new policies and avoid destructive updates. Intuitively, if the new policy assigns a higher probability to an action than the old policy did, the importance weight will be greater than 1, increasing the contribution of that action to the surrogate objective. Conversely, if the new policy assigns a lower probability, the weight will be less than 1, reducing its contribution. This ensures that the surrogate objective accurately reflects the expected return under the new policy, even though the data was collected under the old policy. If we now use this importance sampling ratio to reweight the advantage estimates in the policy gradient estimate, we obtain the surrogate objective function:

$$
\hat{J(\phi)} = \mathbb{E}_{\tau \sim \Pi_{\phi^{old}}} \left[ \sum_{t=0}^{T-1} w_t(a_t, s_t) A_{\pi_{\phi^{old}}}(s_t, a_t) \right]
$$

So we update the critc (value function) as usual using TD learning, but we update the actor (policy) by solving the constrained optimization problem above.

$$
\phi \leftarrow \arg\max_\phi \mathbb{E}_{\tau \sim \Pi_{\phi^{old}}} \left[ \sum_{t=0}^{T-1} \frac{\pi_\phi(a_t | s_t)}{\pi_{\phi^{old}}(a_t | s_t)} A_{\pi_{\phi^{old}}}(s_t, a_t) \right] \quad \text{subject to} \quad D_{KL}(\pi_{\phi^{old}} || \pi_\phi) \leq \delta
$$

The problem with this constrained optimization problem is that it is difficult to solve directly. Instead, TRPO uses a second-order approximation of the KL-divergence constraint and a linear approximation of the surrogate objective to derive a closed-form solution for the policy update. This involves calculating the Fisher Information Matrix, which captures the curvature of the KL-divergence constraint, and using it to compute a natural gradient step that respects the trust region constraint. However, the computation of the Fisher Information Matrix and its inverse can be computationally expensive, especially for high-dimensional policy parameterizations.

### Proximal Policy Optimization (PPO)

Proximal Policy Optimization (PPO) was introduced by OpenAI as a simpler alternative to TRPO that achieves similar performance with much less computational overhead. PPO retains the benefits of TRPO (avoiding destructively large policy updates) and improving sample efficiency, while using only first-order optimization methods like standard stochastic gradient descent (SGD). This makes it significantly easier to implement and tune, which is why it has become one of the most popular algorithms in practice, including being used to improve Large Language Models via Reinforcement Learning from Human Feedback (RLHF) as shown in the InstructGPT paper by OpenAI.

{{< figure 
    src="/images/ml/rlInstructGPT.jpg"
    caption="The training of InstructGPT involves a combination of supervised fine-tuning and reinforcement learning using PPO to align the model with human preferences."
    alt="The training of InstructGPT involves a combination of supervised fine-tuning and reinforcement learning using PPO to align the model with human preferences."
    width="600"
>}}

The key insight of PPO is to replace the hard KL-divergence constraint of TRPO with a clipped surrogate objective that automatically prevents large policy updates. Let $r_t(\phi) = \frac{\pi_\phi(a_t | s_t)}{\pi_{\phi^{old}}(a_t | s_t)}$ denote the probability ratio between the new and old policies. The **clipped surrogate objective** is:

$$
L^{CLIP}(\phi) = \mathbb{E}_t \left[ \min \left( r_t(\phi) A_t, \text{clip}(r_t(\phi), 1-\epsilon, 1+\epsilon) A_t \right) \right]
$$

where $\epsilon$ is a hyperparameter (typically 0.1 or 0.2) that defines how far the new policy can deviate from the old policy. By taking the minimum, the objective function ignores changes in probability that move outside the trust region defined by $[1-\epsilon, 1+\epsilon]$. This prevents the policy from changing too much in a single update, similar to the KL-divergence constraint in TRPO, but without the need for complex second-order optimization.

The intuition behind the $\min$ operation is as follows:
- When the action was better than expected, with advantage $A_t > 0$, we want to increase $r_t(\phi)$ to make this action more likely. However, the clipping prevents $r_t(\phi)$ from exceeding $1+\epsilon$, limiting how much we can increase the probability.
- When the action was worse than expected, with advantage $A_t < 0$, we want to decrease $r_t(\phi)$ to make this action less likely. The clipping prevents $r_t(\phi)$ from falling below $1-\epsilon$, limiting how much we can decrease the probability.

This creates a "trust region" effect similar to TRPO but without the computational complexity of solving a constrained optimization problem, instead we just simply optimize the clipped objective using standard gradient ascent methods:

$$
\phi \leftarrow \phi + \alpha \nabla_\phi L^{CLIP}(\phi)
$$

{{< figure 
    src="/images/ml/rlPPO.png"
    caption="Visualization of the PPO clipped surrogate objective. The clipping prevents large policy updates that would lead to destructive performance drops."
    alt="Visualization of the PPO clipped surrogate objective. The clipping prevents large policy updates that would lead to destructive performance drops."
    width="400"
>}}

```python title="Proximal Policy Optimization (PPO) Algorithm Pseudo-code"
Algorithm: PPO (clipped surrogate) + GAE(lambda) + minibatches

Inputs: policy pi_phi(a|s), value V_theta(s)
Hyperparams: gamma, lambda, clip_eps, actor step alpha_pi, critic step alpha_v
Loop params: episodes_per_epoch E, ppo_epochs N, minibatch_size B

repeat (over epochs):
    D = empty buffer

    # 1) collect on-policy data with current policy
    for episode = 1..E:
        tau = rollout(env, pi_phi)                         # store (s_t, a_t, r_t, done_t, s_{t+1})
        add tau to D
    end for

    # 2) compute values, next values, old log-probs (frozen), GAE advantages, and value targets
    for each trajectory in D:
        T = length(trajectory)
        A_next = 0

        for t = 0..T-1:
            s_t = s[t]; a_t = a[t]
            v[t] = V_theta(s_t)
            logp_old[t] = log pi_phi(a_t | s_t)            # store and detach (freeze)
        end for

        for t = T-1 down to 0:
            if done[t]:
                v_next = 0
            else:
                v_next = v[t+1]                            # value of next state (already computed)
            end if

            delta = r[t] + gamma * v_next - v[t]
            A[t]  = delta + gamma * lambda * A_next        # GAE recursion
            R[t]  = A[t] + v[t]                            # value target
            A_next = A[t]
        end for
    end for

    # (optional but common) normalize advantages across the whole buffer D
    A_mu    = mean(all A[t] in D)
    A_sigma = std(all A[t] in D) + 1e-8
    for all timesteps in D:
        A[t] = (A[t] - A_mu) / A_sigma
    end for

    # 3) PPO optimization: multiple epochs, minibatches
    for ppo_epoch = 1..N:
        shuffle all timesteps in D
        partition D into minibatches of size B

        for minibatch M:
            # recompute new log-probs under current policy
            logp = log pi_phi(a|s) for (s,a) in M

            # importance ratio
            ratio = exp(logp - logp_old)                   # elementwise

            # PPO clipped objective (maximize)
            obj1 = ratio * A
            obj2 = clip(ratio, 1-clip_eps, 1+clip_eps) * A
            L_clip = mean( min(obj1, obj2) )

            # critic loss (minimize)
            L_V = mean( (V_theta(s) - R)^2 )

            # updates (conceptual gradient steps)
            phi   = phi   + alpha_pi * grad_phi   L_clip
            theta = theta - alpha_v  * grad_theta L_V
        end for
    end for
until done
```

### Deep Deterministic Policy Gradient (DDPG)

So far all the policy gradient methods are on-policy methods, except for TRPO and PPO which are kind of in-between on-policy and off-policy. Deep Deterministic Policy Gradient (DDPG) is on the other hand a off-policy actor-critic algorithm designed it does this by combining the ideas from Deep Q-Networks (DQN) with deterministic policy gradients for continuous action spaces (hence the deterministic in the name). Unlike stochastic policy gradient methods, DDPG learns a deterministic policy $\mu_\phi(s)$ that directly outputs the action to take in a given state, rather than a probability distribution over actions. 

DDPG combines ideas from DQN (experience replay and target networks) with the deterministic policy gradient theorem. We start with the 2 networks from DQN, a Q-network $q(s,a; \theta)$ parameterized by $\theta$ to estimate the action-value function, and a target Q-network $q(s,a; \theta^{old})$ parameterized by $\theta^{old}$ to provide stable target values. In addition, we introduce a policy network $\mu_\phi(s)$ parameterized by $\phi$ to represent the deterministic policy. In DQN, we used the bootstrapped squared Bellman error to train the Q-network:

$$
\hat{L}_Q(\theta) = \mathbb{E}_{(s,a,r,s') \sim D} \left[ \left( r + \gamma \max_{a'} q(s', a'; \theta^{old}) - q(s, a; \theta) \right)^2 \right]
$$

In DDPG, we modify this to use our deterministic policy to select the action in the next state for the target:

$$
\hat{L}_Q(\theta) = \mathbb{E}_{(s,a,r,s') \sim D} \left[ \left( r + \gamma q(s', \mu_\phi(s'); \theta^{old}) - q(s, a; \theta) \right)^2 \right]
$$

The goal of the actor is to pick actions that maximize the critic's estimated value. Since the action space is continuous and the Q-function is differentiable with respect to the action $a$, we can use the chain rule to compute the gradient of the expected return with respect to the policy parameters $\phi$:

$$
\nabla_\phi J(\phi) = \mathbb{E}_{s \sim D} \left[ \nabla_a q(s, a; \theta) |_{a=\mu_\phi(s)} \nabla_\phi \mu_\phi(s) \right]
$$

This can be expressed as the following **actor loss** (to be minimized):

$$
L_\mu(\phi) = -\mathbb{E}_{s \sim D} \left[ Q_\theta(s, \mu_\phi(s)) \right]
$$

{{< callout type="proof" >}}
Key insight is that since the policy is deterministic, we can directly differentiate through the Q-function with respect to the action chosen by the policy. This allows us to compute how changes in the policy parameters $\phi$ affect the expected return by considering how they change the actions selected by the policy and how those actions affect the Q-values estimated by the critic.
{{< /callout >}}

However, since the policy is deterministic, we need to ensure sufficient exploration during training. This is typically done by adding noise to the actions selected by the policy during data collection:

$$
a_t = \mu_\phi(s_t) + \epsilon_t \quad \text{where} \quad \epsilon_t \sim \mathcal{N}(0, \sigma^2)
$$

This encourages exploration of the action space while still allowing the policy to learn a deterministic mapping from states to actions.

**Twin Delayed DDPG (TD3)** extends DDPG to address overestimation bias (similar to Double DQN) by using two critic networks and taking the minimum Q-value for the target and delaying policy updates. We also just like before add some noise to the target action to smooth out Q-value estimates.

### Stochastic Value Gradients (SVG)

also off-policy actor-critic extension of DDPG to stochastic policies. Instead of learning a deterministic policy $\mu_\phi(s)$, SVG learns a stochastic policy $\pi_\phi(a | s)$ that outputs a distribution over actions given a state. The key idea is to use the reparameterization trick to express the stochastic policy in terms of a deterministic function and some independent noise:

$$
a = g_\phi(s, \epsilon) \quad \text{where} \quad \epsilon \sim p(\cdot)
$$

where $g_\phi(s, \epsilon)$ is a deterministic function parameterized by $\phi$ that takes the state $s$ and some noise $\epsilon$ sampled from some known distribution $p(\cdot)$ to produce an action $a$. For example, if $p$ is a Gaussian, $a = \mu_\phi(s) + \sigma_\phi(s) \cdot \epsilon$ where $\mu_\phi(s)$ and $\sigma_\phi(s)$ are the mean and standard deviation output by the policy network.

Then we can differentiate the expected Q-value with respect to $\phi$:

$$
\nabla_\phi J(\phi) = \nabla_\phi \mathbb{E}_{\epsilon} [Q(s, g(\epsilon; s, \phi))] = \mathbb{E}_{\epsilon} [\nabla_a Q(s, a)|_{a=g(\epsilon; s, \phi)} \cdot \nabla_\phi g(\epsilon; s, \phi)]
$$

This allows us to train stochastic policies using the low-variance "pathwise" gradients (chain rule) of DDPG rather than the high-variance score function gradients of REINFORCE, while still being off-policy. If the policy is deterministic, SVG reduces to DDPG.

### Maximum Entropy RL (MERL)

Crucial aspect of reinforcement learning is the exploration-exploitation trade-off. Traditional RL methods often struggle with exploration, especially in high-dimensional or sparse reward environments. An idea from information theory called **Maximum Entropy Reinforcement Learning** addresses this by encouraging policies to be as random as possible while still achieving high returns. This is done by augmenting the standard RL objective with an entropy term that promotes exploration:

$$
J(\pi) = \sum_{t=0}^{T} \mathbb{E}_{(s_t, a_t) \sim \rho_\pi} \left[ r(s_t, a_t) + \alpha H(\pi(\cdot | s_t)) \right]
$$

where $\alpha$ is the temperature parameter controlling the trade-off between reward maximization and entropy maximization. Entropy $H(\pi(\cdot | s_t))$ is defined as:

$$
H(\pi(\cdot | s_t)) = -\mathbb{E}_{a_t \sim \pi(\cdot | s_t)} [\log \pi(a_t | s_t)]
$$

Entropy can be interpreted as a measure of uncertainty or randomness in the policy's action distribution. Low entropy means the policy is more deterministic ("peaked"), while high entropy means the policy is more stochastic ("spread out"). By maximizing entropy, we encourage the policy to explore more diverse actions, which can help discover better strategies and avoid premature convergence to suboptimal policies. 

{{< figure 
    src="/images/ml/entropy.jpg"
    caption="Entropy measures the uncertainty in a probability distribution. Higher entropy indicates a more uniform distribution, while lower entropy indicates a more concentrated distribution."
    alt="Entropy measures the uncertainty in a probability distribution. Higher entropy indicates a more uniform distribution, while lower entropy indicates a more concentrated distribution."
    width="400"
>}}

Popular algorithms that utilize this framework include Soft Actor-Critic (SAC) (discussed next) and an entropy-augmented version of Proximal Policy Optimization (PPO):

$$
L_{PPO}(\phi) = \mathbb{E}_t \left[ \min \left( r_t(\phi) A_t, \text{clip}(r_t(\phi), 1-\epsilon, 1+\epsilon) A_t \right) + \alpha H(\pi_\phi(\cdot | s_t)) \right]
$$

Note that the entropy term had a minus sign in the definition, but here it is added to the objective since we are minimizing the loss but aiming to maximize entropy.

{{< callout type="todo" >}}
MDPS can also be used for the reverse diffusion process in generative modeling. Here the maximum entropy principle can help ensure we don't collapse to a single mode but rather explore the full data distribution.
{{< /callout >}}

### Soft Actor-Critic (SAC)

Soft Actor-Critic (SAC) is state of the art off-policy actor-critic algorithm that combines the benefits of maximum entropy reinforcement learning, the stability of actor-critic methods, and the sample efficiency of off-policy learning. 

also MPO which stands for Maximum a Posteriori Policy Optimization.

## Model-Based Reinforcement Learning

In the previous sections, we focused on **model-free** methods (DQN, PPO, SAC) which learn value functions or policies directly from interaction. While effective, these methods are often **sample inefficient**, requiring millions of interactions.

**Model-Based Reinforcement Learning (MBRL)** attempts to address this by learning a **model of the environment** (often called a **World Model**). The agent learns the transition dynamics $s_{t+1} \sim p(s'|s,a)$ and the reward function $r(s,a)$ from data. It then uses this model to **plan** (simulate futures) to select actions or update its policy.

The core advantages are:
1.  **Sample Efficiency:** We can use the learned model to generate infinite "imagined" data for training.
2.  **Safety:** We can predict the consequences of actions before executing them in the real world.
3.  **Exploration:** Uncertainty in the model can drive structured exploration.
4.  
{{< figure 
    src="/images/ml/rlModelBased.png"
    caption="Model-Based Reinforcement Learning (MBRL) involves learning a model of the environment's dynamics and using it for planning and policy optimization."
    alt="Model-Based Reinforcement Learning (MBRL) involves learning a model of the environment's dynamics and using it for planning and policy optimization."
    width="600"
>}}

### Model Predictive Control (MPC)

Assume for a moment that we **know** the dynamics $f$ (where $s_{t+1} = f(s_t, a_t)$) and rewards $r$. How do we select actions? We maximize the discounted return:

$$
\max_{a_{0:\infty}} \sum_{t=0}^{\infty} \gamma^t r(s_t, a_t) \quad \text{subject to} \quad s_{t+1} = f(s_t, a_t)
$$

Since optimizing over an infinite horizon is intractable, we typically use **Model Predictive Control (MPC)/ Receding Horizon Control**. In MPC, at each time step $t$, we:

1.  Observe current state $s_t$.
2.  Plan optimal actions over a **finite horizon** $H$: $a_{t:t+H-1}$.
3.  Execute only the **first action** $a_t$.
4.  Wait for the next state $s_{t+1}$, shift the horizon, and replan.

To account for the truncated horizon, we often append a **terminal value function** $V(s_{t+H})$ to estimate the rewards beyond $H$. The objective becomes:

$$
J_H(a_{t:t+H-1}) = \underbrace{\sum_{\tau=t}^{t+H-1} \gamma^{\tau-t} r(s_\tau, a_\tau)}_{\text{Short-term planning}} + \underbrace{\gamma^H V(s_{t+H})}_{\text{Long-term estimate}}
$$

{{< figure 
    src="/images/ml/rlMPC.png"
    caption="Model Predictive Control (MPC) plans actions over a finite horizon H, executes the first action, and replans at the next time step."
    alt="Model Predictive Control (MPC) plans actions over a finite horizon H, executes the first action, and replans at the next time step."
    width="500"
>}}

This approach balances short-term planning with long-term value estimates, allowing the agent to make informed decisions without needing to plan indefinitely. In particular there is also a nice connection to model-free RL here. If we set the horizon $H=1$, the objective simplifies to:

$$
\arg\max_a \left[ r(s_t, a) + \gamma V(f(s_t, a)) \right]
$$

This effectively recovers the greedy policy used in value-based model-free methods. Thus, MBRL with finite horizons can be seen as a generalization of model-free RL, where we explicitly look ahead $H$ steps before relying on the cached value estimate.

#### Stochastic Dynamics & Trajectory Sampling

If the environment is stochastic, $s_{t+1} \sim p(s'|s,a)$, a sequence of actions induces a distribution over trajectories. We maximize the **expected** return:

$$
\max_{a_{t:t+H-1}} \mathbb{E} \left[ J_H \right]
$$

Since this expectation involves high-dimensional integrals, we approximate it using **Monte Carlo Trajectory Sampling**. We sample $M$ trajectories (rollouts) for a given action sequence and average the returns.

{{< figure 
    src="/images/ml/rlStochaticMPC.png"
    caption="Trajectory Sampling involves simulating multiple rollouts of action sequences through the stochastic model to estimate expected returns."
    alt="Trajectory Sampling involves simulating multiple rollouts of action sequences through the stochastic model to estimate expected returns."
    width="500"
>}}

To optimize the action sequence through these stochastic samples, we use the **Reparameterization Trick** (just like in SVG or VAEs). If the transition can be written as $s_{t+1} = g(s_t, a_t, \epsilon_t)$ where $\epsilon_t$ is independent noise (e.g., Gaussian), we can backpropagate gradients from the reward function through the transition model all the way to the actions.

#### Parametric Policies (Open-Loop vs Closed-Loop)

MPC performs **closed-loop control** (re-planning every step), which is computationally expensive. To speed this up, we can "distill" the planning result into a parametric policy $\pi_\phi(s)$ (similar to the Actor in Actor-Critic).

We train the policy to mimic the planner or directly maximize the $H$-step objective:

$$
\phi^* = \arg\max_\phi \mathbb{E}_{s \sim \mu} \left[ \sum_{t=0}^{H-1} \gamma^t r(s_t, \pi_\phi(s_t)) + \gamma^H Q(s_H, \pi_\phi(s_H)) \right]
$$

This looks exactly like the **DDPG** or **SVG** objective, but expanded over $H$ time steps. By unrolling the model $H$ steps, the gradients flow through the transition dynamics, allowing the policy to anticipate consequences $H$ steps ahead.

### Learning the Model

In reality, $f$ and $r$ are unknown. We typically treat this as a **supervised learning** problem. We collect a replay buffer $\mathcal{D} = \{(s, a, r, s')\}$ and train a function approximator (e.g., a Neural Network or Gaussian Process) to minimize the negative log-likelihood (NLL).

For a **probabilistic** dynamics model (capturing aleatoric uncertainty), we often output a Gaussian:

$$
p_\theta(s_{t+1} | s_t, a_t) = \mathcal{N}(\mu_\theta(s_t, a_t), \Sigma_\theta(s_t, a_t))
$$

We train this by maximizing $\log p_\theta(s'|s,a)$.

In MBRL, distinguishing uncertainty types is crucial:
- **Aleatoric Uncertainty:** Inherent randomness in the system (e.g., noisy wind). Modeled by the output variance $\Sigma_\theta$.
- **Epistemic Uncertainty:** Ignorance about the model itself due to lack of data. This is what drives exploration.

A standard single Neural Network (MLE/MAP estimate) does **not** capture epistemic uncertainty. Planning with a single, potentially incorrect point-estimate model leads to "model exploitation"—the planner finds trajectories where the model erroneously predicts high reward (e.g., walking through walls).

#### Probabilistic Ensembles (PETS)

The **Probabilistic Ensembles with Trajectory Sampling (PETS)** algorithm addresses this by learning an **ensemble** of probabilistic neural networks.
- **Epistemic uncertainty** is captured by the variance *between* the predictions of different ensemble members.
- **Aleatoric uncertainty** is captured by the output variance of individual members.

During planning (trajectory sampling), we sample a bootstrap head (a model from the ensemble) to generate a trajectory. This prevents the planner from exploiting the error of a single model.

{{< figure 
    src="/images/ml/rlPets.png"
    caption="PETS uses an ensemble of probabilistic networks. Trajectory sampling (TS) propagates a particle through a single model for the horizon H to maintain consistent uncertainty correlations."
    alt="Illustration of PETS planning with ensembles"
    width="500"
>}}

### Exploration

Simply adding noise (epsilon-greedy or dithering) to policy actions is insufficient for complex tasks. We need **Directed Exploration** using the model's uncertainty. 

#### Thompson Sampling

A popular Bayesian exploration strategy is **Thompson Sampling**. The idea is to act according to a single model sampled from the posterior distribution over models. A simple, effective heuristic. At the start of an episode:
1.  Sample a single model $\tilde{f}$ from the posterior (e.g., pick one network from the PETS ensemble).
2.  Plan optimally assuming $\tilde{f}$ is the true environment. So we construct a policy $\pi^*$ that maximizes expected return under $\tilde{f}$.
3.  We execute $\pi^*$ in the real environment for the episode and collect the data to update our model posterior.
   
If $\tilde{f}$ was wrong, we visit new states that falsify it, reducing epistemic uncertainty. If $\tilde{f}$ was correct, we gain high reward. Over time, this balances exploration and exploitation.

#### Optimistic Exploration & Hallucination (H-UCRL)

Based on the principle of **Optimism in the Face of Uncertainty (OFU)**. We want to act according to the *best possible* model that is statistically plausible given our data.

**Hallucinated Upper Confidence RL (H-UCRL)** formalizes this. Instead of a fixed transition, the agent is allowed to "choose" the outcome $s_{t+1}$ from the model's confidence interval.
*   Let the model predict a mean $\mu(s,a)$ and uncertainty $\sigma(s,a)$.
*   We introduce a **control variable** ("luck") $\eta_t \in [-1, 1]$.
*   The "optimistic" transition is $s_{t+1} = \mu(s,a) + \eta_t \cdot \beta \sigma(s,a)$.

The agent jointly optimizes actions $a$ and hallucinated luck $\eta$ to maximize reward.

$$
\max_{\pi, \eta} J(\pi, \eta)
$$

Basically, the agent plans: *"If I go to this unknown area, I assume the dynamics will push me exactly where I want to go."* This optimistic assumption drives the agent to unknown areas. If the assumption fails (real dynamics differ), the uncertainty $\sigma$ decreases, and the plan is corrected.

### Safe Exploration

In real-world robotics, exploration must be **safe**. We want to maximize reward subject to keeping cost $c(s)$ below a threshold $\delta$.

$$
\max_\pi J_r(\pi) \quad \text{subject to} \quad J_c(\pi) \le \delta
$$

To be safe and efficient, we combine:
1.  **Optimism for Reward:** Explore regions that *might* yield high returns (as in H-UCRL).
2.  **Pessimism for Safety:** Avoid regions that *might* be unsafe.

We use the model's uncertainty to construct a "worst-case" cost estimate. We assume the "bad luck" outcome for safety constraints.
