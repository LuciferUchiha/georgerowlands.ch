---
title: Tabular Reinforcement Learning
type: docs
weight: 5
---

So far we have assumed that the MDP model (transition and reward functions) is known, allowing us to use planning methods like policy iteration and value iteration to find an optimal policy. However, in many real-world scenarios, the environment is unknown, and the agent must learn to make decisions through interaction with the environment. This is where **reinforcement learning (RL)** comes into play. Specifically in this section, we will focus on **tabular reinforcement learning** methods, which are suitable for small-scale problems with finite state and action spaces where we can represent the value functions and policies using tables, hence the name "tabular".

## Model-Based Methods

For the model-based approach, we first need to learn a model of the environment's dynamics from the experience data. This involves estimating the transition probabilities $\mathbb{P}(s' \mid s, a)$ and the reward function $r(s, a)$ based on the collected experience tuples $(s, a, r, s')$. Once we have learned the model, we can then use **[planning methods](/garden/ml/rl/planning)** like value iteration or policy iteration to compute an optimal policy based on the learned model. 

The most obvious approach is to use maximum likelihood estimation (MLE) to estimate the transition probabilities and rewards. Specifically, we can estimate the transition probabilities as the relative frequencies of transitions observed in the experience data:

$$
\hat{\mathbb{P}}(s' \mid s, a) = \frac{N(s, a, s')}{N(s, a)},
$$

where $N(s, a, s')$ is the number of times we observed a transition from state $s$ to state $s'$ after taking action $a$, and $N(s, a)$ is the total number of times action $a$ was taken in state $s$. So simply put we use an unbiased sample mean estimator of observed transitions, where we count how often we saw a transition from s to s' when taking action a.

For the reward function, we can estimate the expected reward for each state-action pair as the average of the observed rewards:

$$
\hat{r}(s, a) = \frac{1}{N(s, a)} \sum_{i=1}^{N(s, a)} r_i,
$$

where $r_i$ are the rewards observed when taking action $a$ in state $s$. 

Importantly, for these estimates to be valid, we need to ensure that every state-action pair $(s, a)$ is visited sufficiently often in the experience data. This is crucial for the estimates to converge to the true transition probabilities and rewards as the amount of data increases. If some state-action pairs are rarely or never visited, our estimates for those pairs will be poor or undefined, leading to suboptimal policies.

### Epsilon-Greedy Exploration

Because the dynamics $\mathbb{P}$ and rewards $r$ depend only on the environment and not on the agent's policy, therefore any trajectory, regardless of the policy used to generate it, can be used to improve the model. We can use data collected by a random policy, an old policy, or a human expert to update our estimates of the transition probabilities and rewards. Making this an off-policy method.

However, because we are using a policy to gather experience data, we need to address the exploration vs exploitation trade-off here as well. If we only exploit our current knowledge and follow the best-known policy, we may miss out on exploring other actions that could lead to better long-term rewards. Therefore, we need to balance exploration (trying out different actions to gather more information about the environment) and exploitation (using our current knowledge to maximize rewards). This is especially important in the beggining when our model estimates are still poor and uncertain. Here we are especially prone to overfitting our model to early data, leading to suboptimal policies if we do not explore enough. If the agent visits a state once and has a "bad" experience due to stochasticity, the learned model might incorrectly assume that state is always bad, preventing the agent from ever visiting it again to correct the model. This can be thought of just like burning yourself on a hot stove once as a kid and then never going near stoves again.

The simplest exploration strategy is **epsilon-greedy exploration**. In this approach, we do the following:
1. With probability $\epsilon$, we select a random action (exploration).
2. With probability $1 - \epsilon$, we select the action that maximizes the expected return according to our current value function or policy $a^* = \arg\max_a Q(s,a)$ (exploitation).

While effective, this method explores the state space in an "uninformed" manner. It ignores past experience during exploration steps and treats all non-greedy actions as equally useful, which can be inefficient in large state spaces.

The question also arises of whether we explore enough using this method to see all states and actions sufficiently often to learn the optimal policy. This is what leads to idea of **Greedy in the Limit with Infinite Exploration (GLIE)** which defines two conditions for ensuring convergence to the optimal policy:
1. **Infinite Exploration**: Every state-action pair $(s, a)$ must be visited infinitely often as the number of time steps $t \to \infty$. This ensures that the agent gathers enough information about the environment to learn accurate value estimates for all state-action pairs.
2. **Greedy in the Limit**: The policy must converge to a greedy policy with respect to the learned value function as $t \to \infty$. This means that the probability of selecting the optimal action in each state approaches 1 over time.

The simple epsilon-greedy strategy never stops exploring, so it does not converge to a greedy policy. However, we can modify it to satisfy the GLIE conditions by annealing/decaying epsilon over time. For example, we can set $\epsilon_t = \frac{1}{t}$, which ensures that as time progresses, the agent explores less and less, eventually converging to a greedy policy while still ensuring that all state-action pairs are visited infinitely often.

Specifically, if the sequence of epsilon values $\{\epsilon_t\}$ satisfies the Robbins-Monro conditions:
1. $\sum_{t=1}^{\infty} \epsilon_t = \infty$ (ensures infinite exploration)
2. $\sum_{t=1}^{\infty} \epsilon_t^2 < \infty$ (ensures convergence to a greedy policy)
   
then the epsilon-greedy policy with decaying epsilon will satisfy the GLIE conditions such as $\epsilon_t = \frac{1}{t}$ or $\epsilon_t = \frac{1}{\sqrt{t}}$. Remember that a pre-condition for a series to converge is that it is diminishing, i.e., $\epsilon_t \to 0$ as $t \to \infty$.

### Softmax Exploration

Instead of selecting a random action uniformly during exploration steps, we can use a **softmax exploration** strategy also called Boltzmann exploration or just simply soft for short. Here we select actions according to a probability distribution that favors actions with higher estimated values but still allows for exploration of lower-valued actions. Specifically, the policy selects action $a$ in state $s$ with probability:

$$
\pi(a \mid s) = \frac{e^{Q(s,a)/\tau}}{\sum_{a'} e^{Q(s,a')/\tau}},
$$

where $\tau$ is a temperature parameter that controls the level of exploration and the denominator is a normalization factor to ensure the probabilities sum to 1. So we can also see this as a softmax over the Q-values:

$$
\pi(a \mid s) \propto e^{Q(s,a)/\tau}.
$$

A high temperature ($\tau \to \infty$) results in nearly uniform exploration as then all actions have similar probabilities. A low temperature ($\tau \to 0$) makes the policy focus on exploitation, specifically when $\tau \to 0$ then we have a greedy policy, as the action with the highest Q-value will dominate the probabilities. 

Just like with epsilon-greedy exploration, we can anneal the temperature parameter over time to ensure sufficient exploration in the beginning and convergence to a greedy policy later on. For example, we can set $\tau_t = \frac{1}{\log(t+1)}$ (to avoid division by zero at t=0) or $\tau_t = \frac{1}{\sqrt{t}}$.

The epsilon-softmax exploration strategy combines the benefits of both epsilon-greedy and softmax exploration. In this approach, with probability $\epsilon$, we select an action according to the softmax distribution, and with probability $1 - \epsilon$, we select the greedy action. This allows for more informed exploration while still ensuring that the agent explores sufficiently and does not get stuck in suboptimal policies.

{{< figure 
    src="/images/ml/rlEpsilonGreedySoft.webp"
    caption="Visualization of epsilon-greedy and softmax exploration strategies."
    alt="Visualization of epsilon-greedy and softmax exploration strategies."
    width="400"
>}}

### Rmax Algorithm

Epsilon-greedy and softmax exploration are simple and effective strategies, but they do not explicitly account for the uncertainty in the value estimates when deciding which actions to explore, instead they just add "noise" to the action selection process like "dithering". More sophisticated exploration strategies consider the uncertainty in the value estimates and prioritize exploring actions that are less certain but have the potential for high rewards. A common method is to **optimism in the face of uncertainty (OFU)**, where we assign optimistic values to actions with high uncertainty, encouraging the agent to explore them. The intuition is rather simple, if we are uncertain about the value of an action, we should assume it has a high value until proven otherwise, motivating exploration and making a "fairy tale state" where everything is great until we find out otherwise. This leads to more efficient exploration and faster convergence to the optimal policy. 

One well-known algorithm that incorporates the OFU principle is the **Rmax algorithm**. We first initialize the value function optimistically, assuming that all state-action pairs yield some maximum possible reward $R_{max}$. As the agent interacts with the environment and gathers experience, it updates its estimates of the transition probabilities and rewards based on the observed data. 

Specifically, we define a threshold $m$ for the number of visits to each state-action pair. If a state-action pair $(s, a)$ has been visited fewer than $m$ times, we treat it as "unknown" and assign it the optimistic value $R_{max}$. Once a state-action pair has been visited at least $m$ times, we use the empirical estimates of the transition probabilities and rewards for that pair to update the value function using standard planning methods like value iteration.

{{< figure 
    src="/images/ml/rlRmax.png"
    caption="Pseudo-code illustration of the Rmax algorithm."
    alt="Pseudo-code illustration of the Rmax algorithm."
    width="400"
>}}

The question then arises of how to choose the threshold $m$. A common approach is to set $m$ based on the desired level of confidence in the estimates. Specifically, we use Hoeffding's inequality to determine how many samples are needed to ensure that the empirical estimates are within a certain error bound of the true values with high probability. By setting $m$ appropriately, we can balance exploration and exploitation effectively, ensuring that the agent explores sufficiently while still converging to the optimal policy. The Hoeffding's inequality tells us that for the absolute approximation error to be at most $\epsilon$ with probability at least $1 - \delta$, we need:

$$
N(s, a) \geq \frac{R_{max}^2}{2 \epsilon^2} \log\left(\frac{2}{\delta}\right).
$$

Because we only recompute the policy when we have new information (i.e., when a state-action pair transitions from "unknown" to "known"), the Rmax algorithm is more computationally efficient than methods that update the policy at every time step. Specifically we recompute the policy at most $O(|S| * |A|)$ times, as there are that many state-action pairs. However, each policy computation can be expensive, especially for large state spaces, as it involves solving the MDP using planning methods like value iteration or policy iteration which are polynomial in the number of states and actions.

## Model-free Approaches

The problem with model-based methods is that learning an accurate model of the environment can be challenging, especially in complex or high-dimensional environments. Additionally, planning methods like value iteration and policy iteration can be computationally expensive, particularly for large state spaces. Especially since to store the tables of transition probabilities and rewards we need space O(|S|^2 * |A|) which can get large quickly.

Instead we now consider **model-free** reinforcement learning methods, which skip the model learning step entirely and directly learn value functions or policies from experience data. Model-free methods are often more scalable and can handle larger state spaces more effectively, as they do not require storing or updating a full model of the environment.

### Monte Carlo Methods

In the model-based setting we estimated the model from experience and then used planning methods to compute value functions and policies. In the model-free setting we skip the model learning step entirely and directly estimate value functions or policies from experience data. 

If we look at the definition of the value function:

$$
\begin{align*}
v_\pi(s) &= r(s, \pi(s)) + \gamma \sum_{s'} \mathbb{P}(s' \mid s, \pi(s)) v_\pi(s') \\
&= \mathbb{E}_{R_0, S_1} \left[ R_0 + \gamma v_\pi(S_1) \mid S_0 = s, A_0 = \pi(s) \right]
\end{align*}
$$

an initial idea might be to estimate it using a single sample trajectory starting from state $s$ and following policy $\pi$ and seeing $S_0 = s, A_0 = \pi(s), R_0 = r, S_1 = s'$. We can then use the observed reward and next state to update our estimate of the value function as follows:

$$
v_\pi(s) \approx r + \gamma v_\pi(s').
$$

However, this approach requires knowledge of $v_\pi(s')$, which we do not have. So instead we can use a monte carlo approach, where we estimate the value function based on the actual returns observed in the environment. Specifically in **Monte Carlo Policy Evaluation**, we use complete episodes (trajectories including state, action, reward sequences) of experience to estimate the value functions instead of calculating the expected return. This results in an unbiased estimate of the value functions, as we are using actual returns observed in the environment rather than expected returns based on a learned model:

$$
v_\pi(s) = \mathbb{E}_\pi [G_t \mid S_t = s] \approx \frac{1}{N(s)} \sum_{i=1}^{N(s)} G_t^{(i)},
$$

Following the law of large numbers, as the number of visits to state $s$ increases, $N(s) \to \infty$, the estimate converges to the true value function $v_\pi(s)$.

Importantly this requires episodic tasks where we can observe complete episodes from start to finish to calculate the returns. Because we need to wait until the end of the episode to compute the return, Monte Carlo methods are not suitable for continuing tasks without natural episode boundaries unless we artificially create them via some thresholds or time limits. 

Another issue with Monte Carlo methods is that they can have high variance, as the rollouts/trajectories can be very long and also vary significantly due to the stochasticity in the environment. This can lead to slow convergence and instability in the value estimates. 

There are a few different variants of Monte Carlo policy evaluation, including **first-visit** and **every-visit** Monte Carlo methods. In first-visit Monte Carlo, we only consider the first time a state is visited in an episode when updating the value estimate for that state. 

In every-visit Monte Carlo, we consider all visits to a state within an episode when updating the value estimate. Both methods have their pros and cons, with first-visit being simpler and less computationally intensive, while every-visit can provide more data for updating the value estimates:

Lastly we can also use incremental updates to estimate the value function, which allows us to update the value estimates online as we gather experience data, rather than waiting until the end of an episode. Specifically, we can update the value function for a state $s$ incrementally as follows:

$$
v_\pi(s) \leftarrow v_\pi(s) + \alpha \left[G_t - v_\pi(s) \right],
$$

This can be derived as follows:

$$
\begin{align*}
v_\pi(s) &= \frac{1}{N(s)} \sum_{i=1}^{N(s)} G_t^{(i)} \\
&= \frac{N(s) - 1}{N(s)} \cdot \frac{1}{N(s) - 1} \sum_{i=1}^{N(s) - 1} G_t^{(i)} + \frac{1}{N(s)} G_t^{(N(s))} \\
&= \left(1 - \frac{1}{N(s)}\right) v_\pi(s) + \frac{1}{N(s)} G_t^{(N(s))} \\
&= v_\pi(s) + \frac{1}{N(s)} \left[G_t^{(N(s))} - v_\pi(s) \right]
\end{align*}
$$

So we can see that the incremental update is simply a weighted average of the previous estimate and the new return, with the weight determined by the number of visits to the state. This allows us to update the value estimates online as we gather experience data, rather than waiting until the end of an episode. It is common to replace the step size $\frac{1}{N(s)}$ with a constant learning rate $\alpha$ to control the influence of new information on the value estimates, which can help reduce variance and improve stability:

$$
v_\pi(s) \leftarrow v_\pi(s) + \alpha \left[G_t - v_\pi(s) \right],
$$

Once we have estimated the value function using Monte Carlo methods, we can use it to improve the policy in a similar manner as in policy iteration. Specifically, we can derive a greedy policy with respect to the estimated value function.

### Temporal-Difference Learning

This is TD(0), TD(lambda) is like a weighting between MC and TD(0). This is also why then in policy gradients what leads use to GAE (Generalized Advantage Estimation) which is a weighting between MC and TD(0) and the bias-variance trade-off.

One of the key issues with Monte Carlo methods is that they require complete episodes to compute the returns(excluding the incremental version), which can be inefficient and impractical in many scenarios. Additionally, Monte Carlo methods can have high variance due to the stochastic nature of the environment and the long trajectories involved. 

This leads us to **temporal-difference (TD) learning**, which combines ideas from both Monte Carlo methods and dynamic programming. TD learning methods update value estimates based on the difference between consecutive value estimates, allowing for online updates after each time step without waiting for the end of an episode. This results in lower variance and more efficient learning. Specifically, TD learning methods use the concept of **bootstrapping**, where we update our value estimates based on other learned estimates rather than relying solely on actual returns. So we update the value function using the observed reward and the estimated value of the next state as follows:

$$
v_\pi(s_t) \leftarrow (1 - \alpha) v_\pi(s_t) + \alpha \left[ r_{t+1} + \gamma v_\pi(s_{t+1}) \right],
$$

where $\alpha$ is the learning rate. This update rule adjusts the value estimate for the current state $s_t$ towards the observed reward plus the discounted estimated value of the next state $s_{t+1}$. The term in brackets is called the **TD target**, and the difference between the TD target and the current value estimate is known as the **TD error**:

$$
\delta_t = r_{t+1} + \gamma v_\pi(s_{t+1}) - v_\pi(s_t).
$$

This is where the name temporal-difference learning comes from, as we are learning from the difference between consecutive time steps. We can also rewrite the update rule in terms of the TD error:

$$
v_\pi(s_t) \leftarrow v_\pi(s_t) + \alpha \left[r_{t+1} + \gamma v_\pi(s_{t+1}) - v_\pi(s_t) \right] = v_\pi(s_t) + \alpha \delta_t.
$$

Using bootstrapping allows TD learning methods to update value estimates more frequently and efficiently, leading to faster convergence compared to Monte Carlo methods. Additionally, TD learning methods can be applied to both episodic and continuing tasks, making them more versatile. 

Intuitevely, the TD Error meassures how surprised we are about the new information we just received. If the TD error is large, it means that our current value estimate was far off from what we observed, indicating that we need to adjust our estimate significantly. Conversely, if the TD error is small, it suggests that our current estimate was accurate, and only a minor adjustment is needed.

By also weighting the updates with a learning rate $\alpha$, we can control the influence of new information on the value estimates, reducing variance and improving stability. However, the act of bootstrapping also introduces some bias into the estimates, as we are relying on our current value estimates rather than actual returns. This bias-variance trade-off is a key consideration in TD learning methods and can be managed through the choice of the learning rate $\alpha$ and other hyperparameters (e.g., in TD(lambda)).

Again, once we have estimated the value function using TD learning, we can use it to improve the policy in a similar manner as in policy iteration. Specifically, we can derive a greedy policy with respect to the estimated value function. 

### SARSA

In Monte Carlo policy evaluation and TD learning, we focused on estimating the state-value function $v_\pi(s)$ for a given policy $\pi$ and then applied some form of policy improvement to derive a better policy. However, in many scenarios, it is more useful to estimate the action-value function $q_\pi(s,a)$, which gives the expected return for taking action $a$ in state $s$ and then following policy $\pi$. This leads us to **SARSA (State-Action-Reward-State-Action)**, which results in not just estimating the action-value function but also learning a policy simultaneously, making it an on-policy control method. SARAS uses the same TD learning principles but focuses on the action-value function instead of the state-value function by updating the action-value estimates based on the observed reward and the estimated value of the next state-action pair:

$$
q_\pi(s_t, a_t) \leftarrow (1 - \alpha) q_\pi(s_t, a_t) + \alpha \left[ r_{t+1} + \gamma q_\pi(s_{t+1}, a_{t+1}) \right],
$$

or in terms of the TD error:

$$
q_\pi(s_t, a_t) \leftarrow q_\pi(s_t, a_t) + \alpha \left[ r_{t+1} + \gamma q_\pi(s_{t+1}, a_{t+1}) - q_\pi(s_t, a_t) \right].
$$

We can derive the SARSA update rule from the definition of the action-value function:

$$
\begin{align*}
q_\pi(s, a) &= r(s, a) + \gamma \sum_{s'} \mathbb{P}(s' \mid s, a) \sum_{a'} \pi(a' \mid s') q_\pi(s', a') \\
&= \mathbb{E}_{R_0, S_1, A_1} \left[ R_0 + \gamma q_\pi(S_1, A_1) \mid S_0 = s, A_0 = a \right]
\end{align*}
$$

If we then have the two sample transitions $(S_0 = s, A_0 = a, R_0 = r, S_1 = s', A_1 = a', R_1 = r', ... )$ observed from the environment while following policy $\pi$, we can use it to update our estimate of the action-value function for state-action pair $(s_t, a_t)$:

$$
q_\pi(s_t, a_t) \approx r + \gamma q_\pi(s', a').
$$

From this we can see where the name SARSA comes from, as the update depends on the current state $s$, action $a$, reward $r$, next state $s'$, and next action $a'$. We can also see that SARSA is an on-policy method, as the update for the next state-action pair $(s', a')$ is based on the action actually taken by the current policy $\pi$.

Just like with TD learning, SARSA uses bootstrapping to update the action-value estimates more frequently and efficiently, leading to faster convergence compared to Monte Carlo methods. Additionally, SARSA can be applied to both episodic and continuing tasks, making it more versatile.

Having access to the action-value function allows us to derive a policy directly by selecting actions that maximize the estimated action-values. Specifically, we can derive a greedy policy with respect to the estimated action-value function:

$$
\pi(s) = \arg\max_a q_\pi(s,a).
$$

Therefore SARSA combines policy evaluation and policy improvement in a single algorithm, allowing us to learn both the action-value function and the policy simultaneously just like in policy iteration.

This version of SARSA is on-policy as we use the policy to select the next action $a_{t+1}$. There is also an off-policy version called **Expected SARSA**, where instead of using the action actually taken by the current policy, we use the expected value over all possible actions in the next state according to the current policy. The update rule for Expected SARSA is as follows:

$$
q_\pi(s_t, a_t) \leftarrow q_\pi(s_t, a_t) + \alpha \left[ r_{t+1} + \gamma \mathbb{E}_{a' \sim \pi(\cdot \mid s_{t+1})} [q_\pi(s_{t+1}, a')] - q_\pi(s_t, a_t) \right] 
$$

which can be expanded to:

$$
q_\pi(s_t, a_t) \leftarrow q_\pi(s_t, a_t) + \alpha \left[ r_{t+1} + \gamma \sum_{a'} \pi(a' \mid s_{t+1}) q_\pi(s_{t+1}, a') - q_\pi(s_t, a_t) \right].
$$

### Q-Learning

The issue with SARSA is that it is an on-policy method(excluding Expected SARSA), meaning that it learns the value of the policy being followed, which can limit exploration and lead to suboptimal policies and is also not sample efficient as we need to follow a specific policy to gather experience data. To address this, one of the most popular model-free reinforcement learning algorithms is **Q-learning**, which is an off-policy TD control algorithm.

Again instead of estimating the state-value function $v_\pi(s)$, Q-learning estimates the action-value function $q_*(s,a)$ for the optimal policy directly. The key idea in Q-learning is to update the action-value estimates based on the observed reward and the maximum estimated value of the next state over all possible actions. The update rule for Q-learning is as follows:

$$
q(s_t, a_t) \leftarrow (1 - \alpha) q(s_t, a_t) + \alpha \left[ r_{t+1} + \gamma \max_{a'} q(s_{t+1}, a') \right],
$$

or in terms of the TD error:

$$
q(s_t, a_t) \leftarrow q(s_t, a_t) + \alpha \left[ r_{t+1} + \gamma \max_{a'} q(s_{t+1}, a') - q(s_t, a_t) \right].
$$

The key difference between Q-learning and SARSA is that Q-learning uses the maximum estimated value of the next state over all possible actions, rather than the value of the action actually taken by the current policy. This makes Q-learning an off-policy method, as it learns the value of the optimal policy regardless of the policy being followed to gather experience data. This allows for more efficient exploration and can lead to better policies.

We can derive the Q-learning update rule from the definition of the optimal action-value function:

$$
q_*(s, a) = \mathbb{E} \left[ R_{t+1} + \gamma \max_{a'} q_*(S_{t+1}, a') \mid S_t = s, A_t = a \right],
$$

If we then have a single sample transition $(s_t, a_t, r_{t+1}, s_{t+1})$ observed from the environment, we can use it to update our estimate of the optimal action-value function for state-action pair $(s_t, a_t)$:

$$
q_*(s_t, a_t) \approx r_{t+1} + \gamma \max_{a'} q_*(s_{t+1}, a').
$$

Similar to SARSA, having access to the action-value function allows us to derive a policy directly by selecting actions that maximize the estimated action-values. Specifically, we can derive the optimal policy with respect to the estimated action-value function:

$$
\pi_*(s) = \arg\max_a q_*(s,a).
$$

Q-learning combines policy evaluation and policy improvement in a single algorithm, allowing us to learn both the optimal action-value function and the optimal policy simultaneously just like in value iteration.

The space complexity of Q-learning is O(|S|*|A|) as we need to store the Q-values for each state-action pair. The time complexity per update is O(m) since each update involves finding the maximum. However, the overall time to converge to the optimal policy depends on factors such as the exploration strategy, learning rate, and the specific MDP being solved. Generally, Q-learning can take a significant amount of time to converge, especially in large state-action spaces or with sparse rewards. However, in general if we use $T$ time steps and $m$ actions, then it will take $O(Tm)$ time.

In all of the above methods we have discussed it is important to ensure sufficient exploration to learn accurate value estimates and optimal policies. Strategies like epsilon-greedy and softmax exploration can be used to balance exploration and exploitation effectively. For Q-learning specifically there is also an exploration strategy called **Optimistic Q-learning**, where similar to the Rmax algorithm, we initialize the Q-values optimistically, assuming that all state-action pairs yield a non-negative reward:

$$
q_0(s, a) = Q_{max} \geq \frac{R_{max}}{1 - \gamma} \geq \max_{s,a} q_*(s,a).
$$

Since the initial values are impossibly high, every time the agent visits a state-action pair, the observed reward $r + \gamma \max_{a'} q(s', a')$ will likely be lower than the current estimate. Therefore the Q-value will decrease and the greedy policy will naturally favor unvisited or seldom visited actions because their values are still high (optimistic).
