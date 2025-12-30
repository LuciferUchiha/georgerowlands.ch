---
title: Reinforcement Learning
type: docs
weight: 24
---

When learning from data, we are often used to the supervised learning setting where the world appears calm and well behaved: someone has already collected a dataset of input–output pairs, and our only task is to learn a mapping that imitates these known answers. But this picture of learning is deeply limited as we only have a snapshot of a static world. On the other hand, the idea of reinforcement learning is to **learn through interaction rather than instruction**. Instead of being told what the correct answer is, an agent has to discover good behaviour by "fucking around and finding out", i.e performing actions and observing the consequences, and gradually building an understanding of how its actions shape the future. In this sense, reinforcement learning is closer to the way animals and humans learn by acting in the world and learning from experience.

{{< figure 
    src="/images/ml/rlCycle.svg" 
    caption="The reinforcement learning cycle."
    alt="The reinforcement learning cycle."
    width="400"
>}}

For example imagine for a moment that you are trying to learn to ride a bicycle. Nobody hands you a table of “situations” paired with “correct actions”. Instead, you climb onto the bike, wobble, fall, climb again, and gradually figure out which movements keep you upright. The feedback the world gives you is not a label like “correct steering angle is 17 degrees”, but something far more primitive and indirect: you remain balanced (good), or you fall (bad). Reinforcement learning formalizes exactly these sorts of learning signals. Instead of labelled answers, the agent (you) receives **rewards**. Rewards do not tell you what the right action was but simply indicate how good or bad the outcome was. Your goal is then to learn to take actions that maximize the total reward it receives over time or down the road (pun intended).

{{< figure 
    src="/images/ml/rlBicycle.png"
    caption="Learning to ride a bike through trial and error."
    alt="Learning to ride a bike through trial and error."
>}}

Another key difference from supervised learning is that in reinforcement learning, **the data the agent sees is no longer independent and identically distributed**, because each action the agent takes changes the world in which future data will be generated. If you steer left on a bicycle now, the physical situation you will be in a second later depends on that choice. If you make a poor move in a game of chess, the next positions you see will reflect the consequences of that mistake. A move in Chess might seem normal now but lead to a checkmate twenty turns later. This is known as the **credit assignment problem**, determining how much each earlier action contributed to a reward that arrives much later. As the agent improves and starts behaving differently, it begins exploring different parts of the environment, which means it will see different rewards, different transitions, and different future possibilities and tries to exploit what it has learned so far to maximize its cumulative reward. This tight coupling between behaviour and data is exactly what makes reinforcement learning powerful as by **experiencing/sampling trajectories of interaction**, the agent gradually uncovers the structure of the task. But it also introduces new challenges. Because actions influence what the agent sees later, it must constantly balance two competing objectives: Trying things it already believes are good, and experimenting with actions that might lead to better strategies it has not yet discovered. This is the so called **exploration–exploitation dilemma**, which lies at the heart of reinforcement learning.

Although these ideas may seem abstract at first and were initially also heavily doubted by the research community, they underpin some of the most impressive achievements of modern AI. Systems like **AlphaGo** and **AlphaZero** by Google Deepmind were not trained on datasets of "correct" moves. Instead, they learned by interacting with the environment by playing games, receiving sparse signals such as win or lose, and gradually learning strategies. These models began with no human knowledge, no opening book, no endgame heuristics, only the rules of each game. By repeatedly playing against other oponent or against itself and learning from the outcomes, it discovered tactics, structures, and strategic principles that took human experts centuries to develop and even surpassed the best human players. 

{{< figure 
    src="/images/ml/alphaGo.jpg"
    caption="AlphaGo, developed by DeepMind, made history in 2016 by defeating the world champion Go player Lee Sedol."
    alt="AlphaGo, developed by DeepMind, made history in 2016 by defeating the world champion Go player Lee Sedol."
    width="600"
>}}

## Markov Decision Processes

To turn our intuitive picture of reinforcement learning into a precise mathematical framework, we need a way to describe an agent interacting with an environment over time. For this purpose, we extend the [Markov Chain framework](). If a Markov Chain is a stochastic process over states, a Markov Decision Process (MDP) is a stochastic process over states where the transitions are controlled by actions and motivated by rewards. Many ordinary systems fit naturally into this form: a robot navigating a room, an algorithm playing a game, an elevator responding to button presses, or even a user browsing through a website. 

We formally define an MDP as a tuple:

$$
(\mathcal{S}, \mathcal{A}, P, R, p_0, \gamma)
$$

where:
- $\mathcal{S}$ is the set of possible states the environment can be in, also called the **state space**. States can be discrete (e.g., positions on a grid) or continuous (e.g., the position and velocity of a robot). We often also split the state space into **terminal states** $\mathcal{S}_T$ (e.g., game over) and **non-terminal states** $\mathcal{S}_{NT}$ (e.g., ongoing game) and define $\mathcal{S} = \mathcal{S}_T \cup \mathcal{S}_{NT}$.
- $\mathcal{A}(s)$ is the set of actions the agent can take in state $s$, the union over all the states gives the full **action space** $\mathcal{A} = \bigcup_{s \in \mathcal{S}} \mathcal{A}(s)$. Actions can also be discrete (e.g., move left, right, up, down) or continuous (e.g., steering angle, acceleration).
- $\mathbb{P}(s'|s,a)$ is the **transition probability function**, giving the probability of moving to state $s'$ when action $a$ is taken in state $s$.
- $r(s,a, s')$ is the **reward function**, specifying the expected reward received after taking action $a$ in state $s$.
- $p_0(s)$ is the **initial state distribution**, defining the probability of starting in each state at time step $t=0$.
- $\gamma \in [0,1]$ is the **discount factor**, which determines the importance of future rewards.

{{< figure 
    src="/images/ml/rlMDP.png"
    caption="Illustration of an MDP with states, actions, transitions, and rewards."
    alt="Illustration of an MDP with states, actions, transitions, and rewards."
>}}

The process of the MDP can then be seen as a stochastic process over **discrete time steps** $t = 0, 1, 2, ...$, where we start at time step $t=0$ by sampling an initial state $s_0$ from the initial state distribution $p_0(s)$. Then, at each time step the environment is in some state $s_t \in \mathcal{S}$. **The agent observes this state and selects an action** $a_t \in \mathcal{A}$. The set of actions available may depend on the current state, i.e., $\mathcal{A}(s_t)$, for example think of a chess game where a piece can not leave the board. After taking action $a_t$, **the environment transitions to a new state** $s_{t+1}$ according to the transition probabilities $P(s_{t+1}|s_t, a_t)$ and **the agent receives a reward**, $r_t = r(s_t, a_t, s_{t+1})$ (the reward function can also just depend on the state and action, i.e., $r(s_t, a_t)$). This process then repeats at the next time step. Notice that just like Markov Chains, MDPs also satisfy the **Markov property**: the future state $s_{t+1}$ depends only on the current state $s_t$ and action $a_t$, not on any previous states or actions:

$$
\mathbb{P}(s_{t+1}|s_t, a_t, s_{t-1}, a_{t-1}, ..., s_0, a_0) = \mathbb{P}(s_{t+1}|s_t, a_t)
$$

Importantly, these transitions are proper probabilities, meaning that they are all non-negative and sum to one:

$$
\sum_{s' \in \mathcal{S}} P(s'|s,a) = 1 \quad \forall s \in \mathcal{S}, a \in \mathcal{A}
$$

Another important note is that just because a certain action is available in a state, it does not mean that taking that action will always lead to the same next state. The transition function $P(s'|s,a)$ is generally stochastic, meaning that taking the same action in the same state can lead to different next states with certain probabilities. This stochasticity captures the inherent uncertainty allows us to model a wide range of real-world scenarios where outcomes are not deterministic. Think of the game of Blackjack, where the same action (e.g., "hit") can lead to different next states depending on the random draw of the next card.

The choices the agent makes at each time step are determined by its **policy** $\pi(a|s)$, which specifies the probability of taking action $a$ when in state $s$. A policy can be **deterministic**, mapping each state to a specific action, or **stochastic**, assigning probabilities to each possible action in a state:

$$
\pi(a|s) = \mathbb{P}(a_t = a | s_t = s) \text{ stochastic policy } \quad \text{or} \quad a = \pi(s) \text{ deterministic policy }
$$

When the agent follows a policy $\pi$ in an MDP, it generates a sequence of states, actions, and rewards over time. These chains of states, actions, and rewards generated by following this process are called **trajectories** or **episodes**. A trajectory starting at time step $t=0$ and ending at time step $T$ can be represented as a tuple:

$$
\tau = (s_0, a_0, R_1, s_1, a_1, R_2, s_2, a_2, R_3, ..., s_{T-1}, a_{T-1}, R_T, s_T)
$$

The goal of the agent is to maximize the total amount of reward it receives. To understand this better, we need to define how rewards are structured over time. Just like the transitions, the reward can be stochastic, meaning that the same state-action-next state transition can yield different rewards at different times. For example, in a stock trading environment, taking the action "buy" in a certain market state may lead to different profits or losses depending on market fluctuations. We denote the **random variable for the reward** received at time step $t+1$ after taking action $a_t$ in state $s_t$ and transitioning to state $s_{t+1}$ as:

$$
R_{t+1} \sim r(s_t, a_t, s_{t+1})
$$

a **realization** of this random variable is denoted as a lowercase letter $r \in \mathbb{R}$, i.e., the actual scalar reward received at that time step. The actual **expected reward function** $r(s, a, s')$ gives the expected value of this random variable. Or in other words, the average reward we would expect to receive if we were to take action $a$ in state $s$ and transition to state $s'$ multiple times:

$$
r(s, a, s') = \mathbb{E}[R_{t+1} | s_t = s, a_t = a, s_{t+1} = s']
$$

We also often want to know the expected reward for taking action $a$ in state $s$ regardless of the next state. We can compute this by marginalizing over all possible next states which results in the **expected immediate reward function**:

$$
r(s, a) = \mathbb{E}[R_{t+1} | s_t = s, a_t = a] = \sum_{s' \in \mathcal{S}}\mathbb{P}(s'|s,a) \cdot r(s, a, s')
$$

However, what does this mean in practice? Depending on the task, the task may be **episodic**, meaning that it naturally breaks down into separate episodes with a clear starting and ending point (e.g., a game of chess). We can also force a task to be episodic by defining a maximum time horizon or by introducing terminal states. Alternatively, the task may be **continuing**, where the agent interacts with the environment indefinitely without a natural endpoint (e.g., controlling a thermostat). In either case, we define the **return** $G_0$ as a random variable corresponding to the total accumulated reward from time step $t = 0$ onward. In episodic tasks, this is simply the sum of rewards until the end of the episode:

$$
G_0 = \sum_{t=0}^{T-1} R_{t+1} \quad \text{(episodic tasks)}
\qquad \text{or} \qquad
G_0 = \sum_{t=0}^{\infty} R_{t+1} \quad \text{(continuing tasks)}.
$$

However, in continuing tasks, summing all future rewards may lead to an infinite return if the rewards do not diminish over time. To address this, we introduce a **discount factor** (\gamma \in [0,1)) that reduces the importance of future rewards exponentially over time and therefore focuses the agent more on immediate rewards. The discounted return is defined as:

$$
G_0 = \sum_{t=0}^{\infty} \gamma^{t} R_{t+1}
= R_{1} + \gamma R_{2} + \gamma^2 R_{3} + \cdots
\quad \text{(discounted return for continuing tasks)}.
$$

The discount factor (\gamma) controls the trade-off between immediate and future rewards. A value of (\gamma = 0) makes the agent only care about immediate rewards, while a value close to (1) encourages the agent to consider long-term consequences of its actions. In episodic tasks, we can also apply discounting to prioritize earlier rewards within the episode:

$$
G_0 = \sum_{t=0}^{T-1} \gamma^t R_{t+1}
= R_1 + \gamma R_2 + \gamma^2 R_3 + \cdots
\quad \text{(discounted return for episodic tasks)}.
$$

To correctly assign credit to the actions that caused later rewards, we define the **return from time step (t)**, which only includes rewards received after taking action (a_t). This is known as the **reward-to-go**:

$$
\begin{align*}
G_0 &= R_{1} + \gamma R_2 + \gamma^2 R_3 + \cdots, \
G_1 &= R_{2} + \gamma R_3 + \gamma^2 R_4 + \cdots, \
G_2 &= R_{3} + \gamma R_4 + \gamma^2 R_5 + \cdots.
\end{align*}
$$

More generally, the reward-to-go from time step (t) is:

$$
G_t = \sum_{k=0}^{T-t-1} \gamma^{k} R_{t+1+k}
= R_{t+1} + \gamma R_{t+2} + \gamma^{2} R_{t+3} + \cdots.
$$

We can also express the reward-to-go recursively:

$$
G_t = R_{t+1} + \gamma G_{t+1}.
$$

This simple recursive structure is extremely important for reinforcement learning, as it underlies the Bellman equations and forms the basis for most algorithms. The choice of discount factor (\gamma) can significantly impact the agent's behavior and is often treated as a hyperparameter to be tuned based on the specific task and desired trade-off between immediate and future rewards. The same goes for the reward function itself, which needs to be designed carefully to encourage the desired behavior from the agent. This process of designing the reward function is known as **reward shaping** and can be quite challenging, as poorly designed rewards can lead to unintended behaviors. For example, if we don't give the agent a penalty for not doing anything, it might learn to just stay still to avoid negative rewards, in a way becoming "scared" to act and explore the environment.

{{< callout type="example" >}}
A typical example of an MDP is a grid world where an agent can move in four directions (up, down, left, right) on a grid. For simplicity, we assume that the agent can move in any direction unless it hits a wall. The states $\mathcal{S}$ are the grid cells, the actions $\mathcal{A}$ are the four possible movements, and the transition probabilities $P(s'|s,a)$ are deterministic (i.e., moving in a direction always leads to the adjacent cell unless blocked). 

However, there is an exception. We mark 2 special cells $A$ and $B$. If the agent moves into cell $A$, it is teleported to cell $A'$ and receives a reward of $+10$. Similarly, moving into cell $B$ teleports the agent to cell $B'$ with a reward of $+5$. All other movements yield a reward of -1 to encourage the agent to explore. If this were not the case, the agent might learn to just stay still to avoid negative rewards. The initial state distribution $p_0(s)$ is uniform over states.

{{< figure 
    src="/images/ml/rlGridWorld.webp"
    caption="A simple grid world MDP with teleportation states A and B."
    alt="A simple grid world MDP with teleportation states A and B."
>}}
{{< /callout >}}

### Partial Observability

In the definition of MDPs above, we assumed that the agent has full knowledge of the current state $s_t$ at each time step. However, in many real-world scenarios, the agent may not have access to the complete state information. Instead, it receives **observations** that provide partial information about the underlying state. This leads us to the concept of **Partially Observable Markov Decision Processes (POMDPs)**. For example in the Grid World example above, the agent could see the entire grid and know exactly where it is. But what if the agent could only see a 3x3 window around its current position? Or another common example is the game of Poker or Blackjack, where the agent does not know the opponent's cards or the deck composition. In such cases, the agent must make decisions based on incomplete information.

In a POMDP, we extend the MDP framework by introducing an **observation space** $\mathcal{O}$ and an **observation function** $O(o|s)$, which defines the probability of receiving observation $o_t$ when the environment is in state $s_t$. The agent's policy then depends on the history of observations and actions rather than the full state. 

$$
o_t \sim O(o|s_t)
$$

{{< figure 
    src="/images/ml/rlPOMDP.png"
    caption="Illustration of an POMDP with observations, states, actions, transitions, and rewards."
    alt="Illustration of an POMDP with observations, states, actions, transitions, and rewards."
>}}

The biggest challenge in POMDPs is that the observations may no longer satisfy the Markov property, as the current observation may not contain all the necessary information to predict future states. To address this, agents often maintain a **belief state**, which is a probability distribution over possible states given the history of observations and actions. The agent can then use this belief state to make decisions, effectively transforming the POMDP into a fully observable MDP over belief states. However, solving POMDPs is generally more complex than solving MDPs due to the added uncertainty and the need to reason about hidden states.

### Value Functions

To make good decisions, the agent needs a way to evaluate how good different states and actions are in terms of expected future rewards. This is where **value functions** come into play. Value functions provide a way to quantify the long-term desirability of states and actions under a given policy. They essentially answer the question: "If I am in this state (or take this action), how much reward can I expect to accumulate in the future?"

We now no longer just care about immediate rewards or rewards of particular trajectories, but about the **expected return** when following a certain policy. For example eating a cookie now might give you a small immediate reward, but if it leads to health problems later, the long-term expected return might be negative. Conversely, exercising now might be effortful (negative immediate reward), but it could lead to better health and more rewards in the future. 

First we define the so-called **state-value function** or also just **value function**, which tells us how good it is to be in a certain state when following a particular policy $\pi$, or more specifically, the expected return starting from that state and following the policy thereafter:

$$
v_\pi(s) = \mathbb{E}_\pi [ G_t \mid S_t = s ] = \mathbb{E}_\pi \left[ \sum_{k=0}^{\infty} \gamma^k R_{t+1+k} \mid S_t = s \right].
$$

The value function only depends on the policy $\pi$ and the state $s$, not on the specific time step $t$, because of the Markov property and by taking the expectation over all possible future trajectories we take out the randomness. The precise expectation with the sums written out only depends on the discount factor $\gamma$:

$$
v_\pi(s) = \mathbb{E}_\pi [ G_t \mid S_t = s ] = \sum_{a \in \mathcal{A}} \pi(a \mid s) \sum_{s' \in \mathcal{S}, r \in \mathbb{R}} \mathbb{P}(s', r \mid s, a) \left[ r + \gamma \, v_\pi(s') \right]
$$

Or we could write it in terms of the expected immediate reward function (same counts for all other equations below that use the sum over the realizations of the reward):

$$
v_\pi(s) = \sum_{a \in \mathcal{A}} \pi(a \mid s) \left[ r(s,a) + \gamma \sum_{s' \in \mathcal{S}} \mathbb{P}(s'|s,a) v_\pi(s') \right].
$$

Intuitively, $v_\pi(s)$ tells us the long-term expected return if we start in state $s$ and then follow policy $\pi$. If $v_\pi(s)$ is high, it means that being in state $s$ is desirable under policy $\pi$, as it leads to high expected future rewards. Conversely, if $v_\pi(s)$ is low or negative, it indicates that being in state $s$ is not beneficial when following policy $\pi$. Importantly, the value function depends on the policy $\pi$, for example if one policy is a Grandmaster at chess and another is a beginner, the value of being in a certain board position will differ significantly between the two policies because you would expect the Grandmaster to be able to convert that position into a win much more often than the beginner which might blunder away the advantage.

However, the state-value function only tells us how good it is to be in a certain state, it doesn't directly tell us what to do in that state. This leads us to the **action-value function** also called **Q-Function** which tells us the expected return of if we start in state $s$, take action $a$ (which might be subpotimal or not according to the policy), and then follow policy $\pi$ thereafter:

$$
q_\pi(s,a) = \mathbb{E}_\pi [ G_t \mid S_t = s, A_t = a ] = \mathbb{E}_\pi \left[ \sum_{k=0}^{\infty} \gamma^k R_{t+1+k} \mid S_t = s, A_t = a \right].
$$

The distinction is simple but powerful:
- $v_\pi(s)$ tells us the expected return of being in state $s$ and behaving "normally", i.e. following policy $\pi$.
- $q_\pi(s,a)$ tells us the expected return of being in state $s$, taking action $a$ (which might be good or bad), and then behaving "normally" again.

These two functions are crucial. As remember that the agent can't see into the future. The reward signal is delayed and noisy, a decision now might have consequences far later. Value functions compress the entire uncertain future into a **single numeric prediction** and therefore allow us to evaluate how good different states and actions are in terms of expected future rewards, which in turn enables us to make informed decisions about which actions to take in order to maximize the long-term reward. In this sense, value functions provide the agent with a **map of the future**, telling it which states and which decisions are promising.

In fact, the two functions are also closely related. We can easily notice above that the state-value function can be expressed in terms of the action-value function as:

$$
v_\pi(s) = \sum_{a \in \mathcal{A}} \pi(a \mid s) \, q_\pi(s,a).
$$

Intuitively, this is rather straightforward: the value of being in state $s$ is simply the expected value of taking each possible action $a$ in that state, weighted by how likely the policy $\pi$ is to choose that action. 

This relationship is extremely important, as in practice it is often easier to estimate action-values directly, and then derive state-values from them. 

An important note is that if the policy $\pi$ is deterministic, meaning it always chooses the same action $a = \pi(s)$ in state $s$. Because of this marginalizing over the actions disappears and we simply have:

$$
v_\pi(s) = q_\pi(s, \pi(s)).
$$

Similarly, we can also express the action-value function in terms of the state-value function as:

$$
q_\pi(s,a) = \sum_{s' \in \mathcal{S}, r \in \mathbb{R}} \mathbb{P}(s', r \mid s, a) \left[ r + \gamma \, v_\pi(s') \right] = r(s,a) + \gamma \sum_{s' \in \mathcal{S}} \mathbb{P}(s'|s,a) v_\pi(s').
$$

Here, we also clearly see that the action-value function decomposes into the immediate expected reward for taking action $a$ in state $s$, plus the discounted expected value of the next state $s'$ we transition to after taking that action.

{{< callout type="proof" >}}
To see why the action-value function can be expressed in terms of the state-value function, we start from the definition of the action-value function and then rollout the expectation and the linearity of expectation:

$$
\begin{align*}
q_\pi(s,a) &= \mathbb{E}_\pi [ G_t \mid S_t = s, A_t = a ] \\
&= \mathbb{E}_\pi [ R_{t+1} + \gamma G_{t+1} \mid S_t = s, A_t = a ] \\
&= \mathbb{E}_\pi [ R_{t+1} \mid S_t = s, A_t = a ] + \gamma \, \mathbb{E}_\pi [ G_{t+1} \mid S_t = s, A_t = a ] \\
&= \sum_{s' \in \mathcal{S}, r \in \mathbb{R}} \mathbb{P}(s', r \mid s, a) \cdot \mathbb{E}_\pi [ R_{t+1} + \gamma G_{t+1} \mid S_t = s, A_t = a, S_{t+1} = s' ] \\
&= \sum_{s' \in \mathcal{S}, r \in \mathbb{R}} \mathbb{P}(s', r \mid s, a) \left[ r + \gamma \, \mathbb{E}_\pi [ G_{t+1} \mid S_{t+1} = s' ] \right] \\
&= \sum_{s' \in \mathcal{S}, r \in \mathbb{R}} \mathbb{P}(s', r \mid s, a) \left[ r + \gamma \, v_\pi(s') \right].
\end{align*}
$$

{{< /callout >}}

{{< callout type="example" >}}
If we go back to our grid world example, we can see how the value functions help us evaluate states and actions. 

If the agent starts in some state $s$, then:
- $v_\pi(s)$ will be large if the policy tends to reach high-reward teleportation states like $A$ or $B$ quickly.
- If $v_\pi(s)$ is small or negative, it indicates that the policy often leads to long paths with many $-1$ penalties before reaching a teleportation state.

If we look at action-values, $q_\pi(s,a)$ distinguishes between the possible actions in state $s$. For example, if “move down” leads directly into the teleportation cell $A$, then:

- $q_\pi(s,\text{down})$ will be large.
- while $q_\pi(s,\text{left})$ will be smaller because it leads to a longer sequence of penalties.

{{< figure 
    src="/images/ml/rlGridWorldValue.webp"
    caption="The state-value function of a policy in the grid world example."
    alt="The state-value function of a policy in the grid world example."
>}}

From this value function, it is easy to see which states are desirable (those near teleportation states) and which actions are preferable in each state (those leading towards teleportation states). This information can then be used to also induce a policy which always chooses the best action which leads it to a higher value state.
{{< /callout >}}

### Bellman Expectation Equations

The definition of the value functions above is useful for understanding what they represent, but it does not provide a practical way to compute them as they define value as an expectation over all possible future trajectories, which can be infinite in length. This would mean we would theoretically have to simulate the agent in all states and actions for an infinite amount of time to compute the value functions exactly, which would be intractable in all but the simplest cases.

Instead, we can exploit the recursive structure of the return $G_t$ to derive a set of equations known as the **Bellman expectation equations**. These equations express the value functions in terms of themselves, allowing us to compute them iteratively. Specifically, we define the **Bellman expectation equation for the state-value function** as follows:

$$
v_\pi(s) = \sum_{a \in \mathcal{A}} \pi(a \mid s) \sum_{s' \in \mathcal{S}, r \in \mathbb{R}} \mathbb{P}(s', r \mid s, a) \left[ r + \gamma v_\pi(s') \right].
$$

It states that the value of a state under policy $\pi$ is equal to the expected immediate reward plus the expected discounted value of the next state, averaged over all possible actions and next states according to the policy and transition probabilities.

We can also express this more compactly by combining the sums back into a single expectation over the next reward and state:

$$
\begin{align*}
v_\pi(s) &= \sum_{a \in \mathcal{A}} \pi(a \mid s) \sum_{s' \in \mathcal{S}, r \in \mathbb{R}} \mathbb{P}(s', r \mid s, a) \left[ r + \gamma v_\pi(s') \right] \\
&= \mathbb{E}_{a_t \sim \pi(\cdot|s), s_{t+1}, R_{t+1} \sim \mathbb{P}(\cdot|s,a_t)} [ R_{t+1} + \gamma v_\pi(S_{t+1}) \mid S_t = s ] \\
&= \mathbb{E}_\pi [ R_{t+1} + \gamma v_\pi(S_{t+1}) \mid S_t = s ].
\end{align*}
$$

using the definition of the state-value function in terms of the action-value function $v_\pi(s) = \sum_{a \in \mathcal{A}} \pi(a \mid s) \, q_\pi(s,a)$, we can also write the Bellman equation as:

$$
\begin{align*}
v_\pi(s) &= \sum_{a \in \mathcal{A}} \pi(a \mid s) \sum_{s' \in \mathcal{S}, r \in \mathbb{R}} \mathbb{P}(s', r \mid s, a) \left[ r + \gamma v_\pi(s') \right] \\
&= \sum_{a \in \mathcal{A}} \pi(a \mid s) \, q_\pi(s,a).
\end{align*}
$$

{{< callout type="proof" >}}
Remember that the return can be expressed recursively as:

$$
G_t = R_{t+1} + \gamma G_{t+1}.
$$

This can also be observed as follows:

$$
\begin{align*}
v_\pi(s) &= \mathbb{E}_\pi [ G_t \mid S_t = s ] \\
&= \mathbb{E}_\pi [ R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \cdots \mid S_t = s ] \\
&= \mathbb{E}_\pi [ R_{t+1} + \gamma (R_{t+2} + \gamma R_{t+3} + \cdots) \mid S_t = s ] \\
&= \mathbb{E}_\pi [ R_{t+1} + \gamma G_{t+1} \mid S_t = s ].
\end{align*}
$$

So by linearity of expectation we could separate the immediate reward and the future return:

$$
v_\pi(s) = \mathbb{E}_\pi [ R_{t+1} \mid S_t = s ] + \gamma \, \mathbb{E}_\pi [ G_{t+1} \mid S_t = s ].
$$

We can further expand these expectations by "unrolling" the randomness in the environment's dynamics and the agent's policy. First, we use the policy to condition on the action taken at time step $t$:

$$
\begin{align*}
v_\pi(s) &= \mathbb{E}_\pi [ R_{t+1} + \gamma G_{t+1} \mid S_t = s ] \\
&= \sum_{a \in \mathcal{A}} \pi(a \mid s) \, \mathbb{E}_\pi [ R_{t+1} + \gamma G_{t+1} \mid S_t = s, A_t = a ].
\end{align*}
$$

Now we are given the action $a$ taken in state $s$, so we can use these and to get the next state $s'$ and reward $r$ according to the environment's transition probabilities, $\mathbb{P}(S_{t+1} = s', R_{t+1} = r \mid S_t = s, A_t = a)$ and apply the law of total expectation:

$$
\begin{align*}
\mathbb{E} [ R_{t+1} + \gamma G_{t+1} \mid S_t = s, A_t = a ]
&= \sum_{s' \in \mathcal{S}, r \in \mathbb{R}} \mathbb{P}(s', r \mid s, a) \, \mathbb{E} [ R_{t+1} + \gamma G_{t+1} \mid S_t = s, A_t = a, S_{t+1} = s', R_{t+1} = r ] \\
&= \sum_{s' \in \mathcal{S}, r \in \mathbb{R}} \mathbb{P}(s', r \mid s, a) \left[ r + \gamma \, \mathbb{E} [ G_{t+1} \mid S_{t+1} = s' ] \right] 
\end{align*}
$$

Note that above we used the markov property to drop the conditioning on $S_t$ and $A_t$ in the second term. Finally, we can recognize that the expectation of the return from time step $t+1$ given that we are in state $s'$ is just the value function at state $s'$:

$$
\mathbb{E} [ G_{t+1} \mid S_{t+1} = s' ] = v_\pi(s').
$$

Putting everything together, we arrive at the **Bellman expectation equation for the state-value function**:

$$
v_\pi(s) = \sum_{a \in \mathcal{A}} \pi(a \mid s) \sum_{s' \in \mathcal{S}, r \in \mathbb{R}} \mathbb{P}(s', r \mid s, a) \left[ r + \gamma v_\pi(s') \right].
$$
{{< /callout >}}

Importantly we get such a Bellman expectation equation for each state $s \in \mathcal{S}$, resulting in a system of $|\mathcal{S}|$ linear equations with $|\mathcal{S}|$ unknowns (the values $v_\pi(s)$). This system can be solved using standard linear algebra techniques, such as matrix inversion or iterative methods like value iteration. Specifically, if we collect the values $v_\pi(s)$ into a vector $\mathbf{v}_\pi$:

$$
\mathbf{v}_\pi = \begin{bmatrix}
v_\pi(s_1) \\
v_\pi(s_2) \\
\vdots \\
v_\pi(s_{|\mathcal{S}|})
\end{bmatrix},
$$

and define the state transition matrix $\mathbf{P}_\pi$ as an $|\mathcal{S}| \times |\mathcal{S}|$ matrix where each entry $(i,j)$ represents the probability of transitioning from state $s_i$ to state $s_j$ under policy $\pi$:

$$
\mathbf{P}_\pi(i,j) = \sum_{a \in \mathcal{A}} \pi(a \mid s_i) \mathbb{P}(s_j \mid s_i, a) \quad \text{where} \quad \mathbb{P}(s_j \mid s_i, a) = \sum_{r \in \mathbb{R}} \mathbb{P}(s_j, r \mid s_i, a)
$$

and the reward vector $\mathbf{r}_\pi$ as:

$$
\mathbf{r}_\pi = \begin{bmatrix}
r_\pi(s_1) \\
r_\pi(s_2) \\
\vdots \\
r_\pi(s_{|\mathcal{S}|})
\end{bmatrix} \quad \text{where} \quad r_\pi(s) = \sum_{a \in \mathcal{A}} \pi(a \mid s) \sum_{s' \in \mathcal{S}, r \in \mathbb{R}} \mathbb{P}(s', r \mid s, a) \cdot r
$$

we can rewrite the Bellman equation in matrix form as:

$$
\mathbf{v}_\pi = \mathbf{r}_\pi + \gamma \mathbf{P}_\pi \mathbf{v}_\pi.
$$

From the MDP we know all the components on the right-hand side except for $\mathbf{v}_\pi$, so we can rearrange this system so that we can solve for $\mathbf{v}_\pi$:

$$
\begin{align*}
\mathbf{Ax} &= \mathbf{b} \\
(\mathbf{I} - \gamma \mathbf{P}_\pi) \mathbf{v}_\pi &= \mathbf{r}_\pi  \\
\mathbf{v}_\pi &= (\mathbf{I} - \gamma \mathbf{P}_\pi)^{-1} \mathbf{r}_\pi.
\end{align*}
$$

For small problems, we can directly compute the inverse of matrix $\mathbf{A}$ to find the value function but for larger problems, iterative methods like value iteration or policy evaluation are more practical. The inverse $(\mathbf{I} - \gamma \mathbf{P}_\pi)^{-1}$ exists as long as $\gamma < 1$ and the MDP is well-defined (i.e., the transition probabilities are valid summing to one). Don't ask me why though, I am not a mathematician (something with the Bellman operator being a contraction mapping and therefore having a unique fixed point, meaning the linear system has a unique solution and therefore the matrix is invertible).

If we now go back to the action-value function, we can also define a **Bellman expectation equation for the action-value function**:

$$
q_\pi(s,a) = \sum_{s' \in \mathcal{S}, r \in \mathbb{R}} \mathbb{P}(s', r \mid s, a) \left[ r + \gamma \sum_{a' \in \mathcal{A}} \pi(a' \mid s') \, q_\pi(s',a') \right].
$$

which we can also express as an expectation:

$$
q_\pi(s,a) = \mathbb{E}_\pi [ R_{t+1} + \gamma v_\pi(S_{t+1}) \mid S_t = s, A_t = a ].
$$

or again if we use the definition of the state-value function $v_\pi(s) = \sum_{a' \in \mathcal{A}} \pi(a' \mid s) \, q_\pi(s,a')$, we can also write the Bellman expectation equation as:

$$
\begin{align*}
q_\pi(s,a) &= \sum_{s' \in \mathcal{S}, r \in \mathbb{R}} \mathbb{P}(s', r \mid s, a) \left[ r + \gamma \sum_{a' \in \mathcal{A}} \pi(a' \mid s') \, q_\pi(s',a') \right] \\
&= \sum_{s' \in \mathcal{S}, r \in \mathbb{R}} \mathbb{P}(s', r \mid s, a) \left[ r + \gamma v_\pi(s') \right].
\end{align*}
$$

{{< callout type="proof" >}}
We can derive the equation for the action-value function $q_\pi(s,a)$ in a similar manner as we did for the state-value function. Starting from the definition of the action-value function, we have:

$$
\begin{align*}
q_\pi(s,a) &= \mathbb{E}_\pi [ G_t \mid S_t = s, A_t = a ] \\
&= \mathbb{E}_\pi [ R_{t+1} + \gamma G_{t+1} \mid S_t = s, A_t = a ].
\end{align*}
$$

and then again we can expand the expectation by conditioning on the next state $s'$ and reward $r$:

$$
\begin{align*}
q_\pi(s,a) &= \sum_{s' \in \mathcal{S}, r \in \mathbb{R}} \mathbb{P}(s', r \mid s, a) \, \mathbb{E}_\pi [ R_{t+1} + \gamma G_{t+1} \mid S_t = s, A_t = a, S_{t+1} = s', R_{t+1} = r ] \\
&= \sum_{s' \in \mathcal{S}, r \in \mathbb{R}} \mathbb{P}(s', r \mid s, a) \left[ r + \gamma \, \mathbb{E}_\pi [ G_{t+1} \mid S_{t+1} = s' ] \right].
\end{align*}
$$

Just like before, we recognize that the expectation of the return from time step $t+1$ given that we are in state $s'$ is just the value function at state $s'$:

$$
q_\pi(s,a) = \sum_{s' \in \mathcal{S}, r \in \mathbb{R}} \mathbb{P}(s', r \mid s, a) \left[ r + \gamma v_\pi(s') \right].
$$

We can now again use the relationship between state-values and action-values to express this entirely in terms of action-values:

$$
\begin{align*}
q_\pi(s,a) &= \sum_{s' \in \mathcal{S}, r \in \mathbb{R}} \mathbb{P}(s', r \mid s, a) \left[ r + \gamma v_\pi(s') \right] \\
&= \sum_{s' \in \mathcal{S}, r \in \mathbb{R}} \mathbb{P}(s', r \mid s, a) \left[ r + \gamma \sum_{a' \in \mathcal{A}} \pi(a' \mid s') \, q_\pi(s',a') \right].
\end{align*}
$$

Giving us the final form of the **Bellman expectation equation for the action-value function**.
{{< /callout >}}

So in summary, the Bellman equations provide a recursive way to compute the value functions by expressing them in terms of themselves. This allows us to use iterative methods to compute the value functions efficiently, which is crucial for solving MDPs and finding optimal policies. We have also shown that the state-value and action-value functions are closely related, with the state-value function being the expectation of the action-value function under the policy and that we can also compute the value functions using linear algebra techniques for small problems.

- **Bellman expectation state-value equation**:

$$
v_\pi(s) = \sum_{a \in \mathcal{A}} \pi(a \mid s) \sum_{s' \in \mathcal{S}, r \in \mathbb{R}} \mathbb{P}(s', r \mid s, a) \left[ r + \gamma v_\pi(s') \right].
$$

- **Bellman expectation action-value equation**:

$$
q_\pi(s,a) = \sum_{s' \in \mathcal{S}, r \in \mathbb{R}} \mathbb{P}(s', r \mid s, a) \left[ r + \gamma \sum_{a' \in \mathcal{A}} \pi(a' \mid s') \, q_\pi(s',a') \right].
$$

### Bellman Optimality Equations

The Bellman expectation equations allow us to compute the value functions iteratively and therefore evaluate how good a specific policy $\pi$ is. However, our ultimate goal is not just to evaluate a fixed policy, but to find the **optimal policy** $\pi^*$ that maximizes the expected return from every state. So in other words, we want to find the **optimal state-value function** $v_*(s)$ which corresponds to the maximum value achievable from state $s$ under any policy:

$$
v_*(s) = \max_\pi v_\pi(s)
$$

Similarly, we define the **optimal action-value function** $q_*(s,a)$ as the maximum expected return achievable from state $s$ by taking action $a$ and then following the optimal policy thereafter:

$$
q_*(s,a) = \max_\pi q_\pi(s,a)
$$

These value functions correspond to the best possible behaviour achievable in the MDP and induced by at least one optimal policy $\pi^*$ and therefore an optimal policy can be derived from them. Intuitively, an optimal policy shouldn't just average over actions like a random policy but instead should select the best action. This intuition is formalized by the **Bellman Optimality Equations**.

To derive the Bellman optimality equation for the state-value function, we consider the value of a state $s$ under the optimal policy. The value of state $s$ is the expected return of taking the best possible action $a$ in state $s$, and then following the optimal policy thereafter. Therefore, instead of summing over actions weighted by the policy probabilities, we take the maximum over all possible actions:

$$
v_*(s) = \max_a \mathbb{E} [ R_{t+1} + \gamma v_*(S_{t+1}) \mid S_t = s, A_t = a ].
$$

if we expand the expectation as before, we get the **Bellman optimality equation for the state-value function**:

$$
v_*(s) = \max_a \sum_{s' \in \mathcal{S}, r \in \mathbb{R}} \mathbb{P}(s', r \mid s, a) \left[ r + \gamma v_*(s') \right].
$$

We now notice that we can also express this in terms of the optimal action-value function by using the defintion of the action-value function in terms of the state-value function:

$$
q_*(s,a) = \sum_{s' \in \mathcal{S}, r \in \mathbb{R}} \mathbb{P}(s', r \mid s, a) \left[ r + \gamma v_*(s') \right].
$$

Substituting this into the Bellman optimality equation for the state-value function gives us:

$$
v_*(s) = \max_a q_*(s,a).
$$

Similarly, we can derive the Bellman optimality equation for the action-value function. The value of taking action $a$ in state $s$ under the optimal policy is the expected return of taking action $a$, and then following the optimal policy thereafter. Therefore, we again take the maximum over all possible actions in the next state:

$$
q_*(s,a) = \mathbb{E} [ R_{t+1} + \gamma \max_{a'} q_*(S_{t+1}, a') \mid S_t = s, A_t = a ].
$$

Expanding the expectation gives us the **Bellman optimality equation for the action-value function**:

$$
q_*(s,a) = \sum_{s' \in \mathcal{S}, r \in \mathbb{R}} \mathbb{P}(s', r \mid s, a) \left[ r + \gamma \max_{a'} q_*(s',a') \right].
$$

which also matches if we had above replaced $v_*(s')$ with $\max_{a'} q_*(s',a')$ directly. So we can see that if we have one of the optimal value functions, we can easily derive the other using these relationships. Specifically if we have the optimal action-value function, we can easily derive the optimal policy by simply acting greedily with respect to it (i.e., choosing the action with the highest action-value in each state). This is also known as the **greedy policy** with respect to the action-value function:

$$
\pi^*(s) = \arg\max_a q_*(s,a) = \arg\max_a [r(s,a) + \gamma \sum_{s' \in \mathcal{S}} \mathbb{P}(s' \mid s, a) v_*(s')].
$$

This link between the policy being optimal and it being greedy with respect to the optimal action-value function is extremely important, as it allows us to derive optimal policies directly from the value functions.

Importantly we now get non-linear equations because of the max operator, so we can no longer solve this using linear algebra techniques. However, instead we notice that these equations satisfy the conditions for **dynamic programming** methods, which allow us to compute the optimal value functions recursively/iteratively. The idea of dynamic programming is to break down a complex problem into simpler subproblems and solve them recursively where importantly solutions to subproblems are stored and reused to avoid redundant computations and the solution for a subproblem is used to solve larger problems. This matches Bellman's **principle of optimality**, which states that "an optimal policy has the property that whatever the initial state and initial decision are, the remaining decisions must constitute an optimal policy with regard to the state resulting from the first decision." in computer science this is known as **optimal substructure**. 

{{< callout type="example" >}}
In our grid world example, the optimal state-value function $v_*(s)$ will assign high values to states that are close to the teleportation cells $A$ and $B$, as these lead to high rewards quickly. 

So if we are given the optimal value function, we can easily derive the optimal policy $\pi^*$ using the greedy approach, which simply chooses the action that leads to the highest expected value in the next state:

$$
\pi^*(s) = \arg\max_a \sum_{s' \in \mathcal{S}, r \in \mathbb{R}} \mathbb{P(s', r \mid s, a) \left[ r + \gamma v_*(s') \right]}.
$$

In our grid world example, if we take an action the resulting state is deterministic, so the optimal policy simply chooses the action that leads to the neighboring state with the highest value:

$$
\pi^*(s) = \arg\max_a v_*(s').
$$

For example, if the agent is in the state just above teleportation cell $A$, the optimal action would be to move down into cell $A$ to receive the high reward and teleport to $A'$. If there is a tie between multiple actions leading to equally good states, or all neighboring states have low values, the optimal policy can choose any of those actions arbitrarily as it can't stay in place.

{{< figure 
    src="/images/ml/rlGridWorldOptimal.png"
    caption="The optimal state-value function in the grid world example and the corresponding greedy policy."
    alt="The optimal state-value function in the grid world example and the corresponding greedy policy."
>}}
{{< /callout >}}

## Planning and Reinforcement Learning

So far we have assumed that the agent follows some fixed policy $\pi$ and derived the corresponding value functions and Bellman equations. But how do we actually find a good policy in the first place? This is where the concepts of **planning** and **reinforcement learning** come into play. Specifically, planning refers to the process of computing an optimal policy when the MDP model (transition and reward functions) is known, or in other words, when we have **complete knowledge** of the environment. In contrast, reinforcement learning deals with the situation where the environment is unknown and must be learned by interacting with the environment. However, both planning and reinforcement learning share the same underlying goal of finding an optimal policy that maximizes the expected return.

This can be very confussing to differentiate, especially since we can also do planning in reinforcement learning settings, for example by learning a model of the environment and then using that model to plan. However, the key distinction is whether the MDP model is whether you have access to a model of the environment, and you try to solve it via some form of advanced search. It does not require to collect true experience from the real environment, but some planning methods are based on simulated experience from the known (or modeled) environment. It's all in the agent's head, just like when you plan something, hence planning.

Learning is when you do not assume to have a model of the environment, and thus you need true experience to infer anything. And it can be done broadly speaking in two ways: 
- **Model-based learning**: Here, the agent tries to learn a model of the environment's dynamics (i.e., the transition probabilities and reward function) from its interactions with the environment. Once the model is learned, the agent can use planning methods to compute an optimal policy based on this learned model.
- **Model-free learning**: In this approach, the agent directly skips learning a model of the environment and instead focuses on learning the value functions or policies directly from its interactions with the environment. 

{{< figure 
    src="/images/ml/rlModelBasedFree.png"
    caption="Illustration of model-based and model-free reinforcement learning."
    alt="Illustration of model-based and model-free reinforcement learning."
>}}

There are a few other key concepts that are important to understand in the context of planning and reinforcement learning. First, we usually have some data from the environment in the form of **experience tuples** $(s, a, r, s')$, which represent a transition from state $s$ to state $s'$ after taking action $a$ and receiving reward $r$. Based on this experience data, we now want to find an optimal policy. Depending on the setting, we have to differentiate between two main scenarios:
- **Offline (batch) learning**: In this setting, we have a fixed dataset of experience tuples collected from the environment, and we want to learn an optimal policy from this static dataset without further interaction with the environment. We also call it batch learning because we process the data in batches, where each batch comes after some period of data collection using a deployed policy.
- **Online learning**: Here, the agent can interact with the environment in real-time, collecting new experience tuples as it explores. The agent updates its policy and value functions continuously based on this ongoing interaction. In the online setting, we then of course have to deal with our exploration vs. exploitation trade-off, where the agent has to balance between exploring new actions to gather more information about the environment and exploiting its current knowledge to maximize rewards to achieve the best possible performance.

{{< figure 
    src="/images/ml/rlOnlineOffline.png"
    caption="Illustration of offline (batch) and online reinforcement learning."
    alt="Illustration of offline (batch) and online reinforcement learning."
>}}

Another key concept is the idea of whether the agent's policy can use data from other policies or not. Sepcifically, we say the **behaviour policy** is the policy used to generate the experience data as it defines the behaviour of the agent while interacting with the environment. The **target policy** is the policy that we are trying to evaluate or improve. In some cases, the behaviour policy and target policy can be the same, while in other cases they can be different. This leads to the notion of:
- **On-policy learning**: In this case, the agent learns the value function or policy based on the actions taken by its current policy. In other words, the experience data used for learning is generated by the same policy that is being evaluated or improved. 

$$
\pi_{behaviour} = \pi_{target}
$$

These methods typically require the agent to explore the environment according to its current policy to gather relevant experience and update its value estimates or policy accordingly. This can lead to more stable learning, as the agent is always learning from data that is relevant to its current behavior. However, it can also be less efficient, as the agent may not explore the environment effectively and the sample efficiency is low as if we change the policy we need to start collecting new data again. Online and on-policy learning often go hand in hand, as the agent continuously updates its policy based on its ongoing interactions with the environment.
- **Off-policy learning**: Here, the agent can learn from experience data generated by a different policy than the one it is currently following. 

$$
\pi_{behaviour} \neq \pi_{target}
$$

This allows the agent to learn from a broader range of experiences, potentially improving learning and sample efficiency. However, off-policy learning can also be more challenging, as the agent needs to account for the differences between the behaviour policy (the one generating the data) and the target policy (the one being learned). This can lead to instability in learning if not handled properly. 

{{< figure 
    src="/images/ml/rlOnOffPolicy.webp"
    caption="Illustration of on-policy and off-policy reinforcement learning."
    alt="Illustration of on-policy and off-policy reinforcement learning."
>}}

### Policy Evaluation

To be able to find an optimal policy, we first need to be able to evaluate how good a given policy is. This is done through the policy evaluation step, where we compute the value function $v_\pi$ for the current policy. As mentioned earlier, the Bellman Equations gives us a system of linear equations. While we could solve this analytically, it is computationally expensive for large state spaces, specifically it requires inverting a matrix of size $|\mathcal{S}| \times |\mathcal{S}|$ which is $O(|\mathcal{S}|^3)$ in time complexity:

$$
\mathbf{v}_\pi = (\mathbf{I} - \gamma \mathbf{P}_\pi)^{-1} \mathbf{r}_\pi.
$$

Instead, we can use an iterative approach known as **iterative policy evaluation**. The idea is to start with an initial guess for the value function (e.g., all zeros) and then repeatedly update the value function using the Bellman equation until it converges to the true value function for the policy. For this we first define the **Bellman expectation operator** $\mathcal{T}^\pi$ which takes a value function $v$ as input and produces a new value function as output by applying the Bellman equation for policy $\pi$:

$$
(\mathcal{T}^\pi v)(s) = \sum_{a \in \mathcal{A}} \pi(a \mid s) \sum_{s' \in \mathcal{S}, r \in \mathbb{R}} \mathbb{P}(s', r \mid s, a) \left[ r + \gamma v(s') \right].
$$

The bellman equation itself can then be seen as a fixed point of this operator:

$$
v_\pi = \mathcal{T}^\pi v_\pi.
$$

More generally, a fixed point of a function or operator is a point that is mapped to itself by the function $F(x^*) = x^*$. In our case, the value function $v_\pi$ is a fixed point of the Bellman expectation operator $\mathcal{T}^\pi$ because applying the operator to $v_\pi$ yields $v_\pi$ itself. To then find this fixed point, we use **fixed-point iteration** where we start with an initial guess $v_0$ and then repeatedly improve our point by applying the operator:

$$
v_{k+1} = \mathcal{T}^\pi v_k.
$$

This gives us the iterative update rule for the value function:

$$
v_{k+1}(s) = \sum_{a \in \mathcal{A}} \pi(a \mid s) \sum_{s' \in \mathcal{S}, r \in \mathbb{R}} \mathbb{P}(s', r \mid s, a) \left[ r + \gamma v_k(s') \right].
$$

We repeat this update for all states until the value function converges, i.e., the changes in the value function are below a certain threshold $\theta$:

$$
\max_{s \in \mathcal{S}} |v_{k+1}(s) - v_k(s)| < \theta.
$$

Here we can see the connection to dynamic programming again, as we are caching the results of subproblems (the value of each state) and using them to compute the values of other states iteratively. This process is guaranteed to converge to the true value function $v_\pi$ for the policy $\pi$ as long as the discount factor $\gamma < 1$. 

{{< figure 
    src="/images/ml/rlPolicyEvaluation.png"
    caption="Pseudo-code illustration of the iterative policy evaluation process."
    alt="Pseudo-code illustration of the iterative policy evaluation process."
    width="400"
>}}

Policy evaluation and in fact many dynamic programming methods can be understood very cleanly from a **fixed point perspective**. A **fixed point** of a function $f$ is a value $x^*$ such that applying the function does not change it:

$$
f(x^*) = x^*.
$$

In our setting, the object we are trying to compute is the value function $v_\pi$, and the function we repeatedly apply is the **Bellman expectation operator**. So the operator takes a value function as input and produces a new value function as output by applying the Bellman expectation equation. We define the Bellman expectation operator for a fixed policy $\pi$ as follows:

$$
(\mathcal{B}^\pi v)(s) = \sum_{a \in \mathcal{A}} \pi(a \mid s) \left[ r(s,a) + \gamma \sum_{s' \in \mathcal{S}} \mathbb{P}(s' \mid s,a) v(s') \right].
$$

So applying the Bellman expectation operator to a value function produces a new value function according to the Bellman expectation equation. Specifically, if we start with a value function $v$ and apply the operator, we get a new value function $v_{new}$:

$$
\begin{align*}
v_{new}(s) &= \sum_{a \in \mathcal{A}} \pi(a \mid s) \left[ r(s,a) + \gamma \sum_{s' \in \mathcal{S}} \mathbb{P}(s' \mid s,a) v(s') \right] \\
&= (\mathcal{B}^\pi v)(s).
\end{align*}
$$

Therefore, if our goal is to find the value function $v_\pi$ for policy $\pi$, we want to find a fixed point of the Bellman expectation operator such that applying the operator does not change the value function:

$$
v_\pi = \mathcal{B}^\pi v_\pi.
$$

This leads us directly to the **fixed point iteration** which is exactly the iterative policy evaluation procedure we described earlier. Specifically, we start from an initial guess for the value function $v_0$ and repeatedly apply the Bellman expectation operator to obtain a sequence of value functions until convergence:

$$
v_{k+1} = \mathcal{B}^\pi v_k.
$$

our initial guess could be all zeros, i.e., $v_0(s) = 0$ for all states $s$. From the above definition of the Bellman expectation operator, we can see that the update rule is exactly:

$$
v_{k+1}(s) = \sum_{a \in \mathcal{A}} \pi(a \mid s) \left[ r(s,a) + \gamma \sum_{s' \in \mathcal{S}} \mathbb{P}(s' \mid s,a) v_k(s') \right].
$$

which matches the iterative policy evaluation update we had before. Importantly we need to show that this fixed point iteration fullfills the following properties:
- **Convergence**: The sequence of value functions $v_k$ converges to the true value function $v_\pi$ as $k \to \infty$. This convergence especially needs to hold for any initial guess $v_0$.
- **Uniqueness**: The fixed point $v_\pi$ is unique, meaning that there is only one value function that satisfies the Bellman expectation equation for policy $\pi$.

Formally this means that the Bellman expectation operator is a **contraction mapping**. This is the case when the discount factor $\gamma < 1$. Intuitively, a contraction mapping is a function that **pulls things closer together** every time it is applied. More precisely, for any two value functions $v$ and $u$, the distance between their images under the operator is strictly less than the distance between the original functions, scaled by a factor $\beta < 1$:

$$
\begin{align*}
\| \mathcal{B}^\pi v - \mathcal{B}^\pi u \|_\infty &\leq \beta \| v - u \|_\infty \\
\text{where} \quad \beta &= \gamma < 1.
\end{align*}
$$

{{< callout type="proof" >}}
To show that the Bellman expectation operator is a contraction mapping, we will use the max-norm (also known as the infinity norm) to measure the distance between value functions and then show that applying the operator reduces this distance by a factor of $\gamma$. We define the max-norm as $\|v\|_\infty = \max_{s \in \mathcal{S}} |v(s)|$.

Let $u$ and $v$ be any two value functions (vectors in $\mathbb{R}^{|\mathcal{S}|}$). For any state $s$, we have:

$$
\begin{align*}
|(\mathcal{B}^\pi u)(s) - (\mathcal{B}^\pi v)(s)| &= \left| \left( r_\pi(s) + \gamma \sum_{s'} p(s'|s,\pi(s)) u(s') \right) - \left( r_\pi(s) + \gamma \sum_{s'} p(s'|s,\pi(s)) v(s') \right) \right| \\
&= \left| \gamma \sum_{s'} p(s'|s,\pi(s)) (u(s') - v(s')) \right| \\
&\leq \gamma \sum_{s'} p(s'|s,\pi(s)) |u(s') - v(s')| \quad (\text{Triangle Inequality}) \\
&\leq \gamma \sum_{s'} p(s'|s,\pi(s)) \|u - v\|_\infty \quad (\text{Replacing term with max difference}) \\
&= \gamma \|u - v\|_\infty \sum_{s'} p(s'|s,\pi(s)) \\
&= \gamma \|u - v\|_\infty \quad (\text{Since probabilities sum to 1})
\end{align*}
$$

Since this holds for any state $s$, it must hold for the maximum over $s$ which is the max-norm:

$$
\|\mathcal{B}^\pi u - \mathcal{B}^\pi v\|_\infty \leq \gamma \|u - v\|_\infty
$$

Since $0 \le \gamma < 1$, the operator is a contraction. By the **Banach Fixed Point Theorem**, applying this operator iteratively will converge to a unique fixed point $v_\pi$.

Intuitively, you can also think of that at each "sweep" through all states, we are getting information about the rewards which can then be used to update the values of other states in the next sweep, and over time this information propagates through the state space until all values are accurate.
{{< /callout >}}

### Policy Improvement

Once we have evaluated the current policy and obtained its value function $v_\pi$, we can use this information to improve the policy. The idea is to use the value function to identify better actions in each state, leading to a new policy that is at least as good as the current one. This process is known as **policy improvement**.

The idea is rather simple. Consider we are in some state $s$ and the current policy selects action $\pi(s)$. If we then were to find an action which isn't $\pi(s)$ but leads to a higher expected return according to the value function, then we could improve the policy by switching to that action instead, so $q_\pi(s,a) > v_\pi(s)$ for some action $a \neq \pi(s)$. The reason this works is because in both cases after the initial action we follow the same policy $\pi$ thereafter, so if taking action $a$ now leads to a higher expected return than taking action $\pi(s)$ now, then the new policy which takes action $a$ in state $s$ and follows $\pi$ thereafter must be better than the old policy. Formally if we fix any state $s$ then for the current policy $\pi$ we have:

$$
v_\pi(s) = \sum_{a \in \mathcal{A}} \pi(a \mid s) q_\pi(s,a) \leq \max_a q_\pi(s,a) = q_\pi(s, \arg\max_a q_\pi(s,a)).
$$

we can then simply set $\pi'(s) = \arg\max_a q_\pi(s,a)$ to get a new policy $\pi'$ which is guaranteed to be at least as good as $\pi$ in state $s$:

$$
v_\pi(s) \leq q_\pi(s, \pi'(s)) = v_{\pi'}(s).
$$

This is known as the **policy improvement theorem** and the picking of the action that maximizes the action-value function is known as the **greedy policy improvement**, as we are always choosing the action that looks best according to the current value function:

$$
\pi'(s) = \arg\max_a q_\pi(s,a).
$$

However, in practice we don't have the action-value function $q_\pi$ directly, but we can compute it from the value function $v_\pi$ using the Bellman equation for the action-value function:

$$
q_\pi(s,a) = \sum_{s' \in \mathcal{S}, r \in \mathbb{R}} \mathbb{P}(s', r \mid s, a) \left[ r + \gamma v_\pi(s') \right].
$$

Putting everything together, we can improve the policy by updating it to choose the action that maximizes the expected return based on the current value function:

$$
\pi'(s) = \arg\max_a \sum_{s' \in \mathcal{S}, r \in \mathbb{R}} \mathbb{P}(s', r \mid s, a) \left[ r + \gamma v_\pi(s') \right].
$$

If the new policy $\pi'$ is the same as the old policy $\pi$, then we have found the optimal policy and can stop. Otherwise, we can repeat the process of policy improvement and calculating the value function for the new policy until convergence. 

### Policy Iteration

One of the fundamental algorithms for finding an optimal policy in a known MDP is **policy iteration**. This algorithm alternates between two main steps: **policy evaluation** and **policy improvement**. By iteratively evaluating the current policy and then improving it based on the value function, policy iteration converges to the optimal policy.

{{< figure 
    src="/images/ml/rlPolicyIterationCycle.png"
    caption="The cycle of policy iteration slowly converging to the optimal policy."
    alt="The cycle of policy iteration slowly converging to the optimal policy."
    width="500"
>}}

For a finite MDP with a finite number of states and actions, policy iteration is guaranteed to converge to the optimal policy $\pi^*$ in a finite number of iterations. This is because each policy improvement step produces a strictly better policy unless the current policy is already optimal, and there are only a finite number of possible policies. 

The biggest issue with policy iteration is that the policy evaluation step can be computationally expensive, especially for large state spaces, as it requires solving a system of linear equations or performing many iterations of value updates. However, in practice, policy iteration often converges in relatively few iterations, making it a powerful method for finding optimal policies in MDPs.

{{< figure 
    src="/images/ml/rlPolicyIteration.png"
    caption="Pseudo-code illustration of the policy iteration algorithm."
    alt="Pseudo-code illustration of the policy iteration algorithm."
    width="400"
>}}

{{< callout type="proof" >}}
We want to prove that the policy improvement step always produces a policy that is at least as good as the current policy. This is formalized in the **Policy Improvement Theorem**.

Let $\pi$ be a policy and $\pi'$ be a greedy policy with respect to $v_\pi$. Then we want that $v_{\pi'}(s) \ge v_\pi(s)$ for all $s$. If $v_{\pi'}(s) > v_\pi(s)$ for at least one state, the policy has strictly improved.

Let $\pi$ be the current policy and $v_\pi$ be its value function. If we then let $\pi'$ be the greedy policy, such that $\pi'(s) = \arg\max_a q_\pi(s, a)$ we have by definition of the greedy selection:

$$
q_\pi(s, \pi'(s)) \ge q_\pi(s, \pi(s)) = v_\pi(s)
$$

Based on the inequality above ($q_\pi(s, \pi'(s)) \ge v_\pi(s)$), we can see that applying the operator of the new policy to the old value function improves it (or keeps it equal):

$$
(B^{\pi'} v_\pi)(s) = r(s, \pi'(s)) + \gamma \sum_{s'} p(s'|s, \pi'(s)) v_\pi(s') = q_\pi(s, \pi'(s)) \ge v_\pi(s)
$$

So, $B^{\pi'} v_\pi \ge v_\pi$ which means that the Bellman operator for the new policy produces a value function that is at least as good as the old value function. Therefore, $B^{\pi'}$ improves $v_\pi$ and is a monotone operator (if $u \ge v$ then $B^{\pi'} u \ge B^{\pi'} v$). If we then repeatedly apply the Bellman operator for the new policy to the old value function, we get a sequence of value functions that are non-decreasing:

$$
\begin{align*}
v_\pi &\le B^{\pi'} v_\pi \\
&\le B^{\pi'} (B^{\pi'} v_\pi) \\
&\le \dots \\
&\le \lim_{k \to \infty} (B^{\pi'})^k v_\pi
\end{align*}
$$

We also know from the Fixed Point analysis that repeated application of the Bellman operator for $\pi'$ converges to the value function of $\pi'$.

$$
\lim_{k \to \infty} (B^{\pi'})^k v_\pi = v_{\pi'}
$$

Therefore we have that the value function of the new policy is at least as good as the value function of the old policy:

$$
v_\pi(s) \le v_{\pi'}(s)
$$

If the inequality is strict at any step, the policy strictly improves. Convergence is guaranteed because in a finite MDP there are a finite number of deterministic policies.
{{< /callout >}}

{{< callout type="example" >}}
In our grid world example, we can apply policy iteration to find the optimal policy for the agent. Starting with an initial policy (e.g., moving randomly), we first evaluate the policy to compute its value function. Then, we improve the policy by choosing actions that lead to higher expected returns based on the current value function. Repeating this process, we eventually converge to the optimal policy that guides the agent to the teleportation cells $A$ and $B$ efficiently, maximizing its long-term rewards.

In this example, we can see that the policy iteration algorithm quickly converges to the optimal policy, demonstrating its effectiveness in solving MDPs.

{{< figure 
    src="/images/ml/rlGridWorldPolicyIteration.png"
    caption="Visualization of policy iteration in the grid world example."
    alt="Visualization of policy iteration in the grid world example."
>}}
{{< /callout >}}

### Value Iteration

While policy iteration is effective, it has a significant drawback: the policy evaluation step requires solving a linear system or running many iterative updates until convergence just to evaluate a single policy. This can be computationally wasteful, especially since we often only care about identifying the greedy action, not the exact value.

In comparison, **value iteration** streamlines the process by combining policy evaluation and improvement into a single step. Instead of fully evaluating the current policy, value iteration updates the value function using the Bellman optimality equation directly, which inherently incorporates the policy improvement step. Specifically, the update rule for value iteration is:

$$
v_{k+1}(s) = \max_a \sum_{s' \in \mathcal{S}, r \in \mathbb{R}} \mathbb{P(s', r \mid s, a) \left[ r + \gamma v_k(s') \right]}.
$$

Here the inner sum term looks similar to the policy evaluation update, so we are collecting the immediate return for each action $a$ in state $s$ based on the current value function $v_k$, and then we are optimizing over actions by taking the maximum. This effectively performs a greedy policy improvement step at each iteration, as we are always choosing the action that looks best according to the current value function.

Value iteration is generally preferred when the state space is large but the transition dynamics are sparse (i.e., from any state, you can only transition to a few others). This is because value iteration can exploit this sparsity more effectively than policy iteration, which requires evaluating the entire policy at each step. However, value iteration may require more iterations to converge compared to policy iteration, as it does not fully evaluate the policy at each step but instead makes approximate updates, but each iteration is computationally cheaper.

Similarly to policy evaluation, value iteration can also be understood from a fixed point perspective. Here, we define the **Bellman optimality operator** as follows:

$$
(\mathcal{B}^* v)(s) = \max_a \left[ r(s,a) + \gamma \sum_{s' \in \mathcal{S}} \mathbb{P}(s' \mid s,a) v(s') \right].
$$

So applying the Bellman optimality operator to a value function produces a new value function according to the Bellman optimality equation. Specifically, if we start with a value function $v$ and apply the operator, we get a new value function $v_{new}$:

$$
\begin{align*}
v_{new}(s) &= \max_a \left[ r(s,a) + \gamma \sum_{s' \in \mathcal{S}} \mathbb{P}(s' \mid s,a) v(s') \right] \\
&= (\mathcal{B}^* v)(s).
\end{align*}
$$

Therefore, if our goal is to find the optimal value function $v_*$, we want to find a fixed point of the Bellman optimality operator such that applying the operator does not change the value function:

$$
v_* = \mathcal{B}^* v_*.
$$

Note that the Bellman optimality operator is similar to the Bellman expectation operator, but instead of taking an expectation over actions according to a policy, it takes the maximum over all possible actions. Importantly, because of the relation between the state-value and action-value functions, we can also express the Bellman optimality operator in terms of the action-value function:

$$
(\mathcal{B}^* v)(s) = \max_a q_*(s,a),
$$

Another interesting point is that value iteration can be seen as extending the time horizon of the policy being evaluated at each iteration. For example if we initialize the value function to zero, i.e., $v_0(s) = 0$ for all states $s$, then the first iteration of value iteration computes the value function for a policy that only considers immediate rewards (i.e., a one-step lookahead). The second iteration then considers rewards up to two steps into the future, and so on. As the number of iterations increases, the effective time horizon of the policy being evaluated extends further into the future, eventually converging to the optimal policy that considers the entire infinite horizon. This perspective also highlights the fact that value iteration converges asymptotically to the optimal value function, as each iteration refines the value estimates by considering longer-term rewards, whereas policy iteration finds the optimal policy in a finite number of steps if the MDP is finite.

{{< figure 
    src="/images/ml/rlValueIteration.png"
    caption="Pseudo-code illustration of the value iteration algorithm."
    alt="Pseudo-code illustration of the value iteration algorithm."
    width="400"
>}}

{{< callout type="proof" >}}
To show that the Bellman optimality operator is a contraction mapping and therefore that value iteration converges to the optimal value function we use a similar approach as for the Bellman expectation operator. Specifically we want to show that value iteration converges asymptotically to the optimal value function $v_*$.

We have already established the section on policy evaluation that the Bellman expectation operator is a contraction mapping. Because the Bellman optimality operator involves taking a maximum over actions, which is a non-expansive operation, we can extend the contraction property to the Bellman optimality operator as well. Specifically, we can show that $\mathcal{B}^*$ is a contraction mapping with factor $\gamma < 1$ under the max-norm due to the following property:

$$
|\max_a f(a) - \max_a g(a)| \leq \max_a |f(a) - g(a)|
$$

Value iteration is then simply the fixed-point iteration:

$$
v_{k+1} = \mathcal{B}^* v_k
$$

Let $v_*$ be the unique fixed point of $\mathcal{B}^*$ (which is the optimal value function).
The distance between the estimate at step $k$ and the optimal value is:

$$
\| v_{k+1} - v_* \|_\infty = \| \mathcal{B}^* v_k - \mathcal{B}^* v_* \|_\infty
$$

Using the contraction property:

$$
\| \mathcal{B}^* v_k - \mathcal{B}^* v_* \|_\infty \leq \gamma \| v_k - v_* \|_\infty
$$

By induction, applying this $k$ times:

$$
\| v_k - v_* \|_\infty \leq \gamma^k \| v_0 - v_* \|_\infty
$$

Since $\gamma < 1$, as $k \to \infty$, $\gamma^k \to 0$. Therefore, the error approaches 0, and $v_k$ converges to $v_*$.
{{< /callout >}}

{{< callout type="example" >}}
In our grid world example, we can apply value iteration to find the optimal policy for the agent. We can initialize $v_0(s) = 0$ for all states. At the very beginning, the only place where rewards “enter” the system are the immediate rewards near $A$ and $B$.

On the first sweep, $v_1$ will become non-zero only in states that can reach $A$ or $B$ in one step, because we are looking at $r + \gamma v_0(s') = r$ and $v_0$ is zero everywhere. On the second sweep, values propagate one step further back: states that are two steps away from $A$ and $B$ now start to get positive values, and so on.

With each sweep, this “wave” of value flows backwards through the grid, roughly at one step per iteration, until every state’s value reflects the best possible path to $A$ or $B$ (balanced against the $-1$ penalties and discounting). The resulting $v_*$ is identical to the optimal value function you would get from policy iteration, but value iteration jumped straight towards it without explicitly keeping track of a policy during the iterations.

Once the value function has converged, we can derive the optimal policy by choosing the action that maximizes the expected return in each state based on the final value function.
{{< /callout >}}

## Tabular Reinforcement Learning

So far we have assumed that the MDP model (transition and reward functions) is known, allowing us to use planning methods like policy iteration and value iteration to find an optimal policy. However, in many real-world scenarios, the environment is unknown, and the agent must learn to make decisions through interaction with the environment. This is where **reinforcement learning (RL)** comes into play. Specifically in this section, we will focus on **tabular reinforcement learning** methods, which are suitable for small-scale problems with finite state and action spaces where we can represent the value functions and policies using tables, hence the name "tabular".

### Model-Based Methods

For the model-based approach, we first need to learn a model of the environment's dynamics from the experience data. This involves estimating the transition probabilities $\mathbb{P}(s' \mid s, a)$ and the reward function $r(s, a)$ based on the collected experience tuples $(s, a, r, s')$. Once we have learned the model, we can then use planning methods like value iteration or policy iteration to compute an optimal policy based on the learned model. 

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

#### Epsilon-Greedy Exploration

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

#### Softmax Exploration

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

#### Rmax Algorithm

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

### Model-free Approaches

The problem with model-based methods is that learning an accurate model of the environment can be challenging, especially in complex or high-dimensional environments. Additionally, planning methods like value iteration and policy iteration can be computationally expensive, particularly for large state spaces. Especially since to store the tables of transition probabilities and rewards we need space O(|S|^2 * |A|) which can get large quickly.

Instead we now consider **model-free** reinforcement learning methods, which skip the model learning step entirely and directly learn value functions or policies from experience data. Model-free methods are often more scalable and can handle larger state spaces more effectively, as they do not require storing or updating a full model of the environment.

#### Monte Carlo Methods

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

#### Temporal-Difference Learning

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

We can also rewrite the update rule in terms of the TD error:

$$
v_\pi(s_t) \leftarrow v_\pi(s_t) + \alpha \left[r_{t+1} + \gamma v_\pi(s_{t+1}) - v_\pi(s_t) \right] = v_\pi(s_t) + \alpha \delta_t.
$$

Using bootstrapping allows TD learning methods to update value estimates more frequently and efficiently, leading to faster convergence compared to Monte Carlo methods. Additionally, TD learning methods can be applied to both episodic and continuing tasks, making them more versatile. 

Intuitevely, the TD Error meassures how surprised we are about the new information we just received. If the TD error is large, it means that our current value estimate was far off from what we observed, indicating that we need to adjust our estimate significantly. Conversely, if the TD error is small, it suggests that our current estimate was accurate, and only a minor adjustment is needed.

By also weighting the updates with a learning rate $\alpha$, we can control the influence of new information on the value estimates, reducing variance and improving stability. However, the act of bootstrapping also introduces some bias into the estimates, as we are relying on our current value estimates rather than actual returns. This bias-variance trade-off is a key consideration in TD learning methods and can be managed through the choice of the learning rate $\alpha$ and other hyperparameters (e.g., in TD(lambda)).

Again, once we have estimated the value function using TD learning, we can use it to improve the policy in a similar manner as in policy iteration. Specifically, we can derive a greedy policy with respect to the estimated value function. 

#### SARSA

In Monte Carlo policy evaluation and TD learning, we focused on estimating the state-value function $v_\pi(s)$ for a given policy $\pi$ and then applied some form of policy improvement to derive a better policy.

However, in many scenarios, it is more useful to estimate the action-value function $q_\pi(s,a)$, which gives the expected return for taking action $a$ in state $s$ and then following policy $\pi$. This leads us to **SARSA (State-Action-Reward-State-Action)**, which results in not just estimating the action-value function but also learning a policy simultaneously, making it an on-policy control method.

Similarly to TD learning, SARSA updates the action-value estimates based on the observed reward and the estimated value of the next state-action pair. The update rule for SARSA is as follows:

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

#### Q-Learning

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

## Deep Reinforcement Learning

In the tabular case, we can represent the value functions and policies using tables, which is feasible for small state and action spaces. However, in many real-world applications, the state and action spaces are large or continuous, making tabular methods impractical. To address this challenge, we can use function approximation techniques, such as neural networks, to represent the value functions and policies. This leads us to the field of **deep reinforcement learning (deep RL)**, which combines reinforcement learning with deep learning.

Importantly we treat discrete and continuous action spaces differently here. If we for example approximate a policy with a neural network in a discrete action space, we can simply have the network output a probability distribution over the discrete actions. This then becomes a classification problem where the output layer uses a softmax activation to produce probabilities for each action. In contrast, in continuous action spaces, we do not just want it to output a single action like in regression, but rather a distribution over actions. To model this, we can have the neural network output the parameters of a probability distribution (e.g., mean and variance for a Gaussian distribution) from which we can then sample actions. This allows us to capture the uncertainty in action selection and also enables exploration in continuous action spaces.

{{< figure 
    src="/images/ml/rlDiscreteContinousActions.jpg"
    caption="Illustration of deep reinforcement learning in discrete and continuous action spaces."
    alt="Illustration of deep reinforcement learning in discrete and continuous action spaces."
>}}

### Model-Free

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

#### Deep Q-Networks (DQN)

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

#### Policy Gradient Methods

The problem with value-based methods like DQN is that they are only learning the policy indirectly through the value function. This can lead to suboptimal policies, especially in complex environments where the value function may not accurately capture the optimal policy. Additionally, they can struggle in high-dimensional or continuous action spaces, as finding the action that maximizes the Q-value can be challenging. 

Instead we want to directly learn the policy itself, which leads us to **policy search/gradient methods**. These methods optimize the policy directly by adjusting the policy parameters in the direction that maximizes the expected return.

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

#### Actor–Critic

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

#### Trust Region Policy Optimization (TRPO)

This is the clipped version? to increase sample efficiency and stability.

#### Proximal Policy Optimization (PPO)
A stable and practical policy gradient method.

* Avoiding destructive large policy updates
* Clipped surrogate objective
* Why PPO works well in practice

special case of TRPO?

#### Group Related Policy Optimization (GRPO)

### Maximum Entropy RL (MERL)

motivates exploration by augmenting the reward with an entropy term.

soft actor-critic (SAC)?

#### Deep Deterministic Policy Gradient (DDPG)

T3, twin delayed DDPG (TD3)? Similar idea to DQN and double DQN to reduce overestimation bias.

#### Stochastic Value Gradients (SVG)

#### Soft Actor-Critic (SAC)

soft Q-learning?

soft actor-critic (SAC)?

### Model-Based

World models

#### Model Predictive Control (MPC)

random shooting methods

for stochastic dynamics we use:

trajectory sampling methods like thompson sampling?

#### Pilco?

#### Pets?

#### Hallucinated upper confidence reinforcement learning

#### Constrained and Safe RL

## Learning from Preferences

### Reinforcement Learning with Human Feedback (RLHF)

### Direct Preference Optimization (DPO)

