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

The same goes for the reward function $r(s, a, s')$, which results in some scalar reward $R_{t+1}$ when transitioning from state $s_t$ to $s_{t+1}$ after taking action $a_t$. This reward can also be stochastic, meaning that the same state-action-next state transition can yield different rewards at different times. For example, in a stock trading environment, taking the action "buy" in a certain market state may lead to different profits or losses depending on market fluctuations. So we define the **expected reward function** as:

$$
r(s, a, s') = \mathbb{E}[R_{t+1} | s_t = s, a_t = a, s_{t+1} = s'] = \sum_{r \in \mathbb{R}} r \cdot \mathbb{P}(R_{t+1} = r | s_t = s, a_t = a, s_{t+1} = s')
$$

We also often want to know the expected reward for taking action $a$ in state $s$ regardless of the next state. We can compute this by marginalizing over all possible next states which results in the **expected immediate reward function**:

$$
r(s, a) = \mathbb{E}[R_{t+1} | s_t = s, a_t = a] = \sum_{s' \in \mathcal{S}}\mathbb{P}(s'|s,a) \cdot r(s, a, s')
$$

The choices the agent makes at each time step are determined by its **policy** $\pi(a|s)$, which specifies the probability of taking action $a$ when in state $s$. A policy can be **deterministic**, mapping each state to a specific action, or **stochastic**, assigning probabilities to each possible action in a state:

$$
\pi(a|s) = \mathbb{P}(a_t = a | s_t = s) \text{ stochastic policy } \quad \text{or} \quad a = \pi(s) \text{ deterministic policy }
$$

When the agent follows a policy $\pi$ in an MDP, it generates a sequence of states, actions, and rewards over time. These chains of states, actions, and rewards generated by following this process are called **trajectories** or **episodes**. A trajectory starting at time step $t=0$ and ending at time step $T$ can be represented as a tuple:

$$
\tau = (s_0, a_0, R_1, s_1, a_1, R_2, s_2, a_2, R_3, ..., s_{T-1}, a_{T-1}, R_T, s_T)
$$

The goal of the agent is to maximize the total amount of reward it receives. However, what does this mean in practice? Depending on the task, the task may be **episodic**, meaning that it naturally breaks down into separate episodes with a clear starting and ending point (e.g., a game of chess). We can also force a task to be episodic by defining a maximum time horizon or by introducing terminal states. Alternatively, the task may be **continuing**, where the agent interacts with the environment indefinitely without a natural endpoint (e.g., controlling a thermostat). In either case, we define the **return** (G_0) as the total accumulated reward from time step (t = 0) onward. In episodic tasks, this is simply the sum of rewards until the end of the episode:

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

The value function only depends on the policy $\pi$ and the state $s$, not on the specific time step $t$, because of the Markov property and by taking the expectation over all possible future trajectories we take out the randomness. 

Intuitively, $v_\pi(s)$ tells us the long-term expected return if we start in state $s$ and then follow policy $\pi$. If $v_\pi(s)$ is high, it means that being in state $s$ is desirable under policy $\pi$, as it leads to high expected future rewards. Conversely, if $v_\pi(s)$ is low or negative, it indicates that being in state $s$ is not beneficial when following policy $\pi$. Importantly, the value function depends on the policy $\pi$, for example if one policy is a Grandmaster at chess and another is a beginner, the value of being in a certain board position will differ significantly between the two policies because you would expect the Grandmaster to be able to convert that position into a win much more often than the beginner which might blunder away the advantage.

However, the state-value function only tells us how good it is to be in a certain state, it doesn't directly tell us what to do in that state. This leads us to the **action-value function** also called **Q-Function** which tells us the expected return of if we start in state $s$, take action $a$ (which might be subpotimal or not according to the policy), and then follow policy $\pi$ thereafter:

$$
q_\pi(s,a) = \mathbb{E}_\pi [ G_t \mid S_t = s, A_t = a ] = \mathbb{E}_\pi \left[ \sum_{k=0}^{\infty} \gamma^k R_{t+1+k} \mid S_t = s, A_t = a \right].
$$

The distinction is simple but powerful:
- $v_\pi(s)$ tells us the expected return of being in state $s$ and behaving "normally", i.e. following policy $\pi$.
- $q_\pi(s,a)$ tells us the expected return of being in state $s$, taking action $a$ (which might be good or bad), and then behaving "normally" again.

These two functions are crucial. As remember that the agent can't see into the future. The reward signal is delayed and noisy, a decision now might have consequences far later. Value functions compress the entire uncertain future into a **single numeric prediction** and therefore allow us to evaluate how good different states and actions are in terms of expected future rewards, which in turn enables us to make informed decisions about which actions to take in order to maximize the long-term reward. In this sense, value functions provide the agent with a **map of the future**, telling it which states and which decisions are promising.

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

### Bellman Equations

The definition of the value functions above is useful for understanding what they represent, but it does not provide a practical way to compute them as they define value as an expectation over all possible future trajectories, which can be infinite in length. This would mean we would theoretically have to simulate the agent in all states and actions for an infinite amount of time to compute the value functions exactly, which would be intractable in all but the simplest cases.

Instead, we can exploit the recursive structure of the return $G_t$ to derive a set of equations known as the **Bellman equations**. These equations express the value functions in terms of themselves, allowing us to compute them iteratively. Remember that the return can be expressed recursively as:

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

Putting everything together, we arrive at the **Bellman equation for the state-value function**:

$$
v_\pi(s) = \sum_{a \in \mathcal{A}} \pi(a \mid s) \sum_{s' \in \mathcal{S}, r \in \mathbb{R}} \mathbb{P}(s', r \mid s, a) \left[ r + \gamma v_\pi(s') \right].
$$

It states that the value of a state under policy $\pi$ is equal to the expected immediate reward plus the expected discounted value of the next state, averaged over all possible actions and next states according to the policy and transition probabilities. We can also express this more compactly by combining the sums back into a single expectation:

$$
v_\pi(s) = \mathbb{E}_\pi [ R_{t+1} + \gamma v_\pi(S_{t+1}) \mid S_t = s ].
$$

using the definition of the state-value function in terms of the action-value function we can also write the Bellman equation as:

$$
v_\pi(s) = \sum_{a \in \mathcal{A}} \pi(a \mid s) \, q_\pi(s,a).
$$

Importantly we get such an equation for each state $s \in \mathcal{S}$, resulting in a system of $|\mathcal{S}|$ linear equations with $|\mathcal{S}|$ unknowns (the values $v_\pi(s)$). This system can be solved using standard linear algebra techniques, such as matrix inversion or iterative methods like value iteration. Specifically, if we collect the values $v_\pi(s)$ into a vector $\mathbf{v}_\pi$:

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

For small problems, we can directly compute the inverse of matrix $\mathbf{A}$ to find the value function but for larger problems, iterative methods like value iteration or policy evaluation are more practical. The inverse $(\mathbf{I} - \gamma \mathbf{P}_\pi)^{-1}$ exists as long as $\gamma < 1$ and the MDP is well-defined (i.e., the transition probabilities are valid summing to one). Don't ask me why though, I am not a mathematician.

If we now go back to the action-value function, we can derive its Bellman equation in a similar manner. Starting from the definition:

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

Giving us the final form of the **Bellman equation for the action-value function**. Similar to before, we can also express this all as an expectation:

$$
q_\pi(s,a) = \mathbb{E}_\pi [ R_{t+1} + \gamma v_\pi(S_{t+1}) \mid S_t = s, A_t = a ].
$$

It is easy to see that the value functions are related to each other. If in state $s$ the policy $\pi$ chooses action $a$ with probability $\pi(a \mid s)$, then the value function is just the expectation of action-values under the policy:

$$
v_\pi(s) = \sum_{a \in \mathcal{A}} \pi(a \mid s) q_\pi(s,a).
$$

This relationship is extremely important, as in practice it is often easier to estimate action-values directly, and then derive state-values from them. This is the basis for many reinforcement learning algorithms, such as Q-learning and Deep Q-Networks (DQN) etc. 

An important note is that if the policy $\pi$ is deterministic, meaning it always chooses the same action $a = \pi(s)$ in state $s$, then the relationship simplifies to:

$$
v_\pi(s) = q_\pi(s, \pi(s)).
$$

So in summary, the Bellman equations provide a recursive way to compute the value functions by expressing them in terms of themselves. This allows us to use iterative methods to compute the value functions efficiently, which is crucial for solving MDPs and finding optimal policies. We have also shown that the state-value and action-value functions are closely related, with the state-value function being the expectation of the action-value function under the policy and that we can also compute the value functions using linear algebra techniques for small problems.

- **Bellman state-value equation**:

$$
v_\pi(s) = \sum_{a \in \mathcal{A}} \pi(a \mid s) \sum_{s' \in \mathcal{S}, r \in \mathbb{R}} \mathbb{P}(s', r \mid s, a) \left[ r + \gamma v_\pi(s') \right].
$$

- **Bellman action-value equation**:

$$
q_\pi(s,a) = \sum_{s' \in \mathcal{S}, r \in \mathbb{R}} \mathbb{P}(s', r \mid s, a) \left[ r + \gamma \sum_{a' \in \mathcal{A}} \pi(a' \mid s') \, q_\pi(s',a') \right].
$$
- **Relationship between state-value and action-value functions**:

$$
\begin{align*}
v_\pi(s) &= \sum_{a \in \mathcal{A}} \pi(a \mid s) q_\pi(s,a) \\
q_\pi(s,a) &= \sum_{s' \in \mathcal{S}, r \in \mathbb{R}} \mathbb{P}(s', r \mid s, a) \left[ r + \gamma v_\pi(s') \right]
\end{align*}
$$

### Bellman Optimality Equations

The Bellman equations allow us to compute the value functions iteratively and therefore evaluate how good a specific policy $\pi$ is. However, our ultimate goal is not just to evaluate a fixed policy, but to find the **optimal policy** $\pi^*$ that maximizes the expected return from every state. So in other words, we want to find the **optimal state-value function** $v_*(s)$ which corresponds to the maximum value achievable from state $s$ under any policy:

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

## Planning

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

### Policy Evaluation

{{< callout type="todo" >}}
Clearly mention tabular case only, i.e. small finite state and action spaces. Later on we will look at other cases.
{{< /callout >}}

To be able to find an optimal policy, we first need to be able to evaluate how good a given policy is. This is done through the policy evaluation step, where we compute the value function $v_\pi$ for the current policy. As mentioned earlier, the Bellman Equations gives us a system of linear equations. While we could solve this analytically, it is computationally expensive for large state spaces, specifically it requires inverting a matrix of size $|\mathcal{S}| \times |\mathcal{S}|$ which is $O(|\mathcal{S}|^3)$ in time complexity. Instead, we can use an iterative approach known as **iterative policy evaluation**. The idea is to start with an initial guess for the value function (e.g., all zeros) and then repeatedly update the value function using the Bellman equation until it converges to the true value function for the policy. Specifically, we can use the following update rule for each state $s$:

$$
v_{k+1}(s) = \sum_{a \in \mathcal{A}} \pi(a \mid s) \sum_{s' \in \mathcal{S}, r \in \mathbb{R}} \mathbb{P}(s', r \mid s, a) \left[ r + \gamma v_k(s') \right].
$$

We repeat this update for all states until the value function converges, i.e., the changes in the value function are below a certain threshold $\theta$:

$$
\max_{s \in \mathcal{S}} |v_{k+1}(s) - v_k(s)| < \theta.
$$

*How is the above derived?*

Here we can see the connection to dynamic programming again, as we are caching the results of subproblems (the value of each state) and using them to compute the values of other states iteratively. This process is guaranteed to converge to the true value function $v_\pi$ for the policy $\pi$ as long as the discount factor $\gamma < 1$. 

This algorithm can also be viewed as a form of **fixed-point iteration** ??? explanation and formal definition?
Bellman expectation operator?

{{< figure 
    src="/images/ml/rlPolicyEvaluation.png"
    caption="Pseudo-code illustration of the iterative policy evaluation process."
    alt="Pseudo-code illustration of the iterative policy evaluation process."
    width="400"
>}}

### Policy Improvement

Once we have evaluated the current policy and obtained its value function $v_\pi$, we can use this information to improve the policy. The idea is to use the value function to identify better actions in each state, leading to a new policy that is at least as good as the current one. This process is known as **policy improvement**.

The idea is rather simple. Consider we are in some state $s$ and the current policy selects action $\pi(s)$. If we then were to find an action which isn't $\pi(s)$ but leads to a higher expected return according to the value function, then we could improve the policy by switching to that action instead, so $q_\pi(s,a) > v_\pi(s)$ for some action $a \neq \pi(s)$. The reason this works is because in both cases after the initial action we follow the same policy $\pi$ thereafter, so if taking action $a$ now leads to a higher expected return than taking action $\pi(s)$ now, then the new policy which takes action $a$ in state $s$ and follows $\pi$ thereafter must be better than the old policy. 

So if we find a policy $\pi'$ such that for all states $s$:

$$
q_\pi(s, \pi'(s)) \geq v_\pi(s),
$$

then the new policy $\pi'$ is guaranteed to be at least as good as the old policy $\pi$. So how do we find such a policy? The answer is simple: we can simply choose the action that maximizes the action-value function in each state so we get the **greedy policy** with respect to the action-value function:

$$
\pi'(s) = \arg\max_a q_\pi(s,a).
$$

However, we don't have the action-value function $q_\pi$ directly, but we can compute it from the value function $v_\pi$ using the Bellman equation for the action-value function:

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

* Combining iteration and improvement into one step
* When value iteration is preferred
* Convergence characteristics

## Tabular Reinforcement Learning

now we **don’t know** the MDP model (transition/reward functions).

Transition from planning to learning when the model is unknown.

* Online vs offline RL
* On-policy vs off-policy
  
Off-policy methods are therefore more sample-efficient than
on-policy

### Model-Based Methods

Tabular setting

Learning the transition probabilities and rewards (e.g., MLE/MAP)?
Do we then perform planning with the learned model (e.g., value iteration/policy iteration)?

* Exploration vs exploitation strategies

What is the epsilon greedy, softmax/boltzmann exploration in this context?

and then the optimism in the face of uncertainty (OFU) principle to encourage exploration which motivates algorithms like Rmax.

### Model-free Approaches

#### Monte Carlo Learning

model free 

we get trajectories by interacting with the environment.
and calculate average return to estimate value functions and then use policy iteration?

more useful to estimate action-value function Q(s,a) directly.

monte carlo evaluation

for effieciency we use an update rule for when we get a new sample.

constant-alpha monte carlo

monte carlo control

introduces epsilon-greedy policies for exploration.
initialize epsilon-soft policy?

can make a split to behavior policy and target policy to get off-policy learning.
importance sampling for off-policy monte carlo and with coverage?
target no longer epsilon-soft but behavior policy is for exploration. instead our target policy could be deterministic greedy policy.

#### SARSA

expected sarsa?
double sarsa?

#### Temporal-Difference Learning

Learning value functions from experience using bootstrapping.

Bootstrapping and TD(0)

* Difference between TD and Monte Carlo
* TD update rule
* TD as SGD on value approximation

#### Q-Learning

Central model-free control method.

* Off-policy TD control algorithm
* Update rule and intuition
* Convergence intuition

Exploration Revisited
* ε-greedy behavior policies
* Decaying ε
* Tradeoffs

## Deep Reinforcement Learning

### Model-Free

#### Deep Q-Networks (DQN)

Using neural networks as function approximators for Q-values.

linear functions or neural networks to approximate Q-values.

#### Double DQN

* Overestimation problem
* Double Q-learning correction

#### Policy Gradient Methods

Directly optimizing the policy.

* Difficulty of using Q-values in continuous action spaces
* Policy parameterization

Variance Reduction:
* Baselines
* Advantage functions

#### Reinforce Algorithm

* Monte Carlo policy gradient
* Derivation and limitations

#### Actor–Critic

Hybrid methods combining TD learning with policy gradients.

But also a version where we start with Q Learning rather then TD learning?

* Actor updates the policy
* Critic estimates value function
* Lower variance than pure policy gradients

#### Advantage Actor–Critic

Introduce Advantage function to reduce variance.

#### Proximal Policy Optimization (PPO)
A stable and practical policy gradient method.

* Avoiding destructive large policy updates
* Clipped surrogate objective
* Why PPO works well in practice

#### Trust Region Policy Optimization (TRPO)

This is the clipped version

### Maximum Entropy RL (MERL)

motivates exploration by augmenting the reward with an entropy term.

soft actor-critic (SAC)?

#### Deep Deterministic Policy Gradient (DDPG)

#### Stochastic Value Gradients (SVG)

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

