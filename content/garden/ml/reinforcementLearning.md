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

## **The Action-Value Function (q_\pi(s,a))**

Sometimes we want a more fine-grained understanding: not just “how good is state (s)?”, but “how good is it to take this particular action (a) in state (s)?”. This leads to the **action-value function**, defined as:

[
q_\pi(s,a)
= \mathbb{E}_\pi \left[ G_t \mid S_t = s, A_t = a \right].
]

This represents the expected return if the agent **first** takes action (a) in state (s), and then follows the policy (\pi) thereafter.

The distinction is simple but powerful:

* (v_\pi(s)) tells you how good a state is when following (\pi).
* (q_\pi(s,a)) tells you how good a specific action is in that state when following (\pi).

As we’ll later see, optimal behaviour can be extracted directly from (q_*(s,a)), the optimal action-value function.

---

## **Relationship Between (v_\pi) and (q_\pi)**

The state-value and action-value functions are intimately related. If in state (s) the policy chooses action (a) with probability (\pi(a \mid s)), then the value function is just the expectation of action-values under the policy:

[
v_\pi(s)
= \sum_{a \in \mathcal{A}(s)} \pi(a \mid s), q_\pi(s,a).
]

This simply says:
“the value of a state is the average value of the actions the policy takes there”.

This relationship is extremely important, because in many RL algorithms we estimate (q_\pi) directly (for example in Monte Carlo control or Q-learning), and then obtain (v_\pi) or greedy policies from it.

---

## **Value Functions as Long-Term Predictions**

To appreciate why these functions matter, remember that the agent cannot see into the future. The reward signal is delayed and noisy; a decision now might have consequences far later. Value functions compress the entire uncertain future into a **single numeric prediction**.

If:

* (v_\pi(s)) is high, the agent should try to reach state (s);
* if (q_\pi(s,a)) is high, the agent should try to choose action (a) in that state.

In this sense, value functions provide the agent with a **map of the future**: they tell it which states and which decisions are promising.

---

## **Example: Value in a Grid World**

Consider the grid world example. If the agent starts in some non-terminal state (s), then:

* (v_\pi(s)) will be large if the policy tends to reach high-reward teleportation states like (A) or (B);
* (v_\pi(s)) will be small or negative if the policy tends to wander into long paths with many (-1) penalties.

If we look at action-values, (q_\pi(s,a)) distinguishes between the possible actions in state (s). For example, if “move down” leads directly into the teleportation cell (A), then:

* (q_\pi(s,\text{down})) will be large,
* while (q_\pi(s,\text{left})) might be small because it leads to a long sequence of penalties.

This fine-grained structure is essential for improving the policy later on.

---

## **Value Functions and Policy Improvement**

A remarkable consequence of these definitions is that knowing (q_\pi) is enough to produce a strictly better policy. The simplest improvement rule is the **greedy policy**:

[
\pi'(s) = \arg\max_a q_\pi(s,a).
]

This new policy (\pi') is guaranteed to be at least as good as (\pi), and often strictly better. This fact is the foundation of:

* policy improvement,
* policy iteration,
* value iteration,
* Q-learning,
* and modern actor–critic methods.

But to use these tools, we need a way to compute or approximate the value functions themselves. This brings us to the **Bellman equations**, which express the value functions in a recursive form that relates (v_\pi) and (q_\pi) to expectations of future values.

* State-value (V^\pi(s))
* Action-value (Q^\pi(s,a))
* Expected return and discounted expected return
* How value functions imply policies and vice versa

### Bellman Equations

* Deriving Bellman expectation equations for (V^\pi) and (Q^\pi)
* Bellman optimality equations
* Principle of optimality

### Planning

Assumes **complete knowledge** of transition and reward functions. So known MDP.

Dynamic Programming

## Policy Evaluation

* Solving for (V^\pi) exactly
* Iterative (fixed-point) policy evaluation
* Connection to dynamic programming
approximate policy evaluation?

## Policy Improvement

* How to derive a better policy from a value function
* Proof of policy improvement theorem

## Policy Iteration

* Alternating evaluation and improvement
* Convergence properties
* Pros and cons

## Value Iteration

* Combining iteration and improvement into one step
* When value iteration is preferred
* Convergence characteristics

## Reinforcement Learning

now we **don’t know** the MDP model (transition/reward functions).

Transition from planning to learning when the model is unknown.

* Episodic vs continual tasks
* Online vs offline RL

Model-Based RL:

* Learning the transition probabilities and rewards (e.g., MLE/MAP)
* Planning with a learned model
* The Rmax algorithm

Model-Free RL Overview:

* On-policy vs off-policy
* Exploration vs exploitation strategies
* ε-greedy and softmax exploration

### Monte Carlo Methods

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

### Temporal-Difference Learning

Learning value functions from experience using bootstrapping.

Bootstrapping and TD(0)

* Difference between TD and Monte Carlo
* TD update rule
* TD as SGD on value approximation

### Q-Learning

Central model-free control method.

* Off-policy TD control algorithm
* Update rule and intuition
* Convergence intuition

Exploration Revisited
* ε-greedy behavior policies
* Decaying ε
* Tradeoffs

### Deep Q-Networks (DQN)

Using neural networks as function approximators for Q-values.

linear functions or neural networks to approximate Q-values.

## Double DQN

* Overestimation problem
* Double Q-learning correction

### Policy Gradient Methods

Directly optimizing the policy.

* Difficulty of using Q-values in continuous action spaces
* Policy parameterization

### Variance Reduction

* Baselines
* Advantage functions

### REINFORCE

* Monte Carlo policy gradient
* Derivation and limitations

### Actor–Critic Methods

Hybrid methods combining TD learning with policy gradients.

* Actor updates the policy
* Critic estimates value function
* Lower variance than pure policy gradients

### Proximal Policy Optimization (PPO)

A stable and practical policy gradient method.

* Avoiding destructive large policy updates
* Clipped surrogate objective
* Why PPO works well in practice
