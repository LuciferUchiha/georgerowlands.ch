---
title: Reinforcement Learning
type: docs
weight: 24
---

When learning from data, we are often used to the supervised learning setting where the world appears calm and well behaved: someone has already collected a dataset of input–output pairs, and our only task is to learn a mapping that imitates these known answers. But this picture of learning is deeply limited as we only have a snapshot of a static world. On the other hand, the idea of reinforcement learning is to **learn through interaction rather than instruction**. Instead of being told what the correct answer is, an agent has to discover good behaviour by "fucking around and finding out", i.e performing actions and observing the consequences, and gradually building an understanding of how its actions shape the future. In this sense, reinforcement learning is closer to the way animals and humans learn by acting in the world and learning from experience.

For example imagine for a moment that you are trying to learn to ride a bicycle. Nobody hands you a table of “situations” paired with “correct actions”. Instead, you climb onto the bike, wobble, fall, climb again, and gradually figure out which movements keep you upright. The feedback the world gives you is not a label like “correct steering angle is 17 degrees”, but something far more primitive and indirect: you remain balanced (good), or you fall (bad). Reinforcement learning formalizes exactly these sorts of learning signal. Instead of labelled answers, the agent (you) receives **rewards**. Rewards do not tell you what the right action was but simply indicate how good or bad the outcome was. Your goal is then to learn to take actions that maximize the total reward it receives over time.

Another key difference from supervised learning is that in reinforcement learning, **the data the agent sees is no longer independent or identically distributed**, because each action the agent takes changes the world in which future data will be generated. If you steer left on a bicycle now, the physical situation you will be in a second later depends on that choice. If you make a poor move in a game of chess, the next positions you see will reflect the consequences of that mistake. A move in Chess might seem normal now but lead to a checkmate twenty turns later. This is known as the **credit assignment problem**, determining which past action is responsible for the current reward. As the agent improves and starts behaving differently, it begins exploring different parts of the environment, which means it will see different rewards, different transitions, and different future possibilities and tries to exploit what it has learned so far to maximize its cumulative reward. This tight coupling between behaviour and data is exactly what makes reinforcement learning powerful as by **experiencing/sampling trajectories of interaction**, the agent gradually uncovers the structure of the task. But it also introduces new challenges. Because actions influence what the agent sees later, it must constantly balance two competing objectives: Trying things it already believes are good, and experimenting with actions that might lead to better strategies it has not yet discovered. This is the so called **exploration–exploitation dilemma**, which lies at the heart of reinforcement learning.

Although these ideas may seem abstract at first and were initially also heavily doubted by the research community, they underpin some of the most impressive achievements of modern AI. Systems like **AlphaZero** and **AlphaGo** by Google Deepmind were not trained on datasets of "correct" moves. Instead, they learned by interacting with the environment by playing games, receiving sparse signals such as win or lose, and gradually learning strategies. These models began with no human knowledge, no opening book, no endgame heuristics, only the rules of each game. By repeatedly playing against other oponent or against itself and learning from the outcomes, it discovered tactics, structures, and strategic principles that took human experts centuries to develop and even surpassed the best human players. 

What Makes RL Different?

## Markov Decision Processes

To turn our intuitive picture of reinforcement learning into a precise mathematical framework, we need a way to describe an agent interacting with an environment over time. For this purpose, we extend the [Markov Chain framework](). If a Markov Chain is a stochastic process over states, a Markov Decision Process (MDP) is a stochastic process over states where the transitions are controlled by actions and motivated by rewards. Many ordinary systems fit naturally into this form: a robot navigating a room, an algorithm playing a game, an elevator responding to button presses, or even a user browsing through a website. 

{{< figure 
    src="rlMDPCycle.png" 
    caption="An agent interacts with an environment in discrete time steps. At each time step t, the agent observes the current state s_t, selects an action a_t, and receives a reward r_t from the environment. The environment then transitions to a new state s_{t+1} based on the action taken."
    alt="An agent interacts with an environment in discrete time steps. At each time step t, the agent observes the current state s_t, selects an action a_t, and receives a reward r_t from the environment. The environment then transitions to a new state s_{t+1} based on the action taken."
    width="600"
>}}

We formally define an MDP as a tuple:

$$
(\mathcal{S}, \mathcal{A}, P, R, p_0, \gamma)
$$

where:
- $\mathcal{S}$ is the set of possible states the environment can be in, also called the **state space**. States can be discrete (e.g., positions on a grid) or continuous (e.g., the position and velocity of a robot). We often also split the state space into **terminal states** $\mathcal{S}_T$ (e.g., game over) and **non-terminal states** $\mathcal{S}_{NT}$ (e.g., ongoing game) and define $\mathcal{S} = \mathcal{S}_T \cup \mathcal{S}_{NT}$.
- $\mathcal{A}(s)$ is the set of actions the agent can take in state $s$, the union over all the states gives the full **action space** $\mathcal{A} = \bigcup_{s \in \mathcal{S}} \mathcal{A}(s)$. Actions can also be discrete (e.g., move left, right, up, down) or continuous (e.g., steering angle, acceleration).
- $\mathbb{P}(s'|s,a)$ is the **transition probability function**, giving the probability of moving to state $s'$ when action $a$ is taken in state $s$.
- $r(s,a)$ is the **reward function**, specifying the expected reward received after taking action $a in state $s$.
- $p_0(s)$ is the **initial state distribution**, defining the probability of starting in each state at time step $t=0$.
- $\gamma \in [0,1]$ is the **discount factor**, which determines the importance of future rewards.

The process of the MDP can then be seen as a stochastic process over **discrete time steps** $t = 0, 1, 2, ...$, where we start at time step $t=0$ by sampling an initial state $s_0$ from the initial state distribution $p_0(s)$. Then, at each time step the environment is in some state $s_t \in \mathcal{S}$. **The agent observes this state and selects an action** $a_t \in \mathcal{A}$. The set of actions available may depend on the current state, i.e., $\mathcal{A}(s_t)$, for example think of a chess game where a piece can not leave the board. After taking action $a_t$, **the agent receives a reward** based on this state-action pair, $r_t = r(s_t, a_t)$ (the reward function can also depend on the next state, i.e., $r(s,a,s')$, but usually the simpler form is used), and **the environment transitions to a new state** $s_{t+1}$ according to the transition probabilities $P(s_{t+1}|s_t, a_t)$. This process then repeats at the next time step. Notice that just like Markov Chains, MDPs also satisfy the **Markov property**: the future state $s_{t+1}$ depends only on the current state $s_t$ and action $a_t$, not on any previous states or actions:

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
\tau = (s_0, a_0, r_0, s_1, a_1, r_1, ..., s_T)
$$

The goal of the agent is to maximize the total amount of reward it receives. However, we must consider the time horizon.

* Policies

Horizons and Returns:

* Finite vs infinite horizon
* Discounted vs undiscounted return
* Reward vs return vs reward-to-go
* Reward shaping

### Value Functions

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
