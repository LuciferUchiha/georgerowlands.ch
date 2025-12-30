---
title: RL Fundamentals
type: docs
weight: 3
---

Before diving into specific algorithms for solving MDPs, it is crucial to understand the fundamental distinctions in Reinforcement Learning. These concepts define how an agent interacts with the environment, how it learns, and what kind of data it uses. Specifically, we need to distinguish between **Planning vs. Learning**, **Model-Based vs. Model-Free**, **Online vs. Offline**, and **On-Policy vs. Off-Policy**.

## Planning vs. Learning

So far we have assumed that the agent follows some fixed policy $\pi$ and derived the corresponding value functions and Bellman equations. But how do we actually find a good policy in the first place? This is where the concepts of **[planning](/garden/ml/rl/planning)** and **reinforcement learning** come into play. 

**Planning** is the reasoning side of the equation. It assumes we know how the world works (the model) and we just need to figure out the best course of action. Think of it like playing Chess: you know the rules (transitions) and the goal (checkmate/reward), you just need to think ahead to find the optimal moves. Specifically, planning refers to the process of computing an optimal policy when the MDP model (transition and reward functions) is known, or in other words, when we have **complete knowledge** of the environment.

In contrast, **Reinforcement Learning** deals with the situation where the environment is unknown and must be learned by interacting with it. It is like learning to ride a bike. You don't know the physics equations (the model), you just try things, fall over (negative reward), and adjust your behavior. However, both planning and reinforcement learning share the same underlying goal of finding an optimal policy that maximizes the expected return.

This can be very confussing to differentiate, especially since we can also do planning in reinforcement learning settings, for example by learning a model of the environment and then using that model to plan. However, the key distinction is whether the MDP model is whether you have access to a model of the environment, and you try to solve it via some form of advanced search. It does not require to collect true experience from the real environment, but some planning methods are based on simulated experience from the known (or modeled) environment. It's all in the agent's head, just like when you plan something, hence planning.

## Model-Based vs. Model-Free

Learning is when you do not assume to have a model of the environment, and thus you need true experience to infer anything. And it can be done broadly speaking in two ways: 
- **Model-based learning**: Here, the agent tries to learn a model of the environment's dynamics (i.e., the transition probabilities and reward function) from its interactions with the environment. Once the model is learned, the agent can use planning methods to compute an optimal policy based on this learned model.
- **Model-free learning**: In this approach, the agent directly skips learning a model of the environment and instead focuses on learning the value functions or policies directly from its interactions with the environment. 

{{< figure 
    src="/images/ml/rlModelBasedFree.png"
    caption="Illustration of model-based and model-free reinforcement learning."
    alt="Illustration of model-based and model-free reinforcement learning."
>}}

## Online vs. Offline (Batch)

There are a few other key concepts that are important to understand in the context of planning and reinforcement learning. First, we usually have some data from the environment in the form of **experience tuples** $(s, a, r, s')$, which represent a transition from state $s$ to state $s'$ after taking action $a$ and receiving reward $r$. Based on this experience data, we now want to find an optimal policy. Depending on the setting, we have to differentiate between two main scenarios:
- **Offline (batch) learning**: In this setting, we have a fixed dataset of experience tuples collected from the environment, and we want to learn an optimal policy from this static dataset without further interaction with the environment. We also call it batch learning because we process the data in batches, where each batch comes after some period of data collection using a deployed policy.
- **Online learning**: Here, the agent can interact with the environment in real-time, collecting new experience tuples as it explores. The agent updates its policy and value functions continuously based on this ongoing interaction. In the online setting, we then of course have to deal with our exploration vs. exploitation trade-off, where the agent has to balance between exploring new actions to gather more information about the environment and exploiting its current knowledge to maximize rewards to achieve the best possible performance.

{{< figure 
    src="/images/ml/rlOnlineOffline.png"
    caption="Illustration of offline (batch) and online reinforcement learning."
    alt="Illustration of offline (batch) and online reinforcement learning."
>}}

## On-Policy vs. Off-Policy

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
