---
title: Markov Chains & Decision Processes
type: docs
weight: 12
---

Markov Chains are a key concept in probability theory and has many applications in computer science and machine learning. For example the PageRank algorithm used by Google Search to rank web pages is based on a Markov Chain model of web surfing behavior or Diffusion models also use Markov Chains to model the process of gradually transforming random noise into coherent images.

The same goes for Markov Decision Processes, which extend Markov Chains to include the possibility of making decisions at each state, allowing for the modeling of sequential decision-making problems and potential rewards. These decision processes form the basis of many reinforcement learning algorithms, where an agent learns to make optimal decisions by interacting with an environment.

## Markov Chains

First let's start with defining what a stochastic process is. A deterministic process is a process where we start from a given state $s_0$ and can predict with certainty the next state $s_1$ and all future states $s_2, s_3, ...$ based on some rules. We can think of such a process as a function $f$ that maps the current state to the next state and a specific initial state $s_0$:

$$
s_{n+1} = f(s_n) \quad \text{with} \quad s_0 \text{ given}
$$

The idea of such a process is also briefly discussed when discussing the movement of particles in physics as an e

On the other hand, a stochastic process is a process where the next state is not determined solely by the current state, but also involves some randomness. In this case, we can think of the process as a function $f$ that maps the current state to a probability distribution over possible next states:

### Homogeneous Markov Chains

### Regular Markov Chains

## Markov Decision Processes

### Partially Observable