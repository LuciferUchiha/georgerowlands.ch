---
title: Learning from Preferences
type: docs
weight: 7
---

So far rewards been clear but not always the case as can be hard to specify what we want. 

relative ranking rather than absolute reward values. pairwise comparisons/preferences.

Instead we can learn from comparisons of different outcomes, e.g. "I prefer outcome A to outcome B". This is known as learning from preferences. We denote this preference as:

$$
\tau_i_1 \succ \tau_i_2
$$

where $\tau_i_1$ and $\tau_i_2$ are two different trajectories. Intuitively we want to learn a reward function $r_\phi$ which refelects these preferences. So we want:

$$
\tau_i_1 \succ \tau_i_2 \iff r_\phi(\tau_i_1) > r_\phi(\tau_i_2)
$$

where $r_\phi(\tau) = \sum_t r_\phi(s_t, a_t)$ is the cumulative reward of trajectory $\tau$ under reward function $r_\phi$. 

## LLMs as Agents

state is a sequence of tokens (prompt + generated tokens), action is next token to generate, reward could be 1 or 0 for verifiable tasks but what about more subjective tasks like helpfulness or safety? Here we can use human feedback to learn a reward model from comparisons of different outputs.

## Reinforcement Learning from Human Feedback (RLHF)

Aligning or Steering LLMs using human feedback?

use PPO to change the LLM's policy to optimize for the learned reward model.

Feedback is only as good as the reward model, so need to be careful about how we collect data and train the reward model.

what is the issue with just pretraining LLMs, why do we need RL? Helpfulness, safety, etc. InstructGPT the original paper on RLHF for ChatGPT.

Pretraining to get base model. Supervised fine-tuning (SFT) of collection queries and responses. RLHF to further improve helpfulness and safety. 

Is RLHF really judt Bradley-Terry? What is the precise setup of RLHF, e.g. how is the data collected, what is the full training procedure, what is trained when, etc.

Reinforcement Learning from Human Feedback (RLHF) is a technique that trains a reward model directly from human feedback and then uses this model to optimize the policy using reinforcement learning. This approach has been successfully applied to train large language models (LLMs) like ChatGPT to generate more helpful and safe responses.

A common approach is to use a Bradley-Terry model to model the probability of preferring one trajectory over another:

$$
P(\tau_i_1 \succ \tau_i_2) = \frac{exp(r_\phi(\tau_i_1))}{exp(r_\phi(\tau_i_1)) + exp(r_\phi(\tau_i_2))} = \frac{1}{1 + exp(-(r_\phi(\tau_i_1) - r_\phi(\tau_i_2)))}
$$

$$
\mathcal{L}(\phi) = \sum_{\tau_i_1, tau_i_2 \in D} \mu(1)\log P(\tau_i_1 \succ \tau_i_2) + \mu(2)\log P(\tau_i_2 \succ \tau_i_1)
$$

what is meaning of mu and the above loss? where does it come from?

reward gaming? We add a KL penalty to keep the new policy close to the original pre-trained policy:

$$
r_\phi = r_\phi - \beta KL(\pi_\theta(\tau) || \pi_{old}(\tau))
$$

old is pre-trained + fine-tuned with supervised learning

## Direct Preference Optimization (DPO)

This is only in the script.

why use RL at all? Can we directly optimize the policy from preferences without learning a reward model? rather then generating a reward model and then using RL to optimize the policy, we can directly optimize the policy to maximize the likelihood of preferred trajectories. So we get the probability of the 2 trajectories under the policy and just increase the likelihood of the preferred one and decrease the likelihood of the other one. This is known as Direct Preference Optimization (DPO).

$$
L_{DPO}(\theta) = \mathbb{E}_{x, y_1, y_2 \sim D} [\log \sigma(\beta (\log \frac{\pi_\theta(y_1|x)}{\pi_{ref}(y_1|x)} - \log \frac{\pi_\theta(y_2|x)}{\pi_{ref}(y_2|x)}))]
$$

where $\pi_{ref}$ is a reference policy (e.g. the pre-trained LLM) and $\beta$ is a scaling parameter and $y_1$ is preferred to $y_2$. so we want the left term and the right term to decrease. rewrite as KL divergence between the new policy and the reference policy weighted by the preference?

### Group Related Policy Optimization (GRPO)

used by deepseek is this the thinking stuff?