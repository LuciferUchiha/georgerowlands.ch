---
title: Learning from Preferences
type: docs
weight: 7
---

In previous sections, we assumed access to a well-defined reward function $r(s, a)$ that the agent maximizes. However, in many real-world applications, specifying such a reward function is difficult. For example, in tasks like summarizing text, writing code, or generating art, it is hard to write a scalar reward function that captures "quality" or "helpfulness".

Instead of absolute reward values, it is often easier for humans to provide **relative preferences**. Given two trajectories or outcomes $\tau_1$ and $\tau_2$, a human can indicate which one they prefer, denoted as $\tau_1 \succ \tau_2$. **Learning from Preferences** involves using these pairwise comparisons to learn a policy that generates preferred outcomes.

## LLMs as Agents

We can model the generation process of a Large Language Model (LLM) as a reinforcement learning problem:
- **State ($s_t$)**: The current context window, which includes the original prompt and all tokens generated so far.
- **Action ($a_t$)**: The next token chosen from the vocabulary.
- **Policy ($\pi_\theta(a_t|s_t)$)**: The LLM itself, parameterized by weights $\theta$, which outputs a probability distribution over the next token.
- **Reward ($r$)**: A scalar value indicating the quality of the completed sequence (usually given only at the end of the episode). In the case of verifiable tasks, this could be a binary reward (1 for correct, 0 for incorrect). For subjective tasks, we run into the challenge of defining a reward function and instead rely on human preferences.

## Bradley-Terry Model

To learn from preferences, we typically assume that there exists an underlying (latent) reward function $r_\phi(\tau)$ parameterized by $\phi$, such that the probability of preferring $\tau_1$ over $\tau_2$ depends on the difference in their rewards. A common choice is the **Bradley-Terry (BT) model**, which models this probability using the sigmoid function:

$$
P(\tau_1 \succ \tau_2) = \sigma(r_\phi(\tau_1) - r_\phi(\tau_2)) = \frac{1}{1 + \exp(-(r_\phi(\tau_1) - r_\phi(\tau_2)))}
$$

where $\sigma(x) = \frac{1}{1 + e^{-x}}$ is the logistic sigmoid function. Intuitively, if $r_\phi(\tau_1) \gg r_\phi(\tau_2)$, the probability approaches 1. If the rewards are equal, the probability is 0.5 and if $r_\phi(\tau_1) \ll r_\phi(\tau_2)$, the probability approaches 0. This is similar to the softmax function used in multi-class classification but applied to pairwise comparisons.

## Reinforcement Learning from Human Feedback (RLHF)

**Reinforcement Learning from Human Feedback (RLHF)** is a prominent framework that uses this concept to align LLMs with human intent and preferences. The key idea is to train a reward model from human preference data and then use this reward model to fine-tune the LLM via reinforcement learning. This is a widely used approach in training models like InstructGPT and ChatGPT.

You might ask: "Why aren't the pre-trained base model or a Supervised Fine-Tuned (SFT) model enough?" There are two main reasons:

Base Models are Trained to predict the next token on internet text. They are excellent simulators but may simulate incorrect or harmful behaviors (e.g., hallucinating facts because they look plausible, or generating toxic content because it exists in the training data). Even if we use supervised fine-tuning (SFT) on high-quality datasets where we aim to mimic high-quality demonstrations (prompts and desired responses). The models only learn to copy the form of the answer. If a user asks for a summary, SFT might produce a summary that looks like a summary but misses the key point. Furthermore, for subjective tasks like "write a funny joke" or "be helpful", it is often easier for humans to recognize a good result (reward) than to demonstrate it perfectly (SFT). RL allows the model to explore and find strategies that maximize the reward, potentially exceeding the quality of the demonstrations.

The standard RLHF pipeline consists of three main steps:

1.  **Supervised Fine-Tuning (SFT)**: We start with a pre-trained base model (e.g., GPT-3) and fine-tune it on a dataset of high-quality demonstrations (prompts and desired responses) using standard maximum likelihood estimation. This results in a reference policy $\pi_{ref}$ (or $\pi_{SFT}$) that produces coherent and relevant text.

2.  **Reward Modeling**: Then we train a reward model $r_\phi(x, y)$ to predict human preferences. We sample prompts $x$ and generate pairs of responses $(y_1, y_2)$ using the SFT model. Human labelers rank these pairs ($y_w \succ y_l$). The reward model is then trained to maximize the likelihood of these preferences under the Bradley-Terry model:
   
$$
\mathcal{L}_{RM}(\phi) = - \mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma(r_\phi(x, y_w) - r_\phi(x, y_l)) \right]
$$

3.  **RL Fine-Tuning**: Lastly we treat the SFT model as a policy $\pi_\theta$ and optimize it to maximize the learned reward $r_\phi$. The objective is to maximize the expected reward while keeping the policy close to the reference policy $\pi_{ref}$ to prevent "reward hacking" (exploiting the reward model's flaws) and maintain linguistic coherence. The objective is then:
   
$$
J(\theta) = \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_\theta(\cdot|x)} \left[ r_\phi(x, y) - \beta \log \frac{\pi_\theta(y|x)}{\pi_{ref}(y|x)} \right]
$$

Here, the term $\log \frac{\pi_\theta(y|x)}{\pi_{ref}(y|x)}$ is the KL-divergence between the current policy and the reference policy and $\beta$ controls the strength of this penalty. In the original RLHF work, Proximal Policy Optimization (PPO) is used to optimize this objective. 

## Direct Preference Optimization (DPO)

The RLHF pipeline is complex and unstable. It requires training a separate reward model and then running an expensive RL loop (PPO) which involves sampling trajectories and computing gradients through the policy. **Direct Preference Optimization (DPO)** asks: *Why do we need to treat this as an RL problem at all?* Can we optimize the policy directly from preferences without an explicit reward model?

DPO starts by analyzing the analytical solution to the RLHF objective. The objective $J(\theta)$ (reward maximization with KL penalty) can be rewritten as:

$$
\max_{\pi} \mathbb{E}_{x} \left[ \mathbb{E}_{y \sim \pi(\cdot|x)} [r(x, y)] - \beta D_{KL}(\pi(\cdot|x) || \pi_{ref}(\cdot|x)) \right]
$$

It can be shown (using the method of Lagrange multipliers or by recognizing the form of the Gibbs distribution) that the optimal policy $\pi^*$ for this objective is:

$$
\pi^*(y|x) = \frac{1}{Z(x)} \pi_{ref}(y|x) \exp\left( \frac{r(x, y)}{\beta} \right)
$$

where $Z(x) = \sum_y \pi_{ref}(y|x) \exp\left( \frac{r(x, y)}{\beta} \right)$ is the **partition function**. $Z(x)$ acts as a normalization constant to ensure the probabilities sum to 1. Calculating $Z(x)$ is intractable because it requires summing over all possible output sequences $y$ (which is exponentially large).

However, we can rearrange the equation to express the reward function in terms of the optimal policy and the reference policy:

$$
r(x, y) = \beta \log \frac{\pi^*(y|x)}{\pi_{ref}(y|x)} + \beta \log Z(x)
$$

Now, we substitute this expression for $r(x, y)$ into the Bradley-Terry preference model. For a pair $y_w \succ y_l$:

$$
\begin{align*}
P(y_w \succ y_l) &= \sigma(r(x, y_w) - r(x, y_l)) \\
&= \sigma\left( \left( \beta \log \frac{\pi^*(y_w|x)}{\pi_{ref}(y_w|x)} + \beta \log Z(x) \right) - \left( \beta \log \frac{\pi^*(y_l|x)}{\pi_{ref}(y_l|x)} + \beta \log Z(x) \right) \right) \\
&= \sigma\left( \beta \log \frac{\pi^*(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log \frac{\pi^*(y_l|x)}{\pi_{ref}(y_l|x)} \right)
\end{align*}
$$

Crucially, the intractable partition function $Z(x)$ cancels out because it depends only on the prompt $x$, not on the response $y$. This allows us to define a loss function directly on the policy $\pi_\theta$ (which we treat as our approximation of $\pi^*$) without needing the ground truth reward $r$ or the partition function.

The DPO loss is then simply the negative log-likelihood of the preference data under this derived probability model:

$$
\mathcal{L}_{DPO}(\theta) = - \mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)} \right) \right]
$$

This loss increases the likelihood of the preferred response $y_w$ and decreases the likelihood of the dispreferred response $y_l$, weighted by the implicit reward defined by the KL constraint.

## Group Related Policy Optimization (GRPO)