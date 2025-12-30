---
title: Planning
type: docs
weight: 4
---

When the environment's dynamics (transition probabilities and reward function) are fully known, we can use **Planning** algorithms to solve for the optimal policy. Planning is the general problem of finding a policy given a model. **Dynamic Programming (DP)** is a specific collection of algorithms that can be used to solve this planning problem efficiently. By breaking down the complex problem of finding an optimal policy into simpler subproblems (finding the value of a state), DP allows us to compute optimal policies without brute-force search. DP algorithms exploit the recursive structure of the Bellman equations to iteratively compute value functions and improve policies. The two main algorithms in this category are **Policy Iteration** and **Value Iteration**.

## Policy Evaluation

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

## Policy Improvement

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

## Policy Iteration

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

## Value Iteration

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
