---
title: Expectation, Variance and Covariance
type: docs
weight: 5
---

Probability theory doesn't just let us describe which outcomes are possible but we often want to know how a random variable behaves on average. Does it tend to take large values, small values, or something in between? Does it fluctuate wildly, or stick close to a central point? To make these questions precise, we introduce key summary quantities: **expectation**, **variance**, and **covariance**.

## Expectation

The **expectation** (also called the **mean** or **expected value**) of a random variable is a measure of the central tendency of its distribution. Formally, it is the **weighted average** of all possible values that the random variable can take, with weights given by their respective probabilities.

In other words, the expectation tells us the **long-term average** value of the random variable if we were to repeat the random experiment an infinite number of times. It answers: "On average, what do we expect to observe?"

### Discrete Random Variables

Suppose $X: \Omega \to \mathbb{R}$ is a discrete random variable with values in $W$ (finite or countable). The expectation of $X$ is defined as

$$
\mathbb{E}[X] = \sum_{x \in W} x \cdot \mathbb{P}(X = x)
$$

where $W$ is the set of all values $X$ can take, and $\mathbb{P}(X = x)$ is the probability that $X$ takes value $x$. 

This formula is essentially a weighted average: we take each possible value $x$, multiply it by its chance of occurring, and sum over all possibilities. This captures what you would get if you repeated the experiment many times and averaged the results.

Alternatively, you may see the expectation written as a sum over all outcomes $\omega \in \Omega$ in the sample space:

$$
\mathbb{E}[X] = \sum_{\omega \in \Omega} X(\omega) \cdot \mathbb{P}(\{\omega\})
$$

Here, $X(\omega)$ is the value of the random variable in outcome $\omega$, and $\mathbb{P}({\omega})$ is the probability of outcome $\omega$.

This definition captures the idea of "average value," where each possible outcome is weighted by its likelihood. So if we perform the random experiment a large number of times, the arithmetic mean of the observed values will be close to $\mathbb{E}[X]$. In particular, the law of large numbers tells us that as the number of trials increases, the sample average converges to the expectation.

{{< callout type="example" title="Fair Die" >}}
Let $X$ be the outcome when rolling a fair six-sided die. Then $W = \{1,2,3,4,5,6\}$ and $\mathbb{P}(X = x) = \frac{1}{6}$ for all $x$. The expectation is:

$$
\mathbb{E}[X] = \sum_{x=1}^{6} x \cdot \frac{1}{6} = \frac{1}{6}(1 + 2 + 3 + 4 + 5 + 6) = 3.5
$$

So, the "average" roll of a fair die is $3.5$. Of course, $3.5$ is not a possible outcome of a single roll—the expectation is an average, not a possible value! Or in other terms if we kept track of all of our dice rolls whilst playing a game, the average of all rolls would converge to $3.5$ as the number of rolls increases (for example when playing monopoly which takes a long time).
{{< /callout >}}

{{< callout type="example" title="Gambling Game" >}}
Suppose you win or lose depending on the die outcome, with winnings $X(\omega)$ as follows:

$$
\forall \omega \in \Omega: X(\omega) = \begin{cases}
-1 & \text{ if } \omega = 1, 2, 3 \\
0 & \text{ if } \omega = 4 \\
2 & \text{ if } \omega = 5, 6
\end{cases}
$$

Each outcome has probability $\frac{1}{6}$, so

$$
\mathbb{E}[X] = (-1) \cdot \frac{3}{6} + 0 \cdot \frac{1}{6} + 2 \cdot \frac{2}{6} = -\frac{3}{6} + 0 + \frac{4}{6} = \frac{1}{6}
$$

So, on average, you expect to win $\frac{1}{6}$ per game in the long run. So in other words, if you had enough cash and played this game many times, you would expect to end up with a profit over time.
{{< /callout >}}

{{< callout type="example" title="Indicator Function" >}}
Let $A \subseteq \Omega$ be an event, and let $I_A$ be the **indicator random variable** of $A$:

$$
I_A(\omega) = \begin{cases}
1 & \text{ if } \omega \in A \\
0 & \text{ otherwise}
\end{cases}
$$

Then,

$$
\mathbb{E}[I_A] = 1 \cdot \mathbb{P}(A) + 0 \cdot \mathbb{P}(\Omega \setminus A) = \mathbb{P}(A)
$$

So, the expectation of an indicator random variable is just the probability of the event. 
{{< /callout >}}

{{< callout type="example" title="Bernoulli Distribution" >}}
If $X \sim \text{Bernoulli}(p)$, so $X$ takes value $1$ with probability $p$ (success) and $0$ with probability $1-p$ (failure):

$$
\mathbb{E}[X] = 1 \cdot p + 0 \cdot (1-p) = p
$$

We could also interpret a Bernoulli random variable as an indicator variable where $A$ is the event of success. Then $\mathbb{E}[I_A] = \mathbb{P}(A) = p$.
{{< /callout >}}

{{< callout type="example" title="Binomial Distribution" >}}
If $X \sim \mathrm{Binomial}(n, p)$, so $X$ counts the number of successes in $n$ independent Bernoulli($p$) trials:

$$
\mathbb{E}[X] = \sum_{k=0}^n k \cdot \binom{n}{k} p^k (1-p)^{n-k} = n p
$$

So we iterate over all possible values $k$ from $0$ to $n$, representing the number of successes, and multiply each value by its probability $\binom{n}{k} p^k (1-p)^{n-k}$, which is the probability of getting exactly $k$ successes in $n$ trials. The sum gives us the expected number of successes.
{{< /callout >}}

{{< callout type="example" title="Poisson Distribution" >}}
If $X \sim \mathrm{Poisson}(\lambda)$, then

$$
\mathbb{E}[X] = \sum_{k=0}^{\infty} k \cdot \frac{\lambda^k e^{-\lambda}}{k!} = \lambda
$$

This follows from the definition of the Poisson distribution, where $\lambda$ is the average rate of occurrence. The expectation is simply $\lambda$, which represents the average number of events in a fixed interval.
{{< /callout >}}

### Continuous Random Variables

Let's now consider continuous random variables. The intuition remains the same: the expectation is a weighted average, but now with weights given by a probability density function (PDF) $f(x)$. Here, sums are replaced by integrals as we are no longer just summing over discrete outcomes but over a continuous range of values which is why we need to use integrals.

Suppose $X$ is a continuous random variable with PDF $f(x)$ and $X \geq 0$ almost surely (i.e $\mathbb{P}(X < 0) = 0$). Then the expectation is

$$
\mathbb{E}[X] = \int_0^\infty x f(x) \, dx
$$

where $f(x) \geq 0$, $\int_0^\infty f(x) dx = 1$. The reason we say almost surely is that in continuous random variables, the probability of any single point (like $X = 0$) is zero, so we can ignore it without affecting the integral. This definition reflects the same logic as in the discrete case: we weight each possible value $x$ by its likelihood $f(x),dx$ and sum (integrate) over all possible $x$.

For nonnegative random variables, the expectation may be finite or infinite depending on the behavior of the PDF. However, we say it is well defined if the integral converges to a finite value.

It also follows that if $X$ is a non-negative continuous random variable, then

$$
\mathbb{E}[X] \geq 0
$$

Due to the value of $x$ being non-negative and the PDF $f(x)$ being non-negative, the integral will always yield a non-negative result. The equality holds if and only if $f(x) = 0$ for all $x < 0$, meaning $X = 0$ almost surely (i.e., with probability 1). This is because the integral from $0$ to $\infty$ of a non-negative function is zero only if the function is zero almost everywhere in that range.

If $X$ is a non-negative continuous random variable with values in the range from $[a,b]$, then we don't need to integrate from $0$ to $\infty$, but rather from $a$ to $b$:

$$
\mathbb{E}[X] = \int_a^b x f(x) \, dx \text{ if } X(\omega) \in [a,b] \text{ for all } \omega \in \Omega
$$

This makes sense as for all values outside $[a,b]$, the PDF $f(x)$ is $0$, so they do not contribute to the integral.

{{< callout type="example" title="Uniform Distribution" >}}
Let $X \sim \text{Uniform}(0,1)$ so $f(x) = 1$ for $x \in [0,1]$, zero otherwise. 

$$
\mathbb{E}[X] = \int_0^1 x \cdot 1 \, dx = \left[ \frac{x^2}{2} \right]_0^1 = \frac{1^2}{2} - \frac{0^2}{2} = \frac{1}{2}
$$

So the average value of a random number in $[0,1]$ is $\frac{1}{2}$. Which is intuitive since the uniform distribution is symmetric around $\frac{1}{2}$.
{{< /callout >}}

{{< callout type="example" title="Exponential Distribution" >}}
Let $X \sim \mathrm{Exp}(\lambda)$, so $f(x) = \lambda e^{-\lambda x}$ for $x \geq 0$:

$$
\mathbb{E}[X] = \int_0^\infty x \lambda e^{-\lambda x} dx
$$

We integrate by parts with $u = x$, $dv = \lambda e^{-\lambda x} dx$, $du = dx$, $v = -e^{-\lambda x}$ then we have:

$$
\begin{align*}
\mathbb{E}[X] &= \int_0^\infty x \lambda e^{-\lambda x} dx \\
&= \left. -x e^{-\lambda x} \right|_0^\infty + \int_0^\infty e^{-\lambda x} dx \\
&= 0 + \int_0^\infty e^{-\lambda x} dx \\
&= \left[ -\frac{1}{\lambda} e^{-\lambda x} \right]_0^\infty \\
&= \frac{1}{\lambda}
\end{align*}
$$
{{< /callout >}}

If $X$ can take both positive and negative values, we need to ensure the expectation is well-defined. To do this, we decompose $X$ into its positive and negative parts:

$$
X_+(\omega) = \begin{cases}
X(\omega) & \text{if } X(\omega) \geq 0 \\
0 & \text{otherwise}
\end{cases}
\qquad
X_-(\omega) = \begin{cases}
-X(\omega) & \text{if } X(\omega) < 0 \\
0 & \text{otherwise}
\end{cases}
$$

We obviously have $X = X_+ - X_-$, but we also have $|X| = X_+ + X_-$ because the negative part $X_-$ is defined as the absolute value of $X$ when $X$ is negative. So if $\mathbb{E}[|X|] = \mathbb{E}[X_+] + \mathbb{E}[X_-] < \infty$, then we say the expectation is well-defined and set:

$$
\mathbb{E}[X] = \mathbb{E}[X_+] - \mathbb{E}[X_-]
$$

Because of the condition above that $\mathbb{E}[|X|] = \mathbb{E}[X_+] + \mathbb{E}[X_-] < \infty$ it means that both $\mathbb{E}[X_+]$ and $\mathbb{E}[X_-]$ are finite and well-defined. This is crucial because if either expectation diverges to infinity, the difference would be undefined (as $\infty - \infty$ is not a number). If this holds, $\mathbb{E}[X]$ is well-defined (finite). If $X \geq 0$ then the expectation is always defined, but can be infinite. So we can write

$$
\mathbb{E}[X_+] = \int_{-\infty}^{\infty} X_+(\omega) f(x) dx, \qquad \mathbb{E}[X_-] = \int_{-\infty}^{\infty} X_-(\omega) f(x) dx
$$

This means that we can compute the expectation of $X$ as the difference between the expected values of its positive and negative parts so putting it together we have:

$$
\begin{align*}
\mathbb{E}[X] &= \mathbb{E}[X_+] - \mathbb{E}[X_-] \\
&= \int_{-\infty}^{\infty} X_+(\omega) f(x) dx - \int_{-\infty}^{\infty} X_-(\omega) f(x) dx \\
&= \int_{-\infty}^{\infty} (X_+(\omega) - X_-(\omega)) f(x) dx \\
&= \int_{-\infty}^{\infty} X(\omega) f(x) dx 
\end{align*}
$$

{{< callout type="proof" >}}
Let's check that this matches our intuition:

$$
\mathbb{E}[X] = \int_{-\infty}^\infty X(\omega) f(x) dx
= \int_{-\infty}^0 X(\omega) f(x) dx + \int_0^\infty X(\omega) f(x) dx
$$

But for $x<0$, $X(\omega) = x < 0$, so $X_{-(\omega)} = -x > 0$, and for $x \geq 0$, $X_{+(\omega)} = x$. Therefore:

$$
\mathbb{E}[X_+] = \int_0^\infty x f(x) dx, \quad \mathbb{E}[X_-] = \int_{-\infty}^0 (-x) f(x) dx
$$

So

$$
\mathbb{E}[X] = \int_0^\infty x f(x) dx - \int_{-\infty}^0 (-x) f(x) dx = \int_{-\infty}^\infty x f(x) dx
$$

which is the standard formula for expectation, provided the integral is absolutely convergent. 
{{< /callout >}}

{{< callout type="example" >}}
Let $X \sim \text{Uniform}(-2,2)$, so $f(x) = \frac{1}{4}$ for $x \in [-2,2]$. Then

$$
\mathbb{E}[X] = \int_{-2}^2 x \cdot \frac{1}{4} dx = \frac{1}{4} \left[ \frac{x^2}{2} \right]_{-2}^2 = \frac{1}{4} \cdot 0 = 0
$$

So the average value of a uniformly distributed random variable on $[-2,2]$ is $0$. 
{{< /callout >}}

### Linearity of Expectation

One of the most powerful and useful properties of expectation is linearity. The expectation of a sum is the sum of the expectations even if the random variables are dependent! So we have:

$$
\mathbb{E}[\lambda \cdot X] = \lambda \cdot \mathbb{E}[X] \quad \text{and} \quad \mathbb{E}[X + Y] = \mathbb{E}[X] + \mathbb{E}[Y]
$$

Putting these together, we get the **linearity of expectation**:

$$
\mathbb{E}\left[\sum_{i=1}^n \lambda_i X_i\right] = \sum_{i=1}^n \lambda_i \mathbb{E}[X_i]
$$

{{< callout type="proof" >}}
Let's prove this for $n$ random variables by induction. We will focus on the case where the $X_i$ are discrete random variables, but the proof can be adapted for continuous random variables as well. First we show the base case for $n=1$:

$$
\mathbb{E}[\lambda_1 X_1] = \sum_{x_1} \lambda_1 x_1 \cdot \mathbb{P}(X_1 = x_1) = \lambda_1 \sum_{x_1} x_1 \cdot \mathbb{P}(X_1 = x_1) = \lambda_1 \mathbb{E}[X_1]
$$

For our **induction hypothesis** we assume it holds for some $k \geq 1$:

$$
\mathbb{E}\left[\sum_{i=1}^k \lambda_i X_i\right] = \sum_{i=1}^k \lambda_i \mathbb{E}[X_i]
$$

Now for our induction step, we need to show it holds for $n = k + 1$:

$$
\begin{align*}
\mathbb{E}\left[\sum_{i=1}^{k+1} \lambda_i X_i\right] &= \mathbb{E}\left[\sum_{i=1}^k \lambda_i X_i + \lambda_{k+1} X_{k+1}\right] \\
&= \mathbb{E}\left[\sum_{i=1}^k \lambda_i X_i\right] + \mathbb{E}[\lambda_{k+1} X_{k+1}] \\
&= \sum_{i=1}^k \lambda_i \mathbb{E}[X_i] + \lambda_{k+1} \mathbb{E}[X_{k+1}] \quad \text{(by induction hypothesis)} \\
&= \sum_{i=1}^{k+1} \lambda_i \mathbb{E}[X_i]
\end{align*}
$$
{{< /callout >}}

{{< callout type="example" title="Binomial Distribution by Linearity" >}}
Let $X = X_1 + X_2 + \cdots + X_n$ where $X_i$ are independent (or even just identically distributed) Bernoulli($p$) variables. By linearity we have:

$$
\mathbb{E}[X] = \sum_{i=1}^n \mathbb{E}[X_i] = np
$$

Which is much nicer then having to deal with the binomial formula directly! Notice we did not need independence for the sum. The result is true even if the $X_i$ are dependent! 
{{< /callout >}}

### Tailsum Formula

Sometimes, computing the expectation directly from the definition can be cumbersome, especially for discrete random variables with many possible values. The **tailsum formula** offers a different perspective that is not only often easier to compute in practice, but also gives an insightful interpretation of expectation.

Suppose $X$ is a non-negative discrete random variable then the **tailsum formula** says:

$$
\mathbb{E}[X] = \sum_{k=1}^\infty \mathbb{P}(X \geq k)
$$

The previous definition summed over all possible values $m$ of $X$ and weighted them by their probabilities. The tailsum formula, however, sums over all possible thresholds $k$, counting how many times $X$ is at least $k$.

{{< callout type="todo" >}}
Doesnt rly make sense to me why this is the same an intuitively?
{{< /callout >}}
{{< callout type="proof" >}}
Let $X$ be a non-negative integer-valued random variable. We start with the standard definition:

$$
\mathbb{E}[X] = \sum_{m=0}^\infty m \cdot \mathbb{P}(X = m)
$$

But $m$ can be rewritten as a sum:

$$
m = \sum_{k=1}^m 1
$$

Therefore,

$$
\mathbb{E}[X] = \sum_{m=0}^\infty \left( \sum_{k=1}^m 1 \right) \mathbb{P}(X = m)
= \sum_{m=0}^\infty \sum_{k=1}^m \mathbb{P}(X = m)
$$

Now if we swap the order of summation we get:

$$
= \sum_{k=1}^\infty \sum_{m=k}^\infty \mathbb{P}(X = m)
= \sum_{k=1}^\infty \mathbb{P}(X \geq k)
$$
{{< /callout >}}

{{< callout type="example" title="Geometric Distribution via Tailsum" >}}
Let $X \sim \mathrm{Geom}(p)$, so $\mathbb{P}(X \geq k) = (1-p)^{k-1}$ for $k \geq 1$.

$$
\mathbb{E}[X] = \sum_{k=1}^\infty (1-p)^{k-1}
$$

This is a geometric series with first term $1$ and common ratio $1-p$, so we can use the formula for the sum of an infinite geometric series:

$$
\sum_{k=0}^\infty r^k = \frac{1}{1-r} \text{ for } |r| < 1
$$

Thus,

$$
\mathbb{E}[X] = \sum_{k=1}^\infty (1-p)^{k-1} = \frac{1}{1 - (1-p)} = \frac{1}{p}
$$
{{< /callout >}}

If $X$ is a **non-negative continuous random variable** with probability density function $f(x)$, the tailsum formula takes the form of an integral involving the **survival function** (also called the "tail probability function"):

$$
\mathbb{E}[X] = \int_0^\infty \mathbb{P}(X \geq x) dx
$$

or, equivalently,

$$
\mathbb{E}[X] = \int_0^\infty (1 - F(x)) dx
$$

where $F(x) = \mathbb{P}(X \leq x)$ is the cumulative distribution function (CDF), and $1 - F(x)$ is the **survival function**. The intuition here is similar to the discrete case, but now we are integrating over a continuous range of values. For each value $x \geq 0$, the function $\mathbb{P}(X \geq x)$ measures the "probability mass" that remains above $x$. As $x$ increases, this probability decreases.By integrating this "tail" probability across all $x \geq 0$, we effectively add up all the infinitesimal slices of expected "remaining" value, yielding the overall average (expected) value of $X$.

{{< callout type="proof" >}}
To derive the tailsum formula for continuous random variables, we start from the definition of expectation:

$$
\mathbb{E}[X] = \int_0^\infty x f(x) dx
$$

But by integration by parts with $u = x$, $dv = f(x) dx$, we have $du = dx$, and $v = F(x)$ (the CDF). Then,

$$
\mathbb{E}[X] = \left[ x F(x) \right]_0^\infty - \int_0^\infty F(x) dx
$$

As $x \to \infty$, $F(x) \to 1$ (if $X$ is integrable and nonnegative), so the boundary term vanishes. The lower boundary at $x=0$ gives $0$. Therefore,

$$
\mathbb{E}[X] = \int_0^\infty (1 - F(x)) dx = \int_0^\infty \mathbb{P}(X \geq x) dx
$$
{{< /callout >}}

{{< callout type="example" title="Exponential Distribution via Tailsum" >}}
Let $X \sim \mathrm{Exp}(\lambda)$, so $\mathbb{P}(X \geq x) = e^{-\lambda x}$ for $x \geq 0$.

$$
\mathbb{E}[X] = \int_0^\infty e^{-\lambda x} dx = \left[ -\frac{1}{\lambda} e^{-\lambda x} \right]_0^\infty = 0 - \left( -\frac{1}{\lambda} \right) = \frac{1}{\lambda}
$$

So, the expectation of an exponential random variable with parameter $\lambda$ is $\frac{1}{\lambda}$. 
{{< /callout >}}

### Characterization of Distributions

When we study random variables, it is tempting to summarize their behavior with a single number such as their expectation. However, the expectation alone does not uniquely determine the distribution of a random variable. That is, many very different random variables can have the same mean, but differ in almost every other respect (such as their spread, skewness, probability of extreme values, and so on).

Think of expectation as the "center of mass" of a probability distribution, but not its shape or spread. Two random variables can have the same mean, but their outcomes may be clustered tightly around that mean (low variance), or spread very far apart (high variance), or distributed in totally different ways.
Expectation does **not** uniquely characterize a distribution. Different random variables can have the same mean but very different shapes (e.g., uniform and normal distributions with the same mean).

The same idea applies to any transformation of the random variable i.e. applying a function $g:\mathbb{R} \to \mathbb{R}$ to $X$. The value $\mathbb{E}[g(X)]$ is called a moment (when $g(x) = x^k$), and can capture more information about the distribution:

$$
\mathbb{E}[g(X)] = \int_{-\infty}^{\infty} g(x) f(x) dx \quad \text{or} \quad \mathbb{E}[g(X)] = \sum_{x \in W} g(x) \cdot \mathbb{P}(X = x)
$$

This follows from the definition of expectation, where we replace $X$ with $g(X)$, effectively transforming the random variable before computing its expected value. The expectation of a transformed random variable is the average value of the function $g$ applied to the outcomes of $X$, weighted by their probabilities.

However, even knowing all moments $\mathbb{E}[X], \mathbb{E}[X^2], \mathbb{E}[X^3], \ldots$ is not always enough to fully determine the distribution. But in many practical cases, the sequence of all moments uniquely characterizes the distribution, while the mean alone never does.

{{< callout type="example" >}}
Suppose $X$ is a random variable representing the outcome of rolling a fair six-sided die. The expected value of $X$ is $\mathbb{E}[X] = 3.5$. Now, consider the transformation $g(X) = 2X + 1$. The expected value of the transformed random variable is:

$$
\mathbb{E}[g(X)] = \mathbb{E}[2X + 1] = 2\mathbb{E}[X] + 1 = 2 \cdot 3.5 + 1 = 7 + 1 = 8
$$

We can also compute this directly:

$$
\begin{align*}
\mathbb{E}[g(X)] &= \sum_{x=1}^{6} (2x + 1) \cdot \mathbb{P}(X = x) \\
&= \sum_{x=1}^{6} (2x + 1) \cdot \frac{1}{6} \\
&= \frac{1}{6} \left(2(1 + 2 + 3 + 4 + 5 + 6) + 6 \right) \\
&= \frac{1}{6} \left(2 \cdot 21 + 6 \right) \\
&= \frac{1}{6} \cdot 48 = 8
\end{align*}
$$
{{< /callout >}}

### Moments

The $n$-th moment of a random variable $X$ is defined as:

$$
\mathbb{E}[X^n] = \int_{-\infty}^{\infty} x^n f(x) dx \quad \text{(for continuous RVs)}
$$

{{< callout type="todo" >}}
Interpretations and meanings
{{< /callout >}}

### Product of Expectations

When dealing with random variables we have seen that they are linear and can be transformed. It is then natural to ask "what is the expected value of their product?" In general, this can be tricky, because $X$ and $Y$ may interact in complicated ways. However, if $X$ and $Y$ are independent, then their product behaves as simply as possible:

$$
\mathbb{E}[XY] = \mathbb{E}[X] \cdot \mathbb{E}[Y]
$$

Because the random variables are independent it means that knowing the value of $X$ tells you nothing about $Y$, and vice versa. So, the "average value" of $XY$ is just the average of $X$ times the average of $Y$—no unexpected "cross-terms" arise. You can think of this as the "The expected value of $XY$ is what you'd get if you independently draw $X$ and $Y$ many times, multiply each pair, and then average all those products."

Importantly, this property only holds under independence. If $X$ and $Y$ are dependent, then the expectation of their product can be different from the product of their expectations. It is also crucial to note that the converse does not hold:

$$
\mathbb{E}[XY] = \mathbb{E}[X] \cdot \mathbb{E}[Y] \text{ does not imply independence! }
$$

The equality can happen by coincidence, even when the random variables are not independent.

{{< callout type="proof" >}}
Suppose $X$ and $Y$ are discrete random variables, with joint probability mass function $p_{X,Y}(x, y)$. The expectation of their product is

$$
\mathbb{E}[XY] = \sum_{x} \sum_{y} x y \cdot p_{X,Y}(x, y)
$$

If $X$ and $Y$ are independent, then $p_{X,Y}(x, y) = p_X(x) p_Y(y)$, so

$$
\begin{align*}
\mathbb{E}[XY] &= \sum_{x} \sum_{y} x y \cdot p_X(x) p_Y(y) \\
&= \left(\sum_{x} x p_X(x)\right) \left(\sum_{y} y p_Y(y)\right) \\
&= \mathbb{E}[X] \cdot \mathbb{E}[Y]
\end{align*}
$$

We can also extend this to the continuous case.

Suppose $X$ and $Y$ are independent continuous random variables, with joint probability density function $f_{X,Y}(x, y) = f_X(x) f_Y(y)$.

$$
\begin{align*}
\mathbb{E}[XY] &= \int_{-\infty}^{\infty} \int_{-\infty}^{\infty} x y \, f_{X,Y}(x, y) \, dx \, dy \\
&= \int_{-\infty}^{\infty} x f_X(x) dx \cdot \int_{-\infty}^{\infty} y f_Y(y) dy \\
&= \mathbb{E}[X] \cdot \mathbb{E}[Y]
\end{align*}
$$
{{< /callout >}}

We can extend this to functions of independent random variables. If $X$ and $Y$ are independent, then for any functions $g_1(X)$ and $g_2(Y)$,

$$
\mathbb{E}[g_1(X) g_2(Y)] = \mathbb{E}[g_1(X)] \cdot \mathbb{E}[g_2(Y)]
$$

{{< callout type="proof" >}}
$$
\begin{align*}
\mathbb{E}[g_1(X) g_2(Y)] &= \sum_{x} \sum_{y} g_1(x) g_2(y) \cdot p_{X,Y}(x, y) \\
&= \sum_{x} g_1(x) p_X(x) \sum_{y} g_2(y) p_Y(y) \\
&= \mathbb{E}[g_1(X)] \cdot \mathbb{E}[g_2(Y)]
\end{align*}
$$

The continuous case is similar, with integrals replacing sums.
{{< /callout >}}

This also generalizes to the case of multiple independent random variables. If $X_1, \ldots, X_n$ are independent random variables and $g_i(X_i)$ are functions of these variables, then:

$$
\mathbb{E}[g_1(X_1) \cdots g_n(X_n)] = \prod_{i=1}^n \mathbb{E}[g_i(X_i)]
$$

{{< callout type="proof" >}}
We can prove this by induction on the number of variables $n$. The base case $n=2$ was already shown above. We then assume as our **induction hypothesis** that it holds for $n$ independent random variables, and show it holds for $n+1$:

$$
\mathbb{E}\left[ \prod_{i=1}^{n+1} g_i(X_i) \right] 
= \mathbb{E}\left[ \left(\prod_{i=1}^n g_i(X_i)\right) g_{n+1}(X_{n+1}) \right]
$$

Since $X_{n+1}$ is independent of $X_1, \ldots, X_n$, we can separate:

$$
\mathbb{E}\left[ \left(\prod_{i=1}^n g_i(X_i)\right) g_{n+1}(X_{n+1}) \right] = \mathbb{E}\left[ \prod_{i=1}^n g_i(X_i) \right] \cdot \mathbb{E}[ g_{n+1}(X_{n+1}) ]
$$

By the induction hypothesis we then get:

$$
\begin{align*}
\mathbb{E}\left[ \prod_{i=1}^{n+1} g_i(X_i) \right] &= \mathbb{E}\left[ \left(\prod_{i=1}^n g_i(X_i)\right) g_{n+1}(X_{n+1}) \right] \\
&= \prod_{i=1}^n \mathbb{E}[g_i(X_i)] \cdot \mathbb{E}[g_{n+1}(X_{n+1})] \\
&= \prod_{i=1}^{n+1} \mathbb{E}[g_i(X_i)]
\end{align*}
$$
{{< /callout >}}

{{< callout type="example" title="Independent" >}}
Suppose you roll two independent fair six-sided dice, $X$ and $Y$, so $\mathbb{E}[X] = \mathbb{E}[Y] = 3.5$. By independence:

$$
\mathbb{E}[XY] = \mathbb{E}[X] \cdot \mathbb{E}[Y] = 3.5 \times 3.5 = 12.25
$$

Let's verify this by direct calculation:

$$
\mathbb{E}[XY] = \sum_{x=1}^{6} \sum_{y=1}^{6} x y \cdot \frac{1}{6} \cdot \frac{1}{6} = \frac{1}{36} \sum_{x=1}^{6} \sum_{y=1}^{6} x y
$$

But $\sum_{x=1}^{6} x = 21$, and the sum of products $\sum_{x=1}^{6} \sum_{y=1}^{6} x y = (\sum_{x=1}^{6} x)(\sum_{y=1}^{6} y) = 21 \times 21 = 441$.

$$
\mathbb{E}[XY] = \frac{441}{36} = 12.25
$$

which matches $\mathbb{E}[X] \cdot \mathbb{E}[Y]$.
{{< /callout >}}

{{< callout type="example" title="Dependent" >}}
{{< callout type="todo" >}}
Is this correct?
{{< /callout >}}
Suppose you play two independent games: Game A: Flip a fair coin. Win $X = 1$ (heads) or $X = 2$ (tails). Game B: Roll a fair die. Win $Y \in \{1,2,3,4,5,6\}$ equally likely.

Then $\mathbb{E}[X] = (1+2)/2 = 1.5$ and $\mathbb{E}[Y] = (1+2+3+4+5+6)/6 = 3.5$. If your total payout is $Z = X \cdot Y$ (the product), then:

$$
\mathbb{E}[Z] = \mathbb{E}[X] \cdot \mathbb{E}[Y] = 1.5 \times 3.5 = 5.25
$$

If the games are dependent (say, $Y = 7 - X$), this formula fails. For example, if $X = 1$ then $Y = 6$, $X = 2$ then $Y = 5$, and so on. Then,

$$
\mathbb{E}[XY] = (1 \cdot 6 + 2 \cdot 5)/2 = (6 + 10)/2 = 8
$$

but $\mathbb{E}[X] = 1.5$, $\mathbb{E}[Y] = 5.5/2 = 2.75$, so $\mathbb{E}[X] \mathbb{E}[Y] = 4.125 \neq 8$.
{{< /callout >}}

## Variance and Standard Deviation

So far, we have a notion of average for a random variable, but just knowing the expectation doesn't tell us how much the values of a random variable **fluctuate** around that mean. Are most outcomes tightly clustered near the expected value, or can they be wildly far from it? To measure this spread or dispersion, we introduce the concepts of **variance** and **standard deviation**.

Think of a number line: Imagine marking a dot for each possible value of $X$ (weighted by probability). The mean $m = \mathbb{E}[X]$ gives the "center of mass" of these dots. The **variance** tells us how far, on average, these dots are from the center.

A first guess for an average distance from the mean might be the **mean absolute deviation (MAD)**:

$$
\frac{1}{n} \sum_{i=1}^n |x_i - m| = \frac{1}{n} \sum_{i=1}^n |X(\omega_i) - \mathbb{E}[X]|
$$

where the $x_i$ are outcomes of $X$ and $m=\mathbb{E}[X]$ is the mean or expectation. This measures, on average, how far you are from the mean—but it has some issues. It is **not differentiable** at $m$ (the kink at zero), which makes it less useful for calculus-based arguments. It does **not behave nicely under linear transformations** (e.g., scaling).

For these reasons, probability theory prefers to use **variance**. For the variance we measure the **squared distance** from the mean. Why squared? It ensures all differences are positive, and, more importantly, gives us a quantity with nice mathematical properties. So we define the variance of a random variable $X$ as:

$$
\text{Var}(X) = \sigma^2_X = \mathbb{E}\left[(X - m)^2\right] = \mathbb{E}\left[(X - \mathbb{E}[X])^2\right]
$$

where $\sigma^2_X$ is the variance of $X$ and $m = \mathbb{E}[X]$ is the mean. Notice with the process of squaring to not have negative distances, we have also introduced changed the units of measurement whilst also weighting larger deviations more heavily. The variance is now in squared units of $X$. So if $X$ has units (e.g., meters), the variance is in squared units (meters$^2$). We can go back to the original units by using the **standard deviation**:

$$
\sigma_X = \sqrt{\text{Var}(X)} = \sqrt{\mathbb{E}[(X - \mathbb{E}[X])^2]}
$$

This puts the measure of spread back into the original units of $X$. If $X$ is a random variable with $\mathbb{E}[X^2] < \infty$, we can rewrite variance:

$$
\text{Var}(X) = \mathbb{E}[(X - \mathbb{E}[X])^2] = \mathbb{E}[X^2] - (\mathbb{E}[X])^2
$$

This formula is often easier to compute. The condition $ \mathbb{E}[X^2] < \infty $ ensures that the variance is well-defined (finite). 

{{< callout type="proof" >}}
Expand the square:

$$
\text{Var}(X) = \mathbb{E}[(X - m)^2] = \mathbb{E}[X^2 - 2mX + m^2] = \mathbb{E}[X^2] - 2m\mathbb{E}[X] + m^2
$$

But $\mathbb{E}[X] = m$, so:

$$
= \mathbb{E}[X^2] - 2m^2 + m^2 = \mathbb{E}[X^2] - m^2 = \mathbb{E}[X^2] - (\mathbb{E}[X])^2
$$
{{< /callout >}}

{{< callout type="example" title="Variance of a Constant" >}}
If $X = c$ (a constant random variable), then $\mathbb{E}[X] = c$ always, and

$$
\text{Var}(X) = \mathbb{E}[(c - c)^2] = \mathbb{E}[0] = 0
$$

So a constant has zero variance; there is no spread! This makes sense because a constant random variable does not vary at all, it always takes the same value, so there is no deviation from the mean.
{{< /callout >}}

{{< callout type="example" title="Throwing a Die" >}}
Let $X$ be the outcome of a fair six-sided die, $W = \{1,2,3,4,5,6\}$. We already have $\mathbb{E}[X] = 3.5$.

Now, compute $\mathbb{E}[X^2]$:

$$
\mathbb{E}[X^2] = \sum_{x=1}^6 x^2 \cdot \frac{1}{6} = \frac{1^2 + 2^2 + 3^2 + 4^2 + 5^2 + 6^2}{6}
= \frac{1+4+9+16+25+36}{6} = \frac{91}{6} \approx 15.1667
$$

Therefore,

$$
\text{Var}(X) = \mathbb{E}[X^2] - (\mathbb{E}[X])^2 = \frac{91}{6} - (3.5)^2 \approx 15.1667 - 12.25 = 2.9167
$$

Standard deviation:

$$
\sigma_X = \sqrt{2.9167} \approx 1.7078
$$

Interpretation: A die roll deviates from the mean (3.5) by about $1.71$ on average (in original units). This matches our knowledge of a die's spread: most rolls are within 1 or 2 of the mean so between 2.5 and 4.5, but can be as far as 1 or 6.
{{< /callout >}}

{{< callout type="example" title="Uniform Distribution" >}}
Let $X \sim \mathrm{Uniform}(a, b)$. Then the variance is:

$$
\text{Var}(X) = \frac{(b-a)^2}{12}
$$

For $X \sim \mathrm{Uniform}(0,1)$,

$$
\text{Var}(X) = \frac{(1-0)^2}{12} = \frac{1}{12}
$$
{{< /callout >}}
The variance also has some useful properties that make it easier to work with just like the expectation. The first one being that shifting a random variable by a constant does not change its variance. This matches our intuition as we are just moving the entire distribution without changing its spread or mean. So formally adding a constant $c$ to $X$ does not change its variance:

$$
\text{Var}(X + c) = \text{Var}(X)
$$

We can also scale a random variable by a constant without so if $X$ is a random variable and $\lambda$ is a constant:

$$
\text{Var}(\lambda X) = \lambda^2 \text{Var}(X)
$$

The interpretation of this result is that if we stretch or compress the random variable by a factor of $\lambda$, the variance scales by the square of that factor. This is intuitive because if we stretch the distribution, the spread increases, and if we compress it, the spread decreases.

{{< callout type="proof" >}}
Let $m = \mathbb{E}[X]$. Then $\mathbb{E}[\lambda X] = \lambda m$. Now,

$$
\begin{align*}
\text{Var}(\lambda X) &= \mathbb{E}[(\lambda X - \mathbb{E}[\lambda X])^2] \\
&= \mathbb{E}[(\lambda X - \lambda m)^2] \\
&= \mathbb{E}[\lambda^2 (X - m)^2] \\
&= \lambda^2 \mathbb{E}[(X - m)^2] \\
&= \lambda^2 \text{Var}(X)
\end{align*}
$$
{{< /callout >}}

So we can put this together to get a general rule for scaling and shifting:

$$
\text{Var}(aX + b) = a^2 \text{Var}(X)
$$

where $a$ is the scaling factor and $b$ is the shift. 

We can also sum independent random variables and their variances add up. This is a powerful property that allows us to compute the variance of sums of independent random variables without needing to know their joint distribution. So if $X_1, X_2, \ldots, X_n$ are independent random variables, then:

$$
\text{Var}\left(\sum_{i=1}^n X_i\right) = \sum_{i=1}^n \text{Var}(X_i)
$$

{{< callout type="proof" >}}
For $n=2$, by independence and linearity of expectation:

$$
\begin{align*}
\text{Var}(X_1 + X_2) &= \mathbb{E}[(X_1 + X_2 - (\mathbb{E}[X_1] + \mathbb{E}[X_2]))^2] \\
&= \mathbb{E}[(X_1 - \mathbb{E}[X_1])^2] + \mathbb{E}[(X_2 - \mathbb{E}[X_2])^2] + 2\mathbb{E}[(X_1 - \mathbb{E}[X_1])(X_2 - \mathbb{E}[X_2])]
\end{align*}
$$

But if $X_1, X_2$ are independent, the cross-term vanishes:

$$
\mathbb{E}[(X_1 - \mathbb{E}[X_1])(X_2 - \mathbb{E}[X_2])] = \mathbb{E}[X_1 - \mathbb{E}[X_1]] \mathbb{E}[X_2 - \mathbb{E}[X_2]] = 0
$$

So:

$$
\text{Var}(X_1 + X_2) = \text{Var}(X_1) + \text{Var}(X_2)
$$

And similarly for $n > 2$ (by induction).
{{< /callout >}}

{{< callout type="example" title="Binomial Distribution" >}}
Let $X \sim \mathrm{Binomial}(n,p)$: the sum of $n$ independent Bernoulli($p$) random variables. Each has variance $p(1-p)$. Thus we can use additivity of variance:

$$
\text{Var}(X) = n p (1-p)
$$
{{< /callout >}}

## Covariance and Correlation

Variance tells us how much a single random variable fluctuates around its mean. But what if we're interested in how two random variables move together? Do they both tend to be large at the same time, or does one tend to be large when the other is small? For this, we introduce the concepts of **covariance** and **correlation**.

Suppose $X$ and $Y$ are random variables with finite second moments (i.e., $\mathbb{E}[X^2] < \infty$ and $\mathbb{E}[Y^2] < \infty$). The **covariance** of $X$ and $Y$ is defined as:

$$
\text{Cov}(X, Y) = \mathbb{E}[(X - \mathbb{E}[X])(Y - \mathbb{E}[Y])]
$$

Alternatively, by expanding the definition using linearity of expectation, we can write:

$$
\text{Cov}(X, Y) = \mathbb{E}[XY] - \mathbb{E}[X]\mathbb{E}[Y]
$$

Notice that if $X = Y$, then $\text{Cov}(X, Y) = \mathbb{E}[(X - \mathbb{E}[X])^2] = \text{Var}(X)$. Thus, covariance generalizes variance to the case of two random variables. The intuition is that covariance measures how much $X$ and $Y$ vary together.

Formally, covariance captures the **average product of deviations** from the mean. If large values of $X$ are associated with large values of $Y$, this product is positive more often, so the covariance is positive. If large $X$ is paired with small $Y$, covariance is negative. If $X$ and $Y$ fluctuate independently, the positive and negative terms tend to cancel, and covariance is near zero.

The reason we require finite second moments is to ensure that the covariance is well-defined. Specifically, we need $\mathbb{E}[XY]$ and $\mathbb{E}[X]\mathbb{E}[Y]$ to be finite. This is guaranteed if both $\mathbb{E}[X^2]$ and $\mathbb{E}[Y^2]$ are finite, which is a common assumption in probability theory. We can see this using the inequality $2|ab| \leq a^2 + b^2$:

$$
2\mathbb{E}[|XY|] \leq \mathbb{E}[X^2] + \mathbb{E}[Y^2] \implies \mathbb{E}[|XY|] < \infty
$$

Covariance measures whether two variables move together, but its magnitude depends on the scales of $X$ and $Y$. To make this comparison unitless and interpretable, we use [correlation](). So you can think of correlation as a normalized or standardized version of covariance. For example the **Pearson correlation coefficient** between $X$ and $Y$ is:

$$
\rho_{X, Y} = \frac{\text{Cov}(X, Y)}{\sigma_X \sigma_Y}
$$

So we standardize the covariance by the product of the standard deviations of $X$ and $Y$. This gives us a measure that is always between -1 and 1, making it easier to interpret.

Notice that we have actually already seen covariance when summing two (possibly dependent) random variables, the variance of their sum is:

$$
\begin{align*}
\text{Var}(X + Y) &= \mathbb{E}[(X + Y - \mathbb{E}[X + Y])^2] \\
&= \text{Var}(X) + \text{Var}(Y) + \mathbb{E}[(X - \mathbb{E}[X])(Y - \mathbb{E}[Y])] \\
&= \text{Var}(X) + \text{Var}(Y) + \text{Cov}(X, Y)
\end{align*}
$$

So as already mentioned, this term is 0 when $X$ and $Y$ are independent, and it captures the interaction between $X$ and $Y$ when they are dependent. Meaning if $X$ and $Y$ are independent, then:

$$
\mathbb{E}[XY] = \mathbb{E}[X]\mathbb{E}[Y] \implies \text{Cov}(X, Y) = 0
$$

But zero covariance does not imply independence. It only means that $X$ and $Y$ are uncorrelated they might have nonlinear dependencies as correlation only measures linear relationships which is discussed in more detail in the [Correlation]() section.

{{< callout type="example" title="Dependent Variables with Non-Zero Covariance" >}}
Suppose $X$ and $Y$ are random variables with $\mathbb{E}[X] = \mathbb{E}[Y] = 0$ (for simplicity), $\text{Var}(X) = 4$, $\text{Var}(Y) = 9$, and $\text{Cov}(X, Y) = 3$. Then we can compute the variance of their sum:

$$
\text{Var}(X + Y) = 4 + 9 + 2 \cdot 3 = 19
$$

If $X$ and $Y$ are independent ($\text{Cov}(X, Y) = 0$), then $\text{Var}(X + Y) = 13$. 
{{< /callout >}}

{{< callout type="example" title="Dependent Variables with Zero Covariance" >}}
Let $X$ be uniformly distributed on $[-1, 1]$, and let $Y = X^2$. Then $\mathbb{E}[X] = 0$ and $Y$ is always non-negative, but

$$
\text{Cov}(X, Y) = \mathbb{E}[X Y] - \mathbb{E}[X]\mathbb{E}[Y] = \mathbb{E}[X^3] - 0 = 0
$$

However, $X$ and $Y$ are **not independent** (since knowing $Y$ tells you about $X$), but their covariance is zero. 
{{< /callout >}}

## Approximating Expectation and Probabilities

Probability theory not only lets us compute exact values, but also provides powerful tools to **estimate** or **bound** probabilities and expectations, even when we have only partial information about a random variable. This is essential when precise calculations are intractable or we care more about "worst-case" scenarios. In this section, we present several fundamental inequalities and properties for expectations and probabilities, which allow us to make robust statements about random variables.

### Monotonicity of Expectation

One of the basic but surprisingly useful facts about expectation is that it **preserves order.** If a random variable $X$ is always less than or equal to another random variable $Y$—meaning $X(\omega) \leq Y(\omega)$ for all $\omega$ except perhaps a set of probability zero (i.e. almost surely), then the expected value of $X$ cannot exceed the expected value of $Y$:

$$
X \leq Y \text{ almost surely } \implies \mathbb{E}[X] \leq \mathbb{E}[Y]
$$

We know that expectation is a "weighted average." If every possible value of $X$ is at most the corresponding value of $Y$, then the average of $X$ can't be more than the average of $Y$. Again think of the dots on a number line: if every dot for $X$ is to the left of or at the same position as the corresponding dot for $Y$, then the center of mass (the expectation) of $X$ must be to the left of or at the same position as that of $Y$.

{{< callout type="proof" >}}
Let $Z = Y - X$. Since $X \leq Y$ almost surely, we have $Z \geq 0$ almost surely, so $\mathbb{E}[Z] \geq 0$. By linearity:

$$
\mathbb{E}[Z] = \mathbb{E}[Y - X] = \mathbb{E}[Y] - \mathbb{E}[X] \geq 0 \implies \mathbb{E}[X] \leq \mathbb{E}[Y]
$$
{{< /callout >}}

{{< callout type="example" title="Dice Example" >}}
Let $X$ be the roll of a fair die, and $Y = X + 2$. Then for every outcome, $Y \geq X$. We have $\mathbb{E}[X] = 3.5$ and $\mathbb{E}[Y] = 5.5$, so $\mathbb{E}[Y] \geq \mathbb{E}[X]$.
{{< /callout >}}

### Markov's Inequality

Often, we wish to bound the probability that a random variable is "much larger" than its mean, without knowing its detailed distribution. **Markov's inequality** gives a universal upper bound on the tail probabilities of non-negative random variables:

$$
\mathbb{P}(X \geq a) \leq \frac{\mathbb{E}[X]}{a} \qquad \text{for any } X \geq 0,\, a > 0
$$

The idea is that the chance that $X$ gets very large cannot be too big unless its average is also large. Markov's inequality is crude, but it needs almost no information: only that $X$ is non-negative and has a finite expectation.

{{< callout type="proof" >}}
Let $A = \{\omega : X(\omega) \geq a\}$. Then for $\omega \in A$, $X(\omega) \geq a$, and for the rest, $X(\omega) \geq 0$. Then

$$
\mathbb{E}[X] = \int_\Omega X d\mathbb{P} \geq \int_A X d\mathbb{P} \geq \int_A a d\mathbb{P} = a\,\mathbb{P}(X \geq a)
$$
Solving for the probability gives the inequality.
{{< /callout >}}

We can also apply Markov's inequality to transformed random variables. If $g$ is a non-decreasing, non-negative function, then:

$$
\mathbb{P}(X \geq a) = \mathbb{P}\left(g(X) \geq g(a)\right) \leq \frac{\mathbb{E}[g(X)]}{g(a)}
$$
This allows bounding tail probabilities using expectations of other functions of $X$. The reason this works is that if $g$ is non-decreasing, then $g(X) \geq g(a)$ implies $X \geq a$, so the probability of the event is at most the expectation of the transformed variable divided by the transformation applied to the threshold.

{{< callout type="example" >}}
Suppose $X$ is the number of emails you receive in a day, with $\mathbb{E}[X] = 4$. What is the probability you get at least 20 emails today?

$$
\mathbb{P}(X \geq 20) \leq \frac{4}{20} = 0.2
$$

So, no more than 20%. As you can see this is a very loose bound, as the actual probability might be much lower due to the jump from 4 to 20 being quite large compared to the average. But it gives a quick way to estimate the tail probability without knowing the distribution.
{{< /callout >}}

### Chebyshev's Inequality

While Markov's inequality uses only the expectation, **Chebyshev's inequality** exploits both the mean and variance to bound how likely $X$ is to fall *far* from its expectation. If $X$ has finite variance so $\mathbb{E}[X^2] < \infty$, then for any $a > 0$,

$$
\mathbb{P}(|X - \mathbb{E}[X]| \geq a) \leq \frac{\text{Var}(X)}{a^2}
$$

The idea is that if $X$ has low variance (the values cluster near the mean), it is unlikely to have a large deviation from the mean. If $X$ has high variance, large deviations are more probable. So if the distance from the mean is small, then we are likely to be close to the mean and within the standard deviation so we are dividing by a small number, which makes the probability larger. If the distance is large, then we are likely to be far from the mean and dividing by a large number, which makes the probability smaller.

{{< callout type="proof" >}}
Let $Y = (X - \mathbb{E}[X])^2 \geq 0$ and apply Markov's inequality. By definition of variance we have $\text{Var}(X) = \mathbb{E}[(X - \mathbb{E}[X])^2] = \mathbb{E}[Y]$. Then for any $a > 0$:


$$
\mathbb{P}(|X - \mathbb{E}[X]| \geq a) = \mathbb{P}\left((X - \mathbb{E}[X])^2 \geq a^2\right) = \mathbb{P}(Y \geq a^2)
$$

If we apply Markov's inequality to $Y$:

$$
\mathbb{P}(|X - \mathbb{E}[X]| \geq a) = \mathbb{P}(Y \geq a^2) \leq \frac{\mathbb{E}[Y]}{a^2} = \frac{\text{Var}(X)}{a^2}
$$
{{< /callout >}}

{{< callout type="example" title="Height Distribution" >}}
Suppose adult heights have mean $175$cm, standard deviation $10$cm. What is the probability a random person's height is less than $160$cm or more than $190$cm? So we are $15$cm from the mean:

$$
\mathbb{P}(|X - 175| \geq 15) \leq \frac{100}{15^2} = \frac{100}{225} \approx 0.444
$$
So, at most $44.4\%$ can be that far from the mean.  

Actually, if heights are normal, the true probability is much less (about 13.4%), illustrating that Chebyshev is often conservative.
{{< /callout >}}

### Jensen's Inequality

Many applications and bounds in probability rely on **Jensen's inequality**, which links [convexity]() with expectation as convex functions are very useful in optimization and probability. Suppose $g: \mathbb{R} \to \mathbb{R}$ is convex. Then for any integrable random variable $X$ we have:

$$
g(\mathbb{E}[X]) \leq \mathbb{E}[g(X)]
$$

If $g$ is concave, the inequality reverses so:

$$
g(\mathbb{E}[X]) \geq \mathbb{E}[g(X)]
$$

"Convex outside, concave inside:" The expected value of a convex transformation is at least the transformation of the expected value. If you "average then transform" you never exceed "transform then average" for convex $g$. The intuition is that for a convex function, the "curve" always lies above its chords. The average of several points on the curve (i.e., $\mathbb{E}[g(X)]$) will always be at least as large as the value at the average point ($g(\mathbb{E}[X])$).

{{< callout type="proof" >}}
For any finite support (discrete) random variable, suppose $X$ takes values $x_i$ with probabilities $p_i$ (sum to 1). By convexity:  
$$
g\Big(\sum p_i x_i \Big) \leq \sum p_i g(x_i)
$$
which is exactly the Jensen's statement.
{{< /callout >}}

{{< callout type="example" title="Triangle Inequality for Expectation" >}}
If $g(x) = |x|$, then for any integrable $X$:

$$
|\mathbb{E}[X]| \leq \mathbb{E}[|X|]
$$

So, the absolute value of the average cannot exceed the average absolute value. This is the triangle inequality for expectation.
{{< /callout >}}

{{< callout type="example" title="Moments and Jensen" >}}
If $g(x) = x^2$, which is convex, we have:

$$
(\mathbb{E}[X])^2 \leq \mathbb{E}[X^2]
$$

or equivalently:

$$
\mathbb{E}[X^2] \geq (\mathbb{E}[X])^2
$$

this can be used for variance to show that the variance is non-negative and defined as $\text{Var}(X) = \mathbb{E}[X^2] - (\mathbb{E}[X])^2$). This inequality is fundamental in probability and statistics, as it shows that the variance is always non-negative.
{{< /callout >}}

{{< callout type="example" title="Root Mean Square Inequality" >}}
If $g(x) = \sqrt{|x|}$ (concave for $x \geq 0$), then Jensen reverses:

$$
\sqrt{\mathbb{E}[|X|]} \geq \mathbb{E}[\sqrt{|X|}]
$$
{{< /callout >}}

### Chernoff Bounds

What dis? tighter approximation using markov and Chebyshev
