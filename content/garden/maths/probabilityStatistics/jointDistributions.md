---
title: Joint Distributions
type: docs
weight: 6
---

When working with a single random variable, our questions are often of the form: "What is the probability that a person is less than 30 years old?" or "What is the probability that a die shows a 5?" But what if we want to ask **questions about two or more random variables at the same time**? For example:

-"What is the probability that a person is less than 30 and taller than 1.8m?"
-"What is the probability that, when throwing two dice, the first shows an even number and the second is prime?"

These are examples of **joint distributions**. Joint distributions (or **multivariate distributions**) describe the probability structure of multiple random variables together, capturing both their individual behaviors and how they relate to each other.

You can think of a joint distribution as the probability for a **random vector**:

$$
\mathbf{X} = (X_1, X_2, \ldots, X_n)
$$

where each $X_i$ is a random variable. The sample space is thus "cut into boxes" corresponding to all possible combinations of values for $(X_1, X_2, \ldots, X_n)$.

## Discrete Random Variables

Let's start with **two discrete random variables**, $X$ and $Y$, each taking values in (possibly different) countable sets $W_X$ and $W_Y$ (for example, both could be $\{0,1\}$ for coin flips, or $\{1,2,3,4,5,6\}$ for dice). The **joint probability mass function (pmf)** is defined as:

$$
p(x, y) = \mathbb{P}(X = x,\, Y = y),\qquad (x, y) \in W_X \times W_Y
$$

This function assigns a probability to each “box” in the two-dimensional grid of possible pairs. You can think of the comma being an "and" here, so $p(x, y)$ is the probability that $X$ takes the value $x$ **and** $Y$ takes the value $y$. So this is then like a conditional probability, but for two variables at once as we are looking at the probability of both $X$ and $Y$ taking specific values simultaneously:

$$
p(x, y) = \mathbb{P}(X = x, Y = y) = \mathbb{P}(X = x | Y = y) \cdot \mathbb{P}(Y = y) = \mathbb{P}(Y = y | X = x) \cdot \mathbb{P}(X = x)
$$

From the above it is also clear that if one of the variables is fixed, it becomes a normal conditional probability. This obviously also means that if one of the variables has a probability of 0, then the joint probability is also 0. 

Just like for a single random variable we can also define the **joint cumulative distribution function (cdf)** for two variables:

$$
F(x, y) = \mathbb{P}(X \leq x, Y \leq y) = \sum_{x' \leq x} \sum_{y' \leq y} p(x', y')
$$

This gives us the probability that $X$ is less than or equal to $x$ **and** $Y$ is less than or equal to $y$. Again we iterate over all values $x' \in W_X$ and $y' \in W_Y$ that are less than or equal to $x$ and $y$, respectively, summing their joint probabilities.

{{< callout type="example" >}}
Let $X$ and $Y$ be two independent fair coin tosses ($\mathbb{P}(X=0)=\mathbb{P}(X=1)=\frac{1}{2}$ and $\mathbb{P}(Y=0)=\mathbb{P}(Y=1)=\frac{1}{2}$). Our joint distribution is then as follows because there are four equally likely outcomes:

$$
p(x, y) = \mathbb{P}(X = x, Y = y) = \frac{1}{4}, \qquad x, y \in \{0,1\}
$$

You can think of this as a table where each cell corresponds to a pair $(x, y)$:

| $X \backslash Y$ | 0   | 1   |
| -------------------- | --- | --- |
| 0                    | 1/4 | 1/4 |
| 1                    | 1/4 | 1/4 |

So the probability of getting $X=0$ and $Y=1$ is $\mathbb{P}(X=0, Y=1) = 1/4$.

We could also define the joint distribution of $(X, X)$:

| $X \backslash X$ | 0   | 1   |
| -------------------- | --- | --- |
| 0                    | 1/2 | 0   |
| 1                    | 0   | 1/2 |

This makes sense as one $X$ can not be both 0 and 1 at the same time, so the joint distribution is 0 for all pairs where $x \neq y$:

$$
p(x, y) = \mathbb{P}(X = x, Y = y) = \begin{cases}
\frac{1}{2} & \text{if } x = y \\
0 & \text{otherwise}
\end{cases}
$$
{{< /callout >}}

We can also extend this to **$n$ discrete random variables**. For $n$ variables $X_1, X_2, \ldots, X_n$, where each $X_i$ takes values in countable sets $W_i$, the joint distribution is defined as:

$$
p(x_1, \ldots, x_n) = \mathbb{P}(X_1 = x_1,\, X_2 = x_2,\, \ldots,\, X_n = x_n), \qquad (x_1, \ldots, x_n) \in W_1 \times \cdots \times W_n
$$

This captures the probability of every possible combination. Because we have a probability measure the joint distribution must add up to 1 over all combinations of values. So summing over all combinations $(x_1, \ldots, x_n)$ in the Cartesian product of the sets $W_1, \ldots, W_n$, we have:

$$
\sum_{(x_1, \ldots, x_n) \in W_1 \times \cdots \times W_n} p(x_1, \ldots, x_n) = 1
$$

{{< callout type="proof" >}}
Let's define the set $A$ as $\{X_1 \in W_1, X_2 \in W_2, \ldots, X_n \in W_n\}$, which is the event that the vector $(X_1, \ldots, X_n)$ takes *some* value in the sample space. So $A$ is a finite intersection of almost sure events so $\mathbb{P}(A) = 1$. So A covers the entire sample space of the joint distribution. We can write $A$ as the union of all "elementary events" where each $X_i$ takes a specific value:

$$
A = \bigcup_{x_1 \in W_1, ..., x_n \in W_n} \{ X_1 = x_1, ..., X_n = x_n \}
$$

These elementary events are disjoint, meaning they do not overlap. Therefore, the probability of $A$ is the sum of the probabilities of these elementary events by countable additivity:

$$
\mathbb{P}(A) = \sum_{x_1, ..., x_n} \mathbb{P}(X_1 = x_1, ..., X_n = x_n) = 1
$$
{{< /callout >}}

### Transformed Random Variables

Remember that we can [transform random variables]() which is basically creating a new random variable by applying a function to existing ones. We saw that we can also apply a function of **several random variables** to create a new random variable. 

Suppose we have two (or more) random variables $X$ and $Y$ on the same probability space, and we have a function $g: \mathbb{W}^2 \to \R$ (or in general $g: \R^n \to \R$). We can define a new random variable $Z$ by:

$$
Z(\omega) = g(X(\omega), Y(\omega)) \qquad \text{for all } \omega \in \Omega
$$

So this analogous to how we define a joint distribution:

$$
\mathbb{P}(Z = z) = \mathbb{P}(g(X, Y) = z) = \sum_{(x, y) \in W_X \times W_Y} \mathbb{P}(X = x, Y = y) \cdot \mathbb{I}(g(x, y) = z)
$$

where $\mathbb{I}$ is the indicator function that is 1 if the condition holds and 0 otherwise. This means we are summing over all pairs $(x, y)$ where $g(x, y) = z$. That is, we sum the joint probabilities over all points where $g(x_1, ..., x_n) = z$.

{{< callout type="example" >}}
Let $X, Y$ be independent coin flips ($\{0,1\}$), and define $Z = X + Y$.

- $\mathbb{P}(Z = 0) = \sum_{(x,y) \in \{0,1\}^2} \mathbb{P}(X = x, Y = y) \cdot \mathbb{I}(g(x,y) = 0)$ = $\mathbb{P}(X=0, Y=0) = 1/4$
- $\mathbb{P}(Z = 1) = \sum_{(x,y) \in \{0,1\}^2} \mathbb{P}(X = x, Y = y) \cdot \mathbb{I}(g(x,y) = 1)$ = $\mathbb{P}(X=0, Y=1) + \mathbb{P}(X=1, Y=0) = 1/4 + 1/4 = 1/2$
- $\mathbb{P}(Z = 2) = \sum_{(x,y) \in \{0,1\}^2} \mathbb{P}(X = x, Y = y) \cdot \mathbb{I}(g(x,y) = 2)$ = $\mathbb{P}(X=1, Y=1) = 1/4$

So the distribution of $Z$ is:

$$
p_Z(z) = \begin{cases}
\frac{1}{4} & \text{if } z = 0 \\
\frac{1}{2} & \text{if } z = 1 \\
\frac{1}{4} & \text{if } z = 2 \\
0 & \text{otherwise}
\end{cases}
$$

we could then also take the joint distribution of $(X, Z)$:

| $X \backslash Z$ | 0   | 1   | 2   |
|------------------|-----|-----|-----|
| 0                | 1/4 | 1/4 | 0 |
| 1                | 0   | 1/4 | 1/4 |
{{< /callout >}}

## Marginal Distributions (Sum Rule)

Suppose we want to "forget" about one of the variables, say $Y$, and just look at the distribution of $X$ on its own. The resulting distribution is called the **marginal distribution** of $X$. This process is often referred to as **marginalization** or the **sum rule** for random variables, as it corresponds to the Law of Total Probability for events. Formally:

$$
p_X(x) = \mathbb{P}(X = x) = \sum_{y \in W_Y} p(x, y)
$$

We sum over all possible values of $Y$ (think of collapsing the table), so we're aggregating all probabilities in the $X = x$ “row” of the table. This is very practical as we often have access to the joint distribution, but want to analyze just one variable at a time. Intuitively the marginal tells us the "overall" probability for $X$, regardless of the value of $Y$. In conditional probability terms, this is like going over all possible values for the conditioning variable $Y$ and summing the probabilities. In general for $n$ variables we get the marginal of $X_i$ as follows:

$$
p_{X_i}(z) = \sum_{x_1, ..., x_{i-1}, x_{i+1}, ..., x_n} p(x_1, ..., x_{i-1}, z, x_{i+1}, ..., x_n)
$$

This is the sum over all combinations of the other variables, effectively collapsing the joint distribution to just focus on $X_i$. If you have the joint distribution, you can always compute the marginals by summing out the other variables. However, the reverse is not true, if you have only the marginals, you cannot uniquely determine the joint distribution. The idea here is that the joint distribution contains more information than just the marginals, specifically about how the variables interact with each other.

{{< callout type="example" >}}
As an example, consider the joint distribution of two independent fair coins $X$ and $Y$. The marginal distributions are:

$$
p_X(x) = \mathbb{P}(X = x) = \sum_{y \in \{0,1\}} p(x, y) = \frac{1}{2}, \quad x \in \{0,1\}
$$

Show two differently defined joint distributions that have the same marginals. For example, the joint distribution of $(X, Y)$ and $(X, X)$ both have the same marginals?
{{< /callout >}}

## Conditional Distributions and the Product Rule

Just as we defined conditional probability for events, we can define the **conditional probability mass function** for discrete random variables. The probability that $X=x$ given that $Y=y$ is:

$$
p_{X|Y}(x|y) = \mathbb{P}(X=x | Y=y) = \frac{\mathbb{P}(X=x, Y=y)}{\mathbb{P}(Y=y)} = \frac{p(x,y)}{p_Y(y)}
$$

provided $p_Y(y) > 0$. This tells us the probability distribution of $X$ when we know that $Y$ has taken the value $y$.

By rearranging this formula, we obtain the **product rule** (or chain rule) for random variables, which allows us to factorize a joint distribution into a product of conditional and marginal distributions:

$$
p(x, y) = p_{X|Y}(x|y) \cdot p_Y(y)
$$

This is completely analogous to the product rule for events $\mathbb{P}(A \cap B) = \mathbb{P}(A|B)\mathbb{P}(B)$. It is extremely useful because it is often easier to model systems by specifying a marginal distribution for one variable and a conditional distribution for the other, rather than specifying the joint distribution directly.

This rule extends to $n$ random variables as the **chain rule**:

$$
p(x_1, x_2, \ldots, x_n) = p(x_1) \cdot p(x_2 | x_1) \cdot p(x_3 | x_1, x_2) \cdots p(x_n | x_1, \ldots, x_{n-1})
$$

## Expectation of Joint Distributions

When dealing with joint distributions, we often want to compute the **expectation** (or mean) of a function of the random variables involved. This is similar to how we compute the expectation for a single random variable, but instead we look at a function applied to the joint distribution because the expectation is only defined for a single random variable. Suppose $X_1, X_2, \ldots, X_n$ are discrete random variables with joint distribution $p(x_1, x_2, \ldots, x_n)$. The expectation of a function $\varphi$ of these variables is defined as:

$$
\mathbb{E}[\varphi(X_1, ..., X_n)] = \sum_{x_1, ..., x_n} \varphi(x_1, ..., x_n)\, p(x_1, ..., x_n)
$$

So for example to then compute the expectation of one of the marginals $X_i$, we can set $\varphi(x_1, ..., x_n) = x_i$. Or if you set $\varphi(x_1, ..., x_n) = x_1 x_2$, you get the expectation of the product of the first two variables. This is a very general formula that allows us to compute the expectation of any function of the random variables involved.

{{< callout type="example" >}}
For two dice, what is the expected value of their sum. The joint distribution is:

$$
p(x, y) = \frac{1}{36}, \quad x, y \in \{1, 2, 3, 4, 5, 6\}
$$

The expectation of the sum is:

$$
\begin{align*}
\mathbb{E}[X + Y] &= \sum_{x=1}^6 \sum_{y=1}^6 (x + y) p(x, y) \\
&= \sum_{x=1}^6 \sum_{y=1}^6 (x + y) \cdot \frac{1}{36} \\
&= \frac{1}{36} \sum_{x=1}^6 \sum_{y=1}^6 (x + y) \\
&= \frac{1}{36} \left( \sum_{x=1}^6\sum_{y=1}^6 x + \sum_{x=1}^6\sum_{y=1}^6 y \right) \\
&= \frac{1}{36} \left( \sum_{x=1}^6 6x + \sum_{y=1}^6 6y \right) \\
&= \frac{1}{36} \left( 6 \sum_{x=1}^6 x + 6 \sum_{y=1}^6 y \right) \\
&= \frac{1}{36} \left( 6 \cdot 21 + 6 \cdot 21 \right) \\
&= \frac{1}{36} \cdot 252 \\
&= 7
\end{align*}
$$

Which matches other methods of computing the expected value of the sum of two dice.
{{< /callout >}}

## Independence of Joint Distributions

Independence of random variables has a very specific meaning in the context of joint distributions. As already mentioned the joint distribution captures the relationship between multiple random variables. If two random variables are independent, it means that knowing the value of one does not provide any information about the value of the other. In terms of joint distributions, this means that the joint distribution can be expressed as the product of the marginal distributions. So we say $X$ and $Y$ are **independent** if, for all $x \in W_X$, $y \in W_Y$:

$$
p(x, y) = p_X(x) \cdot p_Y(y)
$$

For $n$ variables this means:

$$
p(x_1, ..., x_n) = \prod_{i=1}^n p_{X_i}(x_i)
$$

{{< callout type="example" >}}
We have already seen this in action with the example of two independent fair coins. The joint distribution is:

$$
p(x, y) = \mathbb{P}(X = x, Y = y) = \frac{1}{4}, \qquad x, y \in \{0,1\}
$$

and the marginals are:

$$
p_X(x) = \mathbb{P}(X = x) = \frac{1}{2}, \quad p_Y(y) = \mathbb{P}(Y = y) = \frac{1}{2}
$$

so multiplying the marginals gives us the joint distribution:

$$
p(x, y) = p_X(x) \cdot p_Y(y) = \frac{1}{2} \cdot \frac{1}{2} = \frac{1}{4}
$$

If we look at the joint distribution of two dependent variables, say $X$ and $Y$ where $Y = X + 1$, we can see that the joint distribution cannot be expressed as the product of the marginals. For example, if $X$ can take values $\{0, 1\}$, then $Y$ can take values $\{1, 2\}$, and the joint distribution would be:

| $X \backslash Y$ | 1   | 2   |
|----------------------|-----|-----|
| 0                    | 1/2 | 0   |
| 1                    | 0   | 1/2 |

where the marginal of $X$ is the same as before and the marginal of $Y$ is:

$$
p_Y(y) = \mathbb{P}(Y = y) = \begin{cases}
\frac{1}{2} & \text{if } y = 1 \\
\frac{1}{2} & \text{if } y = 2 \\
0 & \text{otherwise}
\end{cases}
$$

Then the joint distribution is:

$$
p(x, y) = \mathbb{P}(X = x, Y = y) = \begin{cases}
\frac{1}{2} & \text{if } x = 0, y = 1 \neq p_X(0) \cdot p_Y(1) = \frac{1}{2} \cdot \frac{1}{2} = \frac{1}{4} \\
\frac{1}{2} & \text{if } x = 1, y = 2 \neq p_X(1) \cdot p_Y(2) = \frac{1}{2} \cdot \frac{1}{2} = \frac{1}{4} \\
0 & \text{otherwise}
\end{cases}
$$
{{< /callout >}}

## Continuous Joint Distributions

So far, we've talked about joint distributions for discrete random variables (like dice and coins), where probabilities are assigned to individual outcomes. But what if our random variables are **continuous**, for example the height and weight of a randomly chosen person, or the $x$ and $y$ coordinates of a point randomly dropped inside a region? In this case, the probability of any single, exact outcome is zero, and we describe probabilities in terms of densities over regions.

Suppose we have $n$ continuous random variables $X_1, X_2, \ldots, X_n$, and we want to describe the probability structure of all of them together. We do this with a **joint probability density function (pdf)**:

$$
f(x_1, x_2, \ldots, x_n)
$$

This function describes how probability is "spread out" over $\R^n$. To get the probability that the random vector $(X_1, \ldots, X_n)$ falls in a certain region $A$, we integrate the joint density over that region:

$$
\mathbb{P}\big((X_1, \ldots, X_n) \in A\big) = \int_A f(x_1, \ldots, x_n)\,dx_1 \ldots dx_n
$$

Just as in the discrete case, we can also define a **joint cumulative distribution function (cdf)**:

$$
F(x_1, \ldots, x_n) = \mathbb{P}(X_1 \leq x_1, \ldots, X_n \leq x_n) = \int_{-\infty}^{x_1} \cdots \int_{-\infty}^{x_n} f(t_1, \ldots, t_n) dt_n \cdots dt_1
$$

So the probability that all variables fall below their respective thresholds is given by integrating the density up to those values. You can think of $f(x_1, \ldots, x_n),dx_1 \cdots dx_n$ as the probability that the vector $(X_1, \ldots, X_n)$ falls inside a tiny $n$-dimensional "box" of side lengths $dx_1, ..., dx_n$ around the point $(x_1, ..., x_n)$. Because in the continuous case, the probability of landing at a single point is zero, so only *regions* have positive probability. The density tells us how much probability is packed into different areas, and—crucially—how the variables “move together” (for example, if $X$ is big, does $Y$ also tend to be big, or are they independent?)

{{< callout type="example" >}}
Suppose $(X, Y)$ is a random point uniformly distributed in the square $[0,1]^2$. The joint density is simply

$$
f(x, y) = \begin{cases}
1 & \text{if } (x, y) \in [0, 1]^2 \\
0 & \text{otherwise}
\end{cases}
$$

Here, every point in the square is equally likely. To check that this is a valid density, we integrate over the square:

$$
\mathbb{P}((X, Y) \in [0, 1]^2) = \int_0^1 \int_0^1 1\,dy\,dx = 1
$$

So the total probability is 1. If then want to for example see what is the probability that the point is in the lower left quadrant (where $0 < x < 0.5$ and $0 <y < 0.5$), we compute:

$$
\mathbb{P}(X < 0.5, Y < 0.5) = \int_{0}^{0.5} \int_0^{0.5} 1\,dy\,dx = \int_{0}^{0.5} 1 dy \cdot \int_{0}^{0.5} 1 dx = (0.5 - 0) \cdot (0.5 - 0) = 0.25
$$
{{< /callout >}}

{{< callout type="example" >}}
Now, let's pick a random point uniformly in the unit disk $D = {(x, y) \mid x^2 + y^2 \leq 1}$. The joint density is:

$$
f(x, y) = \begin{cases}
\frac{1}{\pi} & \text{if } x^2 + y^2 \leq 1 \\
0 & \text{otherwise}
\end{cases}
$$

We have $\frac{1}{\pi}$ because the area of the unit disk is $\pi$ and we want the total probability to be 1. We also know that the all points in the disk are equally likely. To check this, we integrate over the disk so we have to fulfill that $c \cdot \pi = 1$ then solving for $c$ the equal probability we get 

$$
\iint_D \frac{1}{\pi} dx\,dy = 1
$$
{{< /callout >}}

### Marginal Densities

Just like with discrete random variables, we often want to focus on just one of the continuous random variables in a joint distribution. The **marginal density** of $X$ is obtained by integrating the joint density over all possible values of $Y$:

$$
f_X(x) = \int_{-\infty}^\infty f(x, y) dy
$$

Intuitively, this collapses the two-dimensional distribution down to just $X$, summing up (integrating) the probabilities for all possible values of $Y$. This is the direct analog of summing rows or columns in the discrete case.

We can again extend this to **marginal densities for $n$ continuous random variables**. The marginal density of $X_i$ in a joint distribution of $n$ variables is obtained by integrating out all other variables:

$$
f_{X_i}(z) = \int_{-\infty}^\infty \cdots \int_{-\infty}^\infty f(x_1, ..., x_{i-1}, z, x_{i+1}, ..., x_n) dx_1 \cdots dx_{i-1} dx_{i+1} \cdots dx_n
$$

So we integrate over all other variables except $X_i$. Again just like in the discrete case, if you have the joint density, you can compute all the marginals by integrating out the other variables. However, if you only have the marginals, you cannot reconstruct the joint density uniquely. This is because the joint density contains information about how the variables depend on each other that is not captured by the marginals alone.

{{< callout type="example" >}}
Back to our example of the random point uniformly distributed in the square $[0,1]^2$. The joint density is $f(x, y) = 1$ for $(x, y) \in [0, 1]^2$. To find the marginal density of $X$, we integrate over $Y$ which takes values in $[0, 1]$:

$$
f_X(x) = \int_0^1 1\,dy = 1 \qquad \text{for } x \in [0,1]
$$

So $X$ is uniformly distributed on $[0,1]$. Same for $Y$. You can think of this as collapsing the square down to a line segment, where every point on that segment is equally likely.
{{< /callout >}}
{{< callout type="example" >}}
For the uniform point in the unit disk the marginal density of $X$ is obtained by integrating the joint density over $Y$:

$$
f_X(x) = \int_{-\sqrt{1-x^2}}^{\sqrt{1-x^2}} \frac{1}{\pi} dy = \frac{2}{\pi} \sqrt{1-x^2}
$$

So the marginal density of $X$ is higher near $x=0$ and lower near $x=\pm1$, reflecting the geometry of the disk as the unit circle is centered around the origin so the values near the center are more likely to be chosen. Similarly for $Y$. 
{{< /callout >}}

### Expectation of Joint Continuous Random Variables

Just like in the discrete case, we usually want to compute the expectation (mean) of a function $\varphi(X_1, \ldots, X_n)$ of our random variables—not just $X$ or $Y$ themselves, but perhaps their sum, product, or something more complicated. For continuous random variables, the expectation is defined by integrating over the joint density:

$$
\mathbb{E}[\varphi(X_1, ..., X_n)] = \int_{-\infty}^\infty \cdots \int_{-\infty}^\infty \varphi(x_1, ..., x_n)\, f(x_1, ..., x_n)\, dx_1 \cdots dx_n
$$

{{< callout type="example" >}}
Let $(X, Y)$ be a random point uniformly distributed in $[0, 1]^2$ (as above). Compute $\mathbb{E}[X]$ and $\mathbb{E}[Y]$:

$$
\mathbb{E}[X] = \int_0^1 \int_0^1 x \cdot 1\,dy\,dx = \int_0^1 x \cdot (1)\,dx = \left[\frac{x^2}{2}\right]_0^1 = \frac{1}{2}
$$

Similarly, $\mathbb{E}[Y] = \frac{1}{2}$. So as expected when we collapse the square down to a line segment, the average value of $X$ and $Y$ is $\frac{1}{2}$ matching the uniform distribution over $[0, 1]$ and the symmetry of the square. If we wanted the expectation of the sum, say $\mathbb{E}[X + Y]$, we would compute:

$$
\mathbb{E}[X + Y] = \int_0^1 \int_0^1 (x + y) \cdot 1\,dy\,dx = \int_0^1 x\,dx + \int_0^1 y\,dy = \frac{1}{2} + \frac{1}{2} = 1
$$
{{< /callout >}}

### Independence for Continuous Random Variables

Just as in the discrete case, two continuous random variables $X$ and $Y$ are **independent** if their joint density factors as the product of their marginals:

$$
f(x, y) = f_X(x) \cdot f_Y(y)
$$

More generally, for $n$ variables:

$$
f(x_1, \ldots, x_n) = \prod_{i=1}^n f_{X_i}(x_i)
$$

This means that knowing the value of one variable gives no information about the others.

{{< callout type="example" >}}
For the uniform point in the square, $f(x, y) = 1$ on $[0,1]^2$, and $f_X(x) = 1$, $f_Y(y) = 1$, so $f(x, y) = f_X(x) f_Y(y)$. Thus, $X$ and $Y$ are independent.
{{< /callout >}}

{{< callout type="example" >}}
For the disk, recall $f(x, y) = \frac{1}{\pi}$ for $x^2 + y^2 \leq 1$, $f_X(x) = \frac{2}{\pi} \sqrt{1-x^2}$, $f_Y(y) = \frac{2}{\pi} \sqrt{1-y^2}$. Here,

$$
f(x, y) \neq f_X(x) f_Y(y)
$$

This means $X$ and $Y$ are **not independent**. For example, if $X$ is close to $1$, $Y$ must be close to $0$. If you know one coordinate, it restricts the other. 
{{< /callout >}}
