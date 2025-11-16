---
title: Random Variables
type: docs
weight: 4
---

So far, we have learned to describe and calculate the probability of events. But in practice, we often want to **quantify the outcomes of random experiments**, not just talk about whether an event happens or not. For example, we might care about "the number of heads" in a sequence of coin tosses, or "the profit" in a gambling game.

This is where **random variables** come in: they provide a systematic way to assign numbers to random outcomes, allowing us to model, analyze, and compute probabilities for quantitative questions. More formally we define a random variable (r.v. or RV) $X$ as a function as follows:

$$
\begin{align*}
X: \Omega &\to \mathbb{R} \\
\omega &\mapsto a \in \mathbb{R}
\end{align*}
$$

where $\Omega$ is the sample space of the random experiment and $a$ is a real number. Importantly, for the random variable to be well defined, the function $X$ must be measurable. In other words, for all $a \in \mathbb{R}$, the following set must be measurable:

$$
\{\omega \in \Omega | X(\omega) \leq a\} \in \mathcal{F}
$$

where measurable means that the set is in the sigma-algebra $\mathcal{F}$ of the sample space. Because it is in the sigma-algebra, it means that we can assign a probability to the event that the random variable takes on a value less than or equal to $a$. This is important because it allows us to use the random variable to model the outcomes of the random experiment in a probabilistic way. Here as always we need to be careful if our set of outcomes is not countable as then the sigma-algebra can get complicated and the definition of the random variable can be more complex. 

Depending on the values of the random variable we can define different types of random variables. For example, if the experiment maps to a finite or countable set of values, we can define a **discrete random variable**. If the experiment maps to an uncountable set of values such as a real interval, we can define a **continuous random variable**. We will see more about this later.

{{< callout type="example" >}}
As a simple example let's look at a fun gambling game we can play with our friends. The game is as follows, we throw a die and if the die shows a 1, 2 or 3 we lose 1 point. If we throw a 4 nothing happens. If we throw a 5 or 6 we win 2 points. We can now define a random variable $X$ to quantify our profit and describe the outcome of the game. The random variable $X$ is defined as follows:

$$
\forall \omega \in \Omega: X(\omega) = \begin{cases}
-1 & \text{ if } \omega = 1, 2, 3 \\
0 & \text{ if } \omega = 4 \\
2 & \text{ if } \omega = 5, 6
\end{cases}
$$

So if we throw a 5 we get $X(5) = 2$ points. If we throw a 1 we get $X(1) = -1$ points. Because the random variables range is finite with only 3 possible values, we have a discrete random variable. 

Importantly we said that the for all $a \in \mathbb{R}$ the set $\{\omega \in \Omega | X(\omega) \leq a\}$ must be in our sigma-algebra. So depending on our sigma-algebra a random variable can be well defined for it or not. Consider the following sigma-algebra:

- $\mathcal{F}_1 = \mathcal{P}(\Omega)$, the power set of $\Omega$. This is the largest sigma-algebra and contains all possible events.
- $\mathcal{F}_2 = \{\emptyset, \Omega\}$, the trivial sigma-algebra. This is the smallest sigma-algebra and contains only the empty set and the whole sample space.
- $\mathcal{F}_3 = \{\emptyset, \{1, 2, 3\}, \{4, 5, 6\}, \Omega\}$.
- $\mathcal{F}_4 = \{\emptyset, \{1, 2, 3\}, \{1, 2, 3, 4\}, \{4, 5, 6\}, \{5, 6\}, \{1, 2, 3, 5, 6\}, \{4\}, \Omega\}$.

If we collect the outcomes for all the different random variable values we get the following sets:

$$
\{\omega \in \Omega | X(\omega) \leq a\} = \begin{cases}
\emptyset & \text{ if } a < -1 \\
\{1, 2, 3\} & \text{ if } -1 \leq a < 0 \\
\{1, 2, 3, 4\} & \text{ if } 0 \leq a < 2 \\
\{1, 2, 3, 4, 5, 6\} & \text{ if } a \geq 2
\end{cases}
$$

So we notice that the random variable is not well defined for $\mathcal{F}_2$ as the set $\{1, 2, 3\}$ is not in the sigma-algebra and not for $\mathcal{F}_3$ as the set $\{1, 2, 3, 4\}$ is not in the sigma-algebra. However, it is well defined for $\mathcal{F}_1$ and $\mathcal{F}_4$. So we can use the random variable $X$ to quantify our profit in the game if we use these sigma-algebras. In most cases we will use the power set of the sample space as our sigma-algebra, so we can use any random variable we want. However, it is important to keep in mind that the random variable must be well defined for the sigma-algebra we are using and that we can not always use the power set as our sigma-algebra.
{{< /callout >}}

{{< callout type="example" title="Indicator Variables" >}}
If we have a specific event $A$ in our sample space $\Omega$, we can also define a special random variable called an **indicator variable**. This variable "indicates" if an outcome belongs to an event: 1 for "yes", 0 for "no". The indicator variable is defined as follows:

$$
\forall \omega \in \Omega: 1_A(\omega) = \begin{cases}
1 & \text{ if } \omega \in A \\
0 & \text{ if } \omega \notin A
\end{cases}
$$

Again we only have two possible values for the indicator variable, 0 and 1, hence we have a discrete random variable. If we also analyze the set $\{\omega \in \Omega | 1_A(\omega) \leq a\}$ we get the following sets:

$$
\{\omega \in \Omega | 1_A(\omega) \leq a\} = \begin{cases}
\emptyset & \text{ if } a < 0 \\
A^c & \text{ if } 0 \leq a < 1 \\
\Omega & \text{ if } a \geq 1
\end{cases}
$$

So we notice that for the indicator variable to be well defined we need to have the event $A$ in our sigma-algebra as by the definition of a sigma-algebra we need to have the event $A^c$ in our sigma-algebra as well. So if we have a sigma-algebra that does not contain the event $A$, we can not use the indicator variable to quantify the event. 
{{< /callout >}}

## Probability of Random Variables

So far, we have defined what a random variable is. Next, we'll see how we use random variables to define probabilities and distributions. Specifically we have seen that a random variable is a function that assigns a numerical value to each outcome of a random experiment and we have seen that we can assign a probability to an event. In this section we want to combine these two concepts and assign a probability to a random variable. We have already seen that for all values $a \in \mathbb{R}$ of a random variable $X$ we can define the following set:

$$
\{\omega \in \Omega | X(\omega) \leq a\} \in \mathcal{F}
$$

This set contains all the outcomes of the random experiment for which the random variable $X$ takes on a value less than or equal to $a$. Because it is a set of outcomes, i.e. an event it means we can assign a probability to it. We can do this by using the probability measure $P$ of the sigma-algebra $\mathcal{F}$. More specifically the probability of the random variable $X$ taking on a value less than or equal to $a$ is defined as follows:

$$
\mathbb{P}(\{\omega \in \Omega | X(\omega) \leq a\})
$$

To simplify the notation we usually omit the dependence on the outcome $\omega$ and the brackets and just simply write:

$$
\mathbb{P}(X \leq a)
$$

This is the probability of the random variable $X$ taking on a value less than or equal to $a$. We can also extend this notation to the case where the random variable $X$ takes on a value in an interval $(a, b]$:

$$
\mathbb{P}(X \in (a, b]) = \mathbb{P}(a < X \leq b) = \mathbb{P}(\{\omega \in \Omega | a < X(\omega) \leq b\})
$$

Or also more rarely used to the case where the random variable $X$ takes on a value in some set $A$:

$$
\mathbb{P}(X \in A) = \mathbb{P}(\{\omega \in \Omega | X(\omega) \in A\})
$$

{{< callout type="todo" >}}
Are these valid definitions of the probability of a random variable?
What about equals, less than, greater than, etc?

Are the examples below correct?
{{< /callout >}}

{{< callout type="example" >}}
If we go back to our example of the gambling game we can now calculate the different probabilities of the random variable $X$ taking on a value less than or equal to $a$. We defined the random variable $X$ as follows:

$$
\forall \omega \in \Omega: X(\omega) = \begin{cases}
-1 & \text{ if } \omega = 1, 2, 3 \\
0 & \text{ if } \omega = 4 \\
2 & \text{ if } \omega = 5, 6
\end{cases}
$$

We have also already analyzed the different sets of outcomes for the different values of $a$:

$$
\{\omega \in \Omega | X(\omega) \leq a\} = \begin{cases}
\emptyset & \text{ if } a < -1 \\
\{1, 2, 3\} & \text{ if } -1 \leq a < 0 \\
\{1, 2, 3, 4\} & \text{ if } 0 \leq a < 2 \\
\{1, 2, 3, 4, 5, 6\} & \text{ if } a \geq 2
\end{cases}
$$

Because when throwing a die we have a laplace experiment we know that the probability of each outcome is $\frac{1}{6}$. So we can now calculate the different probabilities:

$$
\begin{align*}
\mathbb{P}(X < -1) &= \mathbb{P}(\emptyset) = 0 \\
\mathbb{P}(-1 \leq X < 0) &= \mathbb{P}(\{1, 2, 3\}) = \frac{3}{6} = \frac{1}{2} \\
\mathbb{P}(0 \leq X < 2) &= \mathbb{P}(\{1, 2, 3, 4\}) = \frac{4}{6} = \frac{2}{3} \\
\mathbb{P}(X \geq 2) &= \mathbb{P}(\{1, 2, 3, 4, 5, 6\}) = \frac{6}{6} = 1 \\
\end{align*}
$$
{{< /callout >}}
{{< callout type="example" >}}
We can also do the same and define the probability of an indicator variable for an event $A$. 

$$
\begin{align*}
\mathbb{P}(1_A \leq 0) = \mathbb{P}(\{\omega \in \Omega | 1_A(\omega) \leq 0\}) = \mathbb{P}(A^c) = 1 - \mathbb{P}(A) \\
\mathbb{P}(1_A \leq 1) = \mathbb{P}(\{\omega \in \Omega | 1_A(\omega) \leq 1\}) = \mathbb{P}(A) \\
\mathbb{P}(1_A \leq 2) = \mathbb{P}(\{\omega \in \Omega | 1_A(\omega) \leq 2\}) = 1 \\
\end{align*}
$$
{{< /callout >}}

### Almost Sure Events

So far we have said that an event $A$ is a sure event if it contains all the outcomes of the sample space $\Omega$ and therefore has the probability $\mathbb{P}(A) = 1$. However, we can also define a different type of sure event called an **almost sure event**, commonly abbreviated as "a.s.". An event $A$ is called almost surely if the following holds:

$$
\mathbb{P}(A) = 1
$$

Therefore an equivalent definition of an almost sure event $A$ is also if the complementary event $A^c$ has the following probability:

$$
\mathbb{P}(A^c) = 0
$$

Now you might be thinking well what is the difference between a sure event and an almost sure event? The difference is that a sure event contains all the outcomes of the sample space $\Omega$ and therefore has the probability $\mathbb{P}(A) = 1$. An almost sure event on the other hand does not contain all the outcomes of the sample space $\Omega$ but has the probability $\mathbb{P}(A) = 1$. This means that an almost sure event can be a subset of the sample space $\Omega$ but still have the probability $\mathbb{P}(A) = 1$. All the outcomes that are not in the event $A$ then just have a probability of 0. Another way you can think about it is that in an infinite sample spaces, events of probability zero can still be possible, but just extremely unlikely (e.g., picking a single real number at random from [0,1]: the chance of hitting any particular number is zero, but it's not impossible). So picking a single outcome from an infinite sample space is almost surely not going to happen, but it is still possible.

{{< callout type="example" title="Tossing a Coin Infinitely Many Times" >}}
Suppose we flip a fair coin infinitely many times and we consider the event $A$ that "at least one head appears" in the sequence of tosses.

Let's compute the probability of the complementary event $A^c$: the event that "no heads ever appear" (i.e., we get only tails forever). For each toss, the probability of getting tails is $\frac{1}{2}$, so for $n$ tosses in a row, the probability of getting all tails is $\left(\frac{1}{2}\right)^n$. In the limit as $n \to \infty$, the probability of getting only tails forever is:

$$
\lim_{n \to \infty} \left(\frac{1}{2}\right)^n = 0
$$

Therefore, the probability of getting at least one head at some point is:

$$
\mathbb{P}(\text{at least one head appears}) = 1 - 0 = 1
$$

So, **almost surely** we will see at least one head in an infinite sequence of tosses, even though there is technically one outcome (all tails) with probability zero where this does not happen. So in infinite sequences, it's possible (but with probability zero) that no head ever appears, but in practice, we are "certain" to see at least one head if we keep tossing forever. This is an almost sure event. 
{{< /callout >}}

{{< callout type="example" title="Infinite Monkey Theorem" >}}
The **Infinite Monkey Theorem** says that if a monkey randomly hits keys on a typewriter for an infinite amount of time, then almost surely the monkey will eventually type out the complete works of Shakespeare (or any given text).

Suppose the keyboard has $k$ possible symbols (e.g., $k=26$ for just lowercase English letters). Suppose the text we want to see is $n$ characters long. The probability that the monkey types the target text in a particular sequence of $n$ keystrokes is $\frac{1}{k^n}$ as each character has a $\frac{1}{k}$ chance of being correct and he needs to get all $n$ characters right.

Suppose $A$ be the event that the monkey types the target text **at least once** somewhere in the infinite sequence. Then:
* The probability that the monkey does **not** type the text in the first $n$ positions is $1 - \frac{1}{k^n}$.
* The probability that the monkey does **not** type the text in the first $m$ blocks of length $n$ is $\left(1 - \frac{1}{k^n}\right)^m$. The idea here is that each block of $n$ characters is independent, so we multiply the probabilities.

As $m \to \infty$, this probability goes to 0:

$$
\lim_{m \to \infty} \left(1 - \frac{1}{k^n}\right)^m = 0
$$

So the probability that the monkey never types the target text is 0. Therefore, the probability that the monkey does type the text at least once (at some point in the infinite sequence) is 1. This is an **almost sure event**.

Even though it is staggeringly unlikely for a monkey to type Shakespeare in any finite period, if you give infinite time, the probability becomes 1 (almost sure), except for a set of probability zero where the monkey never produces the text. 
{{< /callout >}}

Because of this definition on some event we can also extend the idea to random variables. We can for example say that $X \leq a$ almost surely if the following holds:

$$
\mathbb{P}(X \leq a) = 1
$$

This means that the set of outcomes for which the random variable $X$ takes on a value less than or equal to $a$ is almost surely. In other words, the set of outcomes for which the random variable $X$ takes on a value less than or equal to $a$ has the probability $\mathbb{P}(X \leq a) = 1$. 

The same can be done for two random variables $X$ and $Y$. We can say that $X \leq Y$ almost surely if the following holds:

$$
\mathbb{P}(X \leq Y) = 1
$$

The definition of this set of outcomes is a bit more complex as we have two random variables. However, an example of this would be if we defined the random variable $X$ as the outcome of a die throw and the random variable $Y$ boolean value that is true if the die throw is even and false if the die throw is odd. 

## Discrete Random Variables

We have already seen that if a random variable $X$ takes on a finite or countable set of values, we can define a **discrete random variable**. Common examples of discrete random variables are throwing a die, flipping a coin, or counting the number of something happening. We will go into more detail about some special discrete random variables later that have earned their own names.

### Probability Mass Function (PMF)

If the random variable is discrete, so in other words the range of the random variable are countable, we can define the **probability mass function** short PMF, or sometimes also called the **density function** or just simply **discrete distribution of $X$**. The PMF is a function that assigns a probability to each value of the random variable. This means that the PMF maps a value of the range of the random variable to a probability between 0 and 1. More formally we the PMF maps the values as follows:

$$
\begin{align*}
p_X: W &\to [0, 1] \\
a &\mapsto p_X(a)
\end{align*}
$$

The PMF is defined for all values in the countable range $a \in W$ of the random variable $X$ and is defined as follows:

$$
p_X(a) = \mathbb{P}(X = a) = \mathbb{P}(\{\omega \in \Omega | X(\omega) = a\}) = \mathbb{P}(X \in \{a\})
$$
    
where $a$ is a value of the random variable $X$. Sometimes if the random variable is clear from the context we just write $p(a)$. Notice that we orignally defined the random variable $X$ to have a range $a \in \mathbb{R}$, but now we are defining the PMF to only assign probabilities to values in finite or countable sets $W$. This is because the PMF is only defined for discrete random variables, so it only assigns probabilities to values that the random variable can take on. You can think of the PMF as a function that is also defiend for all values in the range of the random variable $X$, but it only assigns a nonzero probability to values that are actually in the range of $X$. In other words, if $a$ is not in the range of $X$, then $p_X(a) = 0$.

Because the PMF is a function that assigns probabilities, it has very similar properties to a probability measure, one of which is that the sum of all probabilities is equal to 1 so we have:

$$
\sum_{a \in X} p_X(a) = 1
$$

{{< callout type="proof" >}}
Let $W$ be the (at most countable) set of all values that $X$ can take. For each $a \in W$, we define the event:

$$
A_a := \{\omega \in \Omega \mid X(\omega) = a\}
$$

This is the set of all outcomes $\omega$ in the sample space $\Omega$ such that the random variable $X$ takes the value $a$. These events are **pairwise disjoint** because if we take two different values $a$ and $b$ in $W$ then the events $A_a$ and $A_b$ are disjoint:

$$
A_a \cap A_b = \emptyset \quad \text{for } a \neq b
$$

This is because if $X(\omega) = a$ for some outcome $\omega$, then $X(\omega)$ cannot simultaneously equal $b$ if. Moreover, since $W$ is the range of $X$, the union of all these events covers the entire sample space:

$$
\bigcup_{a\in W} A_a = \Omega
$$

because for every $\omega \in \Omega$, $X(\omega) = a$ for exactly one $a \in W$. By **countable additivity** of the probability measure we then have:

$$
\mathbb{P}(\Omega) = \mathbb{P}\left(\bigcup_{a\in W} A_a\right) = \sum_{a\in W} \mathbb{P}(A_a)
$$

But by definition, $\mathbb{P}(A_a) = \mathbb{P}(X = a) = p_X(a)$. Thus,

$$
1 = \sum_{a\in W} p_X(a)
$$
{{< /callout >}}

Because it is discrete the graph of the PMF is a series of discrete points, this is commonly shown using a table or a bar graph. 

{{< figure
  src="/images/maths/probabilityPMF.png"
  alt="The bar graph of the PMF of the random variable $X$."
  caption="The bar graph of the PMF of the random variable $X$."
  width="500"
>}}

{{< callout type="info" >}}
It is important to remember that the PMF is only defined for discrete random variables. For continuous random variables we will define a different function called the **probability density function** (PDF) which we will see later. You might be thinking why does this not work for continuous random variables? The reason is rather simple and comes from the definition of a probability measure and the behavior of summing over an uncountable set. 

This has briefly been hit on when we [discuss why we need a sigma-algebra](/maths/probabilityStatistics/probabilitySpaces#sigma-algebras) and the fact that we can not assign a probability to an uncountable set.

We will also see this in more detail when we discuss the PDF and the CDF of continuous random variables.
{{< /callout >}}

### Cumulative Distribution Function (CDF)

For any random variable $X$ we can define the **cumulative distribution function**, short CDF or sometimes also called the **distribution function**. Again the CDF is a function that assigns a probability to the random variable $X$ taking on a value less than or equal to $a$. The CDF is defined for all values of $a \in \mathbb{R}$ and is defined as follows:

$$
\begin{align*}
F_X: \mathbb{R} &\to [0, 1] \\
a &\mapsto F_X(a)
\end{align*}
$$

Importantly the difference between the PMF and the CDF is that the CDF assigns a probability to the random variable $X$ taking on a value less than or equal to $a$ and not just equal to $a$. This also means that it assigns a probability to the entire range of the random variable $X$ and not just to a single value. More formally we can define the CDF as follows:

$$
F_X(a) = \mathbb{P}(X \leq a) = \mathbb{P}(\{\omega \in \Omega | X(\omega) \leq a\})
$$

We will first focus on the case where the random variable is discrete. In this case we can define the CDF as follows:

$$
\mathbb{P}(X \leq a) = F_X(a) = \sum_{y \in X, y\leq a} p_X(y) = \sum_{y \in X, y \leq a} \mathbb{P}(X = y)
$$

Where we sum the probabilities over all values of the random variable $X$ that are less than or equal to $a$. Intuitively this makes sense.

{{< callout type="proof" >}}
We want to show that the CDF can be expressed as a sum of the PMF over all values less than or equal to $a$.
Let $X$ be a discrete random variable with range $W$. By definition, $F_X(a) = \mathbb{P}(X \leq a)$. Let's express this probability as a sum over the values of $X$: 

$$
\begin{align*}
F_X(a) &= \mathbb{P}(X \leq a) \\
&= \mathbb{P}(\{\omega \in \Omega \mid X(\omega) \leq a\}) \\
&= \mathbb{P}\left(\bigcup_{y \in W,\, y \leq a} \{\omega \mid X(\omega) = y\}\right)
\end{align*}
$$

The sets $\{\omega \mid X(\omega) = y\}$ for different $y \in W$ are **disjoint** (since $X$ can only take one value at a time). By countable additivity,

$$
\begin{align*}
F_X(a) &= \mathbb{P}\left(\bigcup_{y \in W,\, y \leq a} \{\omega \mid X(\omega) = y\}\right) \\
&= \sum_{y \in W,\, y \leq a} \mathbb{P}(X = y) \\
&= \sum_{y \in W,\, y \leq a} p_X(y)
\end{align*}
$$
{{< /callout >}}

{{< figure
  src="/images/maths/probabilityCDF.png"
  alt="The CDF of the random variable $X$."
  caption="The CDF of the random variable $X$."
  width="500"
>}}

We can see that the CDF is a step function that increases at the points where the random variable $X$ takes on a value. This is because the CDF is defined as the sum of the probabilities of all values less than or equal to $a$. You can think of the CDF as the accumulated height of the PMF. 

Because it is a step function the CDF is a non-decreasing function, meaning that it can only stay the same or increase as $a$ increases.This comes from the fact that if $a \leq b$ then $\{\omega \in \Omega | X(\omega) \leq a\} \subseteq \{\omega \in \Omega | X(\omega) \leq b\}$, so we have:

$$
\mathbb{P}(X \leq a) \leq \mathbb{P}(X \leq b)
$$

If we have $a < b$ then we can also combine this to get the probability of the random variable $X$ taking on a value in the interval $(a, b]$:

$$
\mathbb{P}(a < X \leq b) = F_X(b) - F_X(a)
$$

This is the so called **basic identity** and can directly be proven by reordering the disjoint union:

$$
\begin{align*}
\{X \leq b\} &= \{X \leq a\} \cup \{a < X \leq b\} \\
\mathbb{P}(X \leq b) &= \mathbb{P}(\{X \leq a\} \cup \{a < X \leq b\}) \\
F_X(b) &= F_X(a) + \mathbb{P}(a < X \leq b) \\
\mathbb{P}(a < X \leq b) &= F_X(b) - F_X(a)
\end{align*}
$$

{{< callout type="example" >}}
If we go back to our example of the gambling game we already saw some of the probabilities of the random variable $X$ taking on a value less than or equal to $a$. We can now define the CDF for the random variable $X$ for all values of $a \in \mathbb{R}$:

$$
F_X(a) = \begin{cases} 
0 & \text{ if } a < -1 \\
\frac{1}{2} & \text{ if } -1 \leq a < 0 \\
\frac{2}{3} & \text{ if } 0 \leq a < 2 \\
1 & \text{ if } a \geq 2
\end{cases}
$$

{{< callout type="todo" >}}
Add the plot of the CDF here.
{{< /callout >}}

We can also see the basic identity in effect. For example if we want to know the probability that we get points, so $X > 0$, we can use the basic identity with $a = 0$ and $b = 2$:

$$
\mathbb{P}(X > 0) = \mathbb{P}(0 < X \leq 2) = F_X(2) - F_X(0) = 1 - \frac{2}{3} = \frac{1}{3}
$$

This also matches with our intuition as we have 2 outcomes that give us points and 4 outcomes that do not give us points. So the probability of getting points is $\frac{2}{6} = \frac{1}{3}$. Visually this can be interpreted as taking the part of the function from the left to the point $b$ and then subtracting the part of the function from the left to the point $a$. 
{{< /callout >}}
{{< callout type="example" >}}
Again we can also do the same and define the CDF for an indicator variable for an event $A$. 

$$
F_X(a) = \begin{cases}
0 & \text{ if } a < 0 \\
1 - \mathbb{P}(A) & \text{ if } 0 \leq a < 1 \\
1 & \text{ if } a \geq 1
\end{cases}
$$
{{< /callout >}}

From the example above and the plot we notice that for a function to be a CDF it fulfills three properties:
1. $F_X(a)$ is a non-decreasing function. 
2. $F_X(a)$ is a right-continuous function.
3. As the variable value $a$ goes to infinity the CDF goes to 1 and as $a$ goes to minus infinity the CDF goes to 0. 

The first property is rather trivial and comes from the fact that probabilities are always between 0 and 1 and that we are looking at values less than or equal to $a$. 

{{< callout type="proof" >}}
There are many ways to prove this but one is by monotonicity of the probability measure. If $a \leq b$ we have $\{\omega \in \Omega | X(\omega) \leq a\} \subseteq \{\omega \in \Omega | X(\omega) \leq b\}$, so we have:

$$
\mathbb{P}(X \leq a) \leq \mathbb{P}(X \leq b) \iff F_X(a) \leq F_X(b)
$$
{{< /callout >}}

The second property is important because it means that the CDF is continuous from the right. This means it can have jumps but that the limits from the right exist and are equal to the value of the function at that point, more formally we can say that for all $a \in \mathbb{R}$ and $b \geq a$ we have:

$$
F_X(a) = \lim_{b \to a^+} F_X(b)
$$

{{< callout type="todo" >}}
Show in a better way that the CDF is right-continuous and that at the jumps the CDF is equal to the PMF.

This helps us to understand the behavior of the CDF at points where the random variable has a probability mass. For discrete random variables, this means that the CDF is constant (flat) between the jumps at points $a \in W$, and jumps by $p_X(a)$ at $a$. These jumps correspond to the PMF values at those points.

For a continous random variable, the CDF is continuous everywhere and does not have jumps, but it still fulfills the right-continuity property. This is then why at a single point is zero and the CDF is continuous everywhere else.
{{< /callout >}}

{{< callout type="proof" >}}
Let $X$ be any random variable and $F_X(a) = \mathbb{P}(X \leq a)$. We claim that for any $a \in \mathbb{R}$, the CDF is right-continuous:

$$
F_X(a) = \lim_{h \to 0} F_X(a + h)
$$

Let $h \to 0^+$, then we can write:

$$
\{X \leq a\} = \bigcap_{n=1}^{\infty} \{X \leq a + h\}
$$

since as $h \to 0$, the sets on the right "shrink" to exactly ${X \leq a}$.

By the **continuity from above** property of probability measures (for decreasing sequences of sets) we have:

$$
\mathbb{P}\left(\bigcap_{n=1}^\infty A_n\right) = \lim_{n \to \infty} \mathbb{P}(A_n)
$$

So putting this together, we get:

$$
F_X(a) = \lim_{h \to 0^+} F_X(a + h) = \lim_{h \to 0^+} \mathbb{P}(X \leq a + h) = \mathbb{P}(X \leq a)
$$

So the CDF is right-continuous at every point $a$. But for discrete random variables, this means that the CDF is constant (flat) between the jumps at points $a \in W$, and jumps by $p_X(a)$ at $a$. These jumps correspond to the PMF values at those points.
{{< /callout >}}

The last property in a way is like with the PMF, to guarantee that the CDF is a valid probability measure. For this to be the case the values of the CDF must be bounded between 0 and 1. This means that the CDF must go to 1 as $a$ goes to infinity and to 0 as $a$ goes to minus infinity. This is important so it can be used to assign probabilities to events.

{{< callout type="proof" >}}
We want to show that the CDF approaches 0 as $a \to -\infty$ and approaches 1 as $a \to \infty$. First let's consider the limit as $a \to -\infty$. For any $\omega \in \Omega$, $X(\omega) \leq a$ is **never true** for $a$ less than the minimum value in $W$. Thus,

$$
F_X(a) = 0 \text{ for } a < \min W
$$

So as $a \to -\infty$, we have:

$$
\lim_{a \to -\infty} F_X(a) = 0
$$

As $a \to \infty$ for large enough $a$, $X(\omega) \leq a$ is **always true** (for all $X(\omega)$), so $F_X(a) = \mathbb{P}(\Omega) = 1$ for $a$ greater than the maximum value in $W$. So,

$$
\lim_{a \to \infty} F_X(a) = 1
$$

So the CDF "starts" at 0 (for very small $a$) and "ends" at 1 (for very large $a$), as the probability that $X$ is less than or equal to $a$ transitions from impossible to certain. 
{{< /callout >}}

## Properties of Random Variables

### Conditional Random Variables

Just as we can talk about the probability of an event given another event (i.e., [conditional probability]()), we can also consider the probability distribution of a random variable given that some event has occurred. This leads us to the idea of a **conditional random variable**. Think of this as "restricting our attention" to a subset of the sample space: only those outcomes that make $B$ true. We then ask: "within that world, how does $X$ behave?"


More formally we can define a conditional random variable $X$ given an event $B$. The idea is that we are looking at the distribution of $X$ but only considering the outcomes where $B$ occurs. So given an event $B$ with $\mathbb{P}(B) > 0$, we can define the **conditional probability** of $X$ taking a value $a$, given $B$:

$$
\mathbb{P}(X = a \mid B) = \frac{\mathbb{P}(X = a \cap B)}{\mathbb{P}(B)}
$$

Alternatively, in terms of the PMF, the **conditional PMF** of $X$ given $B$ is:

$$
p_{X|B}(a) = \mathbb{P}(X = a \mid B) = \frac{\mathbb{P}(X = a \text{ and } B)}{\mathbb{P}(B)}
$$

This describes the "new" probability distribution of $X$ if we know that $B$ occurred. You can think of this as a new random variable, but with the probability measure changed to reflect the conditioning on $B$.

{{< callout type="example" >}}
Suppose we go back to our gambling game example with the random variable $X$ defined as:

$$
\forall \omega \in \Omega: X(\omega) = \begin{cases}
-1 & \text{if } \omega = 1, 2, 3 \\
0 & \text{if } \omega = 4 \\
2 & \text{if } \omega = 5, 6
\end{cases}
$$

Let $B$ be the event "the die roll is even" ($B = {2, 4, 6}$). Then what is the conditional distribution of $X$ given $B$? The event $B$ has outcomes $\omega = 2, 4, 6$, so $\mathbb{P}(B) = \frac{3}{6} = \frac{1}{2}$.

Now, let's compute the conditional PMF:

- $\mathbb{P}(X = -1 \mid B)$: Only $\omega = 2$ contributes (since $2 \in B$ and $X(2) = -1$), so

$$
\mathbb{P}(X = -1 \mid B) = \frac{\mathbb{P}(\{2\})}{\mathbb{P}(B)} = \frac{\frac{1}{6}}{\frac{1}{2}} = \frac{1}{3}
$$

- $\mathbb{P}(X = 0 \mid B)$: Only $\omega = 4$ (since $4 \in B$ and $X(4) = 0$):

$$
\mathbb{P}(X = 0 \mid B) = \frac{\frac{1}{6}}{\frac{1}{2}} = \frac{1}{3}
$$

- $\mathbb{P}(X = 2 \mid B)$: Only $\omega = 6$ (since $6 \in B$ and $X(6) = 2$):

$$
\mathbb{P}(X = 2 \mid B) = \frac{\frac{1}{6}}{\frac{1}{2}} = \frac{1}{3}
$$

So, conditional on rolling an even number, the random variable $X$ is equally likely to be $-1$, $0$, or $2$.

Similarly, we can define the **conditional cumulative distribution function** (CDF):

$$
F_{X|B}(a) = \mathbb{P}(X \leq a \mid B)
$$

For our example, for $a = 0$ the outcomes in $B$ where $X \leq 0$ are $\omega = 2$ ($X = -1$) and $\omega = 4$ ($X = 0$), so:

$$
F_{X|B}(0) = \mathbb{P}(X \leq 0 \mid B) = \frac{\mathbb{P}(\{2, 4\})}{\mathbb{P}(B)} = \frac{2/6}{1/2} = \frac{2}{3}
$$

This means that given we rolled an even number, the probability that $X$ is less than or equal to 0 is $\frac{2}{3}$.
{{< /callout >}}

### Transforming Random Variables

Often we want to study **functions of random variables**. This might be to rescale, shift, combine, or otherwise manipulate random variables. The resulting object is always **another random variable**. This is very useful for example to model **functions** of measurements (e.g., profit as a function of quantity sold), to construct new statistics (e.g., the sample mean is a function of several $X_i$ or to analyze joint behavior (e.g., sums of random variables). This becomes especially clear in the [estimator](/garden/maths/probabilityStatistics/estimators) section.


So suppose $X: \Omega \to W$ is a random variable and $g:W \to \mathbb{R}$ is a (measurable) function. Then $Y = g(X)$ is a new random variable defined by:

$$
Y(\omega) = g(X(\omega)) \qquad \text{for all } \omega \in \Omega
$$

The range of $Y$ is $g$ applied to the range of $X$. So in other words the possible values of $Y$ are $g(a)$ for all $a$ in the range of $X$. The probability distribution of $Y$ can be derived from the distribution of $X$ by summing over all values of $X$ that map to a given value of $Y$. So for a discrete random variable $X$ and $Y = g(X)$, for any $b$ in the range of $Y$ we can define the probability mass function (PMF) of $Y$ as follows:

$$
\mathbb{P}(Y = b) = \sum_{a: g(a) = b} \mathbb{P}(X = a)
$$

The idea is that we sum over all values $a$ in the range of $X$ that map to the value $b$ under the function $g$. This gives us the total probability of $Y$ taking on the value $b$.

{{< callout type="proof" >}}
Let $X$ be a discrete random variable i.e., $X$ maps each $\omega \in \Omega$ to one of the countable values in $W$ (or in the continuous case $W = \mathbb{R}$), and let $g: W \to \mathbb{R}$ be any function. Define $Y = g(X)$, i.e., $Y(\omega) = g(X(\omega))$. To show $Y$ is a random variable, recall a function $Y: \Omega \to \mathbb{R}$ is a random variable if, for every $a \in \mathbb{R}$, the set ${\omega \in \Omega : Y(\omega) \leq a}$ is a valid event (i.e., is in $\mathcal{F}$). For discrete $Y$, this set can be written as:

$$
\{\omega \in \Omega : Y(\omega) \leq a\} = \{\omega : g(X(\omega)) \leq a\}
$$

But $g(X(\omega))$ only takes values in the (at most countable) set $V = {g(w) : w \in W}$. Thus:

$$
\{\omega : g(X(\omega)) \leq a\} = \bigcup_{b \in V,\, b \leq a} \{\omega : g(X(\omega)) = b\}
$$

But ${\omega : g(X(\omega)) = b} = {\omega : X(\omega) = c}$ for all $c \in W$ with $g(c) = b$. So:

$$
\{\omega : Y(\omega) \leq a\} = \bigcup_{\substack{c \in W\\g(c) \leq a}} \{\omega : X(\omega) = c\}
$$

But for any $c \in W$, the set ${\omega : X(\omega) = c}$ is in $\mathcal{F}$ (since $X$ is a random variable and $W$ is countable). Therefore, the union is in $\mathcal{F}$ as sigma-algebras are closed under countable unions.

Therefore $Y = g(X)$ is a (discrete) random variable. 
{{< /callout >}}

{{< callout type="example" >}}
Suppose $X$ is the number on a die, and $Y = X \bmod 2$ (i.e., whether the number is odd or even). Then $Y$ takes values $0$ (even) or $1$ (odd).  

- $\mathbb{P}(Y = 0) = \mathbb{P}(X = 2) + \mathbb{P}(X = 4) + \mathbb{P}(X = 6) = \frac{3}{6} = \frac{1}{2}$
- $\mathbb{P}(Y = 1) = \mathbb{P}(X = 1) + \mathbb{P}(X = 3) + \mathbb{P}(X = 5) = \frac{1}{2}$

So, $Y$ is a random variable indicating whether the die is even or odd, and its distribution is derived from $X$. 
{{< /callout >}}

{{< callout type="example" >}}
Suppose $X$ is the number shown on a die, and let $Y = X^2$. $Y$ is a new random variable:

* For $\omega = 1$, $Y(1) = 1$
* For $\omega = 2$, $Y(2) = 4$
* For $\omega = 3$, $Y(3) = 9$
* etc.

The PMF of $Y$ is given by:

$$
\mathbb{P}(Y = y) = \sum_{x: g(x) = y} \mathbb{P}(X = x)
$$

For example, $\mathbb{P}(Y = 4) = \mathbb{P}(X = 2) = \frac{1}{6}$.
{{< /callout >}}

Just as we can apply a function to a single random variable to create a new one, we can also apply a function of **several random variables** to create a new random variable. 

Suppose we have two (or more) random variables $X$ and $Y$ on the same probability space, and we have a function $g: \mathbb{W}^2 \to \mathbb{R}$ (or in general $g: \mathbb{R}^n \to \mathbb{R}$). We define a new random variable $Z$ by:

$$
Z(\omega) = g(X(\omega), Y(\omega)) \qquad \text{for all } \omega \in \Omega
$$

The possible values of $Z$ are given by all values $g(a, b)$ where $a$ is in the range of $X$ and $b$ in the range of $Y$. $Z$ is a random variable because, for any $c \in \mathbb{R}$, the set ${\omega \mid Z(\omega) \leq c} = {\omega \mid g(X(\omega), Y(\omega)) \leq c}$ can be written in terms of the events associated with $X$ and $Y$.

So for example, we could take:

- $g(x, y) = x + y$ (sum)
- $g(x, y) = \max(x, y)$ (maximum)
- $g(x, y) = x \cdot y$ (product)
- or any other function that takes two numbers and produces another.

This construction allows us to create new random variables out of existing ones, and analyze things like the sum, difference, or maximum of two measurements.

{{< callout type="example" >}}
Suppose $X$ and $Y$ are the outcomes of two independent dice. We can define a new random variable $Z = X + Y$. For every outcome $(x, y)$, $Z(x, y) = x + y$. This random variable represents the sum of the dice. The PMF of $Z$ is computed by considering all pairs $(x, y)$ such that $x + y = z$:

$$
p_Z(z) = \mathbb{P}(X + Y = z) = \sum_{\substack{x, y\\ x + y = z}} \mathbb{P}(X = x)\mathbb{P}(Y = y)
$$

We can compute this for each possible value of $z$, which ranges from 2 (1+1) to 12 (6+6):
$$
p_Z(z) = \begin{cases}
0 & \text{if } z < 2 \text{ or } z > 12 \\
\frac{1}{36} & \text{if } z = 2 \text{ (1,1)} \\
\frac{2}{36} & \text{if } z = 3 \text{ (1,2), (2,1)} \\
\frac{3}{36} & \text{if } z = 4 \text{ (1,3), (2,2), (3,1)} \\
\vdots & \\
\frac{6}{36} & \text{if } z = 7 \text{ (1,6), (2,5), (3,4), (4,3), (5,2), (6,1)} \\
\vdots & \\
\frac{3}{36} & \text{if } z = 10 \text{ (4,6), (5,5), (6,4)} \\
\frac{2}{36} & \text{if } z = 11 \text{ (5,6), (6,5)} \\
\frac{1}{36} & \text{if } z = 12 \text{ (6,6)}
\end{cases}
$$

So we can see the most likely outcome when rolling two dice is 7, which has the highest probability of occurring.
{{< /callout >}}

### Independent Random Variables

Just like with events, we can also define the concept of independence for random variables. We say that $X_1, X_2, \ldots, X_n$ are independent random variables if the following holds:

$$
\mathbb{P}(X_1 \leq a_1, X_2 \leq a_2, \ldots, X_n \leq a_n) = \prod_{i=1}^n \mathbb{P}(X_i \leq a_i) \quad \forall a_1, a_2, \ldots, a_n \in \mathbb{R}
$$

This is very similar to the definition of independence for events. The only difference is that we are looking at the joint probability of the random variables taking on a value less than or equal to their respective values $a_i$. We discuss [joint distributions](/maths/probabilityStatistics/jointDistributions) here in more detail. 

It actually turns out that the random variables $X_1, X_2, \ldots, X_n$ are independent if and only if for any choice of Intervals $I_1, I_2, \ldots, I_n$ the probability that each random variable $X_i$ takes on a value in the interval $I_i$ is equal to the product of the probabilities of each random variable taking on a value in their respective intervals. More formally we can say that:

$$
\mathbb{P}(X_1 \in I_1, X_2 \in I_2, \ldots, X_n \in I_n) = \prod_{i=1}^n \mathbb{P}(X_i \in I_i) \quad \forall I_1, I_2, \ldots, I_n \subseteq \mathbb{R}
$$

{{< callout type="example" >}}
We are throwing two independent dice and are considering the laplace model where $\Omega = \{1, 2, 3, 4, 5, 6\}^2$ and $\mathcal{F} = \mathcal{P}(\Omega)$. We can define the random variables $X$ and $Y$ as the outcome of the first and second die respectively and then $Z$ as the sum of the two dice. So for each outcome $\omega = (x, y)$ we define the random variables as follows:

$$
X(\omega) = x \text{ and } Y(\omega) = y \text{ and } Z(\omega) = x + y
$$

Let's first start by comparing the random variables $X$ and $Y$. To check this we look a the Intervals $I, J \subseteq \{1, 2, 3, 4, 5, 6\}$ and check if they are independent. We can do this by looking at the joint probability of the random variables $X$ and $Y$ taking on a value in the intervals $I$ and $J$. This is like checking if the events $X \in I$ and $Y \in J$ are independent. So we have:

$$
\begin{align*}
\mathbb{P}(X \in I, Y \in J) &= \mathbb{P}(\{(x, y) \in \Omega | x \in I, y \in J\}) = \mathbb{P}(I \times J) \\
&= \frac{|I \times J|}{|\Omega|} = \frac{|I| \cdot |J|}{36} \\ 
&= \frac{|I|}{6} \cdot \frac{|J|}{6} = \frac{|I \times \{1, 2, 3, 4, 5, 6\}|}{36} \cdot \frac{|J \times \{1, 2, 3, 4, 5, 6\}|}{36} \\
&= \mathbb{P}(X \in I) \mathbb{P}(Y \in J)
\end{align*}
$$

So we have shown that the random variables $X$ and $Y$ are independent. Because we can find the intervals $I$ and $J$ for any values $x, y \in \R$ we can also find the sets $\{X \in I \} = \{X \leq x\}$ and $\{Y \in J \} = \{Y \leq y\}$ and therefore we can also show that the random variables $X$ and $Y$ are independent in the sense of the CDF. So we have:

$$
\mathbb{P}(X \leq x, Y \leq y) = \mathbb{P}(X \leq x) \mathbb{P}(Y \leq y)
$$

This also makes intuitively sense as the two dice are physically independent and throwing one die does not affect the outcome of the other die. However, what about the random variable $Z$? We can also check if the random variable $Z$ is independent of one of the other random variables. To show this we can look at a specific example and check if the random variable $Z$ is independent of the random variable $X$. So we can look at the intervals $I = \{1\}$ and $J = \{1,2\}$. We can then check the following:

$$
\begin{align*}
\mathbb{P}(X \leq 1, Z \leq 2) &= \mathbb{P}(\{(1, 1)\}) \\
&= \frac{1}{36} \neq \frac{1}{6} \cdot \frac{2}{36} = \mathbb{P}(X \leq 1) \mathbb{P}(Z \leq 2)
\end{align*}
$$

So we have shown that the random variable $Z$ is not independent of the random variable $X$. This also makes intuitively sense as the random variable $Z$ is dependent on the random variable $X$ and therefore the two random variables are not independent.
{{< /callout >}}

When reading literature on probability and statistics you will often come across the abbreviation **"i.i.d."**. This stands for **independent and identically distributed**. This is a stricter version of independence and means that the random variables or events are independent of each other and have the same distribution. The i.i.d. assumption is often used in statistics and machine learning to simplify the analysis of data and models. More formally we can say that two random variables $X_1$ and $X_2$ are i.i.d. if they are independent and the following holds:

$$
\forall i,j F_{X_i} = F_{X_j}
$$

An example of this would be if we have two coins that are thrown independently of each other. We can define the random variables $X_1$ and $X_2$ as the outcome of the first and second coin respectively. The two coins are then i.i.d. as they are independent of each other and have the same distribution, so they are identically distributed, so they have the same probability of showing heads or tails. Having the i.i.d condition would also allow us to make the coins unfair, so for example the coin has a 60% chance of showing heads and a 40% chance of showing tails. However, both coins would need to have the same probability measure, so they would both need to have a 60% chance of showing heads and a 40% chance of showing tails, hence they are identically distributed. 

{{< callout type="todo" >}}
Also breaking it down using marginal distribuations 

and convolutions if X+Y 
{{< /callout >}}

### Grouping Random Variables

Up to now, we've considered the independence of individual random variables. But sometimes, we want to combine variables into groups, apply functions to each group, and ask: are the results still independent? For instance, imagine we split a collection of independent random variables into several disjoint subgroups and define new random variables by applying (possibly different) functions to each group. Intuitively, the answer is yes and this is a powerful and frequently used property in probability theory. This property is particularly useful because it allows us to break problems into smaller, independent parts which can simplify analysis in statistics or make algorithms more efficient.

Let $X_1, X_2, \ldots, X_n$ be independent random variables. Suppose we partition the indices ${1, \ldots, n}$ into $k$ disjoint, non-overlapping groups:

$$
\begin{align*}
G_1 &= \{i_1, \ldots, i_{m_1}\} \\
G_2 &= \{j_1, \ldots, j_{m_2}\} \\
&\vdots \\
G_k &= \{\ell_1, \ldots, \ell_{m_k}\}
\end{align*}
$$

so that every index appears in exactly one group, and the groups are disjoint. Now, let $g_1, \ldots, g_k$ be arbitrary (measurable) functions, and define new random variables:

$$
\begin{align*}
Y_1 = g_1(X_{i_1}, \ldots, X_{i_{m_1}}) \\
Y_2 = g_2(X_{j_1}, \ldots, X_{j_{m_2}}) \\
\vdots \\
Y_k = g_k(X_{\ell_1}, \ldots, X_{\ell_{m_k}})
\end{align*}
$$
Then the new random variables $Y_1, Y_2, \ldots, Y_k$ are independent. That is, functions of disjoint groups of independent random variables are themselves independent, no matter which (measurable) functions you use. The key intuition is that **independence is preserved under grouping**: If variables are independent, then anything you do to a group of them (as long as you don't "mix" the groups) can't introduce dependence between the groups. You can think of each group as an "island". Whatever happens within a group is independent of what happens in the other groups. Even if you apply complicated, nonlinear, or even random functions inside the group, as long as the groups don't overlap, the results remain independent.

{{< callout type="example" >}}
Suppose you throw three independent dice, and let $X_1$, $X_2$, $X_3$ be the outcomes.

- **Group 1:** $G_1 = {1, 2}$. Let $Y_1 = X_1 + X_2$ (the sum of the first two dice).
- **Group 2:** $G_2 = {3}$. Let $Y_2 = X_3$ (the third die).

Then $Y_1$ and $Y_2$ are independent random variables. In other words, knowing the sum of the first two dice gives you no information about the value of the third die.

More generally, if you split the dice into two non-overlapping groups and take any function of each group (for example, $Y_1 = \max(X_1, X_2)$ and $Y_2 = X_3^2 + 5$), the resulting variables $Y_1$ and $Y_2$ will still be independent. Intuitively, this makes sense because the groups do not share any common variables, and the functions applied to each group do not introduce any new dependencies.
{{< /callout >}}

## Examples of Discrete Random Variables

### Bernoulli Distribution

We have already seen [bernoulli experiments](/maths/probabilityStatistics/probabilitySpaces#bernoulli-experiments) as a special case of a random experiment. A Bernoulli experiment is an experiment that has only two possible outcomes, usually called success and failure. We can also define a random variable and therefore distribution for a Bernoulli experiment. We denote a random variable $X$ as a Bernoulli random variable with parameter $p$ as $X \sim \text{Ber}(p)$, where $p$ is the probability of success. The Bernoulli random variable takes the value 1 with probability $p$ and the value 0 with probability $1 - p$. In other words, we can say that a Bernoulli random variable is a discrete random variable that takes the values in $W=\{0,1\}$ with the following probabilities:

$$
\mathbb{P}(X = 0) = 1 - p \text{ and } \mathbb{P}(X = 1) = p
$$

So the PMF of a Bernoulli random variable is given by:

$$
p_X(a) = \begin{cases}
1 - p & \text{if } a = 0 \\
p & \text{if } a = 1 \\
0 & \text{otherwise}
\end{cases}
$$

and the CDF is given by:

$$
F_X(a) = \begin{cases}
0 & \text{if } a < 0 \\
1 - p & \text{if } 0 \leq a < 1 \\
p & \text{if } 1 \leq a < 2 \\
1 & \text{if } a \geq 2
\end{cases}
$$

The expectation and variance of a Bernoulli random variable are given by:

$$
\mathbb{E}(X) = p \text{ and } \text{Var}(X) = p(1 - p)
$$

The [derivation of the expectation]() and [variance]() can be seen in the section on [expectation, variance and covariance](/garden/maths/probabilityStatistics/expectationVariancCovariance).

{{< callout type="todo" >}}
Plots of the PMF and CDF for the Bernoulli distribution.
{{< /callout >}}

{{< callout type="example" >}}
We can define a Bernoulli random variable for shooting a penalty in a football game. We can define the random variable $X$ as follows:

$$
X \sim \text{Ber}(p) = \begin{cases}
0 & \text{ if the penalty is missed} \\
1 & \text{ if the penalty is scored}
\end{cases}
$$

where $p$ is the probability of scoring the penalty. So if we have a penalty taker that scores 80% of the time, we can define the random variable as $X \sim \text{Ber}(0.8)$
{{< /callout >}}

{{< callout type="example" title="Radamacher Distribution" >}}
There is also the Radamacher distribution which is a special case of the Bernoulli distribution with $p = 0.5$. It takes the values -1 and 1 with equal probability
{{< /callout >}}

### Binomial Distribution

We have already seen [Bernoulli experiments](/maths/probabilityStatistics/probabilitySpaces#bernoulli-experiments) and the Bernoulli random variable $X \sim \text{Ber}(p)$, which models the outcome of a single experiment with two possible results: "success" (with probability $p$) and "failure" (with probability $1-p$).

The **binomial distribution** extends this idea to a sequence of $n$ independent Bernoulli experiments, each with the same success probability $p$. The binomial random variable counts the total number of successes in these $n$ trials. 

Let $X$ be the random variable representing the **number of successes** in $n$ independent Bernoulli trials, each with success probability $p$. We write:

$$
X \sim \mathrm{Bin}(n, p)
$$

where $n$ = number of trials (fixed, known in advance) and $p$ = probability of success on each trial. The possible values of $X$ are $W = {0, 1, 2, ..., n}$: from zero successes (all failures) up to $n$ successes (all successes). So you can also think of a Bernoulli random variable as just a special case of the binomial with $n=1$:

$$
X \sim \mathrm{Bin}(1, p) \equiv \mathrm{Ber}(p)
$$

The probability of observing exactly $k$ successes ($k \in {0, 1, ..., n}$) in $n$ independent trials is given by:

$$
\mathbb{P}(X = k) = p_X(k) = \binom{n}{k} p^k (1-p)^{n-k} 
$$

where $\binom{n}{k}$ is the [binomial coefficient](/maths/commonFunctions/binomialCoefficient), which counts the number of different ways to choose $k$ trials (out of $n$) to be successes. The intuition behind this formula is to have exactly $k$ we need to know the number of ways to choose which $k$ trials are successes, multiplied by the probability of those trials being successes, and the remaining trials being failures. So we get:

- $\binom{n}{k}$ counts the number of different ways $k$ successes could be distributed among the $n$ trials.
- $p^k$ is the probability that $k$ specific trials are all successes,
- $(1-p)^{n-k}$ is the probability that the remaining $n-k$ trials are all failures,

So in a way we choose which $k$ trials out of $n$ will be successes out of the $\binom{n}{k}$ ways. Then the probability of one specific arrangement of successes and failures is given by $p^k (1-p)^{n-k}$ (since each trial is independent) and we need to get $k$ successes and $n-k$ failures. 

The CDF of the binomial is then:

$$
F_X(a) = \mathbb{P}(X \leq a) = \sum_{k = 0}^{\lfloor a \rfloor} \binom{n}{k} p^k (1-p)^{n-k}
$$

The expectation and variance of the binomial random variable $X \sim \text{Bin}(n, p)$ are:

$$
\mathbb{E}[X] = n p \text{ and } \text{Var}(X) = n p (1-p)
$$

This matches the intuition that for a fair coin in 10 coin flips, you expect 5 heads. The derivation of these formulas can be found in the section on [expectation, variance and covariance](/garden/maths/probabilityStatistics/expectationVariancCovariance).

{{< callout type="proof" title="PMF Sums to 1" >}}
First let's check that the PMF sums to 1, which is a requirement for any probability mass function (PMF). So we need to show that the sum of the probabilities over all possible values of $k$ (from 0 to $n$) equals 1:

$$
\sum_{k=0}^n \mathbb{P}(X = k) = \sum_{k=0}^n \binom{n}{k} p^k (1-p)^{n-k} = 1
$$

For this we use the [binomial theorem](/maths/commonFunctions/binomialCoefficient):

$$
(a + b)^n = \sum_{k=0}^n \binom{n}{k} a^k b^{n-k}
$$

Notice the patterns match up:

$$
\begin{align*}
(a + b)^n &= \sum_{k=0}^n \binom{n}{k} a^k b^{n-k} \\
(p + (1-p))^n &= \sum_{k=0}^n \binom{n}{k} p^k (1-p)^{n-k}
&= 1^n = 1
\end{align*}
$$
{{< /callout >}}

{{< callout type="proof" >}}
Next let's prove that the Binomial Distribution is a sum of independent Bernoulli trials. So suppose $X_1, X_2, ..., X_n$ are i.i.d. Bernoulli($p$) random variables (representing each individual trial: 1 = success, 0 = failure). Then

$$
S_n = X_1 + X_2 + ... + X_n \sim \text{Bin}(n, p)
$$

First we check that $S_n$ takes values in $\{0, 1, \ldots, n\}$. This is clear since each $X_i$ can only be 0 or 1, so their sum can only range from 0 (all failures) to $n$ (all successes).

Next we can split the set $\{S_n = k\}$ into all possible combinations of successes and failures:

$$
\{S_n = k\} = \bigcup_{x_1, x_2, \ldots, x_n \in \{0, 1\}, \sum_{i=1}^n x_i = k} \{X_1 = x_1, X_2 = x_2, \ldots, X_n = x_n\}
$$

Because each of these combinations is disjoint, we can sum the probabilities:

$$
\begin{align*}
\mathbb{P}(S_n = k) &= \sum_{x_1, x_2, \ldots, x_n \in \{0, 1\}, \sum_{i=1}^n x_i = k} \mathbb{P}(X_1 = x_1, X_2 = x_2, \ldots, X_n = x_n) \\
&= \sum_{x_1, x_2, \ldots, x_n \in \{0, 1\}, \sum_{i=1}^n x_i = k} \mathbb{P}(X_1 = x_1) \mathbb{P}(X_2 = x_2) \ldots \mathbb{P}(X_n = x_n) \\
&= \sum_{x_1, x_2, \ldots, x_n \in \{0, 1\}, \sum_{i=1}^n x_i = k} p^{\sum_{i=1}^n x_i} (1-p)^{n - \sum_{i=1}^n x_i} \\
&= \binom{n}{k} p^k (1-p)^{n-k}
\end{align*}
$$

Key here is that the independence of the $X_i$ allows us to factor the joint probability into a product of individual probabilities, and then we count how many ways we can choose $k$ successes from $n$ trials using the binomial coefficient.
{{< /callout >}}

The binomial distribution is also additive. So if you have two independent binomial random variables with the same success probability $p$, you can add them together to get another binomial random variable with the sum of the number of trials. Intuitively this makes sense as all the trials are independent and the success probability is the same, so you can just combine them into one larger set of trials. So more formally if $X \sim \text{Bin}(n_1, p)$ and $Y \sim \text{Bin}(n_2, p)$ are independent random variables, then their sum $Z = X + Y$ is also a binomial random variable:

$$
Z \sim \text{Bin}(n_1 + n_2, p)
$$

{{< callout type="proof" title="Sum of Independent Binomial Random Variables" >}}
Let's prove that the sum of two independent binomial random variables (with the **same** $p$) is again binomial. So suppose $X \sim \text{Bin}(n_1, p)$ and $Y \sim \text{Bin}(n_2, p)$, and $X$ and $Y$ are independent.
Let $Z = X + Y$. We want to show $Z \sim \text{Bin}(n_1 + n_2, p)$.

Let $j$ be the number of successes in the first $n_1$ trials ($X = j$), and $k-j$ the number in the next $n_2$ trials ($Y = k-j$) so the missing success such that we have $k$ total successes. So we then just need to sum over all possible values of $j$ that can occur in the first $n_1$ trials. Specifically $j$ must be at least $0$ (since you can't have negative successes in $X$), but also $Y = k-j$ must be at least $0$, i.e., $j \leq k$. Also $j$ can't be more than $n_1$, and $k-j$ can't be more than $n_2$ (since $Y$ can't have more than $n_2$ successes).

So putting this together, we have:

$$
\mathbb{P}(Z = k) = \sum_{j = \max(0, k-n_2)}^{\min(k, n_1)} \mathbb{P}(X = j) \mathbb{P}(Y = k - j)
$$

Using the PMF of the binomial distribution, we can write:

$$
\begin{align*}
\mathbb{P}(X = j) &= \binom{n_1}{j} p^j (1-p)^{n_1 - j} \\
\mathbb{P}(Y = k - j) &= \binom{n_2}{k-j} p^{k-j} (1-p)^{n_2 - (k-j)}
\end{align*}
$$

So,

$$
\mathbb{P}(Z = k) = \sum_{j} \binom{n_1}{j} p^j (1-p)^{n_1 - j} \cdot \binom{n_2}{k-j} p^{k-j} (1-p)^{n_2 - (k-j)}
$$

We can combine the powers of $p$ and $(1-p)$ as $p^j \cdot p^{k-j} = p^k$ and $(1-p)^{n_1 - j} \cdot (1-p)^{n_2 - (k-j)} = (1-p)^{n_1 + n_2 - k}$. Resulting in:

$$
\mathbb{P}(Z = k) = p^k (1-p)^{n_1 + n_2 - k} \sum_{j = \max(0, k-n_2)}^{\min(k, n_1)} \binom{n_1}{j} \binom{n_2}{k-j}
$$

Lastly we use [Vandermonde's identity]() to simplify the sum:

$$
\sum_{j=0}^k \binom{n_1}{j} \binom{n_2}{k-j} = \binom{n_1 + n_2}{k}
$$

the sum from 0 to $k$ counts all ways to choose $k$ successes from $n_1 + n_2$ trials, which is the same as our limits just in a different way of writing it. So we can rewrite the probability as:

$$
\mathbb{P}(Z = k) = \binom{n_1 + n_2}{k} p^k (1-p)^{n_1 + n_2 - k}
$$

Therefore $Z = X + Y \sim \text{Bin}(n_1 + n_2, p)$
{{< /callout >}}

{{< callout type="example" >}}
Suppose you are answering a multiple-choice exam with $n = 10$ questions, each with 4 possible answers (only one correct), and you guess randomly. What is the probability of getting exactly 6 correct (so in Switzerland a passing grade of a 4 (60% correct))?

Let $X$ be the number of correct answers. Each question is a Bernoulli($p$) trial with $p = \frac{1}{4}$.

$$
X \sim \text{Bin}(10, 0.25)
$$

The probability of getting exactly 6 correct is:

$$
\mathbb{P}(X = 6) = \binom{10}{6} (0.25)^6 (0.75)^4 = 210 \cdot 0.00024414 \cdot 0.31640625 \approx 0.0162
$$

So, there is about a 1.62% chance to get exactly 6 right by guessing. So make sure to study! This actually does not include the fact that you might guess more than 6 questions right, so the actual chance of passing is higher. But this is a good lower bound.
{{< /callout >}}
{{< callout type="example" >}}
Suppose you flip a fair coin ($p=0.5$) $n=8$ times. What is the probability of getting exactly 3 heads?

$$
\mathbb{P}(X = 3) = \binom{8}{3} (0.5)^3 (0.5)^{5} = 56 \times (0.5)^8 = 56 \times \frac{1}{256} \approx 0.21875
$$

So, about a 21.9% chance. What about getting at least 3 heads?

$$
\begin{align*}
\mathbb{P}(X \geq 3) &= P(X = 3) + P(X = 4) + \ldots + P(X = 8) \\
&= \sum_{k=3}^{8} \binom{8}{k} (0.5)^k (0.5)^{8-k}
\end{align*}
$$

Or alternatively, using the complement:
$$
\begin{align*}
\mathbb{P}(X \geq 3) &= 1 - \mathbb{P}(X < 3) \\
&= 1 - (\mathbb{P}(X = 0) + \mathbb{P}(X = 1) + \mathbb{P}(X = 2))
\end{align*}
$$
{{< /callout >}}

### Geometric Distribution

We've seen how the Bernoulli and Binomial distributions model the number of successes in a fixed number of independent trials. But what if we flip the question to "how many trials will it take to get the **first** success?" This leads us to the **geometric distribution**

Suppose we let $X_1, X_2, \ldots$ be independent random variables with $X_i \sim \mathrm{Ber}(p)$. Then let $X$ be the random variable representing the **trial number on which the first success occurs**. So we define $X$ as:

$$
X = \min \{ n \in \mathbb{N}: X_n = 1 \}
$$

We say $X$ is a geometric random variable with parameter $p$:

$$
X \sim \text{Geom}(p)
$$

The PMF of $X$ is given by:

$$
\mathbb{P}(X = k) = p_X(k) = (1-p)^{k-1} \cdot p
$$

for $k \in \mathbb{N}^+$. The intuition behind this is that think of flipping a coin over and over: what is the probability that you get your first heads exactly on the $k$th flip? To do this, you must have gotten $k-1$ tails (failures) in a row, followed by a heads (success) on the $k$th flip. Each outcome is independent, so the probability multiplies. This is why the geometric distribution is sometimes called the "waiting time" distribution as it tells you how long you'll wait for the first success.

You can also think of the event as follows:

$$
\{X = k\} = \{X_1 = 0, X_2 = 0, \ldots, X_{k-1} = 0, X_k = 1\}
$$

where each $X_i \sim \mathrm{Ber}(p)$ is the indicator of success on trial $i$. Technically, it's possible (though with probability zero) that we never see a success, i.e., $X=+\infty$. But

$$
\mathbb{P}(X = +\infty) = \lim_{k \to \infty} (1-p)^k = 0
$$

so for all practical purposes, $X$ takes finite values with probability 1. So we say it is **almost surely finite**.

{{< callout type="proof" >}}
Let's check the PMF sums to 1:

$$
\sum_{k=1}^\infty \mathbb{P}(X = k) = \sum_{k=1}^\infty p(1-p)^{k-1} = p \sum_{k=0}^\infty (1-p)^k
$$

This is a geometric series with ratio $r = 1-p < 1$ so we can use the formula for the sum of an infinite geometric series:

$$
\sum_{k=0}^\infty r^k = \frac{1}{1 - r}
$$

So,

$$
p \cdot \frac{1}{1 - (1-p)} = p \cdot \frac{1}{p} = 1
$$
{{< /callout >}}

The CDF of $X$ is the probability that the first success happens by trial $k$:

$$
F_X(k) = \mathbb{P}(X \leq k) = \sum_{j=1}^k p (1-p)^{j-1} = 1 - (1-p)^k
$$

notice again that this uses the formula for the sum of a geometric series, this is also the reason why the geometric distribution is called "geometric" as it is related to the geometric series. We can alternatively also get the **probability that you have not yet succeeded after $n$ trials** is, i.e. the survival function

$$
\mathbb{P}(X > k) = 1 - F_X(k) = 1 - (1 - (1-p)^k) = (1-p)^k
$$

This makes sense because you need $k$ consecutive failures.

{{< callout type="proof" >}}
We can derive this CDF using the PMF. First we calculate that we have $k-1$ failures and then a success on the $k$th trial. So we can write again using the formula for the sum of a geometric series:

$$
\sum_{j=0}^{k-1} (1-p)^j = \frac{1 - (1-p)^k}{1 - (1-p)} = \frac{1 - (1-p)^k}{p}
$$

Multiplying by $p$:

$$
F_X(k) = 1 - (1-p)^k
$$

Notice that for $k=1$, we have $F_X(1) = p$, which is the probability of success on the first trial.

For $k=0$ so we have $F_X(0) = 0$, which is the probability of never seeing a success which again shows that at least one success will happen with probability 1 so almost surely.
{{< /callout >}}

The geometric random variable $X$ has the following **expectation** and **variance**:

$$
\mathbb{E}[X] = \frac{1}{p} \text{ and } \text{Var}(X) = \frac{1-p}{p^2}
$$

Intuitively: if the chance of success is small, you expect to wait longer. If $p = 0.2$, you expect to wait $1/0.2 = 5$ trials for a success. If $p = 0.5$, you expect to wait $1/0.5 = 2$ trials. The proof and derivation for these can be found in the [expectation and variance section](/garden/maths/probabilityStatistics/expectationVariancCovariance).

One of the key properties of the geometric distribution is that it is **memoryless**, also known as the **markov property**. This means that the probability of waiting for $m$ more trials for a success does not depend on how many trials have already been conducted. In other words, if you have already waited $n$ trials without success, the probability of needing $m$ more trials to get a success is the same as if you had just started fresh. So in a way as soon as a trial is conducted it "forgets" it and therefore does not change the probability of success in the next trial.Formally for all $m, n \in \mathbb{N}$:

$$
\mathbb{P}(X > m + n \mid X > n) = \mathbb{P}(X > m)
$$

{{< callout type="proof" >}}
To prove this, we can use the definition of conditional probability:

$$
\begin{align*}
\mathbb{P}(X > m + n \mid X > n) &= \frac{\mathbb{P}(X > m + n)}{\mathbb{P}(X > n)} \\
&= \frac{(1-p)^{m+n}}{(1-p)^n} \\
&= (1-p)^m = \mathbb{P}(X > m)
\end{align*}
$$

It doesn't matter how many failures you've already seen. The probability you have to wait $m$ more trials for a success is always the same.
{{< /callout >}}

{{< callout type="example" >}}
Suppose a basketball player has a $p = 0.3$ chance of making a free throw. What is the probability they make their first free throw on their 4th attempt?

We are looking for:

$$
\mathbb{P}(X = 4) = (1-0.3)^{4-1} \cdot 0.3 = (0.7)^3 \cdot 0.3 = 0.343 \cdot 0.3 = 0.1029
$$

So about a 10.3% chance. What is the expected number of shots until the first make?

$$
\mathbb{E}[X] = \frac{1}{0.3} \approx 3.33
$$

So on average, it will take just over 3 attempts to make a shot. 
{{< /callout >}}

### Poisson Distribution

So far, we've seen discrete random variables such as the Bernoulli, Binomial, and Geometric distributions, which model processes based on repeated, independent trials. But what if we want to model the **number of times an event happens in a fixed interval of time or space**, especially when the probability of an individual event is very small so it is a **rare event** but there are **many opportunities** for it to occur, so the interval is large? This leads us to the **Poisson distribution**.

Think about how many typos you find on a single printed page, the number of emails you receive in one hour, or how many buses arrive at a stop in 10 minutes. Each possible typo, email, or bus is a "rare" event (unlikely per opportunity), but there are lots of chances for it to occur.

The Poisson distribution answers the question: **If events happen at an average rate $\lambda$ per interval (time, space, etc.), what is the probability that we observe exactly $k$ events in that interval?** Formally, we write:

$$
X \sim \mathrm{Poi}(\lambda)
$$

where $\lambda > 0$ is the **rate parameter**: the expected number of occurrences in the interval. The PMF of the Poisson distribution is given by:

$$
\mathbb{P}(X = k) = p_X(k) = \frac{\lambda^k}{k!}e^{-\lambda}
$$

where $k \in \mathbb{N}_0 = {0, 1, 2, \ldots}$ is the number of occurrences. 

This formula comes from the following idea. We use the poisson distribution as follows. Suppose we want to model how many events happen in a fixed interval (say 1 hour), where each possible opportunity for an event (say each minute) has a small probability $p$ of an event. Then we have a large number $n$ of such opportunities (n=60 for 1 hour). The average total number of events we expect in the interval is the rate parameter $\lambda = np$ (so as $n$ increases, $p$ decreases). Notice this is exactly the same as the binomial distribution, where we have $n$ independent trials with success probability $p$ which is $\lambda/n$ in this case. The idea is now that we can think of the Poisson distribution as a limit of the binomial distribution as we split the interval to infinitely many small intervals. Because as $n$ increases, the probability of an event in each small interval becomes very small, but the total number of intervals increases, so we can still expect a fixed number of events $\lambda$ in the whole interval. 

So let $X_n \sim \mathrm{Bin}(n, \frac{\lambda}{n})$ be the binomial random variable counting the number of successes in $n$ independent trials, each with success probability $\frac{\lambda}{n}$ such that over the whole interval we expect $\lambda$ successes. Then as $n \to \infty$ we get the PMF of the Poisson distribution:

$$
\begin{align*}
\mathbb{P}(X_n = k) &= \binom{n}{k} \left(\frac{\lambda}{n}\right)^k \left(1-\frac{\lambda}{n}\right)^{n-k} \\
&= \underbrace{\frac{n!}{k!(n-k)!} \cdot \frac{1}{n^k}}_{\to \frac{1}{k!} \text{ as } n \to \infty}
\cdot \lambda^k
\cdot \underbrace{\left(1-\frac{\lambda}{n}\right)^n}_{\to e^{-\lambda}}
\cdot \underbrace{\left(1-\frac{\lambda}{n}\right)^{-k}}_{\to 1} \\
&= \frac{\lambda^k}{k!}
\cdot
\underbrace{\frac{n(n-1)\cdots(n-k+1)}{n^k}}_{\to 1}
\cdot e^{-\lambda}
\cdot 1 \\
&\xrightarrow{n \to \infty} \frac{\lambda^k}{k!} e^{-\lambda}
\end{align*}
$$

So we have:

$$
\lim_{n \to \infty} \binom{n}{k}
\left(\frac{\lambda}{n}\right)^k
\left(1-\frac{\lambda}{n}\right)^{n-k} =
\frac{\lambda^k}{k!} e^{-\lambda}
$$

{{< callout type="proof" >}}
Let's check that the Poisson PMF is a valid probability mass function (PMF) by showing that it sums to 1 over all possible values of $k$:

$$
\sum_{k=0}^\infty \frac{\lambda^k}{k!}e^{-\lambda} = e^{-\lambda} \sum_{k=0}^\infty \frac{\lambda^k}{k!}
$$

The sum inside is the Taylor series for $e^{\lambda}$, so:

$$
e^{-\lambda} \cdot e^{\lambda} = 1
$$

So, the total probability is 1, as required. 
{{< /callout >}}

As mentioned you can think of the Poisson distribution as a limit of the binomial distribution as $n \to \infty$ and $p = \frac{\lambda}{n} \to 0$, while keeping $\lambda = np$ constant. The binomial distribution was additive, meaning that if you have two independent binomial random variables with the same success probability $p$, you can add them together to get another binomial random variable with the sum of the number of trials. The Poisson distribution is also additive, but in a different way. Suppose we observe two independent Poisson processes such as the number of emails from Alice ($X \sim \mathrm{Poi}(\lambda_1)$) and from Bob ($Y \sim \mathrm{Poi}(\lambda_2)$) in an hour. If we then wanted to know the total number of emails received in that hour, we can add the two independent Poisson random variables together:

$$
Z = X + Y \sim \mathrm{Poi}(\lambda_1 + \lambda_2)
$$

Intuitively this makes sense as over the interval we have two independent sources of events, each with their own rate, and the total number of events is just the sum of the two independent sources.

{{< callout type="proof" >}}
Since $X$ and $Y$ are independent we have:

$$
\mathbb{P}(Z = k) = \sum_{j=0}^k \mathbb{P}(X = j)\mathbb{P}(Y = k-j)
$$

If we then plug in the PMFs of the Poisson distributions we get:
$$
\begin{align*}
\mathbb{P}(Z = k) &= \sum_{j=0}^k \frac{\lambda_1^j}{j!}e^{-\lambda_1} \cdot \frac{\lambda_2^{k-j}}{(k-j)!}e^{-\lambda_2} \\
&= e^{-(\lambda_1 + \lambda_2)} \sum_{j=0}^k \frac{\lambda_1^j}{j!}\frac{\lambda_2^{k-j}}{(k-j)!}
\end{align*}
$$

Notice that by the binomial theorem we can rewrite the sum as:

$$
\begin{align*}
\sum_{j=0}^k \frac{\lambda_1^j}{j!}\frac{\lambda_2^{k-j}}{(k-j)!}
&= \frac{1}{k!} \sum_{j=0}^k \binom{k}{j} \lambda_1^j \lambda_2^{k-j} \\
= \frac{(\lambda_1 + \lambda_2)^k}{k!}
\end{align*}
$$

So putting it all together we have:

$$
\mathbb{P}(Z = k) = e^{-(\lambda_1 + \lambda_2)} \frac{(\lambda_1 + \lambda_2)^k}{k!}
$$

which is the PMF of $\mathrm{Poi}(\lambda_1 + \lambda_2)$. 
{{< /callout >}}

{{< callout type="todo" >}}
Derivation of the expectation and variance.
* **Mean (Expectation):** $\mathbb{E}[X] = \lambda$
* **Variance:** $\text{Var}(X) = \lambda$
{{< /callout >}}

{{< callout type="example" >}}
Suppose a page of a newspaper contains $n = 10^4$ characters, and each character has a chance $p = \frac{10}{n} = 0.001$ of being misprinted. What is the probability there are exactly $k$ misprints on the page?

Let $M$ be the number of misprints, so $M \sim \mathrm{Bin}(n, p)$. For $n$ large, $p$ small, $np = 10$. So the probability of exactly $k$ misprints is given by the Poisson PMF with $\lambda = 10$.

By the Poisson approximation $M \sim \mathrm{Poi}(10)$, we have:

$$
\mathbb{P}(M = k) = \frac{10^k}{k!} e^{-10} \approx 0.0378
$$

So there is about a **3.78% chance** of seeing exactly 10 misprints on the page.
{{< /callout >}}
{{< callout type="example" >}}
Suppose a radioactive material emits particles at a known average rate of $\lambda = 4$ decays per second. What is the probability that exactly 7 particles decay in a single second? Let $X$ be the number of decays in one second. Then $X \sim \mathrm{Poi}(4)$. The probability of exactly 7 decays is:

$$
\mathbb{P}(X = 7) = \frac{4^7}{7!} e^{-4} \approx 0.0595
$$

So there is about a **5.95% chance** of seeing exactly 7 decays in one second.
{{< /callout >}}
{{< callout type="example" >}}
Suppose customers arrive at a bakery at an average rate of $\lambda = 3$ per 10 minutes. What is the probability that **no customers arrive** in a particular 10-minute interval?

Let $X \sim \mathrm{Poi}(3)$, and we want $\mathbb{P}(X = 0)$:

$$
\mathbb{P}(X = 0) = \frac{3^0}{0!}e^{-3} = 1 \cdot e^{-3} \approx 0.0498
$$

So, there is about a **4.98% chance** that no one comes in during a 10-minute period.

What about the probability that 3 appear in the next 20 minutes? Since the rate is 3 per 10 minutes, in 20 minutes we expect $\lambda = 6$ customers. So we can use the Poisson PMF:

$$
\mathbb{P}(X = 3) = \frac{6^3}{3!} e^{-6} \approx 0.0892
$$

So, there is about an **8.92% chance** that exactly 3 customers arrive in the next 20 minutes.
{{< /callout >}}

### Negative Binomial Distribution

### Hypergeometric Distribution

Die Hypergeometrische Verteilung der Zufallsvariable $X$ ist die Verteilung, die beim $n$-maligen Ziehen ohne
Zurücklegen und ohne Reihenfolge aus einer Urne mit $N$ Kugeln, von denen $M$ eine spezielle Eigenschaft haben und wo
die Anzahl der gezogenen Kugeln mit dieser speziellen Eigenschaft gezählt werden.

- Wir schreiben dann $X \sim Hyp(N,M,n)$
- Die Dichtefunktion von $X$ ist $f(k)=\binom{M}{k} \cdot \frac{\binom{N-M}{n-k}}{ \binom{N}{n}}$ wobei $N$ die
  Gesamtanzahl der Kugeln ist, $M$ die Anzahl mit der speziellen Eigenschaft. $n$ ist dann der Umfang der Stichprobe also
  die Anzahl der entnommenen Kugeln und $k$ die Anzahl angestrebte Kugeln mit der speziellen Eigenschaft.
- $E(X) = n \cdot \frac{M}{N}$
- $V(X) = n \cdot \frac{M}{N} \cdot (1 - \frac{M}{N}) \cdot \frac{N-n}{N-1}$

In Matlab haben wir die Funktionen:

- Dichtefunktion: $hygepdf(k,N,M,n)$
- Verteilungsfunktion $hygecdf(k,N,M,n)$

mehr dazu findest du [hier](https://www.youtube.com/watch?v=BoPYslAe8sg).

{{< callout type="example" title="Beispiel Hypergeometrische Verteilung" >}}
Das perfekte Beiepiel dafür ist Lotto, wobei wir 49 nummerierte Kugeln haben, 6 davon werden gezogen, welche in diesem
Falle unsere spezielle Kugeln sind. Wir dürfen 6 Zahlen aufschreiben, also sind das unsere Kugeln die wir
herausnehmen ohne zurücklegen oder die Reihenfolge zu beachten. Was ist nun die Wahrscheinlichkeit das wir 4 von
den 6 richtig haben?

$X \sim Hyp(49,6,6)$
$hygepdf(4,49,6,6) = \frac{645}{665896} \approx 0.09686\%$
{{< /callout >}}

### Coupon Collector Problem

## Continuous Random Variables

Recall that a **random variable** is a function that assigns a real number to each outcome in a sample space. We can have two types of random variables: **discrete** and **continuous**. In the discrete case the image of the random variable is a countable set, while in the continuous case it is an uncountable set (like an interval on the real line).

We also defined that a **cumulative distribution function (CDF)** $F_X$ of a random variable $X$ is defined as:

$$
F_X(a) = \mathbb{P}(X \leq a)
$$

where $a$ is a real number and the CDF $F_X$ is left continuous. The probabilites of a discrete random variable are given by the **probability mass function (PMF)** $p_X(a)$ which are non-zero at the countable set of points where the cdf has jumps. For a continuous random variable the CDF is continous so not just left continuous but also right continuous. Because the CDF is continous there exists a PDF $f_X$ such that:

$$
\mathbb{P}(X \leq a) = F_X(a) = \int_{-\infty}^{a} f_X(x)\,dx
$$

and the probabilities are given by the probability density function (PDF) $f_X(a)$ which is non-negative and integrates to 1 over the whole space.

$$
\int_{-\infty}^{\infty} f_X(x) \, dx = 1
$$

By the **fundamental theorem of calculus**, we can also go from the PDF to the CDF:

$$
\mathbb{P}(a < X \leq b) = \mathbb{P}(X \leq b) - \mathbb{P}(X \leq a) = F_X(b) - F_X(a) = \int_{a}^{b} f_X(x)\,dx
$$

We can also define the **probability of a single point** for a random variable $X$ at a point $a$ as:

$$
\mathbb{P}(X = a) = F_X(a) - \lim_{h \to 0^+} F_X(a-h)
$$

In the case of a continuous random variable, this probability is always zero because the CDF is continuous at every point, meaning there are no jumps. In the case of a discrete random variable, this probability can be non-zero if there is a jump in the CDF at that point. This also becomes clear when you look at the following integral:

$$
\mathbb{P}(X = a) = \int_{a}^{a} f_X(x) \, dx = 0
$$

The intuition behind this is imagine we have the interval between $a$ and $a + h$, then the probability of the random variable being exactly at $a$ is 0 because inbetween $a$ and $a + h$ there are infinitely many points, so the probability of hitting exactly one point is 0. This is why we say that for continuous random variables, the probability of a single point is always 0.

If we are given just the PDF $f_X(a)$, we can also define the CDF by taking the integral of the PDF:

$$
F_X(a) = \int_{-\infty}^{a} f_X(x) \, dx
$$

We can also define the **survival function** $S_X(a)$ which is the probability that the random variable $X$ is greater than $a$:

$$
S_X(a) = \mathbb{P}(X > a) = 1 - F_X(a) = \int_{a}^{\infty} f_X(x) \, dx
$$

Alternatively, we can also go from the CDF to the PDF by taking the derivative in general:

$$
f_X(a) = (F_X(a))'
$$

## Examples of Continuous Random Variables

### Uniform Distribution

We have already seen the **discrete uniform distribution** as a model where all outcomes are equally likely among a finite number of values, such as when throwing a fair die. We call this a [laplace space]().

We now extend this idea to the continuous case, where the random variable takes values in a real interval. For example, a time chosen uniformly at random in the interval $[a, b]$. This gives us the **continuous uniform distribution**.

The intuition is the same, each subinterval of equal length within $[a, b]$ is equally likely to contain the outcome. So, we are modeling complete uncertainty over an interval. We know the outcome lies somewhere in $[a, b]$, but we assume no preference for any subinterval over another. The resulting distribution is flat, or **uniform**, over the interval.

Let $X$ be a continuous random variable defined on the interval $[a, b]$. Then $X$ is said to have a **uniform distribution on $[a, b]$** if its **probability density function (PDF)** is given by:

$$
f_X(x) = \begin{cases}
\frac{1}{b-a} & \text{if } x \in [a, b] \\
0 & \text{otherwise}
\end{cases}
$$

We write:

$$
X \sim \text{Uniform}(a, b)
$$

The PDF is zero outside $[a, b]$, meaning the random variable cannot take values outside this range.

{{< callout type="proof" >}}
We check that the PDF integrates to 1 as it is trivially non-negative because $a < b$:

$$
\int_{-\infty}^{\infty} f_X(x) \, dx = \int_a^b \frac{1}{b-a} dx = \frac{1}{b-a}(b - a) = 1
$$
{{< /callout >}}

The PDF is constant over the interval $[a, b]$, reflecting the fact that all values in this range are equally likely (in terms of density). However, since the distribution is continuous, the probability of hitting any exact point $x$ is still zero:

$$
\mathbb{P}(X = x) = \int_x^x f_X(t) dt = 0
$$

So instead we compute probabilities over intervals using the **cumulative distribution function**. The CDF $F_X(x)$ for a uniform distribution on $[a, b]$ is obtained by integrating the PDF:

$$
F_X(x) = \int_{-\infty}^x f_X(t)\,dt = \begin{cases}
0 & \text{if } x < a \\
\frac{x - a}{b - a} & \text{if } a \leq x \leq b \\
1 & \text{if } x > b
\end{cases}
$$

This function smoothly increases from 0 to 1 as $x$ goes from $a$ to $b$, and is **continuous** and **strictly increasing** on $[a, b]$. It is flat (zero) before $a$, and reaches 1 after $b$.

{{< callout type="proof" >}}
The middle case comes from integrating the constant PDF:

$$
F_X(x) = \int_a^x \frac{1}{b-a} dt = \frac{x-a}{b-a}
\quad \text{for } x \in [a, b]
$$
{{< /callout >}}

For continuous random variables, we compute probabilities by integrating the PDF or using the CDF. For the uniform distribution, we get fo any interval $[c, d]$ within $[a, b]$:

$$
\mathbb{P}(c \leq X \leq d) = \int_c^d f_X(x) dx = F_X(d) - F_X(c) = \frac{d - c}{b - a}\quad \text{for } a \leq c < d \leq b
$$

So the probability is **proportional to the length** of the interval $[c, d]$ with respect to the total length of $[a, b]$. This reflects the uniform nature of the distribution: longer intervals have higher probabilities. We can also use the CDF to compute intervals:

$$
\mathbb{P}(X \leq x) = F_X(x), \quad
\mathbb{P}(X > x) = 1 - F_X(x), \quad
\mathbb{P}(c < X \leq d) = F_X(d) - F_X(c)
$$

The expectation of a uniform random variable $X \sim \text{Uniform}(a, b)$ is simply the midpoint:

$$
\mathbb{E}[X] = \frac{a + b}{2}
$$

and the variance is given by:

$$
\text{Var}(X) = \frac{(b - a)^2}{12}
$$

{{< callout type="proof" title="Derivation of the expectation" >}}
We compute the expectation as:

$$
\mathbb{E}[X] = \int_a^b x \cdot \frac{1}{b - a} dx = \frac{1}{b-a} \cdot \left[\frac{x^2}{2}\right]_a^b = \frac{1}{b-a} \cdot \frac{b^2 - a^2}{2}
$$

Simplify using the identity $b^2 - a^2 = (b-a)(b+a)$:

$$
\mathbb{E}[X] = \frac{1}{b-a} \cdot \frac{(b-a)(b+a)}{2} = \frac{a + b}{2}
$$
{{< /callout >}}

{{< callout type="proof" title="Derivation of the variance" >}}
We compute $\mathbb{E}[X^2]$:

$$
\mathbb{E}[X^2] = \int_a^b x^2 \cdot \frac{1}{b-a} dx = \frac{1}{b-a} \left[ \frac{x^3}{3} \right]_a^b = \frac{1}{b-a} \cdot \frac{b^3 - a^3}{3}
$$

Then the variance is:

$$
\text{Var}(X) = \mathbb{E}[X^2] - (\mathbb{E}[X])^2 = \frac{b^3 - a^3}{3(b - a)} - \left( \frac{a + b}{2} \right)^2
$$

After simplifying, we find:

$$
\text{Var}(X) = \frac{(b - a)^2}{12}
$$
{{< /callout >}}

{{< callout type="example" >}}
Suppose we pick a random time $X$ between 2pm and 4pm (i.e. $X \sim \text{Uniform}(2, 4)$). Then:

- The PDF is $f_X(x) = \frac{1}{2}$ for $x \in [2, 4]$
- The CDF is $F_X(x) = \frac{x - 2}{2}$ for $x \in [2, 4]$
- The mean time is $\mathbb{E}[X] = 3$pm
- The variance is $\text{Var}(X) = \frac{(4 - 2)^2}{12} = \frac{1}{3}$

Probability that we arrive between 2:30 and 3:30pm:

$$
\mathbb{P}(2.5 \leq X \leq 3.5) = \frac{3.5 - 2.5}{2} = \frac{1}{2}
$$

So there's a 50% chance of arriving in that hour-long interval. 
{{< /callout >}}

### Exponential Distribution

We've seen how the geometric distribution models the number of Bernoulli trials until the first success, and how the Poisson distribution models the number of events in a fixed interval, given a constant event rate. Now what if, instead of counting the number of events, we want to model **how long we wait** until the first event occurs in a continuous-time process? This leads us to the **exponential distribution**.

Suppose we have a Poisson process, so events happen randomly, independently, and at a constant average rate in time (for example, arrivals of buses, radioactive decays, phone calls, or failures of light bulbs). If we fix a starting point, the **time we must wait until the next event** is an **exponentially distributed random variable**. So the exponential is to the geometric what the Poisson is to the binomial.

Let $X$ be a continuous random variable modeling the **waiting time until the first event** in a process where events occur at a constant average rate $\lambda > 0$ per unit time. So for example if $\lambda = 2$, we expect 2 events per interval. We write:

$$
X \sim \mathrm{Exp}(\lambda)
$$

The **probability density function (PDF)** is:

$$
f_X(x) = \begin{cases}
\lambda e^{-\lambda x} & x \geq 0 \\
0 & x < 0
\end{cases}
$$

Where the rate parameter $\lambda > 0$ controls the average waiting time until the first event occurs. And the **support** of $X$ is $x \in [0, \infty)$ as we can't have a negative waiting time but we can have a waiting time of 0 (the event happens immediately) or infinite (the event never happens).

Intuitively, the chance you're still waiting after some time $t$ decays **exponentially** fast as $t$ increases. This is the origin of the name. This follows from the poisson distribution where the PDF for a poisson random variable $X$ is given by:

$$
\mathbb{P}(X = k) = \frac{\lambda^k}{k!} e^{-\lambda}
$$

Where $k$ is the number of events that have happened in the interval. Then the probability that no event has happened in the interval is given by:

$$
\mathbb{P}(X = 0) = \frac{\lambda^0}{0!} e^{-\lambda} = e^{-\lambda}
$$

This leads us to the PDF of the exponential distribution as the first event happening after some interval is the same as no events happening in that interval, so we can think of the exponential distribution as the time until the first event in a Poisson process.

{{< callout type="proof" >}}
Let's check the PDF integrates to 1, which is a requirement for it to be a valid probability density function (PDF):

$$
\begin{align*}
\int_{-\infty}^{\infty} f_X(x)\,dx &= \int_0^\infty \lambda e^{-\lambda x} dx \\
&= \lambda \left[ -\frac{1}{\lambda} e^{-\lambda x} \right]_0^\infty \\
&= (0 - (-1)) = 1
\end{align*}
$$
{{< /callout >}}

The CDF gives the probability that the event has occurred by time $x$. Importantly for $x < 0$, the CDF is $F_X(x) = 0$ since we can't have a negative waiting time. For $x \geq 0$, we compute the CDF as follows:

$$
F_X(x) = \mathbb{P}(X \leq x) = \int_0^x \lambda e^{-\lambda t} dt = 1 - e^{-\lambda x}, \quad x \geq 0
$$

As $x \to \infty$, $F_X(x) \to 1$ the event will eventually occur almost surely, with probability 1:

$$
\lim_{x \to \infty} F_X(x) = \lim_{x \to \infty} (1 - e^{-\lambda x}) = 1
$$

For any $0 \leq a < b$,

$$
\mathbb{P}(a < X \leq b) = F_X(b) - F_X(a) = (1 - e^{-\lambda b}) - (1 - e^{-\lambda a}) = e^{-\lambda a} - e^{-\lambda b}
$$

So, the probability the waiting time is between $a$ and $b$ is simply the difference between the survival probabilities at those two times.

Just like the geometric distribution, the exponential distribution is **memoryless**. This means that the probability of waiting an additional $t$ units of time, given that you have already waited $s$ units, is the same as if you had just started waiting. In other words, the process does not "remember" how long you have already waited. Formally, for all $s, t \geq 0$:

$$
\mathbb{P}(X > s + t \mid X > s) = \mathbb{P}(X > t)
$$

Intuitively you can think of this as saying that the waiting time until the next event is independent of how long you have already waited as you don't know when the last event happened, so you can only expect to wait the same amount of time as you would if you just started waiting.

{{< callout type="proof" >}}
By definition of conditional probability:

$$
\mathbb{P}(X > s + t \mid X > s) = \frac{\mathbb{P}(X > s + t)}{\mathbb{P}(X > s)}
$$

Recall that $\mathbb{P}(X > x) = 1 - F_X(x) = e^{-\lambda x}$, so:

$$
\frac{e^{-\lambda(s + t)}}{e^{-\lambda s}} = e^{-\lambda t} = \mathbb{P}(X > t)
$$
{{< /callout >}}

The expectation and variance of an exponentially distributed random variable $X \sim \mathrm{Exp}(\lambda)$ are given by:

$$
\mathbb{E}[X] = \frac{1}{\lambda} \text{ and } \text{Var}(X) = \frac{1}{\lambda^2}
$$

{{< callout type="proof" title="Derivation of the expectation" >}}
Compute the expectation:
$$
\mathbb{E}[X] = \int_0^\infty x \lambda e^{-\lambda x} dx
$$
Integrate by parts, let $u = x$, $dv = \lambda e^{-\lambda x} dx$. Then $du = dx$, $v = -e^{-\lambda x}$:
$$
\mathbb{E}[X] = \left. -x e^{-\lambda x} \right|_0^\infty + \int_0^\infty e^{-\lambda x} dx = 0 + \frac{1}{\lambda} = \frac{1}{\lambda}
$$
{{< /callout >}}

{{< callout type="proof" title="Derivation of the variance" >}}
Similarly, $\mathbb{E}[X^2] = \frac{2}{\lambda^2}$, so:

$$
\text{Var}(X) = \mathbb{E}[X^2] - (\mathbb{E}[X])^2 = \frac{2}{\lambda^2} - \left(\frac{1}{\lambda}\right)^2 = \frac{1}{\lambda^2}
$$
{{< /callout >}}

{{< callout type="example" >}}
Buses arrive at a stop at an average rate of $\lambda = 6$ per hour (i.e., one every 10 minutes on average).  
What is the probability you wait **more than 15 minutes** for the next bus? First we need to convert the time to hours, since our rate is in hours. So $\lambda = 6$ per hour means we expect one bus every $\frac{1}{6}$ hours, or 10 minutes and 15 minutes is $\frac{15}{60} = 0.25$ hours.

$$
\mathbb{P}(X > 0.25) = e^{-\lambda \cdot 0.25} = e^{-6 \times 0.25} = e^{-1.5} \approx 0.2231
$$

So, **about a 22.3% chance** of waiting more than 15 minutes. The expected waiting time for the first bus is:

$$
\mathbb{E}[X] = \frac{1}{\lambda} = \frac{1}{6} \text{ hr} \approx 10 \text{ minutes}
$$

Which would match our average. The idea here is that say you arrive 5 minutes after the last bus left, then you would expect to wait 5 minutes for the next bus, but because the process is memoryless, the expected waiting time is still 10 minutes, because you don't know when the last bus left, so you can only expect to wait 10 minutes on average. 
{{< /callout >}}

## Normal Distribution

One of the most important continuous random variables in probability theory and statistics is the **normal distribution** (also called the **Gaussian distribution**, named after Carl Friedrich Gauss who defined the standardized form). Because of the shape of the function graph it is often also referred to as the **bell curve**.

The normal distribution plays a central role in statistics and probability. It is the chosen model for a huge variety of real-world phenomena, especially when these phenomena result from the accumulation of many small, independent effects such as physical measurements, noise, heights, test scores, and many natural quantities. The reason why it is so prevalent is due to the [Central Limit Theorem](), which states that the sum (or average) of a large number of independent and identically distributed random variables will be approximately normally distributed, regardless of the original distribution of the variables. This is a fundamental result in probability theory and underpins many statistical methods.

Let's now formally define the normal distribution. The random variable $X$ is said to have a **normal distribution** with **mean** $\mu$ and **variance** $\sigma^2$ denoted as:

$$
X \sim \mathcal{N}(\mu, \sigma^2)
$$

if its **probability density function (PDF)** is given by:

$$
f_{X}(x) = \frac{1}{\sqrt{2\pi \sigma^2}} \exp\left(-\frac{(x - \mu)^2}{2\sigma^2}\right)
$$


The expectation and variance of a normal random variable are the parameters $\mu$ and $\sigma^2$ respectively:

$$
\mathbb{E}[X] = \mu, \quad \text{Var}(X) = \sigma^2
$$

So the distribution is completely characterized by its mean and variance. From the graph and definition we can see that the normal distribution is symmetric around the mean $\mu$, and the spread of the distribution is controlled by the standard deviation $\sigma$. The PDF is bell-shaped, with the highest point at $x = \mu$ and tails that approach zero as $x$ moves away from $\mu$. Unlike other distributions, the normal distribution is defined for all real numbers, meaning it has infinite support. The interpretation of the graph is if you **measure the same physical quantity repeatedly**, the errors you get will, under mild conditions, look normal—most values are close to the "true" mean, with fewer large deviations, and very rare extreme deviations. Intuitively as the variance $\sigma^2$ increases, the distribution becomes wider and flatter, while as $\sigma^2$ decreases, it becomes narrower and taller as the values cluster more closely around the mean $\mu$ by the definition of variance.

The **cumulative distribution function (CDF)** of a normal random variable $X \sim \mathcal{N}(\mu, \sigma^2)$ is given by:

$$
F_X(a) = \mathbb{P}(X \leq a) = \int_{-\infty}^a f_X(x)\,dx
$$

There is **no closed form** for this integral in terms of elementary functions, but it is tabulated and implemented in all statistical packages.

{{< callout type="example" title="Probability of Student Height" >}}
Suppose the heights of students in a class are normally distributed with mean $\mu = 175$ cm and standard deviation $\sigma = 10$ cm. What is the probability that a randomly chosen student is shorter than 170 cm?

Let $X \sim \mathcal{N}(175, 10^2)$. We want $\mathbb{P}(X < 170)$.


$$
\mathbb{P}(X < 170) = F_X(170) \approx 0.3085
$$

(Using a standard normal table or calculator.) So about **31%** of students are shorter than 170 cm. 
{{< /callout >}}

Another reason why the normal distribution is so common is that it is **closed under linear transformations**. This means that if you apply a linear transformation to a normally distributed random variable, the result is still normally distributed. So if $X \sim \mathcal{N}(\mu, \sigma^2)$, then for any constant $c$:

$$
X + c \sim \mathcal{N}(\mu + c, \sigma^2)
$$

So adding a constant shifts the mean, but does not affect the variance. This is intuitive if you think of just shifting the entire distribution left or right by $c$ as the PDF is just the PDF of $X$ shifted by $c$.

{{< callout type="proof" >}}
Because the normal distribution is characterized by its mean and variance, we can show this property easily. Using the definition and properties of expectation and variance. Let $Y = X + c$. Then,

$$
\mathbb{E}[Y] = \mathbb{E}[X] + c = \mu + c \qquad \text{Var}(Y) = \text{Var}(X) = \sigma^2
$$
{{< /callout >}}

{{< callout type="example" >}}
Suppose all the students in the previous example were measured with their shoes on, and the shoes add exactly 4 cm to everyone's height. So to correct for this, we subtract 4 cm from each measurement. Then the new variable $Y = X - 4 \sim \mathcal{N}(171, 10^2)$. 

The mean is now 171 cm rather than 175 cm. But because we subtracted everyone's height by the same amount, the variance (spread) is unchanged.
{{< /callout >}}

We can also **scale** the normal distribution. This means multiplying the random variable by a constant $a$. If $X \sim \mathcal{N}(\mu, \sigma^2)$ and $a \in \mathbb{R}$:

$$
a X \sim \mathcal{N}(a\mu, a^2 \sigma^2)
$$

That is, **scaling** by $a$ multiplies the mean by $a$ and the variance by $a^2$.

{{< callout type="proof" >}}
Again, this follows from the properties of expectation and variance. If $Y = aX$, then:

$$
\mathbb{E}[Y] = a\mathbb{E}[X] = a\mu
\qquad \text{Var}(Y) = a^2 \text{Var}(X) = a^2 \sigma^2
$$

So we have $Y \sim \mathcal{N}(a\mu, a^2 \sigma^2)$. The PDF stretches or compresses according to $a$.
{{< /callout >}}

{{< callout type="example" >}}
Suppose the heights of parents are $X \sim \mathcal{N}(180, 10^2)$, but the students are, on average, 80% of their parents' heights. Then $Y = 0.8 X \sim \mathcal{N}(0.8 \times 180, 0.8^2 \times 100) = \mathcal{N}(144, 64)$.
{{< /callout >}}

We have seen for the binomial and poisson distributions that we can add independent random variables together, and the result is also a random variable of the same type. The same holds for the normal distribution. Specifically if $X_1 \sim \mathcal{N}(\mu_1, \sigma_1^2)$ and $X_2 \sim \mathcal{N}(\mu_2, \sigma_2^2)$ are **independent**, then:

$$
X_1 + X_2 \sim \mathcal{N}(\mu_1 + \mu_2, \sigma_1^2 + \sigma_2^2)
$$

That is, **the sum of independent normals is again normal**, with the mean and variances adding.

{{< callout type="proof" >}}
Again this follows from the properties of expectation and variance. If $Y = X_1 + X_2$, where $X_1$ and $X_2$ are independent normal random variables, then:

$$
\mathbb{E}[Y] = \mathbb{E}[X_1] + \mathbb{E}[X_2] = \mu_1 + \mu_2
\qquad
\text{Var}(Y) = \text{Var}(X_1) + \text{Var}(X_2) = \sigma_1^2 + \sigma_2^2
$$
{{< /callout >}}

{{< callout type="example" >}}
Suppose the heights of boys and girls in a school are independent and normally distributed:
- Boys: $X_1 \sim \mathcal{N}(175, 9^2)$
- Girls: $X_2 \sim \mathcal{N}(165, 8^2)$

Then you could model the total height of a pair of a boy and a girl as $Y = X_1 + X_2$. Then the sum $Y = X_1 + X_2 \sim \mathcal{N}(340, 145)$. 
{{< /callout >}}

Combining the above properties we can see that **any linear combination** of independent normal random variables is again normal:

If $X_1, \ldots, X_n$ are independent, and $Y = c_0 + \lambda_1 X_1 + \cdots + \lambda_n X_n$:

$$
Y \sim \mathcal{N}\left( c_0 + \sum_{i=1}^n \lambda_i \mu_i,\, \sum_{i=1}^n \lambda_i^2 \sigma_i^2 \right)
$$

This is the **stability property** of the normal distribution, and is one of the reasons for its fundamental importance.

{{< callout type="proof" >}}
Let $Y = c_0 + \sum_{i=1}^n \lambda_i X_i$. Then we have by the properties of expectation and variance:
- Mean: $\mathbb{E}[Y] = c_0 + \sum_{i=1}^n \lambda_i \mathbb{E}[X_i] = c_0 + \sum_{i=1}^n \lambda_i \mu_i$
- Variance: $\text{Var}(Y) = \sum_{i=1}^n \lambda_i^2 \text{Var}(X_i) = \sum_{i=1}^n \lambda_i^2 \sigma_i^2$
{{< /callout >}}

### Standard Normal Distribution

Because we can scale and shift the normal distribution, we can always reduce any normal distribution to the **standard normal distribution**. This is a special case of the normal distribution where the mean is 0 and the variance is 1, often denoted as $Z$:

$$
Z \sim \mathcal{N}(0, 1)
$$

We standardize it using the so-called **z-score transformation**:

$$
Z = \frac{X - \mu}{\sigma}
$$

This transformation centers the distribution at 0 and rescales the variance to 1. By removing the mean we center the distribution around 0, and by dividing by the standard deviation we ensure that the spread of the distribution is 1 because the variance of $Z$ is:

$$
\text{Var}(Z) = \text{Var}\left(\frac{X - \mu}{\sigma}\right) = \frac{\text{Var}(X)}{\sigma^2} = \frac{\sigma^2}{\sigma^2} = 1
$$

The result is the standard normal distribution. The **probability density function (PDF)** of the standard normal distribution is denoted as $\varphi(z)$ and is given by:

$$
\varphi(z) = \frac{1}{\sqrt{2\pi}} \exp\left(-\frac{z^2}{2}\right)
$$

The **cumulative distribution function (CDF)** is denoted as $\Phi(z)$:

$$
\Phi(z) = \mathbb{P}(Z \leq z) = \int_{-\infty}^z \varphi(t) dt
$$

In practice, **any normal probability can be reduced to a standard normal calculation** using this standardization. That is,

$$
\mathbb{P}(X \leq a) = \mathbb{P}\left( \frac{X - \mu}{\sigma} \leq \frac{a - \mu}{\sigma} \right) = \Phi\left( \frac{a - \mu}{\sigma} \right)
$$

This is actually the key to using the normal distribution in practice: we can always convert any normal random variable to a standard normal variable, compute probabilities using the standard normal CDF, and then interpret those results in terms of the original variable.

{{< callout type="proof" >}}
Let $X \sim \mathcal{N}(\mu, \sigma^2)$ and $Z = \frac{X-\mu}{\sigma}$ then we have:
- $\mathbb{E}[Z] = \frac{\mathbb{E}[X] - \mu}{\sigma} = 0$
- $\text{Var}(Z) = \frac{\text{Var}(X)}{\sigma^2} = 1$

Thus, $Z \sim \mathcal{N}(0,1)$, and we can write $\mathbb{P}(X \leq a) = \Phi\left( \frac{a - \mu}{\sigma} \right)$ as above. 
{{< /callout >}}

{{< callout type="example" title="Standardizing Student Heights" >}}
Suppose the heights of students are $X \sim \mathcal{N}(175, 10^2)$. What is the probability that a student is shorter than 170 cm? First, standardize:

$$
z = \frac{170 - 175}{10} = -0.5
$$

Then we look up $\Phi(-0.5) \approx 0.3085$. So about 31% of students are shorter than 170 cm. 
{{< /callout >}}

#### Quantiles of the Normal Distribution

Using the CDF, we can find the probability that a random variable $X$ takes on a value less than or equal to $a$. The **quantile function** (or inverse CDF) allows us to do the opposite: given a probability $a$, it gives us the value $c$ such that:

$$
\mathbb{P}(X \leq c) = a
$$

This is written as:

$$
c = \mu + \sigma \Phi^{-1}(a)
$$

What is the derivation of this? Because it is symmetric? 

where $\Phi^{-1}$ is the **inverse** of the standard normal CDF. For the standard normal distribution with mean $\mu = 0$ and standard deviation $\sigma = 1$, the quantile function is simply:

$$
c = \Phi^{-1}(a)
$$

For example for the **median** (50th percentile), we have:

$$
c = \mu + \sigma \Phi^{-1}(0.5) = \mu
$$

Quantiles are important for all distributions, but the normal distribution is so central to statistics where we often want to find a threshold for some sort of percentage such as for [hypothesis tests]() or [confidence intervals](). They are also useful for setting cutoffs, such as in outlier detection or acceptance intervals.

{{< callout type="example" >}}
Suppose we want to know the minimum height needed to be in the top 5% of students in the class ($X \sim \mathcal{N}(175, 10^2)$).

We want $c$ such that $\mathbb{P}(X > c) = 0.05$, or $\mathbb{P}(X \leq c) = 0.95$:

$$
c = 175 + 10 \cdot \Phi^{-1}(0.95) \approx 175 + 10 \times 1.645 = 191.45
$$

So a student must be at least **191.5 cm** tall to be in the top 5% of the class. 
{{< /callout >}}

### Three Sigma Rule

One of the most important practical properties of the normal distribution is the **three sigma rule**:

- About **68%** of values lie within **1 standard deviation** ($\mu \pm \sigma$)
- About **95%** within **2 standard deviations** ($\mu \pm 2\sigma$)
- About **99.7%** within **3 standard deviations** ($\mu \pm 3\sigma$)

Formally,

$$
\mathbb{P}(|X - \mu| < k \sigma) = 2\Phi(k) - 1
$$

for $k = 1, 2, 3$.

For example, for $k = 3$:

$$
\mathbb{P}(|X - \mu| \geq 3\sigma) = 1 - \mathbb{P}(|X - \mu| < 3\sigma) \approx 0.0027
$$

So, the probability of being "more than three sigmas away" is about 0.27%, which is why "three sigma events" are considered extremely rare. This fact is used, for example, in quality control, where measurements outside $\mu \pm 3\sigma$ are flagged as anomalies.

Surprisingly, this rule does not depend on the mean $\mu$ or variance $\sigma^2$ as can be seen from the formula above. This is a consequence of the symmetry and shape of the normal distribution, which is why it is so widely used in statistics.

## Constructing Random Variables

Up to now, we have talked about random variables as **functions defined on some probability space** and described their **distribution** via the cumulative distribution function (CDF), $F_X(a) = \mathbb{P}(X \leq a)$. But you might wonder: given any function $F: \mathbb{R} \to [0,1]$ that looks like a "CDF", does there actually exist a random variable $X$ (possibly on some probability space) whose distribution function is exactly $F$? And how could we construct such an $X$ in practice or in simulations?

This question lies at the heart of probability theory: it tells us that we can focus on the distribution of random variables—meaning their PMF, PDF, or CDF, without needing to worry about the detailed structure of the underlying probability space. 

Recall that for a function $F: \mathbb{R} \to [0,1]$ to be the CDF of some random variable, it must satisfy three properties:

1. **Non-decreasing:** $F(a_1) \leq F(a_2)$ whenever $a_1 < a_2$.
2. **Right-continuity:** For all $a$, $\lim_{x \to a^+} F(x) = F(a)$.
3. **Limits at infinity:** $\lim_{a \to -\infty} F(a) = 0$ and $\lim_{a \to +\infty} F(a) = 1$.

These are exactly the properties that arise from the definitions and basic results about probability measures. 

The question now is for any function $F$ with these properties, does there exist a random variable $X$ such that $F_X(a) = F(a)$ for all $a$? The answer is yes. For any function $F$ with these properties, there exists a probability space $(\Omega, \mathcal{F}, \mathbb{P})$ and a random variable $X: \Omega \to \mathbb{R}$ with $F_X(a) = \mathbb{P}(X \leq a) = F(a)$ for all $a$.

This is often called the **Kolmogorov existence theorem**. In practice, this means that when defining or analyzing a random variable, we can simply specify its distribution function (or PMF or PDF), and be assured that a corresponding random variable does exist.

This is useful because in applications (especially in statistics and simulation), we then do not need to describe the probability space $(\Omega, \mathcal{F}, \mathbb{P})$ explicitly. Instead, we work directly with the distributions of random variables. 

The idea of kolmogorov's theorem is as follows. Consider the "canonical" probability space: the infinite product space $\Omega = {0,1}^{\mathbb{N}}$, i.e., all infinite sequences of independent coin flips (where 0 = tails, 1 = heads) and the probability measure $\mathbb{P}$ is defined so that the coordinates are i.i.d. Bernoulli(1/2) random variables. We can now use such sequences to construct a wide variety of random variables, in particular the uniform distribution on $[0,1]$ which, in turn, can be used to generate any other desired distribution.

Let's generate a uniform random variable on $[0,1]$ using this framework, and then show how to use it to construct any other distribution. Suppose $X_1, X_2, \dots$ are i.i.d. Bernoulli(1/2) random variables (i.e., independent fair coin flips):

$$
U(\omega) = \sum_{n=1}^{\infty} \frac{X_n(\omega)}{2^n}
$$

You can think of this as a binary expansion of a number in the interval $[0,1]$. Each $X_n$ contributes either $0$ or $\frac{1}{2^n}$ to the sum, depending on whether the $n$-th coin flip is tails (0) or heads (1) which corresponds to the binary digit at position $n$ in the expansion. So for each $\omega \in \Omega$, this sum converges to a real number in $[0,1]$, whose binary expansion is given by the sequence $X_1, X_2, \dots$. So each binary digit is determined by a fair coin, and the resulting distribution is uniform.

Now, suppose we want to construct a random variable $X$ with any given distribution function $F$ (satisfying the three properties above). The key idea is to use **inverse transform sampling**.

Let $U$ be a uniform random variable on $[0,1]$. Define

$$
X = F^{-1}(U)
$$

where $F^{-1}$ is the **generalized inverse** (sometimes called the quantile function). We know this inverse exists because $F$ is non-decreasing and continuous. The generalized inverse is defined as:

$$
F^{-1}(a) = \inf \{ x \in \mathbb{R} : F(x) \geq a \}, \quad a \in [0,1]
$$

Then $X$ has distribution function $F$:

$$
\mathbb{P}(X \leq x) = \mathbb{P}(F^{-1}(U) \leq x) = \mathbb{P}(U \leq F(x)) = F(x)
$$

This construction works for any CDF, whether $F$ is continuous or has jumps (discrete points). If $F$ is strictly increasing and continuous, $F^{-1}$ is the usual inverse function. The idea is that we are transforming a uniform random variable into one with the desired distribution by mapping the uniform probabilities to the quantiles of the target distribution.
