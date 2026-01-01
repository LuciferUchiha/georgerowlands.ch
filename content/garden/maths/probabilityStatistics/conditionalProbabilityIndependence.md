---
title: Conditional Probability and Independence
type: docs
weight: 3
---

In some cases when performing a random experiment we may already have some prior information or possess incomplete information about the experiment. We may also be interested in knowing the probability of an outcome under certain conditions. This leads us to the concept of **conditional probability**.

The most simple example would be if we threw a die and then a friend told us that the die showed an even number. If we then wanted to know what the probability of the die showing a 6 is, so that we land on Mayfair in Monopoly, we would have to take into account that the die can only show an even number. In this case we have prior information about the die showing an even number which is the same as knowing incomplete information about the die and also wanting to know a probability under certain conditions. This is a conditional probability. 

Formally we want to know the probability of an event $A$ given that another event $B$ has already occurred. We denote this as $\mathbb{P}(A | B)$, which is read as the probability of $A$ given $B$. We define the conditional probability of $A$ given $B$ as follows:

$$
\mathbb{P}(A | B) = \frac{\mathbb{P}(A \cap B)}{\mathbb{P}(B)}
$$

Importantly to avoid division by 0, we need to have $\mathbb{P}(B) \neq 0$. These formula makes sense and can also be intuitively understood. 

{{< callout type="example" >}}
Let's look at this in the case of the die example. We have the following events in a laplace
- $A$: Die shows 6, so $A=\{6\}$
- $B$: Die shows an even number, so $B=\{2,4,6\}$
- $A \cap B$: Die shows 6 and an even number, so $A \cap B=\{6\}$

Because we have a laplace space, the probabilities are simple. But what is the conditional probability actually telling us. In the numerator we have the outcomes that are in both $A$ and $B$, so these are the events that interest us and can happen. In the denominator we have the outcomes that are in $B$, so these are the events that will happen. So the conditional probability is the ratio of the events that interest us and can happen to the events that will happen. For the die example we then get:

$$
\mathbb{P}(A | B) = \frac{\mathbb{P}(A \cap B)}{\mathbb{P}(B)} = \frac{1/6}{3/6} = \frac{1}{3}
$$

This means that the probability of the die showing a 6 given that it shows an even number is $\frac{1}{3}$. This is intuitive as we have 3 possible outcomes that are even and only one of them is a 6. 

{{< callout type="todo" >}}
This can also be nicely be shown with a Venn diagram
{{< /callout >}}
{{< /callout >}}

If it turns out that the prior information we are given happens to be the exact same as the event we are interested in, so $A=B$, then we know that the event $A$ will surely happen and we get the following:

$$
\mathbb{P}(A | A) = \frac{\mathbb{P}(A \cap A)}{\mathbb{P}(A)} = \frac{\mathbb{P}(A)}{\mathbb{P}(A)} = 1
$$

This again makes intuitively sense as if for example we know that the die shows a 6, then the probability of the die showing a 6 is 1. There is a similar scenario where the sample space is the condition. You can think of this as if your friend told you that the die is showing a number between 1 and 6. He might as well have told you nothing. In this case we have the following:

$$
\mathbb{P}(A | \Omega) = \frac{\mathbb{P}(A \cap \Omega)}{\mathbb{P}(\Omega)} = \frac{\mathbb{P}(A)}{1} = \mathbb{P}(A)
$$

We can also have the opposite case where the prior information makes it impossible for the event to happen, so $A \cap B = \emptyset$. For example if I told you that the die shows a 7, then the probability of the die showing a 6 is 0. This is also intuitive as the die can not show a 7. In this case we get:

$$
\mathbb{P}(A | B) = \frac{\mathbb{P}(A \cap B)}{\mathbb{P}(B)} = \frac{0}{\mathbb{P}(B)} = 0
$$

An important thing to notice is that prior information is not symmetric. So in other words one event might "increase" the probability of another event, but the other event does not "increase" the probability of the first event by the same amount if even at all. For example if we have two events $A$ and $B$, where $A$ is the die showing a 6 and $B$ is the die showing an even number, then we have:

$$
\mathbb{P}(A | B) = \frac{\mathbb{P}(A \cap B)}{\mathbb{P}(B)} = \frac{1/6}{3/6} = \frac{1}{3} \text{ and } \mathbb{P}(B | A) = \frac{\mathbb{P}(B \cap A)}{\mathbb{P}(A)} = \frac{1/6}{1/6} = 1
$$

Therefore in general we have that $\mathbb{P}(A | B) \neq \mathbb{P}(B | A)$. This is important to keep in mind as it can lead to confusion.

## Conditional Probability as a Probability Measure

By our [definition of a probability measure](/maths/probabilityStatistics/probabilitySpaces#probability-measure), it turns out that the conditional probability is also a probability measure. More precisely, if $B$ is an event with $\mathbb{P}(B) \neq 0$, then we can define the conditional probability as a probability measure on the event space $\mathcal{F}$ of the sample space $\Omega$ as follows:

$$
\mathbb{P}(\cdot \mid B): \mathcal{F} \to [0,1] \quad \text{with} \quad \mathbb{P}(A \mid B) = \frac{\mathbb{P}(A \cap B)}{\mathbb{P}(B)}
$$

Intuitively this makes sense as conditional probability is a way to "restrict" the universe of discourse to the outcomes where $B$ has occurred. In other words, we are only considering the outcomes that are in $B$, and then calculating the probability of $A$ within that restricted universe. 

To show that $\mathbb{P}(\cdot \mid B)$ is a probability measure, we need to check that it satisfies the Kolmogorov axioms:

- **Non-negativity**: For any event $A \in \mathcal{F}$, we have $\mathbb{P}(A \cap B) \geq 0$ (since probabilities are non-negative), and $\mathbb{P}(B) > 0$ (by assumption). Thus conditional probability is non-negative:

$$
\mathbb{P}(A \mid B) = \frac{\mathbb{P}(A \cap B)}{\mathbb{P}(B)} \geq 0
$$

- **Probability of the sample space**: The probability of the sample space $\Omega$ given $B$ must equal 1. This means we need to show $\mathbb{P}(\Omega \mid B) = 1$. Since $\Omega \cap B = B$ we have:

$$
\mathbb{P}(\Omega \mid B) = \frac{\mathbb{P}(\Omega \cap B)}{\mathbb{P}(B)} = \frac{\mathbb{P}(B)}{\mathbb{P}(B)} = 1
$$
- **Probability of the Empty Set**: Because the empty set $\emptyset$ is disjoint from any event and its probability is 0, we have:

$$
\mathbb{P}(\emptyset \mid B) = \frac{\mathbb{P}(\emptyset \cap B)}{\mathbb{P}(B)} = \frac{\mathbb{P}(\emptyset)}{\mathbb{P}(B)} = 0
$$

- **Countable Additivity**: For any **countable collection** of pairwise disjoint events $(A_i)_{i=1}^\infty$ in $\mathcal{F}$ (i.e., $A_i \cap A_j = \emptyset$ for $i \neq j$), we need to show:

$$
\mathbb{P}\left( \bigcup_{i=1}^{\infty} A_i \mid B \right) = \sum_{i=1}^{\infty} \mathbb{P}(A_i \mid B)
$$

First, note that for all $i \neq j$, $(A_i \cap B) \cap (A_j \cap B) = (A_i \cap A_j) \cap B = \emptyset \cap B = \emptyset$, so the events $A_i \cap B$ are also pairwise disjoint. Therefore,

$$
\mathbb{P}\left( \bigcup_{i=1}^{\infty} A_i \mid B \right)
= \frac{\mathbb{P}\left( \left( \bigcup_{i=1}^{\infty} A_i \right) \cap B \right)}{\mathbb{P}(B)}
= \frac{\mathbb{P}\left( \bigcup_{i=1}^{\infty} (A_i \cap B) \right)}{\mathbb{P}(B)}
$$

But by countable additivity of the original probability measure $\mathbb{P}$:

$$
\mathbb{P}\left( \bigcup_{i=1}^{\infty} (A_i \cap B) \right) = \sum_{i=1}^{\infty} \mathbb{P}(A_i \cap B)
$$

So,

$$
\begin{align*}
\mathbb{P}\left( \bigcup_{i=1}^{\infty} A_i \mid B \right) &= \frac{ \sum_{i=1}^{\infty} \mathbb{P}(A_i \cap B) }{ \mathbb{P}(B) } \\
&= \sum_{i=1}^{\infty} \frac{\mathbb{P}(A_i \cap B)}{\mathbb{P}(B)} \\
&= \sum_{i=1}^{\infty} \mathbb{P}(A_i \mid B)
\end{align*}
$$

## Stochastic Independence

We have seen that for certain events if we know that one event $B$ has already occurred, this can influence the probability of another event $A$. However, depending on the experiment we can also have that the prior information of $B$ has no influence on the event $A$. So the the probability of $A$ is the same as the probability of $A$ given $B$. This is called **stochastic independence** as the two events do not influence each other and can be done independently of each other. More formally we can write this as follows:

$$
\mathbb{P}(A | B) = \mathbb{P}(A) = \frac{\mathbb{P}(A \cap B)}{\mathbb{P}(B)}
$$

Where $\mathbb{P}(B) \neq 0$. This means that the probability of $A$ given $B$ is the same as the probability of $A$ without any prior information. If we multiply both sides by $\mathbb{P}(B)$ we get:

$$
\mathbb{P}(A \cap B) = \mathbb{P}(A) \cdot \mathbb{P}(B)
$$

This formula is preferred as it avoids the division by 0 and is more intuitive. The probability of $A$ and $B$ happening is the same as the probability of $A$ happening and then times the probability of $B$ happening. From this formula we can also see that independence is symmetric, so if $A$ and $B$ are independent, then $B$ and $A$ are also independent. As if we would divide both sides by $\mathbb{P}(A)$ we would get:

$$
\frac{\mathbb{P}(A \cap B)}{\mathbb{P}(A)} = \mathbb{P}(B) \Leftrightarrow \mathbb{P}(B | A) = \mathbb{P}(B)
$$

{{< callout type="example" >}}
An example of this would be if we have two coins and we want to know the probability of the second coin showing heads given that the first coin shows heads. So if we model our experiment as a laplace space, we have the following events:
- $A$: Coin 1 shows heads, so $A=\{(H,H),(H,T)\}$
- $B$: Coin 2 shows heads, so $B=\{(H,H),(T,H)\}$
- $A \cap B$: Coin 1 shows heads and Coin 2 shows heads, so $A \cap B=\{(H,H)\}$

We know that the probability of the second coin showing heads is $\frac{1}{2}$, either it shows heads or tails. If we know that the first coin shows heads, this does not change the probability of the second coin showing heads. intuitively this also makes sense as the two coins are physically independent of each other and we model them as such, so we do not somehow between the two coin tosses become a supernatural being and influence the second coin toss. So we have:

$$
\mathbb{P}(A | B) = \frac{\mathbb{P}(A \cap B)}{\mathbb{P}(B)} = \frac{1/4}{1/2} = \frac{1}{2} = \mathbb{P}(A)
$$

Because we know coin tosses are independent of each other we can then also easily calculate the probability of both coins showing heads. This is simply the probability of a coin showing heads times the probability of the other coin showing heads. So we have:

$$
\mathbb{P}(A \cap B) = \mathbb{P}(A) \cdot \mathbb{P}(B) = \frac{1}{2} \cdot \frac{1}{2} = \frac{1}{4}
$$
{{< /callout >}}

### Independence of Multiple Events

When discussing independence, it is important to extend the idea beyond just two events. For a collection of events $(A_i)_{i\in I}$, we say they are **stochastically independent** if knowing the outcome of any combination of these events does **not** provide information about the others, no matter how many we choose.

Formally, a finite collection $(A_1, \ldots, A_n)$ of events is called **independent** if for **every** subset $J \subseteq {1,\ldots, n}$ (with $|J|\geq 2$), the following holds:

$$
\mathbb{P}\left(\bigcap_{j\in J} A_j\right) = \prod_{j\in J} \mathbb{P}(A_j)
$$

This definition is stronger than just requiring that every pair of events is independent; **every** subset must satisfy the condition. For three events $A$, $B$, and $C$, this means checking:

$$
\begin{align*}
\mathbb{P}(A \cap B) &= \mathbb{P}(A)\cdot\mathbb{P}(B) \\
\mathbb{P}(A \cap C) &= \mathbb{P}(A)\cdot\mathbb{P}(C) \\
\mathbb{P}(B \cap C) &= \mathbb{P}(B)\cdot\mathbb{P}(C) \\
\mathbb{P}(A \cap B \cap C) &= \mathbb{P}(A)\cdot\mathbb{P}(B)\cdot\mathbb{P}(C)
\end{align*}
$$

Simply checking all pairs is not enough; the joint intersection must also match the product of their probabilities. For larger collections, the number of subsets grows rapidly, so the check becomes more involved.

{{< callout type="example" >}}
Suppose we roll two fair six-sided dice and define the following events:

- $A$: The **first die is even**; i.e., $\omega_1 \in {2,4,6}$
- $B$: The **second die is odd**; i.e., $\omega_2 \in {1,3,5}$
- $C$: **Both dice show $\leq 2$**; i.e., $\omega_1 \leq 2$ and $\omega_2 \leq 2$

Let's compute all relevant probabilities and check all required conditions for independence. There are $36$ possible outcomes in total, as each die has $6$ outcomes: $|\Omega|=6 \times 6 = 36$.

**Event $A$: First die is even**: Possible values: 2, 4, 6 for $\omega_1$, any value for $\omega_2$. Number of favorable outcomes: $3$ (even values) $\times 6 = 18$ so:

$$
\mathbb{P}(A) = \frac{18}{36} = \frac{1}{2}
$$

**Event $B$: Second die is odd**: Possible values: 1, 3, 5 for $\omega_2$, any value for $\omega_1$. Number of favorable outcomes: $3 \times 6 = 18$ so:

$$
\mathbb{P}(B) = \frac{18}{36} = \frac{1}{2}
$$

**Event $C$: Both dice are $\leq 2$**: Possible values: 1 or 2 for both dice. Number of favorable outcomes: $2 \times 2 = 4$ so:

$$
\mathbb{P}(C) = \frac{4}{36} = \frac{1}{9}
$$

Next, we compute the pairwise intersections. **$A \cap B$:** First die even **and** second die odd so $\omega_1 \in {2,4,6}$ ($3$ options) and $\omega_2 \in {1,3,5}$ ($3$ options). Number of outcomes: $3 \times 3 = 9$.

$$
\mathbb{P}(A \cap B) = \frac{9}{36} = \frac{1}{4}
$$


**$A \cap C$:** First die even **and** both dice $\leq 2$ so $\omega_1 \in {2}$ (since only 2 is even and $\leq 2$) and $\omega_2 \in {1,2}$. Number of outcomes: $1 \times 2 = 2$ so:

$$
\mathbb{P}(A \cap C) = \frac{2}{36} = \frac{1}{18}
$$

**$B \cap C$:** Second die odd **and** both dice $\leq 2$ so $\omega_2 \in {1}$ (since only 1 is odd and $\leq 2$) and $\omega_1 \in {1,2}$. Number of outcomes: $2 \times 1 = 2$.

$$
\mathbb{P}(B \cap C) = \frac{2}{36} = \frac{1}{18}
$$

Lastly we compute the triple intersection

**$A \cap B \cap C$:** First die even, second die odd, both dice $\leq 2$ so $\omega_1 = 2$ (only value both even and $\leq 2$) and $\omega_2 = 1$ (only value both odd and $\leq 2$). Only 1 outcome: $(2,1)$

$$
\mathbb{P}(A \cap B \cap C) = \frac{1}{36}
$$

Now we can summarize the probabilities and check independence:

| Events              | Probability      | Product of Probabilities                        | Independent? |
| ------------------- | ---------------- | ----------------------------------------------- | ------------ |
| $A$               | $\frac{1}{2}$  |                                                 |              |
| $B$               | $\frac{1}{2}$  |                                                 |              |
| $C$               | $\frac{1}{9}$  |                                                 |              |
| $A \cap B$        | $\frac{1}{4}$  | $\frac{1}{2}\cdot\frac{1}{2}$                 | Yes          |
| $A \cap C$        | $\frac{1}{18}$ | $\frac{1}{2}\cdot\frac{1}{9}$                 | Yes          |
| $B \cap C$        | $\frac{1}{18}$ | $\frac{1}{2}\cdot\frac{1}{9}$                 | Yes          |
| $A \cap B \cap C$ | $\frac{1}{36}$ | $\frac{1}{2}\cdot\frac{1}{2}\cdot\frac{1}{9}$ | Yes          |

So $A$, $B$, and $C$ are all independent events. By intuition you wouldn't think that $A$ and $C$ are independent, or that $B$ and $C$ are independent, but they are. This is because there is an equally likely chance of the first die being even and the second die being odd, regardless of whether both dice are $\leq 2$. 

If we also added the event $D$ for the sum of the two dice being at most 3 we would have the tuples

$$
D = \{(1,1),(1,2),(2,1)\} \text{ with } \mathbb{P}(D) = \frac{3}{36} = \frac{1}{12}
$$

If we then check the independence of $D$ with the other events we would get:

| Events              | Probability      | Product of Probabilities                        | Independent? |
| ------------------- | ---------------- | ----------------------------------------------- | ------------ |
| $A \cap D$        | $\frac{1}{36}$ | $\frac{1}{2}\cdot\frac{1}{12}$                 | No           |
| $B \cap D$        | $\frac{1}{36}$ | $\frac{1}{2}\cdot\frac{1}{12}$                 | No           |
| $C \cap D$        | $\frac{3}{36}$ | $\frac{1}{9}\cdot\frac{1}{12}$                 | No           |

So the events $A$, $B$, and $C$ are independent of each other, but the event $D$ is not independent of any of them. 
{{< /callout >}}

### Probabilistic vs Stochastic

In probability theory, the terms probabilistic and stochastic are often used, sometimes interchangeably, but they do have subtle differences in emphasis.

- Probabilistic simply refers to anything involving or governed by the laws of probability. When we say something is probabilistic, we mean that there is inherent randomness or uncertainty in its behavior or outcome, as opposed to being deterministic.
- Stochastic, on the other hand, typically describes systems or processes that evolve in time (or through multiple stages) where each step involves some random component. A stochastic process is a sequence of random experiments, where each experiment may depend on previous outcomes.

Stochastic independence is thus a special, more precise term: it means that, in a sequence of random experiments (a process), the outcome of one does not influence the others. This distinction is why we talk about stochastic processes (multi-step, random systems) rather than "probabilistic processes," even though both terms involve randomness.

### Physical Independence vs Stochastic Independence

It is important to note that stochastic independence does not have to mean that the events are physically independent of each other. Two events can be independent of each other despite the experiment only being conducted once physically. 

{{< callout type="example" >}}
An interesting example of this is if we throw two dice at the same time. The two dice are physically independent of each other. Now we have the following events:
- $A$: The first die is even, so $A=$\{(2,1),(2,2),(2,3),(2,4),(2,5),(2,6),(4,1),(4,2),(4,3),(4,4),(4,5),(4,6),(6,1),(6,2),(6,3),(6,4),(6,5),(6,6)\}. So $A$ has 18 outcomes out of 36 possible outcomes.
- $B$: The second die has a sum of 7, so $B=$\{(1,6),(2,5),(3,4),(4,3),(5,2),(6,1)\}. So $B$ has 6 outcomes out of 36 possible outcomes.
- $A \cap B$: The first die is even and the second die has a sum of 7, so $A \cap B=\{(2,5),(4,3),(6,1)\}$. So $A \cap B$ has 3 outcomes out of 36 possible outcomes.

You would think that it matters if the first die is even or not, but it does not. These two events are independent of each other despite their being a physical dependency. To show they are independent we just need to show that the $P(A \cap B) = \mathbb{P}(A) \cdot \mathbb{P}(B)$, so we have:

$$
\mathbb{P}(A \cap B) = \frac{3}{36} = \frac{1}{12} \text{ and } \mathbb{P}(A) \cdot \mathbb{P}(B) = \frac{18}{36} \cdot \frac{6}{36} = \frac{108}{1296} = \frac{1}{12}
$$

So the two events are independent of each other. Interestingly if we define the event slightly differently we get a different result:
- $A$: The first die is even, so $A=$\{(2,1),(2,2),(2,3),(2,4),(2,5),(2,6),(4,1),(4,2),(4,3),(4,4),(4,5),(4,6),(6,1),(6,2),(6,3),(6,4),(6,5),(6,6)\}. So $A$ has 18 outcomes out of 36 possible outcomes.
- $C$: The second die has a sum of 6, so $C=$\{(1,5),(2,4),(3,3),(4,2),(5,1)\}. So $C$ has 5 outcomes out of 36 possible outcomes. 
- $A \cap C$: The first die is even and the second die has a sum of 6, so $A \cap C=\{(2,4),(4,2)\}$. So $A \cap C$ has 2 outcomes out of 36 possible outcomes.

Now we have:

$$
\mathbb{P}(A \cap C) = \frac{2}{36} = \frac{1}{18} \text{ and } \mathbb{P}(A) \cdot \mathbb{P}(C) = \frac{18}{36} \cdot \frac{5}{36} = \frac{90}{1296} = \frac{5}{72}
$$

So now clearly the two events are not independent of each other. What is happening here? The reason for this is that within event $B$ the number of outcomes where the event $A$ has an influence is the same as for the ones it doesn't. In other words the number of outcomes where the first die is even is 3 and the number of outcomes where the first die is odd is also 3. So the event has no influence. However, in event $C$ the number of outcomes where the first die is even is 2 and the number of outcomes where the first die is odd is 3. So the parity of the first die matters.
{{< /callout >}}

## Conditional Independence

We have seen that two events $A$ and $B$ are independent if knowing $B$ gives us no information about $A$. However, in the real world, true independence is rare. Often, events are related through some common cause or context. This leads us to the concept of **conditional independence**.

Two events $A$ and $B$ are said to be **conditionally independent** given a third event $C$ if, once we know that $C$ has occurred, knowing $B$ provides no additional information about $A$. Formally, this is defined as:

$$
\mathbb{P}(A \cap B \mid C) = \mathbb{P}(A \mid C) \cdot \mathbb{P}(B \mid C)
$$

or equivalently (if $\mathbb{P}(B \cap C) > 0$):

$$
\mathbb{P}(A \mid B \cap C) = \mathbb{P}(A \mid C)
$$

This is a "weaker" form of independence because $A$ and $B$ might be dependent in general (unconditionally), but become independent when we fix the condition $C$. This is extremely useful in practice, especially in fields like machine learning (e.g., Naive Bayes classifiers) and causal inference, where we model complex systems by assuming that variables are independent given their direct causes.

{{< callout type="example" title="Shoe Size and Reading Ability" >}}
Consider two events for a randomly selected child:
- $A$: The child has a large shoe size.
- $B$: The child has high reading ability.

Are $A$ and $B$ independent? Surprisingly, no! They are positively correlated. If you pick a child with large feet, they are likely to be older, and older children tend to read better. So, knowing someone has large feet does give you information about their reading ability (it suggests they are older).

$$
\mathbb{P}(A \cap B) \neq \mathbb{P}(A) \cdot \mathbb{P}(B)
$$

However, if we condition on the child's **age** (let $C$ be the event "the child is 10 years old"), then shoe size and reading ability likely become independent. Among 10-year-olds, knowing a child has big feet doesn't tell you much about their reading skills.

$$
\mathbb{P}(A \cap B \mid C) = \mathbb{P}(A \mid C) \cdot \mathbb{P}(B \mid C)
$$

Here, $A$ and $B$ are **conditionally independent** given $C$, even though they are dependent unconditionally. The variable "Age" is a **confounder** that creates a spurious correlation between shoe size and reading ability.
{{< /callout >}}

## Product Rule

By rearranging the formula for conditional probability, we actually get a formula for the intersection of two events, so in other words a formula for when two events happen at the same time. This is called the **product rule**. We get the product rule by rearranging the formula for conditional probability by multiplying both sides by $\mathbb{P}(B)$:

$$
\mathbb{P}(A | B) = \frac{\mathbb{P}(A \cap B)}{\mathbb{P}(B)} \Leftrightarrow \mathbb{P}(A \cap B) = \mathbb{P}(A | B) \cdot \mathbb{P}(B)
$$

This is the product rule for two events. However, this also works the other way around. So in other words if we have the probability of $B$ given $A$, we can also rearrange the formula for conditional probability to get the product rule in the other direction: 

$$
\mathbb{P}(B | A) = \frac{\mathbb{P}(A \cap B)}{\mathbb{P}(A)} \Leftrightarrow \mathbb{P}(A \cap B) = \mathbb{P}(B | A) \cdot \mathbb{P}(A)
$$

So from this we can see that the product rule is symmetric or more formally we have:

$$
\mathbb{P}(A \cap B) = \mathbb{P}(A | B) \cdot \mathbb{P}(B) = \mathbb{P}(B | A) \cdot \mathbb{P}(A)
$$

This can be extended to more than two events, referred to as the **chain rule** of probability. For $n$ events $A_1, A_2, \ldots, A_n$ we have:

$$
\mathbb{P}(A_1 \cap A_2 \cap \ldots \cap A_n) = \mathbb{P}(A_1) \cdot \mathbb{P}(A_2 | A_1) \cdot \mathbb{P}(A_3 | A_1 \cap A_2) \cdots \mathbb{P}(A_n | A_1 \cap A_2 \cap \ldots \cap A_{n-1})
$$

{{< callout type="note" >}}
Do not confuse the **chain rule** in probability with the chain rule in calculus (differentiation). While they share the same name and a similar "chaining" structure, they apply to completely different mathematical concepts.
{{< /callout >}}

{{< callout type="proof" >}}
The proof of the above is rather simple if we use the definition of conditional probability we get:

$$
\begin{align*}
\mathbb{P}(A_1 \cap A_2 \cap \ldots \cap A_n) &= \mathbb{P}(A_1) \cdot \mathbb{P}(A_2 | A_1) \cdot \mathbb{P}(A_3 | A_1 \cap A_2) \cdots \mathbb{P}(A_n | A_1 \cap A_2 \cap \ldots \cap A_{n-1}) \\
&= \frac{\mathbb{P}(A_1)}{1} \cdot \frac{\mathbb{P}(A_1 \cap A_2)}{\mathbb{P}(A_1)} \cdot \frac{\mathbb{P}(A_1 \cap A_2 \cap A_3)}{\mathbb{P}(A_1 \cap A_2)} \cdots \frac{\mathbb{P}(A_1 \cap A_2 \cap \ldots \cap A_n)}{\mathbb{P}(A_1 \cap A_2 \cap \ldots \cap A_{n-1})} \\
&= \mathbb{P}(A_1 \cap A_2 \cap \ldots \cap A_n)
\end{align*}
$$

The multiplications become a telescoping product, where each term cancels with the denominator of the next term, leaving us with the original intersection probability.
{{< /callout >}}

{{< callout type="example" >}}
Suppose we have a standard deck of 52 playing cards. We draw two cards from the deck one after the other, without replacement. What is the probability that both cards are aces?

Let $A$ be the event "the first card is an ace" and $B$ be the event "the second card is an ace given that the first card was an ace". There are 4 aces in the deck so the probability that the **first card** is an ace is $\mathbb{P}(A) = \frac{4}{52}$.

If the first card is an ace, there are now 3 aces left out of 51 cards. So, the probability that the **second card** is an ace given the first was an ace is $\mathbb{P}(B \mid A) = \frac{3}{51}$.

By the product rule the probability that both cards drawn are aces is:

$$
\begin{align*}
\mathbb{P}(\text{first card is ace} \cap \text{second card is ace})
&= \mathbb{P}(A \cap B) = \mathbb{P}(A) \cdot \mathbb{P}(B \mid A) \\
&= \frac{4}{52} \cdot \frac{3}{51} = \frac{12}{2652} = \frac{1}{221} \approx 0.45\%
\end{align*}
$$
{{< /callout >}}

### Multi-Stage Random Experiments

A **multi-stage random experiment** consists of several random steps performed in sequence. Each step can depend on the outcomes of previous steps. These can be visualized with **tree diagrams** (or event trees), where each branch represents a possible outcome at a given stage, and each full path represents a sequence of outcomes leading to a final event. We can then use 2 simple rules to calculate the probabilities of events in these multi-stage experiments:

1. **Product Rule:**
   The probability of a specific path (sequence of outcomes) is the **product** of the conditional probabilities along the path. This is because each stage's outcome is "conditioned" on the previous outcomes so it follows directly from the product rule for conditional probabilities:

$$
\mathbb{P}(A_1 \cap A_2 \cap \ldots \cap A_n) = \mathbb{P}(A_1)\cdot\mathbb{P}(A_2|A_1)\cdot\mathbb{P}(A_3|A_1\cap A_2)\cdots\mathbb{P}(A_n|A_1\cap\cdots\cap A_{n-1})
$$

2. **Additivity Rule (Countable Additivity):**
   If several different paths lead to the same final event, the **probability of the event** is the **sum** of the probabilities of those paths. This comes from the axiom of countable additivity, since the different paths are disjoint as they represent distinct sequences of outcomes.

We can summarize the process for constructing a multi-stage experiment as follows:

1. **Model each stage**: Identify all possible outcomes at each step.
2. **Draw the tree**: Each node splits into branches for the possible next outcomes.
3. **Label probabilities**: Assign the correct (possibly conditional) probability to each branch.
4. **Compute path probabilities**: Multiply along the branches for each complete path.
5. **Add if needed**: If an event can happen through several paths, sum their probabilities.

{{< callout type="info" >}}
An important note is that for a probabilistic model the difference between throwing two coins and throwing one coin twice is or two coins one after another is not important. This is because we do not model any dependence between the two coins, so for example the first coin influencing the second coin, which could in practice be the case as the two coins hit each other in the air. So in other words the two events are independent of each other so it does not matter if we throw them at the same time or one after another. This is important to keep in mind as it can lead to confusion.
If you still find it confusing think of the possible tree diagrams and you will see that they are the same.
{{< /callout >}}

{{< callout type="example" title="Independent Stages" >}}
Suppose we draw two balls **with replacement** from an urn containing 3 red and 2 blue balls. So after each draw, we put the ball back into the urn. We want to find the probability of drawing two red balls. There are two stages in this experiment: the first draw and the second draw.

- **Stage 1**: We have two possible outcomes, we either draw a red ball or a blue ball. The probabilities are $P(R_1) = \frac{3}{5}$, $P(B_1) = \frac{2}{5}$.
- **Stage 2**: Because we replace the ball, the probabilities for the second draw are the same: $P(R_2|R_1) = P(R_2|B_1) = \frac{3}{5}$, etc.

The probability of getting two reds can then just be multiplied together.

$$
\mathbb{P}(R_1 \cap R_2) = \mathbb{P}(R_1)\cdot\mathbb{P}(R_2) = \frac{3}{5}\cdot\frac{3}{5} = \frac{9}{25}
$$

Here the stages are independent (replacement resets the urn each time) So it is a simple example without any conditions as the second stage does not depend on the first stage. To show the additivity rule, we can also calculate the probability of getting at least one red ball in two draws. This can happen in three ways: 

1. First red, second blue $\mathbb{P}(R_1 \cap B_2) = \mathbb{P}(R_1)\cdot\mathbb{P}(B_2) = \frac{3}{5}\cdot\frac{2}{5} = \frac{6}{25}$
2. First blue, second red $\mathbb{P}(B_1 \cap R_2) = \mathbb{P}(B_1)\cdot\mathbb{P}(R_2) = \frac{2}{5}\cdot\frac{3}{5} = \frac{6}{25}$
3. Both red $\mathbb{P}(R_1 \cap R_2) = \mathbb{P}(R_1)\cdot\mathbb{P}(R_2) = \frac{3}{5}\cdot\frac{3}{5} = \frac{9}{25}$

The total probability of getting at least one red ball is then:

$$
\mathbb{P}(\text{at least one red}) = \mathbb{P}(R_1 \cap B_2) + \mathbb{P}(B_1 \cap R_2) + \mathbb{P}(R_1 \cap R_2) = \frac{6}{25} + \frac{6}{25} + \frac{9}{25} = \frac{21}{25}
$$
{{< /callout >}}

{{< callout type="example" title="Dependent Stages" >}}
Suppose we draw two balls **without replacement** from an urn containing 2 white and 4 black balls. We want to find the probability of drawing two drawing 2 balls of the same color ($A$) or 2 balls of different colors ($B$)?

- **Stage 1:** We have two possible outcomes, we either draw a white ball or a black ball. The probabilities are:
$P(W_1) = {2 \over 6} = {1 \over 3}$, $P(S_1) = {4 \over 6} = {2 \over 3}$.

- **Stage 2:** After the first draw, only 5 balls remain. So if a white ball was drawn in the first stage we have the following probabilities for the second draw: $P(W|W) = {1 \over 5}$ and $P(S|W) = {4 \over 5}$, and if a black ball was drawn in the first stage we have: $P(W|S) = {2 \over 5}$ and $P(S|S) = {3 \over 5}$.

The probability of drawing two balls of the same color ($A$) can be calculated as follows:

$$
P(A)=P(WW) + P(SS) = {1 \over 3} \cdot {1 \over 5} + {2 \over 3} \cdot {2 \over 5} = {7 \over 15}
$$

For different colored balls:

$$
P(B)=P(WS) + P(SW) = {1 \over 3} \cdot {4 \over 5} + {2 \over 3} \cdot {2 \over 5} = {8 \over 15}
$$

Because drawing two different colored balls is actually the complement of drawing two balls of the same color, we can also write this as:

$$
P(B) = 1 - P(A) = 1 - {7 \over 15} = {8 \over 15}
$$

The tree diagram for this experiment would look like this:

{{< figure
  src="/images/maths/probabilityMultiStage.png"
  alt="Tree diagram for a multi-stage random experiment with dependent stages"
  caption="Tree diagram for a multi-stage random experiment with dependent stages"
>}}
{{< /callout >}}

## Sum Rule (Law of Total Probability)

In some cases we may have a bunch of conditional probabilities so, the probabilities of an event $A$ given that another event $B$ has already occurred but we actually want to know the total probability of $A$. This leads to the **law of total probability**, also often referred to as the **sum rule** or **marginalization**.

More precisely, if we have the exhaustive mutually exclusive events $B_1, B_2, \ldots, B_n$ so in other words $B_1 \cup B_2 \cup \ldots \cup B_n = \Omega$ and $B_i \cap B_j = \emptyset$ for $i \neq j$. You can also think of these events $B_i$ as [partitions](/maths/discrete/setTheory#partitions) of the sample space $\Omega$. Then we can write the total probability of an event $A$ as follows:

$$
\mathbb{P}(A) = \sum_{i=1}^{n} \mathbb{P}(A | B_i) \cdot \mathbb{P}(B_i) = \sum_{i=1}^{n} \mathbb{P}(A \cap B_i)
$$

This comes from the product rule and the fact that the events $A_i$ are mutually exclusive. Each term in the sum represents the probability of $B$ occurring given that one of the events $A_i$ has occurred which from the product rule is the same as the events $A_i$ happening and $B$ happening at the same time, so $A_i \cap B$. Because the events $A_i$ are exhaustive they cover the entire sample space $\Omega$, so when we sum over all the events $A_i$ we get the total probability of $B$. We can also prove this formally using the distributivity of set operations. We have:

$$
A = A \cap \Omega = A \cap (B_1 \cup B_2 \cup \ldots \cup B_n) = (A \cap B_1) \cup (A \cap B_2) \cup \ldots \cup (A \cap B_n)
$$
 
And because the events $B_i$ are mutually exclusive we can use countable additivity to get:

$$
\mathbb{P}(A) = \mathbb{P}(A \cap B_1) + \mathbb{P}(A \cap B_2) + \ldots + \mathbb{P}(A \cap B_n) = \sum_{i=1}^{n} \mathbb{P}(A \cap B_i) = \sum_{i=1}^{n} \mathbb{P}(A | B_i) \cdot \mathbb{P}(B_i)
$$

{{< callout type="example" >}}
As an example for the law of total probability, let's consider a football match. We have the following events:
- $A$: The team wins.
- $B$: The team plays at home.
- $B^c$: The team plays away.

Now if we wanted to know the probability of the team winning, we could use the law of total probability. For this we need to know the probability of the team winning at home and away and the probability of the team playing at home and away. So for example we could have:
- $P(A | B) = 0.8$ for the probability of the team winning at home.
- $P(A | B^c) = 0.3$ for the probability of the team winning away.
- $P(B) = 0.6$ for the probability of the team playing at home.
- $P(B^c) = 1 - P(B) = 0.4$ for the probability of the team playing away.

Then we can use the law of total probability to get:

$$
\mathbb{P}(A) = \mathbb{P}(A | B) \cdot \mathbb{P}(B) + \mathbb{P}(A | B^c) \cdot \mathbb{P}(B^c) = 0.8 \cdot 0.6 + 0.3 \cdot 0.4 = 0.48 + 0.12 = 0.6
$$

This means that the probability of the team winning is 0.6 or 60% with the given probabilities. This is also intuitive from the numbers as the team wins 80% of the time at home and 30% of the time away and they play 60% of the time at home and 40% of the time away. So team is more likely to win.
{{< /callout >}}

## Bayes' Theorem

Another common case is we are given a conditional probability and we want to know the reverse of the conditional probability. So we have the probability of an event $B$ given that another event $A$ has already occurred, so $\mathbb{P}(B | A)$ and we want to know the probability of $A$ given that $B$ has already occurred, so $\mathbb{P}(A | B)$. This is called **Bayes' theorem**. Bayes' theorem follows from the product rule and the sum rule. More formally if we again have the events $B_1, B_2, \ldots, B_n$ which partition the sample space $\Omega$ then we can write Bayes' theorem as follows for the events $A$ and $B_i$:

$$
\mathbb{P}(B_i | A) = \frac{\mathbb{P}(B_i \cap A)}{\mathbb{P}(A)} = \frac{\mathbb{P}(A | B_i) \cdot \mathbb{P}(B_i)}{\sum_{j=1}^{n} \mathbb{P}(A | B_j) \cdot \mathbb{P}(B_j)}
$$

Importantly we need to have $\mathbb{P}(A) \neq 0$ and $\mathbb{P}(B_i) \neq 0$. This is also called the **inverse probability** as we are looking for inverse of the conditional probability that we already have.

{{< callout type="example" title="Sickness Test" >}}
One of the most common examples of Bayes' theorem is the test for a sickness as it can be quiet counter intuitive and paradoxical/scary. We want to develop a test for a rare sickness. We know that the sickness occurs in 0.01% of the population so $\frac{1}{10000}$ people have the sickness. We also know that the test very accurate. If the person really is sick, the test will show positive with a probability of 99%. So we can model this experiment. 

We create our sample space $\Omega$ as a set of tuples $(X,Y)$ where each tuple represents a person and $X$ is the event of the person being sick and $Y$ is the event of the test showing positive. We use boolean values for $X$ and $Y$. So we have the following events:
- $S$: The person is sick, so $S=\{(1,0),(1,1)\}$
- $T$: The test shows positive, so $T=\{(0,1),(1,1)\}$
- etc.

From the information we have we can get the following probabilities:
- $\mathbb{P}(S) = 0.0001$ for the probability of a person being sick.
- $\mathbb{P}(T | S) = 0.99$ for the probability of the test showing positive given that the person is sick.
- $\mathbb{P}(T | S^c) = 0.01$ for the probability of the test showing positive given that the person is not sick. This follows from the fact that if the test is positive, then there are only two possibilities: the person is sick and the test shows positive or the person is not sick and the test shows positive. 

What we now want to know is if I take the test and it shows positive, what is the probability that I am actually sick. So we want to know $\mathbb{P}(S | T)$. We can use Bayes' theorem to get this:

$$
\mathbb{P}(S | T) = \frac{\mathbb{P}(T | S) \cdot \mathbb{P}(S)}{\sum_{i=1}^{2} \mathbb{P}(T | S_i) \cdot \mathbb{P}(S_i)} = \frac{0.99 \cdot 0.0001}{0.99 \cdot 0.0001 + 0.01 \cdot 0.9999} \approx 0.0098 = 0.98\%
$$

No, you are not seeing that wrong. Despite the test being 99% accurate, the probability of you actually being sick is only 0.98%. Why is this happening? We can split the people that will have a positive test result into two groups:
1. People that are sick and have a correct positive test result. Because the sickness is so rare, this group represents roughly $\frac{1}{10000}$ of the population, so 0.01% of the population.
2. The other group is the people that are not sick and have a false positive test result. This group represents roughly $\frac{1}{100}$ of the population, so 1% of the population.

Given that the test is positive, it is much more likely that you are in the second group than in the first group. This is because the test is so accurate, but the sickness is so rare. This is a common example of how Bayes' theorem can lead to counter intuitive results and it is important to keep this in mind when interpreting test results.
{{< /callout >}}

## Birthday Paradox

{{< callout type="example" title="Birthday Problem" >}}
{{< callout type="todo" >}}
This is in exercise sheet 2 and is a very common example of how certain probabilities are often misestimate intuitively.

Write this better and show the solutions and proof clearly.
{{< /callout >}}

We ask: "What is the probability that at least two people in a group of $k$ people share the same birthday?"

To answer this, we first consider the probability that **no** two people share a birthday:

For 2 people: ${365 \over 365} \cdot {364 \over 365}$  
For 3 people: ${365 \over 365} \cdot {364 \over 365} \cdot {363 \over 365}$  
etc.  

This probability approaches 0 as $k$ increases. Thus, we can answer our question as follows:

$$
P(\text{same})=1-P(\text{different}) \Leftrightarrow P(A)=1- \frac{365 \cdot (365-1)\cdot...\cdot (365-n+1)}{365^n}
$$

for a class of 23 people this is already above 50%.
{{< /callout >}}

## Simpson's Paradox

{{< callout type="example" title="Simpson's Paradox" >}}
This was also discussed in the second exercise sheet. With regard to bias towards male or female students in the admission process of a university. Does this belong in this section or further up?
{{< /callout >}}