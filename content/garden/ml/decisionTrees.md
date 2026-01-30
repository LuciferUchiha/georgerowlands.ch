---
title: Decision Trees and Ensemble Methods
type: docs
weight: 10
---

Decision trees are a fundamental class of models for both classification and regression that recursively partition the input space into regions, making predictions based on which region a new input falls into. While individual trees are interpretable and fast, they suffer from several limitations that motivate **ensemble methods**, which combine multiple models to achieve better generalization. Some of the key limitations of single decision trees include:

1. **High Variance**: Small changes in the training data can lead to very different trees. This is because the greedy splitting procedure makes hard decisions at each node and a different split at the root cascades to completely different subtrees.
2. **Bias Toward Features with Many Levels**: Features with more unique values have more potential split points and may be favored regardless of their predictive power.
3. **Axis-Aligned Boundaries**: Standard trees only make splits parallel to feature axes, making them inefficient for diagonal decision boundaries.
4. **Instability**: Because trees make hard partitions of the input space, they cannot express smooth decision boundaries or interpolate between regions.

These limitations, especially high variance, motivate ensemble methods. We will focus primarily on classification trees and then develop the theory of ensembles, including bagging (random forests) and boosting (AdaBoost).

## Decision Trees

A decision tree consists of **decision nodes** (internal nodes) and **leaf nodes** (terminal nodes). Each decision node asks a question about an input feature and splits the data based on the answer. Each leaf node contains a prediction: either a class label (classification) or a continuous value (regression).

Mathematically, a decision tree defines a **partition of the input space** $\mathcal{X} \subseteq \mathbb{R}^d$ into disjoint regions $R_1, R_2, \ldots, R_M$ (one per leaf), where $\bigcup_{m=1}^M R_m = \mathcal{X}$ and $R_i \cap R_j = \emptyset$ for $i \neq j$. Each region $R_m$ corresponds to a unique path from the root to a leaf, defined by a conjunction of feature tests. The prediction function is then:

$$
\hat{f}(\mathbf{x}) = \sum_{m=1}^M c_m \cdot \mathbb{1}[\mathbf{x} \in R_m]
$$

where $c_m$ is the prediction for region $R_m$. The indicator function $\mathbb{1}[\mathbf{x} \in R_m]$ equals 1 if $\mathbf{x}$ belongs to region $R_m$ and 0 otherwise. Since exactly one region contains $\mathbf{x}$, exactly one indicator is 1 and all others are 0, so the sum returns $c_m$ for whichever region $m$ contains $\mathbf{x}$.

For **classification**, $c_m$ is a class label (the majority class in region $R_m$). For **regression**, $c_m$ is a continuous value (typically the mean of all target values in region $R_m$). This makes the prediction function a **piecewise constant function** over the input space. The decision tree approximates the true underlying function by dividing the space into regions and assigning a constant prediction within each region.

{{< callout type="example" >}}
Consider a binary classification tree with $d=2$ features ($x_1, x_2$) and 3 leaf regions:
- $R_1$: $\{x_1 \leq 5\}$ predicts class $+1$ (so $c_1 = +1$)
- $R_2$: $\{x_1 > 5, x_2 \leq 3\}$ predicts class $-1$ (so $c_2 = -1$)
- $R_3$: $\{x_1 > 5, x_2 > 3\}$ predicts class $+1$ (so $c_3 = +1$)

For input $\mathbf{x} = (6, 2)$: Since $x_1 = 6 > 5$ and $x_2 = 2 \leq 3$, we have $\mathbf{x} \in R_2$, so:
$$\hat{f}(\mathbf{x}) = (+1) \cdot 0 + (-1) \cdot 1 + (+1) \cdot 0 = -1$$
{{< /callout >}}

The goal is to construct a tree that partitions the input space such that each partition (leaf) contains examples that are as "pure" as possible, ideally all belonging to the same class. Decision tree learning proceeds by **recursive partitioning**: starting with the entire training dataset $\mathcal{D} = \{(\mathbf{x}_i, y_i)\}_{i=1}^n$ at the root node, we:

1. **Select the best split**: Choose the feature and threshold that maximizes some purity criterion
2. **Partition the data**: Split the data into subsets based on this split
3. **Recurse**: Apply the same procedure to each subset
4. **Stop**: When a stopping criterion is met, create a leaf node

For a **binary split** on a continuous feature $x_j$ with threshold $t$, we partition the data $\mathcal{D}_m$ at node $m$ into:
- Left child: $\mathcal{D}_{m_L} = \{(\mathbf{x}, y) \in \mathcal{D}_m : x_j \leq t\}$
- Right child: $\mathcal{D}_{m_R} = \{(\mathbf{x}, y) \in \mathcal{D}_m : x_j > t\}$

**Stopping criteria (pre-pruning)** control when to stop splitting and create a leaf instead:
- **Maximum depth**: Stop when the tree reaches a specified depth
- **Minimum samples per node**: Stop when a node has fewer than $k$ samples
- **Minimum impurity decrease**: Stop when the best split improves impurity by less than $\epsilon$
- **Pure node**: Stop when all examples belong to the same class

Without proper stopping criteria, trees continue splitting until every leaf is pure, containing only one class or even a single training example. This leads to **deep trees** that achieve perfect training accuracy but severely overfit:
- **High variance**: Extremely sensitive to small changes in training data
- **Poor generalization**: Decision boundaries fit noise rather than true patterns
- **Memorization**: The tree memorizes the training set instead of learning generalizable patterns

The stopping criteria provide **regularization** by limiting tree complexity and controlling the bias-variance tradeoff. Shallow trees have higher bias but lower variance, while deep trees have lower bias but higher variance.

{{< figure
    src="/images/ml/decisionTree.png"
    alt="Example of a decision tree with decision nodes (internal nodes) that test features and leaf nodes (terminal nodes) that make predictions."
    caption="Example of a decision tree with decision nodes (internal nodes) that test features and leaf nodes (terminal nodes) that make predictions."
>}} 

## Classification Trees

In classification, each leaf predicts the majority class among the training examples that reach it. Let $\mathcal{D}_m$ denote the set of training examples at node $m$, with $|\mathcal{D}_m| = n_m$ examples. For a $K$-class problem, we can then define the **class proportions** at node $m$:

$$
\hat{p}_{mk} = \frac{1}{n_m} \sum_{(\mathbf{x}, y) \in \mathcal{D}_m} \mathbb{1}[y = k]
$$

This is the fraction of examples at node $m$ belonging to class $k$. The predicted class at node $m$ is then the majority class:

$$
\hat{y}_m = \arg\max_k \hat{p}_{mk}
$$

{{< figure
    src="/images/ml/classificationTree.gif"
    alt="Animation showing how a classification tree grows and creates decision boundaries. As the tree grows deeper, the decision boundaries become increasingly complex and tightly fit the training data."
    caption="Animation showing how a classification tree grows and creates decision boundaries. As the tree grows deeper, the decision boundaries become increasingly complex and tightly fit the training data."
>}}

As we can see from the figure, decision trees can be quite prone to overfitting if grown too deep. This can be visually seen by the slim decision boundaries that tightly fit the training data, carving out narrow regions to accommodate individual training points. **Outliers and noisy examples** exacerbate this problem: as the tree grows deeper, it creates isolated decision regions around individual points that don't follow the general pattern, learning the noise rather than the underlying decision boundary. To prevent this, we need to carefully choose splits that improve the purity of the nodes while controlling tree complexity through appropriate stopping criteria.

### Impurity Measures

To decide which split is "best," we need to quantify how "pure" or "impure" a node is. A node is pure if all examples belong to the same class ($\hat{p}_{mk} = 1$ for some $k$ and 0 for all others). 

#### Misclassification Rate

The simplest impurity measure is the **misclassification rate**:

$$
E(m) = 1 - \max_k \hat{p}_{mk}
$$

This is the fraction of examples that would be misclassified if we predict the majority class. For binary classification with $p = \hat{p}_{m1}$:

$$
E(m) = \min(p, 1-p)
$$

The misclassification rate is intuitive as it directly measures the error we would make. However, it is **not a good splitting criterion** because it is piecewise linear. Consider two potential splits that both achieve the same majority class in each child but with different class proportions. The misclassification rate may be identical for both, even though one split produces much purer children. The misclassification rate does not reward making both children purer, only the majority class matters.

{{< callout type="example" >}}
Consider a parent node with 100 examples and two candidate splits (assume equal-sized children for simplicity):

**Split A**: Creates children with proportions $(p=0.4, 1-p=0.6)$ and $(p=0.3, 1-p=0.7)$
- Child 1: 40% class 1, 60% class 2, so $E = \min(0.4, 0.6) = 0.4$
- Child 2: 30% class 1, 70% class 2, so $E = \min(0.3, 0.7) = 0.3$
- Total: $(0.4 + 0.3)/2 = 0.35$

**Split B**: Creates children with $(p=0.45, 1-p=0.55)$ and $(p=0.45, 1-p=0.55)$
- Child 1: $E = \min(0.45, 0.55) = 0.45$
- Child 2: $E = \min(0.45, 0.55) = 0.45$
- Total: $(0.45 + 0.45)/2 = 0.45$

Split A has lower weighted misclassification rate (0.35 vs 0.45), but the metric fails to distinguish between splits with the same majority class but different purities. As we'll see, Gini provides a smoother gradient.
{{< /callout >}}

#### Gini Impurity

The **Gini impurity** (or Gini index) measures the probability of misclassifying a randomly chosen example if it were labeled according to the class distribution at the node, where $\hat{p}_{mk}$ is the probability of class $k$ at node $m$:

$$
G(m) = \sum_{k=1}^K \hat{p}_{mk}(1 - \hat{p}_{mk}) = 1 - \sum_{k=1}^K \hat{p}_{mk}^2
$$

For binary classification with $p = \hat{p}_{m1}$ we get from substituting $\hat{p}_{m2} = 1 - p$:

$$
\begin{align*}
G(m) &= p(1 - p) + (1 - p)(1 - (1 - p)) \\
&= p(1 - p) + (1 - p)p \\
&= 2p(1 - p)
\end{align*}
$$

The Gini impurity ranges from 0 (pure) to a maximum when classes are equally distributed. For binary classification, the maximum is $G = 0.5$ when $p = 0.5$.

Intuitevely we are randomly picking two examples from node $m$ with replacement and labeling the second according to the class distribution. The Gini impurity is the probability that they belong to different classes. If the node is pure, this probability is 0. If classes are equally mixed, this probability is maximized.

While both Gini impurity and misclassification rate quantify node impurity and reach their maximum at $p = 0.5$ for binary classification, they differ crucially in their sensitivity to changes in class proportions. The misclassification rate is **piecewise linear** (with a corner at $p=0.5$), while Gini is **strictly concave** (a smooth parabola).

This difference has important implications for splitting. Consider two potential splits in binary classification:
- Split A creates children with $(p=0.4, 1-p=0.6)$ and $(p=0.3, 1-p=0.7)$
- Split B creates children with $(p=0.45, 1-p=0.55)$ and $(p=0.45, 1-p=0.55)$

Using misclassification rate:
- Split A: $E = \min(0.4, 0.6) + \min(0.3, 0.7) = 0.4 + 0.3 = 0.7$
- Split B: $E = \min(0.45, 0.55) + \min(0.45, 0.55) = 0.45 + 0.45 = 0.9$

Using Gini impurity:
- Split A: $G = 2(0.4)(0.6) + 2(0.3)(0.7) = 0.48 + 0.42 = 0.9$
- Split B: $G = 2(0.45)(0.55) + 2(0.45)(0.55) = 0.495 + 0.495 = 0.99$

Both prefer Split A (the purer split), but Gini provides a **smooth gradient** that consistently rewards purer nodes across all probability ranges, not just when the majority class changes. This makes Gini a much better splitting criterion, as it provides a clear signal pushing toward purer nodes throughout the entire probability space.

Gini is often preferred in practice because it is computationally cheap, differentiable everywhere, and tends to produce similar trees to entropy-based methods.

{{< callout type="example" >}}
Using the same splits from the misclassification rate example:

**Split A**: Children with $(p=0.4)$ and $(p=0.3)$
- Child 1: $G = 2(0.4)(0.6) = 0.48$
- Child 2: $G = 2(0.3)(0.7) = 0.42$
- Total: $(0.48 + 0.42)/2 = 0.45$

**Split B**: Children with $(p=0.45)$ and $(p=0.45)$
- Child 1: $G = 2(0.45)(0.55) = 0.495$
- Child 2: $G = 2(0.45)(0.55) = 0.495$
- Total: $(0.495 + 0.495)/2 = 0.495$

Split A has lower Gini impurity (0.45 vs 0.495), correctly identifying it as producing purer children. The difference (0.045) provides a smooth gradient for optimization, unlike the piecewise linear misclassification rate.
{{< /callout >}}

{{< callout type="proof" >}}
For binary classification, the Gini impurity is $G(p) = 2p(1-p)$ where $p \in [0,1]$. To show strict concavity, compute the second derivative:

$$
\frac{dG}{dp} = 2(1-p) - 2p = 2 - 4p
$$

$$
\frac{d^2G}{dp^2} = -4 < 0
$$

Since the second derivative is strictly negative everywhere, $G(p)$ is strictly concave. This means Gini has a smooth parabolic shape that consistently rewards purer splits across all probability ranges, providing clear gradients for optimization.
{{< /callout >}}

#### Entropy and Information Gain

**Entropy** measures the uncertainty or disorder in the class distribution. In information theory, the **Shannon entropy** (see [entropy in variational inference](/garden/ml/bayesian/variationalinference/#entropy)) of a discrete random variable $Z$ with probability mass function $p(z)$ is defined as:

$$
H[Z] = \mathbb{E}[-\log_2 p(Z)] = -\sum_{z} p(z) \log_2 p(z)
$$

This quantifies the expected "surprise" or information content when observing $Z$. The expectation form makes clear that entropy is the average information content across all possible outcomes. Note that entropy is defined for **discrete** random variables. For classification, the class label is naturally discrete. For regression trees with continuous targets, we instead use variance as the splitting criterion, as entropy is not defined for continuous distributions in the same way (differential entropy exists but doesn't have the same properties).

For a classification node $m$, we view the class label as a discrete random variable with distribution given by the empirical class proportions $\hat{p}_{mk}$. The **node entropy** is:

$$
H(m) = -\sum_{k=1}^K \hat{p}_{mk} \log_2 \hat{p}_{mk}
$$

By convention, $0 \log 0 = 0$. For binary classification ($K=2$) with $p = \hat{p}_{m1}$, we can rewrite this using $\hat{p}_{m2} = 1 - p$:

$$
\begin{align*}
H(m) &= -\sum_{k=1}^2 \hat{p}_{mk} \log_2 \hat{p}_{mk} \\
&= -\hat{p}_{m1} \log_2 \hat{p}_{m1} - \hat{p}_{m2} \log_2 \hat{p}_{m2} \\
&= - p \log_2 p - (1 - p) \log_2 (1 - p)
\end{align*}
$$

Entropy has the following properties:
- $H(m) = 0$ when the node is pure (all examples in one class)
- $H(m) = \log_2 K$ when all classes are equally represented (maximum uncertainty)
- For binary classification: $H(m) = 1$ when $p = 0.5$ (maximum uncertainty)

Entropy is more sensitive to changes in class probabilities near 0 and 1 than Gini, due to the logarithm approaching $-\infty$ as its argument approaches 0.

The entropy formula quantifies the expected "surprise" or uncertainty about the class label at node $m$. If we view the class distribution $(\hat{p}_{m1}, \ldots, \hat{p}_{mK})$ as a probability distribution, then $H(m)$ measures how uncertain we are about which class a random example from node $m$ belongs to.

This connection to information theory gives us a principled way to evaluate splits through **information gain**. Recall from [information theory](/garden/ml/bayesian/activelearning/#mutual-information) that the **mutual information** between random variables $C$ (class label) and $S$ (split indicator) is:

$$
I(C; S) = H[C] - H[C \mid S]
$$

This measures the expected reduction in uncertainty about $C$ when we observe $S$. The **conditional entropy** $H[C \mid S]$ is defined as:

$$
H[C \mid S] = \mathbb{E}_{S}[H[C \mid S = s]] = \sum_{s} p(s) H[C \mid S = s]
$$

This is the expected entropy of $C$ after observing $S$, weighted by the probability of each outcome $s$ of $S$.

In the context of decision trees at node $m$, let $C$ denote the class label and consider a candidate split that partitions the node into children. For a **binary split** into left child $m_L$ and right child $m_R$, the split indicator $S$ takes two values corresponding to the two children. Each example at node $m$ goes to either $m_L$ or $m_R$, with probabilities:

$$
p(S = L) = \frac{n_{m_L}}{n_m}, \quad p(S = R) = \frac{n_{m_R}}{n_m}
$$

The conditional entropy after the split is:

$$
H[C \mid S] = \frac{n_{m_L}}{n_m} H(m_L) + \frac{n_{m_R}}{n_m} H(m_R)
$$

This is the **expected entropy** after the split, where we weight each child's entropy by the probability (fraction) of examples that reach that child. Without this weighting, we would treat all splits equally regardless of how they distribute the data, favoring splits that create tiny pure nodes while leaving large impure ones.

The **information gain** of the split is then:

$$
\text{IG}(m) = H(m) - \left[ \frac{n_{m_L}}{n_m} H(m_L) + \frac{n_{m_R}}{n_m} H(m_R) \right]
$$

Without weighting by $n_{m_L}/n_m$ and $n_{m_R}/n_m$, we would favor splits that create very small pure nodes while leaving large impure nodes. Consider a split that isolates a single example (pure, $H=0$) while leaving $n-1$ examples with unchanged entropy. Without weighting, this would appear to have information gain $H(m)/2$, despite barely improving our situation. The weighting ensures we account for the fact that most examples still end up in the impure node.

Equivalently, the weighting computes the **expected** entropy after the split: if we randomly draw an example from node $m$, the expected entropy of the node it lands in is exactly the weighted sum.

### Impurity Reduction

The information gain formulation generalizes to any impurity measure. For a split $S$ that partitions node $m$ into children $m_L$ and $m_R$, the **impurity reduction** is:

$$
\Delta I(S, m) = I(m) - \left[ \frac{n_{m_L}}{n_m} I(m_L) + \frac{n_{m_R}}{n_m} I(m_R) \right]
$$

where $I(\cdot)$ is any impurity measure (entropy, Gini, or misclassification rate). The best split maximizes the impurity reduction:

$$
S^* = \arg\max_S \Delta I(S, m)
$$

For continuous features, we typically sort the values and consider all midpoints between adjacent values as candidate thresholds. If a feature has $n$ unique values, there are $n-1$ candidate thresholds. However, this can be computationally expensive for large datasets, so heuristics or random sampling of thresholds may be used in practice.

## Ensemble Methods

The fundamental insight behind ensemble methods is that combining multiple models can reduce variance and improve generalization. This is sometimes called the **"wisdom of crowds"** principle where aggregating diverse opinions often yields better estimates than any single expert. A classic example is Francis Galton's 1907 observation at a county fair: when 787 people guessed the weight of an ox, the median guess was remarkably accurate (within 1% of the true weight), even though most individual guesses were far off. The errors of individual guesses cancelled out when aggregated, leaving a highly accurate estimate.

To understand why ensembles work, we need to analyze the **bias-variance decomposition** of prediction error. For a regression problem with noisy target observations $y = f(\mathbf{x}) + \varepsilon$ where $\mathbb{E}[\varepsilon] = 0$ and $\text{Var}[\varepsilon] = \sigma^2$, the expected squared error of a predictor $\hat{f}(\mathbf{x})$ can be decomposed as:

$$
\mathbb{E}[(\hat{f}(\mathbf{x}) - y)^2] = \underbrace{(\mathbb{E}[\hat{f}(\mathbf{x})] - f(\mathbf{x}))^2}_{\text{Bias}^2} + \underbrace{\mathbb{E}[(\hat{f}(\mathbf{x}) - \mathbb{E}[\hat{f}(\mathbf{x})])^2]}_{\text{Variance}} + \underbrace{\sigma^2}_{\text{Irreducible Noise}}
$$

This is what leads to the bias-variance tradeoff:
- **Bias**: Error from approximating the true function with a simplified model. High bias means the model is too simple (underfitting).
- **Variance**: Sensitivity to the particular training set. High variance means the model changes significantly with different training data (overfitting).
- **Irreducible noise**: Fundamental randomness in the data that no model can capture.

So we need to balance bias and variance to minimize total error. For example deep decision trees have **low bias** as they can fit complex patterns but have **high variance** as they're sensitive to the training data. Ensemble methods aim to reduce variance while maintaining low bias.

{{< callout type="proof" >}}
Starting from the expected squared error for a fixed point $\mathbf{x}$:

$$
\begin{align*}
\mathbb{E}[(y - \hat{f}(\mathbf{x}))^2] &= \mathbb{E}[(f(\mathbf{x}) + \varepsilon - \hat{f}(\mathbf{x}))^2] \\
&= \mathbb{E}[(f(\mathbf{x}) - \hat{f}(\mathbf{x}))^2] + \mathbb{E}[\varepsilon^2] + 2\mathbb{E}[(f(\mathbf{x}) - \hat{f}(\mathbf{x}))\varepsilon] \\
&= \mathbb{E}[(f(\mathbf{x}) - \hat{f}(\mathbf{x}))^2] + \sigma^2
\end{align*}
$$

where we used $\mathbb{E}[\varepsilon] = 0$ and independence of $\varepsilon$ and $\hat{f}(\mathbf{x})$ to eliminate the cross-term. Now expand the first term by adding and subtracting $\mathbb{E}[\hat{f}(\mathbf{x})]$:

$$
\begin{align*}
\mathbb{E}[(f(\mathbf{x}) - \hat{f}(\mathbf{x}))^2] &= \mathbb{E}[(f(\mathbf{x}) - \mathbb{E}[\hat{f}(\mathbf{x})] + \mathbb{E}[\hat{f}(\mathbf{x})] - \hat{f}(\mathbf{x}))^2] \\
&= (f(\mathbf{x}) - \mathbb{E}[\hat{f}(\mathbf{x})])^2 + \mathbb{E}[(\hat{f}(\mathbf{x}) - \mathbb{E}[\hat{f}(\mathbf{x})])^2] \\
&\quad + 2(f(\mathbf{x}) - \mathbb{E}[\hat{f}(\mathbf{x})])\mathbb{E}[\hat{f}(\mathbf{x}) - \mathbb{E}[\hat{f}(\mathbf{x})]] \\
&= (f(\mathbf{x}) - \mathbb{E}[\hat{f}(\mathbf{x})])^2 + \text{Var}[\hat{f}(\mathbf{x})] \\
&= \text{Bias}^2[\hat{f}(\mathbf{x})] + \text{Var}[\hat{f}(\mathbf{x})]
\end{align*}
$$

The cross-term vanishes because $\mathbb{E}[\hat{f}(\mathbf{x}) - \mathbb{E}[\hat{f}(\mathbf{x})]] = 0$.
{{< /callout >}}

Now let us analyze how averaging multiple estimators affects bias and variance and therefore the overall error. Suppose we have $B$ estimators $\{\hat{f}_1(\mathbf{x}), \ldots, \hat{f}_B(\mathbf{x})\}$ and we average them:

$$
\hat{f}_{\text{avg}}(\mathbf{x}) = \frac{1}{B}\sum_{b=1}^B \hat{f}_b(\mathbf{x})
$$

How does this affect bias and variance? Taking expectations:

$$
\begin{align*}
\mathbb{E}[\hat{f}_{\text{avg}}(\mathbf{x})] &= \mathbb{E}\left[\frac{1}{B}\sum_{b=1}^B \hat{f}_b(\mathbf{x})\right] \\
&= \frac{1}{B}\mathbb{E}\left[\sum_{b=1}^B \hat{f}_b(\mathbf{x})\right] \quad \text{(expectation is linear)} \\
&= \frac{1}{B}\sum_{b=1}^B \mathbb{E}[\hat{f}_b(\mathbf{x})] \quad \text{(expectation of sum = sum of expectations)}
\end{align*}
$$

If all estimators have the same expected value $\mathbb{E}[\hat{f}_b(\mathbf{x})] = \mu(\mathbf{x})$ for all $b$, then:

$$
\mathbb{E}[\hat{f}_{\text{avg}}(\mathbf{x})] = \frac{1}{B}\sum_{b=1}^B \mu(\mathbf{x}) = \frac{1}{B} \cdot B\mu(\mathbf{x}) = \mu(\mathbf{x})
$$

The $1/B$ factor cancels out the sum of $B$ identical terms. Now recall that **bias** is defined as:

$$
\text{Bias}[\hat{f}(\mathbf{x})] = \mathbb{E}[\hat{f}(\mathbf{x})] - f(\mathbf{x})
$$

This measures the systematic error: the difference between the expected prediction and the true value. For the average estimator:

$$
\text{Bias}[\hat{f}_{\text{avg}}(\mathbf{x})] = \mathbb{E}[\hat{f}_{\text{avg}}(\mathbf{x})] - f(\mathbf{x}) = \mu(\mathbf{x}) - f(\mathbf{x})
$$

For an individual estimator:

$$
\text{Bias}[\hat{f}_b(\mathbf{x})] = \mathbb{E}[\hat{f}_b(\mathbf{x})] - f(\mathbf{x}) = \mu(\mathbf{x}) - f(\mathbf{x})
$$

Since both have the same expectation $\mu(\mathbf{x})$, they have the same bias! This means **averaging preserves bias**: if the individual estimators are unbiased (or all have the same bias), the average has exactly the same bias.

Now consider variance. For any random variables, the variance of their sum can be expressed using covariances:

$$
\text{Var}\left[\sum_{b=1}^B \hat{f}_b(\mathbf{x})\right] = \sum_{i=1}^B \sum_{j=1}^B \text{Cov}[\hat{f}_i(\mathbf{x}), \hat{f}_j(\mathbf{x})]
$$

This double sum includes all pairwise covariances. When $i = j$, we have $\text{Cov}[\hat{f}_i(\mathbf{x}), \hat{f}_i(\mathbf{x})] = \text{Var}[\hat{f}_i(\mathbf{x})]$ (there are $B$ such diagonal terms). When $i \neq j$, we have the covariance between different estimators (there are $B(B-1)$ such off-diagonal terms).

For the average, we have a factor of $1/B$:

$$
\text{Var}[\hat{f}_{\text{avg}}(\mathbf{x})] = \text{Var}\left[\frac{1}{B}\sum_{b=1}^B \hat{f}_b(\mathbf{x})\right] = \frac{1}{B^2} \text{Var}\left[\sum_{b=1}^B \hat{f}_b(\mathbf{x})\right] = \frac{1}{B^2}\sum_{i=1}^B \sum_{j=1}^B \text{Cov}[\hat{f}_i(\mathbf{x}), \hat{f}_j(\mathbf{x})]
$$

**Case 1: Independent estimators.** If the estimators are independent and identically distributed with variance $\sigma^2(\mathbf{x})$, then $\text{Cov}[\hat{f}_i(\mathbf{x}), \hat{f}_j(\mathbf{x})] = 0$ for $i \neq j$. Only the $B$ diagonal terms (variance terms) remain:

$$
\text{Var}[\hat{f}_{\text{avg}}(\mathbf{x})] = \frac{1}{B^2}\sum_{b=1}^B \text{Var}[\hat{f}_b(\mathbf{x})] = \frac{1}{B^2} \cdot B\sigma^2(\mathbf{x}) = \frac{\sigma^2(\mathbf{x})}{B}
$$

The $B^2$ in the denominator comes from squaring the $1/B$ factor, and we sum $B$ variance terms (each equal to $\sigma^2(\mathbf{x})$) in the numerator, giving $B\sigma^2(\mathbf{x})/B^2 = \sigma^2(\mathbf{x})/B$. So **averaging reduces variance by a factor of $B$** while preserving bias! This is why ensembles work: if we can train $B$ low-bias, high-variance models (like deep trees) and average them, we get a low-bias, low-variance ensemble.

**Case 2: Correlated estimators.** However, there's a crucial assumption: **independence**. In practice, models trained on the same data are not independent - they tend to make similar errors. If the estimators have equal variance $\sigma^2(\mathbf{x})$ and pairwise correlation $\rho(\mathbf{x})$, then $\text{Cov}[\hat{f}_i(\mathbf{x}), \hat{f}_j(\mathbf{x})] = \rho(\mathbf{x})\sigma^2(\mathbf{x})$ for $i \neq j$, and $\text{Cov}[\hat{f}_i(\mathbf{x}), \hat{f}_i(\mathbf{x})] = \sigma^2(\mathbf{x})$ for $i = j$. The variance becomes:

$$
\begin{align*}
\text{Var}[\hat{f}_{\text{avg}}(\mathbf{x})] &= \frac{1}{B^2}\sum_{i=1}^B \sum_{j=1}^B \text{Cov}[\hat{f}_i(\mathbf{x}), \hat{f}_j(\mathbf{x})] \\
&= \frac{1}{B^2}\left[\sum_{i=1}^B \sigma^2(\mathbf{x}) + \sum_{i=1}^B \sum_{j \neq i} \rho(\mathbf{x})\sigma^2(\mathbf{x})\right] \quad \text{(split diagonal and off-diagonal)} \\
&= \frac{1}{B^2}\left[B\sigma^2(\mathbf{x}) + B(B-1)\rho(\mathbf{x})\sigma^2(\mathbf{x})\right] \quad \text{($B$ diagonal, $B(B-1)$ off-diagonal)} \\
&= \frac{B\sigma^2(\mathbf{x})}{B^2} + \frac{B(B-1)\rho(\mathbf{x})\sigma^2(\mathbf{x})}{B^2} \\
&= \frac{\sigma^2(\mathbf{x})}{B} + \frac{B-1}{B}\rho(\mathbf{x})\sigma^2(\mathbf{x})
\end{align*}
$$

Let's examine this result more carefully. The variance has two terms:
- $\sigma^2(\mathbf{x})/B$: This term decreases to 0 as $B \to \infty$
- $[(B-1)/B]\rho(\mathbf{x})\sigma^2(\mathbf{x})$: This term depends on the correlation

As $B \to \infty$, note that:

$$
\frac{B-1}{B} = \frac{B}{B} - \frac{1}{B} = 1 - \frac{1}{B} \to 1
$$

Therefore:

$$
\begin{align*}
\lim_{B \to \infty} \text{Var}[\hat{f}_{\text{avg}}(\mathbf{x})] &= \lim_{B \to \infty} \left[\frac{\sigma^2(\mathbf{x})}{B} + \frac{B-1}{B}\rho(\mathbf{x})\sigma^2(\mathbf{x})\right] \\
&= 0 + 1 \cdot \rho(\mathbf{x})\sigma^2(\mathbf{x}) \\
&= \rho(\mathbf{x})\sigma^2(\mathbf{x})
\end{align*}
$$

The key insight is that **the variance reduction is limited by the correlation $\rho$ between models**. If models are perfectly correlated ($\rho = 1$), averaging provides no benefit. If models are independent ($\rho = 0$), variance decreases as $1/B$. **To maximize variance reduction, we need diverse (decorrelated) models.** 

## Random Forests

This is what leads us to **random forests** where we aim to create an ensemble of decorrelated decision trees. We do this through two main mechanisms:
1. **Bootstrap Aggregating (Bagging)**: We train each tree on a different bootstrap sample of the data.
2. **Random Feature Selection**: At each split, consider only a random subset of features.

### Bootstrap Sampling

A **bootstrap sample** of size $n$ is drawn by sampling $n$ times **with replacement** from the original dataset $\mathcal{D} = \{(\mathbf{x}_i, y_i)\}_{i=1}^n$. This means:
- Some examples appear multiple times in the bootstrap sample.
- Some examples don't appear at all (called **out-of-bag** or OOB examples).

The idea of bootstrap sampling is to create different training sets for each tree, introducing variability. Each tree sees a slightly different dataset, leading to different splits and predictions. Specifically if we use **with replacement** then each draw is independent and identically distributed. Because each tree also sees different data, they will look different and make different errors, reducing correlation between them, which is crucial for variance reduction!

The probability that a specific example is **not** selected in any of $n$ draws is:

$$
\left(1 - \frac{1}{n}\right)^n \approx e^{-1} \approx 0.368
$$

So about 36.8% of examples are out-of-bag for each tree (not picked), while about 63.2% are included (some multiple times). This means each tree is trained on a different subset of the data, further promoting diversity.

Since each tree is trained on only about 63.2% of the data, we can use the OOB examples for validation. The **OOB error estimate** is computed as follows:

1. For each training example $(\mathbf{x}_i, y_i)$, collect predictions only from trees where $\mathbf{x}_i$ was OOB.
2. Aggregate these predictions (majority vote for classification).
3. Compute the error rate across all training examples.

This OOB error is an unbiased estimate of the test error, so no separate validation set is needed! As the number of trees increases, the OOB error converges to the leave-one-out cross-validation error.

{{< figure
    src="/images/ml/bagging.png"
    alt="Illustration of bootstrap aggregating (bagging). Multiple bootstrap samples are drawn from the original dataset, a model is trained on each, and predictions are aggregated."
    caption="Illustration of bootstrap aggregating (bagging). Multiple bootstrap samples are drawn from the original dataset, a model is trained on each, and predictions are aggregated."
>}}

### Random Feature Selection

A further way to reduce correlation between trees is to introduce randomness in the feature selection at each split. Instead of considering all $d$ features when finding the best split at a node, we randomly select a subset of $m$ features ($m < d$) and find the best split only among those. Typical choices for $m$ are:
- Classification: $m = \sqrt{d}$
- Regression: $m = d/3$

We use this because even with bootstrap sampling, trees trained on similar data tend to select the same strong features at the top levels, making them correlated. By forcing each split to choose from a random subset, we ensure that even if one feature is globally optimal, it won't always be selected, allowing other features to contribute and increasing diversity.

{{< figure
    src="/images/ml/randomForestsFeatureSelection.jpeg"
    alt="Illustration of random feature selection in random forests. At each split, only a random subset of features is considered, increasing diversity between trees."
    caption="Illustration of random feature selection in random forests. At each split, only a random subset of features is considered, increasing diversity between trees."
>}}

### Feature Importance

Random forests provide a natural measure of feature importance based on how much each feature contributes to reducing impurity across all trees in the forest. For each tree $b$, we track the impurity decrease $\Delta I_m$ at each node $m$ where a feature $j$ is used for splitting. Let $n_m$ be the number of training examples that reach node $m$, and $n$ be the total number of training examples. The **impurity-based importance** of feature $j$ is:

$$
\text{Importance}(j) = \frac{1}{B}\sum_{b=1}^B \sum_{m \in T_b} \mathbb{1}[\text{split}_m \text{ uses feature } j] \cdot \Delta I_m \cdot \frac{n_m}{n}
$$

This averages the total impurity reduction from feature $j$ across all trees, weighted by the number of samples at each split. However, this measure has important limitations:

1. **Bias toward high-cardinality features**: Features with many unique values (like continuous variables or IDs) tend to have higher importance scores simply because they offer more potential split points, even if they have no true predictive power.

2. **Unreliable with correlated features**: When two features are correlated, the importance can be arbitrarily divided between them depending on which gets selected first in each tree. This makes the scores unstable and difficult to interpret.

3. **Computed on training data**: The impurity reduction is measured on the training set, which can overstate the importance of features that enable overfitting.

A more reliable alternative is **permutation importance**, which directly measures how much each feature contributes to prediction accuracy. The idea is simple: if a feature is important, randomly shuffling its values should break the relationship with the target and increase prediction error. For each feature $j$:

1. Compute the baseline OOB error on the unmodified test set
2. Randomly permute the values of feature $j$ across all OOB samples (this shuffles $x_j$ while keeping all other features and the labels fixed)
3. Recompute the OOB error with the permuted feature
4. The permutation importance is the increase in error: $\text{PermImportance}(j) = \text{Error}_{\text{permuted}} - \text{Error}_{\text{baseline}}$

For example, if predicting house prices and we permute the square footage feature, predictions become much worse because we have broken the relationship between size and price. However, permuting a random ID variable would not affect predictions because it was never informative.

Permutation importance addresses all three limitations of impurity-based importance: it is unbiased with respect to cardinality, handles correlated features more gracefully (both correlated features will show importance if both are predictive), and is computed on held-out OOB samples, providing a better estimate of true predictive importance.

## Boosting

While bagging reduces variance by averaging independent models, **boosting** takes a fundamentally different approach. It sequentially trains models where each new model focuses on correcting the errors of the ensemble so far. Where bagging primarily reduces variance by decorrelating models through bootstrap sampling, boosting reduces both bias and variance through a sequential adaptive process. The key insight is that we can build a strong classifier by combining many **weak learners**. A weak learner is a classifier whose error rate is only slightly better than random guessing (that is, error rate less than 0.5 for binary classification). Common choices include decision stumps (trees with only one split) and shallow trees (limited depth of 2-4 levels).

Each weak learner has high bias because it is deliberately simple and cannot fit complex patterns. However, by adaptively combining many weak learners, the ensemble builds up the capacity to fit complex decision boundaries, thereby reducing bias. Unlike bagging where we seek diversity through randomization, boosting creates diversity by having each learner specialize in correcting the mistakes of the ensemble so far.

We focus on **AdaBoost** (Adaptive Boosting), introduced by Freund and Schapire in 1995. AdaBoost is a deterministic algorithm that maintains a weight distribution over training examples. Initially all weights are equal, but after each iteration, misclassified examples receive higher weights. This forces subsequent learners to focus on the hard cases that previous learners struggled with. The sequential nature means that weak learners are highly correlated by design (each targets similar hard examples), but this correlation does not harm performance as it does in bagging because boosting operates in a different regime: it builds complexity to reduce bias rather than averaging to reduce variance. Although boosting is deterministic (given the same data, it produces the same model), the weighted voting scheme prevents any single weak learner from dominating the ensemble. As we add more weak learners, the ensemble becomes more stable because no single classifier can dramatically change the final prediction. This provides a form of variance reduction despite the high correlation between learners.

{{< figure
    src="/images/ml/randomForestsBoosting.png"
    alt="Comparison of bagging and boosting approaches. Bagging trains models independently in parallel, while boosting trains models sequentially with each focusing on correcting previous errors."
    caption="Comparison of bagging and boosting approaches. Bagging trains models independently in parallel, while boosting trains models sequentially with each focusing on correcting previous errors."
>}}

Consider binary classification with training data $\mathcal{D} = \{(\mathbf{x}_i, y_i)\}_{i=1}^n$ where labels $y_i \in \{-1, +1\}$. Let $c_b(\mathbf{x})$ denote the output of the $b$-th weak classifier, where $c_b(\mathbf{x}) \in \{-1, +1\}$.

AdaBoost builds a strong classifier as a weighted combination:

$$
\hat{C}(\mathbf{x}) = \text{sgn}\left(\sum_{b=1}^B \alpha_b c_b(\mathbf{x})\right)
$$

where $\alpha_b$ is the weight assigned to classifier $b$ based on its performance. The algorithm starts by initializing all weights of the training examples equally: $w_i^{(1)} = 1/n$ for all $i = 1, \ldots, n$. We then sequentially train $B$ weak classifiers, updating the weights after each iteration:

1. **Train** a weak classifier $c_b(\mathbf{x})$ on the training data using weights $w_i^{(b)}$.

   The weights $w_i^{(b)}$ modify the training objective to focus on harder examples. For decision trees, the weights affect the impurity calculations at each split. Recall that for an unweighted tree, we choose splits to maximize the impurity reduction. With weights, we modify the impurity measures to use weighted counts. For example, the weighted Gini impurity at node $m$ becomes:

   $$
   G_{\text{weighted}}(m) = 1 - \sum_{k=1}^K \left(\frac{\sum_{i \in m} w_i \mathbb{1}[y_i = k]}{\sum_{i \in m} w_i}\right)^2
   $$

   where the class proportions are now weighted. Examples with larger weights contribute more to determining the best split. This means the tree will prioritize reducing errors on high-weight examples (those misclassified by previous classifiers) over low-weight examples (those already correctly classified). In practice, many implementations sample with replacement from the training set according to the weights, creating a dataset where difficult examples appear more frequently.

2. **Compute weighted error**:
   $$
   \epsilon_b = \frac{\sum_{i=1}^n w_i^{(b)} \mathbb{1}[c_b(\mathbf{x}_i) \neq y_i]}{\sum_{i=1}^n w_i^{(b)}}
   $$

    where $\mathbb{1}[\cdot]$ is the indicator function that equals 1 if the condition is true and 0 otherwise. So if classifier $b$ misclassifies example $i$, we add its weight $w_i^{(b)}$ to the numerator. The denominator normalizes by the total weight of all examples. This is the probability of misclassification under the weighted data distribution. The normalization by the total weight ensures that $\epsilon_b$ is a valid probability between 0 and 1.

3. **Compute classifier weight**:
   $$
   \alpha_b = \frac{1}{2}\ln\frac{1 - \epsilon_b}{\epsilon_b}
   $$

   This determines how much we trust classifier $b$ in the final vote. If $\epsilon_b < 0.5$ (better than random), then $\alpha_b > 0$. As $\epsilon_b \to 0$ (perfect classifier), we have $\alpha_b \to \infty$ because the fraction $(1 - \epsilon_b)/\epsilon_b$ becomes very large and thus its logarithm grows without bound. As $\epsilon_b \to 0.5$ (random classifier), we have $\alpha_b \to 0$ because the fraction approaches 1 and $\ln(1) = 0$. If $\epsilon_b > 0.5$ (worse than random), then $\alpha_b < 0$, meaning we would actually subtract this classifier's vote in the final decision.

4. **Update weights**: Lastly, we update the weights for the next iteration:
   $$
   w_i^{(b+1)} = w_i^{(b)} \exp(\alpha_b \mathbb{1}[c_b(\mathbf{x}_i) \neq y_i])
   $$

   Equivalently, we can write this as $w_i^{(b+1)} = w_i^{(b)} \exp(-\alpha_b y_i c_b(\mathbf{x}_i))$ using the fact that $y_i c_b(\mathbf{x}_i) = 1$ when correct and $y_i c_b(\mathbf{x}_i) = -1$ when incorrect.

**Return** the final classifier $\hat{C}(\mathbf{x}) = \text{sgn}\left(\sum_{b=1}^B \alpha_b c_b(\mathbf{x})\right)$. So the more confident classifiers (with larger $\alpha_b$) have more influence on the final decision.

The weight update has an elegant interpretation. When classifier $c_b$ correctly classifies example $i$, the weight is multiplied by $e^{-\alpha_b} < 1$ (decreased). When it misclassifies example $i$, the weight is multiplied by $e^{\alpha_b} > 1$ (increased). The magnitude of the change depends on how confident we are in classifier $b$ through $\alpha_b$. This reweighting scheme ensures that the next classifier focuses more attention on the examples that previous classifiers found difficult.

### Forward Stagewise Additive Modeling

Where do these formulas for $\alpha_b$ and the weight updates come from? We now show that AdaBoost is performing **forward stagewise additive modeling** to minimize the **exponential loss**. Let's define the cumulative sum of weighted classifiers up to iteration $b$:

$$
F_b(\mathbf{x}) = \sum_{t=1}^b \alpha_t c_t(\mathbf{x})
$$

We can think of $F_b(\mathbf{x})$ as the current ensemble's confidence score for classifying $\mathbf{x}$ as class $+1$ with the final prediction is $\text{sgn}(F_B(\mathbf{x}))$. If the sum is very positive, we are confident in class $+1$; if very negative, confident in class $-1$.

Consider the **exponential loss function**:

$$
L(y, F(\mathbf{x})) = \exp(-y F(\mathbf{x}))
$$

For binary classification with $y \in \{-1, +1\}$. When $y F(\mathbf{x}) > 0$ (correct sign), the loss is less than 1. When $y F(\mathbf{x}) < 0$ (wrong sign), the loss exceeds 1 and grows exponentially with the magnitude of the error. The loss is differentiable everywhere, making it suitable for optimization. For a training set $\{(\mathbf{x}_i, y_i)\}_{i=1}^n$, the total loss over the training set is:

$$
\mathcal{L}(F) = \sum_{i=1}^n \exp(-y_i F(\mathbf{x}_i))
$$

In **forward stagewise modeling**, we build $F$ one term at a time. At iteration $b$, given the previous model $F_{b-1}$, we choose the next classifier $c_b$ and weight $\alpha_b$ to make the next cumulative model $F_b = F_{b-1} + \alpha_b c_b$ that minimizes the total loss:

$$
\mathcal{L}(F_{b-1} + \alpha_b c_b) = \sum_{i=1}^n \exp(-y_i(F_{b-1}(\mathbf{x}_i) + \alpha_b c_b(\mathbf{x}_i)))
$$

We can rewrite this by factoring out $\exp(-y_i F_{b-1}(\mathbf{x}_i))$:

$$
\begin{align*}
\mathcal{L}(F_{b-1} + \alpha_b c_b) &= \sum_{i=1}^n \exp(-y_i F_{b-1}(\mathbf{x}_i)) \exp(-y_i \alpha_b c_b(\mathbf{x}_i)) \\
&= \sum_{i=1}^n w_i^{(b)} \exp(-y_i \alpha_b c_b(\mathbf{x}_i))
\end{align*}
$$

where $w_i^{(b)} = \exp(-y_i F_{b-1}(\mathbf{x}_i))$ are exactly the weights for the next iteration from the AdaBoost algorithm (up to normalization). These weights represent how much exponential loss each example contributes based on all previous classifiers.

Since $y_i, c_b(\mathbf{x}_i) \in \{-1, +1\}$, the product $y_i c_b(\mathbf{x}_i)$ equals $+1$ when the classifier is correct and $-1$ when incorrect. We can split the sum:

$$
\mathcal{L}(F_{b-1} + \alpha_b c_b) = \sum_{i: y_i = c_b(\mathbf{x}_i)} w_i^{(b)} e^{-\alpha_b} + \sum_{i: y_i \neq c_b(\mathbf{x}_i)} w_i^{(b)} e^{\alpha_b}
$$

Let $W = \sum_{i=1}^n w_i^{(b)}$ be the total weight. we then define the weighted error rate of classifier $c_b$ as:

$$
\epsilon_b = \frac{1}{W}\sum_{i: y_i \neq c_b(\mathbf{x}_i)} w_i^{(b)}
$$

Then $1 - \epsilon_b = \frac{1}{W}\sum_{i: y_i = c_b(\mathbf{x}_i)} w_i^{(b)}$ is the weighted accuracy. The loss becomes:

$$
\mathcal{L}(F_{b-1} + \alpha_b c_b) = W[(1-\epsilon_b)e^{-\alpha_b} + \epsilon_b e^{\alpha_b}]
$$

**Optimizing over $c_b$**: For fixed $\alpha_b > 0$, we have $e^{\alpha_b} > e^{-\alpha_b}$, so the loss is minimized by choosing $c_b$ to minimize the weighted error $\epsilon_b$. This is precisely what AdaBoost does: train the weak learner to minimize the weighted misclassification rate.

**Optimizing over $\alpha_b$**: For a fixed classifier $c_b$, we minimize the loss by taking the derivative with respect to $\alpha_b$ and setting it to zero:

$$
\begin{align*}
\frac{d}{d\alpha_b}[(1-\epsilon_b)e^{-\alpha_b} + \epsilon_b e^{\alpha_b}] &= -(1-\epsilon_b)e^{-\alpha_b} + \epsilon_b e^{\alpha_b} = 0 \\
\epsilon_b e^{\alpha_b} &= (1-\epsilon_b)e^{-\alpha_b} \\
e^{2\alpha_b} &= \frac{1-\epsilon_b}{\epsilon_b} \\
\alpha_b &= \frac{1}{2}\ln\frac{1-\epsilon_b}{\epsilon_b}
\end{align*}
$$

This is exactly the formula AdaBoost uses for $\alpha_b$. Thus, the AdaBoost reweighting algorithm is equivalent to forward stagewise additive modeling with exponential loss.

### Gradient Boosting

We can also view AdaBoost through the lens of optimization in function space. The exponential loss is:

$$
\mathcal{L}(F) = \sum_{i=1}^n \exp(-y_i F(\mathbf{x}_i))
$$

The gradient of this loss with respect to $F(\mathbf{x}_i)$ is:

$$
-\frac{\partial \mathcal{L}}{\partial F(\mathbf{x}_i)} = y_i \exp(-y_i F(\mathbf{x}_i)) = y_i w_i^{(b)}
$$

This is the negative gradient (direction of steepest descent) for reducing the loss at each training point $\mathbf{x}_i$. In gradient descent, we would update $F(\mathbf{x}_i) \gets F(\mathbf{x}_i) + \eta \cdot y_i w_i^{(b)}$ for some step size $\eta$.

However, we cannot directly follow this gradient because we are constrained to add a function from our weak learner class. Instead, we choose the weak learner $c_b(\mathbf{x})$ that best approximates the negative gradient in a weighted least squares sense:

$$
c_b \approx \arg\min_{c} \sum_{i=1}^n w_i^{(b)} (y_i - c(\mathbf{x}_i))^2
$$

The use of a second-order Taylor approximation in the derivation makes this a **Newton-like update** rather than pure gradient descent. The key insight is that AdaBoost can be viewed as taking Newton steps in function space, where the "function parameter" being optimized is $F$ itself. This perspective generalizes to **Gradient Boosting**, which works with any differentiable loss function by fitting weak learners to approximate the negative gradient of the loss.

### Training Error Bound

A remarkable property of AdaBoost is that the training error decreases exponentially fast. The training error is the fraction of misclassified training examples:

$$
\text{Training Error} = \frac{1}{n}\sum_{i=1}^n \mathbb{1}[y_i \neq \text{sgn}(F_B(\mathbf{x}_i))]
$$

Note that $\mathbb{1}[y_i \neq \text{sgn}(F_B(\mathbf{x}_i))] = \mathbb{1}[y_i F_B(\mathbf{x}_i) \leq 0]$ since the classification is wrong when the sign disagrees. We can bound this indicator by the exponential loss:

$$
\mathbb{1}[y_i F_B(\mathbf{x}_i) \leq 0] \leq \exp(-y_i F_B(\mathbf{x}_i))
$$

This holds because the exponential function is always positive, and when $y_i F_B(\mathbf{x}_i) \leq 0$, we have $\exp(-y_i F_B(\mathbf{x}_i)) \geq \exp(0) = 1$, which upper bounds the indicator that equals 0 or 1.

Therefore:

$$
\begin{align*}
\text{Training Error} &= \frac{1}{n}\sum_{i=1}^n \mathbb{1}[y_i F_B(\mathbf{x}_i) \leq 0] \\
&\leq \frac{1}{n}\sum_{i=1}^n \exp(-y_i F_B(\mathbf{x}_i))
\end{align*}
$$

Now recall that the exponential loss after $B$ iterations is:

$$
\mathcal{L}(F_B) = \sum_{i=1}^n \exp(-y_i F_B(\mathbf{x}_i))
$$

We showed earlier that at each iteration $b$, the loss is:

$$
\mathcal{L}(F_b) = W_b[(1-\epsilon_b)e^{-\alpha_b} + \epsilon_b e^{\alpha_b}]
$$

where $W_b = \sum_i w_i^{(b)}$ is the total weight at iteration $b$. Note that $W_1 = n$ (initial uniform weights sum to $n$), and:

$$
W_{b+1} = \sum_i w_i^{(b+1)} = \sum_i w_i^{(b)} \exp(\alpha_b \mathbb{1}[c_b(\mathbf{x}_i) \neq y_i]) = W_b[(1-\epsilon_b)e^{-\alpha_b} + \epsilon_b e^{\alpha_b}]
$$

Therefore, by telescoping:

$$
\mathcal{L}(F_B) = W_{B+1} = n \prod_{b=1}^B [(1-\epsilon_b)e^{-\alpha_b} + \epsilon_b e^{\alpha_b}]
$$

Substituting $\alpha_b = \frac{1}{2}\ln\frac{1-\epsilon_b}{\epsilon_b}$, we get $e^{-\alpha_b} = \sqrt{\frac{\epsilon_b}{1-\epsilon_b}}$ and $e^{\alpha_b} = \sqrt{\frac{1-\epsilon_b}{\epsilon_b}}$. Thus:

$$
\begin{align*}
(1-\epsilon_b)e^{-\alpha_b} + \epsilon_b e^{\alpha_b} &= (1-\epsilon_b)\sqrt{\frac{\epsilon_b}{1-\epsilon_b}} + \epsilon_b\sqrt{\frac{1-\epsilon_b}{\epsilon_b}} \\
&= \sqrt{\epsilon_b(1-\epsilon_b)} + \sqrt{\epsilon_b(1-\epsilon_b)} \\
&= 2\sqrt{\epsilon_b(1-\epsilon_b)}
\end{align*}
$$

Therefore:

$$
\text{Training Error} \leq \frac{1}{n}\mathcal{L}(F_B) = \prod_{b=1}^B 2\sqrt{\epsilon_b(1-\epsilon_b)}
$$

This shows how the training error is bounded by the product of terms involving the weighted errors of each weak learner.

Now suppose each weak learner achieves weighted error $\epsilon_b \leq \frac{1}{2} - \gamma$ for some $\gamma > 0$ (that is, each is at least $\gamma$ better than random guessing which we assume by using weak learners). Then:

$$
\begin{align*}
\epsilon_b(1-\epsilon_b) &\leq \left(\frac{1}{2} - \gamma\right)\left(\frac{1}{2} + \gamma\right) \\
&= \frac{1}{4} - \gamma^2
\end{align*}
$$

Therefore:

$$
2\sqrt{\epsilon_b(1-\epsilon_b)} \leq 2\sqrt{\frac{1}{4} - \gamma^2} = \sqrt{1 - 4\gamma^2}
$$

Using the Taylor expansion $\sqrt{1-x} \approx e^{-x/2}$ for small $x$ (or more precisely, $\sqrt{1-x} \leq e^{-x/2}$ for $x \in [0, 1]$), we have:

$$
\sqrt{1 - 4\gamma^2} \leq e^{-2\gamma^2}
$$

Thus:

$$
\text{Training Error} \leq \prod_{b=1}^B e^{-2\gamma^2} = e^{-2\gamma^2 B}
$$

**The training error decreases exponentially with the number of weak learners**. If each weak learner is even slightly better than random (that is, $\gamma > 0$), adding more weak learners drives the training error toward zero at an exponential rate. However, we must be cautious as low training error does not automatically guarantee low test error. Interestingly, boosting can sometimes continue to improve test error even after training error reaches zero. This seemingly paradoxical behavior occurs because boosting increases the **margin** of correctly classified examples. Even after all training examples are classified correctly (that is, $y_i F(\mathbf{x}_i) > 0$ for all $i$), continuing to boost can increase $|F(\mathbf{x}_i)|$ for correctly classified examples, making the classifier more confident and improving generalization. The exponential loss encourages not just correct classification but large margins, which provides implicit regularization. However, with too many iterations or in the presence of label noise, boosting will eventually overfit. The exponential loss heavily penalizes misclassified examples, which can cause the algorithm to overfit to noisy labels when boosting for too long.

In practice, boosting with decision trees (such as gradient boosted decision trees or XGBoost) often outperforms random forests on tabular data because it can achieve lower bias while maintaining reasonable variance, particularly when using shallow trees and appropriate regularization.

## Regression Trees

While we've focused on classification, decision trees also handle regression by predicting continuous values. At each leaf, instead of predicting the majority class, we predict the **mean** of the target values:

$$
\hat{y}_m = \frac{1}{n_m}\sum_{(\mathbf{x}, y) \in \mathcal{D}_m} y
$$

For splitting, we minimize **variance** (or equivalently, mean squared error) instead of entropy or Gini:

$$
I(m) = \frac{1}{n_m}\sum_{(\mathbf{x}, y) \in \mathcal{D}_m} (y - \hat{y}_m)^2
$$

The impurity reduction becomes the reduction in variance:

$$
\Delta I(S, m) = \text{Var}(m) - \left[\frac{n_{m_L}}{n_m}\text{Var}(m_L) + \frac{n_{m_R}}{n_m}\text{Var}(m_R)\right]
$$

All ensemble methods (bagging, boosting) apply to regression trees by averaging predictions instead of voting.

{{< figure
    src="/images/ml/regressionTree.gif"
    alt="Example of a regression tree predicting continuous values."
    caption="Example of a regression tree predicting continuous values."
>}}
