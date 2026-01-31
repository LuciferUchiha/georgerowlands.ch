---
title: Counterfactual Invariance
type: docs
weight: 25
---

When we train machine learning models using standard approaches, we typically minimize prediction error on a training dataset. The model learns to exploit any pattern in the data that helps reduce this error, regardless of whether that pattern represents a genuine causal relationship or merely a spurious correlation. However, not all patterns that work well in training are actually the right reasons for making predictions. This leads to what researchers call **shortcut learning**, where the model takes shortcuts by relying on features that happen to be correlated with the target in the training data but do not represent the true underlying causal mechanism. This can cause the model to fail dramatically when deployed in new environments or under different conditions. 

{{< callout type="example" >}}
A classic example of shortcut learning is in image classification. A model trained to distinguish between cows and camels might learn to rely on the background environment (grasslands versus deserts) rather than the actual features of the animals themselves. While this shortcut works well on the training data, it fails when the model encounters a cow in a desert setting or a camel in a grassy field such as a zoo. The model has learned the wrong features, leading to poor generalization.

{{< figure
  src="/images/maths/shortcutLearningCowCamel.png"
  alt="Diagram showing cow images with grass backgrounds and camel images with desert backgrounds, illustrating how a model might learn the wrong features"
  caption="Shortcut learning: A model trained on this biased dataset may learn to classify based on background environment rather than animal features"
>}}
{{< /callout >}}

The concept of **counterfactual invariance** provides a principled framework for thinking about which features a model should rely on. A model that is counterfactually invariant makes predictions that do not change when we hypothetically intervene on variables that should not causally affect the outcome. This means the model has learned to focus on the right features and ignore spurious correlations that might exist in the training data.

### Domain Shift and Distribution Change

The root cause of these failures is what we call a **domain shift**. A domain shift occurs when the test samples come from a different distribution than the training samples. In practice, domain shifts are extremely common. Medical data collected at different hospitals will have different characteristics. Images captured with different cameras or under different lighting conditions will have different statistical properties. Text written by different populations or at different time periods will have different patterns.

Standard machine learning assumes that training and test data are drawn from the same distribution. When this assumption is violated through a domain shift, models that relied on shortcuts will fail because the spurious correlations they exploited during training no longer hold in the new domain. The correlations that were reliable in the training distribution may completely disappear or even reverse in the test distribution.

## Shortcut Learning

Shortcut learning occurs when there is a spurious correlation between causal features and non-causal features in the training dataset, and the resulting model exploits these non-causal features to make predictions. Here, **causal features** are those that have a genuine causal relationship with the target variable, meaning that changing these features would actually change the outcome. In contrast, **non-causal features** may be correlated with the target in the training data but do not have a true causal effect on it. The model essentially finds an easy pattern that works in training but does not generalize because it is not based on the true causal mechanism.

In the example of classifying images of cows versus camels, the background environment provides a strong signal for classification in this dataset. A model might learn to classify images primarily based on whether the background looks like grass or sand, with the actual animal features playing a secondary role.

{{< figure
  src="/images/maths/shortcutLearningCowCamel.png"
  alt="Diagram showing cow images with grass backgrounds and camel images with desert backgrounds, illustrating how a model might learn the wrong features"
  caption="Shortcut learning: A model trained on this biased dataset may learn to classify based on background environment rather than animal features"
>}}

This model will achieve excellent performance on the training data and even on test data that follows the same pattern. However, it has learned the wrong representation. The soil color and background vegetation are not causally related to whether an animal is a cow or a camel. When the model encounters a cow photographed in a desert setting, it will likely misclassify it as a camel because it has learned to rely on the background as a shortcut rather than learning the actual visual features that distinguish cows from camels.

The key insight is that shortcut learning happens when the model finds features that are predictive in the training distribution but whose association with the label changes under distribution shift. An **invariant representation**, by contrast, would focus on features whose relationship with the target remains stable across different environments. The shape, facial features, and body structure of animals remain consistent indicators of species regardless of the background environment. A model using these invariant features would generalize much better to new settings.

## Counterfactual Invariance

To formalize when a model has learned the right features, we introduce the concept of counterfactual invariance. This concept requires us to think about what would happen under hypothetical interventions on variables that should not affect our predictions.

Let $X$ be a random variable representing the features of an object or observation. Let $Y$ be a random variable representing the target that we want to predict from $X$. Now consider a random variable $W$ representing an environmental or contextual factor that influences $X$ but should not influence our predictions about $Y$. For example, $W$ might represent the background environment in which an image was captured, while $Y$ represents the true label we care about.

We can think about counterfactuals by imagining what the features $X$ would have looked like if we had forced the value of $W$ to be different. We denote this counterfactual random variable as $X(w)$, which represents the features we would have observed if we had intervened to set $W$ equal to some specific value $w$. This is the value we would obtain by replacing the actual value of $W$ associated with $X$ with the counterfactual value $w$. Now we can define counterfactual invariance for a prediction function $f$.

We say that a function $f: \mathcal{X} \to \mathcal{Y}$ is **counterfactually invariant** with respect to $W$ if for any two values $w$ and $w'$ in the range of $W$, we have:

$$
f(X(w)) = f(X(w'))
$$

In other words, the prediction does not change when we hypothetically intervene on $W$ and set it to different values. The function $f$ makes the same prediction, which means it has not learned to rely on spurious features that vary with $W$. Counterfactual invariance captures the intuitive idea that our model should not be influenced by factors that do not causally affect the outcome. A model that satisfies this property has learned a representation that depends only on the causally relevant features.

{{< callout type="example" >}}
Going back to our animal classification example, if $W$ represents the background environment (grassland or desert) and $X$ represents the image, then a counterfactually invariant classifier would make the same prediction for $X(\text{grassland})$ and $X(\text{desert})$. The model would correctly identify a cow as a cow regardless of whether we imagine it in a grassy field or a desert setting. This is exactly the behavior we want because the background should not affect the species classification.

{{< figure
  src="/images/maths/counterfactualInvarianceDiagram.jpg"
  alt="Diagram showing the same cow in two different backgrounds with the model making the same prediction"
  caption="Counterfactual invariance: The prediction f(X) should remain the same under interventions on the environment W"
>}}
{{< /callout >}}

## Causal and Anti-Causal Scenarios

To understand when and how we can achieve counterfactual invariance, we need to distinguish between different causal relationships between our features $X$ and target $Y$. There are two main scenarios to consider, based on the direction of causation.

### The Causal Scenario

In the **causal scenario**, the features in $X$ have a causal influence on the target $Y$. More precisely, changing or intervening on the features in $X$ would cause a change in $Y$. This is the case when we are predicting an outcome that is caused by the observed features.

In a causal scenario, we typically have an environmental variable $W$ that influences some of the features in $X$ but should not directly influence our predictions. To understand the structure, we decompose the features $X$ into three disjoint subsets based on their causal relationships:

- $X_W^\perp$: Features that are independent of the environmental variable $W$. The superscript $\perp$ denotes independence from $W$. These features are not influenced by the environment.
- $X_Y^\perp$: Features that are independent of the target $Y$. These features do not causally affect the outcome, even though they may be present in our feature vector.
- $X_{W \& Y}$: Features that depend on both $W$ and $Y$. These features are influenced by the environment and also have a causal effect on the outcome.

The formal structure of the causal scenario is represented by a directed acyclic graph where all arrows flow from left to right, starting from $W$ and ending at $Y$:

$$
W \to X_Y^\perp, \quad W \to X_{W \& Y}, \quad X_W^\perp \to Y, \quad X_{W \& Y} \to Y
$$

The key insight is that in this structure, information flows from the environmental variable $W$ through some features to the target $Y$. The features $X_Y^\perp$ that are influenced by $W$ do not causally affect $Y$, while the features $X_{W \& Y}$ that are influenced by $W$ do causally affect $Y$. Additionally, some features $X_W^\perp$ are independent of the environment but still causally affect the outcome.

{{< figure
  src="/images/maths/causalScenarioDiagram.png"
  alt="Causal graph showing W influencing some features, and features causing Y"
  caption="Causal scenario: Features X cause the target Y. Arrows flow from W to X to Y."
>}}

{{< callout type="example" >}}
Consider predicting whether a person has cancer based on medical features. The target $Y$ is whether the person develops cancer. Our features $X$ might include measurements like CO2 levels in the lungs and whether the person smokes. The environmental variable $W$ represents the city where the person lives.

In this example, smoking and CO2 levels have a direct causal effect on cancer development. These would be part of $X_W^\perp$ or $X_{W \& Y}$ depending on whether the city influences them. The city $W$ itself should not directly cause cancer, though it might correlate with cancer rates in the training data because different cities have different pollution levels or demographic characteristics.

We want our model to be invariant to the city variable $W$ because city itself does not cause cancer, even though it may be correlated with true causal factors. A counterfactually invariant model would make the same cancer prediction regardless of which city we imagine the person living in.
{{< /callout >}}

### The Anti-Causal Scenario

In the **anti-causal scenario**, the target $Y$ has a causal influence on the features in $X$. This might seem backwards at first, but it is actually common in many prediction tasks. We are predicting a cause from its effects.

In an anti-causal scenario, the environmental variable $W$ influences some of the observed features in $X$, but crucially, $W$ does not directly cause the target $Y$. The causal structure is fundamentally different from the causal scenario because the arrows now flow from $Y$ to $X$ rather than from $X$ to $Y$.

The formal structure of the anti-causal scenario is represented by a directed acyclic graph where arrows flow from the target $Y$ to the features $X$, and separately from the environmental variable $W$ to some features in $X$:

$$
Y \to X, \quad W \to X
$$

Importantly, there is no direct causal path from $W$ to $Y$. The environmental variable only influences how the features manifest or are observed, not the underlying target variable itself. The features $X$ are influenced by both the true target $Y$ (which causes them) and potentially by the environmental context $W$ (which may affect how they are measured or reported).

{{< figure
  src="/images/maths/antiCausalScenarioDiagram.png"
  alt="Causal graph showing Y causing features X, with W also influencing some features"
  caption="Anti-causal scenario: Target Y causes features X. Environmental variable W influences feature observation but not Y."
>}}

{{< callout type="example" >}}
Consider diagnosing a disease from symptoms. The disease $Y$ causes the symptoms $X$ that we observe. For instance, having celiac disease causes stomachaches, fatigue, and various other symptoms. In this case, the causal arrow points from $Y$ to $X$, which is why we call it anti-causal prediction.

The environmental variable $W$ might represent the person's job type, which influences which symptoms we observe or how they are reported, but does not directly cause the disease. Someone with a physically demanding job might report fatigue differently than someone with a desk job, even if they have the same underlying disease. We want our diagnostic model to be invariant to job type $W$ because the job does not cause the disease.
{{< /callout >}}

The distinction between causal and anti-causal scenarios is important because it affects how we should think about achieving invariance. In both cases, we want our predictions to be invariant to environmental factors $W$, but the structure of the problem is different, which leads to different conditions for achieving this goal.

## Confounding and Selection Bias

Before we can understand the precise conditions for counterfactual invariance, we need to understand two major ways that spurious correlations can arise in our data. These mechanisms can make features appear to be predictive when they are not actually causal. These concepts are closely related to what we learned about [conditional independence](/maths/probabilityStatistics/conditionalProbabilityIndependence).

### Confounding

**Confounding** occurs when a hidden variable $U$ affects both the environmental variable $W$ and our features or target. This creates a spurious correlation between $W$ and the features or target, even though there is no direct causal relationship between them.

In causal graphs, confounding is represented by a hidden variable $U$ that has directed edges to multiple observed variables. This creates what is called a fork structure: $W \leftarrow U \to X$ (or $W \leftarrow U \to Y$). Even though there is no direct causal path from $W$ to $X$, they become statistically dependent because they share a common cause $U$. This spurious dependence can mislead a model into thinking that $W$ is predictive of the outcome, when in reality both are simply driven by the hidden confounder.

{{< figure
  src="/images/maths/confoundingExample.png"
  alt="Causal graph showing a confounder affecting both observed variables"
  caption="Confounding: A hidden variable U causes both W and X, creating a spurious correlation"
>}}

{{< callout type="example" title="Ice Cream and Shark Attacks" >}}
A classic example is the spurious correlation between ice cream sales and shark attacks. Both of these variables tend to be higher in the summer, not because ice cream causes shark attacks or vice versa, but because a confounding variable (hot weather) causes both. The weather influences people to buy more ice cream and also to go swimming more often, which increases the chance of shark encounters. If we did not account for the confounding variable of weather, we might incorrectly conclude that ice cream consumption is predictive of shark attacks.
{{< /callout >}}

### Selection Bias

**Selection bias** occurs when a hidden variable $S$ filters our training dataset based on multiple observed variables. We only observe data points where $S$ takes a particular value (typically $S = 1$ indicating selection into the dataset). This can create spurious correlations in the observed data that do not exist in the full population.

Mathematically, selection creates a collider structure in the causal graph. Both variables (such as $X$ and $W$) influence whether a data point is selected, with arrows pointing into the selection variable: $X \to S \leftarrow W$. By conditioning on $S = 1$ (observing only selected data), we create a spurious dependency between $X$ and $W$ even if they were independent in the full population. This is the opposite of how confounding works: while confounders create dependencies by being common causes, colliders create dependencies when we condition on them.

{{< figure
  src="/images/maths/selectionBiasDiagram.png"
  alt="Causal graph showing how selection on a collider creates spurious dependencies"
  caption="Selection bias: Conditioning on selection (S) which depends on both variables creates spurious correlations"
>}}

{{< callout type="example" title="LinkedIn Selection Bias" >}}
Suppose someone analyzes their LinkedIn contacts and finds a negative correlation between attractiveness and professional success. Should they conclude that being attractive is bad for your career? No, this is selection bias.

In the general population, attractiveness and professional success might be independent. However, people become LinkedIn contacts either because they are professionally successful or because they are attractive (socially connected). If we imagine the population divided equally into four groups (attractive and successful, attractive and unsuccessful, unattractive and successful, unattractive and unsuccessful), the selection process excludes only those who are neither attractive nor successful.

Among the selected LinkedIn contacts, unattractive people have a 100% success rate (since they had to be successful to be selected), while attractive people have only a 50% success rate (since they could be selected for either reason). This creates an apparent negative correlation in the selected sample, even though the variables were independent in the full population. The selection process created a spurious correlation by conditioning on a collider.
{{< /callout >}}

Selection bias is particularly insidious because our training datasets are almost always selected in some way. Medical datasets come from people who sought treatment. Internet datasets come from people who chose to post or interact online. Any time our data collection process depends on the variables we are studying, we risk introducing selection bias.

## Causal Graphs and d-Separation

To reason formally about these causal relationships and independencies, we use **causal graphs** (directed acyclic graphs). In a causal graph, each variable is represented as a node, and a directed edge from variable $A$ to variable $B$ indicates that $A$ has a direct causal influence on $B$.

Causal graphs let us determine which variables are independent of each other, and which independencies hold conditionally on observing certain other variables. The key tool for reading independencies from a graph is called **d-separation** (directional separation).

The concept of d-separation provides a graphical criterion for determining whether two sets of variables are conditionally independent given a third set. To check whether variables $A$ and $B$ are independent given observed variables $C$, we examine all paths between $A$ and $B$. When traversing these paths, we ignore the direction of the arrows and simply follow the edges in any direction. If every such path is blocked (d-separated), then $A$ and $B$ are conditionally independent given $C$, written as $A \perp B \mid C$. For unconditional independence (when $C$ is empty), we check if all paths are blocked without conditioning on any variables.

Understanding d-separation requires understanding the three basic junction types that can appear along a path. Every path in a complex graph is just a sequence of these three structures.

### The Three Basic Junctions

A path is blocked or open based on whether it contains certain junction patterns and whether the middle nodes in those junctions are observed (in the conditioning set $Z$).

**1. Chain (Mediation)**

Structure: $A \to B \to C$, where $A$ causes $B$, which causes $C$.

- If $B$ is not observed (not in $Z$): The path is **open**. Information flows from $A$ to $C$ through $B$. Learning about $A$ updates our belief about $B$, which updates our belief about $C$.
- If $B$ is observed (in $Z$): The path is **blocked**. Once we know the value of $B$, learning about $A$ tells us nothing new about $C$, because $A$ only affects $C$ through $B$, which we already know.

**2. Fork (Common Cause)**

Structure: $A \leftarrow B \to C$, where $B$ causes both $A$ and $C$.

- If $B$ is not observed (not in $Z$): The path is **open**. This is classic confounding. If $B$ changes, both $A$ and $C$ change simultaneously, making them correlated. Knowing $A$ gives a hint about $B$, which gives a hint about $C$.
- If $B$ is observed (in $Z$): The path is **blocked**. We have conditioned on the common cause. Once we know $B$, variations in $A$ are no longer related to variations in $C$ through this path. We have controlled for the confounder.

**3. Collider (Common Effect)**

Structure: $A \to B \leftarrow C$, where both $A$ and $C$ cause $B$. This behaves opposite to chains and forks.

- If $B$ and all its descendants are not observed (not in $Z$): The path is **blocked**. Since $A$ and $C$ are independent causes, learning about $A$ gives no information about $C$ unless we know the outcome $B$. The information paths collide at $B$ and stop.
- If $B$ or any descendant of $B$ is observed (in $Z$): The path is **open**. This is called explaining away or Berkson's paradox. If we know the result $B$ occurred, learning that cause $A$ did not occur makes it more likely that cause $C$ did occur to explain $B$. Conditioning on any descendant of $B$ also opens the path because the descendant acts as a proxy for $B$.

{{< figure
  src="/images/maths/dSeparationPatterns.jpg"
  alt="Diagram showing the three basic patterns: chain, fork, and collider, with blocking conditions"
  caption="The three fundamental d-separation patterns and their blocking conditions"
>}}

The following table summarizes all cases for a single junction:

| Structure | Middle Node in $Z$? | Path Status |
|-----------|---------------------|-------------|
| Chain: $A \to B \to C$ | No | Open |
| Chain: $A \to B \to C$ | Yes | Blocked |
| Fork: $A \leftarrow B \to C$ | No | Open |
| Fork: $A \leftarrow B \to C$ | Yes | Blocked |
| Collider: $A \to B \leftarrow C$ | No (and no descendants) | Blocked |
| Collider: $A \to B \leftarrow C$ | Yes (or a descendant) | Open |

### Determining d-Separation

A path between $A$ and $B$ is blocked by conditioning set $C$ if it contains at least one of the following:
- A non-collider node (middle node in a chain or fork) that is in $C$, or
- A collider node that is not in $C$ and has no descendants in $C$

Variables $A$ and $B$ are d-separated given $C$ if every path between $A$ and $B$ is blocked by $C$. If even one path remains open, they are d-connected, meaning they are potentially dependent given $C$.

The d-separation criterion gives us a complete graphical test for conditional independence. If $A$ and $B$ are d-separated given $C$ in the graph, then they are conditionally independent given $C$ in any probability distribution that respects the causal graph structure. However, if $A$ and $B$ are not d-separated, we cannot conclude anything definitive about their independence.

## Necessary Conditions for Counterfactual Invariance

Now we can state the precise conditions under which a model can be counterfactually invariant. These conditions tell us what independence properties must hold for the predictions $f(X)$ to be invariant to changes in the environmental variable $W$.

The key insight is that if $f$ is counterfactually invariant, then $f$ can depend only on the parts of $X$ that are not influenced by $W$. We denote by $X_W^\perp$ the components of $X$ that are independent of $W$ (orthogonal to $W$ in a causal sense).

For an estimator $f$ to be counterfactually invariant with respect to $W$, the following conditions are necessary:

**Anti-Causal Scenario** (where $Y$ causes $X$): We need $f(X) \perp W \mid Y$. The prediction must be independent of $W$ when conditioning on the true label $Y$.

In the anti-causal scenario, the causal structure is $Y \to X \leftarrow W$, which forms a collider at $X$. To check if $f(X) \perp W \mid Y$, we examine paths between $f(X)$ (which depends on $X$) and $W$. When we condition on $Y$, we are observing the node that causes $X$. The path $W \to X \leftarrow Y$ is a collider at $X$, and since we are not conditioning on $X$ itself but rather on its cause $Y$, the specific path structure determines whether information flows. By conditioning on $Y$, we block the influence that $Y$ has on creating dependencies, allowing $f(X)$ to be independent of $W$ given $Y$. This means that $f(X) \mid W = w, Y = y$ should be the same for all values of $w$.

{{< figure
  src="/images/maths/counterfactualInvarianceAntiScenario.png"
  alt="Causal graph for anti-causal scenario showing Y causing X, with W also influencing X"
  caption="Anti-causal scenario: Conditioning on Y blocks the influence of W on predictions f(X)"
>}}

**Causal Scenario** (where $X$ causes $Y$): We need $f(X) \perp W$. The prediction must be unconditionally independent of $W$.

In the causal scenario, the causal structure is $W \to X \to Y$. To check if $f(X) \perp W$, we examine paths between $f(X)$ and $W$ without conditioning on any variables. The path $W \to X$ must be blocked for this independence to hold. This is only possible if $f$ depends solely on the components of $X$ that are not influenced by $W$ (the $X_W^\perp$ components). Any components of $X$ that are influenced by $W$ (such as $X_{W \& Y}$) must not be used by $f$ for the prediction to be invariant. This is a stronger condition than the anti-causal case because we need unconditional independence rather than conditional independence.

{{< figure
  src="/images/maths/counterfactualInvarianceCausalScenario.png"
  alt="Causal graph for causal scenario showing W influencing X, which causes Y"
  caption="Causal scenario: f(X) must be unconditionally independent of W, using only X_W^⊥ components"
>}}