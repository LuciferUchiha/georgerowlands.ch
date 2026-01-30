---
title: Support Vector Machines
type: docs
weight: 8
---

Support Vector Machines (SVMs) are a fundamental tool in machine learning that rely heavily on [convex optimization](/garden/maths/calculus/convexOptimization.md) to find optimal decision boundaries between classes. The key insight behind SVMs is to find a hyperplane that separates classes with the maximum possible margin. This approach provides better generalization than simpler methods like the perceptron and produces classifiers that are robust to small perturbations in the data.

{{< figure src="/images/ml/svm3d.gif" caption="3D visualization of SVM separating hyperplane with maximum margin" alt="3D visualization of SVM separating hyperplane with maximum margin" >}}

The perceptron (which you can think of as a single-layer neural network with a step activation function) is a classic algorithm that finds a linear separator for linearly separable data by iteratively adjusting weights until all points are correctly classified. However, the perceptron suffers from a critical instability problem. When data is linearly separable, there exist infinitely many hyperplanes that correctly classify all training points. The perceptron will find one of these hyperplanes, but which one it finds depends on the initialization and the order in which training examples are processed.

This instability means that small changes in the training data or the training procedure can lead to very different decision boundaries. Some of these boundaries may lie very close to the training points, making them sensitive to noise and leading to poor generalization on new data. We need a principled way to select the "best" hyperplane among all possible separating hyperplanes.

## The Maximum Margin Principle

In binary classification, we use a **hyperplane** as our decision boundary to separate the two classes. A hyperplane in $d$-dimensional space is a $(d-1)$-dimensional flat subspace defined by the equation:

$$
w^T x + w_0 = 0
$$

where $w \in \mathbb{R}^d$ is the normal vector perpendicular to the hyperplane and $w_0 \in \mathbb{R}$ is the bias term. Points on one side of the hyperplane (where $w^T x + w_0 > 0$) are classified as positive, while points on the other side (where $w^T x + w_0 < 0$) are classified as negative.

Support Vector Machines address the perceptron's instability by introducing the concept of a **margin**, which measures how far the decision boundary is from the nearest training points. Intuitively, a larger margin means the decision boundary has more "breathing room" around it, making it more robust to noise and more likely to generalize well. The SVM approach is to find the hyperplane that maximizes this margin, giving us a unique solution that depends only on the data, not on the training procedure. The training points that lie exactly on the margin boundaries are called **support vectors**, and remarkably, the optimal hyperplane depends only on these support vectors.

For a linearly separable binary classification problem with labels $y_i \in \{+1, -1\}$ and data points $x_i$, we require that positive examples satisfy $w^T x_i + w_0 > 0$ and negative examples satisfy $w^T x_i + w_0 < 0$. We can combine these constraints into a single condition $y_i(w^T x_i + w_0) > 0$ for all training points, which works because multiplying by $y_i$ flips the sign for negative examples.

The **geometric margin** for a point $x_i$ is the perpendicular distance from that point to the hyperplane. Let $x^+$ be a positive example on the margin boundary and $x^-$ be a negative example on the margin boundary. We define $m(w, w_0)$ as the distance from the separating hyperplane to either margin boundary. The total **margin width** is therefore $2m(w, w_0)$ because it spans the distance between the two parallel hyperplanes on either side of the decision boundary (one containing the closest positive examples and one containing the closest negative examples). The factor of 2 comes from measuring the full width from the positive margin boundary to the negative margin boundary, passing through the central hyperplane.

To compute this distance, we project both points onto the normal vector $w$ of the hyperplane. The projection of $x^+$ is $\frac{w^T x^+}{\|w\|}$ and the projection of $x^-$ is $\frac{w^T x^-}{\|w\|}$. The distance between these projections gives us:

$$
2m(w, w_0) = \left\| \frac{w^T x^+}{\|w\|} - \frac{w^T x^-}{\|w\|} \right\| = \frac{w^T x^+ - w^T x^-}{\|w\|}
$$

{{< figure src="/images/ml/svm.png" caption="Geometric visualization of the margin and support vectors in 2D" alt="Geometric visualization of the margin and support vectors" >}}

### The Normalization Trick

The margin $m(w, w_0)$ does not depend on the magnitude of $w$ or $w_0$. Intuitively, $w$ is the normal vector to the hyperplane, which defines the direction perpendicular to the decision boundary. A normal vector can have any length while still pointing in the same direction. Conventionally, normal vectors are normalized to have unit length ($\|w\| = 1$), but mathematically, $w$ and $\gamma w$ for any $\gamma > 0$ point in the same direction and define the same hyperplane orientation.

To see why scaling doesn't change the solution, consider what happens when we scale both parameters by a positive constant $\gamma > 0$. The hyperplane equation becomes $(\gamma w)^T x + \gamma w_0 = 0$, which simplifies to $\gamma(w^T x + w_0) = 0$. Since $\gamma \neq 0$, dividing both sides by $\gamma$ gives us back $w^T x + w_0 = 0$. The geometric location of the decision boundary remains unchanged.

Similarly, when computing the margin width $\frac{2}{\|w\|}$, if we use $\gamma w$ instead, we get $\frac{2}{\|\gamma w\|} = \frac{2}{\gamma \|w\|}$. However, the constraints $y_i(w^T x_i + w_0) \geq 1$ also get scaled to $y_i((\gamma w)^T x_i + \gamma w_0) \geq \gamma$, so the geometry of the margin remains the same. This **scale invariance** means that infinitely many $(w, w_0)$ pairs represent the same geometric solution.

This scale invariance allows us to use a clever normalization trick. We can choose the scale such that the closest points to the hyperplane satisfy $w^T x^+ + w_0 = 1$ for positive examples and $w^T x^- + w_0 = -1$ for negative examples. With this normalization, the margin simplifies to:

$$
2m(w, w_0) = \frac{|w^T x^+ - w^T x^-|}{\|w\|} = \frac{|1 - (-1)|}{\|w\|} = \frac{2}{\|w\|}
$$

Therefore, maximizing the margin is equivalent to maximizing:

$$
m(w, w_0) = \frac{1}{\|w\|}
$$

which is the same as minimizing $\|w\|$.

## Hard Margin SVM

With the normalization trick, we can now formulate the SVM optimization problem. We want to maximize the margin $\frac{2}{\|w\|}$, which is equivalent to minimizing $\|w\|$ or, more conveniently, minimizing $\frac{1}{2}\|w\|^2$. The factor of $\frac{1}{2}$ simplifies the derivatives we will compute later.

This results in a constrained convex optimization problem called the **hard margin SVM primal problem**:

$$
\begin{aligned}
\min_{w, w_0} \quad & \frac{1}{2} w^T w \\
\text{subject to} \quad & y_i(w^T x_i + w_0) \geq 1 \quad \text{for } i = 1, \ldots, n
\end{aligned}
$$

This is a [convex optimization](/garden/maths/calculus/convexOptimization.md) problem for two reasons. First, the objective function $\frac{1}{2} w^T w = \frac{1}{2} \|w\|^2$ is convex. It is a quadratic form with a positive semidefinite Hessian (the Hessian is the identity matrix), which means it is strictly convex. Geometrically, this function forms a paraboloid that curves upward in all directions.

Second, the feasible region defined by the constraints is convex. Each constraint $y_i(w^T x_i + w_0) \geq 1$ defines a halfspace (the region on one side of a hyperplane in the $(w, w_0)$ space). The intersection of convex sets is convex, so the feasible region is convex. Together, minimizing a convex function over a convex set makes this a convex optimization problem, which guarantees that any local minimum is also a global minimum and that efficient algorithms exist to solve it.

## The Dual Problem

While we could solve the primal problem directly, because it is a convex optimization problem as shown in [Convex Optimization](/garden/maths/calculus/convexOptimization.md), there exists a related **dual problem** that is often easier to solve. The dual formulation has several important advantages for SVMs. First, the dual depends only on inner products between data points, which allows us to apply the kernel trick later. Second, the dual does not depend on the dimensionality of the feature space, only on the number of training examples. Third, solving the dual directly gives us the Lagrange multipliers, which tell us which points are support vectors.

In general, the dual problem provides a lower bound on the optimal value of the primal problem (this is called weak duality). Under certain conditions (which we will verify for SVMs), the dual and primal optimal values are equal (strong duality), allowing us to solve the dual instead of the primal.

To derive the dual, we use the method of **Lagrange multipliers** from [constrained optimization](/garden/maths/calculus/convexOptimization.md). The Lagrangian is a function that combines the objective function with the constraints, weighted by Lagrange multipliers. For each inequality constraint of the form $g_i(x) \geq 0$, we introduce a multiplier $\alpha_i \geq 0$ and add the term $\alpha_i g_i(x)$ to the objective. The Lagrangian converts the constrained optimization problem into an unconstrained one by penalizing constraint violations.

For the SVM primal problem, we rewrite each constraint as $1 - y_i(w^T x_i + w_0) \leq 0$ and introduce Lagrange multipliers $\alpha_i \geq 0$:

$$
L(w, w_0, \alpha) = \frac{1}{2} w^T w + \sum_{i=1}^n \alpha_i [1 - y_i(w^T x_i + w_0)]
$$

Expanding this expression:

$$
L(w, w_0, \alpha) = \frac{1}{2} w^T w + \sum_{i=1}^n \alpha_i - \sum_{i=1}^n \alpha_i y_i w^T x_i - w_0 \sum_{i=1}^n \alpha_i y_i
$$

To find the dual function, we minimize the Lagrangian with respect to the primal variables $w$ and $w_0$. Taking derivatives:

$$
\frac{\partial L}{\partial w} = w - \sum_{i=1}^n \alpha_i y_i x_i = 0
$$

$$
\frac{\partial L}{\partial w_0} = -\sum_{i=1}^n \alpha_i y_i = 0
$$

From the first condition, we obtain that the optimal $w$ is a linear combination of the training points:

$$
w^* = \sum_{i=1}^n \alpha_i y_i x_i
$$

The second condition gives us an additional constraint for the dual:

$$
\sum_{i=1}^n \alpha_i y_i = 0
$$

The first equation is the key insight of SVMs: the optimal weight vector $w^*$ can be expressed entirely in terms of the training data and the Lagrange multipliers $\alpha_i$. It only depends on datapoints and not their dimensionality directly. It also shows that only points with $\alpha_i > 0$ contribute to $w^*$, which will turn out to be the support vectors.

Now we substitute $w = \sum_i \alpha_i y_i x_i$ and $\sum_i \alpha_i y_i = 0$ back into the Lagrangian to eliminate the primal variables. Starting with:

$$
L(w, w_0, \alpha) = \frac{1}{2} w^T w + \sum_{i=1}^n \alpha_i - \sum_{i=1}^n \alpha_i y_i w^T x_i - w_0 \sum_{i=1}^n \alpha_i y_i
$$

The last term vanishes because $\sum_i \alpha_i y_i = 0$. For the first term, we have:

$$
\frac{1}{2} w^T w = \frac{1}{2} \left(\sum_i \alpha_i y_i x_i\right)^T \left(\sum_j \alpha_j y_j x_j\right) = \frac{1}{2} \sum_{i=1}^n \sum_{j=1}^n \alpha_i \alpha_j y_i y_j x_i^T x_j
$$

For the third term:

$$
\sum_{i=1}^n \alpha_i y_i w^T x_i = \sum_{i=1}^n \alpha_i y_i \left(\sum_j \alpha_j y_j x_j\right)^T x_i = \sum_{i=1}^n \sum_{j=1}^n \alpha_i \alpha_j y_i y_j x_j^T x_i
$$

Since $x_i^T x_j = x_j^T x_i$ (the inner product is symmetric), the third term equals the first term. Combining everything:

$$
\theta(\alpha) = \sum_{i=1}^n \alpha_i - \frac{1}{2} \sum_{i=1}^n \sum_{j=1}^n \alpha_i \alpha_j y_i y_j x_i^T x_j
$$

The **dual problem** is then:

$$
\begin{aligned}
\max_{\alpha} \quad & \sum_{i=1}^n \alpha_i - \frac{1}{2} \sum_{i=1}^n \sum_{j=1}^n \alpha_i \alpha_j y_i y_j x_i^T x_j \\
\text{subject to} \quad & \alpha_i \geq 0 \quad \text{for } i = 1, \ldots, n \\
& \sum_{i=1}^n \alpha_i y_i = 0
\end{aligned}
$$

This is also a [convex quadratic programming problem](/garden/maths/calculus/convexOptimization.md), and it can be solved efficiently using standard QP solvers. The dual objective might look complicated, but it has a natural interpretation. The first term $\sum_i \alpha_i$ encourages large values of $\alpha_i$, which correspond to points near or on the margin boundaries (the support vectors). The second term $-\frac{1}{2} \sum_{i,j} \alpha_i \alpha_j y_i y_j x_i^T x_j$ penalizes configurations where support vectors are too close together (measured by their inner products $x_i^T x_j$), weighted by their labels. Together, these terms balance finding a decision boundary with a large margin while ensuring good separation between classes.

### Strong Duality and Slater's Condition

As discussed in [Convex Optimization](/garden/maths/calculus/convexOptimization.md), **weak duality** states that the dual optimal value provides a lower bound on the primal optimal value. This always holds for any optimization problem. However, **strong duality** is the stronger condition where the dual and primal optimal values are exactly equal. Strong duality does not always hold, but for convex optimization problems satisfying certain constraint qualifications (like Slater's condition), strong duality is guaranteed.

For the dual problem to give us the same optimal value as the primal (strong duality), we need Slater's condition to hold. **Slater's condition** is a constraint qualification that requires the existence of a strictly feasible point for the primal problem. For a problem with inequality constraints $g_i(x) \geq 0$, Slater's condition requires that there exists at least one point $x$ such that all inequality constraints are satisfied with strict inequality: $g_i(x) > 0$ for all $i$.

For the SVM primal problem with constraints $y_i(w^T x_i + w_0) \geq 1$, Slater's condition requires finding $(w, w_0)$ such that:

$$
y_i(w^T x_i + w_0) > 1 \quad \text{for all } i = 1, \ldots, n
$$

When does this hold? If the data is linearly separable, there exists some hyperplane $(w', w_0')$ that perfectly separates the classes, meaning $y_i(w'^T x_i + w_0') > 0$ for all $i$. Since the data is finite and the inequality is strict, there exists some $\epsilon > 0$ such that $y_i(w'^T x_i + w_0') \geq \epsilon$ for all $i$.

Now, using the scale invariance property, we can scale $(w', w_0')$ by a factor $\gamma = \frac{1+\delta}{\epsilon}$ for any small $\delta > 0$ to get:

$$
y_i((\gamma w')^T x_i + \gamma w_0') = \gamma \cdot y_i(w'^T x_i + w_0') \geq \gamma \epsilon = 1 + \delta > 1
$$

This shows that $(w, w_0) = (\gamma w', \gamma w_0')$ is strictly feasible, satisfying Slater's condition. Therefore, strong duality holds for hard margin SVMs on linearly separable data, guaranteeing that solving the dual problem gives us the optimal solution to the primal.

### Complementary Slackness and Support Vectors

Strong duality implies that the [KKT (Karush-Kuhn-Tucker) conditions](/garden/maths/calculus/convexOptimization.md) hold at the optimum. One of these conditions is **complementary slackness**:

$$
\alpha_i^* (1 - y_i(w^{*T} x_i + w_0^*)) = 0 \quad \text{for all } i
$$

This condition means that for each training point, either $\alpha_i^* = 0$ or the constraint is active (the point lies exactly on the margin boundary). As mentioned earlier, these points with $\alpha_i^* > 0$ are the **support vectors**. These are the points that lie exactly on the margin boundaries, satisfying $y_i(w^{*T} x_i + w_0^*) = 1$. The optimal hyperplane depends only on these support vectors. Points with $\alpha_i^* = 0$ do not contribute to $w^* = \sum_i \alpha_i y_i x_i$ and could be removed from the training set without changing the solution.

In practice, the solution vector $\alpha^*$ is typically sparse, meaning most $\alpha_i^*$ are zero. This sparsity is a key property of SVMs and makes them efficient at test time since predictions only require computing inner products with the support vectors rather than all training points.

### Computing the Hyperplane

After solving the dual problem, we obtain $\alpha^*$ and can compute $w^* = \sum_i \alpha_i^* y_i x_i$ easily. However, we still need to find $w_0^*$. We can use any support vector for this. If $x^+$ is a positive support vector (with $y = +1$ and $\alpha > 0$), then it satisfies $w^{*T} x^+ + w_0^* = 1$, giving us $w_0^* = 1 - w^{*T} x^+$. Similarly, if $x^-$ is a negative support vector, it satisfies $w^{*T} x^- + w_0^* = -1$, giving us $w_0^* = -1 - w^{*T} x^-$.

While theoretically any support vector gives the correct $w_0^*$, in practice numerical errors from the optimization solver mean that different support vectors may give slightly different values. Averaging over all support vectors provides a more stable and robust estimate that is less sensitive to these numerical inaccuracies. The standard approach is to average over one positive and one negative support vector (or over all support vectors):

$$
w_0^* = -\frac{1}{2}(w^{*T} x^+ + w^{*T} x^-)
$$

where $x^+$ is any positive support vector and $x^-$ is any negative support vector. This can be further improved by averaging over all support vectors, which gives the most numerically stable result.

{{< callout type="example" >}}
Consider a simple 2D dataset with four points:
- Positive class: $x_1 = (1, 1)$ with $y_1 = +1$ and $x_2 = (2, 2)$ with $y_2 = +1$
- Negative class: $x_3 = (-1, -1)$ with $y_3 = -1$ and $x_4 = (-2, -2)$ with $y_4 = -1$

The dual problem for this dataset is:

$$
\max_{\alpha} \sum_{i=1}^4 \alpha_i - \frac{1}{2} \sum_{i=1}^4 \sum_{j=1}^4 \alpha_i \alpha_j y_i y_j x_i^T x_j
$$

subject to $\alpha_i \geq 0$ and $\sum_{i=1}^4 \alpha_i y_i = 0$.

Computing the inner products: $x_1^T x_1 = 2$, $x_1^T x_3 = -2$, $x_3^T x_3 = 2$, etc. The symmetry of this dataset means that the optimal solution has $\alpha_1 = \alpha_3 = \alpha$ for some value $\alpha > 0$, and $\alpha_2 = \alpha_4 = 0$ (the points further from the boundary are not support vectors).

Solving the optimization problem yields $\alpha_1 = \alpha_3 = 0.25$. This gives us:

$$
w^* = \alpha_1 y_1 x_1 + \alpha_3 y_3 x_3 = 0.25 \cdot 1 \cdot (1,1) + 0.25 \cdot (-1) \cdot (-1,-1) = (0.5, 0.5)
$$

Using the support vector $x_1 = (1,1)$:

$$
w_0^* = 1 - w^{*T} x_1 = 1 - 0.5 \cdot 1 - 0.5 \cdot 1 = 0
$$

The optimal hyperplane is $0.5 x_1 + 0.5 x_2 = 0$, or equivalently $x_1 + x_2 = 0$, which perfectly bisects the two classes with support vectors at $(1,1)$ and $(-1,-1)$.

{{< figure src="/images/ml/svmExample.png" caption="Hard margin SVM solution for a simple 2D dataset showing the decision boundary, margins, and support vectors" alt="Hard margin SVM example with support vectors" >}}
{{< /callout >}}

### Making Predictions

Once we have trained the SVM and obtained the optimal parameters $w^*$ and $w_0^*$, we can classify a new data point $x$ by computing the signed distance to the decision boundary. The prediction is given by:

$$
\hat{y} = \text{sign}(w^{*T} x + w_0^*)
$$

Since $w^* = \sum_{i=1}^n \alpha_i^* y_i x_i$, we can write this in terms of the support vectors:

$$
\hat{y} = \text{sign}\left(\sum_{i \in SV} \alpha_i^* y_i x_i^T x + w_0^*\right)
$$

where the sum is only over support vectors (points with $\alpha_i^* > 0$). The magnitude $|w^{*T} x + w_0^*|$ represents the confidence of the prediction, with points far from the decision boundary having larger magnitudes. This formulation in terms of inner products $x_i^T x$ will become important when we introduce kernels.

## Soft Margin SVM

The hard margin SVM requires that the data be linearly separable. In practice, data is often not linearly separable due to noise or inherent overlap between classes. Even if the data is technically separable, enforcing perfect separation may lead to overfitting.

Soft margin SVMs relax the hard margin constraints by introducing **slack variables** $\xi_i \geq 0$ for each training point. The slack variable $\xi_i$ measures how much point $i$ violates the margin constraint. If $\xi_i = 0$, the point satisfies the margin constraint perfectly. If $0 < \xi_i < 1$, the point is correctly classified but within the margin. If $\xi_i > 1$, the point is misclassified as it lies on the wrong side of the hyperplane. The modified constraint becomes:

$$
y_i(w^T x_i + w_0) \geq 1 - \xi_i
$$

We want to minimize the total amount of slack, so we add a penalty term $C \sum_i \xi_i$ to the objective:

$$
\begin{aligned}
\min_{w, w_0, \xi} \quad & \frac{1}{2} w^T w + C \sum_{i=1}^n \xi_i \\
\text{subject to} \quad & y_i(w^T x_i + w_0) \geq 1 - \xi_i \quad \text{for } i = 1, \ldots, n \\
& \xi_i \geq 0 \quad \text{for } i = 1, \ldots, n
\end{aligned}
$$

{{< figure src="/images/ml/svmSoftMargin.png" caption="Soft margin SVM with slack variables allowing margin violations" alt="Soft margin SVM with slack variables allowing margin violations" >}}

The hyperparameter $C > 0$ controls the trade-off between maximizing the margin and minimizing classification errors. A large value of $C$ penalizes slack heavily, pushing the solution toward the hard margin SVM (perfect separation if possible). A small value of $C$ allows more slack, resulting in a wider margin that may misclassify some training points.

You can interpret $C$ as a regularization parameter. When $C$ is large, we prioritize fitting the training data (low regularization). When $C$ is small, we prioritize finding a simple decision boundary with a large margin (high regularization).

{{< figure src="/images/ml/svmSoftMarginHyperparameter.jpg" caption="Effect of the C hyperparameter on the decision boundary and margin" alt="Effect of the C hyperparameter on the decision boundary and margin" >}}

### Soft Margin Dual

The dual problem for soft margin SVMs is similar to the hard margin dual, with one important difference: the Lagrange multipliers are now upper-bounded by $C$. To see where this upper bound comes from, we need to derive the dual carefully.

We form the Lagrangian by introducing multipliers $\alpha_i \geq 0$ for the margin constraints and $\beta_i \geq 0$ for the non-negativity constraints on the slack variables:

$$
L(w, w_0, \xi, \alpha, \beta) = \frac{1}{2} w^T w + C \sum_{i=1}^n \xi_i + \sum_{i=1}^n \alpha_i [1 - \xi_i - y_i(w^T x_i + w_0)] - \sum_{i=1}^n \beta_i \xi_i
$$

To find the dual, we minimize with respect to the primal variables. Taking derivatives:

$$
\frac{\partial L}{\partial w} = w - \sum_{i=1}^n \alpha_i y_i x_i = 0 \implies w = \sum_{i=1}^n \alpha_i y_i x_i
$$

$$
\frac{\partial L}{\partial w_0} = -\sum_{i=1}^n \alpha_i y_i = 0 \implies \sum_{i=1}^n \alpha_i y_i = 0
$$

$$
\frac{\partial L}{\partial \xi_i} = C - \alpha_i - \beta_i = 0 \implies \alpha_i + \beta_i = C
$$

The third condition is crucial. Since $\beta_i \geq 0$, we have $\alpha_i = C - \beta_i \leq C$. Combined with $\alpha_i \geq 0$, this gives us the box constraint $0 \leq \alpha_i \leq C$. Substituting these conditions back into the Lagrangian and simplifying (following the same steps as for the hard margin case), we obtain:

$$
\begin{aligned}
\max_{\alpha} \quad & \sum_{i=1}^n \alpha_i - \frac{1}{2} \sum_{i=1}^n \sum_{j=1}^n \alpha_i \alpha_j y_i y_j x_i^T x_j \\
\text{subject to} \quad & 0 \leq \alpha_i \leq C \quad \text{for } i = 1, \ldots, n \\
& \sum_{i=1}^n \alpha_i y_i = 0
\end{aligned}
$$

The complementary slackness conditions now become more nuanced. For each point:
- If $\alpha_i = 0$, the point is correctly classified and outside the margin
- If $0 < \alpha_i < C$, the point is a support vector on the margin boundary
- If $\alpha_i = C$, the point is either on the wrong side of the margin or misclassified

The soft margin formulation applies even when data is not linearly separable, making it much more practical than the hard margin SVM.

It is important to understand that $C$ does not directly specify the number of errors allowed. A common misconception is that "C=1 allows 1 error", but this is incorrect. Instead, $C$ is a penalty coefficient that controls the trade-off between margin width and constraint violations. Looking at the objective $\frac{1}{2}\|w\|^2 + C \sum_i \xi_i$, we see that $C$ weights how much we care about the total slack $\sum_i \xi_i$ relative to the margin term $\|w\|^2$. The actual number of margin violations and misclassifications that occur depends on the data and the optimization, not on the specific value of $C$. As $C \to \infty$, we approach the hard margin SVM (no slack tolerated), while smaller values of $C$ allow the optimizer to accept more slack in exchange for a larger margin.

## Kernel Methods

So far, we have considered only linear decision boundaries. However, many real-world problems require non-linear decision boundaries. Kernel methods provide an elegant way to extend SVMs to non-linear classification without explicitly working in high-dimensional feature spaces.

The key idea behind non-linear SVMs is the same as in [polynomial regression](/garden/ml/linearregression.md#polynomial-regression): transform the input data $x$ into a higher-dimensional feature space using a feature map $\phi: \mathbb{R}^d \to \mathbb{R}^D$ where typically $D \gg d$. In this higher-dimensional space, the data may become linearly separable even if it was not separable in the original space. We then apply a linear SVM in the transformed feature space, which corresponds to a non-linear decision boundary in the original input space.

For example, consider data in $\mathbb{R}^2$ that is not linearly separable but can be separated by a circle. We can use the feature map:

$$
\phi(x) = (x_1^2, \sqrt{2} x_1 x_2, x_2^2)
$$

This maps 2D points into 3D space, and a linear separator in 3D corresponds to a conic section (circle, ellipse, etc.) in the original 2D space.

### The Kernel Trick

Computing $\phi(x)$ explicitly can be expensive or even impossible when $D$ is very large or infinite. However, notice that the SVM dual only requires computing inner products between data points: $x_i^T x_j$ appears in the dual objective. If we use a feature map, we would need to compute $\phi(x_i)^T \phi(x_j)$.

The **kernel trick** exploits the fact that for many useful feature maps, we can compute the inner product $\phi(x_i)^T \phi(x_j)$ directly without explicitly computing $\phi(x_i)$ and $\phi(x_j)$. A **kernel function** $K(x, x')$ is defined as:

$$
K(x, x') = \phi(x)^T \phi(x')
$$

Not every function can serve as a valid kernel. **Mercer's theorem** provides the necessary and sufficient condition: a symmetric function $K(x, x')$ is a valid kernel (corresponds to an inner product in some feature space) if and only if it is positive semidefinite. This means that for any finite set of points $\{x_1, \ldots, x_n\}$, the kernel matrix $K_{ij} = K(x_i, x_j)$ must be positive semidefinite.

Common kernel functions include the linear kernel $K(x, x') = x^T x'$, polynomial kernels, and the RBF (Radial Basis Function) kernel. For a comprehensive discussion of kernel functions and their properties, see the [kernel functions section](/garden/ml/bayesian/gaussianprocesses.md#kernel-functions) in the Gaussian Processes notes. The choice of kernel and its hyperparameters significantly affects the decision boundary's shape and the model's generalization performance.

{{< figure src="/images/ml/svmKernel.png" caption="Different kernels affecting the decision boundary" alt="Different kernels affecting the decision boundary" >}}

### Training and Prediction with Kernels

To see how kernels are used in SVMs, recall the soft margin dual problem:

$$
\max_{\alpha} \sum_{i=1}^n \alpha_i - \frac{1}{2} \sum_{i=1}^n \sum_{j=1}^n \alpha_i \alpha_j y_i y_j x_i^T x_j
$$

subject to $0 \leq \alpha_i \leq C$ and $\sum_i \alpha_i y_i = 0$. Notice that the objective depends on the data only through inner products $x_i^T x_j$.

Now suppose we want to work in a transformed feature space defined by $\phi$. If we were to derive the dual problem in this transformed space, we would replace each $x$ with $\phi(x)$, giving:

$$
\max_{\alpha} \sum_{i=1}^n \alpha_i - \frac{1}{2} \sum_{i=1}^n \sum_{j=1}^n \alpha_i \alpha_j y_i y_j \phi(x_i)^T \phi(x_j)
$$

The kernel trick is to recognize that we can replace $\phi(x_i)^T \phi(x_j)$ with a kernel function $K(x_i, x_j)$ without ever computing $\phi$ explicitly. This gives us the kernelized dual problem:

$$
\begin{aligned}
\max_{\alpha} \quad & \sum_{i=1}^n \alpha_i - \frac{1}{2} \sum_{i=1}^n \sum_{j=1}^n \alpha_i \alpha_j y_i y_j K(x_i, x_j) \\
\text{subject to} \quad & 0 \leq \alpha_i \leq C \\
& \sum_{i=1}^n \alpha_i y_i = 0
\end{aligned}
$$

For prediction on a new point $x$, we again just need inner products between $x$ and the training points. The prediction rule becomes:

$$
\hat{y} = \text{sign}\left(\sum_{i \in SV} \alpha_i y_i K(x_i, x) + w_0\right)
$$

where the sum is over support vectors only. Notice that we never need to explicitly compute $\phi(x)$ or $w$ in the feature space. Everything is expressed in terms of kernel evaluations.

## Multiclass SVMs

The SVM formulation we have developed is inherently binary (two classes). For multiclass problems with $M > 2$ classes, there are several approaches.

One approach is **one-vs-rest**: train $M$ binary SVMs, where classifier $k$ separates class $k$ from all other classes. At test time, classify a point $x$ according to the classifier with the highest score, where the score for classifier $k$ is the decision function value $w_k^T x + w_{k,0}$. The point is assigned to the class $\arg\max_k (w_k^T x + w_{k,0})$, corresponding to the classifier that is most confident in its positive prediction.

Another approach is **one-vs-one**: train $\binom{M}{2}$ binary SVMs, one for each pair of classes. At test time, use voting to determine the final class.

A more principled approach is to extend the margin concept directly to multiple classes. For each class $z \in \{1, 2, \ldots, M\}$, we introduce a separate weight vector $w_z \in \mathbb{R}^d$ and bias term $w_{z,0}$. Each weight vector $w_z$ lives in the same $d$-dimensional feature space as the original data. The concatenation notation $W = [w_1, w_2, \ldots, w_M] \in \mathbb{R}^{d \times M}$ is simply a compact way to represent all $M$ weight vectors as columns of a matrix, where column $z$ contains the weights for class $z$. There are no "spaces inbetween" - this is purely a mathematical notation for organizing the parameters.

For prediction, we compute a score $w_z^T x + w_{z,0}$ for each class $z$ by taking the inner product of the input $x$ with that class's weight vector $w_z$. The class with the highest score is chosen. For a point with true label $z_i$, we require:

$$
w_{z_i}^T x_i + w_{z_i,0} \geq \max_{z \neq z_i} (w_z^T x_i + w_{z,0}) + 1
$$

This ensures that the score for the correct class exceeds the score for any incorrect class by at least a margin of 1. This formulation leads to a multiclass SVM that can be solved similarly to the binary case.

{{< figure src="/images/ml/svmMulticlass.png" caption="Multiclass SVM decision boundaries separating multiple classes" alt="Multiclass SVM decision boundaries separating multiple classes" >}}