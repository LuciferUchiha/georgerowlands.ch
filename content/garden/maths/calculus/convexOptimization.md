---
title: Convex Optimization
type: docs
weight: 13
---

Optimization is the process of finding the best solution to a problem from a set of possible solutions. In machine learning, we constantly face optimization problems when training models as we want to find parameters that minimize a loss function or maximize a likelihood. Remember we can turn maximization into minimization by negating the objective. While general optimization is difficult and can have many local optima, **convex optimization** is a special class where we can efficiently find the global minima.

Understanding convex optimization is fundamental for machine learning because many important algorithms rely on it. Ridge regression, Lasso, Support Vector Machines, and many neural network architectures all involve solving convex (or approximately convex) optimization problems. The theory we develop here will provide the mathematical foundation for understanding these methods.

## Optimization Problems

Before we dive into convex optimization, let us establish what we mean by an optimization problem in general. An **unconstrained optimization problem** is one where we seek to minimize a function $f: \mathbb{R}^d \to \mathbb{R}$ over all possible inputs:

$$
\min_{w \in \mathbb{R}^d} f(w)
$$

In many practical situations, we cannot choose any arbitrary $w$. For example, when training a classifier, we might want the weights to satisfy certain properties like having a bounded norm or summing to one. This leads us to **constrained optimization** problems, where we minimize $f(w)$ subject to some constraints on $w$. Let $f, g_1, g_2, \ldots, g_m, h_1, h_2, \ldots, h_n$ be functions with domain in $\mathbb{R}^d$ and codomain in $\mathbb{R}$. A constrained optimization problem has the form:

$$
\begin{align}
\min_{w \in \mathbb{R}^d} \quad & f(w) \\
\text{subject to} \quad & g_i(w) = 0, \quad \text{for } i \leq m, \\
& h_j(w) \leq 0, \quad \text{for } j \leq n.
\end{align}
$$

Here $f(w)$ is called the **objective function** (or cost function), the equations $g_i(w) = 0$ are called **equality constraints**, and the inequalities $h_j(w) \leq 0$ are called **inequality constraints**. Together, these constraints define the **feasible region** of the optimization problem.

A point $w \in \mathbb{R}^d$ is called a **feasible solution** if it satisfies all the constraints. Among all feasible solutions, $w^*$ is called an **optimal solution** if $f(w^*) \leq f(w')$ for any other feasible point $w'$.

{{< figure
    src="/images/maths/optimizationProblems.jpg"
    caption="Constrained optimization problems involve finding the minimum of an objective function subject to constraints that define a feasible region."
    alt="Illustration of constrained optimization with feasible region"
>}}

## Convex Functions and Sets

The power of convex optimization comes from the geometric properties of convex functions and convex sets. These properties ensure that any local minimum is also a global minimum, making optimization tractable (the same holds for maximization of concave functions, a concave function is just the negative of a convex function).

We define a **Convex Set** as a set $C \subseteq \mathbb{R}^d$ is convex if for any two points $x, y \in C$ and any $\theta \in [0, 1]$, the point $\theta x + (1 - \theta) y$ is also in $C$. Intuitively, a set is convex if the line segment connecting any two points in the set lies entirely within the set. Examples of convex sets include hyperplanes, half-spaces, balls, and intersections of convex sets.

{{< figure
    src="/images/maths/convexSet.png"
    caption="A set is convex if the line segment between any two points in the set remains entirely within the set."
    alt="Convex and non-convex sets"
>}}

A function $f: \mathbb{R}^d \to \mathbb{R}$ is then convex if its domain is a convex set and for any $x, y$ in the domain and any $\theta \in [0, 1]$:

$$
f(\theta x + (1 - \theta) y) \leq \theta f(x) + (1 - \theta) f(y)
$$

Geometrically, this means that the line segment connecting any two points on the graph of $f$ lies above the graph. Equivalently, a differentiable function is convex if and only if its Hessian matrix is positive semidefinite everywhere. In one dimension, this reduces to the second derivative being non-negative.

Some important properties of convex functions that are useful to remember and are rather intuitive include:
- The sum of convex functions is convex
- Multiplying a convex function by a positive scalar gives a convex function
- The maximum of convex functions is convex
- Linear and affine functions are both convex and concave.

{{< figure
    src="/images/maths/convexFunctions.png"
    caption="A function is convex if the line segment connecting any two points on its graph lies above the graph itself."
    alt="Convex and non-convex functions"
>}}

A constrained optimization problem is a **convex optimization** problem if:
1. The objective function $f$ is convex.
2. The equality constraint functions $g_1, \ldots, g_m$ are affine (of the form $g_i(w) = a_i^T w - b_i$).
3. The inequality constraint functions $h_1, \ldots, h_n$ are convex.
4. The set of feasible solutions is convex.

The crucial property of convex optimization problems is that **any local minimum is a global minimum**. This is because if we are at a local minimum and the function is convex, moving in any direction that maintains feasibility will not decrease the objective value.

{{< callout type="example" title="Drone Problem" >}}
To build intuition for constrained optimization, let us consider a concrete example that we will return to throughout these notes.

Imagine a drone hovering at a fixed position $(x_0, y_0, z_0)$ in three-dimensional space. A person is standing on a flat circular disk of radius $r$ centered at the origin. The disk lies in the plane $z = 0$. The person wants to get as close as possible to the drone but cannot leave the disk (otherwise they fall). We can formulate this as an optimization problem:

$$
\begin{align}
\min_{(x, y, z) \in \mathbb{R}^3} \quad & \|(x, y, z) - (x_0, y_0, z_0)\|^2 \\
\text{subject to} \quad & z = 0 \\
& x^2 + y^2 \leq r
\end{align}
$$

Let us define:
- $f(x, y, z) := \|(x, y, z) - (x_0, y_0, z_0)\|^2$ (squared Euclidean distance to the drone)
- $g(x, y, z) := z$ (equality constraint: must be on the disk plane)
- $h(x, y, z) := x^2 + y^2 - r$ (inequality constraint: must be within the disk radius)

This is a convex optimization problem because:
1. The objective $f$ is the squared Euclidean distance, which is convex
2. The equality constraint $g(x, y, z) = z$ is affine
3. The inequality constraint $h(x, y, z) = x^2 + y^2 - r$ is convex (it defines a disk, which is a convex set)

Before developing the general theory, let us compute the gradients of these functions as they will be central to our analysis:

$$
\begin{align}
\nabla f(x, y, z) &= 2(x - x_0, y - y_0, z - z_0) \\
\nabla g(x, y, z) &= (0, 0, 1) \\
\nabla h(x, y, z) &= (2x, 2y, 0)
\end{align}
$$

Notice that $\nabla g(x, y, z) = (0, 0, 1)$ is always perpendicular to the plane $z = 0$, which makes sense because the gradient points in the direction of steepest increase, and moving in the $z$ direction is the only way to increase $z$. Similarly, $\nabla h(x, y, z) = (2x, 2y, 0)$ points radially outward from the center of the disk and lies in the plane $z = 0$. When $(x, y, z)$ is on the boundary of the disk (where $x^2 + y^2 = r$), this gradient is perpendicular to the disk's edge.

{{< figure
    src="/images/maths/convexOptimizationDrone.jpg"
    caption="The drone problem: a person on a circular disk tries to get as close as possible to a hovering drone. The optimal solution depends on whether the drone is above the interior or edge of the disk."
    alt="Drone optimization problem visualization"
>}}
{{< /callout >}}

## Necessary Conditions for Optimality

How do we find the optimal solution to an optimization problem? In unconstrained optimization, we know that at a minimum, the gradient must be zero: $\nabla f(w^*) = 0$. This is because if the gradient were nonzero, we could move in the direction of $-\nabla f$ to decrease the function value.

For constrained optimization, the situation is more subtle. We cannot simply set the gradient to zero because we must also satisfy the constraints. The key insight is that at an optimum, the gradient of the objective must be a linear combination of the gradients of the active constraints.

Let us build intuition using the drone example. There are several cases to consider depending on where the drone is located:

**Case 1: The drone is on the disk**

If the drone position $(x_0, y_0, z_0)$ happens to lie on the disk (so $z_0 = 0$ and $x_0^2 + y_0^2 \leq r$), then the optimal solution is simply $w^* = (x_0, y_0, 0)$. The person can reach the drone exactly. At this point, we have $f(w^*) = 0$ and thus $\nabla f(w^*) = 0$. Neither constraint is "active" in the sense that small perturbations in any direction are feasible.

**Case 2: The drone is above the disk but not above the edge**

Suppose the drone has $z_0 \neq 0$ but $x_0^2 + y_0^2 < r$. The person cannot reach the drone because they must stay in the plane $z = 0$. The best they can do is move directly below the drone to position $(x_0, y_0, 0)$. At this optimal point, the equality constraint $z = 0$ is active but the inequality constraint is not (the person is strictly inside the disk).

At the optimum, we cannot move freely in any direction because we must maintain $z = 0$. However, we can move in any direction tangent to the plane. If the gradient $\nabla f(w^*)$ had a nonzero component in a tangent direction, we could decrease $f$ by moving that way. Therefore, $\nabla f(w^*)$ must be perpendicular to the plane, meaning it must be parallel to $\nabla g(w^*)$.

This gives us the necessary condition:

$$
\nabla f(w^*) + \lambda \nabla g(w^*) = 0
$$

for some scalar $\lambda \in \mathbb{R}$ called a **Lagrange multiplier**. The sign of $\lambda$ can be positive or negative because the constraint is an equality.

The Lagrange multiplier $\lambda$ has a powerful economic interpretation: it represents the **shadow price** or **marginal cost** of the constraint. Specifically, $\lambda$ tells us how much the optimal objective value would change if we slightly relaxed the constraint. If we perturbed the constraint from $g(w) = 0$ to $g(w) = \epsilon$ for a small $\epsilon$, the new optimal value would be approximately $f(w^*) - \lambda \epsilon$. A large $|\lambda|$ means the constraint is "expensive" in the sense that relaxing it would significantly improve the objective. A small $|\lambda|$ means the constraint barely affects the optimal value. If $\lambda = 0$, the constraint is not binding at all, and removing it would not change the solution.

**Case 3: The drone is above the edge of the disk**

Now suppose $x_0^2 + y_0^2 > r$. The optimal solution is on the boundary of the disk, on the line segment from the origin to the projection of the drone onto the plane. Both constraints are active: the person must be on the plane ($z = 0$) and on the edge of the disk ($x^2 + y^2 = r$).

At this point, the person can only move along the edge of the disk (a circle in the plane). If $\nabla f(w^*)$ had a nonzero component tangent to this circle, we could improve the solution. Therefore, $\nabla f(w^*)$ must lie in the span of $\nabla g(w^*)$ and $\nabla h(w^*)$, giving:

$$
\nabla f(w^*) + \lambda \nabla g(w^*) + \alpha \nabla h(w^*) = 0
$$

for some $\lambda, \alpha \in \mathbb{R}$. Here is the key insight: because $h$ is an inequality constraint, the multiplier $\alpha$ must be non-negative. To see why, note that $\nabla h(w^*)$ points outward from the disk (in the direction where $h$ increases). If we are at the boundary and $h(w^*) = 0$, then $-\nabla f(w^*)$ (the direction we want to go to decrease $f$) must either point inward or tangent to the boundary. If it pointed outward, we would not be at an optimum (we would want to leave the disk, but cannot). For $-\nabla f$ to be a combination of $-\lambda \nabla g$ and $-\alpha \nabla h$ where the component $-\alpha \nabla h$ points inward or is zero, we need $\alpha \geq 0$.

More generally, for inequality constraints, we have the **complementary slackness** condition. The name "complementary" refers to the fact that for each constraint, exactly one of two things must be true: either the constraint is inactive (has "slack") or the multiplier is nonzero. They cannot both be nonzero simultaneously, and they complement each other.

To understand this intuitively, consider the inequality constraint $h_j(w) \leq 0$:

- If $h_j(w^*) < 0$ (the constraint is **inactive** with slack), then we are strictly inside the feasible region with respect to this constraint. We could move slightly in the direction that increases $h_j$ without violating feasibility. Since we have this freedom and we are already at an optimum, this constraint is not limiting our solution. Therefore, it should not contribute to the gradient balance, so we must have $\alpha_j = 0$.

- If $h_j(w^*) = 0$ (the constraint is **active** or **binding**), then we are exactly on the boundary defined by this constraint. The constraint is limiting where we can go. To stay feasible, we cannot move in the direction where $h_j$ increases (that would violate the constraint). This constraint matters for the optimum, so it can have a nonzero multiplier $\alpha_j \geq 0$.

The compact mathematical formulation $\alpha_j h_j(w^*) = 0$ captures both cases: if $\alpha_j > 0$, then we must have $h_j(w^*) = 0$ (so the product is zero). If $h_j(w^*) < 0$, then we must have $\alpha_j = 0$ (so the product is zero). The product of the multiplier and the constraint value is always zero.

These necessary conditions can be unified into a single statement for general convex optimization problems.

## The Method of Lagrange Multipliers

The observations from the drone example generalize to arbitrary constrained optimization problems. The key idea, developed by Joseph-Louis Lagrange in the 1700s, is to convert a constrained optimization problem into an unconstrained one by incorporating the constraints into the objective function using additional variables called **Lagrange multipliers**. For a constrained optimization problem of the form given earlier, the function is called the **Lagrangian**:

$$
\mathcal{L}(\lambda, \alpha, w) = f(w) + \sum_{i=1}^m \lambda_i g_i(w) + \sum_{j=1}^n \alpha_j h_j(w)
$$

where $\lambda = (\lambda_1, \ldots, \lambda_m) \in \mathbb{R}^m$ are the Lagrange multipliers for the equality constraints and $\alpha = (\alpha_1, \ldots, \alpha_n) \in \mathbb{R}^n$ are the Lagrange multipliers for the inequality constraints.

The Lagrangian combines the objective function with weighted versions of all the constraints. The multipliers $\lambda_i$ and $\alpha_j$ tell us how much each constraint "costs" at the optimum. If a constraint has a large multiplier, satisfying that constraint significantly affects the optimal objective value.

### KKT Conditions

The following theorem, which combines the work of Karush, Kuhn, and Tucker, gives necessary conditions that any optimal solution must satisfy. These are often called the **KKT conditions** (Karush-Kuhn-Tucker conditions). The theorem is also often called the **First-Order Necessary Conditions** for optimality. Let $w^*$ be an optimal solution of a convex optimization problem. Then there exist multipliers $\lambda^* \in \mathbb{R}^m$ and $\alpha^* \in \mathbb{R}^n$ such that:

1. **Stationarity**: $\nabla_w \mathcal{L}(\lambda^*, \alpha^*, w^*) = 0$
2. **Primal feasibility**:
   - $g_i(w^*) = 0$ for $i \leq m$
   - $h_j(w^*) \leq 0$ for $j \leq n$
3. **Dual feasibility**: $\alpha_j^* \geq 0$ for $j \leq n$
4. **Complementary slackness**: $\alpha_j^* h_j(w^*) = 0$ for $j \leq n$

The stationarity condition says that the gradient of the Lagrangian with respect to $w$ vanishes at the optimum. Expanding this:

$$
\nabla_w \mathcal{L}(\lambda, \alpha, w) = \nabla f(w) + \sum_{i=1}^m \lambda_i \nabla g_i(w) + \sum_{j=1}^n \alpha_j \nabla h_j(w) = 0
$$

This is exactly the condition we derived geometrically for the drone example. The gradient of the objective must be a linear combination of the constraint gradients.

Primal feasibility simply restates that $w^*$ must satisfy all the original constraints. Dual feasibility requires that multipliers for inequality constraints be non-negative. Complementary slackness says that for each inequality constraint, either the constraint is active ($h_j(w^*) = 0$) or its multiplier is zero ($\alpha_j^* = 0$), or both.

{{< callout type="example" title="Applying KKT Conditions to the Drone Example" >}}
For the drone problem, the Lagrangian is:

$$
\mathcal{L}(\lambda, \alpha, x, y, z) = \|(x, y, z) - (x_0, y_0, z_0)\|^2 + \lambda z + \alpha (x^2 + y^2 - r)
$$

The KKT conditions are:

1. $\nabla_{x,y,z} \mathcal{L} = 0$:
   - $2(x - x_0) + 2\alpha x = 0$
   - $2(y - y_0) + 2\alpha y = 0$
   - $2(z - z_0) + \lambda = 0$

2. Primal feasibility: $z = 0$ and $x^2 + y^2 \leq r$

3. Dual feasibility: $\alpha \geq 0$

4. Complementary slackness: $\alpha(x^2 + y^2 - r) = 0$

From condition 1c and primal feasibility, we get $\lambda = 2z_0$. From 1a and 1b:
$$
x = \frac{x_0}{1 + \alpha}, \quad y = \frac{y_0}{1 + \alpha}
$$

If the optimum is strictly inside the disk ($x^2 + y^2 < r$), then by complementary slackness $\alpha = 0$, giving $(x, y) = (x_0, y_0)$. If the optimum is on the boundary ($x^2 + y^2 = r$), we solve:

$$
\frac{x_0^2 + y_0^2}{(1 + \alpha)^2} = r
$$

for $\alpha$, then compute $x$ and $y$.
{{< /callout >}}
{{< callout type="example" title="Minimizing Distance to a Line" >}}
Suppose we want to find the point on the line $x + y = 2$ that is closest to the origin. This is the optimization problem:

$$
\begin{align}
\min_{x, y} \quad & x^2 + y^2 \\
\text{subject to} \quad & x + y - 2 = 0
\end{align}
$$

The Lagrangian is:
$$
\mathcal{L}(\lambda, x, y) = x^2 + y^2 + \lambda(x + y - 2)
$$

Taking gradients:
$$
\begin{align}
\frac{\partial \mathcal{L}}{\partial x} &= 2x + \lambda = 0 \\
\frac{\partial \mathcal{L}}{\partial y} &= 2y + \lambda = 0 \\
\frac{\partial \mathcal{L}}{\partial \lambda} &= x + y - 2 = 0
\end{align}
$$

From the first two equations, $x = y = -\lambda/2$. Substituting into the constraint: $-\lambda/2 - \lambda/2 = 2$, so $\lambda = -2$. Thus $x = y = 1$ is the optimal solution, and the minimum distance squared is $1^2 + 1^2 = 2$.

Note that $\lambda = -2$ tells us how much the optimal objective value would change if we relaxed the constraint slightly. If we changed the constraint to $x + y = 2 + \epsilon$, the new optimal value would be approximately $2 - 2\epsilon$ (for small $\epsilon$).
{{< /callout >}}

## Lagrangian Duality

While the KKT conditions give us a system of equations to solve, in many cases this system is intractable. Duality theory provides an alternative approach: instead of solving the original the so called **primal problem** directly, we construct and solve a related **dual problem** which may be easier to solve. Under certain conditions, solving the dual gives us the solution to the primal or at least a lower bound on the optimal primal value.

In the primal problem, we were minimizing $f(w)$ subject to constraints. The dual problem involves maximizing a function derived from the Lagrangian over the multipliers $\lambda$ and $\alpha$. The key insight is to view the Lagrangian as a function of the multipliers $\lambda$ and $\alpha$ for a fixed $w$.

For any feasible point $w$, we have $g_i(w) = 0$ and $h_j(w) \leq 0$ by the definition of feasibility (feasible points must satisfy all constraints). If we also require $\alpha \geq 0$ (dual feasibility), then:

$$
\mathcal{L}(\lambda, \alpha, w) = f(w) + \sum_{i=1}^m \lambda_i g_i(w) + \sum_{j=1}^n \alpha_j h_j(w) = f(w) + \sum_{j=1}^n \alpha_j h_j(w) \leq f(w)
$$

The first sum vanishes because $g_i(w) = 0$, and the second sum is non-positive because $\alpha_j \geq 0$ and $h_j(w) \leq 0$. This means that for any feasible $w$ and any $\alpha \geq 0$, the Lagrangian provides a lower bound on the objective:

$$
\mathcal{L}(\lambda, \alpha, w) \leq f(w)
$$

Now, if we take the infimum of the Lagrangian over all $w$ (not just feasible $w$), we get an even smaller value:

$$
\inf_w \mathcal{L}(\lambda, \alpha, w) \leq \mathcal{L}(\lambda, \alpha, w) \leq f(w)
$$

This holds for any feasible $w$, so it holds for the optimal solution $w^*$:

$$
\inf_w \mathcal{L}(\lambda, \alpha, w) \leq f(w^*)
$$

This inequality holds for any $\lambda \in \mathbb{R}^m$ and any $\alpha \geq 0$. To get the best (largest) lower bound, we maximize the left-hand side over all choices of multipliers to get closer to the optimal primal value. So we define the dual function and dual problem as follows:

$$
\theta(\lambda, \alpha) := \inf_w \mathcal{L}(\lambda, \alpha, w) = \inf_w \left[ f(w) + \sum_{i=1}^m \lambda_i g_i(w) + \sum_{j=1}^n \alpha_j h_j(w) \right]
$$

An important property is that $\theta$ is always concave (even if the primal problem is not convex). This is because $\theta$ is the pointwise infimum of a family of affine functions in $(\lambda, \alpha)$, and the pointwise infimum of affine functions is concave.

To understand why, fix any $w$ and consider $\mathcal{L}(\lambda, \alpha, w)$ as a function of $(\lambda, \alpha)$. This is an affine function (linear plus a constant) in $(\lambda, \alpha)$ because the Lagrangian is linear in the multipliers:

$$
\mathcal{L}(\lambda, \alpha, w) = f(w) + \lambda^T g(w) + \alpha^T h(w)
$$

Affine functions are both convex and concave. Now, $\theta(\lambda, \alpha)$ takes the infimum (greatest lower bound) over all $w$ of these affine functions. Geometrically, if you plot several affine functions and take their pointwise minimum, the resulting envelope is concave (it curves downward). This is a general property: the pointwise minimum of any collection of convex functions is convex, and the pointwise maximum of concave functions is concave. Since we are taking the infimum (minimum) of concave functions (the affine functions viewed as functions of $(\lambda, \alpha)$), the result $\theta$ is concave. This is very helpful because maximizing a concave function is a convex optimization problem.

The **dual optimization problem** is then:

$$
\begin{align}
\max_{\lambda, \alpha} \quad & \theta(\lambda, \alpha) \\
\text{subject to} \quad & \alpha \geq 0
\end{align}
$$

or equivalently (since maximizing is the same as minimizing the negative):

$$
\begin{align}
\min_{\lambda, \alpha} \quad & -\theta(\lambda, \alpha) \\
\text{subject to} \quad & \alpha \geq 0
\end{align}
$$

Let $(\lambda^*, \alpha^*)$ be an optimal solution of the dual problem. From our derivation, we have:

$$
\theta(\lambda^*, \alpha^*) \leq f(w^*)
$$

This is called **weak duality**: the optimal dual value is always a lower bound on the optimal primal value. The difference $f(w^*) - \theta(\lambda^*, \alpha^*)$ is called the **duality gap**.

### Strong Duality and Slater's Condition

In some fortunate cases, the inequality becomes an equality. This is called **strong duality** rather than weak duality. A constrained optimization problem satisfies strong duality if:

$$
\theta(\lambda^*, \alpha^*) = f(w^*)
$$

where $w^*$ is an optimal solution of the primal problem and $(\lambda^*, \alpha^*)$ is an optimal solution of the dual problem.

Strong duality is powerful because it means we can solve the dual problem to find the optimal primal value. In many machine learning applications (notably Support Vector Machines), the dual problem is easier to solve than the primal and is a preferred approach to finding the optimal solution as it is a strong duality case. 

The question remains of when strong duality holds. For convex optimization problems, a simple sufficient condition is **Slater's condition** which results in the theorem that **Slater's Condition Implies Strong Duality**. A convex optimization problem satisfies Slater's condition if there exists a feasible point $w_0$ such that all inequality constraints are strictly satisfied:

$$
h_j(w_0) < 0, \quad \text{for all } j \leq n
$$

rather than just $h_j(w_0) \leq 0$ and the equality constraints are satisfied as usual: $g_i(w_0) = 0$ for all $i \leq m$. If such a point exists, we say that the problem has a **strictly feasible point**. Slater's condition essentially says that the feasible region has a nonempty interior (with respect to the inequality constraints). This rules out pathological cases where the feasible region is too "thin."

{{< callout type="example" title="Checking Slater's Condition for the Drone Problem" >}}
For the drone problem, the constraints are $z = 0$ and $x^2 + y^2 - r \leq 0$. To verify Slater's condition, we need a point where $z = 0$ (equality) and $x^2 + y^2 - r < 0$ (inequality strict).

Any point of the form $(x, y, 0)$ with $x^2 + y^2 < r$ satisfies this. For example, $(0, 0, 0)$ works (the center of the disk). Therefore, Slater's condition holds, and we have strong duality.
{{< /callout >}}

### Complementary Slackness Revisited

When strong duality holds, we can derive additional useful properties. The main one is a more formal statement of the complementary slackness condition we discussed earlier. If strong duality holds and $w^*$ is an optimal solution to the primal problem with corresponding dual optimal solution $(\lambda^*, \alpha^*)$, then:

$$
\alpha_j^* h_j(w^*) = 0, \quad \text{for all } j \leq n
$$

this is the complementary slackness condition mentioned before, but now we have a formal proof under strong duality. Complementary slackness is extremely useful in practice. It tells us that at the optimal solution, for each inequality constraint, either:
- The constraint is inactive ($h_j(w^*) < 0$) and the multiplier is zero ($\alpha_j^* = 0$), or
- The constraint is active ($h_j(w^*) = 0$) and the multiplier may be nonzero ($\alpha_j^* \geq 0$)

This dramatically reduces the number of cases we need to consider when solving optimization problems.

{{< callout type="proof" title="Proof of Complementary Slackness under Strong Duality" >}}
$$
f(w^*) = \theta(\lambda^*, \alpha^*) = \inf_w \mathcal{L}(\lambda^*, \alpha^*, w) \leq \mathcal{L}(\lambda^*, \alpha^*, w^*) \leq f(w^*)
$$

The first equality is strong duality. The first inequality holds because $w^*$ is one of the choices in the infimum. The second inequality we derived earlier (for feasible $w^*$ and dual feasible $\alpha^*$, the Lagrangian is at most $f(w^*)$).

Since both inequalities must be equalities, we have:

$$
f(w^*) = \mathcal{L}(\lambda^*, \alpha^*, w^*)
$$

Expanding the Lagrangian:

$$
f(w^*) = f(w^*) + \sum_{i=1}^m \lambda_i^* g_i(w^*) + \sum_{j=1}^n \alpha_j^* h_j(w^*)
$$

The terms $g_i(w^*) = 0$ because $w^*$ is feasible. This gives:

$$
0 = \sum_{j=1}^n \alpha_j^* h_j(w^*)
$$

Since $\alpha_j^* \geq 0$ and $h_j(w^*) \leq 0$, each term $\alpha_j^* h_j(w^*)$ is non-positive. The only way a sum of non-positive terms can equal zero is if each term is zero:

$$
\alpha_j^* h_j(w^*) = 0, \quad \text{for all } j
$$
{{< /callout >}}

## Solving Convex Optimization Problems

We now have a complete toolkit for solving convex optimization problems. If the problem is simple enough, we can directly solve the KKT conditions:

1. Write down the Lagrangian $\mathcal{L}(\lambda, \alpha, w)$
2. Compute $\nabla_w \mathcal{L}(\lambda, \alpha, w) = 0$ (stationarity)
3. Include the primal feasibility constraints: $g_i(w) = 0$ and $h_j(w) \leq 0$
4. Include dual feasibility: $\alpha_j \geq 0$
5. Include complementary slackness: $\alpha_j h_j(w) = 0$

If Slater's condition holds, any solution to this system is optimal for the primal problem.

The complementary slackness conditions often allow us to split into cases. For each inequality constraint $j$, either $\alpha_j = 0$ (constraint inactive) or $h_j(w) = 0$ (constraint active). If there are $n$ inequality constraints, there are at most $2^n$ cases to check.

{{< callout type="example" title="A One-Dimensional Problem" >}}
Consider the problem:

$$
\begin{align}
\min_x \quad & (x - 3)^2 \\
\text{subject to} \quad & x \leq 1
\end{align}
$$

The Lagrangian is $\mathcal{L}(\alpha, x) = (x - 3)^2 + \alpha(x - 1)$.

The KKT conditions are:
1. Stationarity: $2(x - 3) + \alpha = 0$
2. Primal feasibility: $x \leq 1$
3. Dual feasibility: $\alpha \geq 0$
4. Complementary slackness: $\alpha(x - 1) = 0$

**Case 1**: $\alpha = 0$ (constraint inactive). Then $x = 3$ from stationarity. But this violates primal feasibility ($3 > 1$), so this case is infeasible.

**Case 2**: $x = 1$ (constraint active). Then from stationarity: $2(1 - 3) + \alpha = 0$, so $\alpha = 4 > 0$, satisfying dual feasibility.

Therefore, $x^* = 1$ is the optimal solution with multiplier $\alpha^* = 4$.
{{< /callout >}}

When solving the KKT conditions directly is intractable, we can try solving the dual problem:

1. Derive the dual function $\theta(\lambda, \alpha) = \inf_w \mathcal{L}(\lambda, \alpha, w)$. This often involves solving $\nabla_w \mathcal{L}(\lambda, \alpha, w) = 0$ for $w$ as a function of $\lambda$ and $\alpha$, then substituting back.
2. Solve the dual optimization problem: maximize $\theta(\lambda, \alpha)$ subject to $\alpha \geq 0$.
3. If Slater's condition holds (giving strong duality), use the optimal dual solution $(\lambda^*, \alpha^*)$ to recover the primal solution. Often, the primal solution $w^*$ is the argmin in the definition of $\theta(\lambda^*, \alpha^*)$.
4. Verify complementary slackness if needed.

The dual problem is often easier to solve because:
- It has a simpler constraint set (just $\alpha \geq 0$)
- The dual function is always concave, so we are maximizing a concave function

## Connection to Machine Learning

Convex optimization is ubiquitous in machine learning. Many learning algorithms can be formulated as convex optimization problems, which guarantees we can find the global optimum efficiently.

### Ridge Regression as Constrained Optimization

Recall from Bayesian linear regression that Ridge regression minimizes:

$$
\min_{\mathbf{w}} \|\mathbf{y} - \mathbf{X}\mathbf{w}\|^2 + \lambda \|\mathbf{w}\|^2
$$

where $\mathbf{X} \in \mathbb{R}^{n \times d}$ is the design matrix, $\mathbf{y} \in \mathbb{R}^n$ is the target vector, and $\mathbf{w} \in \mathbb{R}^d$ is the weight vector. This can be reformulated as a constrained optimization problem:

$$
\begin{align}
\min_{\mathbf{w}} \quad & \|\mathbf{y} - \mathbf{X}\mathbf{w}\|^2 \\
\text{subject to} \quad & \|\mathbf{w}\|^2 \leq t
\end{align}
$$

for some $t > 0$. So rather than including the regularization term in the objective, we constrain the norm of the weights by some threshold $t$. The Lagrangian is:

$$
\mathcal{L}(\alpha, \mathbf{w}) = \|\mathbf{y} - \mathbf{X}\mathbf{w}\|^2 + \alpha(\|\mathbf{w}\|^2 - t)
$$

Setting $\nabla_{\mathbf{w}} \mathcal{L} = 0$ gives:

$$
-2\mathbf{X}^T(\mathbf{y} - \mathbf{X}\mathbf{w}) + 2\alpha \mathbf{w} = 0 \implies \mathbf{w} = (\mathbf{X}^T \mathbf{X} + \alpha \mathbf{I})^{-1} \mathbf{X}^T \mathbf{y}
$$

This is exactly the Ridge regression solution with regularization parameter $\lambda = \alpha$. The Lagrange multiplier $\alpha$ controls the trade-off between fitting the data and keeping the weights small.
