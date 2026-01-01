---
title: Linear Regression
type: docs
weight: 6
---


Linear regression is one of the most fundamental algorithms in machine learning. It is a linear approach to modeling the relationship between a scalar response (or dependent variable) and one or more explanatory variables (or independent variables). In particular, linear regression attempts to model the relationship by fitting a linear equation to observed data. So we assume that the target value $y$ is a linear combination of the input features $\mathbf{x}$ plus some noise. For a single data point $(\mathbf{x}, y)$, the model is given by:

$$
y = \mathbf{w}^T\mathbf{x} + \epsilon
$$

where:
- $\mathbf{x} \in \mathbb{R}^D$ is the input feature vector (often with a bias term $x_0 = 1$).
- $\mathbf{w} \in \mathbb{R}^D$ is the weight vector (parameters) we want to learn.
- $\epsilon$ is the error term (noise), typically assumed to be Gaussian distributed $\epsilon \sim \mathcal{N}(0, \sigma^2)$.

For a dataset of $N$ samples, we can write this in matrix notation:

$$
\mathbf{y} = \mathbf{X}\mathbf{w} + \boldsymbol{\epsilon}
$$

where:
- $\mathbf{y} \in \mathbb{R}^N$ is the vector of target values.
- $\mathbf{X} \in \mathbb{R}^{N \times D}$ is the design matrix, where the $i$-th row is $\mathbf{x}_i^T$.
- $\boldsymbol{\epsilon} \in \mathbb{R}^N$ is the noise vector.

## Ordinary Least Squares (OLS)

Now that we have defined the model structure, we need a way to estimate the parameters $\mathbf{w}$. The most common method is **Ordinary Least Squares (OLS)**. The intuition is simple: we want to find the line (or hyperplane) that minimizes the discrepancy between the actual target values and the values predicted by our model.

Suppose we are tasked with predicting house prices based on the size of the house. We are given a set of data points $(x_1, y_1), (x_2, y_2), \ldots, (x_N, y_N)$, where:
- $x_i$: Is the measured size of the house (independent variable, or feature) for the $i$-th data point.
- $y_i$: the actual observed price of the house (dependent variable, or outcome) for the $i$-th data point.

{{< figure
  src="/images/maths/leastSquares.png"
  alt="On the x-axis we have some feature such as the size of the house and on the y-axis we have the price of the house. The dots represent the observed data points and the line is the best-fitting line that we are trying to find."
  caption="On the x-axis we have some feature such as the size of the house and on the y-axis we have the price of the house. The dots represent the observed data points and the line is the best-fitting line that we are trying to find."
>}}

If we then assume that there is a linear relationship between the size of the house and its price. So the price of house is dependent on the size of the house. The linear part means that the relationship can be expressed as a line. This is why the problem is also often referred to as "fitting a line to the data". 

So there is some linear function that given the size of the house can predict the price of the house. We aim to find this function. We know that a linear function is defined as follows:

$$
y = w_0 + w_1 x
$$

Where $x$ is our input feature, $y$ is the output and $w_0$ and $w_1$ are the parameters of the function. In our case $x$ is the size of the house and $y$ is the price of the house. So we want to find the parameters $w_0$ and $w_1$ such that the predicted price $\hat{y}_i$ is given by:

$$
\hat{y}_i = w_0 + w_1 x_i,
$$

where the predicted price $\hat{y}_i$ is as close as possible to the observed price $y_i$. The difference between the observed price $y_i$ and the predicted price $\hat{y}_i$ is called the **error** for the $i$-th observation. To find the best-fitting line, we minimize the **sum of squared errors** across all observations:

$$
\min_{w_0, w_1} \sum_{i=1}^N \left(y_i - (w_0 + w_1 x_i)\right)^2.
$$

This least squares formulation has numerous practical applications for modeling any linear relationship between variables. As we are doing linear algebra let us represent the data in terms of matrices and vectors. So we define the following:
- The vector containing the observed outcomes (house prices) is $\mathbf{y} = \begin{bmatrix} y_1 \\ y_2 \\ \vdots \\ y_N \end{bmatrix}$.
- The vector of coefficients (parameters to be determined) is $\mathbf{w} = \begin{bmatrix} w_0 \\ w_1 \end{bmatrix}$.
- The matrix containing the features is $\mathbf{X} = \begin{bmatrix} 1 & x_1 \\ 1 & x_2 \\ \vdots & \vdots \\ 1 & x_N \end{bmatrix}$. The reason why we concatenate a column of ones to the start of the feature matrix for $\mathbf{X}$ is that we want to include the intercept term $w_0$ in our linear model. This allows us to express the linear function in matrix form, where the first column corresponds to the intercept and the second column corresponds to the feature values $x_i$. So we want to just add the term $w_0$ to the term $w_1 x_i$. 

Our predicted outcomes for all inputs are then given by:

$$
\hat{\mathbf{y}} = \mathbf{X}\mathbf{w}.
$$

The error vector, representing the difference between observed outcomes and predicted outcomes can then be written as:

$$
\mathbf{e} = \mathbf{y} - \hat{\mathbf{y}} = \mathbf{y} - \mathbf{X}\mathbf{w}.
$$

The least squares problem is then to find $\mathbf{w}$ that minimizes the squared norm of the error vector:

$$
\min_{\mathbf{w} \in \mathbb{R}^2} \|\mathbf{y} - \mathbf{X}\mathbf{w}\|^2.
$$

Hence the name **least squares**. The goal is to find the coefficients $\mathbf{w}$ that minimize the squared error between the observed prices and the predicted prices based on the linear model. This matrix formulation generalizes easily to cases with multiple features. For example, if we also have the **age of the house** as an additional feature, the matrix $\mathbf{X}$ would include an additional column corresponding to this feature:

$$
\mathbf{X} = \begin{bmatrix} 1 & x_1 & s_1 \\ 1 & x_2 & s_2 \\ \vdots & \vdots & \vdots \\ 1 & x_N & s_N \end{bmatrix},
$$

where $s_i$ is for example the age of the $i$-th house. The least squares problem remains the same, but now we have more coefficients to determine, and the vector $\mathbf{w}$ would be:

$$
\mathbf{w} = \begin{bmatrix} w_0 \\ w_1 \\ w_2 \end{bmatrix},
$$

where $w_2$ is the coefficient for the age of the house. More generally, for a given matrix $\mathbf{X} \in \mathbb{R}^{N \times D}$ (where $N$ is the number of observations and $D$ is the number of features plus one for the intercept) and a vector $\mathbf{y} \in \mathbb{R}^N$, we aim to find the vector $\mathbf{w} \in \mathbb{R}^D$ that minimizes:

$$
\min_{\mathbf{w} \in \mathbb{R}^D} \|\mathbf{y} - \mathbf{X}\mathbf{w}\|^2.
$$

From the above formulation we can see that the when we are trying to minimize the squared norm of the error we are actually looking for the projection of $\mathbf{y}$ onto the column space of $\mathbf{X}$:

$$
\min_{\mathbf{w} \in \mathbb{R}^D} \|\mathbf{y} - \mathbf{X}\mathbf{w}\|^2 = \|\mathbf{y} - \text{proj}_{C(\mathbf{X})}(\mathbf{y})\|^2.
$$

This actually makes sense as we are given some vector and we are trying to create it as a linear combination of the columns of $\mathbf{X}$. Because we have now noticed that we are looking for the projection of $\mathbf{y}$ onto the column space of $\mathbf{X}$ we can use our knowledge of projections to solve the least squares problem. We know that the solution to the least squares problem is derived using the **normal equations**. By setting the gradient of the squared error to zero, we obtain:

$$
\mathbf{X}^T\mathbf{X}\mathbf{w} = \mathbf{X}^T\mathbf{y}.
$$

If we then assume that the columns of $\mathbf{X}$ are linearly independent, we know that $\mathbf{X}^T\mathbf{X}$ is invertible and that we can solve for $\mathbf{w}$:

$$
\mathbf{w} = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}.
$$

To solve the least squares problem, the columns of $\mathbf{X}$ must be **linearly independent**. Linear independence ensures that $\mathbf{X}^T\mathbf{X}$ is invertible, which is crucial for deriving the solution. For example, if all the feature values $x_i$ are identical (e.g., $x_i = x_j$ for all $i \neq j$), then the column would be a multiple of the first column and the columns of $\mathbf{X}$ would not span a sufficiently large space, making it impossible to uniquely determine the coefficients $\mathbf{w}$. Linear independence of the columns of $\mathbf{X}$ ensures that the data provides enough information to determine a unique solution. If the columns are not linearly independent (or close to it, a condition known as multicollinearity), $\mathbf{X}^T\mathbf{X}$ becomes singular or ill-conditioned. This is a key motivation for **regularization**, where we add a term (like $\lambda \mathbf{I}$) to make the matrix invertible.

If we look more closely at the what the matrix multiplication is doing we can actually derive an explicit formula for the coefficients $\mathbf{w}$. First let's remember what the matrix multiplication of $\mathbf{X}^T\mathbf{X}$ looks like:

$$
\begin{align*}
\mathbf{X}^T\mathbf{X} &= \begin{bmatrix}
x_{11} & x_{12} & \cdots & x_{1N} \\
x_{21} & x_{22} & \cdots & x_{2N} \\
\vdots & \vdots & \ddots & \vdots \\
x_{D1} & x_{D2} & \cdots & x_{DN}
\end{bmatrix} 
\begin{bmatrix}
x_{11} & x_{21} & \cdots & x_{D1} \\
x_{12} & x_{22} & \cdots & x_{D2}
\end{bmatrix} \\ 
&= \begin{bmatrix}
(x_{11}^2 + x_{12}^2 + \cdots + x_{1N}^2) & (x_{11}x_{21} + x_{12}x_{22} + \cdots + x_{1N}x_{2N}) & \cdots & (x_{11}x_{D1} + x_{12}x_{D2} + \cdots + x_{1N}x_{DN}) \\
(x_{21}x_{11} + x_{22}x_{12} + \cdots + x_{2N}x_{1N}) & (x_{21}^2 + x_{22}^2 + \cdots + x_{2N}^2) & \cdots & (x_{21}x_{D1} + x_{22}x_{D2} + \cdots + x_{2N}x_{DN}) \\
\vdots & \vdots & \ddots & \vdots \\
(x_{D1}x_{11} + x_{D2}x_{12} + \cdots + x_{DN}x_{1N}) & (x_{D1}x_{21} + x_{D2}x_{22} + \cdots + x_{DN}x_{2N}) & \cdots & (x_{D1}x_{D1} + x_{D2}x_{D2} + \cdots + x_{DN}x_{DN})
\end{bmatrix}
\end{align*}
$$

So we remember that the diagonal elements of $\mathbf{X}^T\mathbf{X}$ are the sums of squares of the columns of $\mathbf{X}$, and the off-diagonal elements are the sums of products of different columns. More specifically for the case where we have one feature (the size of the house) where $\mathbf{X} \in \mathbb{R}^{N \times 2}$, we have that the first column of $\mathbf{X}$ is all ones and the second column is the feature values $x_i$. So the first diagonal element of $\mathbf{X}^T\mathbf{X}$ is the sum of squares of the first column, which is simply $N$ (the number of observations), and the second diagonal element is the sum of squares of the feature values. The off diagonal element is the sum of the feature values multiplied by the ones, which is simply the sum of the feature values. 

If we then just multiply the matrix $\mathbf{X}^T$ with the vector $\mathbf{y}$, representing the observed prices, we get just a linear combination of the observed prices and the feature values. Specifically, the first element is the sum of the observed prices and the second element is the sum of the observed prices multiplied by the feature values. So we can write:

$$
\begin{bmatrix} w_0 \\ w_1 \end{bmatrix} = 
\begin{bmatrix}
N & \sum_{i=1}^{N} x_i \\
\sum_{i=1}^{N} x_i & \sum_{i=1}^{N} x_i^2
\end{bmatrix}^{-1}
\begin{bmatrix}
\sum_{i=1}^{N} y_i \\
\sum_{i=1}^{N} y_i x_i
\end{bmatrix}.
$$

Where $N$ is the number of observations (houses), $\sum_{i=1}^{N} x_i$ is the sum of the feature values (sizes of houses) and $\sum_{i=1}^{N} y_i x_i$ is the sum of the product of observed prices and feature values which together give us the coefficient $w_0$ and $w_1$ for the intercept and the slope of the line respectively.

If the columns of $\mathbf{X}$ are pairwise orthogonal, then the matrix $\mathbf{X}^T\mathbf{X}$ is diagonal as each element is the dot product of two columns which is zero for different columns and the sum of squares for the same column. If $\mathbf{X} \in \mathbb{R}^{N \times 2}$, then the columns are orthogonal if the feature values $x_i$ are such that $\sum_{i=1}^{N} x_i = 0$. This is because when we calculate the dot product of the first column (all ones) and the second column (the feature values), we get:

$$
\sum_{i=1}^{N} 1 \cdot x_i = \sum_{i=1}^{N} x_i.
$$

So if they are orthogonal we can simplify the matrix $\mathbf{X}^T\mathbf{X}$ to:

$$
\begin{bmatrix} w_0 \\ w_1 \end{bmatrix} =
\begin{bmatrix}
N & 0 \\
0 & \sum_{i=1}^{N} x_i^2
\end{bmatrix}^{-1}
\begin{bmatrix}
\sum_{i=1}^{N} y_i \\
\sum_{i=1}^{N} y_i x_i
\end{bmatrix}
$$

And because the inverse of a diagonal matrix is simply the reciprocal of the diagonal elements, we can write:

$$
\begin{bmatrix} w_0 \\ w_1 \end{bmatrix} =
\begin{bmatrix}\frac{1}{N} & 0 \\
0 & \frac{1}{\sum_{i=1}^{N} x_i^2}
\end{bmatrix}
\begin{bmatrix}\sum_{i=1}^{N} y_i \\
\sum_{i=1}^{N} y_i x_i
\end{bmatrix} = 
\begin{bmatrix}\frac{1}{N} \sum_{i=1}^{N} y_i \\
\frac{1}{\sum_{i=1}^{N} x_i^2} \sum_{i=1}^{N} y_i x_i
\end{bmatrix}
$$

{{< callout type="example" title="One Feature Case" >}}
Suppose we have the following data points representing the size of houses and their prices:

$$
\mathbf{X} = \begin{bmatrix} 1 & 50 \\ 1 & 80 \\ 1 & 100 \end{bmatrix}, \quad \mathbf{y} = \begin{bmatrix} 200 \\ 300 \\ 400 \end{bmatrix}
$$

To then find the coefficients $\mathbf{w} = \begin{bmatrix} w_0 \\ w_1 \end{bmatrix}$, we first calculate $\mathbf{X}^T\mathbf{X}$:

$$
\mathbf{X}^T\mathbf{X} = \begin{bmatrix} 1 & 1 & 1 \\ 50 & 80 & 100 \end{bmatrix} \begin{bmatrix} 1 & 50 \\ 1 & 80 \\ 1 & 100 \end{bmatrix} = \begin{bmatrix} 3 & 230 \\ 230 & 18900 \end{bmatrix}
$$

Then we calculate $\mathbf{X}^T\mathbf{y}$:

$$
\mathbf{X}^T\mathbf{y} = \begin{bmatrix} 1 & 1 & 1 \\ 50 & 80 & 100 \end{bmatrix} \begin{bmatrix} 200 \\ 300 \\ 400 \end{bmatrix} = \begin{bmatrix} 900 \\ 74000 \end{bmatrix}
$$

We can then either calculate the inverse of $\mathbf{X}^T\mathbf{X}$ and multiply it with $\mathbf{X}^T\mathbf{y}$ to calculate:

$$
\mathbf{w} = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y} = \begin{bmatrix} 3 & 230 \\ 230 & 18900 \end{bmatrix}^{-1} \begin{bmatrix} 900 \\ 74000 \end{bmatrix}
$$

Or we can solve the normal equations directly so finding the solution to the least squares problem:

$$
\mathbf{X}^T\mathbf{X}\mathbf{w} = \mathbf{X}^T\mathbf{y}
$$

So we can write the system of equations as:

$$
\begin{bmatrix} 3 & 230 \\ 230 & 18900 \end{bmatrix} \begin{bmatrix} w_0 \\ w_1 \end{bmatrix} = \begin{bmatrix} 900 \\ 74000 \end{bmatrix}
$$

solving this system gives us the coefficients $w_0=-\frac{50}{19}$ and $w_1=\frac{75}{19}$. So the best-fitting line is:

$$
\hat{y}_i = -\frac{50}{19} + \frac{75}{19} x_i
$$
{{< /callout >}}

While OLS gives us the optimal parameters for the training data, this is not the end of the story. A model that fits the training data perfectly might fail miserably when predicting on new data. This brings us to the core challenge of machine learning.

## Generalization, Overfitting, and Underfitting

The ultimate goal of machine learning is **generalization**: the ability of a model to perform well on unseen data. We typically train our model on a **training set**, but we care about its performance on a held-out **test set**.

*   **Underfitting**: Occurs when the model is too simple to capture the underlying structure of the data. For example, trying to fit a straight line to data that is clearly quadratic. The model has high **bias**.
*   **Overfitting**: Occurs when the model is too complex and learns the noise in the training data as if it were signal. The model fits the training data perfectly but fails to generalize to new data. The model has high **variance**.

{{< figure src="/images/ml/overfitting_underfitting.png" caption="Illustration of Underfitting (High Bias), Optimal Fit, and Overfitting (High Variance)." >}}

## Bias-Variance Tradeoff

The relationship between model complexity and error is quantified by the **Bias-Variance Decomposition**. The expected test error can be decomposed into three terms: **Bias**, **Variance**, and **Irreducible Error**. 

Let's assume the true relationship is $y = f(\mathbf{x}) + \epsilon$, where $E[\epsilon] = 0$ and $Var(\epsilon) = \sigma^2$. We estimate $f(\mathbf{x})$ with our model $\hat{f}(\mathbf{x})$. We want to analyze the expected squared error at a query point $\mathbf{x}$:

$$
E[(y - \hat{f}(\mathbf{x}))^2]
$$

We can expand the term inside the expectation. Note that $y = f(\mathbf{x}) + \epsilon$ is a random variable due to $\epsilon$, and $\hat{f}(\mathbf{x})$ is a random variable because it depends on the specific training set $\mathcal{D}$ used to learn the weights.

$$
\begin{align*}
E[(y - \hat{f}(\mathbf{x}))^2] &= E[(f(\mathbf{x}) + \epsilon - \hat{f}(\mathbf{x}))^2] \\
&= E[( (f(\mathbf{x}) - \hat{f}(\mathbf{x})) + \epsilon )^2] \\
&= E[(f(\mathbf{x}) - \hat{f}(\mathbf{x}))^2] + E[\epsilon^2] + 2E[(f(\mathbf{x}) - \hat{f}(\mathbf{x}))\epsilon]
\end{align*}
$$

Since $\epsilon$ is independent of $\hat{f}(\mathbf{x})$ and has mean 0, the cross term vanishes: $E[(f(\mathbf{x}) - \hat{f}(\mathbf{x}))\epsilon] = E[f(\mathbf{x}) - \hat{f}(\mathbf{x})]E[\epsilon] = 0$. Also, $E[\epsilon^2] = Var(\epsilon) + (E[\epsilon])^2 = \sigma^2$.

Now we focus on the first term $E[(f(\mathbf{x}) - \hat{f}(\mathbf{x}))^2]$. Let's add and subtract $E[\hat{f}(\mathbf{x})]$ (the average prediction of our model over many training sets):

$$
\begin{align*}
E[(f(\mathbf{x}) - \hat{f}(\mathbf{x}))^2] &= E[(f(\mathbf{x}) - E[\hat{f}(\mathbf{x})] + E[\hat{f}(\mathbf{x})] - \hat{f}(\mathbf{x}))^2] \\
&= E[(f(\mathbf{x}) - E[\hat{f}(\mathbf{x})])^2] + E[(E[\hat{f}(\mathbf{x})] - \hat{f}(\mathbf{x}))^2] \\
&\quad + 2E[(f(\mathbf{x}) - E[\hat{f}(\mathbf{x})])(E[\hat{f}(\mathbf{x})] - \hat{f}(\mathbf{x}))]
\end{align*}
$$

The cross term again vanishes because $f(\mathbf{x}) - E[\hat{f}(\mathbf{x})]$ is constant with respect to the expectation over training sets, and $E[E[\hat{f}(\mathbf{x})] - \hat{f}(\mathbf{x})] = E[\hat{f}(\mathbf{x})] - E[\hat{f}(\mathbf{x})] = 0$.

So we are left with:

$$
E[(y - \hat{f}(\mathbf{x}))^2] = \underbrace{(f(\mathbf{x}) - E[\hat{f}(\mathbf{x})])^2}_{\text{Bias}^2} + \underbrace{E[(\hat{f}(\mathbf{x}) - E[\hat{f}(\mathbf{x})])^2]}_{\text{Variance}} + \underbrace{\sigma^2}_{\text{Irreducible Error}}
$$

We can interpret the three terms as follows:

*   **Bias**: The difference between the average prediction of our model and the correct value. High bias means the model is too simple to capture the underlying structure of the data (underfitting).
*   **Variance**: The variability of the model prediction for a given data point across different realizations of the training set. High variance means the model is too sensitive to the noise in the training data (overfitting).
*   **Irreducible Error**: The noise inherent in the problem itself. We cannot reduce this term.

As we increase model complexity (e.g., adding more features, using higher degree polynomials), bias typically decreases, but variance increases. The goal is to find the sweet spot that minimizes the total error.

### Cross-Validation

To estimate the test error and find the optimal model complexity (or hyperparameters), we use **Cross-Validation**. The most common method is **$k$-fold Cross-Validation**:

1.  Split the training data into $k$ equal-sized folds.
2.  For each fold $i \in \{1, \dots, k\}$:
    *   Train the model on the other $k-1$ folds.
    *   Validate (test) the model on fold $i$.
3.  Average the validation errors to get an estimate of the test error.

This allows us to tune hyperparameters (like the regularization strength $\lambda$ in Ridge/Lasso) without touching the final test set, preventing data leakage.

## Regularization

We established that high variance (overfitting) is a major issue, often caused by the model being too complex or the data being insufficient. In Linear Regression, this often manifests when features are highly correlated (**multicollinearity**), causing the matrix $\mathbf{X}^T\mathbf{X}$ to be close to singular (non-invertible), making the OLS solution unstable. Small changes in the data can lead to large changes in the weights $\mathbf{w}$, which is a symptom of high variance (overfitting).

To combat this, we can add a regularization term to the loss function that penalizes large weights.

### Ridge Regression ($L_2$ Regularization)

Ridge regression adds a penalty proportional to the square of the magnitude of the coefficients.

$$
\mathcal{L}_{Ridge}(\mathbf{w}) = ||\mathbf{y} - \mathbf{X}\mathbf{w}||^2 + \lambda ||\mathbf{w}||_2^2
$$

where $\lambda \geq 0$ is a hyperparameter controlling the strength of regularization.

**Closed Form Solution:**
We can derive the solution by taking the gradient and setting it to zero:

$$
\begin{align*}
\nabla_{\mathbf{w}} \mathcal{L}_{Ridge} &= -2\mathbf{X}^T(\mathbf{y} - \mathbf{X}\mathbf{w}) + 2\lambda\mathbf{w} = 0 \\
\mathbf{X}^T\mathbf{X}\mathbf{w} + \lambda\mathbf{I}\mathbf{w} &= \mathbf{X}^T\mathbf{y} \\
(\mathbf{X}^T\mathbf{X} + \lambda\mathbf{I})\mathbf{w} &= \mathbf{X}^T\mathbf{y}
\end{align*}
$$

$$
\hat{\mathbf{w}}_{Ridge} = (\mathbf{X}^T\mathbf{X} + \lambda\mathbf{I})^{-1}\mathbf{X}^T\mathbf{y}
$$

Adding $\lambda\mathbf{I}$ to $\mathbf{X}^T\mathbf{X}$ ensures that the matrix is always invertible (it adds $\lambda$ to the eigenvalues), solving the multicollinearity problem and reducing variance at the cost of introducing some bias.

**Interpretation:**
The term $\lambda ||\mathbf{w}||_2^2$ penalizes large weights. By forcing weights to be small, we prevent the model from fitting the noise in the training data (overfitting). This directly relates to the **Bias-Variance Tradeoff**:
*   **Low $\lambda$**: The model behaves like OLS. Low bias, high variance (prone to overfitting).
*   **High $\lambda$**: The weights are shrunk towards zero. High bias (underfitting), low variance.

**Bias and Variance of Ridge Estimator:**
The Ridge estimator is $\hat{\mathbf{w}}_{Ridge} = (\mathbf{X}^T\mathbf{X} + \lambda\mathbf{I})^{-1}\mathbf{X}^T\mathbf{y}$.
It can be shown that:
*   **Bias**: $E[\hat{\mathbf{w}}_{Ridge}] - \mathbf{w} = - \lambda (\mathbf{X}^T\mathbf{X} + \lambda\mathbf{I})^{-1} \mathbf{w} \neq 0$. Ridge is a **biased estimator**.
*   **Variance**: $Var(\hat{\mathbf{w}}_{Ridge}) = \sigma^2 (\mathbf{X}^T\mathbf{X} + \lambda\mathbf{I})^{-1} \mathbf{X}^T\mathbf{X} (\mathbf{X}^T\mathbf{X} + \lambda\mathbf{I})^{-1}$.
As $\lambda$ increases, the bias magnitude increases, but the variance decreases (the matrix inverse terms become smaller).

### Lasso Regression ($L_1$ Regularization)

A limitation of Ridge Regression is that it shrinks all weights towards zero but rarely sets them *exactly* to zero. This means the final model still includes all input features, which can be a problem for **interpretability** if we have thousands of features.

**Lasso** (Least Absolute Shrinkage and Selection Operator) addresses this by adding a penalty proportional to the absolute value of the coefficients. This encourages **sparsity**, meaning many weights become exactly zero. This effectively performs **feature selection**, giving us a simpler, more interpretable model that only relies on the most important features.

$$
\mathcal{L}_{Lasso}(\mathbf{w}) = ||\mathbf{y} - \mathbf{X}\mathbf{w}||^2 + \lambda ||\mathbf{w}||_1
$$

**Sparsity:**
Unlike Ridge, Lasso does not have a closed-form solution (due to the non-differentiability of the absolute value at 0). However, it has a very useful property: it promotes **sparsity**. It tends to force the coefficients of less important features to be exactly zero. This effectively performs feature selection.

**Solving Lasso:**
Since the $L_1$ term is not differentiable at zero, we cannot use standard gradient descent or closed-form solutions. Instead, we often use **Coordinate Descent**. The idea is to optimize one weight $w_j$ at a time while holding all others fixed. This has a closed-form solution for each step (using the soft-thresholding operator) and converges to the global minimum.

**Geometric Interpretation:**
Why does $L_1$ lead to sparsity while $L_2$ does not?
Imagine the contours of the OLS loss function (ellipses) and the constraint region defined by the regularization term ($||\mathbf{w}|| \leq C$).
*   **Ridge ($L_2$)**: The constraint region is a circle (or hypersphere). The OLS contours typically touch the circle at a point where $w_j \neq 0$.
*   **Lasso ($L_1$)**: The constraint region is a diamond (or polytope) with corners on the axes. The OLS contours are much more likely to touch the diamond at a **corner** (where one or more $w_j = 0$).

{{< figure src="/images/ml/l1_vs_l2.png" caption="Geometric interpretation of L1 (Lasso) vs L2 (Ridge) regularization. The L1 diamond shape encourages solutions on the axes (sparse)." >}}

## Polynomial Regression

Regularization helps us constrain a model that is too complex. But what if our linear model is too simple (high bias)? What if the relationship between our features and target is fundamentally non-linear? For example, what if the data follows a parabolic curve? We can still use the machinery of linear regression by transforming the input features into a higher-dimensional space using a **feature map** $\phi(\mathbf{x})$. This is known as **basis function expansion**.

{{< figure src="/images/ml/polynomial_regression.png" caption="Fitting a non-linear function using polynomial features." >}}

For a 1D input $x$, we can use polynomial features:

$$
\phi(x) = [1, x, x^2, \dots, x^M]^T
$$

The model becomes:
$$
y = \mathbf{w}^T\phi(x) = w_0 + w_1 x + w_2 x^2 + \dots + w_M x^M
$$

This is still a **linear model** because it is linear in the parameters $\mathbf{w}$, even though it is non-linear in $x$. We can solve this using OLS with the transformed design matrix $\boldsymbol{\Phi}$.

While polynomial regression allows us to fit non-linear data, it has significant drawbacks:

1.  **Overfitting**: High-degree polynomials can fit the training data perfectly but oscillate wildly between data points (Runge's phenomenon), leading to terrible generalization.
2.  **Non-local effects**: Changing a data point in one region affects the polynomial fit everywhere.
3.  **Extrapolation**: Polynomials go to $\pm \infty$ as $x \to \pm \infty$, making them poor for extrapolation.

To address these issues, we often need to carefully tune the degree $M$ or use strong regularization. Later, we will see how **Gaussian Processes** provide a more principled, non-parametric approach to regression that avoids the need to manually specify basis functions.

**Curse of Dimensionality:**
A major practical issue with polynomial regression is the explosion in the number of features. If we have $D$ original features and want to include all polynomial terms up to degree $M$, the number of features grows as $O(D^M)$. This makes the model computationally expensive and prone to severe overfitting without massive amounts of data. 