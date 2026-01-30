---
title: Linear Regression
type: docs
weight: 6
---


Linear regression is one of the most fundamental algorithms in machine learning. It is a linear approach to modeling the relationship between a scalar response (or dependent variable) and one or more explanatory variables (also called features or independent variables). In particular, linear regression attempts to model the relationship by fitting a linear equation to observed data. So we assume that the target value $y$ is a linear combination of the input features $\mathbf{x}$ plus some noise. For a single data point $(\mathbf{x}, y)$, the model is given by:

$$
y = \mathbf{w}^T\mathbf{x} + \epsilon
$$

where:
- $\mathbf{x} \in \mathbb{R}^D$ is the input feature vector (often with a bias term $x_0 = 1$).
- $\mathbf{w} \in \mathbb{R}^D$ is the weight vector (parameters) we want to learn.
- $\epsilon$ is the error term (noise), typically assumed to be Gaussian distributed $\epsilon \sim \mathcal{N}(0, \sigma^2)$. This noise term comes from the assumption that our observations are subject to some random noise or measurement error.

For a dataset of $N$ samples, we can write this in matrix notation:

$$
\mathbf{y} = \mathbf{X}\mathbf{w} + \boldsymbol{\epsilon}
$$

where:
- $\mathbf{y} \in \mathbb{R}^N$ is the vector of target values.
- $\mathbf{X} \in \mathbb{R}^{N \times D}$ is the design matrix, where the $i$-th row is $\mathbf{x}_i^T$.
- $\boldsymbol{\epsilon} \in \mathbb{R}^N$ is the noise vector.

## Ordinary Least Squares (OLS)

Now that we have defined the model structure, we need a way to estimate the parameters $\mathbf{w}$. The most common method is **Ordinary Least Squares (OLS)**. The intuition is simple: we want to find the line (or hyperplane) that minimizes the discrepancy between the actual target values and the values predicted by our model. For now we will ignore the noise term and just focus on fitting the model to the data.

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

where the predicted price $\hat{y}_i$ is as close as possible to the observed price $y_i$. The difference between the observed price $y_i$ and the predicted price $\hat{y}_i$ is called the **error** for the $i$-th observation. To find the best-fitting line, we minimize the **sum of squared errors** across all observations. Note that minimizing the sum of squared errors is equivalent to minimizing the **Mean Squared Error (MSE)**, which is often used as the cost function $J(\mathbf{w})$:

$$
J(\mathbf{w}) = \text{MSE} = \frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2
$$

Since $N$ is a positive constant, minimizing the MSE yields the same parameters as minimizing the sum of squared errors:

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

To solve the least squares problem, the columns of $\mathbf{X}$ must be **linearly independent**. Linear independence ensures that $\mathbf{X}^T\mathbf{X}$ is invertible, which is crucial for deriving the solution. For example, if all the feature values $x_i$ are identical (e.g., $x_i = x_j$ for all $i \neq j$), then the column would be a multiple of the first column and the columns of $\mathbf{X}$ would not span a sufficiently large space, making it impossible to uniquely determine the coefficients $\mathbf{w}$. Linear independence of the columns of $\mathbf{X}$ ensures that the data provides enough information to determine a unique solution. If the columns are not linearly independent (or close to it, a condition known as multicollinearity), $\mathbf{X}^T\mathbf{X}$ becomes non-invertible (singular) or ill-conditioned. This is a key motivation for **regularization**, where we add a term (like $\lambda \mathbf{I}$) to make the matrix invertible.

If we look more closely at the what the matrix multiplication is doing we can actually derive an explicit formula for the coefficients $\mathbf{w}$ when we only have one feature (plus the intercept). This will help us understand what the normal equations are actually doing. First let's remember what the matrix multiplication of $\mathbf{X}^T\mathbf{X}$ looks like:

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

## Overfitting, and Underfitting

The ultimate goal of machine learning is **generalization**: the ability of a model to perform well on unseen data. We typically train our model on a **training set**, but we care about its performance on a held-out **test set**. There are two common pitfalls that can prevent good generalization:
- **Underfitting**: Occurs when the model is too simple to capture the underlying structure of the data. For example, trying to fit a straight line to data that is clearly quadratic. The model has high **bias** and fails to learn the patterns in the training data. The mathmatical definition for a model/estimator is:

$$
\text{Bias} = E[\hat{f}(\mathbf{x})] - f(\mathbf{x})
$$

where $\hat{f}(\mathbf{x})$ is the prediction of the model and $f(\mathbf{x})$ is the true underlying function. High bias means that the model's predictions are systematically off from the true values.

- **Overfitting**: Occurs when the model is too complex and learns the noise in the training data as if it were signal. The model fits the training data perfectly but fails to generalize to new data. The model has high **variance**. The mathematical definition for variance is:

$$
\text{Variance} = E[(\hat{f}(\mathbf{x}) - E[\hat{f}(\mathbf{x})])^2]
$$

where $\hat{f}(\mathbf{x})$ is the prediction of the model. High variance means that the model's predictions vary significantly with different training sets.

{{< figure 
  src="/images/ml/overfittingUnderfitting.svg" 
  caption="Illustration of Underfitting (High Bias), Optimal Fit, and Overfitting (High Variance)." 
  alt="Illustration of Underfitting (High Bias), Optimal Fit, and Overfitting (High Variance)."
  width="600"
>}}

This is also what ultimately leads us to the **Bias-Variance Tradeoff**. As we increase model complexity (e.g., adding more features, using higher degree polynomials), bias typically decreases, but variance increases. The goal is to find the sweet spot that minimizes the total error. The relationship between model complexity and error can also be quantified by the **Bias-Variance Decomposition**. The expected test error can be decomposed into three terms: **Bias**, **Variance**, and **Irreducible Error**. 

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
- **Bias**: The difference between the average prediction of our model and the correct value. High bias means the model is too simple to capture the underlying structure of the data (underfitting).
- **Variance**: The variability of the model prediction for a given data point across different realizations of the training set. High variance means the model is too sensitive to the noise in the training data (overfitting).
- **Irreducible Error**: The noise inherent in the problem itself. We cannot reduce this term.

A common technique to assess model performance and generalization is to split the dataset into training and test sets. The model is trained on the training set, and its performance is evaluated on the test set using metrics like Mean Squared Error (MSE) or R-squared ($R^2$). Performing this split once can lead to high variance in the performance estimate, especially with small datasets, so instead we often use cross-validation, where the data is split multiple times to get a more robust estimate of test error. This helps in selecting model hyperparameters not just based on training performance but also on validation performance. The most common method is **$k$-fold Cross-Validation**:

1.  Split the training data into $k$ equal-sized folds.
2.  For each fold $i \in \{1, \dots, k\}$ we validate (test) the model on fold $i$ and we train the model on the other $k-1$ folds.
3.  Average the validation errors to get an estimate of the test error.

This allows us to tune hyperparameters without touching the final test set, preventing data leakage.

## Bias and Variance of OLS Estimator

We can analyze the properties of the OLS estimator $\hat{\mathbf{w}} = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}$ by looking at its expectation (bias) and variance. Recall that our model assumption is $\mathbf{y} = \mathbf{X}\mathbf{w} + \boldsymbol{\epsilon}$, where $E[\boldsymbol{\epsilon}] = \mathbf{0}$ and $Cov(\boldsymbol{\epsilon}) = \sigma^2\mathbf{I}$.

First, let's check if the estimator is unbiased. We take the expectation of $\hat{\mathbf{w}}$:

$$
\begin{align*}
E[\hat{\mathbf{w}}] &= E[(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}] \\
&= (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T E[\mathbf{y}] \\
&= (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T E[\mathbf{X}\mathbf{w} + \boldsymbol{\epsilon}] \\
&= (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T (\mathbf{X}\mathbf{w} + E[\boldsymbol{\epsilon}]) \\
&= (\mathbf{X}^T\mathbf{X})^{-1}(\mathbf{X}^T\mathbf{X})\mathbf{w} \\
&= \mathbf{I}\mathbf{w} = \mathbf{w}
\end{align*}
$$

Since $E[\hat{\mathbf{w}}] = \mathbf{w}$, the OLS estimator is **unbiased**. This means that on average, across many different training sets, the estimated coefficients will equal the true coefficients.

Next, we calculate the variance of the estimator. The variance of a vector-valued random variable is a covariance matrix.

$$
\begin{align*}
Var(\hat{\mathbf{w}}) &= Var((\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}) \\
&= (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T Var(\mathbf{y}) ((\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T)^T \\
&= (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T (\sigma^2 \mathbf{I}) \mathbf{X}(\mathbf{X}^T\mathbf{X})^{-1} \\
&= \sigma^2 (\mathbf{X}^T\mathbf{X})^{-1}(\mathbf{X}^T \mathbf{X})(\mathbf{X}^T\mathbf{X})^{-1} \\
&= \sigma^2 (\mathbf{X}^T\mathbf{X})^{-1}
\end{align*}
$$

So the covariance matrix of the weights is proportional to the inverse of the covariance matrix of the features (Gram matrix). The **Gauss-Markov Theorem** states that under the assumptions of the linear regression model (errors have expectation zero, are uncorrelated and have equal variances), the OLS estimator is the **Best Linear Unbiased Estimator (BLUE)**.

While OLS has the lowest variance among unbiased estimators, this variance can still be very high. This happens when the features are highly correlated (**multicollinearity**). If features are correlated, the columns of $\mathbf{X}$ are nearly linearly dependent. This means the matrix $\mathbf{X}^T\mathbf{X}$ is close to being singular (non-invertible). In terms of eigendecomposition, $\mathbf{X}^T\mathbf{X}$ will have some very small eigenvalues $\lambda_i \approx 0$. The inverse matrix $(\mathbf{X}^T\mathbf{X})^{-1}$ has eigenvalues $1/\lambda_i$. If $\lambda_i$ is small, $1/\lambda_i$ becomes huge. Since the variance of the weights is given by $\sigma^2 (\mathbf{X}^T\mathbf{X})^{-1}$, these huge eigenvalues translate directly into **huge variance** for the estimated coefficients.

Intuitively, if two features are highly correlated, the model can shift a large positive weight to one and a large negative weight to the other without changing the prediction much. This instability means the specific weights we find are very sensitive to the specific noise in our training data.

## Regularization

So we have established that OLS is unbiased and has the lowest variance among all linear unbiased estimators by Gauss-Markov theorem: For any linear estimator $\tilde{\theta} = \mathbf{c}^T \mathbf{y}$ that is unbiased (i.e., $E[\tilde{\theta}] = \theta$) for $\theta = \mathbf{a}^T \mathbf{w}$, we have:

$$
\mathbf{Var}(\tilde{\theta}) \geq \mathbf{Var}(\hat{\theta})
$$

However, in practice, OLS can still lead to overfitting, especially when when features are highly correlated (**multicollinearity**), causing the matrix $\mathbf{X}^T\mathbf{X}$ to be close to singular (non-invertible), making the OLS solution unstable. So small changes in the data can lead to large changes in the weights $\mathbf{w}$, due to the high variance of the estimator.

To combat this, we can introduce **regularization** to trade introduce some bias in exchange for a significant reduction in variance. Regularization adds a penalty term to the loss function that discourages complex models. This helps prevent overfitting and improves generalization.

### Ridge Regression

An effective way to reduce variance and handle multicollinearity is **Ridge Regression** (also known as $L_2$ regularization). Here, we modify the OLS loss function by adding a penalty term proportional to the square of the $L_2$ norm of the weight vector $\mathbf{w}$. The idea is to discourage complex models by penalizing large coefficients, as if we have a large weight then due to the nature of matrix multiplication a small change in the input features can lead to a large change in the output prediction.

$$
\mathcal{L}_{Ridge}(\mathbf{w}) = ||\mathbf{y} - \mathbf{X}\mathbf{w}||^2 + \lambda ||\mathbf{w}||_2^2
$$

where $\lambda \geq 0$ is a hyperparameter controlling the strength of regularization. Just like for OLS, we can derive a closed-form solution for Ridge Regression. We can derive the solution by taking the gradient and setting it to zero:

$$
\begin{align*}
\nabla_{\mathbf{w}} \mathcal{L}_{Ridge} &= -2\mathbf{X}^T(\mathbf{y} - \mathbf{X}\mathbf{w}) + 2\lambda\mathbf{w} = 0 \\
\mathbf{X}^T\mathbf{X}\mathbf{w} + \lambda\mathbf{I}\mathbf{w} &= \mathbf{X}^T\mathbf{y} \\
(\mathbf{X}^T\mathbf{X} + \lambda\mathbf{I})\mathbf{w} &= \mathbf{X}^T\mathbf{y}
\end{align*}
$$

Solving for $\mathbf{w}$ gives us the Ridge Regression estimator:

$$
\hat{\mathbf{w}}_{Ridge} = (\mathbf{X}^T\mathbf{X} + \lambda\mathbf{I})^{-1}\mathbf{X}^T\mathbf{y}
$$

Here, we can also see why Ridge helps with multicollinearity. Adding $\lambda\mathbf{I}$ to $\mathbf{X}^T\mathbf{X}$ ensures that the matrix is always invertible as it adds $\lambda$ to the diagonal elements, effectively increasing the eigenvalues of the matrix. This stabilizes the inversion process and reduces the sensitivity of the weights to small changes in the data.

We can derive the bias and variance of the Ridge estimator $\hat{\mathbf{w}}_{Ridge} = (\mathbf{X}^T\mathbf{X} + \lambda\mathbf{I})^{-1}\mathbf{X}^T\mathbf{y}$ just like for the OLS estimator. Let $\mathbf{H}_\lambda = (\mathbf{X}^T\mathbf{X} + \lambda\mathbf{I})^{-1}\mathbf{X}^T\mathbf{X}$. We start with the bias:

$$
\begin{align*}
E[\hat{\mathbf{w}}_{Ridge}] &= E[(\mathbf{X}^T\mathbf{X} + \lambda\mathbf{I})^{-1}\mathbf{X}^T\mathbf{y}] \\
&= (\mathbf{X}^T\mathbf{X} + \lambda\mathbf{I})^{-1}\mathbf{X}^T \mathbf{X}\mathbf{w} \\
&= \mathbf{H}_\lambda \mathbf{w}
\end{align*}
$$

Since $\mathbf{H}_\lambda \neq \mathbf{I}$ (unless $\lambda=0$), $E[\hat{\mathbf{w}}_{Ridge}] \neq \mathbf{w}$, so Ridge regression is **biased**. The bias is $E[\hat{\mathbf{w}}_{Ridge}] - \mathbf{w} = (\mathbf{H}_\lambda - \mathbf{I})\mathbf{w}$. Note that as $\lambda$ increases, the bias increases because $\mathbf{H}_\lambda$ shrinks the weights more towards zero. Next, we compute the variance:

$$
\begin{align*}
Var(\hat{\mathbf{w}}_{Ridge}) &= Var((\mathbf{X}^T\mathbf{X} + \lambda\mathbf{I})^{-1}\mathbf{X}^T\mathbf{y}) \\
&= (\mathbf{X}^T\mathbf{X} + \lambda\mathbf{I})^{-1}\mathbf{X}^T Var(\mathbf{y}) ((\mathbf{X}^T\mathbf{X} + \lambda\mathbf{I})^{-1}\mathbf{X}^T)^T \\
&= (\mathbf{X}^T\mathbf{X} + \lambda\mathbf{I})^{-1}\mathbf{X}^T (\sigma^2 \mathbf{I}) \mathbf{X}(\mathbf{X}^T\mathbf{X} + \lambda\mathbf{I})^{-1} \\
&= \sigma^2 (\mathbf{X}^T\mathbf{X} + \lambda\mathbf{I})^{-1}\mathbf{X}^T\mathbf{X}(\mathbf{X}^T\mathbf{X} + \lambda\mathbf{I})^{-1}
\end{align*}
$$

It can be shown that the variance of the Ridge estimator is strictly smaller than the variance of the OLS estimator. By introducing a small amount of bias (controlled by $\lambda$), we can drastically reduce the variance, often leading to a lower Mean Squared Error (MSE). This is the essence of the **Bias-Variance Tradeoff** in regularization.

### Training with Noisy Targets

An interesting property of the squared error loss is that it remains unchanged in expectation when we replace the true targets with random estimates, as long as those estimates are unbiased. This observation has practical applications in scenarios where we cannot directly observe clean target values.

Consider the standard regression problem where we minimize the sum of squared errors:

$$
\min_{\mathbf{w}} \sum_{i=1}^n (y_i - f_{\mathbf{w}}(\mathbf{x}_i))^2
$$

Now suppose instead of the true targets $y_i$, we only have access to noisy estimates $\hat{y}_i$ where $\mathbb{E}[\hat{y}_i] = y_i$. We can show that minimizing the loss with these noisy targets gives us the same expected result.

{{< callout type="proof" >}}
We want to show that the difference between the expected noisy loss and the true loss does not depend on the model parameters. Let's expand the expected loss with noisy targets:

$$
\begin{align*}
\mathbb{E}_{\hat{y}} \left[ \frac{1}{n} \sum_{i=1}^n (\hat{y}_i - f_{\mathbf{w}}(\mathbf{x}_i))^2 \right] &= \frac{1}{n} \sum_{i=1}^n \mathbb{E}_{\hat{y}_i} \left[ \hat{y}_i^2 - 2\hat{y}_i f_{\mathbf{w}}(\mathbf{x}_i) + f_{\mathbf{w}}(\mathbf{x}_i)^2 \right] \\
&= \frac{1}{n} \sum_{i=1}^n \left[ \mathbb{E}[\hat{y}_i^2] - 2f_{\mathbf{w}}(\mathbf{x}_i)\mathbb{E}[\hat{y}_i] + f_{\mathbf{w}}(\mathbf{x}_i)^2 \right] \\
&= \frac{1}{n} \sum_{i=1}^n \left[ \mathbb{E}[\hat{y}_i^2] - 2f_{\mathbf{w}}(\mathbf{x}_i) y_i + f_{\mathbf{w}}(\mathbf{x}_i)^2 \right]
\end{align*}
$$

where we used the fact that $\mathbb{E}[\hat{y}_i] = y_i$. Now using the identity $\mathbb{E}[\hat{y}_i^2] = \text{Var}(\hat{y}_i) + (\mathbb{E}[\hat{y}_i])^2 = \text{Var}(\hat{y}_i) + y_i^2$:

$$
\begin{align*}
&= \frac{1}{n} \sum_{i=1}^n \left[ \text{Var}(\hat{y}_i) + y_i^2 - 2f_{\mathbf{w}}(\mathbf{x}_i) y_i + f_{\mathbf{w}}(\mathbf{x}_i)^2 \right] \\
&= \frac{1}{n} \sum_{i=1}^n (y_i - f_{\mathbf{w}}(\mathbf{x}_i))^2 + \frac{1}{n} \sum_{i=1}^n \text{Var}(\hat{y}_i)
\end{align*}
$$

The key insight is that the difference between the expected noisy loss and the true loss is:

$$
\mathbb{E}_{\hat{y}} \left[ \frac{1}{n} \sum_{i=1}^n (\hat{y}_i - f_{\mathbf{w}}(\mathbf{x}_i))^2 \right] - \frac{1}{n} \sum_{i=1}^n (y_i - f_{\mathbf{w}}(\mathbf{x}_i))^2 = \frac{1}{n} \sum_{i=1}^n \text{Var}(\hat{y}_i)
$$

Importantly, this difference does not depend on the model parameters $\mathbf{w}$. Therefore, the $\mathbf{w}$ that minimizes the expected noisy error is exactly the same $\mathbf{w}$ that minimizes the true error. As we collect more independent noisy samples for each target, the variance term approaches zero by the law of large numbers, and our estimate converges to the true minimum.
{{< /callout >}}

**Practical Applications**: This result has important applications where we cannot directly observe clean target values:

1. **Self-Supervised Learning**: We can train models using corrupted versions of the data as both inputs and targets, as long as the corruption is unbiased.

2. **Data Augmentation**: When the original training set is small, we can generate multiple noisy versions of each target value to increase the effective training set size without introducing bias.

The key requirement in all these applications is that the noise in $\hat{y}_i$ must be unbiased, meaning $\mathbb{E}[\hat{y}_i] = y_i$. If this condition holds, we can train robust models even without access to clean target values.

### Lasso Regression

A limitation of Ridge Regression is that it shrinks all weights towards zero but rarely sets them exactly to zero. This means the final model can still includes all input features, which isn't ideal if features are correlated, irrelevant, or redundant. This can also be a problem for **interpretability** if we have thousands of features.

**Lasso** (Least Absolute Shrinkage and Selection Operator) addresses this by adding a penalty proportional to the absolute value of the coefficients. This encourages **sparsity**, meaning many weights become exactly zero. This effectively performs **feature selection**, giving us a simpler, more interpretable model that only relies on the most important features.

$$
\mathcal{L}_{Lasso}(\mathbf{w}) = ||\mathbf{y} - \mathbf{X}\mathbf{w}||^2 + \lambda ||\mathbf{w}||_1
$$

Unlike Ridge, Lasso does not have a closed-form solution (due to the non-differentiability of the absolute value at 0) Since the $L_1$ term is not differentiable at zero, we cannot use standard gradient descent or closed-form solutions. Instead, we often use **Coordinate Descent**. The idea is to optimize one weight $w_j$ at a time while holding all others fixed. This has a closed-form solution for each step (using the soft-thresholding operator) and converges to the global minimum.

However, it has a very useful property: it promotes **sparsity**. It tends to force the coefficients of less important features to be exactly zero. This effectively performs feature selection.

To understand why Lasso leads to sparsity, let's consider the simplest case: a single feature with an orthonormal design matrix (so $\mathbf{X}^T\mathbf{X} = 1$). The objective function becomes:

$$
\min_{w} \frac{1}{2}(y - w)^2 + \lambda |w|
$$

This is a convex problem, but $|w|$ is not differentiable at $w=0$. We can solve this using subgradients. The optimality condition is that 0 must be in the subdifferential of the objective function with respect to $w$:

$$
-(y - w) + \lambda \partial |w| = 0 \implies w - y + \lambda s = 0
$$

where $s \in \partial |w|$ is the subgradient of the absolute value function:
- $s = \text{sign}(w)$ if $w \neq 0$
- $s \in [-1, 1]$ if $w = 0$

We can analyze the solution $\hat{w}$ based on the value of $y$:
1.  **If $y > \lambda$**: We must have $w > 0$, so $s=1$. Then $w - y + \lambda = 0 \implies \hat{w} = y - \lambda$.
2.  **If $y < -\lambda$**: We must have $w < 0$, so $s=-1$. Then $w - y - \lambda = 0 \implies \hat{w} = y + \lambda$.
3.  **If $-\lambda \le y \le \lambda$**: We can satisfy the condition with $w=0$. The equation becomes $-y + \lambda s = 0 \implies s = y/\lambda$. Since $|y| \le \lambda$, we have $|s| \le 1$, which is a valid subgradient at 0. So $\hat{w} = 0$.

This solution is known as the **Soft Thresholding** operator:
$$
\hat{w} = \text{sign}(y) \max(|y| - \lambda, 0)
$$

Crucially, whenever the OLS estimate $y$ is small (specifically, within the interval $[-\lambda, \lambda]$), the Lasso estimate is **exactly zero**. This logic extends to the multivariate case (via Coordinate Descent), causing many coefficients to be driven to zero.

{{< figure 
  src="/images/ml/ridgeLasso.png" 
  caption="Comparison of Ridge and Lasso Regression. Ridge shrinks coefficients but keeps all features, while Lasso sets some coefficients to zero, effectively performing feature selection." 
  alt="Comparison of Ridge and Lasso Regression. Ridge shrinks coefficients but keeps all features, while Lasso sets some coefficients to zero, effectively performing feature selection."
  width="600"
>}}

In the figure above, we can see how Ridge regression shrinks all coefficients towards zero but keeps them non-zero, while Lasso regression sets some coefficients exactly to zero, effectively selecting a subset of features. Visuallly, if we imagine the contours of the OLS loss function (ellipses) and the constraint region defined by the regularization term ($||\mathbf{w}|| \leq C$).
- **Ridge ($L_2$)**: The constraint region is a circle. The OLS contours typically touch the circle at a point where $w_j \neq 0$.
- **Lasso ($L_1$)**: The constraint region is a diamond with corners on the axes. The OLS contours are much more likely to touch the diamond at a **corner** (where one or more $w_j = 0$).

Unlike OLS and Ridge, Lasso does not have a simple closed-form expression for its bias and variance due to the non-linear selection operation. However, we can characterize them qualitatively:

## Polynomial Regression

Regularization helps us constrain a model that is too complex. But what if our linear model is too simple (high bias)? What if the relationship between our features and target is fundamentally non-linear? For example, what if the data follows a parabolic curve? We can still use the machinery of linear regression by transforming the input features into a higher-dimensional space using a **feature map** $\phi(\mathbf{x})$. This is known as **basis function expansion**.

{{< figure
  src="/images/ml/polynomialRegression.png" 
  caption="The polynomial of degree 15 fits the training data perfectly but generalizes poorly to new data (overfitting). The polynomial of degree 4 captures the underlying trend better."
  alt="The polynomial of degree 15 fits the training data perfectly but generalizes poorly to new data (overfitting). The polynomial of degree 4 captures the underlying trend better."
  width="600"
>}}

If we have a single feature $x$, we can create polynomial features up to degree $d$ by using the feature map:

$$
\phi(x) = [1, x, x^2, \dots, x^d]^T.
$$

The model becomes:
$$
y = \mathbf{w}^T\phi(x) = w_0 + w_1 x + w_2 x^2 + \dots + w_d x^d
$$

This is still a **linear model** because it is linear in the parameters $\mathbf{w}$, even though it is non-linear in $x$. We can solve this using OLS with the transformed design matrix $\boldsymbol{\Phi}$.

While polynomial regression allows us to fit non-linear data, it has significant drawbacks:

1.  **Overfitting**: High-degree polynomials can fit the training data perfectly but oscillate wildly between data points (Runge's phenomenon), leading to terrible generalization.
2.  **Non-local effects**: Changing a data point in one region affects the polynomial fit everywhere. You can think of this if the function where like a piece of string that is fixed at certain points (the data points) but can move freely in between. So if we change one of the fixed points the whole string will be affected.
3.  **Curse of Dimensionality**: The number of polynomial features grows rapidly with the number of original features and the degree of the polynomial, leading to computational challenges and overfitting. Specifically, for $n$ features and polynomial degree $d$, the number of polynomial features is given by the binomial coefficient $\binom{n+d}{d}$, which grows combinatorially with $n$ and $d$, specifically $O(n^d)$, which can become infeasible for even moderate $n$ and $d$.

To address these issues, we often need to carefully tune the degree $d$ or use strong regularization. Later, we will see how **Gaussian Processes** provide a more principled, non-parametric approach to regression that avoids the need to manually specify basis functions.
