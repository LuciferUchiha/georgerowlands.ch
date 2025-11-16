---
title: Correlation
type: docs
weight: 13
---

Correlation is a statistical measure that describes how related two random variables are, for computer scientists, this
is usually how related two vectors or time series are.

## Correlation Coefficients

Most often the correlation coefficient is defined as $\rho$ (rho) or $r$ and is a value between -1 and 1, i.e.
$-1 \leq r \leq 1$. The simplest form of correlation is a linear correlation, which is a linear relationship between
the two variables. The correlation coefficient can then be interpreted in the following way:

- $r = 1$ means that the two variables are perfectly correlated, i.e. if one variable increases, the other variable
increases as well.
- $r = -1$ means that the two variables are perfectly negatively correlated, i.e. if one variable increases, the
other variable decreases.
- $r = 0$ means that the two variables are not correlated at all, i.e. if one variable increases or decreases, the
other variable does not change.

Therefore, the absolute value of the correlation coefficient can be interpreted as the strength of the correlation.

The standard interpretation of the correlation strength is:

- $0.8 \leq |r| \leq 1$ is a strong correlation.
- $0.5 \leq |r| \leq 0.8$ is a moderate correlation.
- $0.1 \leq |r| \leq 0.5$ is a weak correlation.
- $0 \leq |r| \leq 0.1$ is no correlation.

{{< figure
  src="/images/maths/correlationLinear.png"
  alt="Visualizations of linear correlations, positive and negative."
  caption="Visualizations of linear correlations, positive and negative."
>}}

{{< callout type="example" >}}
A simple example of a linear correlation is the air temperature and the number of ice creams sold. If it is hot
then more ice creams are sold, if it is cold then less ice creams are sold. This is a linear correlation, and the
correlation coefficient is close to 1 as the two variables increase and decrease together.
{{< /callout >}}

Non-linear correlations are also possible, but they are more difficult to interpret.

{{< figure
  src="/images/maths/correlationNonLinear.png"
  alt="Visualizations of linear and non-linear correlations. The top left is the linear correlation coeff, the top right is the non-linear correlation coeff."
  caption="Visualizations of linear and non-linear correlations. The top left is the linear correlation coeff, the top right is the non-linear correlation coeff."
  width="600"
>}}

{{< callout type="warning" >}}
The correlation coefficient only measures if there is a correlation between the two variables, it does not say
anything about cause and effect. For example if we have a positive correlation between the number of ice creams
sold and the number of people who drown, it does not mean that eating ice cream causes people to drown. It is
simply that both variables increase during the summer months.
{{< /callout >}}

### Outliers

The correlation coefficient is just like regression sensitive to outliers. If there are outliers in the data, then
the correlation coefficient will be affected by them. So it is important to check for outliers before calculating the
correlation coefficient.

{{< figure
  src="/images/maths/correlationOutlier.png"
  alt="An example of an outlier affecting the correlation coefficient."
  caption="An example of an outlier affecting the correlation coefficient."
  width="500"
>}}

### Pearson's Correlation Coefficient

Pearson's correlation coefficient is the most common correlation coefficient, and is a measure of the linear correlation
between two variables. If we have the two variables $X$ and $Y$, which have $n$ values, then the Pearson's correlation
coefficient is defined as:

$$
\begin{align*}
r_{X,Y} &= \frac{n \cdot \sum_{i=1}^{n}{(x_i - y_i)} - \sum_{i=1}^{n}{x_i} \cdot \sum_{i=1}^{n}{y_i}}{\sqrt{n \cdot \sum_{i=1}^{n}{x_i^2} - (\sum_{i=1}^{n}{x_i})^2} \cdot \sqrt{n \cdot \sum_{i=1}^{n}{y_i^2} - (\sum_{i=1}^{n}{y_i})^2}} \\
&= \frac{\sum_{i=1}^{n}{(x_i - \bar{x}) \cdot (y_i - \bar{y})}}{\sqrt{\sum_{i=1}^{n}{(x_i - \bar{x})^2}} \cdot \sqrt{\sum_{i=1}^{n}{(y_i - \bar{y})^2}}}
\end{align*}
$$

This formula can look a bit scary, but it is actually quite simple. Let us start with a few quick reminders. The mean
value of a variable $X$ is defined as:

$$
\mathbb{E}(X) = \mu_X = \bar{x} = \frac{1}{n} \cdot \sum_{i=1}^{n}{x_i}
$$

The variance of a variable $X$ is defined as:

$$
\begin{align*}
\text{Var}(X) &= \mathbb{E}(X^2) - \mathbb{E}(X)^2 \\
&= \frac{1}{n} \cdot \sum_{i=1}^{n}{x_i^2} - (\frac{1}{n} \cdot \sum_{i=1}^{n}{x_i})^2
\end{align*}
$$

The standard deviation of a variable $X$ is defined as:

$$
\sigma(X) = \sqrt{\text{Var}(X)}
$$

And lastly the covariance between two variables $X$ and $Y$ is defined as:

$$
\begin{align*}
\text{Cov}(X,Y) &= \mathbb{E}((X - \mu_X) \cdot (Y - \mu_Y)) \\
&= \frac{1}{n} \cdot \sum_{i=1}^{n}{(x_i - \bar{x}) \cdot (y_i - \bar{y})}
\end{align*}
$$

{{< callout type="todo" >}}
This can be linked to all the other articles about statistics. Specifically making it clear that correlation is a standardized covariance.
{{< /callout >}}

Now we can see that the nominator of the Pearson's correlation coefficient is the covariance between $X$ and $Y$ times
$n$, and the denominator is the product of the standard deviation of $X$ and $Y$ times $n$. So the Pearson's
correlation coefficient is the covariance between $X$ and $Y$ divided by the product of the standard deviation of $X$
and $Y$. So we can rewrite the formula as:

$$
r_{X,Y} = \frac{\text{Cov}(X,Y)}{\sigma_X \cdot \sigma_Y}
$$

{{< callout type="example" >}}
Let's say we have the hypothesis that students with a higher GPA also have a higher SAT score. We can then check
if there is a correlation between the two variables. We have the following data:

| Student | GPA ($X$) | SAT ($Y$) |
|---------|-----|-----|
| 1 | 3.4 | 595 |
| 2 | 3.2 | 520 |
| 3 | 3.9 | 715 |
| 4 | 2.3 | 405 |
| 5 | 3.9 | 680 |
| 6 | 2.5 | 490 |
| 7 | 3.5 | 565 |

If we plot the data we get the following plot, where we can see a clear correlation:

{{< callout type="todo" >}}
Include plot here.
{{< /callout >}}

By extending the table slightly we can calculate the Pearson's correlation coefficient pretty quickly.

| Student | GPA ($X$) | SAT ($Y$) | $x_i y_i$ | $x_i^2$ | $y_i^2$ |
|---------|-----|-----|-----|-----|-----|
| 1 | 3.4 | 595 | 2023 | 11.56 | 354025 |
| 2 | 3.2 | 520 | 1664 | 10.24 | 270400 |
| 3 | 3.9 | 715 | 2789 | 15.21 | 511225 |
| 4 | 2.3 | 405 | 932 | 5.29 | 164025 |
| 5 | 3.9 | 680 | 2652 | 15.21 | 462400 |
| 6 | 2.5 | 490 | 1225 | 6.25 | 240100 |
| 7 | 3.5 | 565 | 1978 | 12.25 | 319225 |
| **Sum** | 22.7 | 3970 | 13262 | 76.01 | 2322400 |

Now we can calculate the Pearson's correlation coefficient:

$$
\begin{align*}
r_{X,Y} &= \frac{7 \cdot 13262 - 22.7 \cdot 3970}{\sqrt{7 \cdot 76.01 - 22.7^2} \cdot \sqrt{7 \cdot 2322400 - 3970^2}} \\
&= \frac{2715}{2864.22} \approx 0.95
\end{align*}
$$
{{< /callout >}}

### Spearman's Rank Correlation Coefficient

Spearman's rank correlation coefficient measures the monotonic relationship between two variables. As a reminder, a
monotonic function is a function that is either strictly increasing or strictly decreasing. The word rank comes into
play because we first rank the values of the variables with the lowest value of the variable getting the rank 1 and
the highest value of the variable getting the rank $n$, we then have $R(X)$ and $R(Y)$ which are the ranked variables.

Then the Spearman's rank correlation coefficient is actually calculated just as the Pearson's correlation coefficient
between the two ranked variables. Often the Spearman's rank correlation coefficient is denoted as $r_s$.

$$
r_s = \frac{\text{Cov}(R(X),R(Y))}{\sigma_{R(X)} \cdot \sigma_{R(Y)}}
$$

If all the ranks are unique (which is often the case if there aren't any duplicates in the data), then the formula can
be simplified to:

$$
r_s = 1 - \frac{6 \cdot \sum_{i=1}^{n}{(R(X_i) - R(Y_i))^2}}{n \cdot (n^2 - 1)} = 1 - \frac{6 \cdot \sum_{i=1}^{n}{d_i^2}}{n \cdot (n^2 - 1)}
$$

{{< figure
  src="/images/maths/correlationSpearman.png"
  alt="Visualizations of Spearman's rank correlation coefficient compared to Pearson's correlation coefficient."
  caption="Visualizations of Spearman's rank correlation coefficient compared to Pearson's correlation coefficient."
>}}

## Correlation Matrix

Isn't this also used in image classification to see how close classes are?

## Coefficient of determination R-Squared

## Cross Correlation

### Of Vectors

See above.

### Of Functions

Same but the function isnt flipped.

### Of Time Series

How is this interpretetd. Should say somethign about lags no? Because one is slid across the other it is correlation of one time series with the lagged time series of the other.

## Auto Correlation

Is Cross Correlation but the two functions are the same so a cross correlation with itself. How is this interpretetd? Same goes here for time series is this interpreted?
Because it is slid across it is basically the correlation with the lagged version of it self. so each point is correlation of the series with the x lag of itself.

### Partial Auto Correltion

a, b, c. a with b are correlated and b with c, to measure true influence of c on a  need to take out the influence b had on a. Taking out inluence of other timespots on current.

AR model or MA model https://medium.com/@vaibhav1403/ar-model-vs-ma-model-427ee28587a#:~:text=How%20they%20differ%3A,white%20noise%20or%20error%20terms.

1 lag differencing is a way to detrend. subtract lag 1 values.

first order AR model is if lag 1 is used, second order if lag 2 etc. How many points are taken = order, simpler models prefered. Significance threhsold, how to get?

AR model uses partial AC, MA model uses full AC
https://www.youtube.com/watch?v=5Q5p6eVM7zM

Also known as Conditional correlation. Take out all intermediate effects.

https://machinelearningmastery.com/gentle-introduction-autocorrelation-partial-autocorrelation/

correlogram with Confidence intervals are drawn as a cone. By default, this is set to a 95% confidence interval, suggesting that correlation values outside of this code are very likely a correlation and not a statistical fluke.


The partial autocorrelation at lag k is the correlation that results after removing the effect of any correlations due to the terms at shorter lags.

The autocorrelation for an observation and an observation at a prior time step is comprised of both the direct correlation and indirect correlations. These indirect correlations are a linear function of the correlation of the observation, with observations at intervening time steps.

It is these indirect correlations that the partial autocorrelation function seeks to remove. Without going into the math, this is the intuition for the partial autocorrelation

how it is calcualted: https://timeseriesreasoning.com/contents/partial-auto-correlation/

