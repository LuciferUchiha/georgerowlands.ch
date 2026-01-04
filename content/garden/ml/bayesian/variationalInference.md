---
title: Variational Inference
type: docs
weight: 3
---

bayesian learning in general

showing that Z the normalizing constant in the posterior is intractable (partition function?)

same goes for if for predictions/inference of new data points we marginalize over all possible parameter settings

in the case of bayesian linear regression and gp regression we can compute these integrals in closed form due to conjugacy of gaussians

However, in general this is not possible -> we need to resort to approximate inference techniques

Running example of bayesian logistic regression which is classification between two classes but can be seen as bayesian linear regression with a bernoulli likelihood? Ber(y;sigmoid(w^T x)) Introduce it all and then show the problem of intractable posterior

we assume we can evaluate the joint distribution so likelihood times prior p(D, w) = p(D|w)p(w) but not the posterior p(w|D) = p(D,w)/p(D)

Variational inference seeks to approximate the
intractable distribution p by a simple one q that is “as
close as possible”

skip the method of markov chain monte carlo (MCMC) as this is not exam relevant just mention it as an alternative as a future topic that I might cover

laplace approximation: Gaussian approximation to the posterior, obtained from a second-order Taylor expansion around the posterior mode

show f(theta) = log p(theta | y) or given Data? what is this? 

and then show taylor expansion why is this around the mode? some things dissapear because 0 sind theta hat is mode, hessian is negative semi definitie? rewrite with big Lambda?

then get q(theta) = 1/Z exp(f hat of theta) = Gaussian at theta with mean theta hat and covariance Lambda^-1?

Image of laplace approximation

show now for the bayesian logistic regression example how this looks like with 1/Z not mattering for optimization

finally results in the logistic loss plus a regularizer which is just standard l2 regularized logistic regression which can be solved with SGD

normal SGD is quickly shown with a stochastic objective (expectation over x drawn from p) L(theta;x) and then the update step. Gradient is obtained via auto diff

SGD for logistic regression (with mini batch of size 1) Pick data point (x, y) uniformly at random from data D Compute probability of misclassification with current model and then gradien step theta(1 - lambda eta_t) + eta_t yx P(Y = -y | x, theta) where lambda is the regularization strength and eta_t the learning rate at time step t the probability is 1/(1+exp(y theta^T x))

I like the notation with x_1:n and y_1:n for data points    
show how to get the covariance start from big lambda = - nabla nabla log p(\theta hat | x_1:n, y_1:n) = ... = \sigma_p^-2 I + X^T diag(pi_i(1 -pi_i)) X where pi_i = sigmoid(theta hat^T x_i) = p(y_i = 1 | x_i, theta hat)

For covariance he shows a picture where the approximation has same shape but like different heights or is this the width of the gaussian? this then has a big impact on the fitting

laplaceCovariance.png

then show how to do predictions with the laplace approximation so predicting p(y* | x*, D) results in d dim integral again which is intractable but gets rewrrite using f^* = theta^T x* and then approximate f^* as Gaussian with which params? we then get a 1d integral again which can be efficiently computed for example via gauss-hermite quadrature (do not go into details of gauss-hermite quadrature just mention it as a numerical integration technique)

problem with laplace approximation is that Laplace approximation first greedily seeks the mode, and then matches the curvature, which can lead to poor (overcnfident) approximations if the posterior is not close to Gaussian

some of the plots have the MAP drawn in why is this the mode? maybe related back to bayesianLinearRegression.md

instead variational inference idea is to use a simple (tractable) distribution (variational family) that is parameterizable such as gaussians or gaussian with diagonal covariance and then approximate the true posterior by minimizing some measure of divergence between the two distributions

this meassure is often the KL divergence which is then minimized wrt the parameters of the variational distribution

definition and properties of KL divergence is it the forward or backward one. one is easier/better for vi?

show calculations of KL between two gaussians

shows some plot with the difference between forward and backward kl but doesnt make a lot of sense to me? its just how 2 different eliptical contours cover each other one is like the length of the elipsis is inside the other elipsis where as for the other its that the width is inside the other one?

introduce idea of entropy with definitions etc and also "suprise" then show entropy of a distribution and of a product distribution?

also entropy of a gaussian

then the goal is to minimize the kl divergence showing the full evaluations also writting it into a form with the maximizinh the expected log probability of theta given data + entropy of q

then can further rewrite this into argmx of E_q[log p(D, theta)] - Kl of q and p where p is our prior???

show then how starting from the evidence using jensens inequality we can derive the elbo (evidence lower bound) which is what we maximize in variational inference

Inference is then done by optimizing the elbo wrt the parameters of the variational distribution? Inference here means finding the best approximation to the posterior within the variational family from which we can then sample?

variationalVsLaplace.png

in image can see that vi tries to cover the whole posterior where as laplace just fits around the mode

finally show elbo for bayesian logistic regression with diagonal covariance gaussian variational distribution

to get the gradient of the elbo what is it and why do we need it? We need to differentiate through an expectation wrt to q which is tricky especially as q depends on the variational parameters we want to optimize wrt

Two approaches: reparameterization trick or rewriting it so we can use monte carlo estimates (score trick seen in rl and diffusion models)?

the idea of the reparameterization trick in general with change of variables. show and explain the ideas of change of variablse as ive never seen this.

then we basically get that q(\theta | lambda) = phi(epsilon) | \nabla_\epsilon g(lambda, epsilon) | ^-1 where epsilon is random variable from phi and theta is g(lambda, epsilon) and g is some invertible function

then we can rewrite the expectation of theta ~ q as expectation over epsilon ~ phi of f(g(lambda, epsilon))

now the expectation is wrt to a distribution that does not depend on lambda so we can now differentiate inside the expectation so we can obtain stochastic gradients of the elbo wrt lambda via

nabla_lambda E_{theta ~ q}[f(theta)] = E_{epsilon ~ phi}[nabla_lambda f(g(lambda, epsilon))]

show for gaussian variational distribution how the reparameterization trick works with no g? why? covariance becomes CC^T from cholesky decomposition and lambda = (mu, C) then theta = mu + C epsilon with epsilon ~ N(0, I) so phi(epsilon) = N(epsilon; 0, I)

g(epsilon, C, mu) = g(lambda, epsilon) = mu + C epsilon

then it holds that epsilon = C^-1 (theta - mu) and the determinant of the jacobian is |C^-1| = 1/|C| why? so then phi(epsilon) = q(theta | mu, C) |C|? without loss of generality can assume that C is lower triangular or diagonalfrom cholesky decomposition

finally show how reparameterizing the elbo for bayesian logistic regression with gaussian variational distribution looks like

black box stochastic variational inference?