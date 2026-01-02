---
title: Bayesian Learning
type: docs
weight: 1
---

probabilistic inference

bayesian inference with priors and posteriors

conjugate priors

Reminder of MLE and then MAP as MLE with prior as regularization

bayesian linear regression

continos model assumptions?
seen linear regresion using OLS (not connected to probabilistic view yet)

bayesian graphical model introduction and then show the graph for bayesian linear regression

MLE results in linear regression?

OLSE gauss markov theorem?

Then use MAP with gaussian prior to get ridge regression

Ridge regression can be viewed as approximating the full
posterior by (placing all mass on) its mode???

ridge predicts using MAP estimate of weights, Bayesian linear regression predicts using full posterior predictive distribution i.e average over all possible weights weighted by their posterior probability?

somewhere epistemic vs aleatoric uncertainty

lasso regression as laplace prior???

How do we choose hyperparameters like the prior variance or noise variance? one way would be to use cross validation but in bayesian approach we can also do evidence maximization / type II MLE where we treat these hyperparameters as parameters to be estimated by maximizing the marginal likelihood (evidence) of the data integrating out the model parameters. Define evidence and show how to compute it for bayesian linear regression. then show how to optimize it w.r.t hyperparameters?

recursive bayesian learning(Online Bayesian Linear Regression) this is where we update our posterior as new data comes in rather than recomputing from scratch each time what performance benefits does it give us?

If d is large, computing the inverse every round is very expensive. Can you use the recursive structure
you found in the previous question to reduce the computational complexity of every round to O(d^2)? Shermann-Morrison formula or woodbury identity?


Can apply linear method (like BLR) on nonlinearly
transformed data. However, computational cost increases
with dimensionality of the feature space!

kernel trick Express problem s.t. it only depends on inner products
§ Replace inner products by kernels

weight vs function space view?

infinite domains resulting in gaussian processes

so much to cover here!