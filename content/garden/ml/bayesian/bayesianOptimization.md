---
title: Bayesian Optimization
type: docs
weight: 6
---

Numerous applications require trading experimentation
(exploration) and optimization (exploitation) as also discussed in reinforcement learning.

#alternatives >> #trials
▪ experiments are noisy & expensive
▪ similar alternatives have similar performance

Bayesian optimization is a strategy for global optimization of noisy, expensive black-box functions. It is particularly useful when the objective function lacks a closed-form expression and is costly to evaluate. Relate back to active learning and acquisition functions. process of sequentially picking x1 , x2, ... x_t to find the max/ global optimum of an unknown function f(x) based on previous observations with minimal samples

how does this also relate to hyperparameter optimization of machine learning models?

multi armed bandit problem. each arm has unknown reward distribution. goal is to function over time by choosing which arms to pull. trade off exploration of uncertain arms and exploitation of arms known to give high rewards. What is the precise definition, the mathematical formulation of the problem? How is it related to bayesian optimization. It is also a special case of reinforcement learning where there is only one state and multiple actions?

after each selection we observer a noisy sample of the function. Regret and cumulative regret definitions and equations? also instant regret?
sublinear if R_n / n -> 0 as n -> infinity what does this mean. implies that the max t of f(x_t) approaches f(x^*) as n increases? where x^* is the global optimum

Gaussian process bandit optimization

average regret? pick samples to minimize average regret. so goes to 0? Difference to cumulative regret?

upper confindence bound (GB-UCB) algorithm. upper conficence sampling? Focus exploration on plausible maximizers
(upper confidence bound ≥ best lower bound)

What is upper confidence bound= need to pick beta_t to find max x of mu_{t-1}(x) + beta_t * sigma_{t-1}(x), where does this come from? explain the intuition. why does this work? how to pick beta_t? in general not convex optimization problem but can be solved using gradient ascent

Pick input that maximizes upper confidence bound. Naturally trades off exploration and exploitation
Only picks plausible maximizers

bayesian regret of GP-UCB. if we assume that the function is sampled from a GP with known kernel and we pick the correct beta_t then the regret which regert is O^*(sqrt(gamma_n/n)) where gamma_n is the maximum information gain after n observations. explain what this means. especially the O^* notation? so maximum information gain determines the regret? what is the intuition behind this? gamma_n is bounded by o(T) by submodularity of mutual information often decays faster?

Submodularity of mutual info. yields data-dependent bounds

information gain bounds for common kernels? why? what does this have to do with regret? linear O(d log n), squared exponential O((log n)^{d+1}), Matern with nu > 1/2 O(n^{d/2nu + d} (log n)^{2nu/2nu + d}) 

Guarantees sublinear regret / convergence

Regret depends on how quickly we can gain information

frequentist regret for GP-UCB. what does mean? difference to bayesian regret? O^*(sqrt(beta_n gamma_n/n)) and O(||f||_k + gamma_n log^3 n) where ||f||_k is the RKHS norm of f under the kernel k whihc is complexity of f. explain what this means. why does the RKHS norm appear here? what is the intuition behind this?

Alternative acquisition criteria

expected improvement

Thompson sampling

this is also used in other things like reinforcment learning? What is the general idea and benefit of thompson sampling?
idea explanation and intution? at each iteration sample a function from the posterior of p(f | D) and then pick the point that maximizes this sampled function. The randomness in sampling functions naturally balances exploration and exploitation. It is possible to establish regret bounds for Thompson
sampling similar to those for UCB

Hyperparameter tuning with Bayesian optimization

In principle can alternate learning hyperparameters
(e.g., via marginal likelihood maximization) and
observation selection

In practice, there is a specific danger of overfitting
▪ Data sets in BO / active learning are small
▪ Data points are selected dependent on prior observations
▪ Potential solutions
▪ “Being Bayesian” about the hyperparameters (i.e., placing a
hyperprior on them, and marginalizing them out)
▪ Occasionally simply selecting some points at random