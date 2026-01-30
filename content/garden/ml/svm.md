---
title: Support Vector Machines
type: docs
weight: 8
---

Heavily linked to convex optimization.

goal is to find a hyperplane that separates classes with the maximum margin criterion.

margin, support vectors, linear separability

rpojections onto the normal vector of the hyperplane? show calculations, l+ - l- what are these l's?

maximize w, w_0 2 m(w, w_0) s.t. y_i (w^T x_i + w_0) > 0? using dual to strengthen?

normalization trick

formailization of the primal optimization problem

we can rewrite it using dual?

slater's condition holds for svm, strong duality holds

dual does not depend on the dimension of the data, only the number of data points unlike the primaly which can become intractable in high dimensions

complementary slackness conditions from the dual.

## Soft Margin

what if not linearly separable? introduce slack variables xi_i >= 0

define penalty per point C is a hyperparameter

Interpretation of C as a regularization parameter. C is 1 means we allow 1 misclassification worth of slack. if it is infinity we do hard margin.? But if C is 0 we allow infinite slack? and C is 2 means we allow 2 misclassifications worth of slack?

## Kernel SVM

Soft margin is a hacky approach to dealing with non linearly separable data or is it actually related to kernels?

non linear transformations to higher dimensional space to make data linearly separable.

link to bayesian linear regression with feature maps using kernel trick for polynomial rather then linear decision boundaries.

sepcifically RBF. we dont actually need the kenerl function just the dot products?

training and classification using kernel functions.

## Multi-class SVM

hard margin version and soft margin version 
