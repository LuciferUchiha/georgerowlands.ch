---
title: Multivariate Gaussian
type: docs
weight: 8
---

problem with high-dimensional distributions. If we have n binar yvariables (each can take on 2 values), then we need 2^n-1 parameters to fully specify the joint distribution. This grows exponentially with n, making it impractical for large n.

another is computing marginals and conditionals. Is this the part with the Z normalization constant? Is this the same as patition function?

Show the above problems explicitly, ideally even with some example or code snippets not sure what makes sense.

Instead we prefer to use distributions that have nice properties that make them easier to work with. One such distribution is the Gaussian distribution. In particular the multivariate Gaussian distribution as we have already seen in the 1-dimensional case the Gaussian distribution is defined by just 2 parameters (mean and variance) and has the nice properties of?

definition of multivariate Gaussian distribution

now only need $O(n^2)$ parameters to define the covariance matrix rather than $O(2^n)$ parameters to define the full joint distribution. How?