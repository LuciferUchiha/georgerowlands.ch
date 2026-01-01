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

covariance matrix encodes relationships between variables, interpretation etc. other names?

positive semi-definite property, what if it is symmetric or positive definite? isotropic Gaussian?

gaussians are independent if they are uncorrelated, why? what does mean in this context?

gaussian random vector, and then taking a subset of the vector to get marginals and conditionals, how to do this? how are the parameters of the marginals and conditionals related to the original parameters and defined?

multiples of gaussians are also gaussian, linear combinations of gaussians are also gaussian, why? what does this mean in practice? can summarize this with affine transformations of gaussians, is this when we multiply by a matrix?

sums of gaussians are also gaussian, why? what does this mean in practice?

if X and Y are jointly gaussian, then the conditional distribution of X given Y is also gaussian, why? what does this mean in practice? mean linearity depends on y?

Therefore random variable X can be seen as a linear function of Y plus some gaussian noise, what does this mean in practice? The converse als holds. derivation and formulas specifically for mean and covariance of the conditional distribution.