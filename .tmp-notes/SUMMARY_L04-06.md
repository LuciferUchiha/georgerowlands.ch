# Matrix Completion / Collaborative Filtering source summary (CIL 2026, Lectures 4-6)

Target file: `content/garden/ml/collaborativeFiltering.md` (replace the single TODO block).
Notation decision: **bold \mathbf** (match autoencoders.md + eigendecomposition.md).
Matrix A in R^{n x m}: n users (rows), m items (cols). rank k. A ~ U V^T, U in R^{n x k}, V in R^{m x k}.
u_i, v_j in R^k = latent factors (rows of U and V). Singular values sigma_i, sigma_1>=sigma_2>=...
Observation matrix Omega in {0,1}^{n x m}, omega_ij. Projection Pi_Omega(R) = R (hadamard) Omega.

## Lecture 4: CF, rank-1 model, convexity, gradient dynamics, eigenvector solution
- Matrix completion: sparse A, fill missing entries. Netflix (rows users, cols items, 1-5 stars, <1% observed).
- Recommender systems; collaborative filtering (exploit collective data, generalize across users/items) vs content-based; special case of "algorithmic selection".
- No-goal: if entries iid independent, P(A)=prod p(a_ij), mutual info I(a_ij,a_kl)=0, no info one entry about another => ill-posed.
- Minimal dependence: what makes a matrix a matrix = identity of rows & cols. Entries in same row/col not independent.
- Conditional independence: a_ij _|_ {a_kl: k!=i, l!=j} | ({a_il: l!=j} same row) U ({a_kj: k!=i} same col). Does NOT imply stochastic (marginal) indep.
- Conditional vs marginal: partial observation => conditional indep does NOT imply marginal indep; effective indirect couplings between all entries.
- Formalization: rating matrix A in R^{n x m}; observation matrix Omega in {0,1}^{n x m}; sum omega_ij = #observed.
- Preprocessing: row mean mu_row_i = sum_j omega_ij a_ij / max(1, sum_j omega_ij); col mean analog. Centering subtracts rating bias (a user very positive/negative). Variance normalization to 1 => z-score Z=(X-mu)/sigma. Rows OR cols as reference population. Sarwar et al 2001: normalizing items (transposed view) more effective than per-user. "try diff normalizations for project."
- Plan: A ~ U V^T, U in R^{n x k}, V in R^{m x k}. Today fully observed: min_{U,V} 1/2 ||A - UV^T||_F^2. Rank-1: A ~ u v^T. 1-dim: a ~ uv. Next lectures incomplete: min 1/2 ||Pi_Omega(A - UV^T)||_F^2.
- Outer product / bi-linear model: a_ij ~ u_i v_j. Non-identifiability: lambda!=0: u->lambda u, v->(1/lambda)v.
- Objective: phi(u,v) = 1/2 ||Pi_Omega(A - uv^T)||_F^2. ||R||_F^2 = ||vec(R)||_2^2 = sum r_ij^2.
- rank-1 = outer product; every rank-1 matrix is outer product of two vectors & vice versa.
- Minimal minimal (scalar): phi=1/2(a-uv)^2. dphi/du=(a-uv)(-v)=delta v; dphi/dv=(a-uv)(-u)=delta u; delta:=uv-a.
- Gradient field a=1: minima = hyperbola uv=a (two branches). origin = isolated critical pt = saddle. GD converges to hyperbola.
- Hessian scalar: [[v^2, 2uv-a],[2uv-a, u^2]]; at origin [[0,-a],[-a,0]]. char poly lambda^2-a^2=(lambda-a)(lambda+a); eigenvalues +-a => indefinite => non-convex (except a=0).
- Convexity: set convex if segment in set. f convex: f(tx+(1-t)y)<=tf(x)+(1-t)f(y). 1st order: f(x)>=f(y)+grad f(y).(x-y). 2nd order: Hessian PSD on interior. Convexity = demarcation tractable/intractable. Matrix completion non-convex but analyzable/approximable, exact for special cases.
- Rank-1 matrix gradients: R:=(uv^T - A); grad_u phi = R v; grad_v phi = R^T u. (Frobenius rule grad_R 1/2||R||_F^2 = R + chain rule.)
- Hessian map: [[||v||^2 I_n, 2uv^T-A],[(2uv^T-A)^T, ||u||^2 I_m]]; at origin [[0,-A],[-A^T,0]] in R^{(n+m)x(n+m)}. Square, symmetric, trace=0 => sum eigenvalues=0 => can't be PSD unless A=0. Non-convex for all n,m>=1.
- Gradient flow via ODE (small step limit). Balanced init u0=v0 (u,v evolve identically). x=uv=u^2. du/dt=-v(uv-a); dx/dt=du^2/dt=-2uv(uv-a)=-2x(x-a).
- ODE solution x(t)=a + (ac-a^2)/(c e^{2at} + a - c), c=x(0). For a=1,c small: sigmoid-like exponential growth then saturate at a. Init matters (c).
- Fully observed rank-1 rewrite: ||R||_F^2=tr(R^T R). 1/2||A-uv^T||_F^2 = 1/2 tr((A-uv^T)^T(A-uv^T)) = 1/2 tr(A^TA - vu^TA - A^Tuv^T + vu^Tuv^T). tr(A^TA)=const; the two middle = -u^TAv each (scalar = its trace, cyclic); tr(vu^Tuv^T)=tr(u^Tu v^Tv)=||u||^2||v||^2. => phi = 1/2||u||^2||v||^2 - u^TAv (+ const 1/2||A||_F^2).
- Directionality determined by -u^TAv term. u=c1 u~, v=c2 v~, ||u~||=||v~||=1, c=c1 c2.
- Optimality: max u^TAv s.t. ||u||=||v||=1. Lagrangian L=u^TAv - mu1(u.u-1) - mu2(v.v-1). grad_u L = Av - 2mu1 u = 0 => u = Av/||Av||. Similarly v=A^Tu/||A^Tu||.
- Eigenvector eqs: u prop (AA^T)u, v prop (A^TA)v. u = principal eigenvector of AA^T; v = principal eigenvector of A^TA. Use power iteration. A ~ sigma_1 u_1 v_1^T (sigma_1 = sqrt(principal eigenvalue of AA^T = A^TA)). Generalization to k: SVD.
- Self-check: eigenvalues of A^TA and AA^T are the same. rank(UV^T) with k indep cols = k (<= min(m,n)).

## Lecture 5: SVD, norms, Eckart-Young, ALS
- SVD theorem: any A in R^{n x m} = U Sigma V^T, Sigma in R^{n x m} rect-diagonal diag(sigma_1..sigma_min(n,m)), sigma_i>=sigma_{i+1}>=0; U in R^{n x n}, V in R^{m x m} orthogonal. Singular values unique; singular vectors arbitrary sign; multiplicities => unique subspaces (not unique vectors). [link eigendecomposition.md SVD section]
- Geometry: A v_i = sigma_i u_i. "largest stretch" of unit vector = sigma_1 (spectral norm).
- Reduced/compact SVD: prune zero singular values.
- SVD & eigendecomposition: (1) A square symmetric => U,V equal cols up to sign. (2) A also PSD => SVD = eigendecomposition. Any A: AA^T = U Sigma Sigma^T U^T = U diag(sigma_1^2..sigma_n^2) U^T; A^TA = V Sigma^T Sigma V^T = V diag(sigma_1^2..sigma_m^2) V^T. Singular values = sqrt eigenvalues of AA^T (or A^TA). => can get SVD from eigendecomp & vice versa.
- SVD & PCA: SVD of data matrix gives principal eigenvectors of covariance (PCA). [link pca.md]
- SVD & Frobenius: ||A||_F^2 = sum_{i=1}^{min(n,m)} sigma_i^2. Proof: ||A||_F^2=tr(A^TA)=tr(V Sigma^T U^T U Sigma V^T)=tr(Sigma^T Sigma)=sum sigma_i^2 (U^TU=I, cyclic+V^TV=I).
- SVD & spectral norm: ||A||_2 := sup{||Ax||: ||x||=1} = sigma_1. Proof: orthogonal preserve norm; sup||U Sigma V^T x|| = sup||Sigma V^T x|| = sup||Sigma z||(||z||=1) = ||Sigma||_2 = sigma_1 (2-norm of diagonal = largest |entry|).
- Eckart-Young: A_k := U diag(sigma_1..sigma_k) V^T in argmin{||A-B||_F : rank(B)<=k}. Best rank-k = truncate SVD. Read off for any k. [link eigendecomposition.md]
- Eckart-Young corollary: ||A - A_k||_F^2 = sum_{i=k+1}^{rank(A)} sigma_i^2 (error = discarded singular values). A-A_k = U diag(0..0,sigma_{k+1}..) V^T.
- Optimal spectral norm approx: A_k also in argmin{||A-B||_2: rank(B)<=k}; error = sigma_{k+1} (self-check, derive).
- SVD & matrix completion: rank-k A ~ sum_{i=1}^k u_i v_i^T = additive superposition of k rank-1. Fully observed => SVD constructive: A ~ sum_{i=1}^k sigma_i u_i v_i^T. Low-rank approx non-convex even fully observed; SVD computable O(min(nm^2,mn^2)) => solvable non-convex problem.
- Matrix completion (incomplete): SVD w/ imputation (fill row/col mean, then SVD) = BAD. SVD not directly applicable.
- NP-hardness: min 1/2||Pi_Omega(A-B)||_F^2 s.t. rank(B)<=k. Weighted Frobenius min sum w_ij(a_ij-b_ij)^2, rank(B)=k (special case w_ij=omega_ij in {0,1}). NP-hard even rank-1! Fully observed is special.
- ALS: factored param U in R^{n x k}, V in R^{k x m} (lecture; I use V in R^{m x k}, V^T). Regularized non-convex objective: phi(U,V)=1/2||Pi_Omega(A-UV)||_F^2 + lambda/2(||U||_F^2 + ||V||_F^2), lambda>0. Reg => numerical+statistical stability, invertibility.
- Degree-4 polynomial; monomials omega_ij u_ir v_rj u_is v_sj (1<=r,s<=k), omega_ij u_ir v_rj, u_ir^2, v_rj^2. Higher-order terms involve exactly ONE row index i of U and ONE col index j of V => bipartite graph between rows of U and cols of V. Row u_i doesn't interact w/ u_j (i!=j); col v_i doesn't interact w/ v_j.
- Separable subproblem for column v_j (analog row u_i): phi_U(v_j) = 1/2 v_j^T (sum_{i=1}^n omega_ij u_i u_i^T + lambda I_k) v_j - (sum_{i=1}^n omega_ij a_ij u_i^T) v_j. [NOTE: lecture wrote sum_{i=1}^k, that is a typo; correct is sum over all n rows i.]
- Fix U => least squares: v_j* = (sum_{i=1}^n omega_ij u_i u_i^T + lambda I_k)^{-1} (sum_{i=1}^n omega_ij a_ij u_i). Inverse of k x k (k small); reg ensures invertibility.
- ALS algorithm: V_{t+1} = argmin_V phi(U_t,V); U_{t+1}=argmin_U phi(U,V_{t+1}). Highly parallel (n or m independent LS subproblems of dim k). Monotone decrease of objective; converges to fixed point; first-order optimal grad phi=0 but may not be global min (saddle/local). ALS more efficient than GD. Practical: add new user/item = one extra LS problem.

## Lecture 6: projection algorithms, nuclear norm, randomized SVD, exact recovery
- Parameterization (ALS, built-in rank constraint) vs projection methods (unconstrained, project to low rank/norm; use SVD subroutine).
- Singular Value Projection (SVP, Jain et al 2010): projected gradient descent. A^0=0; A^{t+1} = [A^t + eta Pi_Omega(A - A^t)]_k, where [.]_k = best rank-k via SVD, eta>0. Inner = sparse gradient step; outer = SVD projection to rank-k. loss ℓ(B)=1/2||Pi_Omega(A-B)||_F^2 convex in B; grad_B = -Pi_Omega(A-B); non-convexity from rank-k projection.
- Nuclear norm relaxation: convex relaxation of rank constraint. Nuclear (trace) norm ||A||_* = sum_{i=1}^{rank} sigma_i = ||sigma(A)||_1. ||A||_F=||sigma(A)||_2. Generalizes 1-norm-for-sparsity (Lasso): sparse singular values = low rank.
- Convex envelope: largest convex g with g<=f. Theorem (Fazel 2001): convex envelope of rank(A) on {A: ||A||_2<=1} is ||A||_*. Motivates nuclear norm as convex surrogate for rank.
- Relaxation chain: min 1/2||Pi_Omega(A-B)||_F^2 s.t. rank(B)<=k (non-convex domain) -> Lagrange: min 1/2||Pi_Omega(A-B)||_F^2 + mu rank(B) (non-convex loss) -> surrogate min ||B||_* s.t. Pi_Omega(A-B)=0 -> relaxed min tau||B||_* + 1/2||B||_F^2 s.t. Pi_Omega(A-B)=0. As tau->inf, relaxed -> surrogate.
- SVT (Cai et al 2010): shrink_tau(A) := U diag((sigma_i - tau)_+) V^T = argmin_B (1/2||A-B||_F^2 + tau||B||_*). Shrink each sigma by tau, clip at 0. Reduces rank monotonically with tau.
- SVT algorithm: A^0=0; A^{t+1} = A^t + eta_t Pi_Omega(A - shrink_tau(A^t)) (Omega-sparse). Converges (suitable eta_t) to argmin tau||B||_* + 1/2||B||_F^2 s.t. Pi_Omega(A-B)=0: agrees on observed, minimal mixed Frobenius/nuclear. SVD on sparse iterate is fast.
- Conclusion: convex relaxation maintains sparse iterate; SVP maintains rank-k matrix. Both better than dense.
- Randomized SVD (Halko et al 2011): find semi-orthogonal Q in R^{n x 2k}, A ~ QQ^T A. SVD of small B=Q^TA = Utilde Sigmatilde Vtilde^T; extend A ~ (Q Utilde) Sigmatilde Vtilde^T. Find Q: random Gaussian R in R^{m x 2k}; Y=(AA^T)^q A R (e.g q=2); orthonormalize (Gram-Schmidt) => Q. (power-iteration captures dominant subspace; small errors ok since followed by GD.)
- Exact reconstruction: rank-k A, conditions for exact recovery w.h.p.?
- Degrees of freedom: count from SVD. k singular values + left/right singular vectors (A in R^{n x n}): (n-1)+(n-2)+...+(n-k) = nk - k(k+1)/2 free params per side (1st vector unit => n-1, 2nd unit+orth => n-2, etc). Total S >= k + 2(nk - k(k+1)/2) = 2nk - k^2. Rank-k n x n not reconstructable if observed S < 2nk - k^2.
- Coupon collector: n coupons, E[t_i]=n/(n-i+1) geometric; E[T]=sum = n H_n, H_n=log n + O(1). => need ~ n log n random samples just to hit every row/col. (motivates n log n factor.)
- Impossibility / incoherence motivation: if u_1=e_i, v_1=e_j in SVD, must sample a_ij to recover sigma_1. Recovery impossible without regularity: info must be sufficiently spread out.
- Incoherence (Candes-Tao 2010): P=sum u_i u_i^T (col space proj), Q=sum v_i v_i^T (row space proj), E=sum u_i v_i^T. Conditions: |p_ij|,|q_ij| (i!=j) <= mu sqrt(k/n); |p_ii - k/n|,|q_ii-k/n| <= mu sqrt(k/n); |e_ij| <= mu sqrt(k/n). (technical, cf compressed sensing.)
- Reconstruction theorem (Candes-Tao 2010): rank-k A incoherent with mu>=1, S random samples. Universal C: if S >= C mu^2 n k (log n)^6, then w.p. >= 1 - n^{-3}, A = argmin_B ||B||_* s.t. Pi_Omega(B)=Pi_Omega(A). I.e. nuclear norm minimization recovers A exactly.

## Lecture 4 prep = refresh background (cond. indep, info theory, eigen, PSD, Hessian) -> already in existing notes (active learning MI, eigendecomposition, pca). Use lightly.

## Existing notes to link
- /garden/ml/computerVision/autoencoders (linear AE, rank constraint, P=VW, PCA theorem, power method, SGD, trace gradient)
- /garden/maths/linearAlgebra/pca (variance, covariance, principal eigenvectors)
- /garden/maths/linearAlgebra/eigendecomposition (#singular-value-decomposition, #low-rank-approximation Eckart-Young, #the-rayleigh-quotient, spectral thm, PSD, trace, #positive-definite-and-positive-semi-definite-matrices)
- /garden/maths/linearAlgebra/moorePenrose, /matrixRanks, /hadamardProduct, /projectionsLS, /orthogonality
- /garden/ml/gradientDescent ; /garden/ml/bayesian/activelearning/#information-theory (mutual information, entropy)
- /garden/maths/probabilityStatistics ... (geometric distribution / coupon collector) if exists
