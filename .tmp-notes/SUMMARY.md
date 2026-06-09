# Linear Autoencoder / PCA source material summary (CIL 2026, Lectures 1-3 + prep)

Temporary working notes. Source PDFs in `resources/material/`. Goal: write the linear
autoencoder study notes in `content/garden/ml/computerVision/autoencoders.md` (replacing the
big TODO block), matching existing note style and linking to existing linear algebra / PCA notes.

## Notation reconciliation (IMPORTANT)

Course uses: n = ambient dim, m = latent/bottleneck dim, W in R^{m x n} encoder, V in R^{n x m}
decoder, P = VW, Sigma = E[xx^T] covariance, s/N = number of samples.

Existing notes (pca.md, autoencoders.md) use: **D = ambient dim, d = latent dim, n = number of
samples**, x_i data points, S = sample covariance. => MATCH EXISTING NOTES. So:
- Encoder F: R^D -> R^d, z = Wx, W in R^{d x D}
- Decoder G: R^d -> R^D, xhat = Vz, V in R^{D x d}
- Reconstruction map P = VW in R^{D x D}, rank(P) <= d
- Risk R(P) = 1/2 E||x - Px||^2
- Covariance S = E[xx^T] (centered); sample Shat = 1/(n-1)(X - Xbar)(X-Xbar)^T
- Notes use non-bold x, U, P, S (NOT \mathbf), matching pca.md/autoencoders.md actual style.

## Lecture 1: setup
- AE as DNN. Data law x ~ nu (prob measure). Sample set S = {x_i iid ~ nu}.
- E_nu[f] true expectation; E_S[f] empirical (Monte Carlo / sample average).
- Encoder F: R^D->R^d, decoder G: R^d->R^D. Reconstruction map G∘F: R^D->R^D.
- Ideally G∘F = id (unachievable unless data distribution is intrinsically dim d).
- Loss / distortion measure l(x, xhat). Quadratic loss l = 1/2 ||x - xhat||^2 (convenient choice;
  could use log loss / cross-entropy for binary data). 
- Risk = average loss. Empirical risk E_S[l] = 1/(2s) sum ||x_t - (G∘F)(x_t)||^2.
  New-data risk E_nu[l] = integral l(x, (G∘F)(x)) d nu(x) (Lebesgue integral). [George note: "don't have this"]
- Linear AE: F(x)=Wx, G(z)=Vz. Objective R(W,V)=R(P:=VW)=E[1/2||x-Px||^2].
- Composing linear maps = matrix mult; no expressivity gain from stacking linear layers.
- AFFINE = LINEAR for centered data + squared loss. Proof: affine F(x)=Wx+a, G(z)=Vz+b =>
  (G∘F)(x) = VWx + c, c=b+Va. E||x-(Px+c)||^2 = E||x-Px||^2 + ||c||^2 - 2<c, Ex - PEx>.
  Centered => Ex=0 so the cross term =0, leaving E||x-Px||^2 + ||c||^2 >= E||x-Px||^2. So c=0 optimal.
  (Equivalent to "include bias in weight matrix" / centering.)
- Identifiability: (1) is optimal P unique? YES (George note). (2) is parameterization P=VW unique?
  NO: VW = V(A A^{-1})W = (VA)(A^{-1}W) for any invertible A in R^{d x d}. => Don't over-interpret W,V.
- Rank constraint: rank(P)=min(rank V, rank W) <= min(D,d)=d. rank(A)=dim(im(A)). Bottleneck = rank constraint.

## Lecture 2: projection is optimal; representation of P
- Goal min R(P)=1/2 E||x-Px||^2 s.t. rank(P)<=d. Rank constraint + linearity => im(P) is linear
  subspace U of R^D, dim <= d. Split: (1) find optimal subspace U, (2) find optimal map into given U.
- Given U: optimal map Pi_U(x) = argmin_{x' in U} ||x - x'|| is the ORTHOGONAL PROJECTION onto U;
  exists and is unique.
- Projection defn: linear P:V->V is projection onto U if (1) P(x) in U, (2) idempotent P(P(x))=P(x).
  Orthogonal if ker(P) ⊥ im(P)  <=>  <Px,y>=<x,Py> (self-adjointness/symmetry). Else oblique.
- Pi_U proofs: idempotent (Pi_U(u)=u for u in U); orthogonal (Pi_U(x)-x in U^perp, by contradiction
  using ||a+b||^2 with cross term 0); linear (homogeneity + additivity). => Pi_U is orthogonal projection.
- Pi_U(x) = Px (projection matrix). Optimal linear AE represents a projection. Properties of P:
  P^2=P (idempotent), P=P^T (symmetric/self-adjoint).
- Matrix rep: projection to a line (unit u): P_u = u u^T, ||u||=1; P_u x = <u,x> u. Inner product =
  signed length; ||P_u x|| = |<u,x>| <= ||x|| (non-expansive).
- Orthonormal basis {u_1..u_d} of U: P = U U^T, U=[u_1..u_d], Px = sum <u_i,x> u_i. Sum of d rank-1
  matrices. Symmetry P^T = (UU^T)^T = UU^T. Idempotency PP = U(U^T U)U^T = U I_d U^T = UU^T (U^T U = I_d).
- AE projection matrix without parameter tying: P=VW is a projection but NOT necessarily orthonormal form.
  With tying V=W^T => P=VV^T semi-orthogonal. Without tying, for lin indep cols V=[v_1..v_d]:
  P = V V^+, V^+ = (V^T V)^{-1} V^T = LEFT Moore-Penrose pseudo-inverse. P=VV^+ is orthogonal projection
  onto U=span(v_i). Proof idempotency P^2=V(V^TV)^{-1}(V^TV)(V^TV)^{-1}V^T=P; PV=V; symmetry P^T=P.
  Corollary: for any V, V^+ = argmin_W R(W,V), i.e. given decoder V, optimal encoder W is left pseudo-inverse.
- numpy digits example (sklearn load_digits, centered, m=20 bottleneck): random V -> MSE 13.48;
  first m data columns -> MSE 4.40; eigs(X X^T, k=m) -> MSE 1.98 (optimal subspace).
- Section 4 PCA: which U optimal? Bring out total variance: using self-adjoint + idempotent,
  R(P)=1/2 E||x-Px||^2 = 1/2 E||x||^2 - 1/2 E||Px||^2. For centered data:
  R(P) = 1/2 (Var[x] - Var[Px]), Var[x]=E||x||^2 const, Var[Px]=E||Px||^2 (since EPx=PEx=0).
  => minimizing reconstruction loss = MAXIMIZING Var[Px].
- Sufficient statistic: S=E[xx^T] is sufficient statistic for R(P). R(P) = -1/2 tr(P E[xx^T]).
  Proof via trace: <x,y>=x^Ty=tr(x^Ty); trace linear (commutes with E); cyclic tr(ABC)=tr(CAB).
  Var[Px]=E<x,Px>=E tr(x^T P x)=E tr(P x x^T)=tr(P E[xx^T]). [George: cyclic NOT any permutation; for exam]
- Sample covariance Shat = 1/(N-1)(X-Xbar)^T(X-Xbar). Optimal projection fully determined by E[xx^T]
  (+ E[x] for centering) = sufficient statistics.
- PCA Theorem (stated, proved in L3): rank-d projection optimizing squared reconstruction loss on
  centered data is the projection onto the span of the d principal (top) eigenvectors of E[xx^T].

## Lecture 3: PCA theorem proof + algorithms
- Want argmin_{rank(P)<=d} R(P) = U U^T where U = d principal orthonormal eigenvectors of S=E[xx^T].
- DIAGONAL CASE: S = Lambda = diag(lambda_1..lambda_D), lambda_1>=...>=lambda_D>=0.
  Standard basis vectors e_1..e_D are eigenvectors (Lambda e_j = lambda_j e_j), ordered by principality.
  Optimal P = sum_{i=1}^d e_i e_i^T = [[I_d,0],[0,0]]. "Project onto the d axes of largest variance."
  Multiplicity: if eigenvalues tie, any orthonormal basis of that eigenspace works (non-unique).
  Proof: maximize Var[Px]=tr(UU^T Lambda) = sum_i lambda_i (sum_j u_ij^2), s.t. U^T U = I_d.
  Let gamma_i = sum_{j=1}^d u_ij^2. (1) gamma_i <= 1 (else ||Pe_i||^2>1 contradicts non-expansiveness).
  (2) sum_i gamma_i = ||U||_F^2 = tr(U^T U)=tr(I_d)=d. (3) since lambda ordered desc, optimum puts
  gamma_i=1 for i=1..d => P=[[I_d,0],[0,0]]. Max value Var[Px] = sum_{i=1}^d lambda_i.
- GENERAL CASE via Spectral Theorem: symmetric PSD S = Q Lambda Q^T, Lambda desc, Q orthogonal
  (ordered eigenvectors). PCA theorem: P = U U^T, U = Q[I_d;0] = first d columns of Q.
  Proof: Var(Px)=tr(P S)=tr(UU^T Q Lambda Q^T)=tr((Q^T U)(Q^T U)^T Lambda) [cyclic], maximized when
  Q^T U = [I_d;0] => U = Q[I_d;0] = first d eigenvectors of Q. 
- IMPORTANT: linear AE finds the principal SUBSPACE span(Q[I_d;0]) but not necessarily the PCA basis
  (individual eigenvectors), since P=VV^+ uses any basis of U.
- Why eigenvectors: principal unit eigenvector u of symmetric A maximizes ||Av|| over unit v
  (spectral norm); eigenvectors of symmetric matrices with distinct eigenvalues are orthogonal:
  lambda<u,u'> = <Au,u'> = <u,Au'> = lambda'<u,u'> => (lambda-lambda')<u,u'>=0 => orthogonal.
- LEARNING ALGORITHMS:
  1. Matrix method: numpy np.linalg.eigh on covariance (symmetric), take last d columns. Digits MSE 1.984.
     Complexity: covariance estimation O(N D^2); eig decomposition O(D^3). Bad if D large, d<<D.
  2. POWER METHOD for principal eigenvector(s). v^{(0)} ~ N(0,I), v^{(t+1)} = A v^{(t)} / ||A v^{(t)}||.
     Converges to u_1 if <v0,u1> != 0 and lambda_1 > lambda_2. Proof: in eigenbasis v0=sum alpha_i u_i,
     v^{(k)} ∝ alpha_1 u_1 + sum_{i>1} alpha_i (lambda_i/lambda_1)^k u_i, ratios <1 -> 0. Complexity O(T D^2).
     Iterative deflation for next eigenvectors: A_2 = A - lambda_1 q_1 q_1^T (zeros out top eigenvalue),
     reapply power method. d eigenvectors: O(T d D^2) + covariance O(N D^2).
  3. (STOCHASTIC) GRADIENT DESCENT. Matrix gradient of l(x;P)=1/2||x-Px||^2:
     l = 1/2||x||^2 - tr(P x x^T) + 1/2 tr(P^T P x x^T). Trace deriv rules: d/dA tr(AB)=B^T,
     d/dA tr(A^T A B) = AB^T + AB. => grad_P l = (P - I) x x^T.
     Chain rule for P=VW: grad_W l = V^T (P-I) x x^T;  grad_V l = (P-I) x x^T W^T.
     SGD: pick random x, W <- W - eta grad_W, V <- V - eta grad_V. (Batch version uses covariance.)
     Complexity for T iters: O(T(d+k)D^2), k batch size. Better than O(D^3) when d,k << D.
- NON-LINEAR AEs: F,G nonlinear (R^D->R^d, R^d->R^D). Set of nonlinear fns includes linear =>
  reconstruction loss <= linear AE. BUT trained with GD without global convergence guarantees =>
  may end up worse than linear. Digits: m=20 -> PCA 1.984, linear AE 1.985, nonlinear AE 1.360;
  m=2 -> PCA 13.42, linear AE 13.42, nonlinear 5.99. Example: 2D manifold in 3D (nonlinear captures curve).

## Existing notes to link to
- /garden/maths/linearAlgebra/pca (PCA: variance max, Lagrange, covariance, choosing d)
- /garden/maths/linearAlgebra/projectionsLS (projection matrix P=A(A^TA)^{-1}A^T, idempotency, P=QQ^T)
- /garden/maths/linearAlgebra/moorePenrose (left pinv (A^TA)^{-1}A^T, conditions)
- /garden/maths/linearAlgebra/eigendecomposition (#diagonalization, spectral theorem, Rayleigh quotient,
  PSD #positive-definite-and-positive-semi-definite-matrices, trace, SVD, low-rank)
- /garden/maths/linearAlgebra/matrixRanks, /vectorSpaces (#column-space, span, basis), /matrixInverses
- /garden/ml/gradientDescent (SGD), /garden/ml/neuralNetworks, /garden/ml/backprop
- /garden/maths/linearAlgebra/orthogonality (orthonormal, orthogonal complement)

## Existing autoencoders.md structure
- intro; ## Traditional Autoencoders [TODO block + general g_phi/f_theta def, MSE, bottleneck];
  ### What Makes a Good Representation?; ### The Infomax Principle; ## Variational Autoencoders (...).
- Plan: add "## Linear Autoencoders" big section with the above derivation; keep VAE; cross-link pca.md.
