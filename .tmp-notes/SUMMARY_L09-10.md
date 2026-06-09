# Source summary: CIL 2026 Lectures 9-10 (+ 09 prep) — Backprop / GD / Optimizers

Scratch working notes. Source PDFs in `resources/material/`, rendered pages in `.tmp-notes/img/`
as `09_L_pNN.png`, `10_L_pNN.png`, `09_P_pNN.png`. Lectures 9-10 are NOT autoencoders. They are
Deep Neural Networks: compositional models, backpropagation, gradient methods, CNNs.

Target notes (confirmed scope): `content/garden/ml/backprop.md` (expand),
`content/garden/ml/gradientDescent.md` (theory, new), `content/garden/ml/optimizers.md`
(algorithms, new). NN foundations and CNNs out of scope.

Notation for the notes: bold `\mathbf{}` for latin vectors/matrices, `\boldsymbol{\theta}` for the
parameter vector, step size `\eta`, loss `\ell`, smoothness `L`, strong-convexity/PL `\mu`,
eigenvalues `\lambda`, condition number `\kappa`. No em dashes, no semicolons.

## Lecture 9 (54 pp): DNN — compositional models, backprop, gradient methods

### §1 Compositional models / ridge functions / MLP (p4-26)
- Compositionality: `f = g ∘ H_L ∘ ... ∘ H_1`, each `H_l: R^{n_{l-1}} -> R^{n_l}`. Depth L.
- Philosophies: feature engineering (p<<n), feature expansion (p>>n), compositionality (depth).
- Partial maps `H_{l:1}` produce intermediate representations. Markov property: each layer must
  preserve task-relevant info. Layers can only lose information, net gain is accessibility.
- Linear maps `F(x;Θ)=Θx`, closed under composition (no expressivity gain) → need non-linearity.
- Ridge function: `H(x;Θ)=φ(Θx)`, scalar activation φ applied elementwise.
- Activations: sigmoid `σ(z)=1/(1+e^{-z})`, tanh `=2σ(2z)-1`, ReLU `(z)_+=max(0,z)`,
  softplus (smooth ReLU), GELU (modern transformers). ReLU partitions input along hyperplane.
- Universal Approximation Theorem (Cybenko 1989, Hornik 1989): one hidden layer + non-polynomial
  activation = universal approximator (needs enough neurons; depth is more efficient).
- Classical MLP: `f(x;β,Θ)=Σ_j β_j σ(⟨θ_j,x⟩)`, sigmoid hidden + linear output.
- MLP derivatives (squared loss `ℓ=½(f(x)-y)^2`): [09_L_p21,p22]
  - `∂ℓ/∂β_j = (f(x)-y) σ(⟨θ_j,x⟩)`
  - `∂ℓ/∂θ_{ji} = (f(x)-y) β_j σ'(⟨θ_j,x⟩) x_i`
  - key fact `σ'(z)=σ(z)(1-σ(z))=σ(z)σ(-z)`. [ANNOTATION p22: "remember for exam";
    sigmoid maxes 0.5, derivative maxes 0.25; "activations very large/small won't adapt weights
    → vanishing"]
- SGD: minibatch `S_t`, `θ_{t+1}=θ_t - η ∇ε(...; S_t)`, B=|S_t| batch size, mean of gradient.

### §2 Backpropagation (p28-41)
- Single unit `h=φ(⟨θ,z⟩)`, loss ℓ. Error signal / delta `δ := ∂ℓ/∂h`. "How much the loss wants
  to change this activation." [ANNOTATION p30: "how is this different to a gradient? become
  more/less active" — answer in note: δ is gradient of loss w.r.t. an INTERMEDIATE activation,
  not w.r.t. parameters.]
- Single-unit parameter gradient (chain rule): `∇_θ ℓ = (∂ℓ/∂h) ∇_θ h = δ φ'(⟨θ,z⟩) z`.
  = upstream input z × downstream scalar error δ × local sensitivity φ'.
- Jacobi map `[J_H]_{ij}=∂h_i/∂z_j`. Local linearization `H(z+Δz)≈H(z)+J_H(z)Δz`.
- Backprop recurrence [09_L_p33]:
  `δ_k := ∂ℓ/∂H_k = [∂H_{k+1}/∂H_k]^T · ∂ℓ/∂H_{k+1} =: J_{k+1}^T δ_{k+1} = J_{k+1}^T ... J_L^T δ_L`,
  where `J_k = J_{H_k}`. Jacobi of composed map = product of Jacobis of elementary maps.
- Caution [p34]: implicit dependence on x,y. `δ_L=δ_L(x,y)`, `J_k=J_k(z_{k-1})`, `z_{k-1}=z_{k-1}(x)`.
  Evaluating Jacobi maps requires a forward pass.
- Jacobi special cases for ridge map `H(z)=φ(Θz)` [09_L_p35-37]:
  - linear φ=id: `J_H=Θ`
  - ReLU: `J_H = diag(χ)Θ`, `χ=1[Θz>0] ∈{0,1}^m`
  - sigmoid: `J_H = diag(χ)Θ`, `χ=σ'(Θz) ∈(0,1)^m`
- Initial error signals [09_L_p38,p39]:
  - squared loss `ℓ=½‖y-Ψ(x)‖^2`: `δ_L=Ψ(x)-y`
  - logistic loss (binary y∈{-1,1}, n_L=1) `ℓ=-ln σ(yΨ(x))`: `δ_L=-y σ(-yΨ(x))`
- Backprop algorithm [09_L_p40]: 1) forward pass compute z_1..z_L; 2) compute loss error δ_L;
  3) backpropagate via Jacobi-vector products to get δ_{L-1}..δ_1; 4) compute & accumulate
  partial derivatives; 5) minibatch SGD update.
- Autodiff [09_L_p41]: automates gradient computation. Theano (symbolic graph), PyTorch/JAX
  (imperative dynamic). Enables efficient/correct gradients, differentiable sims, frees from
  manual derivation. [ANNOTATION p27: "Cant analytically derive derivatives" motivates this.]

### §3 Gradient methods — convergence (p42-52) → gradientDescent.md
- Quadratic approximation near minimum (Taylor): `ℓ(θ)≈ℓ(θ*)+½(θ-θ*)^T Q (θ-θ*)`, `Q⪰0`.
- Quadratic loss coordinate decoupling [09_L_p44]: `ℓ(θ)=½θ^T Q θ - b^T θ`. Diagonalize
  `Q=UΛU^T`, `U^T U=I`, `Λ=diag(λ_1..λ_d)`. Change basis `ν=U^T θ`, `q=U^T b`. Loss separates:
  `ℓ(ν)=Σ_i ℓ_i(ν_i)`, `ℓ_i(ν_i)=(λ_i/2)ν_i^2 - q_i ν_i`. Each coordinate evolves independently.
- Per-coordinate [09_L_p45,p46]: optimum `ν_i*=q_i/λ_i`, value `ℓ_i(ν_i*)=-q_i^2/(2λ_i)`.
  Shifted loss `ℓ̃_i(ν_i)=(1/2λ_i)(λ_i ν_i - q_i)^2 ≥0`. Gradient step `ν_i^+ = ν_i - η(λ_i ν_i - q_i)`.
  After step: `ℓ̃_i(ν_i^+)=(1-λ_i η)^2 ℓ̃_i(ν_i)`. Decrease iff `|1-λ_i η|<1 ⟺ η<2/λ_i`.
  For all coords `η < 2/λ_max`.
- Optimal step & contraction [09_L_p48]:
  `η* = argmin_η max_i (1-ηλ_i)^2 = argmin_η max{ηλ_max-1, 1-ηλ_min} = 2/(λ_max+λ_min)`.
  Weakest contraction `ρ=(1-λ_min η*)^2 = ((λ_max-λ_min)/(λ_max+λ_min))^2 = ((κ-1)/(κ+1))^2`,
  `κ=λ_max/λ_min`. [ANNOTATION: if max=min then ρ=0 (one step); if max>>min, ρ→1 (slow zigzag).]
  GD converges faster when Q is well-conditioned.

## Lecture 10 (65 pp): §3 GD continued + §4 CNNs (CNN out of scope)

### §3 continued — smoothness, PL, strong convexity, saddles, optimizers
- Smoothness [10_L_p06-09]: `ℓ:R^d->R` is L-smooth if `‖∇ℓ(θ)-∇ℓ(θ')‖ ≤ L‖θ-θ'‖`. Smoothness of
  ℓ ⟺ Lipschitz continuity of ∇ℓ. Hessian bound `∇^2 ℓ(θ) ⪯ L·I` (largest eigenvalue ≤ L).
- Implications [10_L_p11,p12]: Taylor w/ integral remainder gives descent lemma
  `ℓ(θ')-ℓ(θ) ≤ ⟨∇ℓ(θ),θ'-θ⟩ + (L/2)‖θ'-θ‖^2`. For GD step `θ'=θ-η∇ℓ(θ)`:
  `ℓ(θ')-ℓ(θ) ≤ -η(1 - Lη/2)‖∇ℓ(θ)‖^2`. With `η=1/L`: `ℓ(θ')-ℓ(θ) ≤ -(1/2L)‖∇ℓ(θ)‖^2`.
  [η=1/L is optimal for the guaranteed/maximal decrease.]
- ε-critical point [10_L_p14-16]: `‖∇ℓ(θ)‖ ≤ ε`. With C:=ℓ(θ_0)-ℓ*≥0, telescoping the descent
  lemma: `(1/k)Σ_{r=0}^{k-1} ‖∇ℓ(θ_r)‖^2 ≤ 2LC/k`. So at least one iterate has
  `‖∇ℓ(θ')‖^2 ≤ 2LC/k ≤ ε^2 ⟺ k ≥ 2LC/ε^2`. Smoothness alone ⇒ ε-critical in O(ε^{-2}) steps.
  Caveat: critical point, not necessarily global/local min.
- Polyak-Lojasiewicz (PL) [10_L_p18-22]: `½‖∇ℓ(θ)‖^2 ≥ μ(ℓ(θ)-ℓ*)` ∀θ. "Gradient can't be flat
  away from minimum." Theorem: ℓ differentiable, L-smooth, μ-PL ⇒ GD with η=1/L converges
  geometrically: `ℓ(θ^k)-ℓ* ≤ (1-μ/L)^k (ℓ(θ^0)-ℓ*)`.
  Proof [10_L_p21]: 1) descent lemma `ℓ(θ^{k+1})-ℓ(θ^k) ≤ -(1/2L)‖∇ℓ(θ^k)‖^2`;
  2) PL `-‖∇ℓ(θ^k)‖^2 ≤ -2μ(ℓ(θ^k)-ℓ*)`; 3) combine & subtract ℓ*:
  `ℓ(θ^{k+1})-ℓ* ≤ (1-μ/L)(ℓ(θ^k)-ℓ*)`; 4) induction.
  Illustration: 4x^2 is PL (μ=8); x^4 not PL at 0 (gradient too small near min).
- Strong convexity [10_L_p23-28]: `ℓ(θ') ≥ ℓ(θ)+⟨∇ℓ(θ),θ'-θ⟩+(μ/2)‖θ'-θ‖^2`. μ=0 → convex.
  "Curved at least like a parabola of curvature μ." PD quadratic ⇒ μ=λ_min. Twice-diff sandwich
  `0 ≺ μI ⪯ ∇^2 ℓ(θ) ⪯ LI`. Strong convexity bounds smallest eigenvalue, smoothness the largest.
  Strong convexity ⇒ PL with same μ [10_L_p27]: minimize both sides of SC over θ', minimizer
  `θ'=θ-(1/μ)∇ℓ(θ)`, RHS becomes `-(1/2μ)‖∇ℓ(θ)‖^2`, so `ℓ*-ℓ(θ) ≥ -(1/2μ)‖∇ℓ‖^2` = PL.
  In DNNs PL/SC usually hold only locally near a minimum (fast local convergence, no global claim).
- Saddle points [10_L_p29]: ∇ℓ=0 but indefinite Hessian. DNN objectives have them. GD slows down.
  Noisy gradient descent (SGD) is a valid escape strategy.

### §3 Optimizers (p30-40) → optimizers.md
- Heavy ball / momentum (Polyak 1964) [10_L_p31]: `θ^{k+1}=θ^k - η∇ℓ(θ^k) + β(θ^k-θ^{k-1})`,
  β∈(0,1), the β-term is extrapolation. Constant-gradient analysis: increments
  `θ^1-θ^0=-η∇ℓ`, `θ^2-θ^1=-η(1+β)∇ℓ`, ... `lim (θ^k-θ^{k-1}) = -η Σ_{i≥0} β^i ∇ℓ = -(η/(1-β))∇ℓ`.
  Effective step boosted by 1/(1-β). β=0.9 ⇒ ~10x. Too large β → oscillation/instability,
  typical [0.9, 0.95].
- Momentum velocity form (1986) [10_L_p33]: `v^{k+1}=β v^k - η∇ℓ(θ^k)`, `θ^{k+1}=θ^k+v^{k+1}`.
  Moving average of past gradients, accelerates consistent directions, damps oscillations, escapes
  shallow regions. Same idea as heavy ball.
- Nesterov acceleration (1983) [10_L_p34]: evaluate gradient at momentum-extrapolated point.
  `θ'^{k+1}=θ^k+β(θ^k-θ^{k-1})`, `θ^{k+1}=θ'^{k+1}-η∇ℓ(θ'^{k+1})`. Anticipates momentum, adjusts
  before overshoot ("skating on ice"). Optimal in convex case, accelerated in strongly convex.
- AdaGrad (2011) [10_L_p36]: `γ_i^k=γ_i^{k-1}+[∂_i ℓ(θ^k)]^2`, `θ_i^{k+1}=θ_i^k - η_i^k ∂_i ℓ`,
  `η_i^k = η/√(γ_i^k+δ)`. Per-coordinate (not just global LR). δ numerical stability, η base LR.
  [ANNOTATION: big updates grow γ, step shrinks; lots of updates in same direction → stop going.]
  Problem: γ grows indefinitely so step sizes vanish, optimization stalls.
- RMSProp (2012) [10_L_p37]: EMA instead of sum: `γ_i^k=ρ γ_i^{k-1}+(1-ρ)[∂_i ℓ(θ^k)]^2`,
  `η_i^k=η/√(γ_i^k+δ)`. ρ=0.9 typical. Keeps adaptivity without decaying steps to zero.
  [ANNOTATION: "is this weight decay?" — no; clarify in note. Widely used though only in lec notes.]
- Adam (2014) [10_L_p39]: 1st moment (momentum) `g_i^k=β g_i^{k-1}+(1-β)∂_i ℓ(θ^k)`, β≈0.9,
  g^0=∂_i ℓ(θ^0); 2nd moment (RMSProp) `h_i^k=α h_i^{k-1}+(1-α)[∂_i ℓ(θ^k)]^2`, α≈0.999,
  h^0=[∂_i ℓ(θ^0)]^2. Update `θ_i^{k+1}=θ_i^k - η_i^k g_i^k`, `η_i^k=η/√(h_i^k+δ)`, δ≈1e-8,
  η≈0.001. Bias correction: `ĝ_i^k=g_i^k/(1-β^k)`, `ĥ_i^k=h_i^k/(1-α^k)`. [ANNOTATION: bigger steps
  in beginning to correct for zero init — moments start biased toward 0.] Gradient stochastic (SGD).
- Takeaways [10_L_p40]: GD uses loss gradients; momentum/Nesterov smooth updates; adaptive
  (AdaGrad/RMSProp) per-parameter LR; Adam combines momentum+RMSProp; no single best;
  Adam/AdamW strong default. [ANNOTATION: AdamW "decouples weight decay from gradients" — explain.]

## 09 Preparation (35 pp): math foundation
- §1 Multivariate Taylor (p3-10): multi-index notation, 1st/2nd order approximations, vector fields.
  Useful for the local quadratic model in gradientDescent.
- §2 Matrix calculus (p11-20):
  - Numerator vs denominator layout [09_P_p12]. Numerator: `∂y/∂x ∈ R^{m×n}`, `∇f=(∂f/∂x)^T`.
    Be consistent. [ANNOTATION: "in high dimensions there are different layouts", "first dimension
    of matrix given by numerator", "chain rule nicer with numerator"].
  - VJP simple case [09_P_p14]: `L=ℓ(y,ŷ)`, `ŷ=Wx`. `∂L/∂W = (∂L/∂ŷ)(∂ŷ/∂W)`.
  - Backprop chain [09_P_p16]: `j_1 J_2 J_3 ... J_n`, `j_1=∂L/∂x_n ∈ R^{1×d_n}` row vector (loss
    gradient), J_i Jacobian matrices. Lowercase j = vector, uppercase J = matrix.
  - VJP worked [09_P_p19]: y=Wx ⇒ `∂y_i/∂w_{jk}=x_k·1{i=j}` ⇒ `[∂L/∂W]_{jk}=(∂L/∂y_j)x_k` ⇒
    matrix form `∂L/∂W = (∂L/∂y)^T x^T`. "Significantly more efficient than constructing ∂y/∂W."
  - Computational graph [09_P_p20]: figure z=ℓ(w), w=log(x1·x2)·sin(x2). Forward nodes, backward
    nodes MultBackward/LogBackward/SinBackward. "Grads from different paths are added together."
    Source: PyTorch blog. → use for autodiff: addition copies upstream grad (local Jacobian I),
    multiplication swaps the other input, fan-out branches sum.
- §3 Lipschitz continuity (p21-24): `|f(x)-f(y)| ≤ L‖x-y‖`, slope bounded everywhere.
- §4 Convexity & smoothness (p25-28): convex def (chord above graph), strict/strong convexity,
  L-smoothness (L-Lipschitz gradient).
- §5 Saddle points (p29-35): gradient zero but not extremum; problem for GD; escape strategies.

## Handwritten annotations to honor (George's study questions)
- δ vs gradient (backprop): δ is the gradient w.r.t. an intermediate activation.
- σ'(z)=σ(z)σ(-z), maxes at 0.25 → vanishing gradients (remember for exam).
- "Cant analytically derive derivatives" → autodiff motivation.
- κ large → slow zigzag; κ=1 → one-step convergence.
- AdaGrad steps shrink forever; RMSProp fixes with EMA; "is this weight decay?" (no).
- Adam bias correction = bigger steps early (zero-init bias).
- AdamW decouples weight decay from gradients.
