# L7-L8 source summary + plan (scratch, gitignored)

Lectures 7 (17 Apr) and 8 (24 Apr), Boeva, slides adopted from Thomas Hofmann.
Unified theme: **Latent Variable Models**.

## Where each topic goes (decided with user)
- NEW `content/garden/ml/latentVariableModels.md` (weight 13): latent-variable framework,
  mixture models, EM (Jensen -> ELBO -> E/M steps), k-means as EM, GMM.
- NEW `content/garden/ml/topicModels.md` (weight 14): term-document/exchangeability, pLSA + EM,
  LDA (Dirichlet conjugacy), probabilistic matrix decomposition view -> link NMF.
- EXTEND `content/garden/ml/matrixCompletion.md`: add standalone NMF section (model, Frobenius/KL,
  ALS, part-based interpretation). topicModels links here.
- EXTEND `content/garden/ml/llms/textEmbeddings.md`: word2vec (skip-gram + CBOW + neg sampling + PMI),
  GloVe (matrix factorization view, analogies).
- Prose-first, minimal new figures.

## Notation (match garden)
- bold vectors/matrices `\mathbf{}`, transpose `^\top`, bare `^*` for optima (KaTeX).
- mixture/GMM: data x_i, i=1..n; latent z in {1..k}; mixing pi_z (sum 1); component params theta_z
  (GMM: mu_z, Sigma_z); responsibilities q_{iz} = P(Z_i=z | x_i; theta).
- topic models: docs i=1..n, positions t=1..s_i, vocab words w_j j=1..m, topics z in {1..k},
  counts N_{ij}, q_{itz}, u_{jz}=p(w_j|z), v_{zi}=p(z|d_i). N_hat = U V, N_ij ~ s_i N_hat_ij.
- word embeddings (honor lecture): word w=center, v=context. z_w = embedding (when word is predicted/context),
  zeta_w = context vector (when word is center), bias b_w.
  p(v|w) ∝ exp(<zeta_w, z_v> + b_v). Analogy: zeta_queen = argmax_v <zeta_king - zeta_man + zeta_woman, zeta_v>.

## Cross-link targets
- variationalInference.md: ELBO derived there (Bayesian framing, reverse KL, Jensen). EM is same ELBO,
  latent=cluster, objective=data log-likelihood. Link, do not duplicate KL/entropy defs.
- autoencoders.md VAE section: ELBO for deep generative models. Link.
- kMeans.md: k-means as hard EM. Link.
- matrixCompletion.md: ALS, SVD, low-rank, Eckart-Young. NMF lives here.
- tfidf.md: term-document matrix, BM25. Link from topicModels.
- nodeEmbeddings.md (DeepWalk/Node2Vec): reuse skip-gram. Link from textEmbeddings.
- multiVariateGaussian.md: Gaussian marginals/conditionals. pca.md: variance maximisation (NMF vs PCA).

## L7 content (printed)
1. Latent variables: high-dim data, low-dim semantic factors (MNIST digit grid: z1 rotation, z2 stroke).
   Two-step generative process: z ~ P(z) (dense info), x ~ P(x|z) (render, redundant). p(x)=∫p(x|z)p(z)dz.
   Caveats: identifiability (permute labels), no guaranteed interpretability, entanglement, misspecification.
2. Mixture models: Z_t ~ Categ(pi), p(x,z)=pi_z p(x|theta_z), marginal p(x;theta)=sum_z pi_z p(x;theta_z)
   = convex combination. Posterior P(Z=z|x) = pi_z p(x;theta_z)/sum. Hard vs soft clustering.
3. MLE: ell(theta)=sum_t ln sum_z pi_z p(x_t;theta_z). "sum inside log" -> no closed form -> EM.
4. EM: Jensen (ln concave: ln E[X] >= E[ln X]). Variational dist q_z>=0 sum 1.
   ELBO: ell >= sum_t sum_z q_tz[ln pi_z + ln p(x_t;theta_z) - ln q_tz].
   Decomp: expected complete log-lik - KL(q||posterior). E-step: q_tz = posterior (Lagrange mult deriv).
   M-step: pi_z = (1/n) sum_t q_tz; theta_z = argmax sum_t q_tz ln p(x_t;theta_z) (weighted MLE).
   Properties: log-lik never decreases (bound), local max, depends on init. k-means = hard EM.
5. GMM: N(x;mu,Sigma). CLT motivation. E-step responsibilities (general/isotropic).
   M-step: pi_z=(1/n)sum q; mu_z=weighted mean; Sigma_z=weighted scatter. Init (k-means++), choose k (BIC),
   log-sum-exp, singularity regularization.
6. Topic modeling: term-document matrix N (n docs x m words), sufficient stat. Bag-of-words/exchangeability.
   Topic z per WORD OCCURRENCE. p(z|d), p(w|z). Two-stage sampling. pLSA log-lik = sum_ij N_ij ln p(w_j|d_i),
   p(w_j|d_i)=sum_z p(w_j|z)p(z|d_i).

## L8 content (printed)
- pLSA EM: q_itz = p(x_it|z)p(z|d_i)/sum; M: p(w_j|z) = sum I[x_it=w_j]q / sum q ; p(z|d_i)=(1/s_i)sum_t q.
  Limitation (handwriting): only for trained docs, cannot handle new documents -> motivates LDA.
- LDA: u_jz=p(w_j|z), v_zi=p(z|d_i). Put Dirichlet prior over topic-mixture v. Conjugacy: Dirichlet
  conjugate to multinomial/categorical (Beta is k=2 case). Dir(v;alpha) ∝ prod v_z^{alpha_z-1}.
  Marginal P(X;U)=∫ prod_t p(x_t|U,v) p(v|alpha) dv. U shared/global -> generalizes to new docs.
  Learn via collapsed Gibbs / variational EM. NMF vs LDA: NMF fast/simple, LDA Bayesian/generalizes.
- Probabilistic matrix decomposition: N_hat=UV rank<=k, approximates relative freqs N_ij ~ s_i N_hat_ij.
  Non-negativity u,v>=0 + row/col normalization -> special case of NMF with log-likelihood (KL) objective.
- NMF: approx A ~ UV, U,V>=0. Frobenius 1/2||A-UV||^2_F or KL. ALS (fix one, solve other, clip max{0,.}),
  NNDSVD init. Part-based, sparse, additive ("ink can't be erased"). vs PCA holistic eigenfaces w/ cancellation.
- Word embeddings: distributional hypothesis (Firth: "know a word by the company it keeps").
  One-hot -> dense embeddings. Skip-gram: predict context words in window R around center.
  ell = sum_t sum_{l in I} log p(x_{t+l}|x_t). Window I={-R..-1,1..R}.
  Log-bilinear: ln p(v|w)=<z_w,z_v>+const -> softmax over vocab (expensive normalization).
  Refined: p(v|w)=exp(<zeta_w,z_v>+b_v)/Z, params w->(z_w,zeta_w,b_w).
  Suff stat N_vw co-occurrence counts. Negative sampling: binary classification, sigma(<zeta_w,z_v>+b_v).
  S+ observed pairs, S- sampled from noise q(w) ∝ p(w)^alpha, alpha=3/4. r negatives per positive.
  Logistic log-lik = sum_{S+} ln sigma + sum_{S-} ln(1-sigma).
  Bayes classifier P(true)= pi p(v,w)/(pi p(v,w)+(1-pi)q(v,w)). Logit = ln p(v,w)/q(v,w) + ln pi/(1-pi).
  Balanced pi=1/2, alpha=1: logit = ln p(v,w)/(p(v)p(w)) = PMI. word2vec ~ maximize MI. (George: "dont get this")
  GloVe: ell = sum_{v,w:N>0} f(N_vw)(ln N_vw - ln N_hat_vw)^2, N_hat=exp(<zeta_w,z_v>) (+biases).
  f(N)=min(1,(N/Nmax)^alpha), alpha=3/4. Unnormalized (two-sided loss), no partition function.
  Matrix factorization: ln N_hat = U V (U^T context vectors, V embeddings). SGD updates given.
  Analogy: zeta_queen=argmax_v<zeta_king-zeta_man+zeta_woman,zeta_v> (parallelogram). Frogs nearest-neighbours.

## Handwritten annotations to honor (decoded)
- p5: latent z = dense info; x = rendered with redundant info.
- p7: identifiability = can permute cluster labels, not unique; interpretability hoped not guaranteed.
- p12 posterior: numerator "how likely in this cluster", "if high, very likely sampled from cluster z";
  denominator "all classes".
- p13: hard=binary assignment; soft=probability over clusters = uncertainty at boundaries.
- p15: "sum inside the log" is the blocker.
- EM summary p27: "it will converge due to bound"; local max "depends on initialisation, modelling family".
- pLSA EM (L8 p14): M-step is reweighting; "only for documents trained on, cant do anything if new doc comes in".
- LDA (L8 p17/p22): "adds another distribution over the previous distribution"; U shared so new docs OK.
- prob matrix decomp (L8 p28): row/col normalization, actual counts vs relative frequencies.
- NMF faces (L8 p33/34): PCA has negatives/cancellation, NMF part-based/sparse.
- skip-gram (L8 p47): "word2vec"; "CBOW = the opposite, continuous bag of words" (predict center from context).
- bilinear (L8 p49): center=zeta_w, context=z_v, dot product = design choice; softmax over all words "not good".
- PMI (L8 p61): "dont get this" -> explain carefully: neg-sampling logit recovers PMI.
