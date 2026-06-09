---
title: Text Embeddings
type: docs
weight: 1
---

A word is a discrete symbol, but its meaning is not. The natural way to feed a word to a model is the **one-hot vector**, a vector as long as the vocabulary with a single one at the word's index and zeros everywhere else. This representation has two fatal flaws. It is enormous, since a realistic vocabulary has hundreds of thousands or millions of entries, and it carries no notion of similarity, because any two distinct words are orthogonal. Under one-hot encoding "good" and "great" are exactly as dissimilar as "good" and "milk", which throws away everything we know about meaning.

A **word embedding** replaces the one-hot vector with a dense, low-dimensional real vector $\mathbf{z}_w \in \mathbb{R}^m$, with $m$ typically between $100$ and $500$. The goal is for geometry to reflect meaning: words with similar meanings should sit close together, and distances and angles should encode semantic relationships, so that we can reason, find analogies, and cluster in a continuous space. The same idea applies to any discrete symbols, not just words, which is why the same recipe reappears for products, users, genes, and graph nodes (see [node embeddings](/garden/ml/nodeEmbeddings)).

How can we possibly learn meaning from raw text with no labels? The guiding principle is the **distributional hypothesis**, captured by Firth's slogan "you shall know a word by the company it keeps." A word's meaning is reflected in the contexts it appears in, so words used in similar contexts should get similar vectors. This is a [latent variable](/garden/ml/latentVariableModels) view: the embedding of a word is a latent vector whose job is to **predict the words that co-occur with it**. The two classic methods built on this idea, **word2vec** and **GloVe**, differ mainly in how they turn co-occurrence into geometry, and they are close cousins of the matrix-factorization view of [topic models](/garden/ml/topicModels#the-matrix-factorization-view). They have largely been superseded by contextual embeddings from [transformers](/garden/ml/llms/transformers), but the concepts they introduced underpin everything that followed.

## Word2Vec

Word2vec (Mikolov et al., 2013, at Google) is a family of shallow log-linear models that learn word vectors by predicting co-occurrences within a sliding context window. There are two complementary variants. **Skip-gram** predicts the surrounding context from a centre word, and **continuous bag-of-words (CBOW)** predicts the centre word from its surrounding context. We develop skip-gram in full, since it is the version the derivation below targets, then describe CBOW as its mirror image.

### The Skip-Gram Model

Skip-gram slides a window of radius $R$ across the text and, from each **centre word**, tries to predict the **context words** in the window around it. Writing the corpus as a sequence $x_1, \dots, x_T$ and the window offsets as $I = \{-R, \dots, -1, 1, \dots, R\}$, the model maximizes the log-probability of every context word given its centre word, assuming the context predictions are independent:

$$
\ell(\boldsymbol{\theta}; \mathbf{x}) = \sum_{t=1}^T \sum_{\ell \in I} \ln p(x_{t+\ell} \mid x_t; \boldsymbol{\theta})
$$

So for the sentence "the victorious king rode to his castle", with "king" as the centre word, the model is trained to raise the probability of "victorious", "rode", "to", "his" and to make unrelated words like "moon" unlikely.

It remains to choose the conditional $p(v \mid w)$ of a context word $v$ given a centre word $w$. We want the probability to grow when the two embeddings point in a similar direction, so we use a **log-bilinear model**: the log-probability is the inner product of the two word vectors, up to a constant,

$$
\ln p(v \mid w) = \langle \mathbf{z}_w, \mathbf{z}_v \rangle + \text{const} \quad\implies\quad p(v \mid w) = \frac{\exp \langle \mathbf{z}_w, \mathbf{z}_v \rangle}{\sum_{u \in \Sigma} \exp \langle \mathbf{z}_w, \mathbf{z}_u \rangle}
$$

The normalization turns the scores into a probability distribution over the whole vocabulary $\Sigma$ through a softmax. The inner product $\langle \mathbf{z}_w, \mathbf{z}_v \rangle$ is the design choice that ties similarity to geometry, and it plays the same role as the query-key dot product in [attention](/garden/ml/llms/attention).

In practice the model is refined in two small ways. Each word is given **two** vectors, a vector $\boldsymbol{\zeta}_w$ used when it acts as the centre word and a vector $\mathbf{z}_w$ used when it acts as a predicted context word, plus a scalar **bias** $b_w$ that lets the model control each word's marginal frequency. The conditional becomes

$$
p(v \mid w; \boldsymbol{\theta}) = \frac{\exp\!\big(\langle \boldsymbol{\zeta}_w, \mathbf{z}_v \rangle + b_v\big)}{\sum_{u \in \Sigma} \exp\!\big(\langle \boldsymbol{\zeta}_w, \mathbf{z}_u \rangle + b_u\big)}, \qquad w \mapsto \boldsymbol{\theta}_w = (\mathbf{z}_w, \boldsymbol{\zeta}_w, b_w) \in \mathbb{R}^{2m+1}
$$

Since word order is dropped, the data reduces to the **co-occurrence counts** $N_{vw} = |\{t : x_t = w, \, x_{t+\ell} = v, \, \ell \in I\}|$, the number of times $v$ appears in the context of $w$. The log-likelihood then groups by word pair,

$$
\ell(\boldsymbol{\theta}) = \sum_{v, w} N_{vw} \Big[ \langle \boldsymbol{\zeta}_w, \mathbf{z}_v \rangle + b_v - \ln \underbrace{\sum_{u \in \Sigma} \exp\big(\langle \boldsymbol{\zeta}_w, \mathbf{z}_u \rangle + b_u\big)}_{\text{normalization constant}} \Big]
$$

### Continuous Bag-of-Words

CBOW is skip-gram run in the opposite direction. Instead of predicting context from the centre, it predicts the **centre word from its context**. The context embeddings in the window are summed (or averaged) into a single context vector, and the centre word is predicted from it:

$$
p(w \mid \text{context}) = \frac{\exp\!\big\langle \mathbf{z}_w, \, \sum_{\ell \in I} \boldsymbol{\zeta}_{x_{t+\ell}} \big\rangle}{\sum_{u \in \Sigma} \exp\!\big\langle \mathbf{z}_u, \, \sum_{\ell \in I} \boldsymbol{\zeta}_{x_{t+\ell}} \big\rangle}
$$

Bagging the context words together (hence "bag-of-words") makes CBOW faster to train and a little better on frequent words, since it averages out noise across the window. Skip-gram, by treating each context word as a separate prediction, produces a stronger training signal per occurrence and tends to do better on rare words and small corpora. Both produce embeddings of the same flavour.

### Negative Sampling

Both objectives share an expensive flaw: the softmax normalization sums over the **entire vocabulary** for every single prediction, which is hopeless when the vocabulary has a million words. The fix is to change the task. Rather than predicting one word out of all of them, we pose a **binary classification** problem: given a word pair, decide whether it is a genuine co-occurrence or a random one. The prediction task is not fixed, and we are free to design one that induces good vectors while avoiding the partition function.

For a pair $(w, v)$ we model the probability that it is a true co-occurrence with the logistic function applied to the same bilinear score,

$$
p(\text{true} \mid w, v) = \sigma\big(\langle \boldsymbol{\zeta}_w, \mathbf{z}_v \rangle + b_v\big), \qquad \sigma(a) = \frac{1}{1 + e^{-a}}
$$

The **positive set** $S^+$ is the set of word pairs actually observed in the corpus. The **negative set** $S^-$ is built by **negative sampling**: for each positive pair we draw $r$ noise words $v \sim q$ from a fixed noise distribution over the vocabulary and pair them with the same centre word. Training maximizes the logistic log-likelihood, pushing the score up on real pairs and down on noise pairs, which is just a cross-entropy on the two classes (using $1 - \sigma(a) = \sigma(-a)$),

$$
\ell(\boldsymbol{\theta}) = \sum_{(w,v) \in S^+} \ln \sigma\big(\langle \boldsymbol{\zeta}_w, \mathbf{z}_v \rangle + b_v\big) \; + \; \sum_{(w,v) \in S^-} \ln \sigma\big(-\langle \boldsymbol{\zeta}_w, \mathbf{z}_v \rangle - b_v\big)
$$

Each term now costs an inner product rather than a sum over the vocabulary. The noise distribution is chosen as a flattened version of the empirical word frequency,

$$
q(w) \propto p(w)^{\alpha}, \qquad \alpha = \tfrac{3}{4}
$$

Raising the frequency to the power $\tfrac{3}{4}$ damps the most common words and lifts rare ones, much like a temperature, so that frequent function words do not dominate the negative samples.

### Why It Works: Pointwise Mutual Information

Negative sampling can look like an arbitrary trick, so it is worth seeing what the learned scores actually represent. This is the most illuminating part of the story. Set up the classification as distinguishing true pairs, which come from the empirical co-occurrence distribution $p(v, w)$, from noise pairs, which come from the product distribution $q(v, w) = p(w)\,q(v)$ where the context word is drawn independently. If a fraction $\pi$ of pairs are true and $1 - \pi$ are noise, then by Bayes' rule the **optimal** classifier is

$$
P(\text{true} \mid v, w) = \frac{\pi \, p(v, w)}{\pi \, p(v, w) + (1 - \pi)\, q(v, w)}
$$

Our model represents this probability as $\sigma(h_{vw})$ where $h_{vw} = \langle \boldsymbol{\zeta}_w, \mathbf{z}_v \rangle + b_v$ is the score it learns. At the optimum the score must equal the **log-odds** of the Bayes classifier, that is the pre-image of the probability under the logistic function,

$$
h_{vw}^* = \sigma^{-1}\!\big(P(\text{true} \mid v, w)\big) = \ln \frac{p(v, w)}{q(v, w)} + \ln \frac{\pi}{1 - \pi}
$$

Now take **balanced classes** $\pi = \tfrac{1}{2}$ and uniform-exponent noise $\alpha = 1$, so $q(v, w) = p(v)\,p(w)$. The second term vanishes and the score collapses to

$$
h_{vw}^* = \ln \frac{p(v, w)}{p(v)\, p(w)} = \text{PMI}(v, w)
$$

This is the **pointwise mutual information** of the pair. PMI measures how much more often two words co-occur than they would by chance: it is positive when $v$ and $w$ appear together more than independence predicts, zero when they are independent, and negative when they avoid each other. So negative sampling arranges the embeddings such that the inner product $\langle \boldsymbol{\zeta}_w, \mathbf{z}_v \rangle$ approximates the PMI between the words, and word2vec is implicitly performing a low-rank factorization of the PMI matrix (Levy and Goldberg, 2014). Summed over all pairs, training word2vec maximizes the mutual information of word co-occurrences, which is exactly the statistical signal the distributional hypothesis says carries meaning.

## GloVe

**GloVe** (Global Vectors, Pennington, Socher, and Manning, 2014, at Stanford) starts from the same co-occurrence counts but fits them directly with a regression instead of a probabilistic prediction. The previous section showed that the right target is the log co-occurrence, so GloVe simply asks the inner product of two word vectors to reproduce the **log count**, in a weighted least-squares fit,

$$
\ell(\boldsymbol{\theta}; \mathbf{N}) = \sum_{v, w \,:\, N_{vw} > 0} f(N_{vw}) \Big( \ln N_{vw} - \langle \boldsymbol{\zeta}_w, \mathbf{z}_v \rangle - b_w - b_v \Big)^2
$$

The bracket is the residual between the observed log-count (the target) and the model's bilinear score plus biases (the prediction). Two design choices make this work well.

- **Weighting function.** Not all pairs deserve equal weight. Very frequent pairs would otherwise dominate the fit, and very rare pairs are noisy. The weight

$$
f(N) = \min\!\left\{ 1, \left( \frac{N}{N_{\max}} \right)^{\alpha} \right\}, \qquad \alpha = \tfrac{3}{4}
$$

grows with the count for rare pairs but is capped at one for frequent pairs, so common co-occurrences are downweighted rather than allowed to swamp the objective. Pairs that never co-occur are excluded entirely, which is why the sum runs only over $N_{vw} > 0$.

- **No normalization.** Because the objective is a two-sided squared loss between two real numbers rather than a probability, the model $\ln \hat{p}(v, w) = \langle \boldsymbol{\zeta}_w, \mathbf{z}_v \rangle$ does **not** need to be a normalized distribution. There is no partition function to compute, which is GloVe's main efficiency advantage over the softmax skip-gram.

### GloVe as Matrix Factorization

Stacking the centre vectors as the rows of a matrix and the context vectors as the columns makes the connection to the [low-rank factorizations](/garden/ml/matrixCompletion) explicit. Collecting $\mathbf{U}^\top = [\boldsymbol{\zeta}_{w_1}, \dots]$ and $\mathbf{V} = [\mathbf{z}_{w_1}, \dots]$, the bilinear scores form the product $\mathbf{U}\mathbf{V}$, and GloVe fits

$$
\ln \hat{\mathbf{N}} = \mathbf{U}\mathbf{V}
$$

so GloVe is a **low-rank factorization of the log co-occurrence matrix**, where the rank is the embedding dimension $m$. In the simplest case $f(N) = \min\{1, N\}$ and no biases, the objective is plainly

$$
\min_{\mathbf{U}, \mathbf{V}} \sum_{v, w \,:\, N_{vw} > 0} \big( \ln N_{vw} - (\mathbf{U}\mathbf{V})_{vw} \big)^2
$$

This is the same low-rank-factorization idea behind [topic models](/garden/ml/topicModels#the-matrix-factorization-view) and [matrix completion](/garden/ml/matrixCompletion), applied here to the matrix of log co-occurrences. It is optimized by stochastic gradient descent: sample a pair $(v, w)$ and take a step with rate $\eta$,

$$
\boldsymbol{\zeta}_w \leftarrow \boldsymbol{\zeta}_w + 2\eta\, f(N_{vw}) \big( \ln N_{vw} - \langle \boldsymbol{\zeta}_w, \mathbf{z}_v \rangle \big)\, \mathbf{z}_v, \qquad
\mathbf{z}_v \leftarrow \mathbf{z}_v + 2\eta\, f(N_{vw}) \big( \ln N_{vw} - \langle \boldsymbol{\zeta}_w, \mathbf{z}_v \rangle \big)\, \boldsymbol{\zeta}_w
$$

The contrast with word2vec is instructive. Skip-gram with negative sampling learns from **local** windows one at a time and implicitly factorizes the PMI matrix, while GloVe fits the **global** co-occurrence statistics directly with an explicit least-squares objective. Both end up factorizing a transformed co-occurrence matrix into word vectors, and in practice the two give comparably useful embeddings.

## Geometry of Word Vectors

The payoff of both methods is that semantic structure shows up as geometry. Words that occur in similar contexts end up with nearby vectors, so a nearest-neighbour search around a query word retrieves its semantic relatives, for example the neighbours of "frog" being other amphibians and related species names.

More strikingly, **relationships** become consistent directions in the space. The vector difference between a word and its analogue is roughly the same across pairs, so "king $-$ man $+$ woman" lands near "queen", and the same offset connects "uncle" to "aunt" or "emperor" to "empress". Analogy queries are answered by searching for the word whose vector best completes the parallelogram,

$$
\boldsymbol{\zeta}_{\text{queen}} = \arg\max_{v} \big\langle \boldsymbol{\zeta}_{\text{king}} - \boldsymbol{\zeta}_{\text{man}} + \boldsymbol{\zeta}_{\text{woman}}, \; \boldsymbol{\zeta}_v \big\rangle
$$

This **affine structure** is the clearest sign that the embeddings have captured something about meaning rather than mere co-occurrence bookkeeping, and it is what made these models so influential. The translation of co-occurrence structure into geometry is the single idea behind all of them: similar usage becomes proximity, and analogies become directions.

## FastText

## ELMo

## BERT

## Sentence Embeddings

 