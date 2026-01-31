---
title: Attention Mechanism
type: docs
weight: 4
---

The attention mechanism represents a fundamental shift in how we process sequential data. Before attention, neural networks for Natural Language Processing relied on recurrent architectures like [recurrent neural networks (RNNs)](/garden/ml/llms/rnn) that processed text sequentially, struggling to capture long-range dependencies. The paper [Attention is All You Need](https://arxiv.org/abs/1706.03762) introduced the Transformer architecture, which replaced recurrence entirely with attention mechanisms. This innovation enabled the models that power modern large language models like ChatGPT and Claude.

https://jalammar.github.io/illustrated-gpt2/
https://sebastianraschka.com/blog/2023/self-attention-from-scratch.html
https://lilianweng.github.io/posts/2018-06-24-attention/

Consider the task of understanding a sentence. As humans, we naturally focus our attention on different words depending on what we are trying to understand. When reading "The baseball player swung his bat and hit a home run", we automatically associate "bat" with sports equipment rather than the animal. This contextual understanding requires relating words to each other, not just processing them in isolation.

{{< figure
    src="/images/ml/attentionWordRelationships.png"
    caption="Words in a sentence have relationships that determine their meaning in context."
    alt="Diagram showing relationships between words in a sentence"
    width="400"
>}}

Traditional approaches to Natural Language Processing faced a critical limitation. If we simply create an embedding for each word such as with [word2vec or GloVe](/garden/ml/llms/textEmbeddings) and aggregate them (say, by averaging), we lose the relationships between words. A sentence like "The actors are good but the movie is bad" cannot be understood by merely adding up word embeddings. The word "good" relates to "actors" while "bad" relates to "movie", and the word "but" signals a contrast. We need a mechanism that allows each word to find its meaning through other words in the sentence.

This is precisely what the attention mechanism provides. It allows each token in a sequence to selectively **give attention** to other tokens, gathering relevant contextual information to refine its representation. The resulting representation captures not just what the word is, but what it means in this specific context.

{{< figure
    src="/images/ml/attentionBetweenWords.png"
    caption="Attention allows tokens to selectively attend to other tokens in the sequence."
    alt="Visualization of attention weights between words in a sentence"
>}}

## Embedding Matrix

The first step of the attention mechanism is to create the embedding matrix. We begin by tokenizing the input text into a sequence of $N$ tokens $[t_1, t_2, \ldots, t_N]$, where each token $t_i$ is an integer index in the vocabulary of size $S$. The tokenization process is typically done using a pre-trained tokenizer associated with the model, such as [tiktoken](https://github.com/openai/tiktoken) for OpenAI models or [WordPiece](https://arxiv.org/abs/1609.08144) for BERT.

The next step is to convert these discrete tokens into continuous vector representations. Each token is mapped to a dense vector in a high-dimensional space, allowing the model to capture semantic relationships between words as discussed in more detail in [Text Embeddings](/garden/ml/llms/textEmbeddings). We can either use pre-trained embeddings or learn them from scratch during model training.

We define an embedding lookup table $\mathbf{W}_{\text{emb}} \in \mathbb{R}^{S \times d}$ where $S$ is the vocabulary size and $d$ is the embedding dimension (a hyperparameter, often 512 or 768 depending on the model architecture). Each row of $\mathbf{W}_{\text{emb}}$ corresponds to the embedding vector for a particular token in the vocabulary.

For a sequence of $N$ tokens, we look up their embeddings to form the sequence embedding matrix $\mathbf{E} \in \mathbb{R}^{N \times d}$. Each row $\mathbf{e}_i \in \mathbb{R}^d$ is the embedding of the $i$-th token in the sequence. This matrix serves as the input to the attention mechanism.

If the embedding table is learned during training, we initialize it randomly and update it via backpropagation along with the rest of the model parameters. If using pre-trained embeddings, we can either keep them fixed or fine-tune them during training.

## Self-Attention Mechanism

The core insight of attention is that every word finds its meaning through other words in the sentence. To formalize this, we need a mechanism that allows each token to query the other tokens and retrieve relevant information.

### Attention as Information Retrieval

The attention mechanism can be understood through the lens of information retrieval. Consider reading a sentence with an unfamiliar word like "alacrity": "She responded with alacrity, leaping to her feet with energy, unlike his usual torpor in the mornings". To understand "alacrity", you naturally look at surrounding words. The words "leaping" and "energy" provide positive context, while the contrast with "torpor" (slowness) further clarifies that "alacrity" means eagerness or enthusiasm.

This contextual disambiguation is exactly what attention achieves. For each token in the sequence, we need to determine which other tokens are relevant for understanding it. The attention mechanism solves this through a query-key-value framework inspired by information retrieval systems.

In a traditional database, you issue a query, the system matches it against indexed keys, and returns associated values. Attention implements a differentiable version of this process. Each token generates a query vector that encodes what information would be useful for understanding that token. Other tokens generate key vectors that encode what information they contain. We measure the compatibility between queries and keys using dot products, producing attention scores. These scores determine how to weight the value vectors (the actual information) when forming the output representation.

The crucial insight is that we use three separate transformations (query, key, value) rather than a single transformation. This asymmetry allows the model to learn that token A might attend strongly to token B, while token B attends weakly to token A. This directional selectivity is essential for modeling linguistic structure.

### Deriving the Attention Formula

Now let us derive the attention mechanism formally. We start with the sequence embedding matrix $\mathbf{E} \in \mathbb{R}^{N \times d}$ where $N$ is the sequence length and $d$ is the embedding dimension. Each row $\mathbf{e}_i \in \mathbb{R}^d$ is the embedding vector for token $i$.

Our goal is to compute a new representation for each token that incorporates contextual information from other tokens. We want each token to selectively attend to relevant parts of the sequence and aggregate their information.

**First attempt**: The simplest approach is to compute pairwise similarities between all token embeddings using dot products. The similarity matrix is $\mathbf{EE}^T \in \mathbb{R}^{N \times N}$ where entry $(i,j)$ is the dot product $\mathbf{e}_i^T \mathbf{e}_j$, measuring the similarity between tokens $i$ and $j$.

We apply a row-wise softmax function to convert these scores into probability distributions:

$$
\mathbf{S} = \text{softmax}(\mathbf{EE}^T) \in \mathbb{R}^{N \times N}
$$

Each row $\mathbf{s}_i$ of $\mathbf{S}$ sums to 1 and represents the attention distribution for token $i$. We then compute the output as:

$$
\mathbf{A} = \mathbf{SE} \in \mathbb{R}^{N \times d}
$$

Each row of $\mathbf{A}$ is a weighted average of the embedding vectors, where the weights come from the attention scores.

This approach has two critical limitations. First, there are no learnable parameters, so the model cannot adapt the attention mechanism to the task. The similarities are determined entirely by the input embeddings. Second, the similarity matrix is symmetric since $\mathbf{EE}^T = (\mathbf{EE}^T)^T$. But linguistic relationships are often directional and asymmetric. The relationship from "Alice" to "tried" (Alice is the subject performing the action) differs from "tried" to "Alice" (tried describes an action performed by Alice).

**Second attempt**: We introduce learnable parameters by transforming the embeddings before computing similarities. Let $\mathbf{W} \in \mathbb{R}^{d \times d}$ be a learnable weight matrix. We transform the embeddings:

$$
\tilde{\mathbf{E}} = \mathbf{EW} \in \mathbb{R}^{N \times d}
$$

Then compute attention as before:

$$
\mathbf{S} = \text{softmax}(\tilde{\mathbf{E}}\tilde{\mathbf{E}}^T), \quad \mathbf{A} = \mathbf{S}\tilde{\mathbf{E}}
$$

This allows the model to learn which aspects of the embeddings are relevant through the transformation $\mathbf{W}$. However, the symmetry problem persists. Even though $\tilde{\mathbf{E}}$ differs from $\mathbf{E}$, the product $\tilde{\mathbf{E}}\tilde{\mathbf{E}}^T$ is still symmetric by definition of matrix transposition. We are computing dot products within the same transformed space, which necessarily produces a symmetric matrix. This prevents the model from learning directional relationships.

**Final form**: The solution is to use separate transformations for computing similarities and retrieving information. We introduce three separate parameter matrices rather than a single shared transformation.

Let $\mathbf{W}_Q \in \mathbb{R}^{d \times d_k}$, $\mathbf{W}_K \in \mathbb{R}^{d \times d_k}$, and $\mathbf{W}_V \in \mathbb{R}^{d \times d_v}$ be learnable weight matrices. The dimensions $d_k$ and $d_v$ are hyperparameters controlling the size of the intermediate representations. Typically we set $d_k = d_v = d$ for simplicity, though using smaller dimensions can reduce computational cost. The constraint is that $d_k$ must be the same for queries and keys since we will compute dot products between them.

We compute three transformed representations:

$$
\mathbf{Q} = \mathbf{EW}_Q \in \mathbb{R}^{N \times d_k} \quad \mathbf{K} = \mathbf{EW}_K \in \mathbb{R}^{N \times d_k} \quad \mathbf{V} = \mathbf{EW}_V \in \mathbb{R}^{N \times d_v}
$$

The query matrix $\mathbf{Q}$ contains transformed representations for computing attention scores. Each row $\mathbf{q}_i \in \mathbb{R}^{d_k}$ is the query vector for token $i$, encoding what information would be useful for understanding that token. The key matrix $\mathbf{K}$ contains transformed representations to be matched against queries. Each row $\mathbf{k}_j \in \mathbb{R}^{d_k}$ is the key vector for token $j$, encoding what information token $j$ contains. The value matrix $\mathbf{V}$ contains the actual information to aggregate. Each row $\mathbf{v}_j \in \mathbb{R}^{d_v}$ is the value vector for token $j$, representing the information that will be propagated to other tokens.

{{< figure
    src="/images/ml/attention.png"
    caption="The attention mechanism computes queries, keys, and values from input embeddings."
    alt="Diagram showing the computation of query, key, and value matrices from input embeddings"
    width="400"
>}}

Now we compute attention scores by taking dot products between all queries and all keys. The score matrix is:

$$
\mathbf{QK}^T \in \mathbb{R}^{N \times N}
$$

Entry $(i,j)$ of this matrix is $\mathbf{q}_i^T \mathbf{k}_j$, the dot product of the $i$-th query with the $j$-th key. This measures the compatibility between token $i$ and token $j$. A high dot product indicates that token $j$ is relevant to token $i$.

Importantly, this matrix is no longer symmetric. Since $\mathbf{Q}$ and $\mathbf{K}$ come from different transformations, we have $\mathbf{QK}^T \neq (\mathbf{QK}^T)^T$ in general. This allows the model to learn that token $i$ attends strongly to token $j$ while token $j$ attends weakly to token $i$, capturing the directional nature of linguistic relationships.

We then apply a scaling factor and softmax to convert scores into probability distributions:

$$
\mathbf{S} = \text{softmax}\left(\frac{\mathbf{QK}^T}{\sqrt{d_k}}\right) \in \mathbb{R}^{N \times N}
$$

The scaling by $\sqrt{d_k}$ is crucial for numerical stability. Without it, when $d_k$ is large, the dot products can have very large magnitude. This causes the softmax to produce highly peaked distributions (probabilities very close to 0 or 1), leading to vanishingly small gradients during training.

To understand why $\sqrt{d_k}$ is the appropriate scaling factor, assume the components of $\mathbf{q}$ and $\mathbf{k}$ are independent random variables with mean 0 and variance 1. The dot product $\mathbf{q}^T \mathbf{k} = \sum_{i=1}^{d_k} q_i k_i$ is the sum of $d_k$ products of independent random variables. Since each product $q_i k_i$ has mean 0 and variance 1, the sum has mean 0 and variance $d_k$ (the variance of a sum of independent variables is the sum of their variances). Dividing by $\sqrt{d_k}$ normalizes the variance back to 1, keeping the distribution of dot products stable regardless of dimension.

Finally, we compute the output as a weighted combination of the values:

$$
\mathbf{A} = \mathbf{SV} \in \mathbb{R}^{N \times d_v}
$$

Each row $\mathbf{a}_i$ of $\mathbf{A}$ is a weighted average of the value vectors, where the weights come from the attention distribution in row $i$ of $\mathbf{S}$. Specifically:

$$
\mathbf{a}_i = \sum_{j=1}^{N} S_{ij} \mathbf{v}_j
$$

where $S_{ij}$ is the attention weight that token $i$ places on token $j$, and $\mathbf{v}_j$ is the value vector for token $j$.

Putting it all together, the complete attention mechanism is:

$$
\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{QK}^T}{\sqrt{d_k}}\right) \mathbf{V}
$$

The beauty of this mechanism is that it is entirely learned through the parameter matrices $\mathbf{W}_Q, \mathbf{W}_K, \mathbf{W}_V$. The model learns what patterns to encode in queries (what each token needs), what features to encode in keys (what each token offers), and what information to encode in values (what will be propagated).

{{< figure
    src="/images/ml/attentionMechanism.png"
    caption="The complete self-attention mechanism from input embeddings to contextualized output."
    alt="Full diagram of the self-attention mechanism showing all computation steps"
>}}

{{< callout type="example" title="Running Example: Understanding Context" >}}
Consider the sentence "The bat flew out of the cave at night". Let us trace through how the word "bat" might attend to other words.

1. **Query**: The embedding of "bat" is transformed by $W^Q$ to produce a query vector that might encode something like "what type of bat am I?"

2. **Keys**: Each word produces a key vector. The key for "flew" might encode "I am a flying-related action", while "cave" might encode "I am an animal habitat".

3. **Attention scores**: We compute $q_{\text{bat}}^T k_{\text{flew}}$ and $q_{\text{bat}}^T k_{\text{cave}}$. If these dot products are high, it indicates relevance. The word "bat" would likely attend strongly to "flew" and "cave" because they disambiguate the meaning toward the animal rather than sports equipment.

4. **Values**: The value vectors for "flew" and "cave" contain information like "flight-related" and "nocturnal habitat". These get weighted by the attention scores.

5. **Output**: The output representation for "bat" is now enriched with information from "flew" and "cave", resolving the ambiguity. This contextualized representation is then passed to the next layer of the network.
{{< /callout >}}

### Parameter Count

Before discussing variants of attention, let us quantify the number of learnable parameters in a single attention head. The attention mechanism has four weight matrices: $\mathbf{W}_Q \in \mathbb{R}^{d \times d_k}$, $\mathbf{W}_K \in \mathbb{R}^{d \times d_k}$, $\mathbf{W}_V \in \mathbb{R}^{d \times d_v}$, and $\mathbf{W}_O \in \mathbb{R}^{(H \cdot d_v) \times d}$ for the output projection.

In the standard configuration where $d_k = d_v = \frac{d}{H}$ (splitting the model dimension equally across $H$ heads), each of the query, key, and value projection matrices has dimensions $d \times d$, containing $d^2$ parameters each. The output projection matrix $\mathbf{W}_O$ maps from the concatenated multi-head output (dimension $H \cdot \frac{d}{H} = d$) back to dimension $d$, requiring $d \times d = d^2$ parameters.

Therefore, a single multi-head attention block contains exactly:

$$
\text{Parameters} = d^2 + d^2 + d^2 + d^2 = 4d^2
$$

This parameter count is independent of the number of heads $H$ and sequence length $N$. The number of heads affects only how we partition the computation, not the total number of parameters. This is one reason why multi-head attention is efficient: we gain the benefits of multiple specialized attention patterns without increasing the parameter count.

### Computational Cost

Now let us analyze the computational cost, which determines practical limits on sequence length. For a single sequence of length $N$ with embedding dimension $d$, the attention mechanism consists of three main operations.

Computing the query, key, and value matrices requires three matrix multiplications: $\mathbf{Q} = \mathbf{EW}_Q$ multiplies $\mathbf{E} \in \mathbb{R}^{N \times d}$ by $\mathbf{W}_Q \in \mathbb{R}^{d \times d_k}$, costing $O(Nd \cdot d_k)$ operations. Similarly, $\mathbf{K} = \mathbf{EW}_K$ costs $O(Nd \cdot d_k)$ and $\mathbf{V} = \mathbf{EW}_V$ costs $O(Nd \cdot d_v)$. The total for all three projections is $O(Nd(2d_k + d_v))$.

Computing the attention scores $\mathbf{QK}^T$ requires multiplying $\mathbf{Q} \in \mathbb{R}^{N \times d_k}$ by $\mathbf{K}^T \in \mathbb{R}^{d_k \times N}$, involving $N^2$ dot products each of dimension $d_k$, costing $O(N^2 d_k)$. Applying softmax to each of the $N$ rows adds $O(N^2)$ operations.

Computing the output $\mathbf{A} = \mathbf{SV}$ requires multiplying $\mathbf{S} \in \mathbb{R}^{N \times N}$ by $\mathbf{V} \in \mathbb{R}^{N \times d_v}$, costing $O(N^2 d_v)$ operations.

The total time complexity is $O(Nd(2d_k + d_v) + N^2(d_k + d_v))$. In practice, we typically set $d_k = d_v = d$ for simplicity, which gives $O(Nd^2 + N^2 d)$. For short sequences where $N \ll d$, the $O(Nd^2)$ term from linear projections dominates. For long sequences where $N \gg d$, the $O(N^2 d)$ term dominates. This **quadratic scaling in sequence length** is the fundamental bottleneck, as computing all pairwise token interactions inherently requires $O(N^2)$ operations.

Memory consumption is often more constraining. We must store the attention matrix $\mathbf{S} \in \mathbb{R}^{N \times N}$, requiring $O(N^2)$ memory per attention head. For $N = 2'048$ tokens and $H = 16$ heads, a single layer requires $16 \times (2048)^2 \approx 67$ million values, or 268 MB in 32-bit precision. For longer sequences like $N = 8'192$ tokens, this becomes $16 \times (8192)^2 \approx 1.07$ billion values, or 4.3 GB per layer just for attention matrices.

A practical note: in real applications, we rarely process single sequences in isolation. The **batch size** $B$ is the number of sequences processed in parallel (multiple training examples during training, or concurrent user requests during inference). For a batch of $B$ sequences each with length $N$, the matrices become $\mathbf{E} \in \mathbb{R}^{B \times N \times d}$, and attention is computed independently for each sequence. The attention matrix becomes $\mathbf{S} \in \mathbb{R}^{B \times N \times N}$, multiplying memory usage by $B$. With $B = 32$ sequences of length $N = 8'192$ and $H = 16$ heads, a single layer requires $32 \times 16 \times (8192)^2 \approx 34$ billion values, or 137 GB in 32-bit precision.

This memory must reside in GPU VRAM for efficient computation, and VRAM is typically much more limited than system RAM (modern GPUs commonly have 16-80 GB VRAM). This severely constrains batch size for long sequences and motivates memory-efficient attention variants like Flash Attention, which we discuss later.

## Multi-Head Attention

A single attention mechanism learns one set of relationships between tokens. However, language has many different types of relationships operating simultaneously. In "Alice tried golf and tennis and she liked them", we have subject-verb relations, verb-object relations, and coreferences. A single attention mechanism would need to capture all of these patterns in one attention matrix, which would dilute the signal by averaging over fundamentally different types of relationships.

**Multi-head attention** addresses this by running multiple independent attention mechanisms in parallel. Each head has its own learned parameters $\mathbf{W}_Q^h, \mathbf{W}_K^h, \mathbf{W}_V^h$ and produces its own attention pattern. Different heads can specialize in different types of relationships. For example, different heads might focus on:
- Syntactic relationships (subject-verb, verb-object)
- Semantic similarity (synonyms, related concepts)
- Positional relationships (nearby words)
- Coreference (pronouns referring to entities)

Empirically, researchers have visualized attention patterns in trained Transformers and found that different heads indeed learn interpretable patterns. Some heads focus on the next token, some on syntactic heads in dependency trees, and some on rare words or specific parts of speech.

The mathematical motivation for multiple heads is that the optimization landscape for learning query-key-value matrices is non-convex with many local optima. Different random initializations can lead to different attention patterns. By training multiple heads in parallel, we increase the likelihood that at least some heads discover useful patterns. The final output projection then learns to optimally combine information from all heads.

Formally, for $H$ attention heads, we create $H$ independent sets of parameter matrices. For each head $h = 1, 2, \ldots, H$, we have:

$$
\mathbf{Q}_h = \mathbf{EW}_Q^h \in \mathbb{R}^{N \times d_k} \quad \mathbf{K}_h = \mathbf{EW}_K^h \in \mathbb{R}^{N \times d_k} \quad \mathbf{V}_h = \mathbf{EW}_V^h \in \mathbb{R}^{N \times d_v}
$$

where $\mathbf{W}_Q^h \in \mathbb{R}^{d \times d_k}$, $\mathbf{W}_K^h \in \mathbb{R}^{d \times d_k}$, and $\mathbf{W}_V^h \in \mathbb{R}^{d \times d_v}$ are the learnable parameters for head $h$. These are completely independent across heads, allowing each head to learn different transformations.

Each head computes its attention output independently using the standard attention formula:

$$
\mathbf{A}_h = \text{softmax}\left(\frac{\mathbf{Q}_h \mathbf{K}_h^T}{\sqrt{d_k}}\right) \mathbf{V}_h \in \mathbb{R}^{N \times d_v}
$$

The outputs from all $H$ heads are concatenated along the feature dimension. Let $\text{Concat}(\mathbf{A}_1, \mathbf{A}_2, \ldots, \mathbf{A}_H) \in \mathbb{R}^{N \times (H \cdot d_v)}$ denote the concatenation of all head outputs. We then apply a final linear projection to combine information from all heads:

$$
\text{MultiHead}(\mathbf{E}) = \text{Concat}(\mathbf{A}_1, \mathbf{A}_2, \ldots, \mathbf{A}_H) \mathbf{W}_O
$$

where $\mathbf{W}_O \in \mathbb{R}^{(H \cdot d_v) \times d}$ is a learnable output projection matrix. This projection allows the model to learn how to optimally combine the information from different heads.

In practice, we typically set $d_v = \frac{d}{H}$ so that the total dimension after concatenation is $H \cdot \frac{d}{H} = d$. With this choice, $\mathbf{W}_O \in \mathbb{R}^{d \times d}$ maps back to the original embedding dimension, and the output has shape $N \times d$, matching the input shape.

An important consideration is that multi-head attention does not increase the computational cost compared to single-head attention. With $H$ heads and dimension $\frac{d}{H}$ per head, each head computes attention scores $\mathbf{Q}_h \mathbf{K}_h^T$ with cost $O(N^2 \frac{d}{H})$. Computing all $H$ heads gives total cost $O(H \cdot N^2 \frac{d}{H}) = O(N^2 d)$, identical to single-head attention with full dimension $d$. The concatenation and final projection $\mathbf{W}_O$ add $O(Nd^2)$ operations. Thus multi-head attention maintains the same asymptotic complexity while enabling the model to learn diverse attention patterns. Modern hardware can compute all heads in parallel, making multi-head attention highly efficient in practice.

Beyond computational efficiency, multi-head attention provides two key advantages. First, constraining each head to work in a smaller representation subspace (dimension $\frac{d}{H}$) acts as **implicit regularization**, preventing individual heads from overfitting to spurious patterns. Second, combining multiple heads creates an **ensemble effect** similar to [ensemble methods in decision trees](/garden/ml/decisionTrees), where errors in individual heads can be compensated by others, leading to more robust predictions.

{{< figure
    src="/images/ml/attentionMultiHead.webp"
    caption="Multi-head attention runs multiple independent attention mechanisms in parallel, each learning different relationship patterns."
    alt="Diagram illustrating multi-head attention with parallel attention heads"
>}}

## Cross-Attention

So far we have discussed self-attention, where tokens in a sequence attend to other tokens within the same sequence. In many applications, we need to allow tokens in one sequence to attend to tokens in a different sequence. This is called **cross-attention** and is fundamental for tasks like machine translation, question answering, and image captioning.

Consider machine translation from English to German. We want to translate "Alice played tennis but she didn't like it" to "Alice hat Tennis gespielt aber es hat ihr nicht gefallen". The word order and grammatical structure differ between languages. When generating the German word "gespielt" (played), the model needs to attend back to the English word "played" to retrieve the appropriate semantic information, even though the words appear in different positions in their respective sentences.

Cross-attention enables this by maintaining two separate sequences while allowing information to flow from one to the other. We have a source sequence with embeddings $\mathbf{E}_s \in \mathbb{R}^{N_s \times d}$ and a target sequence with embeddings $\mathbf{E}_t \in \mathbb{R}^{N_t \times d}$, where $N_s$ and $N_t$ can differ.

The key modification is that queries come from the target sequence, while keys and values come from the source sequence:

$$
\mathbf{Q} = \mathbf{E}_t \mathbf{W}_Q \in \mathbb{R}^{N_t \times d_k} \quad \mathbf{K} = \mathbf{E}_s \mathbf{W}_K \in \mathbb{R}^{N_s \times d_k} \quad \mathbf{V} = \mathbf{E}_s \mathbf{W}_V \in \mathbb{R}^{N_s \times d_v}
$$

The target tokens generate queries for computing attention scores, while the source tokens provide keys to match against and values to aggregate. The attention computation remains:

$$
\mathbf{A} = \text{softmax}\left(\frac{\mathbf{QK}^T}{\sqrt{d_k}}\right) \mathbf{V}
$$

The output $\mathbf{A} \in \mathbb{R}^{N_t \times d}$ has one vector for each target token, where each vector is a weighted combination of source value vectors. The attention weight matrix $\mathbf{QK}^T \in \mathbb{R}^{N_t \times N_s}$ is now rectangular rather than square, with dimensions $N_t \times N_s$. Entry $(i,j)$ represents how much target token $i$ attends to source token $j$.

Cross-attention appears in several important architectures. In encoder-decoder models like the original Transformer for machine translation, the encoder processes the source sentence using self-attention to produce contextualized representations. The decoder then uses cross-attention to attend to these encoder outputs while generating the target sentence one token at a time. This allows each generated word to selectively attend to the entire source sentence and extract relevant information.

In question answering systems, we can encode a question and a document separately, then use cross-attention to identify which parts of the document are relevant to answering the question. In vision-language models, cross-attention connects different modalities. For example, given an image represented as a sequence of patch embeddings and a text caption, cross-attention allows the caption words to attend to relevant image regions.

The key property of cross-attention is that it maintains two separate sequences with potentially different lengths, semantic roles, or even modalities, while enabling selective information flow from source to target.

{{< figure
    src="/images/ml/attentionCross.webp"
    caption="Cross-attention enables information flow between two different sequences, with queries from the target and keys/values from the source."
    alt="Diagram showing cross-attention between source and target sequences"
>}}

## Masked Self-Attention

In generative language modeling, we generate text autoregressively, predicting one token at a time based on previously generated tokens. Consider generating an answer to the question "Why did the bicycle fall over?". During generation, the model produces tokens sequentially:

1. [prompt] → "Because"
2. [prompt] "Because" → "it"
3. [prompt] "Because it" → "was"
4. [prompt] "Because it was" → "two"
5. [prompt] "Because it was two" → "tired"

At each step $t$, the model can only condition on tokens at positions $1, \ldots, t-1$. It cannot "peek ahead" at future tokens because those do not exist yet during generation.

During training, we have access to the complete target sequence. We could simulate autoregressive generation by running the model $N$ times, once for each prefix. However, this is computationally wasteful. At step 1, we compute attention for 1 token. At step 2, we recompute attention for the first token and compute it for the second. This redundant computation grows quadratically with sequence length.

Instead, we can compute the entire sequence in parallel using **masked self-attention**. We process all tokens simultaneously but mask the attention mechanism so that each position can only attend to previous positions. This gives us the same outputs as sequential generation but with much better computational efficiency during training.

We implement masking by modifying the attention scores before applying softmax. Given the attention score matrix $\mathbf{P} = \mathbf{QK}^T \in \mathbb{R}^{N \times N}$, we define a causal mask $\mathbf{M} \in \{0,1\}^{N \times N}$:

$$
\mathbf{M} = \begin{bmatrix}
1 & 0 & 0 & \cdots & 0 \\
1 & 1 & 0 & \cdots & 0 \\
1 & 1 & 1 & \cdots & 0 \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
1 & 1 & 1 & \cdots & 1
\end{bmatrix}
$$

Entry $M_{ij} = 1$ if $j \leq i$ and $M_{ij} = 0$ otherwise. This is a lower triangular matrix indicating which positions each token is allowed to attend to.

We apply the mask before softmax by setting masked positions to $-\infty$:

$$
P_{\text{masked}}[i,j] = \begin{cases} P[i,j] & \text{if } M[i,j] = 1 \\ -\infty & \text{if } M[i,j] = 0 \end{cases}
$$

When we apply softmax, the exponential of $-\infty$ is zero, so masked positions receive zero attention weight:

$$
\mathbf{S} = \text{softmax}\left(\frac{\mathbf{P}_{\text{masked}}}{\sqrt{d_k}}\right), \quad \mathbf{A} = \mathbf{SV}
$$

This ensures that position $i$ only attends to positions $1, \ldots, i$, preserving the causal structure of autoregressive generation.

Masked attention is essential for causal language modeling. Without masking, the model could exploit future information during training, leading to artificially high training performance that does not transfer to generation. The model would learn to simply look ahead and copy future tokens rather than predict them based on context.

To understand how masked attention enables next token prediction, consider the training process. We have a training sequence like "The cat sat on the mat". The model processes this entire sequence in parallel with masked attention. At position 3 (the word "sat"), the attention mechanism can only attend to positions 1-3 ("The cat sat"), not to future positions 4-6 ("on the mat"). The output of the attention layer at position 3 is a contextualized representation that summarizes "The cat sat". This representation is then passed through feedforward layers and a final linear projection to predict a probability distribution over the vocabulary. The model is trained to predict the next token, which is "on" in this case.

During training, we compute this prediction for every position simultaneously. At position 1, we predict position 2. At position 2, we predict position 3. And so on. The loss function compares these predictions to the actual next tokens and updates the model parameters accordingly. The masking ensures that each position learns to predict the next token using only information from previous positions, exactly matching the autoregressive generation process at inference time where future tokens are not yet available.

The key insight is that masked attention provides the contextualized representations needed for next token prediction, while the masking constraint ensures these representations are computed using only past context. The actual prediction from representation to next token probability distribution is done by subsequent layers (typically a linear projection followed by softmax).

{{< figure
    src="/images/ml/attentionMasked.png"
    caption="Masked self-attention uses a causal mask to prevent tokens from attending to future positions."
    alt="Diagram showing masked attention with a lower triangular attention mask"
>}}


## Flash Attention

