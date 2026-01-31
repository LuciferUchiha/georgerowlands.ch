---
title: Transformers
type: docs
weight: 5
---

The Transformer architecture, introduced in the 2017 paper [Attention is All You Need](https://arxiv.org/abs/1706.03762) by Vaswani et al., represents a fundamental breakthrough in sequence modeling. Before Transformers, the dominant architectures for tasks like machine translation were recurrent networks (RNNs, LSTMs, GRUs). While these models achieved impressive results, they suffered from a critical limitation: sequential processing. Each token had to be processed one at a time, with hidden states passed from one step to the next. This sequential dependency prevented parallel computation and made it difficult to capture long-range dependencies in sequences.

The Transformer eliminates recurrence entirely, relying instead on the [attention mechanism](/garden/ml/llms/attention) to model relationships between all positions in the sequence. This architectural choice enables full parallelization during training, dramatically reducing training time while improving the model's ability to capture dependencies regardless of their distance in the sequence. The architecture has become the foundation for modern large language models like GPT, BERT, and Claude.

Consider translating the English sentence "I am a student" to French: "Je suis étudiant". A recurrent model would process this word by word, maintaining a hidden state that attempts to remember all relevant information as it progresses. By the time it reaches "student", the information about "I am" may have been compressed and transformed through multiple recurrent steps. The Transformer, in contrast, allows the decoder to directly attend to "I" when generating "Je", to "am" when generating "suis", and to "student" when generating "étudiant", computing attention scores between all pairs of words in parallel. This direct access to all source tokens, regardless of their distance, is the key innovation that enables the Transformer to excel at capturing dependencies in sequences.

https://jalammar.github.io/illustrated-transformer/
https://nlp.seas.harvard.edu/2018/04/03/attention.html

## The Architecture Overview

The original Transformer follows an encoder-decoder architecture designed for sequence-to-sequence tasks like machine translation. The encoder processes the input sequence (for example, "I am a student") and produces a sequence of continuous representations, one vector for each input token. These are dense, high-dimensional vectors (typically 512 or 768 dimensions) that encode both the meaning of each token and its context within the sentence. Unlike discrete token IDs, these continuous representations can capture nuanced semantic relationships and are differentiable, enabling gradient-based learning. The decoder then generates the output sequence one token at a time (for example, "Je", "suis", "étudiant"), attending to both the encoder outputs and previously generated tokens.

{{< figure
    src="/images/ml/transformers.webp"
    caption="The Transformer architecture with encoder (left) and decoder (right) stacks."
    alt="Diagram of the full Transformer encoder-decoder architecture"
    width="500"
>}}

Both the encoder and decoder are composed of stacks of identical layers. The original paper uses $N = 6$ identical layers for both encoder and decoder. Each layer contains multiple sub-components built from attention mechanisms and feedforward networks, with residual connections and layer normalization applied throughout.

The key insight is that all operations within a layer can be computed in parallel across all positions in the sequence. Unlike recurrent networks where position $t$ must wait for position $t-1$, the Transformer computes representations for all positions simultaneously. This parallelization is the primary source of the Transformer's computational efficiency during training.

## Positional Encoding

https://kazemnejad.com/blog/transformer_architecture_positional_encoding/

The attention mechanism as described in the [attention notes](/garden/ml/llms/attention) operates on sets rather than sequences. When we compute attention between queries and keys, the operation is permutation-invariant. If we shuffle the input tokens, the attention weights would be shuffled correspondingly, but the mechanism itself has no inherent notion of position. This is problematic for language, where word order fundamentally determines meaning. The sentences "Alice helped Bob" and "Bob helped Alice" have very different meanings despite containing the same words.

Recurrent networks naturally encode position through their sequential processing. The hidden state at step $t$ implicitly knows it comes after step $t-1$. Since Transformers abandon this sequential structure, we must explicitly inject positional information into the input representations.

The Transformer achieves this through **positional encodings**, fixed vectors that are added to the input embeddings before the first layer. Let $\mathbf{E} \in \mathbb{R}^{N \times d}$ be the embedding matrix for a sequence of $N$ tokens, where each row $\mathbf{e}_i \in \mathbb{R}^d$ is the embedding of token $i$. We define a positional encoding matrix $\mathbf{P} \in \mathbb{R}^{N \times d}$ where each row $\mathbf{p}_i \in \mathbb{R}^d$ encodes the position $i$. The input to the first Transformer layer is then:

$$
\mathbf{X} = \mathbf{E} + \mathbf{P} \in \mathbb{R}^{N \times d}
$$

Each row $\mathbf{x}_i = \mathbf{e}_i + \mathbf{p}_i$ combines semantic information (from the token embedding) with positional information (from the positional encoding).

The addition operation itself does not give more weight to certain positions. The positional encodings have similar magnitudes across all positions, so no position is inherently emphasized over others. However, the addition serves a deeper purpose: it allows the model to use both semantic and positional information when computing attention. When we compute queries and keys as $\mathbf{Q} = \mathbf{X}\mathbf{W}_Q$ and $\mathbf{K} = \mathbf{X}\mathbf{W}_K$, the weight matrices $\mathbf{W}_Q$ and $\mathbf{W}_K$ can learn to extract and combine both types of information. For example, the model might learn that when processing a verb, it should attend to subjects that appear earlier in the sequence (using positional information) and that have semantic features compatible with being agents (using semantic information). The model learns through training how to weight the importance of position versus content for different linguistic patterns.

### Sinusoidal Positional Encoding

The original Transformer uses a deterministic function based on sinusoids of different frequencies to generate positional encodings. For position $i$ and dimension $j$ in the embedding space, the positional encoding is:

$$
\mathbf{P}[i,j] = \begin{cases}
\sin\left(\frac{i}{10000^{2k/d}}\right) & \text{if } j = 2k \text{ (even dimension)} \\
\cos\left(\frac{i}{10000^{2k/d}}\right) & \text{if } j = 2k+1 \text{ (odd dimension)}
\end{cases}
$$

where $k = \lfloor j/2 \rfloor$ determines which sine-cosine pair this dimension belongs to. Each dimension uses a different frequency, ranging from $2\pi$ (for low dimensions) to $10000 \times 2\pi$ (for high dimensions).

The intuition behind this design is that each dimension oscillates at a different frequency. Low-frequency components change slowly with position, encoding coarse positional information (for example, whether a word is in the first or second half of the sentence). High-frequency components change rapidly, encoding fine-grained distinctions between nearby positions (for example, distinguishing position 17 from position 18).

To understand why sinusoids are particularly suitable, consider the requirement that the model should be able to learn relative positions. For any fixed offset $k$, we want the encoding at position $i+k$ to be representable as a linear function of the encoding at position $i$. Sinusoidal functions satisfy this property through the trigonometric addition formulas:

$$
\sin(\alpha + \beta) = \sin(\alpha)\cos(\beta) + \cos(\alpha)\sin(\beta)
$$
$$
\cos(\alpha + \beta) = \cos(\alpha)\cos(\beta) - \sin(\alpha)\sin(\beta)
$$

This means that for any fixed offset $k$, the encoding at position $i+k$ can be expressed as a linear transformation of the encoding at position $i$. Specifically, setting $\alpha = i/10000^{2j/d}$ and $\beta = k/10000^{2j/d}$, we have:

$$
\begin{pmatrix}
\sin\left(\frac{i+k}{10000^{2j/d}}\right) \\
\cos\left(\frac{i+k}{10000^{2j/d}}\right)
\end{pmatrix}
= \begin{pmatrix}
\cos\left(\frac{k}{10000^{2j/d}}\right) & \sin\left(\frac{k}{10000^{2j/d}}\right) \\
-\sin\left(\frac{k}{10000^{2j/d}}\right) & \cos\left(\frac{k}{10000^{2j/d}}\right)
\end{pmatrix}
\begin{pmatrix}
\sin\left(\frac{i}{10000^{2j/d}}\right) \\
\cos\left(\frac{i}{10000^{2j/d}}\right)
\end{pmatrix}
$$

This linear relationship allows the model to easily learn to attend to relative positions. For example, when processing position $i$, the model can learn a linear transformation that allows it to identify or attend to position $i-3$ or $i+5$ by leveraging this structure.

Another advantage of sinusoidal encodings is that they naturally generalize to sequence lengths not seen during training. Since the encoding is defined by a mathematical function rather than learned parameters, we can compute positional encodings for arbitrarily long sequences. A learned positional embedding lookup table would be limited to the maximum sequence length seen during training.

An alternative approach is **binary positional encoding**, where each position is represented by its binary representation. For example, position 5 in binary is 101, which could be encoded as the vector $[1, 0, 1, 0, 0, \ldots]$. While this is simple and guarantees unique encodings for each position, it has significant drawbacks. Binary representations do not encode any notion of distance between positions (position 7 and position 8 differ only in the last bit, but 7 and 15 differ in multiple bits). The sinusoidal approach is preferred because nearby positions have similar encodings, and the model can learn to use the smooth variation in frequencies to understand relative positions.

{{< figure
    src="/images/ml/transformerPositionalEncoding.png"
    caption="Sinusoidal positional encodings visualized across positions and dimensions, showing the wave patterns at different frequencies."
    alt="Heatmap visualization of sinusoidal positional encodings"
    width="600"
>}}

## Encoder Architecture

The Transformer encoder consists of a stack of $N$ identical layers (typically $N=6$). Each layer has two main sub-components: a multi-head self-attention mechanism and a position-wise feedforward network. Both sub-components are wrapped with residual connections and layer normalization for stable training and better gradient flow.

{{< figure
    src="/images/ml/transformerEncoder.png"
    caption="A single encoder layer showing multi-head self-attention, feedforward network, and residual connections with layer normalization."
    alt="Detailed diagram of a Transformer encoder layer"
    width="400"
>}}

### Multi-Head Self-Attention

The first sub-component is a multi-head self-attention layer as described in detail in the [attention mechanism notes](/garden/ml/llms/attention#multi-head-attention). The input $\mathbf{X} \in \mathbb{R}^{N \times d}$ (which for the first layer is the sum of embeddings and positional encodings) is transformed through multiple attention heads operating in parallel.

For each head $h$, we compute:

$$
\mathbf{Q}_h = \mathbf{X}\mathbf{W}_Q^h, \quad \mathbf{K}_h = \mathbf{X}\mathbf{W}_K^h, \quad \mathbf{V}_h = \mathbf{X}\mathbf{W}_V^h
$$

where $\mathbf{W}_Q^h, \mathbf{W}_K^h, \mathbf{W}_V^h \in \mathbb{R}^{d \times d_k}$ are learnable parameter matrices. The attention output for head $h$ is:

$$
\mathbf{A}_h = \text{softmax}\left(\frac{\mathbf{Q}_h\mathbf{K}_h^T}{\sqrt{d_k}}\right)\mathbf{V}_h
$$

The outputs from all $H$ heads (typically $H=8$) are concatenated and projected:

$$
\mathbf{Z}_{\text{attn}} = \text{Concat}(\mathbf{A}_1, \ldots, \mathbf{A}_H)\mathbf{W}_O \in \mathbb{R}^{N \times d}
$$

where $\mathbf{W}_O \in \mathbb{R}^{Hd_k \times d}$ is the output projection matrix. In practice, $d_k = d/H$, so the concatenated heads have dimension $d$.

This self-attention mechanism allows each token in the encoder to attend to all tokens in the input sequence, including itself. For our translation example "I am a student", when encoding the word "student", the attention mechanism can gather information from "I am" to understand the context. This contextualized representation of "student" captures that it refers to the speaker in the first person.

### Feedforward Network

After the attention sub-layer, each position is processed independently through an identical feedforward network. This network consists of two linear transformations with a nonlinear activation function (typically ReLU or GELU) in between:

$$
\text{FFN}(\mathbf{z}) = \text{ReLU}(\mathbf{z}\mathbf{W}_1 + \mathbf{b}_1)\mathbf{W}_2 + \mathbf{b}_2
$$

where $\mathbf{W}_1 \in \mathbb{R}^{d \times d_{\text{ff}}}$, $\mathbf{b}_1 \in \mathbb{R}^{d_{\text{ff}}}$, $\mathbf{W}_2 \in \mathbb{R}^{d_{\text{ff}} \times d}$, and $\mathbf{b}_2 \in \mathbb{R}^d$ are learnable parameters. The intermediate dimension $d_{\text{ff}}$ is typically much larger than $d$ (the original paper uses $d_{\text{ff}} = 2048$ with $d = 512$).

This feedforward network is applied independently to each token. The same network with the same parameters processes token 1, token 2, and so on. This is sometimes called "position-wise" because the operation is identical across positions but applied separately to each one. Unlike the attention mechanism which mixes information across tokens, the feedforward network processes each token in isolation.

The role of the feedforward network is to provide additional representational capacity and nonlinearity. While attention is powerful for aggregating information across tokens, it is fundamentally a weighted average operation. The feedforward network allows each token to transform its aggregated representation in a nonlinear way. The large intermediate dimension $d_{\text{ff}}$ provides the model with a high-dimensional space to compute complex functions of the attended representation.

### Residual Connections and Layer Normalization

Both the attention sub-layer and the feedforward sub-layer are wrapped with residual connections and layer normalization. The residual connection allows gradients to flow directly through the network, addressing the vanishing gradient problem in deep networks. For a sub-layer with function $f$, the output is:

$$
\text{Output} = \text{LayerNorm}(\mathbf{X} + f(\mathbf{X}))
$$

The layer normalization normalizes activations across the feature dimension for each token independently. For a vector $\mathbf{x} \in \mathbb{R}^d$, layer normalization computes:

$$
\text{LayerNorm}(\mathbf{x}) = \gamma \odot \frac{\mathbf{x} - \mathbb{E}[\mathbf{x}]}{\sqrt{\text{Var}[\mathbf{x}] + \epsilon}} + \beta
$$

where $\gamma, \beta \in \mathbb{R}^d$ are learnable scale and shift parameters, $\epsilon$ is a small constant for numerical stability, and the mean and variance are computed over the $d$ dimensions of $\mathbf{x}$. This operation is applied independently to each token in the sequence.

Putting it all together, a single encoder layer performs the following operations:

$$
\begin{align*}
\mathbf{Z}_1 &= \text{LayerNorm}(\mathbf{X} + \text{MultiHeadAttention}(\mathbf{X}, \mathbf{X}, \mathbf{X})) \\
\mathbf{Z}_2 &= \text{LayerNorm}(\mathbf{Z}_1 + \text{FFN}(\mathbf{Z}_1))
\end{align*}
$$

where $\mathbf{Z}_2$ is the output of the layer and becomes the input to the next layer. The final encoder layer produces the encoder outputs $\mathbf{H} \in \mathbb{R}^{N \times d}$, a sequence of continuous representations that encode the input sequence.

## Decoder Architecture

The decoder is also composed of a stack of $N$ identical layers (typically $N=6$). Each decoder layer has three sub-components rather than two: masked multi-head self-attention, cross-attention to the encoder outputs, and a position-wise feedforward network. Like the encoder, all sub-components use residual connections and layer normalization.

{{< figure
    src="/images/ml/transformerEncoderDecoder.png"
    caption="The full encoder-decoder architecture showing how the decoder attends to encoder outputs through cross-attention."
    alt="Diagram of encoder-decoder interaction in Transformers"
    width="500"
>}}

### Masked Self-Attention

The first sub-component in each decoder layer is a masked self-attention mechanism. During training, we have access to the complete target sequence, but we must prevent tokens from attending to future tokens. This maintains the autoregressive property required for generation, where each token can only depend on previously generated tokens.

Consider translating "I am a student" to French: "Je suis étudiant". During training, when the decoder processes the token "suis", it should only have access to the previously generated tokens in the target sequence (the start token and "Je"), not to future tokens ("étudiant"). Otherwise, the model could simply learn to copy future target tokens rather than learning to generate them based on context. At inference time, future tokens do not exist yet, so this masking ensures the model learns in a way that matches the generation process.

The masking is implemented exactly as described in the [masked self-attention section](/garden/ml/llms/attention#masked-self-attention) of the attention notes. Before applying softmax, we set future token positions to $-\infty$:

$$
\mathbf{M}[i,j] = \begin{cases}
0 & \text{if } j \leq i \\
-\infty & \text{if } j > i
\end{cases}
$$

The masked attention computation becomes:

$$
\text{MaskedAttention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{QK}^T + \mathbf{M}}{\sqrt{d_k}}\right)\mathbf{V}
$$

After softmax, positions where the mask was $-\infty$ have attention weight exactly 0, ensuring no information leaks from future tokens.

### Cross-Attention

The second sub-component is a cross-attention layer that allows the decoder to attend to the encoder outputs. This is where the decoder extracts relevant information from the source sequence to generate the target sequence.

As explained in the [cross-attention section](/garden/ml/llms/attention#cross-attention) of the attention notes, queries come from the decoder while keys and values come from the encoder. Let $\mathbf{Y} \in \mathbb{R}^{M \times d}$ be the output of the previous decoder sub-layer (the masked self-attention) and $\mathbf{H} \in \mathbb{R}^{N \times d}$ be the encoder outputs. The cross-attention computes:

$$
\mathbf{Q} = \mathbf{Y}\mathbf{W}_Q, \quad \mathbf{K} = \mathbf{H}\mathbf{W}_K, \quad \mathbf{V} = \mathbf{H}\mathbf{W}_V
$$

$$
\text{CrossAttention}(\mathbf{Y}, \mathbf{H}) = \text{softmax}\left(\frac{\mathbf{QK}^T}{\sqrt{d_k}}\right)\mathbf{V}
$$

The attention weight matrix has shape $M \times N$, where entry $(i,j)$ indicates how much target token $i$ attends to source token $j$. This allows the decoder to selectively focus on relevant parts of the input. When generating the French word "étudiant" (student), the decoder can attend strongly to the English word "student" in the encoder outputs to retrieve the appropriate semantic content.

The cross-attention mechanism is what enables the encoder-decoder architecture to perform sequence-to-sequence tasks. The encoder builds representations of the source sequence, and the decoder uses cross-attention to query these representations when generating each target token.

### Feedforward Network

The third sub-component is a feedforward network, identical in structure to the one used in the encoder:

$$
\text{FFN}(\mathbf{z}) = \text{ReLU}(\mathbf{z}\mathbf{W}_1 + \mathbf{b}_1)\mathbf{W}_2 + \mathbf{b}_2
$$

This is applied to each token independently after the cross-attention sub-layer.

### Complete Decoder Layer

A complete decoder layer performs these operations in sequence:

$$
\begin{align*}
\mathbf{Y}_1 &= \text{LayerNorm}(\mathbf{Y} + \text{MaskedMultiHeadAttention}(\mathbf{Y}, \mathbf{Y}, \mathbf{Y})) \\
\mathbf{Y}_2 &= \text{LayerNorm}(\mathbf{Y}_1 + \text{CrossAttention}(\mathbf{Y}_1, \mathbf{H})) \\
\mathbf{Y}_3 &= \text{LayerNorm}(\mathbf{Y}_2 + \text{FFN}(\mathbf{Y}_2))
\end{align*}
$$

where $\mathbf{Y}$ is the input to the layer, $\mathbf{H}$ is the encoder output, and $\mathbf{Y}_3$ is the output that feeds into the next decoder layer.

The first sub-layer allows the decoder to build representations based on previously generated tokens through masked self-attention. The second sub-layer incorporates information from the source sequence through cross-attention. The third sub-layer provides additional nonlinear transformation capacity. Together, these three components enable the decoder to generate the target sequence conditioned on both the source sequence and previously generated target tokens.

After the final decoder layer, we have a sequence of continuous representations $\mathbf{Y}_{\text{final}} \in \mathbb{R}^{M \times d}$, one vector for each token in the target sequence. To convert these to probability distributions over the vocabulary, we apply a final linear transformation followed by softmax:

$$
\mathbf{P}[i,:] = \text{softmax}(\mathbf{y}_i \mathbf{W}_{\text{vocab}} + \mathbf{b}_{\text{vocab}})
$$

where $\mathbf{W}_{\text{vocab}} \in \mathbb{R}^{d \times V}$ and $\mathbf{b}_{\text{vocab}} \in \mathbb{R}^V$ are learnable parameters, $V$ is the vocabulary size, and $\mathbf{P}[i,:]$ is the probability distribution over vocabulary tokens for position $i$.

{{< figure
    src="/images/ml/transformerOutputDistributions.png"
    caption="The decoder outputs probability distributions over the vocabulary for each position in the target sequence."
    alt="Diagram showing output probability distributions for each token position"
    width="500"
>}}

## Training

During training, we have access to both the source sequence and the ground truth target sequence $\mathbf{t} = [t_1, t_2, \ldots, t_M]$. The model is trained to maximize the log probability of the correct tokens using the cross-entropy loss:

$$
\mathcal{L} = -\sum_{i=1}^{M} \log P(t_i \mid t_{<i}, \mathbf{x})
$$

where $t_{<i}$ denotes all target tokens before position $i$, and $\mathbf{x}$ is the source sequence. This is the negative log-likelihood of the target sequence given the source.

The training process uses **teacher forcing**, where the model is fed the ground truth previous tokens when predicting the next token, rather than its own predictions. At position $i$, the decoder input contains the ground truth tokens $[t_1, t_2, \ldots, t_{i-1}]$ even if the model's predictions at earlier positions were incorrect. This significantly speeds up training because we can compute the loss for all positions in parallel. Without teacher forcing, we would need to generate tokens sequentially during training, which would be much slower.

For example, when training on "I am a student" → "Je suis étudiant", the decoder receives the input sequence $[\text{start}, \text{Je}, \text{suis}]$ and must predict the probability distribution for the next token at each position. At position 1 (after the start token), it should predict "Je". At position 2 (after "Je"), it should predict "suis". At position 3 (after "suis"), it should predict "étudiant". The loss compares these predicted distributions to the actual ground truth tokens. Crucially, even if the model incorrectly predicts position 1, at position 2 the decoder still receives the ground truth token "Je" as input, not the model's incorrect prediction. This prevents error accumulation during training.

The masking in the decoder self-attention ensures that token $i$ cannot attend to tokens $i+1, i+2, \ldots$ during training. Combined with parallel computation, this allows us to compute the predictions for all positions simultaneously while maintaining the autoregressive property. The loss at position $i$ is computed using only information from positions $1$ through $i-1$, exactly as would occur during sequential generation at inference time.

The model parameters (all the weight matrices in attention layers, feedforward networks, and the output projection) are updated using backpropagation and gradient descent (typically with the Adam optimizer). The gradients flow backward through the entire network, updating both the encoder and decoder parameters to minimize the cross-entropy loss.

## Inference and Decoding

At inference time, we do not have access to the target sequence. We must generate it autoregressively, one token at a time, with each new token conditioned on the previously generated tokens.

The process begins with a special start-of-sequence token. The encoder processes the source sequence once to produce encoder outputs $\mathbf{H} \in \mathbb{R}^{N \times d}$. The decoder then generates tokens sequentially. At each step, the decoder processes all currently generated tokens through all $N$ layers in a single forward pass, using the representation of the last token to predict the next token:

1. **Step 1**: The decoder input contains only the start token. The decoder produces a distribution over the vocabulary for position 1. We sample or select the most likely token as $\hat{t}_1$.

2. **Step 2**: The decoder input now contains $[\text{start}, \hat{t}_1]$. The decoder produces distributions for positions 1 and 2, but we only care about position 2 since position 1 has already been generated. We select $\hat{t}_2$.

3. **Step 3**: The decoder input is $[\text{start}, \hat{t}_1, \hat{t}_2]$, and we generate $\hat{t}_3$.

This continues until the model generates a special end-of-sequence token or reaches a maximum length.

{{< figure
    src="/images/ml/transformerInference.gif"
    caption="Autoregressive generation process showing how the decoder sequentially produces tokens, with each new token conditioned on all previously generated tokens."
    alt="Animation of the Transformer inference process"
    width="600"
>}}

### Greedy Decoding and Beam Search

The decoding problem is fundamentally a search problem. At each generation step, the model produces a probability distribution over the entire vocabulary (often 30,000+ tokens). For a sequence of length $M$, there are $V^M$ possible sequences, where $V$ is the vocabulary size. For realistic values like $V = 30000$ and $M = 10$, this is approximately $10^{44}$ possible sequences. Exhaustively evaluating all possible sequences is computationally infeasible. We need efficient search strategies to find high-probability sequences without exploring this exponentially large space.

The simplest decoding strategy is **greedy decoding**, where at each step we select the token with the highest probability:

$$
\hat{t}_i = \arg\max_{t} P(t \mid \hat{t}_{<i}, \mathbf{x})
$$

Greedy decoding is fast and deterministic but can be suboptimal. Since it makes locally optimal choices at each step, it may miss globally better sequences. Consider translating "I am" to French. If the model assigns high probability to "Je" at the first position, greedy decoding commits to this choice. However, if "J'" (the contracted form before a vowel, as in "J'ai" meaning "I have") would lead to a better overall sentence for a different English input, greedy decoding cannot reconsider.

**Beam search** addresses this by maintaining multiple hypotheses simultaneously. Instead of keeping only the single best token at each step, we keep the top $K$ complete sequences (the "beam"), where $K$ is the beam width. At each step:

1. For each of the $K$ current hypotheses, compute the probability distribution over the next token.

2. This produces $K \times V$ possible next tokens (where $V$ is vocabulary size).

3. Keep the top $K$ sequences by total probability, where the probability of a sequence is:

$$
P(\hat{t}_{1:i}) = \prod_{j=1}^{i} P(\hat{t}_j \mid \hat{t}_{<j}, \mathbf{x})
$$

In practice, we work with log probabilities to avoid numerical underflow:

$$
\log P(\hat{t}_{1:i}) = \sum_{j=1}^{i} \log P(\hat{t}_j \mid \hat{t}_{<j}, \mathbf{x})
$$

4. Repeat until all $K$ hypotheses produce an end-of-sequence token or reach maximum length.

5. Return the sequence with the highest probability (or log probability).

Beam search explores a larger portion of the search space than greedy decoding while remaining computationally tractable. The beam width $K$ trades off between search quality and computational cost. Larger $K$ explores more hypotheses but requires more computation. Typical values are $K = 4$ or $K = 8$ for machine translation.

{{< figure
    src="/images/ml/transformerBeamSearch.jpeg"
    caption="Beam search maintains multiple hypotheses at each step, keeping the top K sequences by probability."
    alt="Diagram illustrating beam search with multiple hypotheses"
    width="600"
>}}

## Computational Complexity

Understanding the computational complexity of the Transformer is essential for understanding its practical applicability and limitations. We analyze the complexity per layer in terms of sequence length $N$ and embedding dimension $d$. The table below summarizes the key operations:

| Operation | Time Complexity | Memory | Bottleneck |
|-----------|----------------|---------|------------|
| **Self-Attention** | | | |
| Linear projections ($\mathbf{Q}, \mathbf{K}, \mathbf{V}$) | $O(Nd^2)$ | $O(Nd)$ | Dominates for short sequences |
| Attention scores ($\mathbf{QK}^T$) | $O(N^2d)$ | $O(N^2)$ | Dominates for long sequences |
| Weighted sum ($\mathbf{SV}$) | $O(N^2d)$ | $O(Nd)$ | |
| **Feedforward Network** | | | |
| Two linear layers | $O(Ndd_{\text{ff}}) = O(Nd^2)$ | $O(Nd_{\text{ff}})$ | With $d_{\text{ff}} = 4d$ |
| **Total per layer** | $O(Nd^2 + N^2d)$ | $O(N^2 + Nd)$ | |

For self-attention, the linear projections to compute $\mathbf{Q}, \mathbf{K}, \mathbf{V}$ each require a matrix multiplication $\mathbf{X}\mathbf{W}$ where $\mathbf{X} \in \mathbb{R}^{N \times d}$ and $\mathbf{W} \in \mathbb{R}^{d \times d}$, costing $O(Nd^2)$ operations. The attention score computation $\mathbf{QK}^T$ requires $N^2$ dot products of dimension $d$, costing $O(N^2d)$ operations. The weighted sum $\mathbf{SV}$ costs another $O(N^2d)$ operations.

For short sequences where $N \ll d$, the $O(Nd^2)$ term from linear projections dominates. For long sequences where $N \gg d$, the $O(N^2d)$ term dominates. This **quadratic scaling in sequence length** is the fundamental bottleneck of self-attention.

Memory consumption is also critical. The attention matrix $\mathbf{S} \in \mathbb{R}^{N \times N}$ requires $O(N^2)$ memory per head. For $H$ heads, a single layer requires $O(HN^2)$ memory. This grows quadratically with sequence length, limiting the maximum context length that can fit in GPU memory.

The feedforward network performs two linear transformations, with total complexity $O(Ndd_{\text{ff}})$. Since $d_{\text{ff}}$ is typically $4d$ (for example, $d_{\text{ff}} = 2048$ and $d = 512$ in the original paper), this becomes $O(Nd^2)$ operations.

A single Transformer layer (encoder or decoder) has total complexity:

$$
O(Nd^2 + N^2d)
$$

For a Transformer with $L$ layers, the total complexity is:

$$
O(L(Nd^2 + N^2d))
$$

The quadratic dependence on sequence length $N$ in the attention mechanism limits the applicability of Transformers to very long sequences. For sequences of length $N = 512$, the $N^2$ term is manageable. For $N = 8192$ or longer, the memory and computation requirements become prohibitive. This has motivated research into more efficient attention mechanisms such as sparse attention, linear attention, and flash attention, though these are beyond the scope of the original Transformer architecture.

## KV Cache Optimization

The complexity analysis above assumes we compute attention from scratch at each step. However, during autoregressive generation at inference time, we can exploit the sequential nature of token production to dramatically reduce computation through caching.

### The Naive Generation Problem

Without caching, autoregressive generation has a critical inefficiency. When generating token $t$, we must process all $t$ previous tokens through all $L$ layers, which costs $O(L(t^2 d + td^2))$. To generate an entire sequence of length $N$, the total cost is:

$$
\sum_{t=1}^{N} O(L(t^2 d + td^2)) = O(LN^3 d + LN^2 d^2)
$$

using the formula $\sum_{t=1}^{N} t^2 = \frac{N(N+1)(2N+1)}{6} \approx \frac{N^3}{3}$. This **cubic scaling** in sequence length arises because we redundantly recompute the same key and value vectors: token 1's key and value are computed $N$ times, token 2's are computed $N-1$ times, and so on.

### Caching Solution

At inference time, the Transformer produces tokens sequentially, with each new token conditioned on all previously generated tokens. Without optimization, we would recompute the entire attention mechanism from scratch at each generation step. This is wasteful because previously generated tokens and their embeddings remain fixed, yet we would recompute their key and value representations every time.

The encoder outputs $\mathbf{H}$ are fixed for the entire generation process, so we compute them once and reuse them for all decoder steps through cross-attention. The challenge is in the decoder self-attention. Consider generating a translation token by token. At step $t$, we have generated tokens $x_1, \ldots, x_{t-1}$ and want to generate $x_t$. The decoder self-attention computation requires:

$$
\mathbf{Q}_t = \mathbf{E}_t \mathbf{W}_Q \in \mathbb{R}^{1 \times d_k} \quad \mathbf{K}_{1:t} = \mathbf{E}_{1:t} \mathbf{W}_K \in \mathbb{R}^{t \times d_k} \quad \mathbf{V}_{1:t} = \mathbf{E}_{1:t} \mathbf{W}_V \in \mathbb{R}^{t \times d_v}
$$

where $\mathbf{E}_{1:t} \in \mathbb{R}^{t \times d}$ contains embeddings for all tokens from position 1 to $t$. At the next step $t+1$, we need:

$$
\mathbf{Q}_{t+1} = \mathbf{E}_{t+1} \mathbf{W}_Q \in \mathbb{R}^{1 \times d_k} \quad \mathbf{K}_{1:t+1} = \mathbf{E}_{1:t+1} \mathbf{W}_K \in \mathbb{R}^{(t+1) \times d_k} \quad \mathbf{V}_{1:t+1} = \mathbf{E}_{1:t+1} \mathbf{W}_V \in \mathbb{R}^{(t+1) \times d_v}
$$

The matrix $\mathbf{K}_{1:t+1}$ contains $\mathbf{K}_{1:t}$ as its first $t$ rows, and similarly for $\mathbf{V}_{1:t+1}$. Without caching, we recompute these overlapping portions at every step. Over the course of generating $N$ tokens, token 1's key and value are computed $N$ times, token 2's are computed $N-1$ times, and so on. The total number of redundant key-value computations is:

$$
\sum_{i=1}^{N} i = \frac{N(N+1)}{2} = O(N^2)
$$

For a Transformer with $L$ layers, the total cost without caching is $O(LN^2 d^2)$ for the linear projections alone, making long-form generation prohibitively expensive.

The key insight enabling **KV caching** is that keys and values for a given token depend only on the fixed weight matrices $\mathbf{W}_K, \mathbf{W}_V$ and that token's embedding. Once a token is generated, its key and value never change. We can compute and store them once, then reuse them for all subsequent generation steps. At step $t$, we maintain a cache:

$$
\text{Cache}_\mathbf{K}^{(t)} = \mathbf{K}_{1:t} = \begin{bmatrix} \mathbf{k}_1^T \\ \mathbf{k}_2^T \\ \vdots \\ \mathbf{k}_t^T \end{bmatrix} \in \mathbb{R}^{t \times d_k} \quad \text{Cache}_\mathbf{V}^{(t)} = \mathbf{V}_{1:t} = \begin{bmatrix} \mathbf{v}_1^T \\ \mathbf{v}_2^T \\ \vdots \\ \mathbf{v}_t^T \end{bmatrix} \in \mathbb{R}^{t \times d_v}
$$

At step $t+1$, we only compute the key and value for the new token:

$$
\mathbf{k}_{t+1} = \mathbf{E}_{t+1} \mathbf{W}_K \in \mathbb{R}^{1 \times d_k} \quad \mathbf{v}_{t+1} = \mathbf{E}_{t+1} \mathbf{W}_V \in \mathbb{R}^{1 \times d_v}
$$

We then update the cache by concatenation:

$$
\text{Cache}_\mathbf{K}^{(t+1)} = \begin{bmatrix} \text{Cache}_\mathbf{K}^{(t)} \\ \mathbf{k}_{t+1} \end{bmatrix} \in \mathbb{R}^{(t+1) \times d_k} \quad \text{Cache}_\mathbf{V}^{(t+1)} = \begin{bmatrix} \text{Cache}_\mathbf{V}^{(t)} \\ \mathbf{v}_{t+1} \end{bmatrix} \in \mathbb{R}^{(t+1) \times d_v}
$$

The query is always computed fresh for the current token since we only need it for this step:

$$
\mathbf{Q}_{t+1} = \mathbf{E}_{t+1} \mathbf{W}_Q \in \mathbb{R}^{1 \times d_k}
$$

The attention computation at step $t+1$ uses the cached keys and values:

$$
\begin{align*}
\mathbf{S}_{t+1} &= \text{softmax}\left(\frac{\mathbf{Q}_{t+1} (\text{Cache}_\mathbf{K}^{(t+1)})^T}{\sqrt{d_k}}\right) \in \mathbb{R}^{1 \times (t+1)} \\
\mathbf{A}_{t+1} &= \mathbf{S}_{t+1} \cdot \text{Cache}_\mathbf{V}^{(t+1)} \in \mathbb{R}^{1 \times d_v}
\end{align*}
$$

This produces the contextualized representation for token $t+1$, which is passed through feedforward layers to predict token $t+2$.

With KV caching, each token's key and value are computed exactly once. The cost at step $t$ is $O(d \cdot d_k + d \cdot d_v)$ for computing $\mathbf{k}_t$ and $\mathbf{v}_t$, plus $O(t \cdot d_k)$ for the attention score computation $\mathbf{Q}_t (\text{Cache}_\mathbf{K}^{(t)})^T$ and $O(t \cdot d_v)$ for the weighted sum $\mathbf{S}_t \cdot \text{Cache}_\mathbf{V}^{(t)}$. Assuming $d_k = d_v = d$, the cost per step is $O(d^2 + td)$. Summing over $N$ generation steps:

$$
\text{Total cost} = \sum_{t=1}^{N} O(d^2 + td) = O(Nd^2 + \sum_{t=1}^{N} td) = O(Nd^2 + N^2 d)
$$

Comparing to the uncached version with cost $O(N^2 d^2)$, when $d \gg N$ (typical for large models with short generation), caching provides approximately $\frac{N^2 d^2}{Nd^2} = N \times$ speedup. For modern language models generating hundreds of tokens, KV caching provides 10-100× speedup, making interactive generation feasible.

The computational savings come at a memory cost. For a single attention layer with $H$ heads, each head has dimension $d_k = d_v = \frac{d}{H}$. For a sequence of length $N$, we store:

$$
\begin{align*}
\text{Keys per layer:} &\quad N \times H \times \frac{d}{H} = Nd \text{ values} \\
\text{Values per layer:} &\quad N \times H \times \frac{d}{H} = Nd \text{ values} \\
\text{Total per layer:} &\quad 2Nd \text{ values}
\end{align*}
$$

For a Transformer with $L$ decoder layers, the total is $2LNd$ values. Using 16-bit floating point precision (2 bytes per value):

$$
\text{KV Cache Size} = 2LNd \times 2 \text{ bytes} = 4LNd \text{ bytes}
$$

{{< callout type="example" title="KV Cache Memory: Large-Scale Translation" >}}
Consider a large Transformer with $L = 6$ decoder layers, $d = 768$ dimensions, and generating a translation of length $N = 512$ tokens in 16-bit precision.

Per sequence:

$$
\text{KV Cache} = 4 \times 6 \times 512 \times 768 = 9'437'184 \text{ bytes} \approx 9 \text{ MB}
$$

For batch size $B = 32$ (processing 32 translations concurrently):

$$
\text{Total Memory} = 32 \times 9 \text{ MB} = 288 \text{ MB}
$$

This is manageable for most modern GPUs. However, for very large models or longer sequences, the KV cache can become a significant memory bottleneck. For a GPT-3 scale model with $L = 96$ layers, $d = 12'288$ dimensions, and context length $N = 2'048$ tokens, the cache per sequence is approximately 9 GB, making batch processing challenging.
{{< /callout >}}

This memory requirement must reside in GPU VRAM for efficient computation. In production systems serving many users, each request maintains its own KV cache, making memory management across GPUs complex. Systems must carefully batch requests with similar sequence lengths to maximize GPU utilization while avoiding out-of-memory errors.
