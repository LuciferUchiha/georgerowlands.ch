---
title: CLIP
type: docs
weight: 3
---

CLIP (Contrastive Language-Image Pre-training) is a neural network architecture developed by OpenAI that learns to connect images and text through contrastive learning. The model addresses a fundamental challenge in multimodal learning by creating a shared embedding space where semantically similar images and text descriptions have similar representations.

Contrastive learning is a self-supervised learning approach where the model learns by comparing examples rather than from explicit labels. The key idea is to train the model to distinguish between similar (positive) pairs and dissimilar (negative) pairs.

## The Representation Alignment Problem

An autoencoder is a neural network that learns to compress data into a lower-dimensional representation (encoding) and then reconstruct the original data from this compressed form (decoding). The encoder maps inputs to a latent representation that captures the essential features, while the decoder reconstructs the input from this representation. Autoencoders are useful for dimensionality reduction, denoising, and learning meaningful representations of data without supervision.

Traditional autoencoders create their own internal representations independently. When we train an image encoder separately from a text encoder, the representations produced by one autoencoder are seldom understandable by another. Consider training an image encoder that maps pictures of cars to some vector in a high-dimensional space, and separately training a text encoder that maps the word "car" to another vector. These two vectors will likely be far apart in their respective spaces because the encoders were trained independently with different objectives.

This incompatibility prevents us from performing cross-modal tasks. For example, we cannot decode an image representation into text or search for images using text descriptions when the two modalities live in separate embedding spaces.

The core challenge is to harmonize these representations and bring them into a shared space. We need a way to ensure that an image of a dog and the text "dog" produce similar representations in a shared embedding space. CLIP solves this problem by training both encoders jointly using a contrastive objective that pulls matching image-text pairs together while pushing non-matching pairs apart.

{{< figure
    src="/images/ml/clipVectorSpace.png"
    caption="CLIP learns a shared embedding space where semantically related images and text are mapped to nearby points. The text 'sunset over mountains' and images of mountains and sunsets cluster together, while unrelated concepts remain distant."
    alt="CLIP shared embedding space visualization"
>}}

## Architecture

CLIP consists of two main components that operate in parallel. An image encoder processes visual inputs and a text encoder processes language inputs. Both encoders are followed by trainable linear projection layers and normalization layers that map their outputs into a shared embedding space of the same dimensionality.

{{< figure
    src="/images/ml/clip.svg"
    caption="CLIP architecture showing the image encoder and text encoder processing their respective inputs, followed by linear projection layers that map both modalities into a shared embedding space."
    alt="CLIP architecture diagram"
>}}

### Image Encoder

The goal of the image encoder is to extract meaningful visual features from images and map them into a vector space where semantically similar images are close together. This encoder transforms raw pixel values into a compact representation that captures the essential visual content.

The image encoder can be implemented using different architectures. The original CLIP paper explored both convolutional and transformer-based approaches. The convolutional variant uses a modified ResNet architecture where the global average pooling layer is replaced with an [attention pooling mechanism](/garden/ml/attention/). The transformer variant uses a Vision Transformer (ViT) that splits the input image into patches and processes them as a sequence.

Regardless of the specific architecture, the image encoder produces a feature vector $\mathbf{v}_i \in \mathbb{R}^d$ for each input image $I_i$. This feature vector captures the visual content but may have arbitrary magnitude. The vector is then passed through a learnable linear projection $W_I \in \mathbb{R}^{d \times d_e}$ followed by L2 normalization to produce the final image embedding.

$$
\mathbf{v}_i = \frac{W_I \cdot \text{ImageEncoder}(I_i)}{\| W_I \cdot \text{ImageEncoder}(I_i) \|_2}
$$

The linear projection $W_I$ allows the model to learn a transformation that maps the encoder's output space into the shared embedding space with dimension $d_e$. This projection is crucial because the image encoder and text encoder may naturally produce features of different dimensions or characteristics. The projection layer learns to align these spaces during training.

The L2 normalization ensures that all embeddings lie on the unit hypersphere. This normalization serves two important purposes. First, it makes the dot product between embeddings equivalent to the cosine similarity, which measures the angle between vectors rather than their magnitudes. This focuses the model on learning directional relationships in the embedding space. Second, it prevents any single embedding from dominating the similarity computations due to having a large magnitude. All embeddings are treated equally in terms of scale.

### Text Encoder

The text encoder can use a [Transformer](/garden/ml/attention/) architecture to process text descriptions. The input text is tokenized and embedded using learned token embeddings combined with positional encodings. The Transformer processes this sequence using self-attention layers to produce contextualized representations.

The CLIP text encoder uses causal (masked) self-attention rather than bidirectional attention. This means each token can only attend to previous tokens in the sequence, not future ones. While this might seem less expressive than bidirectional attention (as used in BERT), it aligns with the autoregressive nature of language modeling and was found to work well in practice. The causal masking is a design choice that simplifies the architecture while still capturing rich linguistic relationships.

The text encoder extracts a feature vector from the final transformer layer, typically from a special end-of-sequence token that has attended to the entire input sequence. This feature vector is projected through a learnable linear layer $W_T \in \mathbb{R}^{d \times d_e}$ and L2 normalized to produce the final text embedding $\mathbf{w}_j \in \mathbb{R}^{d_e}$.

$$
\mathbf{w}_j = \frac{W_T \cdot \text{TextEncoder}(T_j)}{\| W_T \cdot \text{TextEncoder}(T_j) \|_2}
$$

## Contrastive Learning Objective

The key insight behind CLIP is to use contrastive learning to align the image and text embeddings. During training, we process a batch of $N$ image-text pairs $(I_i, T_i)$ where each image $I_i$ is paired with its corresponding text description $T_i$. The model should learn to produce high similarity between matching pairs and low similarity between non-matching pairs.

### Similarity Matrix

After encoding all images and texts in the batch through their respective encoders and projection layers, we obtain normalized embeddings $\mathbf{v}_i$ and $\mathbf{w}_j$ that all lie on the unit hypersphere. The normalization was applied during the encoding process, ensuring that the dot product between any two embeddings equals their cosine similarity.

We compute the pairwise cosine similarities between all image and text embeddings in the batch. This creates a similarity matrix $S \in \mathbb{R}^{N \times N}$ where each element is:

$$
S_{ij} = \mathbf{v}_i^\top \mathbf{w}_j
$$

The diagonal elements $S_{ii}$ represent the similarity between matching pairs (the image of a dog and the text "dog"), while the off-diagonal elements represent similarities between non-matching pairs (the image of a dog and the text "cat"). The goal is to make the diagonal elements large (close to 1) and the off-diagonal elements small (close to 0 or negative).

{{< figure
    src="/images/ml/clipSimilarityMatrix.png"
    caption="The similarity matrix computed from image and text embeddings. Diagonal elements (blue) represent matching pairs and should have high values, while off-diagonal elements represent non-matching pairs and should have low values."
    alt="CLIP similarity matrix visualization"
>}}

### Cross-Entropy Loss

The objective is to make the similarity matrix resemble the identity matrix $I_N$. For each row $i$, we want $S_{ii}$ to be much larger than all other elements in that row. This is achieved using the cross-entropy loss.

For the image-to-text direction, we treat each row of $S$ as logits for a classification problem where the correct class is the corresponding text (the diagonal entry). The loss for image $i$ is:

$$
\mathcal{L}_i^{\text{i2t}} = -\log \frac{\exp(S_{ii})}{\sum_{j=1}^N \exp(S_{ij})} = -\log \frac{\exp(\mathbf{v}_i^\top \mathbf{w}_i)}{\sum_{j=1}^N \exp(\mathbf{v}_i^\top \mathbf{w}_j)}
$$

This is the standard softmax cross-entropy loss. To understand the connection to probabilities, recall that the softmax function converts the raw similarity scores $S_{ij}$ into a probability distribution. For a given image $I_i$, we interpret the model's predicted probability that text $T_j$ is the correct match as:

$$
p(T_j \mid I_i) = \frac{\exp(S_{ij})}{\sum_{k=1}^N \exp(S_{ik})} = \frac{\exp(\mathbf{v}_i^\top \mathbf{w}_j)}{\sum_{k=1}^N \exp(\mathbf{v}_i^\top \mathbf{w}_k)}
$$

This is a valid probability distribution over all $N$ candidate texts because all values are non-negative and sum to 1. We can now rewrite the loss in terms of this probability distribution:

$$
\begin{align*}
\mathcal{L}_i^{\text{i2t}} &= -\log \frac{\exp(S_{ii})}{\sum_{j=1}^N \exp(S_{ij})} \\
&= -\log p(T_i \mid I_i)
\end{align*}
$$

The true distribution is a one-hot vector where the probability mass is entirely on the correct text $T_i$. The [cross-entropy loss](/garden/ml/bayesian/variationalinference/#cross-entropy) between the predicted distribution and this true distribution is:

$$
\begin{align*}
\mathcal{L}_i^{\text{i2t}} &= -\sum_{j=1}^N \mathbb{1}[j = i] \log p(T_j \mid I_i) \\
&= -\log p(T_i \mid I_i)
\end{align*}
$$

Thus, minimizing this loss encourages the model to assign high probability to the correct matching text.

For the text-to-image direction, we perform the symmetric operation using the columns of $S$. For each text $j$, we compute:

$$
\mathcal{L}_j^{\text{t2i}} = -\log \frac{\exp(S_{jj})}{\sum_{i=1}^N \exp(S_{ij})} = -\log \frac{\exp(\mathbf{w}_j^\top \mathbf{v}_j)}{\sum_{i=1}^N \exp(\mathbf{w}_j^\top \mathbf{v}_i)}
$$

The total CLIP loss is the average of both directions:

$$
\mathcal{L}_{\text{CLIP}} = \frac{1}{2N} \sum_{i=1}^N \mathcal{L}_i^{\text{i2t}} + \frac{1}{2N} \sum_{j=1}^N \mathcal{L}_j^{\text{t2i}}
$$

This symmetric formulation ensures that both encoders receive gradients from both directions. The image encoder learns to produce embeddings that can retrieve the correct text, and the text encoder learns to produce embeddings that can retrieve the correct image.

This is the essence of contrastive learning. For each positive pair (matching image and text), the model contrasts it against $N-1$ negative pairs (the image with all other texts, and the text with all other images). By minimizing this loss, the model learns to maximize the [mutual information](/garden/ml/bayesian/activelearning/#mutual-information) between paired images and texts. Mutual information measures how much knowing one variable reduces uncertainty about another. Maximizing it means the image embedding should tell us as much as possible about the paired text, and vice versa. From a metric learning perspective, we are learning a shared embedding space where the distance between matching pairs is small and the distance between non-matching pairs is large.

### Temperature Scaling

The similarity scores are scaled by a learnable temperature parameter $\tau$ before computing the softmax. This scaling is crucial for controlling the concentration of the probability distribution and the effective learning rate of the contrastive loss. The scaled similarity matrix becomes:

$$
L_{ij} = \frac{S_{ij}}{\tau} = \frac{\mathbf{v}_i^\top \mathbf{w}_j}{\tau}
$$

The temperature parameter determines how sharply the model distinguishes between similar and dissimilar pairs.

{{< callout type="example" title="Temperature Scaling Effect" >}}
Suppose for a dog image, the similarities are: dog text = 0.8, cat text = 0.3, car text = 0.1.

With a small temperature $\tau = 0.01$:
- Scaled logits: [80, 30, 10]
- Softmax probabilities: [1.0, 0.0, 0.0] (extremely confident)

With a large temperature $\tau = 1.0$:
- Scaled logits: [0.8, 0.3, 0.1]
- Softmax probabilities: [0.57, 0.31, 0.12] (more uniform)
{{< /callout >}}

A small temperature (approaching zero) creates very sharp peaks in the softmax distribution, making the model highly confident in its predictions. This can be beneficial when the model is certain about matches, but it can also make training unstable if the model makes mistakes early on. A large temperature creates a more uniform distribution, which can help early in training when the model is uncertain, but may slow down learning if the temperature remains too high.

In CLIP, the temperature is implemented as a learnable parameter initialized to correspond to $\tau = 0.07$. Rather than fixing this value, the model learns the optimal temperature during training. The parameter is constrained to prevent scaling the logits by more than 100 (corresponding to $\tau \geq 0.01$), which helps maintain training stability and prevents numerical issues.

The final loss formulation with temperature scaling for the image-to-text direction is:

$$
\mathcal{L}_i^{\text{i2t}} = -\log \frac{\exp\left(\frac{\mathbf{v}_i^\top \mathbf{w}_i}{\tau}\right)}{\sum_{j=1}^N \exp\left(\frac{\mathbf{v}_i^\top \mathbf{w}_j}{\tau}\right)}
$$

The other direction is analogous.

## Training

CLIP was trained on a dataset of 400 million image-text pairs collected from the internet. The training data consists of images and their associated text (such as captions, alt-text, or surrounding text from web pages). This approach leverages naturally occurring supervision from the internet rather than requiring manually labeled datasets.

The model is trained end-to-end using the contrastive loss described above. Both the image encoder and text encoder are initialized randomly (or from pretrained weights) and updated jointly through backpropagation. The linear projection layers and temperature parameter are also learned during training.

The training procedure can be summarized as follows:

1. **Sample a batch**: Randomly sample $N$ image-text pairs from the training dataset.
2. **Encode images**: Pass all $N$ images through the image encoder, apply the linear projection, and L2 normalize to obtain image embeddings $\{\mathbf{v}_1, \ldots, \mathbf{v}_N\}$.
3. **Encode texts**: Pass all $N$ texts through the text encoder, apply the linear projection, and L2 normalize to obtain text embeddings $\{\mathbf{w}_1, \ldots, \mathbf{w}_N\}$.
4. **Compute similarity matrix**: Calculate all pairwise dot products to form the $N \times N$ similarity matrix $S$.
5. **Apply temperature scaling**: Divide all similarities by the learnable temperature parameter $\tau$ to obtain logits $L = \frac{S}{\tau}$.
6. **Compute loss**: Calculate the cross-entropy loss in both directions (image-to-text and text-to-image) and average them to get the total CLIP loss.
7. **Backpropagate**: Compute gradients of the loss with respect to all parameters (encoder weights, projection weights, temperature) and update them using an optimizer like Adam.

The batch size $N$ plays a critical role in training effectiveness. Each batch provides $N$ positive pairs (the diagonal elements) and $N(N-1)$ negative pairs (the off-diagonal elements). For a batch of 32 images, we get 32 positive pairs but 992 negative pairs. This abundance of negative examples in a single forward pass is key to learning good representations. Larger batch sizes provide more negative examples, which generally leads to better learned representations. The original CLIP was trained with very large batch sizes (up to 32,768) to maximize the number of negative pairs per training step.

This simple training procedure is remarkably effective. The model learns rich visual and textual representations without requiring any task-specific labels or annotations beyond the natural pairing of images and text.

## Zero-Shot Transfer

One of the most powerful capabilities demonstrated in the CLIP paper is zero-shot transfer to downstream tasks. After pretraining, the model can perform image classification on arbitrary categories without any task-specific training. This works by converting the classification problem into an image-text matching problem.

For a classification task with classes $\{c_1, c_2, \ldots, c_K\}$, we create text descriptions for each class. The CLIP paper explored various prompt templates, finding that using descriptive prompts improved performance over simple class names. These descriptions can be simple templates like "a photo of a {class}" or more sophisticated prompts like "a photo of a {class}, a type of {category}". We encode all $K$ text descriptions using the text encoder to obtain text embeddings $\{\mathbf{w}_1, \mathbf{w}_2, \ldots, \mathbf{w}_K\}$.

For a test image $I$, we encode it using the image encoder to obtain $\mathbf{v}$. We then compute the cosine similarity between $\mathbf{v}$ and each of the class text embeddings:

$$
\text{score}(I, c_k) = \mathbf{v}^\top \mathbf{w}_k
$$

The predicted class is the one with the highest similarity:

$$
\hat{c} = \arg\max_{k} \mathbf{v}^\top \mathbf{w}_k
$$

To convert these scores into probabilities, we can apply a softmax function with temperature scaling:

$$
p(c_k \mid I) = \frac{\exp\left(\frac{\mathbf{v}^\top \mathbf{w}_k}{\tau}\right)}{\sum_{j=1}^K \exp\left(\frac{\mathbf{v}^\top \mathbf{w}_j}{\tau}\right)}
$$

This zero-shot classification ability is remarkable because the model can classify images into categories it has never explicitly seen during training. The model leverages its understanding of both visual concepts and language to bridge the gap. If the model has learned good representations of "dogs" and good representations of the word "dog", it can correctly classify dog images even if it never saw labeled dog images during pretraining.

The quality of zero-shot classification depends heavily on the prompt engineering. Using informative prompts like "a photo of a {class}, a type of pet" often works better than simple prompts like "{class}". Ensemble methods that average predictions from multiple prompt templates can further improve performance.

{{< figure
    src="/images/ml/clipClassification.svg"
    caption="Zero-shot classification with CLIP: text descriptions for each class are encoded and compared with the image embedding. The class with the highest similarity is selected as the prediction."
    alt="CLIP zero-shot classification process"
>}}

## Image-Text Retrieval

CLIP naturally supports bidirectional retrieval between images and text. Given a text query, we can retrieve the most relevant images from a database. Given an image query, we can retrieve the most relevant text descriptions.

For text-to-image retrieval, we encode the query text to obtain $\mathbf{w}_{\text{query}}$. We encode all database images to obtain $\{\mathbf{v}_1, \mathbf{v}_2, \ldots, \mathbf{v}_M\}$. We compute similarities and rank images by their similarity to the query:

$$
\text{similarity}(T_{\text{query}}, I_m) = \mathbf{w}_{\text{query}}^\top \mathbf{v}_m
$$

Image-to-text retrieval works symmetrically. These retrieval capabilities enable applications like semantic image search, where users can search for images using natural language descriptions rather than keywords.

{{< figure
    src="/images/ml/clipRetrieval.png"
    caption="CLIP enables bidirectional retrieval: given a text query, find the most similar images, or given an image query, find the most similar text descriptions."
    alt="CLIP retrieval capabilities"
>}}
