---
title: Graph Neural Networks
type: docs
weight: 20
math: true
---

Many real-world phenomena are naturally represented as graphs rather than vectors or grids. Molecules are graphs where atoms are nodes and chemical bonds are edges. Social networks connect people through friendships. Knowledge graphs link entities through semantic relationships. Traffic networks model road intersections and connections. In all these cases, the structure of the data carries essential information that would be lost if we simply flattened it into a vector.

Consider the task of predicting whether a molecule is toxic. Looking at the chemical structures below, we see that molecules like acrylonitrile, cyanobenzene, acetonitrile, and cyanogen chloride are toxic because they contain the cyano group (C≡N), which can release cyanide. Meanwhile, water, ethanol, glucose, and even ferricyanide (which contains six cyano groups bound to iron) are not toxic because the iron atom in ferricyanide holds the cyano groups tightly together, preventing cyanide release. This demonstrates that predicting molecular properties requires understanding both the local features (what atoms are present) and the global structure (how they are connected).

The challenge is that standard neural network architectures cannot directly process graphs. Vectors have fixed dimensions. Images have fixed spatial structure that convolutions can exploit. Text can be tokenized into sequences of fixed vocabulary. But graphs have variable numbers of nodes, variable connectivity patterns, and no canonical ordering of their vertices. We need a new type of neural network architecture designed specifically for graph-structured data.

## Graphs and Annotated Graphs

A **graph** $G = (V, E)$ consists of a non-empty set $V$ of vertices (also called nodes) and a set $E \subseteq V \times V$ of edges connecting pairs of vertices. In an undirected graph, if $(u, v) \in E$, then $(v, u) \in E$ as well, meaning the edge set is symmetric. The **neighborhood** of a vertex $u$, denoted $N(u)$, is the set of all vertices connected to $u$ by an edge. The **degree** of a vertex $\deg(u)$ is the number of edges incident to it, which equals $|N(u)|$ for undirected graphs.

In most applications, we work with **annotated graphs** where vertices and edges carry additional information. An annotated graph $G = (V, E, h^V, h^E, h_0)$ extends the basic graph structure with:

1. A function $h^V: V \to \mathbb{R}^{d_V}$ that assigns a feature vector to each vertex
2. A function $h^E: E \to \mathbb{R}^{d_E}$ that assigns a feature vector to each edge
3. A global feature vector $h_0 \in \mathbb{R}^{d_0}$ for the entire graph

For molecules, the vertex features $h^V$ might encode atom type (carbon, nitrogen, oxygen), charge, and hybridization state. The edge features $h^E$ might encode bond type (single, double, triple) and whether the bond is part of an aromatic ring. The global features $h_0$ might encode overall molecular properties like molecular weight.

For simplicity, we will focus on the case where only vertices have feature vectors, so an annotated graph is $G = (V, E, h^V)$. We also assume undirected graphs throughout, though all formulations extend naturally to directed graphs.

Given a distribution $p^*$ over annotated graphs and an unknown target function $f^*(G) \in \mathbb{R}^m$, our goal is to learn an approximation to $f^*$ from training examples $(G_i, y_i)$ where $G_i \sim p^*$ and $y_i = f^*(G_i)$. This is a supervised learning problem, but the input space consists of graphs rather than fixed-dimensional vectors.

## From Neural Networks to Graph Filters

To understand how to design neural networks for graphs, let us first analyze how standard neural networks process their inputs. In a fully connected network, we iteratively process representations through layers. If $h^{(\ell)}$ is the representation at layer $\ell$, the representation at layer $\ell + 1$ is computed by:

$$
h^{(\ell+1)} = \phi\left(\sum_i (w_i^\top h^{(\ell)} + b_i)\right)
$$

where $w_i$ and $b_i$ are the weights and biases, and $\phi$ is an element-wise activation function. The key operations are: **transforming** the input (multiplying by weights), **aggregating** the transformed values (summing), and **applying a nonlinearity** (the activation function).

Convolutional neural networks (CNNs) apply this same pattern locally. A convolutional filter processes a spatial patch of the image by transforming each pixel value in the patch, aggregating these transformed values with a sum, and applying an activation function. The filter slides across the image, applying the same operation at each location. This weight sharing makes CNNs efficient and translation-invariant.

To generalize this to graphs, we need to define what a "patch" means in a graph and how to aggregate information from it. The natural analog of a spatial neighborhood in an image is the set of neighbors of a vertex in a graph. For each vertex $u$, we define its patch as the vertex itself together with its neighbors $N(u) \cup \{u\}$.

A **graph filter** then processes each vertex by:
1. **Transforming** the feature vectors of the vertex and its neighbors
2. **Aggregating** these transformed features
3. **Applying an activation function**

The crucial constraint is that the aggregation must be **permutation invariant**: since there is no canonical ordering of neighbors, swapping the order of neighbors must not change the output. Functions like sum, mean, and max satisfy this property.

## Graph Convolutional Networks

The **Graph Convolutional Network (GCN)**, introduced by Kipf and Welling in their influential 2017 paper, implements these ideas with a specific choice of transformation and aggregation. Given vertex features $h_u$ for each $u \in V$, the GCN computes new features as:

$$
h_u' = \phi\left(\frac{1}{\sqrt{\deg(u)}} \sum_{v \in N(u) \cup \{u\}} \frac{1}{\sqrt{\deg(v)}} h_v W\right)
$$

Here $W \in \mathbb{R}^{d \times d'}$ is a learnable weight matrix shared across all vertices, and $\phi$ is an activation function (typically ReLU). The normalization by $\frac{1}{\sqrt{\deg(u)\deg(v)}}$ keeps features on the same scale regardless of vertex degree. Without this normalization, vertices with many neighbors would have much larger feature magnitudes than vertices with few neighbors, which could destabilize training.

We can stack multiple such layers to obtain a deep GCN. Let $h_u^{(\ell)}$ denote the features of vertex $u$ at layer $\ell$, with $h_u^{(0)}$ being the initial vertex features. The update from layer $\ell$ to layer $\ell + 1$ is:

$$
h_u^{(\ell+1)} = \phi\left(\frac{1}{\sqrt{\deg(u)}} \sum_{v \in N(u) \cup \{u\}} \frac{1}{\sqrt{\deg(v)}} h_v^{(\ell)} W^{(\ell)}\right)
$$

where $W^{(\ell)}$ is the weight matrix for layer $\ell$.

### Matrix Formulation

This layer-wise update can be written compactly in matrix form. Let $H^{(\ell)} \in \mathbb{R}^{n \times d}$ be the matrix whose rows are the feature vectors of all $n$ vertices at layer $\ell$. Let $A \in \{0, 1\}^{n \times n}$ be the adjacency matrix of the graph, where $A_{uv} = 1$ if $(u, v) \in E$ and $A_{uv} = 0$ otherwise. Define:

$$
\tilde{A} = A + I
$$

This is the adjacency matrix with added self-loops, ensuring each vertex aggregates its own features along with its neighbors'. Let $\tilde{D}$ be the diagonal degree matrix of $\tilde{A}$, where $\tilde{D}_{ii} = 1 + \deg(i)$. Then the GCN update becomes:

$$
H^{(\ell+1)} = \phi\left(\tilde{D}^{-\frac{1}{2}} \tilde{A} \tilde{D}^{-\frac{1}{2}} H^{(\ell)} W^{(\ell)}\right)
$$

The matrix $S = \tilde{D}^{-\frac{1}{2}} \tilde{A} \tilde{D}^{-\frac{1}{2}}$ is the symmetrically normalized adjacency matrix with self-loops. Entry $S_{uv}$ equals $\frac{1}{\sqrt{\tilde{D}_{uu}\tilde{D}_{vv}}}$ if $u$ and $v$ are neighbors (or $u = v$), and zero otherwise. This matrix formulation is efficient because it can be computed with sparse matrix operations, and it integrates seamlessly with automatic differentiation frameworks like PyTorch.

### Understanding the Normalization

Why do we normalize by the degrees? Consider what happens without normalization. The aggregation would simply sum the features of all neighbors:

$$
h_u' = \phi\left(\sum_{v \in N(u) \cup \{u\}} h_v W\right)
$$

A vertex with 100 neighbors would receive 100 times more "signal" than a vertex with 1 neighbor. The features would grow with degree, making it difficult for the network to learn degree-independent patterns. The symmetric normalization $\frac{1}{\sqrt{\deg(u)\deg(v)}}$ addresses this by:

1. Dividing by $\sqrt{\deg(v)}$: A vertex $v$ with high degree sends its signal to many neighbors, so each neighbor receives a fraction proportional to $1/\sqrt{\deg(v)}$. This is the "broadcaster" normalization.

2. Dividing by $\sqrt{\deg(u)}$: A vertex $u$ with high degree receives signals from many neighbors, so the total is normalized by $\sqrt{\deg(u)}$. This is the "receiver" normalization.

The choice of square roots (rather than full degrees) is motivated by spectral graph theory, where this normalization makes the matrix $S$ have eigenvalues in the range $[-1, 1]$, which helps with training stability.

## Message Passing Neural Networks

The GCN is one instance of a broader class of architectures called **Message Passing Neural Networks (MPNNs)**, formalized by Gilmer et al. in their 2017 paper. The MPNN framework provides a unifying view of many graph neural network variants.

An MPNN operates in two phases: a **message passing phase** and a **readout phase**. During message passing, each vertex iteratively updates its hidden state by receiving "messages" from its neighbors. This continues for $T$ steps (layers). The readout phase then aggregates the final vertex representations into a graph-level output.

### Message Passing Phase

At each step $t$, vertices exchange messages along edges. The message from vertex $v$ to vertex $u$ depends on the hidden states of both vertices and optionally the edge features:

$$
m_{uv}^{(t)} = M_t(h_u^{(t)}, h_v^{(t)}, e_{uv})
$$

where $M_t$ is a learnable message function and $e_{uv}$ are edge features. Each vertex then aggregates all incoming messages and updates its hidden state:

$$
h_u^{(t+1)} = U_t\left(h_u^{(t)}, \bigoplus_{v \in N(u)} m_{uv}^{(t)}\right)
$$

Here $\bigoplus$ denotes a permutation-invariant aggregation (sum, mean, or max), and $U_t$ is a learnable update function, typically implemented as a neural network.

An important intuition is that stacking $k$ message passing layers allows each vertex to integrate information from vertices up to $k$ hops away. After one layer, a vertex knows about its immediate neighbors. After two layers, it knows about neighbors of neighbors. This defines the **receptive field** of the GNN, analogous to the receptive field of a CNN that grows with network depth.

### Readout Phase

After $T$ message passing steps, the vertex representations are aggregated into a single graph-level representation:

$$
\hat{y} = R\left(\{h_u^{(T)} \mid u \in V\}\right)
$$

The readout function $R$ must also be permutation invariant since there is no canonical vertex ordering. Common choices include summing all vertex features, taking their mean, or using more sophisticated aggregations like attention-weighted sums.

### Connection to Classical Algorithms

The message passing paradigm is powerful because it encompasses many classical distributed algorithms on graphs. In a message passing algorithm, each vertex acts as a computing unit with local memory and communication channels to its neighbors. The algorithm proceeds in rounds: in each round, vertices send messages to neighbors based on their current state, receive messages, and update their state accordingly.

{{< callout type="example" title="Breadth-First Search as Message Passing" >}}
Consider computing shortest path distances from a source vertex $s$ to all other vertices. We can formulate this as message passing:

**Initialization**: Set $d(s) = 0$ and $d(v) = \infty$ for all $v \neq s$.

**Message**: Each vertex $u$ sends the message $d(u) + 1$ to all neighbors.

**Update**: Each vertex $v$ updates its distance: $d(v) \leftarrow \min(d(v), \min_{u \in N(v)} m_{uv})$.

After $k$ rounds, each vertex knows its distance from $s$ if that distance is at most $k$. This is exactly breadth-first search, implemented as message passing.
{{< /callout >}}

Many other classical graph algorithms follow this pattern: Dijkstra's algorithm for weighted shortest paths, algorithms for computing minimum spanning trees, leader election in distributed systems, and the Bellman-Ford algorithm. The fact that GNNs generalize message passing algorithms provides theoretical support for their expressiveness.

## Expressiveness of Graph Neural Networks

A natural question is: how expressive are GNNs? Can they approximate any function on graphs, analogous to how standard neural networks can approximate any continuous function on vectors?

Rauchwerger et al. demonstrated that message-passing GNNs are **universal approximators** on the space of attributed graphs. The key insight is that GNNs can simulate message passing algorithms, which compute a wide variety of graph functions. However, there are fundamental limits to what any message passing architecture can distinguish.

### The Weisfeiler-Lehman Hierarchy

The expressiveness of GNNs is intimately connected to the **Weisfeiler-Lehman (WL) graph isomorphism test**. The 1-WL algorithm (also called color refinement) iteratively updates vertex "colors" based on their neighbors' colors:

1. Initially, all vertices have the same color (or colors based on initial features)
2. At each iteration, a vertex's new color is determined by its current color and the multiset of its neighbors' colors
3. The algorithm terminates when the coloring stabilizes

Two graphs are distinguished by 1-WL if they have different color histograms after stabilization. It can be shown that standard message passing GNNs are **at most as powerful as 1-WL**: if two graphs cannot be distinguished by 1-WL, no message passing GNN can distinguish them either.

This means there exist non-isomorphic graphs that no standard GNN can tell apart. For example, certain pairs of regular graphs with the same degree sequence but different structures are indistinguishable to GNNs. More expressive architectures (like higher-order GNNs) can overcome some of these limitations, but at increased computational cost.

## The Oversmoothing Problem

A significant challenge with deep GCNs is **oversmoothing**: as we stack more layers, the vertex representations become increasingly similar, eventually converging to nearly identical vectors. This severely limits the discriminative power of the network.

### Mathematical Analysis

To understand oversmoothing mathematically, consider the GCN update without the nonlinearity and weight matrix for simplicity:

$$
H^{(\ell+1)} = S H^{(\ell)}
$$

where $S = \tilde{D}^{-\frac{1}{2}} \tilde{A} \tilde{D}^{-\frac{1}{2}}$ is the normalized adjacency matrix with self-loops. After $\ell$ layers:

$$
H^{(\ell)} = S^\ell H^{(0)}
$$

The matrix $S$ is symmetric (for undirected graphs), so it has an eigendecomposition $S = U \Lambda U^\top$ where $U$ is orthogonal and $\Lambda = \text{diag}(\lambda_1, \lambda_2, \ldots, \lambda_n)$ contains the eigenvalues.

A key result from spectral graph theory is that the eigenvalues of $S$ satisfy:

$$
1 = \lambda_1 > \lambda_2 \geq \cdots \geq \lambda_n > -1
$$

The largest eigenvalue is exactly 1, with corresponding eigenvector $u_1 = \tilde{D}^{1/2} \mathbf{1} / \|\tilde{D}^{1/2} \mathbf{1}\|$, where $\mathbf{1}$ is the all-ones vector. All other eigenvalues have magnitude strictly less than 1 (assuming the graph is connected and non-bipartite).

As $\ell \to \infty$, we have $\Lambda^\ell \to \text{diag}(1, 0, \ldots, 0)$, so:

$$
S^\ell \to u_1 u_1^\top
$$

This means $H^{(\ell)} \to u_1 u_1^\top H^{(0)}$. Every vertex's representation becomes a scalar multiple of $u_1$, collapsing all information into a one-dimensional subspace. The convergence rate is exponential: $\|S^\ell - u_1 u_1^\top\|_2 = \max_{i \geq 2} |\lambda_i|^\ell$, so the collapse happens rapidly.

### Spectral Interpretation

The eigenvectors of $S$ (equivalently, the graph Laplacian $L = I - S$) capture different frequency components of signals on the graph:

The eigenvector $u_1$ corresponding to eigenvalue 1 is the "DC component", a constant signal across the graph. Eigenvectors corresponding to eigenvalues close to 1 vary slowly across the graph (low frequency), while eigenvectors corresponding to eigenvalues close to $-1$ vary rapidly (high frequency).

Repeated multiplication by $S$ progressively attenuates high-frequency components and amplifies the DC component. This is analogous to repeatedly applying a low-pass filter to an image: eventually, all spatial variation is smoothed away, leaving only the average color. In the graph setting, all vertex features converge to a weighted average, losing the ability to distinguish between vertices.

{{< callout type="warning" title="Oversmoothing in Practice" >}}
While the theoretical analysis assumes no nonlinearities or weight matrices, oversmoothing also occurs in practice with full GCN layers. The nonlinearities and learned weights can slow down the convergence but cannot prevent it entirely. Empirically, GCN performance often degrades significantly beyond 2-4 layers, which is a severe limitation for tasks requiring information from distant vertices.
{{< /callout >}}

## Solutions to Oversmoothing

Several approaches have been developed to mitigate oversmoothing and enable deeper graph neural networks.

### Graph Isomorphism Network (GIN)

The **Graph Isomorphism Network** addresses oversmoothing by avoiding the averaging that causes feature collapse. Instead of normalizing by degrees, GIN uses a sum aggregation with a learnable parameter:

$$
h_v^{(\ell+1)} = \text{MLP}^{(\ell)}\left((1 + \epsilon^{(\ell)}) h_v^{(\ell)} + \sum_{u \in N(v)} h_u^{(\ell)}\right)
$$

Here $\epsilon^{(\ell)}$ is a learnable scalar (or can be fixed to 0), and $\text{MLP}^{(\ell)}$ is a multi-layer perceptron. In matrix form:

$$
H^{(\ell+1)} = \text{MLP}^{(\ell)}\left(\hat{A}_\epsilon H^{(\ell)}\right)
$$

where $\hat{A}_\epsilon = (1 + \epsilon) I + A$.

The key difference from GCN is that some eigenvalues of $\hat{A}_\epsilon$ are greater than 1 (since we add rather than normalize). This prevents the exponential decay of non-principal components. The MLP also provides additional expressiveness compared to a single linear transformation.

GIN was shown to be maximally powerful among message passing GNNs: it can distinguish any pair of graphs that 1-WL can distinguish. This makes it a strong baseline for graph classification tasks.

### Laplacian Positional Encodings

Another approach augments the input features with **positional encodings** that are resistant to oversmoothing. The idea is to expand the feature vectors with additional dimensions that preserve information even after many layers of smoothing.

Recall that oversmoothing collapses features toward the principal eigenvector $u_1$. The features most resistant to this collapse are exactly the eigenvectors of $S$: the eigenvector $u_k$ corresponding to eigenvalue $\lambda_k$ satisfies $S u_k = \lambda_k u_k$, so $S^\ell u_k = \lambda_k^\ell u_k$. While this still decays for $|\lambda_k| < 1$, eigenvectors with eigenvalues close to 1 decay slowly.

**Laplacian Positional Encodings** append the first $k$ non-trivial eigenvectors of the graph Laplacian (or equivalently, $S$) to the initial vertex features. Specifically, if $u_2, u_3, \ldots, u_{k+1}$ are the eigenvectors corresponding to the $k$ largest eigenvalues less than 1, we set:

$$
\tilde{h}_v^{(0)} = [h_v^{(0)}, (u_2)_v, (u_3)_v, \ldots, (u_{k+1})_v]
$$

These eigenvectors encode structural information about each vertex's position in the graph. For example, in a ring graph, $u_2$ varies like a sine wave around the ring, providing a "coordinate" for each vertex. In graphs with community structure, the eigenvectors often separate different communities.

{{< callout type="info" title="Connection to Transformer Positional Encodings" >}}
Laplacian positional encodings for graphs are analogous to sinusoidal positional encodings in Transformers. For a sequence (which is a path graph), the Laplacian eigenvectors are exactly the discrete cosine transform basis vectors, which are sinusoidal functions of position. This connection suggests that Laplacian encodings are a natural generalization of sequential positional encodings to arbitrary graph structures.
{{< /callout >}}

### Other Approaches

Several other techniques help with oversmoothing:

**Skip connections** add the input of each layer to its output, similar to ResNets. This allows information to bypass the smoothing operation: $H^{(\ell+1)} = \phi(S H^{(\ell)} W^{(\ell)}) + H^{(\ell)}$.

**Jumping Knowledge Networks** aggregate representations from all layers rather than just the final layer, giving the model access to both local (early layer) and global (late layer) information.

**DropEdge** randomly removes edges during training, reducing the effective receptive field and slowing down oversmoothing.

**PairNorm** normalizes the feature vectors at each layer to maintain their pairwise distances, preventing collapse.

## Graph-Level Predictions

So far, we have focused on computing vertex-level representations. For tasks requiring a prediction for the entire graph (like molecular property prediction), we need to aggregate vertex representations into a single graph-level vector.

The simplest approach is **global pooling**: apply a permutation-invariant function like sum or mean to all vertex features:

$$
h_G = \sum_{v \in V} h_v^{(L)} \quad \text{or} \quad h_G = \frac{1}{|V|} \sum_{v \in V} h_v^{(L)}
$$

The three main aggregation choices have different characteristics:

1. **Sum pooling** captures the total "mass" of features across the graph. It works well when the absolute count of features matters, such as predicting molecular weight or total charge.

2. **Mean pooling** provides a normalized view that is independent of graph size. This is useful when vertices have highly variable degrees or when comparing graphs of different sizes.

3. **Max pooling** highlights the most salient features, regardless of how many vertices have them. This can be effective when the presence (rather than count) of certain patterns is predictive.

There is no universally best choice. The optimal aggregation depends on the specific task and data characteristics.

More sophisticated approaches include:

**Set2Set**: Uses an attention mechanism to iteratively read from the set of vertex features, producing an order-invariant representation.

**Hierarchical pooling**: Coarsens the graph by clustering vertices, applies GNN layers to the coarsened graph, and repeats. This captures multi-scale structure.

**Virtual node**: Adds a special vertex connected to all other vertices. This vertex's representation after message passing summarizes the entire graph and enables communication between distant parts of the graph without the computational overhead of full connectivity.

The choice of pooling method depends on the task. For small molecules, sum pooling often works well because the total "mass" of features matters. For larger graphs where the number of vertices varies widely, mean pooling or attention-based methods may be preferable.

## Practical Considerations

Several practical insights emerge from applying GNNs to real problems like molecular property prediction:

**Shallow networks often suffice.** Despite the theoretical appeal of deep networks with large receptive fields, shallow GNNs (2-4 layers) frequently outperform deeper ones in practice. This is partly due to oversmoothing, but also because many graph properties can be captured with local information. Deeper networks can "dilute" the node representations, making it harder to retain task-relevant information.

**Parameter efficiency matters.** Effective GNN models can achieve strong performance with surprisingly few parameters (sometimes only a few thousand). This is because the weight sharing across vertices provides strong inductive bias. Overly large models may overfit, especially on small graph datasets.

**Communication between graph attributes helps.** Allowing information flow not just between vertex features, but also between vertex, edge, and global features typically improves performance. For example, edge features can be updated based on the vertices they connect, and a global context vector can aggregate information from all vertices and broadcast it back.

**Feature engineering still helps.** While GNNs can learn representations from raw graph structure, incorporating domain-specific features often improves performance. For molecules, this might include chemical properties computed by domain software. For social networks, this might include node centrality measures.

## Summary

Graph Neural Networks extend deep learning to graph-structured data by iteratively aggregating information from local neighborhoods. The GCN applies normalized message passing followed by learned transformations, and can be written compactly in matrix form. The MPNN framework generalizes this to arbitrary message and update functions.

GNNs are provably expressive (universal approximators on attributed graphs) but face fundamental limits tied to the Weisfeiler-Lehman hierarchy. The oversmoothing problem limits the depth of GNNs, but solutions like GIN and positional encodings enable effective deeper architectures.

For graph-level predictions, vertex representations must be aggregated using permutation-invariant pooling operations. The choice of architecture components (message function, aggregation, pooling) depends on the specific task and graph characteristics.

## References

Key papers in the development of Graph Neural Networks:

1. Kipf, T. N., & Welling, M. (2017). [Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/abs/1609.02907). ICLR.

2. Gilmer, J., Schoenholz, S. S., Riley, P. F., Vinyals, O., & Dahl, G. E. (2017). [Neural Message Passing for Quantum Chemistry](https://arxiv.org/abs/1704.01212). ICML.

3. Xu, K., Hu, W., Leskovec, J., & Jegelka, S. (2019). How Powerful are Graph Neural Networks? ICLR.

4. For a comprehensive survey, see: [A review of graph neural networks: concepts, architectures, techniques, challenges, datasets, applications, and future directions](https://link.springer.com/article/10.1186/s40537-023-00876-4). Journal of Big Data, 2024.

5. For an excellent interactive introduction with visualizations, see: [A Gentle Introduction to Graph Neural Networks](https://distill.pub/2021/gnn-intro/). Distill, 2021.
