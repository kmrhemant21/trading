# Orthogonal Complements

Have a look at the following links:

- [Orthogonal complement](https://en.wikipedia.org/wiki/Orthogonal_complement)
- [Orthogonal decomposition](https://en.wikipedia.org/wiki/Orthogonal_decomposition)

## Key Points

If we look at an $n$-dimensional vector space $V$ and a $k$-dimensional subspace $W \subset V$, then the orthogonal complement $W^{\perp}$ is an $(n-k)$-dimensional subspace of $V$ and contains all vectors in $V$ that are orthogonal to every vector in $W$.

Every vector $x \in V$ can be (uniquely) decomposed into:

$$x = \sum_{i=1}^{k} \lambda_i b_i + \sum_{j=1}^{n-k} \psi_j b_j^{\perp}$$

where $\lambda_i, \psi_j \in \mathbb{R}$, $\{b_1, \ldots, b_k\}$ is a basis of $W$, and $\{b_1^{\perp}, \ldots, b_{n-k}^{\perp}\}$ is a basis of $W^{\perp}$.
