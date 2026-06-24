# Algorithm Limits

- K-means assumes compact clusters and struggles with non-convex shapes.
- Gaussian Mixture Models can model ellipses but still prefer separable densities.
- DBSCAN is good for arbitrarily shaped clusters but is sensitive to `eps`.
- Agglomerative clustering can capture topology but linkage choice matters.
- PCA is linear and may miss curved manifolds.
- t-SNE and similar embeddings are useful for visualization but may distort global distances.
- Autoencoders help nonlinear compression but add training variance and runtime cost.
- Multi-view consensus is robust for disagreement, but its benefit depends on the
  diversity and quality of the views.

