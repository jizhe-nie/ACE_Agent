# Clustering Taxonomy Rules

## Routing summary

1. Prefer centroid methods for compact, roughly spherical, balanced clusters.
2. Prefer topology and density methods for non-convex shapes, bridges, or noise.
3. Prefer dimensionality experts when the feature space is high-dimensional or
   manifold structure is likely.
4. Prefer deep representation learning when nonlinear compression is likely to
   improve separability.
5. Prefer multi-view consensus when multiple plausible views disagree and a more
   stable consensus is needed.

## Practical demo rules

- `blobs`: centroid expert is primary, topology expert is baseline.
- `moons`: topology expert is primary, centroid expert is a comparison baseline.
- `smile`: topology expert and multi-view expert are primary, centroid expert is
  demoted to challenger.
- `s_curve`: dimension and deep experts are primary, topology expert is secondary.

