"""
tools/community_discovery.py
=============================
Native graph community discovery — the primary pipeline for graph-connected data.

All algorithms operate on sparse adjacency matrices (not coordinate data).
Cluster count is determined by modularity optimum, not pre-defined k.

Algorithms:
  - Louvain (greedy modularity optimisation, pure numpy/scipy)
  - MCL (Markov Cluster Algorithm — expand/inflate)
  - Label Propagation (sklearn, fallback to pure numpy)
  - Spectral Graph Partition (delegates to GraphBuilder)
  - Random Walk Community Detection (diffusion distance clustering)
  - Leiden fallback (wraps Louvain when leidenalg is absent)
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix, diags, issparse, spmatrix
from scipy.sparse import eye as speye
from scipy.sparse.linalg import eigsh


def _to_dense(adjacency: spmatrix | np.ndarray, *, max_n: int = 15000) -> np.ndarray:
    """Convert to dense if feasible; warn and subsample otherwise."""
    if not issparse(adjacency):
        return np.asarray(adjacency, dtype=float)
    n = adjacency.shape[0]
    if n <= max_n:
        return adjacency.toarray()
    raise ValueError(
        f"Adjacency too large ({n}x{n}) for dense conversion. "
        f"Use a smaller graph or sparse-native methods."
    )


def _modularity(adjacency: spmatrix, labels: np.ndarray, m2: float | None = None) -> float:
    """Compute graph modularity Q = 1/2m * sum_ij[A_ij - k_i*k_j/2m] * delta(c_i, c_j)."""
    labels = np.asarray(labels, dtype=int).ravel()
    if m2 is None:
        m2 = adjacency.sum()
    if m2 == 0:
        return 0.0
    degrees = np.asarray(adjacency.sum(axis=1)).ravel()
    unique_labels = np.unique(labels)
    if issparse(adjacency):
        adjacency = adjacency.tocsr()
    q = 0.0
    for c in unique_labels:
        indices = np.where(labels == c)[0]
        if len(indices) == 0:
            continue
        # Edges within community: convert to int indices for sparse
        sub = adjacency[indices, :][:, indices]
        w_in = sub.sum()
        # Expected edges within community
        d_c = degrees[indices].sum()
        expected = (d_c * d_c) / m2
        q += (w_in - expected) / m2
    return q


def _connected_components(adjacency: spmatrix) -> np.ndarray:
    """BFS-based connected component labelling on sparse adjacency.

    Returns integer label array; number of components = max+1.
    """
    n = adjacency.shape[0]
    labels = np.full(n, -1, dtype=int)
    current_label = 0

    adjacency = adjacency.tocsr() if issparse(adjacency) else adjacency

    for start in range(n):
        if labels[start] != -1:
            continue
        # BFS
        stack = [start]
        labels[start] = current_label
        while stack:
            v = stack.pop()
            if issparse(adjacency):
                neighbors = adjacency[v].indices
            else:
                neighbors = np.where(adjacency[v] > 0)[0]
            for nb in neighbors:
                if labels[nb] == -1:
                    labels[nb] = current_label
                    stack.append(nb)
        current_label += 1

    return labels


# =========================================================================
# Markov Cluster Algorithm (MCL)
# =========================================================================

def mcl_clustering(
    adjacency: spmatrix,
    *,
    expand_factor: int = 2,
    inflate_factor: float = 2.0,
    max_iter: int = 100,
    prune_threshold: float = 1e-6,
    self_loop_weight: float = 1.0,
    random_state: int = 42,
) -> np.ndarray:
    """Markov Cluster Algorithm — graph community detection via flow simulation.

    The MCL algorithm simulates random walks on the graph.  Expansion
    (matrix power) spreads flow along paths; inflation (element-wise power
    + column renorm) sharpens the distribution, driving flow into natural
    clusters.  Connected components of the converged transition matrix
    are the communities.

    Parameters
    ----------
    adjacency : sparse matrix, shape (n, n)
        Undirected graph adjacency (edges may be weighted).
    expand_factor : int
        Matrix power for the expansion step (default 2 = one walk step).
    inflate_factor : float
        Element-wise power for the inflation step (higher = sharper clusters).
    max_iter : int
        Maximum number of expand/inflate cycles.
    prune_threshold : float
        Entries below this value are zeroed after each cycle.
    self_loop_weight : float
        Weight added to the diagonal before normalising.

    Returns
    -------
    labels : ndarray of shape (n,), dtype int
        Community labels (0-indexed).
    """
    n = adjacency.shape[0]
    if n < 2:
        return np.zeros(n, dtype=int)

    # Build stochastic transition matrix with self-loops
    if issparse(adjacency):
        A = adjacency.tocsr().astype(float).copy()
    else:
        A = csr_matrix(adjacency, dtype=float)
    A = A + speye(n, dtype=float) * self_loop_weight

    # Column-normalise to get transition matrix
    col_sums = np.asarray(A.sum(axis=0)).ravel()
    col_sums[col_sums == 0] = 1.0
    D_inv = diags(1.0 / col_sums)
    M = A @ D_inv  # M_ij = P(j→i)
    M = M.tocsc()

    _rng = np.random.RandomState(random_state)

    for _ in range(max_iter):
        prev_nnz = M.nnz

        # Expand: matrix power
        M = (M ** expand_factor).tocsc()

        # Inflate: element-wise power
        M.data = M.data ** inflate_factor

        # Column renormalise
        col_sums = np.asarray(M.sum(axis=0)).ravel()
        col_sums[col_sums == 0] = 1.0
        D_inv = diags(1.0 / col_sums)
        M = M @ D_inv

        # Prune
        M.data[np.abs(M.data) < prune_threshold] = 0
        M.eliminate_zeros()

        # Convergence check
        if M.nnz == prev_nnz:
            break

    # Extract clusters as connected components of M
    # (M is the converged "cluster assignment" matrix)
    return _connected_components(M)


# =========================================================================
# Louvain (greedy modularity optimisation)
# =========================================================================

def louvain_clustering(
    adjacency: spmatrix,
    *,
    resolution: float = 1.0,
    max_passes: int = 20,
    random_state: int = 42,
) -> np.ndarray:
    """Louvain community detection — greedy modularity maximisation.

    Two-phase algorithm:
      1. **Local moves**: each node is moved to the neighbouring community
         that yields the maximum positive modularity gain.
      2. **Aggregation**: nodes in each community are collapsed into a
         super-node; the adjacency of the coarse graph is the sum of
         inter-community edges.

    Repeats phases 1–2 until modularity can no longer be improved.

    This is a pure numpy/scipy implementation that does NOT require the
    ``python-louvain`` or ``leidenalg`` packages.
    """
    n_orig = adjacency.shape[0]
    if n_orig < 2:
        return np.zeros(n_orig, dtype=int)

    if issparse(adjacency):
        A = adjacency.tocsr().astype(float)
    else:
        A = csr_matrix(adjacency, dtype=float)

    # Keep working on undirected graph
    A = A + A.T
    A.data *= 0.5
    m2 = A.sum() * 2
    if m2 == 0:
        return np.zeros(n_orig, dtype=int)

    rng = np.random.RandomState(random_state)

    # Track hierarchy: hierarchy[i] maps nodes_at_level_i → community_ids (0-indexed, contiguous)
    hierarchy_labels: list[np.ndarray] = []
    current_labels = np.arange(A.shape[0], dtype=int)

    while True:
        n_curr = A.shape[0]
        improved = False

        # Phase 1: local moves
        order = rng.permutation(n_curr)
        for v in order:
            v_comm = current_labels[v]
            row = A[v]
            nb_indices = row.indices

            k_v = row.sum()
            if k_v == 0:
                continue

            mask_cur = current_labels == v_comm
            d_cur_total = float(A[mask_cur, :].sum())
            d_cur_no_v = d_cur_total - k_v
            k_v_in_cur = float(row[:, mask_cur].sum())
            gain_remove = -2.0 * (k_v_in_cur - resolution * k_v * d_cur_no_v / m2) / m2

            best_gain = 0.0
            best_comm = v_comm
            seen = {v_comm}

            for nb in nb_indices:
                c = current_labels[nb]
                if c in seen:
                    continue
                seen.add(c)
                mask_c = current_labels == c
                d_c = float(A[mask_c, :].sum())
                k_v_in_c = float(row[:, mask_c].sum())
                gain = gain_remove + 2.0 * (k_v_in_c - resolution * k_v * d_c / m2) / m2
                if gain > best_gain:
                    best_gain = gain
                    best_comm = c

            if best_comm != v_comm:
                current_labels[v] = best_comm
                improved = True

        if not improved:
            break

        # Reindex to contiguous 0..k-1 and store in hierarchy
        uniq = sorted(np.unique(current_labels))
        old_to_new = {old: i for i, old in enumerate(uniq)}
        reindexed = np.array([old_to_new[l] for l in current_labels], dtype=int)
        hierarchy_labels.append(reindexed)

        # Phase 2: aggregation
        n_comm = len(uniq)
        if n_comm >= n_curr - 1 or n_comm < 2:
            break

        rows, cols = A.nonzero()
        data = A.data
        coarse_rows = reindexed[rows]
        coarse_cols = reindexed[cols]
        mask = coarse_rows != coarse_cols
        A = csr_matrix(
            (data[mask], (coarse_rows[mask], coarse_cols[mask])),
            shape=(n_comm, n_comm),
            dtype=float,
        )
        A = A + A.T
        A.data *= 0.5

        current_labels = np.arange(n_comm, dtype=int)

    # Unroll hierarchy from bottom level back to original nodes
    if not hierarchy_labels:
        return np.zeros(n_orig, dtype=int)

    final_labels = hierarchy_labels[0].copy()
    for level_labels in hierarchy_labels[1:]:
        final_labels = np.array([level_labels[l] for l in final_labels], dtype=int)

    _, final_labels = np.unique(final_labels, return_inverse=True)
    return final_labels


# =========================================================================
# Label Propagation (async, pure numpy/scipy)
# =========================================================================

def label_propagation_clustering(
    adjacency: spmatrix,
    *,
    max_iter: int = 100,
    random_state: int = 42,
) -> np.ndarray:
    """Label propagation community detection.

    Each node adopts the majority label of its neighbours, ties broken
    randomly.  Converges quickly on most graphs.

    Falls back to sklearn.semi_supervised.LabelPropagation if available
    for better numerical stability on dense graphs.
    """
    n = adjacency.shape[0]
    if n < 2:
        return np.zeros(n, dtype=int)

    try:
        from sklearn.semi_supervised import LabelPropagation as SKLabelPropagation

        # sklearn's LabelPropagation needs a feature matrix; use adjacency rows
        if issparse(adjacency):
            adj_dense = adjacency.toarray()
        else:
            adj_dense = np.asarray(adjacency, dtype=float)
        # Use 1-step random-walk embedding as features
        degrees = adj_dense.sum(axis=1, keepdims=True)
        degrees[degrees == 0] = 1.0
        features = adj_dense / degrees
        # Sklearn LP needs some labels pre-assigned; use -1 for unlabelled
        model = SKLabelPropagation(kernel='knn', n_neighbors=min(10, n - 1),
                                   max_iter=max_iter)
        # We misuse this slightly — all unlabelled, sklearn will initialise
        y_initial = np.full(n, -1)
        model.fit(features, y_initial)
        labels = model.transduction_
        # Map to contiguous
        _, labels = np.unique(labels, return_inverse=True)
        return labels
    except Exception:
        pass

    # Pure numpy fallback
    rng = np.random.RandomState(random_state)
    labels = np.arange(n, dtype=int)

    if issparse(adjacency):
        A = adjacency.tocsr()
    else:
        A = csr_matrix(adjacency)

    for _ in range(max_iter):
        changed = 0
        order = rng.permutation(n)
        for v in order:
            row = A[v]
            nb_indices = row.indices
            if len(nb_indices) == 0:
                continue
            # Count neighbour labels
            nb_labels = labels[nb_indices]
            label_counts = np.bincount(nb_labels, minlength=n)
            max_count = label_counts.max()
            best_labels = np.where(label_counts == max_count)[0]
            new_label = best_labels[rng.randint(len(best_labels))]
            if new_label != labels[v]:
                labels[v] = new_label
                changed += 1
        if changed == 0:
            break

    _, labels = np.unique(labels, return_inverse=True)
    return labels


# =========================================================================
# Random Walk Community Detection (diffusion-distance clustering)
# =========================================================================

def random_walk_community_detection(
    adjacency: spmatrix,
    *,
    n_clusters: int | None = None,
    diffusion_steps: int = 8,
    random_state: int = 42,
) -> np.ndarray:
    """Community detection via diffusion map embedding + spectral clustering.

    1. Build random-walk transition matrix P = D^{-1}A.
    2. Compute P^t (t-step diffusion).
    3. Compute diffusion distances from P^t.
    4. If n_clusters is None, determine k via modularity sweep of a
       spectral clustering on the diffusion embedding.

    Returns labels where cluster count is driven by modularity.
    """
    from ACE_Agent.tools.graph_builder import GraphBuilder

    n = adjacency.shape[0]
    if n < 2:
        return np.zeros(n, dtype=int)

    if issparse(adjacency):
        A = adjacency.tocsr().astype(float)
    else:
        A = csr_matrix(adjacency, dtype=float)

    # Build transition matrix
    degrees = np.asarray(A.sum(axis=1)).ravel()
    degrees[degrees == 0] = 1.0
    D_inv = diags(1.0 / degrees)
    P = D_inv @ A  # P_ij = prob(i→j) — row-stochastic

    # Diffusion for t steps
    P_t = P
    steps_remaining = diffusion_steps - 1
    while steps_remaining > 0:
        if steps_remaining % 2 == 1:
            P_t = P_t @ P
            steps_remaining -= 1
        else:
            P = P @ P
            steps_remaining //= 2

    # Diffusion map: use right eigenvectors of P_t
    # For stability, use symmetric normalised Laplacian embedding instead
    D_sqrt_inv = diags(1.0 / np.sqrt(degrees))
    L_sym = speye(n) - D_sqrt_inv @ A @ D_sqrt_inv

    k_eig = min(n - 1, 50)
    try:
        vals, vecs = eigsh(L_sym, k=k_eig, which='SM', tol=1e-4)
    except Exception:
        try:
            vals, vecs = eigsh(L_sym, k=min(k_eig, n - 2), which='SM')
        except Exception:
            # Fallback: use SVD of P_t
            from scipy.sparse.linalg import svds
            _, _, vecs = svds(P_t, k=min(20, n - 1))
            vecs = vecs.T

    embedding = vecs[:, :min(16, vecs.shape[1])]

    # Determine k via modularity sweep when not given
    if n_clusters is None:
        n_clusters = GraphBuilder.adaptive_modularity_k(adjacency, max_k=12, min_k=2)

    from sklearn.cluster import KMeans
    labels = KMeans(
        n_clusters=n_clusters, n_init=10, random_state=random_state,
    ).fit_predict(embedding)

    return labels


# =========================================================================
# Modularity-optimal k sweep
# =========================================================================

def modularity_optimal_k(
    adjacency: spmatrix,
    clustering_fn,
    *,
    min_k: int = 2,
    max_k: int = 15,
    random_state: int = 42,
) -> tuple[np.ndarray, int, float]:
    """Run a clustering function for k in [min_k, max_k] and return the
    labels, k, and modularity for the best k.

    ``clustering_fn`` must accept ``n_clusters`` and ``random_state`` kwargs.

    Returns (best_labels, best_k, best_modularity).
    """
    best_k = min_k
    best_q = -999.0
    best_labels = None

    for k in range(min_k, max_k + 1):
        try:
            labels = clustering_fn(n_clusters=k, random_state=random_state)
            labels = np.asarray(labels, dtype=int).ravel()
            q = _modularity(adjacency, labels)
            if q > best_q:
                best_q = q
                best_k = k
                best_labels = labels
        except Exception:
            continue

    if best_labels is None:
        # Fallback: return single cluster
        return np.zeros(adjacency.shape[0], dtype=int), 1, 0.0

    return best_labels, best_k, best_q


# =========================================================================
# Leiden / Infomap stubs (fall back to Louvain)
# =========================================================================

def leiden_clustering(
    adjacency: spmatrix, *, resolution: float = 1.0, n_iterations: int = 2,
    random_state: int = 42,
) -> np.ndarray:
    """Leiden community detection.

    Uses the ``leidenalg`` package if installed; otherwise falls back to
    Louvain (which is the same algorithmic family, just without the
    guaranteed well-connectedness refinement step).
    """
    try:
        import igraph as ig
        import leidenalg

        if issparse(adjacency):
            A_coo = adjacency.tocoo()
            edges = list(zip(A_coo.row.tolist(), A_coo.col.tolist(), strict=False))
            weights = A_coo.data.tolist()
        else:
            adj_dense = np.asarray(adjacency)
            rows, cols = np.where(adj_dense > 0)
            edges = list(zip(rows.tolist(), cols.tolist(), strict=False))
            weights = adj_dense[rows, cols].tolist()

        g = ig.Graph(n=adjacency.shape[0], edges=edges, directed=False)
        if weights:
            g.es['weight'] = weights

        partition = leidenalg.find_partition(
            g, leidenalg.ModularityVertexPartition,
            n_iterations=n_iterations, seed=random_state,
        )
        labels = np.array(partition.membership, dtype=int)
        # Re-label contiguously
        _, labels = np.unique(labels, return_inverse=True)
        return labels
    except ImportError:
        return louvain_clustering(adjacency, resolution=resolution,
                                  random_state=random_state)


def infomap_clustering(
    adjacency: spmatrix, *, random_state: int = 42, **kwargs,
) -> np.ndarray:
    """Infomap community detection.

    Uses the ``infomap`` package if installed; otherwise falls back to
    Louvain with a higher resolution parameter (Infomap tends to find
    finer-grained communities than default Louvain).
    """
    try:
        import infomap as imp

        im = imp.Infomap(f"--seed {random_state}")
        if issparse(adjacency):
            A_coo = adjacency.tocoo()
            for i, j, w in zip(A_coo.row, A_coo.col, A_coo.data, strict=False):
                im.add_link(int(i), int(j), float(w))
        else:
            adj_dense = np.asarray(adjacency)
            rows, cols = np.where(adj_dense > 0)
            for i, j in zip(rows, cols, strict=False):
                im.add_link(int(i), int(j), float(adj_dense[i, j]))

        im.run()
        labels = np.zeros(adjacency.shape[0], dtype=int)
        for node in im.iterTree():
            if node.isLeaf():
                labels[node.node_id] = node.moduleIndex()
        _, labels = np.unique(labels, return_inverse=True)
        return labels
    except ImportError:
        return louvain_clustering(adjacency, resolution=1.5,
                                  random_state=random_state)


# =========================================================================
# Community Discovery Pipeline — primary entry point
# =========================================================================

def discover_communities(
    adjacency: spmatrix,
    *,
    methods: tuple[str, ...] = ("louvain", "mcl", "label_propagation", "spectral"),
    random_state: int = 42,
) -> dict[str, np.ndarray]:
    """Run multiple community discovery algorithms on the same graph.

    Returns a dict mapping method name → labels array.  All algorithms
    operate purely on the graph structure — no coordinate data is used.

    This is the **primary clustering pipeline** for graph-connected data,
    replacing Euclidean-distance-based clustering.
    """
    try:
        from ACE_Agent.tools.graph_builder import GraphBuilder
    except ImportError:
        from tools.graph_builder import GraphBuilder

    results: dict[str, np.ndarray] = {}
    _n = adjacency.shape[0]

    for method in methods:
        try:
            if method == "louvain":
                results[method] = louvain_clustering(adjacency, random_state=random_state)
            elif method == "leiden":
                results[method] = leiden_clustering(adjacency, random_state=random_state)
            elif method == "mcl":
                results[method] = mcl_clustering(adjacency, random_state=random_state)
            elif method == "label_propagation":
                results[method] = label_propagation_clustering(adjacency, random_state=random_state)
            elif method == "random_walk":
                results[method] = random_walk_community_detection(
                    adjacency, random_state=random_state,
                )
            elif method == "infomap":
                results[method] = infomap_clustering(adjacency, random_state=random_state)
            elif method == "spectral":
                # spectral_graph_cut_clustering needs n_clusters — determine via modularity
                k = GraphBuilder.adaptive_modularity_k(adjacency)
                results[method] = GraphBuilder.spectral_graph_cut_clustering(
                    adjacency, n_clusters=k, random_state=random_state,
                )
        except Exception:
            # Algorithm failed — continue with others
            continue

    return results


def select_best_community(
    adjacency: spmatrix,
    community_results: dict[str, np.ndarray],
) -> tuple[str, np.ndarray, float]:
    """Select the best community partition by modularity score.

    Returns (method_name, labels, modularity).
    """
    best_method = ""
    best_labels: np.ndarray | None = None
    best_q = -999.0

    for method, labels in community_results.items():
        labels = np.asarray(labels, dtype=int).ravel()
        if len(np.unique(labels)) < 2:
            continue  # single-cluster result is not useful
        q = _modularity(adjacency, labels)
        if q > best_q:
            best_q = q
            best_method = method
            best_labels = labels

    if best_labels is None:
        # Fallback: single cluster
        return "fallback", np.zeros(adjacency.shape[0], dtype=int), 0.0

    return best_method, best_labels, best_q
