from typing import Literal
import networkx as nx
from itertools import chain
import numpy as np
from scipy.sparse import coo_matrix


def build_adj_matrix(
    edges: np.ndarray,
    num_nodes: int,
    *,
    weights: np.ndarray | None = None,
    dtype: np.dtype = np.float32,
    aggregate: bool = True,
) -> coo_matrix:
    """
    Build a sparse adjacency matrix A from an edge list.

    The input is a 2-column NumPy array of edges `(u, v)` and an optional
    weight array. The function returns a `scipy.sparse.coo_matrix` of shape
    `(num_nodes, num_nodes)`:

        A[i, j] = weight of edge i → j   (or 1.0 if unweighted)

    It is designed to be:
    - **Fast**: uses vectorized operations and integer linearization.
    - **Robust**: can optionally check bounds and aggregate parallel edges.
    - **Flexible**: supports unweighted and weighted edges.

    Parameters
    ----------
    edges:
        A NumPy array of shape `(m, 2)` where each row is an edge `(u, v)`.
        Node indices must be in `[0, num_nodes - 1]`.
    num_nodes:
        Total number of nodes `N` in the graph, defining the shape `(N, N)`.
    weights:
        Optional 1D array of length `m` with edge weights. If `None`, all edges
        get weight `1.0`.
    dtype:
        Data type of the resulting matrix and weights (default: `np.float32`).
    aggregate:
        If True, aggregate parallel edges between the same `(u, v)` pair by
        summing their weights. If False, parallel edges are kept as-is in the
        COO representation (multiple entries with the same `(row, col)`).

    Returns
    -------
    A : coo_matrix
        Sparse adjacency matrix in COO format.

    Notes
    -----
    - Aggregation is done by “linearizing” edge indices into a 1D key
      `key = u * num_nodes + v`, then using `np.unique` + `np.bincount`.
    - This function does *not* symmetrize the matrix; it assumes a directed
    interpretation unless you have already duplicated edges appropriately.
    """
    # Handle trivial case: no edges => zero matrix
    if edges.size == 0:
        return coo_matrix((num_nodes, num_nodes), dtype=dtype)

    # Basic shape validation: must be (m, 2)
    if edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError("Edges must be a (m, 2) array of (u, v) pairs")

    # Extract row/col indices as int64 for robust indexing
    rows = edges[:, 0].astype(np.int64, copy=False)
    cols = edges[:, 1].astype(np.int64, copy=False)

    # Prepare weights
    if weights is None:
        # Unweighted graph: each edge has weight 1
        weights_array = np.ones(len(rows), dtype=dtype)
    else:
        if len(weights) != len(rows):
            raise ValueError("Weights length must match number of edges")
        weights_array = weights.astype(dtype, copy=False)

    # No aggregation: construct COO directly from raw edges
    if not aggregate:
        return coo_matrix(
            (weights_array, (rows, cols)),
            shape=(num_nodes, num_nodes),
            dtype=dtype,
        )

    # Aggregate parallel edges (u, v) by summing their weights.
    # We linearize (u, v) to a single integer key to use bincount.
    edge_keys = rows * np.int64(num_nodes) + cols

    # unique_keys: distinct (u, v) pairs (in linearized form)
    # inverse_indices: for each edge, index of its unique key
    unique_keys, inverse_indices = np.unique(edge_keys, return_inverse=True)

    # Sum weights for each unique key
    aggregated_weights = np.bincount(inverse_indices, weights=weights_array)
    aggregated_weights = aggregated_weights.astype(dtype, copy=False)

    # Recover row/col indices from linearized keys
    unique_rows = (unique_keys // np.int64(num_nodes)).astype(np.int64, copy=False)
    unique_cols = (unique_keys % np.int64(num_nodes)).astype(np.int64, copy=False)

    # Build final aggregated adjacency matrix in COO format
    return coo_matrix(
        (aggregated_weights, (unique_rows, unique_cols)),
        shape=(num_nodes, num_nodes),
        dtype=dtype,
    )


def undirected_graph(
    G: nx.Graph,
    *,
    weight: str = "weight",
    combine: Literal["sum", "max", "mean"] = "sum",
) -> nx.Graph:
    """
    Convert any NetworkX graph into a simple undirected weighted graph.

    This function normalizes a possibly messy input graph (directed, multigraph,
    missing weights) into a clean `nx.Graph` with:

    - No parallel edges (simple graph).
    - Undirected edges (u, v) == (v, u).
    - A single numeric weight per edge, obtained by combining all parallel
    edges according to `combine` ("sum", "max", or "mean").

    It uses vectorized NumPy operations for efficient handling of large graphs.

    Parameters
    ----------
    G:
        Input NetworkX graph. Can be:
        - undirected or directed
        - simple or multigraph
        - with or without edge weights
    weight:
        Name of the edge attribute used as weight when present. If an edge
        lacks this attribute, a default weight of 1.0 is used.
    combine:
        How to combine multiple edges between the same unordered pair {u, v}:
        - "sum"  : sum of all weights
        - "max"  : maximum of all weights
        - "mean" : average of all weights

    Returns
    -------
    result_graph : nx.Graph
        A simple, undirected NetworkX graph with a single `"weight"` attribute
        per edge.

    Notes
    -----
    - For directed graphs, edges are mirrored: both u→v and v→u are collected
    before merging, so the resulting undirected graph reflects the combined
    “interaction strength” in both directions.
    - If you pass in an already simple, undirected graph, this function just
    returns a copy.
    """
    # Fast path: already a simple undirected graph, nothing to normalize.
    if isinstance(G, nx.Graph) and not G.is_multigraph() and not G.is_directed():
        return G.copy(as_view=False)

    # Create node index mapping for efficient NumPy-based aggregation.
    nodes = list(G.nodes())
    node_count = len(nodes)
    node_to_index = {node: idx for idx, node in enumerate(nodes)}

    # Extract edges as (u, v, w) triples.
    # For directed graphs, we include both (u, v) and (v, u) so that the final
    # undirected weight reflects interactions in both directions.
    if not G.is_directed():
        edge_data = (
            (u, v, float(data.get(weight, 1.0)))
            for u, v, data in G.edges(data=True)
        )
    else:
        edge_data = chain(
            (
                (u, v, float(data.get(weight, 1.0)))
                for u, v, data in G.edges(data=True)
            ),
            (
                (v, u, float(data.get(weight, 1.0)))
                for u, v, data in G.edges(data=True)
            ),
        )

    # Pack edges into arrays of integer indices + weights
    source_indices, target_indices, weight_values = [], [], []
    for u, v, w in edge_data:
        source_indices.append(node_to_index[u])
        target_indices.append(node_to_index[v])
        weight_values.append(w)

    # If there are no edges, just return a node-only undirected graph.
    if not source_indices:
        undirected_graph = nx.Graph()
        undirected_graph.add_nodes_from(nodes)
        return undirected_graph

    source_indices = np.asarray(source_indices, dtype=np.int64)
    target_indices = np.asarray(target_indices, dtype=np.int64)
    weight_values = np.asarray(weight_values, dtype=np.float32)

    # Canonicalize edges for undirected representation:
    # We always store (min(u, v), max(u, v)) so {u, v} and {v, u} merge.
    low_nodes = np.minimum(source_indices, target_indices)
    high_nodes = np.maximum(source_indices, target_indices)

    # Linearize unordered pairs into a single integer key.
    edge_keys = low_nodes * np.int64(node_count) + high_nodes
    unique_keys, inverse_indices = np.unique(edge_keys, return_inverse=True)

    # Aggregate weights for each unique undirected edge based on `combine`.
    if combine == "sum":
        aggregated_weights = np.bincount(
            inverse_indices,
            weights=weight_values,
        ).astype(np.float32, copy=False)

    elif combine == "max":
        aggregated_weights = np.full(unique_keys.shape[0], -np.inf, dtype=np.float32)
        np.maximum.at(aggregated_weights, inverse_indices, weight_values)

    elif combine == "mean":
        weight_sums = np.bincount(
            inverse_indices,
            weights=weight_values,
        ).astype(np.float32, copy=False)
        edge_counts = np.bincount(inverse_indices).astype(np.float32, copy=False)
        edge_counts = np.maximum(edge_counts, 1.0)
        aggregated_weights = (weight_sums / edge_counts).astype(
            np.float32, copy=False
        )

    else:
        raise ValueError("combine must be one of {'sum', 'max', 'mean'}")

    # Decode linearized keys back into (u, v) index pairs.
    source_nodes = (unique_keys // np.int64(node_count)).astype(np.int64, copy=False)
    target_nodes = (unique_keys % np.int64(node_count)).astype(np.int64, copy=False)

    # Build the resulting simple undirected weighted graph.
    result_graph = nx.Graph()
    result_graph.add_nodes_from(nodes)

    edges_with_weights = [
        (nodes[int(u)], nodes[int(v)], float(w))
        for u, v, w in zip(source_nodes, target_nodes, aggregated_weights)
    ]
    result_graph.add_weighted_edges_from(edges_with_weights)

    return result_graph
