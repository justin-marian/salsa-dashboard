from typing import Any
import numpy as np
import networkx as nx
from scipy.sparse import coo_matrix

from datasets.clean import clean_edges


def sparse_matrix(edges: np.ndarray, directed: bool = True) -> nx.Graph:
    """
    Build a NetworkX graph from an edge array via a sparse adjacency matrix.

    This helper takes a 2D edge array of integer node IDs and constructs a
    NetworkX graph by:

        1) Computing the number of nodes from the max index in `edges`.
        2) Building a SciPy COO sparse adjacency matrix where
        A[u, v] = 1  for each edge (u, v).
        3) Converting that sparse matrix into a NetworkX graph.

    It is designed for large graphs where going through a sparse matrix can
    be more memory and performance friendly than adding edges one by one.

    Parameters
    ----------
    edges:
        NumPy array of shape (M, 2), where each row is an edge (u, v)
        with node indices >= 0.
    directed:
        If True, build a directed graph (nx.DiGraph). If False, build an
        undirected graph (nx.Graph).

    Returns
    -------
    G:
        NetworkX graph (DiGraph or Graph depending on `directed`).

    Notes
    -----
    - If `edges` is empty, returns an empty graph of the chosen type.
    - No edge weights are assigned here; all edges are unweighted (weight=1
    implicitly). If you need weights, the adjacency matrix build step
    would need to be extended.
    """
    if edges.size == 0:
        # No edges ⇒ empty graph of requested type
        return nx.DiGraph() if directed else nx.Graph()

    # Extract source and target nodes
    source_nodes, target_nodes = edges[:, 0], edges[:, 1]

    # Number of nodes is max index + 1 (assuming dense-ish integer IDs)
    num_nodes = int(max(source_nodes.max(), target_nodes.max())) + 1

    # Create sparse adjacency matrix in COO format with ones on edge positions
    adjacency_matrix = coo_matrix(
        (np.ones(len(source_nodes), dtype=np.uint8), (source_nodes, target_nodes)),
        shape=(num_nodes, num_nodes),
    )

    # Build the requested NetworkX graph type from the sparse matrix
    graph_type = nx.DiGraph() if directed else nx.Graph()
    return nx.from_scipy_sparse_array(adjacency_matrix, create_using=graph_type)


def load_unidir_graph(
    edges: np.ndarray,
    keep_lcc: bool = True,
    relabel_to_int: bool = True,
    remove_self_loops: bool = True,
    directed: bool = True,
) -> nx.Graph:
    """
    End-to-end pipeline: from raw edge list to a cleaned NetworkX graph.

    This wraps the full preprocessing + construction flow for a graph:

        1) `clean_edges`:
        - remove self-loops (optional)
        - remap node IDs if needed (e.g., from arbitrary labels)
        - keep only the largest connected component (optional)
        - relabel nodes to dense integers 0..N-1 (optional)

        2) `sparse_matrix`:
        - build a sparse adjacency matrix
        - convert it into a NetworkX Graph / DiGraph

    Parameters
    ----------
    edges:
        Raw edge list, array-like of shape (M, 2). Node IDs may be arbitrary
        (depending on what `clean_edges` expects).
    keep_lcc:
        If True, keep only the edges belonging to the largest connected
        component (undirected connectivity).
    relabel_to_int:
        If True, ensure final node IDs are dense integers 0..N-1.
    remove_self_loops:
        If True, drop edges (u, u) early in the pipeline.
    directed:
        If True, build a directed graph (DiGraph) at the end; otherwise an
        undirected Graph.

    Returns
    -------
    G:
        Cleaned NetworkX graph (DiGraph or Graph) ready for downstream
        algorithms.
    """
    cleaned_edges = clean_edges(
        edges,
        keep_lcc=keep_lcc,
        relabel_to_int=relabel_to_int,
        remove_loops=remove_self_loops,
    )
    return sparse_matrix(cleaned_edges, directed=directed)


def payload_to_graph(payload: dict[str, Any]) -> nx.DiGraph:
    """
    Convert an OGB-style payload dictionary into a NetworkX DiGraph.

    The Open Graph Benchmark (OGB) often represents graphs as dictionaries
    with fields like:

        {
            "edges": <array-like or tensor of shape (E, 2) or (2, E)>,
            "num_nodes": <int>,  # optional
            ...
        }

    This helper normalizes such a payload into a directed NetworkX graph:

    - Handles missing or empty `edges`.
    - Accepts 1D and 2D edge arrays and reshapes where appropriate.
    - Uses provided `num_nodes` if available; otherwise infers from `edges`.
    - Ensures node indices are integers and contiguous in [0, num_nodes).

    Parameters
    ----------
    payload:
        Dictionary-like structure containing at least an "edges" entry
        (or not; empty edges are allowed) and optionally "num_nodes".

    Returns
    -------
    graph:
        Directed NetworkX DiGraph with nodes [0..num_nodes-1] and edges
        from the payload.

    Raises
    ------
    ValueError
        If `edges` has an unsupported shape (e.g. 1D with size != 0 or 2,
        or 2D with second dimension != 2).
    """
    graph = nx.DiGraph()
    if not payload:
        # Empty payload ⇒ empty graph
        return graph

    edges_raw = payload.get("edges", None)
    num_nodes_raw = payload.get("num_nodes", None)

    # Normalize `edges_raw` to a 2D array of shape (E, 2)
    if edges_raw is None:
        # No edges given ⇒ use an explicit empty (0, 2) array
        edges_array = np.empty((0, 2), dtype=int)
    else:
        edges_array = np.asarray(edges_raw)

    # Edges can be 1D (length 0 or 2) or 2D (E, 2)
    if edges_array.ndim not in (1, 2):
        raise ValueError("Edges must be a 1D or 2D array-like")

    if edges_array.ndim == 1:
        size = edges_array.size
        if size == 0:
            edges_array = edges_array.reshape(0, 2)
        elif size == 2:
            edges_array = edges_array.reshape(1, 2)
        else:
            raise ValueError("1D edges array must have exactly 0 or 2 elements")
    if edges_array.ndim == 2 and edges_array.shape[1] != 2:
        raise ValueError("2D edges array must have shape (m, 2)")

    # Determine number of nodes:
    #   Prefer explicit `num_nodes` from payload if provided.
    #   Otherwise, infer from max node index in `edges_array`.
    #   If there are no edges and no num_nodes, fall back to 0.
    if num_nodes_raw is not None:
        num_nodes = int(num_nodes_raw)
    else:
        if edges_array.size == 0:
            num_nodes = 0
        else:
            num_nodes = int(edges_array.max()) + 1

    # Add nodes [0..num_nodes-1]
    if num_nodes > 0:
        graph.add_nodes_from(range(num_nodes))

    # Add edges (cast to int to be safe)
    if edges_array.size > 0:
        edge_tuples = [
            (int(source), int(target)) for source, target in edges_array
        ]
        graph.add_edges_from(edge_tuples)

    return graph
