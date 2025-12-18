import numpy as np
import networkx as nx
from scipy.sparse.csgraph import connected_components

from .build import build_adj_matrix
from .transform import remap_labels, remap_edges, remove_self_loops


def largest_weakly_component_connection(G: nx.DiGraph) -> nx.DiGraph:
    """
    Extract the largest weakly connected component from a directed graph.

    For directed graphs, *weakly* connected components are defined by treating
    all edges as undirected (direction is ignored). This function finds the
    largest such component and returns it as a new DiGraph.

    Parameters
    ----------
    G:
        Input directed graph (nx.DiGraph).

    Returns
    -------
    H:
        A directed graph containing only the nodes and edges from the largest
        weakly connected component of `G`. If `G` is empty, a copy of `G`
        is returned. If `G` has only one weakly connected component, a full
        copy of `G` is returned.
    """
    node_count = G.number_of_nodes()
    if node_count == 0:
        # Empty graph: just return a shallow copy as a DiGraph.
        return nx.DiGraph(G)

    # Weakly connected components treat the graph as undirected.
    wcc = nx.weakly_connected_components(G)
    largest_component_nodes = max(wcc, key=len)

    # If the largest component already contains all nodes, just copy G.
    if len(largest_component_nodes) == node_count:
        return nx.DiGraph(G)

    # Otherwise, restrict to the largest component.
    return nx.DiGraph(G.subgraph(largest_component_nodes))


def relabel_to_int(G: nx.DiGraph) -> nx.DiGraph:
    """
    Map node labels to compact 0..N-1 integers.

    Many numeric and array-based algorithms assume nodes are indexed from
    0 to N-1 without gaps. This helper normalizes arbitrary node labels
    (strings, tuples, large integers) into a dense integer range.

    Parameters
    ----------
    G:
        Input directed graph with arbitrary hashable node labels.

    Returns
    -------
    H:
        A new DiGraph whose nodes are labeled with consecutive integers
        {0, 1, ..., N-1}. The ordering follows the iteration order of
        `G.nodes()`.
    """
    node_mapping = {
        original_label: new_index
        for new_index, original_label in enumerate(G.nodes())
    }
    return nx.relabel_nodes(G, node_mapping, copy=True)


def edge_index_to_graph(edge_index_torch) -> nx.DiGraph:
    """
    Convert a PyTorch Geometric edge_index tensor (2 x E) to a NetworkX DiGraph.

    Handles both CPU and CUDA tensors, returning a directed NetworkX graph
    whose edges correspond to the columns of `edge_index`.

    Parameters
    ----------
    edge_index_torch:
        A PyTorch tensor of shape (2, E), typically named `edge_index`
        in PyG, where:
            - row 0 contains source node indices
            - row 1 contains target node indices

    Returns
    -------
    G:
        Directed graph (nx.DiGraph) with edges (src, dst) taken from
        `edge_index_torch`.
    """
    # If tensor is on GPU, move it to CPU first.
    if hasattr(edge_index_torch, "is_cuda") and edge_index_torch.is_cuda:
        edge_index_torch = edge_index_torch.cpu()

    edge_index_numpy = edge_index_torch.numpy()
    source_nodes = edge_index_numpy[0].tolist()
    target_nodes = edge_index_numpy[1].tolist()

    graph = nx.DiGraph()
    graph.add_edges_from(zip(source_nodes, target_nodes))
    return graph


def largest_component_connection(edges: np.ndarray, num_nodes: int) -> np.ndarray:
    """
    Keep only edges belonging to the largest (undirected) connected component.

    This is a low-level, array-based version of “take the largest connected
    component” that operates directly on an edge list with N nodes:

    1. Build a sparse adjacency matrix from `edges`.
    2. Symmetrize it to treat the graph as undirected.
    3. Use `scipy.sparse.csgraph.connected_components` to label components.
    4. Keep only edges whose endpoints both lie in the largest component.

    Parameters
    ----------
    edges:
        NumPy array of shape (M, 2), with each row (u, v) representing an edge
        between node indices in [0, num_nodes).
    num_nodes:
        Number of nodes in the graph (the adjacency will be of shape
        `(num_nodes, num_nodes)`).

    Returns
    -------
    filtered_edges:
        Edge array containing only those edges whose endpoints are in the
        largest connected component. If the graph has 0 or 1 components,
        `edges` is returned unchanged.
    """
    if edges.size == 0:
        return edges

    # Build sparse adjacency and symmetrize to treat graph as undirected
    adjacency_matrix = build_adj_matrix(edges, num_nodes)
    undirected_adjacency = (adjacency_matrix + adjacency_matrix.T).astype(bool)
    undirected_adjacency = undirected_adjacency.astype(np.uint8).tocsr()

    # Compute connected components on the undirected adjacency
    component_count, node_labels = connected_components(
        undirected_adjacency,
        directed=False,
        return_labels=True,
    )

    # If there's only one component, nothing to filter out.
    if component_count <= 1:
        return edges

    # Count how many nodes in each component and find the largest one.
    component_sizes = np.bincount(node_labels)
    largest_component_label = np.argmax(component_sizes)
    nodes_in_largest_component = node_labels == largest_component_label

    # Keep edges whose both endpoints lie in the largest component.
    source_nodes, target_nodes = edges[:, 0], edges[:, 1]
    valid_edges_mask = (
        nodes_in_largest_component[source_nodes]
        & nodes_in_largest_component[target_nodes]
    )
    return edges[valid_edges_mask]


def should_relabel_nodes(edges: np.ndarray, num_nodes: int) -> bool:
    """
    Decide whether node IDs need to be remapped to a dense 0..N-1 range.

    This is a cheap diagnostic that avoids unnecessary relabeling work:

    - If edge endpoints already cover the full range [0, num_nodes - 1]
    with no gaps (min = 0 and max = num_nodes - 1), we can skip remapping.
    - Otherwise, we consider labels “sparse” and return True.

    Parameters
    ----------
    edges:
        Edge array of shape (M, 2) with integer node IDs.
    num_nodes:
        Number of distinct nodes expected in the graph.

    Returns
    -------
    needs_relabel:
        True if labels are not dense 0..N-1, False otherwise.
    """
    if edges.size == 0:
        return False

    max_node_index = edges.max()
    min_node_index = edges.min()

    # Dense if:
    #   - min index is 0
    #   - max index is num_nodes - 1
    # Any deviation ⇒ either gaps or shifted indices → relabel.
    return (max_node_index + 1 != num_nodes) or (min_node_index != 0)


def clean_edges(
    edges: np.ndarray,
    keep_lcc: bool = True,
    relabel_to_int: bool = True,
    remove_loops: bool = True,
) -> np.ndarray:
    """
    High-performance edge-cleaning pipeline for graph preprocessing.

    This function applies a sequence of common graph-cleaning operations to an
    edge list in a carefully chosen order to minimize unnecessary work:

    Pipeline
    --------
    1. Normalize input to a 2D integer array of shape (M, 2).
    2. Remove self-loops (edges (u, u)).
    3. Remap arbitrary node IDs to internal contiguous indices if needed (`remap_edges`).
    4. Keep only the largest connected component (`keep_lcc`).
    5. Relabel nodes to a dense 0..N-1 range (`relabel_to_int`).

    Parameters
    ----------
    edges:
        Input edges as array-like; will be converted to shape (M, 2).
    keep_lcc:
        If True, keep only edges belonging to the largest connected component
        (based on undirected connectivity).
    relabel_to_int:
        If True, remap node IDs to dense consecutive integers after any
        component filtering.
    remove_loops:
        If True, drop self-loop edges (u, u) as an initial cheap operation.

    Returns
    -------
    cleaned_edges:
        A 2D NumPy array of dtype int64 and shape (M', 2) with the cleaned
        edge list. May be empty if the graph is fully pruned.

    Notes
    -----
    - `remap_edges` is assumed to convert arbitrary labels to integers but
    preserves relative structure.
    - `remap_labels` is used at the end (if needed) to pack integer labels
    into a dense range without gaps.
    """
    # Early exit: no edges at all
    if edges.size == 0:
        return edges.reshape(0, 2).astype(np.int64)

    # Ensure we have a 2D array of [u, v] pairs.
    edges = np.atleast_2d(np.asarray(edges))

    # Remove self-loops (cheapest operation)
    if remove_loops:
        edges = remove_self_loops(edges)
        if edges.size == 0:
            return edges.reshape(0, 2).astype(np.int64)

    # Remap arbitrary node IDs into some integer index space
    # (e.g., resolve non-int labels or large scattered IDs).
    edges = remap_edges(edges)
    num_nodes = int(edges.max()) + 1 if edges.size else 0

    #  Optionally restrict to the largest connected component.
    if keep_lcc and num_nodes > 0:
        edges = largest_component_connection(edges, num_nodes)
        if edges.size == 0:
            return edges.reshape(0, 2).astype(np.int64)
        num_nodes = int(edges.max()) + 1

    # Optionally relabel to a dense 0..N-1 range if gaps/offsets exist.
    if relabel_to_int and num_nodes > 0:
        if should_relabel_nodes(edges, num_nodes):
            edges = remap_labels(edges)

    return edges.astype(np.int64, copy=False)
