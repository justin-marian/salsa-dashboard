from typing import Callable
import networkx as nx

from .normalize import normalize_outgoing_weights
from .minbd import minimal_back_distance


def distance_based_evaluation(
    graph: nx.DiGraph,
    max_distance: int = 30,
    distance_weight_function: Callable[[int], float] = lambda distance: 1.0 / (1.0 + distance),
) -> nx.DiGraph:
    """
    Metric 3: Distance-based edge reweighting via backward path length.

    For each directed edge (u → v), we look at the *minimal backward distance*:
        d(u → v) = length of the shortest directed path from v back to u
        (capped at `max_distance`).

    Intuition
    ---------
    - If v can return to u in just a few steps, then (u → v) is part of a
    tightly knit, strongly connected region → it should be relatively strong.
    - If v cannot reach u at all (or only via a very long path), (u → v) is
    more "one-way" and we downweight it.

    We then assign a raw edge weight via:
        w_raw(u → v) = f(d(u → v)),

    where `f` is `distance_weight_function` (default: 1 / (1 + d), decreasing
    with d). Distances ≥ `max_distance` and unreachable pairs use
    f(max_distance) as a fallback.

    To avoid exploding weights on high-degree nodes, we divide by a node-based
    normalization constant (out-degree or degree), and finally we call
    `normalize_outgoing_weights` to make each node's outgoing weights
    stochastic (sum to ~1).

    Parameters
    ----------
    graph:
        Input graph. Typically directed. If undirected, degrees are taken
        from `graph.degree`, but the output is still a DiGraph.
    max_distance:
        Maximum backward distance considered; larger distances and unreachable
        edges are treated as `max_distance`.
    distance_weight_function:
        Function mapping an integer distance d ↦ weight w. Must handle
        distances from 0 up to `max_distance`.

    Returns
    -------
    weighted_graph:
        A new directed graph with identical nodes and edges, but each edge
        has a "weight" attribute reflecting distance-based importance, and
        outgoing weights from each node are normalized.
    """
    # Initialize new graph with original node structure + attributes
    weighted_graph = nx.DiGraph()
    weighted_graph.add_nodes_from(graph.nodes(data=True))

    # Per-node normalization factor: out-degree for DiGraph, degree otherwise
    if graph.is_directed():
        normalization_constants = {
            node: max(1, graph.out_degree(node))
            for node in graph.nodes()
        }
    else:
        normalization_constants = {
            node: max(1, graph.degree(node))
            for node in graph.nodes()
        }

    # Precompute minimal backward distances: for each (u → v), distance v → u
    backward_distances = minimal_back_distance(graph, max_distance, verbose=False)

    # Fallback weight for “far” or unreachable edges
    max_distance_weight = distance_weight_function(max_distance)

    # Assign distance-based weights to edges
    for source_node, target_node in graph.edges():
        edge_key = (source_node, target_node)
        backward_distance = backward_distances.get(edge_key)

        # Choose raw weight based on backward distance
        if backward_distance is None or backward_distance >= max_distance:
            # No short return path: use fallback
            edge_weight = max_distance_weight
        else:
            # Prefer short cycles: f(d) with d = minimal backward distance
            edge_weight = distance_weight_function(backward_distance)

        # Normalize by node's (out)degree to control total outgoing mass
        normalized_weight = edge_weight / normalization_constants[source_node]

        weighted_graph.add_edge(
            source_node,
            target_node,
            weight=float(normalized_weight),
        )

    # Final safety normalization: make outgoing weights stochastic per node
    normalize_outgoing_weights(weighted_graph)
    return weighted_graph
