import networkx as nx
from typing import Any


def normalize_outgoing_weights(graph: nx.DiGraph) -> None:
    """
    Normalize outgoing edge weights so each node's out-neighborhood sums to 1.

    Purpose
    -------
    Convert arbitrary edge weights into *row-stochastic* form, i.e. for every node u:

        Σ_v w(u → v) = 1,   over all outgoing neighbors v

    This is the standard requirement for:
        - random-walk style processes,
        - Markov chains,
        - PageRank/SALSA/HITS-like algorithms when interpreted as transitions.

    Behavior
    --------
    - Works in-place: modifies the `"weight"` attribute of edges in `graph`.
    - If an edge has no `"weight"` attribute, it is treated as 0.0 during summation.
    - Nodes with no outgoing edges are left untouched (they just never appear as `source`).
    - If a node's total outgoing weight is 0, its outgoing edges (if any) are set to 0.0.

    Parameters
    ----------
    graph:
        Directed graph whose edge weights (attribute `"weight"`) will be
        normalized per source node.
    """
    outgoing_weight_sums: dict[Any, float] = {}

    # Accumulate total outgoing weight per source node
    for source, _, edge_data in graph.edges(data=True):
        current_weight = float(edge_data.get("weight", 0.0))
        outgoing_weight_sums[source] = outgoing_weight_sums.get(source, 0.0) + current_weight

    # Normalize each edge weight by its source's total
    for source, _, edge_data in graph.edges(data=True):
        node_weight_sum = outgoing_weight_sums.get(source, 1.0)
        current_weight = float(edge_data.get("weight", 0.0))

        if node_weight_sum > 0.0:
            normalized_weight = current_weight / node_weight_sum
        else:
            # Degenerate case: all outgoing weights were 0.0 → keep them at 0.0
            normalized_weight = 0.0

        edge_data["weight"] = float(normalized_weight)
