from typing import Any
import networkx as nx


def strongly_connected_components(Gw: nx.DiGraph) -> dict[Any, int]:
    """
    Compute strongly connected components (SCCs) and return node → component id.

    In a directed graph, a *strongly connected component* is a maximal set of
    nodes such that each node can reach every other node via a directed path.
    Intuitively, within an SCC, there is a directed route between any pair of
    nodes; between different SCCs this is not guaranteed.

    This helper wraps :func:`networkx.strongly_connected_components` and
    flattens its output into a dictionary mapping each node to an integer
    component label in the range ``0 .. (num_components - 1)``. Labels are
    arbitrary but contiguous and deterministic for a given graph.

    Parameters
    ----------
    Gw:
        Directed graph (``nx.DiGraph``) on which to compute SCCs.

    Returns
    -------
    assignments:
        Dictionary mapping each node in ``Gw`` to the index of the strongly
        connected component it belongs to.

    Notes
    -----
    - The order in which components are enumerated (and thus the numeric
    labels) depends on NetworkX's SCC traversal, but is stable for the same
    graph and NetworkX version.
    - SCCs are often used to collapse a directed graph into a DAG of
    components or as “communities” in reachability-based analyses.
    """
    assignments: dict[Any, int] = {}

    # NetworkX returns a generator of sets of nodes, each set being one SCC.
    components = list(nx.strongly_connected_components(Gw))

    # Assign a unique integer id to each component and propagate it to nodes.
    for cid, component in enumerate(components):
        for node in component:
            assignments[node] = cid

    return assignments
