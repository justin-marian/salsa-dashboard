from typing import Any
import networkx as nx


def kcore_communities(
    UG: nx.Graph,
    *,
    k: int | None = None
) -> dict[Any, int]:
    """
    Assign each node to a k-core level as a simple structural “community”.

    The k-core of a graph is the largest subgraph where every node has degree ≥ k.
    Each node has a *core number* (also called coreness), representing the deepest
    k-core it belongs to. Nodes with higher core numbers are typically located in
    the dense, well-connected “core” of the network, whereas peripheral nodes have
    lower core numbers.

    This function uses NetworkX's `core_number(G)` and converts the coreness values
    into a flat assignment: node → core_id

    Two usage modes
    ---------------
    1. k=None  (default)
    - Each node gets its exact coreness value.
        Example: node X with coreness 3 → community id 3.

    2. k is provided
    - Collapse all nodes with coreness < k into a single bucket (k-1).
        This produces a binary-style partition:
        - nodes with coreness ≥ k stay as their own levels
        - all low-coreness nodes get merged together.
    Useful when you want to emphasize dense-core membership vs. everything else.

    Parameters
    ----------
    UG : nx.Graph
        Undirected graph on which to compute core numbers.
    k : int | None
        Optional threshold for collapsing low-coreness nodes. If None, raw core
        numbers are used directly.

    Returns
    -------
    assignments : dict[Any, int]
        Mapping from node to its assigned core-based community label.
    """
    # Compute coreness for all nodes: node → integer core number
    core = nx.core_number(UG)

    # If no threshold is provided, use coreness directly
    if k is None:
        return {node: c for node, c in core.items()}

    # If threshold provided, merge low-coreness nodes into bucket k-1
    assignments: dict[Any, int] = {}
    for node, c in core.items():
        if c >= k:
            assignments[node] = c
        else:
            assignments[node] = k - 1

    return assignments
