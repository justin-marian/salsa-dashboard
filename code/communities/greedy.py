from typing import Any
import networkx as nx
from networkx.algorithms.community import greedy_modularity_communities


def greedy_communities(
    UG: nx.Graph,
    *,
    use_weights: bool = False,
    resolution: float = 1.0,
) -> dict[Any, int]:
    """
    Detect communities using greedy modularity maximization.

    This wraps NetworkX's :func:`greedy_modularity_communities`, which
    implements the Clauset-Newman-Moore (CNM) algorithm:

    High-level idea
    ---------------
    - Start with each node as its own community.
    - At each step, merge the pair of communities that yields the largest
    increase (or smallest decrease) in modularity.
    - Stop when no merge can improve modularity.

    Modularity is a quality score that compares the density of edges inside
    communities to what we would expect at random (given node degrees). Higher
    modularity means a “better” partition into tightly knit clusters with
    relatively few edges between them.

    This function converts the resulting list of communities (each a set of
    nodes) into a flat mapping: node → integer community ID (0..C-1).

    Parameters
    ----------
    UG:
        Undirected graph on which to run community detection.
        (Greedy modularity is defined for undirected graphs.)
    use_weights:
        If True, uses the edge attribute `"weight"` (if present) to compute
        weighted modularity. If False, treats the graph as unweighted and
        ignores any weight attributes.
    resolution:
        Resolution parameter controlling the size/number of communities,
        analogous to that used in Louvain/Leiden:
        - > 1.0 tends to favor smaller, more numerous communities.
        - < 1.0 tends to favor larger, fewer communities.

    Returns
    -------
    assignments:
        Dictionary mapping each node to an integer community label. Labels
        are contiguous and arbitrary (they encode cluster identity, not order).

    Notes
    -----
    - The algorithm is deterministic for a fixed graph and edge weights.
    - Complexity is higher than label propagation but often acceptable for
    medium-sized graphs, making it a good baseline modularity-based method.
    """
    assignments: dict[Any, int] = {}

    # Choose whether to pass a weight attribute to NetworkX.
    # If `use_weights` is False, we pass None so all edges are treated equally.
    wkey = "weight" if use_weights else None

    # NetworkX returns a list of sets of nodes, one set per detected community.
    communities = list(greedy_modularity_communities(UG, weight=wkey, resolution=resolution))

    # Flatten community structure into node → community_id.
    for cid, community in enumerate(communities):
        for node in community:
            assignments[node] = cid

    return assignments
