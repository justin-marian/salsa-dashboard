from typing import Any
import networkx as nx
from networkx.algorithms.community import asyn_lpa_communities


def label_propagation_communities(
    UG: nx.Graph,
    *,
    use_weights: bool = False,
    random_state: int = 42,
) -> dict[Any, int]:
    """
    Detect communities using Asynchronous Label Propagation (LPA).

    The Label Propagation Algorithm is a very fast, heuristic method for
    discovering communities based on local majority voting:

    High-level idea
    ---------------
    1. Initialization:
    - Each node starts with a unique label (often its own id).
    2. Iterative update:
    - Visit nodes in some order (here: asynchronous / random).
    - For each node, look at its neighbors' labels.
    - The node adopts the most frequent label among its neighbors
        (breaking ties at random).
    3. Convergence:
    - Repeat until labels stop changing or a maximum number of steps is
        reached internally.
    - Nodes that end up with the same label form a community.

    Properties
    ----------
    - Extremely fast and scalable: suitable for large graphs.
    - Parameter-free (no explicit resolution / number of communities).
    - Non-deterministic: results can vary across runs due to the random
    update order and tie-breaking (we expose `random_state` to control this).
    - Works on undirected graphs; for directed graphs you typically use an
    undirected projection upstream.

    Parameters
    ----------
    UG:
        Undirected NetworkX graph on which to run label propagation.
    use_weights:
        If True, use the edge attribute "weight" (if present) to give
        higher influence to stronger edges when counting neighbor labels.
        If False, all edges are treated equally.
    random_state:
        Seed controlling the random update order and tie-breaking inside the
        algorithm, making results reproducible for a given graph.

    Returns
    -------
    assignments:
        Dictionary mapping each node to an integer community id (0..C-1),
        derived from the distinct labels output by LPA. Community ids are
        contiguous but otherwise arbitrary.

    Notes
    -----
    - Because LPA is heuristic and order-dependent, it is often used as a
    quick baseline or for exploratory analysis, rather than as a final,
    high-precision community detector.
    """
    assignments: dict[Any, int] = {}

    # Choose whether to pass a weight attribute to NetworkX.
    # If `use_weights` is False, we pass None so all edges have equal influence.
    wkey = "weight" if use_weights else None

    # NetworkX returns an iterable of sets of nodes, each set representing a
    # community discovered by asynchronous label propagation.
    communities = list(asyn_lpa_communities(UG, weight=wkey, seed=random_state))

    # Flatten community structure into node → community_id.
    for cid, community in enumerate(communities):
        for node in community:
            assignments[node] = cid

    return assignments
