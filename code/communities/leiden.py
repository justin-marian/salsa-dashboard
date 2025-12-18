from typing import Any
import networkx as nx
from tqdm.auto import tqdm

import igraph as ig
import leidenalg as la


def leiden_communities(
    UG: nx.Graph,
    *,
    use_weights: bool = False,
    random_state: int = 42,
    iterations: int | None = -1,
    resolution: float = 1.0,
    weight_attr: str = "weight",
    verbose: bool = True,
) -> dict[Any, int]:
    """
    Detect communities using the Leiden algorithm (via igraph + leidenalg).

    The Leiden algorithm is a refinement of Louvain with stronger guarantees:
    - It avoids badly connected communities (each community is internally
    well-connected).
    - It typically finds higher-quality partitions in terms of modularity
    (or related quality functions).
    - It iterates between local node moves and aggregation, refining the
    partition at each level.

    High-level workflow
    -------------------
    1) Convert the NetworkX graph into an igraph.Graph with integer vertex IDs.
    - This keeps the interface flexible (original nodes can be any hashable
        type) while satisfying Leiden's requirement for integer indexing.
    2) Optionally attach edge weights from a given attribute.
    3) Run `leidenalg.find_partition` using the
    `RBConfigurationVertexPartition` model, which supports a resolution
    parameter (Reichardt & Bornholdt configuration model).
    4) Map the resulting partition (list of communities, each a list of vertex
    indices) back to the original node labels, producing: node → community_id

    Parameters
    ----------
    UG:
        Undirected NetworkX graph on which to run Leiden. The algorithm
        conceptually assumes an undirected graph; if you start from a DiGraph,
        convert it beforehand (e.g. `Gw.to_undirected()`).
    use_weights:
        If True, use the edge attribute given by `weight_attr` (default: "weight")
        as the edge weights in the quality function. If False, run an unweighted
        version of the algorithm (all edges weight = 1).
    random_state:
        Random seed to make the Leiden algorithm reproducible.
    iterations:
        Maximum number of refinement iterations:
        - If None or negative, the algorithm runs until convergence according
        to Leiden's internal stopping criteria.
        - If a positive integer, the number of iterations is capped.
    resolution:
        Resolution parameter of the RB configuration model:
        - > 1.0 typically yields more / smaller communities.
        - < 1.0 typically yields fewer / larger communities.
        It plays a similar role to the resolution in Louvain/Leiden literature.
    weight_attr:
        Name of the edge attribute in `UG` that stores the weight values,
        used only when `use_weights=True`.
    verbose:
        If True, prints basic progress and graph size using tqdm.

    Returns
    -------
    assignments:
        Dictionary mapping each original node in `UG` to an integer community id
        in the range 0..(num_communities - 1). Community ids are arbitrary but
        contiguous.

    Notes
    -----
    - We explicitly construct an igraph.Graph from edge lists to decouple the
    internal integer IDs from the original node labels.
    - The choice of `RBConfigurationVertexPartition` corresponds to a
    modularity-like objective based on a configuration model null graph,
    with tunable resolution.
    """
    # Convert NetworkX graph to igraph with integer vertex indices
    # Keep a stable list of nodes so we can map indices ↔ node labels.
    nodes = list(UG.nodes())
    idx = {n: i for i, n in enumerate(nodes)}

    # Convert each (u, v) edge to integer indices (idx[u], idx[v]).
    edges = [(idx[u], idx[v]) for u, v in UG.edges()]

    # Build the igraph.Graph from scratch.
    g = ig.Graph()
    g.add_vertices(len(nodes))
    g.add_edges(edges)

    if verbose:
        tqdm.write(f"[LEIDEN] graph prepared | nodes={g.vcount()} edges={g.ecount()}")

    # Optional edge weights
    if use_weights:
        # Collect weights from the NetworkX graph in the same order as edges.
        # If an edge lacks `weight_attr`, fall back to 1.0.
        weights = [float(UG[u][v].get(weight_attr, 1.0)) for u, v in UG.edges()]
        g.es["weight"] = weights

    # Run Leiden partitioning on the igraph graph
    # We use RBConfigurationVertexPartition:
    #   - resolution_parameter controls community granularity.
    #   - weights (if provided) influence modularity computation.
    #   - n_iterations: -1 means "until convergence" in leidenalg.
    n_iter = iterations if iterations is not None else -1

    part = la.find_partition(
        g,
        la.RBConfigurationVertexPartition,  # Resolution-based configuration model
        weights=g.es["weight"] if use_weights else None,
        resolution_parameter=resolution,
        n_iterations=n_iter,
        seed=random_state,
    )

    # Map partition back to original node labels: node → community_id
    assignments: dict[Any, int] = {}
    for cid, community in enumerate(part):
        for i in community:
            assignments[nodes[i]] = cid

    return assignments
