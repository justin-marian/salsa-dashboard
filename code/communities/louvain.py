from typing import Any
import time
import networkx as nx
from tqdm.auto import tqdm

from community import community_louvain

from datasets.build import undirected_graph


def louvain_partition(
    UG: nx.Graph,
    *,
    use_weights: bool = False,
    random_state: int = 42,
    level: int | None = None,
    resolution: float = 1.0,
    weight_attr: str = "weight",
    verbose: bool = True,
) -> dict[object, int]:
    """
    Run the Louvain algorithm and return a node → community id mapping.

    The Louvain method is a multilevel modularity optimization algorithm:

    High-level idea
    ---------------
    1. Local moving phase:
    - Start with each node in its own community.
    - Iteratively move nodes to neighboring communities if this increases
        modularity (a measure of within-community vs. between-community edges).
    2. Aggregation phase:
    - Collapse each community into a super-node.
    - Build a new, smaller graph whose nodes represent communities.
    - Repeat the local moving phase on this coarse graph.

    This produces a hierarchy (dendrogram) of partitions with increasing
    coarseness. You can either:
        - Take the “best” partition (default, `level=None`), or
        - Cut the hierarchy at a given `level`.

    Parameters
    ----------
    UG:
        Input graph (typically undirected). If the original graph is directed,
        it should already have been projected to an undirected graph upstream.
    use_weights:
        If True, treat the edge attribute `weight_attr` as the edge weight in
        modularity computations. If False, all edges are treated as weight = 1.
    random_state:
        Seed used by the Louvain implementation for reproducible results.
    level:
        Dendrogram level to extract from:
        - If None: use `community_louvain.best_partition` (default “best” cut).
        - If an integer: generate the full dendrogram and pick that level,
        clamped to the range [0, len(dendrogram)-1].
    resolution:
        Resolution parameter that controls community granularity:
        - > 1.0 → more / smaller communities.
        - < 1.0 → fewer / larger communities.
    weight_attr:
        Name of the edge attribute containing weights (when `use_weights=True`).
    verbose:
        If True, print a one-line summary with number of groups and runtime.

    Returns
    -------
    out:
        Dictionary mapping each node to an integer community id (0..C-1).
        Community ids are contiguous but otherwise arbitrary.

    Notes
    -----
    - We first convert to a simple undirected graph via `undirected_graph(UG)`.
    This can remove parallel edges and normalize directionality.
    - If the graph has no edges, the “partition” is trivial: each node is
    its own community.
    """
    if UG.number_of_nodes() == 0:
        # Empty graph: no nodes → empty partition.
        return {}

    started = time.perf_counter()

    # Ensure we have a simple undirected view (removes multi-edges, normalizes).
    G = undirected_graph(UG)

    # If we ignore weights, rebuild G as an unweighted graph by dropping the
    # weight attribute entirely and just retaining topology.
    if not use_weights:
        H = nx.Graph()
        H.add_nodes_from(G.nodes())
        H.add_edges_from(G.edges())
        G = H

    # If there are no edges, Louvain cannot merge anything: each node is alone.
    if G.number_of_edges() == 0:
        out = {u: i for i, u in enumerate(G.nodes())}
        if verbose:
            elapsed = time.perf_counter() - started
            tqdm.write(f"[LOUVAIN] trivial (no edges) | groups={len(out)} | {elapsed:.3f}s")
        return out

    # Base keyword-args for Louvain.
    bp_kwargs: dict[str, Any] = {
        "resolution": resolution,
        "random_state": random_state,
    }
    if use_weights:
        # Tell community_louvain which edge attribute holds weights.
        bp_kwargs["weight"] = weight_attr

    # Either get the best partition directly, or generate the entire dendrogram
    # and pick a specific level.
    if level is None:
        # Best cut (internally picks a partition from the hierarchy).
        part = community_louvain.best_partition(G, **bp_kwargs)
    else:
        # Full hierarchy: list of partitions from fine → coarse.
        dendro = community_louvain.generate_dendrogram(G, **bp_kwargs)
        # Clamp requested level to valid range.
        lvl = max(0, min(level, len(dendro) - 1))
        part = community_louvain.partition_at_level(dendro, lvl)

    # Ensure all community labels are ints (some versions return numpy types).
    out = {u: int(c) for u, c in part.items()}

    elapsed = time.perf_counter() - started
    if verbose:
        tqdm.write(f"[LOUVAIN] groups={len(set(out.values()))} | {elapsed:.3f}s")

    return out


def louvain_communities(
    UG: nx.Graph,
    use_weights: bool = False,
    random_state: int = 42,
    level: int | None = None,
    resolution: float = 1.0,
) -> dict[Any, int]:
    """
    Convenience wrapper to compute Louvain communities.

    This simply calls :func:`louvain_partition` and returns its
    node → community id mapping.

    Parameters
    ----------
    UG:
        Input (typically undirected) graph.
    use_weights:
        Whether to use edge weights in modularity computation.
    random_state:
        Seed for reproducibility.
    level:
        Dendrogram level to extract, or None for the default best partition.
    resolution:
        Resolution parameter controlling the community granularity.

    Returns
    -------
    coll_map:
        Dictionary mapping node → community id.
    """
    coll_map = louvain_partition(
        UG,
        use_weights=use_weights,
        random_state=random_state,
        level=level,
        resolution=resolution,
    )
    return coll_map
