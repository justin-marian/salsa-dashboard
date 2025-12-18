import networkx as nx
from tqdm import tqdm
from collections import Counter, defaultdict
from typing import Any

from communities.communities import assign_communities


def split_large_communities(
    graph: nx.DiGraph,
    community_assignments: dict[Any, int],
    max_size_ratio: float = 0.40,
    resolution_boost: float = 2.0,
    verbose: bool = True,
) -> dict[Any, int]:
    """
    Split communities that are too large compared to the whole graph.

    Motivation
    ----------
    Community detection (e.g. Louvain) often produces one or a few
    “giant” communities that swallow a big fraction of the graph.
    This function detects such oversized communities and tries to
    break them into smaller, more structured subcommunities.

    How it works
    ------------
    1. Measure community sizes from `community_assignments`.
    2. Any community whose size / total_nodes > `max_size_ratio` is
    treated as oversized.
    3. For each oversized community:
    - Extract its induced subgraph.
    - Re-run community detection (Louvain) on that subgraph with a
    higher resolution (`resolution_boost`).
    - Replace the original single community label with a pair
    (parent_id, sub_id) to preserve hierarchy.
    4. After all splits, compress these hierarchical labels back down
    to flat integers 0..K-1 via a label remapping.

    Parameters
    ----------
    graph:
        Original directed graph on which communities were detected.
    community_assignments:
        Mapping node → community id (usually output of `assign_communities`).
    max_size_ratio:
        Maximum allowed size fraction of a single community. Any community
        larger than this (relative to total number of nodes) will be split.
        Example: 0.40 means "no community should contain more than 40% of nodes".
    resolution_boost:
        Resolution parameter passed to the second-stage Louvain. Higher
        resolution tends to produce more, smaller communities.
    verbose:
        If True, logs which communities are being split and how many
        new communities result.

    Returns
    -------
    refined_assignments:
        Mapping node → new community id (flat integers 0..K-1) after splitting.
    """
    refined_assignments: dict[Any, Any] = dict(community_assignments)
    total_nodes = len(community_assignments)
    community_sizes = Counter(community_assignments.values())

    # Identify and process oversized communities
    for community_id, community_size in list(community_sizes.items()):
        if total_nodes == 0:
            continue

        community_ratio = community_size / total_nodes
        if community_ratio <= max_size_ratio:
            continue  # This community is within acceptable bounds

        if verbose:
            tqdm.write(
                f"[COMM-PROCESS] Splitting oversized community {community_id} "
                f"({community_size} nodes, {100 * community_ratio:.1f}% of graph)"
            )

        # Nodes belonging to this oversized community
        community_nodes = [
            node
            for node, comm_id in community_assignments.items()
            if comm_id == community_id
        ]
        # Induced subgraph for local re-clustering
        community_subgraph = graph.subgraph(community_nodes).copy()

        # Re-run community detection with higher resolution
        subcommunity_assignments = assign_communities(
            community_subgraph,
            method="louvain",
            use_weights=False,      # `use_weights` can be adjusted as needed
            random_state=42,
            verbose=False,
            louvain_resolution=resolution_boost,
        )

        # Assign hierarchical labels: (parent_id, sub_id)
        for node, subcommunity_id in subcommunity_assignments.items():
            refined_assignments[node] = (community_id, int(subcommunity_id))

    # Compact hierarchical labels (or original labels) to 0..K-1
    label_remapping: dict[Any, int] = {}
    next_community_id = 0

    for node, community_label in refined_assignments.items():
        if community_label not in label_remapping:
            label_remapping[community_label] = next_community_id
            next_community_id += 1
        refined_assignments[node] = label_remapping[community_label]

    # Ensure plain int values
    assignments: dict[Any, int] = {
        node: int(label) for node, label in refined_assignments.items()
    }
    return assignments


def merge_small_communities(
    graph: nx.DiGraph,
    community_assignments: dict[Any, int],
    min_size_ratio: float = 0.005,
    min_absolute_size: int = 100,
    verbose: bool = True,
) -> dict[object, int]:
    """
    Merge too-small communities into better-connected neighbors.

    Motivation
    ----------
    Some community detection runs produce many tiny, statistically weak
    communities (e.g. 2-3 nodes). These can be noisy or visually
    unhelpful. This function identifies such undersized communities and
    merges them into larger, well-connected neighbors.

    How it works
    ------------
    1. Compute total nodes and community sizes.
    2. Compute the minimum allowed community size as:
    max(min_absolute_size, min_size_ratio * total_nodes).
    3. Build a community-community connectivity map:
    edge_count(C_i, C_j) = number of edges between C_i and C_j.
    4. For each community smaller than the threshold:
    - Find neighboring communities it connects to.
    - Score each neighbor by (edges_between / neighbor_size).
    - Merge into the neighbor with the highest score.
    5. After merging, compress labels to 0..K-1.

    Parameters
    ----------
    graph:
        Directed graph whose edges are used to measure inter-community
        connectivity.
    community_assignments:
        Mapping node → community id (typically output of a previous step).
    min_size_ratio:
        Minimum community size expressed as a fraction of total nodes.
        Effective threshold is max(min_absolute_size, min_size_ratio * N).
    min_absolute_size:
        Hard lower bound on community size, regardless of graph size.
    verbose:
        If True, prints which communities are being merged and into whom.

    Returns
    -------
    refined_assignments:
        Mapping node → merged community id (0..K-1).
    """
    refined_assignments = dict(community_assignments)
    total_nodes = len(community_assignments)

    # Minimum community size threshold
    minimum_size = max(min_absolute_size, int(min_size_ratio * max(total_nodes, 1)))
    community_sizes = Counter(refined_assignments.values())

    # Build community-to-community connectivity (undirected in community space)
    inter_community_edges: dict[int, Counter] = defaultdict(Counter)
    for source, target in graph.edges():
        source_community = refined_assignments[source]
        target_community = refined_assignments[target]

        if source_community != target_community:
            inter_community_edges[source_community][target_community] += 1
            inter_community_edges[target_community][source_community] += 1

    # Process undersized communities
    for community_id, community_size in list(community_sizes.items()):
        if community_size >= minimum_size:
            continue  # Community meets minimum size requirements

        if verbose:
            tqdm.write(
                f"[COMM-PROCESS] Merging undersized community {community_id} "
                f"({community_size} nodes < {minimum_size})"
            )

        # Communities connected to this small community
        neighbor_communities = inter_community_edges.get(community_id, {})
        if not neighbor_communities:
            # Isolated community with no external edges - leave it alone
            continue

        # Find best merge candidate based on normalized connectivity
        best_candidate = None
        best_connection_score = -1.0

        for neighbor_id, edge_count in neighbor_communities.items():
            if neighbor_id == community_id:
                continue

            neighbor_size = community_sizes.get(neighbor_id, 1)
            # Score: edge density to this neighbor normalized by its size
            connection_score = edge_count / max(neighbor_size, 1)

            if connection_score > best_connection_score:
                best_connection_score = connection_score
                best_candidate = neighbor_id

        if best_candidate is None:
            continue

        # Reassign all nodes in small community to best_candidate
        for node, current_community in list(refined_assignments.items()):
            if current_community == community_id:
                refined_assignments[node] = best_candidate

        # Update community size bookkeeping
        community_sizes[best_candidate] += community_size
        community_sizes[community_id] = 0

    # Compact community labels to sequential integers 0..K-1
    label_remapping: dict[int, int] = {}
    next_community_id = 0

    for node, community_id in refined_assignments.items():
        if community_id not in label_remapping:
            label_remapping[community_id] = next_community_id
            next_community_id += 1
        refined_assignments[node] = label_remapping[community_id]

    return refined_assignments


def balance_community_sizes(
    graph: nx.DiGraph,
    community_assignments: dict[Any, int],
    verbose: bool = True,
) -> dict[Any, int]:
    """
    Two-stage community size balancing: split giants, merge dwarfs.

    This is a convenience wrapper that applies both post-processing
    stages in sequence:

        1) split_large_communities(...)
        2) merge_small_communities(...)

    The goal is to:
    - avoid huge, over-dominant communities that hide local structure;
    - avoid tiny, noisy communities that have too few nodes.

    Parameters
    ----------
    graph:
        Directed graph used both for splitting (subgraph extraction)
        and for measuring inter-community connectivity in merging.
    community_assignments:
        Initial node → community id mapping (e.g. raw Louvain output).
    verbose:
        If True, logs the number of communities before and after balancing.

    Returns
    -------
    balanced_assignments:
        Final node → community id (0..K-1) after splitting and merging.
    """
    if verbose:
        tqdm.write("[COMM-PROCESS] Starting community balancing pipeline")

    after_splitting = split_large_communities(
        graph, community_assignments, verbose=verbose
    )
    after_merging = merge_small_communities(
        graph, after_splitting, verbose=verbose
    )

    if verbose:
        initial_count = len(set(community_assignments.values()))
        final_count = len(set(after_merging.values()))
        tqdm.write(
            f"[COMM-PROCESS] Balancing complete: "
            f"{initial_count} -> {final_count} communities"
        )

    return after_merging
