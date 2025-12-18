from typing import Callable, Any
import numpy as np
import networkx as nx

from .normalize import normalize_outgoing_weights


def community_rank_based_weighting(
    graph: nx.DiGraph,
    community_assignments: dict[Any, int],
    teleportation_probability: float = 0.15,
    score_transformation: Callable[[float], float] = lambda x: x,
    intra_community_bias: float = 0.2,
    max_iterations: int = 100,
) -> nx.DiGraph:
    """
    Metric 2: Community-level PageRank → edge weights.

    High-level idea
    ----------------
    1. Compress the node-level graph into a *community graph*:
        - Each community is a node.
        - C[a, b] counts edges from community a to community b.

    2. Run (personalized) PageRank on this community graph to obtain
    a stationary importance score PR[c] for each community c.

    3. Push those community scores back to the original node-level edges:
        - Edges pointing into highly important communities get larger weights.
        - The `intra_community_bias` δ controls how much we boost edges that
        stay inside a community vs edges going across communities.

    More concretely
    ----------------
    - Let PR[c] be the PageRank score of community c.
    - For a node u in community cu, let:
        m_u = # of outgoing edges u → v with cv == cu  (intra-community)
        n_u = # of outgoing edges u → v with cv != cu  (inter-community)
    and define a node-specific normalization:
        C_u = δ * m_u + n_u    (if C_u = 0, we use 1.0 to avoid division-by-zero)

    - For each edge (u → v), with target community cv:
        score = f(PR[cv])    where f = `score_transformation` (e.g. identity, log, etc.)

    Then:
        * If cu == cv (intra-community):
            w_uv = (δ * score) / C_u
        * If cu != cv (inter-community):
            w_uv = score / C_u

    So for node u, the outgoing weights sum to:
        m_u * (δ * score_intra / C_u) + n_u * (score_inter / C_u)
    which is proportional to scores, then later normalized again by
    `normalize_outgoing_weights` for a fully stochastic interpretation.

    Parameters
    ----------
    graph:
        Node-level directed graph.
    community_assignments:
        Mapping node → community id. Nodes not present in the mapping will be
        assigned to new singleton communities.
    teleportation_probability:
        Teleportation parameter alpha in PageRank. The stochastic matrix is:
            T = alpha * U + (1 - alpha) * P
        where U is uniform over communities and P is the row-normalized
        community transition matrix.
    score_transformation:
        Optional non-linear transform applied to community PageRank scores
        before using them on edges, e.g. `lambda x: x**0.5` or `np.log1p`.
    intra_community_bias:
        Bias δ controlling the relative weight of intra-community vs
        inter-community edges. δ > 1 boosts intra-community edges; δ < 1
        boosts cross-community edges.
    max_iterations:
        Maximum number of PageRank iterations on the community-level graph.

    Returns
    -------
    weighted_graph:
        A new DiGraph with the same nodes and edges as `graph`, and a "weight"
        attribute on each edge that reflects both:
        - the PageRank importance of the *target community*, and
        - the intra/inter community structure around the source node.
    """
    # Ensure every node has a community label (fill missing as singleton communities)
    assign = dict(community_assignments)
    if len(assign) != graph.number_of_nodes():
        next_id = (max(assign.values()) if assign else -1) + 1
        for n in graph.nodes():
            if n not in assign:
                assign[n] = next_id
                next_id += 1

    if graph.number_of_nodes() == 0:
        return nx.DiGraph()

    # Initialize new graph with original node structure
    weighted_graph = nx.DiGraph()
    weighted_graph.add_nodes_from(graph.nodes(data=True))

    # Extract all community IDs and build a compact index mapping
    community_ids = sorted(set(assign.values()))
    community_count = len(community_ids)

    # Edge case: no communities (should not happen since we backfilled above)
    if community_count == 0:
        weighted_graph.add_edges_from(
            (u, v, {"weight": 0.0}) for u, v in graph.edges()
        )
        normalize_outgoing_weights(weighted_graph)
        return weighted_graph

    cid_to_idx = {cid: i for i, cid in enumerate(community_ids)}

    # Build community-level transition count matrix C
    C = np.zeros((community_count, community_count), dtype=float)
    for u, v in graph.edges():
        su = cid_to_idx[assign[u]]
        tv = cid_to_idx[assign[v]]
        C[su, tv] += 1.0

    # Row-normalize C to obtain transition probabilities P
    row_sums = C.sum(axis=1, keepdims=True)
    P = np.divide(
        C,
        np.where(row_sums == 0.0, 1.0, row_sums),
        where=row_sums != 0.0,
    )

    # Teleportation mix: T = α U + (1 − α) P
    U = np.full((community_count, community_count), 1.0 / community_count, dtype=float)
    T = teleportation_probability * U + (1.0 - teleportation_probability) * P

    # PageRank on communities
    pr = np.full(community_count, 1.0 / community_count, dtype=float)
    for _ in range(max_iterations):
        nxt = pr @ T
        s = nxt.sum()
        if s > 0:
            nxt /= s
        if np.linalg.norm(nxt - pr, 1) < 1e-9:
            pr = nxt
            break
        pr = nxt

    # Map community index scores back to original community IDs
    community_scores = {cid: float(pr[cid_to_idx[cid]]) for cid in community_ids}

    # Per-node intra/inter counts + C_u normalization constants
    intra_cnt: dict[Any, int] = {}
    inter_cnt: dict[Any, int] = {}
    for n in graph.nodes():
        c = assign[n]
        i = j = 0
        for nb in graph.adj[n]:
            if assign[nb] == c:
                i += 1
            else:
                j += 1
        intra_cnt[n] = i
        inter_cnt[n] = j

    norm_const: dict[Any, float] = {}
    for n in graph.nodes():
        denom = intra_community_bias * intra_cnt[n] + inter_cnt[n]
        norm_const[n] = denom if denom > 0 else 1.0  # avoid division by zero

    # Edge-level weights from community PR + intra/inter bias
    for u, v in graph.edges():
        cu = assign[u]
        cv = assign[v]

        # Apply any non-linear transform to the PR score of the target community
        score = score_transformation(community_scores[cv])

        if cu == cv:
            w = (intra_community_bias * score) / norm_const[u]
        else:
            w = score / norm_const[u]

        weighted_graph.add_edge(u, v, weight=float(w))

    # Final safety normalization: make outgoing weights stochastic per node
    normalize_outgoing_weights(weighted_graph)
    return weighted_graph
