from typing import Any
import networkx as nx
from collections import defaultdict

from .normalize import normalize_outgoing_weights


def community_interlink_amplification(
    graph: nx.Graph,
    community_assignments: dict[Any, int],
    intra_community_bias: float = 0.2,
) -> nx.Graph:
    """
    Metric 1: Community-aware edge reweighting with bias parameter δ.

    For each source node i, we distinguish between:
    - m_i = number of outgoing edges from i that stay inside i's community
    - n_i = number of outgoing edges from i that go to other communities

    We then define a node-specific normalization constant:

        C_i = δ * m_i + n_i

    and assign edge weights as:
    - if edge (i → j) is intra-community:
        w_ij = δ / C_i
    - if edge (i → j) is inter-community:
        w_ij = 1 / C_i

    so that the total outgoing weight from node i is:

        m_i * (δ / C_i) + n_i * (1 / C_i) = (δ * m_i + n_i) / C_i = 1.

    In other words:
    - δ < 1: inter-community edges are relatively stronger than intra-community
    edges (community “bridges” are emphasized).
    - δ > 1: intra-community edges are relatively stronger 
    (community cohesion is emphasized).
    - δ = 1: no bias; all outgoing edges from a node are equally weighted.

    After constructing the new graph, we call `normalize_outgoing_weights`
    as a safety step to maintain stochastic outgoing weights per node.

    Parameters
    ----------
    graph:
        Input graph. If directed, neighbors are interpreted as successors
        and the outgoing weights from each node are normalized.
    community_assignments:
        Mapping node → community id. Nodes missing from the mapping will be
        assigned to new singleton communities so that every node has a label.
    intra_community_bias:
        Bias parameter δ controlling relative strength of intra-community
        vs inter-community edges in the reweighted graph.

    Returns
    -------
    amplified_graph:
        A new graph of the same type as `graph` (e.g. Graph / DiGraph) with
        the same nodes and edges, but with a "weight" attribute reflecting
        the community-aware reweighting.
    """
    # Make sure every node has a community label; assign fallbacks if needed
    assign = dict(community_assignments)
    if len(assign) != graph.number_of_nodes():
        base = (max(assign.values()) if assign else -1) + 1
        for n in graph.nodes():
            if n not in assign:
                assign[n] = base
                base += 1

    # Create an empty graph of the same class, copy node attributes
    amplified_graph = graph.__class__()
    amplified_graph.add_nodes_from(graph.nodes(data=True))

    # Count, for each source node, how many neighbors are in / out of its community
    intra_community_edges = defaultdict(int)
    inter_community_edges = defaultdict(int)

    for source_node in graph.nodes():
        s_comm = assign[source_node]
        for target_node in graph.neighbors(source_node):
            t_comm = assign[target_node]
            if s_comm == t_comm:
                intra_community_edges[source_node] += 1
            else:
                inter_community_edges[source_node] += 1

    # Precompute C_i = δ * m_i + n_i, with a fallback of 1.0 for nodes with no neighbors
    normalization_constants = {}
    for node in graph.nodes():
        intra_count = intra_community_edges[node]
        inter_count = inter_community_edges[node]
        denom = intra_community_bias * intra_count + inter_count
        normalization_constants[node] = denom if denom > 0 else 1.0

    # Assign reweighted edges according to the intra/inter rule above
    for source_node, target_node in graph.edges():
        s_comm = assign[source_node]
        t_comm = assign[target_node]
        if s_comm == t_comm:
            w = intra_community_bias / normalization_constants[source_node]
        else:
            w = 1.0 / normalization_constants[source_node]

        amplified_graph.add_edge(source_node, target_node, weight=float(w))

    # Final safety renormalization of outgoing weights
    normalize_outgoing_weights(amplified_graph)
    return amplified_graph
