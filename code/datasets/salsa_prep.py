import networkx as nx
from typing import Union


def S_identity(G: Union[nx.Graph, nx.DiGraph]) -> Union[nx.Graph, nx.DiGraph]:
    """
    Identity graph transformation.

    This is the "do nothing" structural operator: it simply returns the input
    graph as-is. Use it when you want to plug into a pipeline that expects an
    S_* transformation but you explicitly *do not* want to modify the topology.

    Typical use cases
    -----------------
    - Baseline runs where you trust the raw edges.
    - Comparing against other S_* transforms (e.g., S_break_reciprocals).
    - Keeping semantics of signed / weighted / directed graphs untouched.

    Parameters
    ----------
    G:
        Any NetworkX graph (directed or undirected). Node attributes and edge
        attributes are left unchanged.

    Returns
    -------
    G:
        The same graph object passed in (no copy is made).
    """
    return G


def S_break_reciprocals(G: nx.DiGraph) -> nx.DiGraph:
    """
    Remove one direction from every reciprocal edge pair in a directed graph.

    For directed graphs where edges represent endorsement / trust / votes,
    reciprocal pairs (u→v and v→u) can create small 2-cycles that:
        - amplify scores for tightly coupled pairs, and
        - sometimes slow or destabilize iterative methods (SALSA, HITS, PageRank).

    This transformation **deterministically** breaks each 2-cycle by:
    - keeping exactly one of {u→v, v→u}, chosen by a canonical ordering
    of the node identifiers, and
    - preserving all node attributes and edge attributes on kept edges.

    Rule
    ----
    For each unordered pair {u, v} where both u→v and v→u exist:

        - Convert nodes to strings.
        - If str(u) < str(v), keep edge u→v and drop v→u.
        - If str(v) < str(u), keep edge v→u and drop u→v.

    Parameters
    ----------
    G:
        A directed NetworkX graph (nx.DiGraph) whose reciprocal edges you
        want to break. If the graph is not directed, a ValueError is raised.

    Returns
    -------
    H:
        A new directed graph (nx.DiGraph) with:
            - the same node set and node attributes as `G`,
            - a subset of the original edges, with at most one direction for
            every reciprocal pair.

    Raises
    ------
    ValueError
        If `G` is not a directed graph.
    """
    if not G.is_directed():
        raise ValueError("Only directed graphs are supported!")

    # New graph: keep all nodes + their attributes, but rebuild the edge set.
    transformed_graph = nx.DiGraph()
    transformed_graph.add_nodes_from(G.nodes(data=True))

    # Iterate over edges and only keep one direction for reciprocals.
    for source, target, edge_data in G.edges(data=True):
        # If there is a reciprocal edge, we apply the canonical ordering rule.
        if G.has_edge(target, source):
            # Keep the edge where source < target under string comparison.
            if str(source) < str(target):
                # Avoid duplicating the same edge if we encounter it twice.
                if not transformed_graph.has_edge(source, target):
                    transformed_graph.add_edge(source, target, **edge_data)
        else:
            # No reciprocal edge: keep the original direction as-is.
            transformed_graph.add_edge(source, target, **edge_data)

    return transformed_graph
