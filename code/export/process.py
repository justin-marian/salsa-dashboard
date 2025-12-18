from typing import Any
import networkx as nx

from .sizes import get_size_metrics
from .colors import get_color_metrics, generate_color_palette, get_node_color
from .summarize import build_node_tooltip, build_node_properties
from .visualize import (
    calculate_adaptive_node_size,
    calculate_adaptive_edge_width,
)


def process_nodes(
    graph: nx.Graph | nx.DiGraph,
    hub_scores: dict[Any, float],
    authority_scores: dict[Any, float],
    hits_hub_scores: dict[Any, float],
    hits_authority_scores: dict[Any, float],
    pagerank_scores: dict[Any, float],
    community_assignments: dict[Any, int],
    size_metric: str,
    color_strategy: str,
    node_metadata: dict[Any, dict[str, Any]] | None,
    community_names: dict[int, str],
    out_degree: dict[Any, float],
    in_degree: dict[Any, float],
) -> dict[str, Any]:
    """
    Convert all graph nodes into a ReGraph-compatible payload.

    For each node we compute:
        - a visual size (driven by `size_metric` via adaptive scaling),
        - a display color (driven by `color_strategy`, usually community),
        - a short label (truncated title or node ID),
        - a tooltip with human-readable info,
        - a rich `properties` dict with all scalar metrics.

    The result is ready to be merged into the final JSON export.

    Args:
        graph:
            NetworkX graph (directed or undirected).
        hub_scores, authority_scores:
            SALSA hub/authority scores per node.
        hits_hub_scores, hits_authority_scores:
            HITS hub/authority scores per node.
        pagerank_scores:
            PageRank scores per node.
        community_assignments:
            Mapping node -> community ID.
        size_metric:
            Key selecting which metric controls node size
            (e.g. "pagerank", "deg_out", "salsa_hub").
        color_strategy:
            Either "community" or a metric key for metric-based coloring.
        node_metadata:
            Optional external metadata keyed by node
            (e.g. {"title": ..., "desc": ..., "href": ...}).
        community_names:
            Mapping community ID -> human-readable name.
        out_degree, in_degree:
            Precomputed degree values per node.

    Returns:
        Dictionary of node items keyed by node ID string, e.g.:
        {
            "42": {
                "label": {...},
                "size": 23.0,
                "color": "#4f46e5",
                "tooltip": "...",
                "properties": {...},
                "group": "c3",
            },
            ...
        }
    """
    nodes_info: dict[str, Any] = {}

    size_metrics = get_size_metrics(
        out_degree,
        in_degree,
        hub_scores,
        authority_scores,
        pagerank_scores,
    )
    color_metrics = get_color_metrics(
        out_degree,
        in_degree,
        hub_scores,
        authority_scores,
        pagerank_scores,
    )

    # Choose which metric drives node size; fall back explicitly to pagerank.
    if size_metric in size_metrics:
        size_data = size_metrics[size_metric]
    else:
        size_data = pagerank_scores

    all_size_values = list(size_data.values())

    community_count = (
        len(set(community_assignments.values()))
        if community_assignments
        else 1
    )
    color_palette = generate_color_palette(community_count)

    for node in graph.nodes():
        if node in size_data:
            raw_size_value = float(size_data[node])
        else:
            raw_size_value = 0.0

        visual_size = calculate_adaptive_node_size(
            raw_size_value,
            all_size_values,
        )

        color = get_node_color(
            node,
            color_strategy,
            community_assignments,
            color_metrics,
            color_palette,
        )

        metadata = node_metadata[node] if (node_metadata and node in node_metadata) else {}
        display_title = metadata["title"].strip() if ("title" in metadata and metadata["title"]) else str(node)

        community_id = int(community_assignments[node]) if node in community_assignments else 0
        community_name = (
            community_names[community_id]
            if community_id in community_names
            else f"Community {community_id}"
        )
        community_size = sum(
            1
            for _, comm_id in community_assignments.items()
            if comm_id == community_id
        )

        tooltip = build_node_tooltip(
            node,
            display_title,
            community_name,
            community_id,
            community_size,
            hub_scores,
            authority_scores,
            pagerank_scores,
            metadata,
        )
        properties = build_node_properties(
            node,
            pagerank_scores,
            hub_scores,
            authority_scores,
            hits_hub_scores,
            hits_authority_scores,
            out_degree,
            in_degree,
            community_id,
            community_name,
            community_size,
            metadata,
        )

        short_label = display_title[:60]
        nodes_info[str(node)] = {
            "label": {"text": short_label},
            "size": visual_size,
            "color": color,
            "tooltip": tooltip,
            "properties": properties,
            "group": f"c{community_id}",
        }

    return nodes_info  # 


def process_edges(
    graph: nx.Graph | nx.DiGraph,
    community_assignments: dict[Any, int],
    all_weights: list[float],
) -> dict[str, Any]:
    """
    Convert all graph edges into a ReGraph-compatible payload.

    Each edge entry contains:
    - endpoints (id1, id2),
    - a width scaled from the raw edge weight,
    - a directional arrow flag (for directed graphs),
    - properties describing weight and community relationship.

    Args:
        graph:
            NetworkX graph (directed or undirected).
        community_assignments:
            Mapping node -> community ID.
        all_weights:
            List of all edge weights in the graph, used for
            robust scaling inside `calculate_adaptive_edge_width`.

    Returns:
        Dictionary of edge items keyed as "e0", "e1", ..., e.g.:
        {
            "e0": {
                "id1": "u",
                "id2": "v",
                "arrow": {"to": true},
                "width": 3.2,
                "properties": {...},
            }
        }
    """
    items: dict[str, Any] = {}
    is_directed = graph.is_directed()

    for edge_index, (source, target) in enumerate(graph.edges()):
        raw_weight = float(graph[source][target]["weight"])
        visual_width = calculate_adaptive_edge_width(raw_weight, all_weights)

        source_community = int(community_assignments[source]) if source in community_assignments else -1
        target_community = int(community_assignments[target]) if target in community_assignments else -1
        same_community = source_community == target_community

        items[f"e{edge_index}"] = {
            "id1": str(source),
            "id2": str(target),
            "arrow": {"to": is_directed},
            "width": visual_width,
            "properties": {
                "weight": raw_weight,
                "sameCommunity": same_community,
                "sourceCommunity": source_community,
                "targetCommunity": target_community,
            },
        }

    return items  # 
