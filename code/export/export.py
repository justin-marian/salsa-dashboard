import json
import os
import networkx as nx
from typing import Any

from .names import get_community_names
from .summarize import summarize_communities
from .readable import build_human_readable_community_text
from .process import process_nodes, process_edges
from .utils import get_degree_data


def export_regraph_items(
    graph: nx.Graph | nx.DiGraph,
    hub_scores: dict[Any, float],
    authority_scores: dict[Any, float],
    hits_hub_scores: dict[Any, float],
    hits_authority_scores: dict[Any, float],
    pagerank_scores: dict[Any, float],
    community_assignments: dict[Any, int],
    output_file: str,
    *,
    size_by: str = "pagerank",
    color_by: str = "community",
    node_metadata: dict[Any, dict[str, Any]] | None = None,
) -> None:
    """
    Export graph data to ReGraph-compatible JSON format.
    Converts NetworkX graph with centrality scores and community assignments
    into a comprehensive JSON structure suitable for visualization in ReGraph.
    Includes node styling, edge information, and community metadata.
    """
    out_degree, in_degree = get_degree_data(graph)
    # Get all edge weights for adaptive scaling
    all_weights = []
    for source, target in graph.edges():
        all_weights.append(float(graph[source][target]["weight"]))
    
    community_names = get_community_names(graph, community_assignments, node_metadata)
    community_info = summarize_communities(graph, community_assignments, node_metadata)
    community_text = build_human_readable_community_text(community_info, community_names)
    
    node_items = process_nodes(
        graph, 
        hub_scores, authority_scores,
        hits_hub_scores, hits_authority_scores,
        pagerank_scores, community_assignments,
        size_by, color_by, 
        node_metadata, community_names, 
        out_degree, in_degree
    )
    edge_items = process_edges(graph, community_assignments, all_weights)
    items = {**node_items, **edge_items}
    payload = {
        "items": items,
        "community_info": community_info,
        "community_text": community_text,
    }

    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
