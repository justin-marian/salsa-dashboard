import networkx as nx
from collections import defaultdict, Counter
from typing import Any


def summarize_communities(
    graph: nx.Graph | nx.DiGraph,
    community_assignments: dict[Any, int],
    node_metadata: dict[Any, dict[str, str]] | None = None,
    topk: int = 10
) -> dict[int, dict[str, Any]]:
    """
    Build explanation for each community:
        - size
        - internal density
        - strongest external neighbor
        - representative nodes
    """
    # Group nodes by their community assignments
    community_nodes: dict[int, list[Any]] = defaultdict(list)
    for node, community_id in community_assignments.items():
        community_nodes[community_id].append(node)

    # Analyze internal and external edge patterns
    internal_edges = Counter()
    inter_community_connections: dict[int, Counter] = defaultdict(Counter)    
    for source, target in graph.edges():
        source_community = community_assignments[source]
        target_community = community_assignments[target]
        if source_community == target_community:
            internal_edges[source_community] += 1
        else:
            inter_community_connections[source_community][target_community] += 1
            inter_community_connections[target_community][source_community] += 1

    # Build comprehensive summary for each community
    community_summaries: dict[int, dict[str, Any]] = {}
    for community_id, nodes in community_nodes.items():
        community_size = len(nodes)
        # Calculate internal density (actual edges / possible edges)
        possible_internal_edges = community_size * (community_size - 1)
        if graph.is_directed():
            possible_internal_edges = community_size * (community_size - 1)
        else:
            possible_internal_edges = community_size * (community_size - 1) // 2
        internal_density = internal_edges[community_id] / max(possible_internal_edges, 1)

        # Identify strongest external connection
        external_connections = inter_community_connections[community_id]
        if external_connections:
            strongest_neighbor, connection_strength = external_connections.most_common(1)[0]
        else:
            strongest_neighbor, connection_strength = (None, 0)

        # Select representative nodes based on degree centrality
        representative_nodes = sorted(
            nodes,
            key=lambda node: graph.degree(node), # type: ignore
            reverse=True
        )[:topk]

        # Generate readable titles for representative nodes
        representative_titles = []
        for node in representative_nodes:
            display_text = None
            if node_metadata and node in node_metadata:
                metadata = node_metadata[node]
                if metadata.get("title"):
                    display_text = metadata["title"]
                elif metadata.get("desc"):
                    display_text = metadata["desc"]
            
            if not display_text:
                display_text = str(node)
                
            representative_titles.append(display_text[:80])

        # Calculate additional community metrics
        subgraph = graph.subgraph(nodes)
        average_degree = sum(dict(subgraph.degree()).values()) / max(community_size, 1)
        
        # For directed graphs, calculate in/out degree patterns
        if graph.is_directed():
            in_degrees = [graph.in_degree(node) for node in nodes]  # type: ignore
            out_degrees = [graph.out_degree(node) for node in nodes]  # type: ignore
            avg_in_degree = sum(in_degrees) / max(community_size, 1)
            avg_out_degree = sum(out_degrees) / max(community_size, 1)
        else:
            avg_in_degree = average_degree
            avg_out_degree = average_degree

        community_summaries[community_id] = {
            "size": community_size,
            "internal_density": round(internal_density, 4),
            "strongest_external_neighbor": strongest_neighbor,
            "strongest_external_edges": connection_strength,
            "representatives": representative_titles,
            "average_degree": round(average_degree, 2),
            "average_in_degree": round(avg_in_degree, 2),
            "average_out_degree": round(avg_out_degree, 2),
            "internal_edges": internal_edges[community_id],
            "external_edges": sum(inter_community_connections[community_id].values()),
        }

    return community_summaries


def build_node_tooltip(
    node: Any, 
    display_title: str, 
    community_name: str, 
    community_id: int, 
    community_size: int, 
    hub_scores: dict[Any, float], 
    authority_scores: dict[Any, float], 
    pagerank_scores: dict[Any, float], 
    metadata: dict[str, Any]
) -> str:
    """
    Construct informative node tooltip text.
    Creates a multi-line tooltip showing node information, community details,
    centrality scores, and optional description.
    """
    description = metadata.get("desc", "").strip() if metadata.get("desc") else ""
    tooltip_lines = [
        display_title,
        f"{community_name} (ID {community_id}) · {community_size} nodes",
        f"hub={hub_scores.get(node,0.0):.3f}  "
        f"auth={authority_scores.get(node,0.0):.3f}  "
        f"pr={pagerank_scores.get(node,0.0):.3f}",
    ]
    if description:
        tooltip_lines.append(description[:160])
    return "\n".join(tooltip_lines)


def build_node_properties(
    node: Any, 
    pagerank_scores: dict[Any, float], 
    hub_scores: dict[Any, float], 
    authority_scores: dict[Any, float], 
    hits_hub_scores: dict[Any, float], 
    hits_authority_scores: dict[Any, float], 
    out_degree: dict[Any, float], 
    in_degree: dict[Any, float], 
    community_id: int, 
    community_name: str, 
    community_size: int, 
    metadata: dict[str, Any]
) -> dict[str, Any]:
    """
    Build comprehensive node properties dictionary.
    Compiles all node attributes, centrality scores, and metadata into
    a structured properties dictionary for export.
    """
    properties = {
        "pagerank": float(pagerank_scores.get(node, 0.0)),
        "salsa_hub": float(hub_scores.get(node, 0.0)),
        "salsa_auth": float(authority_scores.get(node, 0.0)),
        "hits_hub": float(hits_hub_scores.get(node, 0.0)),
        "hits_auth": float(hits_authority_scores.get(node, 0.0)),
        "deg_out": out_degree[node],
        "deg_in": in_degree[node],
        "community": community_id,
        "community_name": community_name,
        "community_size": community_size,
    }

    if "href" in metadata:
        properties["href"] = metadata["href"]
    if "desc" in metadata:
        properties["desc"] = metadata["desc"]
    if "title" in metadata and metadata["title"] != str(node):
        properties["title"] = metadata["title"]
    return properties
