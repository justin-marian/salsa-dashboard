import networkx as nx
from collections import Counter
from typing import Any


def get_community_names(
    G: nx.Graph | nx.DiGraph,
    communities: dict[Any, int],
    node_meta: dict[Any, dict[str, str]] | None = None,
    top_k: int = 10,
) -> dict[int, str]:
    """
    Generate descriptive names for communities based on prominent nodes.
    Analyzes community composition by identifying top nodes through degree
    centrality and extracts meaningful names from available metadata. Falls
    back to size-based naming when metadata is unavailable, ensuring all
    communities receive informative labels for visualization purposes.
    """
    comm_nodes = {}
    for node, comm_id in communities.items():
        comm_nodes.setdefault(comm_id, []).append(node)

    comm_names = {}

    for comm_id, nodes in comm_nodes.items():
        top_nodes = sorted(nodes, key=lambda n: G.degree(n), reverse=True)[:top_k]
        names_found = []
        for node in top_nodes:
            if node_meta and node in node_meta:
                meta = node_meta[node]
                title = meta.get("title", "").strip()
                if title:
                    words = title.split()[:3]
                    if words:
                        names_found.append(" ".join(words))
                        continue
                desc = meta.get("desc", "").strip()
                if desc:
                    words = desc.split()[:2]
                    if words:
                        names_found.append(" ".join(words))
        if names_found:
            most_common = Counter(names_found).most_common(1)[0][0]
            comm_names[comm_id] = f"{most_common} Community"
        else:
            # fallback to generic if we have nothing
            comm_names[comm_id] = f"Community {comm_id}"

    return comm_names
