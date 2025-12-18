from typing import Any
import networkx as nx


def range_vals(scores_dict: dict[Any, float]) -> tuple[float, float]:
    """
    Range of values in a dictionary.
    Computes the minimum and maximum values from a dictionary's values,
    with safeguards for empty dictionaries and uniform value distributions.
    """
    if not scores_dict:
        return (0.0, 1.0)
    values = list(scores_dict.values())
    minimum = min(values)
    maximum = max(values)
    if maximum == minimum:
        maximum = minimum + 1e-6
    return (minimum, maximum)


def scale_value(
    value: float, 
    source_range: tuple[float, float], 
    destination_range: tuple[float, float]
) -> float:
    """
    Scale a value from source range to destination range.
    Linearly transforms a value from one numerical range to another,
    handling edge cases like zero-range source intervals.
    """
    source_min, source_max = source_range
    destination_min, destination_max = destination_range
    normalized = (value - source_min) / (source_max - source_min) if (source_max - source_min) != 0 else 0.0
    return destination_min + normalized * (destination_max - destination_min)


def get_degree_data(graph: nx.Graph | nx.DiGraph) -> tuple[dict[Any, float], dict[Any, float]]:
    """
    Extract degree information from graph.
    Computes in-degree and out-degree for directed graphs,
    or uniform degree for undirected graphs.
    """
    if graph.is_directed():
        out_degree = {node: float(graph.out_degree[node]) for node in graph.nodes()}
        in_degree = {node: float(graph.in_degree[node]) for node in graph.nodes()}
    else:
        deg = {node: float(graph.degree[node]) for node in graph.nodes()}
        out_degree = deg.copy()
        in_degree = deg.copy()
    return out_degree, in_degree


def get_edge_weight_range(graph: nx.Graph | nx.DiGraph) -> tuple[float, float]:
    """
    Calculate the range of edge weights in the graph.
    Extracts all edge weights and computes their minimum and maximum values
    with safeguards for uniform weight distributions.
    """
    raw_weights = []
    for source, target in graph.edges():
        raw_weights.append(float(graph[source][target].get("weight", 1.0)))
    if not raw_weights:
        raw_weights = [1.0]
    min_weight = min(raw_weights)
    max_weight = max(raw_weights)
    if max_weight == min_weight:
        max_weight = min_weight + 1e-6
    return min_weight, max_weight
