from typing import Any


def get_size_metrics(
    out_degree: dict[Any, float],
    in_degree: dict[Any, float],
    hub_scores: dict[Any, float],
    authority_scores: dict[Any, float],
    pagerank_scores: dict[Any, float],
) -> dict[str, dict[Any, float]]:
    """
    Collect all node-level metrics that can drive node sizing.

    The returned mapping is the size analogue of `get_color_metrics`:
    each entry is a dictionary keyed by node with a non-negative
    score that can be mapped to a visual radius.

    The keys are used by the `size_by` argument in the export
    pipeline and should match any UI dropdowns for node size.

    Returns:
        {
            "pagerank":   {node -> float},
            "salsa_hub":  {node -> float},
            "salsa_auth": {node -> float},
            "deg_out":    {node -> float},
            "deg_in":     {node -> float},
        }
    """
    return {
        "pagerank": pagerank_scores,
        "salsa_hub": hub_scores,
        "salsa_auth": authority_scores,
        "deg_out": out_degree,
        "deg_in": in_degree,
    }  # 
