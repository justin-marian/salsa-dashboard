from typing import Any

from .utils import range_vals, scale_value


def get_color_metrics(
    out_degree: dict[Any, float],
    in_degree: dict[Any, float],
    hub_scores: dict[Any, float],
    authority_scores: dict[Any, float],
    pagerank_scores: dict[Any, float],
) -> dict[str, dict[Any, float]]:
    """
    Collect all node-level metrics that can drive color encoding.

    This function bundles the scalar node scores that we can map
    to colors into a single dictionary. The keys are short labels
    that match the `color_by` / `color_strategy` argument used in
    the export pipeline.

    Returns:
        A mapping from metric name to per-node score dictionary:
        {
            "pagerank":   {node -> float},
            "salsa_hub":  {node -> float},
            "salsa_auth": {node -> float},
            "deg_in":     {node -> float},
            "deg_out":    {node -> float},
        }
    """
    return {
        "pagerank": pagerank_scores,
        "salsa_hub": hub_scores,
        "salsa_auth": authority_scores,
        "deg_in": in_degree,
        "deg_out": out_degree,
    }  # 


def generate_color_palette(color_count: int) -> list[str]:
    """
    Generate a visually pleasing palette of distinct colors.

    The palette is constructed in HSL/HSV space using a golden-ratio
    based hue sequence and soft saturation/lightness settings. The goal
    is to keep communities visually distinct while avoiding harsh,
    neon-like colors.

    Args:
        color_count: Number of distinct colors requested.

    Returns:
        List of hex color strings (e.g. "#4f46e5").
    """
    import colorsys

    if color_count <= 0:
        return ["#4f46e5"]  # sensible default

    golden_ratio_conjugate = 0.618033988749895
    colors: list[str] = []

    for i in range(color_count):
        # Distribute hues using golden ratio for even coverage.
        hue = (i * golden_ratio_conjugate) % 1.0

        # Use slightly softer saturation and lightness to avoid
        # overly aggressive neon colors. We still vary them a bit
        # so adjacent communities are not all the same "strength".
        saturation = 0.55 + 0.15 * ((i % 4) / 3.0)   # 0.55 .. ~0.70
        value = 0.50 + 0.20 * (((i // 4) % 2))       # 0.50 or 0.70

        red, green, blue = colorsys.hsv_to_rgb(hue, saturation, value)
        hex_color = f"#{int(red * 255):02x}{int(green * 255):02x}{int(blue * 255):02x}"
        colors.append(hex_color)

    return colors  # 


def get_node_color(
    node: Any,
    color_strategy: str,
    community_assignments: dict[Any, int],
    color_metrics: dict[str, dict[Any, float]],
    color_palette: list[str],
) -> str:
    """
    Compute the display color for a single node.

    There are two main modes:

    - ``color_strategy == "community"``:
        Each community ID is mapped directly to a color in the palette.
        This is simple but works very well for visually separating groups.

    - ``color_strategy in color_metrics``:
        The given metric (e.g. "pagerank", "deg_out") is used to
        compute a color index by linearly scaling the metric range
        onto the palette indices [0, len(color_palette)-1].

    Args:
        node:
            Node identifier as used in the NetworkX graph.
        color_strategy:
            Either "community" or the name of a metric present in
            the ``color_metrics`` dictionary.
        community_assignments:
            Mapping from node -> community ID.
        color_metrics:
            Mapping from metric name -> {node -> metric value}.
        color_palette:
            List of hex colors to index into.

    Returns:
        Hex color string suitable for the export JSON.
    """
    palette_len = len(color_palette) or 1

    # Community-based coloring: stable and easy to interpret.
    if color_strategy == "community":
        if node in community_assignments:
            community_id = int(community_assignments[node])
        else:
            community_id = 0
        return color_palette[community_id % palette_len]

    # Metric-based coloring: fall back to "pagerank" if unknown strategy.
    if color_strategy in color_metrics:
        color_data = color_metrics[color_strategy]
    else:
        color_data = color_metrics["pagerank"]

    color_range = range_vals(color_data)
    if node in color_data:
        raw_value = float(color_data[node])
    else:
        raw_value = 0.0

    color_index_float = scale_value(
        raw_value,
        color_range,
        (0.0, float(palette_len - 1)),
    )
    color_index = int(color_index_float)
    return color_palette[color_index % palette_len]
