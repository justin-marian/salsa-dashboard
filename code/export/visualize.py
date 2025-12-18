import numpy as np


def calculate_adaptive_edge_width(raw_weight: float, all_weights: list[float]) -> float:
    """
    Map a raw edge weight to a visually useful stroke width.

    Design goals:
    - Always return a width in [1.5, 8.0].
    - Use robust statistics (IQR) so outliers do not dominate.
    - Make higher weights noticeably thicker, but keep thin edges readable.
    """
    # Fallback for empty / invalid input
    if not all_weights:
        return 3.0

    weights = np.asarray(all_weights, dtype=float)
    if weights.size == 0 or not np.isfinite(weights).any():
        return 3.0

    # Clean NaN/Inf
    weights = weights[np.isfinite(weights)]
    if weights.size == 0:
        return 3.0

    q25, q75 = np.percentile(weights, [25, 75])
    iqr = q75 - q25

    min_width = 1.5
    max_width = 8.0

    # Main path: IQR-based clipping
    if iqr > 0:
        lo = q25 - 1.5 * iqr
        hi = q75 + 1.5 * iqr
        clipped = np.clip(raw_weight, lo, hi)
        denom = hi - lo
        if denom <= 0:
            return (min_width + max_width) * 0.5

        # 0..1 normalised
        normalized = (clipped - lo) / denom

        # gamma < 1: compress low weights, expand high weights
        gamma = 0.65
        scaled = normalized ** gamma
        return float(min_width + scaled * (max_width - min_width))

    # Fallback: almost-constant distribution → use simple min/max scaling
    w_min = float(np.min(weights))
    w_max = float(np.max(weights))
    spread = w_max - w_min
    if spread <= 0:
        # truly constant — neutral but visible
        return (min_width + max_width) * 0.5

    normalized = (raw_weight - w_min) / spread
    normalized = float(np.clip(normalized, 0.0, 1.0))
    gamma = 0.7
    scaled = normalized ** gamma
    return float(min_width + scaled * (max_width - min_width))


def calculate_adaptive_node_size(size_value: float, all_values: list[float]) -> float:
    """
    Map a centrality/score value to a node radius.

    Design goals:
        - Node sizes in [14, 52] so hubs are clearly visible.
        - Use percentiles (q10-q90) to reduce the impact of extreme outliers.
        - Slightly emphasise high values using a gamma < 1.
    """
    if not all_values:
        return 24.0

    vals = np.asarray(all_values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 24.0

    q10, q50, q90 = np.percentile(vals, [10, 50, 90])
    rng = q90 - q10
    min_size = 14.0
    max_size = 52.0

    # If there is essentially no spread, use a mid-sized node
    if rng <= 0:
        return (min_size + max_size) * 0.5

    # Clamp value into [q10, q90] and normalise to [0,1]
    if size_value <= q10:
        normalized = 0.0
    elif size_value >= q90:
        normalized = 1.0
    else:
        normalized = (size_value - q10) / rng

    normalized = float(np.clip(normalized, 0.0, 1.0))

    # Variance-aware gamma:
    #  - if spread is high relative to median, use smaller gamma
    #    to accentuate hubs
    #  - otherwise keep distribution gentler
    denom = abs(q50) + 1e-12
    spread_ratio = rng / denom if denom > 0 else 0.0
    gamma = 0.55 if spread_ratio > 2.0 else 0.8

    scaled = normalized ** gamma
    return float(min_size + scaled * (max_size - min_size))
