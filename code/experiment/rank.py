import numpy as np
from typing import Any


def topk(
    scores: dict[Any, float],
    k: int = 10,
    total_length: int | None = None,
    batch_size: int = 10000,
) -> list[tuple[Any, float]]:
    """
    Return the top-k (item, score) pairs from a score dictionary.

    For large dictionaries this uses NumPy's `argpartition` (O(n) selection)
    to avoid an O(n log n) full sort. For smaller dictionaries it simply
    falls back to Python's built-in sorting.

    Parameters
    ----------
    scores:
        Mapping from item → score (e.g. node → centrality).
    k:
        Number of top elements to return. If k <= 0, returns an empty list.
        If k > len(scores), k is clamped to len(scores).
    total_length:
        Optional “expected” number of items. If provided and larger than
        `batch_size`, we treat this as a “large” case and use the NumPy
        selection path. If None, `len(scores)` is used.
        (In most cases you can ignore this and rely on the default.)
    batch_size:
        Threshold used to choose between the “large” (argpartition) and
        “small” (sorted) code paths. If `total_length > batch_size`, the
        argpartition-based implementation is used.

    Returns
    -------
    top_items:
        List of (item, score) pairs sorted in descending order of score.
        Scores are rounded to 6 decimal places for readability.
    """
    # Handle trivial cases early
    if not scores:
        return []

    # Clamp k to valid range
    n = len(scores)
    if k <= 0:
        return []
    if k > n:
        k = n

    if total_length is None:
        total_length = n

    if total_length > batch_size:
        # Large case: use NumPy argpartition
        items_list = list(scores.items())
        score_values = np.array([score for _, score in items_list], dtype=float)

        # argpartition gets indices of the k largest scores (in arbitrary order)
        top_indices = np.argpartition(score_values, -k)[-k:]
        top_elements = [items_list[i] for i in top_indices]

        # Final sort of just the top-k elements
        sorted_elements = sorted(top_elements, key=lambda item: item[1], reverse=True)
        return [(item, round(score, 6)) for item, score in sorted_elements]

    # Small case: just sort everything
    sorted_items = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:k]
    return [(item, round(score, 6)) for item, score in sorted_items]
