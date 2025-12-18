from tqdm import tqdm
import networkx as nx
import time
from typing import Any
from collections.abc import Callable

from communities.communities import assign_communities


def detect_communities(
    graph: nx.DiGraph,
    method: str,
    use_weights: bool,
    graph_smoothing_function: Callable[[nx.DiGraph], nx.DiGraph],
    baseline_max_iterations: int,
    max_iterations: int,
    community_kwargs: dict[str, Any],
    verbose: bool = True,
) -> dict[Any, int]:
    """
    Run community detection on a (possibly preprocessed) directed graph.

    This is the main entry point for your pipeline's community detection:
    it handles graph preprocessing, a simple method fallback heuristic, and
    consistent integer labeling of community IDs.

    Steps
    -----
    1. Apply `graph_smoothing_function` to the input graph:
    processed_graph = S_coll(graph)
    Typical choices:
        - S_identity           → no structural change
        - S_break_reciprocals  → drop one edge in each reciprocal pair
    If the result is not a DiGraph, it is converted to directed via
    `to_directed()`.

    2. Choose an “effective” method:
    - If `baseline_max_iterations < max_iterations`, use `"label_propagation"` (fast, parameter-free).
    - Otherwise, use the requested `method` (e.g. "louvain", "leiden").

    This gives you a crude performance-aware fallback: when you know
    your baseline (HITS/PageRank) is iteration-limited, you switch to a
    cheaper community method.

    3. Call `assign_communities(...)` to obtain node → community assignments.

    4. Normalize all community labels to plain ints (e.g. ensure NumPy scalars
    become Python ints), returning a mapping node → int community_id.

    Parameters
    ----------
    graph:
        Original directed NetworkX graph.
    method:
        Desired community detection method name to pass to `assign_communities`
        (e.g. "louvain", "leiden", "label_propagation", "kcore", ...).
    use_weights:
        If True, downstream community detection may use edge weights
        (typically stored under the "weight" attribute) where supported.
    graph_smoothing_function:
        A structural transformation to apply before detection, such as
        `S_identity` or `S_break_reciprocals`. It should accept a DiGraph
        and return a graph (directed or undirected).
    baseline_max_iterations:
        Iteration budget used for baselines (HITS/PageRank). If this is
        smaller than `max_iterations`, we fall back to `"label_propagation"`.
    max_iterations:
        Conceptual iteration budget for “full” methods. Used only for the
        method-selection heuristic; it is not passed directly here.
    verbose:
        If True, logs which method is used, and how many communities are found.

    Returns
    -------
    community_assignments:
        Dictionary mapping node → integer community id (0..K-1, but the
        exact range is determined by `assign_communities`).
    """
    # Apply graph smoothing / structural transform
    processed_graph = graph_smoothing_function(graph)
    if not isinstance(processed_graph, nx.DiGraph):
        processed_graph = processed_graph.to_directed()

    effective_method = "label_propagation" if baseline_max_iterations < max_iterations else method
    if verbose:
        tqdm.write(f"[COMMUNITIES] Detecting communities using '{effective_method}' method")

    started = time.perf_counter()

    ck = dict(community_kwargs or {})
    for key in ["random_state", "seed", "rs"]:
        ck.pop(key, None)

    for key in ["use_weight", "use_weights"]:
        ck.pop(key, None)

    # Explicit resolved values
    local_random_state = community_kwargs.get("random_state", 42)
    local_use_weights  = community_kwargs.get("use_weight", use_weights)

    # --- Call assign_communities safely ---
    community_mapping = assign_communities(
        processed_graph,
        method=effective_method,
        use_weights=local_use_weights,
        random_state=local_random_state,
        verbose=verbose,
        **ck,
    )

    community_assignments = {node: int(cid) for node, cid in community_mapping.items()}

    elapsed = time.perf_counter() - started
    if verbose:
        tqdm.write(
            f"[COMMUNITIES] Detected {len(set(community_assignments.values()))} communities "
            f"in {elapsed:.3f} seconds"
        )

    return community_assignments
