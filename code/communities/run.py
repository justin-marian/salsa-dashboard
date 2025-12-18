import networkx as nx
from typing import Any

from .louvain import louvain_communities
from .lpa import label_propagation_communities
from .greedy import greedy_modularity_communities
from .leiden import leiden_communities
from .kcore import kcore_communities
# from .connex import strongly_connected_components


def run_community(
    method: str,
    UG: nx.Graph,
    # Gw: nx.DiGraph,
    use_weights: bool,
    **kwargs: Any,
) -> dict[Any, int]:
    """
    Dispatch to a specific community detection algorithm and run it.

    This function serves as a central router for all supported community
    detection methods. It hides the individual function signatures and exposes
    a simple interface:
        - Choose algorithm by name (``method``).
        - Provide a pre-built (typically undirected) graph ``UG``.
        - Optionally enable ``use_weights``.
        - Pass algorithm-specific hyperparameters via ``**kwargs``.

    Supported method names (aliases)
    --------------------------------
    - "louvain"
        → :func:`louvain_communities`
        Kwargs used:
            - level        : int | None (hierarchy cut; see Louvain doc)
            - resolution   : float (community granularity)
            - random_state : int (reproducibility)

    - "leiden"
        → :func:`leiden_communities`
        Kwargs used:
            - resolution   : float (community granularity)
            - iterations   : int (max refinement iterations; -1 for auto)
            - random_state : int (reproducibility)

    - "label_propagation", "lpa"
        → :func:`label_propagation_communities`
        Kwargs used:
            - random_state : int (seed for update order / tie-breaking)

    - "greedy_modularity", "greedy"
        → :func:`greedy_modularity_communities`
        Kwargs used:
            - resolution   : float (resolution parameter for modularity)

    - "kcore", "core"
        → :func:`kcore_communities`
        Kwargs used:
            - kcore_k      : int | None (threshold for collapsing cores)

    (planned)
    - "scc"
        → :func:`strongly_connected_components` on the directed graph ``Gw``
            (currently commented out).

    Parameters
    ----------
    method:
        Name (or alias) of the chosen community detection algorithm.
        The lookup is case-insensitive and ignores surrounding whitespace.
    UG:
        Undirected graph on which to run the community algorithm. For methods
        that conceptually need a directed graph (e.g. SCC), the directed
        version `Gw` would be used instead (see commented-out code).
    use_weights:
        Global switch that tells weight-aware algorithms whether to read the
        `"weight"` edge attribute:
        - If True: pass `"weight"` to algorithms that support it.
        - If False: they treat all edges as unweighted.
    **kwargs:
        Algorithm-specific hyperparameters, passed through by name. Only a
        subset is read by each method (see mapping above). Common keys:
            - "resolution"
            - "iterations"
            - "level"
            - "kcore_k"
            - "random_state"

    Returns
    -------
    assignments:
        Dictionary mapping each node in ``UG`` to an integer community id
        (0..C-1) as produced by the selected algorithm.

    Raises
    ------
    ValueError
        If an unknown method name is provided.

    Notes
    -----
    - This function is meant to be the single entry point for community
    detection in higher-level pipelines, so that adding/removing methods
    only requires extending the dispatch table below.
    - The `print` at the end logs the full kwargs dict, which is helpful for
    reproducibility/debugging of hyperparameter choices.
    """

    # Dispatch table: method name → zero-argument callable that runs it.
    # We capture `UG`, `use_weights`, and hyperparameters via lambdas so that
    # the selection logic is clean and the call site at the end is uniform.
    method_handlers: dict[str, Any] = {
        "louvain": lambda: louvain_communities(
            UG,
            use_weights=use_weights,
            level=kwargs.get("level"),
            resolution=kwargs.get("resolution", 1.0),
            random_state=kwargs.get("random_state", 42),
        ),
        "leiden": lambda: leiden_communities(
            UG,
            use_weights=use_weights,
            resolution=kwargs.get("resolution", 1.0),
            iterations=kwargs.get("iterations", -1),
            random_state=kwargs.get("random_state", 42),
        ),
        "label_propagation": lambda: label_propagation_communities(
            UG,
            use_weights=use_weights,
            random_state=kwargs.get("random_state", 42),
        ),
        "lpa": lambda: label_propagation_communities(
            UG,
            use_weights=use_weights,
            random_state=kwargs.get("random_state", 42),
        ),
        "greedy_modularity": lambda: greedy_modularity_communities(
            UG,
            weight=("weight" if use_weights else None),
            resolution=kwargs.get("resolution", 1.0),
        ),
        "greedy": lambda: greedy_modularity_communities(
            UG,
            weight=("weight" if use_weights else None),
            resolution=kwargs.get("resolution", 1.0),
        ),
        "kcore": lambda: kcore_communities(
            UG,
            k=kwargs.get("kcore_k"),
        ),
        "core": lambda: kcore_communities(
            UG,
            k=kwargs.get("kcore_k"),
        ),
        # "scc": lambda: strongly_connected_components(Gw),
    }

    # Normalize method name (case/whitespace) before lookup.
    handler = method_handlers.get(method.lower().strip())
    # Helpful debug line: log which hyperparameters were actually passed in.
    print("Hyperparameters passed to community detection:", kwargs)
    if not handler:
        raise ValueError(f"Unknown community detection method: {method}")
    # Execute the selected algorithm and return its node → community mapping.
    return handler()
