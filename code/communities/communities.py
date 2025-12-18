"""
Unified interface for community detection algorithms.

This module wraps several community detection routines behind a single
function (`assign_communities`) and a small benchmark helper
(`compare_community_methods`). The goal is to:

- Always return a mapping: node -> integer community ID (0..C-1).
- Make it easy to swap / compare algorithms in downstream ranking and
graph analytics pipelines.

Supported Methods
-----------------
- louvain (default)
    Multilevel modularity optimization:
    1. Start with each node in its own community.
    2. Repeatedly move nodes to neighbor communities if modularity increases.
    3. Collapse communities into super-nodes and repeat.
    Produces good quality partitions on medium-to-large graphs, often very fast.

- label_propagation / lpa
    Asynchronous label propagation:
    1. Initialize each node with a unique label.
    2. Iteratively update each node's label to the most frequent label among
    its neighbors.
    3. Converges when labels stop changing.
    Extremely fast and parameter-free, but results depend on traversal order
    and random seed.

- greedy_modularity / greedy
    Clauset-Newman-Moore greedy modularity maximization:
    1. Start with each node as its own community.
    2. At each step merge the pair of communities that yields the largest
    increase (or smallest decrease) in modularity.
    3. Continue until no beneficial merges remain.
    Deterministic and conceptually simple; can be slower on large graphs.

- leiden
    Leiden algorithm (via igraph + leidenalg):
    1. Similar spirit to Louvain, but fixes several issues (e.g. badly
    connected communities).
    2. Guarantees that each community is locally well connected.
    Often yields higher-quality partitions than Louvain; best suited for
    small / medium graphs where quality is more important than raw speed.

- kcore / core
    k-core decomposition:
    1. Repeatedly remove nodes with degree < k.
    2. The remaining nodes form the k-core.
    We assign each node to its core index (coreness). This is not a classic
    community algorithm, but produces layers/strata based on how “deep” a
    node sits in the dense core of the graph.

- scc
    Strongly Connected Components (SCCs) for directed graphs:
    1. A strongly connected component is a set of nodes where every node can
    reach every other node via directed paths.
    2. These SCCs are used as communities.
    Naturally suited for directed graphs and flow / reachability analysis.

Design Notes
------------
- Most algorithms are designed for undirected graphs. We therefore project
the input DiGraph `Gw` to an undirected view `UG` before running them.
- We preserve `Gw` only conceptually (e.g. for SCC, which is inherently
directed, but that logic is handled inside `run_community`).
- `use_weights` tells the underlying algorithm whether to consult edge
weights when available; this allows the calling code to swap between
purely topological and weighted partitions.
"""

import time

import networkx as nx
from tqdm.auto import tqdm

from .run import run_community


def assign_communities(
    Gw: nx.DiGraph,
    method: str = "louvain",
    *,
    use_weights: bool = False,
    random_state: int = 42,
    # Louvain ============================================
    louvain_level: int | None = None,
    louvain_resolution: float = 1.0,
    # Greedy Modularity ==================================
    greedy_resolution: float = 1.0,
    # Leiden =============================================
    leiden_resolution: float = 1.0,
    leiden_iterations: int = -1,
    # K-core =============================================
    kcore_k: int | None = None,  # let it decide automatically
    verbose: bool = True,
) -> dict[object, int]:
    """
    Run a chosen community detection algorithm on a directed graph.

    Parameters
    ----------
    Gw:
        Input graph (directed). Most methods operate on an undirected
        projection of this graph; SCC-based methods may use direction.
    method:
        Name of the community detection algorithm:
        {'louvain', 'label_propagation', 'lpa', 'greedy_modularity',
         'greedy', 'leiden', 'kcore', 'core', 'scc'}.
    use_weights:
        If True, algorithms that support edge weights will use them
        (e.g. weighted modularity); otherwise edges are treated as
        unweighted.
    random_state:
        Seed for algorithms with randomness (Louvain, Leiden, LPA, etc.),
        to make runs reproducible.

    Louvain-specific
    ----------------
    louvain_level:
        Optional level of the Louvain hierarchy to return; if None, use the
        final (coarsest) partition.
    louvain_resolution:
        Resolution parameter controlling the size of discovered communities:
        >1 tends to find more/smaller communities, <1 fewer/larger ones.

    Greedy modularity
    -----------------
    greedy_resolution:
        Same conceptual role as `louvain_resolution`, but used with the
        greedy modularity algorithm.

    Leiden-specific
    ---------------
    leiden_resolution:
        Resolution parameter (analogous to Louvain's).
    leiden_iterations:
        Maximum number of refinement iterations. A negative value usually
        means "until convergence" (delegated to `run_community`).

    K-core
    ------
    kcore_k:
        If None, `run_community` will choose k (e.g. maximum coreness).
        Otherwise, a fixed k-core is computed and used as labels.

    verbose:
        If True, prints method, graph size, and timing information.

    Returns
    -------
    assignments:
        Dictionary mapping each node in `Gw` to an integer community ID
        (0..C-1). Community IDs are arbitrary but contiguous.

    Notes
    -----
    - We immediately project the directed graph to an undirected graph
    `UG = Gw.to_undirected()` because many community algorithms assume
    undirected structure. The directed graph `Gw` is still conceptually
    available to `run_community` for methods like SCC that need direction.
    """
    # Most community algorithms expect / perform better on undirected graphs,
    # so we create an undirected view for them.
    UG = Gw.to_undirected()
    method = method.lower().strip()

    started = time.perf_counter()
    if verbose:
        tqdm.write(
            f"[COMM] method={method}, use_weights={use_weights}, "
            f"nodes={UG.number_of_nodes()}, edges={UG.number_of_edges()}"
        )

    # Build method-specific hyperparameters, instead of dumping *everything*.
    common_kwargs: dict = {
        "method": method,
        "UG": UG,
        "use_weights": use_weights,
        "verbose": verbose,
        "random_state": random_state,
    }

    method_kwargs: dict = {}

    if method == "louvain":
        method_kwargs["louvain_level"] = louvain_level
        method_kwargs["louvain_resolution"] = louvain_resolution

    elif method in ("greedy_modularity", "greedy"):
        method_kwargs["greedy_resolution"] = greedy_resolution

    elif method == "leiden":
        method_kwargs["leiden_resolution"] = leiden_resolution
        method_kwargs["leiden_iterations"] = leiden_iterations

    elif method in ("kcore", "core"):
        method_kwargs["kcore_k"] = kcore_k

    # `scc` and label propagation do not need extra hyperparams here

    # Filter out unset / sentinel values so logs stay minimal
    filtered_method_kwargs = {
        k: v
        for k, v in method_kwargs.items()
        if v is not None and v != -1
    }

    if verbose and filtered_method_kwargs:
        tqdm.write(f"[COMM] {method} hyperparameters: {filtered_method_kwargs}")

    # Delegate to the actual implementation
    assignments = run_community(
        **common_kwargs,
        **method_kwargs,   # still pass full dict so defaults are handled there
    )

    elapsed = time.perf_counter() - started
    if verbose:
        num_groups = len(set(assignments.values()))
        tqdm.write(f"[COMM] {method} done in {elapsed:.3f}s | groups={num_groups}")

    return assignments


def compare_community_methods(
    Gw: nx.DiGraph,
    methods: list[str] | None = None,
    use_edge_weights: bool = False,
    verbose: bool = True,
) -> dict[str, dict]:
    """
    Run several community detection methods on the same graph and compare them.

    This is a lightweight benchmarking helper: it calls `assign_communities`
    for each requested method, collects basic statistics, and returns them
    in a single dictionary.

    Parameters
    ----------
    Gw:
        Input directed graph. As in `assign_communities`, most methods will
        operate on an undirected projection of this graph.
    methods:
        List of algorithm names to benchmark. If None, a default set is used:
        - 'louvain'           : strong general-purpose baseline.
        - 'label_propagation' : very fast, sanity-check / rough partition.
        - 'greedy_modularity' : deterministic, modularity-based baseline.
        - 'leiden'            : high-quality partitions when available.
        - 'kcore'             : structural layers based on coreness.
        - 'scc'               : strongly connected components for directed
                                graphs (reachability-based communities).
    use_edge_weights:
        If True, passes `use_weights=True` into `assign_communities`, so
        algorithms that support weighted edges can take advantage of them.
    verbose:
        If True, prints timing and community counts for each method.

    Returns
    -------
    results:
        Dictionary keyed by method name. For each method, we store:
        {
            'communities':   dict[node, int],   # community assignment
            'num_communities': int,             # how many communities found
            'time_seconds':   float,            # runtime in seconds
            'success':        bool              # currently always True
        }

    Notes
    -----
    - This function is intentionally simple; failures in one method should
    ideally be caught and recorded (e.g. with try/except) without breaking
    the others. That behavior can be added inside this loop if needed.
    """
    if methods is None:
        methods = [
            "louvain",
            "label_propagation",
            "greedy_modularity",
            "leiden",
            "kcore",
            "scc",
        ]

    assignments: dict[str, dict] = {}
    for method in methods:
        started = time.perf_counter()

        # Run the chosen algorithm; we suppress verbose output here to keep
        # the comparison log concise, and handle printing ourselves below.
        communities = assign_communities(
            Gw,
            method=method,
            use_weights=use_edge_weights,
            verbose=False,
        )

        num_communities = len(set(communities.values()))
        elapsed = time.perf_counter() - started

        assignments[method] = {
            "communities": communities,
            "num_communities": num_communities,
            "time_seconds": elapsed,
            "success": True,
        }

        if verbose:
            tqdm.write(f"[COMPARE] {method}: {num_communities} communities in {elapsed:.3f}s")

    return assignments
