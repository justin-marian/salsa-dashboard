from tqdm import tqdm
import networkx as nx
import time

from .salsa import SalsaSparse


def compute_salsa_centrality(
    graph: nx.DiGraph,
    use_weight: bool,
    max_iterations: int,
    tolerance: float,
    damping_factor: float,
    verbose: bool = True,
) -> tuple[dict, dict, SalsaSparse]:
    """
    Compute SALSA hub and authority scores on a directed graph.

    This function is a thin wrapper around ``SalsaSparse`` that standardizes
    how you configure and run SALSA in the pipeline. It:

    1. Instantiates ``SalsaSparse`` with the requested iteration limit,
        tolerance, damping factor, and GPU usage.
    2. Chooses whether to use edge weights (attribute ``"weight"``) or
        treat the graph as unweighted.
    3. Runs ``fit(graph, weight=...)``.
    4. Logs convergence information and timing (if ``verbose`` is True).
    5. Returns hub scores, authority scores, and the fitted ``SalsaSparse``
        instance (for introspection/debugging).

    SALSA (Stochastic Approach for Link-Structure Analysis) is similar in
    spirit to HITS: it computes two coupled centralities:

    - hub_scores[node]: quality as a “linker” (points to good authorities)
    - auth_scores[node]: quality as a “content source” (linked by good hubs)

    Parameters
    ----------
    graph:
        Directed NetworkX graph on which SALSA will be run.
    use_weight:
        If True, use edge attribute ``"weight"`` (when present). If False,
        ignore edge weights and treat all edges as having unit weight.
    max_iterations:
        Upper bound on the number of SALSA iterations.
    tolerance:
        Convergence threshold on the iterative updates. Smaller values require
        more iterations to converge.
    damping_factor:
        Damping parameter for SALSA's random walk component (analogous to
        PageRank's alpha, but with SALSA-specific semantics).
    verbose:
        If True, prints progress and convergence information via ``tqdm.write``.

    Returns
    -------
    hub_scores, auth_scores, salsa_algorithm:
        - hub_scores: dict[node, float]
        - auth_scores: dict[node, float]
        - salsa_algorithm: the fitted ``SalsaSparse`` instance, which exposes
        attributes such as:
            - ``converged`` (bool)
            - ``iterations`` (int)
            - ``hub_scores`` / ``auth_scores`` (the same dicts)

    Notes
    -----
    - This function currently always sets ``use_gpu=True`` when constructing
    ``SalsaSparse``. If you want a CPU fallback, that logic should be added
    here (e.g. based on CUDA availability or a config flag).
    """
    if verbose:
        tqdm.write(
            f"[SALSA] Running GPU-accelerated SALSA "
            f"(max_iterations≤{max_iterations}, tolerance={tolerance:.1e})"
        )

    started = time.perf_counter()

    # Initialize and configure SALSA
    salsa_algorithm = SalsaSparse(
        max_iter=max_iterations,
        tol=tolerance,
        damping=damping_factor,
        use_gpu=True,
        verbose=verbose,
    )

    # Choose weight attribute (or None for unweighted)
    weight_attribute = "weight" if use_weight else None

    # Run SALSA on the provided graph
    salsa_algorithm.fit(graph, weight=weight_attribute)

    # Convergence status logging
    convergence_status = "converged" if salsa_algorithm.converged else "did not converge"
    elapsed = time.perf_counter() - started

    if verbose:
        tqdm.write(
            f"[SALSA] Algorithm {convergence_status} in "
            f"{salsa_algorithm.iterations} iterations "
            f"({elapsed:.3f} seconds)"
        )

    return salsa_algorithm.hub_scores, salsa_algorithm.auth_scores, salsa_algorithm
