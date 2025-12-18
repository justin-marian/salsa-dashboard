import networkx as nx
from tqdm import tqdm
import time

from metrics.pagerank_hits import run_baselines_parallel


def compute_baseline_centralities(
    graph: nx.DiGraph,
    use_weights: bool,
    baseline_max_iterations: int,
    verbose: bool = True,
) -> tuple[dict, dict, dict]:
    """
    Compute baseline HITS and PageRank centralities on a directed graph.

    This helper wraps `run_baselines_parallel` and standardizes how you run
    and log the two classic link analysis baselines:
    - HITS (hubs & authorities)
    - PageRank

    It configures iteration limits and tolerances, optionally enabling
    edge-weighted versions of the algorithms.

    Parameters
    ----------
    graph:
        Directed NetworkX graph on which to compute centralities.
    use_weights:
        If True, use edge weights (attribute "weight") when available.
        If False, treat all edges as unweighted.
    baseline_max_iterations:
        Maximum number of iterations for both HITS and PageRank.
    verbose:
        If True, prints progress and timing information via `tqdm.write`.

    Returns
    -------
    hubs, authorities, pagerank:
        - hubs: dict[node, float]
        - authorities: dict[node, float]
        - pagerank: dict[node, float]

        All three dictionaries map node → centrality score.
    """
    if verbose:
        tqdm.write(
            f"[BASELINES] Computing HITS and PageRank centralities "
            f"(max_iterations={baseline_max_iterations}, use_weights={use_weights})"
        )

    started = time.perf_counter()

    baseline_parameters = {
        "hits_params": {
            "max_iter": baseline_max_iterations,
            "tol": 1e-6,
            "weights": use_weights,
        },
        "pr_params": {
            "max_iter": baseline_max_iterations,
            "tol": 1e-6,
            "weights": use_weights,
        },
    }

    # Execute parallel computation of both centrality measures
    baseline_pagerank, baseline_hubs, baseline_authorities = run_baselines_parallel(
        graph,
        hits_params=baseline_parameters["hits_params"],
        pr_params=baseline_parameters["pr_params"],
    )

    elapsed = time.perf_counter() - started
    if verbose:
        tqdm.write(f"[BASELINES] HITS and PageRank completed in {elapsed:.3f} seconds")

    # Note the return order: hubs, authorities, pagerank
    return baseline_hubs, baseline_authorities, baseline_pagerank
