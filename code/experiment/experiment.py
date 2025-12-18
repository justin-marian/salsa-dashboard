import time

import torch
import networkx as nx
from tqdm.auto import tqdm
from typing import Any
from collections.abc import Callable

from .utils import get_device
from .centrality import compute_salsa_centrality
from .detect_communities import detect_communities
from .centrality_baseline import compute_baseline_centralities
from .balance_communities import balance_community_sizes
from .report import report_results


def run_experiment(
    name: str,
    graph: nx.DiGraph,
    community_kwargs: dict[str, Any],
    community_method: str = "louvain",
    graph_smoothing_function: Callable[[Any], Any] = lambda x: x,
    damping_factor: float = 0.05,
    max_iterations: int = 100,
    tolerance: float = 1e-9,
    verbose: bool = True,
    use_weights: bool = True,
) -> tuple[dict, dict, dict, dict, dict, dict]:
    """
    Run the full graph analysis pipeline on a single dataset.
    This function orchestrates the complete workflow for one graph:

    1. Device selection / logging (CPU vs GPU).
    2. Graph cleaning:
        - remove self-loops,
        - drop isolated nodes (keep only nodes with degree > 0).
    3. Centrality computation:
        - SALSA hubs / authorities (GPU-accelerated via SalsaSparse),
        - HITS hubs / authorities,
        - PageRank scores.
    4. Community detection:
        - optional graph “smoothing” step (e.g. S_identity, S_break_reciprocals),
        - community detection via `detect_communities`,
        - post-processing via `balance_community_sizes`.
    5. Reporting:
        - timing, convergence, and top scores via `report_results`.

    Parameters
    ----------
    name:
        Human-readable dataset name used for logging (e.g. "ego-facebook").

    graph:
        Directed NetworkX graph to analyze. Will be cleaned in-place
        (self-loops removed, isolated nodes dropped) and then subgraphed.

    metric:
        Controls SALSA weighting mode; currently:
            - "salsa_unweighted" → ignore edge weights
            - "salsa_weighted"   → use edge attribute "weight" when available.

    community_method:
        Community detection backend to pass to `detect_communities`
        (e.g. "louvain", "leiden", "label_propagation").

    graph_smoothing_function:
        Structural transform applied before community detection, such as:
            - lambda G: G                    (no change),
            - S_identity(G)                  (explicit no-op),
            - S_break_reciprocals(G)         (drop one edge from each 2-cycle).
        Must accept a graph and return a graph.

    damping_factor:
        Damping parameter for SALSA (passed to `SalsaSparse`).

    max_iterations:
        Maximum number of iterations for both SALSA and the baseline
        HITS / PageRank runs.

    tolerance:
        Convergence tolerance for SALSA (and indirectly for baselines via
        `compute_baseline_centralities`).

    verbose:
        If True, prints detailed progress via `tqdm.write`.

    Returns
    -------
    salsa_hubs, salsa_authorities, hits_hubs, hits_authorities, pagerank_scores, balanced_communities:
        - salsa_hubs:         dict[node, float]
        - salsa_authorities:  dict[node, float]
        - hits_hubs:          dict[node, float]
        - hits_authorities:   dict[node, float]
        - pagerank_scores:    dict[node, float]
        - balanced_communities: dict[node, int] (node → community id after balancing)

        If the cleaned graph ends up empty / edgeless, all six outputs are `{}`.
    """
    device = get_device()
    if verbose:
        tqdm.write(f"[{name}] Using computational device: {device}")
        if torch.cuda.is_available():
            gpu_properties = torch.cuda.get_device_properties(0)
            tqdm.write(
                f"[{name}] GPU: {gpu_properties.name} | "
                f"Memory: {gpu_properties.total_memory / 1e9:.1f} GB"
            )

    # Graph preprocessing and cleaning
    started = time.time()
    original_node_count = graph.number_of_nodes()
    tqdm.write(f"[{name}] Original graph nodes: {original_node_count}")

    # Remove self-loops explicitly
    graph.remove_edges_from(list(nx.selfloop_edges(graph)))

    # Keep only nodes that participate in at least one edge
    connected_nodes = [node for node in graph.nodes() if graph.degree(node) > 0]
    graph = graph.subgraph(connected_nodes).copy()
    tqdm.write(f"[{name}] Active nodes after cleaning: {len(connected_nodes)}")

    # If cleaning killed everything, abort early
    if not (graph.number_of_nodes() > 0 and graph.number_of_edges() > 0):
        if verbose:
            tqdm.write(f"[{name}] Graph is empty or edgeless after cleaning")
        return {}, {}, {}, {}, {}, {}

    node_list = list(graph.nodes())
    final_node_count = len(node_list)

    # Centrality computation: SALSA
    salsa_hubs, salsa_authorities, salsa_object = compute_salsa_centrality(
        graph=graph,
        use_weight=use_weights,
        max_iterations=max_iterations,
        tolerance=tolerance,
        damping_factor=damping_factor,
        verbose=verbose,
    )

    # Report SALSA convergence status (if the object exposes the attributes)
    if hasattr(salsa_object, "converged") and hasattr(salsa_object, "iterations"):
        tqdm.write(
            f"[{name}] SALSA convergence: {salsa_object.converged} "
            f"in {salsa_object.iterations} iterations"
        )
    else:
        tqdm.write(f"[{name}] SALSA result object: {salsa_object}")

    # Baseline algorithms: HITS & PageRank
    hits_hubs, hits_authorities, pagerank_scores = compute_baseline_centralities(
        graph=graph,
        use_weights=use_weights,
        baseline_max_iterations=max_iterations,
        verbose=verbose,
    )

    # Community detection + balancing
    raw_communities = detect_communities(
        graph=graph,
        method=community_method,
        use_weights=use_weights,
        graph_smoothing_function=graph_smoothing_function,
        baseline_max_iterations=max_iterations,
        max_iterations=max_iterations,
        community_kwargs=community_kwargs,
        verbose=verbose,
    )

    # Balance community sizes to avoid extreme imbalances
    balanced_communities = balance_community_sizes(
        graph,
        raw_communities,
        verbose=verbose,
    )

    elapsed = time.time() - started
    if verbose:
        report_results(
            name,
            final_node_count,
            graph.number_of_edges(),
            elapsed,
            device,
            salsa_hubs,
            salsa_authorities,
            hits_hubs,
            hits_authorities,
            pagerank_scores,
        )

    return (
        salsa_hubs,
        salsa_authorities,
        hits_hubs,
        hits_authorities,
        pagerank_scores,
        balanced_communities,
    )


def run_experiment_batch(datasets: list[dict[str, Any]]) -> list[tuple]:
    """
    Run the full experiment pipeline on a list of dataset configs.

    Each config in ``datasets`` is expected to be a dict containing the
    keyword arguments for :func:`run_experiment` (e.g. ``name``, ``graph``,
    and optional overrides like ``metric``, ``community_method``, etc.).

    Example
    -------
    datasets = [
        {"name": "ego-facebook", "graph": G1},
        {"name": "bitcoin-otc",  "graph": G2, "metric": "salsa_weighted"},
    ]

    results = run_experiment_batch(datasets)

    Parameters
    ----------
    datasets:
        List of configuration dictionaries. For each dict, we call:
            run_experiment(**dataset_config)

    Returns
    -------
    experiment_results:
        List of tuples, each exactly the return value of `run_experiment`:

            (
                salsa_hubs,
                salsa_authorities,
                hits_hubs,
                hits_authorities,
                pagerank_scores,
                balanced_communities,
            )
    """
    experiment_results: list[tuple] = []
    for dataset_config in datasets:
        result = run_experiment(**dataset_config)
        experiment_results.append(result)
    return experiment_results
